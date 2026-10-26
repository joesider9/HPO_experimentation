import os

import joblib
import pvlib
import pandas as pd
import numpy as np

from eforecast.dataset_creation.nwp_data.dataset_nwp_creator import DatasetNWPsCreator
from eforecast.dataset_creation.nwp_data.dataset_nwp_organizer import DatasetNWPsOrganizer
from eforecast.dataset_creation.nwp_data.dataset_nwp_compressor import DatasetNWPsCompressor
from eforecast.dataset_creation.data_transformations import DataTransformer
from eforecast.dataset_creation.files_manager import FilesManager

from eforecast.common_utils.date_utils import sp_index


class DatasetCreator:
    def __init__(self, static_data, recreate=False, train=True, is_online=False, dates=None):
        self.static_data = static_data
        self.is_online = is_online
        self.train = train
        self.dates = dates
        if self.is_online and self.dates is None:
            raise ValueError('If it runs online you should provide dates')
        self.path_data = self.static_data['path_data']
        self.horizon_type = static_data['horizon_type']
        self.nwp_models = static_data['NWP']
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.transformer = DataTransformer(self.static_data, recreate=recreate, is_online=self.is_online,
                                           train=self.train)
        self.files_manager = FilesManager(self.static_data, is_online=self.is_online, train=self.train)
        if recreate or is_online:
            self.files_manager.remove_row_data_files()
            self.files_manager.remove_nwps()
            self.files_manager.remove_target_files()
            for merge in self.nwp_data_merge:
                self.files_manager.remove_nwps_merged(merge)
                for compress in self.nwp_data_compress:
                    self.files_manager.remove_nwps_compressed(merge, compress)

    def create_nwp_dataset(self):
        if self.nwp_models is not None:
            nwp_data_creator = DatasetNWPsCreator(self.static_data, self.transformer, dates=self.dates,
                                                  is_online=self.is_online)
            nwp_data = self.files_manager.check_if_exists_nwp_data()
            if nwp_data is None:
                nwp_data = nwp_data_creator.make_dataset()
                self.files_manager.save_nwps(nwp_data)

    def merge_nwp_dataset(self):
        nwp_data = self.files_manager.check_if_exists_nwp_data()
        if nwp_data is None:
            raise ImportError('Merge NWP dataset failed due to nwp_data is None, data_nwp_creator seems to fail')
        for merge in self.nwp_data_merge:
            data_merged = self.files_manager.check_if_exists_nwps_merged(merge, get_all=True)
            nwp_data_merged, nwp_metadata = data_merged['data'], data_merged['nwp_metadata']
            if nwp_data_merged is None:
                nwp_data_organizer = DatasetNWPsOrganizer(self.static_data, nwp_data)
                nwp_data_merged, nwp_metadata = nwp_data_organizer.merge(merge)
                self.files_manager.save_nwps_merged(merge, nwp_data_merged, nwp_metadata)

    def compress_nwp_datasets(self):
        print(f"Dataset NWP compressing started for project {self.static_data['_id']}")
        for merge in self.nwp_data_merge:
            data_merged = self.files_manager.check_if_exists_nwps_merged(merge, get_all=True)
            nwp_data_merged, nwp_metadata = data_merged['data'], data_merged['nwp_metadata']
            for compress in self.nwp_data_compress:
                data_compressed = self.files_manager.check_if_exists_nwps_compressed(merge, compress, get_all=True)
                nwp_compressed_all, nwp_compressed, nwp_compressed_distributed, metadata = \
                    data_compressed['data_compressed_all'], \
                    data_compressed['data_compressed'], \
                    data_compressed['data_compressed_distributed'], \
                    data_compressed['nwp_metadata']
                if nwp_compressed is None:
                    nwp_data_compressor = DatasetNWPsCompressor(self.static_data, nwp_data_merged, nwp_metadata,
                                                                compress)
                    nwp_compressed_all, nwp_compressed, nwp_compressed_distributed = nwp_data_compressor.compress()
                    self.files_manager.save_nwps_compressed(merge, compress, nwp_compressed_all, nwp_compressed,
                                                            nwp_compressed_distributed, nwp_metadata)

    def concat_lagged_data(self, data, var_name, var_data, use_diff_between_lags=False):
        data = data.sort_index()
        dates = pd.date_range(data.index[0], data.index[-1],
                              freq=self.static_data['ts_resolution'])
        data_temp = pd.DataFrame(index=dates, columns=data.columns)
        data_temp.loc[data.index] = data
        for lag in var_data['lags']:
            if isinstance(lag, int):
                if var_data['source'] == 'target':
                    data_temp[f'{var_name}_lag_{lag}'] = data_temp[self.static_data['project_name']].shift(-lag)
                else:
                    data_temp[f'{var_name}_lag_{lag}'] = data_temp[var_data['source']].shift(-lag)
            else:
                raise NotImplemented('Last_year_lags')
        if use_diff_between_lags:
            for lag1 in var_data['lags']:
                for lag2 in var_data['lags']:
                    if lag1 > lag2:
                        data_temp[f'Diff_{var_name}_lag{lag1}_lag{lag2}'] = \
                            data_temp[f'{var_name}_lag_{lag1}'] - data_temp[f'{var_name}_lag_{lag2}']
                        data_temp[f'Diff2_{var_name}_lag{lag1}_lag{lag2}'] = np.square(
                            data_temp[f'{var_name}_lag_{lag1}'] - data_temp[f'{var_name}_lag_{lag2}'])
        return data_temp

    def create_autoregressive_dataset(self):
        variables = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                          if var_data['type'] == 'timeseries' and var_data['source'] == 'target'])
        if len(variables) == 0:
            return None
        data_arma = pd.DataFrame()
        for var_name, var_data in variables.items():
            data = self.static_data['data'].copy(deep=True)
            data = self.concat_lagged_data(data, var_name, var_data,
                                           use_diff_between_lags=self.static_data['use_diff_between_lags'])
            data = data.dropna(axis='index')
            data = data.drop(columns=[self.static_data['project_name']])
            data_arma = pd.concat([data_arma, data], axis=1)

        return data_arma

    def create_calendar_dataset(self):
        sin_transformer = lambda x: np.sin(x / period * 2 * np.pi)
        variables_index = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                if var_data['type'] == 'calendar' and var_data['source'] == 'index'])
        data = self.static_data['data'].copy(deep=True)
        index = data.index
        for var_name, var_data in variables_index.items():
            if var_name == 'hour':
                values = index.hour.values
                period = 24
            elif var_name == 'month':
                values = index.month.values
                period = 12
            elif var_name == 'dayweek':
                values = index.dayofweek.values
                period = 7
            elif var_name == 'sp_index':
                values = [sp_index(d, self.static_data['country']) for d in index]
            else:
                raise ValueError(f'Unknown variable {var_name} for index and calendar')

            values = pd.DataFrame(values, columns=[var_name], index=index)
            data = pd.concat([data, values], axis=1)
        variables_astral = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                 if var_data['type'] == 'calendar' and var_data['source'] == 'astral'])
        if len(variables_astral) == 0 and len(variables_index) == 0:
            return None
        for var_name, var_data in variables_astral.items():
            solpos = pvlib.solarposition.get_solarposition(index, self.static_data['coord'][0]
                                                           , self.static_data['coord'][1])
            if var_name not in {'azimuth', 'zenith'}:
                raise ValueError(f'Unknown variable {var_name} for astral and calendar. '
                                 f'Accepted values are azimuth, zenith')
            data[var_name] = solpos[var_name].values
        data = data.dropna(axis='index')
        data = data.drop(columns=[self.static_data['project_name']])

        return data

    def create_extra_ts_datasets(self):
        variables_extra = dict([(var_data['name'], var_data) for var_data in self.static_data['variables']
                                if var_data['type'] == 'timeseries' and var_data['source'] != 'target'])
        if len(variables_extra) == 0:
            return None
        data_extra = pd.DataFrame()
        for var_name, var_data in variables_extra:
            if var_data['source'].endswith('.csv'):
                if os.path.exists(var_data['source']):
                    data = pd.read_csv(var_data['source'], index_col=0, header=0, parse_dates=True)
                else:
                    raise ImportError(f"{var_data['source']} does not exists")
            else:
                data = pd.read_csv(self.static_data['filename'], index_col=0, header=0, parse_dates=True)
                if var_data['source'] not in data.columns:
                    raise ValueError(f"{var_data['source']} does not exists in main file columns. "
                                     f"Filename is {self.static_data['filename']}")
                data = data[var_data['source']].to_frame()
            data = self.concat_lagged_data(data, var_name, var_data)
            data = data.dropna(axis='index')
            data = data.drop(columns=[var_data['source']])
            data_extra = pd.concat([data_extra, data], axis=1)
        return data_extra

    def create_row_datasets(self):
        data_row = self.files_manager.check_if_exists_row_data()
        if data_row is None:
            data_arma = self.create_autoregressive_dataset()
            data_extra = self.create_extra_ts_datasets()
            data_calendar = self.create_calendar_dataset()
            data_row = pd.DataFrame()
            for data in [data_arma, data_extra, data_calendar]:
                if data is not None:
                    data_row = pd.concat([data_row, data], axis=1)
            data_row = data_row.dropna(axis='index')
            self.files_manager.save_row_data(data_row)

    def create_target(self):
        data = self.files_manager.check_if_exists_target()
        if data is None:
            data = self.static_data['data'].copy(deep=True)
            if self.static_data['horizon_type'] == 'multi-output':
                for hor in range(self.static_data['horizon']):
                    data['targ_h'] = data[self.static_data['project_name']].shift(hor)
                data = data.dropna(axis='index')
                data = data.drop(columns=[self.static_data['project_name']])
            else:
                data.columns = ['target']
            self.files_manager.save_target(data)
