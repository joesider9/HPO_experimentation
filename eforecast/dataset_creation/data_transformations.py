import os
import joblib

import numpy as np
import pandas as pd

from eforecast.common_utils.nwp_utils import clear_sky
from eforecast.common_utils.nwp_utils import get_clear_sky

from eforecast.dataset_creation.files_manager import FilesManager


class DataTransformer:
    def __init__(self, static_data, recreate=False, is_online=False, train=False):
        self.transformers = dict()
        self.online = is_online
        self.train = train
        self.filename = os.path.join(static_data['path_model'], 'transformers.pickle')
        if os.path.exists(self.filename):
            try:
                self.transformers = joblib.load(self.filename)
            except:
                self.transformers = dict()
                os.remove(self.filename)
        if recreate:
            self.transformers = dict()
            if os.path.exists(self.filename):
                os.remove(self.filename)
        self.static_data = static_data
        self.variables_index = {var_data['name']: var_data['transformer']
                                for var_data in self.static_data['variables']
                                if var_data['transformer'] is not None}
        self.coord = self.static_data['coord']
        self.local_timezone = self.static_data['local_timezone']
        self.site_timezone = self.static_data['site_timezone']
        self.ts_resolution = self.static_data['ts_resolution']
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.files_manager = FilesManager(static_data, is_online=is_online)

    def save(self):
        joblib.dump(self.transformers, self.filename)

    def fit(self, data, variable, data_dates=None):
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if isinstance(data, pd.DataFrame):
            dates = data.index
        else:
            if data_dates is None:
                raise ValueError('If data is not dataframe, data_dates should be provided')
            dates = data_dates
        transformation = self.variables_index[variable]
        if transformation not in self.transformers.keys():
            if transformation == 'clear_sky':
                ghi = get_clear_sky(dates, self.coord[0], self.coord[1], self.local_timezone, self.site_timezone,
                                    self.ts_resolution)
                self.transformers[transformation] = {'max': ghi.max(),
                                                     'values': ghi}
            else:
                raise NotImplementedError(f'{transformation} transformation is not implemented yet')
        self.save()

    def transform(self, data, variable, data_dates=None):
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if isinstance(data, pd.DataFrame):
            dates = data.index
        else:
            if data_dates is None:
                raise ValueError('If data is not dataframe, data_dates should be provided')
            dates = data_dates
        transformation = self.variables_index[variable]
        if transformation not in self.transformers.keys():
            self.update(data, data_dates=data_dates)
        if transformation == 'clear_sky':
            ghi = self.transformers[transformation]['values']
            dates_diff = dates.difference(ghi.index)
            if dates_diff.shape[0] > 0:
                ghi_new = get_clear_sky(dates_diff, self.coord[0], self.coord[1], self.local_timezone,
                                        self.site_timezone, self.ts_resolution)
                ghi = pd.concat([ghi, ghi_new])
            ghi = ghi.loc[dates]
            rate = np.tile(np.expand_dims((self.transformers[transformation]['max'] / ghi).values,
                                          axis=[i for i in range(data.ndim - 1, 1, -1)]),
                           [1] + list(data.shape[1:]))
            data_transformed = rate * data
        else:
            raise NotImplementedError(f'{transformation} transformation is not implemented yet')
        return data_transformed

    def update(self, data, variable, data_dates=None):
        print(f'Update imputer')
        if not self.online:
            self.fit(data, variable, data_dates=data_dates)
