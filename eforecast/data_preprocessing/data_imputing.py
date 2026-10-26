import os
import joblib

import numpy as np
import pandas as pd

from sklearn.impute import MissingIndicator

from eforecast.dataset_creation.files_manager import FilesManager


class DataImputer:
    def __init__(self, static_data, recreate=False, online=False, train=False):
        self.imputers = dict()
        self.online = online
        self.train = train
        self.filename = os.path.join(static_data['path_model'], 'imputers.pickle')
        if os.path.exists(self.filename):
            try:
                self.imputers = joblib.load(self.filename)
            except:
                self.imputers = set()
                os.remove(self.filename)
        if recreate:
            self.imputers = set()
            if os.path.exists(self.filename):
                os.remove(self.filename)
        self.static_data = static_data
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.files_manager = FilesManager(static_data, is_online=online)

    def save(self):
        joblib.dump(self.imputers, self.filename)

    def fit(self, data, data_dates=None):
        columns = None
        index = None
        missing_indicator = MissingIndicator(features="all")
        shape = data.shape
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if isinstance(data, pd.DataFrame):
            index = data.index
            columns = data.columns
            data = data.values
        if len(shape) > 2:
            index = data_dates
            data = data.reshape(-1, np.prod(shape[1:]))
        if index is None:
            raise ValueError('You should provide dates of numpy array')
        columns = [f'x_{i}' for i in range(data.shape[1])] if columns is None else columns
        flag_missing = missing_indicator.fit_transform(data)
        ind_nan_feature = np.where(np.all(flag_missing, axis=0))[0]
        if len(ind_nan_feature) > 0:
            raise ValueError(f'the feature {columns[ind_nan_feature]} have NaN all their values')
        ind_nan_dates = np.where(np.any(flag_missing, axis=1))[0]
        if len(ind_nan_dates) > 0:
            self.imputers = self.imputers.union(set(index[ind_nan_dates]))
        self.save()

    def transform(self, data, data_dates=None):
        self.update(data, data_dates=data_dates)
        index = None
        columns = None
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if isinstance(data, pd.DataFrame):
            index = data.index
            columns = data.columns
            data = data.values
        if data_dates is not None:
            index = data_dates
        if index is None:
            raise ValueError('You should provide dates of numpy array')

        dates = index.difference(pd.DatetimeIndex(self.imputers))
        ind_dates = index.get_indexer(dates)
        data_transformed = data[ind_dates]
        index = index[ind_dates]

        if index is not None and columns is not None:
            data_transformed = pd.DataFrame(data_transformed, index=index, columns=columns)
        return data_transformed, index

    def update(self, data, data_dates=None):
        print(f'Update imputer')
        if not self.online:
            self.fit(data, data_dates=data_dates)
