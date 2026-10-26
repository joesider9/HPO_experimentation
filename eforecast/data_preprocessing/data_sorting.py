import os
import joblib

import numpy as np
import pandas as pd

from eforecast.common_utils.dataset_utils import concat_by_columns
from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.dataset_creation.files_manager import FilesManager


class DataColumnSorter:
    def __init__(self, static_data, recreate=False, online=False, train=False):
        self.sorters = dict()
        self.online = online
        self.train = train
        self.filename = os.path.join(static_data['path_model'], 'sorters.pickle')
        if os.path.exists(self.filename):
            self.sorters = joblib.load(self.filename)
        if recreate:
            self.sorters = dict()
            if os.path.exists(self.filename):
                os.remove(self.filename)
        self.static_data = static_data
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.files_manager = FilesManager(static_data, is_online=online)

    def fit(self, x, y, dataset_name):
        if f'sorter_{dataset_name}' not in self.sorters.keys():
            x, y = sync_datasets(x, y, name1=dataset_name, name2='target')
            sorter = dict()
            shape = x.shape
            if not isinstance(x, pd.DataFrame):
                raise ValueError('Sorting is performed only to dataframes')
            corr = []
            for f in range(shape[1]):
                corr.append(np.abs(np.corrcoef(x.values[:, f], y.values.ravel())[1, 0]))
            ind = np.argsort(np.array(corr))[::-1]
            sorter['column_index'] = ind
            sorter['columns'] = x.columns[ind].to_list()
            self.sorters[f'sorter_{dataset_name}'] = sorter
            self.save()

    def save(self):
        joblib.dump(self.sorters, self.filename)

    def transform(self, data, dataset_name):
        sorter = self.sorters[f'sorter_{dataset_name}']
        if len(sorter['columns']) != len(data.columns):
            raise ValueError(f'the length of sorter columns list should be the same with dataframe columns')
        for col in data.columns:
            if col not in sorter['columns']:
                raise ValueError(f"Cannot sort dataframe columns. {col} not in sorter columns list")
        try:
            data = data[sorter['columns']]
        except:
            raise ValueError(f"Cannot sort dataframe columns.")
        return data

    def update(self, data, y, dataset_name):
        sorter_name = f'sorter_{dataset_name}'
        if sorter_name not in self.sorters.keys():
            print(f'Update sorter for dataset {dataset_name}')
            if not self.online:
                self.fit(data, y, dataset_name)
            else:
                raise ValueError(f"sorter named {sorter_name} isn't trained")
