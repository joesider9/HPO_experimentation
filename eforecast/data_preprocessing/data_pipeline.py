import numpy as np

from eforecast.common_utils.dataset_utils import concat_by_columns

from eforecast.data_preprocessing.data_scaling import Scaler
from eforecast.data_preprocessing.data_imputing import DataImputer
from eforecast.data_preprocessing.data_sorting import DataColumnSorter
from eforecast.dataset_creation.files_manager import FilesManager


class DataPipeline:
    def __init__(self, static_data, recreate=False, online=False, train=False):
        self.train = train
        self.static_data = static_data
        self.scale_target_method = self.static_data['scale_target_method']
        self.scale_row_method = self.static_data['scale_row_method']
        self.scale_nwp_method = self.static_data['scale_nwp_method']
        self.scaler = Scaler(static_data, recreate=recreate, online=online, train=train)
        self.imputer = DataImputer(static_data, recreate=recreate, online=online, train=train)
        self.sorter = DataColumnSorter(static_data, recreate=recreate, online=online, train=train)
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.files_manager = FilesManager(static_data, is_online=online, train=train)

    def transform_pipe(self, data, dataset_name, what_data=None, data_row=None, data_dates=None, inverse=False):
        if data_row is not None:
            name = f'data_row_{self.scale_row_method}'
            data_row = self.scaler.transform(data_row, name)
            if data_dates is None:
                raise ValueError('You should provide dates of numpy array')
        if what_data == 'target':
            if not inverse:
                data = self.scaler.transform(data, dataset_name)
            return data.dropna(axis=0), data.index
        elif what_data == 'row':
            data = self.scaler.transform(data, dataset_name)
            data, new_dates = self.imputer.transform(data)
            return self.sorter.transform(data, dataset_name), new_dates
        elif what_data in {'row_all', 'row_dict'}:
            data = self.scaler.transform(data, dataset_name)
            data = concat_by_columns(data_row, data, name1='data_row', name2=f'data_{dataset_name}')
            data, new_dates = self.imputer.transform(data)
            return self.sorter.transform(data, dataset_name), new_dates
        elif what_data == 'row_dict_distributed':
            data = self.scaler.transform(data, dataset_name)
            data, new_dates = self.imputer.transform(data, data_dates=data_dates)
            return self.sorter.transform(data, dataset_name), new_dates
        elif what_data == 'cnn':
            data = self.scaler.transform(data, dataset_name)
            data, new_dates = self.imputer.transform(data, data_dates=data_dates)
            return data, new_dates

    def fit_pipe(self):
        if self.train:
            self.fit_row_data_pipe()
            self.fit_nwp_data_pipe()

    def fit_row_data_pipe(self):
        target = self.files_manager.check_if_exists_target()
        if target is None:
            raise ImportError(f'Cannot find target dataset')
        print(f"Fit {self.scale_target_method} scaler for target data")
        dataset_name = f'target_{self.scale_target_method}'
        self.scaler.fit(target, dataset_name)
        target = self.scaler.transform(target, dataset_name)
        target = target.dropna(axis=0)

        data_row = self.files_manager.check_if_exists_row_data()
        if data_row is None:
            raise ImportError(f'Cannot find row dataset')
        print(f"Fit {self.scale_row_method} scaler for row data")

        dataset_name = f'data_row_{self.scale_row_method}'
        self.scaler.fit(data_row, dataset_name)
        data_row= self.scaler.transform(data_row, dataset_name)
        print(f"Fit imputer for data row")
        self.imputer.fit(data_row)
        data_row, _ = self.imputer.transform(data_row)
        print(f"Fit sorter for data row")

        self.sorter.fit(data_row, target, dataset_name)
        self.save()

    def fit_nwp_data_pipe(self):
        target = self.files_manager.check_if_exists_target()
        target = self.scaler.transform(target, f'target_{self.scale_target_method}')
        target = target.dropna(axis=0)
        data_row = self.files_manager.check_if_exists_row_data()
        data_row = self.scaler.transform(data_row, f'data_row_{self.scale_row_method}')
        self.fit_merged_pipe()
        self.fit_compressed_pipe(data_row, target)

    def fit_merged_pipe(self):
        for merge in self.nwp_data_merge:
            data_merged = self.files_manager.check_if_exists_nwps_merged(merge)
            nwp_data_merged, nwp_metadata = data_merged['data'], data_merged['nwp_metadata']
            if nwp_data_merged is None:
                raise ImportError(f'Cannot find merged dataset with merge {merge}')
            groups = nwp_metadata['groups']
            for method in self.scale_nwp_method:
                if len(groups) == 0:
                    data = np.copy(nwp_data_merged)
                    dataset_name = f'{merge}_{method}'
                    print(f"Fit scaler for {dataset_name}")
                    self.scaler.fit(data, dataset_name)
                    data = self.scaler.transform(data, dataset_name)
                    print(f"Fit imputer for {dataset_name}")
                    self.imputer.fit(data, data_dates=nwp_metadata['dates'])

                else:
                    for group in groups:
                        group_name = '_'.join(group) if isinstance(group, tuple) else group
                        data = np.copy(nwp_data_merged[group_name])
                        dataset_name = f'{merge}_{group_name}_{method}'
                        print(f"Fit scaler for {dataset_name}")
                        self.scaler.fit(data, dataset_name)
                        data = self.scaler.transform(data, dataset_name)
                        print(f"Fit imputer for {dataset_name}")
                        self.imputer.fit(data, data_dates=nwp_metadata['dates'])

    def fit_compressed_pipe(self, data_row, target):
        for merge in self.nwp_data_merge:
            for compress in self.nwp_data_compress:
                data_compressed = self.files_manager.check_if_exists_nwps_compressed(merge, compress)
                nwp_compressed_all, nwp_compressed, nwp_compressed_distributed, metadata = \
                    data_compressed['data_compressed_all'], \
                    data_compressed['data_compressed'], \
                    data_compressed['data_compressed_distributed'], \
                    data_compressed['nwp_metadata']
                if nwp_compressed is None:
                    raise ImportError(f'Cannot find compressed dataset with merge {merge} and compress {compress}')
                groups = metadata['groups']
                for method in self.scale_nwp_method:
                    data = nwp_compressed_all.copy()
                    dataset_name = f'{merge}_{compress}_all_{method}'
                    print(f"Fit scaler for {dataset_name}")
                    self.scaler.fit(data, dataset_name)
                    data = self.scaler.transform(data, dataset_name)
                    data = concat_by_columns(data_row, data, name1='data_row', name2=f'data_{dataset_name}')
                    print(f"Fit imputer for {dataset_name}")
                    self.imputer.fit(data)
                    data, _ = self.imputer.transform(data)
                    print(f"Fit sorter for {dataset_name}")
                    self.sorter.fit(data, target, dataset_name)
                    if len(groups) == 0:
                        data = nwp_compressed.copy()
                        dataset_name = f'{merge}_{compress}_{method}'
                        print(f"Fit scaler for {dataset_name}")
                        self.scaler.fit(data, dataset_name)
                        data = self.scaler.transform(data, dataset_name)
                        data = concat_by_columns(data_row, data, name1='data_row', name2=f'data_{dataset_name}')
                        print(f"Fit imputer for {dataset_name}")
                        self.imputer.fit(data)
                        data, _ = self.imputer.transform(data)
                        print(f"Fit sorter for {dataset_name}")
                        self.sorter.fit(data, target, dataset_name)

                        data = np.copy(nwp_compressed_distributed)
                        dataset_name = f'{merge}_{compress}_distributed_{method}'
                        print(f"Fit scaler for {dataset_name}")
                        self.scaler.fit(data, dataset_name)
                        data = self.scaler.transform(data, dataset_name)
                        print(f"Fit imputer for {dataset_name}")
                        self.imputer.fit(data, data_dates=metadata['dates'])
                        data, _ = self.imputer.transform(data, data_dates=metadata['dates'])
                        print(f"Fit sorter for {dataset_name}")
                        self.sorter.fit(data, target, dataset_name)
                    else:
                        for group in groups:
                            group_name = '_'.join(group) if isinstance(group, tuple) else group
                            data = nwp_compressed[group_name].copy()
                            dataset_name = f'{merge}_{compress}_{group_name}_{method}'
                            print(f"Fit scaler for {dataset_name}")
                            self.scaler.fit(data, dataset_name)
                            data = self.scaler.transform(data, dataset_name)
                            data = concat_by_columns(data_row, data, name1='data_row', name2=f'data_{dataset_name}')
                            print(f"Fit imputer for {dataset_name}")
                            self.imputer.fit(data)
                            data, _ = self.imputer.transform(data)
                            print(f"Fit sorter for {dataset_name}")
                            self.sorter.fit(data, target, dataset_name)

                            data = np.copy(nwp_compressed_distributed[group_name])
                            dataset_name = f'{merge}_{compress}_distributed_{group_name}_{method}'
                            self.scaler.fit(data, dataset_name)
                            data = self.scaler.transform(data, dataset_name)
                            print(f"Fit imputer for {dataset_name}")
                            self.imputer.fit(data, data_dates=metadata['dates'])
                            data, _ = self.imputer.transform(data, data_dates=metadata['dates'])

    def save(self):
        self.scaler.save()
        self.imputer.save()
        self.sorter.save()
