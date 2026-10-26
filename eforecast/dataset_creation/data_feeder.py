import os
from eforecast.common_utils.dataset_utils import sync_data_row_with_tensors

from eforecast.dataset_creation.files_manager import FilesManager
from eforecast.data_preprocessing.data_pipeline import DataPipeline
from eforecast.feature_selection.feature_selection_transform import FeatureSelector


class DataFeeder:
    def __init__(self, static_data, online=False, train=False):
        self.static_data = static_data
        self.online = online
        self.train = train
        self.scale_target_method = self.static_data['scale_target_method']
        self.scale_row_method = self.static_data['scale_row_method']
        self.scale_nwp_method = self.static_data['scale_nwp_method']
        self.files_manager = FilesManager(static_data, is_online=online, train=self.train)
        self.pipeline = DataPipeline(self.static_data, online=online, train=self.train)
        self.feature_selector = FeatureSelector(self.static_data, online=online, train=self.train)

    def load_merged(self, merge):
        data_merged = self.files_manager.check_if_exists_nwps_merged(merge)
        if data_merged['data'] is None:
            raise ImportError(f'Cannot find merged dataset with merge {merge}')
        return data_merged

    def load_compressed(self, merge, compress):
        data_compressed = self.files_manager.check_if_exists_nwps_compressed(merge, compress)
        if data_compressed['data_compressed_all'] is None:
            raise ImportError(f'Cannot find compressed dataset with merge {merge} and compress {compress}')
        return data_compressed

    def load_row_data(self):
        data_row = self.files_manager.check_if_exists_row_data()
        if data_row is None:
            raise ImportError(f'Cannot find row dataset')
        return data_row

    def load_target(self, offline=False):
        target = None
        if not self.online or offline:
            target = self.files_manager.check_if_exists_target()
            if target is None:
                raise ImportError(f'Cannot find target dataset')
        else:
            raise ImportError(f'Cannot import target dataset when is online True and offline False')
        return target

    def feed_inputs(self, merge=None, compress=None, scale_nwp_method=None, what_data=None,
                    feature_selection_method=None, cluster=None):
        if cluster is None:
            cluster = dict()
            cluster['cluster_name'] = 'Distributed'
            cluster['cluster_path'] = os.path.join(self.static_data['path_model'], 'Distributed')
        if merge is None or scale_nwp_method is None or what_data is None:
            raise ValueError('Some parameters are None. they all should be defined')
        if merge not in {'all', 'by_area', 'by_area_variable', 'by_variable'}:
            raise ValueError(f"Merge method should be one of 'all', 'by_area', 'by_area_variable', 'by_variable'."
                             f" Not {merge}")
        if compress not in {None, 'dense', 'semi_full', 'full', 'load'}:
            raise ValueError(f"Compress method should be one of dense or semi_full or full or load. Not {compress}")
        if scale_nwp_method not in {'minmax', 'standard'}:
            raise ValueError(f"Scale method should be one of 'minmax', 'standard'. Not {scale_nwp_method}")
        if what_data not in {'row', 'row_all', 'row_dict', 'row_dict_distributed', 'cnn', 'lstm'}:
            raise ValueError(f"Merge method should be one of 'row_all', 'row_dict', 'row_dict_distributed', 'cnn', "
                             f"'lstm'. Not {what_data}")
        data_row = self.load_row_data()
        dates = None
        if what_data in {'row'}:
            dataset_name = f'data_row_{self.scale_row_method}'
            data, dates = self.pipeline.transform_pipe(data_row, dataset_name, what_data=what_data)
            return data, dates
        elif what_data in {'row_all', 'row_dict', 'row_dict_distributed'}:
            data_compressed = self.load_compressed(merge, compress)
            if what_data == 'row_all':
                dataset_name = f'{merge}_{compress}_all_{scale_nwp_method}'
                print(f'Feed row_all data with {merge} and {compress}_all')
                data, dates = self.pipeline.transform_pipe(data_compressed['data_compressed_all'],
                                                           dataset_name, what_data=what_data, data_row=data_row,
                                                           data_dates=data_compressed['nwp_metadata']['dates'])
                data_compressed['nwp_metadata']['dates'] = dates
                data = self.feature_selector.transform(cluster, data, feature_selection_method, merge, compress,
                                                       scale_nwp_method, what_data,
                                                       data_compressed['nwp_metadata']['groups'])
                return data, data_compressed['nwp_metadata']
            elif what_data == 'row_dict':
                groups = data_compressed['nwp_metadata']['groups']
                if len(groups) == 0:
                    print(f'Feed row_dict data with {merge} and {compress}')
                    dataset_name = f'{merge}_{compress}_{scale_nwp_method}'
                    data, dates = self.pipeline.transform_pipe(data_compressed['data_compressed'],
                                                               dataset_name, what_data=what_data, data_row=data_row,
                                                               data_dates=data_compressed['nwp_metadata']['dates'])
                    data_compressed['nwp_metadata']['dates'] = dates
                    data = self.feature_selector.transform(cluster, data, feature_selection_method, merge,
                                                           compress,
                                                           scale_nwp_method, what_data,
                                                           data_compressed['nwp_metadata']['groups'])
                    return data, data_compressed['nwp_metadata']
                else:
                    data = dict()
                    for group in groups:
                        print(f'Feed row_dict data with {merge} and {compress} of {group}')
                        group_name = '_'.join(group) if isinstance(group, tuple) else group
                        dataset_name = f'{merge}_{compress}_{group_name}_{scale_nwp_method}'
                        data[group_name], dates = self.pipeline.transform_pipe(
                            data_compressed['data_compressed'][group_name], dataset_name,
                            what_data=what_data, data_row=data_row,
                            data_dates=data_compressed['nwp_metadata']['dates'])
                    data_compressed['nwp_metadata']['dates'] = dates
                    data = self.feature_selector.transform(cluster, data,
                                                           feature_selection_method,
                                                           merge, compress,
                                                           scale_nwp_method, what_data,
                                                           data_compressed['nwp_metadata']['groups'])
                    return data, data_compressed['nwp_metadata']

            elif what_data == 'row_dict_distributed':
                groups = data_compressed['nwp_metadata']['groups']
                if len(groups) == 0:
                    print(f'Feed row_dict data with {merge} and {compress}_distributed')
                    dataset_name = f'{merge}_{compress}_distributed_{scale_nwp_method}'
                    data, dates = self.pipeline.transform_pipe(data_compressed['data_compressed_distributed'],
                                                               dataset_name, what_data=what_data, data_row=data_row,
                                                               data_dates=data_compressed['nwp_metadata']['dates'])
                    data_compressed['nwp_metadata']['dates'] = dates
                    return data, data_compressed['nwp_metadata']
                else:
                    for group in groups:
                        print(f'Feed row_dict data with {merge} and {compress}_distributed of {group}')
                        group_name = '_'.join(group) if isinstance(group, tuple) else group
                        dataset_name = f'{merge}_{compress}_distributed_{group_name}_{scale_nwp_method}'
                        data_compressed['data_compressed_distributed'][group_name], dates = \
                            self.pipeline.transform_pipe(data_compressed['data_compressed_distributed'][group_name],
                                                         dataset_name,
                                                         what_data=what_data,
                                                         data_row=data_row,
                                                         data_dates=data_compressed['nwp_metadata']['dates'])
                    data_compressed['nwp_metadata']['dates'] = dates
                    return data_compressed['data_compressed_distributed'], data_compressed['nwp_metadata']
            else:
                raise ValueError(f"Unknown {what_data}, what_data should be 'row_all', 'row_dict', "
                                 f"'row_dict_distributed', 'cnn', 'lstm'")
        elif what_data == 'cnn':
            print(f'Feed cnn data with {merge}')
            data_merged = self.load_merged(merge)
            nwp_data_merged, nwp_metadata = data_merged['data'], data_merged['nwp_metadata']
            groups = nwp_metadata['groups']
            if len(groups) == 0:
                dataset_name = f'{merge}_{scale_nwp_method}'
                nwp_data_merged, dates = self.pipeline.transform_pipe(nwp_data_merged,
                                                                      dataset_name, what_data=what_data,
                                                                      data_row=data_row,
                                                                      data_dates=nwp_metadata['dates'])
            else:
                for group in groups:
                    print(f'Feed row_dict data with {merge} of {group}')
                    group_name = '_'.join(group) if isinstance(group, tuple) else group
                    dataset_name = f'{merge}_{group_name}_{scale_nwp_method}'
                    nwp_data_merged[group_name], dates = self.pipeline.transform_pipe(nwp_data_merged[group_name],
                                                                                      dataset_name,
                                                                                      what_data=what_data,
                                                                                      data_row=data_row,
                                                                                      data_dates=nwp_metadata['dates'])

            dataset_name = f'data_row_{self.scale_row_method}'
            data_row, dates_row = self.pipeline.transform_pipe(data_row, dataset_name, what_data=what_data)
            nwp_data_merged, data_row, dates = sync_data_row_with_tensors(data_tensor=nwp_data_merged,
                                                                          dates_tensor=dates,
                                                                          data_row=data_row,
                                                                          dates_row=dates_row)
            nwp_metadata['dates'] = dates
            return [nwp_data_merged, data_row], nwp_metadata
        elif what_data == 'lstm':
            print(f'Feed lstm data with {merge} and {compress}')
            raise NotImplementedError(f'Concatenate {what_data} data for lstm can not feed. '
                                      f'It is not implemented yet')

    def feed_target(self, inverse=False):
        y = self.load_target()
        data, dates = self.pipeline.transform_pipe(y, f'target_{self.scale_target_method}', what_data='target',
                                                   inverse=inverse)
        return data, dates
