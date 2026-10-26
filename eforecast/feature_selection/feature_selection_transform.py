import os
import joblib

import numpy as np
import pandas as pd

from eforecast.dataset_creation.files_manager import FilesManager


class FeatureSelector:
    def __init__(self, static_data, recreate=False, online=False, train=False):
        self.feature_selectors = dict()
        self.online = online
        self.train = train
        self.static_data = static_data
        self.is_Fuzzy = self.static_data['is_Fuzzy']
        if self.is_Fuzzy:
            if os.path.exists(os.path.join(static_data['path_model'], 'clusters.pickle')):
                self.clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
            else:
                self.clusters = dict()

        self.is_Global = self.static_data['is_Global']
        if self.is_Global:
            self.clusters.update({'global': ''})
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.scale_nwp_method = self.static_data['scale_nwp_method']
        self.data_structure = self.static_data['data_structure']
        self.feature_selection_methods = self.static_data['feature_selection_methods']
        self.files_manager = FilesManager(static_data, is_online=online)

    def transform(self, cluster, data, method, merge, compress, scale_method, what_data, groups=None):
        cluster_path = cluster['cluster_path']
        cluster_name = cluster['cluster_name']
        filename = os.path.join(cluster_path, 'feature_selectors.pickle')
        if os.path.exists(filename):
            self.feature_selectors.update(joblib.load(filename))
        if method is not None:
            if len(groups) == 0 or what_data == 'row_all':
                dataset_name = f'{merge}_{compress}_{scale_method}_{what_data}'
                feature_selector = self.feature_selectors[f'feature_selector_{cluster_name}_{method}_{dataset_name}']
                data_new = data[feature_selector['names']]
            else:
                data_new = dict()
                for group in groups:
                    group_name = '_'.join(group) if isinstance(group, tuple) else group
                    dataset_name = f'{merge}_{compress}_{scale_method}_{group_name}_{what_data}'
                    feature_selector = self.feature_selectors[f'feature_selector_{cluster_name}_{method}_{dataset_name}']
                    data_new[group_name] = data[group_name][feature_selector['names']]
        else:
            data_new = data
        return data_new
