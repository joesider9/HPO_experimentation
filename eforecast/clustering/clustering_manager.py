import os
import shutil

import joblib
import numpy as np
import pandas as pd

from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.clustering_utils import check_if_all_nans
from eforecast.clustering.tf_rbf_clusterer import TfRBFClusterer
from eforecast.clustering.ga_fuzzy_clusterer import GAFuzzyClusterer
from eforecast.clustering.hebo_ga_fuzzy_clusterer import HeboFuzzyClusterer

from eforecast.dataset_creation.data_feeder import DataFeeder
from eforecast.data_preprocessing.data_split import Splitter
from eforecast.dataset_creation.files_manager import FilesManager


class ClusterOrganizer:

    def __init__(self, static_data, is_online=False, train=False, refit=False):
        self.is_online = is_online
        self.train = train
        self.refit = refit
        self.sampled_data = None
        self.static_data = static_data
        self.path_model = self.static_data['path_model']
        self.is_Fuzzy = self.static_data['is_Fuzzy']
        self.thres_act = self.static_data['clustering']['thres_act']
        self.scale_method = self.static_data['clustering']['data_type']['scaling']
        self.merge = self.static_data['clustering']['data_type']['merge']
        self.compress = self.static_data['clustering']['data_type']['compress']
        self.methods = self.static_data['clustering']['methods']
        self.make_clusters_for_method = self.static_data['clustering']['clusters_for_method']
        self.data_feeder = DataFeeder(static_data, online=is_online, train=train)
        self.file_manager = FilesManager(static_data, is_online=is_online, train=train)

    def feed_data(self):
        print('Read data for Clustering....')
        X, metadata = self.data_feeder.feed_inputs(merge=self.merge, compress=self.compress,
                                                   scale_nwp_method=self.scale_method,
                                                   what_data='row_all')
        if self.train:
            y, _ = self.data_feeder.feed_target()
            X, y = sync_datasets(X, y, name1='inputs_for_fuzzy', name2='target_for_fuzzy')
            metadata['dates'] = X.index
            return X, y, metadata
        else:
            return X, metadata

    @staticmethod
    def create_cluster_folders(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.makedirs(path)

    def fit(self):
        if self.is_Fuzzy:
            x, y, metadata = self.feed_data()
            cv_mask = self.file_manager.check_if_exists_cv_data()
            for method in self.methods:
                if method == 'RBF':
                    clusterer = TfRBFClusterer(self.static_data, refit=self.refit)
                elif method == 'GA':
                    clusterer = GAFuzzyClusterer(self.static_data, refit=self.refit)
                elif method == 'HEBO':
                    clusterer = HeboFuzzyClusterer(self.static_data, refit=self.refit)
                else:
                    raise NotImplementedError(
                        f"Clustering method {self.static_data['clustering']['method']} not implemented")
                if not clusterer.is_trained or self.refit:
                    clusterer.run(x, y, cv_mask, metadata)

    def cluster_input_dates(self, clusterer, x, metadata):
        clustered_dates = dict()
        activations = clusterer.compute_activations(x, metadata)
        activations = check_if_all_nans(activations, self.thres_act)
        for cluster in clusterer.rule_names:
            indices = np.where(activations[cluster] >= self.thres_act)[0]
            clustered_dates[cluster] = activations.index[indices]
        return clustered_dates, activations

    def predict(self, method):
        if method not in self.methods:
            raise ValueError(f'{method} is not in clustering methods')
        if method == 'RBF':
            clusterer = TfRBFClusterer(self.static_data)
        elif method == 'GA':
            clusterer = GAFuzzyClusterer(self.static_data)
        elif method == 'HEBO':
            clusterer = HeboFuzzyClusterer(self.static_data)
        else:
            raise NotImplementedError(
                f"Clustering method {self.static_data['clustering']['method']} not implemented")
        if self.train:
            x, y, metadata = self.feed_data()
        else:
            x, metadata = self.feed_data()
        predictions, activations = clusterer.compute_activations(x, metadata, with_predictions=True)
        return predictions, activations

    def compute_activations(self, method):
        if method not in self.methods:
            raise ValueError(f'{method} is not in clustering methods')
        if method == 'RBF':
            clusterer = TfRBFClusterer(self.static_data)
        elif method == 'GA':
            clusterer = GAFuzzyClusterer(self.static_data)
        elif method == 'HEBO':
            clusterer = HeboFuzzyClusterer(self.static_data)
        else:
            raise NotImplementedError(
                f"Clustering method {self.static_data['clustering']['method']} not implemented")
        if self.train:
            x, y, metadata = self.feed_data()
        else:
            x, metadata = self.feed_data()
        clustered_dates, activations = self.cluster_input_dates(clusterer, x, metadata)
        return activations, clustered_dates

    def create_clusters_and_cvs(self):
        if self.is_Fuzzy:
            clusters = dict()
            x, y, metadata = self.feed_data()
            for method in self.methods:
                if method == 'RBF':
                    clusterer = TfRBFClusterer(self.static_data)
                elif method == 'GA':
                    clusterer = GAFuzzyClusterer(self.static_data)
                elif method == 'HEBO':
                    clusterer = GAFuzzyClusterer(self.static_data)
                else:
                    raise NotImplementedError(
                        f"Clustering method {self.static_data['clustering']['method']} not implemented")
                if method == self.make_clusters_for_method or self.make_clusters_for_method == 'both':
                    clustered_dates, activations = self.cluster_input_dates(clusterer, x, metadata)
                    splitter = Splitter(self.static_data, is_online=self.is_online, train=self.train)
                    cv_mask = splitter.split_cluster_data(activations)
                    for cluster in clusterer.rule_names:
                        path = os.path.join(clusterer.path_fuzzy, cluster)
                        if os.path.exists(os.path.join(path, 'cv_mask.pickle')) and not self.refit:
                            return
                        self.create_cluster_folders(path)
                        mask_train = clustered_dates[cluster].intersection(cv_mask[0])
                        mask_val = clustered_dates[cluster].intersection(cv_mask[1])
                        mask_test = clustered_dates[cluster].intersection(cv_mask[2])
                        joblib.dump([mask_train, mask_val, mask_test], os.path.join(path, 'cv_mask.pickle'))
                        clusters[f'{method}_{cluster}'] = path
            joblib.dump(clusters, os.path.join(self.path_model, 'clusters.pickle'))
