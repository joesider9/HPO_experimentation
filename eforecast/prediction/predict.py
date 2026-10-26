import os
import joblib

import numpy as np
import pandas as pd

from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.dataset_utils import sync_target_with_tensors
from eforecast.dataset_creation.data_feeder import DataFeeder
from eforecast.dataset_creation.files_manager import FilesManager

from eforecast.clustering.clustering_manager import ClusterOrganizer
from eforecast.shallow_models.shallow_model import ShallowModel
from eforecast.deep_models.tf_1x.network import DeepNetwork
from eforecast.deep_models.tf_1x.global_network import DistributedDeepNetwork

gpu_methods = ['CNN', 'LSTM', 'RBFNN', 'MLP', 'RBF-CNN']
cpu_methods = ['CatBoost', 'RF', 'lasso', 'RBFols', 'GA_RBFols']


class Predictor:
    def __init__(self, static_data, online=False, train=False):
        self.static_data = static_data
        self.online = online
        self.train = train
        self.is_Fuzzy = self.static_data['is_Fuzzy']
        self.is_Global = self.static_data['is_Global']
        self.predictions = dict()
        self.methods = [method for method, values in static_data['project_methods'].items()
                        if values and (method in cpu_methods or method in gpu_methods)]
        self.regressors = []
        if self.is_Fuzzy:
            self.cluster_dates = dict()
            self.clusterer = ClusterOrganizer(static_data, is_online=online, train=train)
            self.predictions['clusterer'] = dict()
            self.clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
            self.predictions['clusters'] = dict()
            for m in self.clusterer.methods:
                if m in self.static_data['clustering']['prediction_for_method'] \
                        and self.static_data['horizon_type'] != 'multi-output':
                    self.regressors.append({'method': m})
                if m == self.clusterer.make_clusters_for_method or self.clusterer.make_clusters_for_method == 'both':
                    self.predictions['clusters'][m] = dict()
                    _, self.cluster_dates[m] = self.clusterer.compute_activations(m)
            for cluster_name, cluster_path in self.clusters.items():
                clusterer_method = [m for m in self.clusterer.methods if m in cluster_name]
                if len(clusterer_method) == 0:
                    raise ValueError(f'Cannot correspond the cluster {cluster_name} with a known clusterer method')
                clusterer_method = clusterer_method[0]
                self.predictions['clusters'][clusterer_method][cluster_name] = dict()
                for method in self.methods:
                    file_regressor = os.path.join(cluster_path, method, 'net_weights.pickle')
                    if not os.path.exists(file_regressor):
                        raise ImportError(f"Cannot find model {method} of cluster {cluster_name}")
                    self.regressors.append({'file': file_regressor,
                                            'method': method,
                                            'clusterer_method': clusterer_method,
                                            'cluster_name': cluster_name,
                                            'cluster_path': cluster_path})
        if self.is_Global:
            self.predictions['distributed'] = dict()
            global_path = os.path.join(static_data['path_model'], 'Distributed')
            for path in os.listdir(global_path):
                regressor_path = os.path.join(global_path, path)
                if os.path.isdir(regressor_path):
                    file_regressor = os.path.join(regressor_path, 'distributed_model.pickle')
                    if not os.path.exists(file_regressor):
                        raise ImportError(f"Cannot find distributed model {path}")
                    self.regressors.append({'file': file_regressor,
                                            'method': 'distributed',
                                            'model_name': path,
                                            'model_path': regressor_path})
        self.data_feeder = DataFeeder(static_data, online=self.online, train=self.train)

    def init_method(self, regressor):
        if regressor['method'] == 'distributed':
            return DistributedDeepNetwork(self.static_data, regressor['model_path'], is_online=self.online,
                                          train=self.train)
        elif regressor['method'] in gpu_methods:
            return DeepNetwork(self.static_data, os.path.join(regressor['cluster_path'], regressor['method']))

        elif regressor['method'] in cpu_methods:
            return ShallowModel(self.static_data, os.path.join(regressor['cluster_path'], regressor['method']))
        elif regressor['method'] in self.clusterer.methods:
            return regressor['method']
        else:
            raise ValueError(f"Unknown method for prediction {regressor['method']}")

    def predict_cluster(self, regressor):
        model = self.init_method(regressor)
        params = model.params
        X, metadata = self.data_feeder.feed_inputs(merge=params['merge'], compress=params['compress'],
                                                   scale_nwp_method=params['scale_nwp_method'],
                                                   what_data=params['what_data'],
                                                   feature_selection_method=params['feature_selection_method'],
                                                   cluster={'cluster_name': regressor['cluster_name'],
                                                            'cluster_path': regressor['cluster_path']})
        name = '_'.join(regressor['cluster_name'].split('_')[1:])
        cluster_dates = self.cluster_dates[regressor['clusterer_method']][name]
        return model.predict(X, metadata, cluster_dates=cluster_dates)

    def predict_global(self, regressor):
        model = self.init_method(regressor)
        params = model.params
        X, metadata = self.data_feeder.feed_inputs(merge=params['merge'], compress=params['compress'],
                                                   scale_nwp_method=params['scale_nwp_method'],
                                                   what_data=params['what_data'],
                                                   feature_selection_method=params['feature_selection_method'])
        return model.predict(X, metadata)

    def predict_clusterer(self):
        pred, _ = self.clusterer.compute_activations('RBF')
        return pred

    def predict_regressors(self):
        for regressor in self.regressors:
            if 'clusterer_method' in regressor.keys():
                clusterer_method = regressor['clusterer_method']
                cluster_name = regressor['cluster_name']
                method = regressor['method']
                self.predictions['clusters'][clusterer_method][cluster_name][method] = self.predict_cluster(regressor)
            elif regressor['method'] == 'distributed':
                self.predictions['distributed'][regressor['model_name']] = self.predict_global(regressor)
            else:
                self.predictions['clusterer'][regressor['method']] = self.predict_clusterer()
        if self.train:
            joblib.dump(self.predictions, os.path.join(self.static_data['path_data'],
                                                       'predictions_regressors_train.pickle'))
