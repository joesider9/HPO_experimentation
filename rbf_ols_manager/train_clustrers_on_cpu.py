import os
import gc
import glob
import time
import shutil
import joblib
import optuna

import numpy as np
import pandas as pd

from joblib import Parallel
from joblib import delayed
from optuna.samplers import TPESampler

from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.dataset_utils import sync_target_with_tensors
from eforecast.common_utils.train_utils import distance
from eforecast.common_utils.train_utils import find_free_cpus

from eforecast.dataset_creation.data_feeder import DataFeeder
from eforecast.shallow_models.shallow_model import ShallowModel

from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor


class Objective(object):
    def __init__(self, static_data, cluster_name, cluster_dir, method, n_jobs):
        self.static_data = static_data
        self.method = method
        self.cluster_dir = cluster_dir
        self.cluster_name = cluster_name
        self.n_jobs = n_jobs

        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.scale_nwp_methods = self.static_data['scale_nwp_method']
        self.data_structure = self.static_data['data_structure']
        self.feature_selection_methods = self.static_data['feature_selection_methods']
        self.data_feeder = DataFeeder(self.static_data, online=False, train=True)

    def load_data(self, merge, compress, scale_method, what_data, feature_selection_method=None):
        X, metadata = self.data_feeder.feed_inputs(merge=merge, compress=compress,
                                                   scale_nwp_method=scale_method,
                                                   what_data=what_data,
                                                   feature_selection_method=feature_selection_method,
                                                   cluster={'cluster_name': self.cluster_name,
                                                            'cluster_path': self.cluster_dir})
        y, _ = self.data_feeder.feed_target(inverse=False)
        if isinstance(X, pd.DataFrame):
            X, y = sync_datasets(X, y, name1='inputs', name2='target')
        elif isinstance(X, list):
            data_row = X[1]
            data = X[0]
            X, y = sync_target_with_tensors(target=y, data_tensor=data, dates_tensor=metadata['dates'],
                                            data_row=data_row)
        elif isinstance(X, dict):
            X, y = sync_target_with_tensors(target=y, data_tensor=X, dates_tensor=metadata['dates'])
        else:
            X, y = sync_target_with_tensors(target=y, data_tensor=X, dates_tensor=metadata['dates'])
        metadata['dates'] = X.index
        return X, y, metadata

    def __call__(self, trial):
        param_init = find_init_params(self.static_data, self.method)
        if len(self.nwp_data_merge) > 1:
            merge = trial.suggest_categorical('merge', self.nwp_data_merge)
        else:
            merge = list(self.nwp_data_merge)[0]
        if len(self.nwp_data_compress) > 1:
            compress = trial.suggest_categorical('compress', self.nwp_data_compress)
        else:
            compress = list(self.nwp_data_compress)[0]
        if len(self.feature_selection_methods) > 1:
            feature_selection_method = trial.suggest_categorical('feature_selection_method',
                                                                 self.feature_selection_methods)
        else:
            feature_selection_method = self.feature_selection_methods[0]
        data_structure = [data_struct for data_struct in self.data_structure
                          if data_struct in {'row', 'row_all'}]
        if len(data_structure) == 0:
            raise ValueError('Cannot find what_data structure to fit')
        elif len(data_structure) > 1:
            what_data = trial.suggest_categorical('what_data', data_structure)
        else:
            what_data = data_structure[0]

        if len(self.scale_nwp_methods) > 1:
            scale = trial.suggest_categorical('scale', self.scale_nwp_methods)
        else:
            scale = list(self.scale_nwp_methods)[0]

        X, y, metadata = self.load_data(merge, compress, scale, what_data,
                                        feature_selection_method=feature_selection_method)

        cv_masks = joblib.load(os.path.join(self.cluster_dir, 'cv_mask.pickle'))

        experiment_params = {'method': self.method,
                             'name': self.cluster_name,
                             'merge': merge,
                             'compress': compress,
                             'what_data': what_data,
                             'feature_selection_method': feature_selection_method,
                             'scale_nwp_method': scale,
                             'groups': metadata['groups']}
        for param, value in self.static_data[self.method].items():
            if isinstance(value, set):
                if param in param_init.keys():
                    value.add(param_init[param])
                if len(value) > 1:
                    v = trial.suggest_categorical(param, list(value))
                else:
                    v = list(value)[0]
            elif isinstance(value, list):
                if len(value) > 1:
                    if param in param_init.keys():
                        if param_init[param] < value[0]:
                            value[0] = param_init[param]
                        if param_init[param] > value[1]:
                            value[1] = param_init[param]
                    if isinstance(value[0], int):
                        v = trial.suggest_int(param, value[0], value[-1])
                    else:
                        v = trial.suggest_float(param, value[0], value[-1])
                else:
                    v = value[0]
            else:
                v = value
            experiment_params[param] = v
        if 'bootstrap_type' in experiment_params.keys():
            if experiment_params["bootstrap_type"] == "Bayesian":
                experiment_params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
            elif experiment_params["bootstrap_type"] == "Bernoulli":
                experiment_params["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

        path_weights = os.path.join(self.cluster_dir, self.method, f'test_{trial.number}')
        if not os.path.exists(path_weights):
            os.makedirs(path_weights)
        else:
            shutil.rmtree(path_weights)
            os.makedirs(path_weights)
        model = ShallowModel(self.static_data, path_weights, params=experiment_params, n_jobs=self.n_jobs)
        acc = model.fit(X, y, cv_masks, metadata)
        trial.set_user_attr('best_mae_test', model.best_mae_test)

        del model
        gc.collect()

        if isinstance(acc, np.ndarray):
            acc = float(np.mean(acc))

        return acc


def get_param_names(static_data, method):
    param_names = []
    for param, value in static_data[method].items():
        if isinstance(value, set):
            if len(value) > 1:
                param_names.append(param)
        elif isinstance(value, list):
            if len(value) > 1:
                param_names.append(param)

    if 'bootstrap_type' in static_data[method].items():
        if static_data[method]["bootstrap_type"] == "Bayesian":
            param_names.append("bagging_temperature")
        elif static_data[method]["bootstrap_type"] == "Bernoulli":
            param_names.append("subsample")
    return param_names


def find_init_params(static_data, method):
    if method == 'RF':
        model = RandomForestRegressor()
    elif method == 'CatBoost':
        model = CatBoostRegressor()
    elif method == 'lasso':
        if static_data['horizon_type'] == 'multi-output':
            model = MultiTaskLassoCV(max_iter=150000)
        else:
            model = LassoCV(max_iter=150000)
    else:
        raise ValueError(f'Unknown method {method} for shallow models')
    param_names = get_param_names(static_data, method)
    return {param: value for param, value in model.get_params().items() if param in param_names}


def optuna_thread(project_id, static_data, cluster_name, cluster_dir, method):
    path_group = static_data['path_group']
    n_jobs = find_free_cpus(path_group)
    print(f'CPU methods starts running on {n_jobs} cpus')
    print(f'{method} Model of {cluster_name} of {project_id} is starts.....')

    if not os.path.exists(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv')):
        if not os.path.exists(os.path.join(cluster_dir, f'study_{method}.pickle')):
            study = optuna.create_study(sampler=TPESampler(seed=42, consider_magic_clip=True, n_startup_trials=4,
                                                           n_ei_candidates=4))
            joblib.dump(study, os.path.join(cluster_dir, f'study_{method}.pickle'))
        else:
            try:
                study = joblib.load(os.path.join(cluster_dir, f'study_{method}.pickle'))
            except:
                study = optuna.create_study(sampler=TPESampler(seed=42, consider_magic_clip=True, n_startup_trials=4,
                                                               n_ei_candidates=4))

        study.enqueue_trial(find_init_params(static_data, method))
        study.optimize(Objective(static_data, cluster_name, cluster_dir, method, n_jobs),
                       n_trials=static_data[method]['n_trials'],
                       gc_after_trial=True)
        results = study.trials_dataframe().sort_values(by='value')
        model_dir = os.path.join(cluster_dir, method)
        test_dir = os.path.join(cluster_dir, method, f'test_{study.best_trial.number}')
        for filename in glob.glob(os.path.join(test_dir, '*.*')):
            print(filename)
            shutil.copy(filename, model_dir)
        results.to_csv(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv'))
        for number in range(len(study.get_trials())):
            test_dir = os.path.join(cluster_dir, method, f'test_{number}')
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)


def CPU_thread(static_data, method):
    clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
    for cluster_name, cluster_dir in clusters.items():
        optuna_thread(static_data['_id'], static_data, cluster_name, cluster_dir, method)
    # n_workers = int(n_jobs)
    # res = Parallel(n_jobs=n_workers)(delayed(optuna_thread)(static_data['_id'], static_data, cluster_name, cluster_dir,
    #                                                         method)
    #                                  for cluster_name, cluster_dir in clusters.items())


def train_clusters_on_cpus(static_data):
    time.sleep(10)
    gpu_methods = ['CNN', 'LSTM', 'RBFNN', 'MLP', 'RBF-CNN']
    cpu_methods = ['CatBoost', 'RF', 'lasso', 'RBFols', 'GA_RBFols']
    methods = []
    for method, values in static_data['project_methods'].items():
        if values and method in cpu_methods:
            methods.append(method)
        else:
            if method not in gpu_methods:
                raise ValueError(f'Regression method {method} unknown')

    for method in methods:
        CPU_thread(static_data, method)
        print(f'Training of {method} ends successfully')
