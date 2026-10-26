import os
import gc
import glob
import time
import shutil
import joblib
from joblib import Parallel
from joblib import delayed

import numpy as np
import pandas as pd
import multiprocessing as mp

from hebo_trial.optimizer_mace import MACEBO


from numpy.random import SeedSequence, default_rng

from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.dataset_utils import sync_target_with_tensors
from eforecast.common_utils.train_utils import find_free_cpus

from eforecast.dataset_creation.data_feeder import DataFeeder
from eforecast.shallow_models.shallow_model import ShallowModel

from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor


class Objective(object):
    def __init__(self, static_data, cluster_name, cluster_dir, method, n_jobs=1, refit=False):
        self.static_data = static_data
        self.method = method
        self.cluster_dir = cluster_dir
        self.cluster_name = cluster_name
        self.n_jobs = n_jobs

        self.warming = self.static_data[method]['warming_iterations']
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.scale_nwp_methods = self.static_data['scale_nwp_method']
        self.data_structure = self.static_data['data_structure']
        self.feature_selection_methods = self.static_data['feature_selection_methods']
        self.data_feeder = DataFeeder(self.static_data, online=False, train=True)

        self.initialize(refit)

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

    def load_datasets(self):
        data_structure = [data_struct for data_struct in self.data_structure
                          if data_struct in {'row', 'row_all'}]
        self.datasets = dict()
        for merge in self.nwp_data_merge:
            for compress in self.nwp_data_compress:
                for what_data in data_structure:
                    for scale in self.scale_nwp_methods:
                        for feature_selection_method in self.feature_selection_methods:
                            name_dataset = f'{merge}_{compress}_{what_data}_{scale}_{feature_selection_method}'
                            self.datasets[name_dataset] = self.load_data(merge, compress, scale, what_data,
                                                                         feature_selection_method=
                                                                         feature_selection_method)
        self.cv_masks = joblib.load(os.path.join(self.cluster_dir, 'cv_mask.pickle'))

    def get_param(self, name, type, dtype, space=None, range=None, values=None):
        return {name: {'type': type,
                       'dtype': dtype,
                       'space': space,
                       'range': range,
                       'values': values}}

    def define_scape_rbf(self, n_features, mean_X):
        self.space = dict()
        self.fix_params = dict()
        range_width = [0.5 * mean_X, 18 * mean_X]
        if 'GA' in self.method:
            for i in range(n_features):
                self.space.update(self.get_param(f'width_{i}', 'real', 'float', values=range_width))
        else:
            self.space.update(self.get_param(f'width', 'real', 'float', values=range_width))
        self.space.update(self.get_param('keep', 'int', 'int', values=[2, 6]))

    def define_space(self):
        self.param_init = find_init_params(self.static_data, self.method)
        self.param_init_nans_values = dict()
        self.space = dict()
        self.fix_params = dict()
        if len(self.nwp_data_merge) > 1:
            self.space.update(self.get_param('merge', 'cat', 'string', values=self.nwp_data_merge))
        else:
            self.fix_params['merge'] = list(self.nwp_data_merge)[0]
        self.param_init['merge'] = list(self.nwp_data_merge)[0]
        if len(self.nwp_data_compress) > 1:
            self.space.update(self.get_param('compress', 'cat', 'string', values=self.nwp_data_compress))
        else:
            self.fix_params['compress'] = list(self.nwp_data_compress)[0]
        self.param_init['compress'] = list(self.nwp_data_compress)[0]
        if len(self.feature_selection_methods) > 1:
            self.space.update(
                self.get_param('feature_selection_method', 'cat', 'string', values=self.feature_selection_methods))
        else:
            self.fix_params['feature_selection_method'] = self.feature_selection_methods[0]
        self.param_init['feature_selection_method'] = list(self.feature_selection_methods)[0]

        data_structure = [data_struct for data_struct in self.data_structure
                          if data_struct in {'row', 'row_all'}]
        if len(data_structure) == 0:
            raise ValueError('Cannot find what_data structure to fit')
        elif len(data_structure) > 1:
            self.space.update(self.get_param('what_data', 'cat', 'string', values=data_structure))
        else:
            self.fix_params['what_data'] = data_structure[0]
        self.param_init['what_data'] = data_structure[0]

        if len(self.scale_nwp_methods) > 1:
            self.space.update(self.get_param('scale', 'cat', 'string', values=self.scale_nwp_methods))
        else:
            self.fix_params['scale'] = list(self.scale_nwp_methods)[0]
        self.param_init['scale'] = list(self.scale_nwp_methods)[0]

        for param, value in self.static_data[self.method].items():
            if isinstance(value, set):
                if param in self.param_init.keys():
                    if self.param_init[param] is not None:
                        value.add(self.param_init[param])
                    else:
                        value.add(-1)
                        self.param_init_nans_values[param] = -1
                else:
                    self.param_init[param] = list(value)[0]
                if len(value) > 1:
                    if isinstance(list(value)[0], str):
                        self.space.update(self.get_param(param, 'cat', 'string', values=list(value)))
                    elif isinstance(list(value)[0], int):
                        self.space.update(self.get_param(param, 'cat', 'int', values=list(value)))
                    else:
                        self.space.update(self.get_param(param, 'cat', 'float', values=list(value)))
                else:
                    self.fix_params[param] = list(value)[0]
            elif isinstance(value, list):
                if len(value) > 1:
                    if param in self.param_init.keys():
                        if self.param_init[param] is None:
                            if isinstance(value[0], int):
                                self.param_init[param] = value[0] + 1
                                self.param_init_nans_values[param] = value[0] + 1
                            else:
                                self.param_init[param] = value[0] + 0.01
                                self.param_init_nans_values[param] = value[0] + 0.01
                        if self.param_init[param] < value[0]:
                            value[0] = self.param_init[param]
                        if self.param_init[param] > value[1]:
                            value[1] = self.param_init[param]
                    else:
                        self.param_init[param] = value[0]
                    if isinstance(value[0], int):
                        self.space.update(self.get_param(param, 'int', 'int', range=value))
                    else:
                        self.space.update(self.get_param(param, 'real', 'float', range=value))
                else:
                    self.fix_params[param] = value[0]
            else:
                self.fix_params[param] = value
        self.param_names = []
        for param_name, param_attr in self.space.items():
            self.param_names.append(param_name)

    def initialize(self, refit=False):
        self.load_datasets()
        self.define_space()

    def fit_trial(self, trial_number, random_state, trials, init_params=None):
        # TODO to check if param in self.param_init_nans_values and its value is equal with self.param_init_nans_values[param]

        optimizer = MACEBO(self.space)
        if len(trials) > 0:
            y_trial = np.array([trial['value'] for trial in trials]).reshape(-1, 1)
            X_trial = []
            for trial in trials:
                param_dict = dict()
                for key in self.param_names:
                    param_dict[key] = trial[key]
                X_trial.append(param_dict)

            X_trial = pd.DataFrame(X_trial)
            optimizer.observe(X_trial, y_trial)
        trial = optimizer.suggest(n_suggestions=1, random_state=random_state, warming=self.warming)[0]
        merge = trial['merge'] if 'merge' in trial.keys() else self.fix_params['merge']
        compress = trial['compress'] if 'compress' in trial.keys() else self.fix_params['compress']
        what_data = trial['what_data'] if 'what_data' in trial.keys() else self.fix_params['what_data']
        scale = trial['scale'] if 'scale' in trial.keys() else self.fix_params['scale']
        feature_selection_method = trial['feature_selection_method'] if 'feature_selection_method' in trial.keys() \
            else self.fix_params['feature_selection_method']
        name_dataset = f'{merge}_{compress}_{what_data}_{scale}_{feature_selection_method}'
        X, y, metadata = self.datasets[name_dataset]

        experiment_params = {'trial_number': trial_number,
                             'method': self.method,
                             'name': self.cluster_name,
                             'merge': merge,
                             'compress': compress,
                             'what_data': what_data,
                             'feature_selection_method': feature_selection_method,
                             'scale_nwp_method': scale,
                             'groups': metadata['groups']}
        for param, value in trial.items():
            if param not in experiment_params.keys():
                experiment_params[param] = value
        experiment_params.update(self.fix_params)

        if 'bootstrap_type' in experiment_params.keys():
            if experiment_params["bootstrap_type"] == "Bayesian":
                experiment_params["subsample"] = None
            elif experiment_params["bootstrap_type"] == "Bernoulli":
                experiment_params["bagging_temperature"] = None

        if init_params is not None:
            experiment_params.update(init_params)

        path_weights = os.path.join(self.cluster_dir, self.method, f'test_{trial_number}')
        if not os.path.exists(path_weights):
            os.makedirs(path_weights)
        else:
            shutil.rmtree(path_weights)
            os.makedirs(path_weights)
        model = ShallowModel(self.static_data, path_weights, params=experiment_params, n_jobs=self.n_jobs)
        acc = model.fit(X, y, self.cv_masks, metadata)
        experiment_params['value'] = acc
        experiment_params['best_mae_test'] = model.best_mae_test
        columns = ['trial_number', 'value', 'best_mae_test'] + self.param_names
        trials.append({key: value for key, value in experiment_params.items() if key in columns})
        del model
        gc.collect()


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
        return dict()
    param_names = get_param_names(static_data, method)
    return {param: value for param, value in model.get_params().items() if param in param_names}


def run_optimization(project_id, static_data, cluster_name, cluster_dir, method, refit):
    path_group = static_data['path_group']
    n_jobs = 4  # find_free_cpus(path_group)
    print(f'CPU methods starts running on {n_jobs} cpus')
    print(f'{method} Model of {cluster_name} of {project_id} is starts.....')

    if not os.path.exists(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv')):
        ss = SeedSequence(12345)
        random_states = [default_rng(s) for s in ss.spawn(static_data[method]['n_trials'])]
        objective = Objective(static_data, cluster_name, cluster_dir, method, n_jobs=1, refit=refit)
        manager = mp.Manager()
        shared_trials = manager.list()
        with Parallel(n_jobs=n_jobs) as parallel:
            start = 0
            if 'RBF' not in method:
                init_params = find_init_params(static_data, method)
                parallel(delayed(objective.fit_trial)(0, random_states[0], shared_trials, init_params=init_params))
                start = 1
            parallel(delayed(objective.fit_trial)(trial_number, random_states[trial_number], shared_trials)
                     for trial_number in range(start, static_data[method]['n_trials']))
        trials = []
        for trial in shared_trials:
            param_dict = dict()
            for key in trial.keys():
                param_dict[key] = trial[key]
            trials.append(param_dict)

        trials = pd.DataFrame(trials)
        results = trials.sort_values(by='value')
        best_trial = trials['best_mae_test'].idxmin()
        model_dir = os.path.join(cluster_dir, method)
        test_dir = os.path.join(cluster_dir, method, f'test_{best_trial}')
        for filename in glob.glob(os.path.join(test_dir, '*.*')):
            print(filename)
            shutil.copy(filename, model_dir)
        results.to_csv(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv'))
        for number in range(static_data[method]['n_trials']):
            test_dir = os.path.join(cluster_dir, method, f'test_{number}')
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)


def CPU_thread(static_data, method, cluster=None, refit=False):
    if cluster is not None:
        for cluster_name, cluster_dir in cluster.items():
            run_optimization(static_data['_id'], static_data, cluster_name, cluster_dir, method, refit)
    else:
        clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
        for cluster_name, cluster_dir in clusters.items():
            run_optimization(static_data['_id'], static_data, cluster_name, cluster_dir, method, refit)


def train_clusters_on_cpus(static_data, cluster=None, method=None, refit=False):
    print('cpu')
    time.sleep(10)
    gpu_methods = ['CNN', 'LSTM', 'RBFNN', 'MLP', 'RBF-CNN']
    cpu_methods = ['CatBoost', 'RF', 'lasso', 'RBFols', 'GA_RBFols']

    methods = []
    if method is None:
        for m, values in static_data['project_methods'].items():
            if values and m in cpu_methods:
                methods.append(m)
            else:
                if values and m not in gpu_methods:
                    raise ValueError(f'Regression method {m} unknown')

    if method is None and cluster is None:
        for method in methods:
            CPU_thread(static_data, method, refit=refit)
            print(f'Training of {method} ends successfully')
    else:
        if method is not None and cluster is None:
            CPU_thread(static_data, method, refit=refit)
            print(f'Training of {method} ends successfully')
        elif method is None and cluster is not None:
            for method in methods:
                CPU_thread(static_data, method, cluster=cluster, refit=refit)
                print(f'Training of {method} ends successfully')
        else:
            CPU_thread(static_data, method, cluster=cluster, refit=refit)
