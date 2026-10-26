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

from GPUtil import getGPUs

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor

from hebo_trial.optimizer_mace import MACEBO

from numpy.random import SeedSequence, default_rng

from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.dataset_utils import sync_target_with_tensors
from eforecast.common_utils.train_utils import distance

from eforecast.dataset_creation.data_feeder import DataFeeder
from eforecast.deep_models.tf_1x.network import DeepNetwork


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            self._cconn.send(-1)

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class Objective():
    def __init__(self, static_data, cluster_name, cluster_dir, method, refit=False):
        self.space_structure = None
        self.static_data = static_data
        self.method = method
        self.cluster_dir = cluster_dir
        self.cluster_name = cluster_name
        self.refit = refit
        self.warming = self.static_data[method]['warming_iterations']

        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.scale_nwp_methods = self.static_data['scale_nwp_method']
        self.data_structure = self.static_data['data_structure']
        self.feature_selection_methods = self.static_data['feature_selection_methods']
        self.data_feeder = DataFeeder(self.static_data, online=False, train=True)
        self.initialize(refit)

    def initialize(self, refit=False):
        self.define_space()
        self.define_structure_space()

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
        metadata['dates'] = y.index
        return X, y, metadata

    def get_param(self, name, type, dtype, space=None, range=None, values=None):
        return {name: {'type': type,
                       'dtype': dtype,
                       'space': space,
                       'range': range,
                       'values': values}}

    def select_structure(self, trial_structure, experiment_tag, exp):
        exp_sel = dict()
        for key, layers in exp.items():
            exp_sel[key] = []
            for i, layer in enumerate(layers):
                layer_type = layer[0]
                param = f'{experiment_tag}_{key}_{layer_type}_{i}'
                size = trial_structure[param] if param in trial_structure.keys() \
                    else self.fix_params_structure[experiment_tag][param]
                if isinstance(size, str):
                    if size not in 'linear':
                        size = float(size)
                elif isinstance(size, int):
                    size = float(size)
                exp_sel[key].append((layer_type, size))
        return exp_sel

    def define_structure_space(self):
        experiment_tags = list(self.static_data[self.method]['experiment_tag'])
        self.space_structure = dict()
        self.fix_params_structure = dict()
        self.param_layer_names = dict()
        for experiment_tag in experiment_tags:
            exp = self.static_data['experiments'][experiment_tag]
            self.space_structure[experiment_tag] = dict()
            self.fix_params_structure[experiment_tag] = dict()
            self.param_layer_names[experiment_tag] = []
            for key, layers in exp.items():
                for i, layer in enumerate(layers):
                    layer_type = layer[0]
                    sizes = layer[1]
                    param = f'{experiment_tag}_{key}_{layer_type}_{i}'
                    if isinstance(sizes, list):
                        if len(sizes) > 1:
                            self.space_structure[experiment_tag].update(
                                self.get_param(param, 'real', 'float', range=list(sizes)))
                        elif len(sizes) == 1:
                            self.fix_params_structure[experiment_tag][param] = list(sizes)[0]
                        else:
                            self.fix_params_structure[experiment_tag][param] = sizes
                    else:
                        if len(sizes) > 1:
                            self.space_structure[experiment_tag].update(
                                self.get_param(param, 'cat', 'float', values=list(sizes)))
                        elif len(sizes) == 1:
                            self.fix_params_structure[experiment_tag][param] = sizes[0]
                        else:
                            self.fix_params_structure[experiment_tag][param] = sizes
        self.param_structure_names = dict()
        for experiment_tag, experiment in self.space_structure.items():
            self.param_structure_names[experiment_tag] = []
            for param_name, param_attr in experiment.items():
                self.param_structure_names[experiment_tag].append(param_name)

    def define_space(self):
        self.space = dict()
        self.fix_params = dict()
        if len(self.nwp_data_merge) > 1 and self.method not in {'LSTM'}:
            self.space.update(self.get_param('merge', 'cat', 'string', values=self.nwp_data_merge))
        elif self.method in {'LSTM'}:
            self.fix_params['merge'] = 'all'
        else:
            self.fix_params['merge'] = list(self.nwp_data_merge)[0]
        if len(self.nwp_data_compress) > 1 and self.method not in {'LSTM', 'CNN'}:
            self.space.update(self.get_param('compress', 'cat', 'string', values=self.nwp_data_compress))
        elif self.method in {'LSTM'}:
            self.fix_params['compress'] = 'load'
        elif self.method in {'CNN'}:
            self.fix_params['compress'] = None
        else:
            self.fix_params['compress'] = list(self.nwp_data_compress)[0]
        if len(self.feature_selection_methods) > 1 and self.method not in {'CNN'}:
            self.space.update(
                self.get_param('feature_selection_method', 'cat', 'string', values=self.feature_selection_methods))
        elif self.method in {'CNN'}:
            self.fix_params['feature_selection_method'] = None
        else:
            self.fix_params['feature_selection_method'] = self.feature_selection_methods[0]

        data_structure = [data_struct for data_struct in self.data_structure
                          if data_struct in {'row', 'row_all',
                                             'row_dict',
                                             'row_dict_distributed'}]
        if len(data_structure) == 0 and self.method not in {'LSTM', 'CNN'}:
            raise ValueError('Cannot find what_data structure to fit')
        elif len(data_structure) > 1 and self.method not in {'LSTM', 'CNN'}:
            self.space.update(self.get_param('what_data', 'cat', 'string', values=data_structure))
        elif self.method in {'LSTM', 'CNN'}:
            self.fix_params['what_data'] = str.lower(self.method)
        else:
            self.fix_params['what_data'] = data_structure[0]

        if len(self.scale_nwp_methods) > 1:
            self.space.update(self.get_param('scale', 'cat', 'string', values=self.scale_nwp_methods))
        else:
            self.fix_params['scale'] = list(self.scale_nwp_methods)[0]

        for param, value in self.static_data[self.method].items():
            if isinstance(value, set):
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

    @staticmethod
    def _fit(model, X, y, cv_mask, metadata, gpu_i):
        model.fit(X, y, cv_mask, metadata, gpu_id=gpu_i)

    def fit_trial(self, trial_number, random_state, trials, gpu_i):
        print(trials)
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
        X, y, metadata = self.load_data(merge, compress, scale, what_data,
                                        feature_selection_method=feature_selection_method)

        experiment_tag = trial['experiment_tag'] if 'experiment_tag' in trial.keys() \
            else self.fix_params['experiment_tag']

        cv_masks = joblib.load(os.path.join(self.cluster_dir, 'cv_mask.pickle'))

        experiment_params = {'trial_number': trial_number,
                             'method': self.method,
                             'name': self.cluster_name,
                             'merge': merge,
                             'compress': compress,
                             'what_data': what_data,
                             'feature_selection_method': feature_selection_method,
                             'scale_nwp_method': scale,
                             'groups': metadata['groups'],
                             'experiment_tag': experiment_tag}
        for param, value in trial.items():
            if param not in experiment_params.keys():
                experiment_params[param] = value
        experiment_params.update(self.fix_params)

        optimizer_structure = MACEBO(self.space_structure[experiment_tag])
        if len(trials) > 0:
            y_trial_structure = np.array([trial['value'] for trial in trials]).reshape(-1, 1)
            indices = []
            X_trial_structure = []
            for i, trial in enumerate(trials):
                param_dict = dict()
                for key in self.param_structure_names[experiment_tag]:
                    if key in trial.keys():
                        param_dict[key] = trial[key]
                if len(param_dict) > 0:
                    indices.append(i)
                    X_trial_structure.append(param_dict)
            if len(X_trial_structure) > 0:
                X_trial_structure = pd.DataFrame(X_trial_structure)
                optimizer_structure.observe(X_trial_structure, y_trial_structure[indices])
        trial_structure = optimizer_structure.suggest(n_suggestions=1, random_state=random_state,
                                                      warming=self.warming)[0]

        experiment_params['experiment'] = self.select_structure(trial_structure, experiment_tag,
                                                                self.static_data['experiments'][experiment_tag])

        path_weights = os.path.join(self.cluster_dir, self.method, f'test_{trial_number}')
        if os.path.exists(path_weights) and self.refit:
            shutil.rmtree(path_weights)
        if not os.path.exists(path_weights):
            os.makedirs(path_weights)
        acc = np.inf
        model = DeepNetwork(self.static_data, path_weights, experiment_params)
        if model.is_trained:
            acc = model.best_mae_test
            for param in model.params.keys():
                if param in trial.keys():
                    trial[param] = model.params[param]
        else:
            while True:
                gpus = getGPUs()
                gpuUtil = gpus[gpu_i].load
                if gpuUtil < 0.9:
                    break
                else:
                    time.sleep(10)

            while True:
                p = Process(target=self._fit, args=(model, X, y, cv_masks, metadata, gpu_i))
                p.start()
                p.join()
                gpus = getGPUs()
                memory_util = gpus[gpu_i].memoryUtil
                if p.exception or not os.path.exists(os.path.join(path_weights, 'net_weights.pickle')):
                    if memory_util < 0.12:
                        print('Trial aboard due to gpu memory utilization')
                        print()
                        model.best_weights = {}
                        model.best_mae_test = np.inf
                        model.best_mae_val = np.inf
                        model.results = pd.DataFrame()
                        model.is_trained = True
                        model.save()
                        return np.inf

                    print('Network do not fit to GPU memory')
                    time.sleep(30)
                    continue
                else:

                    break
            model.load()
            acc = model.best_mae_test
        print(acc)
        experiment_params['value'] = acc
        experiment_params['mae_test'] = model.best_mae_test
        experiment_params['mae_val'] = model.best_mae_val
        experiment_params['sse_test'] = model.best_sse_test
        experiment_params['sse_val'] = model.best_sse_val

        columns = ['trial_number', 'value', 'mae_test', 'mae_val', 'sse_val', 'sse_test'] + self.param_names
        trial = {key: value for key, value in experiment_params.items() if key in columns}
        trial.update(trial_structure)
        trials.append(trial)
        del model
        gc.collect()


def get_results(static_data, cluster_dir, trial, method):
    path_weights = os.path.join(cluster_dir, method, f'test_{trial}')
    model = DeepNetwork(static_data, path_weights)

    return model.results


def run_optimization(project_id, static_data, cluster_name, cluster_dir, method, n_jobs, refit, gpu_i):
    print(f'{method} Model of {cluster_name} of {project_id} is starts.....')
    if not os.path.exists(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv')) or refit:
        ss = SeedSequence(12345)
        random_states = [default_rng(s) for s in ss.spawn(static_data[method]['n_trials'])]
        objective = Objective(static_data, cluster_name, cluster_dir, method, refit=refit)
        manager = mp.Manager()
        shared_trials = manager.list()
        with Parallel(n_jobs=7, prefer='threads') as parallel:
            parallel(delayed(objective.fit_trial)(trial_number, random_states[trial_number], shared_trials, gpu_i)
                     for trial_number in range(static_data[method]['n_trials']))
        trials = []
        for trial in shared_trials:
            param_dict = dict()
            for key in trial.keys():
                param_dict[key] = trial[key]
            trials.append(param_dict)

        trials = pd.DataFrame(trials)
        results = trials.sort_values(by='value')
        cols = ['mae_test', 'mae_val',
                'sse_test', 'sse_val']
        res = results[cols]
        res['diff_mae'] = res['mae_test'].subtract(res['mae_val']).abs()
        res['diff_sse'] = res['sse_test'].subtract(res['sse_val']).abs()
        res_old, res_max, res_min = np.inf * np.ones(6), np.inf * np.ones(6), np.inf * np.ones(6)
        i = 0
        best_trials = []
        weights = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.05])
        while res.shape[0] > 0:
            flag_res, res_old, res_max, res_min = distance(res.iloc[i].values, res_old, res_max, res_min,
                                                           weights=weights)
            if flag_res:
                best = i
            i += 1
            if i == res.shape[0]:
                best_trials.append(res.index[best])
                i = 0
                res_old, res_max, res_min = np.inf * np.ones(6), np.inf * np.ones(6), np.inf * np.ones(6)
                res = res.drop(index=res.index[best])
        results = results.loc[best_trials]
        results.to_csv(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv'))
    else:
        results = pd.read_csv(os.path.join(cluster_dir, f'results_{cluster_name}_{method}.csv'), index_col=0, header=0)
    return results.number.values


def GPU_thread(static_data, n_gpus, n_jobs, cluster=None, method=None, refit=False):
    if cluster is not None:
        clusters = cluster
    else:
        clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))

    res = []
    gpu_ids = {cluster_name: i % n_gpus for i, cluster_name in enumerate(clusters.keys())}
    with ThreadPoolExecutor(n_gpus) as executor:
        futures = [executor.submit(run_optimization, static_data['_id'], static_data, cluster_name, cluster_dir, method,
                                   n_jobs, refit, gpu_ids[cluster_name])
                   for cluster_name, cluster_dir in clusters.items()]
        for future in as_completed(futures):
            res.append(future.result())

    best_trials = dict()
    i = 0
    for cluster_name, cluster_dir in clusters.items():
        best_trials[cluster_name] = dict()
        best_trials[cluster_name]['best'] = res[i][0]
        best_trials[cluster_name]['results'] = res[i]
        best_trials[cluster_name]['path'] = cluster_dir
        i += 1

    return best_trials


def train_clusters_on_gpus(static_data, cluster=None, method=None, refit=False):
    print('gpu')
    gpu_methods = ['CNN', 'LSTM', 'RBFNN', 'MLP', 'RBF-CNN']
    path_group = static_data['path_group']
    joblib.dump(np.array([0]), os.path.join(path_group, 'freeze_for_gpu.pickle'))
    methods = [method for method, values in static_data['project_methods'].items() if values and method in gpu_methods]
    n_gpus = static_data['n_gpus']

    if method is None:
        ordered_method = [i for i, method in enumerate(gpu_methods) if method in methods]
    else:
        ordered_method = [i for i, m in enumerate(gpu_methods) if m == method]
    for order in ordered_method:
        method = gpu_methods[order]
        gpu_status = 1
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))

        n_jobs = static_data[method]['n_jobs']

        best_trials = GPU_thread(static_data, n_gpus, n_jobs, method=method, cluster=cluster, refit=refit)

        save_deep_models(best_trials, method)
        gpu_status = 0
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))

        print(f'Training of {method} ends successfully')


def save_deep_models(results, method):
    for cluster_name, res in results.items():
        model_dir = os.path.join(res['path'], method)
        test_dir = os.path.join(model_dir, 'test_' + str(res['best']))
        for filename in glob.glob(os.path.join(test_dir, '*.*')):
            print(filename)
            shutil.copy(filename, model_dir)
        for test_dir_name in os.listdir(model_dir):
            if 'test' in test_dir_name:
                test_dir = os.path.join(model_dir, test_dir_name)
                if os.path.exists(test_dir):
                    shutil.rmtree(test_dir)
