import os
import gc
import glob
import shutil
import joblib
import optuna

import numpy as np
import pandas as pd
import multiprocessing as mp

from joblib import Parallel
from joblib import delayed
from optuna.samplers import TPESampler

from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.dataset_utils import sync_target_with_tensors
from eforecast.common_utils.train_utils import distance

from eforecast.common_utils.train_utils import GpuQueue
from eforecast.dataset_creation.data_feeder import DataFeeder
from eforecast.dataset_creation.files_manager import FilesManager

from eforecast.deep_models.tf_1x.global_network import DistributedDeepNetwork


class Objective(object):
    def __init__(self, static_data, n_gpus):
        self.gpu_queue = GpuQueue(n_gpus, all_gpus=True)
        self.static_data = static_data
        self.param_space = self.static_data['Global']
        self.data_structure = self.static_data['data_structure']
        self.feature_selection_methods = self.static_data['feature_selection_methods']
        self.data_feeder = DataFeeder(self.static_data, online=False, train=True)
        self.file_manager = FilesManager(self.static_data, is_online=False, train=True)

    def select_structure(self, experiment_tag):
        exp = self.static_data['experiments'][experiment_tag]
        exp_sel = dict()
        for key, layers in exp.items():
            exp_sel[key] = []
            for i, layer in enumerate(layers):
                layer_type = layer[0]
                sizes = layer[1]
                size = sizes
                exp_sel[key].append((layer_type, size))
        return exp_sel

    def load_data(self, merge, compress, scale_method, what_data, feature_selection_method=None):
        X, metadata = self.data_feeder.feed_inputs(merge=merge, compress=compress,
                                                   scale_nwp_method=scale_method,
                                                   what_data=what_data,
                                                   feature_selection_method=feature_selection_method)
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
        if len(self.param_space['nwp_data_merge']) > 1:
            merge = trial.suggest_categorical('merge', self.param_space['nwp_data_merge'])
        else:
            merge = list(self.param_space['nwp_data_merge'])[0]
        if len(self.param_space['experiment_tag']) > 1:
            experiment_tag = trial.suggest_categorical('experiment_tag', self.param_space['experiment_tag'])
        else:
            experiment_tag = list(self.param_space['experiment_tag'])[0]
        experiment = self.select_structure(experiment_tag)

        if 'mlp' in experiment_tag:
            method = 'MLP'
            if len(self.param_space['compress_data']) > 1:
                compress = trial.suggest_categorical('compress', self.param_space['compress_data'])
            else:
                compress = list(self.param_space['compress_data'])[0]
            if len(self.feature_selection_methods) > 1:
                feature_selection_method = trial.suggest_categorical('feature_selection_method',
                                                                     self.feature_selection_methods)
            else:
                feature_selection_method = self.feature_selection_methods[0]
            data_structure = [data_struct for data_struct in self.data_structure
                              if data_struct in {'row', 'row_all',
                                                 'row_dict',
                                                 'row_dict_distributed'}]
            if len(data_structure) == 0:
                raise ValueError('Cannot find what_data structure to fit MLP')
            elif len(data_structure) > 1:
                what_data = trial.suggest_categorical('what_data', data_structure)
            else:
                what_data = data_structure[0]
        else:
            if 'cnn' in experiment_tag:
                method = 'CNN'
            elif 'lstm' in experiment_tag:
                method = 'LSTM'
            else:
                raise ValueError(f'Unknown method {experiment_tag} in experiment_tag')
            compress = None
            feature_selection_method = None
            what_data = str.lower(method)
        if len(self.param_space['scale_nwp_method']) > 1:
            scale = trial.suggest_categorical('scale', self.param_space['scale_nwp_method'])
        else:
            scale = list(self.param_space['scale_nwp_method'])[0]

        if len(self.param_space['is_fuzzy']) > 1:
            is_fuzzy = trial.suggest_categorical('is_fuzzy', self.param_space['is_fuzzy'])
        else:
            is_fuzzy = list(self.param_space['is_fuzzy'])[0]
        if is_fuzzy:
            if len(self.param_space['n_rules']) > 1:
                n_rules = trial.suggest_categorical('n_rules', self.param_space['n_rules'])
            else:
                n_rules = list(self.param_space['n_rules'])[0]
            clustering_method = None
        else:
            n_rules = None
            if len(self.static_data['clustering']['methods']) > 1:
                clustering_method = trial.suggest_categorical('clustering_method',
                                                              self.static_data['clustering']['methods'])
            else:
                clustering_method = self.static_data['clustering']['methods'][0]
        X, y, metadata = self.load_data(merge, compress, scale, what_data,
                                        feature_selection_method=feature_selection_method)

        cv_mask = self.file_manager.check_if_exists_cv_data()
        name = f'Distributed_{trial.number}'
        experiment_params = {'name': name,
                             'experiment_tag': experiment_tag,
                             'experiment': experiment,
                             'merge': merge,
                             'compress': compress,
                             'what_data': what_data,
                             'feature_selection_method': feature_selection_method,
                             'scale_nwp_method': scale,
                             'is_fuzzy': is_fuzzy,
                             'n_rules': n_rules,
                             'clustering_method': clustering_method,
                             'rbf_var_imp': self.param_space['rbf_var_imp'],
                             'data_type': self.param_space['data_type']}
        for param, value in self.static_data['Global'].items():
            if param not in {'experiment_tag', 'keep_n_models', 'n_trials', 'nwp_data_merge', 'compress_data',
                             'scale_nwp_method', 'n_rules', 'is_fuzzy', 'clustering_method', 'rbf_var_imp',
                             'data_type'}:
                if isinstance(value, set):
                    if len(value) > 1:
                        v = trial.suggest_categorical(param, list(value))
                    else:
                        v = list(value)[0]
                elif isinstance(value, list):
                    if len(value) > 1:
                        v = trial.suggest_int(param, value[0], value[-1])
                    else:
                        v = value[0]
                else:
                    v = value
                experiment_params[param] = v

        path_weights = os.path.join(self.static_data['path_model'], 'Distributed', f'Distributed_{trial.number}')
        if not os.path.exists(path_weights):
            os.makedirs(path_weights)
        model = DistributedDeepNetwork(self.static_data, path_weights, is_online=False, train=True,
                                       params=experiment_params)
        if model.is_trained:
            acc = model.best_mae_test
            for param in trial.params.keys():
                if param in model.params.keys():
                    trial.params[param] = model.params[param]
        else:
            with self.gpu_queue.one_gpu_per_process() as gpu_i:
                model.fit(X, y, cv_mask, metadata, gpu_id=gpu_i)
                acc = model.best_mae_test
        for key, value in model.results.items():
            trial.set_user_attr(key, value)
        del model
        gc.collect()
        if isinstance(acc, np.ndarray):
            acc = float(np.mean(acc))
        return acc


def get_results(static_data, cluster_dir, trial, method):
    path_weights = os.path.join(cluster_dir, method, f'test_{trial}')
    model = DistributedDeepNetwork(static_data, path_weights)

    return model.results


def optuna_thread(project_id, static_data, n_gpus):
    print(f'Distributed Model HPO of {project_id} starts.....')
    if not os.path.exists(os.path.join(static_data['path_model'], 'Distributed', f'results.csv')):
        study = optuna.create_study(sampler=TPESampler(seed=42, consider_magic_clip=True, n_startup_trials=2,
                                                       n_ei_candidates=2))

        study.optimize(Objective(static_data, n_gpus), n_trials=static_data['Global']['n_trials'],
                       gc_after_trial=True, n_jobs=n_gpus)
        results = study.trials_dataframe().sort_values(by='value')
        cols = ['user_attrs_mae_test_out', 'user_attrs_mae_val_out',
                'user_attrs_sse_test_out', 'user_attrs_sse_val_out']
        res = results[cols]
        res['diff_mae'] = res['user_attrs_mae_test_out'].subtract(res['user_attrs_mae_val_out']).abs()
        res['diff_sse'] = res['user_attrs_sse_test_out'].subtract(res['user_attrs_sse_val_out']).abs()
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
        results.to_csv(os.path.join(static_data['path_model'], 'Distributed', f'results.csv'))
    else:
        results = pd.read_csv(os.path.join(static_data['path_model'], 'Distributed', f'results.csv'),
                              index_col=0, header=0)
    return results.number.values


def train_distributed_on_gpus(static_data):
    if static_data['is_Global']:
        path_group = static_data['path_group']
        joblib.dump(np.array([0]), os.path.join(path_group, 'freeze_for_gpu.pickle'))
        n_gpus = static_data['n_gpus']
        gpu_status = 1
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))

        best_trials = optuna_thread(static_data['_id'], static_data, n_gpus)
        path = os.path.join(static_data['path_model'], 'Distributed')
        remove_worst_models(best_trials, static_data['Global']['keep_n_models'], path)
        gpu_status = 0
        joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))

        print(f'Training of Distributed ends successfully')


def remove_worst_models(results, keep_n_models, path):
    remove_paths = [os.path.join(path, f'Distributed_{trial}') for trial in results[keep_n_models:]]
    for directory in remove_paths:
        print(directory)
        shutil.rmtree(directory)
