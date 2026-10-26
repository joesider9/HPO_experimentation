import gc
import os
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tqdm import tqdm

from eforecast.common_utils.train_utils import distance
from eforecast.common_utils.train_utils import feed_data_eval
from eforecast.common_utils.train_utils import get_tf_config
from eforecast.common_utils.train_utils import lr_schedule
from eforecast.common_utils.train_utils import split_validation_set
from eforecast.deep_models.tf_1x.builders import build_graph
from eforecast.deep_models.tf_1x.builders import create_placeholders
from eforecast.deep_models.tf_1x.global_builders import check_rbf_bounds
from eforecast.deep_models.tf_1x.global_builders import create_centroids
from eforecast.deep_models.tf_1x.global_builders import get_rbf
from eforecast.deep_models.tf_1x.global_builders import get_stratify_batches_by_act
from eforecast.deep_models.tf_1x.optimizers import optimize
from eforecast.deep_models.tf_1x.trainer import gather_weights
from eforecast.deep_models.tf_1x.trainer import train_schedule_fuzzy
from eforecast.deep_models.tf_1x.trainer import train_schedule_global
from eforecast.deep_models.tf_1x.trainer import train_step
from eforecast.deep_models.tf_1x.trainer import validation_step

pd.set_option('display.expand_frame_repr', False)


class DeepNetwork:
    def __init__(self, static_data, path_weights, params=None, is_global=False, is_fuzzy=False, is_for_cluster=False,
                 refit=False):
        self.results = None
        self.best_sse_val = None
        self.best_sse_test = None
        self.best_min_act = None
        self.best_max_act = None
        self.best_mean_act = None
        self.best_sum_act = None
        self.best_mae_val = None
        self.best_mae_test = None
        self.best_weights = None
        self.n_batch = None
        self.n_out = None
        self.is_trained = False
        self.refit = refit
        self.is_global = is_global
        self.is_fuzzy = is_fuzzy
        self.is_for_cluster = is_for_cluster
        self.static_data = static_data
        self.rated = static_data['rated']
        if params is not None:
            self.params = params
            self.method = self.params['method']
            self.name = self.params['name']
            self.model_layers = self.params['experiment']
            self.merge = self.params['merge']
            self.what_data = self.params['what_data']
            self.compress = self.params['compress']
            self.scale_nwp_method = self.params['scale_nwp_method']
            self.groups = self.params['groups']
            self.epochs = self.params['max_iterations']
            self.learning_rate = self.params['learning_rate']
            self.batch_size = self.params['batch_size']
        self.path_weights = path_weights

        try:
            if not self.refit:
                self.load()
        except:
            pass
        self.refit = refit
        if not hasattr(self, 'params'):
            raise RuntimeError('The network has no params. You should train the net providing params')

    def get_slice(self, x, mask, meta_data, y=None, x_imp=None, act=None, rules=None):
        group_layers = []
        dates = meta_data['dates']
        mask = mask.intersection(dates)
        indices = dates.get_indexer(mask)
        y_slice = y.iloc[indices].values if y is not None else None
        if isinstance(x, list):  # These data is for cnn method because must have data_row
            X_slice = dict()
            data, data_row = x
            if len(self.groups) == 0:
                group_layers += ['input']
                X_slice['input'] = data[indices].squeeze()
            else:
                group_layers += self.merge.split('_')[1:]
                for group in self.groups:
                    group_name = '_'.join(group) if isinstance(group, tuple) else group
                    X_slice[group_name] = data[group_name][indices].squeeze()
            X_slice['data_row'] = data_row.iloc[indices].values
        elif isinstance(x, dict) and len(self.groups) > 0:  # These data is for lstm and mlp method
            X_slice = dict()
            group_layers += self.merge.split('_')[1:]
            for group in self.groups:
                group_name = '_'.join(group) if isinstance(group, tuple) else group
                X_slice[group_name] = x[group_name].iloc[indices].values
        elif isinstance(x, pd.DataFrame):  # These data is for mlp method
            X_slice = x.iloc[indices].values
        elif isinstance(x, np.ndarray):
            X_slice = x[indices]  # These data is for lstm method
        else:
            raise ValueError('Wrong type of input X')
        if x_imp is None and rules is None:
            return X_slice, y_slice, group_layers
        else:
            if (self.is_global and self.is_fuzzy) and (x_imp is None or act is not None):
                raise ValueError('If the model is_global and is_fuzzy, you should provide data for clustering x_imp '
                                 'and activations should be None')
            if (self.is_global and not self.is_fuzzy) and (x_imp is not None or act is None):
                raise ValueError('If the model is_global but not is_fuzzy, you should provide data for clustering'
                                 ' activations and x_imp should be None')
            if rules is not None:
                if x_imp is None and act is None:
                    raise ValueError('If you provide rules, you should also provide data for clustering x_imp '
                                     'or activations')
                if not isinstance(x_imp, pd.DataFrame) and act is None:
                    raise TypeError('Data for clustering x_imp should be a dataframe')
                X_slice_rules = dict()
                group_layers = ['clustering'] + group_layers
                if x_imp is not None:
                    X_slice_rules['clustering'] = x_imp.iloc[indices].values
                elif act is not None:
                    for rule in act.columns:
                        X_slice_rules[f'act_{rule}'] = act[rule].iloc[indices].values.reshape(-1, 1)
                else:
                    raise ValueError('If you provide rules, you should also provide data for clustering x_imp '
                                     'or activations')
                for rule in rules:
                    if isinstance(X_slice, dict):
                        if 'data_row' in X_slice.keys():
                            X_slice_rules['data_row'] = X_slice['data_row']
                        for key, value in X_slice.items():
                            if key != 'data_row':
                                X_slice_rules['_'.join([rule, key])] = value
                    else:
                        X_slice_rules[rule] = X_slice

                return X_slice_rules, y_slice, group_layers
            else:
                raise ValueError('If you provide data for clustering x_imp, you should also provide rules')

    def fit(self, X, y, cv_masks, meta_data, activations=None, gpu_id=0, X_imp=None):
        if self.is_trained and not self.refit:
            return self.best_mae_test
        self.params['n_out'] = y.shape[1]
        if self.is_fuzzy:
            if self.refit or ('rules' not in self.params.keys() and 'centroids' not in self.params.keys()):
                self.params = create_centroids(X_imp.loc[cv_masks[0].intersection(X_imp.index)], self.params)
                self.save()
        elif self.is_global:
            if activations is None:
                raise ValueError('Provide activations or turn is_fuzzy attribute to True')
            self.params['rules'] = activations.columns
        else:
            self.params['rules'] = None
        thres_act = self.params['thres_act'] if 'thres_act' in self.params.keys() else None
        X_train, y_train, group_layers = self.get_slice(X, cv_masks[0], meta_data, y=y, x_imp=X_imp, act=activations,
                                                        rules=self.params['rules'])
        X_val, y_val, _ = self.get_slice(X, cv_masks[1], meta_data, y=y, x_imp=X_imp, act=activations,
                                         rules=self.params['rules'])
        X_test, y_test, _ = self.get_slice(X, cv_masks[2], meta_data, y=y, x_imp=X_imp, act=activations,
                                           rules=self.params['rules'])

        if isinstance(X_train, dict):
            self.params['scopes'] = [scope for scope in X_train.keys() if 'act' not in scope]
        else:
            self.params['scopes'] = ['input']

        self.params['group_layers'] = group_layers
        with open(os.path.join(self.path_weights, 'parameters.txt'), 'w') as file:
            file.write(yaml.dump(self.params, default_flow_style=False, sort_keys=False))
        self.n_out = y_train.shape[1]
        N = cv_masks[0].intersection(meta_data['dates']).shape[0]
        self.batch_size = np.minimum(self.batch_size, int(N / 7.5))
        self.n_batch = int(N / self.batch_size)
        batches = [np.random.choice(N, self.batch_size, replace=False) for _ in range(self.n_batch + 1)]

        ind_val_list = split_validation_set(X_val)
        ind_test_list = split_validation_set(X_test)

        config = get_tf_config(self.static_data['n_jobs'])

        tf.compat.v1.reset_default_graph()
        graph_cnn = tf.Graph()
        print('Create graph....')
        with graph_cnn.as_default():
            with tf.device('/device:GPU:' + str(gpu_id)):
                x_pl, y_pl = create_placeholders(X_train, self.params, train=True, is_global=self.is_global,
                                                 is_fuzzy=self.is_fuzzy)
                model_output, model_layers_built, cluster_outputs, \
                    act_pl, act_nan_err = build_graph(x_pl,
                                                      self.model_layers,
                                                      self.params,
                                                      is_fuzzy=
                                                      self.is_fuzzy,
                                                      is_global=
                                                      self.is_global,
                                                      is_for_cluster=
                                                      self.is_for_cluster,
                                                      thres_act=thres_act)
                trainers, MAEs, SSEs, learning_rate = optimize(model_output, y_pl,
                                                               cluster_outputs=cluster_outputs,
                                                               act_all=act_pl,
                                                               act_nan_err=act_nan_err,
                                                               rated=self.rated,
                                                               is_global=self.is_global,
                                                               is_fuzzy=self.is_fuzzy,
                                                               is_for_clustering=self.is_for_cluster,
                                                               thres_act=thres_act,
                                                               rules=self.params['rules'])

        len_performers = 2
        mae_old, sse_old = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
        mae_max, sse_max = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
        mae_min, sse_min = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)

        best_mae_val, best_mae_test = np.inf, np.inf
        train_flag, best_weights, best_clusters, init_clusters = True, None, None, None
        warm = self.params['warming_iterations']
        wait, best_iteration, best_tot_iteration, loops, n_iter = 0, 0, 0, 0, 0
        epochs = self.epochs
        patience, exam_period = int(self.epochs / 3), int(self.epochs / 5)

        results_columns = ['Iteration', 'best_iteration', 'best_mae_val', 'best_mae_test', 'mae_val_out',
                           'mae_test_out', 'sse_val_out', 'sse_test_out']
        if self.is_global:
            results_columns += ['cl_val_out', 'cl_test_out', 'sum_activations', 'min_activations', 'max_activations',
                                'mean_activations', 'mae_lin_val', 'mae_lin_test']
            if self.is_fuzzy:
                if epochs <= 400:
                    raise ValueError('epochs should be greater than 400 when it is fuzzy')
                mae_old_lin, sse_old_lin = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
                mae_max_lin, sse_max_lin = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
                mae_min_lin, sse_min_lin = np.inf * np.ones(len_performers), np.inf * np.ones(len_performers)
        results = pd.DataFrame(columns=results_columns)

        print(f"Start training of {self.name} using {self.method} with {self.n_batch} batches and {self.epochs} epochs")

        with tf.compat.v1.Session(graph=graph_cnn, config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            variables = [v for v in tf.compat.v1.trainable_variables()]
            if self.is_global:
                init_clusters = get_rbf(sess)
                best_clusters = get_rbf(sess)
            while train_flag:
                for epoch in tqdm(range(epochs)):

                    lr = lr_schedule(epoch, lr=self.learning_rate)
                    if not self.is_global and not self.is_fuzzy:
                        random.shuffle(batches)
                        train_step(sess, trainers['bulk'], batches, x_pl, y_pl, X_train, y_train, learning_rate, lr)
                    else:
                        batches = get_stratify_batches_by_act(sess, act_pl, N, x_pl, y_pl, X_train, y_train,
                                                              learning_rate, lr, self.params['thres_act'],
                                                              self.batch_size, self.n_batch)
                        if self.is_global and not self.is_fuzzy:
                            train_schedule_global(sess, trainers, self.params['rules'], batches, x_pl, y_pl, X_train,
                                                  y_train, learning_rate, lr)
                        elif self.is_global and self.is_fuzzy:
                            train_schedule_fuzzy(sess, epoch, trainers, self.params['rules'], batches, x_pl, y_pl,
                                                 X_train, y_train,
                                                 learning_rate, lr, warm)

                    warm = 0

                    if self.is_global:
                        mae_val_lin, mae_test_lin, sse_val_lin, sse_test_lin, \
                            sum_act, min_act, max_act, mean_act, warm = \
                            check_rbf_bounds(sess, act_pl, N, x_pl, y_pl,
                                             X_train, y_train,
                                             X_val, y_val,
                                             X_test, y_test,
                                             learning_rate, lr,
                                             self.params, init_clusters, wait)
                        if self.is_fuzzy:
                            if epoch == 0:
                                init_mae_val_lin, init_mae_test_lin = mae_val_lin, mae_test_lin
                            mae_lin = np.hstack([mae_val_lin, mae_test_lin])
                            flag_mae_lin, mae_old_lin, mae_max_lin, mae_min_lin = distance(mae_lin, mae_old_lin,
                                                                                           mae_max_lin, mae_min_lin)
                            sse_lin = np.hstack([sse_val_lin, sse_test_lin])
                            flag_sse_lin, sse_old_lin, sse_max_lin, sse_min_lin = distance(sse_lin, sse_old_lin,
                                                                                           sse_max_lin, sse_min_lin)
                            if flag_mae_lin and flag_sse_lin:
                                best_clusters = get_rbf(sess)
                            # if init_mae_val_lin + 0.001 < mae_val_lin and init_mae_test_lin + 0.001 < mae_test_lin:
                            #     assign_rbf(sess, best_clusters)
                            #     warm = 4

                    sse_val = validation_step(sess, SSEs, np.arange(y_val.shape[0]), x_pl, y_pl, X_val,
                                              y_val, learning_rate, lr)
                    mae_val = validation_step(sess, MAEs, np.arange(y_val.shape[0]), x_pl, y_pl, X_val,
                                              y_val, learning_rate, lr)
                    sse_test = validation_step(sess, SSEs, np.arange(y_test.shape[0]), x_pl, y_pl, X_test,
                                               y_test, learning_rate, lr)
                    mae_test = validation_step(sess, MAEs, np.arange(y_test.shape[0]), x_pl, y_pl, X_test,
                                               y_test, learning_rate, lr)
                    mae = np.hstack([mae_val[-1], mae_test[-1]])
                    flag_mae, mae_old, mae_max, mae_min = distance(mae, mae_old, mae_max, mae_min)
                    sse = np.hstack([sse_val[-1], sse_test[-1]])
                    flag_sse, sse_old, sse_max, sse_min = distance(sse / 10, sse_old, sse_max, sse_min)
                    flag_best = flag_mae and flag_sse
                    if flag_best:
                        best_weights = gather_weights(sess, variables)
                        best_tot_iteration = n_iter
                        best_iteration = epoch
                        wait = 0
                    else:
                        wait += 1
                    best_mae_val = mae_val[-1] if best_mae_val >= mae_val[-1] else best_mae_val
                    best_mae_test = mae_test[-1] if best_mae_test >= mae_test[-1] else best_mae_test
                    evaluation = np.array([n_iter, best_tot_iteration, best_mae_val, best_mae_test,
                                           mae_val[-1], mae_test[-1], sse_val[-1], sse_test[-1]])

                    print_columns = ['best_mae_val', 'best_mae_test', 'mae_val_out', 'mae_test_out']
                    if self.is_global:
                        evaluation = np.concatenate([evaluation, np.array([mae_val[0], mae_test[0]])])
                        print_columns += ['cl_val_out', 'cl_test_out']

                        if self.is_global:
                            evaluation = np.concatenate([evaluation, np.array([sum_act,
                                                                               min_act,
                                                                               max_act, mean_act,
                                                                               mae_val_lin, mae_test_lin])])
                            print_columns += ['mae_lin_val', 'mae_lin_test']

                    res = pd.DataFrame(evaluation.reshape(-1, 1).T, index=[n_iter], columns=results_columns)
                    results = pd.concat([results, res])
                    n_iter += 1
                    print(res[print_columns])
                    if wait > patience and epoch > 400:
                        train_flag = False
                        break
                if (epochs - best_iteration) <= exam_period and epoch > 400:
                    if loops > 3:
                        train_flag = False
                    else:
                        epochs, exam_period = patience, int(patience / 3)
                        best_iteration = 0
                        loops += 1
                else:
                    train_flag = False
            sess.close()
        self.best_weights = best_weights
        self.best_mae_test = results['mae_test_out'].iloc[best_tot_iteration]
        self.best_mae_val = results['mae_val_out'].iloc[best_tot_iteration]
        self.best_sse_test = results['sse_test_out'].iloc[best_tot_iteration]
        self.best_sse_val = results['sse_val_out'].iloc[best_tot_iteration]
        self.results = results.iloc[best_tot_iteration]
        results.to_csv(os.path.join(self.path_weights, 'results.csv'))
        self.is_trained = True
        self.save()
        gc.collect()
        if self.is_fuzzy:
            self.best_sum_act = results['sum_activations'].iloc[best_tot_iteration]
            self.best_min_act = results['min_activations'].iloc[best_tot_iteration]
            self.best_max_act = results['max_activations'].iloc[best_tot_iteration]
            self.best_mean_act = results['mean_activations'].iloc[best_tot_iteration]
            print(f'SUM OF ACTIVATIONS IS {self.best_sum_act}')
            print(f'MIN OF ACTIVATIONS IS {self.best_min_act}')
            print(f'MAX OF ACTIVATIONS IS {self.best_max_act}')
            print(f'MEAN OF ACTIVATIONS IS {self.best_mean_act}')
        print(f"Total accuracy of validation: {self.best_mae_val} and of testing {self.best_mae_test}")

    def predict(self, X, metadata, cluster_dates=None, X_imp=None, activations=None, with_activations=False):
        self.load()
        thres_act = self.params['thres_act'] if 'thres_act' in self.params.keys() else None
        if not hasattr(self, 'best_weights'):
            raise NotImplementedError(f'The {self.method} network is not train. '
                                      f'The location path is {self.path_weights}')
        cluster_dates = metadata['dates'] if cluster_dates is None else cluster_dates
        inp_x, _, _ = self.get_slice(X, cluster_dates, metadata, x_imp=X_imp, act=activations, rules=self.params['rules'])
        tf.compat.v1.reset_default_graph()
        graph_net = tf.Graph()
        print('Create graph....')
        with graph_net.as_default():
            x_pl = create_placeholders(inp_x, self.params, train=False, is_global=self.is_global,
                                       is_fuzzy=self.is_fuzzy)
            model_output, model_layers_built, cluster_outputs, act_pl, act_nan_err = build_graph(x_pl,
                                                                                                 self.model_layers,
                                                                                                 self.params,
                                                                                                 is_fuzzy=self.is_fuzzy,
                                                                                                 is_global=
                                                                                                 self.is_global,
                                                                                                 is_for_cluster=
                                                                                                 self.is_for_cluster,
                                                                                                 thres_act=thres_act,
                                                                                                 train=False)

        config = get_tf_config(self.static_data['n_jobs'])
        with tf.compat.v1.Session(graph=graph_net, config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for variable in tf.compat.v1.trainable_variables():
                sess.run(tf.compat.v1.assign(variable, self.best_weights[variable.name]))
            feed_dict = feed_data_eval(x_pl, inp_x)
            y_pred = sess.run([model_output], feed_dict=feed_dict)
            if with_activations:
                activations = sess.run([act_pl], feed_dict=feed_dict)
            sess.close()
        if self.static_data['horizon_type'] == 'multi-output':
            cols = [f'hour_ahead_{h}' for h in range(self.static_data['horizon'])]
        else:
            cols = [self.method]
        y_pred = pd.DataFrame(y_pred[0], index=cluster_dates, columns=cols)
        if with_activations:
            activations = np.concatenate(activations[0], axis=1) if self.is_fuzzy else activations[0]
            activations = pd.DataFrame(activations, index=cluster_dates, columns=sorted(self.params['rules']))
            return y_pred, activations
        else:
            return y_pred

    def load(self):
        if os.path.exists(os.path.join(self.path_weights, 'net_weights.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.path_weights, 'net_weights.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot load weights for cnn model' + self.path_weights)
        else:
            raise ImportError('Cannot load weights for cnn model' + self.path_weights)

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['static_data', 'path_weights', 'refit']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict, os.path.join(self.path_weights, 'net_weights.pickle'), compress=9)
