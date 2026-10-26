import os
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNetCV, MultiTaskElasticNetCV

from eforecast.common_utils.logger import create_logger
from eforecast.common_utils.dataset_utils import sync_data_with_dates

from eforecast.deep_models.tf_1x.network import DeepNetwork

warnings.filterwarnings("ignore", category=FutureWarning)


class TfRBFClusterer:

    def __init__(self, static_data, refit=False):
        self.logger = None
        self.fuzzy_models = None
        self.rule_names = None
        self.is_trained = False
        self.refit = refit
        self.static_data = static_data
        self.rated = static_data['rated']
        self.n_jobs = self.static_data['clustering']['n_jobs']
        self.type = static_data['type']
        self.thres_act = self.static_data['clustering']['thres_act']
        self.var_fuzz = self.static_data['clustering']['rbf_var_imp']
        self.n_var_lin = self.static_data['clustering']['n_var_lin']
        self.min_samples = self.static_data['clustering']['min_samples']
        self.max_samples_ratio = self.static_data['clustering']['max_samples_ratio']
        self.abbreviations = self.static_data['clustering']['Gauss_abbreviations']
        self.experiment_tag = self.static_data['clustering']['params']['experiment_tag']
        self.params = self.static_data['clustering']['params']
        self.params['experiment'] = self.static_data['experiments'][self.experiment_tag]
        self.params['scale_nwp_method'] = self.static_data['clustering']['data_type']['scaling']
        self.params['merge'] = self.static_data['clustering']['data_type']['merge']
        self.params['compress'] = self.static_data['clustering']['data_type']['compress']
        self.params['what_data'] = 'row_all'
        self.params['n_rules'] = self.static_data['clustering']['n_rules']
        self.path_fuzzy = os.path.join(static_data['path_model'], 'cluster_organizer', 'RBF')
        try:
            if not self.refit:
                self.load()
        except:
            pass
        self.path_fuzzy = os.path.join(static_data['path_model'], 'cluster_organizer', 'RBF')
        if not os.path.exists(self.path_fuzzy):
            os.makedirs(self.path_fuzzy)

    def compute_activations(self, x, metadata, with_predictions=False):
        self.params['what_data'] = 'row_all'
        if not hasattr(self, 'fuzzy_models'):
            raise ValueError('clusterer is not trained, fuzzy_models is not exists')
        activations = None
        predictions = None
        var_del = []
        for i, fuzzy_model in enumerate(self.fuzzy_models):
            var_imp = fuzzy_model['var_imp']
            var_lin = fuzzy_model['var_lin']
            for var_name in var_imp:
                if var_name not in x.columns:
                    var_names = [c for c in x.columns if var_name.lower() in c.lower()]
                    if len(var_names) == 0:
                        raise ValueError(f'Cannot find variables associated with {var_name}')
                    x[var_name] = x.loc[:, var_names].mean(axis=1)
                    var_del.append(var_name)
            network = DeepNetwork(self.static_data, self.path_fuzzy, self.params, is_global=True, is_fuzzy=True,
                                  refit=False)
            y_pred, act = network.predict(x[var_lin], metadata, X_imp=x[var_imp], with_activations=True)
            activations = act if activations is None else pd.concat([activations, act], axis=1)
            predictions = y_pred if predictions is None else pd.concat([predictions, y_pred], axis=1)

        activations.columns = self.rule_names
        predictions.columns = [f'rbf_clusterer_{i}' for i in range(len(self.fuzzy_models))]
        if len(var_del) > 0:
            x = x.drop(columns=var_del)
        if with_predictions:
            return predictions, activations
        else:
            return activations

    def run(self, x, y, cv_mask, metadata):
        if not self.refit and self.is_trained:
            return
        x_train = pd.concat([sync_data_with_dates(x, cv_mask[0]), sync_data_with_dates(x, cv_mask[1])])
        y_train = pd.concat([sync_data_with_dates(y, cv_mask[0]), sync_data_with_dates(y, cv_mask[1])])

        x_test = sync_data_with_dates(x, cv_mask[2])
        y_test = sync_data_with_dates(y, cv_mask[2])
        self.logger = create_logger(logger_name='log_fuzzy.log', abs_path=self.path_fuzzy,
                                    logger_path='log_fuzzy.log', write_type='a')

        if len(y_test.shape) > 1:
            n_target = y_test.shape[1]
            if n_target > 1:
                self.rated = y_test.values if self.rated is None else 1
            else:
                self.rated = y_test.values.ravel() if self.rated is None else 1
        else:
            n_target = 1
            self.rated = y_test.values if self.rated is None else 1
        var_del = []
        fuzzy_models = []
        activations = None
        for n_case, case in enumerate(self.var_fuzz):
            print(f'{n_case}th Case')
            var_lin = [c for c in x_train.columns[:self.n_var_lin + 1]]
            print(f'Variables for linear regression of {n_case}th Case')
            print(var_lin)
            if isinstance(case, dict):
                raise ValueError('Fuzzy variables should be in list for rbf clustering')

            var_imp = case

            for var_name in case:
                var_names = [c for c in x_train.columns if var_name.lower() in c.lower()]
                if var_name not in x_train.columns:
                    if len(var_names) == 0:
                        raise ValueError(f'Cannot find variables associated with {var_name}')
                    x_train[var_name] = x_train.loc[:, var_names].mean(axis=1)
                    x_test[var_name] = x_test.loc[:, var_names].mean(axis=1)
                    x[var_name] = x.loc[:, var_names].mean(axis=1)
                    var_del.append(var_name)
                    var_lin += var_names
                    if var_name not in var_lin:
                        var_lin.append(var_name)
            var_lin = list(set(var_lin))
            if n_target > 1:
                lin_models = MultiTaskElasticNetCV(cv=5).fit(x_train[var_lin].values, y_train.values)
                pred = lin_models.predict(x_test[var_lin].values)
                err = (pred - y_test.values) / self.rated
            else:
                lin_models = ElasticNetCV(cv=5).fit(x_train[var_lin].values, y_train.values.ravel())
                pred = lin_models.predict(x_test[var_lin].values)
                err = (pred.ravel() - y_test.values.ravel()) / self.rated

            rms_before = np.sqrt(np.mean(np.square(err)))
            mae_before = np.mean(np.abs(err))
            print('rms = %s', rms_before)
            print('mae = %s', mae_before)
            self.logger.info("Objective before train: %s", mae_before)

            self.params['name'] = 'RBF_clustering'
            self.params['var_imp'] = var_imp
            self.params['thres_act'] = self.thres_act
            self.params['min_samples'] = self.min_samples
            self.params['max_samples_ratio'] = self.max_samples_ratio
            self.params['groups'] = metadata['groups']
            self.params['method'] = 'Fuzzy-MLP'
            network = DeepNetwork(self.static_data, self.path_fuzzy, params=self.params, is_global=True, is_fuzzy=True,
                                  is_for_cluster=True, refit=self.refit)

            network.fit(x[var_lin], y, cv_mask, metadata, X_imp=x[var_imp])

            y_pred, act = network.predict(x_test[var_lin], X_imp=x_test[var_imp], with_activations=True)
            activations = act if activations is None else np.concatenate([activations, act], axis=1)
            self.logger.info("Objective after train: %s", str(network.best_mae_test))
            fuzzy_models.append({'var_imp': var_imp, 'var_lin': var_lin})

        self.rule_names = ['rule_' + str(i) for i in range(activations.shape[1])]
        self.fuzzy_models = fuzzy_models
        if len(var_del) > 0:
            x_train = x_train.drop(columns=var_del)
            x_test = x_test.drop(columns=var_del)
        self.is_trained = True
        self.save()

    def load(self):
        if os.path.exists(os.path.join(self.path_fuzzy, 'fuzzy_model.pickle')):
            try:
                f = open(os.path.join(self.path_fuzzy, 'fuzzy_model.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open fuzzy model')
        else:
            raise ImportError('Cannot find fuzzy model')

    def save(self):
        f = open(os.path.join(self.path_fuzzy, 'fuzzy_model.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'refit', 'path_fuzzy']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()
