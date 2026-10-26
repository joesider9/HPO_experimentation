import os
import numpy as np
import pandas as pd
import joblib
from deap import base, creator, tools, algorithms
from itertools import repeat
from collections import Sequence
from .rbf_ols_hebo import rbf_ols_module

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

CategoricalFeatures = ['dayweek', 'hour', 'month', 'sp_index']


class RbfOlsManager(object):
    def __init__(self, static_data, path_weights, params=None, n_jobs=1, refit=False):

        self.models = dict()
        self.best_mae_val = None
        self.best_mae_test = None
        self.is_trained = False
        self.refit = refit
        self.static_data = static_data
        self.rated = static_data['rated']
        self.n_jobs = n_jobs
        self.cluster_name = path_weights.split('/')[-1]
        if params is not None:
            self.params = params
            self.method = self.params['method']
            self.name = self.params['name']
            self.merge = self.params['merge']
            self.compress = self.params['compress']
            self.scale_nwp_method = self.params['scale_nwp_method']
            self.groups = self.params['groups']

        self.path_weights = path_weights

        self.refit = refit
        if not hasattr(self, 'params'):
            raise RuntimeError('The network has no params. You should train the net providing params')

    def get_rated(self, y):
        if self.static_data['rated'] is not None:
            norm_val = 1
        else:
            norm_val = y
        return norm_val

    @staticmethod
    def get_slice(x, mask, meta_data, y=None):
        dates = meta_data['dates']
        mask = mask.intersection(dates)
        indices = dates.get_indexer(mask)
        y_slice = y.iloc[indices] if y is not None else None
        if isinstance(x, pd.DataFrame):
            X_slice = x.iloc[indices]
        else:
            raise ValueError('Wrong type of input X for shallow models')
        return X_slice, y_slice

    def fit(self, X, y, cv_masks, meta_data):
        norm_val = self.get_rated(y)
        cat_feats = [v_name for v_name in X.columns
                     for c_feats in CategoricalFeatures if c_feats in v_name]
        X[cat_feats] = X[cat_feats].astype('int')
        X_train, y_train = self.get_slice(X, cv_masks[0], meta_data, y=y)
        X_val, y_val = self.get_slice(X, cv_masks[1], meta_data, y=y)
        X_test, y_test = self.get_slice(X, cv_masks[2], meta_data, y=y)

        model_rbf_ols = rbf_ols_module(self.static_data, self.path_weights, self.static_data['rated'],
                                       self.static_data['n_jobs'], GA=False
                                       , path_group=self.static_data['path_group'])
        model_rbf_ga = rbf_ols_module(self.static_data, self.path_weights, self.static_data['rated'],
                                      self.static_data['n_jobs'], GA=True
                                      , path_group=self.static_data['path_group'])
        if not model_rbf_ols.is_trained:
            max_samples = 1500
            print('Train RBFOLS ', self.cluster_name)
            self.models['RBF_OLS'] = model_rbf_ols.optimize_rbf(X_train.values, y_train.values,
                                                                X_val.values, y_val.values,
                                                                X_test.values, y_test.values,
                                                                max_samples=max_samples, n_jobs=self.static_data['n_jobs'])

        else:
            self.models['RBF_OLS'] = model_rbf_ols.to_dict()
        if not model_rbf_ga.is_trained:
            max_samples = 1500

            print('Train GA-RBF ', self.cluster_name)
            self.models['GA_RBF_OLS'] = model_rbf_ga.optimize_rbf(X_train.values, y_train.values,
                                                                  X_val.values, y_val.values,
                                                                  X_test.values, y_test.values,
                                                                  max_samples=max_samples, n_jobs=self.static_data['n_jobs'])

        else:
            self.models['GA_RBF_OLS'] = model_rbf_ga.to_dict()

        self.is_trained = True
        self.save()
        return 'Done'

    def load(self):
        if os.path.exists(os.path.join(self.path_weights, 'RBFolsManager.pickle')):
            try:
                tmp_dict = joblib.load(os.path.join(self.path_weights, 'RBFolsManager.pickle'))
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open CNN model')
        else:
            raise ImportError('Cannot find CNN model')

    def save(self):
        tmp_dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'path_weights', 'cluster_dir', 'data_dir']:
                tmp_dict[k] = self.__dict__[k]
        joblib.dump(tmp_dict, os.path.join(self.path_weights, 'RBFolsManager.pickle'))
