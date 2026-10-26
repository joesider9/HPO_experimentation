import os
import joblib

import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from catboost import Pool


pd.set_option('display.expand_frame_repr', False)

CategoricalFeatures = ['dayweek', 'hour', 'month', 'sp_index']


class ShallowModel:
    def __init__(self, static_data, path_weights, params=None, n_jobs=1, refit=False):
        self.best_mae_val = None
        self.best_mae_test = None
        self.is_trained = False
        self.refit = refit
        self.static_data = static_data
        self.rated = static_data['rated']
        self.n_jobs = n_jobs
        if params is not None:
            self.params = params
            self.method = self.params['method']
            self.name = self.params['name']
            self.merge = self.params['merge']
            self.compress = self.params['compress']
            self.scale_nwp_method = self.params['scale_nwp_method']
            self.groups = self.params['groups']
            if self.method == 'RF':
                self.model = RandomForestRegressor(n_jobs=self.n_jobs)
            elif self.method == 'CatBoost':
                self.model = CatBoostRegressor(thread_count=self.n_jobs, allow_writing_files=False)
            elif self.method == 'lasso':
                if self.static_data['horizon_type'] == 'multi-output':
                    self.model = MultiTaskLassoCV(max_iter=150000, n_jobs=self.n_jobs)
                else:
                    self.model = LassoCV(max_iter=150000, n_jobs=self.n_jobs)
            else:
                raise ValueError(f'Unknown method {self.method} for shallow models')
            self.best_params = self.model.get_params()
            for param, value in self.params.items():
                if param in self.best_params.keys():
                    self.best_params[param] = value
            self.model.set_params(**self.best_params)
        self.path_weights = path_weights
        try:
            if not self.refit:
                self.load()
        except:
            pass
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

        if self.method in {'lasso', 'RF'}:
            X_train = pd.concat([X_train, X_val]).values
            y_train = pd.concat([y_train, y_val]).values
            if y_train.shape[1] == 1:
                y_train = y_train.ravel()
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
        else:
            self.model.fit(X_train, y_train, cat_features=cat_feats, use_best_model=True, eval_set=[(X_val, y_val)],
                           verbose=False,
                           early_stopping_rounds=100)
            y_pred = self.model.predict(Pool(X_test, cat_features=cat_feats))

        y_test = y_test.values
        if len(y_pred.shape) != len(y_test.shape):
            y_test = y_test.ravel()
        self.best_mae_test = np.mean(np.abs(y_pred - y_test) / norm_val)
        self.is_trained = True
        self.save()
        return np.sum(np.square((y_pred - y_test) / norm_val))

    def predict(self, X, metadata, cluster_dates=None):
        index = X.index
        cat_feats = [v_name for v_name in X.columns for c_feats in CategoricalFeatures if c_feats in v_name]
        X[cat_feats] = X[cat_feats].astype('int')
        cluster_dates = metadata['dates'] if cluster_dates is None else cluster_dates
        X, _ = self.get_slice(X, cluster_dates, metadata)
        if self.static_data['horizon_type'] == 'multi-output':
            cols = [f'hour_ahead_{h}' for h in range(self.static_data['horizon'])]
        else:
            cols = [self.method]
        if self.method == 'CatBoost':
            return pd.DataFrame(self.model.predict(Pool(X, cat_features=cat_feats)), index=cluster_dates, columns=cols)
        else:
            return pd.DataFrame(self.model.predict(X.values), index=cluster_dates, columns=cols)

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
        joblib.dump(tmp_dict, os.path.join(self.path_weights, 'net_weights.pickle'))
