import os
import joblib

import numpy as np
import pandas as pd

from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.dataset_utils import sync_target_with_tensors
from eforecast.common_utils.dataset_utils import sync_data_with_dates
from eforecast.dataset_creation.files_manager import FilesManager
from eforecast.dataset_creation.data_feeder import DataFeeder

from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from catboost import Pool

CategoricalFeatures = ['dayweek', 'hour', 'month', 'sp_index']


class FeatureSelector:
    def __init__(self, static_data, recreate=False, online=False, train=False):
        self.feature_selectors = dict()
        self.online = online
        self.train = train
        self.static_data = static_data
        self.is_Fuzzy = self.static_data['is_Fuzzy']
        if self.is_Fuzzy:
            self.clusters = joblib.load(os.path.join(static_data['path_model'], 'clusters.pickle'))
            for cluster_name, cluster_path in self.clusters.items():
                filename = os.path.join(cluster_path, 'feature_selectors.pickle')
                if os.path.exists(filename):
                    if recreate:
                        self.feature_selectors = dict()
                        if os.path.exists(filename):
                            os.remove(filename)
                    else:
                        self.feature_selectors.update(joblib.load(filename))
        self.is_Global = self.static_data['is_Global']
        if self.is_Global:
            cluster_path = os.path.join(static_data['path_model'], 'Distributed')
            self.clusters.update({'Distributed': cluster_path})
            filename = os.path.join(cluster_path, 'feature_selectors.pickle')
            if os.path.exists(filename):
                if recreate:
                    self.feature_selectors = dict()
                    if os.path.exists(filename):
                        os.remove(filename)
                else:
                    self.feature_selectors.update(joblib.load(filename))
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']
        self.scale_nwp_method = self.static_data['scale_nwp_method']
        self.data_structure = self.static_data['data_structure']
        self.feature_selection_methods = self.static_data['feature_selection_methods']
        self.files_manager = FilesManager(static_data, is_online=online)

    def load_data(self, merge, compress, scale_method, what_data):
        data_feeder = DataFeeder(self.static_data, online=self.online, train=self.train)

        X, metadata = data_feeder.feed_inputs(merge=merge, compress=compress,
                                              scale_nwp_method=scale_method,
                                              what_data=what_data)
        y, _ = data_feeder.feed_target()
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

    @staticmethod
    def estimator(method, alpha=None, multi_output=False):
        if method == 'lasso':
            if alpha is None:
                if multi_output:
                    return MultiTaskLassoCV(max_iter=150000, n_jobs=-1)
                else:
                    return LassoCV(max_iter=150000, positive=True, n_jobs=-1)
            else:
                if multi_output:
                    return MultiTaskLasso(max_iter=150000, alpha=alpha)
                else:
                    return Lasso(max_iter=150000, positive=True, alpha=alpha)
        elif method == 'FeatureImportance':
            return RandomForestRegressor(n_estimators=1000, n_jobs=-1)
        elif method == 'ShapValues':
            return CatBoostRegressor(iterations=600, allow_writing_files=False)
        else:
            raise ValueError(f'Unknown feature selection method {method}')

    @staticmethod
    def importance_vector(method, estimator, train_pool=None):
        if method == 'lasso':
            return np.abs(estimator.coef_)
        elif method == 'FeatureImportance':
            return estimator.feature_importances_
        elif method == 'ShapValues':
            return estimator.get_feature_importance(data=train_pool,
                                                    type=method,
                                                    prettified=True).abs().mean(axis=0).values[:-1]
        else:
            raise ValueError(f'Unknown feature selection method {method}')

    def fit_method(self, method, x_train, y_train, x_test, y_test):
        if y_train.shape[1] > 1:
            cols = y_train.columns
            multi_output = True
        else:
            cols = y_train.columns[0]
            multi_output = False
        feature_selector = dict()
        thresholds = [0] + np.logspace(-6, -1, 6).tolist() + [1]
        selector = self.estimator(method, multi_output=multi_output)
        cat_feats = [v_name for v_name in x_train.columns
                     for c_feats in CategoricalFeatures if c_feats in v_name]
        x_train[cat_feats] = x_train[cat_feats].astype('int')
        if method == 'ShapValues':
            selector.fit(x_train, y_train[cols], cat_features=cat_feats, verbose=False)
        else:
            selector.fit(x_train, y_train[cols])
        if method == 'lasso':
            alpha = selector.alpha_
        else:
            alpha = None
        mae = []
        importance = self.importance_vector(method, selector, Pool(x_train, y_train[cols], cat_features=cat_feats))
        for threshold in thresholds:
            indices = np.where(importance > threshold)[0]
            if indices.shape[0] > 4:
                names = x_train.columns[indices]
                selector_temp = self.estimator(method, alpha=alpha, multi_output=multi_output)
                cat_feats = [v_name for v_name in names for c_feats in CategoricalFeatures if c_feats in v_name]
                x_train[cat_feats] = x_train[cat_feats].astype('int')
                x_test[cat_feats] = x_test[cat_feats].astype('int')
                if method == 'ShapValues':
                    selector_temp.fit(x_train[names], y_train[cols], cat_features=cat_feats, verbose=False)
                    pred = selector_temp.predict(Pool(x_test[names], cat_features=cat_feats))
                else:
                    selector_temp.fit(x_train[names], y_train[cols])
                    pred = selector_temp.predict(x_test[names])
                mae.append(np.mean(np.abs(pred - y_test[cols].values)))
            else:
                mae.append(np.inf)
        best_threshold = thresholds[np.argmin(mae)]
        feature_indices = np.where(importance > best_threshold)[0]
        feature_names = x_train.columns[feature_indices]
        feature_selector['indices'] = feature_indices
        feature_selector['names'] = feature_names
        return feature_selector

    def _fit(self, x, y, cv_mask, dataset_name, cluster_name, feature_selectors):
        x_train = pd.concat([sync_data_with_dates(x, cv_mask[0]), sync_data_with_dates(x, cv_mask[1])])
        y_train = pd.concat([sync_data_with_dates(y, cv_mask[0]), sync_data_with_dates(y, cv_mask[1])])

        x_test = sync_data_with_dates(x, cv_mask[2])
        y_test = sync_data_with_dates(y, cv_mask[2])

        for method in self.feature_selection_methods:
            if method is not None:
                if f'feature_selector_{cluster_name}_{method}_{dataset_name}' not in self.feature_selectors.keys():
                    print(f'Fitting feature_selector_{cluster_name}_{method}_{dataset_name}')
                    feature_selectors[f'feature_selector_{cluster_name}_{method}_{dataset_name}'] = \
                        self.fit_method(method, x_train, y_train, x_test, y_test)
        return feature_selectors

    def fit(self):
        data_structure = [data_struct for data_struct in self.data_structure
                          if data_struct in {'row', 'row_all',
                                             'row_dict',
                                             'row_dict_distributed'}]
        if len(data_structure) == 0:
            raise ValueError('Cannot find what_data structure to fit MLP')
        for cluster_name, cluster_path in self.clusters.items():
            feature_selectors = dict()
            if cluster_name == 'Distributed':
                cv_mask = self.files_manager.check_if_exists_cv_data()
            else:
                cv_mask = joblib.load(os.path.join(cluster_path, 'cv_mask.pickle'))
            for merge in self.nwp_data_merge:
                for compress in self.nwp_data_compress:
                    for scale_method in self.scale_nwp_method:
                        for what_data in data_structure:
                            x, y, metadata = self.load_data(merge, compress, scale_method, what_data)
                            groups = metadata['groups']
                            if len(groups) == 0 or what_data == 'row_all':
                                dataset_name = f'{merge}_{compress}_{scale_method}_{what_data}'
                                feature_selectors = self._fit(x, y, cv_mask, dataset_name, cluster_name,
                                                              feature_selectors)
                            else:
                                for group in groups:
                                    group_name = '_'.join(group) if isinstance(group, tuple) else group
                                    dataset_name = f'{merge}_{compress}_{scale_method}_{group_name}_{what_data}'
                                    feature_selectors = self._fit(x[group_name], y, cv_mask, dataset_name, cluster_name,
                                                                  feature_selectors)
            self.save(cluster_path, feature_selectors)

    @staticmethod
    def save(cluster_path, feature_selectors):
        filename = os.path.join(cluster_path, 'feature_selectors.pickle')
        joblib.dump(feature_selectors, filename)

    def transform(self, cluster_name, data, method, merge, compress, scale_method, what_data, groups=None):
        if method is not None:
            if len(groups) == 0 or what_data == 'row_all':
                dataset_name = f'{merge}_{compress}_{scale_method}_{what_data}'
                feature_selector = self.feature_selectors[f'feature_selector_{cluster_name}_{method}_{dataset_name}']
                data_new = data[feature_selector['names']]
            else:
                data_new = dict()
                for group in groups:
                    group_name = '_'.join(group) if isinstance(group, tuple) else group
                    dataset_name = f'{merge}_{compress}_{scale_method}_{group_name}_{what_data}'
                    feature_selector = self.feature_selectors[
                        f'feature_selector_{cluster_name}_{method}_{dataset_name}']
                    data_new[group_name] = data[group_name][feature_selector['names']]
        else:
            data_new = data
        return data_new
