import os
import joblib
import numpy as np
import pandas as pd
import dask.dataframe as dd

from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.dataset_creation.data_feeder import DataFeeder

CategoricalFeatures = ['hour', 'month', 'sp_index']


class Combiner:
    def __init__(self, static_data, refit=False):
        self.hour_feat_flag = False
        self.dates_kdtree = dict()
        self.X_tr = None
        self.num_feats = []
        self.kdtree = dict()
        self.labels = None
        self.cat_feats = []
        self.metadata = None
        self.X = None
        self.y = None
        self.static_data = static_data
        self.refit = refit
        self.scale_method = self.static_data['combining']['data_type']['scaling']
        self.merge = self.static_data['combining']['data_type']['merge']
        self.compress = self.static_data['combining']['data_type']['compress']
        self.what_data = 'row_all'
        self.problem_type = self.static_data['type']
        self.n_jobs = self.static_data['n_jobs']
        self.predictions = joblib.load(os.path.join(self.static_data['path_data'],
                                                    'predictions_regressors_train.pickle'))
        self.feed_data()
        self.apply_kmeans_X()
        self.apply_kdtree_X()

    def feed_data(self):
        print('Read data for Clustering....')
        data_feeder = DataFeeder(self.static_data, train=True)
        self.X, self.metadata = data_feeder.feed_inputs(merge=self.merge, compress=self.compress,
                                                        scale_nwp_method=self.scale_method,
                                                        what_data=self.what_data)
        self.y, _ = data_feeder.feed_target()
        self.X, self.y = sync_datasets(self.X, self.y, name1='inputs_for_combine', name2='target_for_combine')
        self.metadata['dates'] = self.X.index
        self.cat_feats = [v_name for v_name in self.X.columns
                          for c_feats in CategoricalFeatures if c_feats in v_name]
        self.num_feats = [v_name for v_name in self.X.columns if v_name not in self.cat_feats]

    def apply_kmeans_X(self):
        n_clusters = 5 if self.problem_type == 'pv' else 12
        cat_feats = [f for f in self.cat_feats if 'hour' not in f]
        if len(cat_feats):
            kmeans = KMeans(n_clusters=n_clusters)
            self.labels = pd.Series(kmeans.fit_predict(self.X[cat_feats].values), index=self.X.index, name='labels')

    def apply_kdtree_X(self):
        print('Train KDtree for resampling')
        self.hour_feat_flag = any(['hour' in f for f in self.cat_feats])

        if self.labels is None and not self.hour_feat_flag:
            self.kdtree['main'] = KDTree(self.X.values)
            self.dates_kdtree['main'] = self.X.index
        elif self.labels is not None:
            for label in self.labels.unique():
                dates = self.labels[self.labels == label].index
                name_slice = f'label_{label}'
                if self.hour_feat_flag:
                    for h in range(24):
                        dates_h = dates[dates.hour == h]
                        name_slice_h = name_slice + f'_hour_{h}'
                        leaf_size = 40 if dates_h.shape[0] > 400 else int(dates_h.shape[0] / 10)
                        self.kdtree[name_slice_h] = KDTree(self.X.loc[dates_h].values,
                                                           leaf_size=leaf_size if leaf_size > 2 else 2)
                        self.dates_kdtree[name_slice_h] = dates_h
                else:
                    leaf_size = 40 if dates.shape[0] > 400 else int(dates.shape[0] / 10)
                    self.kdtree[name_slice] = KDTree(self.X.loc[dates].values,
                                                     leaf_size=leaf_size if leaf_size > 2 else 2)
                    self.dates_kdtree[name_slice] = dates

    def resample_y(self, sample):
        date = sample['date']
        if self.labels is not None:
            label = sample['labels']
            if self.hour_feat_flag:
                h = date.hour
                name_slice = f'label_{label}_hour_{h}'
            else:
                name_slice = f'label_{label}'
        else:
            name_slice = 'main'
        kdtree = self.kdtree[name_slice]
        if 'labels' in sample.index:
            sample = sample.drop(index=['labels'])
        sample = sample.drop(index=['date'])

        dist, nearest = kdtree.query(sample.values.reshape(1, -1), k=2, sort_results=True)
        nearest = nearest[0]
        date_loc = self.dates_kdtree[name_slice][nearest[1]]
        y_swap = self.y.loc[date_loc].values

        dist, closest = kdtree.query(sample.values.reshape(1, -1), k=20, sort_results=True)
        closest = closest[0]
        date_loc = self.dates_kdtree[name_slice][closest[1:]]
        ks = KernelDensity(bandwidth=0.01, leaf_size=2)
        ks.fit(self.y.loc[date_loc].values)
        y_ks = ks.sample()[0]

        dist, closest = kdtree.query(sample.values.reshape(1, -1), k=50, sort_results=True)
        closest = closest[0]
        closest = np.random.choice(closest[1:], size=10, replace=False)
        date_loc = self.dates_kdtree[name_slice][closest]
        pca = PCA(n_components=10)
        pca.fit(self.X.loc[date_loc].values)
        X_tr = pca.transform(self.X.loc[date_loc].values)
        lr = LinearRegression().fit(X_tr, self.y.loc[date_loc].values.ravel())
        y_lr = lr.predict(pca.transform(sample.values.reshape(1, -1)))
        if y_lr < np.min(self.y.loc[date_loc].values.ravel()):
            y_lr = np.array([np.min(self.y.loc[date_loc].values.ravel())])
        if y_lr > np.max(self.y.loc[date_loc].values.ravel()):
            y_lr = np.array([np.max(self.y.loc[date_loc].values.ravel())])
        return pd.DataFrame(np.array([y_swap, y_ks, y_lr]).reshape(1, -1), index=[date],
                            columns=['swap', 'kernel_density', 'linear_reg'])

    def resampling(self):
        if self.labels is not None:
            X_ = pd.concat([self.X, self.labels], axis=1)
        else:
            X_ = self.X.copy(deep=True)
        X_['date'] = X_.index
        ddf = dd.from_pandas(X_, npartitions=self.n_jobs)
        ddf_update = ddf.map_partitions(lambda df: df.apply(self.resample_y, axis=1), meta=pd.DataFrame).\
            compute(scheduler='processes')
        res = ddf_update.to_list()
        self.y_resample = pd.concat(res)

        print('found')
