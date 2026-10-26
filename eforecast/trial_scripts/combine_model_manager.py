import logging
import os

import joblib
import numpy as np
import pandas as pd
import tqdm

from joblib import Parallel
from joblib import delayed
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from Fuzzy_clustering.version4.train_manager.combine_manager.combine_cluster_manager import ClusterCombiner
from Fuzzy_clustering.version4.train_manager.common_utils.utils_for_forecast import calculate_cpus


class ModelCombiner:
    def __init__(self, static_data):
        self.predictions = dict()
        self.models = []
        self.is_trained = False
        self.static_data = static_data
        self.rated = static_data['rated']
        self.model_dir = os.path.join(self.static_data['path_model'], 'Combine_module')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        try:
            self.load()
        except:
            pass
        self.static_data = static_data
        self.model_dir = os.path.join(self.static_data['path_model'], 'Combine_module')
        self.combine_methods = self.static_data['combine_methods']
        self.methods = []
        for method in [method for method, values in static_data['project_methods'].items() if values['train']]:
            if method == 'RBFols':
                self.methods.extend(['RBF_OLS', 'GA_RBF_OLS'])
            elif method == 'RBF-CNN':
                self.methods.extend(['RBF-CNN', 'RBF-RF'])
            else:
                self.methods.append(method)

        self.data_dir = self.static_data['path_data']
        self.n_jobs = static_data['sklearn']['n_jobs']

    def concat_predictions(self, pred_cluster, cluster_data):
        predictions = dict()

        result_clust = pd.DataFrame()
        if 'target' in cluster_data.keys():
            dates_all = cluster_data['target'].index
        elif 'inputs' in cluster_data.keys():
            dates_all = cluster_data['inputs'].index
        else:
            raise ImportError('Error in cluster_data. cannot found target or inputs')
        for clust in pred_cluster.keys():
            dates = cluster_data[clust]['dates']
            for method in pred_cluster[clust].keys():
                if method not in {'dates', 'index', 'metrics'}:
                    if method not in predictions.keys():
                        if self.static_data['horizon_type'] == 'multi-output':
                            predictions[method] = [pd.DataFrame(index=dates_all,
                                                                columns=[cl for cl in pred_cluster.keys()]) for _ in
                                                   range(self.static_data['horizon'])]
                        else:
                            predictions[method] = [pd.DataFrame(index=dates_all,
                                                                columns=[cl for cl in pred_cluster.keys()])]
                    if self.static_data['horizon_type'] == 'multi-output':
                        for h in range(self.static_data['horizon']):
                            predictions[method][h].loc[dates, clust] = pred_cluster[clust][method]['h' + str(h)].values.ravel()
                    else:
                        predictions[method][0].loc[dates, clust] = pred_cluster[clust][
                            method].values.ravel()
                elif method in {'metrics'}:
                    result_clust = pd.concat([result_clust, pred_cluster[clust][method]['mae'].rename(clust)], axis=1)

        if result_clust.shape[0] > 0:
            result_clust.to_csv(os.path.join(self.data_dir, 'result_of_clusters.csv'))
        return predictions

    def fit(self):
        self.models = []
        if len(self.methods) > 1 and not self.is_trained:
            if self.static_data['horizon_type'] == 'multi-output':
                horizon = np.arange(self.static_data['horizon'])
            else:
                horizon = [0]
            cluster_combiner = ClusterCombiner(self.static_data)
            cluster_predictions, cluster_data = cluster_combiner.predict(what_data='training')

            predictions = self.concat_predictions(cluster_predictions, cluster_data)

            if 'target' in cluster_data.keys():
                y = cluster_data['target']
            else:
                raise ImportError('Error in cluster_data. cannot found target or inputs')
            if 'inputs' in cluster_data.keys():
                X = cluster_data['inputs']
            else:
                raise ImportError('Error in cluster_data. cannot found target or inputs')

            for hor in horizon:
                models = dict()
                combine_method = 'bcp'

                for method in self.combine_methods:
                    print(f'Train combine method {combine_method} and method {method}')
                    models['bcp_' + method] = self.bcp_fit(predictions[method][hor].values.astype('float'),
                                                           y.values[:, hor].reshape(-1, 1))
                combine_method = 'kmeans'

                for method in self.combine_methods:
                    print(f'Train combine method {combine_method} and method {method}')
                    models['kmeans_' + method] = self.kmeans_combine_fit(predictions[method][hor].values.astype('float'), X.values,
                                                       y.values[:, hor].reshape(-1, 1))
                self.models.append(models)
        else:
            self.combine_methods = ['average']
            models = None
            self.models.append(models)
        self.is_trained = True
        self.save()

    def predict(self, data=None, what_data=None, return_y=False):
        if self.is_trained:
            cluster_combiner = ClusterCombiner(self.static_data)
            cluster_predictions, cluster_data = cluster_combiner.predict(data=data, what_data=what_data)

            X = cluster_data['inputs']

            predictions = self.concat_predictions(cluster_predictions, cluster_data)

            combine_method = 'average'
            for method in self.methods + self.combine_methods:
                if method in predictions.keys():
                    if self.static_data['horizon_type'] == 'multi-output':
                        columns = ['h' + str(i) for i in range(self.static_data['horizon'])]
                    else:
                        columns = ['average_' + method]

                    self.predictions['average_' + method] = pd.DataFrame(index=predictions[method][0].index, columns=columns)
                    for hor, col in enumerate(columns):
                        pred = predictions[method][hor].mean(axis=1).astype('float')
                        pred = pred.clip(0, 1)
                        self.predictions['average_' + method][col] = pred

            combine_method = 'bcp'
            if self.models[0] is not None:
                for method in self.combine_methods:
                    if method in predictions.keys():
                        if self.static_data['horizon_type'] == 'multi-output':
                            columns = ['h' + str(i) for i in range(self.static_data['horizon'])]
                        else:
                            columns = ['bcp_' + method]
                        self.predictions['bcp_' + method] = pd.DataFrame(index=predictions[method][0].index,
                                                                          columns=columns)

                        for hor, col in enumerate(columns):
                            if 'bcp_' + method in self.models[hor].keys():
                                pred = self.bcp_predict(predictions[method][hor].values.astype('float')
                                                        , self.models[hor]['bcp_' + method])
                                pred = np.clip(pred, 0, 1)
                                self.predictions['bcp_' + method][col] = pred
            combine_method = 'kmeans'
            if self.models[0] is not None:
                for method in self.combine_methods:
                    if method in predictions.keys():
                        if self.static_data['horizon_type'] == 'multi-output':
                            columns = ['h' + str(i) for i in range(self.static_data['horizon'])]
                        else:
                            columns = ['kmeans_' + method]
                        self.predictions['kmeans_' + method] = pd.DataFrame(index=predictions[method][0].index,
                                                                         columns=columns)

                        for hor, col in enumerate(columns):
                            if 'kmeans_' + method in self.models[hor].keys():
                                pred = self.kmeans_combine_predict(predictions[method][hor].values.astype('float'), X
                                                        , self.models[hor]['kmeans_' + method], 1)
                                pred = np.clip(pred, 0, 1)
                                self.predictions['kmeans_' + method][col] = pred

        else:
            raise ImportError('Combine overall model seems not trained')
        if 'target' in cluster_data.keys():
            self.predictions['target'] = cluster_data['target']
        else:
            scale_y = joblib.load(os.path.join(self.static_data['path_data'], 'Y_scaler.pickle'))
            for key, values in self.predictions.items():
                self.predictions[key] = pd.DataFrame(scale_y.inverse_transform(values.values), index=values.index,
                                                     columns=values.columns)

    def create_dataset_for_probabilistic(self, what_data='training'):
        cluster_combiner = ClusterCombiner(self.static_data)
        cluster_predictions, cluster_data = cluster_combiner.predict(what_data=what_data)
        self.predict(what_data=what_data)
        return self.predictions, cluster_predictions, cluster_data

    @staticmethod
    def compute_metrics(pred, y, rated):
        if len(y.shape) > 1:
            if y.shape[1] == 1:
                y = y.ravel()
        if len(pred.shape) > 1:
            if pred.shape[1] == 1:
                pred = pred.ravel()
        rated = y if rated is None else rated
        err = np.abs(pred - y) / rated
        sse = np.sum(np.square(err))
        rms = np.sqrt(np.nanmean(np.square(err)))
        mae = np.nanmean(err)
        mse = sse / y.shape[0]
        return [sse, rms, mae, mse]

    def evaluate(self, what_data='testing'):
        self.predict(what_data=what_data)
        scale_y = joblib.load(os.path.join(self.static_data['path_data'], 'Y_scaler.pickle'))
        for key, values in self.predictions.items():
            self.predictions[key] = pd.DataFrame(scale_y.inverse_transform(values.values), index=values.index, columns=values.columns)
        y = self.predictions['target'].sort_index()
        if self.static_data['horizon_type'] == 'multi-output':
            columns = ['mae_h' + str(i) for i in range(self.static_data['horizon'])]
        else:
            columns = ['sse', 'rms', 'mae', 'mse']
        result = pd.DataFrame(index=[method for method in self.predictions.keys() if method != 'target'], columns=columns)
        for method, pred in self.predictions.items():
            if method != 'target':
                pred = pred.sort_index()
                if self.static_data['horizon_type'] == 'multi-output':
                    for i in range(self.static_data['horizon']):
                        try:
                            result.loc[method, 'mae_h' + str(i)] = self.compute_metrics(pred['h' + str(i)].values
                                                                                    , y['targ_' + str(i)].values, self.rated)[2]
                        except:
                            result.loc[method, 'mae_h' + str(i)] = self.compute_metrics(pred['h' + str(i)].values
                                                                                        , y['targ_h' + str(i)].values, self.rated)[2]
                else:
                    result.loc[method] = self.compute_metrics(pred.values, y.values, self.rated)

        result.to_csv(os.path.join(self.data_dir, f'result_final{what_data}.csv'))
        joblib.dump(self.predictions, os.path.join(self.data_dir, f'predictions_final{what_data}.pickle'))

    @staticmethod
    def bcp_predict(X, w):
        preds = []
        for inp in X:
            inp = inp.reshape(-1, 1)
            mask = ~np.isnan(inp)
            pred = np.matmul(w[mask.T] / np.sum(w[mask.T]), inp[mask])
            preds.append(pred)

        return np.array(preds)

    @staticmethod
    def bcp_fit(X, y):
        sigma = np.nanstd((y - X).astype(float), axis=0).reshape(-1, 1)
        err = []
        preds = []
        w = np.ones([1, X.shape[1]]) / X.shape[1]
        count = 0
        for inp, targ in tqdm.tqdm(zip(X, y)):
            inp = inp.reshape(-1, 1)
            mask = ~np.isnan(inp)
            pred = np.matmul(w[mask.T] / np.sum(w[mask.T]), inp[mask])
            preds.append(pred)
            e = targ - pred
            err.append(e)

            p = np.exp(-1 * np.square((targ - inp[mask].T) / (np.sqrt(2 * np.pi) * sigma[mask])))
            w[mask.T] = ((w[mask.T] * p) / np.sum(w[mask.T] * p))

            count += 1
        return w

    def find_best(self, x1, y1, size, ratio=1.25):
        count = np.zeros([1, size])
        mask = np.where(~np.isnan(x1))[0]
        count[0, mask] = count[0, mask] + 1
        p1 = np.zeros([1, size])
        err = x1[mask] - y1
        ind_pos = np.where(err>=0)[0]
        ind_neg = np.where(err<0)[0]
        if ind_pos.shape[0] > 0:
            i_pos = np.array([ind_pos[np.argmin(err[ind_pos])]])
        else:
            i_pos = np.array([])
        if ind_neg.shape[0] > 0:
            i_neg = np.array([ind_neg[np.argmin(np.abs(err[ind_neg]))]])
        else:
            i_neg = np.array([])
        if i_pos.shape[0] > 0 and i_neg.shape[0] > 0:
            err_min = np.minimum(err[i_pos], np.abs(err[i_neg]))
            ind_min = np.array([i_pos[0], i_neg[0]])
            i_min = np.where(np.abs(err[ind_min]) <= ratio * err_min)[0]
            p1[0, mask[ind_min[i_min]]] = 1
        elif i_pos.shape[0] > 0 and not i_neg.shape[0] > 0:
            p1[0, mask[i_pos[0]]] = 1
        elif not i_pos.shape[0] > 0 and i_neg.shape[0] > 0:
            p1[0, mask[i_neg[0]]] = 1
        else:
            raise ValueError('None best found')

        return p1, count

    def kmeans_fit(self, n_cluster, X, X_test, y):
        cluster_model = KMeans(n_clusters=n_cluster).fit(X_test)
        labels = cluster_model.predict(X_test)
        mae_all = []
        probs_all = []
        ratios = [1.1] if self.static_data['type'] != 'load' else [1.25]
        X = np.nan_to_num(X, nan=-1)
        for ratio in ratios:
            probs = dict()
            for label in range(cluster_model.n_clusters):
                probs[str(label)] = dict()
                ind = np.where(labels == label)[0]
                if ind.shape[0] > 10:
                    x_ = X[ind]
                    y_ = y[ind]
                    best_x = np.argmin(np.abs(x_-y_), axis=1)
                    predictors = [predictor for predictor in range(x_.shape[1]) if predictor in best_x]
                    p = KNeighborsClassifier(n_neighbors=12)
                    try:
                        p.fit(x_, best_x)
                    except:
                        return np.inf, None, None
                else:
                    return np.inf, None, None
                probs[str(label)]['model'] = p
                probs[str(label)]['predictors'] = predictors
            pred = []
            for label, x_pred in zip(labels, X):
                pr = probs[str(label)]['model'].predict_proba(x_pred.reshape(1, -1))
                p = x_pred[probs[str(label)]['predictors']]
                ind = np.where(p != -1)
                pred.append(np.sum(pr[0, ind]*p[ind]) / np.sum(pr[0, ind]))

            pred = np.array(pred)
            mae = np.mean(np.abs(pred - y))
            mae_all.append(mae)
            probs_all.append(probs)
        best = np.argmin(mae_all)
        print(f'Best ratio {ratios[best]} for n clusters {n_cluster}')
        return mae_all[best], probs_all[best], cluster_model


    def kmeans_combine_fit(self, X, X_test, y):
        kmeans_models = []
        probs_all = []
        mae_all = []

        print('Kmeans fit for several n_clusters')
        n_clusters = [5, 8, 10, 12, 16]
        # results = Parallel(n_jobs=self.n_jobs)(
        #     delayed(self.kmeans_fit)(n_cluster, X, X_test, y) for n_cluster in n_clusters)
        for n_cluster in n_clusters:
            res = self.kmeans_fit(n_cluster, X, X_test, y)
            mae_all.append(res[0])
            probs_all.append(res[1])
            kmeans_models.append(res[2])
        best = np.argmin(mae_all)
        print(f'Best ncluster {n_clusters[best]}')
        model = {
            'probs': probs_all[best],
            'Kmean': kmeans_models[best]
        }
        return model


    def kmeans_combine_predict(self, X, X_test, model, n_jobs):
        probs = model['probs']
        cluster_model = model['Kmean']
        labels = cluster_model.predict(X_test)
        X = np.nan_to_num(X, nan=-1)
        # pred = Parallel(n_jobs=n_jobs)(
        #     delayed(self.kmeans_matmul)(x_pred, label, probs) for label, x_pred in zip(labels, X))
        pred = []
        for label, x_pred in zip(labels, X):
            pr = probs[str(label)]['model'].predict_proba(x_pred.reshape(1, -1))
            p = x_pred[probs[str(label)]['predictors']]
            ind = np.where(p != -1)
            pred.append(np.sum(pr[0, ind]*p[ind]) / np.sum(pr[0, ind]))
        pred = np.array(pred)
        return pred

    def load(self):
        if os.path.exists(os.path.join(self.model_dir, 'combine_models.pickle')):
            try:
                f = open(os.path.join(self.model_dir, 'combine_models.pickle'), 'rb')
                tmp_dict = joblib.load(f)
                f.close()
                del tmp_dict['model_dir']
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open RLS model')
        else:
            raise ImportError('Cannot find RLS model')

    def save(self):
        f = open(os.path.join(self.model_dir, 'combine_models.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'cluster_dir']:
                dict[k] = self.__dict__[k]
        joblib.dump(dict, f)
        f.close()
