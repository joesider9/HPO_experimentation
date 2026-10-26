import os

import joblib
import numpy as np
import pandas as pd
import tqdm

from joblib import Parallel
from joblib import delayed

from sklearn.linear_model import ElasticNetCV
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

from Fuzzy_clustering.version4.sklearn_models.sklearn_models_deap import SkLearnModel
from Fuzzy_clustering.version4.data_manager.data_preprocessing_manager.data_organizer import DataOrganizer
from Fuzzy_clustering.version4.data_manager.data_preprocessing_manager.cluster_object import ClusterObject
from Fuzzy_clustering.version4.train_manager.for_prediction.predictors import Predictors


class ClusterCombiner:
    def __init__(self, static_data):
        self.is_trained = False
        self.static_data = static_data
        self.model_dir = os.path.join(self.static_data['path_model'], 'Combine_module')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        try:
            self.load()
        except:
            pass
        self.static_data = static_data
        self.model_dir = os.path.join(self.static_data['path_model'], 'Combine_module')

    def fit(self):
        if not self.is_trained:
            data_organiser = DataOrganizer(self.static_data)
            self.clusters = data_organiser.clusters
            data_organiser.retrieve_validation_data()

            if data_organiser.sampled_data is not None:
                predict_sampled_set = Predictors(self.static_data, self.clusters, data_organiser.sampled_data)
                predict_sampled_set.predict()
                cluster_predictions = predict_sampled_set.cluster_predictions
            else:
                predict_valid_set = Predictors(self.static_data, self.clusters, data_organiser.cluster_data)
                predict_valid_set.predict()
                cluster_predictions = predict_valid_set.cluster_predictions

            for cluster_name, cluster_dir in self.clusters.items():
                combiner = Combiner(self.static_data, cluster_dir)
                if not combiner.is_trained:
                    if data_organiser.sampled_data is not None:
                        print(f'Combiner training for {cluster_name}')
                        combiner.train(cluster_predictions[cluster_name], data_organiser.sampled_data[cluster_name])
                    else:
                        combiner.train(cluster_predictions[cluster_name], data_organiser.cluster_data[cluster_name])
            self.is_trained = True
            self.save()

    def predict(self, data=None, what_data=None):
        if self.is_trained:
            data_organiser = DataOrganizer(self.static_data)
            clusters = data_organiser.clusters
            self.clusters = dict()
            for cluster_name, cluster_dir in clusters.items():
                cluster = ClusterObject(self.static_data, cluster_name)
                self.clusters[cluster_name] = cluster.cluster_dir
            if data is None:
                if what_data == 'testing':
                    data_organiser.distribute_test_data()
                elif what_data == 'validation':
                    data_organiser.retrieve_validation_data()
                elif what_data == 'training':
                    data_organiser.distribute_train_data()
                else:
                    raise NotImplementedError(f'Parameter what_data {what_data} to load data. testing and validation'
                                              f' is only implemented, otherwise provide data')
                cluster_data = data_organiser.cluster_data
            else:
                cluster_data = data
            predict_data = Predictors(self.static_data, self.clusters, cluster_data)
            predict_data.predict()
            cluster_predictions = predict_data.cluster_predictions
            for cluster_name, cluster_dir in self.clusters.items():
                combiner = Combiner(self.static_data, cluster_dir)
                predict_combine, data = combiner.predict(cluster_predictions[cluster_name], cluster_data[cluster_name])
                cluster_data[cluster_name] = data
                if cluster_name == 'global' and self.static_data['satellite_use']:
                    ind = [d for d in cluster_data['target'].index if d not in data['dates']]
                    cluster_data['target'] = cluster_data['target'].drop(ind)
                cluster_predictions[cluster_name].update(predict_combine)
        else:
            raise RuntimeError(f'Cluster combine manager seems to be not trained {self.model_dir}')
        return cluster_predictions, cluster_data

    def load(self):
        if os.path.exists(os.path.join(self.model_dir, 'combine_cluster_manager.pickle')):
            try:
                f = open(os.path.join(self.model_dir, 'combine_cluster_manager.pickle'), 'rb')
                tmp_dict = joblib.load(f)
                f.close()
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open combine_cluster_manager')
        else:
            raise ImportError('Cannot find combine_cluster_manager')

    def save(self):
        f = open(os.path.join(self.model_dir, 'combine_cluster_manager.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'cluster_dir', 'static_data', 'model_dir']:
                dict[k] = self.__dict__[k]
        joblib.dump(dict, f)
        f.close()

from sklearn.ensemble import AdaBoostRegressor
class Combiner:

    def __init__(self, static_data, cluster_dir):
        self.is_trained = False
        self.model_dir = os.path.join(cluster_dir, 'Combine')
        try:
            self.load()
        except:
            pass
        self.static_data = static_data
        self.cluster_dir = cluster_dir
        self.cluster_name = os.path.basename(self.cluster_dir)
        self.model_dir = os.path.join(self.cluster_dir, 'Combine')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_type = static_data['type']
        self.methods = []
        for method in [method for method, values in static_data['project_methods'].items() if values['train']]:
            if method == 'RBFols':
                self.methods.extend(['RBF_OLS', 'GA_RBF_OLS'])
            else:
                self.methods.append(method)
        self.combine_methods = static_data['combine_methods']
        self.n_jobs = static_data['sklearn']['n_jobs']

    @staticmethod
    def compute_metrics(pred, y):
        if len(y.shape) > 1:
            if y.shape[1] == 1:
                y = y.ravel()
        if len(pred.shape) > 1:
            if pred.shape[1] == 1:
                pred = pred.ravel()
        err = np.abs(pred - y)
        sse = np.sum(np.square(pred - y))
        rms = np.sqrt(np.nanmean(np.square(err)))
        mae = np.nanmean(err)
        mse = sse / y.shape[0]
        return [sse, rms, mae, mse]

    def evaluate(self, pred_all, y, horizon):
        result = pd.DataFrame(index=[method for method in pred_all.keys()], columns=['sse', 'rms', 'mae', 'mse'])
        for method, pred in pred_all.items():
            result.loc[method] = self.compute_metrics(pred.values[:, horizon], y[:, horizon])
        return result

    def train(self, predictions, data):
        if self.static_data['satellite_use']:
            if 'CNN' in predictions.keys() or 'UNET' in predictions.keys():
                ind = [i for i, d in enumerate(data['dates']) if d in predictions['CNN'].index]
                data['dates'] = data['dates'][ind]
                data['X'] = data['X'][ind]
                data['y'] = data['y'][ind]
                data['X_extra'] = data['X_extra'][ind] if data['X_extra'] is not None else None
                data['X_cnn'] = data['X_cnn'][ind]
        X = data['X']
        y = data['y']
        if self.static_data['horizon_type'] == 'multi-output':
            horizon = np.arange(self.static_data['horizon'])
        else:
            horizon = [0]
        self.best_of_best = []
        self.combine_models = []
        combine_models = []
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        for hor in horizon:
            results = self.evaluate(predictions, y, hor)
            results = results.astype(float)
            if len(self.methods) > 1 and not self.is_trained:
                best_methods = results.nsmallest(6, 'mae').index.tolist()
                self.best_of_best.append(best_methods)

                X_pred = np.array([])
                for method in sorted(best_methods):
                    if X_pred.shape[0] == 0:
                        X_pred = predictions[method].values[:, hor].reshape(-1, 1)
                    else:
                        X_pred = np.hstack((X_pred, predictions[method].values[:, hor].reshape(-1, 1)))
                X_pred[np.where(X_pred < 0)] = 0
                self.weight_size = len(best_methods)
                model = dict()
                for combine_method in self.combine_methods:
                    if combine_method == 'rls':
                        print('RLS training')
                        model[combine_method] = dict()
                        w = self.rls_fit(X_pred, y[:, hor].reshape(-1, 1))

                        model[combine_method]['w'] = w

                    elif combine_method == 'bcp':
                        print('BCP training')
                        model[combine_method] = dict()
                        w = self.bcp_fit(X_pred, y[:, hor].reshape(-1, 1))
                        model[combine_method]['w'] = w

                    elif combine_method == 'knn':
                        print('classified training')
                        model[combine_method] = dict()
                        best_x = np.argmin(np.abs(X_pred -y[:, hor].reshape(-1, 1)), axis=1)
                        classifier = KNeighborsClassifier(n_neighbors=8)
                        # classifier.fit(np.concatenate([X_pred, X], axis=1), best_x)
                        classifier.fit(X_pred, best_x)
                        model[combine_method] = classifier

                    elif combine_method == 'mlp':
                        print('classified training')
                        model[combine_method] = dict()
                        best_x = np.argmin(np.abs(X_pred -y[:, hor].reshape(-1, 1)), axis=1)
                        parameters = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                        'solver': ['adam', 'lbfgs'], 'alpha': 10.0 ** -np.arange(1, 10, 4), 'hidden_layer_sizes':np.arange(10, 150, 25)}
                        clf = RandomizedSearchCV(MLPClassifier(max_iter=10000), parameters, n_iter=8, n_jobs=8, cv=2)
                        clf.fit(X_pred, best_x)
                        print(clf.best_params_)
                        classifier = clf.best_estimator_
                        # classifier.fit(np.concatenate([X_pred, X], axis=1), best_x)

                        model[combine_method] = classifier

                    elif combine_method == 'kmeans':
                        print('Kmeans training')
                        model[combine_method] = dict()
                        kmeans_model = self.kmeans_combine_fit(X_pred, X, y[:, hor].reshape(-1, 1))
                        model[combine_method] = kmeans_model

                    elif combine_method == 'elastic_net':
                        print('elastic_net training')
                        model[combine_method] = ElasticNetCV(cv=5)
                        model[combine_method].fit(X_pred, y[:, hor])

                    elif combine_method == 'rf':
                        print('rf training')
                        model[combine_method] = AdaBoostRegressor(n_estimators=200)
                        model[combine_method].fit(X_pred, y[:, hor].reshape(-1, 1))
                self.combine_models.append(model)
                print('End of combine models training')
            else:
                self.combine_methods = ['average']
                model = None
                self.combine_models.append(model)
        self.is_trained = True
        self.save()

        return 'Done'

    def predict(self, predictions, data):
        if self.static_data['satellite_use']:
            if 'CNN' in predictions.keys() or 'UNET' in predictions.keys():
                ind = [i for i, d in enumerate(data['dates']) if d in predictions['CNN'].index]
                data['dates'] = data['dates'][ind]
                data['X'] = data['X'][ind]
                data['y'] = data['y'][ind]
                data['X_extra'] = data['X_extra'][ind] if data['X_extra'] is not None else None
                data['X_cnn'] = data['X_cnn'][ind]
        X_inputs = data['X']
        pred_combine = dict()
        if self.static_data['horizon_type'] == 'multi-output':
            horizon = np.arange(self.static_data['horizon'])
        else:
            horizon = [0]
        if self.is_trained:
            if len(self.combine_methods) > 1:
                if not hasattr(self, 'combine_models'):
                    raise ValueError('The combine models does not exist')
            dates = predictions[self.methods[0]].index
            for hor in horizon:
                model = self.combine_models[hor]

                if len(self.methods) > 1:
                    best_methods = self.best_of_best[hor]
                    X_pred = np.array([])
                    for method in sorted(best_methods):
                        if X_pred.shape[0] == 0:
                            X_pred = predictions[method].values[:, hor].reshape(-1, 1)
                        else:
                            X_pred = np.hstack((X_pred, predictions[method].values[:, hor].reshape(-1, 1)))

                    for combine_method in self.combine_methods:
                        if X_pred.shape[0] > 0:
                            if combine_method == 'rls':
                                pred = np.matmul(model[combine_method]['w'], X_pred.T).T
                            elif combine_method in {'knn', 'mlp'}:
                                classifier = model[combine_method]
                                # labels = classifier.predict_proba(np.concatenate([X_pred, X_inputs], axis=1))
                                labels = classifier.predict_proba(X_pred)
                                pred = np.sum(X_pred * labels, axis=1).reshape(-1, 1)

                            elif combine_method == 'kmeans':
                                kmeans_model = model[combine_method]
                                labels = kmeans_model['Kmean'].predict(X_inputs)
                                pred = []
                                for label, x_pred in zip(labels, X_pred):
                                    p = np.matmul(kmeans_model['probs'][str(label)]['model'].predict_proba(x_pred.reshape(1, -1)),
                                                  x_pred[kmeans_model['probs'][str(label)]['predictors']])[0]
                                    if np.isnan(p):
                                        print(p)
                                    pred.append(p)
                                pred = np.array(pred)

                            elif combine_method == 'bcp':
                                pred = np.matmul(model[combine_method]['w'], X_pred.T).T / np.sum(
                                    model[combine_method]['w'])

                            elif combine_method == 'elastic_net':
                                pred = model[combine_method].predict(X_pred)

                            elif combine_method == 'rf':
                                pred = model[combine_method].predict(X_pred)

                            else:
                                pred = np.mean(X_pred, axis=1).reshape(-1, 1)

                            if len(pred.shape) == 1:
                                pred = pred.reshape(-1, 1)

                            pred = np.clip(pred, 0, 1)

                            if combine_method not in pred_combine.keys():
                                pred_combine[combine_method] = pred
                            else:
                                pred_combine[combine_method] = np.hstack((pred_combine[combine_method], pred))
                        else:
                            pred_combine[combine_method] = np.array([])
                else:
                    pred = predictions[self.methods[0]].values[:, hor].reshape(-1, 1)
                    pred[np.where(pred < 0)] = 0
                    if 'average' not in pred_combine.keys():
                        pred_combine['average'] = pred
                    else:
                        pred_combine['average'] = np.hstack((pred_combine['average'], pred))
        else:
            raise ValueError('combine model not trained for %s of %s', self.cluster_name, self.static_data['_id'])
        for combine_method in pred_combine.keys():
            if self.static_data['horizon_type'] == 'multi-output':
                columns = ['h' + str(i) for i in range(self.static_data['horizon'])]
            else:
                columns = [combine_method]
            if dates.shape[0]>0:
                pred_combine[combine_method] = pd.DataFrame(pred_combine[combine_method], index=dates, columns=columns)
            else:
                pred_combine[combine_method] = pd.DataFrame([], columns=columns)
        return pred_combine, data

    def simple_stack(self, x, y):
        if x.shape[0] == 0:
            x = y
        else:
            x = np.vstack((x, y))
        return x

    def rls_func(self, l, weight_size, X, y):
        P = 1e-4 * np.eye(weight_size)
        w = np.ones([1, weight_size]) / weight_size

        count = 0
        err = []
        for inp, targ in tqdm.tqdm(zip(X, y)):
            inp = inp.reshape(-1, 1)
            pred = np.matmul(w, inp)
            e = targ - pred
            err.append(e)
            if len(err) == 1:
                sigma = 0.01
            else:
                sigma = np.square(np.std(np.array(err)))

            c = np.square(e) * np.matmul(np.matmul(np.transpose(inp), P), inp) / (
                    sigma * (1 + np.matmul(np.matmul(np.transpose(inp), P), inp)))

            P = (1 / l) * (P - (np.matmul(np.matmul(P, np.matmul(inp, np.transpose(inp))), P)) / (
                    l + np.matmul(np.matmul(np.transpose(inp), P), inp)))
            w += np.transpose(np.matmul(P, inp) * e)
            if np.sum(w) > 1.05 or np.sum(w) < 0.95:
                w /= np.sum(w)
            count += 1
        return w

    def rls_fit(self, X, y):

        mae_all = []
        w_all = []
        lrs = [0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.9999]
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.rls_func)(l, self.weight_size, X, y) for l in lrs)
        for w in results:
            pred = np.matmul(w, X.T).T
            mae = np.mean(np.abs(pred - y))
            mae_all.append(mae)
            w_all.append(w)
        best = np.argmin(mae_all)

        return w_all[best]

    def kmeans_fit(self, n_cluster, X, X_test, y):
        cluster_model = KMeans(n_clusters=n_cluster).fit(X_test)
        labels = cluster_model.predict(X_test)
        mae_all = []
        probs_all = []
        ratios = [1.1] if self.static_data['type'] != 'load' else [1.25]
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
                    p = KNeighborsClassifier()
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
                if isinstance(probs[str(label)]['model'], np.ndarray):
                    pred.append(np.matmul(probs[str(label)]['model'], x_pred)[0])
                else:
                    pred.append(np.matmul(probs[str(label)]['model'].predict_proba(x_pred.reshape(1, -1)),
                                      x_pred[probs[str(label)]['predictors']])[0])
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
        n_clusters = [1, 3, 5, 8, 10]
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

    def bcp_fit(self, X, y):
        sigma = np.nanstd(y - X, axis=0).reshape(-1, 1)
        err = []
        preds = []
        w = np.ones([1, self.weight_size]) / self.weight_size
        count = 0
        for inp, targ in tqdm.tqdm(zip(X, y)):
            inp = inp.reshape(-1, 1)
            mask = ~np.isnan(inp)
            pred = np.matmul(w[mask.T], inp[mask])
            preds.append(pred)
            e = targ - pred
            err.append(e)

            p = np.exp(-1 * np.square((targ - inp[mask].T) / (np.sqrt(2 * np.pi) * sigma[mask])))
            p = p / sum(p)
            w[mask.T] = ((w[mask.T] * p) / np.sum(w[mask.T] * p))
            w[np.where(w < 0)] = 0
            w /= np.sum(w)

            count += 1
        return w

    def load(self):
        if os.path.exists(os.path.join(self.model_dir, 'combine_models.pickle')):
            try:
                f = open(os.path.join(self.model_dir, 'combine_models.pickle'), 'rb')
                tmp_dict = joblib.load(f)
                f.close()
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open Combiner model')
        else:
            raise ImportError('Cannot find Combiner model')

    def save(self):
        f = open(os.path.join(self.model_dir, 'combine_models.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'cluster_dir', 'static_data', 'model_dir' ]:
                dict[k] = self.__dict__[k]
        joblib.dump(dict, f)
        f.close()
