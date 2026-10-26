import numpy as np
import pandas as pd


class RBFols:
    def __init__(self, engine, table):
        self.engine = engine
        self.table = table
        self.model = dict()

    def predict(self, x):
        return self._predict(self.model, x)

    def _predict(self, model, x):
        v = (np.atleast_2d(x)[:, np.newaxis] - model['centroids'][np.newaxis, :]) * model['Radius']
        v = np.sqrt((v ** 2.).sum(-1))
        v = np.exp(-(v ** 3.))
        if isinstance(model['W'], np.ndarray):
            v = np.matmul(v, model['W'][:-1]) + model['W'][-1]
        else:
            v = model['W'].predict(v)
        return v

    def compute_metrics(self, pred, y, rated):
        if rated is None:
            rated = y.ravel()
        else:
            rated = 1
        if y.shape[1] > 1:
            err = np.abs((pred - y).ravel()) / rated
        else:
            err = np.abs(pred.ravel() - y.ravel()) / rated

        sse = np.sum(np.square(err))
        rms = np.sqrt(np.mean(np.square(err)))
        mae = np.mean(err)
        mse = sse / y.shape[0]

        return sse, rms, mae, mse

    def _distance(self, obj_new, obj_old, obj_max, obj_min):
        if np.any(np.isinf(obj_old)):
            obj_old = obj_new.copy()
            obj_max = obj_new.copy()
            return True, obj_old, obj_max, obj_min
        if np.any(np.isinf(obj_min)) and not np.all(obj_max == obj_new):
            obj_min = obj_new.copy()
        d = 0
        for i in range(obj_new.shape[0]):
            if obj_max[i] < obj_new[i]:
                obj_max[i] = obj_new[i]
            if obj_min[i] > obj_new[i]:
                obj_min[i] = obj_new[i]

            d += (obj_new[i] - obj_old[i]) / (obj_max[i] - obj_min[i])
        if d < 0:
            obj_old = obj_new.copy()
            return True, obj_old, obj_max, obj_min
        else:
            return False, obj_old, obj_max, obj_min

    def fit(self, I, O, X_val, y_val, X_test, y_test, rated, gw=1.0, keep=3, verbose=False):
        obj_old = np.inf * np.ones(4)
        obj_max = np.inf * np.ones(4)
        obj_min = np.inf * np.ones(4)
        model = dict()
        model_temp = dict()
        remains = 0
        mse_best = np.inf
        k = np.sqrt(-np.log(0.5)) / gw
        m, d = O.shape
        d *= m
        idx = np.arange(m)
        P = np.exp(-(np.sqrt((((I[np.newaxis, :] - I[:, np.newaxis]) * k) ** 2.0).sum(-1))) ** 2.0)
        G = np.array(P)
        D = (O * O).sum(0)
        e = ((np.matmul(P.T, O) ** 2.) / np.matmul((P * P).sum(0)[:, np.newaxis], D[np.newaxis, :])).sum(1)
        next = e.argmax()
        used = np.array([next])
        idx = np.delete(idx, next)
        W = P[:, next, np.newaxis]
        P = np.delete(P, next, 1)
        G1 = G[:, used]
        v = (np.atleast_2d(I)[:, np.newaxis] - I[used][np.newaxis, :]) * k
        v = np.sqrt((v ** 2.).sum(-1))
        v = np.exp(-(v ** 3.))
        try:
            out_layer = np.linalg.lstsq(np.hstack([v, np.ones([v.shape[0], 1])]), O, rcond=None)[0]
            # out_layer = DecisionTreeRegressor(random_state=42)
            # out_layer.fit(v, O)
        except:
            out_layer = np.linalg.lstsq(np.hstack([v, np.ones([v.shape[0], 1])]), O, rcond=None)[0]
        centers = I[used]
        ibias = k
        model_temp['centroids'] = I[used]
        model_temp['W'] = out_layer
        model_temp['Radius'] = k

        pred_test = self._predict(model_temp, X_test)
        pred_val = self._predict(model_temp, X_val)
        sse_test, rms_test, mae_test, mse_test = self.compute_metrics(pred_test, y_test, rated)
        sse_val, rms_val, mae_val, mse_val = self.compute_metrics(pred_val, y_val, rated)
        metrics = np.array(
            [0.6 * mae_test + 0.4 * mae_val, 0.6 * sse_test + 0.4 * sse_val, 0.6 * rms_test + 0.4 * rms_val,
             0.6 * mse_test + 0.4 * mse_val])

        flag, obj_old, obj_max, obj_min = self._distance(metrics, obj_old, obj_max, obj_min)
        if not flag:
            remains += 1
        else:
            remains = 0
            centers = I[used]
            ibias = k
        while remains < keep and P.shape[1] > 0:
            if verbose:
                print(0.6 * mae_test + 0.4 * mae_val, m - P.shape[1])
            wj = W[:, -1:]
            a = np.matmul(wj.T, P) / np.matmul(wj.T, wj)
            P = P - wj * a
            not_zero = np.ones((P.shape[1])) * np.finfo(np.float64).eps
            e = ((np.matmul(P.T, O) ** 2.) / np.matmul((P * P).sum(0)[:, np.newaxis], D[np.newaxis, :])).sum(1)
            next = e.argmax()
            W = np.append(W, P[:, next, np.newaxis], axis=1)
            used = np.append(used, idx[next])
            P = np.delete(P, next, 1)
            idx = np.delete(idx, next)
            v = (np.atleast_2d(I)[:, np.newaxis] - I[used][np.newaxis, :]) * ibias
            v = np.sqrt((v ** 2.).sum(-1))
            v = np.exp(-(v ** 3.))
            try:
                out_layer = np.linalg.lstsq(np.hstack([v, np.ones([v.shape[0], 1])]), O, rcond=None)[0]
            except:
                out_layer = np.linalg.lstsq(np.hstack([v, np.ones([v.shape[0], 1])]), O, rcond=None)[0]
            model_temp['centroids'] = I[used]
            model_temp['W'] = out_layer
            model_temp['Radius'] = k
            pred_test = self._predict(model_temp, X_test)
            pred_val = self._predict(model_temp, X_val)
            sse_test, rms_test, mae_test, mse_test = self.compute_metrics(pred_test, y_test, rated)
            sse_val, rms_val, mae_val, mse_val = self.compute_metrics(pred_val, y_val, rated)
            metrics = np.array(
                [0.6 * mae_test + 0.4 * mae_val, 0.6 * sse_test + 0.4 * sse_val, 0.6 * rms_test + 0.4 * rms_val,
                 0.6 * mse_test + 0.4 * mse_val])

            flag, obj_old, obj_max, obj_min = self._distance(metrics, obj_old, obj_max, obj_min)
            if not flag:
                remains += 1
            else:
                remains = 0
                centers = I[used]
                ibias = k
        if centers.shape[0] < keep:
            centers = I[used]
            ibias = k

        if verbose:
            print(0.6 * mae_test + 0.4 * mae_val, m - P.shape[1])
        model['centroids'] = centers
        model['Radius'] = ibias

        v = (np.atleast_2d(I)[:, np.newaxis] - model['centroids'][np.newaxis, :]) * model['Radius']
        v = np.sqrt((v ** 2.).sum(-1))
        v = np.exp(-(v ** 3.))
        try:
            out_layer = np.linalg.lstsq(np.hstack([v, np.ones([v.shape[0], 1])]), O, rcond=None)[0]
        except:
            return np.inf
        model['W'] = out_layer
        pred_test = self._predict(model, X_test)
        pred_val = self._predict(model, X_val)
        sse_test, rms_test, mae_test, mse_test = self.compute_metrics(pred_test, y_test, rated)
        sse_val, rms_val, mae_val, mse_val = self.compute_metrics(pred_val, y_val, rated)
        metrics = np.array(
            [0.6 * mae_test + 0.4 * mae_val, 0.6 * sse_test + 0.4 * sse_val, 0.6 * rms_test + 0.4 * rms_val,
             0.6 * mse_test + 0.4 * mse_val])
        self.model = model
        self.ncenters = model['centroids'].shape[0]
        self.err = 0.6 * mae_test + 0.4 * mae_val
        return metrics