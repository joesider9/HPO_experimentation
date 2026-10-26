import copy

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.cluster import KMeans

from eforecast.deep_models.tf_1x.layers import layers_func
from eforecast.deep_models.tf_1x.trainer import compute_tensors
from eforecast.deep_models.tf_1x.trainer import evaluate_activations

from eforecast.common_utils.train_utils import linear_output


def check_rbf_bounds(sess, act_pl, N, x_pl, y_pl, X_train, y_train, X_val, y_val, X_test, y_test, learning_rate, lr,
                     params, best_clusters, wait):
    warm = 0
    if best_clusters is None:
        raise ValueError('best_clusters is not computed')
    act_train = compute_tensors(sess, act_pl, np.arange(N), x_pl, y_pl, X_train, y_train,
                                learning_rate, lr)
    act_train = np.concatenate(act_train, axis=1)
    act_val = compute_tensors(sess, act_pl, np.arange(y_val.shape[0]), x_pl, y_pl, X_val, y_val,
                              learning_rate, lr)
    act_val = np.concatenate(act_val, axis=1)
    act_test = compute_tensors(sess, act_pl, np.arange(y_test.shape[0]), x_pl, y_pl, X_test, y_test,
                               learning_rate, lr)
    act_test = np.concatenate(act_test, axis=1)
    mae_val_lin, mae_test_lin, sse_val_lin, sse_test_lin = linear_output(X_train, X_val, X_test, y_train, y_val, y_test,
                                                                         act_train, act_val, act_test, params)
    sum_act, min_act, max_act, mean_act, id_min, id_max = evaluate_activations(sess, act_pl,
                                                                               np.arange(N),
                                                                               x_pl, y_pl,
                                                                               X_train, y_train,
                                                                               learning_rate,
                                                                               lr,
                                                                               params[
                                                                                   'thres_act'])
    min_samples = params['min_samples']
    max_samples = int(params['max_samples_ratio'] * y_train.shape[0])
    if min_act < min_samples:
        for variable in tf.compat.v1.trainable_variables():
            if f'centroid_{id_min}' in variable.name or f'RBF_variance_{id_min}' in variable.name:
                sess.run(tf.compat.v1.assign(variable, best_clusters[variable.name]))
                warm = 3
    if max_act > max_samples:
        for variable in tf.compat.v1.trainable_variables():
            if f'centroid_{id_max}' in variable.name or f'RBF_variance_{id_max}' in variable.name:
                sess.run(tf.compat.v1.assign(variable, best_clusters[variable.name]))
                warm = 3

    return mae_val_lin, mae_test_lin, sse_val_lin, sse_test_lin, sum_act, min_act, max_act, mean_act, warm


def assign_rbf(sess, best_clusters):
    for variable in tf.compat.v1.trainable_variables():
        if 'centroid' in variable.name or f'RBF_variance' in variable.name:
            sess.run(tf.compat.v1.assign(variable, best_clusters[variable.name]))


def get_rbf(sess):
    best_clusters = dict()
    for variable in tf.compat.v1.trainable_variables():
        if 'centroid' in variable.name or 'RBF_variance' in variable.name:
            best_clusters[variable.name] = sess.run(variable)
    return best_clusters


def get_stratify_batches_by_act(sess, act_pl, N, x_pl, y_pl, X_train, y_train, learning_rate, lr, thres_act,
                                batch_size, n_batch):
    act = compute_tensors(sess, act_pl, np.arange(N), x_pl, y_pl, X_train, y_train, learning_rate,
                          lr)
    act = np.concatenate(act, axis=1)
    act[act >= thres_act] = 1
    act[act < thres_act] = 0
    prob = act.sum(axis=0) / act.shape[0]
    probs = prob[act.argmax(axis=1)]
    batches = [np.random.choice(N, batch_size, replace=False, p=probs / probs.sum())
               for _ in range(n_batch + 1)]
    return batches


def find_min_max_var(inputs, n_rules, n_var, centroids, var, thres_act):
    s = np.shape(inputs)
    phi = []
    for n in range(n_rules):
        d1 = inputs.values - np.tile(centroids[n], [s[0], 1])
        d = np.sqrt(np.sum(np.power(d1 / np.tile(var.values[n], [s[0], 1]), 2), axis=1))
        phi.append(np.expand_dims(np.exp(-1 * np.square(d)), axis=1))
    act_all_eval = np.concatenate(phi, axis=1)
    act_all_eval[act_all_eval >= thres_act] = 1
    act_all_eval[act_all_eval < thres_act] = 0
    return act_all_eval.sum(axis=0)


def check_VAR_if_all_nans(inputs, n_rules, n_var, centroids, var, thres_act):
    s = np.shape(inputs)
    dist = []
    phi = []
    for n in range(n_rules):
        d1 = inputs.values - np.tile(centroids[n], [s[0], 1])
        dist.append(np.expand_dims(np.sqrt(np.sum(np.square(d1), axis=1)), axis=1))
        d = np.sqrt(np.sum(np.power(d1 / np.tile(var.values[n], [s[0], 1]), 2), axis=1))
        phi.append(np.expand_dims(np.exp(-1 * np.square(d)), axis=1))
    dist = np.concatenate(dist, axis=1)
    activations = np.concatenate(phi, axis=1)
    indices = np.where(np.all(activations <= thres_act, axis=1))[0]
    len_nan = 0
    while indices.shape[0] > 0:
        d = dist[indices[0]]
        clust = np.argmin(d)
        var.values[clust] += thres_act
        dist = []
        phi = []
        for n in range(n_rules):
            d1 = inputs.values - np.tile(centroids[n], [s[0], 1])
            dist.append(np.expand_dims(np.sqrt(np.sum(np.square(d1), axis=1)), axis=1))
            d = np.sqrt(np.sum(np.power(d1 / np.tile(var.values[n], [s[0], 1]), 2), axis=1))
            phi.append(np.expand_dims(np.exp(-1 * np.square(d)), axis=1))
        dist = np.concatenate(dist, axis=1)
        activations = np.concatenate(phi, axis=1)
        indices = np.where(np.all(activations <= thres_act, axis=1))[0]
    return var


def create_centroids(X_train, params):
    if X_train is None:
        raise ValueError('X_train is not provided')
    c_best = None
    inertia = np.inf
    for _ in range(5):
        c = KMeans(n_clusters=params['n_rules'], random_state=0).fit(X_train)
        if c.inertia_ < inertia:
            c_best = copy.deepcopy(c)
            inertia = c.inertia_
    if c_best is not None:
        min_samples = params['min_samples']
        max_samples = int(params['max_samples_ratio'] * X_train.shape[0])
        centroids = c_best.cluster_centers_.astype(np.float32)
        cnt = pd.DataFrame(centroids, index=['c' + str(i) for i in range(centroids.shape[0])],
                           columns=['v' + str(i) for i in range(centroids.shape[1])])
        var_init = pd.DataFrame(columns=['v' + str(i) for i in range(centroids.shape[1])])
        for r in cnt.index:
            v = (cnt.loc[r] - cnt.drop(r)).abs().max() / 4
            v.name = r
            var_init = var_init.append(v)
        var_init = check_VAR_if_all_nans(X_train, params['n_rules'], centroids.shape[1], centroids, var_init,
                                         params['thres_act'])
        n_samples = find_min_max_var(X_train, params['n_rules'], centroids.shape[1], centroids, var_init,
                                     params['thres_act'])
        ind_small = np.where(n_samples < min_samples)[0]
        ind_large = np.where(n_samples > max_samples)[0]
        while ind_small.shape[0] != 0:
            var_init.iloc[ind_small] += 0.001
            n_samples = find_min_max_var(X_train, params['n_rules'], centroids.shape[1], centroids, var_init,
                                         params['thres_act'])
            ind_small = np.where(n_samples < min_samples)[0]
        while ind_large.shape[0] != 0:
            var_init.iloc[ind_large] -= 0.001
            n_samples = find_min_max_var(X_train, params['n_rules'], centroids.shape[1], centroids, var_init,
                                         params['thres_act'])
            ind_large = np.where(n_samples > max_samples)[0]
        params['centroids'] = centroids
        params['var_init'] = var_init
        params['rules'] = [f'rule_{i}' for i in range(params['n_rules'])]

    return params


class RBFLayer(tf.keras.layers.Layer):
    def __init__(self, thres_act, centroids, var_init):
        super(RBFLayer, self).__init__()
        self.var = []
        self.centroids = []
        self.thres_act = thres_act
        self.n_rules = centroids.shape[0]
        self.n_var = centroids.shape[1]
        self.var_init = []
        self.centroids_init = []
        for n in range(self.n_rules):
            self.var_init.append(var_init.iloc[n].values.reshape(1, -1))
            self.centroids_init.append(centroids[n].reshape(1, -1))

    def build(self, input_shape):
        for n in range(self.n_rules):
            centroids_init = tf.keras.initializers.constant(self.centroids_init[n])
            self.centroids.append(self.add_weight(f'centroid_{n}',
                                                  shape=[1, self.n_var],
                                                  dtype=tf.float32,
                                                  initializer=centroids_init,
                                                  trainable=False))
            var_init = tf.keras.initializers.constant(self.var_init[n])
            self.var.append(self.add_weight(f'RBF_variance_{n}',
                                            shape=[1, self.n_var],
                                            dtype=tf.float32,
                                            initializer=var_init,
                                            trainable=True))
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        s = tf.shape(inputs)
        phi = []
        for n in range(self.n_rules):
            d1 = inputs - tf.tile(self.centroids[n], [s[0], 1])
            d = tf.sqrt(tf.reduce_sum(tf.pow(tf.divide(d1, tf.tile(self.var[n],
                                                                     [s[0], 1])),
                                             2), axis=1))
            phi.append(tf.expand_dims(tf.exp(tf.multiply(tf.constant(-1, dtype=tf.float32), tf.square(d))), axis=1))

        return phi


def build_fuzzy(fuzzy_inp, params):
    fuzzy_layer = RBFLayer(params['thres_act'], params['centroids'], params['var_init'])
    activations = fuzzy_layer(fuzzy_inp)
    return activations, fuzzy_layer


def cluster_optimize_build(model_output_dict, model_layers_built, n_out, train=True):
    cluster_outputs = []
    layers_functions = layers_func()
    for name_scope in sorted(model_output_dict.keys()):
        with tf.name_scope(f'{name_scope}_output') as scope:
            output_cluster, layer_output_cluster = layers_functions['dense'](model_output_dict[name_scope],
                                                                             {'act_func': None},
                                                                             n_out,
                                                                             f'{name_scope}_cluster_output',
                                                                             train=train)
            model_layers_built[f'{name_scope}_output'] = layer_output_cluster
            cluster_outputs.append(output_cluster)
    # with tf.name_scope('aggregate_clusters') as scope:
    #     cluster_outputs = tf.concat(cluster_outputs, axis=1, name='cluster_outputs')
    return cluster_outputs, model_layers_built


def apply_activations(model_output, act_all, thres_act):
    thres_act_tf = tf.constant(thres_act, tf.float32, name='thres_act')
    thres_act_tf_up = tf.constant(thres_act + thres_act / 10, tf.float32, name='thres_act')
    thres_act_tf_ml = tf.constant(10 / thres_act, tf.float32, name='thres_act')
    act_all_clip = tf.concat(act_all, axis=1)
    output_shape = model_output.get_shape().as_list()[-1]
    n_rules = act_all_clip.get_shape().as_list()[-1]
    act_all_clip = tf.multiply(thres_act_tf_ml, tf.subtract(tf.clip_by_value(act_all_clip, thres_act_tf,
                                                                             thres_act_tf_up),
                                                            thres_act_tf))
    act_sum = tf.reduce_sum(act_all_clip, axis=1)
    act_sum_clipped = tf.clip_by_value(act_sum, 0.00000000001, n_rules + 1)
    act_nan_err = tf.multiply(tf.subtract(act_sum_clipped, act_sum), tf.constant(1e11, tf.float32))
    act_all_weighted = tf.divide(act_all_clip, tf.tile(tf.expand_dims(act_sum_clipped, axis=1), [1, n_rules]))
    cluster_output_size = int(output_shape / n_rules)
    a_norm = tf.reshape(tf.tile(tf.expand_dims(act_all_weighted, -1), [1, 1, cluster_output_size]), [-1, output_shape])
    model_output = tf.multiply(a_norm, model_output)
    # model_output = tf.add(model_output, tf.multiply(model_output, tf.tile(tf.expand_dims(act_nan_err, axis=1),
    #                                                                       [1, n_rules])))
    return model_output, tf.reduce_sum(act_nan_err)


def gauss_mf(x, mean, sigma):
    """
    Gaussian fuzzy membership function.

    Parameters
    ----------
    x : 1d tensor or iterable
        Independent variable.
    mean : float tensor constant
        Gaussian parameter for center (mean) value.
    sigma : float tensor constant
        Gaussian parameter for standard deviation.

    Returns
    -------
    y : 1d tensor
        Gaussian membership function for x.
    """

    return tf.exp(tf.divide(tf.multiply(tf.constant(-1, dtype=tf.float32), tf.square(tf.subtract(x, mean))),
                            tf.multiply(tf.constant(2, dtype=tf.float32), tf.square(sigma))))


def gbell_mf(x, a, b, c):
    """
        Generalized Bell function fuzzy membership generator.

    Parameters
    ----------
    x : 1d array
        Independent variable.
    a : float
        Bell function parameter controlling width. See Note for definition.
    b : float
        Bell function parameter controlling slope. See Note for definition.
    c : float
        Bell function parameter defining the center. See Note for definition.

    Returns
    -------
    y : 1d array
        Generalized Bell fuzzy membership function.

    Notes
    -----
    Definition of Generalized Bell function is:

        y(x) = 1 / (1 + abs([x - c] / a) ** [2 * b])
    """

    div = tf.abs(tf.divide(tf.subtract(x, c), a))
    p = tf.pow(div, 2 * b)
    value = tf.add(tf.constant(1, dtype=tf.float32), p)
    return tf.divide(tf.constant(1, dtype=tf.float32), value)


class FuzzyLayer(tf.keras.layers.Layer):
    def __init__(self, rules, thres_act):
        super(FuzzyLayer, self).__init__()
        self.fuzzy_vars = None
        self.rules = rules
        self.thres_act = thres_act

    def build(self, n):
        self.fuzzy_vars = dict()
        for rule_name, rule in self.rules.items():
            for mf in rule:
                self.fuzzy_vars['var_' + mf['name']] = dict()
                if mf['type'] == 'gauss':
                    var_init = tf.keras.initializers.constant(mf['param'][0])
                    self.fuzzy_vars['var_' + mf['name']]['mean'] = self.add_weight('var_' + mf['name'],
                                                                                   shape=[1],
                                                                                   dtype=tf.float32,
                                                                                   initializer=var_init,
                                                                                   trainable=True)

                    var_init = tf.keras.initializers.constant(mf['param'][1])
                    self.fuzzy_vars['var_' + mf['name']]['sigma'] = self.add_weight('var_' + mf['name'],
                                                                                    shape=[1],
                                                                                    dtype=tf.float32,
                                                                                    initializer=var_init,
                                                                                    trainable=True)

                else:
                    var_init = tf.keras.initializers.constant(mf['param'][0])
                    self.fuzzy_vars['var_' + mf['name']]['a'] = self.add_weight('var_' + mf['name'],
                                                                                shape=[1],
                                                                                dtype=tf.float32,
                                                                                initializer=var_init,
                                                                                trainable=True)

                    var_init = tf.keras.initializers.constant(mf['param'][1])
                    self.fuzzy_vars['var_' + mf['name']]['b'] = self.add_weight('var_' + mf['name'],
                                                                                shape=[1],
                                                                                dtype=tf.float32,
                                                                                initializer=var_init,
                                                                                trainable=True)
                    var_init = tf.keras.initializers.constant(mf['param'][2])
                    self.fuzzy_vars['var_' + mf['name']]['c'] = self.add_weight('var_' + mf['name'],
                                                                                shape=[1],
                                                                                dtype=tf.float32,
                                                                                initializer=var_init,
                                                                                trainable=True)
        super(FuzzyLayer, self).build(n)

    def call(self, fuzzy_inp, **kwargs):
        activations = None
        for rule_name, rule in self.rules.items():
            act_rule = None
            for mf in rule:
                if mf['type'] == 'gauss':
                    act = gauss_mf(fuzzy_inp[mf['name']], self.fuzzy_vars['var_' + mf['name']]['mean'],
                                   self.fuzzy_vars['var_' + mf['name']]['sigma'])
                else:
                    act = gbell_mf(fuzzy_inp[mf['name']], self.fuzzy_vars['var_' + mf['name']]['a'],
                                   self.fuzzy_vars['var_' + mf['name']]['b'],
                                   self.fuzzy_vars['var_' + mf['name']]['c'])
                act_rule = act if act_rule is None else tf.concat([act_rule, act], axis=1)
            act_rule = tf.reduce_prod(act_rule, axis=1, keepdims=True, name='act_' + rule_name)
            activations = act_rule if activations is None else tf.concat([activations, act_rule], axis=1)

        return activations
