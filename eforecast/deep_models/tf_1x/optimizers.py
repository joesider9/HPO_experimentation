import tensorflow as tf


def get_rated(rated, y):
    if rated is not None:
        norm_val = tf.constant(1, tf.float32, name='rated')
    else:
        norm_val = y
    return norm_val


def optimize_rule_branch(cluster_outputs, act_all, y, learning_rate, thres_act, rules):
    rules = sorted(rules)
    trainers_cl = dict()
    mae_cl_list = []
    sse_cl_list = []
    thres_act_tf = tf.constant(thres_act, tf.float32, name='thres_act')
    thres_act_tf_up = tf.constant(thres_act + thres_act / 10, tf.float32, name='thres_act')
    thres_act_tf_ml = tf.constant(10 / thres_act, tf.float32, name='thres_act')
    act_list = []
    act_clip_list = []
    act_sum_list = []
    for rule, cluster_output, act in zip(rules, cluster_outputs, act_all):
        shape = tf.reduce_sum(tf.ones_like(act))
        act_list.append(act)
        act_clip = tf.multiply(thres_act_tf_ml, tf.subtract(tf.clip_by_value(act, thres_act_tf, thres_act_tf_up),
                                                            thres_act_tf))
        act_clip_list.append(act_clip)
        act_sum = tf.reduce_sum(act_clip)
        act_sum_list.append(tf.expand_dims(act_sum, axis=0))
        err_cl = tf.multiply(tf.subtract(cluster_output, y), act_clip)
        cost_cl = tf.reduce_sum(tf.reduce_sum(tf.square(err_cl)))
        sse_cl_list.append(cost_cl)
        mae_cl_list.append(tf.divide(tf.reduce_sum(tf.reduce_sum(tf.abs(err_cl))), tf.reduce_sum(act_clip)))
        optimizer_cl_out = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        train_cl_out = optimizer_cl_out.minimize(cost_cl, var_list=[v for v in tf.compat.v1.trainable_variables()
                                                                    if 'centroid' not in v.name
                                                                    and 'RBF_variance' not in v.name])
        trainers_cl[rule] = train_cl_out
    return trainers_cl, tf.reduce_mean(mae_cl_list), tf.reduce_mean(sse_cl_list), tf.reduce_mean(sse_cl_list)


def optimize_output(model_output, y, learning_rate):
    err_out = tf.abs(model_output - y)
    cost_out = tf.reduce_sum(tf.reduce_sum(tf.square(err_out)))
    optimizer_all = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_out = optimizer_all.minimize(cost_out, var_list=[v for v in tf.compat.v1.trainable_variables()
                                                           if 'output' in v.name])
    return train_out


def optimize_fuzzy(model_output, y, learning_rate):
    err_out = tf.abs(model_output - y)
    cost_out = tf.reduce_sum(tf.reduce_sum(tf.square(err_out)))
    optimizer_all = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_out = optimizer_all.minimize(cost_out, var_list=[v for v in tf.compat.v1.trainable_variables()
                                                           if 'centroid' in v.name
                                                           or 'RBF_variance' in v.name])
    return train_out


def optimize_bulk(model_output, y, learning_rate, norm_val, act_nan_err=None):
    err_out = tf.abs(model_output - y)
    cost_out = tf.reduce_sum(tf.reduce_sum(tf.square(err_out)))
    optimizer_all = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_out = optimizer_all.minimize(cost_out, var_list=[v for v in tf.compat.v1.trainable_variables()])
    err = tf.divide(tf.abs(model_output - y), norm_val)
    if act_nan_err is None:
        accuracy_out = tf.reduce_mean(tf.reduce_mean(err))
        sse_out = tf.reduce_sum(tf.reduce_sum(tf.square(err)))
    else:
        accuracy_out = tf.add(tf.reduce_mean(tf.reduce_mean(err)), act_nan_err)
        sse_out = tf.add(tf.reduce_sum(tf.reduce_sum(tf.square(err))), act_nan_err)
    return train_out, accuracy_out, sse_out


def optimize_not_fuzzy(model_output, y, learning_rate):
    err_out = tf.abs(model_output - y)
    cost_out = tf.reduce_sum(tf.reduce_sum(tf.square(err_out)))
    optimizer_all = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_out = optimizer_all.minimize(cost_out, var_list=[v for v in tf.compat.v1.trainable_variables()
                                                           if 'centroid' not in v.name
                                                           and 'RBF_variance' not in v.name])
    return train_out


def optimize(model_output, y, cluster_outputs=None, act_all=None, act_nan_err=None, rated=None, is_global=False,
             is_fuzzy=False, is_for_clustering=False, thres_act=None, rules=None):
    trainers = dict()
    MAEs = []
    SSEs = []
    norm_val = get_rated(rated, y)
    with tf.name_scope("optimizers") as scope:
        learning_rate = tf.compat.v1.placeholder('float', shape=[], name='learning_rate')
        if not is_global and not is_fuzzy:
            train_all, accuracy_all, sse_all = optimize_bulk(model_output, y, learning_rate, norm_val)
            trainers['bulk'] = train_all
            MAEs.append(accuracy_all)
            SSEs.append(sse_all)
        elif is_global and not is_fuzzy:
            trainers_cl, accuracy_cl, sse_cl, cost_cl = optimize_rule_branch(cluster_outputs, act_all, y, learning_rate,
                                                                             thres_act, rules)
            MAEs.append(accuracy_cl)
            SSEs.append(sse_cl)
            trainers.update(trainers_cl)
            trainers['output'] = optimize_output(model_output, y, learning_rate)
            train_all, accuracy_all, sse_all = optimize_bulk(model_output, y, learning_rate, norm_val)
            trainers['bulk'] = train_all
            MAEs.append(accuracy_all)
            SSEs.append(sse_all)
        elif is_global and is_fuzzy:
            trainers_cl, accuracy_cl, sse_cl, cost_cl = optimize_rule_branch(cluster_outputs, act_all, y,
                                                                             learning_rate,
                                                                             thres_act, rules)
            trainers.update(trainers_cl)
            trainers['not_fuzzy'] = optimize_not_fuzzy(model_output, y, learning_rate)
            trainers['fuzzy'] = optimize_fuzzy(model_output, y, learning_rate)
            MAEs.append(accuracy_cl)
            SSEs.append(sse_cl)
            train_all, accuracy_all, sse_all = optimize_bulk(model_output, y, learning_rate, norm_val,
                                                             act_nan_err=act_nan_err)
            trainers['bulk'] = train_all
            MAEs.append(accuracy_all)
            SSEs.append(sse_all)

    return trainers, MAEs, SSEs, learning_rate
