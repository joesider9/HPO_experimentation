import psutil
import numpy as np


def calculate_cpus(n_cpus):
    warm = psutil.cpu_percent()
    average_load = np.mean(psutil.cpu_percent(interval=5, percpu=True)[:n_cpus])

    return n_cpus - int(n_cpus * average_load / 100)


def feed_data(rules, batch, x, y, fuzzy_var, data, target, var_imp, learning_rate, lr):
    feed_dict = dict()
    for rule_name in rules:
        feed_dict.update({x[rule_name]: data.values[batch]})
    feed_dict.update({fuzzy_var: data[var_imp].values[batch]})

    feed_dict.update({y: target.values[batch]})
    feed_dict.update({learning_rate: lr})
    return feed_dict


def feed_data_eval(rules, x, fuzzy_var, data, var_imp):
    feed_dict = dict()
    for rule_name in rules:
        feed_dict.update({x[rule_name]: data.values})

    feed_dict.update({fuzzy_var: data[var_imp].values})

    return feed_dict


def feed_data_fuzzy(rules, batch, x, y, fuzzy_var, data, target, learning_rate, lr):
    feed_dict = dict()
    for rule_name in rules.keys():
        feed_dict.update({x[rule_name]: data.values[batch]})
        for mf in rules[rule_name]:
            feed_dict.update({fuzzy_var[mf['name']]: data[mf['var_name']].values[batch].reshape(-1, 1)})

    feed_dict.update({y: target.values[batch]})
    feed_dict.update({learning_rate: lr})
    return feed_dict


def feed_data_fuzzy_eval(rules, x, fuzzy_var, data):
    feed_dict = dict()
    for rule_name in rules.keys():
        feed_dict.update({x[rule_name]: data.values})
        for mf in rules[rule_name]:
            feed_dict.update({fuzzy_var[mf['name']]: data[mf['var_name']].values.reshape(-1, 1)})
    return feed_dict


def distance(obj_new, obj_old, obj_max, obj_min):
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


def split_validation_set(x):
    partitions = 250
    ind_list = []
    ind = np.arange(x.shape[0])
    for i in range(0, x.shape[0], partitions):
        if (i + partitions + 1) > x.shape[0]:
            ind_list.append(ind[i:])
        else:
            ind_list.append(ind[i:i + partitions])
    return ind_list
