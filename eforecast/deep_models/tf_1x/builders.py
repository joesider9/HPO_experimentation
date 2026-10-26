import copy

import numpy as np
import tensorflow as tf

from eforecast.deep_models.tf_1x.layers import layers_func
from eforecast.deep_models.tf_1x.global_builders import build_fuzzy
from eforecast.deep_models.tf_1x.global_builders import apply_activations
from eforecast.deep_models.tf_1x.global_builders import cluster_optimize_build

layers_functions = layers_func()


def create_placeholders(inp_x, params, train=True, is_global=False, is_fuzzy=False):
    if isinstance(inp_x, dict):
        x = dict()
        for key, inp in inp_x.items():
            if key != 'clustering':
                x[key] = tf.compat.v1.placeholder('float', shape=[None, *inp.shape[1:]], name=f'input_{key}')
        if is_global and is_fuzzy:
            x['clustering'] = tf.compat.v1.placeholder('float', shape=[None, inp_x['clustering'].shape[1]],
                                                       name='cluster_inp')
    else:
        x = tf.compat.v1.placeholder('float', shape=[None, *inp_x.shape[1:]], name='inputs')
        if is_global:
            raise ValueError('inp_x should be dictionary when it is global')

    if train:
        y = tf.compat.v1.placeholder(tf.float32, shape=[None, params['n_out']], name='target')
        return x, y
    else:
        return x


def get_size(size, layer_id, layers, output_shape):
    if isinstance(size, set):
        size = list(size)
    if isinstance(size, list):
        size = size[0]
    if size == 'linear':
        if layer_id + 1 >= len(layers):
            raise ValueError('Cannot be linear the last layer')
        size_next = layers[layer_id + 1][1]
        if isinstance(size_next, set):
            size_next = list(size_next)
        if isinstance(size_next, list):
            size_next = size_next[0]
        return int((size_next + np.prod(output_shape[1:])) / 2)
    elif size < 8:
        return int(size * np.prod(output_shape[1:]))
    else:
        return size


def build_layer(x, layers, name_scope, params, train=True, is_for_cluster=False):
    output_layer = x
    with tf.name_scope(name_scope) as scope:
        print(f'Graph of {name_scope} building')
        layers_built = dict()
        for layer_id, layer_tuple in enumerate(layers):
            layer_name, size = layer_tuple
            output_shape = output_layer.get_shape().as_list()
            if layer_name == 'dense' and not is_for_cluster:
                size = get_size(size, layer_id, layers, output_shape)
            if isinstance(size, set):
                size = list(size)
            if isinstance(size, list):
                if len(size) > 0:
                    size = size[0]

            if layer_name not in {'Flatten', 'Dropout', 'Reshape'}:
                output_layer, layer_built = layers_functions[layer_name](output_layer, params, size,
                                                                         str(layer_id), train=train)
                layers_built[layer_built['name']] = layer_built
            elif layer_name == 'Reshape':
                output_layer = layers_functions[layer_name](output_layer,
                                                            [32, int(output_layer.get_shape()[1].value / 32)])
            elif layer_name == 'Dropout':
                output_layer = layers_functions[layer_name](output_layer, size)
            elif layer_name == 'Flatten':
                output_layer = layers_functions[layer_name](output_layer)
            else:
                names = [name for name in layers_functions.keys()]
                raise ValueError(f"Unknown layer name {layer_name}. Valid names {names}")

    return output_layer, layers_built


def build_input_branch(x, model_layers, params, train=True, is_for_cluster=False):
    model_output_dict = dict()
    name_scope_list = []
    model_layers_built = dict()
    for name_scope in sorted(params['scopes']):
        if name_scope not in {'data_row', 'clustering'}:
            layer = model_layers['input']
            params_temp = copy.deepcopy(params)
            # if is_for_cluster:
            #     params_temp['act_func'] = None
            output_layer, layers_built = build_layer(x[name_scope] if isinstance(x, dict) else x,
                                                     layer,
                                                     f'prime_{name_scope}',
                                                     params_temp,
                                                     train=train, is_for_cluster=is_for_cluster)
            if 'data_row' in params['scopes']:
                with tf.name_scope(f'{name_scope}_data_row') as scope:
                    output_data_row, layer_output_data_row = build_layer(x['data_row'], model_layers['data_row'],
                                                                         f'prime_{name_scope}_data_row', params,
                                                                         train=train)
                    layers_built.update(layer_output_data_row)
                    output_layer = tf.concat([output_layer, output_data_row], axis=1,
                                             name=f'prime_output_{name_scope}_row')
            model_output_dict[name_scope] = output_layer
            name_scope_list.append(name_scope)
            model_layers_built[name_scope] = layers_built
    return model_output_dict, name_scope_list, model_layers_built


def build_group_branches(params, model_layers, model_output_dict, name_scope_list, model_layers_built, is_global,
                         train=True):
    if len(params['group_layers']) > 1 and 'input' not in params['group_layers']:
        while True:
            name_scope_list = list(set(name_scope_list))
            new_model_output_dict = dict()
            new_name_scope_list = []
            group_names = set()
            for name_branch in sorted(name_scope_list):
                name_split = name_branch.split('_')
                if name_split[0] == 'rule':
                    name_split = ['_'.join(name_split[:2])] + name_split[2:]
                if 'input' in name_split:
                    raise ValueError("keyword 'input' is not allowed to name a group or name_scope when have more than "
                                     "one group_layer")
                if len(name_split) > 1:
                    new_name_scope = '_'.join(name_split[:-1])
                    new_name_scope_list.append(new_name_scope)
                    group_names.add(new_name_scope)
            if len(new_name_scope_list) > 0:
                name_scope_list = copy.deepcopy(new_name_scope_list)
            else:
                break
            for group_name in sorted(group_names):
                output_branches = [model_output_dict[name_branch] for name_branch in sorted(model_output_dict.keys())
                                   if group_name + '_' in name_branch]
                input_branch = tf.concat(output_branches, axis=1, name=f'concat_input_{group_name}')
                output_branch, layers_branch_built = build_layer(input_branch, model_layers['output'], group_name,
                                                                 params, train=train)
                new_model_output_dict[group_name] = output_branch
                model_layers_built[group_name] = layers_branch_built
            if len(new_model_output_dict) > 0:
                model_output_dict = new_model_output_dict
    cluster_outputs = None
    model_output_list = [model_output_dict[group_name] for group_name in sorted(model_output_dict.keys())]
    if len(model_output_dict) > 1:
        if is_global:
            cluster_outputs, model_layers_built = cluster_optimize_build(model_output_dict,
                                                                         model_layers_built, params['n_out'],
                                                                         train=train)
        model_output = tf.concat(model_output_list, axis=1, name=f'concat_input_for_output')
    elif len(model_output_dict) == 1:
        model_output = model_output_list[0]
    else:
        raise RuntimeError('Failed to build model output')
    return model_output, model_layers_built, cluster_outputs


def build_graph(x, model_layers, params, is_fuzzy=False, is_global=False, is_for_cluster=False, train=True,
                thres_act=None):
    model_output_dict, name_scope_list, model_layers_built = build_input_branch(x, model_layers, params, train=train,
                                                                                is_for_cluster=is_for_cluster)
    model_output, model_layers_built, cluster_outputs = build_group_branches(params, model_layers, model_output_dict,
                                                                             name_scope_list, model_layers_built,
                                                                             is_global, train=train)

    if is_global:
        with tf.name_scope("clustering") as scope:
            if is_fuzzy:
                act_all, fuzzy_layer = build_fuzzy(x['clustering'], params)
                model_layers_built['clustering'] = fuzzy_layer
            else:
                act_all = [x[f'act_{rule}'] for rule in sorted(params['rules'])]
            model_output, act_nan_err = apply_activations(model_output, act_all, thres_act)
    else:
        act_all = None
        act_nan_err = None
    if not is_for_cluster:
        name_scope = 'output_scope'
        model_output, layers_built = build_layer(model_output, model_layers['output'], name_scope, params, train=train)
        model_layers_built[name_scope] = layers_built
        name_scope = 'output'
        with tf.name_scope(name_scope) as scope:
            model_output, layer_output = layers_functions['dense'](model_output,
                                                                   {'act_func': None}, params['n_out'],
                                                                   'output', train=train)
        model_layers_built['output'] = dict()
        model_layers_built['output'][layer_output['name']] = layer_output
    else:
        model_output = tf.expand_dims(tf.reduce_sum(model_output, axis=1), axis=1)
    return model_output, model_layers_built, cluster_outputs, act_all, act_nan_err
