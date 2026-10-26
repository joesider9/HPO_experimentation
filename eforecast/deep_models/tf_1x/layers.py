import numpy as np
import tensorflow as tf

act_funcs = {'elu': tf.nn.elu,
             'sigmoid': tf.nn.sigmoid,
             'tanh': tf.nn.tanh}


def conv_2d(x, params, size, layer_id, train=True):
    x_shape1 = x.get_shape().as_list()
    if len(x_shape1) == 3:
        x = tf.expand_dims(x, axis=-1)
        x_shape1 = x.get_shape().as_list()
    x_shape1 = np.array(x_shape1[1:3])
    x_shape = x_shape1 // int(size)
    x_shape[x_shape <= 1] = 2
    x_shape = np.minimum(x_shape, x_shape1)
    kernels = x_shape.tolist()
    act_func = act_funcs[params['act_func']]
    x_shape = x_shape // 2
    x_shape[x_shape == 0] = 1
    pool_size = x_shape.tolist()
    pool_size = [int(p) for p in pool_size]
    kernels = [int(k) for k in kernels]
    layer = dict()
    conv = tf.keras.layers.Conv2D(filters=int(params['filters']),
                                  kernel_size=kernels,
                                  padding="valid",
                                  name='conv_2d_' + layer_id,
                                  activation=act_func)

    pool = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=1, name='pool' + layer_id)
    layer['name'] = 'conv_2d_' + layer_id
    layer['layer'] = conv
    cnn_output = pool(layer['layer'](x))
    print(f"layer {layer['name']} has shape {cnn_output.get_shape().as_list()}")
    print(f"layer weights {layer['name']} has shape {layer['layer'].trainable_weights[0].get_shape().as_list()}")
    return cnn_output, layer


def conv_3d(x, params, size, layer_id, train=True):
    x_shape1 = x.get_shape().as_list()
    if len(x_shape1) == 3:
        x = tf.expand_dims(x, axis=-1)
        x_shape1 = x.get_shape().as_list()
    x_shape1 = np.array(x_shape1[1:4])
    x_shape = x_shape1 // int(size)
    x_shape[x_shape <= 1] = 2
    x_shape = np.minimum(x_shape, x_shape1)
    kernels = x_shape.tolist()
    if len(kernels) != 3:
        kernels = [1] + kernels

    act_func = act_funcs[params['act_func']]
    x_shape = x_shape // 2
    x_shape[x_shape == 0] = 1
    pool_size = x_shape.tolist()
    x_shape = x_shape // 2
    x_shape[x_shape == 0] = 1
    pool_size = x_shape.tolist()
    pool_size = [int(p) for p in pool_size]
    kernels = [int(k) for k in kernels]
    layer = dict()
    conv = tf.keras.layers.Conv3D(filters=int(params['filters']),
                                  kernel_size=kernels,
                                  padding="valid",
                                  name='conv_3d_' + layer_id,
                                  activation=act_func)

    pool = tf.keras.layers.AveragePooling3D(pool_size=pool_size, strides=1, name='pool' + layer_id)
    layer['name'] = 'conv_3d_' + layer_id
    layer['layer'] = conv
    if len(x.get_shape().as_list()) == 4:
        cnn_output = pool(layer['layer'](tf.expand_dims(x, axis=4)))
    else:
        cnn_output = pool(layer['layer'](x))
    print(f"layer {layer['name']} has shape {cnn_output.get_shape().as_list()}")
    print(f"layer weights {layer['name']} has shape {layer['layer'].trainable_weights[0].get_shape().as_list()}")
    return cnn_output, layer


def time_distr_conv_2d(x, params, size, layer_id, train=True):
    x_shape1 = x.get_shape().as_list()
    if len(x_shape1) == 3:
        x = tf.expand_dims(x, axis=-1)
        x_shape1 = x.get_shape().as_list()
    if len(x_shape1) == 4:
        x = tf.expand_dims(x, axis=-1)
    x_shape1 = np.array(x_shape1[2:4])
    x_shape = x_shape1 // int(size)
    x_shape[x_shape <= 1] = 2
    x_shape = np.minimum(x_shape, x_shape1)
    kernels = x_shape.tolist()
    act_func = act_funcs[params['act_func']]
    x_shape = x_shape // 2
    x_shape[x_shape == 0] = 1
    pool_size = x_shape.tolist()
    pool_size = [int(p) for p in pool_size]
    kernels = [int(k) for k in kernels]
    layer = dict()
    conv = tf.keras.layers.Conv2D(filters=int(params['filters']),
                                  kernel_size=kernels,
                                  padding="same",
                                  name='conv_' + layer_id,
                                  activation=act_func)

    pool = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=1, name='pool' + layer_id)
    time_distributed = tf.keras.layers.TimeDistributed(conv, name='time_distr_conv_2d_' + layer_id)
    layer['name'] = 'time_distr_conv_2d_' + layer_id
    layer['layer'] = time_distributed
    cnn_output = layer['layer'](x)
    cnn_output = tf.keras.layers.TimeDistributed(pool)(cnn_output)
    print(f"layer {layer['name']} has shape {cnn_output.get_shape().as_list()}")
    print(f"layer weights {layer['name']} has shape {layer['layer'].trainable_weights[0].get_shape().as_list()}")
    return cnn_output, layer


def time_distr_conv_3d(x, params, size, layer_id, train=True):
    x_shape1 = x.get_shape().as_list()
    if len(x_shape1) == 3:
        x = tf.expand_dims(x, axis=-1)
        x_shape1 = x.get_shape().as_list()
    if len(x_shape1) == 4:
        x = tf.expand_dims(x, axis=-1)
    x_shape1 = np.array(x_shape1[2:5])
    x_shape = x_shape1 // int(size)
    x_shape[x_shape <= 1] = 2
    x_shape = np.minimum(x_shape, x_shape1)
    kernels = x_shape.tolist()
    act_func = act_funcs[params['act_func']]
    x_shape = x_shape // 2
    x_shape[x_shape == 0] = 1
    pool_size = x_shape.tolist()
    pool_size = [int(p) for p in pool_size]
    kernels = [int(k) for k in kernels]
    layer = dict()
    conv = tf.keras.layers.Conv3D(filters=int(params['filters']),
                                  kernel_size=kernels,
                                  padding="same",
                                  name='conv_' + layer_id,
                                  activation=act_func)

    pool = tf.keras.layers.AveragePooling3D(pool_size=pool_size, strides=1, name='pool' + layer_id)
    layer['name'] = 'time_distr_conv_3d_' + layer_id
    layer['layer'] = conv
    if len(x.get_shape().as_list()) == 5:
        cnn_output = tf.keras.layers.TimeDistributed(layer['layer'], name='time_distr_conv_3d_' + layer_id) \
            (tf.expand_dims(x, axis=-1))
    else:
        cnn_output = tf.keras.layers.TimeDistributed(layer['layer'], name='time_distr_conv_3d_' + layer_id)(x)

    cnn_output = tf.keras.layers.TimeDistributed(pool)(cnn_output)
    print(f"layer {layer['name']} has shape {cnn_output.get_shape().as_list()}")
    print(f"layer weights {layer['name']} has shape {layer['layer'].trainable_weights[0].get_shape().as_list()}")
    return cnn_output, layer


def lstm(x, params, size, layer_id, train=True):
    layer = dict()
    act_func = act_funcs['tanh']
    x_shape = x.get_shape().as_list()
    if len(x_shape) != 3:
        x = tf.reshape(x, [-1, x_shape[1], np.prod(x_shape[2:])])
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(
        x_shape[1],
        name='lstm_' + layer_id,
        return_sequences=True), name='bidirectional_' + layer_id)
    layer['name'] = 'bidirectional_' + layer_id + 'lstm_' + layer_id
    layer['layer'] = lstm
    lstm_output = layer['layer'](x)
    if len(x_shape) != 3:
        lstm_output = tf.reshape(lstm_output, [-1, x_shape[1], *x_shape[2:-1]])
    weights_lstm = lstm.trainable_weights
    layer['weights'] = weights_lstm

    print(f"layer {layer['name']} has shape {lstm_output.get_shape().as_list()}")
    print(f"layer weights {layer['name']} has shape {layer['layer'].trainable_weights[0].get_shape().as_list()}")

    return lstm_output, layer


def Flatten(x):
    flat = tf.keras.layers.Flatten()
    return flat(x)


def Dropout(x, hold_prob):
    dropout = tf.keras.layers.Dropout(1 - hold_prob)
    return dropout(x)


def Reshape(x, params):
    return tf.reshape(x, [-1, params[0], params[1]])


def hidden_dense(x, params, size, layer_id, train=True):
    act_func = act_funcs[params['act_func']]
    layer = dict()
    dense = tf.keras.layers.Dense(units=13 ** 3, activation=act_func, name='hidden_dense_' + layer_id)
    layer['name'] = 'hidden_dense_' + layer_id
    layer['layer'] = dense
    out_dense = layer['layer'](x)
    out_dense = tf.reshape(out_dense, [-1, 13, 13, 13], name='reshape_' + layer_id)

    print(f"layer {layer['name']} has shape {out_dense.get_shape().as_list()}")
    print(f"layer weights {layer['name']} has shape {layer['layer'].trainable_weights[0].get_shape().as_list()}")
    return out_dense, layer


def dense(x, params, size, layer_id, train=True):
    if params['act_func'] is not None:
        act_func = act_funcs[params['act_func']]
    else:
        act_func = None
    layer = dict()
    dense = tf.keras.layers.Dense(units=size, activation=act_func, name='dense_' + layer_id)
    layer['name'] = 'dense_' + layer_id
    layer['layer'] = dense
    out_dense = layer['layer'](x)

    print(f"layer {layer['name']} has shape {out_dense.get_shape().as_list()}")
    print(f"layer weights {layer['name']} has shape {layer['layer'].trainable_weights[0].get_shape().as_list()}")
    return out_dense, layer


def layers_func():
    layers = {'conv_2d': conv_2d,
              'time_distr_conv_2d': time_distr_conv_2d,
              'conv_3d': conv_3d,
              'time_distr_conv_3d': time_distr_conv_3d,
              'bidirectional_lstm': lstm,
              'hidden_dense': hidden_dense,
              'dense': dense,
              'Flatten': Flatten,
              'Dropout': Dropout,
              'Reshape': Reshape}
    return layers
