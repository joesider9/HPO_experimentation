import time
import numpy as np
import tensorflow as tf

from eforecast.common_utils.train_utils import feed_data


def train_schedule_for_cluster(sess, trainers, rules, batches, x_pl, y_pl, X_train, y_train, learning_rate, lr, warm):
    if warm > 0:
        for _ in range(warm):
            print('TRAINING non Fuzzy')
            train_step(sess, trainers['not_fuzzy'], batches, x_pl, y_pl, X_train, y_train, learning_rate, 10 * lr)
    print('TRAINING bulk')
    train_step(sess, trainers['bulk'], batches, x_pl, y_pl, X_train, y_train, learning_rate, lr)


def train_schedule_fuzzy(sess, epoch, trainers, rules, batches, x_pl, y_pl, X_train, y_train, learning_rate, lr, warm):
    if warm > 0:
        for s in range(warm):
            print(f'WARMING STEP {s}')
            train_step(sess, trainers['not_fuzzy'], batches, x_pl, y_pl, X_train, y_train, learning_rate, lr)
            # print('TRAINING output')
            # train_step(sess, trainers['output'], batches, x_pl, y_pl, X_train, y_train, learning_rate, 10 * lr)
            # train_step(sess, trainers['output'], batches, x_pl, y_pl, X_train, y_train, learning_rate, 10 * lr)
            # train_step(sess, trainers['output'], batches, x_pl, y_pl, X_train, y_train, learning_rate, 10 * lr)
            # train_step(sess, trainers['output'], batches, x_pl, y_pl, X_train, y_train, learning_rate, 10 * lr)
    # if epoch < 300:
    print('TRAINING fuzzy')
    train_step(sess, trainers['fuzzy'], batches, x_pl, y_pl, X_train, y_train, learning_rate, 10 * lr)
    for s in range(3):
        print(f'TRAINING STEP {s}')
        print('TRAINING non Fuzzy')
        train_step(sess, trainers['not_fuzzy'], batches, x_pl, y_pl, X_train, y_train, learning_rate, lr)


def train_schedule_global(sess, trainers, rules, batches, x_pl, y_pl, X_train, y_train, learning_rate, lr):
    # for rule in rules:
    #     print(f'TRAINING {rule}')
    #     train_step(sess, trainers[rule], batches, x_pl, y_pl, X_train, y_train, learning_rate, lr)

    print('TRAINING bulk')
    train_step(sess, trainers['bulk'], batches, x_pl, y_pl, X_train, y_train, learning_rate, lr)
    # print('TRAINING output')
    # train_step(sess, trainers['output'], batches, x_pl, y_pl, X_train, y_train, learning_rate, lr)


def train_step(sess, trainer, batches, x_pl, y_pl, x, y, lr_pl, lr):
    start = time.time()
    for batch in batches:
        feed_dict = feed_data(batch, x_pl, y_pl, x, y, lr_pl, lr)
        sess.run([trainer], feed_dict=feed_dict)
    end = time.time()
    sec_per_iter = (end - start) / len(batches)
    if sec_per_iter > 1:
        print(f'Run training step with {sec_per_iter}sec/iter')
    elif sec_per_iter > 0:
        print(f'Run training step with {1 / sec_per_iter}iter/sec')


def validation_step(sess, performer, ind_val_list, x_pl, y_pl, x, y, lr_pl, lr):
    feed_dict = feed_data(ind_val_list, x_pl, y_pl, x, y, lr_pl, lr)
    res = sess.run(performer, feed_dict=feed_dict)
    return np.array(res)


def compute_tensors(sess, tensors, ind_val_list, x_pl, y_pl, x, y, lr_pl, lr):
    feed_dict = feed_data(ind_val_list, x_pl, y_pl, x, y, lr_pl, lr)
    np_arrays = sess.run(tensors, feed_dict=feed_dict)
    return np_arrays


def evaluate_activations(sess, act_pl, ind_train, x_pl, y_pl, x, y, lr_pl, lr, thres_act):
    feed_dict = feed_data(ind_train, x_pl, y_pl, x, y, lr_pl, lr)
    act_all_eval = sess.run(act_pl, feed_dict=feed_dict)
    act_all_eval = np.concatenate(act_all_eval, axis=1)
    act_all_eval[act_all_eval >= thres_act] = 1
    act_all_eval[act_all_eval < thres_act] = 0
    print(f'SHAPE OF ACTIVATIONS IS {act_all_eval.shape}')
    print(f'SUM OF ACTIVATIONS IS {act_all_eval.sum()}')
    print(f'MIN OF ACTIVATIONS IS {act_all_eval.sum(axis=0).min()}')
    print(f'MAX OF ACTIVATIONS IS {act_all_eval.sum(axis=0).max()}')
    print(f'MEAN OF ACTIVATIONS IS {act_all_eval.sum(axis=0).mean()}')
    return act_all_eval.sum(), act_all_eval.sum(axis=0).min(), act_all_eval.sum(axis=0).max(),\
        act_all_eval.sum(axis=0).mean(), act_all_eval.sum(axis=0).argmin(), act_all_eval.sum(axis=0).argmax()


def gather_weights(sess, variables):
    best_layers = dict()
    weights = sess.run(variables)
    for variable, w in zip(variables, weights):
        best_layers[variable.name] = w
    return best_layers
