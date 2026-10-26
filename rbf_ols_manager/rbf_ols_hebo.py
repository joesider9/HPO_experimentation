import logging
import multiprocessing as mp
import os
import pickle
import random
import numpy as np
import pandas as pd
from sqlalchemy.types import Integer
from sqlalchemy.types import String
from sqlalchemy.types import FLOAT
from threadpoolctl import threadpool_limits

import sqlalchemy as db
from .rbf_ols_network import rbf_ols


class rbf_ols_module(object):
    def __init__(self, static_data, cluster_name, cluster_dir, GA=False, path_group=None):
        self.static_data = static_data
        self.cluster_name = cluster_name
        self.path_group = path_group
        self.n_jobs = self.static_data['n_jobs']
        self.rated = self.static_data['rated']
        self.GA = GA
        self.is_trained = False
        if not GA:
            self.model_dir = os.path.join(cluster_dir, 'RBF_OLS')
        else:
            self.model_dir = os.path.join(cluster_dir, 'GA_RBF_OLS')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        try:
            self.load(self.model_dir)
        except:
            pass

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

    def hidden_act(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        model = self.models
        v = (np.atleast_2d(x)[:, np.newaxis] - model['centroids'][np.newaxis, :]) * model['Radius']
        s = np.sign(v)
        v = s * (v ** 2.)
        v = np.sqrt(v ** 2.)
        v = s * np.exp(-(v ** 3.))
        # v[v < 0] = 2.001 + v[v < 0]
        return v

    def predict(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        model = self.models
        v = (np.atleast_2d(x)[:, np.newaxis] - model['centroids'][np.newaxis, :]) * model['Radius']
        v = np.sqrt((v ** 2.).sum(-1))
        v = np.exp(-(v ** 3.))
        if isinstance(model['W'], np.ndarray):
            v = np.matmul(v, model['W'][:-1]) + model['W'][-1]
        else:
            v = model['W'].predict(v)
        pred = v
        pred[np.where(pred < 0)] = 0
        return pred

    def load(self, pathname):
        # creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
        # creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        # toolbox = base.Toolbox()
        cluster_dir = pathname
        if os.path.exists(os.path.join(cluster_dir, 'rbf_ols.pickle')):
            try:
                f = open(os.path.join(cluster_dir, 'rbf_ols.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                self.__dict__.update(tmp_dict)
            except:
                raise ImportError('Cannot open RBFols model')
        else:
            raise ImportError('Cannot find RBFols model')

    def save(self, pathname):
        f = open(os.path.join(pathname, 'rbf_ols.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'cluster_dir']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()

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

    def optimize_rbf(self, X_train, y_train, X_val, y_val, X_test, y_test, max_samples=3000, n_jobs=1):

        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)

        if X_train.shape[0] > max_samples:
            ind = np.random.randint(low=0, high=X_train.shape[0], size=max_samples)
            X_train = X_train[ind]
            y_train = y_train[ind]
        if self.GA:

            self.optimize_with_deep(X_train, y_train, X_val, y_val, X_test, y_test, self.rated, n_jobs=n_jobs)
        else:
            self.optimize_common_width(X_train, y_train, X_val, y_val, X_test, y_test, self.rated, n_jobs=n_jobs)
        return self.to_dict()

    def to_dict(self):
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger']:
                dict[k] = self.__dict__[k]
        return dict

    def rbf_obj(self, individual, net, X, y, X_val, y_val, X_test, y_test, rated):
        with threadpool_limits(limits=1):
            err_train = net.fit(X, y, X_val, y_val, X_test, y_test, rated, gw=np.array(individual), keep=4,
                                verbose=False)

            pred_test = net.predict(X_test)
            pred_val = net.predict(X_val)
            sse_test, rms_test, mae_test, mse_test = net.compute_metrics(pred_test, y_test, rated)
            sse_val, rms_val, mae_val, mse_val = net.compute_metrics(pred_val, y_val, rated)
            metrics = np.array(
                [0.6 * mae_test + 0.4 * mae_val, 0.6 * sse_test + 0.4 * sse_val, 0.6 * rms_test + 0.4 * rms_val,
                 0.6 * mse_test + 0.4 * mse_val])
        return (metrics)

    def optimize_common_width(self, X, y, X_val, y_val, X_test, y_test, rated, n_jobs=1):
        logger = logging.getLogger('optimize_common_width')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.model_dir, 'log_train_' + self.cluster_name + '.log'), 'a')
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.info('RBF single width optimization running for rule %s....', self.cluster_name)
        print('RBF single width optimization running for rule ', self.cluster_name, ' .....')

        engine = db.create_engine("postgresql://smartrue:chourios48@localhost/hp_optimization")
        table = f'CommnWidthRBF_{self.cluster_name}'
        df = pd.DataFrame(columns=['n_trials', 'width', 'value'])
        with engine.connect() as conn:
            df.to_sql(table, conn, if_exists='replace', schema='rbf_single', dtype={'n_trials': Integer(),
                                                                                    'width': FLOAT(),
                                                                                    'value': FLOAT()})
        net = rbf_ols(engine, table)

        obj_old = np.inf * np.ones(4)
        obj_max = np.inf * np.ones(4)
        obj_min = np.inf * np.ones(4)

        widths = pd.DataFrame(np.arange(0.5, 18, 0.05) * np.mean(X), columns=['width'])

        for w in widths.values.ravel():
            self.rbf_obj(X, y, X_val, y_val, X_test, y_test, w, 3, rated)
        pool = mp.Pool(n_jobs)
        result = [pool.apply_async(self.rbf_obj, args=(w, net, X, y, X_val, y_val, X_test, y_test, rated)) for w
                  in widths]

        results = [p.get() for p in result]
        pool.close()
        pool.terminate()
        pool.join()

        # results = Parallel(n_jobs=self.n_jobs)(
        #     delayed(rbf_optim_single)(net, X, y, X_val, y_val, X_test, y_test, w, 4, rated) for
        #     w in widths)

        res = dict()
        for p in results:
            flag, obj_old, obj_max, obj_min = self._distance(p[0], obj_old, obj_max, obj_min)
            res[str(p[1]['Radius'])] = p[1]['acc']
            if flag:
                self.models = p[1]

        res = pd.DataFrame.from_dict(res, orient='index')
        res.to_csv(os.path.join(self.model_dir, 'single_rbf_res.csv'))
        p = self.models
        logger.info('Best model Radius %s centers %s acc %s', p['Radius'], p['centroids'].shape[0], p['acc'])
        self.is_trained = True
        self.save(self.model_dir)
        logger.info('\n')

    def optimize_with_deep(self, X, y, X_val, y_val, X_test, y_test, rated, n_jobs=1):
        logger = logging.getLogger('optimize_with_deep')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.model_dir, 'log_train_' + self.cluster_name + '.log'), 'a')
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.info('GA rbf running for rule %s....', self.cluster_name)
        print('GA rbf running for rule ', self.cluster_name, '....')

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        # pool = mp.Pool(self.n_jobs)
        # toolbox.register("map", pool.map)
        toolbox.register("attribute1", random.uniform, 0.5, 18)
        toolbox.register("individual1", tools.initRepeat, creator.Individual, toolbox.attribute1, n=X.shape[1])
        toolbox.register("attribute2", random.uniform, 0.5, 18)
        toolbox.register("individual2", tools.initRepeat, creator.Individual, toolbox.attribute2, n=X.shape[1])
        toolbox.register("population", tools.initCycle, list, (toolbox.individual1, toolbox.individual2), n=24)

        toolbox.register("mate", cx_fun, alpha=0.5)
        toolbox.register("mutate", mut_fun, mu=0, sigma=.5, eta=0.8, low=0.5, up=18, indpb=0.8)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", rbf_optim)
        hof = tools.ParetoFront(lambda x, y: (x == y).all())
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("Avg", np.mean)
        stats.register("Std", np.std)
        stats.register("Min", np.min)
        stats.register("Max", np.max)
        toolbox.decorate("mate", checkBounds(0.5, 18))
        toolbox.decorate("mutate", checkBounds(0.5, 18))
        net = rbf_ols()
        best_ind, logbook = optimize(toolbox.population(), toolbox, 40, 60, net, X, y, X_val, y_val, X_test, y_test,
                                     rated, n_jobs, self.path_group, cxpb=0.6, mutpb=0.4,
                                     ngen=48, stats=stats, halloffame=hof, verbose=False)
        logbook = pd.DataFrame(logbook)
        logbook.to_csv(os.path.join(self.model_dir, 'GA_results.csv'))

        net = rbf_ols()

        err_train = net.train_ols(X, y, X_val, y_val, X_test, y_test, rated, gw=np.array(best_ind), keep=3,
                                  verbose=True)

        pred_test = net.predict(X_test)
        pred_val = net.predict(X_val)
        sse_test, rms_test, mae_test, mse_test = net.compute_metrics(pred_test, y_test, rated)
        sse_val, rms_val, mae_val, mse_val = net.compute_metrics(pred_val, y_val, rated)
        logger.info('Best model centers %s acc %s', net.model['centroids'].shape[0],
                    float(0.6 * mae_test + 0.4 * mae_val))
        self.models = ({'centroids': net.model['centroids'], 'Radius': net.model['Radius'], 'W': net.model['W'],
                        'acc': float(0.6 * mae_test + 0.4 * mae_val)})

        self.is_trained = True
        self.save(self.model_dir)
        logger.info('\n')


def optimize(population, toolbox, mu, lambda_, net, X, y, X_val, y_val, X_test, y_test, rated, n_jobs, path_group, cxpb,
             mutpb, ngen,
             stats=None, halloffame=None, verbose=__debug__):
    assert lambda_ >= mu, "lambda must be greater or equal to mu."
    perf = np.inf
    front_best = None
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    # fitnesses=[]
    # for individual in invalid_ind:
    #     m=rbf_optim(individual, X, y, X_val, y_val, X_test, y_test)
    #     fitnesses.append(m)

    pool = mp.Pool(n_jobs)
    results = [
        pool.apply_async(rbf_optim, args=(np.array(individual).ravel(), net, X, y, X_val, y_val, X_test, y_test, rated))
        for individual in invalid_ind]
    fitnesses = [p.get() for p in results]
    pool.close()
    pool.terminate()
    pool.join()
    # fitnesses = Parallel(n_jobs=n_jobs)(delayed(rbf_optim)(np.array(individual).ravel(), net, X, y, X_val, y_val, X_test, y_test, rated) for
    #                                    individual in invalid_ind)

    # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    logbook = tools.Logbook()
    # Gather all the fitnesses in one list and compute the stats
    fits = np.array([ind.fitness.values for ind in population])

    maximums = np.nanmax(fits, axis=0)
    minimums = np.nanmin(fits, axis=0)
    logbook.header = ['gen', 'nevals'] + ['Max_mae:', 'Min_mae:', 'Max_rms:', 'Min_rms:']

    record = {'Max_mae:': maximums[0], 'Min_mae:': minimums[0], 'Max_rms:': maximums[2], 'Min_rms:': minimums[2]}
    print('GA rbf running generation 0')
    print(record)

    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    with elapsed_timer() as eval_elapsed:
        for gen in range(1, ngen + 1):
            # Vary the population
            offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            n_jobs = 18

            pool = mp.Pool(n_jobs)
            results = [pool.apply_async(rbf_optim,
                                        args=(
                                            np.array(individual).ravel(), net, X, y, X_val, y_val, X_test, y_test,
                                            rated))
                       for individual in invalid_ind]
            fitnesses = [p.get() for p in results]
            pool.close()
            pool.terminate()
            pool.join()
            # fitnesses = Parallel(n_jobs=n_jobs)(
            #     delayed(rbf_optim)(np.array(individual).ravel(), net, X, y, X_val, y_val, X_test, y_test, rated) for
            #     individual in invalid_ind)
            # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            fits = np.array([ind.fitness.values for ind in population])

            maximums = np.nanmax(fits, axis=0)
            minimums = np.nanmin(fits, axis=0)
            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Select the next generation population
            population[:] = toolbox.select(offspring, mu)

            # Update the statistics with the new population
            record = {'Max_mae:': maximums[0], 'Min_mae:': minimums[0], 'Max_rms:': maximums[2],
                      'Min_rms:': minimums[2]}
            print('GA rbf running generation ', str(gen))
            print(record)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            front = population
            for i in range(len(front)):
                if front[i].fitness.getValues()[0] < perf:
                    front_best = front[i]
                    perf = front[i].fitness.getValues()[0]
            if verbose:
                print(logbook.stream)
            if eval_elapsed() > 3600:
                break
    return front_best, logbook
