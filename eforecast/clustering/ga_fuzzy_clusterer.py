import copy
import os
import pickle
import random
import warnings

import joblib
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from joblib import Parallel
from joblib import delayed
from sklearn.linear_model import ElasticNetCV, MultiTaskElasticNetCV
from sklearn.linear_model import LinearRegression

from eforecast.common_utils.logger import create_logger
from eforecast.common_utils.clustering_utils import create_mfs
from eforecast.common_utils.clustering_utils import create_rules
from eforecast.common_utils.clustering_utils import cx_fun
from eforecast.common_utils.clustering_utils import mut_fun
from eforecast.common_utils.clustering_utils import checkBounds
from eforecast.common_utils.dataset_utils import sync_data_with_dates

warnings.filterwarnings("ignore", category=FutureWarning)


class GAFuzzyClusterer:

    def __init__(self, static_data, refit=False):
        self.rule_names = None
        self.is_trained = False
        self.refit = refit
        self.static_data = static_data
        self.rated = static_data['rated']
        self.n_jobs = self.static_data['clustering']['n_jobs']
        self.type = static_data['type']
        self.thres_act = self.static_data['clustering']['thres_act']
        self.var_fuzz = self.static_data['clustering']['fuzzy_var_imp']
        self.n_var_lin = self.static_data['clustering']['n_var_lin']
        self.min_samples = self.static_data['clustering']['min_samples']
        self.max_samples_ratio = self.static_data['clustering']['max_samples_ratio']
        self.abbreviations = self.static_data['clustering']['Gauss_abbreviations']
        self.pop = self.static_data['clustering']['pop']
        self.gen = self.static_data['clustering']['gen']
        self.path_fuzzy = os.path.join(static_data['path_model'], 'cluster_organizer', 'GA')
        try:
            self.load()
            self.rules = self.best_fuzzy_model['rules']
        except:
            pass
        self.path_fuzzy = os.path.join(static_data['path_model'], 'cluster_organizer', 'GA')
        if not os.path.exists(self.path_fuzzy):
            os.makedirs(self.path_fuzzy)
        self.logger = create_logger(logger_name='log_fuzzy.log', abs_path=self.path_fuzzy,
                                    logger_path='log_fuzzy.log', write_type='a')

    def compute_activations(self, x, metadata, with_predictions=False):
        activations = pd.DataFrame(index=x.index, columns=[i for i in sorted(self.best_fuzzy_model['rules'].keys())])
        preds = pd.DataFrame(index=x.index, columns=[i for i in sorted(self.best_fuzzy_model['rules'].keys())])
        var_del = []
        for rule in sorted(self.best_fuzzy_model['rules'].keys()):
            act = []
            for mf in self.best_fuzzy_model['rules'][rule]:
                if mf['var_name'] not in x.columns:
                    var_names = [c for c in x.columns if mf['var_name'].lower() in c.lower()]
                    x[mf['var_name']] = x.loc[:, var_names].mean(axis=1)
                    var_del.append(mf['var_name'])
                act.append(fuzz.interp_membership(mf['universe'], mf['func'], x[mf['var_name']]))
            activations[rule] = np.prod(np.array(act), axis=0)
            indices = activations[rule].index[activations[rule] >= self.thres_act].tolist()
            if len(indices) != 0:
                X1 = x[self.var_lin].loc[indices].values
                preds.loc[indices, rule] = self.best_fuzzy_model['models'][rule].predict(X1).ravel()

        predictions = preds.mean(axis=1)
        if len(var_del) > 0:
            x = x.drop(columns=var_del)
        if with_predictions:
            return predictions, activations
        else:
            return activations

    def run(self, x, y, cv_mask, metadata):
        if not self.refit and self.is_trained:
            return
        x_train = pd.concat([sync_data_with_dates(x, cv_mask[0]), sync_data_with_dates(x, cv_mask[1])])
        y_train = pd.concat([sync_data_with_dates(y, cv_mask[0]), sync_data_with_dates(y, cv_mask[1])])

        x_test = sync_data_with_dates(x, cv_mask[2])
        y_test = sync_data_with_dates(y, cv_mask[2])
        self.var_lin = [c for c in x_train.columns[:self.n_var_lin + 1]]
        print(f'Variables for linear regression')
        print(self.var_lin)
        if len(y_test.shape) > 1:
            n_target = y_test.shape[1]
            if n_target > 1:
                y_train = y_train.mean(axis=1).values
                y_test = y_test.mean(axis=1).values
            self.rated = y_test.values.ravel() if self.rated is None else 1
        else:
            n_target = 1
            self.rated = y_test.values if self.rated is None else 1
        var_del = []
        fuzzy_models = []
        for n_case, case in enumerate(self.var_fuzz):
            print(f'{n_case}th Case')
            base_name = [v for v in sorted(case.keys())][0]
            var_names = [v for v in sorted(case[base_name][0].keys())]
            num_base_mfs = len(case[base_name])
            base_mfs = dict()
            base_mfs = create_mfs(base_mfs, base_name, num_base_mfs, 0, self.abbreviations)

            fuzzy_model = dict()
            fuzzy_model['mfs'] = dict()
            for n in range(len(base_mfs[base_name])):
                fuzzy_model['mfs'][base_name + str(n)] = dict()
                fuzzy_model['mfs'][base_name + str(n)][base_name] = [base_mfs[base_name][n]]

            for var_name in var_names:
                old_num_mf = 0
                for n, base_case in enumerate(case[base_name]):
                    n_mf = base_case[var_name]['mfs']
                    fuzzy_model['mfs'][base_name + str(n)] = create_mfs(
                        fuzzy_model['mfs'][base_name + str(n)]
                        , var_name, n_mf, old_num_mf, self.abbreviations)
                    old_num_mf += n_mf

            for var_name in var_names + [base_name]:
                if var_name not in x_train.columns:
                    var_names = [c for c in x_train.columns if var_name.lower() in c.lower()]
                    if len(var_names) == 0:
                        raise ValueError(f'Cannot find variables associated with {var_name}')
                    x_train[var_name] = x_train.loc[:, var_names].mean(axis=1)
                    x_test[var_name] = x_test.loc[:, var_names].mean(axis=1)
                    var_del.append(var_name)
                if var_name not in self.var_lin:
                    self.var_lin.append(var_name)
            if n_target > 1:
                lin_models = MultiTaskElasticNetCV(cv=5).fit(x_train[self.var_lin].values, y_train.values)
                pred = lin_models.predict(x_test[self.var_lin].values)
                err = (pred - y_test.values) / self.rated
            else:
                lin_models = ElasticNetCV(cv=5).fit(x_train[self.var_lin].values, y_train.values.ravel())
                pred = lin_models.predict(x_test[self.var_lin].values)
                err = (pred.ravel() - y_test.values.ravel()) / self.rated

            rms_before = np.sqrt(np.mean(np.square(err)))
            mae_before = np.mean(np.abs(err))
            print('rms = %s', rms_before)
            print('mae = %s', mae_before)
            self.logger.info("Objective before train: %s", mae_before)
            problem = ClusterProblem(fuzzy_model['mfs'], self.static_data)

            problem.run(x_train[self.var_lin], y_train, x_test[self.var_lin], y_test, 75, 100)

            fuzzy_model = problem.fmodel
            self.logger.info("Objective after train: %s", str(fuzzy_model['result']))
            fuzzy_models.append(fuzzy_model)

            joblib.dump(fuzzy_model, os.path.join(self.path_fuzzy, f'fuzzy_model{n_case}.pickle'))
        fmodel = dict()
        fmodel['mfs'] = dict()
        fmodel['models'] = dict()
        fmodel['rules'] = dict()
        fmodel['result'] = []
        num = 0
        for fuzzy_model in fuzzy_models:
            i = 0
            for rule in fuzzy_model['rules']:
                fmodel['rules']['rule_' + str(num + i)] = fuzzy_model['rules'][rule]
                fmodel['models']['rule_' + str(num + i)] = fuzzy_model['models'][rule]
                i += 1
            fmodel['result'].append(fuzzy_model['result'])
            num += len(fuzzy_model['rules'])
        self.best_fuzzy_model = copy.deepcopy(fmodel)
        self.rule_names = [rule for rule in self.best_fuzzy_model['rules'].keys()]
        if len(var_del) > 0:
            x_train = x_train.drop(columns=var_del)
            x_test = x_test.drop(columns=var_del)
        self.is_trained = True
        self.save()

    def load(self):
        if os.path.exists(os.path.join(self.path_fuzzy, 'fuzzy_model.pickle')):
            try:
                f = open(os.path.join(self.path_fuzzy, 'fuzzy_model.pickle'), 'rb')
                tmp_dict = pickle.load(f)
                f.close()
                tdict = {}
                for k in tmp_dict.keys():
                    tdict[k] = tmp_dict[k]
                self.__dict__.update(tdict)
            except:
                raise ImportError('Cannot open fuzzy model')
        else:
            raise ImportError('Cannot find fuzzy model')

    def save(self):
        f = open(os.path.join(self.path_fuzzy, 'fuzzy_model.pickle'), 'wb')
        dict = {}
        for k in self.__dict__.keys():
            if k not in ['logger', 'static_data', 'data_dir', 'path_fuzzy', 'refit']:
                dict[k] = self.__dict__[k]
        pickle.dump(dict, f)
        f.close()


class ClusterProblem:

    def __init__(self, mfs, static_data):
        self.static_data = static_data
        self.rated = static_data['rated']
        self.n_jobs = self.static_data['clustering']['n_jobs']
        self.type = static_data['type']
        self.thres_act = self.static_data['clustering']['thres_act']
        self.var_fuzz = self.static_data['clustering']['fuzzy_var_imp']
        self.min_samples = self.static_data['clustering']['min_samples']
        self.max_samples_ratio = self.static_data['clustering']['max_samples_ratio']
        self.pop = self.static_data['clustering']['pop']
        self.gen = self.static_data['clustering']['gen']
        self.path_fuzzy = os.path.join(static_data['path_model'], 'cluster_organizer', 'GA')
        self.logger = create_logger(logger_name='log_fuzzy.log', abs_path=self.path_fuzzy,
                                    logger_path='log_fuzzy.log', write_type='a')
        self.mfs = mfs
        self.rules = dict()
        for base_case in self.mfs.keys():
            self.rules = create_rules(self.rules, self.mfs[base_case])
        init_individual = []
        self.lower_bound = []
        self.upper_bound = []
        self.sigma = []
        self.index_constrains = []
        self.number_of_constraints = 0
        for rule_name, rule in sorted(self.rules.items()):
            for mf in rule:
                param = mf['param']
                xrange = [mf['universe'][0], mf['universe'][-1]]
                prange = mf['prange']
                init_individual = init_individual + param
                if len(param) == 2:
                    self.index_constrains.append(np.arange(len(init_individual) - 2, len(init_individual)))
                    self.number_of_constraints = self.number_of_constraints + 3

                    lo = param[0] - prange if (param[0] - prange) > xrange[0] else xrange[0]
                    up = param[0] + prange if (param[0] + prange) < xrange[1] else xrange[1]

                    self.lower_bound.extend([lo, 0.0001])
                    self.upper_bound.extend([up, prange])
                    self.sigma.extend([prange, prange])
                    mf['lower'] = [lo, 0.0001]
                    mf['upper'] = [up, prange]
                elif len(param) == 4:
                    self.index_constrains.append(np.arange(len(init_individual) - 4, len(init_individual)))
                    self.number_of_constraints = self.number_of_constraints + 7
                    lower = []
                    upper = []
                    for i in param:
                        lo = param[0] - prange if (param[0] - prange) > xrange[0] else xrange[0]
                        up = param[3] + prange if (param[3] + prange) < xrange[1] else xrange[1]
                        self.lower_bound.append(lo)
                        self.upper_bound.append(up)
                        self.sigma.append(prange)
                        lower.append(lo)
                        upper.append(up)
                    mf['lower'] = lower
                    mf['upper'] = upper
        self.number_of_variables = len(init_individual)
        self.number_of_objectives = 2
        self.init_individual = init_individual
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()

        attributes = []
        for i in range(self.number_of_variables):
            self.toolbox.register("attribute" + str(i), random.gauss, self.lower_bound[i], self.upper_bound[i])
            attributes.append(self.toolbox.__getattribute__("attribute" + str(i)))

        self.toolbox.register("individual1", tools.initCycle, creator.Individual, tuple(attributes), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual1, n=self.pop)

        self.toolbox.register("mate", cx_fun, alpha=0.5)
        self.toolbox.register("mutate", mut_fun, mu=0, sigma=self.sigma, eta=0.8, low=self.lower_bound,
                              up=self.upper_bound, indpb=0.6)
        self.toolbox.register("select", tools.selTournament, tournsize=4)
        self.toolbox.register("evaluate", evaluate)
        self.hof = tools.ParetoFront(lambda x, y: (x == y).all())
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("Avg", np.mean)
        self.stats.register("Std", np.std)
        self.stats.register("Min", np.min)
        self.stats.register("Max", np.max)
        self.toolbox.decorate("mate", checkBounds(self.lower_bound, self.upper_bound))
        self.toolbox.decorate("mutate", checkBounds(self.lower_bound, self.upper_bound))

        self.population = self.toolbox.population()

    def run(self, x, y, x_test, y_test, mu, lambda_, cxpb=0.6, mutpb=0.4, ):
        perf = np.inf
        front_best = None
        param_ind = creator.Individual(self.init_individual)
        self.population.pop()
        self.population.insert(len(self.population), param_ind)
        i = 0
        while i < 0.25 * len(self.population):
            param_ind = mut_fun(self.init_individual, 0, self.sigma, 0.8, self.lower_bound, self.upper_bound, 0.6)
            param_ind = creator.Individual(param_ind[0])
            self.population.pop(i)
            self.population.insert(i, param_ind)
            i += 1
        assert lambda_ >= mu, "lambda must be greater or equal to mu."

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        rules = copy.deepcopy(self.rules)
        constaints = {'thres_act': self.thres_act, 'min_samples': self.min_samples,
                      'max_samples_ratio': self.max_samples_ratio, 'rated': self.rated}
        fit1 = evaluate(np.array(invalid_ind[-1]).ravel(), rules, x, y, x_test, y_test, constaints)
        print('initial candidate error ', fit1[1])
        fitnesses = Parallel(n_jobs=self.n_jobs)(delayed(evaluate)(np.array(individual).ravel(), rules,
                                                                   x, y, x_test, y_test, constaints) for individual in
                                                 invalid_ind)

        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        if self.hof is not None:
            self.hof.update(self.population)

        self.logbook = tools.Logbook()
        # Gather all the fitnesses in one list and compute the stats
        fits = np.array([ind.fitness.values for ind in self.population])

        maximums = np.nanmax(fits, axis=0)
        minimums = np.nanmin(fits, axis=0)
        self.logbook.header = ['gen', 'nevals'] + ['Max_sse:', 'Min_sse:', 'Max_mae:', 'Min_mae:']
        self.logger.info('Iter: %s, Max_sse: %s, Min_mae: %s', 0, *minimums)
        record = {'Max_sse:': maximums[0], 'Min_sse:': minimums[0], 'Max_mae:': maximums[1], 'Min_mae:': minimums[1]}
        print('Fuzzy running generation 0')
        print(record)
        self.logger.info('Fuzzy running generation 0')
        self.logger.info(record)

        self.logbook.record(gen=0, nevals=len(invalid_ind), **record)

        print(self.logbook.stream)

        # Begin the generational process
        for gen in range(1, self.gen + 1):
            # Vary the population
            offspring = algorithms.varOr(self.population, self.toolbox, lambda_, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = Parallel(n_jobs=self.n_jobs)(delayed(evaluate)(np.array(individual).ravel(), rules,
                                                                       x, y, x_test, y_test, constaints) for individual
                                                     in
                                                     invalid_ind)
            # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            fits = np.array([ind.fitness.values for ind in self.population])

            maximums = np.nanmax(fits, axis=0)
            minimums = np.nanmin(fits, axis=0)
            # Update the hall of fame with the generated individuals
            if self.hof is not None:
                self.hof.update(self.population)

            # Select the next generation population
            self.population[:] = self.toolbox.select(offspring, mu)

            # Update the statistics with the new population
            record = {'Max_sse:': maximums[0], 'Min_sse:': minimums[0], 'Max_mae:': maximums[1],
                      'Min_mae:': minimums[1]}

            print('Fuzzy running generation ', str(gen))
            print(record)
            self.logger.info('Fuzzy running generation %s', str(gen))
            self.logger.info(record)

            self.logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            front = self.population
            for i in range(len(front)):
                if front[i].fitness.getValues()[0] < perf:
                    front_best = front[i]
                    perf = front[i].fitness.getValues()[0]
        self.logger.info('Iter: %s, Max_sse: %s, Min_mae: %s', str(gen), *minimums)
        self.fmodel = evaluate(np.array(front_best).ravel(), rules, x, y, x_test, y_test, constaints, train=False)
        self.rules = self.fmodel['rules']


def evaluate(individual, rules, x, y, x_test, y_test, constaints, train=True):
    i = 0
    for rule_name, rule in sorted(rules.items()):
        for mf in rule:
            if mf['type'] == 'gauss':
                mf['param'] = individual[i:i + 2]
                mf['func'] = fuzz.gaussmf(mf['universe'],
                                          mf['param'][0],
                                          np.abs(mf['param'][1]))
                i += 2
            elif mf['type'] == 'trap':
                mf['param'] = sorted(individual[i:i + 4])
                mf['func'] = fuzz.trapmf(mf['universe'], mf['param'])
                i += 4
        rules[rule_name] = rule
    activations = pd.DataFrame(index=x.index, columns=[rule for rule in sorted(rules.keys())])
    for rule in sorted(rules.keys()):
        act = []
        for mf in rules[rule]:
            act.append(fuzz.interp_membership(mf['universe'], mf['func'], x[mf['var_name']]))
        activations[rule] = np.prod(np.array(act), axis=0)

    lin_models = dict()
    total = 0
    for rule in sorted(activations.columns):
        indices = activations[rule].index[activations[rule] >= constaints['thres_act']].tolist()
        if constaints['min_samples'] < len(indices) < constaints['max_samples_ratio'] * x.shape[0]:
            X1 = x.loc[indices].values
            y1 = y.loc[indices].values

            lin_models[rule] = LinearRegression().fit(X1, y1.ravel())

        else:
            del rules[rule]

        if not train:
            print(len(indices))
            print(f"Number of samples of rule {rule} is {len(indices)}")
            total += len(indices)
    if not train:
        print(total)
        print(f"Number of samples of dataset with {x.shape[0]} is {total}")

    activations_test = pd.DataFrame(index=x_test.index,
                                    columns=[rule for rule in sorted(rules.keys())])
    for rule in sorted(rules.keys()):
        act = []
        for mf in rules[rule]:
            act.append(fuzz.interp_membership(mf['universe'], mf['func'], x_test[mf['var_name']]))
        activations_test[rule] = np.prod(np.array(act), axis=0)

    preds = pd.DataFrame(index=x_test.index, columns=sorted(lin_models.keys()))
    for rule in sorted(rules.keys()):
        indices = activations_test[rule].index[activations_test[rule] >= constaints['thres_act']].tolist()
        if len(indices) != 0:
            X1 = x_test.loc[indices].values
            preds.loc[indices, rule] = lin_models[rule].predict(X1).ravel()

    pred = preds.mean(axis=1)
    pred[pred.isnull()] = 1e+15
    rated = y_test.values.ravel() if constaints['rated'] is None else 1
    err = (pred.values.ravel() - y_test.values.ravel()) / rated
    objectives = [np.sum(np.square(err)), np.mean(np.abs(err))]
    if train:
        return objectives
    else:
        fmodel = dict()
        fmodel['rules'] = rules
        fmodel['models'] = lin_models
        fmodel['result'] = objectives[1]
        print('Error = ', objectives[1])
        return fmodel
