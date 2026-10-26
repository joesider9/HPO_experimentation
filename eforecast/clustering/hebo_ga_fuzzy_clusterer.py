import copy
import os
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import multiprocessing as mp
from numpy.random import SeedSequence, default_rng
from joblib import Parallel
from joblib import delayed

from sklearn.linear_model import ElasticNetCV, MultiTaskElasticNetCV
from sklearn.linear_model import LinearRegression

from eforecast.common_utils.logger import create_logger
from eforecast.common_utils.clustering_utils import create_mfs
from eforecast.common_utils.clustering_utils import create_rules
# from hebo_trial.optimizer_mace import MACEBO
from turbo_trial.optimizer_turbo import TurboOptimizer
from eforecast.common_utils.dataset_utils import sync_data_with_dates
from eforecast.common_utils.train_utils import find_free_cpus

warnings.filterwarnings("ignore", category=FutureWarning)


class HeboFuzzyClusterer:

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
        self.path_fuzzy = os.path.join(static_data['path_model'], 'cluster_organizer', 'HEBO')
        try:
            self.load()
            self.rules = self.best_fuzzy_model['rules']
        except:
            pass
        self.path_fuzzy = os.path.join(static_data['path_model'], 'cluster_organizer', 'HEBO')
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

            problem.run(x_train[self.var_lin], y_train, x_test[self.var_lin], y_test)

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
        self.warming = self.static_data['clustering']['warming_iterations']
        self.type = static_data['type']
        self.thres_act = self.static_data['clustering']['thres_act']
        self.var_fuzz = self.static_data['clustering']['fuzzy_var_imp']
        self.min_samples = self.static_data['clustering']['min_samples']
        self.max_samples_ratio = self.static_data['clustering']['max_samples_ratio']
        self.gen = self.static_data['clustering']['gen'] * 5
        self.path_fuzzy = os.path.join(static_data['path_model'], 'cluster_organizer', 'HEBO')
        self.logger = create_logger(logger_name='log_fuzzy.log', abs_path=self.path_fuzzy,
                                    logger_path='log_fuzzy.log', write_type='a')
        self.mfs = mfs
        self.rules = dict()
        for base_case in self.mfs.keys():
            self.rules = create_rules(self.rules, self.mfs[base_case])
        self.space = dict()
        self.param_names = []
        for rule_name, rule in sorted(self.rules.items()):
            for mf in rule:
                name = mf['name']
                param = mf['param']
                xrange = [mf['universe'][0], mf['universe'][-1]]
                prange = mf['prange']
                if len(param) == 2:
                    lo = param[0] - prange if (param[0] - prange) > xrange[0] else xrange[0]
                    up = param[0] + prange if (param[0] + prange) < xrange[1] else xrange[1]

                    self.space.update(self.get_param(f'{name}_mean', 'real', 'float', range=[lo, up]))
                    self.space.update(self.get_param(f'{name}_width', 'real', 'float', range=[0.0001, prange]))
                    self.param_names.append(f'{name}_mean')
                    self.param_names.append(f'{name}_width')
                elif len(param) == 4:
                    for i in range(4):
                        lo = param[0] - prange if (param[0] - prange) > xrange[0] else xrange[0]
                        up = param[3] + prange if (param[3] + prange) < xrange[1] else xrange[1]
                        self.space.update(self.get_param(f'{name}_{i}', 'real', 'float', range=[lo, up]))
                        self.param_names.append(f'{name}_{i}')


    def get_param(self, name, type, dtype, space=None, range=None, values=None):
        return {name: {'type': type,
                       'dtype': dtype,
                       'space': space,
                       'range': range,
                       'values': values}}

    def run(self, x, y, x_test, y_test):
        rules = copy.deepcopy(self.rules)
        constaints = {'thres_act': self.thres_act, 'min_samples': self.min_samples,
                      'max_samples_ratio': self.max_samples_ratio, 'rated': self.rated}
        ss = SeedSequence(12345)
        random_states = [default_rng(s) for s in ss.spawn(self.gen)]
        n_jobs = find_free_cpus(path_group=self.static_data['path_group'])
        manager = mp.Manager()
        shared_trials = manager.list()
        models = Parallel(n_jobs=n_jobs)(delayed(self.evaluate)(trial_number, random_states[trial_number], shared_trials, rules,
                                            x, y, x_test, y_test, constaints)
                     for trial_number in range(self.gen))
        trials = []
        for trial in shared_trials:
            param_dict = dict()
            for key in trial.keys():
                param_dict[key] = trial[key]
            trials.append(param_dict)
        trials = pd.DataFrame(trials)
        trials.to_csv(os.path.join(self.path_fuzzy, f'results_HEBO.csv'))
        best_trial = trials['best_mae_test'].idxmin()

        self.fmodel = [trial[1] for trial in models if trial[0] == best_trial][0]
        self.rules = self.fmodel['rules']

    def evaluate(self, trial_number, random_state, trials, rules, x, y, x_test, y_test, constaints, train=True):
        optimizer = TurboOptimizer(self.space, model_name='gpy')
        if len(trials) > 0:
            y_trial = np.array([trial['value'] for trial in trials]).reshape(-1, 1)
            X_trial = []
            for trial in trials:
                param_dict = dict()
                for key in self.param_names:
                    param_dict[key] = trial[key]
                X_trial.append(param_dict)

            X_trial = pd.DataFrame(X_trial)
            optimizer.observe(X_trial, y_trial, random_state=random_state)
        trial = optimizer.suggest(n_suggestions=1, random_state=random_state, warming=self.warming)[0]
        for rule_name, rule in sorted(rules.items()):
            for mf in rule:
                name = mf['name']
                if mf['type'] == 'gauss':
                    mf['param'] = np.array([trial[f'{name}_mean'], trial[f'{name}_width']])
                    mf['func'] = fuzz.gaussmf(mf['universe'],
                                              mf['param'][0],
                                              np.abs(mf['param'][1]))
                elif mf['type'] == 'trap':
                    individual = np.array([trial[f'{name}_{i}'] for i in range(4)])
                    mf['param'] = sorted(individual)
                    mf['func'] = fuzz.trapmf(mf['universe'], mf['param'])
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

        trial['value'] = objectives[0]
        trial['best_mae_test'] = objectives[1]
        trial['trial_number'] = trial_number
        trials.append(trial)

        fmodel = dict()
        fmodel['rules'] = rules
        fmodel['models'] = lin_models
        fmodel['result'] = objectives[1]
        print(f'Trial {trial_number}, Error = {objectives[1]}')
        return (trial_number, fmodel)
