from configuration.config_utils import define_n_jobs
from configuration.config_project import TYPE
from configuration.experiments_base import experiments

PROJECT_METHODS = {'CNN': False,
                   'LSTM': False,
                   'MLP': False,
                   'RF': False,
                   'CatBoost': False,
                   'lasso': False,
                   'RBFols': True,
                   'GA_RBFols': False,
                   }

FEATURE_SELECTION_METHODS = [None, 'lasso', 'ShapValues']  # Could be None, 'lasso','FeatureImportance', 'ShapValues'

combine_methods = ['rls', 'bcp', 'elastic_net', 'kmeans', 'knn', 'mlp', 'rf',
                   'average']  # 'rls', 'bcp', 'mlp', 'bayesian_ridge', 'elastic_net', 'ridge',


# 'average', 'kmeans'

def fuzzy_variables():
    if TYPE == 'pv':
        var_imp = [{'hour': [{'flux': {'mfs': 2}},
                             {'flux': {'mfs': 3}},
                             {'flux': {'mfs': 4}},
                             {'flux': {'mfs': 3}},
                             {'flux': {'mfs': 2}},
                             ]}]
    elif TYPE == 'wind':
        var_imp = [{'direction': [{'wind': {'mfs': 6}},
                                  {'wind': {'mfs': 6}},
                                  # {'wind': {'mfs': 3}},
                                  # {'wind': {'mfs': 3}},
                                  ]}]
    elif TYPE == 'load':
        var_imp = [
            {'sp_index': [
                {'hour': {'mfs': 8}, 'month': {'mfs': 3}},
                {'hour': {'mfs': 2}, 'month': {'mfs': 2}},
                {'hour': {'mfs': 2}, 'month': {'mfs': 1}},
            ]
            },
            # {'sp_index': [
            #     {'hour': {'mfs': 4}, 'month': {'mfs': 6}},
            #     {'hour': {'mfs': 2}, 'month': {'mfs': 2}},
            #     {'hour': {'mfs': 1}, 'month': {'mfs': 2}},
            # ]
            # },
            {'sp_index': [
                {'hour': {'mfs': 4}, 'Temp': {'mfs': 6}},
                {'hour': {'mfs': 2}, 'Temp': {'mfs': 2}},
                {'hour': {'mfs': 1}, 'Temp': {'mfs': 2}},
            ]
            }
        ]
    elif TYPE == 'fa':
        var_imp = [
            {'sp_index': [
                {'month': {'mfs': 4}},
                {'month': {'mfs': 1}},
            ]
            },
            {'sp_index': [
                {'temp_max': {'mfs': 4}},
                {'temp_max': {'mfs': 1}},
            ]
            },
            {'sp_index': [
                {'dayweek': {'mfs': 4}},
                {'dayweek': {'mfs': 1}},
            ]
            }
        ]
    else:
        var_imp = []
    return var_imp


def rbf_variables():
    if TYPE == 'pv':
        var_imp = [['hour', 'cloud', 'flux']]
    elif TYPE == 'wind':
        var_imp = [['direction', 'wind']]
    elif TYPE == 'load':
        var_imp = [['sp_index', 'hour', 'month', 'Temp']]
    elif TYPE == 'fa':
        var_imp = [['sp_index', 'month', 'temp_max', 'dayweek']]
    else:
        var_imp = [[]]
    return var_imp


def config_methods():
    static_data = dict()
    static_data['project_methods'] = PROJECT_METHODS
    static_data['feature_selection_methods'] = FEATURE_SELECTION_METHODS
    static_data['experiments'] = experiments
    static_data['resampling'] = True
    static_data['val_test_ratio'] = 0.15

    n_jobs = define_n_jobs()
    static_data['n_jobs'] = n_jobs['n_jobs']
    static_data['n_gpus'] = n_jobs['n_gpus']
    static_data['intra_op'] = n_jobs['intra_op']

    static_data['CNN'] = {'experiment_tag': {'cnn1', 'cnn2', 'cnn3'},
                          'n_trials': 30,
                          'filters': 12,
                          'batch_size': 128,
                          'max_iterations': 100,
                          'warming_iterations': 4,
                          'learning_rate': {1e-4, 1e-3, 1e-5},
                          'act_func': {'tanh', 'sigmoid'},
                          'n_jobs': n_jobs['n_jobs_cnn_3d']
                          }
    static_data['LSTM'] = {'units': 24,
                           'n_trials': 6,
                           'experiment_tag': {'lstm1', 'lstm2', 'lstm3', 'lstm4'},
                           'batch_size': 512,
                           'act_func': {'elu', 'sigmoid', 'tanh'},
                           'max_iterations': 100,
                           'warming_iterations': 4,
                           'learning_rate': {1e-3},
                           'n_jobs': n_jobs['n_jobs_lstm']
                           }
    static_data['MLP'] = {'experiment_tag': {'mlp1', 'mlp2', 'mlp3'},
                          'n_trials': 4,
                          'hold_prob': 1,
                          'batch_size': 256,
                          'act_func': {'elu', 'sigmoid', 'tanh'},
                          'max_iterations': 600,
                          'warming_iterations': 4,
                          'learning_rate': {1e-4},
                          'n_jobs': n_jobs['n_jobs_mlp'],
                          }
    static_data['Global'] = {'experiment_tag': {'distributed_cnn1', 'distributed_mlp1', 'distributed_mlp2'},
                             'keep_n_models': 4,
                             'n_trials': 8,
                             'hold_prob': 1,
                             'batch_size': {32, 64, 128},
                             'filters': 6,
                             'nwp_data_merge': {'by_area', 'by_area_variable'},
                             'compress_data': {'semi_full', 'dense'},
                             'scale_nwp_method': {'standard'},
                             'act_func': {'sigmoid'},
                             'max_iterations': 600,
                             'warming_iterations': 4,
                             'learning_rate': {1e-4, 0.5e-3, 1e-3},
                             'thres_act': 0.001,
                             'min_samples': 200,
                             'max_samples_ratio': 0.8,
                             'train_schedule': 'simple',  # simple or complex
                             'n_rules': {9},
                             'is_fuzzy': {True, False},  # If True creates its own activations
                             'clustering_method': {None, 'RBF', 'GA'},  # None RBF or GA
                             'rbf_var_imp': rbf_variables()[0],
                             'data_type': {'scaling': 'minmax',
                                           'merge': 'by_area',
                                           'compress': 'dense'},  # Data type used for clustering when it is Fuzzy
                             'n_jobs': n_jobs['n_jobs_cnn_3d'],
                             }
    CLUSTERING_METHOD = ['RBF', 'GA', 'HEBO']  # RBF or GA
    static_data['clustering'] = {'n_jobs': n_jobs['n_jobs'],
                                 'data_type': {'scaling': 'minmax',
                                               'merge': 'by_area',
                                               'compress': 'dense'},
                                 'methods': CLUSTERING_METHOD,
                                 'clusters_for_method': 'both',  # RBF or GA or both
                                 'prediction_for_method': ['RBF', 'GA'],  # RBF or GA or both
                                 'thres_act': 0.001,
                                 'Gauss_abbreviations': {'hdd_h', 'temp', 'flux', 'wind', 'temp', 'load', 'power',
                                                         'u10', 'v10' 'gb_h0'},
                                 'fuzzy_var_imp': fuzzy_variables(),
                                 'rbf_var_imp': rbf_variables(),
                                 'n_var_lin': 5,
                                 'min_samples': 500,
                                 'max_samples_ratio': 0.6,
                                 'pop': 100,
                                 'gen': 300,
                                 'warming_iterations': 10,
                                 'n_rules': 9,
                                 'params': {'experiment_tag': 'exp_fuzzy1',
                                            'n_trials': 1,
                                            'hold_prob': 1,
                                            'batch_size': 128,
                                            'filters': 12,
                                            'train_schedule': 'simple',  # Simple or complex
                                            'act_func': None,
                                            'max_iterations': 1000,
                                            'warming_iterations': 100,
                                            'learning_rate': 0.25e-3,
                                            'n_jobs': 1,
                                            }
                                 }
    static_data['RF'] = {'n_trials': 4,
                         'warming_iterations': 4,
                         'max_depth': [1, 48],
                         'max_features': [0.2, 1],
                         'min_samples_leaf': [2, 200],
                         'min_samples_split': [2, 100],
                         'max_samples': [0.3, 1],
                         'oob_score': True
                         }
    static_data['CatBoost'] = {'n_trials': 4,
                               'warming_iterations': 4,
                               'iterations': 1000,
                               'learning_rate': [0.0005, 0.2],
                               'l2_leaf_reg': [1, 10],
                               "objective": {"RMSE", "MAE"},
                               'min_data_in_leaf': [2, 100],
                               "colsample_bytree": [0.3, 0.9],
                               "depth": [1, 48],
                               "boosting_type": {"Ordered", "Plain"},
                               "bootstrap_type": {"Bayesian", "Bernoulli", "MVS"},
                               "bagging_temperature": [0, 10],
                               "subsample": [0.1, 1],
                               "eval_metric": "MAE",
                               }
    static_data['lasso'] = {'n_trials': 4,
                            'warming_iterations': 4,
                            'max_iter': 150000,
                            'eps': [0.005, 2],
                            'fit_intercept': {True, False},
                            "selection": {"cyclic", "random"},
                            }
    static_data['RBFols'] = {'n_trials': 10,
                             'warming_iterations': 4,
                              'width': [0.01, 15],
                              'keep': [2, 6]
                              }
    static_data['GA_RBFols'] = {'n_trials': 1000,
                                'warming_iterations': 4,
                                'width': [0.01, 15],
                                'keep': [2, 6]
                                }
    static_data['combining'] = {'methods': combine_methods,
                                'data_type': {'scaling': 'minmax',
                                              'merge': 'by_area',
                                              'compress': 'dense'},
                                }
    return static_data
