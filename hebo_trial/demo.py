# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
from sklearn.metrics import r2_score
from hebo.design_space.design_space import DesignSpace
from hebo.models.gp.gp import GP
from hebo.acquisitions.acq import LCB
from hebo.optimizers.general import A
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
param_net = [
        {'name' : 'lr',          'type' : 'pow', 'range': [10, -4, -2]},
        {'name' : 'momentum',    'type' : 'num', 'range': [0.1, 0.9]},
        {'name' : 'num_hiddens', 'type' : 'int', 'range': [32, 128]},
        {'name' : 'activation',  'type' : 'cat', 'range': ['relu', 'activation']}
    ]
def f(df_param):
    X, y = load_boston(return_X_y = True)
    tr_x, tst_x, tr_y, tst_y = train_test_split(X, y, shuffle = True, random_state = 42)
    ys  = []
    for i, row in df_param.iterrows():
        model = MLPRegressor(
                solver             = 'sgd', 
                learning_rate_init = row.lr, 
                activation         = row.activation, 
                hidden_layer_sizes = (row.num_hiddens, ), 
                momentum           = row.momentum
                ).fit(tr_x, tr_y)
        py = model.predict(tst_x)
        ys.append(r2_score(tst_y, py))
    return pd.DataFrame(ys, columns = 'obj')


import pandas as pd
import numpy  as np

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

def obj(params : pd.DataFrame) -> np.ndarray:
    return ((params.values - 0.37)**2).sum(axis = 1).reshape(-1, 1)

space = DesignSpace().parse([{'name' : 'x', 'type' : 'num', 'lb' : -3, 'ub' : 3}])
opt   = HEBO(space)
for i in range(5):
    rec = opt.suggest(n_suggestions = 4)
    opt.observe(rec, obj(rec))
    print('After %d iterations, best obj is %.2f' % (i, opt.y.min()))
