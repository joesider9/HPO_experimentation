import numpy as np
import pandas as pd


def concat_by_columns(df1, df2, name1=None, name2=None):
    if name1 is None or name1 is None:
        raise ValueError('Provide some names for datasets')
    dates = df1.index.intersection(df2.index)
    if len(dates) == 0:
        raise ValueError('Cannot sync datasets. there is no common dates')
    print(f'Merge datasets {name1} and {name2}. {dates.shape[0]} common dates ')
    return pd.concat([df1.loc[dates], df2.loc[dates]], axis=1)


def concatenate_numpy(data1, dates1, data2, dates2):
    dates = dates1.intersection(dates2)
    data1 = data1[dates1.get_indexer(dates)]
    data2 = data2[dates2.get_indexer(dates)]
    return np.concatenate([data1, data2], axis=1)


def sync_data_row_with_tensors(data_tensor=None, dates_tensor=None, data_row=None, dates_row=None):
    dates = dates_tensor.intersection(dates_row)
    if not isinstance(data_row, pd.DataFrame):
        raise ValueError('data_row should be dataframe')
    if isinstance(data_tensor, dict):
        for key, data in data_tensor.items():
            if not isinstance(data, np.ndarray):
                raise ValueError('data_tensor should be np.ndarray')
            data_tensor[key] = data[dates_tensor.get_indexer(dates)]
    else:
        data_tensor = data_tensor[dates_tensor.get_indexer(dates)]
    data_row = data_row.loc[dates]
    return data_tensor, data_row, dates


def sync_target_with_tensors(target=None, data_tensor=None, dates_tensor=None, data_row=None):
    dates = dates_tensor.intersection(target.index)
    if isinstance(data_tensor, dict):
        for key, data in data_tensor.items():
            if isinstance(data, np.ndarray):
                data_tensor[key] = data[dates_tensor.get_indexer(dates)]
            else:
                data_tensor[key] = data.iloc[dates_tensor.get_indexer(dates)]
    else:
        data_tensor = data_tensor[dates_tensor.get_indexer(dates)]
    target = target.loc[dates]
    if data_row is not None:
        data_row = data_row.loc[dates]
        return [data_tensor, data_row], target
    else:
        return data_tensor, target


def sync_datasets(df1, df2, name1=None, name2=None):
    if name1 is None or name1 is None:
        raise ValueError('Provide some names for datasets')
    dates = df1.index.intersection(df2.index)
    if len(dates) == 0:
        raise ValueError('Cannot sync datasets. there is no common dates')
    print(f'Merge datasets {name1} and {name2}. {dates.shape[0]} common dates ')
    return df1.loc[dates], df2.loc[dates]


def get_slice(data, ind):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = get_slice(value, ind)
    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.iloc[ind]
    else:
        return data[ind]
    return data


def sync_data_with_dates(x, dates, dates_x=None):
    if dates_x is not None:
        dates_new = dates_x.intersection(dates)
        ind = dates_x.get_indexer(dates_new)
        return x[ind]
    else:
        dates_new = x.index.intersection(dates)
        return x.loc[dates_new]


def split_data_val(activations, test_size=0.15, random_state=42, continuous=False, dates_freeze=None, thes_act=0.01):
    rng = np.random.RandomState(random_state)

    N = activations.shape[0]

    N_val = int(test_size * N)

    rules = activations.sum(axis=0).sort_values().index.to_list()
    activations[activations > thes_act] = 1
    N_rules = activations.sum(axis=0).sort_values().values
    weights = N_rules / N
    dates = pd.DatetimeIndex([])
    for rule_name, n_rule, w_rule in zip(rules, N_rules, weights):
        act_rule = activations[rule_name].iloc[np.where(activations[rule_name] >= thes_act)]
        common_dates = act_rule.index.intersection(dates)
        common_rate = common_dates.shape[0] / N_val
        if common_dates.shape[0] > 0:
            act_rule = act_rule.drop(index=common_dates)
        n_val = int((w_rule - common_rate) * N_val)
        if n_val > 0:
            if dates_freeze is not None:
                act_rule = act_rule.drop(index=act_rule.index.intersection(dates_freeze))
            if continuous:
                dates_rule = act_rule.index[-n_val:]
            else:
                dates_rule = act_rule.sample(n=n_val, random_state=rng).index
            dates = dates.append(dates_rule).unique()
        print(dates.shape[0])
    val_rate = dates.shape[0] / N
    diff_rate = val_rate - test_size
    if diff_rate > 0:
        n_keep = int((1 - diff_rate) * N_val)
        ind = np.random.choice(dates.shape[0], n_keep, replace=False)
        dates = dates[ind]
    dates_train = activations.index.difference(dates)
    if dates_freeze is not None:
        dates_train = dates_train.difference(dates_freeze)
    return dates_train, dates
