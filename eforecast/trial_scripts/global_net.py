import os

import numpy as np
import pandas as pd

from configuration.config import config
from eforecast.init.initialize import initializer

from eforecast.dataset_creation.data_feeder import DataFeeder
from eforecast.dataset_creation.files_manager import FilesManager
from eforecast.data_preprocessing.data_scaling import Scaler

from eforecast.deep_models.tf_1x.global_network import DistributedDeepNetwork
from eforecast.clustering.clustering_manager import ClusterOrganizer

from eforecast.common_utils.dataset_utils import sync_datasets
from eforecast.common_utils.dataset_utils import sync_target_with_tensors

static_data = initializer(config(), read_data=False)
method = list(static_data['Global']['experiment_tag'])[0]
merge = 'by_area'
compress = None if 'cnn' in method else 'dense'
scale_method = 'minmax'
what_data = 'cnn' if 'cnn' in method else 'row_dict'
ID = 6
path_network = os.path.join(static_data['path_model'], 'Distributed', f'Distributed_{ID}')


def load_data(train=True):
    data_feeder = DataFeeder(static_data, online=False, train=train)

    X, metadata = data_feeder.feed_inputs(merge=merge, compress=compress,
                                          scale_nwp_method=scale_method,
                                          what_data=what_data)
    inverse = False if train else True
    y, _ = data_feeder.feed_target(inverse=inverse)
    if isinstance(X, pd.DataFrame):
        X, y = sync_datasets(X, y, name1='inputs', name2='target')
    elif isinstance(X, list):
        data_row = X[1]
        data = X[0]
        X, y = sync_target_with_tensors(target=y, data_tensor=data, dates_tensor=metadata['dates'], data_row=data_row)
    elif isinstance(X, dict):
        X, y = sync_target_with_tensors(target=y, data_tensor=X, dates_tensor=metadata['dates'])
    else:
        X, y = sync_target_with_tensors(target=y, data_tensor=X, dates_tensor=metadata['dates'])
    metadata['dates'] = X.index
    return X, y, metadata


def fit(X, y, cv_mask, metadata):
    experiment_params = {'name': f'Distributed_{method}',
                         'merge': merge,
                         'compress': compress,
                         'scale_nwp_method': scale_method}
    for param, value in static_data['Global'].items():
        if isinstance(value, set):
            v = list(value)[0]
        elif isinstance(value, list):
            v = value[0]
        else:
            v = value
        experiment_params[param] = v
    experiment_params['experiment'] = static_data['experiments'][method]
    network = DistributedDeepNetwork(static_data, path_network, params=experiment_params, train=True, refit=True)
    network.fit(X, y, cv_mask, metadata)


def predict(X):
    network = DistributedDeepNetwork(static_data, path_network, train=False)
    return network.predict(X)


def predict_rbf_clusterer():
    cluster_organizer = ClusterOrganizer(static_data, is_online=False, train=False)
    predictions, activations = cluster_organizer.predict('RBF')
    return predictions


if __name__ == '__main__':
    if not os.path.exists(path_network):
        os.makedirs(path_network)
    scalers = Scaler(static_data)
    file_manager = FilesManager(static_data, is_online=False, train=True)
    cv_mask_train = file_manager.check_if_exists_cv_data()
    X_train, y_train, metadata_train = load_data()
    fit(X_train, y_train, cv_mask_train, metadata_train)

    X_test, y_test, metadata_test = load_data(train=False)
    y_ = predict(X_test)
    y_ = pd.DataFrame(y_, index=metadata_test['dates'], columns=['global_net'])

    y_rbf_ = predict_rbf_clusterer()
    print('Global')
    print(np.mean(np.abs(scalers.inverse_transform_data(y_, f"target_{static_data['scale_target_method']}").values -
                         y_test.values) / static_data['rated']))
    print('clusterer')
    print(np.mean(np.abs(scalers.inverse_transform_data(y_rbf_, f"target_{static_data['scale_target_method']}").values -
                         y_test.values) / static_data['rated']))




