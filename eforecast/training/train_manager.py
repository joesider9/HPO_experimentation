import os
import pickle
import joblib
import traceback
from concurrent.futures import ThreadPoolExecutor

from eforecast.training.train_clusters_on_gpu import train_clusters_on_gpus
from eforecast.training.train_rbfnns_on_cpus import train_rbfnn_on_cpus
from eforecast.training.train_clustrers_on_cpu import train_clusters_on_cpus
# from eforecast.training.train_distributed_on_gpu import train_distributed_on_gpus


def fit_clusters(static_data, cluster=None, method=None, refit=False):
    path_group = static_data['path_group']
    gpu_status = 0
    joblib.dump(gpu_status, os.path.join(path_group, 'gpu_status.pickle'))
    train_rbfnn_on_cpus(static_data, cluster = cluster, method=method, refit=refit)
    # with ThreadPoolExecutor(2) as executor:
    #     _ = executor.submit(train_clusters_on_gpus, static_data, cluster=cluster, method=method, refit=refit)
    #     _ = executor.submit(train_clusters_on_cpus, static_data, cluster=cluster, method=method, refit=refit)

    # try:
    #     procs = []
    #     procs.append(Process(target=train_on_cpus, args=(self.static_data_base, self.projects)))
    #     procs.append(Process(target=train_on_gpus, args=(self.static_data_base, self.projects)))
    #     for p in procs:
    #         p.daemon = False
    #         p.start()
    #     for p in procs:
    #         p.join()
    # except Exception as e:
    #     tb = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
    #     print("".join(tb))
    #     return "".join(tb)

    return 'Done'
