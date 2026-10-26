# import os
# import pickle
# import joblib
# import traceback
# from multiprocessing import Process
#
# from eforecast.training.train_clusters_on_gpu import train_clusters_on_gpus
from train_clustrers_on_cpu import train_clusters_on_cpus
# from eforecast.training.train_distributed_on_gpu import train_distributed_on_gpus


def fit_clusters(static_data):

    # train_distributed_on_gpus(static_data)
    # train_clusters_on_gpus(static_data)
    train_clusters_on_cpus(static_data)
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
