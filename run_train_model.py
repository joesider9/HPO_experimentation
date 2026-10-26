from configuration.config import config
from eforecast.init.initialize import initializer
from eforecast.clustering.clustering_manager import ClusterOrganizer

from eforecast.training.train_manager import fit_clusters

static_data = initializer(config())
cluster_name = 'RBF_rule_5'
cluster_name1 = 'RBF_rule_6'
method = 'GA_RBFols'
n_jobs = 4

cluster_dir = '/media/smartrue/HHD1/George/models/my_projects/IPTO_ver6_ver0/pv/APE_net/intra-ahead/' \
              'model_ver0/cluster_organizer/RBF/rule_5'
cluster_dir1 = '/media/smartrue/HHD1/George/models/my_projects/IPTO_ver6_ver0/pv/APE_net/intra-ahead/' \
              'model_ver0/cluster_organizer/RBF/rule_6'

if __name__ == '__main__':
    # cluster_organizer = ClusterOrganizer(static_data, is_online=False, train=True, refit=False)
    # cluster_organizer.fit()
    fit_clusters(static_data, cluster={cluster_name: cluster_dir, cluster_name1: cluster_dir1}, method=method, refit=True)
