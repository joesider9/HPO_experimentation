"""
Functions for configuration files

AUTHOR: G.SIDERATOS

Date: September 2022
"""
import os
import sys


def find_pycharm_path():
    if sys.platform == 'linux':
        pycharm_path = '/home/smartrue/Dropbox/current_codes/PycharmProjects'
    else:
        if os.path.exists('D:/'):
            pycharm_path = 'D:/Dropbox/current_codes/PycharmProjects'
        else:
            pycharm_path = 'C:/Dropbox/current_codes/PycharmProjects'
    return pycharm_path


def config_folders(docker):
    """
    Define the folders for your PC
    - pycharm_path: Path of your codes where the code of projects group is located
    - sys_folder: Path where the model weights are saved
    - nwp_folder: Path where nwp grib files are located e.g. nwp_folder + /ECMWF
    param docker: Runtime environment
    return: dict with pycharm_path, sys_folder, nwp_folder
    """
    folders = dict()
    folders['pycharm_path'] = find_pycharm_path()
    if docker:
        folders['sys_folder'] = '/models/'
        folders['nwp_folder'] = '/nwp/'
    else:
        if sys.platform == 'linux':
            folders['sys_folder'] = '/media/smartrue/HHD1/George/models/'
            folders['nwp_folder'] = '/media/smartrue/HHD2/'
        else:
            if os.path.exists('D:/'):
                folders['sys_folder'] = 'D:/models/'
                folders['nwp_folder'] = 'D:/Dropbox/'
            else:
                folders['sys_folder'] = 'C:/models/'
                folders['nwp_folder'] = 'C:/Dropbox/'
    for folder in ['sys_folder', 'nwp_folder', 'pycharm_path']:
        try:
            assert os.path.exists(folders[folder])
        except AssertionError:
            raise FileNotFoundError(f'{folders[folder]} does not exist')
    return folders


def find_nwp_attrs(model, path_nwp_source):
    """
    Returns the nwp grib folder
    param model: str Name of the nwp model
    param path_nwp_source: root path that the folders with the original nwp grib files of all models exists
    return: dict path_nwp: the folder of the selected nwp model and resolution
    """
    if model == 'skiron':
        path_nwp = os.path.join(path_nwp_source, 'SKIRON')
        resolution = 0.05
    elif model == 'ecmwf':
        path_nwp = os.path.join(path_nwp_source, 'ECMWF')
        resolution = 0.1
    elif model == 'gfs':
        path_nwp = os.path.join(path_nwp_source, 'GFS')
        resolution = 0.25
    else:
        path_nwp = None
        resolution = None
    if path_nwp is not None:
        try:
            assert os.path.exists(path_nwp)
        except AssertionError:
            raise FileNotFoundError(f'{path_nwp} does not exist')
    return {'model': model, 'resolution': resolution,
            'path_nwp_source': path_nwp}


def define_n_jobs():
    if sys.platform != 'linux':
        jobs = {'n_cpus': 8,  # ALL CPUS
                'n_jobs': 6,
                'n_jobs_rbfnn': 4,
                'n_jobs_lstm': 3,
                'n_jobs_cnn_3d': 1,
                'n_jobs_cnn': 1,
                'n_jobs_mlp': 2,
                'intra_op': 2,
                'n_gpus': 1}
    else:
        jobs = {'n_cpus': 20,
                'n_jobs': 16,  # ALL CPUS
                'n_jobs_rbfnn': 3,
                'n_jobs_lstm': 2,
                'n_jobs_cnn_3d': 2,
                'n_jobs_cnn': 2,
                'n_jobs_mlp': 3,
                'intra_op': 2,
                'n_gpus': 2}
    return jobs
