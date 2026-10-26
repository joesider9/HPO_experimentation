import numpy as np
from configuration.config_project import config_project

static_data = config_project()
TYPE = static_data['type']
NWP_DATA_MERGE = ['by_area', 'by_area_variable']  # 'all', 'by_area', 'by_area_variable', 'by_variable',
# by_nwp_provider

DATA_COMPRESS = ['semi_full', 'dense', 'full']  # dense or semi_full or full or load

DATA_NWP_SCALE = ['minmax', 'standard']

flag_row_dict = all(['by_' in merge_method for merge_method in NWP_DATA_MERGE])
if flag_row_dict:
    what_data = ['row_all', 'row_dict']
else:
    what_data = ['row_all']
if static_data['horizon_type'] == 'multi-output':
    what_data += ['row_dict_distributed']
DATA_STRUCTURE = ['row', 'lstm'] if TYPE == 'load' else what_data + ['cnn']

DATA_TARGET_SCALE = 'max_abs'
DATA_ROW_SCALE = 'minmax'
USE_DATA_BEFORE_AND_AFTER_TARGET = True

REMOVE_NIGHT_HOURS = False

USE_DIFF_BETWEEN_LAGS = False

NWP_MODELS = static_data['NWP']

HORIZON = static_data['horizon']


def variables():
    if TYPE == 'pv':
        # Labels for NWP variables: Flux, Cloud, Temperature
        variable_list = [
            variable_wrapper('Flux', nwp_provider='ALL', transformer='clear_sky'),
            variable_wrapper('Cloud', nwp_provider='ecmwf'),
            variable_wrapper('azimuth', input_type='calendar', source='astral'),
            variable_wrapper('zenith', input_type='calendar', source='astral'),
            variable_wrapper('hour', input_type='calendar', source='index'),
            variable_wrapper('month', input_type='calendar', source='index')
        ]
        if HORIZON > 0:
            var_obs = variable_wrapper('Obs', input_type='timeseries', source='target', lags=3,
                                       timezone=static_data['local_timezone'])
            variable_list.append(var_obs)
    elif TYPE == 'wind':
        # Labels for NWP variables: Uwind, Vwind, WS, WD, Temperature
        variable_list = [
            variable_wrapper('WS', nwp_provider='ALL'),
            variable_wrapper('WD', nwp_provider='ecmwf'),
            variable_wrapper('azimuth', input_type='calendar', source='astral'),
            variable_wrapper('zenith', input_type='calendar', source='astral')
        ]
        if HORIZON > 0:
            var_obs = variable_wrapper('Obs', input_type='timeseries', source='target', lags=3,
                                       timezone=static_data['local_timezone'])
            variable_list.append(var_obs)
    elif TYPE == 'load':
        if HORIZON > 0:
            lags = [-i for i in range(1, 13)] + [-i for i in range(22, 28)] + [-i for i in range(47, 53)] + \
                   [-i for i in range(166, 176)] + [-192]

            lags_days = [-24 * i for i in range(0, 8)]
        else:
            lags = [-i for i in range(48, 60)] + [-i for i in range(72, 77)] + [-i for i in range(96, 100)] + \
                   [-i for i in range(120, 122)] + [-i for i in range(144, 146)] + [-i for i in range(166, 176)] + \
                   [-i for i in range(190, 192)] + [-216] + ['last_year_lags']

            lags_days = [-24 * i for i in range(0, 8)]

        variable_list = [
            variable_wrapper('load', input_type='timeseries', source='target', lags=lags,
                             timezone=static_data['local_timezone']),
            variable_wrapper('Temp_max', nwp_provider='ALL', lags=lags_days),
            variable_wrapper('Temp_min', nwp_provider='ALL', lags=lags_days),
            variable_wrapper('Temperature', nwp_provider='ALL', lags=[i for i in range(0, 13, 2)]),
            variable_wrapper('Cloud', nwp_provider='ALL'),
            variable_wrapper('WS', nwp_provider='ALL'),
            variable_wrapper('WD', nwp_provider='ALL'),
            variable_wrapper('dayweek', input_type='calendar', source='index'),
            variable_wrapper('sp_index', input_type='calendar', source='index', lags=lags_days),
            variable_wrapper('hour', input_type='calendar', source='index'),
            variable_wrapper('month', input_type='calendar', source='index'),
            variable_wrapper('azimuth', input_type='calendar', source='astral'),
            variable_wrapper('zenith', input_type='calendar', source='astral')
        ]
    else:
        raise NotImplementedError(f'Define variables for type {TYPE}')
    return variable_list


def variable_wrapper(name, input_type='nwp', source='grib', lags=None, timezone='UTC', nwp_provider=None,
                     transformer=None):
    if nwp_provider is not None:
        if nwp_provider == 'ALL':
            providers = [nwp_model['model'] for nwp_model in NWP_MODELS]
        else:
            providers = [nwp_model['model'] for nwp_model in NWP_MODELS if nwp_model['model'] == nwp_provider]
    else:
        providers = None

    return {'name': name,
            'type': input_type,  # nwp or timeseries or calendar
            'source': source,  # use 'target' for the main timeseries otherwise 'grib', 'database' for nwps,
            # a column label of input file csv or a csv file extra, 'index' for calendar variables,
            # 'astral' for zenith, azimuth
            'lags': define_variable_lags(name, input_type, lags),
            'timezone': timezone,
            'transformer': transformer,
            'nwp_provider': providers
            }


def define_variable_lags(name, input_type, lags):
    if HORIZON == 0:
        if lags is None or lags == 0:
            lags = [0]
        elif isinstance(lags, int):
            lags = [-i for i in range(lags)]
        elif isinstance(lags, list):
            pass
        else:
            raise ValueError(f'lags should be None or int or list')
    else:
        if lags is None or lags == 0:
            lags = [i for i in range(HORIZON)]
        elif isinstance(lags, int):
            lags = [-i for i in range(lags)] + [i for i in range(1, HORIZON)]
        elif isinstance(lags, list):
            lags += [i for i in range(1, HORIZON)]
        else:
            raise ValueError(f'lags should be None or int or list')
    if name in {'flux', 'wind'}:
        if USE_DATA_BEFORE_AND_AFTER_TARGET:
            max_lag = np.max(lags)
            min_lag = np.min(lags)
            lags += [min_lag - 1, max_lag + 1]
    return lags


def config_data():
    static_input_data = {'nwp_data_merge': NWP_DATA_MERGE if TYPE != 'load' else [None],
                         'compress_data': DATA_COMPRESS if TYPE != 'load' else ['load'],
                         'use_data_before_and_after_target': USE_DATA_BEFORE_AND_AFTER_TARGET,
                         'remove_night_hours': REMOVE_NIGHT_HOURS,
                         'use_diff_between_lags': USE_DIFF_BETWEEN_LAGS,
                         'variables': variables(),
                         'scale_row_method': DATA_ROW_SCALE,
                         'scale_nwp_method': DATA_NWP_SCALE,
                         'scale_target_method': DATA_TARGET_SCALE,
                         'data_structure': DATA_STRUCTURE}
    return static_input_data
