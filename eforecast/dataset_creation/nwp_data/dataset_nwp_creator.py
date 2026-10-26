import os

import joblib
from joblib import Parallel
from joblib import delayed
from tqdm import tqdm
import numpy as np
import pandas as pd

from eforecast.common_utils.date_utils import convert_timezone_dates
from eforecast.common_utils.nwp_utils import check_empty_multiple_nwp
from eforecast.common_utils.nwp_utils import get_lats_longs_by_date


class DatasetNWPsCreator:
    def __init__(self, static_data, transformer, dates=None, is_online=False):
        self.static_data = static_data
        self.transformer = transformer
        self.is_online = is_online
        if self.is_online:
            self.dates = dates
        else:
            self.dates = static_data['data'].index.round('D').unique()

        self.path_group_nwp = static_data['path_group_nwp']
        self.nwp_models = static_data['NWP']
        self.area_group = static_data['area_group']
        self.areas = self.static_data['NWP'][0]['area']
        self.coord = static_data['coord']
        self.n_jobs = static_data['n_jobs']
        self.variables = dict([(var_data['name'], var_data) for var_data in static_data['variables']
                               if var_data['type'] == 'nwp'])
        print(f"Dataset NWP creation started for project {self.static_data['_id']}")

    def make_dataset(self):
        if not isinstance(self.static_data['horizon'], int):
            raise ValueError('horizon parameter of static_data for multiple output (batch mode) should be integer')
        if isinstance(self.areas, dict):
            areas = [area for area in self.areas.keys()]
        else:
            areas = [self.static_data['project_name']]

        nwp_daily = Parallel(n_jobs=self.n_jobs)(
            delayed(self.stack_daily_nwps)(t, areas) for t in tqdm(self.dates))

        x_3d = dict()
        for area in areas:
            x_3d[area] = dict()
            for var in self.variables.keys():
                x_3d[area][var] = dict()
                for nwp in self.nwp_models:
                    x_3d[area][var][nwp['model']] = dict()
                    data = []
                    dates = pd.DatetimeIndex([])
                    for res in nwp_daily:
                        if res[area][var] is not None:
                            data.append(res[area][var][nwp['model']]['data'])
                            dates = dates.append(res[area][var][nwp['model']]['dates'])
                    data = np.array(data)
                    shape = data.shape
                    data = data.reshape([np.prod(shape[:2]), *shape[2:]])
                    if var in self.transformer.variables_index.keys():
                        self.transformer.fit(data, var, data_dates=dates)
                        data = self.transformer.transform(data, var, data_dates=dates)
                    x_3d[area][var][nwp['model']]['data'] = data
                    x_3d[area][var][nwp['model']]['dates'] = dates

        return x_3d

    def stack_daily_nwps(self, t, areas):
        nwp_data = self.read_nwp_pickle(t)
        if self.static_data['horizon_type'] in {'intra-ahead', 'multi-output'}:
            p_dates = pd.date_range(t + pd.DateOffset(hours=1), t + pd.DateOffset(hours=24), freq='H')
        else:
            p_dates = pd.date_range(t + pd.DateOffset(hours=24), t + pd.DateOffset(hours=47), freq='H')

        x_3d = self.create_inp_area(t, p_dates, nwp_data, areas)

        return x_3d

    def create_sample(self, nwp, lats, longs):
        x0 = nwp[np.ix_(lats, longs)]
        return x0

    def create_inp_lag(self, date, variable, nwps, area=None):
        dates_nwp = [date.round('H') + pd.DateOffset(hours=lag)
                     for lag in variable['lags']]
        try:
            if self.static_data['local_timezone'] != 'UTC':
                dates_nwp = convert_timezone_dates(pd.DatetimeIndex(dates_nwp),
                                                   timezone1=self.static_data['local_timezone'],
                                                   timezone2='UTC')
        except:
            return None
        inp_lag = []
        for date_nwp in dates_nwp:
            nwp = self.select_nwp(date, date_nwp, nwps)

            if check_empty_multiple_nwp(nwp, self.variables):
                if not isinstance(nwps['lats'], dict):
                    inp = self.create_sample(nwp[variable['name']], nwps['lats'], nwps['longs'])
                else:
                    inp = self.create_sample(nwp[variable['name']], nwps['lats'][area], nwps['longs'][area])
                inp_lag.append(inp)
            else:
                return None
        inp_lag = np.stack(inp_lag)
        return inp_lag

    def create_inp_hor(self, date, variable, nwps, area=None):
        inp_hor = []
        horizon = 1 if self.static_data['horizon'] == 0 else self.static_data['horizon']

        for hor in range(horizon):
            inp = self.create_inp_lag(date + pd.DateOffset(hours=hor), variable, nwps, area=area)
            if inp is not None:
                inp_hor.append(inp)
        inp_hor = np.stack(inp_hor)
        return inp_hor

    def create_inp_dates(self, p_dates, variable, nwps, area=None):
        inp_dates = []
        dates = pd.DatetimeIndex([])
        for date in p_dates:
            inp = self.create_inp_hor(date, variable, nwps, area=area)
            if len(inp.shape) > 0:
                inp_dates.append(inp)
                dates = dates.append(pd.DatetimeIndex([date]))
        inp_dates = np.stack(inp_dates)
        return inp_dates, dates

    def create_inp_nwp_provider(self, t, p_dates, variable, nwp_data, area=None):
        inp_nwp_provider = dict()
        for id_nwp, nwps in enumerate(nwp_data):
            if nwps is None:
                print(f"NWP data not found for date {t}")
                inp_nwp_provider = None
                break
            else:
                try:
                    inp_nwp_provider[nwps['nwp_provider']] = dict()
                    if nwps['nwp_provider'] in variable['nwp_provider']:
                        inp_nwp_provider[nwps['nwp_provider']]['data'], inp_nwp_provider[nwps['nwp_provider']][
                            'dates'] = \
                            self.create_inp_dates(p_dates, variable, nwps, area=area)
                except:
                    inp_nwp_provider = None
                    break
        return inp_nwp_provider

    def create_inp_variables(self, t, p_dates, nwp_data, area=None):
        inp_var = dict()
        for name, variable in self.variables.items():
            inp_var[name] = self.create_inp_nwp_provider(t, p_dates, variable, nwp_data, area=area)
        return inp_var

    def create_inp_area(self, t, p_dates, nwp_data, areas):
        inp_area = dict()
        for area in areas:
            if area != 'area_group':
                inp_area[area] = self.create_inp_variables(t, p_dates, nwp_data, area=area)
        return inp_area

    def read_nwp_pickle(self, t):
        nwp_data = []

        for nwp_provider in self.nwp_models:
            if self.static_data['horizon_type'] in {'intra-ahead', 'multi-output'}:
                file_name1 = os.path.join(self.path_group_nwp,
                                          nwp_provider['model'] + '_' + (t - pd.DateOffset(days=1)).strftime(
                                              '%d%m%y') + '.pickle')
                if os.path.exists(file_name1):
                    try:
                        nwps_prev = joblib.load(file_name1)
                    except:
                        os.remove(file_name1)
                        print(file_name1)
                        raise ImportError('Restart the training process. Some nwp files should reconstruct')
                else:
                    nwps_prev = None
            else:
                nwps_prev = None
            file_name = os.path.join(self.path_group_nwp,
                                     nwp_provider['model'] + '_' + t.strftime('%d%m%y') + '.pickle')
            if os.path.exists(file_name):
                try:
                    nwps = joblib.load(file_name)
                except:
                    os.remove(file_name)
                    print(file_name)
                    raise ImportError('Restart the training process. Some nwp files should reconstruct')
                p_dates = pd.date_range(t + pd.DateOffset(hours=1), t + pd.DateOffset(hours=48), freq='H').strftime(
                    '%d%m%y%H%M')
                lats, longs = get_lats_longs_by_date(nwps, p_dates, self.area_group, nwp_provider['area'],
                                                     nwp_provider['resolution'])
            else:
                nwps = None
                lats = None
                longs = None
                print(f"NWP file not found for date {t}")

            nwp_data.append({'nwp_provider': nwp_provider['model'], 'data': nwps, 'data_prev': nwps_prev,
                             'lats': lats, 'longs': longs})
        return nwp_data

    def select_nwp(self, date, date_nwp, nwps):
        if self.static_data['horizon'] > 0:
            if date.hour <= 10:
                data_day = 'data_prev'
            else:
                data_day = 'data'
            nwp = nwps[data_day][date_nwp.strftime('%d%m%y%H%M')]
        else:
            data_day = 'data'
            if date_nwp.strftime('%d%m%y%H%M') in nwps[data_day].keys():
                nwp = nwps[data_day][date_nwp.strftime('%d%m%y%H%M')]
            else:
                nwp = nwps['data_prev'][date_nwp.strftime('%d%m%y%H%M')]
        return nwp
