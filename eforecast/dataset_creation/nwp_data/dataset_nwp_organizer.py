import numpy as np
import pandas as pd


class DatasetNWPsOrganizer:
    def __init__(self, static_data, nwp_data):
        self.static_data = static_data
        self.nwp_data = nwp_data
        self.path_data = self.static_data['path_data']
        self.horizon_type = static_data['horizon_type']
        self.nwp_models = static_data['NWP']
        self.area_group = static_data['area_group']
        self.areas = self.static_data['NWP'][0]['area']
        self.variables = dict([(var_data['name'], var_data) for var_data in static_data['variables']
                               if var_data['type'] == 'nwp'])
        print(f"Dataset NWP preprocessing started for project {self.static_data['_id']}")

    def merge(self, merge):
        if merge == 'all':
            nwp_data_merged, nwp_dates = self.merge_all()
        elif merge == 'by_area':
            nwp_data_merged, nwp_dates = self.merge_by_area()
        elif merge == 'by_variable':
            nwp_data_merged, nwp_dates = self.merge_by_variable()
        elif merge == 'by_area_variable':
            nwp_data_merged, nwp_dates = self.merge_by_area_variable()
        elif merge == 'by_nwp_provider':
            raise NotImplementedError(f'Merge method {merge} not implemented yet')
            nwp_data_merged = self.merge_by_nwp_provider()
        else:
            raise NotImplementedError(f'Merge method {merge} not implemented yet')

        return nwp_data_merged, nwp_dates

    def merge_all(self):
        nwp_data = []
        nwp_dates = pd.DatetimeIndex([])
        nwp_metadata = dict()
        nwp_metadata['groups'] = []
        nwp_metadata['axis'] = []
        for area, area_data in self.nwp_data.items():
            for variable, var_data in area_data.items():
                for vendor, nwp_provide_data in var_data.items():
                    if nwp_dates.shape[0] == 0:
                        nwp_dates = nwp_dates.append(nwp_provide_data['dates'])
                        data = nwp_provide_data['data']
                    else:
                        if not nwp_dates.equals(nwp_provide_data['dates']):
                            dates = nwp_dates.intersection(nwp_provide_data['dates'])
                            ind_new = nwp_provide_data['dates'].get_indexer(dates)
                            ind_old = nwp_dates.get_indexer(dates)
                            nwp_dates = dates.copy(deep=True)
                            for i in range(len(nwp_data)):
                                nwp_data[i] = nwp_data[i][ind_old]
                            data = nwp_provide_data['data'][ind_new]
                        else:
                            data = nwp_provide_data['data']
                    nwp_data.append(data)
                    axis_names = []
                    for l in self.variables[variable]['lags']:
                        axis_names.append([area, variable + '_' + str(l), vendor])
                    nwp_metadata['axis'].extend(axis_names)
        try:
            nwp_data = np.concatenate(nwp_data, axis=2)
            nwp_data = np.moveaxis(nwp_data, 2, -1)
            nwp_metadata['dates'] = nwp_dates
        except:
            raise NotImplemented('Cannot merge data with ALL method, try by_area or by variable or both')

        return nwp_data, nwp_metadata

    def merge_by_area(self):
        nwp_data = dict()
        nwp_metadata = dict()
        nwp_metadata['groups'] = [area for area in self.nwp_data.keys()]
        nwp_dates = pd.DatetimeIndex([])
        nwp_metadata['axis'] = dict()
        for area, area_data in self.nwp_data.items():
            nwp_metadata['axis'][area] = []
            nwp_data[area] = []
            for variable, var_data in area_data.items():
                for vendor, nwp_provide_data in var_data.items():
                    if nwp_dates.shape[0] == 0:
                        nwp_dates = nwp_dates.append(nwp_provide_data['dates'])
                        data = nwp_provide_data['data']
                    else:
                        if not nwp_dates.equals(nwp_provide_data['dates']):
                            dates = nwp_dates.intersection(nwp_provide_data['dates'])
                            ind_new = nwp_provide_data['dates'].get_indexer(dates)
                            ind_old = nwp_dates.get_indexer(dates)
                            nwp_dates = dates.copy(deep=True)
                            for i in range(len(nwp_data[area])):
                                nwp_data[area][i] = nwp_data[area][i][ind_old]
                            data = nwp_provide_data['data'][ind_new]
                        else:
                            data = nwp_provide_data['data']
                    nwp_data[area].append(data)
                    axis_names = []
                    for l in self.variables[variable]['lags']:
                        axis_names.append([area, variable + '_' + str(l), vendor])
                    nwp_metadata['axis'][area].extend(axis_names)

            nwp_data[area] = np.concatenate(nwp_data[area], axis=2)
            nwp_data[area] = np.moveaxis(nwp_data[area], 2, -1)
        nwp_metadata['dates'] = nwp_dates
        return nwp_data, nwp_metadata

    def merge_by_variable(self):
        nwp_data = dict()
        nwp_metadata = dict()
        nwp_metadata['groups'] = set()
        nwp_dates = pd.DatetimeIndex([])
        axis = dict()
        for area, area_data in self.nwp_data.items():
            for variable, var_data in area_data.items():
                nwp_metadata['groups'].add(variable)
                nwp_data[variable] = []
                axis[variable] = []
        nwp_metadata['groups'] = list(nwp_metadata['groups'])

        for area, area_data in self.nwp_data.items():
            for variable, var_data in area_data.items():
                for vendor, nwp_provide_data in var_data.items():
                    if nwp_dates.shape[0] == 0:
                        nwp_dates = nwp_dates.append(nwp_provide_data['dates'])
                        data = nwp_provide_data['data']
                    else:
                        if not nwp_dates.equals(nwp_provide_data['dates']):
                            dates = nwp_dates.intersection(nwp_provide_data['dates'])
                            ind_new = nwp_provide_data['dates'].get_indexer(dates)
                            ind_old = nwp_dates.get_indexer(dates)
                            nwp_dates = dates.copy(deep=True)
                            for i in range(len(nwp_data[variable])):
                                nwp_data[variable][i] = nwp_data[variable][i][ind_old]
                            data = nwp_provide_data['data'][ind_new]
                        else:
                            data = nwp_provide_data['data']
                    nwp_data[variable].append(data)
                    axis_names = []
                    for l in self.variables[variable]['lags']:
                        axis_names.append([area, variable + '_' + str(l), vendor])
                    nwp_metadata['axis'][variable].extend(axis_names)
                nwp_data[variable] = np.concatenate(nwp_data[variable], axis=2)
                nwp_data[variable] = np.moveaxis(nwp_data[variable], 2, -1)
        nwp_metadata['dates'] = nwp_dates
        nwp_metadata['axis'] = axis
        return nwp_data, nwp_metadata

    def merge_by_area_variable(self):
        nwp_data = dict()
        nwp_metadata = dict()
        nwp_metadata['groups'] = []
        nwp_metadata['axis'] = dict()
        for area, area_data in self.nwp_data.items():
            for variable, var_data in area_data.items():
                nwp_metadata['groups'].append((area, variable))
        nwp_dates = pd.DatetimeIndex([])
        for area, area_data in self.nwp_data.items():
            for variable, var_data in area_data.items():
                nwp_data[area + '_' + variable] = []
                nwp_metadata['axis'][area + '_' + variable] = []
                for vendor, nwp_provide_data in var_data.items():
                    if nwp_dates.shape[0] == 0:
                        nwp_dates = nwp_dates.append(nwp_provide_data['dates'])
                        data = nwp_provide_data['data']
                    else:
                        if not nwp_dates.equals(nwp_provide_data['dates']):
                            dates = nwp_dates.intersection(nwp_provide_data['dates'])
                            ind_new = nwp_provide_data['dates'].get_indexer(dates)
                            ind_old = nwp_dates.get_indexer(dates)
                            nwp_dates = dates.copy(deep=True)
                            for i in range(len(nwp_data[area + '_' + variable])):
                                nwp_data[area + '_' + variable][i] = nwp_data[area + '_' + variable][i][ind_old]
                            data = nwp_provide_data['data'][ind_new]
                        else:
                            data = nwp_provide_data['data']

                    nwp_data[area + '_' + variable].append(data)
                    axis_names = []
                    for l in self.variables[variable]['lags']:
                        axis_names.append([area, variable + '_' + str(l), vendor])
                    nwp_metadata['axis'][area + '_' + variable].extend(axis_names)
                nwp_data[area + '_' + variable] = np.concatenate(nwp_data[area + '_' + variable], axis=2)
                nwp_data[area + '_' + variable] = np.moveaxis(nwp_data[area + '_' + variable], 2, -1)
        nwp_metadata['dates'] = nwp_dates
        return nwp_data, nwp_metadata
