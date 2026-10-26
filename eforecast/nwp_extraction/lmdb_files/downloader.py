#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ee
import requests
import time
import zipfile
import io
import os
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import tqdm

Satellite_name = 'NOAA/GFS0P25'

# ee.Authenticate()
try:
    ee.Initialize()
except:
    pass


def select_maximum_datetime(sceneList, dates):
    indices = pd.DataFrame(columns=['indices'])
    for i, scene in enumerate(sceneList):
        date, hor = scene.split('F')
        nhor = int(hor)
        date = (pd.to_datetime(date, format='%Y%m%d%H') + pd.DateOffset(hours=nhor)).strftime(
            '%d%m%y%H%M')
        if date in dates:
            ind = pd.DataFrame(i, index=[date], columns=['indices'])
            indices = indices.append(ind)
    files_date_data = indices[~indices.index.duplicated(keep='last')]
    files_date_data = files_date_data.sort_index()
    max_index = files_date_data.values.ravel().tolist()

    return [sceneList[i] for i in max_index]


def pick_other_scene(sceneList, s, geometry):
    date1, hor1 = s.split('F')
    date1 = (pd.to_datetime(date1, format='%Y%m%d%H') + pd.DateOffset(hours=int(hor1))).strftime(
        '%d%m%y%H%M')
    slist = [s]
    r = None
    for _ in range(8):
        indices = pd.DataFrame(columns=['indices'])
        for i, scene in enumerate(sceneList):
            if scene not in slist:
                date, hor = scene.split('F')  # 2011111218F200
                nhor = int(hor)
                date = (pd.to_datetime(date, format='%Y%m%d%H') + pd.DateOffset(hours=nhor)).strftime(
                    '%d%m%y%H%M')
                if date == date1:
                    ind = pd.DataFrame(i, index=[date], columns=['indices'])
                    indices = indices.append(ind)
        s_new = sceneList[indices.iloc[-1].values[0]]
        scene_name = os.path.join(Satellite_name, s_new).replace('\\', '/')
        layer = ee.Image(scene_name).clip(geometry)
        url = layer.getDownloadURL()
        r = requests.get(url, stream=True)
        time.sleep(5)
        if r.ok:
            break
    return r


def get_file_metadata(file_name):
    year = str(file_name[:4])
    month = str(file_name[4:6])
    day = str(file_name[6:8])
    cycle = str(file_name[8:10])
    hour = int(file_name[-3:])
    return year, month, day, cycle, hour


def check_hours_consecutivity():
    pass


def check_cycle_accuracy():
    pass


def find_last_cycle(valid_list):
    date_upds = max([pd.to_datetime(s.split('F')[0], format='%Y%m%d%H') for s in valid_list])

    return date_upds, str(date_upds.year), str(date_upds.month).zfill(2), \
           str(date_upds.day).zfill(2), str(date_upds.hour).zfill(2)


def store_files_in_gfs_filesystem(area, date=None, path_nwp=None):
    geometry = ee.Geometry.Rectangle([area[0][1], area[0][0],
                                      area[1][1], area[1][0]])
    if date is None:
        dataset = ee.FeatureCollection(Satellite_name). \
            filter(
            ee.Filter.date((dt.today() - timedelta(days=2)).strftime("%Y-%m-%d"),
                           (dt.today() + timedelta(days=1)).strftime("%Y-%m-%d"))). \
            filterBounds(geometry)
    else:
        dataset = ee.FeatureCollection(Satellite_name). \
            filter(
            ee.Filter.date((date - timedelta(days=2)).strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'))). \
            filterBounds(geometry)
    dates = pd.date_range(date + pd.DateOffset(hours=6), date + pd.DateOffset(hours=85), freq='H').strftime(
        '%d%m%y%H%M')
    sceneList = dataset.aggregate_array('system:index').getInfo()
    valid_list = select_maximum_datetime(sceneList, dates)
    date_upd, year, month, day, cycle = find_last_cycle(valid_list)
    print(f'Download... {date}')
    for s in tqdm.tqdm(valid_list):
        date_dnld, hor = s.split('F')
        nhor = int(hor)
        date_dnld = (pd.to_datetime(date_dnld, format='%Y%m%d%H') + pd.DateOffset(hours=nhor)).strftime(
            '%d%m%y%H%M')
        year1, month1, day1, cycle1, hour1 = get_file_metadata(s)
        if month1 != month or day1 != day or cycle1 != cycle:
            hour = int((pd.to_datetime(date_dnld, format='%d%m%y%H%M') - date_upd) / pd.Timedelta(hours=1))
        else:
            hour = hour1
        scene_name = os.path.join(Satellite_name, s).replace('\\', '/')
        layer = ee.Image(scene_name).clip(geometry)
        url = layer.getDownloadURL()
        folder_dir = os.path.join(path_nwp, Satellite_name, f'{year}{month}{day}{cycle}')
        file_dir = os.path.join(folder_dir, f'{hour}')
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        count = 0
        while count < 3:
            try:
                r = requests.get(url, stream=True)
                if not r.ok:
                    r = pick_other_scene(sceneList, s, geometry)
                time.sleep(5)
                break
            except:
                time.sleep(30)
                print('sleep...30')
                count += 1
                continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path=file_dir)
        z.close()
    return folder_dir
