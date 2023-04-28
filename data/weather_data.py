import os
import sys

from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.getcwd()))
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import data_const
from utils.time import utc2central, central2utc

class WeatherData():
    def __init__(self, dir, start_time = None, end_time = None) -> None:
        self.dir = dir
        self.raw_weather_data = load_raw_ASOS_data(dir = dir)

    def get_wind_speed(self, start_time = None, end_time = None):#TODO
        """
        Get wind speed data from ASOS during the specified time period
        :param start_time: start time of the time period
        :param end_time: end time of the time period
        :return wspd: {'Time': [time], 'WSPD_100m': {(lat, lon, lat, lon): [wind speed]}
        """
        # convert wind speed from knot to m/s and from 10m to 100m
        wspd = {}
        return wspd
    
    def get_solar_radiation(self, start_time = None, end_time = None):
        """
        Get solar irradiance data from ASOS during the specified time period
        :param start_time: start time of the time period
        :param end_time: end time of the time period
        :return solar: {'Time': [time], 'GHI': {(lat, lon, lat, lon): [solar irradiance]}
        """
        solar = {}
        return solar
    
    def get_environmental_data(self, start_time = None, end_time = None):
        """
        :return env_data : { 'Time':[datetime], ''
        }
        """
        if start_time is None:
            start_time = datetime(2018, 1, 1, 0)
        if end_time is None:
            end_time = datetime(2018, 12, 31, 23)
        env = {}
        env['Time'] = pd.date_range(start_time, end_time, freq = 'H')
        env['Temperature'], env['DewPoint'], env['RelativeHumidity'], env['WindSpeed'] = {}, {}, {}, {}
        dropped_stations = []
        for key in self.raw_weather_data:
            data_tmp = self.raw_weather_data[key].reindex(env['Time']).fillna(np.nan)
            if data_tmp.isna().any(axis=1).sum()/data_tmp.shape[0] > 0.05: # if more than 5% of data is missing
                dropped_stations.append(key)
                continue
            env['Temperature'][key] = data_tmp['tmpc'].values
            env['DewPoint'][key] = data_tmp['dwpc'].values
            env['RelativeHumidity'][key] = data_tmp['relh'].values
            env['WindSpeed'][key] = data_tmp['wsmps'].values
            # a=0
        print('{n_drop} out of {n} stations are dropped.'.format(n_drop=len(dropped_stations), n=len(self.raw_weather_data)))
        return env

def load_raw_ASOS_data(dir = None, file = 'ALL'):
    raw_weather_data = {}
    if file == 'ALL': # read all files in the directory
        blocked_files = []
        for file in os.listdir(dir):
            if file.endswith(".csv"):
                lat, lon, df_weather = load_single_raw_ASOS_data(os.path.join(dir, file))
                raw_weather_data[(lat, lon, lat, lon)] = df_weather
            else:
                blocked_files.append(file)
        return raw_weather_data
    elif file.endswith(".csv"): # read the file
        lat, lon, df_weather = load_single_raw_ASOS_data(os.path.join(dir, file))
        raw_weather_data[(lat, lon, lat, lon)] = df_weather
        return raw_weather_data
    raise NotImplementedError

def load_single_raw_ASOS_data(path):
    df_tmp = pd.read_csv(path, index_col='datetime', parse_dates=True)
    lat = df_tmp['lat'].values[0]
    lon = df_tmp['lon'].values[0]
    df_weather = df_tmp[['tmpc','dwpc','relh','wsmps']]
    return lat, lon, df_weather


if __name__ =="__main__":
    weatherdata = WeatherData(
        dir = os.path.join(data_const.ROOT_DIR, 'ASOS')
    )
    env = weatherdata.get_environmental_data()
    input()