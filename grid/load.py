import os
import sys

from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.getcwd()))
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import data_const
from utils import grid_const
from utils.map import county2coordinate
from utils.map import get_closest_coordinate
from utils.time import time2season
from utils.formulate import env2x, load2y, merge_envs_to_env
from data import ClimateData, WeatherData, ERCOTData, GridData


class LoadZones():
    def __init__(self, state : str, year : int, data_root : str):
        self.state = state
        self.year = year
        self.get_load_info(state, year, data_root)
        self.init_load_zones()
    
    def get_load_info(self, state, year, data_root):
        """Get load data for each state"""
        file_dir =  os.path.join(data_root, 'ERCOT', 'Load')
        file_name = "state_load_{state}_{year}.pkl".format(state=state, year=year)
        file_path = os.path.join(file_dir, file_name)
        # Get state load data
        if os.path.isfile(file_path):
            print("Reading state load from {path}".format(path=file_path))
            state_load_zone_info = _read_state_load_info(file_path)
        else:
            print("Creating state load file at {path}".format(path=file_path))
            state_load_zone_info = get_state_load_zone_info(state, year, data_root)
            _save_state_load_info(file_path, state_load_zone_info)
        self.state_load_zone_info = state_load_zone_info
    
    def init_load_zones(self):
        print("Creating {num} load zones...".format(num=len(self.state_load_zone_info)))
        self.dict_load_zones = {}
        for load_zone_info in tqdm(self.state_load_zone_info):
            self.add_load_zone(load_zone_info)

    def add_load_zone(self, load_zone_info):
        self.dict_load_zones[load_zone_info['Zone']] = LoadZone(load_zone_info)
    
    def add_env(self,
            data_root : str = data_const.ROOT_DIR,
            data_source :str = 'iHESP',
            data_file : str = 'CESM_LR_RCP85_ENS01_Climate-Power_CONUS.nc',
            start_time : datetime = None,
            end_time : datetime = None,
        ):
        self.env = get_state_env(data_root, data_source, data_file, start_time, end_time)

    def get_power(self, data_source = 'iHESP', expansion_factor = None):
        env = self.env
        self.time = env['Time']
        self.power = np.zeros(env['Time'].shape)
        for zone_id, load_zone in self.dict_load_zones.items():
            power_tmp = load_zone.get_power(env, expansion_factor, data_source)
            self.power += power_tmp
        return self.power

class LoadZone():
    def __init__(self, load_zone_info):
        self.state = load_zone_info['State']
        self.zone = load_zone_info['Zone']
        # self.counties = load_zone_info['Counties']
        self.coordinates = load_zone_info['Coordinates']
        self.share = load_zone_info['Share']
        self.load_zone_model = load_zone_info['LoadZoneModel']

    def get_power(self, env_data : dict, expansion_factor = None, env_source = 'iHESP'):
        power = self.load_zone_model.predict(env_data, env_source, expansion_factor)
        self.power = power
        return self.power

class LoadZoneModel():
    def __init__(self, state, zone, model_root, coordinates, share):
        self.state = state
        self.zone = zone
        self.coordinates = coordinates
        self.share = share
        self.model_file = 'load_zone_model_{state}_{zone}.pkl'.format(state=state, zone=zone)
        self.model_path = os.path.join(model_root, self.model_file)
    
    def train(self, overwrite = False):
        print("Creating load model at {path}".format(path=self.model_path))
        start_time = None
        end_time = None
        x, time_x = self.get_x(start_time, end_time)
        y, time_y = self.get_y(start_time, end_time)
        np.testing.assert_array_equal(time_x, time_y)
        mask = np.any(np.isnan(x), axis=1) | np.any(np.isnan(y), axis=1)
        x = x[~mask]
        y = y[~mask]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        self._train_model(X_train, y_train)
        self._validate_model(X_test, y_test)
        self._save_model(self.model_path)
    
    def _train_model(self, X_train, y_train):
        self.scaler_x = MinMaxScaler().fit(X_train)
        self.scaler_y = MinMaxScaler().fit(y_train)
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 100,),
            activation = 'relu',
            batch_size = 64,
            early_stopping=True,
            max_iter = 1000,
            random_state=42,
            alpha = 0.0001,
        )
        x = self.scaler_x.transform(X_train)
        y = self.scaler_y.transform(y_train)
        self.model.fit(x, y.ravel())
    
    def _validate_model(self, X_test, y_test):
        x = self.scaler_x.transform(X_test)
        y_pred = self.model.predict(x)
        if len(y_pred.shape) == 1:
            y_pred = y_pred[:, np.newaxis]
        y_pred = self.scaler_y.inverse_transform(y_pred)
        print("Mean absolute error: {mae}".format(mae=mean_absolute_error(y_test, y_pred)))
        print("Mean squared error: {mse}".format(mse=mean_squared_error(y_test, y_pred)))
        print("R2 score: {r2}".format(r2=r2_score(y_test, y_pred)))
    
    def get_x(self, start_time = None, end_time = None):#TODO
        env_data = get_state_env(
            data_root = 'H:/Climate_grid_reliability/data',
            data_source = 'ASOS',
            data_file = None,
            start_time = start_time,
            end_time = end_time,
        )
        env_data = merge_envs_to_env(
            env_data, 
            self.coordinates, 
            self.share
        )
        x, time = env2x(
            time = env_data['Time'],
            temperature = env_data['Temperature'],
            dewpoint = env_data['DewPoint'],
            relative_humidity = env_data['RelativeHumidity'],
            wind_speed = env_data['WindSpeed'],
            convert_knot_to_mps = False,
            convert_F_to_C = False,
            convert_percent_to_fraction = True,
        )
        return x, time

    def get_y(self, start_time = None, end_time = None):
        load_data = get_state_load(
            data_root = 'H:/Climate_grid_reliability/data',
            data_source = 'ERCOT',
            start_time = start_time,
            end_time = end_time,
            zone = self.zone,
        )
        y, time = load2y(
            time = load_data['Time'],
            load = load_data['Load'],
            convert_UTC_to_CT = False,
            convert_MW_to_GW = False,
        )
        return y, time

    def predict(self, env_data, env_source = 'iHESP', expansion_factor = None):#TODO
        env_data = merge_envs_to_env(
            env_data, 
            self.coordinates, 
            self.share
        )
        x, _  = env2x(
            time = env_data['Time'],
            temperature = env_data['Temperature'],
            dewpoint = env_data['DewPoint'],
            relative_humidity = env_data['RelativeHumidity'],
            wind_speed = env_data['WindSpeed'],
            convert_knot_to_mps = False,
            convert_F_to_C = False,
            convert_percent_to_fraction=True,
            env_source = env_source,
        )
        x = self.scaler_x.transform(x)
        y_pred = self.model.predict(x)
        if len(y_pred.shape) == 1:
            y_pred = y_pred[:, np.newaxis]
        if expansion_factor is None:
            power = self.scaler_y.inverse_transform(y_pred)
        else:
            data_min = self.scaler_y.data_min_*expansion_factor['valley']
            data_max = self.scaler_y.data_max_*expansion_factor['peak']
            power = data_min + (data_max - data_min)*y_pred
        return power.flatten()
    
    def _save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

def get_state_load_zone_info(state, year, data_root):
    """Get load zone data for one state"""
    griddata = GridData(state, year, data_root)
    load_model_root = os.path.join(data_root, 'ERCOT')
    state_load_info = []
    for zone in griddata.zones:
        load_zone_info = {}
        load_zone_info['State'] = state
        load_zone_info['Zone'] = zone
        load_zone_info['Coordinates'] = griddata.zone_coordinate[zone]
        load_zone_info['Share'] = griddata.zone_share[zone]
        load_zone_info['LoadZoneModel'] = get_load_zone_model(
            state, zone, load_model_root,
            load_zone_info['Coordinates'], load_zone_info['Share']
        )
        state_load_info.append(load_zone_info)
    return state_load_info

def get_state_env(
        data_root:str = 'H:/Climate_grid_reliability/data',
        data_source:str = 'iHESP',
        data_file:str = None,
        start_time = None,
        end_time = None,
    ):
    if data_source == 'iHESP':
        path = os.path.join(data_root, 'iHESP', data_file)
        data = ClimateData(path, start_time, end_time)
        return data.get_environmental_data(start_time, end_time)
    elif data_source == 'ASOS':
        dir = os.path.join(data_root, 'ASOS')
        data = WeatherData(dir)
        return data.get_environmental_data(start_time, end_time)
    else:
        raise ValueError("Invalid env data source: {source}".format(source=data_source))

def get_state_load(
        data_root:str = 'H:/Climate_grid_reliability/data',
        data_source:str = 'ERCOT',
        start_time = None,
        end_time = None,
        zone = None,
    ):
    def _convert_zone_name(zone):
        if zone in grid_const.list_zones_full:
            return grid_const.list_zones_short[grid_const.list_zones_full.index(zone)]
        elif zone in grid_const.list_zones_short:
            return zone
        raise ValueError("Invalid zone name: {zone}".format(zone=zone))
    if data_source == 'ERCOT':
        dir = os.path.join(data_root, 'ERCOT', 'Load')
        ercotdata = ERCOTData(dir)
        zone = _convert_zone_name(zone)
        load_data = ercotdata.get_load_data(start_time, end_time, zone)
        load_data['Load'] = load_data['Load'][zone]
        return load_data
    else:
        raise ValueError("Invalid load data source: {source}".format(source=data_source))

def get_load_zone_model(state, zone, load_model_root, coordinates, share):
    model_file = 'load_zone_model_{state}_{zone}.pkl'.format(state=state, zone=zone)
    model_path = os.path.join(load_model_root, model_file)
    if os.path.isfile(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        load_zone_model = LoadZoneModel(state, zone, load_model_root, coordinates, share)
        load_zone_model.train()
        return load_zone_model


def _read_state_load_info(file_path):
    with open(file_path, 'rb') as f:
        state_load_info = pickle.load(f)
    return state_load_info

def _save_state_load_info(file_path, state_load_info):
    with open(file_path, 'wb') as f:
        pickle.dump(state_load_info, f)
    return


if __name__=='__main__':
    loadzones = LoadZones(
        state = 'TX',
        year = 2018,
        data_root = 'H:/Climate_grid_reliability/data',
    )
    data_root = 'H:/Climate_grid_reliability/data'
    data_file = 'CESM_LR_RCP85_ENS01_Climate-Power_CONUS.nc'
    path = os.path.join(data_root, 'iHESP', data_file)
    climatedata = ClimateData(path)
    loadzones.add_env(
        data_root = data_const.ROOT_DIR,
        data_source = 'iHESP',
        data_file = 'CESM_LR_RCP85_ENS01_Climate-Power_CONUS.nc',
    )
    power = loadzones.get_power()
    input()