import os
import sys

from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.getcwd()))
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import solar_const
from utils import data_const
from utils import county2coordinate
from utils import get_closest_coordinate
from utils import time2season
from data import ClimateData, WeatherData

class PVFarms():
    def __init__(self, state : str, year : int, data_root : str):
        self.state = state
        self.year = year
        self.get_PV_info(state, year, data_root)
        self.init_PV_farms()

    def get_PV_info(self, state, year, data_root):
        """Get solar plant data for each state"""
        file_dir =  os.path.join(data_root, 'EIA')
        file_name = "state_solar_plant_{state}_{year}.pkl".format(state=state, year=year)
        file_path = os.path.join(file_dir, file_name)
        # Get state PV plant data
        if os.path.isfile(file_path):
            print("Reading state solar plants from {path}".format(path=file_path))
            state_PV_info = _read_state_PV_info(file_path)
        else:
            print("Creating state solar plants file at {path}".format(path=file_path))
            state_PV_info = get_state_PV_info(state, year, data_root)
            _save_state_PV_info(file_path, state_PV_info)
        self.state_PV_info = state_PV_info
    
    def init_PV_farms(self):
        print("Creating {num} PV farms...".format(num=len(self.state_PV_info)))
        self.dict_PV_farms = {}
        for PV_info in tqdm(self.state_PV_info):
            self.add_PV_farm(PV_info)
    
    def add_PV_farm(self, PV_info):
        plant_id = PV_info['ID']
        if plant_id in self.dict_PV_farms:
            print("PV farm {id} already exists".format(id=plant_id))
        self.dict_PV_farms[plant_id] = PVFarm(PV_info)
    
    def add_solar_radiation(self,
            data_root : str = data_const.ROOT_DIR,
            data_source :str = 'iHESP',
            data_file : str = 'CESM_LR_RCP85_ENS01_Climate-Power_CONUS.nc',
            start_time : datetime = None,
            end_time : datetime = None,
        ):
        self.srd = get_state_solar_radiation( data_root, data_source, data_file, start_time, end_time)

    def get_power(self, method = 'naive', expansion_factor=None, planned_outage_factor=None):
        if expansion_factor is None:
            expansion_factor = 1
        if planned_outage_factor is not None:
            time_target = np.array([dt.strftime("%d-%m %H") for dt in self.srd['Time']])
            time_source = np.array([dt.strftime("%d-%m %H") for dt in planned_outage_factor['time']])
            match_indices = np.where(np.isin(time_target, time_source))[0]
            planned_outage_factor = planned_outage_factor['solar'][match_indices]
            assert planned_outage_factor.shape[0] == self.srd['Time'].shape[0]
        srd = self.srd
        self.time = srd['Time']
        self.power = np.zeros(self.time.shape)
        for plant_id, PV_farm in self.dict_PV_farms.items():
            power_tmp = PV_farm.get_power(srd = srd, method = method)
            if planned_outage_factor is None:
                self.power += power_tmp * expansion_factor
            else:
                self.power += power_tmp * expansion_factor * (1 - planned_outage_factor)
        return self.power
    
class PVFarm():
    def __init__(self, PV_info):
        self.state = PV_info['State']
        self.county = PV_info['County']
        self.capacity = PV_info['Capacity']
        self.azimuth_angle = PV_info['Azimuth_Angle']
        self.tilt_angle = PV_info['Tilt_Angle']
        self.coordinate = county2coordinate(self.state, self.county)
    
    def get_power(self, srd : dict, scheduled_on = None, method = 'naive'):
        if scheduled_on is None:
            scheduled_on = np.ones(srd['Time'].shape[0])
        capacity = self.capacity * scheduled_on
        srd_coordinate = list(srd['NetFlux'].keys())
        srd_coordinate = get_closest_coordinate(self.coordinate, srd_coordinate)
        srd_tmp = {
            'NetFlux':srd['NetFlux'][srd_coordinate],
            'DirectFlux':srd['DirectFlux'][srd_coordinate],
        }
        if method == 'naive':
            solar_power_norm = self.get_power_norm_naive(srd_tmp)
        else:
            raise ValueError("method {method} not supported".format(method=method))
        self.power = np.multiply(solar_power_norm, capacity)
        return self.power
    
    def get_power_norm_naive(self, srd):
        net_flux = srd['NetFlux']
        direct_flux = srd['DirectFlux']
        diffuse_flux = net_flux - direct_flux
        power_norm = direct_flux/direct_flux.max()
        return power_norm
    
    def get_power_norm_complex(self, srd):
        raise NotImplementedError('get_power_norm_complex not implemented yet')

def get_form_860(
        data_root : str = 'H:/Climate_grid_reliability/data',
        year : int = 2019,
    ) -> pd.DataFrame:
    """Read data for EIA Form 860.

    :param str data_dir: data directory.
    :param int year: EIA data year to get.
    :return: (*pandas.DataFrame*) -- dataframe with Form 860 data.
    """
    data_dir = os.path.join(data_root, 'EIA')
    if not os.path.isdir(data_dir):
        raise ValueError("data_dir is not a valid directory")
    form_860_folder = "eia860{year}".format(year=year)
    form_860_filename = "3_3_Solar_Y{year}.xlsx".format(year=year)
    form_860_path = os.path.join(data_dir, form_860_folder, form_860_filename)
    if not os.path.isfile(form_860_path):
        raise ValueError("Form 860 data of year {year} not found at {path}".format(year=year, path=form_860_path))
    form_860 = pd.read_excel(form_860_path, sheet_name = 'Operable', skiprows=1)
    return form_860

def get_state_PV_info(
        state : str = 'TX',
        year : int = 2019,
        data_root : str = 'H:/Climate_grid_reliability/data',
    ):
    form_860 = get_form_860(data_root, year)
    state_solar_plants = form_860[form_860['State'] == state]
    state_solar_plants = state_solar_plants[state_solar_plants['Prime Mover'] == 'PV']
    state_PV_info = []
    for i, f in enumerate(state_solar_plants.index):
        # Look up attributes from Form 860
        plant_capacity = state_solar_plants[solar_const.capacity_col].iloc[i]
        county = state_solar_plants[solar_const.county_col].iloc[i]
        plant_id = state_solar_plants[solar_const.plant_id_col].iloc[i]
        generator_id = state_solar_plants[solar_const.generator_id_col].iloc[i]
        azimuth_angle = state_solar_plants[solar_const.azimuth_angle_col].iloc[i]
        tilt_angle = state_solar_plants[solar_const.tilt_angle_col].iloc[i]
        ID = "{plant_id}_{generator_id}".format(plant_id = plant_id, generator_id = generator_id)
        # Integrate solar plant data
        PV_info = {
            "State": state,
            "County": county,
            "Capacity": plant_capacity,
            "ID": ID,
            "Azimuth_Angle": azimuth_angle,
            "Tilt_Angle": tilt_angle,
        }
        state_PV_info.append(PV_info)
    return state_PV_info

def get_state_solar_radiation(
        data_root:str = 'H:/Climate_grid_reliability/data',
        data_source:str = 'iHESP',
        data_file:str = None,
        start_time = None,
        end_time = None,
    ):
    """Get solar irradiance data during the specified time period"""
    if data_source == 'iHESP':
        path = os.path.join(data_root, 'iHESP', data_file)
        data = ClimateData(path, start_time, end_time)
        return data.get_solar_radiation(start_time, end_time)
    elif data_source == 'NSRDB':#TODO
        path = os.path.join(data_root, 'NSRDB', data_file)
        data = WeatherData(path)
        return data.get_solar_radiation(start_time, end_time)
    else:
        raise ValueError("data_source must be 'iHESP' or 'NSRDB'")

#%% utils
def _read_state_PV_info(file_path):
    with open(file_path, 'rb') as f:
        state_PV_info = pickle.load(f)
    return state_PV_info

def _save_state_PV_info(file_path, state_PV_info):
    with open(file_path, 'wb') as f:
        pickle.dump(state_PV_info, f)

# def _get_closest_coordinate(coordinate, list_coordinate):
#     """Return the closest coordinate in list_coordinate to coordinate"""
#     lat_target, lon_target = coordinate
#     for lat_ll, lon_ll, lat_ur, lon_ur in list_coordinate:
#         if lat_ll <= lat_target <= lat_ur and lon_ll <= lon_target <= lon_ur:
#             return (lat_ll, lon_ll, lat_ur, lon_ur)
#     dist_min = np.inf
#     for lat_ll, lon_ll, lat_ur, lon_ur in list_coordinate:
#         dist = np.sqrt((lat_target-lat_ll)**2 + (lon_target-lon_ll)**2)
#         if dist < dist_min:
#             dist_min = dist
#             coordinate_min = (lat_ll, lon_ll, lat_ur, lon_ur)
#     if dist_min == np.inf:
#         raise ValueError("No coordinate found")
#     return coordinate_min

if __name__=="__main__":
    PVfarms = PVFarms(
        state = 'TX',
        year = 2019,
        data_root = data_const.ROOT_DIR,
    )
    PVfarms.add_solar_radiation(
        data_root = data_const.ROOT_DIR,
        data_source = 'iHESP',
        data_file = 'CESM_LR_RCP85_ENS01_Climate-Power_CONUS.nc',
    )
    PVfarms.get_power()
    input("Press Enter to continue...")