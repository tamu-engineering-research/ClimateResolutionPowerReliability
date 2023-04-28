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

from utils import thermal_const
from utils import data_const
from utils import county2coordinate
from utils import get_closest_coordinate
from utils import time2season
from data import ClimateData, WeatherData

class ThermalPlants():
    def __init__(self, state : str, year : int, data_root : str):
        self.state = state
        self.year = year
        self.get_thermal_info(state, year, data_root)
        self.init_thermal_plants()

    def get_thermal_info(self, state, year, data_root):
        """Get thermal plant data for each state"""
        file_dir =  os.path.join(data_root, 'EIA')
        file_name = "state_thermal_plant_{state}_{year}.pkl".format(state=state, year=year)
        file_path = os.path.join(file_dir, file_name)
        # Get state thermal plant data
        if os.path.isfile(file_path):
            print("Reading state thermal plants from {path}".format(path=file_path))
            state_thermal_info = _read_state_thermal_info(file_path)
        else:
            print("Creating state thermal plants file at {path}".format(path=file_path))
            state_thermal_info = get_state_thermal_info(state, year, data_root)
            _save_state_thermal_info(file_path, state_thermal_info)
        self.state_thermal_info = state_thermal_info
    
    def init_thermal_plants(self):
        print("Creating {num} thermal plants...".format(num=len(self.state_thermal_info)))
        self.dict_thermal_plants = {}
        for thermal_info in tqdm(self.state_thermal_info):
            self.add_thermal_plant(thermal_info)
    
    def add_thermal_plant(self, thermal_info):
        plant_id = thermal_info['ID']
        if plant_id in self.dict_thermal_plants:
            print("Thermal plant {id} already exists".format(id=plant_id))
        self.dict_thermal_plants[plant_id] = ThermalPlant(thermal_info)

    def add_para(self,
            data_root : str = data_const.ROOT_DIR,
            data_source :str = 'iHESP',
            data_file : str = 'CESM_LR_RCP85_ENS01_Climate-Power_CONUS.nc',
            start_time : datetime = None,
            end_time : datetime = None,
        ):
        self.para = get_para_time(data_root, data_source, data_file, start_time, end_time)

    def get_power(self, expansion_factor=None, planned_outage_factor=None):
        if expansion_factor is None:
            expansion_factor = 1
        if planned_outage_factor is not None:
            time_target = np.array([dt.strftime("%d-%m %H") for dt in self.para['Time']])
            time_source = np.array([dt.strftime("%d-%m %H") for dt in planned_outage_factor['time']])
            match_indices = np.where(np.isin(time_target, time_source))[0]
            planned_outage_factor = planned_outage_factor['thermal'][match_indices]
            assert planned_outage_factor.shape[0] == self.para['Time'].shape[0]
        para = self.para
        self.time = self.para['Time']
        self.power = np.zeros(self.time.shape)
        for plant_id, thermal_plant in self.dict_thermal_plants.items():
            power_tmp = thermal_plant.get_power(para = para)
            if planned_outage_factor is None:
                self.power += power_tmp * expansion_factor
            else:
                self.power += power_tmp * expansion_factor * (1 - planned_outage_factor)
        return self.power
    
class ThermalPlant():
    def __init__(self, thermal_info):
        self.state = thermal_info['State']
        self.county = thermal_info['County']
        self.nameplatecapacity = thermal_info['NamePlateCapacity']
        self.nameplatepowerfactor = thermal_info['NamePlatePowerFactor']
        self.summercapacity = thermal_info['SummerCapacity']
        self.wintercapacity = thermal_info['WinterCapacity']
        self.minimumload = thermal_info['MinimumLoad']
        self.summercapacityfactored = self.summercapacity*self.nameplatepowerfactor
        self.wintercapacityfactored = self.wintercapacity*self.nameplatepowerfactor
        self.springcapacityfactored = self.summercapacityfactored
        self.autumncapacityfactored = self.wintercapacityfactored
        self.coordinate = county2coordinate(self.state, self.county)
    
    def get_power(self, para : dict, scheduled_on = None,):
        if scheduled_on is None:
            scheduled_on = np.ones(para['Time'].shape[0])
        season_array = time2season(para['Time'])   
        capacity = np.zeros(para['Time'].shape)
        capacity[np.where(season_array==1)] = self.springcapacityfactored
        capacity[np.where(season_array==2)] = self.summercapacityfactored
        capacity[np.where(season_array==3)] = self.autumncapacityfactored
        capacity[np.where(season_array==4)] = self.wintercapacityfactored
        self.power = np.multiply(scheduled_on, capacity)
        return self.power

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
    form_860_filename = "3_1_Generator_Y{year}.xlsx".format(year=year)
    form_860_path = os.path.join(data_dir, form_860_folder, form_860_filename)
    if not os.path.isfile(form_860_path):
        raise ValueError("Form 860 data of year {year} not found at {path}".format(year=year, path=form_860_path))
    form_860 = pd.read_excel(form_860_path, sheet_name = 'Operable', skiprows=1)
    return form_860

def get_state_thermal_info(
        state : str = 'TX',
        year : int = 2019,
        data_root : str = 'H:/Climate_grid_reliability/data',
    ):
    """In this study, thermal includes coal, natural gas, petroleum, and nuclear."""
    form_860 = get_form_860(data_root, year)
    state_condition = form_860['State'] == state
    status_condition = form_860['Status'] == 'OP'
    technology_condition = ~(form_860['Technology'].isin(thermal_const.non_thermal_technology))
    state_thermal_plants = form_860[state_condition & status_condition & technology_condition]
    state_thermal_info = []
    for i, f in enumerate(state_thermal_plants.index):
        # Look up attributes from Form 860
        nameplate_capacity = state_thermal_plants[thermal_const.nameplate_capacity_col].iloc[i]
        nameplate_power_factor = state_thermal_plants[thermal_const.nameplate_power_factor_col].iloc[i]
        summer_capacity = state_thermal_plants[thermal_const.summer_capacity_col].iloc[i]
        winter_capacity = state_thermal_plants[thermal_const.winter_capacity_col].iloc[i]
        minimum_load = state_thermal_plants[thermal_const.minimum_load_col].iloc[i]
        county = state_thermal_plants[thermal_const.county_col].iloc[i]
        plant_id = state_thermal_plants[thermal_const.plant_id_col].iloc[i]
        generator_id = state_thermal_plants[thermal_const.generator_id_col].iloc[i]
        ID = "{plant_id}_{generator_id}".format(plant_id = plant_id, generator_id = generator_id)
        # Integrate thermal plant data
        thermal_info = {
            "State": state,
            "County": county,
            "NamePlateCapacity": nameplate_capacity,
            "NamePlatePowerFactor": nameplate_power_factor,
            "SummerCapacity": summer_capacity,
            "WinterCapacity": winter_capacity,
            "MinimumLoad": minimum_load,
            "ID": ID,
        }
        state_thermal_info.append(thermal_info)
    return state_thermal_info

def get_para_time(
        data_root : str = 'H:/Climate_grid_reliability/data',
        data_source :str = 'iHESP',
        data_file : str = 'CESM_LR_RCP85_ENS01_Climate-Power_CONUS.nc',
        start_time : datetime = None,
        end_time : datetime = None,
    ):
    if data_source == 'iHESP':
        path = os.path.join(data_root, 'iHESP', data_file)
        data = ClimateData(path, start_time, end_time)
        return data.get_para_time(start_time, end_time)
    else:
        raise ValueError("data_source {data_source} not supported".format(data_source = data_source))

#%% utils
def _read_state_thermal_info(file_path):
    with open(file_path, 'rb') as f:
        state_thermal_info = pickle.load(f)
    return state_thermal_info

def _save_state_thermal_info(file_path, state_thermal_info):
    with open(file_path, 'wb') as f:
        pickle.dump(state_thermal_info, f)

#%% test
if __name__=="__main__":
    thermalplants = ThermalPlants(
        state = 'TX',
        year = 2019,
        data_root = data_const.ROOT_DIR,
    )
    thermalplants.add_para(
        data_root = data_const.ROOT_DIR,
        data_source = 'iHESP',
        data_file = 'CESM_LR_RCP85_ENS01_Climate-Power_CONUS.nc',
    )
    thermalplants.get_power()
    input("Press Enter to continue...")