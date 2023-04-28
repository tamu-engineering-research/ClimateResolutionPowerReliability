import os
import sys

from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.getcwd()))
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import data_const
from utils.time import time2season
from grid.load import LoadZones, LoadZone, LoadZoneModel
from grid.thermal import ThermalPlants
from grid.hydro import HydroPlants
from grid.solar import PVFarms
from grid.wind import WindFarms
from grid.outage import PlannedUnitOutage
from grid.expansion import LoadExpansion, CapacityExpansion

def cal_yearly_adequacy(data_source : str, data_file : str, start_time : datetime, end_time : datetime, year : int):
    # load
    loadzones = LoadZones(state = 'TX',year = 2018,data_root = data_const.ROOT_DIR,)
    loadzones.add_env(
        data_root = data_const.ROOT_DIR,
        data_source = data_source,
        data_file = data_file,
        start_time= start_time,
        end_time = end_time,
    )
    # solar
    PVfarms = PVFarms(state = 'TX',year = 2019,data_root = data_const.ROOT_DIR,)
    PVfarms.add_solar_radiation(
        data_root = data_const.ROOT_DIR,
        data_source = data_source,
        data_file = data_file,
        start_time= start_time,
        end_time = end_time,
    )
    # wind
    windfarms = WindFarms(state='TX', year=2019, data_root=data_const.ROOT_DIR,)
    windfarms.add_wind_speed(
        data_root = data_const.ROOT_DIR,
        data_source = data_source,
        data_file = data_file,
        start_time= start_time,
        end_time = end_time,
    )
    # thermal
    thermalplants = ThermalPlants(state='TX', year=2019,data_root=data_const.ROOT_DIR,)
    thermalplants.add_para(
        data_root = data_const.ROOT_DIR,
        data_source = data_source,
        data_file = data_file,
        start_time= start_time,
        end_time = end_time,
    )
    #hydro
    hydroplants = HydroPlants(state = 'TX',year = 2019,data_root = data_const.ROOT_DIR,)
    hydroplants.add_para(
        data_root = data_const.ROOT_DIR,
        data_source = data_source,
        data_file = data_file,
        start_time= start_time,
        end_time = end_time,
    )
    # load expansion
    loadexpansion = LoadExpansion(state='TX', data_root=data_const.ROOT_DIR,)
    load_expansion_factor = loadexpansion.get_load_expansion_naive()
    load_expansion_factor = load_expansion_factor[year]#{'peak', 'valley'}
    # capacity expansion
    capacityexpansion = CapacityExpansion(state='TX', data_root=data_const.ROOT_DIR,)
    capacity_expansion_factor = capacityexpansion.get_capacity_expansion()
    capacity_expansion_factor = capacity_expansion_factor[year]#{'wind', 'solar', 'thermal', 'hydro'}
    # outage
    plannedunitoutage = PlannedUnitOutage('TX', data_const.ROOT_DIR)
    planned_outage_factor = plannedunitoutage.get_planned_outage_factor()

    # calculate the grid adequacy
    solar_power = PVfarms.get_power(
        expansion_factor = capacity_expansion_factor['solar'],
        planned_outage_factor=planned_outage_factor,
    )
    wind_power = windfarms.get_power(
        expansion_factor = capacity_expansion_factor['wind'],
        planned_outage_factor=planned_outage_factor,
    )
    thermal_power = thermalplants.get_power(
        expansion_factor = capacity_expansion_factor['thermal'],
        planned_outage_factor= planned_outage_factor,
    )
    hydro_power = hydroplants.get_power(
        expansion_factor =  capacity_expansion_factor['hydro'],
        planned_outage_factor = planned_outage_factor,
    )
    load_power = loadzones.get_power(
        expansion_factor = load_expansion_factor
    )
    adequacy = solar_power + wind_power + thermal_power + hydro_power - load_power
    time = loadzones.time
    results = {
        'time': loadzones.time,
        'solar_power': solar_power,
        'wind_power': wind_power,
        'thermal_power': thermal_power,
        'hydro_power': hydro_power,
        'load_power': load_power,
        'adequacy': adequacy,
    }
    return results

def cal_adequacy(climate_data, solar, wind, thermal, hydro, load):
    """
    Calculate the adequacy of the grid
    :param climate_data: a three dimensional array of shape (L, V, T), [location, variable, time]
    :param solar: function object 
        :param climate data (L, V, T) 
        :return solar power (T,)
    :param wind: function object
        :param climate data (L, V, T) 
        :return wind power (T,)
    :param thermal: function object
        :param climate data (L, V, T) 
        :return thermal power (T,)
    :param hydro: function object
        :param climate data (L, V, T) 
        :return hydro power (T,)
    :param load: function object
        :param climate data (L, V, T) 
        :return load power (T,)
    :return: array of grid adequacy (T,)
    """
    return solar(climate_data) + wind(climate_data) + thermal(climate_data) + hydro(climate_data) - load(climate_data)


def save_adequacy(results, save_dir, file_name):
    with open(os.path.join(save_dir, file_name), 'wb') as f:
        pickle.dump(results, f)

def load_adequacy(save_dir, file_name):
    with open(os.path.join(save_dir, file_name), 'rb') as f:
        results = pickle.load(f)
    return results

def adequacy_sensitivity(climate_data, solar, wind, thermal, hydro, load):
    """
    Calculate the sensitivity of the grid adequacy to climate change
    :return: array of grid adequacy's sensitivity w.r.t. climate data (L, V, T,)
    """
    Adequacy = cal_adequacy(climate_data, solar, wind, thermal, hydro, load)
    Ad_Arr = Adequacy.values
    timeDifference = Adequacy.diff()
    TimeDiff_arr = (timeDifference).values
    step = TimeDiff_arr[2]

    P_Ad_P_t = Partial_derivative(Ad_Arr , step)
    #w.r.t Climate data
    column_names = list(climate_data.dtype.names)
    P_Climate_P_t = pd.DataFrame().reindex_like(climate_data).fillna(0)
    P_Ad_P_V = pd.DataFrame().reindex_like(climate_data).fillna(0)

    for V in column_names:
        L = climate_data.shape[2]
        for loc in range(L):
            arr = (Adequacy[: , V , loc]).values
            PD = Partial_derivative(arr , step)
            P_Climate_P_t.loc[:, V, loc] = PD
            P_Ad_P_V.loc[: , V , loc] = PD*P_Ad_P_t
            
    return P_Ad_P_t, P_Climate_P_t, P_Ad_P_V

def Partial_derivative(Input_data , step):
    f = np.array(Input_data)
    # using Stencil Method
    t1 = np.roll(f, 2)
    f2 = np.concatenate((np.zeros(2, dtype=int), t1[2:len(f)]))
    t2 = np.roll(f, 1)
    f1 = np.concatenate((np.zeros(1, dtype=int), t2[1:len(f)]))
    t3 = np.roll(f, -1)
    f_1 = np.concatenate(t3[:len(f)-1] , np.zeros(1, dtype=int))
    t4 = np.roll(f, -2)
    f_2 = np.concatenate(t4[:len(f)-2] , np.zeros(2, dtype=int))

    return (-f2 + 8*f1 - 8*f_1 + f_2)/(12*step)

if __name__=="__main__":
    #%%
    # compare_adequacy_curve()
    compare_adequacy_barplot()
    input()
    #%%
    data_source = 'iHESP'
    data_file = 'CESM_HR_RCP85_ENS01_Climate-Power_CONUS.nc'
    for year in tqdm(range(2033, 2044)):
        start_time = datetime(int(year), 1, 1, 0, 0, 0)
        end_time = datetime(int(year), 12, 31, 18, 0, 0)
        results = cal_yearly_adequacy(data_source, data_file, start_time, end_time, year)
        save_adequacy(
            results, 
            save_dir = os.path.join(data_const.ROOT_DIR, 'Results'), 
            file_name = 'HR_adequacy_{year}.pkl'.format(year=year)
        )
    input()

