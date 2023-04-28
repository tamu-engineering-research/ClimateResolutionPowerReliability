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

from utils import wind_const
from utils import data_const
from utils import county2coordinate
from utils import get_closest_coordinate
from utils import time2season
from data import ClimateData, WeatherData

class WindFarms():
    def __init__(
            self,
            state : str = 'TX',
            year : int = 2019,
            data_root : str = data_const.ROOT_DIR,
        ):
        self.state = state
        self.year = year
        self.get_power_curves(state, year, data_root)
        self.init_wind_farms()

    def get_power_curves(self, state, year, data_root):
        # Create state power curves directory
        file_dir = os.path.join(data_root, 'EIA')
        file_name = "state_wind_farms_{state}_{year}.pkl".format(state=state, year=year)
        file_path = os.path.join(file_dir, file_name)
        # Get state wind power curves
        if os.path.isfile(file_path):
            print("Reading state wind power curves from {path}".format(path=file_path))
            state_power_curves = _read_state_wind_power_curves(file_path)
        else:
            print("Creating state wind power curves file at {path}".format(path=file_path))
            state_power_curves = get_state_wind_power_curves(state = state, year = year, data_root = data_root,)
            _save_state_wind_power_curves(file_path, state_power_curves)
        self.state_power_curves = state_power_curves

    def init_wind_farms(self):
        print("Creating {num} wind farms...".format(num=len(self.state_power_curves)))
        self.dict_wind_farms = {}
        for power_curve in tqdm(self.state_power_curves):
            self.add_wind_farm(power_curve)
    
    def add_wind_farm(self, power_curve : dict):
        plant_id = power_curve['ID']
        if plant_id in self.dict_wind_farms:
            print("Wind farm {id} already exists".format(id=plant_id))
        self.dict_wind_farms[plant_id] = WindFarm(power_curve)

    def add_wind_speed(self,
            data_root : str = data_const.ROOT_DIR,
            data_source :str = 'iHESP',
            data_file : str = 'CESM_LR_RCP85_ENS01_Climate-Power_CONUS.nc',
            start_time : datetime = None,
            end_time : datetime = None,
        ):
        self.wspd = get_state_wind_speed(data_root, data_source, data_file, start_time, end_time)
    
    def get_power(self, expansion_factor=None, planned_outage_factor=None):
        if expansion_factor is None:
            expansion_factor = 1
        if planned_outage_factor is not None:
            time_target = np.array([dt.strftime("%d-%m %H") for dt in self.wspd['Time']])
            time_source = np.array([dt.strftime("%d-%m %H") for dt in planned_outage_factor['time']])
            match_indices = np.where(np.isin(time_target, time_source))[0]
            planned_outage_factor = planned_outage_factor['wind'][match_indices]
            assert planned_outage_factor.shape[0] == self.wspd['Time'].shape[0]
        wspd = self.wspd
        self.time = wspd['Time']
        self.power = np.zeros(len(self.time))
        for plant_id, wind_farm in self.dict_wind_farms.items():
            power_tmp = wind_farm.get_power(wspd)
            if planned_outage_factor is None:
                self.power += power_tmp * expansion_factor
            else:
                self.power += power_tmp * expansion_factor * (1 - planned_outage_factor)
        return self.power

class WindFarm():
    def __init__(self, power_curve : dict,):
        self.state = power_curve['State']
        self.county = power_curve['County']
        self.capacity = power_curve['Capacity']
        self.hub_height = power_curve['HubHeight']
        self.power_curve = power_curve['ShiftedPowerCurve']
        self.coordinate = county2coordinate(self.state, self.county)
    
    def get_power(self, wspd : dict, scheduled_on = None):
        if scheduled_on is None:
            scheduled_on = np.ones(wspd['Time'].shape[0])
        wspd_coordinate = list(wspd['WSPD_100m'].keys())
        wspd_coordinate_tmp = get_closest_coordinate(self.coordinate, wspd_coordinate)
        wspd_tmp = wspd['WSPD_100m'][wspd_coordinate_tmp]
        wind_power_norm = np.interp(wspd_tmp, self.power_curve.index.values, self.power_curve.values, left=0, right=0)
        capacity = self.capacity * scheduled_on
        self.power = np.multiply(wind_power_norm, capacity)
        return self.power


def shift_turbine_curve(turbine_curve, hub_height, maxspd, new_curve_res):
    """Shift a turbine curve based on a given hub height, x:wind speed, y: .

    :param pandas.Series turbine_curve: power curve data, wind speed index.
    :param float hub_height: height to shift power curve to.
    :param float maxspd: Extent of new curve (m/s).
    :param float new_curve_res: Resolution of new curve (m/s).
    """
    curve_x = np.arange(0, maxspd + new_curve_res, new_curve_res)
    wspd_scale_factor = (wind_const.wspd_height_base_100m / hub_height) ** wind_const.wspd_exp
    shifted_x = turbine_curve.index * wspd_scale_factor
    shifted_curve = np.interp(curve_x, shifted_x, turbine_curve, left=0, right=0)
    shifted_curve = pd.Series(data=shifted_curve, index=curve_x)
    shifted_curve.index.name = "Speed bin (m/s)"
    return shifted_curve

def get_turbine_power_curves(
        data_root : str = 'H:/Climate_grid_reliability/data',
        filename : str = "WindTurbinePowerCurves.csv"
    ) -> pd.DataFrame:
    """Load turbine power curves from csv.

    :param str filename: filename (not path) of csv file to read from.
    :return: (*pandas.DataFrame*) -- normalized turbine power curves.
    """
    data_dir = os.path.join(data_root, 'EIA') 
    powercurves_path = os.path.join(data_dir, filename)
    if not os.path.isfile(powercurves_path):
        raise ValueError("Power curves data by manufacturer not found at {path}".format(path=powercurves_path))
    power_curves = pd.read_csv(powercurves_path, index_col=0, header=None).T
    power_curves.set_index("Speed bin (m/s)", inplace=True)
    return power_curves

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
    form_860_filename = "3_2_Wind_Y{year}.xlsx".format(year=year)
    form_860_path = os.path.join(data_dir, form_860_folder, form_860_filename)
    if not os.path.isfile(form_860_path):
        raise ValueError("Form 860 data of year {year} not found at {path}".format(year=year, path=form_860_path))
    form_860 = pd.read_excel(form_860_path, sheet_name = 'Operable', skiprows=1)
    return form_860

def get_state_wind_power_curves(
        state : str = 'TX',
        year : int = 2019,
        default : str = "IEC class 2",
        maxspd : float = 30,
        data_root : str = 'H:/Climate_grid_reliability/data',
    ):
    if state != 'TX':
        raise Warning("Only Texas is supported at this time")
    turbine_power_curves = get_turbine_power_curves()
    form_860 = get_form_860(data_root = data_root, year = year)
    state_wind_farms = form_860.loc[form_860['State'] == state]
    state_power_curves = []
    for i in tqdm(range(len(state_wind_farms.index))):
        # Look up attributes from Form 860
        farm_capacity = state_wind_farms[wind_const.capacity_col].iloc[i]
        hub_height = state_wind_farms[wind_const.hub_height_col].iloc[i]
        turbine_mfg = state_wind_farms[wind_const.mfg_col].iloc[i]
        turbine_model = state_wind_farms[wind_const.model_col].iloc[i]
        county = state_wind_farms[wind_const.county_col].iloc[i]
        plant_id = state_wind_farms[wind_const.plant_id_col].iloc[i]
        generator_id = state_wind_farms[wind_const.generator_id_col].iloc[i]
        # Look up turbine-specific power curve (or default)
        turbine_name = " ".join([turbine_mfg, turbine_model])
        if turbine_name not in turbine_power_curves.columns:
            turbine_name = default
        turbine_curve = turbine_power_curves[turbine_name]
        # Shift based on farm-specific hub height
        shifted_curve = shift_turbine_curve(
            turbine_curve, hub_height, maxspd, wind_const.new_curve_res
        )
        # integrate all information
        power_curve = {
            "State": state,
            "County": county,
            "Capacity": farm_capacity,
            "HubHeight": hub_height,
            "ShiftedPowerCurve": shifted_curve,
            "ID": "{plant_id}_{generator_id}".format(plant_id = plant_id, generator_id = generator_id),
        }
        state_power_curves.append(power_curve)
    return state_power_curves

def get_state_wind_speed(
        data_root:str = 'H:/Climate_grid_reliability/data',
        data_source:str = 'ASOS',
        data_file:str = None,
        start_time = None,
        end_time = None,
    ):
    """Get wind speed data from ASOS or iHESP"""
    if data_source == 'iHESP':
        path = os.path.join(data_root, 'iHESP', data_file)
        data = ClimateData(path, start_time, end_time)
        return data.get_wind_speed(start_time, end_time)
    elif data_source == 'ASOS':#TODO
        path = os.path.join(data_root, 'ASOS', data_file)
        data = WeatherData(path)
        return data.get_wind_speed(start_time, end_time)
    else:
        raise ValueError("data_source must be 'iHESP' or 'ASOS'")


#%% utils
def _read_state_wind_power_curves(file_path):
    with open(file_path, 'rb') as f:
        state_power_curves = pickle.load(f)
    return state_power_curves

def _save_state_wind_power_curves(file_path, state_power_curves):
    with open(file_path, 'wb') as f:
        pickle.dump(state_power_curves, f)

if __name__=="__main__":
    windfarms = WindFarms(
        state = 'TX',
        year = 2019,
        data_root = data_const.ROOT_DIR,
    )
    windfarms.add_wind_speed(
        data_root = 'H:/Climate_grid_reliability/data',
        data_source = 'iHESP',
        data_file = 'CESM_LR_RCP85_ENS01_Climate-Power_CONUS.nc',
    )
    windfarms.get_power()

    input("Press Enter to continue...")