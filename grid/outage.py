import os
import sys

from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['pdf.fonttype'] = 42
from scipy.optimize import curve_fit, minimize
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, StandardScaler

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.getcwd()))
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.ERCOT_data import ERCOTData
from utils import thermal_const
from utils import hydro_const
from utils import solar_const
from utils import wind_const
from utils import data_const


class PlannedUnitOutage():
    def __init__(self, state:str, data_root:str):
        self.state = state
        self.data_root = data_root
    
    def get_planned_outage_factor(self, year=2019):
        ERCOTdata =  ERCOTData(self.data_root)
        planned_outage_data_dir = os.path.join(self.data_root, 'ERCOT', 'PlannedUnitOutage')
        planned_outage_data = ERCOTdata.get_yearly_planned_outage(planned_outage_data_dir, year) 
        capacity = self.get_historical_capacity_yearly(year)
        capacity_renewable = capacity['solar'] + capacity['wind']
        capacity_non_renewable = capacity['thermal'] + capacity['hydro']
        planned_outage_factor_renewable = planned_outage_data['Renewable'] / capacity_renewable
        planned_outage_factor_non_renewable = planned_outage_data['Non_Renewable'] / capacity_non_renewable
        planned_outage_factor = {}
        planned_outage_factor['time'] = planned_outage_data['Time']
        planned_outage_factor['wind'] = planned_outage_factor_renewable
        planned_outage_factor['solar'] = planned_outage_factor_renewable
        planned_outage_factor['thermal'] = planned_outage_factor_non_renewable
        planned_outage_factor['hydro'] = planned_outage_factor_non_renewable
        return planned_outage_factor
    
    def get_historical_capacity_yearly(self, year):
        capacity = {}
        form_860 = get_form_860(year=year)
        state_condition = form_860['State'] == 'TX'
        status_condition = form_860['Status'] == 'OP'
        thermal_condition = ~(form_860['Technology'].isin(thermal_const.non_thermal_technology))
        hydro_condition = form_860['Technology'].isin(hydro_const.hydro_technology)
        solar_condition = form_860['Technology'].isin(solar_const.solar_technology)
        wind_condition = form_860['Technology'].isin(wind_const.wind_technology)
        capacity['thermal'] = form_860[thermal_condition & state_condition & status_condition]['Nameplate Capacity (MW)'].sum()
        capacity['hydro'] = form_860[hydro_condition & state_condition & status_condition]['Nameplate Capacity (MW)'].sum()
        capacity['solar'] = form_860[solar_condition & state_condition & status_condition]['Nameplate Capacity (MW)'].sum()
        capacity['wind'] = form_860[wind_condition & state_condition & status_condition]['Nameplate Capacity (MW)'].sum()
        return capacity
    
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
    if year>=2013:
        form_860_filename = "3_1_Generator_Y{year}.xlsx".format(year=year)
    elif year>=2010:
        form_860_filename = "GeneratorsY{year}.xls".format(year=year)
    elif year==2009:
        form_860_filename = "GenY{year}.xlsx".format(year=year)
    form_860_path = os.path.join(data_dir, form_860_folder, form_860_filename)
    if not os.path.isfile(form_860_path):
        raise ValueError("Form 860 data of year {year} not found at {path}".format(year=year, path=form_860_path))
    form_860 = pd.read_excel(form_860_path, sheet_name = 'Operable', skiprows=1)
    return form_860

def _plot_outage_factor(planned_outage_factor):
    time = planned_outage_factor['time']
    wind = planned_outage_factor['wind']
    thermal = planned_outage_factor['thermal']

    plt.figure(figsize=(6, 6))
    plt.plot(time, wind*100, color='#001219', 
            linestyle='solid',  
            label = 'Planned outage rate of intermittent resources',
            alpha=0.9
    )
    plt.plot(time, thermal*100, color='#9B2226', 
            linestyle='dashed',
            label = 'Planned outage rate of traditional resources',
            alpha=0.9,
    )
    plt.legend()
    date_format = mdates.DateFormatter('%m/%d')
    plt.gca().xaxis.set_major_formatter(date_format)
    # plt.xlim([2023, 2043])
    # plt.xticks(np.arange(2023, 2044, 4))
    # plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
    plt.xlabel('Day of year')
    plt.ylabel('Planned outage rate (%)')
    plt.savefig('./figure/outage_factor.pdf', format='pdf', dpi=400, bbox_inches='tight')
    input()

if __name__ == "__main__":
    plannedunitoutage = PlannedUnitOutage('TX', data_const.ROOT_DIR)
    planned_outage_factor = plannedunitoutage.get_planned_outage_factor()
    _plot_outage_factor(planned_outage_factor)