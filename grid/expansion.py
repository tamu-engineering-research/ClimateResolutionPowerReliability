import os
import sys

from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
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
from utils import county2coordinate
from utils import get_closest_coordinate
from utils import time2season


class CapacityExpansion():#TODO
    def __init__(self, state : str, data_root : str):
        self.state = state
        self.data_root = data_root
        self.historical_capacity = self.get_historical_capacity()
    
    def get_capacity_expansion(self, base_year = 2019):
        wind_base = self.historical_capacity[base_year]['wind']
        solar_base = self.historical_capacity[base_year]['solar']
        hydro_base = self.historical_capacity[base_year]['hydro']
        thermal_base = self.historical_capacity[base_year]['thermal']
        x, y_wind, y_solar, y_hydro, y_thermal = [], [], [], [], []
        for year in self.historical_capacity:
            x.append(year)
            y_wind.append(self.historical_capacity[year]['wind']/wind_base)
            y_solar.append(self.historical_capacity[year]['solar']/solar_base)
            y_hydro.append(self.historical_capacity[year]['hydro']/hydro_base)
            y_thermal.append(self.historical_capacity[year]['thermal']/thermal_base)
        x, y_wind, y_solar, y_hydro, y_thermal = np.array(x), np.array(y_wind), np.array(y_solar), np.array(y_hydro), np.array(y_thermal)
        
        # self._solar_regression_L2(x-2013, y_solar, np.arange(2022, 2044))
        
        # fit
        coefficients_wind = np.polyfit(x-2015, y_wind, deg=1)
        coefficients_solar = np.polyfit(x, y_solar, deg=2)
        coefficients_hydro = np.polyfit(x-2015, y_hydro, deg=1)
        coefficients_thermal = np.polyfit(x-2015, y_thermal, deg=1)
        p_wind = np.poly1d(coefficients_wind)
        p_solar = np.poly1d(coefficients_solar)
        p_hydro = np.poly1d(coefficients_hydro)
        p_thermal = np.poly1d(coefficients_thermal)        
        
        
        # predict
        x_pred = np.arange(2022, 2044)
        y_wind_pred = p_wind(x_pred-2015)
        y_solar_pred = p_solar(x_pred)
        y_hydro_pred = p_hydro(x_pred-2015)
        y_thermal_pred = p_thermal(x_pred-2015)

        # aggregate
        x_all = np.concatenate((x, x_pred))
        y_wind_all = np.concatenate((y_wind, y_wind_pred))
        y_solar_all = np.concatenate((y_solar, y_solar_pred))
        y_hydro_all = np.concatenate((y_hydro, y_hydro_pred))
        y_thermal_all = np.concatenate((y_thermal, y_thermal_pred))
        

        # capacity expansion factor
        capacity_expansion_factor = {}
        for index in range(x_all.shape[0]):
            year = x_all[index]
            capacity_expansion_factor[year] = {}
            capacity_expansion_factor[year]['wind'] = y_wind_all[index]
            capacity_expansion_factor[year]['solar'] = y_solar_all[index]
            capacity_expansion_factor[year]['hydro'] = y_hydro_all[index]
            capacity_expansion_factor[year]['thermal'] = y_thermal_all[index]
        
        # plot
        _plot_capacity_expansion(capacity_expansion_factor, wind_base, solar_base, hydro_base, thermal_base)
        
        return capacity_expansion_factor

    def _solar_regression(self, x, y_solar, x_pred):
        def logistic_growth(t, a, b, c):
            return c / (1 + a * np.exp(-b*t))
        # fit
        scaler_x = StandardScaler().fit(x[:, np.newaxis])
        scaler_y = StandardScaler().fit(y_solar[:, np.newaxis])
        popt_solar, _ = curve_fit(
            logistic_growth, 
            x-2013,
            # scaler_x.transform(x[:, np.newaxis]).ravel(), 
            y_solar,
            # scaler_y.transform(y_solar[:, np.newaxis]).ravel(),
            p0 = [1, 1, 10],
        )
        # predict
        x_pred = np.arange(2022, 2044)
        y_solar_pred = logistic_growth(x_pred-2013, *popt_solar)
        y_solar_pred = logistic_growth(scaler_x.transform(x_pred[:, np.newaxis]), *popt_solar)
        y_solar_pred = scaler_y.inverse_transform(y_solar_pred)
        y_solar_pred = y_solar_pred.ravel()
        return y_solar_pred
    
    def _solar_regression_L2(self, x, y, x_pred ):
        def custom_function(x, a, b, c):
            return c / (1 + a * np.exp(-b*x))
        def custom_function_grad(x, a, b, c):
            return np.array([a*b*c*np.exp(-b*x)/((1+a*np.exp(-b*x))**2), np.ones_like(x)])
        def loss_function(params, x, y, l2_lambda):
            a, b, c = params
            pred = custom_function(x, a, b, c)
            loss = np.sum((y - pred) ** 2) + l2_lambda * np.sum(params ** 2)
            return loss
        def loss_function_grad(params, x, y, l2_lambda):
            a, b, c = params
            pred = custom_function(x, a, b, c)
            grad_a = np.sum(2 * (y - pred) * (1/(1 + a * np.exp(-b*x))))
            grad_b = np.sum(2 * (y - pred) * (a*c*x*np.exp(-b*x)/((1+a*np.exp(-b*x))**2)))
            grad_c = np.sum(2 * (y - pred) * (-c*np.exp(-b*x)/((1+a*np.exp(-b*x))**2)))
            grad = np.array([grad_a, grad_b, grad_c]) + 2 * l2_lambda * params
            return grad
        l2_lambda = 0.1
        params_init = np.array([1, 1, 10])
        result = minimize(loss_function, params_init, args=(x, y, l2_lambda), method='L-BFGS-B', jac=loss_function_grad)
        return

    def get_historical_capacity(self):
        file_name = "capacity_by_fuel_over_years.pkl"
        file_path = os.path.join(self.data_root, 'EIA', file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                historical_capacity = pickle.load(f)
            return historical_capacity
        historical_capacity = {}
        for year in tqdm(range(2014, 2022)):
            historical_capacity[year] = self.get_historical_capacity_yearly(year)
        with open(file_path, 'wb') as f:
            pickle.dump(historical_capacity, f)
        return historical_capacity
    
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
    

class LoadExpansion():
    def __init__(self, state : str, data_root : str):
        self.state = state
        self.data_root = data_root
        self.forecasted_load = self.get_forecasted_load()
        self.historical_load = self.get_historical_load()
    
    def get_load_expansion(self,):
        raise NotImplementedError
    
    def get_load_expansion_naive(self, based_year = 2018):
        """
        Naive load expansion model: 
        uniformly scaling up the load by x% per year based on ERCOT report
        """
        def logistic_growth(t, a, b, c):
            return c / (1 + a * np.exp(-b*t))
        base_peak = max(self.historical_load[based_year].values())
        base_valley = min(self.historical_load[based_year].values())
        x, y_peak, y_valley = [], [], []

        for key in self.forecasted_load:
            x.append(key)
            y_peak.append(max(self.forecasted_load[key].values())/base_peak)
            y_valley.append(min(self.forecasted_load[key].values())/base_valley)
        x, y_peak, y_valley = np.array(x), np.array(y_peak), np.array(y_valley)
        
        # fit version 2
        popt_peak, pcov_peak = curve_fit(logistic_growth, x-2023, y_peak)
        popt_valley, pcov_valley = curve_fit(logistic_growth, x-2023, y_valley)
        # predict
        x_to_predict = np.arange(2033, 2044)
        y_peak_to_predict = logistic_growth(x_to_predict-2023, *popt_peak)
        y_valley_to_predict = logistic_growth(x_to_predict-2023, *popt_valley)
        
        # aggregate
        x_all = np.concatenate((x, x_to_predict))
        y_peak_all = np.concatenate((y_peak, y_peak_to_predict))
        y_valley_all = np.concatenate((y_valley, y_valley_to_predict))
        load_expansion_factor = {}
        for index in range(x_all.shape[0]):
            load_expansion_factor[x_all[index]] = {}
            load_expansion_factor[x_all[index]]['peak'] = y_peak_all[index]
            load_expansion_factor[x_all[index]]['valley'] = y_valley_all[index]
        
        # plot
        _plot_load_expansion(x_all, y_peak_all*74665, y_valley_all*27612)
        return load_expansion_factor
    
    def get_historical_load(self):
        """ load historically recorded by ERCOT """
        ERCOTdata = ERCOTData(dir = os.path.join(self.data_root, 'ERCOT', 'Load'))
        list_years = list(np.arange(2018, 2022))
        historical_load = {}
        for year in list_years:
            month = list(np.arange(1, 13))
            historical_load[year] = ERCOTdata.get_monthly_peak_load(year, month)
        return historical_load
    
    def get_forecasted_load(self):
        """ load forecasted by ERCOT """
        file_name = "ERCOT-Monthly-Peak-Demand-and-Energy-Forecast-2023-2032.xlsx"
        path = os.path.join(self.data_root, 'ERCOT', 'LongTermForecast', file_name)
        df = pd.read_excel(path, sheet_name = 'Sheet1', header = 2)
        df['Year'] = df['Year'].astype(int)
        df['Month'] = df['Month'].astype(int)
        forecasted_load = {}
        list_years = sorted(df['Year'].unique())
        for year in list_years:
            forecasted_load[year] = {}
            df_year = df[df['Year'] == year]
            list_months = sorted(df_year['Month'].unique())
            assert len(list_months) == 12
            for month in list_months:
                df_month = df_year[df_year['Month'] == month]
                load = df_month['Peak (MW)'].values[0]
                forecasted_load[year][month] = load
        return forecasted_load

def _plot_load_expansion(x_all, y_peak_all, y_valley_all):
    x_all = x_all.astype(int)
    mask_ercot = np.where(x_all <= 2032)[0]
    mask_predict = np.where(x_all > 2032)[0]
    plt.figure(figsize=(6, 6))
    plt.plot(x_all[mask_ercot], y_peak_all[mask_ercot], color='#001219', linestyle='solid', marker='o', label = 'Maximum monthly peak by ERCOT')
    plt.plot(x_all[mask_predict], y_peak_all[mask_predict], color='#001219', linestyle='dashed', marker='o', label = 'Maximum monthly peak by our study')

    plt.plot(x_all[mask_ercot], y_valley_all[mask_ercot], color='#9B2226', linestyle='solid', marker='*', label = 'Minimum monthly peak by ERCOT')
    plt.plot(x_all[mask_predict], y_valley_all[mask_predict], color='#9B2226', linestyle='dashed', marker='*', label = 'Minimum monthly peak by our study')
    plt.legend()
    plt.xticks(np.arange(2023, 2044, 4))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
    plt.xlabel('Year')
    plt.ylabel('Load demand (MW)')
    plt.savefig('./figure/load_expansion_v2.pdf', format='pdf', dpi=400,)
    input()

def _plot_capacity_expansion(capacity_expansion_factor, wind_base, solar_base, hydro_base, thermal_base):
    x = np.array(list(capacity_expansion_factor.keys()))
    y_wind = np.array([capacity_expansion_factor[year]['wind'] for year in x])
    y_solar = np.array([capacity_expansion_factor[year]['solar'] for year in x])
    y_hydro = np.array([capacity_expansion_factor[year]['hydro'] for year in x])
    y_thermal = np.array([capacity_expansion_factor[year]['thermal'] for year in x])
    y_wind = y_wind*wind_base
    y_solar = y_solar*solar_base
    y_hydro = y_hydro*hydro_base
    y_thermal = y_thermal*thermal_base
    mask = np.where(x <= 2022)[0]
    mask2 = np.where(x > 2022)[0]
    plt.figure(figsize=(6, 6))
    plt.plot(x[mask], y_wind[mask], color='#001219', linestyle='solid', marker='*', label = 'Wind generation capacity (historical)')
    plt.plot(x[mask], y_solar[mask], color='#9B2226', linestyle='solid',  marker='o', label = 'Solar generation capacity (historical)')
    plt.plot(x[mask], y_hydro[mask], color='#5F4B8B', linestyle='solid',  marker='^', label = 'Hydro generation capacity (historical)')
    plt.plot(x[mask], y_thermal[mask], color='#FFB703', linestyle='solid',  marker='<', label = 'Thermal generation capacity (historical)')
    
    plt.plot(x[mask2], y_wind[mask2], color='#001219', linestyle='dashed', marker='*', label = 'Wind generation capacity (predicted)')
    plt.plot(x[mask2], y_solar[mask2], color='#9B2226', linestyle='dashed', marker='o', label = 'Solar generation capacity (predicted)')
    plt.plot(x[mask2], y_hydro[mask2], color='#5F4B8B', linestyle='dashed', marker='^', label = 'Hydro generation capacity (predicted)')
    plt.plot(x[mask2], y_thermal[mask2], color='#FFB703', linestyle='dashed', marker='<', label = 'Thermal generation capacity (predicted)')
    plt.legend()
    plt.xticks(np.arange(2014, 2044, 3))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
    plt.xlabel('Year')
    plt.ylabel('Generation capacity (MW)')
    plt.savefig('./figure/capacity_expansion.pdf', format='pdf', dpi=400, bbox_inches='tight')
    return

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

if __name__=="__main__":
    # capacityexpansion = CapacityExpansion(
    #     state = 'TX',
    #     data_root = data_const.ROOT_DIR,
    # )
    # capacityexpansion.get_capacity_expansion()

    loadexpansion = LoadExpansion(
        state = 'TX',
        data_root = data_const.ROOT_DIR,
    )
    loadexpansion.get_load_expansion_naive()
    input()