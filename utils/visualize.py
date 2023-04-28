import os
import sys

from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
from mpl_toolkits.basemap import Basemap

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

def visualize_data_on_map(lats, lons, temps):
    # Set up the map projection
    m = Basemap(projection='merc', llcrnrlon=-106.8, llcrnrlat=25, urcrnrlon=-93.5, urcrnrlat=37, resolution='i')

    # Add map elements
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    m.drawcounties()

    # # Plot the temperature data as a contour
    # levels = list(np.arange(-40,40,2))
    
    # Plot the wind data as a contour
    levels = list(np.arange(0,20,2))
    x, y = m(lons, lats)
    m.contourf(x, y, temps, cmap='RdBu_r', levels=levels, alpha=0.5)

    plt.colorbar()
    plt.show()
    return

def convert_data_for_map(list_lats, list_lons, list_temps):
    set_lats = sorted(set(list_lats))
    set_lons = sorted(set(list_lons))
    temps = np.zeros((len(set_lats), len(set_lons)))
    lats = np.zeros((len(set_lats), len(set_lons)))
    lons = np.zeros((len(set_lats), len(set_lons)))
    for i in range(len(list_lats)):
        lat_index = set_lats.index(list_lats[i])
        lon_index = set_lons.index(list_lons[i])
        temps[lat_index][lon_index] = list_temps[i]
        lats[lat_index][lon_index] = list_lats[i]
        lons[lat_index][lon_index] = list_lons[i]
    return lats, lons, temps

def extract_data_at_time(env_data, target_time):
    index = np.where(env_data['Time'] == target_time)[0][0]
    list_lats, list_lons, list_temps, list_winds = [], [],  [], []
    for coordinate in env_data['Temperature']:
        list_lats.append(coordinate[0])
        list_lons.append(coordinate[1])
        list_temps.append(env_data['Temperature'][coordinate][index])
        list_winds.append(env_data['WindSpeed'][coordinate][index])
    return list_lats, list_lons, list_temps, list_winds

def load_adequacy(save_dir, file_name):
        with open(os.path.join(save_dir, file_name), 'rb') as f:
            results = pickle.load(f)
        return results


def compare_adequacy_curve():
    save_dir = os.path.join(data_const.ROOT_DIR, 'Results',)
    HR_adequacy, HR_time = None, None 
    LR_adequacy, LR_time = None, None
    for year in range(2033, 2044):
        file_name = 'HR_adequacy_{}.pkl'.format(year)
        results_tmp = load_adequacy(save_dir, file_name)
        if HR_adequacy is None:
            HR_adequacy, HR_time = results_tmp['adequacy'], results_tmp['time']
        else:
            HR_adequacy = np.concatenate((HR_adequacy, results_tmp['adequacy']), axis = 0)
            HR_time = np.concatenate((HR_time, results_tmp['time']), axis = 0)
        file_name = 'LR_adequacy_{}.pkl'.format(year)
        results_tmp = load_adequacy(save_dir, file_name)
        if LR_adequacy is None:
            LR_adequacy, LR_time = results_tmp['adequacy'], results_tmp['time']
        else:
            LR_adequacy = np.concatenate((LR_adequacy, results_tmp['adequacy']), axis = 0)
            LR_time = np.concatenate((LR_time, results_tmp['time']), axis = 0)
    plt.plot(HR_time, HR_adequacy, label = 'HR')
    plt.plot(LR_time, LR_adequacy, label = 'LR')
    plt.plot(LR_time, 2300*np.ones(LR_time.shape), color= 'red', linestyle='dashed',  label='EEA Threshold')
    plt.legend()
    plt.xlabel('Time (Year)')
    plt.ylabel('Resource Adequacy (MW)')
    plt.show()
    return

def compare_adequacy_barplot_seasonal(ax = None):
    save_dir = os.path.join(data_const.ROOT_DIR, 'Results',)
    thresholds = [0, 2300, 2300*2,]
    years = np.arange(2033, 2044)
    length = years.shape[0]
    HR_L0, HR_L1, HR_L2 = np.zeros((length, 4)), np.zeros((length, 4)), np.zeros((length, 4))
    LR_L0, LR_L1, LR_L2 = np.zeros((length, 4)), np.zeros((length, 4)), np.zeros((length, 4))
    for year_index, year in enumerate(range(2033, 2044)):
        file_name = 'HR_adequacy_{}.pkl'.format(year)
        HR_result_tmp = load_adequacy(save_dir, file_name)
        HR_season = time2season(HR_result_tmp['time'])
        HR_result_spring = HR_result_tmp['adequacy'][HR_season==1]
        HR_result_summer = HR_result_tmp['adequacy'][HR_season==2]
        HR_result_autumn = HR_result_tmp['adequacy'][HR_season==3]
        HR_result_winter = HR_result_tmp['adequacy'][HR_season==4]
        file_name = 'LR_adequacy_{}.pkl'.format(year)
        LR_result_tmp = load_adequacy(save_dir, file_name)
        LR_season = time2season(LR_result_tmp['time'])
        LR_result_spring = LR_result_tmp['adequacy'][LR_season==1]
        LR_result_summer = LR_result_tmp['adequacy'][LR_season==2]
        LR_result_autumn = LR_result_tmp['adequacy'][LR_season==3]
        LR_result_winter = LR_result_tmp['adequacy'][LR_season==4]
        for index, threshold in enumerate(thresholds):
            if index==0:
                HR_L0[year_index][0] = np.sum(HR_result_spring<thresholds[index])
                HR_L0[year_index][1] = np.sum(HR_result_summer<thresholds[index])
                HR_L0[year_index][2] = np.sum(HR_result_autumn<thresholds[index])
                HR_L0[year_index][3] = np.sum(HR_result_winter<thresholds[index])
                LR_L0[year_index][0] = np.sum(LR_result_spring<thresholds[index])
                LR_L0[year_index][1] = np.sum(LR_result_summer<thresholds[index])
                LR_L0[year_index][2] = np.sum(LR_result_autumn<thresholds[index])
                LR_L0[year_index][3] = np.sum(LR_result_winter<thresholds[index])
            elif index==1:
                HR_L1[year_index][0] = np.sum((HR_result_spring>=thresholds[index-1])&(HR_result_spring<thresholds[index]))
                HR_L1[year_index][1] = np.sum((HR_result_summer>=thresholds[index-1])&(HR_result_summer<thresholds[index]))
                HR_L1[year_index][2] = np.sum((HR_result_autumn>=thresholds[index-1])&(HR_result_autumn<thresholds[index]))
                HR_L1[year_index][3] = np.sum((HR_result_winter>=thresholds[index-1])&(HR_result_winter<thresholds[index]))
                LR_L1[year_index][0] = np.sum((LR_result_spring>=thresholds[index-1])&(LR_result_spring<thresholds[index]))
                LR_L1[year_index][1] = np.sum((LR_result_summer>=thresholds[index-1])&(LR_result_summer<thresholds[index]))
                LR_L1[year_index][2] = np.sum((LR_result_autumn>=thresholds[index-1])&(LR_result_autumn<thresholds[index]))
                LR_L1[year_index][3] = np.sum((LR_result_winter>=thresholds[index-1])&(LR_result_winter<thresholds[index]))
            elif index==2:
                HR_L2[year_index][0] = np.sum((HR_result_spring>=thresholds[index-1])&(HR_result_spring<thresholds[index]))
                HR_L2[year_index][1] = np.sum((HR_result_summer>=thresholds[index-1])&(HR_result_summer<thresholds[index]))
                HR_L2[year_index][2] = np.sum((HR_result_autumn>=thresholds[index-1])&(HR_result_autumn<thresholds[index]))
                HR_L2[year_index][3] = np.sum((HR_result_winter>=thresholds[index-1])&(HR_result_winter<thresholds[index]))
                LR_L2[year_index][0] = np.sum((LR_result_spring>=thresholds[index-1])&(LR_result_spring<thresholds[index]))
                LR_L2[year_index][1] = np.sum((LR_result_summer>=thresholds[index-1])&(LR_result_summer<thresholds[index]))
                LR_L2[year_index][2] = np.sum((LR_result_autumn>=thresholds[index-1])&(LR_result_autumn<thresholds[index]))
                LR_L2[year_index][3] = np.sum((LR_result_winter>=thresholds[index-1])&(LR_result_winter<thresholds[index]))

    HR_L0_all, HR_L1_all, HR_L2_all = np.sum(HR_L0, axis=0), np.sum(HR_L1, axis=0), np.sum(HR_L2, axis=0)
    LR_L0_all, LR_L1_all, LR_L2_all = np.sum(LR_L0, axis=0), np.sum(LR_L1, axis=0), np.sum(LR_L2, axis=0)
    all_results = np.column_stack((HR_L0_all,  HR_L1_all, HR_L2_all, LR_L0_all, LR_L1_all, LR_L2_all))#merge results
    all_results = np.where(all_results==0, 0.2, all_results)
    labels = ['HR: Blackout',  'HR: Emergency', 'HR: Warning',
              'LR: Blackout',  'LR: Emergency', 'LR: Warning']
    colors = ['#013A63','#014F86','#2A6F97','#212529','#495057','#ADB5BD'] # contrast color
    pos = np.arange(all_results.shape[0])*1.5
    width = 0.15
    xlabels = ['Spring', 'Summer', 'Fall', 'Winter']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 2.5),)
    for i in range(len(labels)):
        if i<=2:
            ax.bar(pos+i*width, all_results[:, i].ravel(), width, label=labels[i], color=colors[i])
        else:
            ax.bar(pos+(i+1)*width, all_results[:, i].ravel(), width, label=labels[i], color=colors[i])
    ax.set_xticks(pos+(len(labels)-0)/2*width)
    ax.set_xticklabels(xlabels)
    ax.set_ylim([0, 30])
    ax.set_xlabel('All years')
    # ax.set_ylabel('Count of events')
    # ax.legend()
    # plt.savefig('./figure/adequacy_barplot_seasonal.pdf', format='pdf', dpi=400,)
    return

def compare_adequacy_barplot():
    save_dir = os.path.join(data_const.ROOT_DIR, 'Results',)
    thresholds = [0, 2300, 2300*2,]
    years = np.arange(2033, 2044)
    HR_L0, HR_L1, HR_L2 = np.zeros(years.shape), np.zeros(years.shape), np.zeros(years.shape)
    LR_L0, LR_L1, LR_L2 = np.zeros(years.shape), np.zeros(years.shape), np.zeros(years.shape)
    for year_index, year in enumerate(range(2033, 2044)):
        file_name = 'HR_adequacy_{}.pkl'.format(year)
        HR_result_tmp = load_adequacy(save_dir, file_name)
        file_name = 'LR_adequacy_{}.pkl'.format(year)
        LR_result_tmp = load_adequacy(save_dir, file_name)
        for index, threshold in enumerate(thresholds):
            if index==0:
                HR_L0[year_index] = np.sum(HR_result_tmp['adequacy']<thresholds[index])
                LR_L0[year_index] = np.sum(LR_result_tmp['adequacy']<thresholds[index])
            elif index==1:
                HR_L1[year_index] = np.sum((HR_result_tmp['adequacy']>=thresholds[index-1])&(HR_result_tmp['adequacy']<thresholds[index]))
                LR_L1[year_index] = np.sum((LR_result_tmp['adequacy']>=thresholds[index-1])&(LR_result_tmp['adequacy']<thresholds[index]))
            elif index==2:
                HR_L2[year_index] = np.sum((HR_result_tmp['adequacy']>=thresholds[index-1])&(HR_result_tmp['adequacy']<thresholds[index]))
                LR_L2[year_index] = np.sum((LR_result_tmp['adequacy']>=thresholds[index-1])&(LR_result_tmp['adequacy']<thresholds[index]))

    all_results = np.column_stack((HR_L0,  HR_L1, HR_L2, LR_L0, LR_L1, LR_L2))#merge results
    result_all_years = np.sum(all_results, axis=0).ravel() # sum over years
    # all_results = np.append(all_results, np.sum(all_results, axis=0).reshape(1,-1), axis=0) # add total
    all_results = np.where(all_results==0, 0.1, all_results) # avoid zero
    labels = ['High resolution: Blackout cases',  'High resolution: Emergency cases', 'High resolution: Warning cases',
              'Low resolution: Blackout cases',  'Low resolution: Emergency cases', 'Low resolution: Warning cases']
    # colors = ['#001219','#005F73','#0A9396','#9B2226','#BB3E03','#EE9B00'] # contrast color
    # colors = ['#013A63','#014F86','#2A6F97','#212529','#495057','#ADB5BD'] # contrast color
    colors = ['#013A63','#014F86','#2A6F97','#212529','#495057','#ADB5BD'] # contrast color
    pos = np.arange(all_results.shape[0])*1.5
    width = 0.15
    
    fig, ax = plt.subplots(nrows =1, ncols=3, figsize=(8.5, 2.5), gridspec_kw={'width_ratios': [8, 1, 4]})
    for i in range(len(labels)):
        if i<=2:
            ax[0].bar(pos+i*width, all_results[:, i].ravel(), width, label=labels[i], color=colors[i])
        else:
            ax[0].bar(pos+(i+1)*width, all_results[:, i].ravel(), width, label=labels[i], color=colors[i])
    ax[0].set_xticks(pos+(len(labels)-0)/2*width)
    ax[0].set_xticklabels(list(years))
    ax[0].set_ylim([0, 10])
    ax[0].set_xlabel('Year')
    ax[0].set_ylabel('Count of events')
    ax[0].legend()
    
    pos = np.array([0])
    for i in range(len(labels)):
        if i<=2:
            ax[1].bar(pos+i*width, result_all_years[i], width, label=labels[i], color=colors[i])
        else:
            ax[1].bar(pos+(i+1)*width, result_all_years[i], width, label=labels[i], color=colors[i])
    ax[1].set_xticks(pos+(len(labels)-0)/2*width)
    ax[1].set_xticklabels(['2033 - 2043'])
    ax[1].set_ylim([0, 35])
    ax[1].set_yticks([0,5,10,15,20,25,30,35])
    ax[1].set_xlabel('All years')
    compare_adequacy_barplot_seasonal(ax[2])
    plt.savefig('./figure/adequacy_barplot_v2.pdf', format='pdf', dpi=400,)
    return

def worst_blackout():
    save_dir = os.path.join(data_const.ROOT_DIR, 'Results',)
    HR_adequacy, HR_time, HR_wind, HR_solar, HR_thermal, HR_load = None, None, None, None, None, None
    LR_adequacy, LR_time, LR_wind, LR_solar, LR_thermal, LR_load = None, None, None, None, None, None
    for year in range(2033, 2044):
        file_name = 'HR_adequacy_{}.pkl'.format(year)
        results_tmp = load_adequacy(save_dir, file_name)
        if HR_adequacy is None:
            HR_adequacy =  results_tmp['adequacy']
            HR_time = results_tmp['time']
            HR_wind = results_tmp['wind_power']
            HR_solar = results_tmp['solar_power']
            HR_thermal = results_tmp['thermal_power']
            HR_load = results_tmp['load_power']
        else:
            HR_adequacy = np.concatenate((HR_adequacy, results_tmp['adequacy']), axis = 0)
            HR_time = np.concatenate((HR_time, results_tmp['time']), axis = 0)
            HR_wind = np.concatenate((HR_wind, results_tmp['wind_power']), axis = 0)
            HR_solar = np.concatenate((HR_solar, results_tmp['solar_power']), axis = 0)
            HR_thermal = np.concatenate((HR_thermal, results_tmp['thermal_power']), axis = 0)
            HR_load = np.concatenate((HR_load, results_tmp['load_power']), axis = 0)
        file_name = 'LR_adequacy_{}.pkl'.format(year)
        results_tmp = load_adequacy(save_dir, file_name)
        if LR_adequacy is None:
            LR_adequacy =  results_tmp['adequacy']
            LR_time = results_tmp['time']
            LR_wind = results_tmp['wind_power']
            LR_solar = results_tmp['solar_power']
            LR_thermal = results_tmp['thermal_power']
            LR_load = results_tmp['load_power']
        else:
            LR_adequacy = np.concatenate((LR_adequacy, results_tmp['adequacy']), axis = 0)
            LR_time = np.concatenate((LR_time, results_tmp['time']), axis = 0)
            LR_wind = np.concatenate((LR_wind, results_tmp['wind_power']), axis = 0)
            LR_solar = np.concatenate((LR_solar, results_tmp['solar_power']), axis = 0)
            LR_thermal = np.concatenate((LR_thermal, results_tmp['thermal_power']), axis = 0)
            LR_load = np.concatenate((LR_load, results_tmp['load_power']), axis = 0)
    index_HR = np.argmin(HR_adequacy, axis = 0)
    index_LR = np.argmin(LR_adequacy, axis = 0)

    HR_HR = np.array([HR_wind[index_HR], HR_solar[index_HR], HR_thermal[index_HR], HR_load[index_HR]])
    LR_HR = np.array([LR_wind[index_HR], LR_solar[index_HR], LR_thermal[index_HR], LR_load[index_HR]])
    HR_LR = np.array([HR_wind[index_LR], HR_solar[index_LR], HR_thermal[index_LR], HR_load[index_LR]])
    LR_LR = np.array([LR_wind[index_LR], LR_solar[index_LR], LR_thermal[index_LR], LR_load[index_LR]])
    HR_HR = np.where(HR_HR==0, 1000, HR_HR)
    LR_HR = np.where(LR_HR==0, 1000, LR_HR)
    HR_LR = np.where(HR_LR==0, 1000, HR_LR)
    LR_LR = np.where(LR_LR==0, 1000, LR_LR)
    labels = ['Wind', 'Solar', 'Thermal', 'Load']
    # colors = ['#001219', '#9B2226', '#5F4B8B', '#FFB703']
    #harmonic 4-colors
    colors = ['#001219', '#005F73', '#0A9396', '#94D2BD']
    width = 0.6
    pos = np.array([0, 1, 2, 3])
    fig, ax = plt.subplots(1, 4, figsize=(8, 2))
    for i in range(len(labels)):
        ax[0].bar(pos[i], HR_HR[i], width, label=labels[i], color=colors[i])
        ax[1].bar(pos[i], LR_HR[i], width, label=labels[i], color=colors[i])
        ax[2].bar(pos[i], HR_LR[i], width, label=labels[i], color=colors[i])
        ax[3].bar(pos[i], LR_LR[i], width, label=labels[i], color=colors[i])
    for i in range(4):
        ax[i].set_xticks(pos)
        ax[i].set_xticklabels(labels)
        ax[i].set_ylim([0, 85000])
        ax[i].yaxis.set_label_position('right')
        ax[i].yaxis.tick_right()
    plt.show()
    return


if __name__ == "__main__":
    worst_blackout()
    input()
    # compare_adequacy_curve()
    # input()
    # compare_adequacy_barplot()
    # input()

    from data import ClimateData
    data_root = 'H:/Climate_grid_reliability/data'
    data_file = 'CESM_HR_RCP85_ENS01_Climate-Power_CONUS.nc'
    path = os.path.join(data_root, 'iHESP', data_file)
    year = 2036
    start_time=datetime(int(year), 1, 1, 0, 0, 0)
    end_time=datetime(int(year), 12, 30, 0, 0, 0)
    climatedata = ClimateData(
        path,
        start_time=start_time,
        end_time=end_time,
    )
    env_data = climatedata.get_environmental_data(start_time, end_time)
    target_time = datetime(int(year), 2, 5, 12, 0, 0)
    list_lats, list_lons, list_temps, list_winds = extract_data_at_time(env_data, target_time)
    lats, lons, temps = convert_data_for_map(list_lats, list_lons, list_temps)
    lats, lons, winds = convert_data_for_map(list_lats, list_lons, list_winds)
    # visualize_data_on_map(lats, lons, temps)
    visualize_data_on_map(lats, lons, winds)
    input()
