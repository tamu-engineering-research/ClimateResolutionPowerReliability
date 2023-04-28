import os
import sys

import cftime
from datetime import datetime
import numpy as np
import pandas as pd
import xarray


class ClimateData():
    def __init__(self, path, start_time = None, end_time = None):
        """
        :path: path to NetCDF file
        
        :return
        :self.temperature: 2m temperature (Celcius)
        :self.dew_point: 2m dew point (Celcius)
        :self.relative_humidity: 2m relative humidity (%)
        :self.direct_flux: direct flux (W/m2)
        :self.net_flux: Net flux (W/m2)
        :self.wind_speed_10m: 10m wind speed (m/s)
        :self.wind_speed_100m: 100m wind speed (m/s)
        """
        if 'CESM_HR' in path:
            ds = load_HR_data(path, start_time, end_time)
        elif 'CESM_LR' in path:
            ds = load_LR_data(path, start_time, end_time)
        else:
            raise ValueError('Invalid path')
        self.temperature = self._convert_temp(ds['T2m'].values)
        self.dew_point = self._convert_temp(ds['TD2m'].values)
        self.relative_humidity = ds['RH2m'].values
        self.direct_flux = ds['FSDS'].values
        self.net_flux = ds['FSNS'].values
        self.wind_speed_10m = ds['WSPD10'].values
        self.wind_speed_100m = ds['WSPD100'].values
        self.lon = self._convert_longitude(ds['lon'].values)
        self.lat = ds['lat'].values
        self.time = self._convert_time(ds['time'].values)
    
    def get_wind_speed(self, start_time = None, end_time = None):
        """
        Get wind speed data from iHESP during the specified time period
        :param start_time: start time of the time period
        :param end_time: end time of the time period
        :return wspd: {'Time': [time], 'WSPD_100m': {(lat, lon, lat, lon): [wind speed]}
        """
        if start_time is None:
            start_time = self.time[0]
        if end_time is None:
            end_time = self.time[-1]
        start_index = np.where(self.time == start_time)[0][0]
        end_index = np.where(self.time == end_time)[0][0]+1
        wspd = {}
        wspd['Time'] = self.time[start_index:end_index]
        wspd['WSPD_100m'] = {}
        for i in range(self.lat.shape[0]):
            for j in range(self.lon.shape[0]):
                lat = self.lat[i]
                lon = self.lon[j]
                coordinate = (lat, lon, lat, lon)
                wspd['WSPD_100m'][coordinate] = self.wind_speed_100m[start_index:end_index, i, j].flatten()
        return wspd
    
    def get_solar_radiation(self, start_time = None, end_time = None):
        """
        Get solar radiation data from iHESP during the specified time period
        :param start_time: start time of the time period
        :param end_time: end time of the time period
        :return solar: {
            'Time': [time],
            'DirectFlux': {(lat, lon, lat, lon): [solar irradiance],
            'NetFlux': {(lat, lon, lat, lon): [solar irradiance]
        }
        """
        if start_time is None:
            start_time = self.time[0]
        if end_time is None:
            end_time = self.time[-1]
        start_index = np.where(self.time == start_time)[0][0]
        end_index = np.where(self.time == end_time)[0][0]+1
        srd = {}
        srd['Time'] = self.time[start_index:end_index]
        srd['NetFlux'], srd['DirectFlux'] = {}, {}
        for i in range(self.lat.shape[0]):
            for j in range(self.lon.shape[0]):
                lat = self.lat[i]
                lon = self.lon[j]
                coordinate = (lat, lon, lat, lon)
                srd['NetFlux'][coordinate] = self.net_flux[start_index:end_index, i, j].flatten()
                srd['DirectFlux'][coordinate] = self.direct_flux[start_index:end_index, i, j].flatten()
        return srd
    
    def get_environmental_data(self, start_time = None, end_time = None):
        """Get environmental data from iHESP during the specified time period"""
        if start_time is None:
            start_time = self.time[0]
        if end_time is None:
            end_time = self.time[-1]
        start_index = np.where(self.time == start_time)[0][0]
        end_index = np.where(self.time == end_time)[0][0]+1
        env = {}
        env['Time'] = self.time[start_index:end_index]
        env['Temperature'], env['DewPoint'], env['RelativeHumidity'], env['WindSpeed'] = {}, {}, {}, {}
        for i in range(self.lat.shape[0]):
            for j in range(self.lon.shape[0]):
                lat = self.lat[i]
                lon = self.lon[j]
                coordinate = (lat, lon, lat, lon)
                env['Temperature'][coordinate] = self.temperature[start_index:end_index, i, j].flatten()
                env['DewPoint'][coordinate] = self.dew_point[start_index:end_index, i, j].flatten()
                env['RelativeHumidity'][coordinate] = self.relative_humidity[start_index:end_index, i, j].flatten()
                env['WindSpeed'][coordinate] = self.wind_speed_100m[start_index:end_index, i, j].flatten()
        return env

    def get_para_time(self, start_time=None, end_time=None):
        if start_time is None:
            start_time = self.time[0]
        if end_time is None:
            end_time = self.time[-1]
        start_index = np.where(self.time == start_time)[0][0]
        end_index = np.where(self.time == end_time)[0][0]+1
        para = {}
        para['Time'] = self.time[start_index:end_index]
        return para

    def _read_ds(self, path):
        """Read the netCDF file"""
        ds = xarray.open_dataset(path, decode_times = True)
        return ds
    
    def _convert_time(self, time):
        """Convert the time from cftime to datetime"""
        return np.array([datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in time])
    
    def _convert_temp(self, temp):
        """Convert the temperature from Kelvin to Celsius"""
        return temp - 273.15
    
    def _convert_longitude(self, lon):
        """Convert the longitude from 0-360 to -180-180"""
        return np.where(lon > 180, lon - 360, lon)

def load_HR_data(path, start_time, end_time):
    """Load the HR data from the netCDF file"""
    ds = xarray.open_dataset(path, decode_times = True)
    if isinstance(start_time, datetime):
        start_time = cftime.DatetimeNoLeap(start_time.year, start_time.month, start_time.day, start_time.hour,)
    if isinstance(end_time, datetime):
        end_time = cftime.DatetimeNoLeap(end_time.year, end_time.month, end_time.day, end_time.hour,)
    ds_selected = ds.sel(time = slice(start_time, end_time))
    ds.close()
    return ds_selected

def load_LR_data(path, start_time, end_time):
    """Load the LR data from the netCDF file"""
    ds = xarray.open_dataset(path, decode_times = True)
    if isinstance(start_time, datetime):
        start_time = cftime.DatetimeNoLeap(start_time.year, start_time.month, start_time.day, start_time.hour,)
    if isinstance(end_time, datetime):
        end_time = cftime.DatetimeNoLeap(end_time.year, end_time.month, end_time.day, end_time.hour,)
    ds_selected = ds.sel(time = slice(start_time, end_time))
    ds.close()
    return ds_selected

def load_lonlat(path):
    """Load the lonlat data from the netCDF file"""
    ds = xarray.open_dataset(path)
    return ds

if __name__ == '__main__':
    root_dir = r'H:\Climate_grid_reliability\data\iHESP'
    # # Load the HR data
    # ds = load_HR_data(os.path.join(root_dir, 'CESM_HR_RCP85_ENS01_Climate-Power_CONUS.nc'))
    # print(ds)

    # # Load the LR data
    # ds = load_LR_data(os.path.join(root_dir, 'CESM_LR_RCP85_ENS01_Climate-Power_CONUS.nc'))
    # print(ds)

    # # Load the lonlat data
    # ds = load_lonlat(os.path.join(root_dir, 'wrf_lonlat.nc'))
    # lon, lat = ds['XLONG'].values, ds['XLAT'].values

    # Load the ClimateData class
    ds = ClimateData(os.path.join(root_dir, 'CESM_LR_RCP85_ENS01_Climate-Power_CONUS.nc'))
    
    
    input("Press enter to continue...")