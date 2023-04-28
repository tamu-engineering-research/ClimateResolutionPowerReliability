import os
import sys

import cftime
import numpy as np
import pandas as pd
import requests
import pickle
import holidays

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.getcwd()))
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.time import utc2central, central2utc, cfttime2datetime
from utils.map import get_closest_coordinate

def env2x(
        time = None,
        temperature = None,
        dewpoint = None,
        relative_humidity = None,
        wind_speed = None,
        convert_UTC_to_CT = False,
        convert_6hour_to_1hour = False,
        convert_knot_to_mps = False,
        convert_F_to_C = False,
        convert_percent_to_fraction = False,
        move_3hour_backward = False,
        env_source = 'ASOS',
    ):
    """
    : param time : datetime in Central Time
    : param temperature : float in Celsius
    : param dewpoint : float in Celsius
    : param relative_humidity : float in fraction
    : param wind_speed : float in m/s
    """
    if isinstance(time[0], cftime.datetime):
        time = cfttime2datetime(time)
    if env_source == 'iHESP': # iHESP data is in UTC and average of pervious 6 hours
        convert_UTC_to_CT = True
        move_3hour_backward = True
    if convert_UTC_to_CT:
        time = utc2central(time)
    if move_3hour_backward:
        time = time - pd.Timedelta(hours = 3)
    if convert_F_to_C:
        temperature = F2C(temperature)
        dewpoint = F2C(dewpoint)
    if convert_knot_to_mps:
        wind_speed = knots2mps(wind_speed)
    if convert_percent_to_fraction:
        relative_humidity = percent2fraction(relative_humidity)
    if len(temperature.shape)==1:
        temperature = temperature.reshape(temperature.shape[0], 1)
    if len(dewpoint.shape)==1:
        dewpoint = dewpoint.reshape(dewpoint.shape[0], 1)
    if len(relative_humidity.shape)==1:
        relative_humidity = relative_humidity.reshape(relative_humidity.shape[0], 1)
    if len(wind_speed.shape)==1:
        wind_speed = wind_speed.reshape(wind_speed.shape[0], 1)
    
    time_array = convert_time_to_array(time)
    # if convert_6hour_to_1hour:#convert 6-hourly time to hourly time
    #     pass
    x = np.concatenate((time_array, temperature, dewpoint, relative_humidity, wind_speed), axis = 1)
    return x, time

def merge_envs_to_env(envs, coordinates, share):
    dict_env_coordinate = {} # {coordinate: share}
    env_coordinates = list(envs['Temperature'].keys())
    for i, coordinate in enumerate(coordinates):# Collect all used env coordinates
        env_coordinate_tmp = get_closest_coordinate(coordinate, env_coordinates)
        if env_coordinate_tmp in dict_env_coordinate:
            dict_env_coordinate[env_coordinate_tmp] += share[i]
        else:
            dict_env_coordinate[env_coordinate_tmp] = share[i]
    env = {}
    env['Time'] = envs['Time']
    list_keys = ['Temperature', 'DewPoint', 'RelativeHumidity', 'WindSpeed']
    for key in list_keys:
        env[key] = None
        for env_coordinate in dict_env_coordinate:
            if env[key] is None:
                env[key] = envs[key][env_coordinate] * dict_env_coordinate[env_coordinate]
            else:
                env[key] += envs[key][env_coordinate] * dict_env_coordinate[env_coordinate]
    return env


def load2y(
        time= None,
        load = None,
        convert_UTC_to_CT = False,
        convert_MW_to_GW = False,
    ):
    if isinstance(time[0], cftime.datetime):
        time = cfttime2datetime(time)
    y = load
    if convert_UTC_to_CT:
        time = utc2central(time)
    if convert_MW_to_GW:
        y = y / 1000
    if len(y.shape)==1:
        y = y.reshape(y.shape[0], 1)
    return y, time

def convert_time_to_array(time):
    """
    : param time : datetime
    : return array : [month_day, hour, weekday, holiday]
    """
    array = np.zeros((len(time), 4))
    for i, t in enumerate(time):
        array[i, 0] = t.month + t.day/31
        array[i, 1] = t.hour/24
        array[i, 2] = t.weekday()
        array[i, 3] = 1.0 if t.date() in holidays.US() else 0.0
    return array

def knots2mps(wind_speed):
    """Convert wind speed from knots to m/s"""
    return wind_speed * 0.514444

def F2C(temperature):
    """Convert temperature from Fahrenheit to Celsius"""
    return (temperature - 32) / 1.8

def percent2fraction(relative_humidity):
    """Convert relative humidity from percent to fraction"""
    return relative_humidity / 100