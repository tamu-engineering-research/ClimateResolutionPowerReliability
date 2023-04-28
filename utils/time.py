from datetime import datetime
import numpy as np
import pytz

def time2season(time_array):
    """Convert time to season"""
    season_array = np.zeros(time_array.shape)
    for i, time_tmp in enumerate(time_array):
        m = time_tmp.month
        if m in [12, 1, 2]:
            season_array[i] = 4
        elif m in [3, 4, 5]:
            season_array[i] = 1
        elif m in [6, 7, 8, 9]:
            season_array[i] = 2
        elif m in [10, 11]:
            season_array[i] = 3
    return season_array


def utc2central(time_array):#TODO
    """Convert UTC time to Central time"""
    utc_tz = pytz.timezone('UTC')
    central_tz = pytz.timezone('US/Central')
    time_array_converted = np.array([
        central_tz.normalize(dt_utc.astimezone(utc_tz))
        for dt_utc in map(utc_tz.localize, time_array)
    ])
    return time_array_converted

def central2utc(time_array):#TODO
    """Convert Central time to UTC time"""
    utc_tz = pytz.timezone('UTC')
    central_tz = pytz.timezone('US/Central')
    time_array_converted = np.array([
        utc_tz.normalize(dt_ct.astimezone(central_tz))
        for dt_ct in map(central_tz.localize, time_array)
    ])
    return time_array_converted

def cfttime2datetime(time):
    """Convert the time from cftime to datetime"""
    return np.array([datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in time])