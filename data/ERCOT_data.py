import os
import sys

from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import pickle
import zipfile
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.getcwd()))
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import data_const
from utils.time import utc2central, central2utc

class ERCOTData():
    def __init__(self, dir : str):
        self.dir = dir
    
    def get_load_data(self, start_time = None, end_time = None, zone = None):
        if start_time is None:
            start_time = datetime(2018, 1, 1, 0)
        if end_time is None:
            end_time = datetime(2018, 12, 31, 23)
        start_year = start_time.year
        end_year = end_time.year
        for year in range(start_year, end_year + 1):
            data_path = os.path.join(self.dir, 'Native_Load_{year}.xlsx'.format(year=year))
            df_load_tmp = get_load(data_path)
            if year == start_year:
                df_load_tmp = df_load_tmp[start_time:]
            if year == end_year:
                df_load_tmp = df_load_tmp[:end_time]
            if zone is not None:
                if isinstance(zone, str):
                    zone = [zone]
                df_load_tmp = df_load_tmp[zone]
            if year == start_year:
                df_load = df_load_tmp
            else:
                df_load = pd.concat([df_load, df_load_tmp])
        load = {}
        load['Time'] = pd.date_range(start_time, end_time, freq = 'H')
        df_load = df_load[~df_load.index.duplicated(keep='first')]#remove duplicated index
        df_load = df_load.reindex(load['Time']).fillna(np.nan)
        load['Load'] = {}
        for zone in df_load.columns:
            load['Load'][zone] = df_load[zone].values
        return load
    
    def get_monthly_peak_load(self, year, month, zone = None):
        if zone is None:
            zone = 'ERCOT'
        if isinstance(month, int):
            month = [month]
        data_path = os.path.join(self.dir, 'Native_Load_{year}.xlsx'.format(year=year))
        df_load = get_load(data_path)
        monthly_peak_load = {}
        for m in month:
            start_time = datetime(year, m, 1, 0)
            end_time = datetime(year, m, 1, 0) + relativedelta(months=1)
            df_load_moth = df_load[start_time:end_time]
            monthly_peak_load[m] = df_load_moth[zone].max()
        return monthly_peak_load
    
    def get_yearly_energy(self, year, zone =None):
        data_path = os.path.join(self.dir, 'Native_Load_{year}.xlsx'.format(year=year))
        df_load_tmp = get_load(data_path)
        return
    
    def get_yearly_planned_outage(self, dir, year, start_time = None, end_time = None,):
        if start_time is None:
            start_time = datetime(2019, 1, 1, 0)
        if end_time is None:
            end_time = datetime(2019, 12, 31, 23)
        folder = os.path.join(
            dir,
            'Hourly Resource Outage Capacity_{year}'.format(year=year)
        )
        file_path = os.path.join(
            dir, 'planned_outage_{year}.pkl'.format(year=year)
        )
        df_planned_outage = get_planned_outage(folder, file_path)
        planned_outage = {}
        planned_outage['Time'] = pd.date_range(start_time, end_time, freq = 'H')
        df_planned_outage = df_planned_outage.reindex(planned_outage['Time']).fillna(np.nan)
        df_planned_outage = df_planned_outage.interpolate(method='linear', limit_direction='both')
        for columns in df_planned_outage.columns:
            planned_outage[columns] = df_planned_outage[columns].values
        return planned_outage

def get_load(path:str):
    """
    Issues of ERCOT load data:
    1. The time zone is Central Time (CT)
    2. It has 1:00-24:00, rather than 0:00-23:00
    """
    def _convert_time(str_time):
        df_datetime = pd.DataFrame(columns=['str_time','date', 'time', 'datetime', 'nan'])
        df_datetime['str_time'] = str_time
        df_datetime[['date','time', 'nan']] = df_datetime['str_time'].str.split(expand=True)
        df_datetime['datetime'] = (pd.to_datetime(df_datetime.pop('date'), format='%m/%d/%Y') + 
                  pd.to_timedelta(df_datetime.pop('time') + ':00'))# convert 1-24 to 0-23
        return df_datetime['datetime']
    try:
        load_data = pd.read_excel(
            path, sheet_name='Native Load Report', header=0, index_col=0, parse_dates=True
        )
    except:
        load_data = pd.read_excel(
            path, sheet_name='Sheet1', header=0, index_col=0, parse_dates=True
        )
    load_data.index = _convert_time(load_data.index)
    return load_data

def get_planned_outage(folder:str, file_path:str):
    def _convert_time(str_time):
        df_datetime = pd.DataFrame(columns=['str_time','date', 'time', 'datetime', 'nan'])
        df_datetime['str_time'] = str_time
        df_datetime[['date','time', 'nan']] = df_datetime['str_time'].str.split(expand=True)
        df_datetime['datetime'] = (pd.to_datetime(df_datetime.pop('date'), format='%m/%d/%Y') + 
                  pd.to_timedelta(df_datetime.pop('time') + ':00'))# convert 1-24 to 0-23
        return df_datetime['datetime']
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    planned_outage_data = pd.DataFrame(columns = ['Renewable','Non_Renewable'])
    for root, dir, files in os.walk(folder):
        for filename in files:
            if filename.endswith('_csv.zip'):
                csv_file = filename[:-8] + '.csv'
                with zipfile.ZipFile(os.path.join(root, filename)) as z:
                    for csv_file in z.namelist():
                        with z.open(csv_file) as f:
                            df = pd.read_csv(f, header=0, index_col=0, parse_dates=True)
                            deltatime = '{h}:00:00'.format(h=df.iloc[0]['HourEnding'])
                            index = df.index[0]+pd.to_timedelta(deltatime.zfill(8))
                            renewable = df.iloc[0]['TotalIRRMW']
                            non_renewable = df.iloc[0]['TotalResourceMW']
                            planned_outage_data.loc[index] = [renewable, non_renewable] 
    with open(file_path, 'wb') as f:
        pickle.dump(planned_outage_data, f)
    return planned_outage_data

if __name__=="__main__":
    # ercotdata = ERCOTData('TX', 2019, data_const.ROOT_DIR)
    # load = ercotdata.get_load_data()
    
    ercotdata = ERCOTData( data_const.ROOT_DIR)
    planned_outage = ercotdata.get_yearly_planned_outage(
        dir = os.path.join(data_const.ROOT_DIR, 'ERCOT', 'PlannedUnitOutage'),
        year=2019
    )
    
    
    
    # folder_plannedoutage= os.path.join(data_const.ROOT_DIR, 'ERCOT', 'PlannedUnitOutage', 'Hourly Resource Outage Capacity_2019')
    # get_planned_outage(
    #     folder_plannedoutage,
    #     os.path.join(data_const.ROOT_DIR, 'ERCOT', 'PlannedUnitOutage', 'planned_outage_2019.pkl')
    # )
    input()