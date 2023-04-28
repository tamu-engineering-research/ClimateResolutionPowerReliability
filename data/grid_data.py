import os
import sys

import cftime
from datetime import datetime
import numpy as np
import pandas as pd
import xarray

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.getcwd()))
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import data_const
from utils import grid_const

class GridData():
    def __init__(self, state : str, year : int, data_root : str):
        self.state = state
        self.year = year
        self.data_root = data_root
        self.bus_info = self.get_bus_info(os.path.join(data_root, 'ERCOT', 'Grid', 'bus_info.csv'))
        self.zones = self.get_zones()
        self.zone_share, self.zone_coordinate = self.get_zone_bus_info()

    def get_bus_info(self, bus_info_path):
        bus_info = pd.read_csv(bus_info_path)
        bus_info['Load MW'] = bus_info['Load MW'].fillna(0)
        return bus_info
    
    def get_zones(self):
        zones = grid_const.list_zones_full
        return zones

    def get_zone_bus_info(self):
        """Get the coordinates of buses in each zone"""
        zone_coordinate = {}
        zone_share = {}
        for zone in self.zones:
            zone_bus_info = self.bus_info[self.bus_info['Area Name'] == zone]
            if len(zone_bus_info.index) == 0:
                print("No bus in zone {zone}".format(zone=zone))
            else:
                zone_coordinate[zone] = [] 
                zone_share[zone] = []
                zone_P_sum = 0
                for index, row in zone_bus_info.iterrows():
                    zone_coordinate[zone].append((row['Substation Latitude'], row['Substation Longitude']))
                    zone_share[zone].append(row['Load MW'])
                    zone_P_sum += row['Load MW']
                zone_share[zone] = np.array(zone_share[zone]) / zone_P_sum
        return zone_share, zone_coordinate
    
    def get_zone_counties(self, zone):#TODO
        return
    
if __name__=="__main__":
    griddata = GridData('TX', 2019, data_const.ROOT_DIR)
    input()