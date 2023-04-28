import os
import sys

import numpy as np
import pandas as pd
import requests
import pickle
import json

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.getcwd()))
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import data_const

# data_root = data_const.ROOT_DIR
# county_boundary_path = os.path.join(data_root, 'CountyGeography', 'us-county-boundaries.json')
# with open(county_boundary_path, 'r') as f:
#     county_boundary = json.load(f)
#     input()

# Define a function to get the bounding box for a county
def get_county_bbox(state, county):
    root_dir = data_const.ROOT_DIR
    file_name = 'county_bbox_{state}_{county}.pkl'.format(state = state, county = county)
    file_path = os.path.join(root_dir, 'CountyGeography', file_name)
    if os.path.isfile(file_path):
        return _load_county_bbox(file_path)
    url = f'https://nominatim.openstreetmap.org/search.php?q={county}+county,+{state}&polygon_geojson=1&format=json'
    response = requests.get(url).json()
    bbox = response[0]['boundingbox']
    _save_county_bbox(file_path, bbox)
    return bbox

def get_county_bbox_offline(state, county):
    lat, lon = [], []
    return lat, lon

def _save_county_bbox(path, bbox):
    with open(path, 'wb') as f:
        pickle.dump(bbox, f)

def _load_county_bbox(path):
    with open(path, 'rb') as f:
        county_bbox = pickle.load(f)
    return county_bbox


# Get a coordinate (lat, lon) for a county
def county2coordinate(state, county, method = 'center'):
    if method == 'center':
        bbox = get_county_bbox(state, county)
        lat = (float(bbox[0]) + float(bbox[1])) / 2
        lon = (float(bbox[2]) + float(bbox[3])) / 2
        return (lat, lon) 
    raise ValueError('Invalid method')


def get_closest_coordinate(coordinate, list_coordinate):
    """Return the closest coordinate in list_coordinate to coordinate"""
    lat_target, lon_target = coordinate
    for lat_ll, lon_ll, lat_ur, lon_ur in list_coordinate:
        if lat_ll <= lat_target <= lat_ur and lon_ll <= lon_target <= lon_ur:
            return (lat_ll, lon_ll, lat_ur, lon_ur)
    dist_min = np.inf
    for lat_ll, lon_ll, lat_ur, lon_ur in list_coordinate:
        dist = np.sqrt((lat_target-lat_ll)**2 + (lon_target-lon_ll)**2)
        if dist < dist_min:
            dist_min = dist
            coordinate_min = (lat_ll, lon_ll, lat_ur, lon_ur)
    if dist_min == np.inf:
        raise ValueError("No coordinate found")
    return coordinate_min

def county2zone(county):#TODO
    return