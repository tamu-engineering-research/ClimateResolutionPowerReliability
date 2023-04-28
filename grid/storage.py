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

from utils import solar_const
from utils import data_const
from utils import county2coordinate
from utils import get_closest_coordinate
from utils import time2season
from data import ClimateData, WeatherData


#TODO add storage models