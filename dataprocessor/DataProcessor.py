import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import linregress
from decimal import Decimal
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

import statsmodels.api as sm
import statsmodels.formula.api as smf

from data_objects.cow_data import CowData
from data_objects.feed_history_data import FeedHistoryData
from data_objects.weight_history_data import WeightHistoryData


class DataProcessing:
    """
    A class for processing dairy cow data.
    
    This class loads JSON data files, cleans number formats in dictionaries,
    and casts the raw data into corresponding object types.
    """
    def __init__(self, main_folder='./actual-data', analysis_folder='./analysis-dec-2024/'):
        # Folders
        self.main_folder = main_folder
        self.analysis_folder = analysis_folder
        
        # JSON file names
        self.cow_weight_history_data = 'KC51sJ30yRPUgRKZsvoI-cowweighthistory.json'
        self.cow_feed_history_data = 'KC51sJ30yRPUgRKZsvoI-feedhistory.json'
        self.cows_data = 'KC51sJ30yRPUgRKZsvoI-cows.json'
        self.historic_cows_data = 'KC51sJ30yRPUgRKZsvoI-historic-cows.json'
        
        # Other constants
        self.date_format = '%Y-%m-%d'

    def load_json_data(self, file_name, folder=None):
        """
        Loads JSON data from a specified file.

        Args:
            file_name (str): Name of the JSON file.
            folder (str, optional): Folder path where the file is located. Defaults to main_folder.

        Returns:
            dict: Parsed JSON data as a dictionary, or an empty dictionary on failure.
        """
        if folder is None:
            folder = self.main_folder
        file_path = os.path.join(folder, file_name)
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File {file_name} not found in {folder}.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from the file {file_name}.")
            return {}

    def fix_numbers_dic_of_dic(self, dic_of_dicts):
        """
        Recursively processes a dictionary (or dictionary of dictionaries/lists) to convert
        number-like strings into rounded floats, and converts specific string literals to their
        corresponding types (None, True, False).

        Args:
            dic_of_dicts (dict): The dictionary to process.

        Returns:
            dict: The processed dictionary with fixed number formats.
        """
        for key in dic_of_dicts:
            # If the value is a dictionary, process it recursively.
            if isinstance(dic_of_dicts[key], dict):
                dic_of_dicts[key] = self.fix_numbers_dic_of_dic(dic_of_dicts[key])

            # If the value is a list, process each element.
            if isinstance(dic_of_dicts[key], list):
                for x in range(len(dic_of_dicts[key])):
                    if isinstance(dic_of_dicts[key][x], dict):
                        dic_of_dicts[key][x] = self.fix_numbers_dic_of_dic(dic_of_dicts[key][x])
                    else:
                        try:
                            dic_of_dicts[key][x] = round(float(dic_of_dicts[key][x]), 2)
                        except Exception:
                            if dic_of_dicts[key][x] == 'None':
                                dic_of_dicts[key][x] = None
                            elif dic_of_dicts[key][x] == 'True':
                                dic_of_dicts[key][x] = True
                            elif dic_of_dicts[key][x] == 'False':
                                dic_of_dicts[key][x] = False

            # Attempt to convert the value itself
            try:
                dic_of_dicts[key] = round(float(dic_of_dicts[key]), 2)
            except Exception:
                if dic_of_dicts[key] == 'None':
                    dic_of_dicts[key] = None
                elif dic_of_dicts[key] == 'True':
                    dic_of_dicts[key] = True
                elif dic_of_dicts[key] == 'False':
                    dic_of_dicts[key] = False
        return dic_of_dicts

    def cast_to_obj(self, cows, weight_histories, feed_histories, medical_histories):
        """
        Casts the raw dictionary data into specific data objects for each cow.

        Args:
            cows (dict): Dictionary containing cows data.
            weight_histories (dict): Dictionary containing weight history data.
            feed_histories (dict): Dictionary containing feed history data.
            medical_histories (dict): Dictionary containing medical history data.

        Returns:
            dict: A dictionary of cow objects with their associated data objects.
        """
        ret_dict = {}
        for cow_id in cows:
            # Skip a specific cow by its ID if necessary.
            if cow_id != "rexFmUY8QHCvB0TsjnbB":
                ret_dict[cow_id] = {}
                ret_dict[cow_id]['cow_data'] = CowData(cows[cow_id])
                ret_dict[cow_id]['weight_history_data'] = WeightHistoryData(weight_histories[cow_id])
                ret_dict[cow_id]['feed_history_data'] = FeedHistoryData(feed_histories[cow_id]) if cow_id in feed_histories else None
        return ret_dict

    def get_data(self):
        """
        Loads and processes the JSON data files, fixes number formats, and casts the data
        into corresponding objects.

        Returns:
            dict: Dictionary of cow data objects with their associated histories.
        """
        cows = self.load_json_data(self.cows_data)
        weight_histories = self.load_json_data(self.cow_weight_history_data)
        feed_histories = self.load_json_data(self.cow_feed_history_data)
        historic_cows = self.load_json_data(self.historic_cows_data)

        cows = cows | historic_cows

        # Optionally clean the data if needed.
        cows = self.fix_numbers_dic_of_dic(cows)
        weight_histories = self.fix_numbers_dic_of_dic(weight_histories)
        feed_histories = self.fix_numbers_dic_of_dic(feed_histories)

        return self.cast_to_obj(cows, weight_histories, feed_histories, medical_histories)
