import os
import json
from datetime import datetime
import pandas as pd
import numpy as np


from data_objects.cow_data import CowData
from data_objects.feed_history_data import FeedHistoryData
from data_objects.weight_history_data import WeightHistoryData

from consts.consts import tdn_table, costs_per_dm, sales_price
from data_processor.FeedProcessor import FeedProcessor 

class DataProcessing:
    """
    A class for processing dairy cow data.
    
    This class loads JSON data files, cleans number formats in dictionaries,
    and casts the raw data into corresponding object types.
    """
    def __init__(self, main_folder='./data', analysis_folder='./analysis-dec-2024/'):
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

        self.objects = None
        self.dfs = {}

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

    def cast_to_obj(self, cows, weight_histories, feed_histories):
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

        self.objects = self.cast_to_obj(cows, weight_histories, feed_histories )
        return self.objects 
     
    def _process_single_window(self, cow_data, weight_history, feed_history, x, n_weighing):
        """
        Processes a single non-overlapping window for a cow.
        
        Args:
            cow_data: CowData object
            weight_history: WeightHistoryData object
            feed_history: FeedHistoryData object
            x (int): Starting index in weight history
            n_weighing (int): Number of weighings in this window
            
        Returns:
            dict or None: Dictionary of features, or None if window should be skipped
        """
        entry = weight_history.data[x]
        ret_dict = {}
        
        # ===== SUPER PRIMITIVES =====
        target_weighing = x + n_weighing
        ret_dict['pred_date'] = weight_history.data[target_weighing]['date']
        ret_dict['date'] = entry['date']
        
        ret_dict['day_diff'] = (datetime.strptime(ret_dict['pred_date'], "%Y-%m-%d") - 
                               datetime.strptime(ret_dict['date'], "%Y-%m-%d")).days
        ret_dict['day_diff_2'] = ret_dict['day_diff']**2
        
        ret_dict['weight'] = entry['weight']
        ret_dict['cattleId'] = cow_data.cattleId
        ret_dict['originWeight'] = cow_data.originWeight
        ret_dict['breed'] = cow_data.breed
        
        # Breed indicators
        ret_dict['isLimousine'] = (ret_dict['breed'] == 'Limousin')
        ret_dict['isSimental'] = (ret_dict['breed'] == 'Simental')
        
        if ret_dict['breed'] not in ['Limousin', 'Simental']:
            ret_dict['breed'] = 'Other'
        
        ret_dict['entryWeight'] = cow_data.entryWeight
        
        # ===== PROCESS FEED DATA =====
        feed_processor = FeedProcessor(feed_history, weight_history, x, n_weighing)
        
        # Skip if required feeds not present
        if not feed_processor.has_required_feeds:
            return None
        
        # Get all feed features
        feed_features = feed_processor.get_all_features()
        ret_dict.update(feed_features)
        
        # ===== TARGET BASICS =====
        ret_dict['pred_weight'] = weight_history.data[target_weighing]['weight']
        ret_dict['pred_weight_gain'] = ret_dict['pred_weight'] - ret_dict['weight']
        ret_dict['pred_adgLatest_average'] = ret_dict['pred_weight_gain'] / ret_dict['day_diff']
        ret_dict['pred_adgLatest_average_inverse_hyperbolic'] = (
            np.log(ret_dict['pred_adgLatest_average'] + 
                   (ret_dict['pred_adgLatest_average']**2 + 1)**0.5) * 0.5
        )
        ret_dict['pred_fcrLatest_average'] = (
            (ret_dict['pred_weight_gain'] / ret_dict['total_dm_intake']) * 100
        )
        
        # ===== PRIMITIVES =====
        ret_dict['metabolic_weight'] = ret_dict['weight']**0.75
        ret_dict['pred_adgLatest_average_mw'] = (
            (ret_dict['pred_weight_gain'] / ret_dict['day_diff']) * ret_dict['metabolic_weight']
        )
        ret_dict['originWeight_mw'] = ret_dict['originWeight'] * ret_dict['metabolic_weight']
        
        # Breed-specific metabolic weights
        for breed_name in ['Simental', 'Limousin', 'Other']:
            ret_dict[f'metabolic_weight_{breed_name}'] = 0
            ret_dict[f'metabolic_weight_{breed_name}_mw'] = 0
        
        ret_dict[f'metabolic_weight_{ret_dict["breed"]}'] = ret_dict['metabolic_weight']
        ret_dict[f'metabolic_weight_{ret_dict["breed"]}_mw'] = ret_dict['metabolic_weight']**2
        
        # Days on feed
        ret_dict['daysOnFeedNow'] = (
            datetime.strptime(ret_dict['date'], "%Y-%m-%d") - 
            datetime.strptime(cow_data.entryDate, "%Y-%m-%d")
        ).days
        ret_dict['daysOnFeedNow_2'] = ret_dict['daysOnFeedNow']**2
        ret_dict['daysOnFeed_then'] = ret_dict['daysOnFeedNow'] + ret_dict['day_diff']
        
        ret_dict['originWeight_ddmi'] = ret_dict['originWeight'] / ret_dict['avg_dm_intake_per_day']
        
        # Breed-specific with ddmi
        for breed_name in ['Simental', 'Limousin', 'Other']:
            ret_dict[f'metabolic_weight_{breed_name}_ddmi'] = (
                ret_dict[f'metabolic_weight_{breed_name}'] / ret_dict['avg_dm_intake_per_day']
            )
        
        # Filter by minimum days on feed and breed
        if ret_dict['daysOnFeedNow'] < 10 or ret_dict['breed'] == 'Other':
            return None
        
        return ret_dict


    def get_variables(self, n_weighing):
        """
        Processes cow data objects to extract features for modeling.
        Uses non-overlapping windows to ensure statistical independence.
        
        Args:
            n_weighing (int): Number of weighings ahead to predict
        
        Returns:
            list: List of dictionaries containing features for each observation
        """
        if self.objects is None:
            self.get_data()

        ret_arr = []

        for cow_id, cow_dict in self.objects.items():
            cow_data = cow_dict['cow_data']
            weight_history = cow_dict['weight_history_data']
            feed_history = cow_dict['feed_history_data']
            
            # Skip if no feed history
            if feed_history is None:
                continue
            
            # Iterate through weight history using non-overlapping windows
            for x in range(0, len(weight_history.data) - n_weighing, n_weighing):
                window_data = self._process_single_window(
                    cow_data, weight_history, feed_history, x, n_weighing
                )
                
                if window_data is not None:
                    ret_arr.append(window_data)
        
        return ret_arr


    def get_dfs(self, n_weighings: list):
        """
        Generate DataFrames for multiple n_weighing values.
        
        Args:
            n_weighings (list): List of integers representing different weighing intervals
            
        Returns:
            dict: Dictionary mapping n_weighing values to their corresponding DataFrames
        """
        for n in n_weighings:
            arr = self.get_variables(n)
            
            df = pd.DataFrame(arr)
            
            self.dfs[n] = df
        
        return self.dfs
