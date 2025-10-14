import os
import json
from datetime import datetime
import pandas as pd
import numpy as np


from data_objects.cow_data import CowData
from data_objects.feed_history_data import FeedHistoryData
from data_objects.weight_history_data import WeightHistoryData
from utils.data_utils import postprocess_orthogonalize

from consts.consts import tdn_table, costs_per_dm, sales_price
from data_processor.FeedProcessor import FeedProcessor 
from data_processor.KalmanSmoother import KalmanSmoother


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

    def apply_kalman_smoothing(self, measurement_noise=400, process_noise=None):
        """
        Apply Kalman smoothing to weight data for all cows.
        This should be called after get_data() but before get_variables().
        
        Args:
            measurement_noise: Expected measurement error variance (default: 400 = 20^2)
            process_noise: Process noise variance (None = auto-estimate)
        
        Returns:
            dict: Smoothed weight data added to self.objects
        """
        if self.objects is None:
            raise ValueError("Must call get_data() first")
        
        print("Applying Kalman smoothing to cow weights...")
       
        fix_measurement_noise = False 
        use_trend = True

        # Create smoother
        smoother = KalmanSmoother(
            measurement_noise=measurement_noise,
            process_noise=process_noise,
            fix_measurement_noise=fix_measurement_noise,
            use_trend=use_trend 
        )

        
        # Prepare data for smoothing
        weight_records = []
        for cow_id, cow_dict in self.objects.items():
            weight_history = cow_dict['weight_history_data']
            
            for idx, entry in enumerate(weight_history.data):
                weight_records.append({
                    'cow_id': cow_id,
                    'date': entry['date'],
                    'weight': entry['weight'],
                    'index': idx  # Keep track of original index
                })
        
        # Create DataFrame
        weight_df = pd.DataFrame(weight_records)
        weight_df['date'] = pd.to_datetime(weight_df['date'])
        
        # Apply smoothing
        smoothed_df = smoother.filter(
            df=weight_df,
            target_attr='weight',
            group_attr='cow_id',
            time_attr='date'
        )
        
        # Print summary
        smoother.print_summary()
        
        # Add smoothed weights back to objects
        for cow_id, cow_dict in self.objects.items():
            cow_smoothed = smoothed_df[smoothed_df['cow_id'] == cow_id].copy()
            cow_smoothed = cow_smoothed.sort_values('date').reset_index(drop=True)
            
            # Add smoothed values to weight history
            weight_history = cow_dict['weight_history_data']
            for i, entry in enumerate(weight_history.data):
                if i < len(cow_smoothed):
                    entry['weight_smoothed'] = cow_smoothed.iloc[i]['weight_smoothed']
                    entry['weight_smoothed_se'] = cow_smoothed.iloc[i]['weight_smoothed_se']
                    entry['weight_filtered'] = cow_smoothed.iloc[i]['weight_filtered']
        
        print("Kalman smoothing complete! Added 'weight_smoothed' to weight history data.")
        return self.objects
         
    def _process_single_window(self, cow_data, weight_history, feed_history, x, n_weighing, use_smoothed=True):
        """
        Processes a single non-overlapping window for a cow.
        
        Args:
            cow_data: CowData object
            weight_history: WeightHistoryData object
            feed_history: FeedHistoryData object
            x (int): Starting index in weight history
            n_weighing (int): Number of weighings in this window
            use_smoothed (bool): If True, use smoothed weights if available
            
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
        ret_dict['day_diff_recp'] = ret_dict['day_diff']**2
      
        ret_dict['theoritical_error_adg'] = 20/ret_dict['day_diff']

        # USE SMOOTHED WEIGHT IF AVAILABLE
        if use_smoothed and 'weight_smoothed' in entry:
            ret_dict['weight'] = entry['weight_smoothed']
            ret_dict['weight_raw'] = entry['weight']  # Keep original
        else:
            ret_dict['weight'] = entry['weight']
        
        ret_dict['cattleId'] = cow_data.cattleId
        ret_dict['originWeight'] = cow_data.originWeight
        ret_dict['hipHeight'] = cow_data.hipHeight
        ret_dict['breed'] = cow_data.breed

        #if ret_dict['breed'] == 'Limousin X':
        #    ret_dict['breed'] = 'Limousin'

       
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
        # USE SMOOTHED PRED_WEIGHT IF AVAILABLE
        target_entry = weight_history.data[target_weighing]
        if use_smoothed and 'weight_smoothed' in target_entry:
            ret_dict['pred_weight'] = target_entry['weight_smoothed']
            ret_dict['pred_weight_raw'] = target_entry['weight']
        else:
            ret_dict['pred_weight'] = target_entry['weight']
        
        # NOW ALL THESE CALCULATIONS USE SMOOTHED WEIGHTS
        ret_dict['pred_weight_gain'] = ret_dict['pred_weight'] - ret_dict['weight']
       
        if use_smoothed:
            ret_dict['pred_weight_gain_raw'] = ret_dict['pred_weight_raw'] - ret_dict['weight_raw']
        ret_dict['pred_adgLatest_average'] = ret_dict['pred_weight_gain'] / ret_dict['day_diff']
        ret_dict['pred_adgLatest_average_inverse_hyperbolic'] = (
            np.log(ret_dict['pred_adgLatest_average'] + 
                   (ret_dict['pred_adgLatest_average']**2 + 1)**0.5) * 0.5
        )
        ret_dict['pred_fcrLatest_average'] = (
            (ret_dict['pred_weight_gain'] / ret_dict['total_dm_intake']) * 100
        )
        
        # ===== PRIMITIVES (now using smoothed weight) =====
        ret_dict['metabolic_weight'] = (ret_dict['weight']*0.96)**0.75
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
        
        ret_dict['tdn_slobber_daysonfeed'] = ret_dict['tdn_slobber_over_mw_dt']*ret_dict['daysOnFeedNow']

        ret_dict['originWeight_ddmi'] = ret_dict['originWeight'] / ret_dict['avg_dm_intake_per_day']
        
        # Breed-specific with ddmi
        for breed_name in ['Simental', 'Limousin', 'Other']:
            ret_dict[f'metabolic_weight_{breed_name}_ddmi'] = (
                ret_dict[f'metabolic_weight_{breed_name}'] / ret_dict['avg_dm_intake_per_day']
            )
        

        ret_dict['mw_per_ddmi'] = ret_dict['metabolic_weight']/ret_dict['avg_dm_intake_per_day']
        ret_dict['mw_dmi_dt'] = (ret_dict['metabolic_weight']*ret_dict['total_dmi'])/ret_dict['day_diff']

        ret_dict['day_diff_2_dmi'] = ret_dict['day_diff_2'] * ret_dict['total_dmi']
        ret_dict['day_diff_dmi'] = ret_dict['day_diff'] * ret_dict['total_dmi']

        ret_dict['mw_dmi'] = (ret_dict['metabolic_weight']*ret_dict['total_dmi'])
        if ret_dict['breed'] == 'Other':
            return None

        return ret_dict

    def get_variables(self, n_weighing, use_smoothed=True):
        """
        Processes cow data objects to extract features for modeling.
        Uses non-overlapping windows to ensure statistical independence.
        
        Args:
            n_weighing (int): Number of weighings ahead to predict
            use_smoothed (bool): If True, use smoothed weights from apply_kalman_smoothing()
        
        Returns:
            list: List of dictionaries containing features for each observation
        """
        if self.objects is None:
            self.get_data()

        ret_arr = []

        total_limo = 0
        total_sim = 0
        for cow_id, cow_dict in self.objects.items():
            cow_data = cow_dict['cow_data']
            weight_history = cow_dict['weight_history_data']
            feed_history = cow_dict['feed_history_data']
            
            # Skip if no feed history
            if feed_history is None:
                continue
           
            last_window = None
            time = 0
            
            print(f"cow_id: {cow_id}, breed: {cow_data.breed}")


            for x in range(3, len(weight_history.data) - n_weighing, n_weighing):


                window_data = self._process_single_window(
                    cow_data, weight_history, feed_history, x, n_weighing, 
                    use_smoothed=use_smoothed
                )
              
                if window_data is None:
                    continue
                
                window_data['cow_id'] = cow_data.cattleId 
                window_data['time'] = time
                time += 1
                ret_arr.append(window_data)
            
                last_window = window_data

        return ret_arr

    def get_dfs(self, n_weighings: list, measurement_noise=400, apply_smoothing=True):
        """
        Generate dataframes with optional Kalman smoothing applied BEFORE feature engineering.
        
        Args:
            n_weighings: List of prediction horizons
            measurement_noise: Expected measurement error variance
            apply_smoothing: If True, applies Kalman smoothing to raw weights first
        """
        # STEP 0: Load data first if not already loaded
        if self.objects is None:
            self.get_data()

        # STEP 1: Apply smoothing to raw weight data if requested
        if apply_smoothing:
            print("\n=== Applying Kalman smoothing to raw weight data ===")
            self.apply_kalman_smoothing(measurement_noise=measurement_noise)
        
        # STEP 2: Generate features (will use smoothed weights if available)
        for n in n_weighings:
            print(f"\n=== Generating features for n={n} ===")
            arr = self.get_variables(n, use_smoothed=apply_smoothing)
            df = pd.DataFrame(arr)

            df['pred_date'] = pd.to_datetime(df['pred_date'])

            self.dfs[n] = df
        
        return self.dfs
