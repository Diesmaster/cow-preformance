import os
import json
import math
from datetime import datetime
import pandas as pd
import numpy as np

from data_objects.cow_data import CowData
from data_objects.feed_history_data import FeedHistoryData
from data_objects.medical_history_data import MedicalHistoryData
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
        self.medical_history_data = 'KC51sJ30yRPUgRKZsvoI-medicalHistory.json'
        
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
                ret_dict[cow_id]['medical_history_data'] = MedicalHistoryData(medical_histories[cow_id]) if cow_id in medical_histories else None
        return ret_dict

    def get_data(self):
        """
        Loads and processes the JSON data files, fixes number formats, and casts the data
        into corresponding objects.

        Returns:
            dict: Dictionary of cow data objects with their associated histories.
        """
        cows = self.load_json_data(self.cows_data)
        print(f"cows: {len(cows)}")
        weight_histories = self.load_json_data(self.cow_weight_history_data)
        feed_histories = self.load_json_data(self.cow_feed_history_data)
        historic_cows = self.load_json_data(self.historic_cows_data)
        print(f"his cows: {len(historic_cows)}")
        medical_histories = self.load_json_data(self.medical_history_data)

        cows = cows | historic_cows

        # Optionally clean the data if needed.
        cows = self.fix_numbers_dic_of_dic(cows)
        weight_histories = self.fix_numbers_dic_of_dic(weight_histories)
        feed_histories = self.fix_numbers_dic_of_dic(feed_histories)
        medical_histories = self.fix_numbers_dic_of_dic(medical_histories)

        self.objects = self.cast_to_obj(cows, weight_histories, feed_histories, medical_histories)
        return self.objects 

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
            medical_history = cow_dict['medical_history_data']
           
            #print(medical_history)

            # Skip if no feed history
            if feed_history is None:
                continue
           
            last_window = None
            time = 0
            
            print(f"cow_id: {cow_data.cattleId}, breed: {cow_data.breed}")


            n_start = 1

            for x in range(n_start, len(weight_history.data) - n_weighing, n_weighing):


                window_data = self._process_single_window(
                    cow_data, weight_history, feed_history, medical_history, x, n_weighing, 
                    use_smoothed=use_smoothed,
                    n_start=n_start
                )
             

                if window_data is None:
                    continue
                
                window_data['tdn_silage_over_mw_daysinfeedlot_dt'] = window_data['tdn_silage_dt']/(window_data['metabolic_weight'])
                window_data['tdn_rumput_over_mw_daysinfeedlot_dt'] = window_data['tdn_rumput_dt']/(window_data['metabolic_weight'])
                window_data['tdn_slobber_over_mw_daysinfeedlot_dt'] =window_data['tdn_slobber_dt']/(window_data['metabolic_weight'])
                window_data['tdn_SP2A_over_mw_daysinfeedlot_dt'] =window_data['tdn_SP2A_dt']/(window_data['metabolic_weight'])
                window_data['tdn_ricehay_over_mw_daysinfeedlot_dt'] =window_data['tdn_ricehay_dt']/(window_data['metabolic_weight'])
               
                window_data['tdn_silage_dt_ratio'] = window_data['tdn_silage_dt']/window_data['exp_increase_ratio']
                window_data['tdn_rumput_dt_ratio'] = window_data['tdn_rumput_dt']/window_data['exp_increase_ratio']
                window_data['tdn_slobber_dt_ratio'] =window_data['tdn_slobber_dt']/window_data['exp_increase_ratio']
                window_data['tdn_SP2A_dt_ratio'] =window_data['tdn_SP2A_dt']/window_data['exp_increase_ratio']
                window_data['tdn_ricehay_dt_ratio'] =window_data['tdn_ricehay_dt']/window_data['exp_increase_ratio']
                window_data['cow_id'] = cow_data.cattleId 
                window_data['time'] = time
                time += 1
                ret_arr.append(window_data)
           

                last_window = window_data
            print(len(ret_arr))

        return ret_arr

    def signed_log_transform(self, x):
        """
        Apply a sign-preserving logarithmic transformation.
        For positive values: ln(x + 1)
        For negative values: -ln(|x| + 1)
        For zero: 0
        """
        return np.sign(x) * np.log1p(np.abs(x))

    def apply_kalman_smoothing(self, measurement_noise=None, process_noise_per_day=None, 
                              estimate_drift=True, fixed_drift=None, auto_tune=True):
        """
        Apply Kalman smoothing to weight data with proper time-varying dynamics.
        
        This uses a Brownian motion with drift model:
            X_t = X_{t-1} + μ·Δt + ε_t    where Var(ε_t) = σ²·Δt
            Y_t = X_t + η_t                where Var(η_t) = R (constant)
        
        Args:
            measurement_noise: Measurement error variance. If None and auto_tune=True, 
                              will be estimated from data. Default: None
            
            process_noise_per_day: Process noise variance per day. If None and auto_tune=True,
                                  will be estimated from data. Default: None
            
            estimate_drift: If True, estimates growth rate per cow via linear regression.
                           If False, uses fixed_drift parameter. Default: True
            
            fixed_drift: Fixed drift rate in kg/day for all cows (only used if estimate_drift=False).
                        Example: fixed_drift=1.5 means all cows grow at 1.5 kg/day
            
            auto_tune: If True, automatically estimates noise parameters via Maximum Likelihood.
                      Parameters set to None will be estimated. Default: True
        
        Returns:
            dict: self.objects with smoothed weights added to weight history
        
        Examples:
            # Full auto-tuning (recommended)
            processor.apply_kalman_smoothing(auto_tune=True)
            
            # Fix measurement noise, estimate process noise
            processor.apply_kalman_smoothing(measurement_noise=400, auto_tune=True)
            
            # Manual parameters (no auto-tuning)
            processor.apply_kalman_smoothing(
                measurement_noise=400,
                process_noise_per_day=2.0,
                auto_tune=False
            )
        """
        if self.objects is None:
            raise ValueError("Must call get_data() first")
        
        # Determine what mode we're in
        is_auto_tuning = auto_tune and (measurement_noise is None or process_noise_per_day is None)
        
        # Set defaults only if NOT auto-tuning and values are None
        if not auto_tune:
            if measurement_noise is None:
                measurement_noise = 400.0
                print("⚠️  Using default measurement_noise = 400 (auto_tune=False)")
            if process_noise_per_day is None:
                process_noise_per_day = 2.0
                print("⚠️  Using default process_noise_per_day = 2.0 (auto_tune=False)")
        
        # Validate non-None parameters
        if measurement_noise is not None:
            if not isinstance(measurement_noise, (int, float)) or measurement_noise <= 0:
                raise ValueError(f"measurement_noise must be a positive number, got {measurement_noise}")
        
        if process_noise_per_day is not None:
            if not isinstance(process_noise_per_day, (int, float)) or process_noise_per_day <= 0:
                raise ValueError(f"process_noise_per_day must be a positive number, got {process_noise_per_day}")
        
        if not isinstance(estimate_drift, bool):
            raise ValueError(f"estimate_drift must be a boolean, got {estimate_drift}")
        
        if fixed_drift is not None and not isinstance(fixed_drift, (int, float)):
            raise ValueError(f"fixed_drift must be a number or None, got {fixed_drift}")
        
        # Print header
        print("\n" + "="*80)
        if is_auto_tuning:
            print("APPLYING KALMAN SMOOTHING WITH AUTO-TUNED PARAMETERS")
            print("="*80)
            if measurement_noise is not None:
                print(f"Fixed measurement_noise: {measurement_noise} (±{np.sqrt(measurement_noise):.1f} kg)")
            else:
                print("Estimating measurement_noise from data...")
            if process_noise_per_day is not None:
                print(f"Fixed process_noise_per_day: {process_noise_per_day} (±{np.sqrt(process_noise_per_day):.2f} kg/day)")
            else:
                print("Estimating process_noise_per_day from data...")
        else:
            print("APPLYING KALMAN SMOOTHING WITH USER-SPECIFIED PARAMETERS")
            print("="*80)
            print(f"Measurement noise: {measurement_noise} (±{np.sqrt(measurement_noise):.1f} kg CONSTANT)")
            print(f"Process noise per day: {process_noise_per_day} (±{np.sqrt(process_noise_per_day):.2f} kg/√day)")
        
        print(f"Drift estimation: {'Per-cow linear regression' if estimate_drift else f'Fixed at {fixed_drift} kg/day'}")
        
        if not is_auto_tuning:
            print("\nProcess noise scales with time interval:")
            for days in [7, 14, 28]:
                variance = process_noise_per_day * days
                std = np.sqrt(variance)
                print(f"  {days:2d} days: variance = {variance:6.1f}, std = ±{std:5.2f} kg")
        
        print("="*80 + "\n")
        
        # Prepare data for smoothing
        weight_records = []
        for cow_id, cow_dict in self.objects.items():
            weight_history = cow_dict['weight_history_data']
            
            for idx, entry in enumerate(weight_history.data):
                weight_records.append({
                    'cow_id': cow_id,
                    'date': entry['date'],
                    'weight': entry['weight'],
                    'index': idx
                })
        
        # Create DataFrame
        weight_df = pd.DataFrame(weight_records)
        weight_df['date'] = pd.to_datetime(weight_df['date'])
        
        print(f"Processing {len(weight_df)} weight measurements across {weight_df['cow_id'].nunique()} cows...")
        
        # Create smoother - auto-tuning will happen inside filter() if needed
        smoother = KalmanSmoother(
            auto_tune=True
        )
        
        # Apply smoothing with proper time handling
        smoothed_df = smoother.smooth(weight_df, 'weight', 'cow_id', 'date')       
        smoother.plot_all_entities(smoothed_df, 'weight', 'cow_id', 'date', 
                           save_path='all_cattle_weights.png')

        # Add smoothed weights back to objects
        for cow_id, cow_dict in self.objects.items():
            cow_smoothed = smoothed_df[smoothed_df['cow_id'] == cow_id].copy()
            cow_smoothed = cow_smoothed.sort_values('date').reset_index(drop=True)
            
            # Add smoothed values to weight history
            weight_history = cow_dict['weight_history_data']
            for i, entry in enumerate(weight_history.data):
                if i < len(cow_smoothed):
                    

                    # RTS smoothed estimate (non-causal - uses future info)
                    entry['weight_smoothed'] = cow_smoothed.iloc[i]['weight_smoothed']
                    entry['weight_smoothed_se'] = cow_smoothed.iloc[i]['weight_smoothed_se']
                    
                    # Forward-pass filtered estimate (CAUSAL - no future info!)
                    entry['weight_filtered'] = cow_smoothed.iloc[i]['weight_filtered']
                    entry['weight_filtered_se'] = cow_smoothed.iloc[i]['weight_filtered_se']
                    
                    # Drift rate for this cow
                    
                    # Keep original weight unchanged
                    # entry['weight'] stays as raw measurement
        
        print("\n" + "="*80)
        print("KALMAN SMOOTHING COMPLETE!")
        print("="*80)
        print("Added to weight history:")
        print("  - 'weight_filtered': Forward-pass filtered (CAUSAL - use for prediction!)")
        print("  - 'weight_filtered_se': Standard error of filtered estimate")
        print("  - 'weight_smoothed': RTS smoothed (non-causal - visualization only)")
        print("  - 'weight_smoothed_se': Standard error of smoothed estimate")
        print("  - 'weight': Original raw measurement (unchanged)")
        print("\n⚠️  IMPORTANT: Use 'weight_filtered' for prediction to avoid data leakage!")
        print("="*80 + "\n")
        
        return self.objects


    def get_dfs(self, n_weighings: list, measurement_noise=None, process_noise_per_day=None,
                estimate_drift=True, auto_tune=True, apply_smoothing=True, cut_tails=False):
        """
        Generate dataframes with optional Kalman smoothing applied BEFORE feature engineering.
        
        Args:
            n_weighings: List of prediction horizons (e.g., [1, 2, 3])
            
            measurement_noise: Measurement error variance. If None and auto_tune=True,
                              will be estimated from data. Default: None
            
            process_noise_per_day: Process noise per day. If None and auto_tune=True,
                                  will be estimated from data. Default: None
            
            estimate_drift: If True, estimates growth rate per cow. Default: True
            
            auto_tune: If True, automatically estimates None parameters via MLE. Default: True
            
            apply_smoothing: If True, applies Kalman smoothing to raw weights first. Default: True
            
            cut_tails: If True, removes bottom and top 2.5% of pred_adgLatest_average. Default: False
        
        Returns:
            dict: Dictionary of DataFrames keyed by n_weighing value
        
        Examples:
            # Full auto-tuning (recommended)
            dfs = processor.get_dfs([1, 2, 3], auto_tune=True, apply_smoothing=True)
            
            # Fix scale error, estimate biological variation
            dfs = processor.get_dfs(
                [1, 2, 3],
                measurement_noise=400,  # Trust your scale
                auto_tune=True          # Estimate process noise
            )
            
            # Manual parameters
            dfs = processor.get_dfs(
                [1, 2, 3],
                measurement_noise=400,
                process_noise_per_day=2.0,
                auto_tune=False
            )
            
            # No smoothing (use raw data)
            dfs = processor.get_dfs([1, 2, 3], apply_smoothing=False)
        """
        # STEP 0: Load data first if not already loaded
        if self.objects is None:
            self.get_data()
       
        # STEP 1: Apply smoothing to raw weight data if requested
        if apply_smoothing:
            print("\n" + "="*80)
            print("STEP 1: KALMAN SMOOTHING")
            print("="*80)
            self.apply_kalman_smoothing(
                measurement_noise=measurement_noise,
                process_noise_per_day=process_noise_per_day,
                estimate_drift=estimate_drift,
                auto_tune=auto_tune
            )
        
        # STEP 2: Generate features (will use smoothed weights if available)
        print("\n" + "="*80)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*80)
        
        for n in n_weighings:
            print(f"\n--- Generating features for n={n} weighings ahead ---")
            arr = self.get_variables(n, use_smoothed=apply_smoothing)
            df = pd.DataFrame(arr)
            df['pred_date'] = pd.to_datetime(df['pred_date'])
            
            # STEP 3: Cut tails if requested
            if cut_tails:
                original_len = len(df)
                
                # Calculate percentiles
                lower_percentile = df['pred_adgLatest_average'].quantile(0.025)
                upper_percentile = df['pred_adgLatest_average'].quantile(0.975)
                
                # Identify rows to remove
                tail_mask = (df['pred_adgLatest_average'] < lower_percentile) | \
                            (df['pred_adgLatest_average'] > upper_percentile)
                removed_df = df[tail_mask].copy()
                
                # Print information about removed data
                print(f"\n⚠️ Cutting tails: Removing bottom and top 2.5% of pred_adgLatest_average")
                print("=" * 80)
                print(f"Original dataset size: {original_len}")
                print(f"Lower 2.5% threshold: {lower_percentile:.4f}")
                print(f"Upper 97.5% threshold: {upper_percentile:.4f}")
                print(f"Rows removed: {len(removed_df)} ({len(removed_df)/original_len*100:.2f}%)")
                
                if len(removed_df) > 0:
                    cow_col = 'cow_id' if 'cow_id' in df.columns else 'cattleId'
                    date_col = 'pred_date' if 'pred_date' in df.columns else 'date'
                    
                    print(f"\nRemoved data points:")
                    print(f"{'Cow ID':<20} {'Date':<20} {'Value':<12} {'Reason':<15}")
                    print("-" * 80)
                    
                    for idx, row in removed_df.iterrows():
                        cow_id = row.get(cow_col, 'Unknown')
                        date = row.get(date_col, 'Unknown')
                        value = row['pred_adgLatest_average']
                        reason = 'Bottom 2.5%' if value < lower_percentile else 'Top 2.5%'
                        print(f"{str(cow_id):<20} {str(date):<20} {value:<12.4f} {reason:<15}")
                
                # Filter the dataframe
                df = df[~tail_mask].copy()
                print(f"\nFinal dataset size: {len(df)}")
                print("=" * 80)
            
            self.dfs[n] = df
        
        print("\n" + "="*80)
        print("DATAFRAME GENERATION COMPLETE")
        print("="*80)
        print(f"Generated {len(self.dfs)} dataframes:")
        for n, df in self.dfs.items():
            print(f"  n={n}: {len(df)} observations")
        print("="*80 + "\n")
        
        return self.dfs


    def _process_single_window(self, cow_data, weight_history, feed_history, medical_history, 
                               x, n_weighing, use_smoothed=True, n_start=2):
        """
        Processes a single non-overlapping window for a cow.
        
        ⚠️  IMPORTANT: Uses 'weight_filtered' (causal) instead of 'weight_smoothed' 
                      to avoid data leakage!
        
        Args:
            cow_data: CowData object
            weight_history: WeightHistoryData object
            feed_history: FeedHistoryData object
            medical_history: MedicalHistoryData object
            x (int): Starting index in weight history
            n_weighing (int): Number of weighings in this window
            use_smoothed (bool): If True, use FILTERED weights (not smoothed!)
            n_start (int): Starting index for processing
            
        Returns:
            dict or None: Dictionary of features, or None if window should be skipped
        """
        entry = weight_history.data[x]
        ret_dict = {}
        
        # ===== SUPER PRIMITIVES =====
        target_weighing = x + n_weighing
        ret_dict['pred_date'] = weight_history.data[target_weighing]['date']

        ret_dict['date'] = entry['date']
        ret_dict['startWeight'] = weight_history.data[0]['weight']

        ret_dict['day_diff'] = (datetime.strptime(ret_dict['pred_date'], "%Y-%m-%d") - 
                               datetime.strptime(ret_dict['date'], "%Y-%m-%d")).days
        ret_dict['day_diff_2'] = ret_dict['day_diff']**2
        ret_dict['day_diff_recp'] = ret_dict['day_diff']**2
      
        ret_dict['theoritical_error_adg'] = 20/ret_dict['day_diff']

        # ⚠️  CRITICAL: USE FILTERED (CAUSAL) NOT SMOOTHED (NON-CAUSAL)
        # Filtered = forward pass only = no future information = no data leakage
        # Smoothed = RTS backward pass = uses future information = DATA LEAKAGE!
        if use_smoothed and 'weight_filtered' in entry:
            ret_dict['weight'] = entry['weight_filtered']  # CAUSAL estimate
            ret_dict['weight_caus'] = entry['weight_filtered']  # CAUSAL estimate
            ret_dict['weight_raw'] = entry['weight']  # Original measurement
            ret_dict['weight_se'] = entry.get('weight_filtered_se', 0)  # Filtered Standard error
        else:
            ret_dict['weight'] = entry['weight']  # Raw measurement
        
        ret_dict['cattleId'] = cow_data.cattleId
        ret_dict['originWeight'] = cow_data.originWeight
        ret_dict['originWeight_dt'] = cow_data.originWeight/ret_dict['day_diff']
        ret_dict['hipHeight'] = cow_data.hipHeight
        ret_dict['breed'] = cow_data.breed

        # Breed indicators
        ret_dict['isLimousine'] = (ret_dict['breed'] == 'Limousin') or (ret_dict['breed'] == 'Limousine')
        ret_dict['isSimental'] = (ret_dict['breed'] == 'Simental') or (ret_dict['breed'] == 'Simmental')
       
        if ret_dict['breed'] not in ['Limousin', 'Simental', 'Limousine', 'Simmental']:
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
        # ⚠️  CRITICAL: USE FILTERED (CAUSAL) FOR TARGET TOO
        target_entry = weight_history.data[target_weighing]
        if use_smoothed and 'weight_filtered' in target_entry:
            ret_dict['pred_weight'] = target_entry['weight_filtered']  # CAUSAL
            ret_dict['pred_weight_raw'] = target_entry['weight']  # Raw
            ret_dict['pred_weight_se'] = target_entry.get('weight_filtered_se', 0)
        else:
            ret_dict['pred_weight'] = target_entry['weight']
        
        # NOW ALL CALCULATIONS USE FILTERED (CAUSAL) WEIGHTS
        ret_dict['pred_weight_gain'] = ret_dict['pred_weight'] - ret_dict['weight']
       
        if use_smoothed and 'weight_filtered' in entry:
            ret_dict['pred_weight_gain_raw'] = ret_dict['pred_weight_raw'] - ret_dict['weight_raw']
        
        ret_dict['pred_adgLatest_average'] = ret_dict['pred_weight_gain'] / ret_dict['day_diff']
        ret_dict['pred_adgLatest_average_log'] = self.signed_log_transform(ret_dict['pred_adgLatest_average']) 
        ret_dict['pred_adgLatest_average_2'] = ret_dict['pred_adgLatest_average']**2 


        ret_dict['pred_adgLatest_average_inverse_hyperbolic'] = (
            np.log(ret_dict['pred_adgLatest_average'] + 
                   (ret_dict['pred_adgLatest_average']**2 + 1)**0.5) * 0.5
        )
        ret_dict['pred_fcrLatest_average'] = (
            (ret_dict['pred_weight_gain'] / ret_dict['total_dm_intake']) * 100
        )
        
        # ===== PRIMITIVES (now using filtered weight) =====
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

        ret_dict['mw_dmi_dt_dstartweight'] = ret_dict['mw_dmi_dt']/ret_dict['startWeight']

        ret_dict['increase_ratio'] = ret_dict['weight']/ret_dict['startWeight']
        ret_dict['increase_ratio_dt'] = ret_dict['increase_ratio']/ret_dict['day_diff'] 
        ret_dict['ln_increase_ratio_dt'] = math.log(ret_dict['increase_ratio'])/ret_dict['day_diff'] 
        ret_dict['exp_increase_ratio'] = math.exp(ret_dict['weight']/ret_dict['startWeight'])
        ret_dict['mw_ratio'] = ret_dict['metabolic_weight']*ret_dict['increase_ratio']
        ret_dict['mw_ratio_dmi'] = ret_dict['increase_ratio']*ret_dict['total_dmi'] 
        ret_dict['mw_ratio_dmi_dt'] = ret_dict['mw_ratio_dmi']/ret_dict['day_diff']

        ret_dict['mw_dmi_dt_ratio'] = ret_dict['mw_dmi_dt']*ret_dict['increase_ratio']
        ret_dict['ln_mw_dmi_dt_ratio'] = np.log(ret_dict['mw_dmi_dt']*ret_dict['increase_ratio'])
        ret_dict['ln_mw_dmi_dt_ratio_2'] = (np.log(ret_dict['mw_dmi_dt']*ret_dict['increase_ratio']))**2

        ret_dict['mw_dmi_dt_2'] = ret_dict['mw_dmi_dt']**2 
        ret_dict['day_diff_2_dmi'] = ret_dict['day_diff_2'] * ret_dict['total_dmi']
        ret_dict['day_diff_dmi'] = ret_dict['day_diff'] * ret_dict['total_dmi']
        ret_dict['day_diff_dmi_log'] = np.log(ret_dict['day_diff'] * ret_dict['total_dmi'])

        ret_dict['mw_dmi'] = (ret_dict['metabolic_weight']*ret_dict['total_dmi'])
        
        if ret_dict['breed'] == 'Other':
            return None

        # ===== MEDICAL DATA =====
        if medical_history is not None and medical_history.data is not None:
            ret_dict['hasBEF'] = medical_history.has_matching_agenda_entry(
                ret_dict['date'], ret_dict['pred_date'], 'BEF', False, not_contains=['suspected']
            )
            ret_dict['gotHNMVaccination'] = medical_history.has_matching_agenda_entry(
                ret_dict['date'], ret_dict['pred_date'], 'HNM Vaccination', False
            )
            ret_dict['gotDewormed'] = medical_history.has_matching_agenda_entry(
                ret_dict['date'], ret_dict['pred_date'], 'Worm Medication', False
            )
            ret_dict['gotAppetiteBoost'] = medical_history.has_matching_agenda_entry(
                ret_dict['date'], ret_dict['pred_date'], 'Appetite Boost', False
            )
            if ret_dict['gotDewormed']:
                ret_dict['gotWorms'] = False
            else:
                ret_dict['gotWorms'] = medical_history.has_matching_agenda_entry(
                    ret_dict['date'], ret_dict['pred_date'], 'Worm', False
                )
            ret_dict['DaysSinceDewormed'] = medical_history.days_since_last_matching_agenda_entry(
                ret_dict['pred_date'], 'Worm Medication', False
            )
        else:
            ret_dict['gotAppetiteBoost'] = False 
            ret_dict['hasBEF'] = False 
            ret_dict['gotWorms'] = False
            ret_dict['gotHNMVaccination'] = False
            ret_dict['gotDewormed'] = False
            ret_dict['DaysSinceDewormed'] = 0

        ret_dict['hasBEF_dmi_dt'] = (int(ret_dict['hasBEF'])/ret_dict['day_diff'])*ret_dict['total_dmi']
        ret_dict['hasBEF_dt'] = (int(ret_dict['hasBEF'])/ret_dict['day_diff'])
        ret_dict['gotAppetiteBoost_dmi_dt'] = (int(ret_dict['gotAppetiteBoost'])/ret_dict['day_diff'])*ret_dict['total_dmi']
        ret_dict['gotHNMVaccination_dmi_dt'] = (int(ret_dict['gotHNMVaccination'])/ret_dict['day_diff'])*ret_dict['total_dmi']
        ret_dict['gotHNMVaccination_dt'] = (int(ret_dict['gotHNMVaccination'])/ret_dict['day_diff'])
        ret_dict['gotAppetiteBoost_dt'] = (int(ret_dict['gotAppetiteBoost'])/ret_dict['day_diff'])



        if ret_dict['hasBEF_dmi_dt'] == 0:
            ret_dict['hasBEF_dmi_dt_log'] = 0
        else:
            ret_dict['hasBEF_dmi_dt_log'] = np.log((int(ret_dict['hasBEF'])/ret_dict['day_diff'])*ret_dict['total_dmi'])

        ret_dict['hasBEF_dmi'] = (int(ret_dict['hasBEF']))*ret_dict['total_dmi']
        ret_dict['hasBEF_dmi_dt_2'] = ret_dict['hasBEF_dmi_dt']**2 
        ret_dict['hasBEF_ddmi'] = int(ret_dict['hasBEF'])*ret_dict['total_dmi']
        ret_dict['gotDewormed_dt'] = int(ret_dict['gotDewormed'])/ret_dict['day_diff']
        ret_dict['DaysSinceDewormed_dt'] = int(ret_dict['DaysSinceDewormed'])/ret_dict['day_diff']


        if ret_dict.get('pred_adgLatest_average') is not None and ret_dict['pred_adgLatest_average'] > 1.5 or ret_dict['pred_adgLatest_average'] < 0.6:
            with open('check.txt', 'a', encoding='utf-8') as f:
                f.write(json.dumps(ret_dict, indent=2))
                f.write('\n' + '-' * 80 + '\n')


        if ret_dict['pred_adgLatest_average'] < 0.6:
            pass
            #if not medical_history == None:
                #res = medical_history.get_matching_agenda_entry(
                #    ret_dict['date'], ret_dict['pred_date'], '', False
                #)
                
                #for entry in res:
                #    print(entry['agenda'])

            #print(ret_dict)

        return ret_dict
