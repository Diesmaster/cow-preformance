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

from consts.consts import costs_per_dm, sales_price
from data_processor.FeedProcessor import FeedProcessor
from data_processor.KalmanSmoother import KalmanSmoother


class DataProcessing:
    """
    A class for processing dairy cow data.
    """

    def __init__(self, main_folder='./data', analysis_folder='./analysis-dec-2024/'):
        self.main_folder = main_folder
        self.analysis_folder = analysis_folder

        self.cow_weight_history_data = 'KC51sJ30yRPUgRKZsvoI-cowweighthistory.json'
        self.cow_feed_history_data   = 'KC51sJ30yRPUgRKZsvoI-feedhistory.json'
        self.cows_data               = 'KC51sJ30yRPUgRKZsvoI-cows.json'
        self.historic_cows_data      = 'KC51sJ30yRPUgRKZsvoI-historic-cows.json'
        self.medical_history_data    = 'KC51sJ30yRPUgRKZsvoI-medicalHistory.json'
        self.ingredients_data        = 'KC51sJ30yRPUgRKZsvoI-ingredients.json'
        self.recipes_data            = 'KC51sJ30yRPUgRKZsvoI-recipes.json'
        self.historic_recipes_data   = 'KC51sJ30yRPUgRKZsvoI-historic-recipes.json'
        self.effects                 = 'effects.json'

        self.date_format = '%Y-%m-%d'
        self.objects = None
        self.dfs = {}

        self.ingredient_nutrition = {}

        # ---- hormone decay parameters (Synovex-S tuned) ----
        self.T_MIN  = 3
        self.T_PEAK = 30
        self.T_MAX  = 120
        self.LAMBDA = 0.02

    # ------------------------------------------------------------------ #
    #  Hormone decay                                                        #
    # ------------------------------------------------------------------ #

    def hormone_decay(self, t):
        if t is None or t < self.T_MIN:
            return 0.0
        if t < self.T_PEAK:
            return (t - self.T_MIN) / (self.T_PEAK - self.T_MIN)
        return math.exp(-self.LAMBDA * (t - self.T_PEAK))

    # ------------------------------------------------------------------ #
    #  Ingredient nutrition lookup                                          #
    # ------------------------------------------------------------------ #

    def _build_nutrition_lookup(self, ingredients_raw: dict) -> dict:
        _REFERENCE_NUTRITION = {
            'Jagung':                  {'cf': 0.022, 'fat': 0.038, 'ash': 0.014},
            'Silase Jagung':           {'cf': 0.260, 'fat': 0.030, 'ash': 0.045},
            'Rumput':                  {'cf': 0.280, 'fat': 0.025, 'ash': 0.090},
            'Pakchong Grass - Qurban': {'cf': 0.300, 'fat': 0.020, 'ash': 0.080},
            'Rice Hay':                {'cf': 0.320, 'fat': 0.013, 'ash': 0.120},
            'Slobber Mix':             {'cf': 0.120, 'fat': 0.040, 'ash': 0.080},
            'SP2A Mix':                {'cf': 0.168, 'fat': 0.052, 'ash': 0.070},
            'SP2A':                    {'cf': 0.168, 'fat': 0.052, 'ash': 0.070},
            'SP2B Mix':                {'cf': 0.308, 'fat': 0.052, 'ash': 0.080},
            'SP2B':                    {'cf': 0.308, 'fat': 0.052, 'ash': 0.080},
            'SMG Mixfeed S14':         {'cf': 0.120, 'fat': 0.060, 'ash': 0.080},
            'Konsentrat':              {'cf': 0.168, 'fat': 0.028, 'ash': 0.070},
            'DDGS':                    {'cf': 0.090, 'fat': 0.100, 'ash': 0.040},
            'Ampas Tahu':              {'cf': 0.195, 'fat': 0.056, 'ash': 0.040},
            'Mollases':                {'cf': 0.000, 'fat': 0.001, 'ash': 0.080, 'cp': 0.040},
            'Tepung Gaplek':           {'cf': 0.010, 'fat': 0.003, 'ash': 0.025},
            'FML':                     {'cf': 0.125, 'fat': 0.015, 'ash': 0.080},
            'Garam':                   {'cf': 0.000, 'fat': 0.000, 'ash': 0.980, 'cp': 0.000},
        }

        def _val(val) -> float | None:
            try:
                return float(val) / 100.0
            except (TypeError, ValueError):
                return None

        def _val_or_ref(val, name, nutrient) -> float | None:
            v = _val(val)
            if v is not None and v >= 0:
                return v
            return _REFERENCE_NUTRITION.get(name, {}).get(nutrient)

        def _calc_betn(cp, cf, fat, ash) -> float | None:
            if any(v is None for v in [cp, cf, fat]):
                return None
            if ash is None:
                betn = 1.0 - cp - cf - fat
            else:
                betn = 1.0 - cp - cf - fat - ash
            return betn if betn > 0 else None

        lookup = {}
        for doc_id, doc in ingredients_raw.items():
            name = doc.get('PAKAN')
            if not name:
                continue

            cp  = _val_or_ref(doc.get('PK'),  name, 'cp')
            cf  = _val_or_ref(doc.get('SK'),  name, 'cf')
            fat = _val_or_ref(doc.get('LK'),  name, 'fat') or _val_or_ref(doc.get('EE'), name, 'fat')
            ash = _val_or_ref(doc.get('Abu'), name, 'ash') or _val_or_ref(doc.get('Ash'), name, 'ash')
            nfe = _val(doc.get('NFE'))

            betn = nfe if (nfe is not None and nfe > 0) else _calc_betn(cp, cf, fat, ash)

            print(f"{name:<30} cp={cp} cf={cf} fat={fat} ash={ash} nfe={nfe} → betn={betn}")

            lookup[name] = {
                'tdn':  _val(doc.get('TDN')),
                'cp':   cp,
                'rdp':  _val(doc.get('RDP')),
                'cf':   cf,
                'fat':  fat,
                'betn': betn,
            }
        return lookup

    def _build_recipe_nutrition_lookup(self, recipes_raw: dict) -> dict:
        """
        Builds a nutrition lookup from recipe JSON data.
        Recipes store top-level proximate analysis as percentages on DM basis —
        same convention as ingredients.
        """
        def _val(val) -> float | None:
            try:
                v = float(val)
                return v / 100.0 if v != 0 else None
            except (TypeError, ValueError):
                return None

        def _calc_betn(cp, cf, fat) -> float | None:
            if any(v is None for v in [cp, cf, fat]):
                return None
            betn = 1.0 - cp - cf - fat
            return betn if betn > 0 else None

        lookup = {}
        for doc_id, doc in recipes_raw.items():
            name = doc.get('title')
            if not name:
                continue

            tdn = _val(doc.get('TDN'))
            cp  = _val(doc.get('PK'))
            cf  = _val(doc.get('SK'))
            fat = _val(doc.get('LK')) or _val(doc.get('EE'))
            nfe = _val(doc.get('NFE'))
            rdp = _val(doc.get('RDP'))

            betn = nfe if (nfe is not None and nfe > 0) else _calc_betn(cp, cf, fat)

            print(f"[recipe] {name:<30} tdn={tdn} cp={cp} cf={cf} fat={fat} → betn={betn}")

            lookup[name] = {
                'tdn':  tdn,
                'cp':   cp,
                'rdp':  rdp,
                'cf':   cf,
                'fat':  fat,
                'betn': betn,
            }
        return lookup

    # ------------------------------------------------------------------ #
    #  JSON loading                                                         #
    # ------------------------------------------------------------------ #

    def load_json_data(self, file_name, folder=None):
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

    # ------------------------------------------------------------------ #
    #  Number cleaning                                                      #
    # ------------------------------------------------------------------ #

    def fix_numbers_dic_of_dic(self, dic_of_dicts):
        for key in dic_of_dicts:
            if isinstance(dic_of_dicts[key], dict):
                dic_of_dicts[key] = self.fix_numbers_dic_of_dic(dic_of_dicts[key])

            if isinstance(dic_of_dicts[key], list):
                for x in range(len(dic_of_dicts[key])):
                    if isinstance(dic_of_dicts[key][x], dict):
                        dic_of_dicts[key][x] = self.fix_numbers_dic_of_dic(dic_of_dicts[key][x])
                    else:
                        try:
                            dic_of_dicts[key][x] = round(float(dic_of_dicts[key][x]), 2)
                        except Exception:
                            if dic_of_dicts[key][x] == 'None':  dic_of_dicts[key][x] = None
                            elif dic_of_dicts[key][x] == 'True': dic_of_dicts[key][x] = True
                            elif dic_of_dicts[key][x] == 'False': dic_of_dicts[key][x] = False

            try:
                dic_of_dicts[key] = round(float(dic_of_dicts[key]), 2)
            except Exception:
                if dic_of_dicts[key] == 'None':  dic_of_dicts[key] = None
                elif dic_of_dicts[key] == 'True': dic_of_dicts[key] = True
                elif dic_of_dicts[key] == 'False': dic_of_dicts[key] = False

        return dic_of_dicts

    # ------------------------------------------------------------------ #
    #  Cast raw dicts to data objects                                       #
    # ------------------------------------------------------------------ #

    def cast_to_obj(self, cows, weight_histories, feed_histories, medical_histories, effects):
        ret_dict = {}
        for cow_id in cows:
            if cow_id != "rexFmUY8QHCvB0TsjnbB":
                ret_dict[cow_id] = {}
                cattleId = cows[cow_id]['cattleId']

                ret_dict[cow_id]['effect'] = effects[cattleId] if cattleId in effects else None
                ret_dict[cow_id]['cow_data'] = CowData(cows[cow_id])
                ret_dict[cow_id]['weight_history_data'] = WeightHistoryData(weight_histories[cow_id])
                ret_dict[cow_id]['feed_history_data'] = (
                    FeedHistoryData(feed_histories[cow_id]) if cow_id in feed_histories else None
                )
                ret_dict[cow_id]['medical_history_data'] = (
                    MedicalHistoryData(medical_histories[cow_id]) if cow_id in medical_histories else None
                )
        return ret_dict

    # ------------------------------------------------------------------ #
    #  Main data loader                                                     #
    # ------------------------------------------------------------------ #

    def get_data(self):
        cows              = self.load_json_data(self.cows_data)
        weight_histories  = self.load_json_data(self.cow_weight_history_data)
        feed_histories    = self.load_json_data(self.cow_feed_history_data)
        historic_cows     = self.load_json_data(self.historic_cows_data)
        medical_histories = self.load_json_data(self.medical_history_data)
        effects           = self.load_json_data(self.effects)

        # Load all three nutrition sources separately
        ingredients_raw      = self.load_json_data(self.ingredients_data)
        recipes_raw          = self.load_json_data(self.recipes_data)
        historic_recipes_raw = self.load_json_data(self.historic_recipes_data)

        cows = cows | historic_cows

        cows              = self.fix_numbers_dic_of_dic(cows)
        weight_histories  = self.fix_numbers_dic_of_dic(weight_histories)
        feed_histories    = self.fix_numbers_dic_of_dic(feed_histories)
        medical_histories = self.fix_numbers_dic_of_dic(medical_histories)

        # Build ingredient lookup first, then overlay recipe lookups.
        # Recipes win on name collision since they are more specific.
        ingredient_lookup = self._build_nutrition_lookup(ingredients_raw)

        all_recipes_raw = recipes_raw | historic_recipes_raw
        recipe_lookup   = self._build_recipe_nutrition_lookup(all_recipes_raw)

        self.ingredient_nutrition = ingredient_lookup | recipe_lookup

        print(
            f"Loaded nutrition data: {len(ingredient_lookup)} ingredients + "
            f"{len(recipe_lookup)} recipes = {len(self.ingredient_nutrition)} total."
        )

        self.objects = self.cast_to_obj(
            cows, weight_histories, feed_histories, medical_histories, effects
        )
        return self.objects

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                   #
    # ------------------------------------------------------------------ #

    def get_variables(self, n_weighing, use_smoothed=True):
        if self.objects is None:
            self.get_data()

        ret_arr = []

        for cow_id, cow_dict in self.objects.items():
            cow_data        = cow_dict['cow_data']
            weight_history  = cow_dict['weight_history_data']
            feed_history    = cow_dict['feed_history_data']
            medical_history = cow_dict['medical_history_data']
            effect          = cow_dict['effect']

            if feed_history is None:
                continue

            time = 0
            n_start = 1

            for x in range(n_start, len(weight_history.data) - n_weighing, n_weighing):
                window_data = self._process_single_window(
                    cow_data, weight_history, feed_history, medical_history, x, n_weighing,
                    use_smoothed=use_smoothed,
                    n_start=n_start
                )
                if window_data is None:
                    continue

                window_data['tdn_silage_over_mw_daysinfeedlot_dt']  = window_data['tdn_silage_dt']  / window_data['metabolic_weight']
                window_data['tdn_rumput_over_mw_daysinfeedlot_dt']  = window_data['tdn_rumput_dt']  / window_data['metabolic_weight']
                window_data['tdn_slobber_over_mw_daysinfeedlot_dt'] = window_data['tdn_slobber_dt'] / window_data['metabolic_weight']
                window_data['tdn_SP2A_over_mw_daysinfeedlot_dt']    = window_data['tdn_SP2A_dt']    / window_data['metabolic_weight']
                window_data['tdn_ricehay_over_mw_daysinfeedlot_dt'] = window_data['tdn_ricehay_dt'] / window_data['metabolic_weight']

                window_data['tdn_silage_dt_ratio']  = window_data['tdn_silage_dt']  / window_data['exp_increase_ratio']
                window_data['tdn_rumput_dt_ratio']  = window_data['tdn_rumput_dt']  / window_data['exp_increase_ratio']
                window_data['tdn_slobber_dt_ratio'] = window_data['tdn_slobber_dt'] / window_data['exp_increase_ratio']
                window_data['tdn_SP2A_dt_ratio']    = window_data['tdn_SP2A_dt']    / window_data['exp_increase_ratio']
                window_data['tdn_ricehay_dt_ratio'] = window_data['tdn_ricehay_dt'] / window_data['exp_increase_ratio']

                window_data['cow_id'] = cow_data.cattleId
                window_data['time']   = time
                window_data['effect'] = effect
                window_data['docId']  = cow_id
                time += 1
                ret_arr.append(window_data)

        return ret_arr

    def signed_log_transform(self, x):
        return np.sign(x) * np.log1p(np.abs(x))

    # ------------------------------------------------------------------ #
    #  Kalman smoothing                                                     #
    # ------------------------------------------------------------------ #

    def apply_kalman_smoothing(self, measurement_noise=None, process_noise_per_day=None,
                               estimate_drift=True, fixed_drift=None, auto_tune=True):
        if self.objects is None:
            raise ValueError("Must call get_data() first")

        is_auto_tuning = auto_tune and (measurement_noise is None or process_noise_per_day is None)

        if not auto_tune:
            if measurement_noise is None:
                measurement_noise = 400.0
                print("⚠️  Using default measurement_noise = 400 (auto_tune=False)")
            if process_noise_per_day is None:
                process_noise_per_day = 2.0
                print("⚠️  Using default process_noise_per_day = 2.0 (auto_tune=False)")

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

        print("\n" + "=" * 80)
        if is_auto_tuning:
            print("APPLYING KALMAN SMOOTHING WITH AUTO-TUNED PARAMETERS")
            print("=" * 80)
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
            print("=" * 80)
            print(f"Measurement noise: {measurement_noise} (±{np.sqrt(measurement_noise):.1f} kg CONSTANT)")
            print(f"Process noise per day: {process_noise_per_day} (±{np.sqrt(process_noise_per_day):.2f} kg/√day)")

        print(f"Drift estimation: {'Per-cow linear regression' if estimate_drift else f'Fixed at {fixed_drift} kg/day'}")

        if not is_auto_tuning:
            print("\nProcess noise scales with time interval:")
            for days in [7, 14, 28]:
                variance = process_noise_per_day * days
                print(f"  {days:2d} days: variance = {variance:6.1f}, std = ±{np.sqrt(variance):5.2f} kg")

        print("=" * 80 + "\n")

        weight_records = []
        for cow_id, cow_dict in self.objects.items():
            weight_history = cow_dict['weight_history_data']
            for idx, entry in enumerate(weight_history.data):
                weight_records.append({
                    'cow_id': cow_id,
                    'date':   entry['date'],
                    'weight': entry['weight'],
                    'index':  idx
                })

        weight_df = pd.DataFrame(weight_records)
        weight_df['date'] = pd.to_datetime(weight_df['date'])

        print(f"Processing {len(weight_df)} weight measurements across {weight_df['cow_id'].nunique()} cows...")

        smoother = KalmanSmoother(auto_tune=True)
        smoothed_df = smoother.smooth(weight_df, 'weight', 'cow_id', 'date')
        smoother.plot_all_entities(smoothed_df, 'weight', 'cow_id', 'date',
                                   save_path='all_cattle_weights.png')

        for cow_id, cow_dict in self.objects.items():
            cow_smoothed = smoothed_df[smoothed_df['cow_id'] == cow_id].copy()
            cow_smoothed = cow_smoothed.sort_values('date').reset_index(drop=True)

            weight_history = cow_dict['weight_history_data']
            for i, entry in enumerate(weight_history.data):
                if i < len(cow_smoothed):
                    entry['weight_smoothed']    = cow_smoothed.iloc[i]['weight_smoothed']
                    entry['weight_smoothed_se'] = cow_smoothed.iloc[i]['weight_smoothed_se']
                    entry['weight_filtered']    = cow_smoothed.iloc[i]['weight_filtered']
                    entry['weight_filtered_se'] = cow_smoothed.iloc[i]['weight_filtered_se']

        print("\n" + "=" * 80)
        print("KALMAN SMOOTHING COMPLETE!")
        print("=" * 80)
        print("  - 'weight_filtered': Forward-pass filtered (CAUSAL - use for prediction!)")
        print("  - 'weight_filtered_se': Standard error of filtered estimate")
        print("  - 'weight_smoothed': RTS smoothed (non-causal - visualization only)")
        print("  - 'weight_smoothed_se': Standard error of smoothed estimate")
        print("  - 'weight': Original raw measurement (unchanged)")
        print("\n⚠️  IMPORTANT: Use 'weight_filtered' for prediction to avoid data leakage!")
        print("=" * 80 + "\n")

        return self.objects

    # ------------------------------------------------------------------ #
    #  get_dfs                                                              #
    # ------------------------------------------------------------------ #

    def get_dfs(self, n_weighings: list, measurement_noise=None, process_noise_per_day=None,
                estimate_drift=True, auto_tune=True, apply_smoothing=True, cut_tails=False):
        if self.objects is None:
            self.get_data()

        if apply_smoothing:
            print("\n" + "=" * 80)
            print("STEP 1: KALMAN SMOOTHING")
            print("=" * 80)
            self.apply_kalman_smoothing(
                measurement_noise=measurement_noise,
                process_noise_per_day=process_noise_per_day,
                estimate_drift=estimate_drift,
                auto_tune=auto_tune
            )

        print("\n" + "=" * 80)
        print("STEP 2: FEATURE ENGINEERING")
        print("=" * 80)

        for n in n_weighings:
            print(f"\n--- Generating features for n={n} weighings ahead ---")
            arr = self.get_variables(n, use_smoothed=apply_smoothing)
            df = pd.DataFrame(arr)
            df['pred_date'] = pd.to_datetime(df['pred_date'])

            if cut_tails:
                original_len     = len(df)
                lower_percentile = df['pred_adgLatest_average'].quantile(0.025)
                upper_percentile = df['pred_adgLatest_average'].quantile(0.975)

                tail_mask  = (
                    (df['pred_adgLatest_average'] < lower_percentile) |
                    (df['pred_adgLatest_average'] > upper_percentile)
                )
                removed_df = df[tail_mask].copy()

                print(f"\n⚠️ Cutting tails: Removing bottom and top 2.5% of pred_adgLatest_average")
                print("=" * 80)
                print(f"Original dataset size: {original_len}")
                print(f"Lower 2.5% threshold: {lower_percentile:.4f}")
                print(f"Upper 97.5% threshold: {upper_percentile:.4f}")
                print(f"Rows removed: {len(removed_df)} ({len(removed_df)/original_len*100:.2f}%)")

                if len(removed_df) > 0:
                    cow_col  = 'cow_id'    if 'cow_id'    in df.columns else 'cattleId'
                    date_col = 'pred_date' if 'pred_date' in df.columns else 'date'
                    print(f"\n{'Cow ID':<20} {'Date':<20} {'Value':<12} {'Reason':<15}")
                    print("-" * 80)
                    for idx, row in removed_df.iterrows():
                        value  = row['pred_adgLatest_average']
                        reason = 'Bottom 2.5%' if value < lower_percentile else 'Top 2.5%'
                        print(f"{str(row.get(cow_col,'Unknown')):<20} {str(row.get(date_col,'Unknown')):<20} {value:<12.4f} {reason:<15}")

                df = df[~tail_mask].copy()
                print(f"\nFinal dataset size: {len(df)}")
                print("=" * 80)

            self.dfs[n] = df

        print("\n" + "=" * 80)
        print("DATAFRAME GENERATION COMPLETE")
        print("=" * 80)
        for n, df in self.dfs.items():
            print(f"  n={n}: {len(df)} observations")
        print("=" * 80 + "\n")

        return self.dfs

    # ------------------------------------------------------------------ #
    #  Single window processing                                             #
    # ------------------------------------------------------------------ #

    def _process_single_window(self, cow_data, weight_history, feed_history, medical_history,
                               x, n_weighing, use_smoothed=True, n_start=2):
        from data_objects.recipe_data import RecipeData as _RecipeData

        entry    = weight_history.data[x]
        ret_dict = {}

        target_weighing = x + n_weighing
        ret_dict['pred_date']   = weight_history.data[target_weighing]['date']
        ret_dict['date']        = entry['date']
        ret_dict['startWeight'] = weight_history.data[0]['weight']

        ret_dict['day_diff'] = (
            datetime.strptime(ret_dict['pred_date'], "%Y-%m-%d") -
            datetime.strptime(ret_dict['date'],      "%Y-%m-%d")
        ).days
        ret_dict['day_diff_2']    = ret_dict['day_diff'] ** 2
        ret_dict['day_diff_recp'] = ret_dict['day_diff'] ** 2

        ret_dict['theoritical_error_adg'] = 20 / ret_dict['day_diff']

        if use_smoothed and 'weight_filtered' in entry:
            ret_dict['weight']      = entry['weight_smoothed']
            ret_dict['weight_caus'] = entry['weight_filtered']
            ret_dict['weight_raw']  = entry['weight']
            ret_dict['weight_se']   = entry.get('weight_filtered_se', 0)
        else:
            ret_dict['weight'] = entry['weight']

        ret_dict['cattleId']        = cow_data.cattleId
        ret_dict['originWeight']    = cow_data.originWeight
        ret_dict['originWeight_dt'] = cow_data.originWeight / ret_dict['day_diff']
        ret_dict['hipHeight']       = cow_data.hipHeight
        ret_dict['breed']           = cow_data.breed

        ret_dict['isLimousine'] = ret_dict['breed'] in ('Limousin', 'Limousine')
        ret_dict['isSimental']  = ret_dict['breed'] in ('Simental', 'Simmental')

        if ret_dict['breed'] not in ['Limousin', 'Simental', 'Limousine', 'Simmental']:
            ret_dict['breed'] = 'Other'

        ret_dict['entryWeight'] = cow_data.entryWeight

        # ===== FEED DATA =====

        # --- inject any RecipeData objects from the ration into nutrition lookup ---
        enriched_nutrition = dict(self.ingredient_nutrition)
        try:
            ration_snapshot = feed_history.get_diet(
                weight_history.data[x]['date'],
                weight_history.data[x + n_weighing]['date']
            )
            for feed_name, feed_item in ration_snapshot.items():
                if isinstance(feed_item, dict):
                    ingredient_obj = feed_item.get('ingredient')
                    if isinstance(ingredient_obj, _RecipeData):
                        enriched_nutrition[f'__recipe_{feed_name}'] = ingredient_obj
        except Exception as e:
            print(f"Warning: could not enrich nutrition from ration for window {x}: {e}")

        feed_processor = FeedProcessor(
            feed_history, weight_history, x, n_weighing,
            ingredient_nutrition=enriched_nutrition
        )

        if not feed_processor.has_required_feeds:
            return None

        feed_features = feed_processor.get_all_features()
        ret_dict.update(feed_features)

        # ===== TARGET =====
        target_entry = weight_history.data[target_weighing]
        if use_smoothed and 'weight_filtered' in target_entry:
            ret_dict['pred_weight']     = target_entry['weight_filtered']
            ret_dict['pred_weight_raw'] = target_entry['weight']
            ret_dict['pred_weight_se']  = target_entry.get('weight_filtered_se', 0)
        else:
            ret_dict['pred_weight'] = target_entry['weight']

        ret_dict['pred_weight_gain'] = ret_dict['pred_weight'] - ret_dict['weight']

        if use_smoothed and 'weight_filtered' in entry:
            ret_dict['pred_weight_gain_raw'] = ret_dict['pred_weight_raw'] - ret_dict['weight_raw']

        ret_dict['pred_adgLatest_average']     = ret_dict['pred_weight_gain'] / ret_dict['day_diff']
        ret_dict['pred_adgLatest_average_log'] = self.signed_log_transform(ret_dict['pred_adgLatest_average'])
        ret_dict['pred_adgLatest_average_2']   = ret_dict['pred_adgLatest_average'] ** 2

        ret_dict['pred_adgLatest_average_inverse_hyperbolic'] = (
            np.log(ret_dict['pred_adgLatest_average'] +
                   (ret_dict['pred_adgLatest_average'] ** 2 + 1) ** 0.5) * 0.5
        )
        ret_dict['pred_fcrLatest_average'] = (
            (ret_dict['pred_weight_gain'] / ret_dict['total_dm_intake']) * 100
        )

        ret_dict['metabolic_weight']          = (ret_dict['weight'] * 0.96) ** 0.75
        ret_dict['pred_adgLatest_average_mw'] = (
            (ret_dict['pred_weight_gain'] / ret_dict['day_diff']) * ret_dict['metabolic_weight']
        )
        ret_dict['originWeight_mw'] = ret_dict['originWeight'] * ret_dict['metabolic_weight']

        for breed_name in ['Simental', 'Limousin', 'Other']:
            ret_dict[f'metabolic_weight_{breed_name}']    = 0
            ret_dict[f'metabolic_weight_{breed_name}_mw'] = 0

        ret_dict['metabolic_weight_2'] = ret_dict['metabolic_weight'] ** 2

        ret_dict['daysOnFeedNow'] = (
            datetime.strptime(ret_dict['date'],   "%Y-%m-%d") -
            datetime.strptime(cow_data.entryDate, "%Y-%m-%d")
        ).days
        ret_dict['daysOnFeedNow_2'] = ret_dict['daysOnFeedNow'] ** 2
        ret_dict['daysOnFeedNow_r'] = ret_dict['daysOnFeedNow'] ** 0.5
        ret_dict['daysOnFeed_then'] = ret_dict['daysOnFeedNow'] + ret_dict['day_diff']

        ret_dict['tdn_slobber_daysonfeed'] = ret_dict['tdn_slobber_over_mw_dt'] * ret_dict['daysOnFeedNow']
        ret_dict['originWeight_ddmi']      = ret_dict['originWeight'] / ret_dict['avg_dm_intake_per_day']

        for breed_name in ['Simental', 'Limousin', 'Other']:
            ret_dict[f'metabolic_weight_{breed_name}_ddmi'] = (
                ret_dict[f'metabolic_weight_{breed_name}'] / ret_dict['avg_dm_intake_per_day']
            )

        ret_dict['mw_per_ddmi']            = ret_dict['metabolic_weight'] / ret_dict['avg_dm_intake_per_day']
        ret_dict['mw_dmi_dt']              = (ret_dict['metabolic_weight'] * ret_dict['total_dmi']) / ret_dict['day_diff']
        ret_dict['mw_dmi_dt_dstartweight'] = ret_dict['mw_dmi_dt'] / ret_dict['startWeight']

        ret_dict['increase_ratio']          = ret_dict['weight'] / ret_dict['startWeight']
        ret_dict['increase_ratio_r']        = ret_dict['increase_ratio'] ** 0.5
        ret_dict['increase_ratio_dt']       = ret_dict['increase_ratio'] / ret_dict['day_diff']
        ret_dict['increase_ratio_dt_r']     = (ret_dict['increase_ratio'] / ret_dict['day_diff']) ** 0.5
        ret_dict['increase_ratio_dt_dmi']   = (ret_dict['increase_ratio'] / ret_dict['day_diff']) * ret_dict['total_dmi']
        ret_dict['increase_ratio_dt_r_dmi'] = (ret_dict['increase_ratio'] / ret_dict['day_diff']) ** 0.5 * ret_dict['total_dmi']
        ret_dict['ln_increase_ratio_dt']    = math.log(ret_dict['increase_ratio']) / ret_dict['day_diff']
        ret_dict['exp_increase_ratio']      = math.exp(ret_dict['weight'] / ret_dict['startWeight'])
        ret_dict['mw_ratio']                = ret_dict['metabolic_weight'] * ret_dict['increase_ratio']
        ret_dict['mw_ratio_dt']             = ret_dict['mw_ratio'] * ret_dict['day_diff']
        ret_dict['mw_ratio_dmi']            = ret_dict['increase_ratio'] * ret_dict['total_dmi']
        ret_dict['mw_ratio_dmi_dt']         = ret_dict['mw_ratio_dmi'] / ret_dict['day_diff']

        ret_dict['tdn_silage_dt_r_dmi_mw']  = ret_dict['tdn_silage_dt_r_dmi_mw']
        ret_dict['tdn_tahu_dt_r_dmi_mw']    = ret_dict['tdn_tahu_dt_r_dmi_mw']
        ret_dict['tdn_rumput_dt_r_dmi_mw']  = ret_dict['tdn_rumput_dt_r_dmi_mw']
        ret_dict['tdn_slobber_dt_r_dmi_mw'] = ret_dict['tdn_slobber_dt_r_dmi_mw']
        ret_dict['tdn_SP2A_dt_r_dmi_mw']    = ret_dict['tdn_SP2A_dt_r_dmi_mw']
        ret_dict['tdn_ricehay_dt_r_dmi_mw'] = ret_dict['tdn_ricehay_dt_r_dmi_mw']

        ret_dict['mw_dmi_dt_ratio']      = ret_dict['mw_dmi_dt'] * ret_dict['increase_ratio']
        ret_dict['mw_dmi_dt_ratio_2']    = (ret_dict['mw_dmi_dt'] * ret_dict['increase_ratio']) ** 2
        ret_dict['ln_mw_dmi_dt_ratio']   = np.log(ret_dict['mw_dmi_dt'] * ret_dict['increase_ratio'])
        ret_dict['ln_mw_dmi_dt_ratio_2'] = (np.log(ret_dict['mw_dmi_dt'] * ret_dict['increase_ratio'])) ** 2

        ret_dict['mw_dmi_dt_2']      = ret_dict['mw_dmi_dt'] ** 2
        ret_dict['day_diff_2_dmi']   = ret_dict['day_diff_2'] * ret_dict['total_dmi']
        ret_dict['day_diff_dmi']     = ret_dict['day_diff']   * ret_dict['total_dmi']
        ret_dict['day_diff_dmi_log'] = np.log(ret_dict['day_diff'] * ret_dict['total_dmi'])
        ret_dict['mw_dmi']           = ret_dict['metabolic_weight'] * ret_dict['total_dmi']

        if ret_dict['breed'] == 'Other':
            return None

        # ===== MEDICAL DATA =====
        if medical_history is not None and medical_history.data is not None:
            ret_dict['hasBEF'] = medical_history.has_matching_agenda_entry(
                ret_dict['date'], ret_dict['pred_date'], 'BEF', False, not_contains=['suspected']
            )
            ret_dict['gotHormonesLast'] = medical_history.days_since_last_matching_agenda_entry(
                ret_dict['date'], 'Hormone Implant', True
            )

            if ret_dict['gotHormonesLast'] >= 14:
                ret_dict['1_over_hormones']    = 1 / ret_dict['gotHormonesLast']
                ret_dict['hormone_adjustment'] = 0
                ret_dict['gotHormones']        = True
                ret_dict['hormone_effect']     = self.hormone_decay(ret_dict['gotHormonesLast'])
            else:
                ret_dict['1_over_hormones']    = 0
                ret_dict['gotHormones']        = False
                ret_dict['hormone_effect']     = 0
                ret_dict['hormone_adjustment'] = 1 if 0 <= ret_dict['gotHormonesLast'] < 14 else 0

            ret_dict['gotHNMVaccination'] = (
                medical_history.has_matching_agenda_entry(
                    ret_dict['date'], ret_dict['pred_date'], 'HNM Vaccination', False
                ) or
                medical_history.has_matching_agenda_entry(
                    ret_dict['date'], ret_dict['pred_date'], 'FMD Vaccination', False
                )
            )
            ret_dict['gotDewormed'] = medical_history.has_matching_agenda_entry(
                ret_dict['date'], ret_dict['pred_date'], 'Worm Medication', False
            )
            ret_dict['gotAppetiteBoost'] = medical_history.has_matching_agenda_entry(
                ret_dict['date'], ret_dict['pred_date'], 'Appetite Boost', False
            )
            ret_dict['gotWorms'] = (
                False if ret_dict['gotDewormed']
                else medical_history.has_matching_agenda_entry(
                    ret_dict['date'], ret_dict['pred_date'], 'Worm', False
                )
            )
            ret_dict['DaysSinceDewormed'] = medical_history.days_since_last_matching_agenda_entry(
                ret_dict['pred_date'], 'Worm Medication', False
            )
        else:
            ret_dict['gotAppetiteBoost']  = False
            ret_dict['hasBEF']            = False
            ret_dict['gotWorms']          = False
            ret_dict['gotHNMVaccination'] = False
            ret_dict['gotDewormed']       = False
            ret_dict['DaysSinceDewormed'] = 0
            ret_dict['gotHormones']       = 0
            ret_dict['1_over_hormones']   = 0
            ret_dict['hormone_effect']    = 0
            ret_dict['hormone_adjustment']= 0

        ret_dict['hasBEF_dmi_dt']           = (int(ret_dict['hasBEF'])            / ret_dict['day_diff']) * ret_dict['total_dmi']
        ret_dict['gotDewormed_dmi_dt']       = (int(ret_dict['gotDewormed'])       / ret_dict['day_diff']) * ret_dict['total_dmi']
        ret_dict['gotHormones_dmi_dt']       = (int(ret_dict['gotHormones'])       / ret_dict['day_diff']) * ret_dict['total_dmi']
        ret_dict['hasBEF_dt']               =  int(ret_dict['hasBEF'])             / ret_dict['day_diff']
        ret_dict['gotAppetiteBoost_dmi_dt']  = (int(ret_dict['gotAppetiteBoost'])  / ret_dict['day_diff']) * ret_dict['total_dmi']
        ret_dict['gotHNMVaccination_dmi_dt'] = (int(ret_dict['gotHNMVaccination']) / ret_dict['day_diff']) * ret_dict['total_dmi']
        ret_dict['gotHNMVaccination_dt']     =  int(ret_dict['gotHNMVaccination'])  / ret_dict['day_diff']
        ret_dict['gotAppetiteBoost_dt']      =  int(ret_dict['gotAppetiteBoost'])   / ret_dict['day_diff']

        ret_dict['1_over_hormones_dt']    = ret_dict['1_over_hormones']    / ret_dict['day_diff']
        ret_dict['hormone_adjustment_dt'] = ret_dict['hormone_adjustment']  / ret_dict['day_diff']
        ret_dict['gotHormones_dt']        = int(ret_dict['gotHormones'])    / ret_dict['day_diff']
        ret_dict['hormone_effect_dt']     = ret_dict['hormone_effect']      / ret_dict['day_diff']
        ret_dict['gotHormones_dt_mw']     = ret_dict['gotHormones_dt']      * ret_dict['metabolic_weight']
        ret_dict['hormone_effect_dt_dmi'] = (ret_dict['hormone_effect'] / ret_dict['day_diff']) * ret_dict['total_dmi']

        ret_dict['hasBEF_dmi_dt_log'] = (
            np.log((int(ret_dict['hasBEF']) / ret_dict['day_diff']) * ret_dict['total_dmi'])
            if ret_dict['hasBEF_dmi_dt'] != 0 else 0
        )
        ret_dict['hasBEF_dmi']           = int(ret_dict['hasBEF']) * ret_dict['total_dmi']
        ret_dict['hasBEF_dmi_dt_2']      = ret_dict['hasBEF_dmi_dt'] ** 2
        ret_dict['hasBEF_ddmi']          = int(ret_dict['hasBEF']) * ret_dict['total_dmi']
        ret_dict['gotDewormed_dt']        = int(ret_dict['gotDewormed'])      / ret_dict['day_diff']
        ret_dict['DaysSinceDewormed_dt']  = int(ret_dict['DaysSinceDewormed'])/ ret_dict['day_diff']

        if (ret_dict.get('pred_adgLatest_average') is not None and
                (ret_dict['pred_adgLatest_average'] > 1.5 or ret_dict['pred_adgLatest_average'] < 0.6)):
            with open('check.txt', 'a', encoding='utf-8') as f:
                f.write(json.dumps(ret_dict, indent=2))
                f.write('\n' + '-' * 80 + '\n')

        return ret_dict
