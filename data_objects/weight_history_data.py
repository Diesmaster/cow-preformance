import json
import re

from datetime import datetime, timedelta
from data_models.models import Models  # Use Models to reference dynamic field structures
from utils.model_utils import ModelUtils
from utils.metric_utils import MetricUtils
from types import NoneType


class WeightHistoryData:
    """
    A class representing the Weight History of a cow, with specific attributes as fields.
    The weight history contains an array of weight history entry objects.
    """

    def __init__(self, weight_history_data):
        """
        Initializes an instance of WeightHistoryData with the provided dictionary.

        Args:
            weight_history_data (dict): A dictionary containing the weight history.

        Raises:
            ValueError: If the dictionary is missing any required fields or contains None values.
        """
        # Validate the main weight history data
        if not self.validate(weight_history_data):
            raise ValueError(f"Invalid weight history data. Missing or None values for required fields.")

        # Dynamically assign fields from the Models definition
        for field in Models.weight_history_model.keys():
            setattr(self, field, weight_history_data.get(field))

        # Ensure each entry in the `data` array is validated using Models.weight_history_entry_model
        self.data = [self._validate_and_init_entry(entry) for entry in self.data]

    def _validate_and_init_entry(self, entry):
        """
        Validates and initializes a weight history entry.

        Args:
            entry (dict): A dictionary representing a single weight history entry.

        Returns:
            dict: A validated weight history entry.

        Raises:
            ValueError: If the entry does not match the weight_history_entry_model.
        """
        if not ModelUtils.validate_entry(entry, Models.weight_history_entry_model):
            raise ValueError(f"Invalid weight history entry data. Missing or None values for required fields: {entry}")
        
        date = entry.get('date')
        if date and not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
            raise ValueError(f"Invalid date format for entry. Expected YYYY-MM-DD but got: {date}")
        
        return entry

    @staticmethod
    def init_weight_history_entry(update_date, new_weight):
        """
        Creates a new weight history entry based on the given parameters.

        Args:
            last_date (str): The last recorded date in 'YYYY-MM-DD' format.
            update_date (str): The update date for the new entry in 'YYYY-MM-DD' format.
            last_weight (float): The last recorded weight of the cow.
            new_weight (float): The updated weight of the cow.
            dm_intake (float): The dry matter intake during the update period.
            entry_date (str): The date when the cow was first weighed (entry date) in 'YYYY-MM-DD' format.
            entry_weight (float): The initial entry weight of the cow.
            total_dm_intake (float): The total dry matter intake for the cow since entry.

        Returns:
            dict: A dictionary representing the new weight history entry.
        
        Raises:
            ValueError: If the input parameters are invalid or if the calculated values are incorrect.
        """
        try:
            weight_rounded = MetricUtils.cast_to_float(new_weight)

            # Create the weight history entry using ModelUtils.create_object
            weight_history_entry = ModelUtils.create_object(
                Models.weight_history_entry_model,
                date=update_date,
                weight=weight_rounded,
                adgLatest=MetricUtils.cast_to_float(0),
                adgAverage=MetricUtils.cast_to_float(0),
                fcrLatest=MetricUtils.cast_to_float(0),
                fcrAverage=MetricUtils.cast_to_float(0)
            )

            print(f"Created weight history entry: {weight_history_entry}")
            return WeightHistoryData({'data':[weight_history_entry]})

        except Exception as e:
            raise ValueError(f"Error creating weight history entry: {e}")

    @staticmethod
    def create_weight_history_entry(last_date, update_date, last_weight, new_weight, dm_intake, entry_date, entry_weight, total_dm_intake):
        """
        Creates a new weight history entry based on the given parameters.

        Args:
            last_date (str): The last recorded date in 'YYYY-MM-DD' format.
            update_date (str): The update date for the new entry in 'YYYY-MM-DD' format.
            last_weight (float): The last recorded weight of the cow.
            new_weight (float): The updated weight of the cow.
            dm_intake (float): The dry matter intake during the update period.
            entry_date (str): The date when the cow was first weighed (entry date) in 'YYYY-MM-DD' format.
            entry_weight (float): The initial entry weight of the cow.
            total_dm_intake (float): The total dry matter intake for the cow since entry.

        Returns:
            dict: A dictionary representing the new weight history entry.
        
        Raises:
            ValueError: If the input parameters are invalid or if the calculated values are incorrect.
        """
        try:
            # Validate date format for last_date, update_date, and entry_date
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', last_date):
                raise ValueError(f"Invalid date format for last_date. Expected YYYY-MM-DD but got: {last_date}")
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', update_date):
                raise ValueError(f"Invalid date format for update_date. Expected YYYY-MM-DD but got: {update_date}")
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', entry_date):
                raise ValueError(f"Invalid date format for entry_date. Expected YYYY-MM-DD but got: {entry_date}")

            # Convert dates to datetime objects for calculations
            last_date_obj = datetime.strptime(last_date, '%Y-%m-%d')
            update_date_obj = datetime.strptime(update_date, '%Y-%m-%d')
            entry_date_obj = datetime.strptime(entry_date, '%Y-%m-%d')

            # Calculate the number of days between the last and update dates
            days_between = (update_date_obj - last_date_obj).days
            if days_between <= 0:
                raise ValueError(f"Update date ({update_date}) must be later than last date ({last_date}).")

            # Calculate the total number of days since entry
            total_days_on_feed = (update_date_obj - entry_date_obj).days
            if total_days_on_feed <= 0:
                raise ValueError(f"Update date ({update_date}) must be later than entry date ({entry_date}).")

            # Calculate ADG (Average Daily Gain)
            weight_gain_latest = new_weight - last_weight
            weight_gain_total = new_weight - entry_weight

            # Calculate ADG (Average Daily Gain)
            adg_latest = weight_gain_latest / days_between if days_between > 0 else 0
            adg_average = weight_gain_total / total_days_on_feed if total_days_on_feed > 0 else 0

            # Calculate FCR (Feed Conversion Ratio)
            fcr_latest =  (weight_gain_latest / dm_intake) * 100
            fcr_average = (weight_gain_total / total_dm_intake) * 100 

            # Round values for consistency and precision
            adg_latest = MetricUtils.cast_to_float(adg_latest)
            adg_average = MetricUtils.cast_to_float(adg_average)
            fcr_latest = MetricUtils.cast_to_float(fcr_latest)
            fcr_average = MetricUtils.cast_to_float(fcr_average)
            weight_rounded = MetricUtils.cast_to_float(new_weight)

            # Create the weight history entry using ModelUtils.create_object
            weight_history_entry = ModelUtils.create_object(
                Models.weight_history_entry_model,
                date=update_date,
                weight=weight_rounded,
                adgLatest=adg_latest,
                adgAverage=adg_average,
                fcrLatest=fcr_latest,
                fcrAverage=fcr_average
            )

            print(f"Created weight history entry: {weight_history_entry}")
            return WeightHistoryData({'data':[weight_history_entry]})

        except Exception as e:
            raise ValueError(f"Error creating weight history entry: {e}")

    def get_cow_historical_weight(self, date):
        """
        Finds the weight history entry where entry[i]['date'] < date < entry[i+1]['date'].

        Args:
            date (str): The target date in 'YYYY-MM-DD' format.

        Returns:
            dict: The weight history entry that matches the condition.

        Raises:
            ValueError: If the date is not in the correct format or no entry is found.
        """
        try:
            # Ensure the date is in the correct format
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
                raise ValueError(f"Invalid date format. Expected YYYY-MM-DD but got: {date}")

            target_date = datetime.strptime(date, '%Y-%m-%d')

            # Sort data to ensure it's in chronological order 
            sorted_data = self.data #sorted(self.data, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))

            for i in range(len(sorted_data) - 1):
                entry_date = datetime.strptime(sorted_data[i]['date'], '%Y-%m-%d')
                next_entry_date = datetime.strptime(sorted_data[i + 1]['date'], '%Y-%m-%d')

                if entry_date <= target_date < next_entry_date:
                    return sorted_data[i]

            # If no entry is found, return the most recent entry that is before the given date
            latest_entry = max(
                (entry for entry in sorted_data if datetime.strptime(entry['date'], '%Y-%m-%d') <= target_date),
                key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'),
                default=None
            )

            if latest_entry:
                return latest_entry

            raise ValueError(f"No historical weight entry found for date: {date}")
        
        except Exception as e:
            raise ValueError(f"Error retrieving historical weight for date {date}: {e}")



    def to_dict(self):
        """Converts the WeightHistoryData object to a dictionary."""
        return {'data': self.data}

    def to_doc(self):
        """Prepares the WeightHistoryData object for Firestore."""
        return {'data': [entry for entry in self.data]}

    def to_json(self):
        """Converts the WeightHistoryData object to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def validate(weight_history_data):
        """
        Validates if the given dictionary contains all required fields and their values are not None.

        Args:
            weight_history_data (dict): A dictionary containing the weight history.

        Returns:
            bool: True if all required fields are present and their types are correct, otherwise False.
        """
        data_entries = weight_history_data.get('data', [])
        return ModelUtils.validate_list_of_objects(data_entries, Models.weight_history_entry_model)

    @staticmethod
    def load(farm_id, weight_history_id, firebase_manager):
        """
        Fetches a weight history from Firebase and initializes a WeightHistoryData object.

        Args:
            farm_id (str): The ID of the farm.
            weight_history_id (str): The document ID of the weight history.
            firebase_manager (FirebaseManager): An instance of FirebaseManager for accessing Firebase.

        Returns:
            WeightHistoryData: A WeightHistoryData object if the weight history is found, otherwise None.
        """
        try:
            weight_history_data = {'data': firebase_manager.get_weight_history(farm_id, weight_history_id)}
            if weight_history_data:
                return WeightHistoryData(weight_history_data)
            else:
                print(f"No weight history found for ID: {weight_history_id}")
                return None
        except Exception as e:
            raise ValueError(f"Error fetching weight history with ID {weight_history_id}: {e}")

    @staticmethod
    def filter_fields(weight_history_list, fields_to_keep=None):
        """
        Filters a list of WeightHistoryData objects, keeping only specified fields.

        Args:
            weight_history_list (list): A list of WeightHistoryData objects.
            fields_to_keep (list, optional): Fields to keep.

        Returns:
            list: A list of filtered weight history dictionaries.
        """
        if fields_to_keep is None:
            fields_to_keep = list(Models.weight_history_model.keys())
        return [{key: history.to_dict().get(key) for key in fields_to_keep} for history in weight_history_list]
