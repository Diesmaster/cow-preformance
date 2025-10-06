import json
import re 

from datetime import datetime, timedelta


from data_models.models import Models  # Use Models to reference dynamic field structures
from utils.model_utils import ModelUtils
from utils.metric_utils import MetricUtils
from types import NoneType

class FeedHistoryData:
    """
    A class representing a Feed History with specific attributes as fields.
    The feed history contains an array of feed history entry objects.
    """

    def __init__(self, feed_history_data):
    	"""
    	Initializes an instance of FeedHistoryData with the provided dictionary.

    	Args:
    		feed_history_data (dict): A dictionary containing the feed history.

    	Raises:
    		ValueError: If the dictionary is missing any required fields or contains None values.
    	"""
    	# Validate the main feed history data
    	if not self.validate(feed_history_data):
    		raise ValueError(f"Invalid feed history data. Missing or None values for required fields.")

    	# Dynamically assign fields from the Models definition
    	for field in Models.feed_history_model.keys():
    		setattr(self, field, feed_history_data.get(field))

    	# Ensure each entry in the `data` array is validated using Models.feed_history_entry_model
    	self.data = [self._validate_and_init_entry(entry) for entry in self.data]



    def get_dry_matter_intake(self, start_date, end_date):
        """
        Calculate the number of days each entry in the feed history is active within the given date range.
        Prints how many days each entry contributes to the calculation.

        Args:
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.
        """
        try:
            # Convert start and end dates to datetime objects
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=1)

            # Total number of days in the range (inclusive)
            total_days = (end_date_obj - start_date_obj).days + 1
            total_dry_matter_intake = 0


            # Loop through each entry in the feed history data
            for i, entry in enumerate(self.data):
                # Extract the entry's date and convert to datetime
                entry_date_obj = datetime.strptime(entry['date'], '%Y-%m-%d')

                # Calculate the date of the next entry (if it exists) or use end_date
                if i + 1 < len(self.data):
                    next_entry_date_obj = datetime.strptime(self.data[i + 1]['date'], '%Y-%m-%d')
                else:
                    next_entry_date_obj = end_date_obj

                # Determine the entry's active period within the range
                entry_start = max(start_date_obj, entry_date_obj)
                entry_end = min(end_date_obj, next_entry_date_obj - timedelta(days=1))  # Exclude next entry's start date
                day_difference = max(0, (entry_end - entry_start).days + 1)  # Ensure non-negative and inclusive days

                # Print how many days this entry contributes
                if day_difference > 0:
                    total_dry_matter_intake += day_difference*entry['dryMatterIntakePerCow']

            return total_dry_matter_intake
        except Exception as e:
            raise ValueError(f"Error calculating dry matter intake: {e}")

    def change_price_feed_history(self, batch_id, element_id, old_price, new_price):
        """
        Updates the total and daily feed cost in the feed history when the price of a batch changes.

        Args:
            batch_id (str): The ID of the batch whose price is changing.
            element_id (str): The ID of the ingredient or element associated with the batch.
            old_price (float): The old price per unit of the batch.
            new_price (float): The new price per unit of the batch.

        Returns:
            None: Updates `self.data` in place.
        """
        if not isinstance(batch_id, str) or not isinstance(element_id, str):
            raise ValueError("batch_id and element_id must be strings.")
        if not isinstance(old_price, (int, float)) or not isinstance(new_price, (int, float)):
            raise ValueError("old_price and new_price must be numeric values.")

        for x in range(len(self.data)):
            if element_id in self.data[x].get("batches", {}):
                if batch_id in self.data[x]["batches"][element_id]:
                    amount = self.data[x]["batches"][element_id][batch_id]
                    old_total_price = amount * old_price
                    new_total_price = amount * new_price

                    self.data[x]["totalFeedCost"] -= old_total_price
                    self.data[x]["totalFeedCost"] += new_total_price

                    # Calculate daily feed cost
                    now = datetime.today()
                    try:
                        feed_date = datetime.strptime(self.data[x]["date"], "%Y-%m-%d")  # Ensure correct format
                        diff = (now - feed_date).days + 1  # Ensure minimum of 1 day

                        # Avoid division by zero
                        if diff > 0:
                            self.data[x]["dailyFeedCost"] = self.data[x]["totalFeedCost"] / diff
                        else:
                            self.data[x]["dailyFeedCost"] = self.data[x]["totalFeedCost"]  # If same day, total cost applies
                    except ValueError:
                        raise ValueError(f"Invalid date format in feed history entry: {self.data[x].get('date')}")

    def get_ingredients(self):
        """
        Extracts ingredients from the feed history's latest rationDetails.
        If an element has rationDetails that is not empty or None, it recursively extracts ingredients.
        If rationDetails is empty or None, it returns that element.
        """
        def extract_ingredients(ration_list):
            ingredients = []
            for entry in ration_list:
                if not isinstance(entry, dict):
                    continue  # Skip invalid entries
                if entry.get("rationDetails") not in [None, []]:
                    # Recursively extract ingredients from rationDetails
                    ingredients.extend(extract_ingredients(entry["rationDetails"]))
                else:
                    # If rationDetails is None or empty, add the entry itself
                    ingredients.append(entry)
            return ingredients

        if not isinstance(self.data, list) or not self.data:
            raise ValueError("feed history data must contain a list of entries.")

        latest_entry = self.data[-1]  # Get the most recent feed history entry

        if "rationDetails" not in latest_entry or not isinstance(latest_entry["rationDetails"], list):
            raise ValueError("Latest feed history entry must contain 'rationDetails' as a list.")

        return extract_ingredients(latest_entry["rationDetails"])


    def add_batch_consumption(self, item_doc_id, batch, amount):
        """
        Updates the latest feed history entry to track batch consumption and calculates daily feed cost.

        Args:
            item_doc_id (str): The item ID associated with the batch.
            batch (dict): The batch dictionary containing 'docId' and pricing details.
            amount (float): The amount of feed consumed from the batch.

        Returns:
            None: Updates `self.data` in place.
        """
        if not isinstance(item_doc_id, str):
            raise ValueError("item_doc_id must be a string.")
        if not isinstance(batch, dict) or "docId" not in batch:
            raise ValueError("batch must be a dictionary with a 'docId' key.")
        if not isinstance(amount, (int, float)) or amount < 0:
            raise ValueError("amount must be a positive number.")
        if not self.data or not isinstance(self.data, list):
            raise ValueError("Feed history data must be a non-empty list.")

        # Get the latest feed history entry
        latest_entry = self.data[-1]

        # Ensure 'batches' key exists
        latest_entry.setdefault('batches', {})
        latest_entry['batches'].setdefault(item_doc_id, {})

        # Update batch consumption
        batch_doc_id = batch['docId']
        latest_entry['batches'][item_doc_id][batch_doc_id] = (
            latest_entry['batches'][item_doc_id].get(batch_doc_id, 0) + amount
        )

        # Ensure 'totalFeedCost' exists
        latest_entry.setdefault('totalFeedCost', 0)
        price = batch.get('actualPricePerUnit') or batch.get('pricePerUnit', 0)

        latest_entry['totalFeedCost'] += amount * price

        # Calculate daily feed cost
        now = datetime.today()
        try:
            feed_date = datetime.strptime(latest_entry['date'], "%Y-%m-%d")  # Ensure date format is correct
            diff = (now - feed_date).days + 1  # Ensure at least 1 day difference

            # Avoid division by zero
            if diff > 0:
                latest_entry['dailyFeedCost'] = latest_entry['totalFeedCost'] / diff
            else:
                latest_entry['dailyFeedCost'] = latest_entry['totalFeedCost']  # Set to total if same day

            self.data[-1] = latest_entry

        except ValueError:
            raise ValueError(f"Invalid date format in feed history entry: {latest_entry.get('date')}")


    @staticmethod
    def calc_diet_ingredients(dic, entry, days, parent=False):

        #print(f"number of days: {days}")


        for ration_entry in entry:
            if not ration_entry['name'] in dic:
                dic[ration_entry['name']] = {}
                dic[ration_entry['name']]['asFedIntakePerCow'] = 0
                dic[ration_entry['name']]['dryMatterIntakePerCow'] = 0

                if ration_entry['asFedIntakePerCow'] > 0:
                    #print(ration_entry['name'])
                    #print(ration_entry['asFedIntakePerCow'])
                    #print(dic[ration_entry['name']]['asFedIntakePerCow'])
                    dic[ration_entry['name']]['asFedIntakePerCow'] = ration_entry['asFedIntakePerCow']*days
                    #print(dic[ration_entry['name']]['asFedIntakePerCow']) 
                
                if ration_entry['dryMatterIntakePerCow'] > 0:
                    dic[ration_entry['name']]['dryMatterIntakePerCow'] = ration_entry['dryMatterIntakePerCow']*days
            else:
                if ration_entry['asFedIntakePerCow'] > 0:
                    #print("##############")
                    #print(ration_entry['name'])
                    #print(ration_entry['asFedIntakePerCow'])
                    #print(dic[ration_entry['name']]['asFedIntakePerCow'])
                    dic[ration_entry['name']]['asFedIntakePerCow'] += ration_entry['asFedIntakePerCow']*days 
                    #print("after")
                    #print(dic[ration_entry['name']]['asFedIntakePerCow'])
                    #print("#####################")

                if ration_entry['dryMatterIntakePerCow'] > 0: 
                    dic[ration_entry['name']]['dryMatterIntakePerCow'] += ration_entry['dryMatterIntakePerCow']*days

            if not ration_entry['rationDetails'] == None and not ration_entry['rationDetails'] == []:
                dic = FeedHistoryData.calc_diet_ingredients(dic, ration_entry['rationDetails'], days, True)

        return dic

    def get_diet(self, start_date, end_date):
        """
        Calculate the number of days each entry in the feed history is active within the given date range.
        Prints how many days each entry contributes to the calculation.

        Args:
        	start_date (str): The start date in 'YYYY-MM-DD' format.
        	end_date (str): The end date in 'YYYY-MM-DD' format.
        """
        try:
            # Convert start and end dates to datetime objects
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')  - timedelta(days=1)
            

            # Total number of days in the range (inclusive)
            total_days = (end_date_obj - start_date_obj).days  + 1
            diet = {}

            # Loop through each entry in the feed history data
            for i, entry in enumerate(self.data):
                # Extract the entry's date and convert to datetime
                entry_date_obj = datetime.strptime(entry['date'], '%Y-%m-%d')

                # Calculate the date of the next entry (if it exists) or use end_date
                if i + 1 < len(self.data):
                    next_entry_date_obj = datetime.strptime(self.data[i + 1]['date'], '%Y-%m-%d')
                else:
                    next_entry_date_obj = end_date_obj

                # Determine the entry's active period within the range
                entry_start = max(start_date_obj, entry_date_obj)
                entry_end = min(end_date_obj, next_entry_date_obj - timedelta(days=1))  # Exclude next entry's start date
                day_difference = max(0, (entry_end - entry_start).days + 1)  # Ensure non-negative and inclusive days

                # Print how many days this entry contributes
                if day_difference > 0:
                    diet = FeedHistoryData.calc_diet_ingredients(diet, entry['rationDetails'], day_difference)

            return diet
        except Exception as e:
            raise ValueError(f"Error calculating dry matter intake: {e}")


    def _validate_and_init_entry(self, entry):
    	"""
    	Validates and initializes a feed history entry.

    	Args:
    		entry (dict): A dictionary representing a single feed history entry.

    	Returns:
    		dict: A validated feed history entry.

    	Raises:
    		ValueError: If the entry does not match the feed_history_entry_model.
    	"""
    	if not ModelUtils.validate_entry(entry, Models.feed_history_entry_model):
    		raise ValueError(f"Invalid feed history entry data. Missing or None values for required fields: {entry}")
    	
    	date = entry.get('date')
    	if date and not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
    		raise ValueError(f"Invalid date format for entry. Expected YYYY-MM-DD but got: {date}")

    	for detail in entry['rationDetails']:
    		ModelUtils.validate_entry(detail, Models.ration_detail_fields)

    	return entry

    def __str__(self):
    	"""
    	Returns the JSON representation of the FeedHistoryData object as a string.

    	Returns:
    		str: A JSON string representation of the object.
    	"""
    	return self.to_json()

    def to_ingredient_history(self):
    	ret = []

    	for entry in self.data:
    		dic = {}
    		dic = FeedHistoryData.calc_diet_ingredients(dic, entry['rationDetails'], 1)			
    		dic['date'] = entry['date']
    		dic['dryMatterIntakePerCow'] = MetricUtils.cast_to_float(entry['dryMatterIntakePerCow'])

    		ret.append(dic)

    	return ret

    def to_dict(self):
    	"""
    	Converts the FeedHistoryData object to a dictionary.

    	Returns:
    		dict: A dictionary representation of the FeedHistoryData object.
    	"""
    	return {
    		'data': self.data
    	}

    def to_doc(self):
    	"""
    	Converts the FeedHistoryData object to a dictionary, excluding any non-primitive attributes.
    	
    	Non-primitive attributes (classes, objects, etc.) are not included in the final dictionary.

    	Returns:
    		dict: A dictionary representation of the FeedHistoryData object without non-primitive attributes.
    	"""
    	def is_primitive(value):
    		"""Check if the value is a primitive type (str, int, float, bool, None, list, dict, or tuple)."""
    		return isinstance(value, (str, int, float, bool, NoneType, list, dict, tuple))

    	# Filter non-primitive attributes and convert to dictionary
    	data = {
    		field: value 
    		for field, value in self.__dict__.items() 
    		if is_primitive(value) 
    	}

    	# Ensure the `data` list is converted properly
    	data['data'] = [entry for entry in self.data]
    	
    	return data

    def to_json(self):
    	"""
    	Converts the FeedHistoryData object to a JSON string.

    	Returns:
    		str: A JSON string representation of the FeedHistoryData object.
    	"""
    	return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def create_feed_history_entry(**kwargs):
        """
        Creates a ration detail dictionary dynamically based on the ration_detail_fields.

        Args:
            kwargs: Key-value pairs for the ration detail fields.

        Returns:
            dict: A validated ration detail dictionary.

        Raises:
            ValueError: If any required field is missing or invalid.
        """
        return [ModelUtils.create_object(Models.feed_history_entry_model, **kwargs)]

    @staticmethod
    def validate(feed_history_data):
    	"""
    	Validates if the given dictionary contains all required fields and their values are not None.

    	Args:
    		feed_history_data (dict): A dictionary containing the feed history.

    	Returns:
    		bool: True if all required fields are present and their types are correct, otherwise False.
    	"""
    	# Validate each entry in the `data` array
    	data_entries = feed_history_data.get('data', [])
    	return ModelUtils.validate_list_of_objects(data_entries, Models.feed_history_entry_model)

    @staticmethod
    def validate_entry(feed_history_entry):
    	return ModelUtils.validate_entry(feed_history_entry, Models.feed_history_entry_model)

    @staticmethod
    def filter_fields(feed_history_list, fields_to_keep=None):
    	"""
    	Filters a list of FeedHistoryData objects, keeping only the specified fields in their dictionary representation.

    	Args:
    		feed_history_list (list): A list of FeedHistoryData objects.
    		fields_to_keep (list, optional): A list of fields to keep. Defaults to all fields.

    	Returns:
    		list: A list of filtered feed history dictionaries.
    	"""
    	if fields_to_keep is None:
    		fields_to_keep = list(Models.feed_history_model.keys())

    	return [{key: feed_history.to_dict().get(key) for key in fields_to_keep} for feed_history in feed_history_list]

    @staticmethod
    def filter_ingredients(ingredient_history):
        """
        Filters out ingredients from the feed history where `asFedIntakePerCow` is -1.

        Args:
            ingredient_history (list): A list of dictionaries representing the ingredient history.

        Returns:
            list: The filtered ingredient history with unwanted entries removed.
        """
        for entry in ingredient_history:
            for key in list(entry.keys()):  # Iterate over a copy of the keys to safely delete
                if isinstance(entry[key], dict):
                    if 'asFedIntakePerCow' in entry[key]:
                        if entry[key]['asFedIntakePerCow'] == -1:
                            del entry[key]  # Remove the key if the condition is met

        return ingredient_history


    @staticmethod
    def load(farm_id, feed_history_id, firebase_manager):
        """
        Fetches a feed history from Firebase using the FirebaseManager, initializes a FeedHistoryData object, and returns it.

        Args:
            farm_id (str): The ID of the farm.
            feed_history_id (str): The document ID of the feed history entry.
            firebase_manager (FirebaseManager): An instance of FirebaseManager for accessing Firebase.

        Returns:
            FeedHistoryData: A FeedHistoryData object if the feed history is found, otherwise None.
        """
        try:
            # Retrieve the feed history using the FirebaseManager
            feed_history_data = {}
            feed_history_data['data'] = firebase_manager.get_cow_feedhistory(farm_id, feed_history_id)

            # Check if the feed history data exists
            if feed_history_data:
                # Initialize and return the FeedHistoryData object

                feed_his = FeedHistoryData(feed_history_data)
                return feed_his
            else:
                print(f"No feed history found for ID: {feed_history_id}")
                return None

        except Exception as e:
            print(f"Error fetching feed history with ID {feed_history_id}: {e}")
            return None
