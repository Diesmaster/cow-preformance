import json

from datetime import datetime, timedelta

from models.models import Models  # Use Models to reference dynamic cow feed history fields
from utils.model_utils import ModelUtils
from utils.dict_utils import DictUtils
from utils.metric_utils import MetricUtils
from types import NoneType

from data_objects.feed_history_data import FeedHistoryData

class CowData:
	"""
	A class representing a Cow's feed history with specific attributes as fields.
	This class is designed as a simple object to hold cow data.
	"""

	def __init__(self, cow_data):
		"""
		Initializes an instance of CowData with the provided dictionary.

		Args:
			cow_data (dict): A dictionary containing cow feed history attributes.

		Raises:
			ValueError: If the dictionary is missing any required fields or contains None values.
		"""
		# Validate the input dictionary using Models.cow_feed_history_model
		if not self.validate(cow_data):
			raise ValueError(f"Invalid cow feed history data. Missing or None values for required fields.")

		# Dynamically assign fields from the Models definition
		for field in Models.cow_model.keys():
			setattr(self, field, cow_data.get(field))

	def __str__(self):
		"""
		Returns the JSON representation of the CowData object as a string.

		Returns:
			str: A JSON string representation of the object.
		"""
		return self.to_json()

	def to_dict(self):
		"""
		Converts the CowData object to a dictionary.

		Returns:
			dict: A dictionary representation of the CowData object.
		"""
		return {field: getattr(self, field) for field in Models.cow_model.keys()}

	def to_doc(self):
		"""
		Converts the CowData object to a dictionary, excluding any non-primitive attributes.
		
		Non-primitive attributes (classes, objects, etc.) are not included in the final dictionary.

		Returns:
			dict: A dictionary representation of the CowData object without non-primitive attributes.
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
		
		return data

	def to_json(self):
		"""
		Converts the CowData object to a JSON string.

		Returns:
			str: A JSON string representation of the CowData object.
		"""
		return json.dumps(self.to_dict(), ensure_ascii=False)

	# todo validate the date validation
	@staticmethod
	def validate(cow_data):
		"""
		Validates if the given dictionary contains all required fields and their values are not None.

		Args:
			cow_data (dict): A dictionary containing cow feed history attributes.

		Returns:
			bool: True if all required fields are present and their types are correct, otherwise False.
		"""
		return ModelUtils.validate_entry(cow_data, Models.cow_model)

	@staticmethod
	def filter_fields(cow_list, fields_to_keep=None):
	    """
	    Filters a list of CowData objects or dictionaries, keeping only the specified fields in their representation.

	    Args:
	        cow_list (list): A list of CowData objects or dictionaries.
	        fields_to_keep (list, optional): A list of fields to keep. Defaults to all fields from Models.cow_model.

	    Returns:
	        list: A list of filtered cow data dictionaries.
	    """
	    if fields_to_keep is None:
	        fields_to_keep = list(Models.cow_model.keys())

	    filtered_list = []
	    for cow in cow_list:
	        if isinstance(cow, dict):
	            # If the cow is already a dictionary, filter directly
	            filtered_list.append({key: cow.get(key) for key in fields_to_keep})
	        else:
	            # Otherwise, convert the object to a dictionary first
	            filtered_list.append({key: cow.to_dict().get(key) for key in fields_to_keep})

	    return filtered_list

	@staticmethod
	def load(farm_id, cow_id, firebase_manager):
	    """
	    Fetches a cow's feed history from Firebase using the FirebaseManager, initializes a CowData object, and returns it.

	    Args:
	        farm_id (str): The ID of the farm.
	        cow_id (str): The document ID of the cow's feed history.
	        firebase_manager (FirebaseManager): An instance of FirebaseManager for accessing Firebase.

	    Returns:
	        CowData: A CowData object if the feed history is found, otherwise None.
	    """
	    try:
	        # Retrieve the cow feed history using the FirebaseManager
	        cow_data = firebase_manager.get_cow_plain(farm_id, cow_id)

	        # Check if the cow data exists
	        if cow_data:
	            # Initialize and return the CowData object
	            return CowData(cow_data)
	        else:
	            print(f"No cow feed history found for ID: {cow_id}")
	            return None

	    except Exception as e:
	        print(f"Error fetching cow feed history with ID {cow_id}: {e}")
	        return None

	@staticmethod
	def filter_and_sort_cows(cow_dicts):
	    """
	    Filters an array of cow dictionaries to include only 'cattleId' and 'docId',
	    then sorts them alphabetically by 'cattleId'.

	    Args:
	        cow_dicts (list): List of dictionaries containing cow data.

	    Returns:
	        list: A sorted list of filtered cow dictionaries with 'cattleId' and 'docId'.
	    """
	    # Filter to keep only 'cattleId' and 'docId' keys
	    filtered_cows = [{'cattleId': cow.get('cattleId'), 'docId': cow.get('docId')} for cow in cow_dicts]

	    # Sort the filtered cows alphabetically by 'cattleId'
	    sorted_cows = sorted(filtered_cows, key=lambda cow: cow['cattleId'].lower() if cow['cattleId'] else '')

	    return sorted_cows

	def update_cow_feed_details(self, date, ration_details):
		"""
		Update the CowData object with the new rationDetails, rationDetailsUpdateDate, 
		and calculate the day difference between the previous and new rationDetailsUpdateDate.

		Args:
			date (str): The date when the rationDetails were updated (in 'YYYY-MM-DD' format).
			ration_details (dict): The updated rationDetails data to be applied to the CowData.

		Returns:
			int: The number of days between the old and new rationDetailsUpdateDate.

		Raises:
			ValueError: If the date is not in 'YYYY-MM-DD' format, if the ration_details is not a dictionary, 
						or if the date is not greater than the existing rationDetailsUpdateDate.
		"""
		try:
			# Ensure the input date is a valid date string
			if not isinstance(date, str):
				raise ValueError(f"Date must be a string in 'YYYY-MM-DD' format, but got {type(date)}")

			# Ensure the date follows 'YYYY-MM-DD' format
			try:
				input_date = datetime.strptime(date, '%Y-%m-%d')
			except ValueError:
				raise ValueError(f"Date must be in 'YYYY-MM-DD' format, but got {date}")

			# Ensure ration_details is a dictionary
			if not isinstance(ration_details, list):
				raise ValueError(f"rationDetails must be a dictionary, but got {type(ration_details)}")

			day_difference = 0  # Default to 0 if no previous update date exists

			# Check if rationDetailsUpdateDate exists and ensure the new date is greater
			if hasattr(self, 'rationDetailsUpdateDate') and self.rationDetailsUpdateDate:
				try:
					current_update_date = datetime.strptime(self.rationDetailsUpdateDate, '%Y-%m-%d')
				
				except ValueError:
					raise ValueError(f"Existing rationDetailsUpdateDate ({self.rationDetailsUpdateDate}) is not a valid date.")
			
				if input_date == current_update_date:
					self.rationDetails = ration_details
					return self.to_doc()

				if input_date < current_update_date:
					raise ValueError(f"New date ({date}) must be greater than the existing rationDetailsUpdateDate ({self.rationDetailsUpdateDate}).")
				
				# Calculate the day difference
				day_difference = (input_date - current_update_date).days	

			consumed_feed = {}
			FeedHistoryData.calc_diet_ingredients(consumed_feed, self.rationDetails, day_difference)

			actual_consumed_feed = {}

			for key in consumed_feed:
				if consumed_feed[key]['asFedIntakePerCow'] != 1:
					actual_consumed_feed[key] = consumed_feed[key]['asFedIntakePerCow']

			total_dm = 0

			for entry in self.rationDetails:
				total_dm += entry['dryMatterIntakePerCow']*day_difference

			actual_consumed_feed['total_dry_matter_intake'] = total_dm

			# Update the attributes
			self.rationDetails = ration_details  # Update rationDetails
			self.rationDetailsUpdateDate = date  # Update the date of the change

			self.totalFeedConsumption = DictUtils.sum_two_dicts(self.totalFeedConsumption, actual_consumed_feed)

			print(f"Updated rationDetails for CowData on {date} with new rationDetails.")
			print(f"Day difference between previous date ({current_update_date}) and new date ({date}): {day_difference} days")

			return self.to_doc()

		except Exception as e:
			raise ValueError(f"Error updating cow feed details: {e}")

	def get_total_dm_intake(self, current_date):
		"""
		Calculate the total dry matter intake (DMI) for the cow.

		The total DMI is calculated as the sum of the total dry matter intake 
		stored in totalFeedConsumption and the total dry matter intake from the current rationDetails.

		Returns:
			float: The total dry matter intake (DMI) for the cow.
		"""
		try:
			# Initialize the total DMI from the totalFeedConsumption
			total_dm = 0
			if not self.totalFeedConsumption == None:
				total_dm = self.totalFeedConsumption.get('TotalDryMatterIntake', 0)

			day_difference = (datetime.strptime(current_date, "%Y-%m-%d")  - datetime.strptime(self.rationDetailsUpdateDate, "%Y-%m-%d")).days

			# Sum up the dry matter intake from the current ration details
			for entry in self.rationDetails:
				if 'dryMatterIntakePerCow' in entry and isinstance(entry['dryMatterIntakePerCow'], (int, float)):
					total_dm += entry['dryMatterIntakePerCow']*day_difference
			
			return MetricUtils.cast_to_float(total_dm)  # Round to 2 decimal places for precision

		except Exception as e:
			raise ValueError(f"Error calculating total dry matter intake: {e}")


	def update_sales(self, sales_price, sales_date, insurance_payout):
	    """
	    Updates the sales information for the cow, including insurance payout.

	    Args:
	        sales_price (float): The sale price of the cow.
	        sales_date (str): The date of the sale in 'YYYY-MM-DD' format.
	        insurance_payout (float): The amount of insurance payout for the cow.

	    Returns:
	        dict: Updated cow data with sales information.

	    Raises:
	        ValueError: If sales_price, sales_date, or insurance_payout is not valid, or if required cow attributes are missing.
	    """
	    try:
	        # 1️⃣ Validate inputs
	        if not isinstance(sales_price, (int, float)) or sales_price < 0:
	            raise ValueError(f"Sales price must be a positive number, but got {sales_price}.")
	        
	        if not isinstance(insurance_payout, (int, float)) or insurance_payout < 0:
	            raise ValueError(f"Insurance payout must be a positive number, but got {insurance_payout}.")
	        
	        try:
	            # Validate and parse the sales date
	            sales_date_obj = datetime.strptime(sales_date, '%Y-%m-%d')
	        except ValueError:
	            raise ValueError(f"Sales date must be in 'YYYY-MM-DD' format, but got {sales_date}.")
	        
	        # 2️⃣ Validate cow attributes
	        entry_date = getattr(self, 'entryDate', None)
	        weight = getattr(self, 'weight', 0)
	        
	        if not entry_date:
	            raise ValueError(f"entryDate is missing for cow. Cannot calculate days on feed.")
	        
	        if weight <= 0:
	            raise ValueError(f"Weight must be a positive number, but got {weight}.")
	        
	        try:
	            entry_date_obj = datetime.strptime(entry_date, '%Y-%m-%d')
	        except ValueError:
	            raise ValueError(f"entryDate must be in 'YYYY-MM-DD' format, but got {entry_date}.")
	        
	        # Ensure sales date is after the entry date
	        if sales_date_obj <= entry_date_obj:
	            raise ValueError(f"Sales date ({sales_date}) must be after the entry date ({entry_date}).")
	        
	        # 3️⃣ Calculate derived metrics
	        days_on_feed = (sales_date_obj - entry_date_obj).days  # Calculate days on feed
	        sales_price_per_kilo = sales_price / weight if weight > 0 else 0  # Calculate sales price per kilo
	        total_sales = sales_price + insurance_payout  # Calculate total sales

	        # 4️⃣ Update cow data attributes
	        self.salesPrice = sales_price
	        self.salesDate = sales_date
	        self.insurancePayout = insurance_payout
	        self.daysOnFeed = days_on_feed
	        self.salesPricePerKilo = round(sales_price_per_kilo, 2)  # Round to 2 decimal places
	        self.totalSales = round(total_sales, 2)  # Round to 2 decimal places

	        # 5️⃣ Prepare the updated cow data to return (used to update Firestore if needed)
	        updated_cow_data = {
	            'salesPrice': self.salesPrice,
	            'salesDate': self.salesDate,
	            'insurancePayout': self.insurancePayout,
	            'daysOnFeed': self.daysOnFeed,
	            'salesPricePerKilo': self.salesPricePerKilo,
	            'totalSales': self.totalSales
	        }

	        print(f"Successfully updated sales data for cow: {updated_cow_data}")
	        return updated_cow_data

	    except Exception as e:
	        raise ValueError(f"Error updating sales for cow: {e}")
