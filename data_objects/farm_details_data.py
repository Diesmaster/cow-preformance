import json
import re

from types import NoneType
from datetime import datetime

from models.models import Models  # Use Models to reference dynamic farm fields
from utils.model_utils import ModelUtils
from utils.metric_utils import MetricUtils
from utils.feed_utils import FeedUtils
from utils.dict_utils import DictUtils



class FarmDetailsData:
	"""
	A class representing Farm Details with specific attributes as fields.
	This class dynamically initializes fields from the farm model.
	"""

	def __init__(self, farm_data):
		"""
		Initializes an instance of FarmDetailsData with the provided dictionary.

		Args:
			farm_data (dict): A dictionary containing farm details attributes.

		Raises:
			ValueError: If the dictionary is missing any required fields or contains None values.
		"""
		# Validate the input dictionary using the farm details model

		self.model = Models.get_farm_model()

		if not self.validate_quick(farm_data):
			raise ValueError("Invalid farm data. Missing or None values for required fields.")

		# Dynamically assign fields from the Models.get_farm_model definition
		for field in self.model.keys():
			setattr(self, field, farm_data.get(field))

	def __str__(self):
		"""
		Returns the JSON representation of the FarmDetailsData object as a string.

		Returns:
			str: A JSON string representation of the object.
		"""
		return self.to_json()

	def to_dict(self):
		"""
		Converts the FarmDetailsData object to a dictionary.

		Returns:
			dict: A dictionary representation of the FarmDetailsData object.
		"""
		return {field: getattr(self, field) for field in self.model.keys()}

	def to_doc(self):
		"""
		Converts the FarmDetailsData object to a dictionary, excluding any non-primitive attributes.

		Returns:
			dict: A dictionary representation of the FarmDetailsData object without non-primitive attributes.
		"""
		def is_primitive(value):
			"""Check if the value is a primitive type."""
			return isinstance(value, (str, int, float, bool, NoneType, list, dict, tuple))

		return {field: value for field, value in self.__dict__.items() if field != 'model' and is_primitive(value)}

	def to_json(self):
		"""
		Converts the FarmDetailsData object to a JSON string.

		Returns:
			str: A JSON string representation of the FarmDetailsData object.
		"""
		return json.dumps(self.to_dict(), ensure_ascii=False)

	def validate_quick(self, farm_data):
		"""
		Validates if the given dictionary contains all required fields and their values are not None.

		Args:
			farm_data (dict): A dictionary containing farm details attributes.

		Returns:
			bool: True if all required fields are present and their types are correct, otherwise False.
		"""
		return ModelUtils.validate_entry(farm_data, self.model)

	@staticmethod
	def validate(farm_data):
		"""
		Validates if the given dictionary contains all required fields and their values are not None.

		Args:
			farm_data (dict): A dictionary containing farm details attributes.

		Returns:
			bool: True if all required fields are present and their types are correct, otherwise False.
		"""
		return ModelUtils.validate_entry(farm_data, Models.get_farm_model())

	@staticmethod
	def load(farm_id, doc_id, firebase_manager):
		"""
		Fetches farm details from Firebase using the FirebaseManager and initializes a FarmDetailsData object.

		Args:
			farm_id (str): The ID of the farm.
			firebase_manager (FirebaseManager): An instance of FirebaseManager for accessing Firebase.

		Returns:
			FarmDetailsData: A FarmDetailsData object if the data is found, otherwise None.
		"""
		try:
			# Retrieve farm details using the FirebaseManager
			farm_data = firebase_manager.get_farm(farm_id)

			# Check if the farm data exists
			if farm_data:
				return FarmDetailsData(farm_data)
			else:
				print(f"No farm details found for ID: {farm_id}")
				return None

		except Exception as e:
			print(f"Error fetching farm details with ID {farm_id}: {e}")
			return None

	def get_weight_history_entry(self, date):
		farm_dict = self.to_dict()

		ret_dict = {}

		for field in Models.farm_weight_history_entry_model:
			if field == 'date':
				ret_dict[field] = date
			else:
				ret_dict[field] = farm_dict[field]

		return ret_dict

	def get_feed_history_entry(self, date):
		farm_dict = self.to_dict()

		ret_dict = {}

		for field in Models.farm_feed_history_entry_model:
			if field == 'date':
				ret_dict[field] = date
			else:
				ret_dict[field] = farm_dict[field]

		return ret_dict

	def update_with_new_cow(self, new_cow_data):
		"""
		Updates the current farm details instance with a new cow's data 
		by recalculating averages, totals, nominal values, and other fields.

		Args:
			new_cow_data (dict): Dictionary containing the new cow's details.

		Returns:
			None: Updates the instance in place.
		"""
		try: 
			# Increment the cow count
			self.numberOfCows = getattr(self, 'numberOfCows', 0) + 1

			print(new_cow_data)

			# Iterate over fields dynamically based on the cow_model
			for field in Models.cow_model:
				# Handle numeric fields for averages and totals
				if Models.cow_model[field] == (int, float):
					if isinstance(new_cow_data.get(field, ''), str):
						new_cow_data[field] = 0
					# Update average
					average_field = f'Average {field}'
					setattr(
						self,
						average_field,
						MetricUtils.add_to_average(
							self.numberOfCows - 1, getattr(self, average_field, 0), new_cow_data[field]
						)
					)
					# Update total
					total_field = f'Total {field}'
					MetricUtils.add_total_number_to_dict(self.__dict__, total_field, new_cow_data[field])

				# Handle nominal values
				elif (Models.cow_model[field] == str or Models.cow_model[field] == (str, NoneType)) and (
					field not in ["docId", "salesDate", "insurancePolis", "cattleId", "recordingDate"]
					and new_cow_data.get(field) != 'tbf'
				):
					breakdown_field = f'Breakdown {field}'
					MetricUtils.add_nominal_number_to_dict(self.__dict__, breakdown_field, new_cow_data[field])

					# Handle date fields dynamically
					if re.match(r'.*(date|Date).*', field, re.IGNORECASE):
						average_date_field = f'Average {field}'
						if not hasattr(self, average_date_field):
							setattr(self, average_date_field, [new_cow_data.get(field, '')])
						else:
							new_avg = MetricUtils.add_to_average_date(self.numberOfCows, getattr(self, average_date_field), new_cow_data.get(field, ''))
							setattr(self, average_date_field, new_avg)

				# Handle total feed consumption
				elif Models.cow_model[field] == (dict, NoneType) and field == 'totalFeedConsumption':
					if not new_cow_data.get('totalFeedConsumption') == None:
						FeedUtils.update_total_feed_consumption(new_cow_data.get('totalFeedConsumption', {}), self.__dict__)

				# Handle ration details
				elif Models.cow_model[field] == (list, NoneType) and field == 'rationDetails':
					if not new_cow_data.get('rationDetails') == None:
						FeedUtils.update_feed_ration(new_cow_data.get('rationDetails'), self.__dict__)
						for ration in new_cow_data['rationDetails']:
							MetricUtils.add_total_number_to_dict(
								self.rationDetails, 'dryMatterIntakePerDay', ration.get('dryMatterIntakePerCow', 0)
							)

			# Calculate average entry dates
			for key, value in self.__dict__.items():
				if isinstance(value, list) and re.match(r'^Average .*date.*$', key, re.IGNORECASE):
					print(value)
					setattr(self, key, MetricUtils.calculate_average_date(value))

		except Exception as e:
			if field:
				raise ValueError(f"Failed to update farm on field {field}: {str(e)}")
			elif key: 
				raise ValueError(f"Failed to update farm on key {key}: {str(e)}")
			else:
				raise ValueError(f"Failed to update farm on general: {str(e)}")

	def update_after_weighing(self, old_cow_data, updates, date):
		"""
		Updates farm details with new cow data after weighing by recalculating averages and totals.

		Args:
			old_cow_data (list): List of dictionaries representing previous cow details before updates.
			updates (dict): Dictionary with cow IDs as keys and updated attributes as values.
			date (str): The date of the weighing update.

		Returns:
			None: Updates the instance in place.
		"""
		# Step 1: Get the total number of cows in the farm
		cow_count = getattr(self, 'numberOfCows', 0)

		# Step 2: Iterate through the updates and recalculate averages and totals
		for old_cow in old_cow_data:
			old_cow = old_cow['cow']
			cow_id = old_cow['cattleId']
			if cow_id in updates:
				updated_data = updates[cow_id]

				# Handle numeric updates for averages
				for field in Models.cow_model:
					if Models.cow_model[field] == (int, float):
						prev_value = old_cow.get(field, 0)
						new_value = updated_data.get(field, 0)
						average_field = f"Average {field}"

						# Recalculate the average
						setattr(
							self,
							average_field,
							MetricUtils.update_average(
								old_value=prev_value,
								new_value=new_value,
								average=getattr(self, average_field, 0),
								number_of_cows=cow_count
							)
						)

						# Update totals
						total_field = f"Total {field}"
						setattr(
							self,
							total_field,
							MetricUtils.update_total(
								old_total=getattr(self, total_field, 0),
								prev_cow_value=prev_value,
								new_cow_value=new_value
							)
						)

	def update_feed_details(self, cows, current_date):
		"""
		Updates the total feed consumption and feed ration for the farm based on cow data.

		Args:
			cows (list): A list of cow dictionaries containing feed ration data.
			current_date (str): The current date in 'YYYY-MM-DD' format.
		"""
		# Step 1: Retrieve the last feed ration update date
		last_feed_update_date = getattr(self, 'feedRationUpdateDate', None)

		if not last_feed_update_date:
			last_feed_update_date = current_date

		# Step 2: Calculate new feed consumption for the period
		new_feed_consumption = FeedUtils.calculate_total(
			self.rationDetails, last_feed_update_date, current_date
		)

		# Step 3: Update total feed consumption
		self.totalFeedConsumption = DictUtils.sum_two_dicts(
			getattr(self, 'total_feed_consumption', {}),
			new_feed_consumption
		)

		# Step 4: Update feed ration using cow data
		for cow in cows:
			FeedUtils.update_feed_ration(cow['rationDetails'], self.rationDetails)

		# Step 5: Update the feed ration update date
		self.feedRationUpdateDate = current_date


	def remove_cow_data(self, cow_data):
		"""
		Removes a cow's data by recalculating averages, totals, nominal values, and entry date.

		Args:
			cow_data (dict): Dictionary containing the cow's details to be removed.
		"""
		# Decrement the cow count
		cow_count = self.numberOfCows - 1 if self.numberOfCows > 0 else 0

		# Update averages
		for field in Models.cow_model:
			if Models.cow_model[field] == (int, float):
				MetricUtils.remove_average_number_from_dict(
					self.__dict__, f'Average {field}', cow_data.get(field, 0), self.numberOfCows
				)

		# Update totals
		for field in Models.cow_model:
			if Models.cow_model[field] == (int, float):
				MetricUtils.remove_total_number_from_dict(
					self.__dict__, f'Total {field}', cow_data.get(field, 0)
				)

		# Update nominal values
		for field in Models.cow_model:
			if Models.cow_model[field] in [str, (str, NoneType)] and field != 'docId':
				MetricUtils.remove_nominal_number_from_dict(
					self.__dict__, f'Breakdown {field}', cow_data.get(field, '')
				)

		# Update average entry date
		if 'entryDate' in cow_data:
			self.average_entry_date = MetricUtils.remove_from_average_date(
				self.numberOfCows, getattr(self, 'average_entry_date', None), cow_data['entryDate']
			)

		# Update cow count
		self.numberOfCows = cow_count
	
	def update_measurements(self, old_cow_data, updates):
		"""
		Updates the averages and totals in the farm details based on updated cow measurement data.

		Args:
			old_cow_data (list): List of previous cow details before updates.
			updates (dict): Dictionary with cow IDs as keys and updated attributes as values.
		"""
		# Get the total number of cows in the farm
		cow_count = self.numberOfCows

		# Update averages for measurement fields
		for old_cow in old_cow_data:
			cow_id = old_cow['cattleId']
			if cow_id in updates:
				updated_data = updates[cow_id]

				for field in Models.cow_model:
					if Models.cow_model[field] == (int, float):  # Process only numeric measurement fields
						prev_value = old_cow.get(field, 0)
						new_value = updated_data.get(field, 0)
						average_field = f"Average {field}"

						self.__dict__[average_field] = MetricUtils.update_average(
							old_value=prev_value,
							new_value=new_value,
							average=self.__dict__.get(average_field, 0),
							number_of_cows=cow_count
						)

	def update_feed_subset(self, rations_to_update, current_date):
		"""
		Updates the feed rations by replacing specified old rations with new rations.

		Args:
			rations_to_update (list): A list of dictionaries containing 'old_ration' and 'new_ration'.
			current_date (str): The current date in 'YYYY-MM-DD' format.
		"""
		# Initialize the feed ration dictionary if not present
		if 'rationDetails' not in self.__dict__:
			self.rationDetails = {}

		# Get the last feed ration update date
		last_feed_update_date = self.__dict__.get('feedRationUpdateDate', None)
		if not last_feed_update_date:
			last_feed_update_date = current_date

		# Replace the old rations with the new ones
		test = FeedUtils.replace_rations(self.rationDetails, rations_to_update)

		self.rationDetails = test

		# Update the feed ration update date
		self.feed_ration_update_date = current_date


	@staticmethod
	def generate_farm_details(cows_dicts):
		"""
		Generates farm details by calculating averages, totals, nominal values,
		feed consumption, feed ration, and average entry date.

		Args:
			cows_dicts (list): A list of cow details dictionaries.

		Returns:
			dict: A dictionary containing the full farm details.
		"""
		# Step 1: Initialize the farm details dictionary
		cow_count = len(cows_dicts) + 1
		farm_details_dict = {
			'numberOfCows': cow_count,
			'totalFeedConsumption': {},
			'rationDetails': {}
		}

		# Step 2: Calculate averages, totals, and nominal values
		for cow in cows_dicts:
			for field in Models.cow_model:
				if Models.cow_model[field] == (int, float):
					if isinstance(cow.get(field, ''), str):
						cow[field] = 0
					MetricUtils.add_average_number_to_dict(
						farm_details_dict, f'Average {field}', cow.get(field, ''), cow_count
					)
					MetricUtils.add_total_number_to_dict(
						farm_details_dict, f'Total {field}', cow.get(field, '')
					)
				elif (Models.cow_model[field] == str or Models.cow_model[field] == (str, NoneType)) and (not field == "docId" and not cow[field] == 'tbf' and not field == 'salesDate' and not field == 'insurancePolis' and not field == 'cattleId' and not field == 'recordingDate'):
					MetricUtils.add_nominal_number_to_dict(
						farm_details_dict, f'Breakdown {field}', cow.get(field, '')
					)

					key_pattern = re.compile(r'.*(date|Date).*')
					if key_pattern.match(field):
						farm_key = 'Average ' + field
						if not farm_key in farm_details_dict:
							farm_details_dict[farm_key] = [cow.get(field, '')]
						else:
							farm_details_dict[farm_key].append(cow.get(field, ''))

				elif Models.cow_model[field] == (dict, NoneType):
					if field == 'totalFeedConsumption':
						FeedUtils.update_total_feed_consumption(cow.get('totalFeedConsumption', cow.get('total_feed_consumption')), farm_details_dict)
				elif Models.cow_model[field] == (list, NoneType):
					FeedUtils.update_feed_ration(cow.get('rationDetails'), farm_details_dict)
					    # Add the total dry matter intake to the feed ration dictionary
					
					for ration in cow['rationDetails']:
						MetricUtils.add_total_number_to_dict(farm_details_dict['rationDetails'], 'dryMatterIntakePerDay', ration['dryMatterIntakePerCow'])			

		for key, value in farm_details_dict.items():
			if isinstance(value, list) and re.match(r'^Average .*date.*$', key, re.IGNORECASE):
				if not None in value:
					farm_details_dict[key] = MetricUtils.calculate_average_date(value)
				else:
					farm_details_dict[key] = None
		farm_details_dict['feedRationUpdateDate'] = datetime.now().strftime('%Y-%m-%d')
		farm_details_dict['numberOfCows'] = cow_count - 1

		return FarmDetailsData(farm_details_dict)



