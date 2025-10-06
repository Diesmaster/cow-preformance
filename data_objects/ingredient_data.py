import json
from models.models import Models  # Use Models to reference dynamic ingredient fields
from utils.model_utils import ModelUtils
from types import NoneType

class IngredientData:
	"""
	A class representing an Ingredient with specific attributes as fields.
	This class is designed as a simple object to hold ingredient data.
	"""

	def __init__(self, ingredient_data):
		"""
		Initializes an instance of IngredientData with the provided dictionary.

		Args:
			ingredient_data (dict): A dictionary containing ingredient attributes.

		Raises:
			ValueError: If the dictionary is missing any required fields or contains None values.
		"""
		# Validate the input dictionary using Models.ingredient_fields
		if not self.validate(ingredient_data):
			raise ValueError(f"Invalid ingredient data. Missing or None values for required fields.")

		# Dynamically assign fields from the Models definition
		for field in Models.ingredient_fields.keys():
			setattr(self, field, ingredient_data.get(field))

	def __str__(self):
		"""
		Returns the JSON representation of the IngredientData object as a string.

		Returns:
			str: A JSON string representation of the object.
		"""
		return self.to_json()

	def to_dict(self):
		"""
		Converts the IngredientData object to a dictionary.

		Returns:
			dict: A dictionary representation of the IngredientData object.
		"""
		return {field: getattr(self, field) for field in Models.ingredient_fields.keys()}

	def to_doc(self):
		"""
		Converts the IngredientData object to a dictionary, excluding any non-primitive attributes.
		
		Non-primitive attributes (classes, objects, etc.) are not included in the final dictionary.

		Returns:
			dict: A dictionary representation of the IngredientData object without non-primitive attributes.
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
		Converts the IngredientData object to a JSON string.

		Returns:
			str: A JSON string representation of the IngredientData object.
		"""
		return json.dumps(self.to_dict(), ensure_ascii=False)

	@staticmethod
	def validate(ingredient_data):
		"""
		Validates if the given dictionary contains all required fields and their values are not None.

		Args:
			ingredient_data (dict): A dictionary containing ingredient attributes.

		Returns:
			bool: True if all required fields are present and their types are correct, otherwise False.
		"""
		return ModelUtils.validate_entry(ingredient_data, Models.ingredient_fields)

	@staticmethod
	def filter_fields(ingredient_list, fields_to_keep=None):
		"""
		Filters a list of IngredientData objects, keeping only the specified fields in their dictionary representation.

		Args:
			ingredient_list (list): A list of IngredientData objects.
			fields_to_keep (list, optional): A list of fields to keep. Defaults to ['PAKAN', 'Harga', 'docId'].

		Returns:
			list: A list of filtered ingredient dictionaries.
		"""
		if fields_to_keep is None:
			fields_to_keep = ['PAKAN', 'Harga', 'docId']

		return [{key: ingredient.to_dict().get(key) for key in fields_to_keep} for ingredient in ingredient_list]

	@staticmethod
	def load(farm_id, ingredient_id, firebase_dispatcher):
	    """
	    Fetches an ingredient from Firebase using the FirebaseManager, initializes an IngredientData object, and returns it.

	    Args:
	        farm_id (str): The ID of the farm.
	        ingredient_id (str): The document ID of the ingredient.
	        firebase_manager (FirebaseManager): An instance of FirebaseManager for accessing Firebase.

	    Returns:
	        IngredientData: An IngredientData object if the ingredient is found, otherwise None.
	    """
	    try:
	        # Retrieve the ingredient using the FirebaseManager
	        ingredient_data = firebase_dispatcher.get_ingredients(ingredient_id, farm_id)

	        # Check if the ingredient data exists
	        if ingredient_data:
	            # Initialize and return the IngredientData object
	            return IngredientData(ingredient_data)
	        else:
	            print(f"No ingredient found for ID: {ingredient_id}")
	            return None

	    except Exception as e:
	        print(f"Error fetching ingredient with ID {ingredient_id}: {e}")
	        return None
