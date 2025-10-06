from __future__ import annotations
import json
from types import NoneType

from data_objects.ingredient_data import IngredientData
from utils.metric_utils import MetricUtils
from utils.feed_utils import FeedUtils
from utils.model_utils import ModelUtils
from data_models.models import Models 

class RecipeData:
	"""
	A class representing a Recipe with dynamically handled required fields.
	Includes nested IngredientData objects and validation for ingredients and additives.
	"""

	def __init__(self, recipe_data: dict):
	    """
	    Initializes an instance of RecipeData with the provided dictionary.

	    Args:
	        recipe_data (dict): A dictionary containing recipe attributes, ingredients, and additives.

	    Raises:
	        ValueError: If the dictionary is missing any required fields or contains None values,
	                    or if the ingredients/additives are not valid.
	    """
	    # Validate the input dictionary for missing or None required fields
	    missing_fields = [
	        field for field in Models.recipe_fields
	        if (field not in recipe_data or recipe_data[field] is None) and field != 'ingredients' and field != 'additives'
	    ]
	    if missing_fields:
	        raise ValueError(f"Missing or None values for required fields: {', '.join(missing_fields)}")

	    # Extract and validate ingredients from recipe_data
	    ingredients = recipe_data.get('ingredients', [])
	    if not ingredients == []:
		    if not ModelUtils.validate_list_of_objects(ingredients, Models.recipe_ingredient_fields):
		        raise ValueError(f"All ingredients must match the structure: {Models.recipe_ingredient_fields}")

	    # Extract and validate additives from recipe_data
	    additives = recipe_data.get('additives', [])
	    if not additives == []:
		    if additives and not ModelUtils.validate_list_of_objects(additives, Models.additive_fields):
		        raise ValueError(f"All additives must match the structure: {Models.additive_fields}")

	    # Assign ingredients and additives (default to empty lists if not provided)
	    self.ingredients = ingredients if ingredients is not None else []
	    self.additives = additives if additives is not None else []

	    # Dynamically assign other recipe fields from the recipe_data
	    for field in Models.recipe_fields:
	        if field not in ['ingredients', 'additives']:  # Skip 'ingredients' and 'additives' as they're handled separately
	            setattr(self, field, recipe_data.get(field))


	def create_ration_detail(self, **kwargs):
	    """
	    Creates a ration detail dictionary dynamically based on the ration_detail_fields.

	    Args:
	        kwargs: Key-value pairs for the ration detail fields.

	    Returns:
	        dict: A validated ration detail dictionary.

	    Raises:
	        ValueError: If any required field is missing or invalid.
	    """
	    return ModelUtils.create_object(Models.ration_detail_fields, **kwargs)

	## Conversion Functions

	def to_dict(self):
	    """
	    Converts the RecipeData object to a dictionary.

	    Returns:
	        dict: A dictionary representation of the RecipeData object.
	    """
	    data = {field: getattr(self, field) for field in Models.recipe_fields if field != 'ingredients'}

	    data['ingredients'] = []

	    for ingredient in self.ingredients:
	    	ingredient_obj = {}
	    	for key in Models.recipe_ingredient_fields:
	    		if key == 'ingredient':
	    			if 'ingredient' in ingredient:
		    			if ingredient[key] not in (None, ''):
		    				ingredient_obj[key] = ingredient[key].to_dict()
	    		else:
	    			ingredient_obj[key] = ingredient[key]		
	    	data['ingredients'].append(ingredient_obj)

	    # Handle additives
	    data['additives'] = []
	        
	    for ingredient in self.additives:
	    	ingredient_obj = {}
	    	for key in Models.additive_fields:
	    		if key == 'ingredient':
	    			if 'ingredient' in ingredient:
		    			if ingredient[key] not in (None, ''):
		    				ingredient_obj[key] = ingredient[key].to_dict()
	    		else:
	    			ingredient_obj[key] = ingredient[key]
	    	data['ingredients'].append(ingredient_obj)
	    

	    return data

	def to_doc(self):
	    """
	    Converts the RecipeData object to a dictionary, excluding any non-primitive attributes.
	    
	    Non-primitive attributes (classes, objects, etc.) are not included in the final dictionary.

	    Returns:
	        dict: A dictionary representation of the RecipeData object without non-primitive attributes.
	    """
	    def is_primitive(value):
	        """Check if the value is a primitive type (str, int, float, bool, None, list, dict, or tuple)."""
	        return isinstance(value, (str, int, float, bool, NoneType, list, dict))

	    # Handle main recipe fields (excluding ingredients and classes)
	    data = {
	        field: getattr(self, field) 
	        for field in Models.recipe_fields 
	        if field != 'ingredients' and is_primitive(getattr(self, field))
	    }

	    # Handle ingredients, excluding non-primitive attributes
	    data['ingredients'] = [
	        {
	            key: (
	                ingredient[key].to_dict() if key == 'ingredient' and ingredient[key] not in (None, '') else ingredient[key]
	            ) for key in Models.recipe_ingredient_fields if is_primitive(ingredient.get(key))
	        }
	        for ingredient in self.ingredients
	    ]

	    # Handle additives, excluding non-primitive attributes
	    data['additives'] = [
	        {
	            key: (
	                additive[key].to_dict() if key == 'ingredient' and additive[key] not in (None, '') else additive[key]
	            ) for key in Models.additive_fields if is_primitive(additive.get(key))
	        }
	        for additive in self.additives
	    ]

	    return data


	def to_json(self) -> str:
	    """
	    Converts the RecipeData object to a JSON string.

	    Returns:
	        str: A JSON string representation of the RecipeData object.
	    """
	    return json.dumps(self.to_dict(), ensure_ascii=False)

	
	def calculate_ration(self, total_group_weight, percentage_of_bodyweight, number_of_cows, total_dry_matter_intake=None):
	    """
	    Calculates the ration details for each ingredient and additive based on dry matter intake and as-fed basis.

	    Args:
	        total_group_weight (float): Total weight of the group in kg.
	        percentage_of_bodyweight (float): Percentage of body weight for feed.
	        number_of_cows (int): Number of cows in the group.

	    Returns:
	        dict: A dictionary containing ration details for each ingredient and additive.
	    """
	    # Initial total dry matter intake based on group weight and percentage of body weight
	    if total_dry_matter_intake is None:
	        total_dry_matter_intake = FeedUtils.calculate_total_dry_matter_intake(percentage_of_bodyweight, total_group_weight)

	    # Process ingredients
	    ingredient_details, ingredient_dry_matter_total = self._process_ingredients(
	        total_dry_matter_intake, number_of_cows
	    )

	    # Process additives
	    additive_details, additive_dry_matter_total = self._process_additives(
	        number_of_cows
	    )

	    # Add additive dry matter intake to total
	    total_dry_matter_intake += additive_dry_matter_total

	    # Combine ration details
	    ration_details = ingredient_details + additive_details

	    return {
	        'totalDryMatterIntake': MetricUtils.cast_to_float(total_dry_matter_intake),
	        'rationDetails': ration_details
	    }

	def _process_ingredients(self, total_dry_matter_intake, number_of_cows):
		"""
		Processes the ingredients to calculate their ration details.

		Args:
		    total_dry_matter_intake (float): Total dry matter intake for the group.
		    number_of_cows (int): Number of cows in the group.

		Returns:
		    tuple: A list of ingredient ration details and the total dry matter intake from ingredients.
		"""
		ingredient_details = []
		ingredient_dry_matter_total = 0

		for ingredient in self.ingredients:
			if isinstance(ingredient['ingredient'], RecipeData):
				# Ensure the 'recipe' field is set correctly
				if not ingredient.get('recipe', False):
					raise ValueError(f"The 'recipe' field must be True for RecipeData ingredients.")

				this_ingredient_intake = total_dry_matter_intake * ingredient.get('totalBK') / 100
				



				this_ingredient_intake_per_cow = MetricUtils.cast_to_float(this_ingredient_intake / number_of_cows)
				
				if this_ingredient_intake_per_cow > 0.5:
					this_ingredient_as_fed_per_cow = MetricUtils.get_feed_rounding(this_ingredient_intake_per_cow)
					this_ingredient_as_fed = MetricUtils.cast_to_float(this_ingredient_as_fed_per_cow * number_of_cows)


				# Recursive calculation for nested recipes
				nested_ration = ingredient['ingredient'].calculate_ration(
				    total_group_weight=None,
				    percentage_of_bodyweight=None,
				    number_of_cows=number_of_cows,
				    total_dry_matter_intake=this_ingredient_intake
				)

				ingredient_details.append(
				    self.create_ration_detail(
				        name=ingredient['name'],
				        docId=ingredient['docId'],
				        dryMatterIntake=MetricUtils.cast_to_float(this_ingredient_intake),
				        asFedIntake=MetricUtils.cast_to_float(this_ingredient_as_fed),
				        dryMatterIntakePerCow=this_ingredient_intake_per_cow,
				        asFedIntakePerCow=this_ingredient_as_fed_per_cow,
				        rationDetails=nested_ration['rationDetails']
				    )
				)
			else:
				# Ensure the 'recipe' field is set to False for IngredientData
				if ingredient.get('recipe', True):
					raise ValueError(f"The 'recipe' field must be False for IngredientData ingredients.")

				this_ingredient_intake = total_dry_matter_intake * ingredient['totalBK'] / 100
				this_ingredient_as_fed = this_ingredient_intake * (100 / ingredient['BK'])

				this_ingredient_intake_per_cow = MetricUtils.cast_to_float(this_ingredient_intake / number_of_cows)

				this_ingredient_as_fed_per_cow = MetricUtils.cast_to_float(this_ingredient_as_fed / number_of_cows)
				
				if this_ingredient_as_fed_per_cow * number_of_cows > 0.5:
					this_ingredient_as_fed_per_cow = MetricUtils.get_feed_rounding(this_ingredient_as_fed_per_cow)
					this_ingredient_as_fed = MetricUtils.cast_to_float(this_ingredient_as_fed_per_cow * number_of_cows)
					

				

				ingredient_details.append(
				    self.create_ration_detail(
				        name=ingredient['name'],
				        docId=ingredient.get('docId'),
				        dryMatterIntake=MetricUtils.cast_to_float(this_ingredient_intake),
				        asFedIntake=MetricUtils.cast_to_float(this_ingredient_as_fed),
				        dryMatterIntakePerCow=this_ingredient_intake_per_cow,
				        asFedIntakePerCow=this_ingredient_as_fed_per_cow,
				        rationDetails=None
				    )
				)

		return ingredient_details, ingredient_dry_matter_total


	def _process_additives(self, number_of_cows):
	    """
	    Processes the additives to calculate their ration details.

	    Args:
	        number_of_cows (int): Number of cows in the group.

	    Returns:
	        tuple: A list of additive ration details and the total dry matter intake from additives.
	    """
	    additive_details = []
	    additive_dry_matter_total = 0

	    for additive in self.additives:
	        if additive['per'] == 'group':
	            this_additive_as_fed = additive['amount'] / number_of_cows
	        elif additive['per'] == 'head':
	            this_additive_as_fed = additive['amount']
	        elif additive['per'] == 'total':
	        	this_additive_as_fed = additive['amount']

	        # Calculate dry matter intake for additive
	        this_additive_dry_matter = this_additive_as_fed * (additive['ingredient'].BK / 100)

	        # Accumulate dry matter intake
	        additive_dry_matter_total += this_additive_dry_matter

	        additive_details.append(
	            self.create_ration_detail(
	                name=additive['name'],
	                docId=additive.get('docId'),
	                dryMatterIntake=MetricUtils.cast_to_float(this_additive_dry_matter),
	                asFedIntake=MetricUtils.cast_to_float(this_additive_as_fed),
	                dryMatterIntakePerCow=MetricUtils.cast_to_float(this_additive_dry_matter / number_of_cows),
	                asFedIntakePerCow=MetricUtils.cast_to_float(this_additive_as_fed / number_of_cows),
	                rationDetails=None
	            )
	        )

	    return additive_details, additive_dry_matter_total

	def calculate_allowance(self, number_of_cows):
	    """
	    Calculates the allowance details for each ingredient and additive based on the 'qty' field 
	    to determine as-fed intake, and derives other values accordingly.

	    Args:
	        number_of_cows (int): Number of cows in the group.

	    Returns:
	        dict: A dictionary containing allowance details for each ingredient and additive.
	    """
	    # Process ingredients
	    ingredient_details, total_as_fed, total_dry_matter_intake = self._process_allowance_ingredients(number_of_cows)

	    # Process additives
	    additive_details, total_dry_matter_additives = self._process_additives(number_of_cows)

	    # Combine total as-fed intake
	    total_dry_matter_intake += total_dry_matter_additives

	    # Combine allowance details
	    allowance_details = ingredient_details + additive_details

	    return {
	        'totalAsFed': MetricUtils.cast_to_float(total_as_fed),
	        'rationDetails': allowance_details,
	        'totalDryMatterIntake': total_dry_matter_intake
	    }

	def _process_allowance_ingredients(self, number_of_cows):
	    """
	    Processes the ingredients to calculate their allowance details.

	    Args:
	        number_of_cows (int): Number of cows in the group.

	    Returns:
	        tuple: A list of ingredient allowance details and the total as-fed intake from ingredients.
	    """
	    ingredient_details = []
	    total_as_fed = 0
	    total_dry_matter_intake = 0

	    for ingredient in self.ingredients:
	        if isinstance(ingredient['ingredient'], RecipeData):
	            # Ensure the 'recipe' field is set correctly
	            if not ingredient.get('recipe', False):
	                raise ValueError(f"The 'recipe' field must be True for RecipeData ingredients.")

	            # As-fed intake is determined directly from the qty field
	            this_as_fed = MetricUtils.cast_to_float(ingredient['qty']*number_of_cows) 
	            this_dry_matter = this_as_fed * (ingredient['BK'] / 100)

	            total_dry_matter_intake += this_dry_matter

	            this_as_fed_per_cow = ingredient['qty'] 
	            this_dry_matter_per_cow = MetricUtils.cast_to_float(this_dry_matter / number_of_cows)

	            # Recursive calculation for nested recipes
	            nested_allowance = ingredient['ingredient'].calculate_allowance(
	                number_of_cows=number_of_cows
	            )

	            ingredient_details.append(
	                self.create_ration_detail(
	                    name=ingredient['name'],
	                    docId=ingredient['docId'],
	                    asFedIntake=MetricUtils.cast_to_float(this_as_fed),
	                    dryMatterIntake=MetricUtils.cast_to_float(this_dry_matter),
	                    asFedIntakePerCow=this_as_fed_per_cow,
	                    dryMatterIntakePerCow=this_dry_matter_per_cow,
	                    rationDetails=nested_allowance['rationDetails']
	                )
	            )

	        else:
	            # Ensure the 'recipe' field is set to False for IngredientData
	            if ingredient.get('recipe', True):
	                raise ValueError(f"The 'recipe' field must be False for IngredientData ingredients.")

	            # As-fed intake is directly from qty
	            this_as_fed = MetricUtils.cast_to_float(ingredient['qty']*number_of_cows)
	            this_dry_matter = this_as_fed * (ingredient['BK'] / 100)

	            total_dry_matter_intake += this_dry_matter

	            this_as_fed_per_cow = ingredient['qty']
	            this_dry_matter_per_cow = MetricUtils.cast_to_float(this_dry_matter / number_of_cows)

	            ingredient_details.append(
	                self.create_ration_detail(
	                    name=ingredient['name'],
	                    docId=ingredient.get('docId'),
	                    asFedIntake=MetricUtils.cast_to_float(this_as_fed),
	                    dryMatterIntake=MetricUtils.cast_to_float(this_dry_matter),
	                    asFedIntakePerCow=this_as_fed_per_cow,
	                    dryMatterIntakePerCow=this_dry_matter_per_cow,
	                    rationDetails=None
	                )
	            )

	        total_as_fed += this_as_fed

	    return ingredient_details, total_as_fed, total_dry_matter_intake


	@staticmethod
	def load(farm_id, recipe_id, firebase_dispatcher):
	    """
	    Fetches a recipe from Firebase using the FirebaseManager, initializes a RecipeData object, 
	    and loads its ingredient and additive data using IngredientData or RecipeData based on the 'recipe' flag.

	    Args:
	        farm_id (str): The ID of the farm.
	        recipe_id (str): The document ID of the recipe.
	        firebase_manager (FirebaseManager): An instance of FirebaseManager for accessing Firebase.

	    Returns:
	        RecipeData: A RecipeData object if the recipe is found, otherwise None.
	    """
	    try:
	        # Retrieve the recipe using the FirebaseManager
	        recipe_data = firebase_dispatcher.get_recipes(recipe_id, farm_id)

	        # Check if the recipe data exists
	        if recipe_data:
	            #print(f"Recipe found: {recipe_data}")

	            # Loop through all ingredients and load them as RecipeData or IngredientData
	            for index, ingredient in enumerate(recipe_data.get('ingredients', [])):
	                ingredient_doc_id = ingredient.get('docId')
	                
	                if not ingredient_doc_id:
	                    #print(f"No docId found for ingredient at index {index}: {ingredient}")
	                    recipe_data['ingredients'][index]['ingredient'] = ingredient  # Store the original ingredient
	                    continue
	                
	                if ingredient.get('recipe') is True:
	                    # Load the ingredient as a RecipeData object (nested recipe)
	                    loaded_recipe = RecipeData.load(farm_id, ingredient_doc_id, firebase_dispatcher)
	                    if loaded_recipe:
	                        #print(f"Loaded nested recipe at index {index}: {loaded_recipe.to_dict()}")
	                        recipe_data['ingredients'][index]['ingredient'] = loaded_recipe
	                    else:
	                        #print(f"Failed to load nested recipe with ID: {ingredient_doc_id} at index {index}")
	                        recipe_data['ingredients'][index]['ingredient'] = ingredient
	                else:
	                    # Load the ingredient as an IngredientData object
	                    loaded_ingredient = IngredientData.load(farm_id, ingredient_doc_id, firebase_dispatcher)
	                    if loaded_ingredient:
	                        #print(f"Loaded ingredient at index {index}: {loaded_ingredient.to_dict()}")
	                        recipe_data['ingredients'][index]['ingredient'] = loaded_ingredient
	                    else:
	                        #print(f"Failed to load ingredient with ID: {ingredient_doc_id} at index {index}")
	                        recipe_data['ingredients'][index]['ingredient'] = ingredient

	            # Loop through all additives and load them as RecipeData or IngredientData
	            for index, additive in enumerate(recipe_data.get('additives', [])):
	                additive_doc_id = additive.get('docId')
	                
	                if not additive_doc_id:
	                    #print(f"No docId found for additive at index {index}: {additive}")
	                    recipe_data['additives'][index]['ingredient'] = additive  # Store the original additive
	                    continue
	                
	                if additive.get('recipe') is True:
	                    # Load the additive as a RecipeData object (nested recipe)
	                    loaded_recipe = RecipeData.load(farm_id, additive_doc_id, firebase_dispatcher)
	                    if loaded_recipe:
	                        #print(f"Loaded nested recipe at index {index}: {loaded_recipe.to_dict()}")
	                        recipe_data['additives'][index]['ingredient'] = loaded_recipe
	                    else:
	                        #print(f"Failed to load nested recipe with ID: {additive_doc_id} at index {index}")
	                        recipe_data['additives'][index]['ingredient'] = additive
	                else:
	                    # Load the additive as an IngredientData object
	                    loaded_additive = IngredientData.load(farm_id, additive_doc_id, firebase_dispatcher)
	                    if loaded_additive:
	                        #print(f"Loaded additive at index {index}: {loaded_additive.to_dict()}")
	                        recipe_data['additives'][index]['ingredient'] = loaded_additive
	                    else:
	                        #print(f"Failed to load additive with ID: {additive_doc_id} at index {index}")
	                        recipe_data['additives'][index]['ingredient'] = additive

	            # Initialize and return the RecipeData object
	            return RecipeData(recipe_data)
	        else:
	            print(f"No recipe found for ID: {recipe_id}")
	            return None

	    except Exception as e:
	        print(f"Error fetching recipe with ID {recipe_id}: {e}")
	        return None
