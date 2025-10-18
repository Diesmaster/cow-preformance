from types import NoneType
import re

class Models:
    """
    A class to store the data models for Ingredient, Recipe, Additives, and Ration Details.
    This class serves as a centralized location to maintain the structure of required fields 
    and validation rules for each model used in the application.
    """

    # Ingredient field structure
    recipe_ingredient_fields = {
        'ingredient': ('IngredientData', 'RecipeData', NoneType),  # Ingredient can be an IngredientData, RecipeData, or None
        'qty': (int, float),  # Quantity of the ingredient
        'totalBK': (int, float),  # Total dry matter
        'asFedPercentage': (int, float),  # As-fed percentage
        'BK': (int, float),  # Dry matter content
        'name': str,  # Name of the ingredient
        'recipe': bool,  # Indicates if the ingredient is a recipe
        'docId': (str, NoneType)  # Optional document ID
    }

    # Required fields for the recipe
    recipe_fields = {
        'BK': (int, float), 
        'Ca': (int, float), 
        'LK': (int, float), 
        'NDF': (int, float), 
        'P': (int, float), 
        'PK': (int, float), 
        'Qty': (int, float), 
        'SK': (int, float), 
        'TDN': (int, float),
        'hargaPerKiloAsFed': (int, float), 
        'hargaPerKiloDryWeight': (int, float), 
        'ingredients': list,
        'additives': (list, NoneType), 
        'title': str
    }

    # Additive field structure
    additive_fields = {
        'ingredient': ('IngredientData', 'RecipeData', NoneType),  # Ingredient can be IngredientData, RecipeData, or None
        'per': {'head', 'group', 'total'},  # Options for "per" - head, group, or total
        'amount': (int, float),  # Amount of additive
        'name': str,  # Name of the additive
        'docId': (str, NoneType)  # Optional document ID
    }

    # Ration detail fields structure
    ration_detail_fields = {
        'name': str,  # Name of the ingredient or recipe
        'docId': (str, NoneType),  # Optional document ID
        'dryMatterIntake': (int, float),  # Amount of dry matter intake
        'dryMatterIntakePerCow': (int, float),  # Dry matter intake per cow
        'asFedIntake': (int, float),  # Amount of as-fed intake
        'asFedIntakePerCow': (int, float),  # As-fed intake per cow
        'rationDetails': (list, NoneType)  # Nested details for sub-rations, if any
    }

    ingredient_fields = {
        'PAKAN':str, 
        'Harga':(int, float), 
        'BK':(int, float), 
        'TDN':(int, float), 
        'PK':(int, float), 
        'LK':(int, float), 
        'SK':(int, float), 
        'NDF':(int, float), 
        'Ca':(int, float), 
        'P':(int, float),
        'docId': (str, NoneType)
    }

    feed_history_model = {
        'data': list
    }

    feed_history_entry_model = {
        'date': str, # is there a better way?
        'total_weight_of_group': (int, float),
        'cows_in_group': (int, float),
        'feedPercentageOfWeight': (int, float),
        'selectedRecipeDocId': str,
        'targetKiloAdage': (int, float),
        'selectedRecipeName': str,
        'dryMatterIntake': (int, float),
        'dryMatterIntakePerCow': (int, float),
        'rationDetails': list,
        'batches': dict,
        'totalFeedCost': (int, float),
        'dailyFeedCost': (int, float)
    }


    weight_history_model = {
        'data': list
    }

    weight_history_entry_model = {
        'date': str, # is there a better way?
        'weight': (int, float),
        'adgAverage': (int, float),
        'adgLatest': (int, float),
        'fcrAverage': (int, float),
        'fcrLatest': (int, float),
    }

    cow_model = {
        'adgLatest': (int, float),
        'estimatedWeight': (int, float),
        'shrinkage': (int, float),
        'salesDate': str,
        'rationDetails': (list, NoneType),
        'purchasingPriceKg': (int, float),
        'purchasingPrice': (int, float),
        'lastWeighing': str,
        'bodyDepth': (int, float),
        'entryDate': str,
        'typeOfOrigin': str,
        'daysOnFeed': (int, float),
        'backHeight': (int, float),
        'targetDaysOnFeed': (int, float),
        'topline': (int, float),
        'adgAverage': (int, float),
        'fcrAverage': (int, float),
        'rationDetailsUpdateDate': str,
        'cannonBoneCircumference': (int, float),
        'weight': (int, float),
        'withersHeight': (int, float),
        'adgSecondLatest': (int, float),
        'chestDepth': (int, float),
        'insurancePolis': str,
        'fcrLatest': (int, float),
        'farm': (str, NoneType),
        'origin': str,
        'hipHeight': (int, float),
        'totalFeedConsumption': (dict, NoneType),
        'date': str,
        'bodyLength': (int, float),
        'originWeight': (int, float),
        'rumpWidth': (int, float),
        'purchasingPriceKgEntry': (int, float),
        'entryWeight': (int, float),
        'breed': str,
        'totalWeightGained': (int, float),
        'cattleId': str,
        'salesPrice': (int, float, NoneType),
        'insurancePayout':(int, float, NoneType),
        'salesPricePerKilo':(int, float, NoneType),
        'totalSales':(int, float, NoneType),
        'totalMedicalCost':(int, float, NoneType),
        'docId': (str, NoneType)
    }

    cow_measurements_model = {
        'withersHeight':(int, float), 
        'hipHeight':(int, float), 
        'backHeight':(int, float), 
        'chestDepth':(int, float),
        'bodyDepth':(int, float), 
        'bodyLength':(int, float), 
        'topline':(int, float), 
        'rumpWidth':(int, float), 
        'cannonBoneCircumference':(int, float), 
        'docId':str, 
        'cattleId':str
    }

    farm_weight_history_entry_model = {
        'Average adgAverage': (int, float),
        'Average adgLatest': (int, float),
        'Average weight': (int, float),
        'Average totalWeightGained': (int, float),
        'Average fcrAverage': (int, float),
        'date': str,
    }

    farm_feed_history_entry_model = {
        'rationDetails': (dict, NoneType),
        'totalFeedConsumption': (dict, NoneType),
        'date': str
    }

    @staticmethod
    def get_farm_model():
        farm_model = {}
        for key in Models.cow_model:
            if Models.cow_model[key] == (int, float):
                farm_key = "Average " + key
                farm_model[farm_key] = (int, float)
                farm_key = "Total " + key
                farm_model[farm_key] = (int, float)
            elif Models.cow_model[key] == (list, NoneType):
                farm_key = key
                farm_model[farm_key] = (dict, NoneType)
            elif (Models.cow_model[key] == str or Models.cow_model[key] == (str, NoneType)) and (not key == "docId" and not key == 'salesDate' and not key == 'insurancePolis' and not key == 'cattleId' and not key == 'recordingDate'):
                key_pattern = re.compile(r'.*(date|Date).*')
                if key_pattern.match(key):
                    farm_key = 'Average ' + key
                    farm_model[farm_key] = str
                farm_key = "Breakdown " + key
                farm_model[farm_key] = dict
            elif Models.cow_model[key] == (dict, NoneType):
                farm_key = key
                farm_model[farm_key] = (dict, NoneType)

        farm_model['numberOfCows'] = int
        farm_model['feedRationUpdateDate'] = str

        del farm_model['Average rationDetailsUpdateDate']
        del farm_model['Breakdown rationDetailsUpdateDate']

        return farm_model

    medical_history_entry_model = {
        'docId': (str, NoneType),
        'date': str,
        'agenda': str,
        'treatment': str,
        'totalCost': (int, float),
        'medicinesUsed': dict,
        'scheduled': bool
    }

    medical_history_model = {
        'data': list,
    }
