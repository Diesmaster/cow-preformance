# feed_utils.py

from utils.metric_utils import MetricUtils

from datetime import datetime

class FeedUtils:
    """
    A utility class that provides methods for feed calculations and data handling.
    """

    @staticmethod
    def calculate_as_fed(total_dry_matter_intake, ingredient_total_bk, ingredient_bk):
        """
        Calculates the as-fed value for a given ingredient based on its dry matter intake.

        Args:
            total_dry_matter_intake (float): Total dry matter intake for the feed.
            ingredient_total_bk (float): The total dry matter percentage (totalBK) of the ingredient.
            ingredient_bk (float): The percentage of dry matter (BK) of the specific ingredient.

        Returns:
            float: The as-fed value for the ingredient.
        """
        try:
            # Calculate the ingredient's dry matter intake (BK)
            ingredient_bk_value = float(total_dry_matter_intake) * ( float(ingredient_total_bk) / 100)

            # Calculate the as-fed value based on the dry matter intake and BK percentage
            as_fed_value = ingredient_bk_value / ( float(ingredient_bk) / 100)
            return as_fed_value, ingredient_bk_value

        except ZeroDivisionError:
            raise ValueError("Ingredient BK percentage cannot be zero.")
        except Exception as e:
            raise ValueError(f"Error calculating as-fed value: {e}")

    @staticmethod
    def calculate_dry_matter(as_fed_value, ingredient_bk):
        """
        Calculates the total dry matter intake and the ingredient's dry matter intake (BK) 
        based on the as-fed value.

        Args:
            as_fed_value (float): The as-fed value for the ingredient.
            ingredient_total_bk (float): The total dry matter percentage (totalBK) of the ingredient.
            ingredient_bk (float): The percentage of dry matter (BK) of the specific ingredient.

        Returns:
            tuple: A tuple containing the total dry matter intake and the ingredient's dry matter intake.
        """
        try:
            # Calculate the ingredient's dry matter intake (BK)
            ingredient_bk_value = float(as_fed_value) * float(ingredient_bk)/100

            return ingredient_bk_value

        except ZeroDivisionError:
            raise ValueError("Ingredient BK percentage and total BK percentage cannot be zero.")
        except Exception as e:
            raise ValueError(f"Error calculating dry matter intake: {e}")

    @staticmethod
    def calculate_total_dry_matter(ingredients):
        """
        Calculates the total BK intake for a list of ingredients.

        Args:
            ingredients (list): A list of dictionaries, each containing 'as_fed_kg_per_cow' and 'BK'.

        Returns:
            float: The total BK intake for all ingredients.
        """
        total_bk_intake = 0.0  # Initialize total BK intake

        try:
            for ingredient in ingredients:
                as_fed_value = float(ingredient.get('as_fed_per_cow', 0))  # Get the as-fed value
                ingredient_bk = float(ingredient.get('BK', 0))  # Get the BK percentage

                # Use the calculate_dry_matter function to compute the ingredient's BK intake
                ingredient_bk_value = FeedUtils.calculate_dry_matter(as_fed_value, ingredient_bk)

                # Accumulate the total BK intake
                total_bk_intake += ingredient_bk_value

            return total_bk_intake

        except Exception as e:
            raise ValueError(f"Error calculating total BK intake: {e}")


    @staticmethod
    def calculate_total_dry_matter_intake(percentage_of_bodyweight, total_weight):
        """
        Calculates the total dry matter intake for a group of cows based on percentage of body weight.

        Args:
            percentage_of_bodyweight (float): The percentage of the total body weight used for feed intake.
            total_weight (float): The total weight of the group of cows.

        Returns:
            float: Total dry matter intake for the group of cows.
        
        Raises:
            ValueError: If percentage_of_bodyweight or total_weight is zero or negative.
        """
        try:
            # Validate inputs
            if float(percentage_of_bodyweight) <= 0 or float(total_weight) <= 0:
                raise ValueError("Percentage of bodyweight and total weight must be positive values.")

            # Calculate total dry matter intake for the group
            total_dry_matter_intake = (float(percentage_of_bodyweight) / 100) * float(total_weight)
            return round(total_dry_matter_intake, 2)

        except Exception as e:
            raise ValueError(f"Error calculating total dry matter intake: {e}")




    @staticmethod
    def start_calculate_feed_allowance(number_of_cows, feed_allowance):
        """
        Starts the calculation process for feed allowances. This function calculates the 
        total feed allowance based on as-fed values.

        Args:
            number_of_cows (int): Number of cows in the group.
            feed_allowance (dict): Feed allowance data containing the ingredient details.

        Returns:
            dict: The generated feed allowance details including total amounts.
        """
        ingredient_details = []
        total_as_fed_per_cow = 0

        for ingredient in feed_allowance['ingredients']:
            as_fed_per_cow = ingredient['as_fed_kg_per_cow']
            
            # Append the calculated data for the ingredient
            ingredient_details.append(FeedUtils._start_calculate_feed_allowance(as_fed_per_cow, number_of_cows, ingredient))

            # Accumulate the total as-fed per cow
            total_as_fed_per_cow += as_fed_per_cow

        # Generate a result dictionary containing all the computed details
        return {
            'ingredients': ingredient_details,
        }

    @staticmethod
    def _start_calculate_feed_allowance(as_fed_per_cow, number_of_cows, main_ingredient):
        """
        Processes individual ingredient data for feed allowance calculations.

        Args:
            as_fed_per_cow (float): The as-fed amount per cow.
            total_as_fed (float): The total as-fed amount for the group.
            main_ingredient (dict): The main ingredient details.

        Returns:
            dict: The processed ingredient data including total amounts.
        """
        # Prepare the main ingredient data

        total_as_fed = as_fed_per_cow * number_of_cows  # Calculate total as-fed for the group

        ingredient_data = {
            'ingredient_name': main_ingredient['ingredient_name'],
            'ingredient_id': main_ingredient['ingredient_id'],
            'as_fed_per_cow': as_fed_per_cow,
            'as_fed': total_as_fed
        }

        # Check if the main ingredient has subingredients
        if 'ingredients' in main_ingredient:
            subingredient_data = []
            for subingredient in main_ingredient['ingredients']:
                # Calculate the subingredient values recursively
                sub_as_fed_per_cow = subingredient['as_fed_kg_per_cow']
                sub_total_as_fed = sub_as_fed_per_cow * number_of_cows  # Assuming number_of_cows is available here
                subingredient_data.append(FeedUtils._start_calculate_feed_allowance(sub_as_fed_per_cow, sub_total_as_fed, subingredient))
            ingredient_data['subingredient_data'] = subingredient_data  # Add subingredient data to the main ingredient data

        return ingredient_data



    @staticmethod
    def progress_estimate(total_weight, days_difference, target_kilo_adage, cows_in_group):
        """
        Estimates the new total weight based on the number of days and target daily gain.

        Args:
            total_weight (float): The initial total weight of the group of cows.
            days_difference (int): The number of days since the last update.
            target_kilo_adage (float): Target daily weight gain per cow (in kg).
            cows_in_group (int): Number of cows in the group.

        Returns:
            float: The estimated new total weight of the group after the given period.
        """
        try:
            # Update total weight based on the number of days provided
            total_weight += (int(days_difference) * float(target_kilo_adage) * int(cows_in_group))/2
            return total_weight
        except Exception as e:
            raise ValueError(f"Error calculating progress estimate: {e}")

    @staticmethod
    def calculate_total(feed_ration, start_date, end_date):
        """
        Recursively calculates the total as-fed consumption for each ingredient in the feed ration over a given period.

        Args:
            feed_ration (dict): A dictionary representing the feed ration, containing ingredients and their details.
            start_date (str): The start date of the period in 'YYYY-MM-DD' format.
            end_date (str): The end date of the period in 'YYYY-MM-DD' format.

        Returns:
            dict: A dictionary where keys are ingredient names and values are the total as-fed consumption.
        """
        try:
            # Calculate the number of days between start and end date using MetricUtils
            day_difference = MetricUtils.calculate_day_difference(start_date, end_date)

            # Initialize an empty dictionary to hold total as-fed values for each ingredient
            total_feed_consumption = {}

            # Process the top-level ingredients in the feed ration
            for ingredient in feed_ration.get('ingredients', []):
                FeedUtils._calculate_ingredient_total(ingredient, day_difference, total_feed_consumption)

            return total_feed_consumption

        except Exception as e:
            raise ValueError(f"Failed to calculate total feed consumption: {str(e)}")

    @staticmethod
    def _calculate_ingredient_total(ingredient, day_difference, total_feed_consumption):
        """
        A recursive helper function that calculates the total as-fed consumption for an ingredient and its sub-ingredients.

        Args:
            ingredient (dict): A dictionary representing a single ingredient or sub-ingredient.
            day_difference (int): The number of days over which to calculate the total as-fed consumption.
            total_feed_consumption (dict): The dictionary to store the calculated total consumption.

        Returns:
            None: The results are stored in the total_feed_consumption dictionary.
        """
        try:
            # Retrieve the ingredient name and daily as-fed value per cow
            ingredient_name = ingredient.get('ingredient_name')
            daily_as_fed_per_cow = ingredient.get('as_fed_per_cow', 0)

            # Calculate the total as-fed value for the ingredient over the given period
            total_as_fed = daily_as_fed_per_cow * day_difference

            # Store the total in the total_feed_consumption dictionary
            if ingredient_name in total_feed_consumption:
                total_feed_consumption[ingredient_name] += total_as_fed
            else:
                total_feed_consumption[ingredient_name] = total_as_fed

            # Recursively process sub-ingredients if present
            for subingredient in ingredient.get('subingredient_feed_info', []):
                FeedUtils._calculate_ingredient_total(subingredient, day_difference, total_feed_consumption)

        except Exception as e:
            raise ValueError(f"Error calculating total for ingredient '{ingredient_name}': {str(e)}")

    @staticmethod
    def get_ingredient_percentage(ingredient, percentage):
        """
        Recursively calculates the total as-fed and total as-fed per cow values for a given ingredient
        based on a specified percentage of the daily ration.

        Args:
            ingredient (dict): A dictionary representing a single ingredient or sub-ingredient.
            percentage (float): The percentage of the ingredient to be used from the daily ration.

        Returns:
            dict: A dictionary containing the total as-fed and total as-fed per cow values for the ingredient and its sub-ingredients.
            Format: {
                'total_as_fed': float,
                'total_as_fed_per_cow': float,
                'subingredients': [{...}, {...}, ...]
            }
        """
        try:
            # Calculate the total as-fed value for the given percentage
            total_as_fed = ingredient.get('asFedIntake', 0) * (percentage / 100)
            total_as_fed_per_cow = ingredient.get('asFedIntakePerCow', 0) * (percentage / 100)

            # Store the results in a dictionary
            result = {
                'name': ingredient.get('name', 'Unknown Ingredient'),
                'asFedIntake': round(total_as_fed, 2),
                'asFedIntakePerCow': round(total_as_fed_per_cow, 2),
                'rationDetails': []
            }

            # Recursively process sub-ingredients if present
            subingredients = ingredient.get('rationDetails', [])

            if subingredients == None:
                subingredients = []


            for subingredient in subingredients:
                subingredient_result = FeedUtils.get_ingredient_percentage(subingredient, percentage)
                result['rationDetails'].append(subingredient_result)

            return result


        except Exception as e:
            raise ValueError(f"Error calculating ingredient percentage for '{ingredient.get('name', 'Unknown Ingredient')}': {str(e)}")

    @staticmethod
    def calculate_total_feed_values(feeding_schedule):
        """
        Processes the feeding schedule and calculates total as-fed values for each time.
        For each ingredient (including subingredients), sums the values and removes the subingredients.

        Args:
            feeding_schedule (dict): A dictionary structured by times, each containing groups and their ingredients.

        Returns:
            dict: The processed feeding schedule with subingredients removed and totals calculated for each time.
        """
        # Iterate through each feeding time in the schedule
        for time, groups in feeding_schedule.items():
            # Initialize a dictionary to store total values for each ingredient at this time slot
            total_values = {}

            # Iterate through each group within the time slot
            for group, ingredients in groups.items():
                # Process each ingredient in the group
                for ingredient in ingredients:
                    if not ingredient['name'] in total_values:
                        total_values[ingredient['name']] = 0

                    FeedUtils._process_ingredient(ingredient, total_values)

            # Add the totals dictionary to the current time slot
            for key in total_values:
                total_values[key] = MetricUtils.cast_to_float(total_values[key])
            feeding_schedule[time]['totals'] = dict(total_values)

        return feeding_schedule

    @staticmethod
    def _process_ingredient(ingredient, total_values):
        """
        Recursively calculates the total as-fed values for each ingredient and its subingredients.
        After calculating the totals, it removes the subingredient details and adds to the totals.

        Args:
            ingredient (dict): The ingredient object to process.
            total_values (dict): Dictionary to accumulate the total as-fed values for each ingredient.

        Returns:
            None: Modifies the ingredient object in place and updates totals.
        """
        # Base case: If no subingredients, update totals and return
        if not ingredient.get('subingredients'):
            total_values[ingredient['name']] += ingredient['asFedIntake']
            return

        # Initialize variables to accumulate total values
        total_as_fed = ingredient['asFedIntake']
        total_as_fed_per_cow = ingredient['asFedIntakePerCow']

        # Recursively calculate the totals for each subingredient
        for subingredient in ingredient['rationDetails']:

            if not subingredient['name'] in total_values:
                total_values[subingredient['name']] = 0


            FeedUtils._process_ingredient(subingredient, total_values)

        # Update the main ingredient with the aggregated totals
        ingredient['asFedIntake'] = round(total_as_fed, 2)
        ingredient['asFedIntakePerCow'] = round(total_as_fed_per_cow, 2)

        # Remove the subingredients after processing
        ingredient.pop('rationDetails')

        # Add the aggregated totals to the total_values dictionary
        total_values[ingredient['name']] += ingredient['asFedIntake']
    
    @staticmethod
    def update_total_feed_consumption(feed_consumption, farm_details_dict):
        """
        Updates the farm details dictionary with total feed consumption values based on the cows' data.

        Args:
            cows (list): A list of cow dictionaries containing feed consumption data.
            farm_details_dict (dict): The farm details dictionary to update.

        Returns:
            None
        """    
        for ingredient, consumption in feed_consumption.items():
            MetricUtils.add_total_number_to_dict(farm_details_dict['totalFeedConsumption'], f'{ingredient}_Total', consumption)



    @staticmethod
    def update_feed_ration(rationDetails, farm_details_dict):
        """
        Updates the farm details dictionary with feed ration values based on the cows' data.

        Args:
            cows (list): A list of cow dictionaries containing feed ration data.
            farm_details_dict (dict): The farm details dictionary to update.

        Returns:
            None
        """
        for ration in rationDetails:
            ration_name = ration['name']

            # Add the ingredient total to the feed ration dictionary
            if not ration['asFedIntakePerCow'] == -1:
                MetricUtils.add_total_number_to_dict(farm_details_dict, f'{ration_name}PerDay', ration['asFedIntakePerCow'])

            if ration['rationDetails'] == None:
                ration['rationDetails'] = []

            if len(ration['rationDetails']) > 0:
                return FeedUtils.update_feed_ration(ration['rationDetails'], farm_details_dict)

            return farm_details_dict

    @staticmethod
    def remove_feed_ration(rationDetails, farm_details_dict):
        """
        Updates the farm details dictionary with feed ration values based on the cows' data.

        Args:
            cows (list): A list of cow dictionaries containing feed ration data.
            farm_details_dict (dict): The farm details dictionary to update.

        Returns:
            None
        """
        for ration in rationDetails:
            ration_name = ration['name']

            # Add the ingredient total to the feed ration dictionary
            if not ration['asFedIntakePerCow'] == -1:
                MetricUtils.remove_total_number_from_dict(farm_details_dict, f'{ration_name}PerDay', ration['asFedIntakePerCow'])

            if len(ration['rationDetails']) > 0:
                return FeedUtils.remove_feed_ration(ration['rationDetails'], farm_details_dict)

            return farm_details_dict

    @staticmethod
    def get_dry_matter_intake(start_date, end_date, feed_history_data):
        """
        Calculates the total dry matter intake over a given date range from the feed history data.

        Args:
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.
            feed_history_data (list): A list of feed history dictionaries containing 'date' and 'dry_matter_intake' keys.

        Returns:
            float: The total dry matter intake over the specified period.
        """
        try:
            # Convert start and end dates to datetime objects
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

            # Sort the feed history data by date
            feed_history_data.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))

            total_dry_matter_intake = 0.0

            for i, entry in enumerate(feed_history_data):
                entry_date_obj = datetime.strptime(entry['date'], '%Y-%m-%d')

                # Find the start date in the history data
                if entry_date_obj >= start_date_obj:
                    # If next entry exists and is within the end date range
                    if i + 1 < len(feed_history_data):
                        next_entry_date_obj = datetime.strptime(feed_history_data[i + 1]['date'], '%Y-%m-%d')
                    else:
                        next_entry_date_obj = end_date_obj

                    # Extract dry matter intake value (either per cow or total)
                    
                    # Extract dry matter intake value, prioritizing dry_matter_intake_per_cow
                    dry_matter_intake = entry['feed_ration'].get('dry_matter_intake_per_cow')
                    if dry_matter_intake is None:
                        dry_matter_intake = entry['feed_ration'].get('total_dry_matter_intake')

                    # If entry date is <= start date and next entry date is >= end date
                    if entry_date_obj <= start_date_obj and next_entry_date_obj >= end_date_obj:
                        day_difference = (end_date_obj - start_date_obj).days
                        total_dry_matter_intake += dry_matter_intake * day_difference
                        break

                    # If entry date is > start date and within the range of start and end dates
                    elif start_date_obj <= entry_date_obj < next_entry_date_obj <= end_date_obj:
                        day_difference = (next_entry_date_obj - entry_date_obj).days
                        total_dry_matter_intake += dry_matter_intake * day_difference

                    # If next entry date is greater than end date, add the remaining days till the end date
                    elif start_date_obj <= entry_date_obj <= end_date_obj and next_entry_date_obj > end_date_obj:
                        day_difference = (end_date_obj - entry_date_obj).days
                        total_dry_matter_intake += dry_matter_intake * day_difference
                        break

            last_date_in_feedhistory = entry_date_obj = datetime.strptime(feed_history_data[-1]['date'], '%Y-%m-%d')
            last_dry_matter_intake = feed_history_data[-1]['feed_ration'].get('dry_matter_intake_per_cow', feed_history_data[-1]['feed_ration'].get('total_dry_matter_intake'))

            if start_date_obj > last_date_in_feedhistory:
                day_difference = (end_date_obj - start_date_obj).days
                total_dry_matter_intake += last_dry_matter_intake * day_difference


            return round(total_dry_matter_intake, 2)

        except Exception as e:
            raise ValueError(f"Error calculating dry matter intake: {e}")

    @staticmethod
    def replace_rations(feed_ration, rations_to_update):
        """
        Replaces specified old rations with new rations in the feed ration dictionary.

        Args:
            feed_ration (dict): The current feed ration dictionary with ingredient IDs as keys.
            rations_to_update (list): A list of dictionaries containing 'old_ration' and 'new_ration'.

        Returns:
            None: The feed_ration is updated in place.
        """
        for ration in rations_to_update:
            old_ration = ration.get('old_ration', {})
            new_ration = ration.get('new_ration', {})

            if old_ration != {} and new_ration != {}:
                FeedUtils.remove_feed_ration( [old_ration], feed_ration )
                FeedUtils.update_feed_ration( [ new_ration ], feed_ration )

        
        return feed_ration