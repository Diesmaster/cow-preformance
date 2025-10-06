from datetime import datetime, timedelta
from decimal import Decimal

class MetricUtils:
    """
    A utility class for performing various metric calculations related to cow management.
    """

    @staticmethod
    def calculate_adg(start_weight, start_date, end_weight, end_date):
        """
        Calculates the Average Daily Gain (ADG) for a cow between two dates.

        Args:
            start_weight (float): Initial weight of the cow.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_weight (float): Final weight of the cow.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            float: Average Daily Gain (ADG).
        """
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        days_diff = (end_date_obj - start_date_obj).days

        if days_diff == 0:
            days_diff = 1

        # Calculate ADG
        if days_diff > 0:
            return (float(end_weight) - float(start_weight)) / days_diff
        else:
            return 0

    @staticmethod
    def calculate_fcr(total_daily_dm, adg):
        """
        Calculate the Feed Conversion Ratio (FCR) based on total daily dry matter intake and ADG.

        Args:
            total_daily_dm (float): Total dry matter intake (in kg) per day for the given period.
            adg (float): Average daily gain (in kg/day) for the cow.

        Returns:
            float: The calculated Feed Conversion Ratio (FCR).
        """
        if total_daily_dm == 0 or adg == 0.0:
            return 0

        try:
            # Calculate FCR
            fcr = (Decimal(adg) / Decimal(total_daily_dm)) * 100

            return round(float(fcr), 3)
        except Exception as e:
            print(f"Error calculating FCR: {e}")
            return 0

    @staticmethod
    def calculate_total_drymatter_intake(feed_ration_update_date, current_date, total_dry_intake=0, daily_dry_matter_intake=0):
        """
        Calculate the total dry matter intake over a given period.

        Args:
            feed_ration_update_date (str): The date of the last feed ration update in 'YYYY-MM-DD' format.
            current_date (str): The current date in 'YYYY-MM-DD' format.
            total_dry_intake (float, optional): Total dry matter intake recorded up to the last feed ration update.
            daily_dry_matter_intake (float, optional): Daily dry matter intake per cow since the last update.

        Returns:
            float: The calculated total dry matter intake.
        """
        try:
            # Convert dates to datetime objects
            if not feed_ration_update_date:
                days_since_last_ration = 0
            else:
                last_update_date = datetime.strptime(feed_ration_update_date, '%Y-%m-%d')
                days_since_last_ration = (datetime.strptime(current_date, '%Y-%m-%d') - last_update_date).days

            # Calculate the total dry matter intake
            total_dry_matter = total_dry_intake + (daily_dry_matter_intake * days_since_last_ration)

            return round(total_dry_matter, 3)
        except Exception as e:
            print(f"Error calculating total dry matter intake: {e}")
            return 0.0

    @staticmethod
    def cast_to_float(number):
        return round(float(number),2)

    @staticmethod
    def calculate_per_cow(total_value, number_of_cows):
        """
        Calculates the per cow value by dividing a total value by the number of cows.

        Args:
            total_value (float): The total value to be distributed per cow.
            number_of_cows (int): The total number of cows in the group.

        Returns:
            float: The value per cow.
        
        Raises:
            ValueError: If number_of_cows is zero or negative.
        """
        try:
            # Ensure the number of cows is a positive integer
            if number_of_cows <= 0:
                raise ValueError("Number of cows must be a positive integer.")
            
            # Calculate the per cow value
            per_cow_value = total_value / number_of_cows
            return round(per_cow_value, 2)
        
        except ZeroDivisionError:
            raise ValueError("Number of cows cannot be zero.")
        except Exception as e:
            raise ValueError(f"Error calculating per cow value: {e}")

    @staticmethod
    def calculate_day_difference(start_date, end_date):
        """
        Calculates the number of days between two given dates.

        Args:
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.

        Returns:
            int: The number of days between the start and end dates.

        Raises:
            ValueError: If the date format is incorrect or if start_date is after end_date.
        """
        try:
            # Convert the start and end dates to datetime objects
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

            # Calculate the difference in days
            days_diff = (end_date_obj - start_date_obj).days

            return days_diff
        except Exception as e:
            raise ValueError(f"Error calculating day difference: {str(e)}")

    @staticmethod
    def add_average_number_to_dict(dictionary, key, value, cow_count):
        """
        Adds or updates the average value of a given key in the provided dictionary.

        Args:
            dictionary (dict): The dictionary to update.
            key (str): The key to be averaged and added to the dictionary.
            value (str or Decimal): The value to be averaged. It should be a numerical value.
            cow_count (int): The total number of cows to calculate the average.

        Returns:
            None
        """
        if value == 'tbf':
            value = 0

        if value == "-":
            value = 0

        try:
            average_value = Decimal(str(value)) / Decimal(str(cow_count))
            if key not in dictionary:
                dictionary[key] = average_value
            else:
                dictionary[key] = Decimal(str(dictionary[key]))
                dictionary[key] += average_value

            dictionary[key] = MetricUtils.cast_to_float(dictionary[key])
        except Exception as e:
            print("whats going on?")
            print(key)
            print(value)
            print(f"Error adding average number to dictionary for key '{key}': {e}")

    @staticmethod
    def add_total_number_to_dict(dictionary, key, value):
        """
        Adds or updates the average value of a given key in the provided dictionary.

        Args:
            dictionary (dict): The dictionary to update.
            key (str): The key to be averaged and added to the dictionary.
            value (str or Decimal): The value to be averaged. It should be a numerical value.

        Returns:
            None
        """
        try:
            value = Decimal(value)

            if key not in dictionary:
                dictionary[key] = value
            else:
                dictionary[key] = Decimal(str(dictionary[key]))
                dictionary[key] += value

            dictionary[key] = MetricUtils.cast_to_float(dictionary[key])
        except Exception as e:
            print("whats going on?")
            print(key)
            print(value)
            print(f"Error adding average number to dictionary for key '{key}': {e}")

    @staticmethod
    def add_nominal_number_to_dict(dictionary, parent_key, key):
        """
        Adds or updates a nominal count of a given key in the provided dictionary.

        Args:
            dictionary (dict): The dictionary to update.
            parent_key (str): The parent key in the dictionary.
            key (str): The key to be counted and added to the dictionary.

        Returns:
            None
        """
        if parent_key not in dictionary:
            dictionary[parent_key] = {}

        try:
            if key not in dictionary[parent_key]:
                dictionary[parent_key][key] = 1
            else:
                dictionary[parent_key][key] += 1

        except Exception as e:
            print(f"Error adding nominal number to dictionary for key '{key}': {e}")

    @staticmethod
    def remove_average_number_from_dict(dictionary, key, value, cow_count):
        """
        Removes or updates the average value of a given key in the provided dictionary.

        Args:
            dictionary (dict): The dictionary to update.
            key (str): The key to be averaged and removed from the dictionary.
            value (str or Decimal): The value to be averaged. It should be a numerical value.
            cow_count (int): The total number of cows to calculate the average.

        Returns:
            None
        """
        if value == 'tbf':
            value = 0

        if value == "-":
            value = 0

        try:
            average_value = Decimal(str(value)) / Decimal(str(cow_count))
            if key in dictionary:
                dictionary[key] = Decimal(str(dictionary[key])) - average_value
                
                # Remove the key if the value becomes zero or negative
                if dictionary[key] <= 0:
                    del dictionary[key]

                dictionary[key] = MetricUtils.cast_to_float(dictionary[key])
        except Exception as e:
            print(f"Error removing average number from dictionary for key '{key}': {e}")

    @staticmethod
    def remove_total_number_from_dict(dictionary, key, value):
        """
        Removes or updates the total value of a given key in the provided dictionary.

        Args:
            dictionary (dict): The dictionary to update.
            key (str): The key to be totaled and removed from the dictionary.
            value (str or Decimal): The value to be totaled. It should be a numerical value.

        Returns:
            None
        """
        try:
            value = Decimal(value)

            if key in dictionary:
                dictionary[key] = Decimal(str(dictionary[key])) - value
                
                # Remove the key if the value becomes zero or negative
                if dictionary[key] <= 0:
                    del dictionary[key]

                dictionary[key] = MetricUtils.cast_to_float(dictionary[key])
        except Exception as e:
            print(f"Error removing total number from dictionary for key '{key}': {e}")

    @staticmethod
    def remove_nominal_number_from_dict(dictionary, parent_key, key):
        """
        Removes or updates a nominal count of a given key in the provided dictionary.

        Args:
            dictionary (dict): The dictionary to update.
            parent_key (str): The parent key in the dictionary.
            key (str): The key to be counted and removed from the dictionary.

        Returns:
            None
        """
        if parent_key in dictionary:
            try:
                if key in dictionary[parent_key]:
                    dictionary[parent_key][key] -= 1
                    
                    # Remove the key if the count reaches zero
                    if dictionary[parent_key][key] <= 0:
                        del dictionary[parent_key][key]
                        
                    # Remove parent_key if it becomes empty
                    if not dictionary[parent_key]:
                        del dictionary[parent_key]

            except Exception as e:
                print(f"Error removing nominal number from dictionary for key '{key}': {e}")

    @staticmethod
    def remove_from_average_date(number_of_cows, old_average_date, removed_date):
        """
        Updates the average entry date for the farm based on cow removal.

        Args:
            number_of_cows (int): The current number of cows.
            old_average_date (str): The current average entry date as a string (format '%Y-%m-%d').
            removed_date (str): The cow removal date as a string (format '%Y-%m-%d').

        Returns:
            str: The updated average entry date as a string or None if there are no cows left.
        """
        # Convert the old average date and removed date to datetime objects
        old_average_date_dt = datetime.strptime(old_average_date, '%Y-%m-%d') if old_average_date else None
        removed_date_dt = datetime.strptime(removed_date, '%Y-%m-%d')

        if not old_average_date_dt:
            return None  # If no previous average, return None

        if number_of_cows <= 1:
            return None  # If removing this cow leaves no cows, return None

        # Calculate the new average date in days
        old_average_days = (old_average_date_dt - datetime(1970, 1, 1)).days
        removed_days = (removed_date_dt - datetime(1970, 1, 1)).days
        average_days = (old_average_days * number_of_cows - removed_days) / (number_of_cows - 1)

        # Convert back to a datetime object and return as a string
        updated_average_date = datetime(1970, 1, 1) + timedelta(days=average_days)
        return updated_average_date.strftime('%Y-%m-%d')



    @staticmethod
    def calculate_average_date(dates):
        """
        Calculate the average date from a list of date strings or datetime objects.

        Args:
            dates (list): List of date strings or datetime objects.

        Returns:
            str: The average date in 'YYYY-MM-DD' format.
        """

        # Ensure all dates are converted to datetime objects
        converted_dates = []
        for date in dates:
            if isinstance(date, str):
                try:
                    # Parse the date string into a datetime object
                    converted_date = datetime.strptime(date, '%Y-%m-%d')
                    converted_dates.append(converted_date)
                except ValueError:
                    raise ValueError(f"Invalid date format: {date}. Expected 'YYYY-MM-DD'.")
            elif isinstance(date, datetime):
                converted_dates.append(date)
            else:
                raise TypeError(f"Invalid date type: {type(date)}. Expected str or datetime.")

        # Calculate total timestamp and average timestamp
        total_timestamp = sum([date.timestamp() for date in converted_dates])
        average_timestamp = total_timestamp / len(converted_dates)
        average_date = datetime.fromtimestamp(average_timestamp)

        return average_date.strftime('%Y-%m-%d')


    @staticmethod
    def add_to_average(number_of_cows, old_average, added_value):
        """
        Calculate the new average when adding a value to the existing average using Decimal for precision.

        Args:
            number_of_cows (int): The current number of cows.
            old_average (float): The current average value.
            added_value (float): The new value to be added.

        Returns:
            float: The updated average value as a float.
        """
        # Use Decimal for more precise calculations
        old_average_decimal = Decimal(str(old_average))
        added_value_decimal = Decimal(str(added_value))
        number_of_cows_decimal = Decimal(str(number_of_cows))

        # Calculate the new average using the Decimal values
        new_average = (old_average_decimal * number_of_cows_decimal + added_value_decimal) / (number_of_cows_decimal + 1)

        # Convert the result back to float and round it to 2 decimal places
        return MetricUtils.cast_to_float(new_average)

    @staticmethod
    def add_to_average_date(number_of_cows, old_average_date, new_date):
        """
        Updates the average entry date for the farm based on new cow entry.

        Args:
            number_of_cows (int): The current number of cows.
            old_average_date (str): The current average entry date as a string (format '%Y-%m-%d').
            new_date (str): The new cow entry date as a string (format '%Y-%m-%d').

        Returns:
            str: The updated average entry date as a string.
        """
        # Convert the old average date and new date to datetime objects
        print(old_average_date)
        print(type(old_average_date))
        old_average_date_dt = datetime.strptime(old_average_date, '%Y-%m-%d') if old_average_date else None
        new_date_dt = datetime.strptime(new_date, '%Y-%m-%d')

        if not old_average_date_dt:
            return new_date  # If no previous average, return the new date

        # Calculate the new average date in days
        old_average_days = (old_average_date_dt - datetime(1970, 1, 1)).days
        new_days = (new_date_dt - datetime(1970, 1, 1)).days
        average_days = (old_average_days * number_of_cows + new_days) / (number_of_cows + 1)

        # Convert back to a datetime object and return as a string
        updated_average_date = datetime(1970, 1, 1) + timedelta(days=average_days)
        return updated_average_date.strftime('%Y-%m-%d')

    @staticmethod
    def update_average(old_value, new_value, average, number_of_cows):
        """
        Updates the average value when replacing an existing value with a new value.

        Args:
            old_value (str, float, or int): The old value to be replaced (can be a string or non-numeric, treated as 0).
            new_value (float or int): The new value to update.
            average (float): The current average value.
            number_of_cows (int): The total number of cows considered in the average.

        Returns:
            float: The updated average value as a float.
        """
        # Convert the old value to Decimal, treating non-numeric strings as zero
        try:
            old_value_decimal = Decimal(str(old_value))
        except (ValueError, TypeError):
            old_value_decimal = Decimal(0)

        # Convert new value and average to Decimal for precise calculations
        new_value_decimal = Decimal(str(new_value))
        average_decimal = Decimal(str(average))
        number_of_cows_decimal = Decimal(number_of_cows)

        # Calculate the new average using the formula
        updated_average = (((average_decimal * number_of_cows_decimal) - old_value_decimal) + new_value_decimal) / number_of_cows_decimal

        # Convert back to float and round it to 2 decimal places
        return MetricUtils.cast_to_float(updated_average)

    @staticmethod
    def update_total(old_total, prev_cow_value, new_cow_value):
        """
        Update the total value for a given field.

        Args:
            old_total (float): The current total value.
            prev_cow_value (str or float): The previous value of the cow for the field.
            new_cow_value (str or float): The new value of the cow for the field.

        Returns:
            float: The updated total value.
        """
        # Handle cases where the previous value is None or not numeric
        prev_cow_value = Decimal(str(prev_cow_value)) if str(prev_cow_value).replace('.', '', 1).isdigit() else Decimal(0)
        new_cow_value = Decimal(str(new_cow_value)) if str(new_cow_value).replace('.', '', 1).isdigit() else Decimal(0)

        # Convert old total to Decimal
        old_total = Decimal(str(old_total)) if str(old_total).replace('.', '', 1).isdigit() else Decimal(0)

        # Calculate the updated total
        updated_total = old_total - prev_cow_value + new_cow_value

        # Return the updated total as a float
        return MetricUtils.cast_to_float(updated_total)

