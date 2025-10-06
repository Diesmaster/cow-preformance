from utils.metric_utils import MetricUtils

class DictUtils:
    """
    A utility class for handling common dictionary operations.
    """

    @staticmethod
    def make_dicts_uniform(dicts):
        """
        Ensures that all dictionaries in the given list have the same keys.
        Missing keys are filled with the value 0.

        Args:
            dicts (list): A list of dictionaries to be made uniform.

        Returns:
            list: A list of dictionaries with uniform keys.
        """
        # Find all unique keys across all dictionaries
        all_keys = set()
        for d in dicts:
            all_keys.update(d.keys())

        # Ensure every dictionary contains every key, with missing keys set to 0
        uniform_dicts = [{key: d.get(key, 0) for key in all_keys} for d in dicts]

        return uniform_dicts

    @staticmethod
    def total_dict_attribute(dicts_array, attribute_name):
        """
        Calculates the total sum of a specific attribute across an array of dictionaries.

        Args:
            dicts_array (list): A list of dictionaries.
            attribute_name (str): The attribute name whose values need to be summed.

        Returns:
            float: The total sum of the attribute values.
        """
        try:
            total = sum(float(d.get(attribute_name, 0)) for d in dicts_array)
            return total
        except Exception as e:
            raise ValueError(f"Failed to calculate total for attribute '{attribute_name}': {e}")

    @staticmethod
    def sum_two_dicts(dict1, dict2):
        """
        Sums the values of two dictionaries. If a key in `dict2` is present in `dict1`,
        adds the values together. If a key is not in `dict1`, it adds the key and value
        from `dict2` to `dict1`.

        Args:
            dict1 (dict): The first dictionary to be updated.
            dict2 (dict): The second dictionary whose values are added to `dict1`.

        Returns:
            dict: The updated dictionary with summed values.
        """
        for key, value in dict2.items():
            if key in dict1:
                dict1[key] += MetricUtils.cast_to_float(value)
            else:
                dict1[key] = MetricUtils.cast_to_float(value)
        return dict1

    @staticmethod
    def filter_array_of_dicts(array: list[dict], fields_to_keep: list) -> list[dict]:
        """
        Filters an array of dictionaries, only keeping the specified fields for each dictionary.

        Args:
            array (list[dict]): List of dictionaries to be filtered.
            fields_to_keep (list): List of fields to keep in the dictionaries.

        Returns:
            list[dict]: A list of dictionaries with only the specified fields retained.
        """
        if not isinstance(array, list) or not all(isinstance(item, dict) for item in array):
            raise ValueError("The array must be a list of dictionaries.")
        
        if not isinstance(fields_to_keep, list) or not all(isinstance(field, str) for field in fields_to_keep):
            raise ValueError("fields_to_keep must be a list of strings.")
        
        filtered_array = [
            {key: item[key] for key in fields_to_keep if key in item} 
            for item in array
        ]
        return filtered_array