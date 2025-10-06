class SortingUtils:
	@staticmethod
	def group_cows_minimize_difference(cows, max_range):
		"""
		Distribute cows into the minimum number of groups such that the weight difference
		within each group does not exceed the max_range. Groups are arranged heaviest to lightest.

		:param cows: List of cow objects, each containing a 'weight' key.
		:param max_range: Maximum allowed weight difference within each group.
		:return: A list of groups, each containing a subset of cows.
		"""
		if not isinstance(cows, list) or not all(isinstance(cow, dict) and 'weight' in cow for cow in cows):
			raise ValueError("Cows must be a list of dictionaries containing 'weight' keys.")
		if not isinstance(max_range, (int, float)) or max_range <= 0:
			raise ValueError("The max_range must be a positive number.")

		# Step 1: Sort cows by weight in descending order
		sorted_cows = sorted(cows, key=lambda cow: float(cow['weight']), reverse=True)

		# Step 2: Group cows based on max_range
		groups = []
		current_group = []
		current_max_weight = None

		for cow in sorted_cows:
			weight = float(cow['weight'])
			if not current_group:
				# Start a new group
				current_group.append(cow)
				current_max_weight = weight
			elif current_max_weight - weight <= max_range:
				# Add cow to the current group
				current_group.append(cow)
			else:
				# Current group is full, start a new group
				groups.append(current_group)
				current_group = [cow]
				current_max_weight = weight

		# Add the last group if not empty
		if current_group:
			groups.append(current_group)

		return groups

	@staticmethod
	def sort_dicts_by_key(dicts, key, ascending=True):
	    """
	    Sorts a list of dictionaries by a specific key.

	    Args:
	        dicts (list): A list of dictionaries to sort.
	        key (str): The key to sort the dictionaries by.
	        ascending (bool): Whether to sort in ascending order. Default is True.

	    Returns:
	        list: A sorted list of dictionaries.

	    Raises:
	        ValueError: If `dicts` is not a list of dictionaries or if the key is missing.
	    """
	    if not isinstance(dicts, list) or not all(isinstance(d, dict) for d in dicts):
	        raise ValueError("Input must be a list of dictionaries.")
	    if not isinstance(key, str):
	        raise ValueError("Key must be a string.")
	    if not all(key in d for d in dicts):
	        raise ValueError(f"Key '{key}' not found in all dictionaries.")

	    return sorted(dicts, key=lambda d: d.get(key, float('inf')), reverse=not ascending)