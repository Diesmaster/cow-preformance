from types import NoneType


class ModelUtils:
	"""
	A utility class for validating entries against dynamic field structures.
	"""

	@staticmethod
	def validate_entry(entry: dict, field_structure: dict) -> bool:
	    """
	    Dynamically validates an entry against a given field structure by comparing the type names.

	    Args:
	        entry (dict): The dictionary entry to validate.
	        field_structure (dict): The field structure to validate against.

	    Returns:
	        bool: True if the entry is valid, otherwise False.

	    Raises:
	        ValueError: If a field's value does not match its expected type, 
	                    with details on the field name, expected type, and actual type/value.
	    """
	    if not isinstance(entry, dict):
	        raise ValueError(f"Expected entry to be of type 'dict', but got {type(entry).__name__} with value: {entry}")

	    for field, expected_type in field_structure.items():
	        value = entry.get(field)

	        # Handle missing required fields
	        if value is None:
	            if NoneType not in (expected_type if isinstance(expected_type, tuple) else (expected_type,)):
	                raise ValueError(
	                    f"Field '{field}' is missing or has a None value, but it is required. "
	                    f"Expected types: {expected_type}, but got None"
	                )
	            continue

	        # Handle set of allowed values
	        if isinstance(expected_type, set):
	            if value not in expected_type:
	                raise ValueError(
	                    f"Field '{field}' has an invalid value. Expected one of {expected_type}, but got {value} (type: {type(value).__name__})"
	                )

	        # Compare the type names
	        elif isinstance(expected_type, (tuple, list)):
	            # Get the names of all expected types
	            expected_type_names = {t if isinstance(t, str) else t.__name__ for t in expected_type}
	            if type(value).__name__ not in expected_type_names:
	                raise ValueError(
	                    f"Field '{field}' has an invalid type. Expected one of {expected_type_names}, but got {type(value).__name__} with value: {value}"
	                )
	        else:
	            expected_type_name = expected_type if isinstance(expected_type, str) else expected_type.__name__
	            if type(value).__name__ != expected_type_name:
	                raise ValueError(
	                    f"Field '{field}' has an invalid type. Expected '{expected_type_name}', but got {type(value).__name__} with value: {value}"
	                )

	    return True

	@staticmethod
	def validate_subset(subset: dict, field_structure: dict) -> bool:
	    """
	    Validates if the given subset of data matches the types defined in the field structure.
	    This method only validates the keys present in the subset, not requiring all fields in the model.

	    Args:
	        subset (dict): The dictionary subset to validate.
	        field_structure (dict): The field structure to validate against.

	    Returns:
	        bool: True if the subset is valid, otherwise False.

	    Raises:
	        ValueError: If a field's value does not match its expected type, 
	                    with details on the field name, expected type, and actual type/value.
	    """
	    if not isinstance(subset, dict):
	        raise ValueError(f"Expected subset to be of type 'dict', but got {type(subset).__name__} with value: {subset}")

	    for field, value in subset.items():
	        if field not in field_structure:
	            raise ValueError(f"Field '{field}' is not present in the field structure model.")
	        
	        expected_type = field_structure[field]

	        # Handle None values
	        if value is None:
	            if NoneType not in (expected_type if isinstance(expected_type, tuple) else (expected_type,)):
	                raise ValueError(
	                    f"Field '{field}' is None, but it is required. "
	                    f"Expected types: {expected_type}, but got None"
	                )
	            continue

	        # Handle set of allowed values
	        if isinstance(expected_type, set):
	            if value not in expected_type:
	                raise ValueError(
	                    f"Field '{field}' has an invalid value. Expected one of {expected_type}, but got {value} (type: {type(value).__name__})"
	                )

	        # Compare the type names
	        elif isinstance(expected_type, (tuple, list)):
	            # Get the names of all expected types
	            expected_type_names = {t if isinstance(t, str) else t.__name__ for t in expected_type}
	            if type(value).__name__ not in expected_type_names:
	                raise ValueError(
	                    f"Field '{field}' has an invalid type. Expected one of {expected_type_names}, but got {type(value).__name__} with value: {value}"
	                )
	        else:
	            expected_type_name = expected_type if isinstance(expected_type, str) else expected_type.__name__
	            if type(value).__name__ != expected_type_name:
	                raise ValueError(
	                    f"Field '{field}' has an invalid type. Expected '{expected_type_name}', but got {type(value).__name__} with value: {value}"
	                )

	    return True

	@staticmethod
	def create_object(field_structure: dict, **kwargs) -> dict:
	    """
	    Dynamically creates an object based on the provided field structure by comparing type names.
	    Args:
	        field_structure (dict): The field structure defining expected fields and their types.
	        kwargs: Key-value pairs for the fields.
	    Returns:
	        dict: A validated object matching the field structure.
	    Raises:
	        ValueError: If any required field is missing or invalid.
	    """
	    obj = {}
	    for field, expected_type in field_structure.items():
	        value = kwargs.get(field)
	        if value is None and NoneType not in (expected_type if isinstance(expected_type, tuple) else (expected_type,)):
	            raise ValueError(f"Missing required field '{field}' for object.")

	        # Validate the value by comparing type names
	        if value is not None:
	            if isinstance(expected_type, (tuple, list)):
	                expected_type_names = {t if isinstance(t, str) else t.__name__ for t in expected_type}
	                if type(value).__name__ not in expected_type_names:
	                    raise ValueError(f"Field '{field}' must be one of {expected_type_names}, got {type(value).__name__}.")
	            else:
	                expected_type_name = expected_type if isinstance(expected_type, str) else expected_type.__name__
	                if type(value).__name__ != expected_type_name:
	                    raise ValueError(f"Field '{field}' must be of type {expected_type_name}, got {type(value).__name__}.")

	        obj[field] = value
	    return obj

	@staticmethod
	def validate_list_of_objects(entries: list[dict], field_structure: dict) -> bool:
	    """
	    Validates a list of objects against a given field structure by comparing type names.
	    Args:
	        entries (list): A list of dictionaries to validate.
	        field_structure (dict): The field structure to validate against.
	    Returns:
	        bool: True if all entries are valid, otherwise False.
	    """
	    return all(ModelUtils.validate_entry(entry, field_structure) for entry in entries)
