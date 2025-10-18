import json
import re
from datetime import datetime
from types import NoneType

from data_models.models import Models  # Use Models to reference dynamic field structures
from utils.model_utils import ModelUtils

class MedicalHistoryData:
    """
    A class representing a Medical History with specific attributes as fields.
    The medical history contains an array of medical history entry objects.
    """

    def __init__(self, medical_history_data):
        """
        Initializes an instance of MedicalHistoryData with the provided dictionary.

        Args:
            medical_history_data (dict): A dictionary containing the medical history.

        Raises:
            ValueError: If the dictionary is missing any required fields or contains None values.
        """
        if not self.validate(medical_history_data):
            raise ValueError("Invalid medical history data. Missing or None values for required fields.")

        for field in Models.medical_history_model.keys():
            setattr(self, field, medical_history_data.get(field))

        self.data = [self._validate_and_init_entry(entry) for entry in self.data]

    def get_total_cost(self, start_date, end_date):
        """
        Calculate the total medical cost within the given date range.

        Args:
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.
        """
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            total_cost = 0

            for entry in self.data:
                entry_date_obj = datetime.strptime(entry['date'], '%Y-%m-%d')
                if start_date_obj <= entry_date_obj <= end_date_obj:
                    total_cost += entry['totalCost']

            return total_cost
        except Exception as e:
            raise ValueError(f"Error calculating total medical cost: {e}")

    def add_medical_history_entry(self, new_entry):
        """
        Adds a medical history entry ensuring it is placed correctly based on date order.
        
        Args:
            new_entry (dict): The new medical history entry to add.
        
        Raises:
            ValueError: If the new entry's date is not in a valid position.
        """
        if not self.data:
            self.data.append(self._validate_and_init_entry(new_entry))
            return
        
        new_date = datetime.strptime(new_entry['date'], '%Y-%m-%d')
        
        for i in range(len(self.data)):
            entry_date = datetime.strptime(self.data[i]['date'], '%Y-%m-%d')

            if i > 0:
                prev_date = datetime.strptime(self.data[i - 1]['date'], '%Y-%m-%d')
                if prev_date <= new_date <= entry_date:
                    self.data.insert(i, self._validate_and_init_entry(new_entry))
                    return
            elif new_date <= entry_date:
                self.data.insert(0, self._validate_and_init_entry(new_entry))
                return
        
        if new_date >= datetime.strptime(self.data[-1]['date'], '%Y-%m-%d'):
            self.data.append(self._validate_and_init_entry(new_entry))
        else:
            raise ValueError("Invalid date position for new entry.")

    def has_matching_agenda_entry(self, start_date, end_date, query_string, exact_match, not_contains=None):
        """
        Checks if there is at least one medical history entry within the given date range
        whose 'agenda' field matches the provided string criteria.

        Args:
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.
            query_string (str): The string to compare against the 'agenda' field.
            exact_match (bool): If True, an exact match is required; if False, a case-insensitive
                                substring match is used.

        Returns:
            bool: True if a matching entry is found; False otherwise.
        """
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
       
        not_contains = []

        for entry in self.data:
            try:
                entry_date_obj = datetime.strptime(entry.get('date', ''), '%Y-%m-%d')
            except Exception:
                continue  # Skip entries with invalid or missing date formats

            if start_date_obj <= entry_date_obj <= end_date_obj:
                agenda = entry.get('agenda', '')
                if exact_match:
                    if agenda == query_string:
                        return True
                else:
                    if query_string.lower() in agenda.lower():
                        if not any(excl.lower() in agenda.lower() for excl in not_contains):
                            return True
                       
        return False

    def days_since_last_matching_agenda_entry(self, end_date, query_string, exact_match):
        """
        Finds the last matching agenda entry from the beginning of the entries up until the given end_date,
        and returns the day difference between the end_date and the date of that entry.

        Args:
            end_date (str): The end date in 'YYYY-MM-DD' format.
            query_string (str): The string to compare against the 'agenda' field.
            exact_match (bool): If True, requires an exact match; if False, a case-insensitive substring match is used.

        Returns:
            int: The difference in days between the end_date and the last matching agenda entry date.
                 Returns None if no matching entry is found.

        Raises:
            ValueError: If the end_date is not in a valid 'YYYY-MM-DD' format.
        """
        try:
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        except Exception as e:
            raise ValueError(f"Invalid end_date format. Expected 'YYYY-MM-DD', got {end_date}. Error: {e}")

        last_matching_date = None

        for entry in self.data:
            try:
                entry_date_obj = datetime.strptime(entry.get('date', ''), '%Y-%m-%d')
            except Exception:
                continue  # Skip entries with invalid or missing dates

            # Only consider entries that occur on or before the given end_date.
            if entry_date_obj <= end_date_obj:
                agenda = entry.get('agenda', '')
                if exact_match:
                    if agenda == query_string:
                        if last_matching_date is None or entry_date_obj > last_matching_date:
                            last_matching_date = entry_date_obj
                else:
                    if query_string.lower() in agenda.lower():
                        if last_matching_date is None or entry_date_obj > last_matching_date:
                            last_matching_date = entry_date_obj

        if last_matching_date is not None:
            return (end_date_obj - last_matching_date).days + 1
        else:
            return 0


    def _validate_and_init_entry(self, entry):
        """
        Validates and initializes a medical history entry.

        Args:
            entry (dict): A dictionary representing a single medical history entry.

        Returns:
            dict: A validated medical history entry.

        Raises:
            ValueError: If the entry does not match the medical_history_entry_model.
        """
        if not ModelUtils.validate_entry(entry, Models.medical_history_entry_model):
            raise ValueError(f"Invalid medical history entry data: {entry}")

        date = entry.get('date')
        if date and not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
            raise ValueError(f"Invalid date format. Expected YYYY-MM-DD but got: {date}")

        return entry

    def to_dict(self):
        """
        Converts the MedicalHistoryData object to a dictionary.
        """
        return {'data': self.data}

    def to_doc(self):
        return self.to_dict()

    def to_json(self):
        """
        Converts the MedicalHistoryData object to a JSON string.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def validate(medical_history_data):
        """
        Validates if the given dictionary contains all required fields and their values are not None.
        """
        data_entries = medical_history_data.get('data', [])
        return ModelUtils.validate_list_of_objects(data_entries, Models.medical_history_entry_model)

    @staticmethod
    def create_medical_history():
        return {'data':[]}

    @staticmethod
    def create_medical_history_entry(**kwargs):
        """
        Creates a medical history entry dictionary dynamically based on the medical_history_entry_model.
        """
        return [ModelUtils.create_object(Models.medical_history_entry_model, **kwargs)]
