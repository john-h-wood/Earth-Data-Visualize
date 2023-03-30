"""
The globals module houses all global variables. These are initialized when edcl is imported.
"""

from .info_classes import Info
import json
from importlib.resources import files


# Instance of the Info class which stores all metadata
def init_load_info() -> Info:
    """
    Loads information from the info.json file to the global Info object.

    Returns:
        None
    """
    return Info.from_json(json.loads(files('edcl').joinpath('info.json').read_text()))


info = init_load_info()

# Which (grid) data is currently loaded? The answer and data are kept to avoid loading data unnecessarily
loaded_data = None
loaded_dataset = None
loaded_variable = None
loaded_year = None
loaded_month = None
