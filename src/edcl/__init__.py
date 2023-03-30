import json
from importlib.resources import files
from .info_classes import *

def init_load_info() -> Info:
    """
    Loads information from the info.json file to the global Info object.

    Returns:
        None
    """
    return Info.from_json(json.loads(files('edcl').joinpath('info.json').read_text()))

info = init_load_info()

from .util import *
from .types import *
from .collections import *
from .grid_data_io_util import *