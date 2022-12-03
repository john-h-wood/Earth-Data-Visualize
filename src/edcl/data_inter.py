"""
The data interpretation (data_inter) manages all interactions with data.

The module uses info.json to read and store metadata and uses the definitions of the Variable, Dataset,
Info and Data Collection classes. These classes, and the functions in this module, are to be used either by scripts
or GUIs to interact with Earth data.

A functional and object-oriented style is used for maintainability.

Usage of this module with most often start with the creation of a DataCollection object. get_data_collection_names is
particularly useful for this. Other functions, like get_years are useful to find available data. From there,
statistical methods can be used to generate new data collections. Plotting is performed with plot_data_collection.

General Documentation:
    All coordinate limits formatted as (lat_min, lat_max, lon_min, lon_max).

    All times tuples (not simple year/month/day/hour integers) formatted as (year, month, day, hour). Each element
    can be None to specify a period of time:

    - year is None and month is not None: data for the given month over all available years.
    - year is None:                       data over all available years.
    - month is None:                      data over the given year.
    - day is None:                        data over the given month.
    - hour is None:                       data over the given day.

Credits
    JSON serializing and deserializing from https://yuchen52.medium.com/serialize-and-deserialize-complex-json-in-python
    -205ecc636caa by Yuchen Z. Accessed June 13, 2022

    Reshaping numpy array to compute percentileofsocre without nested for loops from https://stackoverflow.com/questions
    /48650645/how-to-calculate-percentile-of-score-along-z-axis-of-3d-array by KRKirov. Accessed June 23, 2022

    Calculating contour areas adapted from https://stackoverflow.com/questions/48634934/contour-area-calculation-using-
    matplotlib-path by Thomas Kühn. Accessed July 23, 2022

    Counting number of non-nan elements in array from https://stackoverflow.com/questions/21778118/counting-the-number-
    of-non-nan-elements-in-a-numpy-ndarray-in-python by M4rtini. Accessed November 18, 2022.

"""

# <editor-fold desc="IMPORTS">
# ============ IMPORTS =================================================================================================
import json
import pickle
import datetime
import warnings

from math import inf
from glob import glob
from matplotlib.path import Path
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod
from os.path import basename, isfile
from importlib.resources import files
from scipy.stats import percentileofscore
from typing import Optional, Callable
from matplotlib.figure import Figure as matFigure

import numpy as np
import scipy.io as sio
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
# ======================================================================================================================
# </editor-fold>

# <editor-fold desc="CONSTANTS">
# ============ CONSTANTS ===============================================================================================
# Typing constants
POINT_TYPE = tuple[float, float]
POINT_INDEX_TYPE = tuple[int, int]
LIMIT_TYPE = tuple[float, float, float, float]
TIME_TYPE = tuple[Optional[int], Optional[int], Optional[int], Optional[int]]
DATA_TYPE = list[ArrayLike, ...]
COMPONENT_TYPE = POINT_TYPE | ArrayLike | Path

# Constant values
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']
RADIUS_OF_EARTH = 6_371  # km

loaded_data = None
loaded_path = None
loaded_dataset = None
# ======================================================================================================================
# </editor-fold>

# <editor-fold desc="CORE CLASSES">
# =========== CORE CLASSES =============================================================================================
class Variable:
    """
    Class representing a variable and its metadata, including type, name, etc.
    """

    def __init__(self, name: str, kind: str, is_combo: bool, identifier: int, key: Optional[str],
                 equation: Optional[str], file_identifier: Optional[str]) -> None:
        self.name = name
        self.kind = kind
        self.is_combo = is_combo
        self.identifier = identifier
        self.key = key
        self.equation = equation
        self.file_identifier = file_identifier

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_json(cls, data):
        """
        Converts dictionary from JSON read to object.
        Args:
            data: The dictionary.

        Returns:
            The object.

        """
        return cls(**data)


class Dataset:
    """
    Class representing a dataset and its metadata, including file location and variables.
    """

    def __init__(self, directory: str, name: str, is_unified: bool, file_prefix: str, file_suffix: str,
                 variables: tuple[Variable, ...]) -> None:
        self.directory = directory
        self.name = name
        self.is_unified = is_unified
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.variables = variables

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_json(cls, data):
        """
        Converts dictionary from JSON read to object.
        Args:
            data: The dictionary.

        Returns:
            The object.

        """
        variables = list(map(Variable.from_json, data['variables']))
        return cls(data['directory'], data['name'], data['is_unified'], data['file_prefix'], data['file_suffix'],
                   variables)


class Info:
    """
    Class storing all information for available data and tools, including file location, and a list of all datasets.
    Includes visualisation methods such as projections and figure output modes.
    """

    def __init__(self, directory: str, projections: tuple[str, ...], graph_styles: tuple[str, ...],
                 graph_out_modes: tuple[str, ...], datasets: tuple[Dataset, ...]) -> None:
        self.directory = directory
        self.projections = projections
        self.graph_styles = graph_styles
        self.graph_out_modes = graph_out_modes
        self.datasets = datasets

    def __str__(self) -> str:
        return f'Info object of directory \'{self.directory}\''

    @classmethod
    def from_json(cls, data):
        """
        Converts dictionary from JSON read to object.
        Args:
            data: The dictionary.

        Returns:
            The object.

        """
        datasets = list(map(Dataset.from_json, data['datasets']))
        return cls(data['directory'], data['projections'], data['graph_styles'], data['graph_out_modes'], datasets)


class Graphable(ABC):
    """
    Abstract base class for a graphable object. These have data with spatial and time limits, and titles for each
    time stamp.
    """

    @abstractmethod
    def get_time_length(self) -> float:
        ...

    @abstractmethod
    def get_time_title(self, time_index) -> str:
        ...

    @abstractmethod
    def get_limits(self) -> LIMIT_TYPE:
        ...
# ======================================================================================================================
# </editor-fold>


# ============ DERIVED CLASSES =========================================================================================
class PointCollection(Graphable):
    """
    Class storing all information for a point, including dataset, variable, time, position over time, etc.
    """

    def __init__(self, dataset: Dataset, variable: Variable, time: TIME_TYPE, latitudes: tuple[float, ...],
                 longitudes: tuple[float, ...], title_prefix: str, title_suffix: str,
                 time_stamps: Optional[tuple[TIME_TYPE, ...]]):
        self.dataset = dataset
        self.variable = variable
        self.time = time
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.title_prefix = title_prefix
        self.title_suffix = title_suffix

        # Check and set time stamps
        if time_stamps is None:
            self.time_stamps = get_time_stamps(dataset, variable, time)
        else:
            if len(time_stamps) == self.get_time_length():
                self.time_stamps = time_stamps
            else:
                raise ValueError('The number of custom time stamps must be equal to the length of data along the '
                                 'time axis.')

    def __str__(self):
        return f'{self.title_prefix}{_time_suffix(self.time)}{self.title_suffix}'

    def get_time_length(self) -> int:
        """
        The time length of the point data.
        Returns:
            The time length.

        """
        return len(self.latitudes)

    def get_time_title(self, time_index) -> str:
        """
        The point title for a given time.
        Args:
            time_index: The time index.

        Returns:
            The time title.

        """
        return f'{self.title_prefix}{_time_suffix(self.time_stamps[time_index])}{self.title_suffix}'

    def get_limits(self) -> LIMIT_TYPE:
        """
        The spatial limits of the point. These are the maximal limits over time.
        Returns:
            The limits.

        """
        return min(self.latitudes), max(self.latitudes), min(self.longitudes), max(self.longitudes)

    def get_component(self, time_index) -> POINT_TYPE:
        """
        The value of the point at a certain time.
        Args:
            time_index:  The time index.

        Returns:
            The value of the point.

        """
        return self.latitudes[time_index], self.longitudes[time_index]


class PathCollection(Graphable):
    """
    Class storing all information for a path, including dataset, variable, time, path points over time, etc.
    """

    def __init__(self, dataset: Dataset, variable: Variable, time: TIME_TYPE, paths: tuple[Path, ...],
                 title_prefix: str, title_suffix: str, time_stamps: Optional[tuple[TIME_TYPE, ...]]):
        self.dataset = dataset
        self.variable = variable
        self.time = time
        self.paths = paths
        self.title_prefix = title_prefix
        self.title_suffix = title_suffix

        if time_stamps is None:
            self.time_stamps = get_time_stamps(dataset, variable, time)
        else:
            if len(time_stamps) == self.get_time_length():
                self.time_stamps = time_stamps
            else:
                raise ValueError('The number of custom time stamps must be equal to the length of data along the '
                                 'time axis.')

    def __str__(self):
        return f'{self.title_prefix}{_time_suffix(self.time)}{self.title_suffix}'

    def get_time_length(self) -> int:
        """
        The time length of the point data.
        Returns:
            The time length.

        """
        return len(self.paths)

    def get_time_title(self, time_index) -> str:
        """
        The title of the point at a certain time.
        Args:
            time_index: The time index.

        Returns:
            The title.

        """
        return f'{self.title_prefix}{_time_suffix(self.time_stamps[time_index])}{self.title_suffix}'

    def get_limits(self) -> LIMIT_TYPE:
        """
        The spatial limits of the point. These are the maximal limits over time.
        Returns:
            The spatial limits.

        """
        limits = list()
        for path in self.paths:
            ex = path.get_extents()
            limits.append((ex.ymin, ex.ymax, ex.xmin, ex.xmax))

        return _maximal_limits(tuple(limits))

    def contains_point(self, time_index: int, point: POINT_TYPE) -> bool:
        """
        Whether the path, at a certain time, contains a point.
        Args:
            time_index: The time index.
            point: The point.

        Returns:
            Whether the point is contained.

        """
        lat, lon = point
        return self.paths[time_index].contains_point((lon, lat))

    def get_area(self, time_index: int, point_condition: Optional[Callable[[float, float], bool]]) -> float:
        """
        The area of the path at a given time, perhaps refined to include only those points meeting a condition.

        The optional point_condition parameter is a function which determines whether each point is included in the
        area calculation. Eliminating points is referred to as 'refining' the area. Each point has an area that is
        approximated using spherical integration, with latitude and longitude values obtained from the path's dataset.
        Args:
            time_index: The time index.
            point_condition: The point_conditions. Possibly None.

        Returns:
            The (possibly refined) area.

        """

        coo_index, latitude, longitude = _get_coordinate_information(self.dataset, self.get_limits())
        area_grid = _compute_area_grid(latitude, longitude)

        is_contained = list()

        for lat in latitude:
            for lon in longitude:
                if self.contains_point(time_index, (lat, lon)):
                    if point_condition is not None:
                        is_contained.append(int(point_condition(lat, lon)))
                    else:
                        is_contained.append(1)
                else:
                    is_contained.append(0)

        is_contained = np.reshape(is_contained, (len(latitude), len(longitude)))
        return float(np.sum(np.multiply(is_contained, area_grid)))

    def get_component(self, time_index: int) -> Path:
        """
        The path at a given time.
        Args:
            time_index: The time index.

        Returns:
            The path.

        """
        return self.paths[time_index]


class DataCollection(Graphable):
    """
    Class storing all information for certain data, including its dataset, variable, time, etc. This might,
    for example, store wind speed over time for a square region on the Earth.
    """

    def __init__(self, dataset: Dataset, variable: Variable, time: TIME_TYPE, limits: Optional[LIMIT_TYPE],
                 data: DATA_TYPE, latitude: ArrayLike, longitude: ArrayLike, title_prefix: str, title_suffix: str,
                 time_stamps: Optional[tuple[TIME_TYPE]]):
        self.dataset = dataset
        self.variable = variable
        self.time = time

        if limits is None:
            raise ValueError('Automatic limits not yet supported')
        else:
            self.limits = limits

        self.data = data
        self.latitude = latitude
        self.longitude = longitude
        self.title_prefix = title_prefix
        self.title_suffix = title_suffix

        # Set shape, spread and vector dimension (quantity of components)
        self.shape = np.shape(data[0])
        self.spread = (len(latitude), len(longitude))
        self.dimension = len(data)

        # Check and set time stamps
        if time_stamps is None:
            self.time_stamps = get_time_stamps(dataset, variable, time)
        else:
            if len(time_stamps) == self.get_time_length():
                self.time_stamps = time_stamps
            else:
                raise ValueError('The number of custom time stamps must be equal to the length of data along the '
                                 'time axis.')

    def __str__(self):
        return f'{self.title_prefix}{_time_suffix(self.time)}{self.title_suffix}'

    def get_time_length(self) -> int:
        """
        The time length of the data.
        Returns:
            The time length.

        """
        return self.shape[0]

    def get_time_title(self, time_index) -> str:
        """
        The title of the data at a certain time.
        Args:
            time_index: The time index.

        Returns:
            The title.

        """
        return f'{self.title_prefix}{_time_suffix(self.time_stamps[time_index])}{self.title_suffix}'

    def get_limits(self) -> LIMIT_TYPE:
        """
        The spatial limits of the data. These are the maximal limits over time.
        Returns:
            The limits.

        """
        return self.limits

    def get_component(self, time_index, component_index) -> ArrayLike:
        """
        A certain component of the data at a certain time.
        Examples:
            If the DataCollection is storing wind vectors, component index 0 might be the horizontal component while
            index 1 might be the vertical component.
        Args:
            time_index: The time index.
            component_index: The component index.

        Returns:
            The data component.

        """
        if component_index >= len(self.data):
            raise ValueError('The given component index is too large.')

        if time_index is None:
            return self.data[component_index]

        if time_index >= self.get_time_length():
            raise ValueError('The given time index is too large.')
        return self.data[component_index][time_index]

    def get_coordinate_value(self, time_index, component_index, latitude, longitude) -> ArrayLike:
        """
        The value of the data for a certain component, time, and position.
        Args:
            time_index: The time index.
            component_index: The component index.
            latitude: The latitude.
            longitude: The longitude.

        Returns:
            The data value.

        Raises:
            ValueError: No data for the given coordinates.
            ValueError: The given component index is too large.
            ValueError: The given time index is too large.

        """
        lat_idx = np.asarray(self.latitude == latitude).nonzero()[0]
        lon_idx = np.asarray(self.longitude == longitude).nonzero()[0]

        if len(lat_idx) == 0 or len(lon_idx) == 0:
            raise ValueError('No data for the given coordinates.')

        lat_idx = lat_idx[0]
        lon_idx = lon_idx[0]

        if component_index >= len(self.data):
            raise ValueError('The given component index is too large.')

        if time_index is None:
            return self.data[component_index][:, lat_idx, lon_idx]

        if time_index >= self.get_time_length():
            raise ValueError('The given time index is too large.')
        # noinspection PyTypeChecker
        return self.data[component_index][time_index, lat_idx, lon_idx]

    def get_vector_dimension(self) -> int:
        """
        The number of components for the data.
        Returns: The number of components.

        """
        return len(self.data)

    def time_order(self) -> None:
        for component in self.data:
            # noinspection PyUnresolvedReferences
            component.sort(axis=0)
        self.title_prefix = 'Ordered ' + self.title_prefix

# ======================================================================================================================

# <editor-fold desc="SETUP">
# ============ SETUP ===================================================================================================
def init_load_info() -> Info:
    """
    Loads information from the info.json file to the core Info object.

    Returns:
        The core Info object.

    """
    return Info.from_json(json.loads(files('edcl').joinpath('info.json').read_text()))


def _function_call():
    """
    Increments the function call count and prints the new count if it is a mutiple of function_call_multiple. The
    function call count is used to see that the module is indeed running when performing long tasks.

    Returns:
        None.

    """
    global function_calls
    function_calls += 1
    if function_calls % function_call_multiple == 0:
        print(function_calls)


info = init_load_info()
function_calls = 0
function_call_multiple = 1_000
# ======================================================================================================================
# </editor-fold>





















