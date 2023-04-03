"""
The data interpretation (data_inter) module manages all interactions with data.

The module uses info.json to read and store metadata and contains the definitions of the Variable, Dataset,
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

    Creating video from series of images from https://stackoverflow.com/a/62434934 by trygvrad and Mr_and_Mrs_D.
    Accessed March 7, 2023. Adapted using moviepy documentation.

"""

import json
import pickle
import os.path
import datetime
import warnings
import numpy as np
import scipy.io as sio
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches

from math import inf
from glob import glob
from natsort import os_sorted
from matplotlib.path import Path
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod
from os.path import basename, isfile
from importlib.resources import files
from scipy.stats import percentileofscore
from typing import Union, Optional, Callable
from matplotlib.figure import Figure as matFigure
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

POINT_TYPE = tuple[float, float]
POINT_INDEX_TYPE = tuple[int, int]
LIMIT_TYPE = tuple[float, float, float, float]
TIME_TYPE = tuple[Optional[int], Optional[int], Optional[int], Optional[int]]
DATA_TYPE = list[ArrayLike] | list[ArrayLike, ArrayLike]
COMPONENT_TYPE = POINT_TYPE | ArrayLike | Path


# ======================== CLASSES =====================================================================================
class Variable:
    """
    Class representing a variable and its metadata, including type, name, etc.
    """

    def __init__(self, name: str, kind: str, is_combo: bool, identifier: int, key: Union[str, None], equation: Union[
        str, None], file_identifier: Union[str, None]) -> None:
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

    def __init__(self, directory: str, name: str, is_unified: bool, file_prefix: str, file_suffix: str, variables:
    list[Variable]) -> None:
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

    def __init__(self, directory: str, projections: list[str], graph_styles: list[str], graph_out_modes: list[str],
                 datasets: list[Dataset]) -> None:
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

    def __init__(self, dataset: Dataset, variable: Variable, time: TIME_TYPE, paths: tuple[Path, ...], title_prefix:
    str, title_suffix: str, time_stamps: Optional[tuple[TIME_TYPE, ...]]):
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
            self.limits = (latitude[-1], latitude[0], longitude[0], longitude[-1])
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

        return self.get_coordinate_index(time_index, component_index, lat_idx, lon_idx)


    def get_coordinate_index(self, time_index, component_index, lat_index, lon_index):
        if component_index >= len(self.data):
            raise ValueError('The given component index is too large.')

        if time_index is None:
            return self.data[component_index][:, lat_index, lon_index]

        if time_index >= self.get_time_length():
            raise ValueError('The given time index is too large.')
            # noinspection PyTypeChecker
        return self.data[component_index][time_index, lat_index, lon_index]

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
        # TODO remove time stamps sinnce they are meaningless after time ordering of data
        # TODO also allow for data to be loaded without time stamps (negative variable ids are time ordered)
        self.title_prefix = 'Ordered ' + self.title_prefix


# ======================== CONSTANTS AND GLOBALS  ======================================================================
def init_load_info() -> Info:
    """
    Loads information from the info.json file to the global Info object.

    Returns:
        None

    """
    return Info.from_json(json.loads(files('edcl').joinpath('info.json').read_text()))


def _function_call():
    """
    Increments the function call count and prints the new count if it is a multiple of five. The function call count
    is used to see that the module is indeed running when performing long tasks.

    Returns:
        None

    """
    global function_calls
    function_calls += 1
    if function_calls % 1_000 == 0:
        print(function_calls)


info = init_load_info()

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']
RADIUS_OF_EARTH = 6_371  # km

loaded_data = None
loaded_path = None
loaded_dataset = None
function_calls = 0


# ======================== USEFUL FUNCTIONS (CONVERT, ETC) =============================================================
def _month_convert(value: str | int) -> str | int:
    """
    Converts a month number to a month name and vice-versa.

    Args:
        value: Either the month number or month name.

    Examples:
        _month_convert(4) -> April
        _month_convert('April') -> 4

    Returns:
        Either the month number or the month name.

    """
    _function_call()
    if isinstance(value, str):
        return MONTHS.index(value) + 1
    else:
        return MONTHS[value - 1]


def iterable_to_words(values: tuple) -> str:
    """
    Returns a formatted string representation of a tuple of elements.

    Formats the elements in English form, with the Oxford comma. If the tuple is empty, 'No value' is returned. If
    the tuple contains a single element, that element is returned as a string.

    Examples:
        ('A', 'B', 'C') -> 'A, B and C'
        (1) -> '1'
        () -> 'No value'

    Args:
        values: The values.

    Returns:
        The string representation.

    """
    _function_call()

    if len(values) == 0:
        return 'No value'
    elif len(values) == 1:
        return str(values[0])
    else:
        words = str()
        for i in range(len(values) - 1):
            words += str(values[i]) + ', '
        words += 'and ' + str(values[-1])
        return words


def _month_format(month: int) -> str:
    """
    Formats a month number as a two-character-long string.

    Particularly useful for file path generation where months are represented with two characters.

    Example:
        The input 3 returns '03' while 12 returns '12'.

    Args:
        month: The month.

    Returns:
        The formatted string.

    """
    _function_call()

    res = str(month)
    return '0' + res if len(res) == 1 else res


def _time_suffix(time: TIME_TYPE) -> str:
    """
    Returns a title suffix for analysis over periods of time, or at a specific time.

    Uses time period convention detailed in the documentation for _get_data.

    Examples:
        An example is given for each possible outcome:
        year: None,   month: 4,      day: None,   hour: None   -> 'over April's'
        year: None,   month: None,   day: None,   hour: None   -> 'over all time'
        year: 2006,   month: None,   day: None,   hour: None   -> 'over 2006'
        year: 2007,   month: 3,      day: None,   hour: None   -> 'over March 2007'
        year: 2012,   month: 2,      day: 20,     hour: None   -> 'over February  20, 2012'
        year: 2018,   month: 6,      day: 11,     hour: 12     -> 'on June 11, 2018 at 12H'

    Args:
        time: The time period as (year, month, day, hour), all elements possibly being None.

    Returns:
        The suffix.

    """
    _function_call()
    year, month, day, hour = time

    if (year is None) and (month is not None):
        return f'over {_month_convert(month)}\'s'
    elif year is None:
        return 'over all time'
    elif month is None:
        return f'over {year}'
    elif day is None:
        return f'over {_month_convert(month)} {year}'
    elif hour is None:
        return f'over {_month_convert(month)} {day}, {year}'
    elif all((year is not None, month is not None, day is not None, hour is not None)):
        return f'on {_month_convert(month)} {day}, {year} at {hour}H'


def _time_stamps_to_filenames(time_stamps: tuple[TIME_TYPE, ...]) -> tuple[str]:
    """
    Converts a tuple of time stamps to file names.

    Useful for generating automatic file names for plotting.

    Args:
        time_stamps: The time stamps.

    Returns:
        The file names.

    """
    names = list()

    for stamp in time_stamps:
        name = str()
        for piece in stamp:
            if piece is None:
                name += '-_'
            else:
                name += f'{str(piece)}_'

        names.append(name[:-1] + '.png')

    return tuple(names)


def _coordinates_to_formatted(latitude: float, longitude: float) -> str:
    """
    Formats a point's coordinates to a string with degree symbol.

    Examples:
        -9.6, 80 -> 9.6° S, 80° E
        10.2, -5.1 -> 10.2° N, 5.1° W
    Args:
        latitude: The latitude.
        longitude: The longitude.

    Returns:
        The formatted coordinate.

    """
    result = str()

    if latitude >= 0:
        result += f'{latitude}\N{DEGREE SIGN} N, '
    else:
        result += f'{abs(latitude)}\N{DEGREE SIGN} S, '

    if longitude >= 0:
        result += f'{longitude}\N{DEGREE SIGN} E'
    else:
        result += f'{abs(longitude)}\N{DEGREE SIGN} W'

    return result


def load_pickle(file_path: str):
    """
    Load a pickle file.
    Args:
        file_path: The file path.

    Returns:
        The pickled object.

    """
    with open(file_path, 'rb') as temp_file:
        return pickle.load(temp_file)


def save_pickle(obj: object, file_path: str) -> None:
    """
    Pickle an object using the highest protocol.
    Args:
        obj: The object
        file_path: The file path.

    Returns:
        None.

    """
    with open(file_path, 'wb') as temp_file:
        pickle.dump(obj, temp_file, pickle.HIGHEST_PROTOCOL)


def _to_tuple(element: object) -> tuple:
    """
    Convert an object to a tuple, if it not already one.
    Args:
        element: The object.

    Returns:
        The object as a tuple, if it is not already one. Otherwise, the object.

    """
    if not isinstance(element, tuple):
        return element,
    else:
        return element


def _output_figure(out_mode: str, save_title: Optional[str]) -> matFigure or None:
    """
    Output manager for the current matplotlib plot.

    Actions that can be performed are:
    out_mode = 'show' -> shows the plot. Returns None.
    out_mode = 'save' -> saves the plot using the given title (support for .png, .ps and .eps). Returns None.
    out_mode=  'fig'  -> returns the current figure.

    Args:
        out_mode: The output mode.
        save_title: The save title. Possibly None.

    Returns:
        None or matplotlib Figure.

    """
    if out_mode == 'show':
        plt.show()
    elif out_mode == 'save':
        if save_title.endswith('.png') or save_title.endswith('.jpg'):
            plt.savefig(save_title, dpi=300, backend='AGG')
        elif save_title.endswith('.eps') or save_title.endswith('.ps'):
            plt.savefig(save_title, format='eps', backend='PS')
        else:
            plt.savefig(save_title)
    elif out_mode == 'fig':
        return plt.gcf()


# ======================== GET FUNCTIONS ===============================================================================
def get_dataset_name(name: str) -> Dataset:
    """
    Get the dataset object with the given name.

    Accesses the global Info object.

    Args:
        name: The name of the dataset.

    Returns:
        The dataset.

    Raises:
        ValueError: No such dataset was found.

    """
    _function_call()

    for dataset in info.datasets:
        if dataset.name == name:
            return dataset
    raise ValueError('No such dataset was found.')


def get_variable_name(dataset, name: str) -> Variable:
    """
    Get the first variable object with the given name from a dataset.

    Accesses the global Info object. Ideally, the variables within each dataset have unique names and so this
    function returns * the * variable with the given name.

    Args:
        dataset: The dataset.
        name: The name of the variable.

    Returns:
        The variable.

    Raises:
        ValueError: No such variable was found.

    """
    _function_call()

    for variable in dataset.variables:
        if variable.name == name:
            return variable
    raise ValueError('No such variable was found.')


def get_data_collection_names(dataset_name: str, variable_name: str, limits: Optional[LIMIT_TYPE], time: TIME_TYPE) -> \
        DataCollection:
    """
    Get a data collection with the given information. Useful for scripts and GUIs to avoid having to get variable and
    dataset objects.

    Args:
        dataset_name: The name of the dataset.
        variable_name: The name of the variable
        limits: The limits.
        time: The time.

    Returns:
        The data collection.

    """
    dataset = get_dataset_name(dataset_name)
    variable = get_variable_name(dataset, variable_name)
    return _get_data(dataset, variable, limits, time)


def _get_variable_identifier(dataset, identifier: int) -> Variable:
    """
    Get the first variable object with the given identifier from a dataset.

    Accesses the global Info object. The variables within each dataset should have unique identifiers,
    so this function should return * the * variable with the given identifier.

    Args:
        dataset: The dataset.
        identifier: The identifier.

    Returns:
        The variable.

    Raises:
        ValueError: No such variable was found.

    """
    _function_call()

    for variable in dataset.variables:
        if variable.identifier == identifier:
            return variable
    raise ValueError('No such variable was found.')


def get_projection_name(name: str, limits: LIMIT_TYPE = None) -> ccrs.Projection:
    """
    Returns the cartopy.crs object for the given projection.

    If coordinates are not none and the selected projection is Lambert Conformal, the projection is centred on the
    central latitudes and longitudes.

    Args:
        name: The name of projection.
        limits: The limits.

    Returns: The object.

    """
    _function_call()

    if name == 'Lambert':
        if limits is not None:
            return ccrs.LambertConformal(central_longitude=np.mean(limits[2:]), central_latitude=np.mean(limits[:2]))
        return ccrs.LambertConformal(central_longitude=-35.0, central_latitude=54)
    elif name == 'Robinson':
        return ccrs.Robinson()
    return ccrs.PlateCarree()


# ======================== COMPUTED METADATA ===========================================================================
def get_years(dataset: Dataset, variable: Optional[Variable] = None) -> list[int]:
    """
    Returns a sorted list of years for which any data, or a specific variable, is available.

    If the variable is None, then years for which any data is available are included. Otherwise, only years which
    have some data for the variable are included.

    Args:
        dataset: The dataset.
        variable: The year. Possibly None.

    Returns:
        The years.

    """
    _function_call()

    years = list()

    # Rely on organization within main dataset directory being in years
    # Note, not all files in year directories have the correct prefix, so check for that too

    for path in glob(f'{info.directory}/{dataset.directory}/*'):
        # determine if valid year
        test_year = int(basename(path))
        if variable is None:
            for sub_path in glob(f'{path}/*'):
                if variable is None and basename(sub_path).startswith(dataset.file_prefix):
                    years.append(test_year)
                    break
        else:
            if len(get_months(dataset, test_year, variable)) > 0:
                years.append(test_year)

    return sorted(years)


def get_months(dataset: Dataset, year: int, variable: Optional[Variable] = None) -> list[int]:
    """
    Returns a sorted list of months for which any data, or a specific variable, is available.

    If the variable is None, then months for which any data is available are included. Otherwise, only months which
    have some data for the variable are included.

    Args:
        dataset: The dataset.
        year: The year.
        variable: The variable. Possibly None.

    Returns:
        The months.

    """
    _function_call()

    months = list()
    # TODO add special case for non-unified datasets. Variable must available for a month to be included, unless the
    #  inputted variable is None

    # Access files within the given year's directory
    # Note, not all files in year directories have the correct prefix, so check for that too

    for path in glob(f'{info.directory}/{dataset.directory}/{year}/*'):
        sub_path = basename(path)
        test_month = int(sub_path[sub_path.rindex('_') - 8:sub_path.rindex('_') - 6])  # relies heavily on formatting!
        if sub_path.startswith(dataset.file_prefix) and (variable is None or isfile(_get_path(dataset, year,
                                                                                              test_month, variable))):
            months.append(test_month)

    # TODO create information document on data assumptions and formatting

    return sorted(list(set(months)))


def get_days(dataset: Dataset, year: int, month: int) -> list[int]:
    """
    Returns a sorted list of days available within a month.

    This function has no Variable parameter because the specified month already takes the variable of interest into
    account.

    Args:
        dataset: The dataset.
        year: The year.
        month: The month.

    Returns:
        The days.

    """
    _function_call()

    data = _load(dataset, year, month, None)
    # noinspection PyTypeChecker
    return np.unique(data['day_ts']).tolist()


def get_hours(dataset: Dataset, year: int, month: int, day: int) -> list[int]:
    """
    Returns a sorted list of hours available within a day.

    This function has no Variable parameter because the specified month already takes the variable of interest into
    account.

    Args:
        dataset: The dataset.
        year: The year.
        month: The month.
        day: The day.

    Returns:
        The hours.

    """
    # Find indices of the given day and the associated hours
    _function_call()

    data = _load(dataset, year, month, None)
    days = data['day_ts']
    hours = data['hour_ts']
    hour_inds = np.asarray(days == day).nonzero()[0]

    return hours[hour_inds[0]:hour_inds[-1] + 1].tolist()


def get_time_stamps(dataset: Dataset, variable: Variable, time: TIME_TYPE) -> tuple[TIME_TYPE, ...]:
    """
    Creates and returns timestamps for the data collection's data. There is one time stamp for each index of
    the data's time axis. So, for a data collection dc, len(dc.get_time_stamps) == np.shape(dc.data)[0].

    Time stamps are formatted as a tuple: (year, month, day, hour).

    Returns:
        The time stamps.

    """
    time_stamps = list()
    range_year, range_month, range_day, range_hour = time

    if (range_month is not None) and (range_year is None):
        years = [x for x in get_years(dataset) if range_month in get_months(dataset, x, variable)]
        for year in years:
            for day in get_days(dataset, year, range_month):
                for hour in get_hours(dataset, year, range_month, day):
                    time_stamps.append((year, range_month, day, hour))
    elif range_year is None:
        for year in get_years(dataset, variable):
            for month in get_months(dataset, year, variable):
                for day in get_days(dataset, year, month):
                    for hour in get_hours(dataset, year, month, day):
                        time_stamps.append((year, month, day, hour))
    elif range_month is None:
        for month in get_months(dataset, range_year, variable):
            for day in get_days(dataset, range_year, month):
                for hour in get_hours(dataset, range_year, month, day):
                    time_stamps.append((range_year, month, day, hour))
    elif range_day is None:
        for day in get_days(dataset, range_year, range_month):
            for hour in get_hours(dataset, range_year, range_month, day):
                time_stamps.append((range_year, range_month, day, hour))
    elif range_hour is None:
        for hour in get_hours(dataset, range_year, range_month, range_day):
            time_stamps.append((range_year, range_month, range_day, hour))
    else:
        time_stamps.append(time)

    return tuple(time_stamps)

def is_available_names(dataset_name: str, variable_name: str, time: TIME_TYPE) -> bool:
    dataset = get_dataset_name(dataset_name)
    variable = get_variable_name(dataset, variable_name)

    year, month, day, hour = time
    if year not in get_years(dataset, variable):
        return False
    if month not in get_months(dataset, year, variable):
        return False
    if day not in get_days(dataset, year, month):
        return False
    if hour not in get_hours(dataset, year, month, day):
        return False
    return True


# ======================== DATA LOADING ================================================================================
def _get_path(dataset: Dataset, year: Optional[int], month: Optional[int], variable: Optional[Variable]) -> str:
    """
    Get the filepath for selected data.

    The variable being None makes no difference if the dataset is unified. Otherwise, an arbitrary, non-combo variable
    is found to find some filepath for the specified dataset, year and month.

    Args:
        dataset: The dataset.
        year: The year.
        month: The month.
        variable: The variable. Possibly None.

    Returns:
        The filepath.

    Raises:
        ValueError: No available non-combo variable found.

    """
    _function_call()

    if year is None and month is None:
        year = get_years(dataset, variable)[0]
        month = get_months(dataset, year, variable)[0]

    path = f'{info.directory}/{dataset.directory}/{year}/{dataset.file_prefix}'

    # Dataset is not unified
    if not dataset.is_unified:
        # Find arbitrary, non-combo, variable if none is provided
        if variable is None:
            for v in dataset.variables:
                if (not v.is_combo) and (isfile(_get_path(dataset, year, month, v))):
                    variable = v
                    break

            if variable is None:
                raise ValueError('No available non-combo variable found.')

        path += f'_{variable.file_identifier}'

    return path + f'_m{_month_format(month)}_y{year}_{dataset.file_suffix}.mat'


def _load(dataset: Dataset, year: Optional[int], month: Optional[int], variable: Optional[Variable]) -> dict:
    """
    Loads desired data, filepath and dataset to global variables. Returns the loaded data.

    Desired data is specified with a dataset, date, time and variable. If no variable is provided and the dataset
    is not unified, one is chosen by the get_path function. A None variable should be used for when the used
    variable does not matter, for example to gather latitude and longitude information. Using this function
    increases efficiency by ensuring that dats is not needlessly loaded.

    Args:
        dataset: The dataset.
        year: The year.
        month: The month.
        variable: The variable. Possibly None.

    Returns:
        The data as a dictionary.

    """
    # TODO There are other cases where new data shouldn't be loaded. If the variable is none, but year and month are
    #  the same, keep current data (should introduce error on trying to time-index sorted wind data)
    _function_call()

    global loaded_path, loaded_data, loaded_dataset

    if (year is None) and (month is None) and (variable is None) and (dataset == loaded_dataset):
        return loaded_data

    path = _get_path(dataset, year, month, variable)
    if path != loaded_path:
        loaded_data = sio.loadmat(path, squeeze_me=True)
        loaded_path = path
        loaded_dataset = dataset

    return loaded_data


def _get_time_index(dataset: Dataset, year: int, month: int, day: int, hour: int) -> int:
    """
    Get the time index of the specified dataset, date and time.

    Data is stored in matrices which have their first dimension as an index for each available hour. This
    function finds that index, to then be used to get hour-specific data.

    Args:
        dataset: The dataset.
        year: The year.
        month: The month,
        day: The day.
        hour: The hour.

    Returns:
        The index.

    """
    _function_call()

    data = _load(dataset, year, month, None)

    days = data['day_ts']
    hours = data['hour_ts']

    hour_inds = np.asarray(days == day).nonzero()[0]
    hours_sub = hours[hour_inds[0]:hour_inds[-1] + 1].tolist()

    return hour_inds[0] + hours_sub.index(hour)


def _get_coordinate_information(dataset: Dataset, limits: Optional[LIMIT_TYPE]) -> tuple[LIMIT_TYPE, ArrayLike,
ArrayLike]:
    # TODO efficiency could probably be improved, with the assumption that coordinates only increase
    """
    Get the coordinate indices corresponding to given coordinate limits, and coordinates cut to those limits.

    Data is stored in matrices which have their first dimension as time. The second and third dimensions refer,
    respectively, to latitude and longitude. This function returns the index limits for latitude and longitude,
    given coordinate limits. That is, the upper and lower indices for which both latitude and longitude are within or
    equal to specified bounds. One is added to the upper indices so that a call such as lat[lat_ind_min:lat_ind_max]
    yields the expected latitudes.

    Limits are formatted as (lat_min, lat_max, lon_min, lon_max). Return is a tuple with similar ordering,
    but with indices, the cut latitudes, and the cut longitudes.

    If the limits are None, the returned indices correspond to all corrinate elements and the returned lat/lon
    vectors constitute all available coordinates.

    Examples:
        Limits: (-2, 0, 0, 1)
        Latitude: [-5, -4, -3, -2, -1, 0, 1, 2, 3]
        Longitude: [-5, -4, -3, -2, -1, 0, 1, 2, 3]

        Return: (3, 6, 5, 7)

    Args:
        dataset: The dataset.
        limits: The limits. Possibly None.

    Returns:
        The indices, and latitude and longitude cut to the limits.

    """
    _function_call()

    data = _load(dataset, None, None, None)

    latitude = data['lat']
    longitude = data['lon']

    if limits is None:
        return (0, len(latitude), 0, len(longitude)), latitude, longitude

    lat_idx = np.asarray((latitude >= limits[0]) & (latitude <= limits[1])).nonzero()[0]
    lon_idx = np.asarray((longitude >= limits[2]) & (longitude <= limits[3])).nonzero()[0]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise IndexError('No data found for the given limits')

    coo_index = lat_idx[0], lat_idx[-1] + 1, lon_idx[0], lon_idx[-1] + 1
    latitude = latitude[coo_index[0]:coo_index[1]]
    longitude = longitude[coo_index[2]:coo_index[3]]

    return coo_index, latitude, longitude


def _compute_variable_equation(equation_type: str, x: ArrayLike, y: ArrayLike) -> list[ArrayLike] | list[ArrayLike,
ArrayLike]:
    """
    Get the result of a combo variable equation.

    Takes an equation type and two component variables to yield a single combo variable.

    Supported equations are:
    - 'norm': np.sqrt(np.square(x) + np.square(y))
    - 'component': (x, y)
    - 'polar': (x * np.sin(np.deg2rad(y)), x * np.cos(np.deg2rad(y)))
    - 'direction': np.rad2deg(np.arctan2(y, x)) - 90

    Args:
        equation_type: The equation.
        x: The first component.
        y: The second component.

    Returns:
        The combo variable as a tuple of Numpy arrays, one for each resulting component.

    """
    _function_call()

    if equation_type == 'component':
        return [x, y]
    elif equation_type == 'polar':
        return [x * np.sin(np.deg2rad(y)), x * np.cos(np.deg2rad(y)), ]
    elif equation_type == 'norm':
        return [np.sqrt(np.square(x) + np.square(y)), ]
    elif equation_type == 'direction':
        return [np.rad2deg(np.arctan2(y, x)) - 90, ]  # TODO use oceanographic convention. This equation is wrong.


def _get_data(dataset: Dataset, variable: Variable, limits: Optional[LIMIT_TYPE], time: TIME_TYPE) -> DataCollection:
    """
    Gathers data for a variable, dataset, time, and coordinate limits. Includes
    coordinates. Performs any calculations coming from combo variables.

    Args:
        dataset: The dataset.
        variable: The variable.
        time: The time.

    Returns:
        A tuple containing latitudes, longitudes and list of datas (ordered by input variables)

    """
    _function_call()

    coo_index, latitude, longitude = _get_coordinate_information(dataset, limits)
    cut_interpret_data = _cut_interpret_data(dataset, variable, coo_index, time)

    title_prefix = f'{dataset.name}: {variable.name} '

    return DataCollection(dataset, variable, time, limits, cut_interpret_data, latitude, longitude, title_prefix, '',
                          None)


def _cut_interpret_data(dataset: Dataset, variable: Variable, coo_index: tuple[float, float, float, float],
                        time: TIME_TYPE) -> list[ArrayLike] | list[ArrayLike, ArrayLike]:
    """
    Gathers data for a variable, dataset, time, and cut to coordinate limits.

    Performs any calculations coming from combo variables. Coordinate limit indices formatted as (lat_min, lat_max,
    lon_min, lon_max). Data is always in a three-dimensional Numpy array, as required by the data collection object.
    This function is separate from _get_data because it is recursively called for combo variable calculations.


    Args:
        dataset: The dataset.
        variable: The variable.
        coo_index: The coordinate limit indices.
        time: The time.

    Returns:
        The data

    """
    _function_call()

    if variable.is_combo:
        equation_type, x_identifier, y_identifier = variable.equation.split('_')

        x_variable = _get_variable_identifier(dataset, int(x_identifier))
        y_variable = _get_variable_identifier(dataset, int(y_identifier))

        x = _cut_interpret_data(dataset, x_variable, coo_index, time)[0]
        y = _cut_interpret_data(dataset, y_variable, coo_index, time)[0]

        return _compute_variable_equation(equation_type, x, y)

    else:

        year, month, day, hour = time

        if (year is None) and (month is not None):
            years = [x for x in get_years(dataset) if month in get_months(dataset, x, variable)]
            month_datas = list()

            for year in years:
                variable_data = _load(dataset, year, month, variable)[variable.key]
                month_datas.append(variable_data[:, coo_index[0]:coo_index[1], coo_index[2]:coo_index[3]])

            return [np.concatenate(month_datas), ]

        elif year is None:
            years = get_years(dataset, variable)
            year_datas = list()

            for year in years:
                for month in get_months(dataset, year, variable):
                    variable_data = _load(dataset, year, month, variable)[variable.key]
                    year_datas.append(variable_data[:, coo_index[0]:coo_index[1], coo_index[2]:coo_index[3]])
            return [np.concatenate(year_datas), ]

        elif month is None:
            months = get_months(dataset, year, variable)

            month_datas = list()

            for month in months:
                variable_data = _load(dataset, year, month, variable)[variable.key]
                month_datas.append(variable_data[:, coo_index[0]:coo_index[1], coo_index[2]:coo_index[3]])
            return [np.concatenate(month_datas), ]

        elif day is None:
            variable_data = _load(dataset, year, month, variable)[variable.key]
            return [variable_data[:, coo_index[0]:coo_index[1], coo_index[2]:coo_index[3]], ]

        elif hour is None:
            hours = get_hours(dataset, year, month, day)

            variable_data = _load(dataset, year, month, variable)[variable.key]
            start_time_index = _get_time_index(dataset, year, month, day, hours[0])
            end_time_index = _get_time_index(dataset, year, month, day, hours[-1])

            return [variable_data[start_time_index:end_time_index + 1, coo_index[0]:coo_index[1], coo_index[
                                                                                                      2]:coo_index[
                                                                                                      3]], ]

        else:
            variable_data = _load(dataset, year, month, variable)[variable.key]
            # noinspection PyTypeChecker
            time_index = _get_time_index(dataset, year, month, day, hour)
            data = variable_data[time_index, coo_index[0]:coo_index[1], coo_index[2]:coo_index[3]]

            # Expand to 3D array
            data = np.expand_dims(data, axis=0)
            return [data, ]


# ======================== PLOTTING ====================================================================================
def plot_graphables(graphables: Graphable | tuple[Graphable, ...], styles: str | tuple[str, ...],
                    projection: ccrs.Projection, limits: Optional[LIMIT_TYPE], ticks: Optional[tuple[float, ...]],
                    skip: Optional[int], size: POINT_TYPE, titles: Optional[tuple[str, ...] | str], out_mode: str,
                    directory: Optional[str], save_titles: Optional[str | tuple[str, ...]], font_size: int) -> None:
    """
    Plot a series of Graphable objects.

    Should each Graphable have the same time length, one plot is generated for each time stamp.

    Supported styles:
    For points: [colour]_[marker]_[marker size] with each from matplotlib. For example: black_X_12
    For paths: [colour]_[alpha] with each from matplotlib. For example: red_0.5
    For scalars: heat_[cmap] with cmap from matplotlib. For example: heat_jet
                 contour_[integer number of levels] For example: contour_9
    For vectors: quiver

    The number of styles should equal the number of graphables.
    Should no limits be given, the maximal limits from all graphables is used.
    Should no tick markers be given, they are automatically found.
    Should titles be given, there should be as the time length of the graphables. If they are not given,
    automatic titles are generates using the time titles from the graphables.
    Should no directory be given and figures saved, the current directory is used.
    Should no save titles be given, they are automatically generated. Otherwise, there should be the same number as
    the time length of the graphables.


    Args:
        graphables: A single Graphable or tuple of Graphables to plot.
        styles: The styles.
        projection: The projection.
        limits: The limits of the plot. Possibly None.
        ticks: Tick markers for use with heat map. Possibly None.
        skip: The number of points to skip in each direction for quiver plots. Possibly None.
        size: The size of the figure.
        titles: The titles. Possibly None.
        out_mode: The output mode. See _output_figure.
        directory: The directory for plot figures. Possibly None.
        save_titles: The save titles.
        font_size: The font size.

    Returns:
        None. Note, each plot is processed through _output_figure.

    """
    # Convert possible single objects to tuples
    graphables = _to_tuple(graphables)
    styles = _to_tuple(styles)
    if titles is not None:
        titles = _to_tuple(titles)
    if save_titles is not None:
        save_titles = _to_tuple(save_titles)

    min_time = min([g.get_time_length() for g in graphables])
    # If only infinite times, just plot one figure
    if min_time == inf:
        min_time = 1
    min_time = int(min_time)

    # Compute automatic values (those which are required but inputted as None). Automatic titles done in time loop
    if limits is None:
        limits = _maximal_limits(tuple([g.get_limits() for g in graphables]))
    if save_titles is None:
        save_titles = tuple([str(i) + '.png' for i in range(min_time)])

    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'Times',
        'font.size': font_size
    })

    for time_index in range(min_time):
        print(f'Plotting figure {time_index} of {min_time - 1}')

        # Close all plots at each pass
        plt.close('all')

        fig = plt.figure(figsize=size)
        ax = plt.axes(projection=projection)
        ax.coastlines(resolution='50m')

        ax.set_extent([limits[2], limits[3], limits[0], limits[1]], crs=ccrs.PlateCarree())

        gl = ax.gridlines(draw_labels=True)
        gl.right_labels = False
        gl.bottom_labels = False

        if titles is None:
            title = iterable_to_words(tuple([f'{g.get_time_title(time_index)} ({style})' for g, style in zip(
                graphables, styles)]))
            title += f'\nMax limits in deg.: {limits[0]}$\leq$lat$\leq${limits[1]}, {limits[2]}$\leq$lon$\leq$\
             {limits[3]}'
            if skip is not None:
                title += f'\n Vector skip: {skip}'
        else:
            title = titles[time_index]

        plt.title(title)

        for g, style in zip(graphables, styles):
            if isinstance(g, DataCollection):
                if style.startswith('heat_'):
                    cmap = style.split('_')[1]
                    if ticks is not None:
                        p = ax.pcolormesh(g.longitude, g.latitude, g.get_component(time_index, 0),
                                          transform=ccrs.PlateCarree(),
                                          cmap=cmap,
                                          shading='nearest',
                                          vmin=ticks[0], vmax=ticks[-1])
                        cbar = fig.colorbar(p, orientation='vertical', ticks=ticks)
                        cbar.ax.set_yticklabels([str(tick) for tick in ticks])
                    else:
                        p = ax.pcolormesh(g.longitude, g.latitude, g.get_component(time_index, 0),
                                          transform=ccrs.PlateCarree(), cmap=cmap,
                                          shading='nearest')
                        fig.colorbar(p, orientation='vertical')

                elif style == 'quiver':
                    ax.quiver(g.longitude[::skip], g.latitude[::skip], g.get_component(time_index, 0)[::skip, ::skip],
                              g.get_component(time_index, 1)[::skip, ::skip], transform=ccrs.PlateCarree())

                elif style.startswith('contour_'):

                    # specified number of levels or automatic
                    levels = style.split('_')[1]
                    if levels == 'None' or levels == 'auto':
                        levels = None
                    else:
                        levels = int(levels)

                    cs = ax.contour(g.longitude, g.latitude, g.get_component(time_index, 0),
                                    transform=ccrs.PlateCarree(), colors='k', levels=levels)
                    ax.clabel(cs, inline=True, fontsize=9)

            elif isinstance(g, PathCollection):
                colour, alpha = style.split('_')
                patch = patches.PathPatch(g.get_component(time_index), facecolor=colour, alpha=float(alpha), lw=0,
                                          transform=ccrs.PlateCarree())
                ax.add_patch(patch)

            elif isinstance(g, PointCollection):
                colour, marker, marker_size = style.split('_')
                latitude, longitude = g.get_component(time_index)
                ax.scatter(longitude, latitude, s=int(marker_size), c=colour, marker=marker,
                           transform=ccrs.PlateCarree())

        # Output figure
        plt.tight_layout()
        save_title = directory + '/' + save_titles[time_index] if directory is not None else save_titles[time_index]
        _output_figure(out_mode, save_title)

    return None


def plot_point_data_over_time(data_collection: DataCollection, title: Optional[str], tick_mode: Optional[str], size:
POINT_TYPE, out_mode: str, save_title: Optional[str], y_line: Optional[float], font_size:
int = 12) -> matFigure | None:
    """
    Plots the data of a single point over time.

    The inputted DataCollection must be of a single point, have a single component (be a scalar), and be over a time 
    series.
    Should no title be given, one is automatically generated.
    Should no tick mode be given, matplolib's AutoDateLocator and AutoDataFormatter is used. Otherwise, 
    supported tick modes are:

    month_day -> Interval of 5 days with month and day marked
    month -> Each month
    auto -> Equivalent to passing no tick mode

    Should no save title be given, one is automatically generated.
    The optional y_line is a grey horizontal dashed line drawn at the given value.

    Args:
        data_collection: The data collection.
        title: The figure title. Possibly None.
        tick_mode: The tick mode. Possibly None.
        size: The figure size.
        out_mode: The output mode. See _output_figure.
        save_title: The save title. Possibly None.
        y_line: The y value for the horizontal line. Possibly None.
        font_size: The font size.

    Returns:
        None. Note, the figure is processed through _output_figure.

    Raises:
        ValueError: The data collection is not of a single point
        ValueError: The data collection must have a single component (be of a scalar)
        ValueError: The data collection is not over a time series

    """
    # Check that the data collection is for a single point
    if not data_collection.spread == (1, 1):
        raise ValueError(f'The data collection is not of a single point')

    # Check that there is a single coordinate
    if not data_collection.dimension == 1:
        raise ValueError(f'The data collection must have a single component (be of a scalar)')

    # Check that the data collection is a time series
    year, month, day, hour = data_collection.time_stamps[0]
    if any((year is None, month is None, day is None, hour is None)):
        raise ValueError(f'The data collection is not over a time series')

    plt.close('all')

    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'Times',
        'font.size': font_size
    })

    # fig = plt.figure(figsize=size)
    ax = plt.axes()

    if title is None:
        title = f'{str(data_collection)} ' \
                f'at {_coordinates_to_formatted(data_collection.latitude[0], data_collection.longitude[0])}'
    if save_title is None and out_mode == 'save':
        save_title = _time_stamps_to_filenames((data_collection.time,))[0]

    time_axis = [datetime.datetime(*stamp) for stamp in data_collection.time_stamps]

    if tick_mode == 'month_day':
        locator = mdates.DayLocator(interval=5)
        formatter = mdates.DateFormatter('%b %d')
    elif tick_mode == 'month':
        locator = mdates.MonthLocator()
        formatter = mdates.DateFormatter('%b')
    elif tick_mode == 'auto' or tick_mode is None:
        locator = mdates.AutoDateLocator()
        formatter = mdates.AutoDateFormatter(locator, defaultfmt='%Y')
    else:
        raise ValueError('No valid tick format given')

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    ax.plot(time_axis, data_collection.data[0][:, 0, 0])
    if y_line is not None:
        ax.axhline(y_line, color='grey', lw=1.5, ls='--')

    ax.set(title=title, ylabel=data_collection.variable.name)

    _output_figure(out_mode, save_title)

    return None


def _maximal_limits(limits: tuple[LIMIT_TYPE, ...]) -> LIMIT_TYPE:
    """
    Find the limits which encompass all coordinates of a set of limits.

    Examples:
        For input limits (-10, 15, -20, 25) and (-30, 7, -90, 40), the result is (-30, 15, -90, 40)

    Args:
        limits: The set of limits.

    Returns:
        The encompassing limits.

    """
    _function_call()

    limit_matrix = np.reshape(limits, (len(limits), 4))
    lat_min = np.amin(limit_matrix[:, 0])
    lat_max = np.amax(limit_matrix[:, 1])
    lon_min = np.amin(limit_matrix[:, 2])
    lon_max = np.amax(limit_matrix[:, 3])

    return lat_min, lat_max, lon_min, lon_max

def images_to_video(image_directory: str, fps: int, video_path: str) -> None:
    image_paths = [os.path.join(image_directory, path) for path in os_sorted(os.listdir(image_directory)) if
                   path.endswith('.png')]
    clip = ImageSequenceClip(image_paths, fps=fps)
    clip.write_videofile(video_path, codec='mpeg4')


# ======================== STATISTICS ==================================================================================
def average(data_collection: DataCollection) -> DataCollection:
    """
    Averages a data collection's data over the time axis.

    Args:
        data_collection: The data collection.

    Returns:
        A new data_collection with averaged data. The dataset, variable, time, limits, latitude and longitude are all
        the same as from the original data collection. The title is updated, for a data collection dc, to

        {dc.dataset}: {dc.variable} averaged {time suffix for dc's time}

    """
    averaged_data = list()
    for component in data_collection.data:
        averaged_data.append(np.expand_dims(np.mean(component, axis=0), axis=0))

    title_prefix = f'{data_collection.dataset}: {data_collection.variable} averaged '

    # noinspection PyTypeChecker
    return DataCollection(data_collection.dataset, data_collection.variable, data_collection.time,
                          data_collection.limits, averaged_data, data_collection.latitude, data_collection.longitude,
                          title_prefix, '', (data_collection.time,))


def bound_frequency(data_collection: DataCollection, lower_bound: float, upper_bound: float) -> DataCollection:
    """
    Finds the relative frequency of a data collection's data being greater than or equal to a lower bound and less
    than or equal to an upper bound. This is done point-wise over the time axis. That is, for each coordinate point
    over time.

    Args:
        data_collection: The data collection.
        lower_bound: The lower bound.
        upper_bound: The upper bound.

    Returns:
        A new data_collection with resulting data. The dataset, variable, time, limits, latitude and longitude are all
        the same as from the original data collection. The title is updated, for a data collection dc, to

        {dc.dataset}: Relative frequency of {dc.variable} where {lower_bound} <= x <= {upper_bound} {time suffix for
        dc's time}

    """
    bound_data = list()
    for component in data_collection.data:
        flat_bound = np.count_nonzero((lower_bound <= component) & (component <= upper_bound), axis=0) / (np.shape(
            component)[0])
        bound_data.append(np.expand_dims(flat_bound, axis=0))

    title_prefix = f'{data_collection.dataset}: Relative frequency of {data_collection.variable} where {lower_bound} ' \
                   f'$<= x <=$ {upper_bound} '

    # noinspection PyTypeChecker
    return DataCollection(data_collection.dataset, data_collection.variable, data_collection.time,
                          data_collection.limits, bound_data, data_collection.latitude, data_collection.longitude,
                          title_prefix, '', (data_collection.time,))


def percentile(data_collection: DataCollection, perc: float) -> DataCollection:
    """
    Finds the given percentile of a data collection's data. This is done point-wise over the time axis. That is, for
    each coordinate point over time.

    Args:
        data_collection: The data collection.
        perc: The percentile.

    Returns:
        A new data_collection with resulting data. The dataset, variable, time, limits, latitude and longitude are all
        the same as from the original data collection. The title is updated, for a data collection dc, to

        {dc.dataset}: {percentile}-th percentile of {dc.variable} {time suffix for dc's time}

    """
    percentile_data = list()
    for i in range(data_collection.get_vector_dimension()):
        flat_percentile = np.percentile(data_collection.get_component(None, i), perc, axis=0)
        percentile_data.append(np.expand_dims(flat_percentile, axis=0))

    title_prefix = f'{data_collection.dataset}: {perc}-th percentile of {data_collection.variable} '

    return DataCollection(data_collection.dataset, data_collection.variable, data_collection.time,
                          data_collection.get_limits(), percentile_data, data_collection.latitude,
                          data_collection.longitude,
                          title_prefix, '', (data_collection.time,))


def percentile_date(spec_data_collection: DataCollection, ref_data_collection: DataCollection) -> \
        DataCollection:
    """
    Returns the percentile of a (specific) data collection's data relative to another (reference) data_collection's
    data. This is done point-wise over the time axis of the reference data collection. That is, for each coordinate
    of the specific collection over time in the reference collection.

    The two data collection must have the same latitude and longitude values. They must also have the same vector
    dimension (number of data components).

    Each 'time-slice' from the specific collection undergoes finding its percentile relative to the reference
    collection. This results in a two-dimensional array. These arrays are staked in order to create a
    three-dimensional array.

    Args:
        spec_data_collection: The specific data collection.
        ref_data_collection: The reference data collection.

    Returns:
        A new data_collection with resulting data. The dataset, variable, time, limits, latitude and longitude are all
        the same as from the original data collection. The title is updated. For a specific collection dc_spec and
        reference collection dc_ref, the title is

        Percentile of {dc_spec.variable} {time suffix for dc_spec's time} ({dc_spec.dataset}) wrt {dc_ref.variable} {
        time suffix for dc_ref's time} ({dc_ref.dataset})

    Raises:
        ValueError: The specific and reference data collections must have the same latitude and longitude values.
        ValueError: The specific and reference data collections must have the same vector dimensions.
    """

    if not np.array_equal(spec_data_collection.latitude, ref_data_collection.latitude) or not np.array_equal(
            spec_data_collection.longitude, ref_data_collection.longitude):
        raise ValueError('The specific and reference data collections must have the same latitude and longitude '
                         'values.')
    if spec_data_collection.dimension != ref_data_collection.dimension:
        raise ValueError('The specific and reference data collections must have the same vector dimensions.')

    percentile_data = list()

    for spec_component, ref_component in zip(spec_data_collection.data, ref_data_collection.data):
        z, x, y = np.shape(ref_component)
        ref_component = np.reshape(ref_component, (z, x * y))
        time_percentiles = list()
        for spec_component_time, time_stamp in zip(spec_component, spec_data_collection.time_stamps):
            print(f'Percentile date for {time_stamp}')

            spec_component_time = np.reshape(spec_component_time, x * y)

            percentiles = [percentileofscore(ref_component[:, i], spec_component_time[i]) for i in range(x * y)]
            percentiles = np.reshape(np.array(percentiles), (x, y))
            time_percentiles.append(np.expand_dims(percentiles, axis=0))

        percentile_data.append(np.concatenate(time_percentiles))

    title_prefix = f'Percentile of {spec_data_collection.variable} '
    title_suffix = f' ({spec_data_collection.dataset.name}) wrt {ref_data_collection.variable.name} ' \
                   f'{_time_suffix(ref_data_collection.time)} ({ref_data_collection.dataset.name})'

    # noinspection PyTypeChecker
    return DataCollection(spec_data_collection.dataset,
                          spec_data_collection.variable, spec_data_collection.time, spec_data_collection.limits,
                          percentile_data, spec_data_collection.latitude, spec_data_collection.longitude,
                          title_prefix, title_suffix, spec_data_collection.time_stamps)


def count_non_nan(dc: DataCollection) -> DataCollection:
    non_nan_time_count = [np.expand_dims(np.count_nonzero(~np.isnan(c), axis=0), axis=0) for c in dc.data]

    title_prefix = 'Non-nan count of ' + dc.title_prefix

    return DataCollection(dc.dataset, dc.variable, dc.time, dc.limits, non_nan_time_count, dc.latitude, dc.longitude,
                          title_prefix, dc.title_suffix, (dc.time,))


def fraction_below_ordered(spec_data_collection: DataCollection, ref_data_collection: DataCollection, non_nan_count:
DataCollection) -> DataCollection:
    """
    TODO: Documentation
    """
    if not ref_data_collection.title_prefix.startswith('Ordered '):
        warnings.warn('Reference data may not be ordered')
    if not np.array_equal(spec_data_collection.latitude, ref_data_collection.latitude) or not np.array_equal(
            spec_data_collection.longitude, ref_data_collection.longitude):
        raise ValueError('The specific and reference data collections must have the same latitude and longitude '
                         'values.')
    if spec_data_collection.dimension != ref_data_collection.dimension:
        raise ValueError('The specific and reference data collections must have the same vector dimensions.')

    len_lat, len_lon = spec_data_collection.spread
    results = np.empty((spec_data_collection.get_vector_dimension(), spec_data_collection.get_time_length(),
                        len_lat, len_lon))

    # For each component at each time step of the specific data
    for component, (spec_component, ref_component) in enumerate(zip(spec_data_collection.data,
                                                                    ref_data_collection.data)):
        for time_idx, (spec_component_time, time_stamp) in enumerate(zip(spec_component,
                                                                         spec_data_collection.time_stamps)):
            print(f'Fraction below for component {component} at {time_stamp}')

            # For each point
            for lat_idx in range(len_lat):
                for lon_idx in range(len_lon):
                    # Find the fraction of points from reference data below value
                    score = spec_component_time[lat_idx, lon_idx]
                    total = np.searchsorted(ref_component[:, lat_idx, lon_idx], score)
                    # noinspection PyTypeChecker
                    results[component, time_idx, lat_idx, lon_idx] = total / non_nan_count.data[component][0,
                    lat_idx,
                    lon_idx]

    title_prefix = f'Fraction of {ref_data_collection} below {spec_data_collection.variable} '
    title_suffix = f' ({spec_data_collection.dataset})'

    return DataCollection(spec_data_collection.dataset,
                          spec_data_collection.variable, spec_data_collection.time, spec_data_collection.limits,
                          [r for r in results], spec_data_collection.latitude, spec_data_collection.longitude,
                          title_prefix, title_suffix, spec_data_collection.time_stamps)


def fraction_scalar_below_all_sorted_memory(spec_data_collection: DataCollection, ref_non_nan_count):
    """
    TODO: Documentation
    """
    # Verify assumptions of function
    if spec_data_collection.get_vector_dimension() != 1:
        return ValueError('Specific data must be of a scalar.')

    # Results array
    len_lat, len_lon = spec_data_collection.spread
    tally_below = np.zeros((spec_data_collection.get_time_length(), len_lat, len_lon))

    # Corresponding sorted variable
    dataset = spec_data_collection.dataset
    sorted_variable = _get_variable_identifier(dataset, -1 * spec_data_collection.variable.identifier)

    # Update results after loading in one year of reference data
    for year in get_years(dataset, sorted_variable):
        for month in get_months(dataset, year, sorted_variable):
            sorted_month = _get_data(dataset, sorted_variable, spec_data_collection.limits, (year, month, None, None))
            print(f'Tallying ref data from {year=}, {month=} for spec dc timed {spec_data_collection.time}')

            # For each coordinate
            for lat_idx in range(len_lat):
                for lon_idx in range(len_lon):
                    sorted_coordinate_data = sorted_month.get_coordinate_index(None, 0, lat_idx, lon_idx)
                    coordinate_data = spec_data_collection.get_coordinate_index(None, 0, lat_idx, lon_idx)
                    tally_below[:, lat_idx, lon_idx] += np.searchsorted(sorted_coordinate_data, coordinate_data)

    # Divide tally by non-nan-count for results array
    results = np.divide(tally_below, ref_non_nan_count)

    title_prefix = f'Fraction of all avaiable associated sorted data below' \
                   f' {spec_data_collection.variable} '
    title_suffix = f' ({spec_data_collection.dataset})'

    return DataCollection(spec_data_collection.dataset,
                          spec_data_collection.variable, spec_data_collection.time, spec_data_collection.limits,
                          [results,], spec_data_collection.latitude, spec_data_collection.longitude,
                          title_prefix, title_suffix, spec_data_collection.time_stamps)


def max_dc(data_collection: DataCollection, per_time_slice: bool) -> tuple[PointCollection, ArrayLike] | \
                                                                     tuple[PointCollection, TIME_TYPE,
                                                                     ArrayLike]:
    """
    Find the maximum data value of a data collection.

    The given data collection must have a single component.

    Args:
        data_collection: The data collection.
        per_time_slice: Whether to compute the maximum at each time (True) or over all time (False).

    Returns:
        The maximum.

    """
    if not data_collection.dimension == 1:
        raise ValueError('Max point only valid for single component (scalar) data')
    if per_time_slice:
        latitudes = list()
        longitudes = list()
        values = list()

        for time_idx in range(data_collection.get_time_length()):
            time_data = data_collection.get_component(time_idx, 0)
            lat_idx, lon_idx = np.unravel_index(np.argmax(time_data), data_collection.spread)
            latitudes.append(data_collection.latitude[lat_idx])
            longitudes.append(data_collection.longitude[lon_idx])

            max_value = np.max(time_data)
            values.append(max_value)

        return PointCollection(data_collection.dataset, data_collection.variable, data_collection.time,
                               tuple(latitudes), tuple(longitudes), f'Time-dependent max of {data_collection} ', '',
                               data_collection.time_stamps), values

    time_idx, lat_idx, lon_idx = np.unravel_index(np.argmax(data_collection.get_component(None, 0)),
                                                  data_collection.shape)
    time = data_collection.time_stamps[time_idx]
    lat, lon = data_collection.latitude[lat_idx], data_collection.longitude[lon_idx]
    max_value = np.max(data_collection.get_component(None, 0))
    point = PointCollection(data_collection.dataset, data_collection.variable, data_collection.time, (lat,),
                            (lon,), f'Time-independent max of {data_collection} ', '', (data_collection.time,))

    return point, time, max_value


# ======================== AREA COMPUTATION ============================================================================
def contour_size_data_collection(data_collection: DataCollection, contour: float) -> tuple[str, tuple[ArrayLike]]:
    """
    Calculates the area of a certain contour level of a data collection's data.

    Accounts for spherical geometry. This function's method is not reliable. Supports data collections with data over
    periods of time. Data must be of a scalar (only have once component). Otherwise, a contour plot is impossible to
    create.

    The returned text report is saved to text_report.txt. The returned tuple of areas is saved as comma-separated values
    to area_list.txt.

    Args:
        data_collection: The data collection.
        contour: The contour level.

    Returns:
        A text report detailing contour areas for each time stamp from the data collection. The minimum, maximum,
        mean and standard deviation are appended. A tuple of all areas is also returned.

    Raises:
        ValueError: The data collection must be of a scalar.

    """
    if data_collection.dimension != 1:
        raise ValueError('The data collection must be of a scalar.')

    plt.close('all')
    # fig = plt.figure()
    ax = plt.axes()

    area_grid = _compute_area_grid(data_collection.latitude, data_collection.longitude)
    text_report = str()
    areas = list()

    for time_data, time in zip(data_collection.data[0], data_collection.time_stamps):
        print(f'Contour sizing for {time}')
        cs = ax.contour(data_collection.longitude, data_collection.latitude, time_data, levels=(contour,))

        contour_object = cs.collections[0]
        paths = contour_object.get_paths()

        text_report += f'{time} ==============================================================\n'

        # determine the area of each path
        for path in paths:
            is_contained = list()
            for lat in data_collection.latitude:
                for lon in data_collection.longitude:
                    is_contained.append(int(path.contains_point((lon, lat))))

            is_contained = np.reshape(is_contained, data_collection.spread)
            area = np.sum(np.multiply(is_contained, area_grid))
            areas.append(area)
            text_report += (str(area) + '\n')

    text_report += f'\n\nMin: {np.min(areas)}\n'
    text_report += f'Max: {np.max(areas)}\n'
    text_report += f'Mean: {np.mean(areas)}\n'
    text_report += f'Std.: {np.std(areas, ddof=1)}\n'

    with open('text_report.txt', 'w') as file:
        file.write(text_report)

    with open('area_list.txt', 'w') as file:
        areas_str = [str(area) for area in areas]
        file.write(','.join(areas_str))

    # noinspection PyTypeChecker
    return text_report, tuple(areas)


def find_contour_path_data_collection(data_collection: DataCollection, contour: float, path_index: int) -> \
        PathCollection:
    """
    Finds a specific contour path of a data collection.

    The data collection must have a single component and be of a single time.

    Args:
        data_collection: The data collection.
        contour: The contour value.
        path_index: The contour path index.

    Returns:
        The path.

    Raises:
        ValueError: The data collection must have a single component (be a scalar).
        ValueError: The data collection must have a time length of 1.
        IndexError: Contour path index too large.

    """
    if data_collection.dimension != 1:
        raise ValueError('The data collection must have a single component (be a scalar).')
    if data_collection.get_time_length() != 1:
        raise ValueError('The data collection must have a time length of 1.')

    plt.close('all')
    # fig = plt.figure()
    ax = plt.axes()

    cs = ax.contour(data_collection.longitude, data_collection.latitude, data_collection.get_component(0, 0),
                    levels=(contour,))

    contour_object = cs.collections[0]

    if path_index >= len(contour_object.get_paths()):
        raise IndexError('Contour path index too large.')
    path_data = contour_object.get_paths()[path_index]

    title = f'{path_index + 1}-th path of the {contour}-th contour of {data_collection} '

    return PathCollection(data_collection.dataset, data_collection.variable, data_collection.time, (path_data,),
                          title, '', data_collection.time_stamps)


def _compute_area_grid(latitude, longitude):
    """
    Computes a latitude by longitude matrix of 'infinitesimal' spherical surface area elements.

    Assumes the Earth is a perfect sphere, and uses the RADIUS_OF_EARTH constant. Uses difference between successive
    latitude and longitude values.

    Args:
        latitude: Latitude values.
        longitude: Longitude values.

    Returns:
        The area matrix.

    """
    _function_call()

    lat_diff = np.radians(np.abs(np.diff(latitude, append=latitude[-2])))
    lon_diff = np.radians(np.abs(np.diff(longitude, append=longitude[-2])))

    sin_latitude = np.sin(np.radians(90 - latitude))
    sin_latitude = np.reshape(sin_latitude, (len(sin_latitude), 1))

    lat_diff_grid, lon_diff_grid = np.meshgrid(lat_diff, lon_diff, indexing='ij')

    area_grid = np.multiply(np.multiply(lat_diff_grid, lon_diff_grid), sin_latitude) * (RADIUS_OF_EARTH ** 2)

    return area_grid


def refine_area_to_illustration(path_collection: PathCollection, point_condition: Callable[[float, float], bool]) -> \
        DataCollection:
    """
    Illustrates a refined path.

    The point_condition takes in a point (latitude, longitude pair) and returns a boolean representing whether that
    point should be shown.

    The returned DataCollection has 1's where the point meets the point condition and 0's otherwise.

    Args:
        path_collection: The path to illustrate.
        point_condition: The point condition.

    Returns:
        A DataCollection with the illustration data as binary data values.

    """
    coo_index, latitude, longitude = _get_coordinate_information(path_collection.dataset, path_collection.get_limits())
    show_points = np.zeros((path_collection.get_time_length(), len(latitude), len(longitude)))

    for time_idx in range(path_collection.get_time_length()):
        for lat_idx, lat in enumerate(latitude):
            for lon_idx, lon in enumerate(longitude):
                # noinspection PyTypeChecker
                if path_collection.contains_point(time_idx, (lat, lon)) and point_condition(lat, lon):
                    show_points[time_idx, lat_idx, lon_idx] = 1

    return DataCollection(path_collection.dataset, path_collection.variable, path_collection.time,
                          path_collection.get_limits(),
                          [show_points, ], latitude, longitude, path_collection.title_prefix,
                          path_collection.title_suffix + ' (refined)', path_collection.time_stamps)


# ======================== ATMOSPHERIC PHYSICS & RESEARCH CONVENTIONS ==================================================
def get_winter_year(time: TIME_TYPE) -> int:
    """
    Get the winter year of a winter time. Uses the convention that November and December refer to the next year.

    Only months 1, 2, 3, 4, 11, and 12 are winter months.

    Examples:
        (2022, 11, 12, 1) -> 2022
        (2013, 4, None, None) -> 2013

    Args:
        time: The winter time.

    Returns:
        The winter year.

    Raises:
        ValueError: The year and months must not be None.
        ValueError: The given month is not recognized as a winter month.

    """
    year, month, day, hour = time
    if month is None or year is None:
        raise ValueError('The year and months must not be None.')
    if month in (11, 12):
        return year + 1
    elif month in (1, 2, 3, 4):
        return year
    else:
        raise ValueError('The given month is not recognized as a winter month.')


def get_defaults() -> tuple[LIMIT_TYPE, ccrs.Projection]:
    """
    Default values for limits and projection.

    The default limits are (40, 68, -60, -10) and the default projection is Lambert, centred on those limits.
    Returns:
        Tuple containing default limits and projection.

    Raises:
        NameError: The default projection, Lambert, is not available according to info.json.

    """
    limits = (40, 68, -60, -10)
    if 'Lambert' in info.projections:
        return limits, get_projection_name('Lambert', limits)
    else:
        raise NameError('The default projection, Lambert, is not available according to info.json.')

# ======================== MAIN ========================================================================================
def main():
    print('Welcome to edcl!')


if __name__ == '__main__':
    main()
