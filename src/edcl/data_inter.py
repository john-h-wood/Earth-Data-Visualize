import datetime
import json
import pickle
from glob import glob
from math import inf
from abc import ABC, abstractmethod
from os.path import basename, isfile
from importlib.resources import files
from typing import Union, Optional, Callable

import cartopy.crs as ccrs
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.figure import Figure as matFigure
from matplotlib.path import Path
from numpy.typing import ArrayLike
from scipy.stats import percentileofscore

"""
The data interpretation (data_module) manages all interactions with data.

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
    
TODO: Ideas:
- Auto range info (e.g. what years for April?)

    

"""

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

    # @staticmethod
    # def __validate_attributes(is_combo: bool, equation: str, file_identifier: str) -> None:
    #     """ Checks the validity of parameters, namely for inconsistencies arising from combo variables.
    #
    #     A combo variable must have an associated equation and no file identifiers. Only combo variables should have
    #     equations. A ValueError is raised if the given parameters do not meet these conditions.
    #
    #     Args:
    #         is_combo: Whether the variable is a combo variable.
    #         equation: The variable's equation equation. Possibly None.
    #         file_identifier: The variable's file identifier. Possibly None
    #
    #     Returns:
    #         None.
    #
    #     Raises:
    #         ValueError: If the parameters are not valid.
    #
    #     """
    #     if is_combo == (equation is None) or (is_combo and file_identifier is not None):
    #         raise ValueError('Combo variables, and only those variables, must have equations. Combo variables cannot '
    #                          'have file identifiers')

    def __init__(self, name: str, kind: str, is_combo: bool, identifier: int, key: Union[str, None], equation: Union[
        str, None], file_identifier: Union[str, None]) -> None:
        # Variable.__validate_attributes(is_combo, equation, file_identifier)
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
    def __init__(self, dataset: Dataset, variable: Variable, time: TIME_TYPE, latitudes: tuple[float, ...],
                 longitudes: tuple[float, ...], title_prefix: str, title_suffix: str, time_stamps: Optional[tuple[
                 TIME_TYPE, ...]]):
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
        return len(self.latitudes)

    def get_time_title(self, time_index) -> str:
        return f'{self.title_prefix}{_time_suffix(self.time_stamps[time_index])}{self.title_suffix}'

    def get_limits(self) -> LIMIT_TYPE:
        return min(self.latitudes), max(self.latitudes), min(self.longitudes), max(self.longitudes)

    def get_component(self, time_index):
        return self.latitudes[time_index], self.longitudes[time_index]


class PathCollection(Graphable):
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
        return len(self.paths)

    def get_time_title(self, time_index) -> str:
        return f'{self.title_prefix}{_time_suffix(self.time_stamps[time_index])}{self.title_suffix}'

    def get_limits(self) -> LIMIT_TYPE:
        limits = list()
        for path in self.paths:
            ex = path.get_extents()
            limits.append((ex.ymin, ex.ymax, ex.xmin, ex.xmax))

        return _maximal_limits(tuple(limits))

    def contains_point(self, time_index: int, point: POINT_TYPE) -> bool:
        lat, lon = point
        return self.paths[time_index].contains_point((lon, lat))

    def get_area(self, time_index: int, point_condition: Optional[Callable[[float, float], bool]]) -> float:

        coo_index, latitude, longitude = _get_coordinate_information(self.dataset, self.get_limits())
        area_grid = _compute_area_grid(latitude, longitude)

        is_containted = list()

        for lat in latitude:
            for lon in longitude:
                if self.contains_point(time_index, (lat, lon)):
                    if point_condition is not None:
                        is_containted.append(int(point_condition(lat, lon)))
                    else:
                        is_containted.append(1)
                else:
                    is_containted.append(0)

        is_contained = np.reshape(is_containted, (len(latitude), len(longitude)))
        return float(np.sum(np.multiply(is_contained, area_grid)))

    def get_component(self, time_index: int) -> Path:
        return self.paths[time_index]


class DataCollection(Graphable):

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
        return self.shape[0]

    def get_time_title(self, time_index) -> str:
        return f'{self.title_prefix}{_time_suffix(self.time_stamps[time_index])}{self.title_suffix}'

    def get_limits(self) -> LIMIT_TYPE:
        return self.limits

    def get_component(self, time_index, component_index) -> ArrayLike:
        if component_index >= len(self.data):
            raise ValueError('The given component index is too large.')

        if time_index is None:
            return self.data[component_index]

        if time_index >= self.get_time_length():
            raise ValueError('The given time index is too large.')
        return self.data[component_index][time_index]

    def get_coordinate_value(self, time_index, component_index, latitude, longitude) -> ArrayLike:
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
        return self.data[component_index][time_index, lat_idx, lon_idx]

    def get_vector_dimension(self) -> int:
        return len(self.data)


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
def get_defaults() -> tuple[LIMIT_TYPE, ccrs.Projection]:
    limits = (40, 68, -60, -10)
    if 'Lambert' in info.projections:
        return limits, get_projection_name('Lambert', limits)
    else:
        raise NameError('The default projection, Lambert, is not available according to info.json.')


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


def load_pickle(file_name: str):
    with open(file_name, 'rb') as temp_file:
        return pickle.load(temp_file)


def save_pickle(obj: object, file_name: str) -> None:
    with open(file_name, 'wb') as temp_file:
        pickle.dump(obj, temp_file, pickle.HIGHEST_PROTOCOL)


def _to_tuple(element: object) -> tuple:
    if not isinstance(element, tuple):
        return element,
    else:
        return element


def _output_figure(out_mode: str, save_title: Optional[str]) -> matFigure or None:
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


def get_data_collection_names(dataset_name: str, variable_name: str, limits: LIMIT_TYPE, time: TIME_TYPE) -> \
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
    so this function should return * the * variable with the given nidentifier.

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

    If the variable is None, then months for which any data is available are inncluded. Otherwise, only months which
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
                print(year, month, dataset)
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


def _get_coordinate_information(dataset: Dataset, limits: LIMIT_TYPE) -> tuple[LIMIT_TYPE, ArrayLike, ArrayLike]:
    # TODO efficiency could probably be improved, with the assumption that coordinates only increase
    """
    Get the coordinate indices corresponding to given coordinate limits.

    Data is stored in matrices which have their first dimension as time. The second and third dimensions refer,
    respectively, to latitude and longitude. This function returns the index limits for latitude and longitude,
    given coordinate limits. That is, the upper and lower indices for which both latitude and longitude are within or
    equal to specified bounds.

    Limits are formatted as (lat_min, lat_max, lon_min, lon_max). Return is a tuple with similar ordering,
    but with indices.

    Examples:
        Limits: (-2, 0, 0, 1)
        Latitude: [-5, -4, -3, -2, -1, 0, 1, 2, 3]
        Longitude: [-5, -4, -3, -2, -1, 0, 1, 2, 3]

        Return: (3, 5, 5, 6)

    Args:
        dataset: The dataset.
        limits: The limits.

    Returns:
        The indices.

    """
    _function_call()

    data = _load(dataset, None, None, None)

    latitude = data['lat']
    longitude = data['lon']

    lat_idx = np.asarray((latitude >= limits[0]) & (latitude <= limits[1])).nonzero()[0]
    lon_idx = np.asarray((longitude >= limits[2]) & (longitude <= limits[3])).nonzero()[0]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise IndexError('No data found for the given limits')

    coo_index = lat_idx[0], lat_idx[-1] + 1, lon_idx[0], lon_idx[-1] + 1
    latitude = data['lat'][coo_index[0]:coo_index[1]]
    longitude = data['lon'][coo_index[2]:coo_index[3]]

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


def _get_data(dataset: Dataset, variable: Variable, limits: LIMIT_TYPE, time: TIME_TYPE) -> DataCollection:
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
                                                                                                     2]:coo_index[3]], ]

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
                                          shading='auto',
                                          vmin=ticks[0], vmax=ticks[-1])
                        cbar = fig.colorbar(p, orientation='vertical', ticks=ticks)
                        cbar.ax.set_yticklabels([str(tick) for tick in ticks])
                    else:
                        p = ax.pcolormesh(g.longitude, g.latitude, g.get_component(time_index, 0),
                                          transform=ccrs.PlateCarree(), cmap=cmap,
                                          shading='auto')
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
        save_title = directory + '/' + save_titles[time_index] if directory is not None else save_titles[time_index]
        _output_figure(out_mode, save_title)

    return None


def plot_point_data_over_time(data_collection: DataCollection, title: Optional[str], tick_mode: Optional[str], size:
                              POINT_TYPE, out_mode: str, save_title: Optional[str], y_line: Optional[float], font_size:
                              int = 12) -> matFigure | None:
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

    fig = plt.figure(figsize=size)
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
        return fig


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


# ======================== STATISTICS ==================================================================================
def average_data_collection(data_collection: DataCollection) -> DataCollection:
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


def bound_frequency_data_collection(data_collection: DataCollection, lower_bound: float,
                                    upper_bound: float) -> DataCollection:
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


def percentile_data_collection(data_collection: DataCollection, percentile: float) -> DataCollection:
    """
    Finds the given percentile of a data collection's data. This is done point-wise over the time axis. That is, for
    each coordinate point over time.

    Args:
        data_collection: The data collection.
        percentile: The percentile.

    Returns:
        A new data_collection with resulting data. The dataset, variable, time, limits, latitude and longitude are all
        the same as from the original data collection. The title is updated, for a data collection dc, to

        {dc.dataset}: {percentile}-th percentile of {dc.variable} {time suffix for dc's time}

    """
    percentile_data = list()
    for i in range(data_collection.get_vector_dimension()):
        flat_percentile = np.percentile(data_collection.get_component(None, i), percentile, axis=0)
        percentile_data.append(np.expand_dims(flat_percentile, axis=0))

    title_prefix = f'{data_collection.dataset}: {percentile}-th percentile of {data_collection.variable} '

    return DataCollection(data_collection.dataset, data_collection.variable, data_collection.time,
                          data_collection.get_limits(), percentile_data, data_collection.latitude, data_collection.longitude,
                          title_prefix, '', (data_collection.time,))


def percentile_date_data_collection(spec_data_collection: DataCollection, ref_data_collection: DataCollection) -> \
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


def max_data_collection(data_collection: DataCollection, per_time_slice: bool) -> tuple[PointCollection, ArrayLike] |\
                                                                                  tuple[PointCollection, TIME_TYPE,
                                                                                        ArrayLike]:
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
            is_containted = list()
            for lat in data_collection.latitude:
                for lon in data_collection.longitude:
                    is_containted.append(int(path.contains_point((lon, lat))))

            is_contained = np.reshape(is_containted, data_collection.spread)
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
    if data_collection.dimension != 1:
        raise ValueError('The data collection must be of a scalar.')
    if data_collection.get_time_length() != 1:
        raise ValueError('The data collection must have a time length of 1.')

    plt.close('all')
    # fig = plt.figure()
    ax = plt.axes()

    cs = ax.contour(data_collection.longitude, data_collection.latitude, data_collection.get_component(0, 0),
                    levels=(contour,))

    contour_object = cs.collections[0]

    if path_index >= len(contour_object.get_paths()):
        raise IndexError('Contour path index too large')
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


def refine_area_to_illustration(path_collection: PathCollection, point_condition: Callable[[int, int], bool]) -> \
        DataCollection:
    coo_index, latitude, longitude = _get_coordinate_information(path_collection.dataset, path_collection.get_limits())
    show_points = np.zeros((path_collection.get_time_length(), len(latitude), len(longitude)))

    for time_idx in range(path_collection.get_time_length()):
        for lat_idx, lat in enumerate(latitude):
            for lon_idx, lon in enumerate(longitude):
                if path_collection.contains_point(time_idx, (lat, lon)) and point_condition(lat, lon):
                    show_points[time_idx, lat_idx, lon_idx] = 1

    return DataCollection(path_collection.dataset, path_collection.variable, path_collection.time, path_collection.get_limits(),
                          [show_points, ], latitude, longitude, path_collection.title_prefix,
                          path_collection.title_suffix + ' (refined)', path_collection.time_stamps)


# ======================== ATMOSPHERIC PHYSICS CONVENTIONS =============================================================
def get_winter_year(time: TIME_TYPE) -> int:
    year, month, day, hour = time
    if month is None or year is None:
        raise ValueError('The year and months must not be None.')
    if month in (11, 12):
        return year + 1
    elif month in (1, 2, 3, 4):
        return year
    else:
        raise ValueError('The given month is not recognized as a winter month')


# ======================== MAIN ========================================================================================
def example_function(lat, lon) -> bool:
    return True


def main():
    limits, projection = get_defaults()
    print(get_winter_year((1980, 1, None, None)))


if __name__ == '__main__':
    main()
