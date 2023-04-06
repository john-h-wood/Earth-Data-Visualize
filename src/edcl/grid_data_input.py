"""
The grid_data_input module houses all functions dealing with information *gathered* from data files for Earth data.
These include functions for loading this data into np arrays and functions which read metadata from the
files (available years, for example).
"""

import numpy as np
from glob import glob
from scipy.io import loadmat
from os.path import basename, isfile

from typing import Optional
from . import config as cfg
from .formatting import format_month
from .info_classes import Dataset, Variable
from .util import get_variable_identifier
from .types import LIMITS, IDX_LIMITS, TIME, TIME_STAMPS, VECTOR_GRID_IN_TIME, SCALAR_GRID_IN_TIME, time_is_supported


# ======================================================================================================================
# PATHS AND LOADING
# ======================================================================================================================
def get_path(dataset: Dataset, year: int, month: int, variable: Optional[Variable]) -> str:
    """
    Get the formatted filepath for specified grid data. Importantly, this path may or may not exist.

    The variable does not have to specified if the dataset is unified.

    Args:
        dataset: The dataset.
        year: The year.
        month: The month.
        variable: The variable. Possibly None.

    Returns:
        The filepath.
    """
    # Validate parameters
    if not dataset.is_unified and variable is None:
        raise ValueError('The variable must be specified for non-unified datasets.')

    if dataset.is_unified:
        return f'{cfg.info.directory}/{dataset.directory}/{year}/{dataset.file_prefix}_' \
               f'm{format_month(month)}_y{year}_{dataset.file_suffix}.mat'
    else:
        return f'{cfg.info.directory}/{dataset.directory}/{year}/{dataset.file_prefix}_{variable.file_identifier}_' \
               f'm{format_month(month)}_y{year}_{dataset.file_suffix}.mat'


def load(dataset: Optional[Dataset], variable: Optional[Variable], year: Optional[int], month: Optional[int]) -> dict:
    """
    Loads desired data, filepath and dataset to global variables. Returns the loaded data.

    Desired data is specified with a dataset, date, time and variable. If one or more of these is None, they will be
    selected in to avoid re-loading data if possible. A None variable should be used for when the used
    variable does not matter, for example to gather latitude and longitude information. Using this function increases
    efficiency by ensuring that dats is not needlessly loaded. Since variables are specific to dataset, the variable
    not being None means the dataset cannot be None. Only descending nullity is supported. If one parameter is None,
    every parameter after it must be None.

    Args:
        dataset: The dataset. Possibly None.
        year: The year. Possibly None.
        month: The month. Possibly None.
        variable: The variable. Possibly None.

    Returns:
        The data as a dictionary.

    Raises:
        ValueError: The variable not being None means the dataset cannot be None.
    """
    # Validate parameters
    givens = (dataset, variable, year, month)

    this_is_none = dataset is None
    for item in givens[1:]:
        if this_is_none and (item is not None):
            raise NotImplementedError('Only descending nullity is supported.')
        this_is_none = item is None

    # Populate unspecified values, making them the loaded values if those exist
    if dataset is None:
        if cfg.loaded_dataset is None:
            cfg.loaded_dataset = cfg.info.datasets[0]
        dataset = cfg.loaded_dataset

    if variable is None:
        if cfg.loaded_dataset is dataset:
            if cfg.loaded_variable is None:
                cfg.loaded_variable = dataset.variables[0]
            variable = cfg.loaded_variable
        else:
            variable = dataset.variables[0]

    if year is None:
        if cfg.loaded_dataset is dataset and (cfg.loaded_variable is variable or dataset.is_unified):
            if cfg.loaded_year is None:
                cfg.loaded_year = get_years(dataset, variable)[0]
            year = cfg.loaded_year
        else:
            year = get_years(dataset, variable)[0]

    if month is None:
        if cfg.loaded_dataset is dataset and (cfg.loaded_variable is variable or dataset.is_unified) and \
           cfg.loaded_year == year:
            if cfg.loaded_month is None:
                cfg.loaded_month = get_months(dataset, year, variable)
            month = cfg.loaded_month
        else:
            month = get_months(dataset, year, variable)[0]

    # Special case for unified datasets
    if not all((cfg.loaded_dataset is dataset, (cfg.loaded_variable is variable or dataset.is_unified),
                cfg.loaded_year == year, cfg.loaded_month == month)):
        # Check if requested data is available
        path = get_path(dataset, year, month, variable)
        if not isfile(path):
            print(path)
            raise ValueError('The requested data is not available.')
        cfg.loaded_dataset = dataset
        cfg.loaded_variable = variable
        cfg.loaded_year = year
        cfg.loaded_month = month
        cfg.loaded_data = loadmat(path, squeeze_me=True)

    # Requested data could be the same as loaded data params, but loaded data may not have been initialized
    if cfg.loaded_data is None:
        # Check if requested data is available
        path = get_path(dataset, year, month, variable)
        if not isfile(path):
            raise ValueError('The requested data is not available.')
        cfg.loaded_data = loadmat(path, squeeze_me=True)

    return cfg.loaded_data


# ======================================================================================================================
# COMPUTED METADATA
# ======================================================================================================================
def get_years(dataset: Dataset, variable: Optional[Variable]) -> list[int]:
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
    # Special case for combo variable: result is union of recursively found years for constituent variables
    if variable is not None and variable.is_combo:
        equation_type, x_identifier, y_identifier = variable.equation.split('_')
        x_variable = get_variable_identifier(dataset, int(x_identifier))
        y_variable = get_variable_identifier(dataset, int(y_identifier))
        x_years = get_years(dataset, x_variable)
        y_years = get_years(dataset, y_variable)

        return np.union1d(x_years, y_years).tolist()

    years = list()

    # Rely on organization within main dataset directory being in years

    for path in glob(f'{cfg.info.directory}/{dataset.directory}/*'):
        # determine if valid year
        test_year = int(basename(path))
        if variable is None:
            years.append(test_year)
        else:
            if len(get_months(dataset, test_year, variable)) > 0:
                years.append(test_year)

    return sorted(years)


def get_months(dataset: Dataset, year: int, variable: Optional[Variable]) -> list[int]:
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
    # Special case for combo variable: result is union of recursively found months for constituent variables
    if variable is not None and variable.is_combo:
        equation_type, x_identifier, y_identifier = variable.equation.split('_')
        x_variable = get_variable_identifier(dataset, int(x_identifier))
        y_variable = get_variable_identifier(dataset, int(y_identifier))
        x_months = get_months(dataset, year, x_variable)
        y_months = get_months(dataset, year, y_variable)

        return np.union1d(x_months, y_months).tolist()

    checked_months = list()
    valid_months = list()

    # Access files within the given year's directory
    for path in glob(f'{cfg.info.directory}/{dataset.directory}/{year}/*'):
        sub_path = basename(path)
        test_month = int(sub_path[sub_path.rindex('_') - 8:sub_path.rindex('_') - 6])  # relies heavily on formatting!

        # Has this month been added already? If so, move on
        if test_month in checked_months:
            continue

        checked_months.append(test_month)

        # Tests for whether to include the month
        if variable is None:
            valid_months.append(test_month)
            continue
        if dataset.is_unified:
            valid_months.append(test_month)
            continue
        if (not dataset.is_unified) and isfile(get_path(dataset, year, test_month, variable)):
            valid_months.append(test_month)
            continue

    return sorted(valid_months)


def get_days(dataset: Dataset, variable: Optional[Variable], year: int, month: int) -> list[int]:
    """
    Returns a sorted list of days available within a month of a variable's data.

    The variable may only be unspecified if the data is unified.

    Args:
        dataset: The dataset.
        variable: The variable. Possibly None
        year: The year.
        month: The month.

    Returns:
        The days.

    Raises:
        ValueError: The variable must be specified for non-unified datasets.
    """
    # Validate parameters
    if not dataset.is_unified and variable is None:
        raise ValueError('The variable must be specified for non-unified datasets.')
    if variable is None:
        variable = dataset.variables[0]

    # Special case for combo variable: result is union of recursively found days for constituent variables
    if variable.is_combo:
        equation_type, x_identifier, y_identifier = variable.equation.split('_')
        x_variable = get_variable_identifier(dataset, int(x_identifier))
        y_variable = get_variable_identifier(dataset, int(y_identifier))
        x_days = get_days(dataset, x_variable, year, month)
        y_days = get_days(dataset, y_variable, year, month)

        return np.union1d(x_days, y_days).tolist()

    data = load(dataset, variable, year, month)
    # noinspection PyTypeChecker
    return np.unique(data['day_ts']).tolist()


def get_hours(dataset: Dataset, variable: Variable, year: int, month: int, day: int) -> list[int]:
    """
    Returns a sorted list of hours available within a day of a variable's data.

    The variable may only be unspecified for non-unified datasets.

    Args:
        dataset: The dataset.
        variable: The variable. Possibly None.
        year: The year.
        month: The month.
        day: The day.

    Returns:
        The hours.

    Raises:
        ValueError: The variable must be specified for non-unified datasets.
        ValueError: The requested day is not available.
    """
    # Validate parameters
    if not dataset.is_unified and variable is None:
        raise ValueError('The variable must be specified for non-unified datasets.')
    if variable is None:
        variable = dataset.variables[0]

    # Special case for combo variable: result is union of recursively found hours for constituent variables
    if variable.is_combo:
        equation_type, x_identifier, y_identifier = variable.equation.split('_')
        x_variable = get_variable_identifier(dataset, int(x_identifier))
        y_variable = get_variable_identifier(dataset, int(y_identifier))
        x_hours = get_hours(dataset, x_variable, year, month, day)
        y_hours = get_hours(dataset, y_variable, year, month, day)

        return np.union1d(x_hours, y_hours).tolist()

    data = load(dataset, variable, year, month)
    days = data['day_ts']

    # Check that the requested day is available
    if day not in days:
        raise ValueError('The requested day is not available.')

    hours = data['hour_ts']
    hour_inds = np.asarray(days == day).nonzero()[0]

    return hours[hour_inds[0]:hour_inds[-1] + 1].tolist()


def get_time_stamps(dataset: Dataset, variable: Variable, time: TIME) -> TIME_STAMPS:
    """
    Creates and returns timestamps for a dataset, variable, annd time. There is one time stamp for each index of
    the data's time axis.

    Time stamps are formatted as a tuple: (year, month, day, hour).

    Returns:
        The time stamps.

    Raises:
        ValueError: The given time type is not supported.
    """
    # Validate parameters
    if not time_is_supported(time):
        raise ValueError('The given time type is not supported.')
    time_stamps = list()
    range_year, range_month, range_day, range_hour = time

    if (range_month is not None) and (range_year is None):
        years = [x for x in get_years(dataset, None) if range_month in get_months(dataset, x, variable)]
        for year in years:
            for day in get_days(dataset, variable, year, range_month):
                for hour in get_hours(dataset, variable, year, range_month, day):
                    time_stamps.append((year, range_month, day, hour))
    elif range_year is None:
        for year in get_years(dataset, variable):
            for month in get_months(dataset, year, variable):
                for day in get_days(dataset, variable, year, month):
                    for hour in get_hours(dataset, variable, year, month, day):
                        time_stamps.append((year, month, day, hour))
    elif range_month is None:
        for month in get_months(dataset, range_year, variable):
            for day in get_days(dataset, variable, range_year, month):
                for hour in get_hours(dataset, variable, range_year, month, day):
                    time_stamps.append((range_year, month, day, hour))
    elif range_day is None:
        for day in get_days(dataset, variable, range_year, range_month):
            for hour in get_hours(dataset, variable, range_year, range_month, day):
                time_stamps.append((range_year, range_month, day, hour))
    elif range_hour is None:
        for hour in get_hours(dataset, variable, range_year, range_month, range_day):
            time_stamps.append((range_year, range_month, range_day, hour))
    else:
        time_stamps.append(time)

    return tuple(time_stamps)


# ======================================================================================================================
# CUTTING AND INTERPRETING
# ======================================================================================================================
def get_time_index(dataset: Dataset, variable: Optional[Variable], year: int, month: int, day: int, hour: int) -> int:
    """
        Get the time index of the specified dataset, date and time.

        Data is stored in matrices which have their first dimension as an index for each available hour. This
        function finds that index, to then be used to get hour-specific data. The variable may only be None for
        non-unified datasets.

        Args:
            dataset: The dataset.
            variable: The variable. Possibly None.
            year: The year.
            month: The month,
            day: The day.
            hour: The hour.

        Returns:
            The index.

        Raises:
            ValueError: The variable must be specified for non-unified datasets.
        """
    # Validate parameters
    if not dataset.is_unified and variable is None:
        raise ValueError('The variable must be specified for non-unified datasets.')

    data = load(dataset, variable, year, month)
    days = data['day_ts']
    hours = data['hour_ts']

    hour_inds = np.asarray(days == day).nonzero()[0]
    hours_sub = hours[hour_inds[0]:hour_inds[-1] + 1].tolist()

    return hour_inds[0] + hours_sub.index(hour)


def get_coordinate_information(dataset: Dataset, limits: LIMITS) -> IDX_LIMITS:
    """
        Get the coordinate indices corresponding to given coordinate limits.

        Data is stored in matrices which have their first dimension as time. The second and third dimensions refer,
        respectively, to latitude and longitude. This function returns the index limits for latitude and longitude,
        given coordinate limits. That is, the upper and lower indices for which both latitude and longitude are
        within or
        equal to specified bounds. One is added to the upper indices so that a call such as lat[lat_ind_min:lat_ind_max]
        yields the expected latitudes.

        Limits are formatted as (lat_min, lat_max, lon_min, lon_max). Return is a tuple with similar ordering,
        but with indices.

        If the limits are None, the returned indices correspond to all coordinate elements.

        Examples:
            Limits: (-2, 0, 0, 1)
            Latitude: [-5, -4, -3, -2, -1, 0, 1, 2, 3]
            Longitude: [-5, -4, -3, -2, -1, 0, 1, 2, 3]

            Return: (3, 6, 5, 7)

        Args:
            dataset: The dataset.
            limits: The limits. Possibly None.

        Returns:
            The indices.

        Raises:
            ValueError: The given coordinates are malformed.
            ValueError: No data is available within the given latitude range.
            ValueError: No data is available within the given longitude range.
        """
    data = load(dataset, None, None, None)
    latitude = np.flip(data['lat']) # TODO very important assumption about coordinate data ordering. Check this
    longitude = data['lon']

    # Check this first. It wouldn't pass the parameter validation because of nullity
    if limits is None:
        return 0, len(latitude), 0, len(longitude)

    # Validate parameters (is there any data available for the given limits?)
    # This check is especially important because a binary bisection for oder preserving is used.
    lat_min, lat_max, lon_min, lon_max = limits
    if lat_min > lat_max or lon_min > lon_max:
        raise ValueError('The given limits are not formatted correctly.')
    if lat_min > latitude[-1] or lat_max < latitude[0]:
        raise ValueError('No data is available within the given latitude range.')
    if lon_min > longitude[-1] or lon_max < longitude[0]:
        raise ValueError('No data is available within the given longitude range.')

    lat_min_idx, lat_max_idx = np.searchsorted(latitude, (lat_min, lat_max))
    lon_min_idx, lon_max_idx = np.searchsorted(longitude, (lon_min, lon_max))

    # Covert latitude values to un-flipped indices
    unflipped_lat_min_idx = (len(latitude) - 1) - lat_max_idx
    unflipped_lat_max_idx = (len(latitude) - 1) - lat_min_idx

    return unflipped_lat_min_idx, unflipped_lat_max_idx + 1, lon_min_idx, lon_max_idx + 1


def compute_combo_equation(equation_type: str, x: SCALAR_GRID_IN_TIME, y: SCALAR_GRID_IN_TIME) -> \
                          tuple[SCALAR_GRID_IN_TIME, ...]:
    """
        Get the result of a combo variable equation.

        Takes an equation type and two component variables to yield a single combo variable. The components must have
        the same size. They mare or may not have the same number of coordinates; a combo variable could be defined
        which has a two-component grid as one of its constituent variables.

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

        Raises:
            ValueError: The given equation type is not supported.
        """
    if equation_type == 'component':
        return x, y
    elif equation_type == 'polar':
        return x * np.sin(np.deg2rad(y)), x * np.cos(np.deg2rad(y))
    elif equation_type == 'norm':
        return np.sqrt(np.square(x) + np.square(y)),
    elif equation_type == 'direction':
        return np.rad2deg(np.arctan2(y, x)) - 90,  # TODO use oceanographic convention. This equation is wrong.
    else:
        raise ValueError('The given equation type is not supported.')


def get_interpreted_grid(dataset: Dataset, variable: Variable, time: TIME, idx_limits: IDX_LIMITS) -> \
                        tuple[SCALAR_GRID_IN_TIME, ...]:
    """
       Gathers grid data in time for a variable, dataset, time, and cut to coordinate limits.

       Performs any calculations coming from combo variables.
       This function is separate from others, including get_data, because it uses recursion for combo_variable
       computation. If the time has no None elements, the time index of the data within the month may be specified.
       Otherwise, it will be calculated.


       Args:
           dataset: The dataset.
           variable: The variable.
           time: The time.
           idx_limits: The coordinate limit indices.
           time_index: The time index. Possibly None.

       Returns:
           The cut and interpreted data.

        Raises:
            ValueError: The given time type is not supported.
            ValueError: The time index may only be specified if the time does not have a None element.
       """
    # Validate parameters
    if not time_is_supported(time):
        raise ValueError('The given time type is not supported.')

    if variable.is_combo:
        equation_type, x_identifier, y_identifier = variable.equation.split('_')
        x_variable = get_variable_identifier(dataset, int(x_identifier))
        y_variable = get_variable_identifier(dataset, int(y_identifier))
        x_data = get_interpreted_grid(dataset, x_variable, time, idx_limits)
        y_data = get_interpreted_grid(dataset, y_variable, time, idx_limits)

        # Input first (and only) coordinate to equation computation. Due to recursion, x_data must have single
        # component.
        return compute_combo_equation(equation_type, x_data[0], y_data[0])

    else:
        year, month, day, hour = time

        if (year is None) and (month is not None):
            month_datas = list()
            years = [x for x in get_years(dataset, None) if month in get_months(dataset, x, variable)]
            for year in years:
                variable_data = load(dataset, variable, year, month)[variable.key]
                month_datas.append(variable_data[:, idx_limits[0]:idx_limits[1], idx_limits[2]:idx_limits[3]])

            return np.concatenate(month_datas),

        elif year is None:
            year_datas = list()

            for year in get_years(dataset, variable):
                for month in get_months(dataset, year, variable):
                    variable_data = load(dataset, variable, year, month)[variable.key]
                    year_datas.append(variable_data[:, idx_limits[0]:idx_limits[1], idx_limits[2]:idx_limits[3]])
            return np.concatenate(year_datas),

        elif month is None:
            month_datas = list()

            for month in get_months(dataset, year, variable):
                variable_data = load(dataset, variable, year, month)[variable.key]
                month_datas.append(variable_data[:, idx_limits[0]:idx_limits[1], idx_limits[2]:idx_limits[3]])
            return np.concatenate(month_datas),

        elif day is None:
            variable_data = load(dataset, variable, year, month)[variable.key]
            return variable_data[:, idx_limits[0]:idx_limits[1], idx_limits[2]:idx_limits[3]],

        elif hour is None:
            hours = get_hours(dataset, variable, year, month, day)
            variable_data = load(dataset, variable, year, month)[variable.key]
            start_time_index = get_time_index(dataset, variable, year, month, day, hours[0])
            end_time_index = get_time_index(dataset, variable, year, month, day, hours[-1])

            return variable_data[start_time_index:end_time_index + 1,
                                 idx_limits[0]:idx_limits[1], idx_limits[2]:idx_limits[3]],

        else:
            variable_data = load(dataset, variable, year, month)[variable.key]
            time_index = get_time_index(dataset, variable, year, month, day, hour)
            data = variable_data[time_index, idx_limits[0]:idx_limits[1], idx_limits[2]:idx_limits[3]]

            # Expand to 3D array
            data = np.expand_dims(data, axis=0)
            return data,


def re_shape_grids(data: tuple[SCALAR_GRID_IN_TIME, ...]) -> VECTOR_GRID_IN_TIME:
    return np.swapaxes(np.array(data), 0, 1)
