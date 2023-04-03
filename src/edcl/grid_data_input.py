"""
The grid_data_input module houses all functions dealing with information *gathered* from data files for Earth data.
These include functions for loading this data into GridCollection objects and functions which read metadata from the
files (available years, for example).
"""

from scipy.io import loadmat
from .info_classes import Dataset, Variable
from typing import Optional
from . import config as cfg
from .formatting import format_month
from glob import glob
from os.path import basename, isfile
import numpy as np


# ======================================================================================================================
# LOADING
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

    if not all((cfg.loaded_dataset is dataset, (cfg.loaded_variable is variable or dataset.is_unified),
                cfg.loaded_year == year, cfg.loaded_month == month)):
        # Check if requested data is available
        path = get_path(dataset, year, month, variable)
        if not isfile(path):
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

    data = load(dataset, variable, year, month)
    days = data['day_ts']

    # Check that the requested day is available
    if day not in days:
        raise ValueError('The requested day is not available.')

    hours = data['hour_ts']
    hour_inds = np.asarray(days == day).nonzero()[0]

    return hours[hour_inds[0]:hour_inds[-1] + 1].tolist()
