"""
The grid_data_io_util module houses all functions dealing with information gathered from data files for Earth data.
These include functions for loading this data into GridCollection objects and functions which read metadata from the
files (available years, for example).
"""

from scipy.io import loadmat
from .info_classes import Dataset, Variable
from typing import Optional
from . import info
from .formatting import format_month
from glob import glob
from os.path import basename, isfile


def get_path(dataset: Dataset, year: int, month: int, variable: Variable) -> str:
    """
    Get the formatted filepath for specified grid data. Importantly, this path may or may not exist.

    Args:
        dataset: The dataset.
        year: The year.
        month: The month.
        variable: The variable.

    Returns:
        The filepath.
    """
    return f'{info.directory}/{dataset.directory}/{year}/{dataset.file_prefix}_{variable.file_identifier}_' \
           f'm{format_month(month)}_y{year}_{dataset.file_suffix}.mat'


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

    for path in glob(f'{info.directory}/{dataset.directory}/*'):
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
    # TODO add special case for non-unified datasets. Variable must available for a month to be included, unless the
    #  inputted variable is None

    # Access files within the given year's directory

    for path in glob(f'{info.directory}/{dataset.directory}/{year}/*'):
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
