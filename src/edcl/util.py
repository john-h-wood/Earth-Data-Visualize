"""
The util module houses useful functions which do not return or produce results for their right; utilities,
and which do not entirely and closely relate to the scope of another module. This includes, say, a function to get the
Dataset object corresponding to an identifier, or a function to compute maximal limits.
"""

import numpy as np
import cartopy.crs as ccrs

from . import info
from .types import LIMITS, PROJECTION
from . import config
from .info_classes import Dataset, Variable


def print_loaded() -> None:
    """
    Prints metadata for the the currently loaded grid collection dictionary.

    Only one relating dictionary from scipy.io.loadmat is kept at a time an avoid needlessly loading data.

    Returns:
        None.
    """
    print(f'Loaded Dictionary: {config.loaded_dataset}, {config.loaded_variable}, {config.loaded_year}, '
          f'{config.loaded_month}')


def maximal_limits(limits: tuple[LIMITS]) -> LIMITS:
    """
    Find the limits which encompass all coordinates of a set of limits.

    Examples:
        For input limits (-10, 15, -20, 25) and (-30, 7, -90, 40), the result is (-30, 15, -90, 40)

    Args:
        limits: The set of limits.

    Returns:
        The encompassing limits.
    """
    limit_matrix = np.reshape(limits, (len(limits), 4))
    lat_min = np.amin(limit_matrix[:, 0])
    lat_max = np.amax(limit_matrix[:, 1])
    lon_min = np.amin(limit_matrix[:, 2])
    lon_max = np.amax(limit_matrix[:, 3])

    return lat_min, lat_max, lon_min, lon_max


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
    for dataset in info.datasets:
        if dataset.name == name:
            return dataset
    raise ValueError('No such dataset was found.')


def get_variable_name(dataset: Dataset, name: str) -> Variable:
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
    for variable in dataset.variables:
        if variable.name == name:
            return variable
    raise ValueError('No such variable was found.')


def get_variable_identifier(dataset: Dataset, identifier: int) -> Variable:
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
    for variable in dataset.variables:
        if variable.identifier == identifier:
            return variable
    raise ValueError('No such variable was found.')


def get_projection_name(name: str, limits: LIMITS = None) -> PROJECTION:
    """
    Returns the cartopy.crs object for the given projection.

    If coordinates are not none and the selected projection is Lambert Conformal, the projection is centred on the
    central latitudes and longitudes.

    Args:
        name: The name of projection.
        limits: The limits.

    Returns: The object.

    Raises:
        ValueError: Limits must be given for the Lambert projection.
        ValueError: No such projection found.
    """
    if name == 'Lambert':
        if limits is not None:
            return ccrs.LambertConformal(central_longitude=np.mean(limits[2:]), central_latitude=np.mean(limits[:2]))
        else: raise ValueError('Limits must be given for the Lambert projection.')
    elif name == 'Robinson':
        return ccrs.Robinson()
    elif name == 'Plate Carree':
        return ccrs.PlateCarree()
    else: raise ValueError('No such projection found.')
