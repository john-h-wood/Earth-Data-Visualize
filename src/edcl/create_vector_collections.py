"""
The create_vector_collections module houses all functions which create VectorCollections.
"""
from typing import Optional

from .types import LIMITS, TIME, TIME_STAMPS
from .info_classes import Dataset, Variable
from .grid_data_input import get_coordinate_information, get_time_stamps, get_interpreted_grid, re_shape_grids, load
from .util import get_dataset_name, get_variable_name
from .collections import VectorCollection


def get_vector_collection(dataset: Dataset, variable: Variable, time: TIME, limits: Optional[LIMITS]) -> \
                         VectorCollection:
    """
    Gathers grid data for a variable, dataset, time, and coordinate limits and returns it as a VectorCollection.

    If the given coordinate limits are None, data for all coordinates is returned.

    Args:
        dataset: The dataset.
        variable: The variable.
        time: The time.
        limits: The coordinate limits. Possibly None.

    Returns:
        A VectorCollection for the data.
    """
    idx_limits = get_coordinate_information(dataset, limits)
    interpreted_data = get_interpreted_grid(dataset, variable, time, idx_limits)
    title_prefix = f'{dataset.name}: {variable.name} '
    time_stamps = get_time_stamps(dataset, variable, time)

    # Get latitude and longitude information, which is only dataset-specific
    data = load(dataset, None, None, None)
    latitude = data['lat'][idx_limits[0]:idx_limits[1]]
    longitude = data['lon'][idx_limits[2]:idx_limits[3]]

    # Re-shape interpreted data from tuple of SCALAR_GRID_IN_TIME to VECTOR_GRID_IN_TIME
    interpreted_data = re_shape_grids(interpreted_data)

    return VectorCollection(dataset, variable, time, time_stamps, title_prefix, '', interpreted_data, latitude,
                            longitude)


# ======================================================================================================================
# CONVENIENCE FUNCTIONS
# ======================================================================================================================
def get_vector_collection_names(dataset_name: str, variable_name: str, time: TIME, limits: Optional[LIMITS]) -> \
                               VectorCollection:
    """
    Convenience function which converts a dataset and variable name, time, and limits to a grid collection via
    conversion methods and the get_vector_collection_method.

    Args:
        dataset_name: The name of the dataset.
        variable_name: The name of the variable.
        time: The time.
        limits: The limits. Possibly None.

    Returns:
        The vector collection.
    """
    dataset = get_dataset_name(dataset_name)
    variable = get_variable_name(dataset, variable_name)
    return get_vector_collection(dataset, variable, time, limits)
