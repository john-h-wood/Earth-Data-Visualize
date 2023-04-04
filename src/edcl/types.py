"""
The edcl_types (so named to avoid conflict with Python's types module) module houses typing definitions.
"""
from typing import Optional
from matplotlib.path import Path
from numpy.typing import ArrayLike
from cartopy.crs import Projection

# A point on the Earth's surface specified by a pair of indices. These refer to the index of the point in a regular
# grid of latitude/longitude values
POINT_INDEX = tuple[int, int]

# Limits defining a region on the Earth's surface in latitude/longitude values. They are formatted as
# (lat_min, lat_max, lon_min, lon_max)
LIMITS = tuple[float, float, float, float]

# Limits defining a region on the Earth's surface in the space of indices of the latitude/longitude vectors. They are
# formatted as the LIMITS type, but with indices
IDX_LIMITS = tuple[int, int, int, int]

# A time, specified by a year, month, day and hour. These may all be None to refer to periods of time
TIME = tuple[Optional[int], Optional[int], Optional[int], Optional[int]]

# A list of times, meant to give a time-title for each slice of data along the time axis in a collection
TIME_STAMPS = tuple[TIME, ...]

# Earth data, which is a grid of values for each component of data. Wind speed, for example, is a singular 2D grid,
# but wind vectors are a list of two such grids, one each for the horizontal and vertical components
GRID = ArrayLike | tuple[ArrayLike, ArrayLike]

# A point on the Earth's surface specified by a pair of latitude/longitude coordinates
POINT = tuple[float, float]

# A path on the Earth's surface, specified by the latitude/longitude coordinates of a series of points. See
PATH = Path

# A tuple of GRIDs, each associated with a specific time
GRID_IN_TIME = tuple[ArrayLike, ...] | tuple[tuple[ArrayLike, ArrayLike], ...]

# A tuple of POINTs, each associated with a specific time
POINT_IN_TIME = tuple[tuple[float, float], ...]

# A tuple of PATHS, each associated with a specific time
PATH_IN_TIME = tuple[Path, ...]

# Generic data
DATA = GRID | POINT | PATH

# A tuple of DATAs, all of the same type, each associated with a specific time
DATA_IN_TIME = GRID_IN_TIME | POINT_IN_TIME | PATH_IN_TIME

# Matrix of coordinates (latitude or longitude)
COORDINATES = ArrayLike

# Map projection from Cartopy
PROJECTION = Projection


def grid_in_time_components(grid_in_time: GRID_IN_TIME) -> int:
    """
    Determine the number of components in a grid in time. A grid in time for scalars would yield 1 for example.

    Assumes, from the type definition of GRID_IN_TIME, that each grid in the time tuple has the same number components.

    Args:
        grid_in_time: The grid in time.

    Returns:
        The number of components. Zero if the grid in time is empty or if its grids are empty.
    """
    if len(grid_in_time) == 0:
        return 0
    else:
        example_grid = grid_in_time[0]
        if isinstance(example_grid, tuple):
            return len(example_grid) # should always just be two
        else:
            return 1


def time_is_supported(time: TIME) -> bool:
    """
    Whether the given time is supported. This is determined by the nullity of the time tuple elements.

    In terms of nullity, supported time types are, where given is the same as not none:
    Year          Month          Day          Hour
    given         given          given        given
    given         given          given        none
    given         given          none         none
    given         none           none         none
    none          none           none         none
    none          given          none         none

    Args:
        time: The time.

    Returns:
        Whether the time is supported.
    """
    year, month, day, hour = time
    # Special case for given month only
    if month is not None and year is None and day is None and hour is None:
        return True

    # Otherwise, only descending nullity is allowed
    else:
        this_is_none = year is None
        for item in time[1:]:
            if this_is_none and (item is not None):
                return False
            this_is_none = item is None
    return True
