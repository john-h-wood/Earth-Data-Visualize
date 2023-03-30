"""
The edcl_types (so named to avoid conflict with Python's types module) module houses typing definitions.
"""
from typing import Optional
from numpy.typing import ArrayLike
from matplotlib.path import Path

# A point on the Earth's surface specified by a pair of indices. These refer to the index of the point in a regular
# grid of latitude/longitude values
POINT_INDEX = tuple[int, int]

# Limits defining a region on the Earth's surface in latitude/longitude values. They are formatted as
# (lat_min, lat_max, lon_min, lon_max)
LIMITS = tuple[float, float, float, float]

# A time, specified by a year, month, day and hour. These may all be None to refer to periods of time
TIME = tuple[Optional[int], Optional[int], Optional[int], Optional[int]]

# A list of times, meant to give a time-title for each slice of data along the time axis in a collection
TIME_STAMPS = tuple[TIME]

# Earth data, which is a grid of values for each component of data. Wind speed, for example, is a singular 2D grid,
# but wind vectors are a list of two such grids, one each for the horizontal and vertical components
GRID = ArrayLike | tuple[ArrayLike, ArrayLike]

# A point on the Earth's surface specified by a pair of latitude/longitude coordinates
POINT = tuple[float, float]

# A path on the Earth's surface, specified by the latitude/longitude coordinates of a series of points. See
PATH = Path

# A tuple of GRIDs, each associated with a specific time
GRID_IN_TIME = tuple[GRID]

# A tuple of POINTs, each associated with a specific time
POINT_IN_TIME = tuple[POINT]

# A tuple of PATHS, each associated with a specific time
PATH_IN_TIME = tuple[PATH]

# Generic data
DATA = GRID | POINT | PATH

# A tuple of DATAs, all of the same type, each associated with a specific time
DATA_IN_TIME = GRID_IN_TIME | POINT_IN_TIME | PATH_IN_TIME

# Matrix of coordinates (latitude or longitude)
COORDINATES = ArrayLike
