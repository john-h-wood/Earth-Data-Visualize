"""
The edcl_types (so named to avoid conflict with Python's types module) module houses typing definitions.
"""
from typing import Optional
from numpy.typing import ArrayLike

# A point on the Earth's surface specified by a pair of latitude/longitude coordinates
POINT = tuple[float, float]

# A point on the Earth's surface specified by a pair of indices. These refer to the index of the point in a regular
# grid of latitude/longitude values
POINT_INDEX = tuple[int, int]

# Limits defining a region on the Earth's surface
LIMITS = tuple[float, float, float, float]

# A time, specified by a year, month, day and hour. These may all be None to refer to periods of time
TIME = tuple[Optional[int], Optional[int], Optional[int], Optional[int]]

# Earth data, which is a grid of values for each component of data. Wind speed, for example, is a singular 2D grid,
# but wind vectors are a list of two such grids, one each for the horizontal and vertical components
GRID_DATA = tuple[ArrayLike] | tuple[ArrayLike, ArrayLike]



