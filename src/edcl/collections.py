"""
The ollections module house Collection classes, which define objects with data and metadata for a variable at a
given location and over a given time period.
"""

from abc import ABC, abstractmethod

from . import maximal_limits
from .info_classes import Dataset, Variable
from .formatting import time_to_suffix
from .types import *
import numpy as np


class Collection(ABC):
    """
    The parent class for all collections of data at a given location and over a given time period. Includes metadata.
    """
    def __init__(self, dataset: Dataset, variable: Variable, time: TIME, time_stamps: TIME_STAMPS, title_prefix: str,
                 title_suffix: str, data_in_time: DATA_IN_TIME):
        """
        Constructor method.
        Args:
            dataset: The dataset.
            variable: The variable.
            time: The time period over which the data spans.
            time_stamps: List of times, one for each time slice of data, which describe a time for each slice.
            title_prefix: Prefix for describing each time slice of data.
            title_suffix: Suffix for describing each time slice of data.
            data_in_time: The data (over time).
        """
        self.dataset = dataset
        self.variable = variable
        self.time = time
        self.time_stamps = time_stamps
        self.title_prefix = title_prefix
        self.title_suffix = title_suffix
        self.data_in_time = data_in_time

        # Parameter validation
        if len(time_stamps) != len(data_in_time): raise ValueError('The lengths of the Collection time stamps and its'
                                                                   'data in time must be equal.')

    def __str__(self):
        return f'{self.title_prefix}{time_to_suffix(self.time)}{self.title_suffix}'

    def get_time_title(self, time_index) -> str:
        """
        The title of the Collection at a certain time.
        Args:
            time_index: The time index.

        Returns:
            The title.
        """
        return f'{self.title_prefix}{time_to_suffix(self.time_stamps[time_index])}{self.title_suffix}'

    def get_time_length(self) -> int:
        """
        The length of the Collection's data in time.

        Returns:
            The time length.
        """
        return len(self.data_in_time)

    @abstractmethod
    def get_time_data(self, time_index: int) -> DATA:
        ...

    @abstractmethod
    def get_limits(self) -> LIMITS:
        ...


class GridCollection(Collection):
    """
    Class storing all information for Earth data, including its dataset, variable, time, etc. This might,
    for example, store wind speed over time for a square region on the Earth.
    """
    def __init__(self, dataset: Dataset, variable: Variable, time: TIME, time_stamps: TIME_STAMPS, title_prefix: str,
                 title_suffix: str, data_in_time: GRID_IN_TIME, latitude: COORDINATES, longitude: COORDINATES,
                 dimension: int):
        super().__init__(dataset, variable, time, time_stamps, title_prefix, title_suffix, data_in_time)
        self.latitude = latitude
        self.longitude = longitude
        self.dimension = dimension

    def get_time_data(self, time_index: int) -> GRID:
        return self.data_in_time[time_index]

    def get_limits(self) -> LIMITS:
        """
        The spatial limits of the data. These are the maximal limits over time.

        Returns:
            The limits.
        """
        return np.min(self.latitude), np.max(self.latitude), np.min(self.longitude), np.max(self.longitude)


class PointCollection(Collection):
    """
    Class storing all information for a point, including dataset, variable, time, position over time, etc.
    """
    def __init__(self, dataset: Dataset, variable: Variable, time: TIME, time_stamps: TIME_STAMPS, title_prefix: str,
                 title_suffix: str, data_in_time: POINT_IN_TIME):
        super().__init__(dataset, variable, time, time_stamps, title_prefix, title_suffix, data_in_time)

    def get_time_data(self, time_index: int) -> POINT:
        return self.data_in_time[time_index]

    def get_limits(self) -> LIMITS:
        """
        The spatial limits of the point. These are the maximal limits over time.

        Returns:
            The limits.
        """
        latitudes, longitudes = np.transpose(np.array(self.data_in_time))
        return np.min(latitudes), np.max(latitudes), np.min(longitudes), np.max(longitudes)


class PathCollection(Collection):
    """
    Class storing all information for a path, including dataset, variable, time, path points over time, etc.
    """
    def __init__(self, dataset: Dataset, variable: Variable, time: TIME, time_stamps: TIME_STAMPS, title_prefix: str,
                 title_suffix: str, data_in_time: POINT_IN_TIME):
        super().__init__(dataset, variable, time, time_stamps, title_prefix, title_suffix, data_in_time)

    def get_time_data(self, time_index: int) -> PATH:
        return self.data_in_time[time_index]

    def get_limits(self) -> LIMITS:
        """
        The spatial limits of the point. These are the maximal limits over time.

        Returns:
            The limits.
        """
        limits = list()
        for path in self.data_in_time:
            ex = path.get_extents()
            limits.append((ex.ymin, ex.ymax, ex.xmin, ex.xmax))

        return maximal_limits(tuple(limits))

