"""
The edcl_collections module house Collection classes, which define objects with data and metadata for a variable at a
given location and given time. The Graphable abstract base class is also here, which defines the common attributes
the Collection classes have which enable them to be plotted.
"""

from abc import ABC, abstractmethod
from .edcl_info_classes import Dataset, Variable
from .edcl_formatting import time_to_suffix
from .edcl_types import *


class Collection(ABC):
    def __init__(self, dataset: Dataset, variable: Variable, time: TIME, time_stamps: TIME_STAMPS, title_prefix: str,
                 title_suffix: str, data_in_time: DATA_IN_TIME):
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

    def get_time_data(self, time_index) -> DATA:
        return self.data_in_time[time_index]

    @abstractmethod
    def get_limits(self) -> LIMITS:
        ...




