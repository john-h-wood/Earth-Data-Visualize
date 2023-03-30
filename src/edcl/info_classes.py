"""
The edcl_info_classes module houses classes which bring the information from info.json into objects. This information is
about available data, for example the names of datasets and their corresponding variables.
"""

from typing import Optional

class Variable:
    """
    Class representing a variable and its metadata, including type, name, etc.
    """

    def __init__(self, name: str, kind: str, is_combo: bool, identifier: int, key: Optional[str],
                 equation: Optional[str], file_identifier: Optional[str]) -> None:
        """
        Constructor method.
        Args:
            name: The name of the variable.
            kind: String specifying what kind of data the variable is expressed in (e.g. 'scalar' or 'vector').
            is_combo: Whether the variable's data is not directly available, and must insteasd be calculated from
                      other variables.
            identifier: Integer key to identify the variable, which is unique up to the dataset.
            key: The key at which the variable's data may be found in a MATLAB file.
            equation: If is_combo is True, the equation for calculating the variable's data, including the
                      identifiers of other variables needed for this calculation.
            file_identifier: Substring use to denote the variable in path of a MATLAB file.
        """
        self.name = name
        self.kind = kind
        self.is_combo = is_combo
        self.identifier = identifier
        self.key = key
        self.equation = equation
        self.file_identifier = file_identifier

        # Parameter validation. The nullity of key, equation, and file_identifier depend on is_combo
        if is_combo:
            if key is not None: raise ValueError('The key for a combo variable should be None.')
            if file_identifier is not None: raise ValueError('The file_identifier for a combo variable should be None.')
            if equation is None: raise ValueError('The equation for a combo variable should not be None.')
            # TODO more advanced validation for equations
        else:
            if key is None: raise ValueError('The key for a non-combo variable should not be None.')
            if file_identifier is None: raise ValueError('The file identifier for a non-combo variable should not be '
                                                         'None.')
            if equation is not None: raise ValueError('The equation for a non-combo variable should be None.')

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_json(cls, data):
        """
        Converts dictionary from JSON read to object.

        Args:
            data: The dictionary.

        Returns:
            The object.
        """
        return cls(**data)


class Dataset:
    """
    Class representing a dataset and its metadata, including file location and variables.
    """

    def __init__(self, directory: str, name: str, is_unified: bool, file_prefix: str, file_suffix: str,
                 variables: tuple[Variable]) -> None:
        """
        Constructor method.
        Args:
            directory: The directory within the main data directory in which files for the dataset are found.
            name: The dataset's name.
            is_unified: Whether all variables for a given time are stored in a single file. Otherwise, data is stored on
                        a per-variable and per-time basis.
            file_prefix: A prefix substring present in paths for files for the dataset.
            file_suffix: A suffix substring present in paths for files for the dataset.
            variables: A tuple of available variables in the dataset.
        """
        self.directory = directory
        self.name = name
        self.is_unified = is_unified
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.variables = variables

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_json(cls, data):
        """
        Converts dictionary from JSON read to object.
        Args:
            data: The dictionary.

        Returns:
            The object.
        """
        variables = tuple(map(Variable.from_json, data['variables']))
        return cls(data['directory'], data['name'], data['is_unified'], data['file_prefix'], data['file_suffix'],
                   variables)


class Info:
    """
    Class storing all information for available data and tools, including file location, and a list of all datasets.
    Includes visualisation methods such as projections and figure output modes.
    """

    def __init__(self, directory: str, projections: tuple[str], graph_styles: tuple[str], graph_out_modes: tuple[str],
                 datasets: tuple[Dataset]) -> None:
        """
        Constructor method.
        Args:
            directory: The main data directory.
            projections: Available projections for plotting.
            graph_styles: Available styles for plotting data.
            graph_out_modes: Available methods of output for graphs.
            datasets: Available datasets.
        """
        self.directory = directory
        self.projections = projections
        self.graph_styles = graph_styles
        self.graph_out_modes = graph_out_modes
        self.datasets = datasets

    def __str__(self) -> str:
        return f'Info object of directory \'{self.directory}\''

    @classmethod
    def from_json(cls, data):
        """
        Converts dictionary from JSON read to object.
        Args:
            data: The dictionary.

        Returns:
            The object.
        """
        datasets = tuple(map(Dataset.from_json, data['datasets']))
        return cls(data['directory'], tuple(data['projections']), tuple(data['graph_styles']),
                   tuple(data['graph_out_modes']), datasets)
