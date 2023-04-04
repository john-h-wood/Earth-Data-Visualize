"""
The visualize module houses all functions which put data into visual formats.
"""
from typing import Optional
from matplotlib.colors import CSS4_COLORS
import matplotlib.pyplot as plt

from .collections import DataCollection
from .types import PROJECTION, LIMITS
from .util import to_tuple, convertable_to_float, convertable_to_int
from .config import info


def plot_data_collections(data_collections: DataCollection | tuple[DataCollection, ...],
                          styles: str | tuple[str, ...], projection: PROJECTION, limits: Optional[LIMITS],
                          size: tuple[float, float], titles: Optional[str | tuple[str, ...]], font_size: int,
                          directory: str, path_names: Optional[str | tuple[str, ...]], out_mode: str) -> None:
    """
    Plot a series of DataCollections.

    Should each DataCollection have the same time length, one plot is generated for each time stamp.

    Supported styles:
    For points: [colour]_[marker]_[marker size] with each from matplotlib. For example: black_X_12
    For paths: [colour]_[alpha] with each from matplotlib. For example: red_0.5
    For scalars: heat_[cmap] with cmap from matplotlib. For example: heat_jet
                 contour_[integer number of levels] For example: contour_9
    For vectors: quiver

    The number of styles should equal the number of graphables.
    Each plots' limits, if not given, are the maximal limits from all DataCollections used.
    Should no tick markers be given, they are automatically found.
    Should titles be given, there should be as the time length of the graphables. If they are not given,
    automatic titles are generates using the time titles from the graphables.
    Should no save titles be given, they are automatically generated. Otherwise, there should be the same number as
    the time length of the graphables.

    Args:
        data_collections: A single or tuple of DataCollections to plot.
        styles: A single or list of styles.
        projection: The projection of the plots.
        limits: The limits of the plot. Possibly None.
        size: The size of the plot. Formatted as (horizontal, vertical) in inches.
        titles: The plot titles. Possibly None.
        font_size: The font size for the plots.
        directory:
        path_names: The path names for saving the plots. Possibly None.
        out_mode: The output mode for the plots.

    Returns:
        None. Note, each plot is processed through edcl.visualize.output_figure.
    """
    # Convert non-tuples parameters to tuples
    data_collections = to_tuple(data_collections)
    styles = to_tuple(styles)
    if titles is not None: titles = to_tuple(titles)
    if path_names is not None: path_names = to_tuple(path_names)

    # Validate parameters
    if len(data_collections) == 0: raise ValueError('At least one DataCollection must be passed to plot.')
    if len(data_collections) != len(styles): raise ValueError('There must be the same number of DataCollections and '
                                                              'styles.')
    if out_mode not in info.graph_out_modes: raise ValueError('The given plot output mode is not supported.')


def validate_style(style: str) -> None:
    """
    Validates a style. Raises relevant Errors if needed.
    Args:
        style: The style.

    Returns:
        None.
    """
    colours = CSS4_COLORS.keys()
    colour_maps = plt.colormaps()
    items = style.split('_')

    if items[0] not in info.graph_styles: raise ValueError(f'Unsupported style {style}.')

    elif items[0] == 'heat':
        if len(items) < 3:
            raise ValueError('Incorrect number of arguments for heat style.')
        if items[1] not in colour_maps:
            raise ValueError(f'Invalid colour map {items[1]}')
        if len(items) == 3 and items[2] != 'None':
            raise ValueError(f'Single tick argument must be None.')
        for tick in items[2:]:
            if not convertable_to_float(tick):
                raise ValueError(f'Invalid tick {tick}.')

    elif items[0] == 'contour':
        if len(items) != 3:
            raise ValueError('Incorrect number of arguments for contour style.')
        if items[1] not in colours:
            raise ValueError(f'Invalid colour {items[1]}.')
        if not convertable_to_int(items[2]) or int(items[2]) <= 0:
            raise ValueError(f'Invalid number of levels {items[2]}. Must be a positive integer.')

    elif items[0] == 'quiver':
        if len(items) != 3:
            raise ValueError('Incorrect number of arguments for quiver style.')
        if items[1] not in colours:
            raise ValueError(f'Invalid colour {items[1]}.')
        if items[2] != 'None' and (not convertable_to_int(items[2]) or int(items[2]) <= 0):
            raise ValueError(f'Invalid skip {items[2]}. Must be a positive integer or None.')



