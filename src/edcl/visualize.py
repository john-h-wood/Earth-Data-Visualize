"""
The visualize module houses all functions which put data into visual formats.
"""
import numpy as np
import os.path as osp
import cartopy.crs as ccrs
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import CSS4_COLORS
from matplotlib.markers import MarkerStyle

from .config import info
from .types import PROJECTION, LIMITS
from .formatting import tuple_to_string
from .util import to_tuple, convertable_to_float, convertable_to_int
from .collections import DataCollection, GridCollection, PathCollection, PointCollection

def plot_data_collections(data_collections: DataCollection | tuple[DataCollection, ...],
                          styles: str | tuple[str, ...], projection: PROJECTION, limits: Optional[LIMITS],
                          size: tuple[float, float], titles: Optional[str | tuple[str, ...]], font_size: int,
                          directory: Optional[str], path_names: Optional[str | tuple[str, ...]], out_mode: str,
                          dpi: Optional[float]) -> None:
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
        dpi: The dpi of the saved plot. Possibly None.

    Returns:
        None. Note, each plot is processed through edcl.visualize.output_figure.
    """
    # Convert non-tuples parameters to tuples
    data_collections = to_tuple(data_collections)
    styles = to_tuple(styles)
    if titles is not None: titles = to_tuple(titles)
    if path_names is not None: path_names = to_tuple(path_names)

    # Validate parameters
    if len(data_collections) == 0:
        raise ValueError('At least one DataCollection must be passed to plot.')
    if len(data_collections) != len(styles):
        raise ValueError('There must be the same number of data collections and styles.')
    if out_mode not in info.graph_out_modes:
        raise ValueError('The given plot output mode is not supported.')
    for data_collection, style in zip(data_collections, styles):
        validate_data_collection_style_pair(data_collection, style)
    if any(data_collections[i] != data_collections[i + 1] for i in range(len(data_collections) - 1)):
        raise ValueError('The DataCollections to plot do not have the same time length.')
    if directory is not None and not osp.isdir(directory):
        raise NotADirectoryError(f'The directory {directory} was not found.')
    # TODO Validate format of path_names if its given
    if titles is not None and len(titles) != len(data_collections):
        raise ValueError('If they are given, there must be the same number of titles as data collections.')
    if path_names is not None and len(path_names) != len(data_collections):
        raise ValueError('If they are given, there must be the same number of path names as data collections')
    if dpi is not None and dpi < 1:
        raise ValueError('Dpi must be greater than zero.')

    # Fill required none values
    if dpi is None:
        dpi = 100.

    # Matplotlib preferences
    plt.rcParams.update({'font.family': 'Times', 'font.size': font_size})

    # Pass through each time slice
    for time_index in range(data_collections[0].get_time_length()):
        print(f'Plotting figure {time_index + 1} of {data_collections[0].get_time_length()}')

        # Close all previous plots and initialize this one
        plt.close('all')
        fig = plt.figure(figsize=size, dpi=dpi)
        ax = plt.axes(projection=projection)

        # Format the plot (coastlines, labels, title)
        ax.coastlines(resolution='50m')
        if limits is not None: ax.set_extent([limits[2], limits[3], limits[0], limits[1]], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True)
        gl.right_labels = False
        gl.bottom_labels = False

        plt.title(titles[time_index] if titles is not None else generate_default_plot_title(data_collections, styles,
                                                                                            time_index))
        # Plot the data
        for dc, style in zip(data_collections, styles):
            # Interpret null entries in style items
            style_items = style.split('_')
            for i, item in enumerate(style_items):
                if item == 'None':
                    style_items[i] = None

            # Heatmap. Check for instance of GridCollection is to satisfy type inspection. This is validated before
            if style_items[0] == 'heat' and isinstance(dc, GridCollection):
                # Extract ticks
                if len(style_items) == 3:
                    ticks = None
                else:
                    ticks = [float(item) for item in style_items[2:]]
                if ticks is None:
                    vmin, vmax = None, None
                else:
                    vmin, vmax = ticks[0], ticks[-1]

                p = ax.pcolormesh(dc.longitude, dc.latitude, dc.get_time_data(time_index),
                                  transform=ccrs.PlateCarree(), cmap=style_items[1], shading='nearest', vmin=vmin,
                                  vmax=vmax)
                fig.colorbar(p, orientation='vertical', ticks=ticks)

            # Contour plot
            elif style_items[0] == 'contour' and isinstance(dc, GridCollection):
                if style_items[2] is not None:
                    levels = int(style_items[2])
                else:
                    levels = None

                cs = ax.contour(dc.longitude, dc.latitude, dc.get_time_data(time_index), transform=ccrs.PlateCarree(),
                                colors=style_items[1], levels=levels)
                ax.clabel(cs, inline=True, fontsize=int(style_items[3]))

            # Quiver plot
            elif style_items[0] == 'quiver' and isinstance(dc, GridCollection):
                skip = int(style_items[2])
                ax.quiver(dc.longitude[::skip], dc.latitude[::skip], dc.get_time_data(time_index)[0][::skip, ::skip],
                          dc.get_time_data(time_index)[1][::skip, ::skip], color=style_items[1],
                          transform=ccrs.PlateCarree())

            # Patch
            elif style_items[0] == 'patch' and isinstance(dc, PathCollection):
                patch = mpatches.PathPatch(dc.get_time_data(time_index), fc=style_items[1], ec=style_items[2],
                                           lw=style_items[3], alpha=style_items[4], transform=ccrs.PlateCarree())
                ax.add_patch(patch)

            # Point
            elif style_items[0] == 'point' and isinstance(dc, PointCollection):
                data = np.array(dc.get_time_data(time_index))
                latitude = data[:, 0]
                longitude=  data[:, 1]
                ax.scatter(longitude, latitude, s=int(style_items[3]), c=style_items[2], marker=style_items[1],
                           transform=ccrs.PlateCarree())










def validate_style(style: str) -> None:
    """
    Validates a style. Raises relevant Errors if needed.

    Args:
        style: The style.

    Returns:
        None.

    Raises:
        ValueError: Invalid number of arguments for heat style.
        ValueError: Invalid colour map.
        ValueError: Single tick argument must be None.
        ValueError: Invalid tick.
        ValueError: Invalid ticks in style. Ticks must have increasing magnitude.
        ValueError: Invalid number of arguments for contour style.
        ValueError: Invalid number of levels. Must be a positive integer or None.
        ValueError: Invalid label font size. Must be an integer than or equal to zero.
        ValueError: Invalid number of arguments for quiver style.
        ValueError: Invalid skip. Must be an integer greater than zero.
        ValueError: Invalid number of arguments for patch style.
        ValueError: Invalid line width. Must be a positive float.
        ValueError: Invalid colour. Can be None.
        ValueError: Invalid alpha. Must be a float between 0 and 1 inclusive.
        ValueError: Incorrect number of arguments for point style.
        ValueError: Invalid marker.
        ValueError: Invalid marker size. Must be a float greater than or equal to zero.
    """
    colours = CSS4_COLORS.keys()
    colour_maps = plt.colormaps()
    markers = MarkerStyle.markers.keys()
    items = style.split('_')

    if items[0] not in info.graph_styles: raise ValueError(f'Unsupported style {style}.')

    elif items[0] == 'heat':
        if len(items) < 3:
            raise ValueError('Invalid number of arguments for heat style.')
        if items[1] not in colour_maps:
            raise ValueError(f'Invalid colour map {items[1]}')
        if len(items) == 3 and items[2] != 'None':
            raise ValueError(f'Single tick argument must be None.')
        if len(items) > 3:
            for tick in items[2:]:
                if not convertable_to_float(tick):
                    raise ValueError(f'Invalid tick {tick}.')
            if not all(float(items[i]) < float(items[i + 1]) for i in range(2, len(items) - 1)):
                # Condition's code adapted from https://stackoverflow.com/a/3755251
                raise ValueError(f'Invalid ticks in style {style}. Ticks must have increasing magnitude.')

    elif items[0] == 'contour':
        if len(items) != 4:
            raise ValueError('Invalid number of arguments for contour style.')
        if items[1] not in colours:
            raise ValueError(f'Invalid colour {items[1]}.')
        if (not convertable_to_int(items[2]) or int(items[2]) <= 0) and (items[2] != 'None'):
            raise ValueError(f'Invalid number of levels {items[2]}. Must be a positive integer or None.')
        if not convertable_to_int(items[3]) or int(items[3]) < 0:
            raise ValueError(f'Invalid label font size {items[3]}. Must be an integer greater than or equal to zero.')

    elif items[0] == 'quiver':
        if len(items) != 3:
            raise ValueError('Invalid number of arguments for quiver style.')
        if items[1] not in colours:
            raise ValueError(f'Invalid colour {items[1]}.')
        if not convertable_to_int(items[2]) or int(items[2]) < 1:
            raise ValueError(f'Invalid skip {items[2]}. Must be an integer greater than or equal to 1.')

    elif items[0] == 'patch':
        if len(items) != 5:
            raise ValueError('Invalid number of arguments for patch style.')
        if items[1] not in colours and items[1] != 'None':
            raise ValueError(f'Invalid colour {items[1]}. Can be None.')
        if not items[2] in colours and items[2] != 'None':
            raise ValueError(f'Invalid colour {items[2]}. Can be None.')
        if not convertable_to_float(items[3]) or float(items[3]) < 0:
            raise ValueError(f'Invalid line width {items[3]}. Must be a positive float.')
        if not convertable_to_float(items[4]) or not (0 <= float(items[4]) <= 1):
            raise ValueError(f'Invalid alpha {items[4]}. Must be a float between 0 and 1 inclusive.')

    elif items[0] == 'point':
        if len(items) != 4:
            raise ValueError('Incorrect number of arguments for point style.')
        if items[1] not in markers:
            raise ValueError(f'Invalid marker {items[1]}.')
        if items[2] not in colours:
            raise ValueError(f'Invalid colour {items[2]}.')
        if not convertable_to_float(items[3]) or float(items[3]) < 0:
            raise ValueError(f'Invalid marker size {items[3]}. Must be a float greater than or equal to zero.')


def validate_data_collection_style_pair(data_collection: DataCollection, style: str) -> None:
    """
    Validates a DataCollection-style pair. Check if the data collection can sensibly be plotted in the given style.

    Raises relevant error if needed.

    Args:
        data_collection: The DataCollection.
        style: The style.

    Returns:
        None.

    Raises:
        ValueError: Heat and contour style is only valid for one-dimensional GridCollections.
        ValueError: Quiver style is only valid for two-dimensional GridCollections.
        ValueError: Path style is only valid for PathCollections.
        ValueError: Point style is only valid for PointCollections.
        ValueError: The style family is not supported.
    """
    # Validate parameters
    validate_style(style)

    style = style[:style.index('_')]
    if style == 'heat' or style == 'contour':
        if not (isinstance(data_collection, GridCollection) and data_collection.dimension == 1):
            raise ValueError('Heat and contour styles are only valid for one-dimensional GridCollections.')
    elif style == 'quiver':
        if not (isinstance(data_collection, GridCollection) and data_collection.dimension == 2):
            raise ValueError('Quiver style is only valid for two-dimensional GridCollections.')
    elif style == 'patch':
        if not isinstance(data_collection, PathCollection):
            raise ValueError('Path style is only valid for PathCollections.')
    elif style == 'point':
        if not isinstance(data_collection, PointCollection):
            raise ValueError('Point style is only valid for PointCollections.')


def generate_default_plot_title(data_collections: tuple[DataCollection], styles: tuple[str], time_index: int) -> str:
    sub_titles = list()
    for data_collection, style in zip(data_collections, styles):
        working_title = ''
        style_items = style.split('_')

        working_title += data_collection.get_time_title(time_index)

        if style_items[0] == 'heat':
            working_title += f' ({style_items[1]} heatmap)'
        elif style_items[0] == 'contour':
            working_title += f' ({style_items[1]} contours)'
        elif style_items[0] == 'quiver':
            working_title += f' ({style_items[1]} quiver, skip {style_items[2]})'
        elif style_items[0] == 'patch':
            working_title += ' ('
            if style_items[1] == 'None' and style_items[2] == 'None':
                working_title += 'invisible patch'
            elif style_items[1] == style_items[2]:
                working_title += f'{style_items[1]} patch'
            else:
                if style_items[1] != 'None':
                    working_title += f'{style_items[1]} fill'
                if style_items[2] != 'None':
                    if style_items[1] != 'None':
                        working_title += f' {style_items[2]} edge'
                    else:
                        working_title += f'{style_items[2]} edge'
            working_title += ')'
        elif style_items[0] == 'point':
            working_title += f' ({style_items[2]} {style_items[1]})'

        sub_titles.append(working_title)

    return tuple_to_string(tuple(sub_titles))
