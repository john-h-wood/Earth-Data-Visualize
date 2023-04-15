"""
The stats module houses all functions relating to statistics on DataCollections.
"""
import numpy as np

from .types import VECTOR_GRID_IN_TIME, POINT
from .collections import VectorCollection, VirtualVectorCollection, PointCollection


def time_average_vector_collection(vc: VectorCollection | VirtualVectorCollection,
                                   ignore_nan: bool = True) -> VectorCollection:
    if isinstance(vc, VectorCollection):
        if ignore_nan:
            av_data = np.expand_dims(np.nanmean(vc.get_all_data(), axis=0), axis=0)
        else:
            av_data = np.expand_dims(np.mean(vc.get_all_data(), axis=0), axis=0)
    else:
        cum_sum = np.zeros((vc.get_dimension(), len(vc.latitude), len(vc.longitude)))
        if ignore_nan:
            quan = np.zeros((vc.get_dimension(), len(vc.latitude), len(vc.longitude)))
            for block in vc.get_all_data_iter():
                cum_sum += np.nansum(block, axis=0)
                quan += np.sum(np.logical_not(np.isnan(block)), axis=0)
            av_data = np.expand_dims(cum_sum / quan, axis=0)
        else:
            quan = 0
            for block in vc.get_all_data_iter():
                cum_sum += np.sum(block, axis=0)
                quan += len(block)
            av_data = np.expand_dims(cum_sum / quan, axis=0)

    if ignore_nan:
        title_prefix = 'Time Nan-Averaged ' + vc.title_prefix
    else:
        title_prefix = 'Time Averaged ' + vc.title_prefix

    return VectorCollection(vc.dataset, vc.variable, vc.time, (vc.time,), title_prefix, vc.title_suffix, av_data,
                            vc.latitude, vc.longitude)


# ======================================================================================================================
# ARG EXTREMUM (Position of the maximum / minimum)
# ======================================================================================================================
def arg_extremum_in_space_grid_in_time(vc: VectorCollection | VirtualVectorCollection,
                                       grid_in_time: VECTOR_GRID_IN_TIME, ignore_nan: bool, multiplier: int) -> \
                                      tuple[POINT]:
    arg_extrema = list()
    if ignore_nan:
        for grid in grid_in_time:
            # Index grid at zero to get rid of component axis. This leaves grid[0] as two-dimensional, lat, lon
            lat_idx, lon_idx = np.unravel_index(np.nanargmax(multiplier * grid[0]), np.shape(grid[0]))
            arg_extrema.append((vc.latitude[lat_idx], vc.longitude[lon_idx]))
    else:
        for grid in grid_in_time:
            # Index grid at zero to get rid of component axis. This leaves grid[0] as two-dimensional, lat, lon
            lat_idx, lon_idx = np.unravel_index(np.argmax(multiplier * grid[0]), np.shape(grid[0]))
            arg_extrema.append((vc.latitude[lat_idx], vc.longitude[lon_idx]))
    return tuple(arg_extrema)


def arg_extremum_in_space_vector_collection(vc: VectorCollection | VirtualVectorCollection, ignore_nan: bool,
                                            multiplier: int, title: str) -> PointCollection:
    # Validate parameters
    if vc.get_dimension() != 1:
        raise ValueError('Extremum computation is only allowed for one-dimensional (scalar) vector collections.')

    if isinstance(vc, VectorCollection):
        arg_extrema = arg_extremum_in_space_grid_in_time(vc, vc.get_all_data(), ignore_nan, multiplier)
    else:
        arg_extrema = list()
        for grid_in_time in vc.get_all_data_iter():
            arg_extrema.extend(arg_extremum_in_space_grid_in_time(vc, grid_in_time, ignore_nan, multiplier))
        arg_extrema = tuple(arg_extrema)

    if ignore_nan:
        title_prefix = f'Timed Ignore Nan-{title} Loc. of ' + vc.title_prefix
    else:
        title_prefix = f'Timed {title} Loc. of ' + vc.title_prefix

    return PointCollection(vc.dataset, vc.variable, vc.time, vc.time_stamps, title_prefix, '', arg_extrema)


def arg_extremum_in_space_time_vector_collection(vc: VectorCollection | VirtualVectorCollection, ignore_nan: bool,
                                                 multiplier: int, title: str) -> PointCollection:
    # Validate parameter
    if vc.get_dimension() != 1:
        raise ValueError('Maximum computation is only allowed for one-dimensional (scalar) vector collections.')

    if isinstance(vc, VectorCollection):
        data = multiplier * vc.get_all_data()[:, 0, :, :]
        if ignore_nan:
            lat_idx, lon_idx = np.unravel_index(np.nanargmax(data), np.shape(data))[1:]
        else:
            lat_idx, lon_idx = np.unravel_index(np.argmax(data), np.shape(data))[1:]
    else:
        current_max = None
        lat_idx = None
        lon_idx = None
        for grid_in_time in vc.get_all_data_iter():
            data = multiplier * grid_in_time[:, 0, :, :]
            # Get this grid_in_time's max info
            if ignore_nan:
                time_idx, this_lat_idx, this_lon_idx = np.unravel_index(np.nanargmax(data), np.shape(data))
            else:
                time_idx, this_lat_idx, this_lon_idx = np.unravel_index(np.argmax(data), np.shape(data))
            this_max = data[time_idx, this_lat_idx, this_lon_idx]

            # If the max for this grid in time is bigger, make it the working max
            if (current_max is None) or (this_max > current_max):
                current_max = this_max
                lat_idx = this_lat_idx
                lon_idx = this_lon_idx

    point_data = ((vc.latitude[lat_idx], vc.longitude[lon_idx]),)

    if ignore_nan:
        title_prefix = f'Ignore Nan-{title} Loc. of ' + vc.title_prefix
    else:
        title_prefix = f'{title} Loc. of ' + vc.title_prefix

    return PointCollection(vc.dataset, vc.variable, vc.time, (vc.time,), title_prefix, '', point_data)


def arg_max_in_space_vector_collection(vc: VectorCollection | VirtualVectorCollection, ignore_nan: bool = True):
    return arg_extremum_in_space_vector_collection(vc, ignore_nan, 1, 'Max')


def arg_min_in_space_vector_collection(vc: VectorCollection | VirtualVectorCollection, ignore_nan: bool = True):
    return arg_extremum_in_space_vector_collection(vc, ignore_nan, -1, 'Min')


def arg_max_in_space_time_vector_collection(vc: VectorCollection | VirtualVectorCollection, ignore_nan: bool = True):
    return arg_extremum_in_space_time_vector_collection(vc, ignore_nan, 1, 'Max')


def arg_min_in_space_time_vector_collection(vc: VectorCollection | VirtualVectorCollection, ignore_nan: bool = True):
    return arg_extremum_in_space_time_vector_collection(vc, ignore_nan, -1, 'Min')




