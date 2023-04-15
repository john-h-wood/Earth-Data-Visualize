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
def arg_max_in_space_vector_collection(vc: VectorCollection | VirtualVectorCollection,
                                       ignore_nan: bool = True) -> PointCollection:
    # Validate parameters
    if vc.get_dimension() != 1:
        raise ValueError('Extremum computation is only allowed for one-dimensional (scalar) vector collections.')

    arg_max = list()
    if isinstance(vc, VectorCollection):
        if ignore_nan:
            for grid in vc.get_all_data():
                # Index grid at zero to get rid of component axis. This leaves grid[0] as two-dimensional, lat, lon
                lat_idx, lon_idx = np.unravel_index(np.nanargmax(grid[0]), np.shape(grid[0]))
                arg_max.append((vc.latitude[lat_idx], vc.longitude[lon_idx]))
        else:
            for grid in vc.get_all_data():
                lat_idx, lon_idx = np.unravel_index(np.argmax(grid[0]), np.shape(grid[0]))
                arg_max.append((vc.latitude[lat_idx], vc.longitude[lon_idx]))

    else:
        for grid_in_time in vc.get_all_data_iter():
            if ignore_nan:
                for grid in grid_in_time:
                    # Index grid at zero to get rid of component axis. This leaves grid[0] as two-dimensional, lat, lon
                    lat_idx, lon_idx = np.unravel_index(np.nanargmax(grid[0]), np.shape(grid[0]))
                    arg_max.append((vc.latitude[lat_idx], vc.longitude[lon_idx]))
            else:
                for grid in grid_in_time:
                    lat_idx, lon_idx = np.unravel_index(np.argmax(grid[0]), np.shape(grid[0]))
                    arg_max.append((vc.latitude[lat_idx], vc.longitude[lon_idx]))

    arg_max = tuple(arg_max)

    if ignore_nan:
        title_prefix = f'Timed Ignore-Nan Max Loc. of ' + vc.title_prefix
    else:
        title_prefix = f'Timed Max Loc. of ' + vc.title_prefix

    return PointCollection(vc.dataset, vc.variable, vc.time, vc.time_stamps, title_prefix, '', arg_max)


def arg_min_in_space_vector_collection(vc: VectorCollection | VirtualVectorCollection,
                                       ignore_nan: bool = True) -> PointCollection:
    # Validate parameters
    if vc.get_dimension() != 1:
        raise ValueError('Extremum computation is only allowed for one-dimensional (scalar) vector collections.')

    arg_min = list()
    if isinstance(vc, VectorCollection):
        if ignore_nan:
            for grid in vc.get_all_data():
                # Index grid at zero to get rid of component axis. This leaves grid[0] as two-dimensional, lat, lon
                lat_idx, lon_idx = np.unravel_index(np.nanargmin(grid[0]), np.shape(grid[0]))
                arg_min.append((vc.latitude[lat_idx], vc.longitude[lon_idx]))
        else:
            for grid in vc.get_all_data():
                lat_idx, lon_idx = np.unravel_index(np.argmin(grid[0]), np.shape(grid[0]))
                arg_min.append((vc.latitude[lat_idx], vc.longitude[lon_idx]))

    else:
        for grid_in_time in vc.get_all_data_iter():
            if ignore_nan:
                for grid in grid_in_time:
                    # Index grid at zero to get rid of component axis. This leaves grid[0] as two-dimensional, lat, lon
                    lat_idx, lon_idx = np.unravel_index(np.nanargmin(grid[0]), np.shape(grid[0]))
                    arg_min.append((vc.latitude[lat_idx], vc.longitude[lon_idx]))
            else:
                for grid in grid_in_time:
                    lat_idx, lon_idx = np.unravel_index(np.argmin(grid[0]), np.shape(grid[0]))
                    arg_min.append((vc.latitude[lat_idx], vc.longitude[lon_idx]))

    arg_min = tuple(arg_min)

    if ignore_nan:
        title_prefix = f'Timed Ignore-Nan Min Loc. of ' + vc.title_prefix
    else:
        title_prefix = f'Timed Min Loc. of ' + vc.title_prefix

    return PointCollection(vc.dataset, vc.variable, vc.time, vc.time_stamps, title_prefix, '', arg_min)


def arg_max_in_space_time_vector_collection(vc: VectorCollection | VirtualVectorCollection,
                                            ignore_nan: bool = True) -> PointCollection:
    # Validate parameter
    if vc.get_dimension() != 1:
        raise ValueError('Maximum computation is only allowed for one-dimensional (scalar) vector collections.')

    if isinstance(vc, VectorCollection):
        data = vc.get_all_data()[:, 0, :, :]
        if ignore_nan:
            lat_idx, lon_idx = np.unravel_index(np.nanargmax(data), np.shape(data))[1:]
        else:
            lat_idx, lon_idx = np.unravel_index(np.argmax(data), np.shape(data))[1:]
    else:
        current_max = None
        lat_idx = None
        lon_idx = None
        for grid_in_time in vc.get_all_data_iter():
            data = grid_in_time[:, 0, :, :]
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
        title_prefix = f'Ignore-Nan Max Loc. of {vc}'
    else:
        title_prefix = f'Max Loc. of {vc}'

    return PointCollection(vc.dataset, vc.variable, vc.time, (vc.time,), title_prefix, '', point_data)


def arg_min_in_space_time_vector_collection(vc: VectorCollection | VirtualVectorCollection,
                                            ignore_nan: bool = True) -> PointCollection:
    # Validate parameter
    if vc.get_dimension() != 1:
        raise ValueError('Maximum computation is only allowed for one-dimensional (scalar) vector collections.')

    if isinstance(vc, VectorCollection):
        data = vc.get_all_data()[:, 0, :, :]
        if ignore_nan:
            lat_idx, lon_idx = np.unravel_index(np.nanargmin(data), np.shape(data))[1:]
        else:
            lat_idx, lon_idx = np.unravel_index(np.argmin(data), np.shape(data))[1:]
    else:
        current_min = None
        lat_idx = None
        lon_idx = None
        for grid_in_time in vc.get_all_data_iter():
            data = grid_in_time[:, 0, :, :]
            # Get this grid_in_time's min info
            if ignore_nan:
                time_idx, this_lat_idx, this_lon_idx = np.unravel_index(np.nanargmin(data), np.shape(data))
            else:
                time_idx, this_lat_idx, this_lon_idx = np.unravel_index(np.argmin(data), np.shape(data))
            this_min = data[time_idx, this_lat_idx, this_lon_idx]

            # If the min for this grid in time is bigger, make it the working min
            if (current_min is None) or (this_min < current_min):
                current_min = this_min
                lat_idx = this_lat_idx
                lon_idx = this_lon_idx

    point_data = ((vc.latitude[lat_idx], vc.longitude[lon_idx]),)

    if ignore_nan:
        title_prefix = f'Ignore-Nan Min Loc. of {vc}'
    else:
        title_prefix = f'Min Loc. of {vc}'

    return PointCollection(vc.dataset, vc.variable, vc.time, (vc.time,), title_prefix, '', point_data)


# ======================================================================================================================
# EXTREMUM (Value of the maximum / minimum)
# ======================================================================================================================
def max_in_time_vector_collection(vc: VectorCollection | VirtualVectorCollection,
                                  ignore_nan: bool = True) -> VectorCollection:
    if isinstance(vc, VectorCollection):
        if ignore_nan:
            data = np.nanmax(vc.get_all_data(), axis=0)
        else:
            data = np.max(vc.get_all_data(), axis=0)
    else:
        data = None
        if ignore_nan:
            for grid_in_time in vc.get_all_data_iter():
                if data is None:
                    data = np.nanmax(grid_in_time, axis=0)
                else:
                    this_max = np.nanmax(grid_in_time, axis=0)
                    data = np.nanmax(np.stack((data, this_max), axis=0), axis=0)
        else:
            for grid_in_time in vc.get_all_data_iter():
                if data is None:
                    data = np.max(grid_in_time, axis=0)
                else:
                    this_max = np.max(grid_in_time, axis=0)
                    data = np.max(np.stack((data, this_max), axis=0), axis=0)
    data = np.expand_dims(data, axis=0)

    if ignore_nan:
        title_prefix = f'Point-wise ignore-Nan Max of {vc} '
    else:
        title_prefix = f'Point-wise Max of {vc} '

    return VectorCollection(vc.dataset, vc.variable, vc.time, (vc.time,), title_prefix, '', data, vc.latitude,
                            vc.longitude)


def min_in_time_vector_collection(vc: VectorCollection | VirtualVectorCollection,
                                  ignore_nan: bool = True) -> VectorCollection:
    if isinstance(vc, VectorCollection):
        if ignore_nan:
            data = np.nanmin(vc.get_all_data(), axis=0)
        else:
            data = np.min(vc.get_all_data(), axis=0)
    else:
        data = None
        if ignore_nan:
            for grid_in_time in vc.get_all_data_iter():
                if data is None:
                    data = np.nanmin(grid_in_time, axis=0)
                else:
                    this_min = np.nanmin(grid_in_time, axis=0)
                    data = np.nanmin(np.stack((data, this_min), axis=0), axis=0)
        else:
            for grid_in_time in vc.get_all_data_iter():
                if data is None:
                    data = np.min(grid_in_time, axis=0)
                else:
                    this_min = np.min(grid_in_time, axis=0)
                    data = np.min(np.stack((data, this_min), axis=0), axis=0)
    data = np.expand_dims(data, axis=0)

    if ignore_nan:
        title_prefix = f'Point-wise ignore-Nan Min of {vc} '
    else:
        title_prefix = f'Point-wise Min of {vc} '

    return VectorCollection(vc.dataset, vc.variable, vc.time, (vc.time,), title_prefix, '', data, vc.latitude,
                            vc.longitude)


