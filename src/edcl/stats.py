"""
The stats module houses all functions relating to statistics on DataCollections.
"""
import numpy as np

from .collections import VectorCollection, VirtualVectorCollection, PointCollection


def time_average_vector_collection(vc: VectorCollection | VirtualVectorCollection,
                                   ignore_nan: bool) -> VectorCollection:
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
# ARG MAX (Position of the maximum)
# ======================================================================================================================
def argmax_in_time_vector_collection(vc: VectorCollection | VirtualVectorCollection, ignore_nan: bool) -> \
                                     PointCollection:
    # Validate parameters
    if vc.get_dimension() != 1:
        raise ValueError('Maximum computation is only allowed for one-dimensional (scalar) vector collections.')

    maxima = list()
    if isinstance(vc, VectorCollection):
        if ignore_nan:
            for grid in vc.get_all_data():
                # Index grid at zero to get rid of component axis. This leaves grid[0] as two-dimensional
                lat_idx, lon_idx = np.unravel_index(np.nanargmax(grid[0]), np.shape(grid[0]))
                maxima.append((vc.latitude[lat_idx], vc.longitude[lon_idx]))
        else:
            for grid in vc.get_all_data():
                # Index grid at zero to get rid of component axis. This leaves grid[0] as two-dimensional
                lat_idx, lon_idx = np.unravel_index(np.argmax(grid[0]), np.shape(grid[0]))
                maxima.append((vc.latitude[lat_idx], vc.longitude[lon_idx]))
    else:
        if ignore_nan:
            for block in vc.get_all_data_iter():
                for grid in block:
                    # Index grid at zero to get rid of component axis. This leaves grid[0] as two-dimensional
                    lat_idx, lon_idx = np.unravel_index(np.nanargmax(grid[0]), np.shape(grid[0]))
                    maxima.append((vc.latitude[lat_idx], vc.longitude[lon_idx]))
        else:
            for block in vc.get_all_data_iter():
                for grid in block:
                    # Index grid at zero to get rid of component axis. This leaves grid[0] as two-dimensional
                    lat_idx, lon_idx = np.unravel_index(np.nanargmax(grid[0]), np.shape(grid[0]))
                    maxima.append((vc.latitude[lat_idx], vc.longitude[lon_idx]))

    data = tuple(maxima)
    if ignore_nan:
        title_prefix = 'Timed Nan-Max Loc. of ' + vc.title_prefix
    else:
        title_prefix = 'Timed Max Loc. of ' + vc.title_prefix

    return PointCollection(vc.dataset, vc.variable, vc.time, vc.time_stamps, title_prefix, '', data)


def argmax_in_space_time_vector_collection(vc: VectorCollection | VirtualVectorCollection, ignore_nan: bool) -> \
        None:
    # Validate parameter
    if vc.get_dimension() != 1:
        raise ValueError('Maximum computation is only allowed for one-dimensional (scalar) vector collections.')

    if isinstance(vc, VectorCollection):
        if ignore_nan:
            data = vc.get_all_data()[:, 0, :, :]
            lat_idx, lon_idx = np.unravel_index(np.nanargmax(data), np.shape(data))

