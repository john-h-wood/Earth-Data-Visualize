"""
The stats module houses all functions relating to statistics on DataCollections.
"""
import numpy as np

from .collections import VectorCollection, VirtualVectorCollection


def time_average_vector_collection(vc: VectorCollection | VirtualVectorCollection,
                                   ignore_nan: bool) -> VectorCollection:
    if isinstance(vc, VectorCollection):
        if ignore_nan:
            av_data = np.expand_dims(np.nanmean(vc.get_all_data(), axis=0), axis=0)
        else:
            av_data = np.expand_dims(np.mean(vc.get_all_data(), axis=0), axis=0)
    else:
        cumsum = np.zeros((vc.get_dimension(), len(vc.latitude), len(vc.longitude)))
        if ignore_nan:
            quan = np.zeros((vc.get_dimension(), len(vc.latitude), len(vc.longitude)))
            for block in vc.get_all_data_iter():
                cumsum += np.nansum(block, axis=0)
                quan += np.sum(np.logical_not(np.isnan(block)), axis=0)
            av_data = np.expand_dims(cumsum / quan, axis=0)
        else:
            quan = 0
            for block in vc.get_all_data_iter():
                cumsum += np.sum(block, axis=0)
                quan += len(block)
            av_data = np.expand_dims(cumsum / quan, axis=0)

    if ignore_nan:
        title_prefix = 'Time Nan-Averaged ' + vc.title_prefix
    else:
        title_prefix = 'Time Averaged ' + vc.title_prefix

    return VectorCollection(vc.dataset, vc.variable, vc.time, (vc.time,), title_prefix, vc.title_suffix, av_data,
                            vc.latitude, vc.longitude)
