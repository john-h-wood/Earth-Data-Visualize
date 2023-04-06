"""
The stats module houses all functions relating to statistics on DataCollections.
"""
import numpy as np

from .collections import VectorCollection, VirtualVectorCollection

def time_average_vector_collection(vector_collection: VectorCollection | VirtualVectorCollection) -> VectorCollection:
    if isinstance(vector_collection, VectorCollection):
        av_data = np.mean()