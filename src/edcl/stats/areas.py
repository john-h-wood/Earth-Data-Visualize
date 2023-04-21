"""The areas module houses all functions for getting statistically defined areas PathCollections and information about
them, for example, their areas."""
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.path import Path
from numpy.typing import ArrayLike

from edcl.types import COORDINATES
from edcl import constants as cnst
from edcl.collections import VectorCollection, VirtualVectorCollection

# Were more specific typing available, this would be a two-dimensional np array of floats, representing a vale for
# each point on a grid. The first coordinate is latitude and the second in longitude.
SCALAR_GRID = ArrayLike


def contour_regions_vector_collection(vc: VectorCollection | VirtualVectorCollection, bottom_contour: float,
                                      top_contour: float) ->\
                                     tuple[tuple[tuple[Path, ...]], tuple[tuple[tuple[Path, ...]]]]:
    # With great thanks to https://stackoverflow.com/a/48695006. Answer by Thomas Kühn.
    # Validate parameters
    if vc.get_dimension() != 1:
        raise ValueError('Contour regions only available for one dimensional (scalar) vector collections.')
    if top_contour <= bottom_contour:
        raise ValueError('Top contour must be strictly greater than the bottom contour.')

    return_pluses = list()
    return_minuses = list()

    if isinstance(vc, VectorCollection):
        for time_idx in range(vc.get_time_length()):
            this_time_pluses = list()
            this_time_minuses = list()

            plt.close('all')
            fig, ax = plt.subplots()
            cs = ax.contourf(vc.longitude, vc.latitude, vc.get_time_data(time_idx)[0])
            paths = cs.collections[0].get_paths()

            for p in paths:
                this_path_minuses = list()
                sign = 1
                verts = p.vertices
                codes = p.codes
                idx = np.where(codes == Path.MOVETO)[0]
                vert_segs = np.split(verts, idx)[1:]
                code_segs = np.split(codes, idx)[1:]

                for code, vert in zip(code_segs, vert_segs):
                    path = Path(vert, code)
                    # Add first sub-path to pluses, and others to minuses
                    if sign == 1: this_time_pluses.append(path)
                    else:         this_path_minuses.append(path)
                    sign = -1

                # Add the minuses for each path to this_time_minuses
                this_time_minuses.append(tuple(this_path_minuses))

            return_pluses.append(tuple(this_time_pluses))
            return_minuses.append(tuple(this_time_minuses))

    return tuple(return_pluses), tuple(return_minuses)


def get_area_grid(latitude: COORDINATES, longitude: COORDINATES, lat_side: Optional[float],
                  lon_side: Optional[float]) -> SCALAR_GRID:
    # Validate parameters
    if len(latitude) < 2 or len(longitude) < 2:
        raise ValueError('The latitude and longitude vectors must each have at least two elements.')

    # Expand and find mid-points of latitude and longitude
    lat_mean_diff = np.mean(np.diff(latitude)) if lat_side is None else lat_side
    expanded_lat = np.insert(latitude, (0, len(latitude)), (latitude[0] - lat_mean_diff, latitude[-1] + lat_mean_diff))
    lat_mid = np.deg2rad(90 - (expanded_lat[:-1] + 0.5 * np.diff(expanded_lat)))

    lon_mean_diff = np.mean(np.diff(longitude)) if lon_side is None else lon_side
    expanded_lon = np.insert(longitude, (0, len(longitude)), (longitude[0] - lon_mean_diff, longitude[-1] +
                                                              lon_mean_diff))
    lon_mid = np.deg2rad(expanded_lon[:-1] + 0.5 * np.diff(expanded_lon))

    # Build up area grid from zeros, first doing longitude then latitude then r^2 contributions.
    area_grid = np.ones((len(latitude), len(longitude)))
    area_grid *= np.diff(lon_mid)
    cos_diff_lat_mid = np.cos(lat_mid[:-1]) - np.cos(lat_mid[1:])
    area_grid *= cos_diff_lat_mid[:, np.newaxis]
    area_grid *= (cnst.r_e_vol ** 2)

    return np.abs(area_grid)



