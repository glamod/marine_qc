"""Some generally helpful location control functions for base QC."""

from __future__ import annotations
from typing import overload

import numpy as np

from .auxiliary import is_scalar_like


__all__ = [
    "mds_lat_to_yindex",
    "mds_lon_to_xindex",
]


@overload
def mds_lat_to_yindex(lat: float, res: float) -> int: ...  # numpydoc ignore=GL08
@overload
def mds_lat_to_yindex(lat: np.ndarray, res: float) -> np.ndarray: ...  # numpydoc ignore=GL08
def mds_lat_to_yindex(lat: float | np.ndarray, res: float) -> int | np.ndarray:
    """
    For a given latitude return the y-index as it was in MDS2/3 in a 1x1 global grid.

    Parameters
    ----------
    lat : numpy.ndarray
        Latitude(s) of observation in degrees.
    res : float
        Resolution of grid in degrees.

    Returns
    -------
    int or numpy.ndarray
        Grid box index(es).

    Notes
    -----
    In the northern hemisphere, borderline latitudes which fall on grid boundaries are pushed north, except
    90 which goes south. In the southern hemisphere, they are pushed south, except -90 which goes north.
    At 0 degrees they are pushed south.

    Expects that latitudes run from 90N to 90S

    In previous versions, ``res`` had the default value 1.0.
    """
    lat_arr = np.atleast_1d(lat)
    lat_local = lat_arr
    lat_local[lat_local == -90] = lat_local[lat_local == -90] + 0.001
    lat_local[lat_local == 90] = lat_local[lat_local == 90] - 0.001

    index = np.zeros_like(lat_local.astype(int))

    index[lat_arr > 0] = (90 / res - 1 - (lat_local[lat_arr > 0] / res)).astype(int)
    index[lat_arr <= 0] = (90 / res - (lat_local[lat_arr <= 0] / res)).astype(int)

    if is_scalar_like(lat):
        return int(index[0])

    return index


@overload
def mds_lon_to_xindex(lon: float, res: float) -> int: ...  # numpydoc ignore=GL08
@overload
def mds_lon_to_xindex(lon: np.ndarray, res: float) -> np.ndarray: ...  # numpydoc ignore=GL08
def mds_lon_to_xindex(lon: float | np.ndarray, res: float) -> int | np.ndarray:
    """
    For a given longitude return the x-index as it was in MDS2/3 in a 1x1 global grid.

    Parameters
    ----------
    lon : numpy.ndarray
        Longitude(s) of observation in degrees.
    res : float
        Resolution of grid in degrees.

    Returns
    -------
    int or numpy.ndarray
        Grid box index(es).

    Notes
    -----
    In the western hemisphere, borderline longitudes which fall on grid boundaries are pushed west, except
    -180 which goes east. In the eastern hemisphere, they are pushed east, except 180 which goes west.
    At 0 degrees they are pushed west.

    In previous versions, ``res`` had the default value 1.0.
    """
    lon_arr = np.atleast_1d(lon)
    long_local = lon_arr
    long_local[long_local == -180] = long_local[long_local == -180] + 0.001
    long_local[long_local == 180] = long_local[long_local == 180] - 0.001

    index = np.zeros_like(long_local.astype(int))

    index[lon_arr > 0.0] = ((long_local[lon_arr > 0.0] / res) + 180 / res).astype(int)
    index[lon_arr <= 0.0] = ((long_local[lon_arr <= 0.0] / res) + 180 / res - 1).astype(int)

    if is_scalar_like(lon):
        return int(index[0])

    return index
