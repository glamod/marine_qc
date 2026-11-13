"""Some generally helpful location control functions for base QC."""

from __future__ import annotations
from typing import overload

import numpy as np

from .auxiliary import is_scalar_like


__all__ = [
    "get_x_index",
    "get_y_index",
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
    lat_arr[lat_arr == -90] = lat_arr[lat_arr == -90] + 0.001
    lat_arr[lat_arr == 90] = lat_arr[lat_arr == 90] - 0.001

    index = np.zeros_like(lat_arr.astype(int))

    index[lat_arr > 0] = (90 / res - 1 - (lat_arr[lat_arr > 0] / res).astype(int)).astype(int)
    index[lat_arr <= 0] = (90 / res - (lat_arr[lat_arr <= 0] / res).astype(int)).astype(int)

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
    lon_arr[lon_arr == -180] = lon_arr[lon_arr == -180] + 0.001
    lon_arr[lon_arr == 180] = lon_arr[lon_arr == 180] - 0.001

    index = np.zeros_like(lon_arr.astype(int))

    index[lon_arr > 0.0] = ((lon_arr[lon_arr > 0.0] / res).astype(int) + 180 / res).astype(int)
    index[lon_arr <= 0.0] = ((lon_arr[lon_arr <= 0.0] / res).astype(int) + 180 / res - 1).astype(int)

    if is_scalar_like(lon):
        return int(index[0])

    return index


def get_y_index(lat: float | np.ndarray, lat_axis: np.ndarray) -> np.ndarray:
    """
    Convert an array of latitudes to an array of indices for the grid.

    Parameters
    ----------
    lat : float or numpy.ndarray
        Latitude(s) to select.
    lat_axis : numpy.ndarray
        Array containing the latitude axis.

    Returns
    -------
    numpy.ndarray
        Array of indices.
    """
    lat_arr = np.atleast_1d(lat)
    lat_axis_0 = lat_axis[0]
    lat_axis_delta = lat_axis[1] - lat_axis[0]

    # Need to know if grid cells are defined by centres or by lower edges...
    if lat_axis_0 not in [-90.0, 90.0]:
        if lat_axis_0 == -90 + lat_axis_delta / 2.0:
            lat_axis_0 = -90.0
        elif lat_axis_0 == 90 + lat_axis_delta / 2.0:
            lat_axis_0 = 90.0
        else:
            raise RuntimeError("I can't work this grid out, grid box boundaries are not at +-90 or +-(90-delta/2)")

    y_index: np.ndarray = ((lat_arr - lat_axis_0) / lat_axis_delta).astype(int)

    y_index[y_index >= len(lat_axis)] = len(lat_axis) - 1

    return y_index


def get_x_index(lon: float | np.ndarray, lon_axis: np.ndarray) -> np.ndarray:
    """
    Convert an array of longitudes to an array of indices for the grid.

    Parameters
    ----------
    lon : float np.ndarray
        Longitude(s) to select.
    lon_axis : np.ndarray
        Array containing the longitude axis.

    Returns
    -------
    np.ndarray
        Array of indices.
    """
    lon_arr = np.atleast_1d(lon)
    lon_axis_0 = lon_axis[0]
    lon_axis_delta = lon_axis[1] - lon_axis[0]

    # Need to know if grid cells are defined by centres or by lower edges...
    if lon_axis_0 not in [-180.0, 180.0]:
        if lon_axis_0 == -180 + lon_axis_delta / 2.0:
            lon_axis_0 = -180.0
        elif lon_axis_0 == 180 + lon_axis_delta / 2.0:
            lon_axis_0 = 180.0
        else:
            raise RuntimeError("I can't work this grid out, grid box boundaries are not at +-180 or +-(180-delta/2)")

    x_index: np.ndarray = ((lon_arr - lon_axis_0) / lon_axis_delta).astype(int)

    x_index[x_index >= len(lon_axis)] = len(lon_axis) - 1

    return x_index
