"""Some generally helpful location control functions for base QC."""

from __future__ import annotations

import numpy as np

from .auxiliary import isvalid
from .statistics import missing_mean


__all__ = [
    "mds_lat_to_yindex",
    "mds_lat_to_yindex_fast",
    "mds_lon_to_xindex",
    "mds_lon_to_xindex_fast",
]


def mds_lat_to_yindex(lat: float, res: float) -> int:
    """
    For a given latitude return the y-index as it was in MDS2/3 in a 1x1 global grid.

    Parameters
    ----------
    lat : float
        Latitude of the point.
    res : float
        Resolution of grid in degrees.

    Returns
    -------
    int
        Grid box index.

    Notes
    -----
    In the northern hemisphere, borderline latitudes which fall on grid boundaries are pushed north, except
    90 which goes south. In the southern hemisphere, they are pushed south, except -90 which goes north.
    At 0 degrees they are pushed south.

    Expects that latitudes run from 90N to 90S

    In previous versions, ``res`` had the default value 1.0.
    """
    lat_local = lat  # round(lat,1)

    if lat_local == -90:
        lat_local += 0.001
    if lat_local == 90:
        lat_local -= 0.001

    if lat > 0.0:
        return int(90 / res - 1 - int(lat_local / res))
    return int(90 / res - int(lat_local / res))


def mds_lat_to_yindex_fast(lat: np.ndarray, res: float) -> np.ndarray:
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
    numpy.ndarray
        Grid box indexes.

    Notes
    -----
    In the northern hemisphere, borderline latitudes which fall on grid boundaries are pushed north, except
    90 which goes south. In the southern hemisphere, they are pushed south, except -90 which goes north.
    At 0 degrees they are pushed south.

    Expects that latitudes run from 90N to 90S

    In previous versions, ``res`` had the default value 1.0.
    """
    lat_local = lat
    lat_local[lat_local == -90] = lat_local[lat_local == -90] + 0.001
    lat_local[lat_local == 90] = lat_local[lat_local == 90] - 0.001

    index = np.zeros_like(lat_local.astype(int))

    index[lat > 0] = (90 / res - 1 - (lat_local[lat > 0] / res).astype(int)).astype(int)
    index[lat <= 0] = (90 / res - (lat_local[lat <= 0] / res).astype(int)).astype(int)

    return index


def mds_lon_to_xindex(lon: float, res: float) -> int:
    """
    For a given longitude return the x-index as it was in MDS2/3 in a 1x1 global grid.

    Parameters
    ----------
    lon : float
        Longitude of the point.
    res : float
        Resolution of grid in degrees.

    Returns
    -------
    int
        Grid box index.

    Notes
    -----
    In the western hemisphere, borderline longitudes which fall on grid boundaries are pushed west, except
    -180 which goes east. In the eastern hemisphere, they are pushed east, except 180 which goes west.
    At 0 degrees they are pushed west.

    In previous versions, ``res`` had the default value 1.0.
    """
    long_local = lon  # round(lon,1)

    if long_local == -180:
        long_local += 0.001
    if long_local == 180:
        long_local -= 0.001
    if long_local > 0.0:
        return int(int(long_local / res) + 180 / res)
    return int(int(long_local / res) + 180 / res - 1)


def mds_lon_to_xindex_fast(lon: np.ndarray, res: float) -> np.ndarray:
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
    numpy.ndarray
        Grid box indexes.

    Notes
    -----
    In the western hemisphere, borderline longitudes which fall on grid boundaries are pushed west, except
    -180 which goes east. In the eastern hemisphere, they are pushed east, except 180 which goes west.
    At 0 degrees they are pushed west.

    In previous versions, ``res`` had the default value 1.0.
    """
    long_local = lon
    long_local[long_local == -180] = long_local[long_local == -180] + 0.001
    long_local[long_local == 180] = long_local[long_local == 180] - 0.001

    index = np.zeros_like(long_local.astype(int))

    index[lon > 0.0] = ((long_local[lon > 0.0] / res).astype(int) + 180 / res).astype(int)
    index[lon <= 0.0] = ((long_local[lon <= 0.0] / res).astype(int) + 180 / res - 1).astype(int)

    return index
