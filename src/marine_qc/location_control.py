"""Some generally helpful location control functions for base QC."""

from __future__ import annotations

import numpy as np

from .auxiliary import ValueFloatType, ValueIntType, isvalid
from .statistics import missing_mean


def yindex_to_lat(yindex: int, res: float) -> float:
    """
    Convert yindex to latitude.

    Parameters
    ----------
    yindex : int
        Index of the latitude.
    res : float
        Resolution of grid in degrees.

    Returns
    -------
    float
        Latitude (degrees).

    Notes
    -----
    In previous versions, ``res`` had the default value 1.0.
    """
    if yindex < 0:
        raise ValueError(f"Invalid yindex: {yindex}. Must be positive.")
    if not (yindex < 180 / res):
        raise ValueError(f"Invalid yindex: {yindex}. Must be less than {180 / res}.")
    return 90.0 - yindex * res - res / 2.0


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
    lat : np.ndarray
        Latitude(s) of observation in degrees.
    res : float
        Resolution of grid in degrees.

    Returns
    -------
    np.ndarray
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


def lat_to_yindex(lat: float, res: float) -> int:
    """
    For a given latitude return the y index in a 1x1x5-day global grid.

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
    The routine assumes that the structure of the SST array is a grid that is 360 x 180 x 73
    i.e. one year of 1degree lat x 1degree lon data split up into pentads. The west-most box is at 180degrees with
    index 0 and the northernmost box also has index zero. Inputs on the border between grid cells are pushed south.

    In previous versions, ``res`` had the default value 1.0.
    """
    yindex = int((90 - lat) / res)
    if yindex >= 180 / res:
        yindex = int(180 / res - 1)
    return max(yindex, 0)


def xindex_to_lon(xindex: int, res: float) -> float:
    """
    Convert xindex to longitude.

    Parameters
    ----------
    xindex : int
        Index of the longitude.
    res : float
        Resolution of grid in degrees.

    Returns
    -------
    float
        Longitude (degrees).

    Notes
    -----
    In previous versions, ``res`` had the default value 1.0.
    """
    if xindex < 0:
        raise ValueError(f"Invalid xindex: {xindex}. Must be positive.")
    if not (xindex < 360 / res):
        raise ValueError(f"Invalid xindex: {xindex}. Must be less than {360 / res}.")
    return xindex * res - 180.0 + res / 2.0


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
    lon : np.ndarray
        Longitude(s) of observation in degrees.
    res : float
        Resolution of grid in degrees.

    Returns
    -------
    np.ndarray
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


def lon_to_xindex(lon: float, res: float) -> int:
    """
    For a given longitude return the x index in a 1x1x5-day global grid.

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
    The routine assumes that the structure of the SST array is a grid that is 360 x 180 x 73
    i.e. one year of 1degree lat x 1degree lon data split up into pentads. The west-most box is at 180degrees W with
    index 0 and the northernmost box also has index zero. Inputs on the border between grid cells are pushed east.

    In previous versions, ``res`` had the default value 1.0.
    """
    inlon = lon
    if inlon >= 180.0:
        inlon = -180.0 + (inlon - 180.0)
    if inlon < -180.0:
        inlon = inlon + 360.0
    xindex = int((inlon + 180.0) / res)
    while xindex >= 360 / res:
        xindex -= int(360 / res)
    return int(xindex)


def filler(
    value_to_fill: float | None, 
    neighbour1: float | None, 
    neighbour2: float | None, 
    opposite: float | None,
) -> float | None:
    """
    Fill invalid values.

    If the value_to_fill is invalid it is replaced with the mean of the neighbours and if it is still invalid then
    it is replaced with the value from the opposite member.

    Parameters
    ----------
    value_to_fill : float
        The value to fill.

    neighbour1 : float
        The first neighbour.

    neighbour2 : float
        The second neighbour.

    opposite : float
        The opposite member.

    Returns
    -------
    float
        Filled invalid input values.
    """
    if not isvalid(value_to_fill):
        value_to_fill = missing_mean([neighbour1, neighbour2])
    if not isvalid(value_to_fill):
        value_to_fill = opposite
    return value_to_fill


def fill_missing_vals(
    q11: float | None, 
    q12: float | None, 
    q21: float | None, 
    q22: float | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Fill missing values.

    For a group of four neighbouring grid boxes which form a square, with values q11, q12, q21, q22,
    fill gaps using means of neighbours.

    Parameters
    ----------
    q11 : float
        Value of first gridbox.
    q12 : float
        Value of second gridbox.
    q21 : float
        Value of third gridbox.
    q22 : float
        Value of fourth gridbox.

    Returns
    -------
    tuple of float
        A tuple of four floats representing neighbour means.
    """
    outq11 = q11
    outq12 = q12
    outq21 = q21
    outq22 = q22

    outq11 = filler(outq11, q12, q21, q22)
    outq22 = filler(outq22, q12, q21, q11)
    outq12 = filler(outq12, q11, q22, q21)
    outq21 = filler(outq21, q11, q22, q12)

    return outq11, outq12, outq21, outq22


def get_four_surrounding_points(lat: float, lon: float, res: int, max90: bool = True) -> tuple[float, float, float, float]:
    """
    Get the four surrounding points of a specified latitude and longitude point.

    Parameters
    ----------
    lat : float
        Latitude of point.
    lon : float
        Longitude of point.
    res : int
        Resolution of the grid in degrees.
    max90 : bool, default: True
        If True then cap latitude at 90.0, otherwise don't cap latitude.

    Returns
    -------
    tuple of floats
        A tuple of floats representing the longitudes of the leftmost and rightmost pairs of points,
        and the latitudes of the topmost and bottommost pairs of points.
    """
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Invalid latitude: {lat}. Must be between -90 and 90.")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Invalid longitude: {lon}. Must be between -180 and 180.")

    x2_index = lon_to_xindex(lon + 0.5, res=res)
    x2 = xindex_to_lon(x2_index, res=res)
    if x2 < lon:
        x2 += 360.0

    x1_index = lon_to_xindex(lon - 0.5, res=res)
    x1 = xindex_to_lon(x1_index, res=res)
    if x1 > lon:
        x1 -= 360.0

    if lat + 0.5 <= 90:
        y2_index = lat_to_yindex(lat + 0.5, res=res)
        y2 = yindex_to_lat(y2_index, res=res)
    else:
        y2 = 89.5
        if not max90:
            y2 = 90.5

    if lat - 0.5 >= -90:
        y1_index = lat_to_yindex(lat - 0.5, res=res)
        y1 = yindex_to_lat(y1_index, res=res)
    else:
        y1 = -89.5
        if not max90:
            y1 = -90.5

    return x1, x2, y1, y2
