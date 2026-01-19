"""
Quality control suite spherical geometry module.

The spherical geometry module is a simple collection of calculations on a sphere
Sourced from https://edwilliams.org/avform147.htm formerly williams.best.vwh.net/avform.htm
"""

from __future__ import annotations

import numpy as np
from pyproj import Geod

from .auxiliary import (
    SequenceFloatType,
    convert_to,
    earths_radius,
    inspect_arrays,
    is_scalar_like,
    isvalid,
    post_format_return_type,
)


radians_per_degree = np.pi / 180.0
geod = Geod(a=earths_radius, b=earths_radius)


def _geod_inv(
    lon1: SequenceFloatType,
    lat1: SequenceFloatType,
    lon2: SequenceFloatType,
    lat2: SequenceFloatType,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute forward azimuth, back azimuth, and distance between two points using an ellipsoidal model.

    Parameters
    ----------
    lon1 : SequenceFloatType
        Longitude of the first point in degrees.
    lat1 : SequenceFloatType
        Latitude of the first point in degrees.
    lon2 : SequenceFloatType
        Longitude of the second point in degrees.
    lat2 : SequenceFloatType
        Latitude of the second point in degrees.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        A tuple containing:
        - Forward azimuth(s) from point 1 to point 2 in degrees.
        - Back azimuth(s) from point 2 to point 1 in degrees.
        - Distance(s) between the points in meters.
        The outputs have the same shape as the broadcasted inputs.
    """
    fwd_az, back_az, dist = geod.inv(lon1, lat1, lon2, lat2)
    return fwd_az, back_az, dist


@post_format_return_type(["lat1", "lon1", "lat2", "lon2"], dtype=float)
@inspect_arrays(["lat1", "lon1", "lat2", "lon2"])
def angular_distance(
    lat1: SequenceFloatType,
    lon1: SequenceFloatType,
    lat2: SequenceFloatType,
    lon2: SequenceFloatType,
) -> np.ndarray:
    """
    Calculate the great-circle angular distance between two points on a sphere.

    Input latitudes and longitudes should be in degrees.
    Output distance is returned in radians.

    Parameters
    ----------
    lat1 : SequenceFloatType
        Latitude of the first point in degrees.
    lon1 : SequenceFloatType
        Longitude of the first point in degrees.
    lat2 : SequenceFloatType
        Latitude of the second point in degrees.
    lon2 : SequenceFloatType
        Longitude of the second point in degrees.

    Returns
    -------
    np.ndarray
        Angular great-circle distance between the two points in radians.
        NaN is returned for any invalid input values.

    Raises
    ------
    ValueError
        - If `inspect_arrays` does not return np.ndarrays.
        - If any of lat1, lat2, lon1, or lon2 is numerically invalid or None.
    """
    if not isinstance(lat1, np.ndarray):
        raise TypeError(f"'lat1' must be a numpy.ndarray, got {type(lat1).__name__}")
    if not isinstance(lon1, np.ndarray):
        raise TypeError(f"'lon1' must be a numpy.ndarray, got {type(lon1).__name__}")
    if not isinstance(lat2, np.ndarray):
        raise TypeError(f"'lat2' must be a numpy.ndarray, got {type(lat2).__name__}")
    if not isinstance(lon2, np.ndarray):
        raise TypeError(f"'lon2' must be a numpy.ndarray, got {type(lon2).__name__}")

    valid = isvalid(lon1) & isvalid(lat1) & isvalid(lon2) & isvalid(lat2)

    result = np.full(lat1.shape, np.nan, dtype=float)  # np.ndarray

    result[valid] = _geod_inv(lon1[valid], lat1[valid], lon2[valid], lat2[valid])[2] / earths_radius
    return result


@post_format_return_type(["lat1", "lon1", "lat2", "lon2"], dtype=float)
@inspect_arrays(["lat1", "lon1", "lat2", "lon2"])
def sphere_distance(
    lat1: SequenceFloatType,
    lon1: SequenceFloatType,
    lat2: SequenceFloatType,
    lon2: SequenceFloatType,
) -> np.ndarray:
    """
    Calculate the great circle angular distance between two points on a sphere.

    Input latitudes and longitudes should be in degrees.
    Output distance is returned in radians.

    The great circle distance is the shortest distance between any two points on the Earths surface.
    The calculation is done by first calculating the angular distance between the points and then
    multiplying that by the radius of the Earth. The angular distance calculation is handled by
    another function.

    Parameters
    ----------
    lat1 : SequenceFloatType
        Latitude of the first point in degrees.
    lon1 : SequenceFloatType
        Longitude of the first point in degrees.
    lat2 : SequenceFloatType
        Latitude of the second point in degrees.
    lon2 : SequenceFloatType
        Longitude of the second point in degrees.

    Returns
    -------
    np.ndarray
        Angular great-circle distance between the two points in kilometres.

    Raises
    ------
    ValueError
        If `inspect_arrays` does not return np.ndarrays.
    """
    if not isinstance(lat1, np.ndarray):
        raise TypeError(f"'lat1' must be a numpy.ndarray, got {type(lat1).__name__}")
    if not isinstance(lon1, np.ndarray):
        raise TypeError(f"'lon1' must be a numpy.ndarray, got {type(lon1).__name__}")
    if not isinstance(lat2, np.ndarray):
        raise TypeError(f"'lat2' must be a numpy.ndarray, got {type(lat2).__name__}")
    if not isinstance(lon2, np.ndarray):
        raise TypeError(f"'lon2' must be a numpy.ndarray, got {type(lon2).__name__}")

    valid = isvalid(lon1) & isvalid(lat1) & isvalid(lon2) & isvalid(lat2)

    result = np.full(lat1.shape, np.nan, dtype=float)  # np.ndarray

    result[valid] = _geod_inv(lon1[valid], lat1[valid], lon2[valid], lat2[valid])[2] / 1000.0
    return result


@post_format_return_type(["lat1", "lon1", "lat2", "lon2", "f"], dtype=float, multiple=True)
@inspect_arrays(["lat1", "lon1", "lat2", "lon2", "f"])
def intermediate_point(
    lat1: SequenceFloatType,
    lon1: SequenceFloatType,
    lat2: SequenceFloatType,
    lon2: SequenceFloatType,
    f: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the intermediate point along the great-circle path between two points.

    Given two lat,lon points find the latitude and longitude that are a fraction f
    of the great circle distance between them https://edwilliams.org/avform147.htm formerly
    williams.best.vwh.net/avform.htm#Intermediate

    Parameters
    ----------
    lat1 : SequenceFloatType
        Latitude of the first point in degrees.
    lon1 : SequenceFloatType
        Longitude of the first point in degrees.
    lat2 : SequenceFloatType
        Latitude of the second point in degrees.
    lon2 : SequenceFloatType
        Longitude of the second point in degrees.
    f : float
        Fraction of distance between the two points.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        A tuple containing:
        - Latitude(s) of the intermediate point(s) in degrees.
        - Longitude(s) of the intermediate point(s) in degrees.
        The outputs have the same shape as the broadcasted inputs.

    Raises
    ------
    ValueError
        If `inspect_arrays` does not return np.ndarrays.
    """
    if not isinstance(lat1, np.ndarray):
        raise TypeError(f"'lat1' must be a numpy.ndarray, got {type(lat1).__name__}")
    if not isinstance(lon1, np.ndarray):
        raise TypeError(f"'lon1' must be a numpy.ndarray, got {type(lon1).__name__}")
    if not isinstance(lat2, np.ndarray):
        raise TypeError(f"'lat2' must be a numpy.ndarray, got {type(lat2).__name__}")
    if not isinstance(lon2, np.ndarray):
        raise TypeError(f"'lon2' must be a numpy.ndarray, got {type(lon2).__name__}")

    valid = isvalid(lon1) & isvalid(lat1) & isvalid(lon2) & isvalid(lat2)
    valid &= f <= 1.0
    valid &= f >= 0.0

    lon_f = np.full(lat1.shape, np.nan, dtype=float)  # np.ndarray
    lat_f = np.full(lat1.shape, np.nan, dtype=float)  # np.ndarray

    fwd_az, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    distance_at_f = dist * f

    lon_f[valid], lat_f[valid], _ = geod.fwd(lon1[valid], lat1[valid], fwd_az[valid], distance_at_f[valid])
    return lat_f, lon_f


@post_format_return_type(["lat1", "lon1", "lat2", "lon2"], dtype=float)
@inspect_arrays(["lat1", "lon1", "lat2", "lon2"])
def course_between_points(
    lat1: SequenceFloatType,
    lon1: SequenceFloatType,
    lat2: SequenceFloatType,
    lon2: SequenceFloatType,
) -> SequenceFloatType:
    """
    Given two points find the initial true course at point1 inputs are in degrees and output is in degrees.

    Parameters
    ----------
    lat1 : SequenceFloatType
        Latitude of the first point in degrees.
    lon1 : SequenceFloatType
        Longitude of the first point in degrees.
    lat2 : SequenceFloatType
        Latitude of the second point in degrees.
    lon2 : SequenceFloatType
        Longitude of the second point in degrees.

    Returns
    -------
    SequenceFloatType
        Initial true course in degrees at point one along the great circle between point
        one and point two.
    """
    fwd_azimuth, _, _ = geod.inv(lon1, lat1, lon2, lat2)
    return fwd_azimuth


@post_format_return_type(["lat1", "lon1"], dtype=float, multiple=True)
@inspect_arrays(
    [
        "lat1",
        "lon1",
    ]
)
def lat_lon_from_course_and_distance(
    lat1: SequenceFloatType,
    lon1: SequenceFloatType,
    tc: float,
    d: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate latitude and longitude given a starting point, true course and distance.

    Uses spherical trigonometry formulas from https://edwilliams.org/avform147.htm
    to compute the endpoint given a starting latitude and longitude, a true coure
    (bearing), and a distance traveled along a great-circle path.

    Parameters
    ----------
    lat1 : SequenceFloatType
        Latitude of the first point in degrees.
    lon1 : SequenceFloatType
        Longitude of the first point in degrees.
    tc : float
        True course measured clockwise from north in degrees.
    d : float
        Distance travelled in kilometres.

    Returns
    -------
    tuple of (SequenceFloatType, SequenceFloatType)
        A tuple containing:
        - Latitude(s) of the intermediate point(s) in degrees.
        - Longitude(s) of the intermediate point(s) in degrees.
        The outputs have the same shape as the broadcasted inputs.

    Raises
    ------
    ValueError
        If `inspect_arrays` does not return np.ndarrays.
    """
    if not isinstance(lat1, np.ndarray):
        raise TypeError(f"'lat1' must be a numpy.ndarray, got {type(lat1).__name__}")
    if not isinstance(lon1, np.ndarray):
        raise TypeError(f"'lon1' must be a numpy.ndarray, got {type(lon1).__name__}")

    lat_rad: np.ndarray = np.asarray(convert_to(lat1, "deg", "rad"), dtype=float)
    lon_rad: np.ndarray = np.asarray(convert_to(lon1, "deg", "rad"), dtype=float)

    tc_converted = convert_to(tc, "deg", "rad")
    if tc_converted is None:
        tc_converted = [np.nan]
    elif is_scalar_like(tc_converted):
        tc_converted = [tc_converted]
    tc_rad: np.ndarray = np.array(tc_converted, dtype=float)

    d_rad = d / earths_radius * 1000

    lat_trig = np.arcsin(np.sin(lat_rad) * np.cos(d_rad) + np.cos(lat_rad) * np.sin(d_rad) * np.cos(tc_rad))
    dlon = np.arctan2(np.sin(tc_rad) * np.sin(d_rad) * np.cos(lat_rad), np.cos(d_rad) - np.sin(lat_rad) * np.sin(lat_trig))
    lon_trig = np.mod(lon_rad + dlon + np.pi, 2.0 * np.pi) - np.pi

    lat_deg: np.ndarray = np.array(convert_to(lat_trig, "rad", "deg"), dtype=float)
    lon_deg: np.ndarray = np.array(convert_to(lon_trig, "rad", "deg"), dtype=float)

    return lat_deg, lon_deg
