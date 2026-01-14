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
    isvalid,
    post_format_return_type,
)


radians_per_degree = np.pi / 180.0
geod = Geod(a=earths_radius, b=earths_radius)


def _geod_inv(
    lat1: SequenceFloatType,
    lon1: SequenceFloatType,
    lat2: SequenceFloatType,
    lon2: SequenceFloatType,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute forward azimuth, back azimuth, and distance between two points using an ellipsoidal model.

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
        If any of lat1, lat2, lon1, or lon2 is numerically invalid or None.
    """
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
    """
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
    """
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
) -> tuple[SequenceFloatType, SequenceFloatType]:
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
    """
    lat1 = convert_to(lat1, "deg", "rad")
    lon1 = convert_to(lon1, "deg", "rad")
    tcr = convert_to(tc, "deg", "rad")

    dr = d / earths_radius * 1000

    lat = np.arcsin(np.sin(lat1) * np.cos(dr) + np.cos(lat1) * np.sin(dr) * np.cos(tcr))
    dlon = np.arctan2(np.sin(tcr) * np.sin(dr) * np.cos(lat1), np.cos(dr) - np.sin(lat1) * np.sin(lat))
    lon = np.mod(lon1 + dlon + np.pi, 2.0 * np.pi) - np.pi

    lat = convert_to(lat, "rad", "deg")
    lon = convert_to(lon, "rad", "deg")

    return lat, lon
