"""
Quality control suite spherical geometry module.

The spherical geometry module is a simple collection of calculations on a sphere
Sourced from https://edwilliams.org/avform147.htm formerly williams.best.vwh.net/avform.htm
"""

from __future__ import annotations

import numpy as np
from pyproj import Geod

from .auxiliary import (
    earths_radius,
    isvalid,
    convert_to,
    inspect_arrays,
    post_format_return_type,
)

radians_per_degree = np.pi / 180.0
geod = Geod(a=earths_radius, b=earths_radius)


@post_format_return_type(["lat1", "lon1", "lat2", "lon2"], dtype=float)
@inspect_arrays(["lat1", "lon1", "lat2", "lon2"])
def angular_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on a sphere  input latitudes and longitudes should be in degrees
    output is in radians

    Parameters
    ----------
    lat1 : float
        latitude of first point
    lon1 : float
        longitude of first point
    lat2 : float
        latitude of second point
    lon2 : float
        longitude of second point

    Returns
    -------
    float
        Return the angular great circle distance between the two points in radians

    Raises
    ------
    ValueError
        If either lat1, lat2, lon1 or lon2 is numerically invalid or None.
    """
    # docstring !!!
    valid = isvalid(lon1) & isvalid(lat1) & isvalid(lon2) & isvalid(lat2)

    result = np.full(lat1.shape, np.nan, dtype=float)  # np.ndarray

    result[valid] = (
        geod_inv(lon1[valid], lat1[valid], lon2[valid], lat2[valid])[2] / earths_radius
    )
    return result


@post_format_return_type(["lat1", "lon1", "lat2", "lon2"], dtype=float)
@inspect_arrays(["lat1", "lon1", "lat2", "lon2"])
def sphere_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the sphere designated
    by their latitude and longitude

    The great circle distance is the shortest distance between any two points on the Earths surface.
    The calculation is done by first calculating the angular distance between the points and then
    multiplying that by the radius of the Earth. The angular distance calculation is handled by
    another function.

    Parameters
    ----------
    lat1 : float
        latitude of first point
    lon1 : float
        longitude of first point
    lat2 : float
        latitude of second point
    lon2 : float
        longitude of second point

    Returns
    -------
    float
        Return the great circle distance in kilometres between the two points
    """
    # docstring !!!
    valid = isvalid(lon1) & isvalid(lat1) & isvalid(lon2) & isvalid(lat2)

    result = np.full(lat1.shape, np.nan, dtype=float)  # np.ndarray

    result[valid] = (
        geod_inv(lon1[valid], lat1[valid], lon2[valid], lat2[valid])[2] / 1000.0
    )
    return result


@post_format_return_type(
    ["lat1", "lon1", "lat2", "lon2", "f"], dtype=float, multiple=True
)
@inspect_arrays(["lat1", "lon1", "lat2", "lon2", "f"])
def intermediate_point(lat1, lon1, lat2, lon2, f):
    """Given two lat,lon point find the latitude and longitude that are a fraction f
    of the great circle distance between them https://edwilliams.org/avform147.htm formerly
    williams.best.vwh.net/avform.htm#Intermediate

    Parameters
    ----------
    lat1 : float
        latitude of first point
    lon1 : float
        longitude of first point
    lat2 : float
        latitude of second point
    lon2 : float
        longitude of second point
    f : float
        fraction of distance between the two points

    Returns
    -------
    float, float
        return the latitude and longitude of the point a fraction f along the great circle between the
        first and second points.
    """
    # docstring !!!
    valid = isvalid(lon1) & isvalid(lat1) & isvalid(lon2) & isvalid(lat2)
    valid &= f <= 1.0
    valid &= f >= 0.0

    lon_f = np.full(lat1.shape, np.nan, dtype=float)  # np.ndarray
    lat_f = np.full(lat1.shape, np.nan, dtype=float)  # np.ndarray

    fwd_az, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    distance_at_f = dist * f

    lon_f[valid], lat_f[valid], _ = geod.fwd(
        lon1[valid], lat1[valid], fwd_az[valid], distance_at_f[valid]
    )
    return lat_f, lon_f


@post_format_return_type(["lat1", "lon1", "lat2", "lon2"], dtype=float)
@inspect_arrays(["lat1", "lon1", "lat2", "lon2"])
def course_between_points(lat1, lon1, lat2, lon2):
    """Given two points find the initial true course at point1 inputs are in degrees and output is in degrees

    Parameters
    ----------
    lat1 : float
        latitude of first point
    lon1 : float
        longitude of first point
    lat2 : float
        latitude of second point
    lon2 : float
        longitude of second point

    Returns
    -------
    float
        return the initial true course in degrees at point one along the great circle between point
        one and point two
    """
    # docstring !!!
    fwd_azimuth, _, _ = geod.inv(lon1, lat1, lon2, lat2)
    fwd_azimuth[fwd_azimuth < 0] = fwd_azimuth[fwd_azimuth < 0] + 360.0
    return fwd_azimuth


@post_format_return_type(["lat1", "lon1"], dtype=float, multiple=True)
@inspect_arrays(
    [
        "lat1",
        "lon1",
    ]
)
def lat_lon_from_course_and_distance(lat1, lon1, tc, d):
    """
    Calculate a latitude and longitude given a starting point, course (in radians) and
    angular distance (also in radians) from https://edwilliams.org/avform147.htm
    formerly williams.best.vwh.net/avform.htm#LL

    Parameters
    ----------
    lat1 : float
        latitude of first point in degrees
    lon1 : float
        longitude of first point in degrees
    tc : float
        true course measured clockwise from north in degrees
    d : float
        distance travelled in kilometres

    Returns
    -------
    float, float
        return the new latitude and longitude
    """
    # docstring !!!
    lat1 = convert_to(lat1, "deg", "rad")
    lon1 = convert_to(lon1, "deg", "rad")
    tcr = convert_to(tc, "deg", "rad")

    dr = d / earths_radius * 1000

    lat = np.arcsin(np.sin(lat1) * np.cos(dr) + np.cos(lat1) * np.sin(dr) * np.cos(tcr))
    dlon = np.arctan2(
        np.sin(tcr) * np.sin(dr) * np.cos(lat1), np.cos(dr) - np.sin(lat1) * np.sin(lat)
    )
    lon = np.mod(lon1 + dlon + np.pi, 2.0 * np.pi) - np.pi

    lat = convert_to(lat, "rad", "deg")
    lon = convert_to(lon, "rad", "deg")

    return lat, lon


def geod_inv(lon1, lat1, lon2, lat2):
    """Returns forward azimuth, back azimuth, and distance  using the ellipsoidal model."""
    fwd_az, back_az, dist = geod.inv(lon1, lat1, lon2, lat2)
    return fwd_az, back_az, dist
