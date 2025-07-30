"""
Quality control suite spherical geometry module.

The spherical geometry module is a simple collection of calculations on a sphere
Sourced from https://edwilliams.org/avform147.htm formerly williams.best.vwh.net/avform.htm
"""

from __future__ import annotations

import math

import numpy as np

from .auxiliary import isvalid

earths_radius = 6371.0088
radians_per_degree = np.pi / 180.0


def sphere_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
    return angular_distance(lat1, lon1, lat2, lon2) * earths_radius


def angular_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
    if not isvalid(lat1):
        raise ValueError("First latitude point is missing of non-finite")
    if not isvalid(lat2):
        raise ValueError("Second latitude point is missing of non-finite")
    if not isvalid(lon1):
        raise ValueError("First longitude point is missing of non-finite")
    if not isvalid(lon2):
        raise ValueError("Second longitude point is missing of non-finite")

    # convert degrees to radians
    lat1 = lat1 * radians_per_degree
    lon1 = lon1 * radians_per_degree
    lat2 = lat2 * radians_per_degree
    lon2 = lon2 * radians_per_degree

    delta_lambda = abs(lon1 - lon2)
    bit1 = np.cos(lat2) * np.sin(delta_lambda)
    bit1 = bit1 * bit1
    bit2 = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(
        delta_lambda
    )
    bit2 = bit2 * bit2
    top_bit = bit1 + bit2
    top_bit = np.sqrt(top_bit)
    bottom_bit = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(
        delta_lambda
    )
    return np.arctan2(top_bit, bottom_bit)


def lat_lon_from_course_and_distance(
    lat1: float, lon1: float, tc: float, d: float
) -> tuple[float, float]:
    """
    calculate a latitude and longitude given a starting point, course (in radians) and
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
    lat1 = lat1 * radians_per_degree
    lon1 = lon1 * radians_per_degree
    tcr = tc * radians_per_degree

    dr = d / earths_radius

    lat = np.arcsin(np.sin(lat1) * np.cos(dr) + np.cos(lat1) * np.sin(dr) * np.cos(tcr))
    dlon = np.arctan2(
        np.sin(tcr) * np.sin(dr) * np.cos(lat1), np.cos(dr) - np.sin(lat1) * np.sin(lat)
    )
    lon = math.fmod(lon1 + dlon + np.pi, 2.0 * np.pi) - np.pi

    lat = lat / radians_per_degree
    lon = lon / radians_per_degree

    return lat, lon


def course_between_points(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
    d = angular_distance(lat1, lon1, lat2, lon2)

    if d != 0:
        lat1 = lat1 * radians_per_degree
        lon1 = lon1 * radians_per_degree
        lat2 = lat2 * radians_per_degree
        lon2 = lon2 * radians_per_degree

        if np.cos(lat1) < 0.0000001:
            if lat1 > 0:
                tc1 = np.pi
            else:
                tc1 = 2.0 * np.pi
        else:
            if np.sin(lon2 - lon1) > 0:
                if (
                    1.0
                    >= (np.sin(lat2) - np.sin(lat1) * np.cos(d))
                    / (np.sin(d) * np.cos(lat1))
                    >= -1.0
                ):
                    tc1 = np.arccos(
                        (np.sin(lat2) - np.sin(lat1) * np.cos(d))
                        / (np.sin(d) * np.cos(lat1))
                    )
                else:
                    tc1 = float("nan")
            else:
                if (
                    1.0
                    >= (np.sin(lat2) - np.sin(lat1) * np.cos(d))
                    / (np.sin(d) * np.cos(lat1))
                    >= -1.0
                ):
                    tc1 = 2.0 * np.pi - np.arccos(
                        (np.sin(lat2) - np.sin(lat1) * np.cos(d))
                        / (np.sin(d) * np.cos(lat1))
                    )
                else:
                    tc1 = float("nan")

        if math.isnan(tc1):
            tc1 = math.fmod(
                np.arctan2(
                    np.sin(lon1 - lon2) * np.cos(lat2),
                    np.cos(lat1) * np.sin(lat2)
                    - np.sin(lat1) * np.cos(lat2) * np.cos(lon1 - lon2),
                ),
                2 * np.pi,
            )

    else:
        tc1 = 0.0

    return tc1 / radians_per_degree


def intermediate_point(
    lat1: float, lon1: float, lat2: float, lon2: float, f: float
) -> tuple[float, float]:
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

    Raises
    ------
    ValueError
        If f is greater than 1.
    """
    if f > 1.0:
        raise ValueError(f"f greater than 1.0 {f}")

    d = angular_distance(lat1, lon1, lat2, lon2)

    if d != 0.0:
        # convert degrees to radians
        lat1 = lat1 * radians_per_degree
        lon1 = lon1 * radians_per_degree
        lat2 = lat2 * radians_per_degree
        lon2 = lon2 * radians_per_degree

        a = np.sin((1 - f) * d) / np.sin(d)
        b = np.sin(f * d) / np.sin(d)
        x = a * np.cos(lat1) * np.cos(lon1) + b * np.cos(lat2) * np.cos(lon2)
        y = a * np.cos(lat1) * np.sin(lon1) + b * np.cos(lat2) * np.sin(lon2)
        z = a * np.sin(lat1) + b * np.sin(lat2)
        lat = np.arctan2(z, np.sqrt(x * x + y * y)) / radians_per_degree
        lon = np.arctan2(y, x) / radians_per_degree
    else:
        lat = lat1
        lon = lon1

    return lat, lon
