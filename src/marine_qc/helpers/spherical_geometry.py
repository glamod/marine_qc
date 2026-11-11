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
    SequenceNumberType,
    ensure_arrays,
    inspect_arrays,
    isvalid,
    post_format_return_type,
)


_all__ = [
    "angular_distance",
    "course_between_points",
    "intermediate_point",
    "lat_lon_from_course_and_distance",
    "sphere_distance",
]

earths_radius = 6371008.8  # IUGG mean Earth radius (m)
geod = Geod(a=earths_radius, b=earths_radius)


def _geod_inv(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute forward azimuth, back azimuth, and distance between two points using an ellipsoidal model.

    Parameters
    ----------
    lon1 : np.ndarray
        Longitude of the first point in degrees.
    lat1 : np.ndarray
        Latitude of the first point in degrees.
    lon2 : np.ndarray
        Longitude of the second point in degrees.
    lat2 : np.ndarray
        Latitude of the second point in degrees.

    Returns
    -------
    fwd_az: np.ndarray
        Forward azimuth(s) from point 1 to point 2 in degrees.
    back_az: np.ndarray
        Back azimuth(s) from point 2 to point 1 in degrees.
    dist: np.ndarray
        Distance(s) between the points in meters.
    """
    fwd_az, back_az, dist = geod.inv(lon1, lat1, lon2, lat2)
    return fwd_az, back_az, dist


def _geod_fwd(
    lon1: np.ndarray,
    lat1: np.ndarray,
    fwd_az: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute destination longitude, latitude, and back azimuth using an ellipsoidal model.

    Parameters
    ----------
    lon1 : np.ndarray
        Longitude of the starting point in degrees.
    lat1 : np.ndarray
        Latitude of the starting point in degrees.
    fwd_az : np.ndarray
        Forward azimuth (true course) in degrees.
    dist : np.ndarray
        Distance travelled in meters.

    Returns
    -------
    lon2: np.ndarray
        Destination longitude(s) in degrees.
    lat2: np.ndarray
        Destination latitude(s) in degrees.
    back_az: np.ndarray
        Back azimuth(s) in degrees.
    """
    lon2, lat2, back_az = geod.fwd(lon1, lat1, fwd_az, dist)

    return lon2, lat2, back_az


@post_format_return_type("lat1", "lon1", "lat2", "lon2", dtype=float)
@inspect_arrays("lat1", "lon1", "lat2", "lon2")
def angular_distance(
    lat1: SequenceNumberType,
    lon1: SequenceNumberType,
    lat2: SequenceNumberType,
    lon2: SequenceNumberType,
) -> np.ndarray:
    """
    Calculate the great-circle angular distance between two points on a sphere.

    Parameters
    ----------
    lat1 : :py:obj:`~marine_qc.SequenceNumberType`
        Latitude of the first point in degrees.
    lon1 : :py:obj:`~marine_qc.SequenceNumberType`
        Longitude of the first point in degrees.
    lat2 : :py:obj:`~marine_qc.SequenceNumberType`
        Latitude of the second point in degrees.
    lon2 : :py:obj:`~marine_qc.SequenceNumberType`
        Longitude of the second point in degrees.

    Returns
    -------
    numpy.ndarray
        Angular great-circle distance between the two points in radians.
        NaN is returned for any invalid input values.

    Raises
    ------
    TypeError
        If `inspect_arrays` does not return numpy ndarrays.
    """
    lat1_arr, lon1_arr, lat2_arr, lon2_arr = ensure_arrays(  # pylint: disable=unbalanced-tuple-unpacking
        lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2
    )

    valid = isvalid(lon1_arr) & isvalid(lat1_arr) & isvalid(lon2_arr) & isvalid(lat2_arr)

    result = np.full(lat1_arr.shape, np.nan, dtype=float)

    _, _, dist = _geod_inv(lon1_arr[valid], lat1_arr[valid], lon2_arr[valid], lat2_arr[valid])
    result[valid] = dist / earths_radius
    return result


@post_format_return_type("lat1", "lon1", "lat2", "lon2", dtype=float)
@inspect_arrays("lat1", "lon1", "lat2", "lon2")
def sphere_distance(
    lat1: SequenceNumberType,
    lon1: SequenceNumberType,
    lat2: SequenceNumberType,
    lon2: SequenceNumberType,
) -> np.ndarray:
    """
    Calculate the great circle angular distance between two points on a sphere.

    Parameters
    ----------
    lat1 : :py:obj:`~marine_qc.SequenceNumberType`
        Latitude of the first point in degrees.
    lon1 : :py:obj:`~marine_qc.SequenceNumberType`
        Longitude of the first point in degrees.
    lat2 : :py:obj:`~marine_qc.SequenceNumberType`
        Latitude of the second point in degrees.
    lon2 : :py:obj:`~marine_qc.SequenceNumberType`
        Longitude of the second point in degrees.

    Returns
    -------
    numpy.ndarray
        Angular great-circle distance between the two points in kilometres.

    Raises
    ------
    TypeError
        If `inspect_arrays` does not return numpy ndarrays.
    """
    lat1_arr, lon1_arr, lat2_arr, lon2_arr = ensure_arrays(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2)  # pylint: disable=unbalanced-tuple-unpacking

    valid = isvalid(lon1_arr) & isvalid(lat1_arr) & isvalid(lon2_arr) & isvalid(lat2_arr)

    result = np.full(lat1_arr.shape, np.nan, dtype=float)

    _, _, dist = _geod_inv(lon1_arr[valid], lat1_arr[valid], lon2_arr[valid], lat2_arr[valid])
    result[valid] = dist / 1000.0
    return result


@post_format_return_type("lat1", "lon1", "lat2", "lon2", "f", dtype=float, multiple=True)
@inspect_arrays("lat1", "lon1", "lat2", "lon2", "f")
def intermediate_point(
    lat1: SequenceNumberType,
    lon1: SequenceNumberType,
    lat2: SequenceNumberType,
    lon2: SequenceNumberType,
    f: SequenceNumberType,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the intermediate point along the great-circle path between two points.

    Parameters
    ----------
    lat1 : :py:obj:`~marine_qc.SequenceNumberType`
        Latitude of the first point in degrees.
    lon1 : :py:obj:`~marine_qc.SequenceNumberType`
        Longitude of the first point in degrees.
    lat2 : :py:obj:`~marine_qc.SequenceNumberType`
        Latitude of the second point in degrees.
    lon2 : :py:obj:`~marine_qc.SequenceNumberType`
        Longitude of the second point in degrees.
    f : :py:obj:`~marine_qc.SequenceNumberType`
        Fraction of distance between the points.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        A tuple containing:
        - Latitude(s) of the intermediate point(s) in degrees.
        - Longitude(s) of the intermediate point(s) in degrees.
        The outputs have the same shape as the broadcasted inputs.

    Raises
    ------
    TypeError
        If `inspect_arrays` does not return numpy ndarrays.
    """
    lat1_arr, lon1_arr, lat2_arr, lon2_arr, f_arr = ensure_arrays(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2, f=f)  # pylint: disable=unbalanced-tuple-unpacking

    valid = isvalid(lon1_arr) & isvalid(lat1_arr) & isvalid(lon2_arr) & isvalid(lat2_arr)
    valid &= (f_arr >= 0.0) & (f_arr <= 1.0)

    result_lon = np.full(lat1_arr.shape, np.nan, dtype=float)
    result_lat = np.full(lat1_arr.shape, np.nan, dtype=float)

    fwd_az, _, dist = _geod_inv(lon1_arr[valid], lat1_arr[valid], lon2_arr[valid], lat2_arr[valid])

    result_lon[valid], result_lat[valid], _ = _geod_fwd(lon1_arr[valid], lat1_arr[valid], fwd_az, dist * f_arr[valid])
    return result_lat, result_lon


@post_format_return_type("lat1", "lon1", "lat2", "lon2", dtype=float)
@inspect_arrays("lat1", "lon1", "lat2", "lon2")
def course_between_points(
    lat1: SequenceNumberType,
    lon1: SequenceNumberType,
    lat2: SequenceNumberType,
    lon2: SequenceNumberType,
) -> SequenceFloatType:
    """
    Given two points, calculate the initial true course at point1.

    Parameters
    ----------
    lat1 : :py:obj:`~marine_qc.SequenceNumberType`
        Latitude of the first point in degrees.
    lon1 : :py:obj:`~marine_qc.SequenceNumberType`
        Longitude of the first point in degrees.
    lat2 : :py:obj:`~marine_qc.SequenceNumberType`
        Latitude of the second point in degrees.
    lon2 : :py:obj:`~marine_qc.SequenceNumberType`
        Longitude of the second point in degrees.

    Returns
    -------
    :py:obj:`~marine_qc.SequenceFloatType`
        Initial true course in degrees at point one along the great circle between point
        one and point two.

    Raises
    ------
    TypeError
        If `inspect_arrays` does not return numpy ndarrays.
    """
    lat1_arr, lon1_arr, lat2_arr, lon2_arr = ensure_arrays(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2)  # pylint: disable=unbalanced-tuple-unpacking

    fwd_azimuth, _, _ = _geod_inv(lon1_arr, lat1_arr, lon2_arr, lat2_arr)
    return fwd_azimuth


@post_format_return_type("lat1", "lon1", dtype=float, multiple=True)
@inspect_arrays("lat1", "lon1", "tc", "d")
def lat_lon_from_course_and_distance(
    lat1: SequenceNumberType,
    lon1: SequenceNumberType,
    tc: SequenceNumberType,
    d: SequenceNumberType,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a starting point, true course and distance, calculate the destination point.

    Parameters
    ----------
    lat1 : :py:obj:`~marine_qc.SequenceNumberType`
        Latitude of the first point in degrees.
    lon1 : :py:obj:`~marine_qc.SequenceNumberType`
        Longitude of the first point in degrees.
    tc : :py:obj:`~marine_qc.SequenceNumberType`
        True course measured clockwise from north in degrees.
    d : :py:obj:`~marine_qc.SequenceNumberType`
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
    TypeError
        If `inspect_arrays` does not return numpy ndarrays.
    """
    lat1_arr, lon1_arr, tc_arr, d_arr = ensure_arrays(lat1=lat1, lon1=lon1, tc=tc, d=d)  # pylint: disable=unbalanced-tuple-unpacking

    lon2, lat2, _ = _geod_fwd(lon1_arr, lat1_arr, tc_arr, d_arr * 1000)
    return lat2, lon2
