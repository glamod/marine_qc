"""Some generally helpful astronomical geometry functions for base QC."""

from __future__ import annotations

import math

import numpy as np

from .time_control import (
    convert_time_in_hours,
    leap_year_correction,
    relative_year_number,
)

# Conversion factor between degrees and radians
degrad = np.pi / 180.0


def sun_position(time: float) -> float:
    """Find position of sun in celestial sphere, assuming circular orbit (radians).

    Parameters
    ----------
    time: float

    Returns
    -------
    float
        Position of the sun.
    """
    return (360.0 * time / 365.25) * degrad


def mean_earth_anomaly(time: float, theta: float) -> float:
    """Calculate mean anomaly of earth (g).

    Parameters
    ----------
    time: float
    theta: float

    Returns
    -------
    float
        Mean anomaly of earth (g).
    """
    return -0.031271 - 4.5396e-7 * time + theta


def sun_longitude(time: float) -> float:
    """Get longitude of sun.

    Parameters
    ----------
    time: float

    Returns
    -------
    float
        Longitude of sun.
    """
    theta = sun_position(time)
    mean_anomaly = mean_earth_anomaly(time, theta)
    return (
        4.900968
        + 3.6747e-7 * time
        + (0.033434 - 2.3e-9 * time) * math.sin(mean_anomaly)
        + 0.000349 * math.sin(2.0 * mean_anomaly)
        + theta
    )


def elliptic_angle(time: float) -> float:
    """Get angle plane of elliptic to plane of celestial equator.

    Parameters
    ----------
    time: float

    Returns
    -------
    float
        Angle plane of elliptic to plane of celestial equator.
    """
    return 0.409140 - 6.2149e-9 * time


def sun_ascension(
    long_of_sun: float, sin_long_of_sun: float, angle_of_elliptic: float
) -> float:
    """Calculate right ascension.

    Parameters
    ----------
    long_of_sun: float

    sin_long_of_sun: float

    angle_of_elliptic: float

    Returns
    -------
    float
        Right ascension.
    """
    a1 = sin_long_of_sun * math.cos(angle_of_elliptic)
    a2 = math.cos(long_of_sun)
    right_ascension = math.atan2(a1, a2)
    if right_ascension < 0.0:
        right_ascension = right_ascension + 2 * np.pi
    return right_ascension


def sun_declination(sin_long_of_sun: float, angle_of_elliptic: float) -> float:
    """Calculate declination of sun.

    Parameters
    ----------
    sin_long_of_sun: float
    angle_of_elliptic: float

    Returns
    -------
    float
        Declination of sun.
    """
    return math.asin(sin_long_of_sun * math.sin(angle_of_elliptic))


def calculate_sun_parameters(time: float) -> tuple[float, float]:
    """Calculate both right ascension and declination of sun.

    Parameters
    ----------
    time: float

    Returns
    -------
    tuple of float
        A tuple of two floats representing right ascension and declination of sun.
    """
    long_of_sun = sun_longitude(time)
    angle_of_elliptic = elliptic_angle(time)
    sin_long_of_sun = math.sin(long_of_sun)
    rta = sun_ascension(long_of_sun, sin_long_of_sun, angle_of_elliptic)
    dec = sun_declination(sin_long_of_sun, angle_of_elliptic)
    return rta, dec


def to_siderial_time(time: float, delyear: int) -> float:
    """Convert to siderial time.

    Parameters
    ----------
    time: float
    delyear: int

    Returns
    -------
    float
        Siderial time.
    """
    sid = 1.759335 + 2 * np.pi * (time / 365.25 - delyear) + 3.694e-7 * time
    if sid >= 2 * np.pi:
        sid = sid - 2 * np.pi
    return sid


def to_local_siderial_time(
    time: float, time_in_hours: float, delyear: int, lon: float
) -> float:
    """Convert to local siderial time.

    Parameters
    ----------
    time: float
    time_in_hours: float
    delyear: int
    lon: float

    Returns
    -------
    float
        Local siderial time.
    """
    siderial_time = to_siderial_time(time, delyear)
    lsid = siderial_time + (time_in_hours * 15.0 + lon) * degrad
    if lsid >= 2 * np.pi:
        lsid = lsid - 2 * np.pi
    return lsid


def sun_hour_angle(local_siderial_time: float, right_ascension: float) -> float:
    """Get hour angle.

    Parameters
    ----------
    local_siderial_time: float
    right_ascension: float

    Returns
    -------
    float
        Hour angle.
    """
    hra = local_siderial_time - right_ascension
    if hra < 0:
        hra = hra + 2 * np.pi
    return hra


def sin_of_elevation(phi: float, declination: float, hour_angle: float) -> float:
    """Get sinus of geometric elevation.

    Parameters
    ----------
    phi: float
    declination: float
    hour_angle: float

    Returns
    -------
    float
        Sinus of geometric elevation.
    """
    sin_elevation = math.sin(phi) * math.sin(declination) + math.cos(phi) * math.cos(
        declination
    ) * math.cos(hour_angle)
    sin_elevation = min(sin_elevation, 1.0)
    sin_elevation = max(sin_elevation, -1.0)
    return sin_elevation


def sun_azimuth(phi: float, declination: float) -> float:
    """Get azimuth.

    Parameters
    ----------
    phi: float
    declination: float

    Returns
    -------
    float
        Azimuth.
    """
    if (phi - declination) > 0:
        return 0
    return 180


def convert_degrees(deg: float) -> float:
    """Convert degrees.

    Parameters
    ----------
    deg: float

    Returns
    -------
    float
        Degree (from 0 to 360).
    """
    if deg < 0.0:
        deg = 360.0 + deg
    return deg


def calculate_azimuth(
    declination: float, hour_angle: float, elevation: float, phi: float
) -> float:
    """Calculate azimuth.

    Parameters
    ----------
    declination: float
    hour_angle: float
    elevation: float
    phi: float

    Returns
    -------
    float
        Azimuth.
    """
    val_to_asin = math.cos(declination) * math.sin(hour_angle) / math.cos(elevation)
    val_to_asin = min(val_to_asin, 1.0)
    val_to_asin = max(val_to_asin, -1.0)
    azimuth = math.asin(val_to_asin) / degrad
    if math.sin(elevation) < math.sin(declination) / math.sin(phi):
        azimuth = convert_degrees(azimuth)
        azimuth = azimuth - 180.0
    return 180.0 + azimuth


def azimuth_elevation(
    lat: float, declination: float, hour_angle: float
) -> tuple[float, float]:
    """Get both azimuth and geometric elevation of sun.

    Parameters
    ----------
    lat: float
    declination: float
    hour_angle: float

    Returns
    -------
    tuple of float
        A tuple of two floats representing azimuth and geometric elevation of sun.
    """
    phi = lat * degrad
    sin_elevation = sin_of_elevation(phi, declination, hour_angle)
    elevation = math.asin(sin_elevation)
    azimuth = sun_azimuth(phi, declination)
    # If sun is not very near the zenith, leave as 0 or 180
    if abs(elevation - 2 * np.pi / 4.0) > 0.000001:
        # Protect against rounding error near +/-90 degrees.
        azimuth = calculate_azimuth(declination, hour_angle, elevation, phi)
    return azimuth, elevation


def sunangle(
    year: int,
    day: int,
    hour: int,
    minute: int,
    sec: int,
    zone: int,
    dasvtm: int,
    lat: float,
    lon: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Calculate the local azimuth and elevation of the sun at a specified location and time.

    Parameters
    ----------
    year: int
        Year.
    day: int
        Day number of year starting with 1 for Jan 1st and running up to 365/6.
    hour: int
        Hour.
    minute: int
        Minute.
    sec: int
        Second.
    zone: int
        The local international time zone, counted westward from Greenwich.
    dasvtm: int
        1 if daylight saving time is in effect, otherwise 0.
    lat: float
        Latitude in degrees, north is positive.
    lon: float
        Longitude in degrees, east is positive.

    Returns
    -------
    tuple of float
        A tuple of six floats representing Azimuth angle of the sun (degrees east of north), Elevation of sun (degrees),
        Right ascension of sun (degrees), Hour angle of sun (degrees), Hour angle of
        local siderial time (degrees) and Declination of sun (degrees)

    Note
    ----
    Copied from Rob Hackett's area 28 Apr 1998 by J.Arnott.
    Add protection for ASIN near +/- 90 degrees 07 Jan 2002 by J.Arnott.
    Pythonised 25/09/2015 by J.J. Kennedy

    The Python version gets within a fraction of a degree of the original Fortran code from which it was ported
    for a range of values. The differences are larger if single precision values are used suggesting that this
    is not the most numerically robust scheme.
    """
    assert 0 < day <= 366
    assert 0 <= hour < 24
    assert 0 <= minute < 60
    assert 0 <= sec < 60
    assert 90 >= lat >= -90

    # Find number of whole years since end of 1979 (reference point)
    delyear = relative_year_number(year)
    # Find time in whole hours since midnight (allow for "daylight saving").
    time_in_hours = convert_time_in_hours(hour, minute, sec, zone, dasvtm)
    # Make leap year correction
    time = leap_year_correction(time_in_hours, day, delyear)
    # Get sun parameters
    right_ascension, declination = calculate_sun_parameters(time)
    local_siderial_time = to_local_siderial_time(time, time_in_hours, delyear, lon)
    # Hour Angle
    hour_angle = sun_hour_angle(local_siderial_time, right_ascension)
    # Geometric elevation and sun azimuth
    azimuth, elevation = azimuth_elevation(lat, declination, hour_angle)

    elevation = elevation / degrad  # Convert elevation to degrees
    declination = declination / degrad  # Convert declination to degrees

    assert 180 >= elevation >= -180

    rta = right_ascension / degrad
    sid = local_siderial_time / degrad
    sid = convert_degrees(sid)
    hra = hour_angle / degrad
    hra = convert_degrees(hra)

    return azimuth, elevation, rta, hra, sid, declination
