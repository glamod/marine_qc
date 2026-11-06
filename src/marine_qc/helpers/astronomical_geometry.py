"""Some generally helpful astronomical geometry functions for base QC."""

from __future__ import annotations

from astropy import units
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time

from .time_control import get_utc_time


__all__ = [
    "elevation_of_sun",
]


def elevation_of_sun(
    year: int,
    day: int,
    hour: int,
    minute: int,
    sec: int,
    zone: int,
    dasvtm: int,
    lat: float,
    lon: float,
) -> float:
    """
    Calculate the elevation of the sun at a specified location and time.

    Parameters
    ----------
    year : int
        Year.
    day : int
        Day number of year starting with 1 for Jan 1st and running up to 365/6.
    hour : int
        Hour.
    minute : int
        Minute.
    sec : int
        Second.
    zone : int
        The local international time zone, counted westward from Greenwich.
    dasvtm : int
        1 if daylight saving time is in effect, otherwise 0.
    lat : float
        Latitude in degrees, north is positive.
    lon : float
        Longitude in degrees, east is positive.

    Returns
    -------
    float
        Elevation of sun (degrees).
    """
    if not (0 < day <= 366):
        raise ValueError(f"Invalid day: {day}. Must be between 1 and 366.")
    if not (0 <= hour < 24):
        raise ValueError(f"Invalid hour: {hour}. Must be between 0 and 23.")
    if not (0 <= minute < 60):
        raise ValueError(f"Invalid minute: {minute}. Must be between 0 and 59.")
    if not (0 <= sec < 60):
        raise ValueError(f"Invalid second: {sec}. Must be between 0 and 59.")
    if not (-90 <= lat <= 90):
        raise ValueError(f"Invalid latitude: {lat}. Must be between -90 and 90.")
    if not -180 <= lon <= 180:
        raise ValueError(f"Invalid longitude: {lon}. Must be between -180 and 180.")
    if dasvtm not in (0, 1):
        raise ValueError("dasvtm must be 0 or 1.")

    # Build the local clock from day-of-year, then convert to UTC.
    utc_time = get_utc_time(year, day, hour, minute, sec, zone, dasvtm)

    location = EarthLocation(lat=lat * units.deg, lon=lon * units.deg)
    sun = get_sun(Time(utc_time))
    horizontal = sun.transform_to(
        AltAz(
            obstime=sun.obstime,
            location=location,
            pressure=0 * units.hPa,  # disable refraction: geometric altitude
        )
    )
    elevation = float(horizontal.alt.to_value(units.deg))

    if not -90 <= elevation <= 90:
        raise ValueError(f"Unexpected solar elevation: {elevation} degrees.")

    return elevation
