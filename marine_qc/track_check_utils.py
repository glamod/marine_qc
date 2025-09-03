"""
The New Track Check QC module provides the functions needed to perform the
track check. The main routine is mds_full_track_check which takes a
list of class`.MarineReport` from a single ship and runs the track check on them.
This is an update of the MDS system track check in that it assumes the Earth is a
sphere. In practice, it gives similar results to the cylindrical earth formerly
assumed.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from . import spherical_geometry as sph, spherical_geometry as sg, time_control
from .auxiliary import (
    convert_to,
    isvalid,
    inspect_arrays,
    convert_units,
    SequenceFloatType,
    SequenceDatetimeType,
)
from .spherical_geometry import (
    sphere_distance_array,
    course_between_points_array,
    intermediate_point_array,
)
from .time_control import time_differences_array


def modal_speed(speeds: list) -> float:
    """Calculate the modal speed from the input array in 3 knot bins. Returns the
    bin-centre for the modal group.

    The data are binned into 3-knot bins with the first from 0-3 knots having a
    bin centre of 1.5 and the highest containing all speed in excess of 33 knots
    with a bin centre of 34.5. The bin with the most speeds in it is found. The higher of
    the modal speed or 8.5 is returned:

    Bins-   0-3, 3-6, 6-9, 9-12, 12-15, 15-18, 18-21, 21-24, 24-27, 27-30, 30-33, 33-36
    Centres-1.5, 4.5, 7.5, 10.5, 13.5,  16.5,  19.5,  22.5,  25.5,  28.5,  31.5,  34.5

    Parameters
    ----------
    speeds : list
        Input speeds in km/h

    Returns
    -------
    float
        Bin-centre speed (expressed in km/h) for the 3 knot bin which contains most speeds in
        input array, or 8.5, whichever is higher
    """
    # if there is one or no observations then return None
    # if the speed is on a bin edge then it rounds up to higher bin
    # if the modal speed is less than 8.50 then it is set to 8.50
    # anything exceeding 36 knots is assigned to the top bin
    if len(speeds) <= 1:
        return np.nan

    # Convert km/h to knots
    speeds = np.asarray(speeds)
    speeds = convert_to(speeds, "km/h", "knots")

    # Bin edges: [0, 3, 6, ..., 36], 12 bins
    bins = np.arange(0, 37, 3)

    # Digitize returns bin index starting from 1
    bin_indices = np.digitize(speeds, bins, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, 11)

    # Count occurrences in each bin
    counts = np.bincount(bin_indices, minlength=12)

    # Find the modal bin (first one in case of tie)
    modal_bin = np.argmax(counts)

    # Bin centres: 1.5, 4.5, ..., 34.5
    bin_centres = bins[:-1] + 1.5
    modal_speed_knots = max(bin_centres[modal_bin], 8.5)

    # Convert back to km/h
    return convert_to(modal_speed_knots, "knots", "km/h")


def set_speed_limits(amode: float) -> (float, float, float):
    """Takes a modal speed and calculates speed limits for the track checker

    Parameters
    ----------
    amode : float
        modal speed in kmk/h

    Returns
    -------
    (float, float, float)
        max speed, maximum max speed and min speed
    """
    amax = convert_to(15.0, "knots", "km/h")
    amaxx = convert_to(20.0, "knots", "km/h")
    amin = 0.00

    if not isvalid(amode):
        return amax, amaxx, amin
    if amode <= convert_to(8.51, "knots", "km/h"):
        return amax, amaxx, amin

    return amode * 1.25, convert_to(30.0, "knots", "km/h"), amode * 0.75


def increment_position(
    alat1: float, alon1: float, avs: float, ads: float, timdif: float
) -> (float, float):
    """Increment_position takes latitudes and longitude, a speed, a direction and a time difference and returns
    increments of latitude and longitude which correspond to half the time difference.

    Parameters
    ----------
    alat1 : float
        Latitude at starting point in degrees
    alon1 : float
        Longitude at starting point in degrees
    avs : float
        speed of ship in km/h
    ads : float
        heading of ship in degrees
    timdif : float
        time difference between the points in hours

    Returns
    -------
    (float, float) or (None, None)
        Returns latitude and longitude increment or None and None if timedif is None
    """
    lat = np.nan
    lon = np.nan
    if isvalid(timdif):
        distance = avs * timdif / 2.0
        lat, lon = sph.lat_lon_from_course_and_distance(alat1, alon1, ads, distance)
        lat -= alat1
        lon -= alon1

    return lat, lon


def direction_continuity_array(dsi, ship_directions, max_direction_change=60.0):

    allowed_list = [0, 45, 90, 135, 180, 225, 270, 315, 360]

    dsi_filtered = np.empty(len(dsi))
    selection = np.isin(dsi, allowed_list)
    dsi_filtered[selection] = dsi[selection]

    dsi_previous = np.roll(dsi, 1)
    dsi_previous[0] = np.nan

    selection1 = max_direction_change < abs(dsi - ship_directions)
    selection2 = abs(dsi - ship_directions) < (360 - max_direction_change)
    selection3 = max_direction_change < abs(dsi_previous - ship_directions)
    selection4 = abs(dsi_previous - ship_directions) < (360 - max_direction_change)

    result = np.zeros(len(dsi))
    result[np.logical_and(selection1, selection2)] = 10.0
    result[np.logical_and(selection3, selection4)] = 10.0

    return result


def direction_continuity(
    dsi: float,
    dsi_previous: float,
    ship_directions: float,
    max_direction_change: float = 60.0,
) -> float:
    """Check that the reported direction at the previous time step and the actual
    direction taken are within max_direction_change degrees of one another.

    Parameters
    ----------
    dsi : float
        heading at current time step in degrees
    dsi_previous : float
        heading at previous time step in degrees
    ship_directions : float
        calculated ship direction from reported positions in degrees
    max_direction_change : float
        Largest deviations that will not be flagged in degrees

    Returns
    -------
    float
        Returns 10.0 if the difference between reported and calculated direction is greater than 60 degrees,
        0.0 otherwise
    """
    # Check for continuity of direction. Error if actual direction is not within 60 degrees of reported direction
    # of travel or previous reported direction of travel.
    result = 0.0

    if not isvalid(dsi) or not isvalid(dsi_previous) or not isvalid(ship_directions):
        return result

    allowed_list = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    if dsi not in allowed_list:
        raise ValueError(f"dsi not one of allowed values {dsi}")
    if dsi_previous not in allowed_list:
        raise ValueError(f"dsi_previous not one of allowed values {dsi_previous}")

    if (
        max_direction_change < abs(dsi - ship_directions) < 360 - max_direction_change
    ) or (
        max_direction_change
        < abs(dsi_previous - ship_directions)
        < 360 - max_direction_change
    ):
        result = 10.0

    return result


def speed_continuity_array(vsi, speeds, max_speed_change=10.0):
    result = np.zeros(len(vsi))
    vsi_previous = np.roll(vsi, 1)
    selection1 = abs(vsi - speeds) > max_speed_change
    selection2 = abs(vsi_previous - speeds) > max_speed_change
    result[np.logical_and(selection1, selection2)] = 10.0
    return result


def speed_continuity(
    vsi: float, vsi_previous: float, speeds: float, max_speed_change: float = 10.0
) -> float:
    """Check if reported speed at this and previous time step is within max_speed_change
    knots of calculated speed between those two time steps

    Parameters
    ----------
    vsi : float
        Reported speed in km/h at current time step
    vsi_previous : float
        Reported speed in km/h at previous time step
    speeds : float
        Speed of ship calculated from locations at current and previous time steps in km/h
    max_speed_change : float
        Largest change of speed that will not raise flag in km/h, default 10

    Returns
    -------
    float
        Returns 10 if the reported and calculated speeds differ by more than 10 knots, 0 otherwise
    """
    result = 0.0
    if not isvalid(vsi) or not isvalid(vsi_previous) or not isvalid(speeds):
        return result

    if (
        abs(vsi - speeds) > max_speed_change
        and abs(vsi_previous - speeds) > max_speed_change
    ):
        result = 10.0

    return result


def check_distance_from_estimate_array(
    vsi, time_differences, fwd_diff_from_estimated, rev_diff_from_estimated
):

    vsi_previous = np.roll(vsi, 1)
    vsi_previous[0] = np.nan

    alwdis = time_differences * ((vsi + vsi_previous) / 2.0)

    selection = fwd_diff_from_estimated > alwdis
    selection = np.logical_and(selection, rev_diff_from_estimated > alwdis)
    selection = np.logical_and(selection, vsi > 0)
    selection = np.logical_and(selection, vsi_previous > 0)
    selection = np.logical_and(selection, time_differences > 0)

    result = np.zeros(len(vsi))
    result[selection] = 10.0

    return result


def check_distance_from_estimate(
    vsi: float,
    vsi_previous: float,
    time_differences: float,
    fwd_diff_from_estimated: float,
    rev_diff_from_estimated: float,
) -> float:
    """Check that distances from estimated positions (calculated forward and backwards in time) are less than
    time difference multiplied by the average reported speeds

    Parameters
    ----------
    vsi : float
        reported speed in km/h at current time step
    vsi_previous : float
        reported speed in km/h at previous time step
    time_differences : float
        calculated time differences between reports in hours
    fwd_diff_from_estimated : float
        distance in km from estimated position, estimates made forward in time
    rev_diff_from_estimated : float
        distance in km from estimated position, estimates made backward in time

    Returns
    -------
    float
        Returns 10 if estimated and reported positions differ by more than the reported speed multiplied by the
        calculated time difference, 0 otherwise
    """
    # Quality-control by examining the distance between the calculated and reported second position.
    result = 0.0

    if (
        not isvalid(vsi)
        or not isvalid(vsi_previous)
        or not isvalid(time_differences)
        or not isvalid(fwd_diff_from_estimated)
        or not isvalid(rev_diff_from_estimated)
    ):
        return result

    if vsi > 0 and vsi_previous > 0 and time_differences > 0:
        alwdis = time_differences * ((vsi + vsi_previous) / 2.0)

        if fwd_diff_from_estimated > alwdis and rev_diff_from_estimated > alwdis:
            result = 10.0

    return result


def increment_position_array(alat1, alon1, avs, ads, timediff):
    """Increment_position takes latitudes and longitude, a speed, a direction and a time difference and returns
    increments of latitude and longitude which correspond to half the time difference.
    """
    distance = avs * timediff / 2.0
    lat, lon = sph.lat_lon_from_course_and_distance_array(alat1, alon1, ads, distance)
    lat = lat - alat1
    lon = lon - alon1

    return lat, lon


def calculate_speed_course_distance_time_difference_array(
    lat, lon, date, alternating=False
):

    if alternating:
        distance = sphere_distance_array(
            np.roll(lat, 1), np.roll(lon, 1), np.roll(lat, -1), np.roll(lon, -1)
        )
        timediff = time_differences_array(np.roll(date, -1), np.roll(date, 1))
        course = course_between_points_array(
            np.roll(lat, 1), np.roll(lon, 1), np.roll(lat, -1), np.roll(lon, -1)
        )
        # Alternating estimates are unavailable for the first and last elements
        distance[0] = np.nan
        distance[-1] = np.nan
        timediff[0] = np.nan
        timediff[-1] = np.nan
        course[0] = np.nan
        course[-1] = np.nan
    else:
        distance = sphere_distance_array(np.roll(lat, 1), np.roll(lon, 1), lat, lon)
        timediff = time_differences_array(date, np.roll(date, 1))
        course = course_between_points_array(np.roll(lat, 1), np.roll(lon, 1), lat, lon)
        # With the regular first differences, we don't have anything for the first element
        distance[0] = np.nan
        timediff[0] = np.nan
        course[0] = np.nan

    speed = distance / timediff
    speed[timediff == 0.0] = 0.0

    return speed, distance, course, timediff


@inspect_arrays(["vsi", "dsi", "lat", "lon", "date"], sortby="date")
@convert_units(vsi="km/h", dsi="degrees", lat="degrees", lon="degrees")
def forward_discrepancy_array(
    lat: SequenceFloatType,
    lon: SequenceFloatType,
    date: SequenceDatetimeType,
    vsi: SequenceFloatType,
    dsi: SequenceFloatType,
) -> SequenceFloatType:
    """"""

    timediff = time_differences_array(date, np.roll(date, 1))
    lat1, lon1 = increment_position_array(
        np.roll(lat, 1), np.roll(lon, 1), np.roll(vsi, 1), dsi, timediff
    )

    lat2, lon2 = increment_position_array(lat, lon, vsi, dsi, timediff)

    updated_latitude = np.roll(lat, 1) + lat1 + lat2
    updated_longitude = np.roll(lon, 1) + lon1 + lon2

    # calculate distance between calculated position and the second reported position
    distance_from_est_location = sphere_distance_array(
        lat, lon, updated_latitude, updated_longitude
    )

    distance_from_est_location[0] = np.nan

    return distance_from_est_location


@inspect_arrays(["vsi", "dsi", "lat", "lon", "date"], sortby="date")
@convert_units(vsi="km/h", dsi="degrees", lat="degrees", lon="degrees")
def backward_discrepancy_array(
    lat: SequenceFloatType,
    lon: SequenceFloatType,
    date: SequenceDatetimeType,
    vsi: SequenceFloatType,
    dsi: SequenceFloatType,
) -> SequenceFloatType:
    """"""

    timediff = time_differences_array(date, np.roll(date, 1))
    lat2, lon2 = increment_position_array(
        np.roll(lat, 1),
        np.roll(lon, 1),
        np.roll(vsi, 1),
        np.roll(dsi, 1) - 180,
        timediff,
    )

    lat1, lon1 = increment_position_array(lat, lon, vsi, dsi - 180, timediff)

    updated_latitude = lat + lat1 + lat2
    updated_longitude = lon + lon1 + lon2

    # calculate distance between calculated position and the second reported position
    distance_from_est_location = sphere_distance_array(
        np.roll(lat, 1), np.roll(lon, 1), updated_latitude, updated_longitude
    )

    distance_from_est_location[-1] = np.nan

    return distance_from_est_location


def calculate_midpoint_array(lat, lon, timediff):
    number_of_obs = len(lat)
    midpoint_discrepancies = np.asarray([np.nan] * number_of_obs)  # type: np.ndarray

    t0 = timediff
    t1 = np.roll(timediff, -1)
    fraction_of_time_diff = t0 / (t0 + t1)
    fraction_of_time_diff[t0 + t1 == 0] = 0.0
    fraction_of_time_diff[np.isnan(t0)] = 0.0
    fraction_of_time_diff[np.isnan(t1)] = 0.0

    est_midpoint_lat, est_midpoint_lon = intermediate_point_array(
        np.roll(lat, 1),
        np.roll(lon, 1),
        np.roll(lat, -1),
        np.roll(lon, -1),
        fraction_of_time_diff,
    )

    est_midpoint_lat[0] = np.nan
    est_midpoint_lat[-1] = np.nan
    est_midpoint_lon[0] = np.nan
    est_midpoint_lon[-1] = np.nan

    midpoint_discrepancies = sphere_distance_array(
        lat, lon, est_midpoint_lat, est_midpoint_lon
    )

    return midpoint_discrepancies


@convert_units(
    lat_later="degrees",
    lat_earlier="degrees",
    lon_later="degrees",
    lon_earlier="degrees",
)
def calculate_course_parameters(
    lat_later: float,
    lat_earlier: float,
    lon_later: float,
    lon_earlier: float,
    date_later: datetime,
    date_earlier: datetime,
) -> tuple[float, float, float, float]:
    """Calculate course parameters.

    Parameters
    ----------
    lat_later: float
        Latitude in degrees of later timestamp.
    lat_earlier:float
        Latitude in degrees of earlier timestamp.
    lon_later: float
        Longitude in degrees of later timestamp.
    lon_earlier: float
        Longitude in degrees of earlier timestamp.
    date_later: datetime
        Date of later timestamp.
    date_earlier: datetime
        Date of earlier timestamp.

    Returns
    -------
    tuple of float
        A tuple of four floats representing the speed, distance, course and time difference
    """
    distance = sg.sphere_distance(lat_later, lon_later, lat_earlier, lon_earlier)
    date_earlier = pd.Timestamp(date_earlier)
    date_later = pd.Timestamp(date_later)

    timediff = time_control.time_difference(
        date_earlier.year,
        date_earlier.month,
        date_earlier.day,
        date_earlier.hour + date_earlier.minute / 60.0,
        date_later.year,
        date_later.month,
        date_later.day,
        date_later.hour + date_later.minute / 60.0,
    )
    if timediff != 0 and isvalid(timediff):
        speed = distance / abs(timediff)
    else:
        timediff = 0.0
        speed = distance

    course = sg.course_between_points(lat_earlier, lon_earlier, lat_later, lon_later)

    return speed, distance, course, timediff


@inspect_arrays(["lat", "lon", "date"])
@convert_units(lat="degrees", lon="degrees")
def calculate_speed_course_distance_time_difference(
    lat: SequenceFloatType,
    lon: SequenceFloatType,
    date: SequenceDatetimeType,
    alternating: bool = False,
) -> tuple[SequenceFloatType, SequenceFloatType, SequenceFloatType, SequenceFloatType]:
    """
    Calculates speeds, courses, distances and time differences using consecutive reports.

    Parameters
    ----------
    lat : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional latitude array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    lon : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional longitude array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    date : sequence of datetime, 1D np.ndarray of datetime, or pd.Series of datetime, shape (n,)
        One-dimensional date array.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    alternating : bool, default: False
        Whether to use alternating reports for calculation.

    Returns
    -------
    tuple of same types as input, each with float values, shape (n,)
        A tuple containing four one-dimensional arrays, sequences, or pandas Series of floats representing:
        speed, distance, course, and time difference.


    Raises
    ------
    ValueError
        If either input is not 1-dimensional or if their lengths do not match.
    """
    number_of_obs = len(lat)

    speed = np.empty(number_of_obs)  # type: np.ndarray
    course = np.empty(number_of_obs)  # type: np.ndarray
    distance = np.empty(number_of_obs)  # type: np.ndarray
    timediff = np.empty(number_of_obs)  # type: np.ndarray

    speed.fill(np.nan)
    course.fill(np.nan)
    distance.fill(np.nan)
    timediff.fill(np.nan)

    if number_of_obs == 1:
        return speed, distance, course, timediff

    range_end = number_of_obs
    first_entry_offset = 0
    second_entry_offset = -1
    if alternating:
        range_end = number_of_obs - 1
        first_entry_offset = 1

    for i in range(1, range_end):
        fe = i + first_entry_offset
        se = i + second_entry_offset
        ship_speed, ship_distance, ship_direction, ship_time_difference = (
            calculate_course_parameters(
                lat[fe], lat[se], lon[fe], lon[se], date[fe], date[se]
            )
        )

        speed[i] = ship_speed
        course[i] = ship_direction
        distance[i] = ship_distance
        timediff[i] = ship_time_difference

    return speed, distance, course, timediff


@inspect_arrays(["vsi", "dsi", "lat", "lon", "date"], sortby="date")
@convert_units(vsi="km/h", dsi="degrees", lat="degrees", lon="degrees")
def forward_discrepancy(
    lat: SequenceFloatType,
    lon: SequenceFloatType,
    date: SequenceDatetimeType,
    vsi: SequenceFloatType,
    dsi: SequenceFloatType,
) -> SequenceFloatType:
    """Calculate what the distance is between the projected position (based on the reported
    speed and heading at the current and previous time steps) and the actual position. The
    observations are taken in time order.

    This takes the speed and direction reported by the ship and projects it forwards half a
    time step, it then projects it forwards another half time-step using the speed and
    direction for the next report, to which the projected location
    is then compared. The distances between the projected and actual locations is returned

    Parameters
    ----------
    vsi : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional reported speed array in km/h.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    dsi : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional reported heading array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    lat : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional latitude array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    lon : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional longitude array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    date : sequence of datetime, 1D np.ndarray of datetime, or pd.Series of datetime, shape (n,)
        One-dimensional date array.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    Returns
    -------
    Same type as input, but with float values, shape (n,)
        One-dimensional array, sequence, or pandas Series containing distances from estimated positions.

    Raises
    ------
    ValueError
        If either input is not 1-dimensional or if their lengths do not match.
    """
    number_of_obs = len(lat)

    distance_from_est_location = np.asarray(
        [np.nan] * number_of_obs
    )  # type: np.ndarray

    for i in range(1, number_of_obs):

        vsi_current = vsi[i]
        vsi_previous = vsi[i - 1]
        dsi_current = dsi[i]
        dsi_previous = dsi[i - 1]
        tsi_current = pd.Timestamp(date[i])
        tsi_previous = pd.Timestamp(date[i - 1])
        lat_current = lat[i]
        lat_previous = lat[i - 1]
        lon_current = lon[i]
        lon_previous = lon[i - 1]

        if False in [
            isvalid(x)
            for x in [
                vsi_current,
                dsi_current,
                vsi_previous,
                dsi_previous,
                tsi_current,
                tsi_previous,
                lat_current,
                lat_previous,
                lon_current,
                lon_previous,
            ]
        ]:
            continue

        timediff = (tsi_current - tsi_previous).total_seconds() / 3600
        # get increment from initial position
        lat1, lon1 = increment_position(
            lat_previous,
            lon_previous,
            vsi_previous,
            dsi_current,
            timediff,
        )

        lat2, lon2 = increment_position(
            lat_current,
            lon_current,
            vsi_current,
            dsi_current,
            timediff,
        )

        # apply increments to the lat and lon at i-1
        updated_latitude = lat_previous + lat1 + lat2
        updated_longitude = lon_previous + lon1 + lon2

        # calculate distance between calculated position and the second reported position
        discrepancy = sg.sphere_distance(
            lat_current, lon_current, updated_latitude, updated_longitude
        )

        distance_from_est_location[i] = discrepancy

    return distance_from_est_location


@inspect_arrays(["vsi", "dsi", "lat", "lon", "date"], sortby="date")
@convert_units(vsi="km/h", dsi="degrees", lat="degrees", lon="degrees")
def backward_discrepancy(
    lat: SequenceFloatType,
    lon: SequenceFloatType,
    date: SequenceDatetimeType,
    vsi: SequenceFloatType,
    dsi: SequenceFloatType,
) -> SequenceFloatType:
    """Calculate what the distance is between the projected position (based on the reported speed and
    heading at the current and previous time steps) and the actual position. The calculation proceeds from the
    final, later observation to the first (in contrast to distr1 which runs in time order)

    This takes the speed and direction reported by the ship and projects it forwards half a time step, it then
    projects it forwards another half-time step using the speed and direction for the next report, to which the
    projected location is then compared. The distances between the projected and actual locations is returned

    Parameters
    ----------
    vsi : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional reported speed array in km/h.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    dsi : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional reported heading array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    lat : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional latitude array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    lon : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional longitude array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    date : sequence of datetime, 1D np.ndarray of datetime, or pd.Series of datetime, shape (n,)
        One-dimensional date array.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    Returns
    -------
    Same type as input, but with float values, shape (n,)
        One-dimensional array, sequence, or pandas Series containing distances from estimated positions.

    Raises
    ------
    ValueError
        If either input is not 1-dimensional or if their lengths do not match.
    """
    number_of_obs = len(lat)

    distance_from_est_location = np.asarray(
        [np.nan] * number_of_obs
    )  # type: np.ndarray

    for i in range(number_of_obs - 1, 0, -1):

        vsi_current = vsi[i]
        vsi_previous = vsi[i - 1]
        dsi_current = dsi[i]
        dsi_previous = dsi[i - 1]
        tsi_current = pd.Timestamp(date[i])
        tsi_previous = pd.Timestamp(date[i - 1])
        lat_current = lat[i]
        lat_previous = lat[i - 1]
        lon_current = lon[i]
        lon_previous = lon[i - 1]

        if False in [
            isvalid(x)
            for x in [
                vsi_current,
                dsi_current,
                vsi_previous,
                dsi_previous,
                tsi_current,
                tsi_previous,
                lat_current,
                lat_previous,
                lon_current,
                lon_previous,
            ]
        ]:
            continue

        timediff = (tsi_current - tsi_previous).total_seconds() / 3600
        # get increment from initial position - backwards in time means reversing the direction by 180 degrees
        lat1, lon1 = increment_position(
            lat_current,
            lon_current,
            vsi_current,
            dsi_current - 180.0,
            timediff,
        )

        lat2, lon2 = increment_position(
            lat_previous,
            lon_previous,
            vsi_previous,
            dsi_previous - 180.0,
            timediff,
        )

        # apply increments to the lat and lon at i-1
        updated_latitude = lat_current + lat1 + lat2
        updated_longitude = lon_current + lon1 + lon2

        # calculate distance between calculated position and the second reported position
        discrepancy = sg.sphere_distance(
            lat_previous, lon_previous, updated_latitude, updated_longitude
        )
        distance_from_est_location[i] = discrepancy

    # that fancy bit at the end reverses the array
    return distance_from_est_location[::-1]


@inspect_arrays(["lat", "lon", "timediff"])
@convert_units(lat="degrees", lon="degrees")
def calculate_midpoint(
    lat: SequenceFloatType,
    lon: SequenceFloatType,
    timediff: SequenceDatetimeType,
) -> SequenceFloatType:
    """Interpolate between alternate reports and compare the interpolated location to the actual location. e.g.
    take difference between reports 2 and 4 and interpolate to get an estimate for the position at the time
    of report 3. Then compare the estimated and actual positions at the time of report 3.

    The calculation linearly interpolates the latitudes and longitudes (allowing for wrapping around the
    dateline and so on).

    Parameters
    ----------
    lat : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional latitude array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    lon : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional longitude array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    timediff : sequence of datetime, 1D np.ndarray of datetime, or pd.Series of datetime, shape (n,)
        One-dimensional time difference array.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    Returns
    -------
    Same type as input, but with float values, shape (n,)
        One-dimensional array, sequence, or pandas Series of distances from estimated positions in kilometers.


    Raises
    ------
    ValueError
        If either input is not 1-dimensional or if their lengths do not match.
    """
    number_of_obs = len(lat)

    midpoint_discrepancies = np.asarray([np.nan] * number_of_obs)  # type: np.ndarray

    for i in range(1, number_of_obs - 1):
        t0 = timediff[i]
        t1 = timediff[i + 1]
        if t0 is not None and t1 is not None:
            if t0 + t1 != 0:
                fraction_of_time_diff = t0 / (t0 + t1)
            else:
                fraction_of_time_diff = 0.0
        else:
            fraction_of_time_diff = 0.0

        estimated_lat_at_midpoint, estimated_lon_at_midpoint = sg.intermediate_point(
            lat[i - 1],
            lon[i - 1],
            lat[i + 1],
            lon[i + 1],
            fraction_of_time_diff,
        )

        discrepancy = sg.sphere_distance(
            lat[i],
            lon[i],
            estimated_lat_at_midpoint,
            estimated_lon_at_midpoint,
        )

        midpoint_discrepancies[i] = discrepancy

    return midpoint_discrepancies
