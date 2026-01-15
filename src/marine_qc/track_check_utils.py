"""
The New Track Check QC module provides the functions needed to perform the track check.

The main routine is mds_full_track_check which takes a
list of class`.MarineReport` from a single ship and runs the track check on them.
This is an update of the MDS system track check in that it assumes the Earth is a
sphere. In practice, it gives similar results to the cylindrical earth formerly
assumed.
"""

from __future__ import annotations
from datetime import datetime

import numpy as np
import pandas as pd

from . import spherical_geometry as sg
from . import spherical_geometry as sph
from . import time_control
from .auxiliary import (
    SequenceDatetimeType,
    SequenceFloatType,
    convert_to,
    convert_units,
    inspect_arrays,
    isvalid,
    post_format_return_type,
)
from .spherical_geometry import (
    course_between_points,
    intermediate_point,
    sphere_distance,
)
from .time_control import time_difference


def modal_speed(speeds: list[float]) -> float:
    """
    Calculate the modal speed from the input array in 3 knot bins.

    Returns thebin-centre for the modal group.

    The data are binned into 3-knot bins with the first from 0-3 knots having a
    bin centre of 1.5 and the highest containing all speed in excess of 33 knots
    with a bin centre of 34.5. The bin with the most speeds in it is found. The higher of
    the modal speed or 8.5 is returned:

    Bins-   0-3, 3-6, 6-9, 9-12, 12-15, 15-18, 18-21, 21-24, 24-27, 27-30, 30-33, 33-36
    Centres-1.5, 4.5, 7.5, 10.5, 13.5,  16.5,  19.5,  22.5,  25.5,  28.5,  31.5,  34.5

    Parameters
    ----------
    speeds : list
        Input speeds in km/h.

    Returns
    -------
    float
        Bin-centre speed (expressed in km/h) for the 3 knot bin which contains most speeds in
        input array, or 8.5, whichever is higher.
    """
    # if there is one or no observations then return None
    # if the speed is on a bin edge then it rounds up to higher bin
    # if the modal speed is less than 8.50 then it is set to 8.50
    # anything exceeding 36 knots is assigned to the top bin
    if len(speeds) <= 1:
        return float(np.nan)

    # Convert km/h to knots
    speeds = np.asarray(speeds)
    speeds = np.asarray(convert_to(speeds, "km/h", "knots"), dtype=float)

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
    modal_speed_kmh = np.asarray(convert_to(modal_speed_knots, "knots", "km/h"), dtype=float)
    return float(modal_speed_kmh)


def set_speed_limits(amode: float) -> tuple[float, float, float]:
    """
    Take a modal speed and calculate speed limits for the track checker.

    Parameters
    ----------
    amode : float
        Modal speed in km/h.

    Returns
    -------
    (float, float, float)
        Max speed, maximum max speed and min speed.
    """
    amax = float(np.asarray(convert_to(15.0, "knots", "km/h"), dtype=float))
    amaxx = float(np.asarray(convert_to(20.0, "knots", "km/h"), dtype=float))
    amin = 0.00

    if not isvalid(amode):
        return amax, amaxx, amin

    if amode <= float(np.asarray(convert_to(8.51, "knots", "km/h"), dtype=float)):
        return amax, amaxx, amin

    return amode * 1.25, float(np.asarray(convert_to(30.0, "knots", "km/h"), dtype=float)), amode * 0.75


@post_format_return_type(["alat1"], dtype=float, multiple=True)
@inspect_arrays(["alat1", "alon1", "avs", "ads", "timediff"])
def increment_position(
    alat1: np.ndarray,
    alon1: np.ndarray,
    avs: np.ndarray,
    ads: np.ndarray,
    timediff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute latitude and longitude increments over half a time interval.

    This function takes latitudes and longitude, a speed, a direction and a time difference
    and returns increments of latitude and longitude which correspond to half the time difference.

    Parameters
    ----------
    alat1 : 1D np.ndarray of float
      One-dimensional array of Latitude at starting point in degrees.
    alon1 : 1D np.ndarray of float
      One-dimensional array of Longitude at starting point in degrees.
    avs : 1D np.ndarray of float
      One-dimensional array of speed of ship in km/h.
    ads : 1D np.ndarray of float
      One-dimensional array of heading of ship in degrees.
    timediff : 1D np.ndarray of float
      One-dimensional array of time difference between the points in hours.

    Returns
    -------
    1D np.ndarray of float
        Returns latitude and longitude increment or None and None if timediff is None.
    """
    alat1 = alat1.astype(float)
    alon1 = alon1.astype(float)
    avs = avs.astype(float)
    ads = ads.astype(float)
    timediff = timediff.astype(float)

    distance = avs * timediff / 2.0
    lat, lon = sph.lat_lon_from_course_and_distance(alat1, alon1, ads, distance)
    lat = lat - alat1
    lon = lon - alon1

    return lat, lon


@post_format_return_type(["dsi"], dtype=float)
@inspect_arrays(["dsi", "directions"])
def direction_continuity(
    dsi: np.ndarray,
    directions: np.ndarray,
    dsi_previous: np.ndarray | None = None,
    max_direction_change: float = 60.0,
) -> np.ndarray:
    """
    Check if reported and calculated directions are within the allowed change.

    This function compares the heading at the previous time step with the
    calculated ship direction from reported positions, flagging differences
    that exceed the maximum allowed direction change.

    Parameters
    ----------
    dsi : 1D np.ndarray of float
        Heading at current time step in degrees.
    directions : 1D np.ndarray of float
        Calculated ship direction from reported positions in degrees.
    dsi_previous : 1D np.ndarray of float
        Heading at previous time step in degrees.
        If None, get dsi_previous from dsi.
    max_direction_change : float
        Largest deviations that will not be flagged in degrees.

    Returns
    -------
    np.ndarray
        Returned array elements are 10.0 if the difference between reported and calculated direction is greater
        than the max_direction_change (default, 60 degrees), 0.0 otherwise.
    """
    allowed_list = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    result = np.zeros(len(dsi))
    if not isvalid(max_direction_change):
        return result

    valid = np.isin(dsi, allowed_list)

    if dsi_previous is None:
        dsi_previous = np.roll(dsi, 1)
        dsi_previous = dsi_previous.astype(float)
        dsi_previous[0] = np.nan
    else:
        valid &= np.isin(dsi_previous, allowed_list)

    dsi = dsi.astype(float)
    dsi_previous = np.atleast_1d(dsi_previous).astype(float)
    directions = directions.astype(float)

    selection1 = max_direction_change < abs(dsi - directions)
    selection2 = abs(dsi - directions) < (360 - max_direction_change)
    selection3 = max_direction_change < abs(dsi_previous - directions)
    selection4 = abs(dsi_previous - directions) < (360 - max_direction_change)

    result[np.logical_and(selection1, selection2)] = 10.0
    result[np.logical_and(selection3, selection4)] = 10.0

    result[~valid] = np.nan

    return result


@post_format_return_type(["vsi"], dtype=float)
@inspect_arrays(["vsi", "speeds"])
def speed_continuity(
    vsi: np.ndarray,
    speeds: np.ndarray,
    vsi_previous: np.ndarray | None = None,
    max_speed_change: float | None = 10.0,
) -> np.ndarray:
    """
    Check if reported speeds are within the allowed change from calculated speeds.

    This function compares the reported speed at the current and previous time steps
    with the speed calculated from positions. Flags positions where the change
    exceeds the maximum allowed speed change.

    Parameters
    ----------
    vsi : 1D np.ndarray of float
        One-dimensional array of reported speed in km/h at current time step.
    speeds : 1D np.ndarray of float
        One-dimensional array of speed of ship calculated from locations
        at current and previous time steps in km/h.
    vsi_previous : 1D np.ndarray of float, optional
        One-dimensional array of reported speed in km/h at previous time step.
        If None, get vsi_previous from vsi.
    max_speed_change : float, optional
        Largest change of speed that will not raise flag in km/h, default 10 (km/h).

    Returns
    -------
    np.ndarray
        Returned array elements are 10 if the reported and calculated speeds differ by more than 10 knots,
        0 otherwise.
    """
    result = np.zeros(len(vsi))
    if not isvalid(max_speed_change):
        return result

    valid = isvalid(vsi)

    if vsi_previous is None:
        vsi_previous = np.roll(vsi, 1)
        vsi_previous = vsi_previous.astype(float)
        vsi_previous[0] = np.nan
    else:
        valid &= isvalid(vsi_previous)

    vsi = vsi.astype(float)
    vsi_previous = np.atleast_1d(vsi_previous).astype(float)
    speeds = speeds.astype(float)

    selection1 = abs(vsi - speeds) > max_speed_change
    selection2 = abs(vsi_previous - speeds) > max_speed_change
    result[np.logical_and(selection1, selection2)] = 10.0

    result[~valid] = np.nan

    return result


@post_format_return_type(["vsi"], dtype=float)
@inspect_arrays(["vsi", "time_differences", "fwd_diff_from_estimated", "rev_diff_from_estimated"])
def check_distance_from_estimate(
    vsi: np.ndarray,
    time_differences: np.ndarray,
    fwd_diff_from_estimated: np.ndarray,
    rev_diff_from_estimated: np.ndarray,
    vsi_previous: np.ndarray | None = None,
) -> np.ndarray:
    """
    Check that distances from estimated positions are less than calculated distance.

    The estimated positions are calculated forward and backwards in time.
    The calculated distance is the time difference multiplied by the average reported speeds.

    Parameters
    ----------
    vsi : 1D np.ndarray of float
        Reported speed in km/h at current time step.
    time_differences : 1D np.ndarray of float
        Calculated time differences between reports in hours.
    fwd_diff_from_estimated : 1D np.ndarray of float
        Distance in km from estimated position, estimates made forward in time.
    rev_diff_from_estimated : 1D np.ndarray of float
        Distance in km from estimated position, estimates made backward in time.
    vsi_previous : 1D np.ndarray of float, optional
        One-dimensional array of reported speed in km/h at previous time step.
        If None, get vsi_previous from vsi.

    Returns
    -------
    np.ndarray
        Returned array elements set to 10 if estimated and reported positions differ by more than the reported
        speed multiplied by the calculated time difference, 0 otherwise.
    """
    valid = isvalid(vsi)

    if vsi_previous is None:
        vsi_previous = np.roll(vsi, 1)
        vsi_previous[0] = np.nan
    else:
        valid &= isvalid(vsi_previous)

    vsi = vsi.astype(float)
    time_differences = time_differences.astype(float)
    fwd_diff_from_estimated = fwd_diff_from_estimated.astype(float)
    rev_diff_from_estimated = rev_diff_from_estimated.astype(float)
    vsi_previous = np.atleast_1d(vsi_previous).astype(float)

    alwdis = time_differences * ((vsi + vsi_previous) / 2.0)

    selection = fwd_diff_from_estimated > alwdis
    selection = np.logical_and(selection, rev_diff_from_estimated > alwdis)
    selection = np.logical_and(selection, vsi > 0)
    selection = np.logical_and(selection, vsi_previous > 0)
    selection = np.logical_and(selection, time_differences > 0)

    result = np.zeros(len(vsi))
    result[selection] = 10.0

    return result


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
    """
    Calculate course parameters.

    Parameters
    ----------
    lat_later : float
        Latitude in degrees of later timestamp.
    lat_earlier : float
        Latitude in degrees of earlier timestamp.
    lon_later : float
        Longitude in degrees of later timestamp.
    lon_earlier : float
        Longitude in degrees of earlier timestamp.
    date_later : datetime
        Date of later timestamp.
    date_earlier : datetime
        Date of earlier timestamp.

    Returns
    -------
    tuple of float
        A tuple of four floats representing the speed, distance, course and time difference.
    """
    distance = sg.sphere_distance(lat_later, lon_later, lat_earlier, lon_earlier)
    date_earlier = pd.Timestamp(date_earlier)
    date_later = pd.Timestamp(date_later)

    timediff = time_control.time_difference(
        date_earlier,
        date_later,
    )
    if timediff != 0 and isvalid(timediff):
        speed = distance / abs(timediff)
    else:
        timediff = 0.0
        speed = distance

    course = sg.course_between_points(lat_earlier, lon_earlier, lat_later, lon_later)

    return speed, distance, course, timediff


@post_format_return_type(["lat"], dtype=float, multiple=True)
@inspect_arrays(["lat", "lon", "date"])
@convert_units(lat="degrees", lon="degrees")
def calculate_speed_course_distance_time_difference(
    lat: np.ndarray,
    lon: np.ndarray,
    date: np.ndarray,
    alternating: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate speeds, courses, distances and time differences using consecutive reports.

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
    tuple of np.ndarray, each with float values, shape (n,)
        A tuple containing four one-dimensional arrays representing: speed, distance, course, and time difference.
    """
    if alternating:
        distance = sphere_distance(np.roll(lat, 1), np.roll(lon, 1), np.roll(lat, -1), np.roll(lon, -1))
        timediff = time_difference(np.roll(date, 1), np.roll(date, -1))
        course = course_between_points(np.roll(lat, 1), np.roll(lon, 1), np.roll(lat, -1), np.roll(lon, -1))
        # Alternating estimates are unavailable for the first and last elements
        distance[0] = np.nan
        distance[-1] = np.nan
        timediff[0] = np.nan
        timediff[-1] = np.nan
        course[0] = np.nan
        course[-1] = np.nan
    else:
        distance = sphere_distance(np.roll(lat, 1), np.roll(lon, 1), lat, lon)
        timediff = time_difference(np.roll(date, 1), date)
        course = course_between_points(np.roll(lat, 1), np.roll(lon, 1), lat, lon)
        # With the regular first differences, we don't have anything for the first element
        distance[0] = np.nan
        timediff[0] = np.nan
        course[0] = np.nan

    speed = np.zeros_like(timediff)
    valid = timediff != 0.0
    speed[valid] = distance[valid] / timediff[valid]

    return speed, distance, course, timediff


@post_format_return_type(["vsi"], dtype=float)
@inspect_arrays(["lat", "lon", "date", "vsi", "dsi"], sortby="date")
@convert_units(vsi="km/h", dsi="degrees", lat="degrees", lon="degrees")
def forward_discrepancy(
    lat: SequenceFloatType,
    lon: SequenceFloatType,
    date: SequenceDatetimeType,
    vsi: SequenceFloatType,
    dsi: SequenceFloatType,
) -> SequenceFloatType:
    """
    Calculate the distance between the projected position and the actual position.

    The projected position is based on the reported speed and heading at the current
    and previous time steps. The observations are taken in time order.

    This takes the speed and direction reported by the ship and projects it forwards half a
    time step, it then projects it forwards another half time-step using the speed and
    direction for the next report, to which the projected location
    is then compared. The distances between the projected and actual locations is returned

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

    vsi : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional reported speed array in km/h.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    dsi : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional reported heading array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    Returns
    -------
    Same type as input, but with float values, shape (n,)
        One-dimensional array, sequence, or pandas Series containing distances from estimated positions.

    Raises
    ------
    ValueError
        - If either input is not 1-dimensional or if their lengths do not match.
        - If decorator `inspect_arrays` does not return np.ndarrays.
    """
    if not isinstance(lat, np.ndarray):
        raise TypeError(f"'lat' must be a numpy.ndarray, got {type(lat).__name__}")
    if not isinstance(lon, np.ndarray):
        raise TypeError(f"'lon' must be a numpy.ndarray, got {type(lon).__name__}")
    if not isinstance(date, np.ndarray):
        raise TypeError(f"'date' must be a numpy.ndarray, got {type(date).__name__}")
    if not isinstance(vsi, np.ndarray):
        raise TypeError(f"'vsi' must be a numpy.ndarray, got {type(vsi).__name__}")
    if not isinstance(dsi, np.ndarray):
        raise TypeError(f"'dsi' must be a numpy.ndarray, got {type(dsi).__name__}")

    timediff = time_difference(np.roll(date, 1), date)
    lat1, lon1 = increment_position(np.roll(lat, 1), np.roll(lon, 1), np.roll(vsi, 1), dsi, timediff)

    lat2, lon2 = increment_position(lat, lon, vsi, dsi, timediff)

    updated_latitude = np.roll(lat, 1) + lat1 + lat2
    updated_longitude = np.roll(lon, 1) + lon1 + lon2

    # calculate distance between calculated position and the second reported position
    distance_from_est_location = sphere_distance(lat, lon, updated_latitude, updated_longitude)

    distance_from_est_location[0] = np.nan

    return distance_from_est_location


@post_format_return_type(["vsi"], dtype=float)
@inspect_arrays(["lat", "lon", "date", "vsi", "dsi"], sortby="date")
@convert_units(vsi="km/h", dsi="degrees", lat="degrees", lon="degrees")
def backward_discrepancy(
    lat: SequenceFloatType,
    lon: SequenceFloatType,
    date: SequenceDatetimeType,
    vsi: SequenceFloatType,
    dsi: SequenceFloatType,
) -> SequenceFloatType:
    """
    Calculate the distance between the projected position and the actual position.

    The projected position is based on the reported speed and heading at the current
    and previous time steps. The calculation proceeds from the
    final, later observation to the first (in contrast to distr1 which runs in time order)

    This takes the speed and direction reported by the ship and projects it forwards half a time step, it then
    projects it forwards another half-time step using the speed and direction for the next report, to which the
    projected location is then compared. The distances between the projected and actual locations is returned

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

    vsi : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional reported speed array in km/h.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    dsi : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional reported heading array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    Returns
    -------
    Same type as input, but with float values, shape (n,)
        One-dimensional array, sequence, or pandas Series containing distances from estimated positions.

    Raises
    ------
    ValueError
        - If either input is not 1-dimensional or if their lengths do not match.
        - If decorator `inspect_arrays` does not return np.ndarrays.
    """
    if not isinstance(lat, np.ndarray):
        raise TypeError(f"'lat' must be a numpy.ndarray, got {type(lat).__name__}")
    if not isinstance(lon, np.ndarray):
        raise TypeError(f"'lon' must be a numpy.ndarray, got {type(lon).__name__}")
    if not isinstance(date, np.ndarray):
        raise TypeError(f"'date' must be a numpy.ndarray, got {type(date).__name__}")
    if not isinstance(vsi, np.ndarray):
        raise TypeError(f"'vsi' must be a numpy.ndarray, got {type(vsi).__name__}")
    if not isinstance(dsi, np.ndarray):
        raise TypeError(f"'dsi' must be a numpy.ndarray, got {type(dsi).__name__}")

    timediff = time_difference(np.roll(date, 1), date)
    lat2, lon2 = increment_position(
        np.roll(lat, 1),
        np.roll(lon, 1),
        np.roll(vsi, 1),
        np.roll(dsi, 1) - 180,
        timediff,
    )

    lat1, lon1 = increment_position(lat, lon, vsi, dsi - 180, timediff)

    updated_latitude = lat + lat1 + lat2
    updated_longitude = lon + lon1 + lon2

    # calculate distance between calculated position and the second reported position
    distance_from_est_location = sphere_distance(np.roll(lat, 1), np.roll(lon, 1), updated_latitude, updated_longitude)

    distance_from_est_location[-1] = np.nan

    return distance_from_est_location


@post_format_return_type(["lat"], dtype=float)
@inspect_arrays(["lat", "lon", "timediff"])
@convert_units(lat="degrees", lon="degrees")
def calculate_midpoint(
    lat: np.ndarray,
    lon: np.ndarray,
    timediff: np.ndarray,
) -> np.ndarray:
    """
    Interpolate between alternate reports and compare the interpolated location to the actual location.

    E.g. take difference between reports 2 and 4 and interpolate to get an estimate for the position
    at the time of report 3. Then compare the estimated and actual positions at the time of report 3.

    The calculation linearly interpolates the latitudes and longitudes (allowing for wrapping around the
    dateline and so on).

    Parameters
    ----------
    lat : 1D np.ndarray of float
        One-dimensional latitude array in degrees.

    lon : 1D np.ndarray of float
        One-dimensional longitude array in degrees.

    timediff : 1D np.ndarray of datetime
        One-dimensional time difference array.

    Returns
    -------
    1D np.ndarray of float
        One-dimensional array of distances from estimated positions in kilometers.

    Raises
    ------
    ValueError
        If either input is not 1-dimensional or if their lengths do not match.
    """
    lat = lat.astype(float)
    lon = lon.astype(float)
    timediff = timediff.astype(float)

    number_of_obs = len(lat)
    midpoint_discrepancies = np.asarray([np.nan] * number_of_obs)  # type: np.ndarray

    t0 = timediff
    t1 = np.roll(timediff, -1)
    fraction_of_time_diff = np.zeros_like(t0)
    valid = (t0 + t1 != 0) & isvalid(t0) & isvalid(t1)
    fraction_of_time_diff[valid] = t0[valid] / (t0[valid] + t1[valid])

    est_midpoint_lat, est_midpoint_lon = intermediate_point(
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

    midpoint_discrepancies = sphere_distance(lat, lon, est_midpoint_lat, est_midpoint_lon)

    return midpoint_discrepancies
