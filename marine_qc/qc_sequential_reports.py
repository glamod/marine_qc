"""
QC of sequential reports
========================

Module containing QC functions for track checking which could be applied on a DataBundle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.ndimage import label

from .auxiliary import (
    SequenceDatetimeType,
    SequenceFloatType,
    SequenceIntType,
    convert_units,
    failed,
    inspect_arrays,
    isvalid,
    passed,
    post_format_return_type,
)
from .spherical_geometry import sphere_distance
from .track_check_utils import (
    calculate_course_parameters,
    calculate_speed_course_distance_time_difference,
    forward_discrepancy,
    backward_discrepancy,
    calculate_midpoint,
    modal_speed,
    set_speed_limits,
    check_distance_from_estimate,
    direction_continuity,
    speed_continuity,
)
from .time_control import time_difference


@post_format_return_type(["value"])
@inspect_arrays(["value", "lat", "lon", "date"], sortby="date")
@convert_units(lat="degrees", lon="degrees")
def do_spike_check(
    value: SequenceFloatType,
    lat: SequenceFloatType,
    lon: SequenceFloatType,
    date: SequenceDatetimeType,
    max_gradient_space: float,
    max_gradient_time: float,
    delta_t: float,
    n_neighbours: int,
) -> SequenceIntType:
    """Perform IQUAM-like spike check.

    Parameters
    ----------
    value : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
      One-dimensional array of values to be analyzed.
      Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    lat : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional array of latitudes in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    lon : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional array of longitudes in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    date : sequence of datetime, 1D np.ndarray of datetime, or pd.Series of datetime, shape (n,)
        One-dimensional array of datetime values.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    max_gradient_space : float, default: 0.5
        Maximum allowed spatial gradient.
        The unit is "units of value" per kilometer.

    max_gradient_time : float, default: 1.0
        Maximum allowed temporal gradient.
        The unit is "units of value" per hour.

    delta_t : float, default: 2.0
        Temperature delta used in the comparison.
        Typically set to 2.0 for ships and 1.0 for drifting buoys.

    n_neighbours : int, default: 5
        Number of neighboring points considered in the analysis.

    Returns
    -------
    Same type as input, but with integer values, shape (n,)
        One-dimensional array, sequence, or pandas Series of integer QC flags.
        - Returns array/sequence/Series of 1s if the spike check fails.
        - Returns array/sequence/Series of 0s otherwise.

    Raises
    ------
    ValueError
        If either input is not 1-dimensional or if their lengths do not match.

    Note
    ----
    In previous versions, default values for the parameters were:

    * max_gradient_space: float = 0.5
    * max_gradient_time: float = 1.0
    * delta_t: float = 2.0
    * n_neighbours: int = 5
    """
    gradient_violations = []
    count_gradient_violations = []

    number_of_obs = len(value)

    spike_qc = np.asarray([passed] * number_of_obs)  # type: np.ndarray

    for t1 in range(number_of_obs):

        violations_for_this_report = []
        count_violations_this_report = 0.0

        lo = max(0, t1 - n_neighbours)
        hi = min(number_of_obs, t1 + n_neighbours + 1)

        for t2 in range(lo, hi):

            if not isvalid(value[t1]) or not isvalid(value[t2]):
                continue

            distance = sphere_distance(lat[t1], lon[t1], lat[t2], lon[t2])
            delta = pd.Timestamp(date[t2]) - pd.Timestamp(date[t1])
            time_diff = abs(delta.days * 24 + delta.seconds / 3600.0)
            val_change = abs(value[t2] - value[t1])

            iquam_condition = max(
                [
                    delta_t,
                    abs(distance) * max_gradient_space,
                    abs(time_diff) * max_gradient_time,
                ]
            )

            if val_change > iquam_condition:
                violations_for_this_report.append(t2)
                count_violations_this_report += 1.0

        gradient_violations.append(violations_for_this_report)
        count_gradient_violations.append(count_violations_this_report)

    while np.sum(count_gradient_violations) > 0.0:
        most_fails = int(np.argmax(count_gradient_violations))
        spike_qc[most_fails] = failed

        for index in gradient_violations[most_fails]:
            if most_fails in gradient_violations[index]:
                gradient_violations[index].remove(most_fails)
                count_gradient_violations[index] -= 1.0

        count_gradient_violations[most_fails] = 0

    return spike_qc


@post_format_return_type(["vsi"])
@inspect_arrays(["vsi", "dsi", "lat", "lon", "date"], sortby="date")
@convert_units(vsi="km/h", dsi="degrees", lat="degrees", lon="degrees")
def do_track_check(
    vsi: SequenceFloatType,
    dsi: SequenceFloatType,
    lat: SequenceFloatType,
    lon: SequenceFloatType,
    date: SequenceDatetimeType,
    max_direction_change: float,
    max_speed_change: float,
    max_absolute_speed: float,
    max_midpoint_discrepancy: float,
) -> SequenceIntType:
    """Perform one pass of the track check.  This is an implementation of the MDS track check code
    which was originally written in the 1990s. I don't know why this piece of historic trivia so exercises
    my mind, but it does: the 1990s! I wish my code would last so long.

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

    max_direction_change : float, default: 60.0
      Maximum valid direction change in degrees.

    max_speed_change : float, default: 10.0
      Maximum valid speed change in km/h.

    max_absolute_speed : float, default: 40.0
      Maximum valid absolute speed in km/h.

    max_midpoint_discrepancy : float, default: 150.0
      Maximum valid midpoint discrepancy in meters.

    Returns
    -------
    Same type as input, but with integer values, shape (n,)
      One-dimensional array, sequence, or pandas Series of integer QC flags.
      - Returns array/sequence/Series of 1s if the track check fails.
      - Returns array/sequence/Series of 0s otherwise.

    Raises
    ------
    ValueError
        If either input is not 1-dimensional or if their lengths do not match.

    Note
    ----
    If number of observations is less than three, the track check always passes.

    In previous versions, the default values of the parameters were:

    * max_direction_change = 60.0
    * max_speed_change = 10.0
    * max_absolute_speed =  40.0
    * max_midpoint_discrepancy = 150.0
    """
    number_of_obs = len(lat)

    # no obs in, no qc outcomes out
    if number_of_obs == 0:
        return np.asarray([])

    # fewer than three obs - set the fewsome flag
    if number_of_obs < 3:
        return np.asarray([passed] * number_of_obs)

    # work out speeds and distances between alternating points
    speed_alt, _distance_alt, _course_alt, _timediff_alt = (
        calculate_speed_course_distance_time_difference(
            lat=lat,
            lon=lon,
            date=date,
            alternating=True,
        )
    )
    speed, _distance, course, timediff = (
        calculate_speed_course_distance_time_difference(
            lat=lat,
            lon=lon,
            date=date,
        )
    )

    # what are the mean and mode speeds?
    ms = modal_speed(speed)

    # set speed limits based on modal speed
    amax, _amaxx, _amin = set_speed_limits(ms)

    # compare reported speeds and positions if we have them
    forward_diff_from_estimated = forward_discrepancy(
        lat=lat,
        lon=lon,
        date=date,
        vsi=vsi,
        dsi=dsi,
    )
    reverse_diff_from_estimated = backward_discrepancy(
        lat=lat,
        lon=lon,
        date=date,
        vsi=vsi,
        dsi=dsi,
    )

    midpoint_diff_from_estimated = calculate_midpoint(
        lat=lat,
        lon=lon,
        timediff=timediff,
    )

    thisqc_a = np.zeros(number_of_obs)
    thisqc_b = np.zeros(number_of_obs)

    speed_alt_previous = np.roll(speed_alt, 1)
    speed_alt_next = np.roll(speed_alt, -1)
    speed_next = np.roll(speed, -1)

    selection1 = speed > amax
    selection2 = speed_alt_previous > amax
    selection_a = np.logical_and(selection1, selection2)

    selection1 = speed_next > amax
    selection2 = speed_alt_next > amax
    selection_b = np.logical_and(selection1, selection2)

    selection1 = speed > amax
    selection2 = speed_next > amax
    selection_c = np.logical_and(selection1, selection2)

    thisqc_a[selection_c] = thisqc_a[selection_c] + 3.00
    thisqc_a[selection_b] = thisqc_a[selection_b] + 2.00
    thisqc_a[selection_a] = thisqc_a[selection_a] + 1.00

    # Quality-control by examining the distance
    # between the calculated and reported second position.
    thisqc_b += check_distance_from_estimate(
        vsi, timediff, forward_diff_from_estimated, reverse_diff_from_estimated
    )
    # Check for continuity of direction
    thisqc_b += direction_continuity(dsi, course, max_direction_change)
    # Check for continuity of speed.
    thisqc_b += speed_continuity(vsi, speed, max_speed_change)

    thisqc_b[speed > max_absolute_speed] = thisqc_b[speed > max_absolute_speed] + 10.0

    fails = (
        (midpoint_diff_from_estimated > max_midpoint_discrepancy)
        & (thisqc_a > 0)
        & (thisqc_b > 0)
    )

    trk = np.where(fails, failed, passed)

    return trk


@post_format_return_type(["value"])
@inspect_arrays(["value"])
def do_few_check(
    value: SequenceFloatType,
) -> SequenceIntType:
    """Checks if number of observations is less than 3.

    Parameters
    ----------
    value: sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
        One-dimensional array of values to be analyzed.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    Returns
    -------
    Same type as input, but with integer values, shape (n,)
      One-dimensional array, sequence, or pandas Series of integer QC flags.
      - Returns array/sequence/Series of 1s if number of observations is less than 3.
      - Returns array/sequence/Series of 0s otherwise.

    Raises
    ------
    ValueError
        If either input is not 1-dimensional.
    """
    number_of_obs = len(value)

    # no obs in, no qc outcomes out
    if number_of_obs == 0:
        return [failed] * number_of_obs

    # fewer than three obs - set the fewsome flag
    if number_of_obs < 3:
        return [failed] * number_of_obs

    return [passed] * number_of_obs


@post_format_return_type(["at"])
@inspect_arrays(["at", "dpt", "lat", "lon", "date"], sortby="date")
@convert_units(at="K", dpt="K", lat="degrees", lon="degrees")
def find_saturated_runs(
    at: SequenceFloatType,
    dpt: SequenceFloatType,
    lat: SequenceFloatType,
    lon: SequenceFloatType,
    date: SequenceDatetimeType,
    min_time_threshold: float,
    shortest_run: int,
) -> SequenceIntType:
    """Perform checks on persistence of 100% rh while going through the voyage.
    While going through the voyage repeated strings of 100 %rh (AT == DPT) are noted.
    If a string extends beyond 20 reports and two days/48 hrs in time then all values are set to
    fail the repsat qc flag.

    Parameters
    ----------
    at : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
      One-dimensional air temperature array.
      Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    dpt : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
      One-dimensional dew point temperature array.
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

    min_time_threshold : float, default: 48.0
      Minimum time threshold in hours.

    shortest_run : int, default: 4
      Shortest number of observations.

    Returns
    -------
    Same type as input, but with integer values, shape (n,)
      One-dimensional array, sequence, or pandas Series of integer QC flags.
      - Returns array/sequence/Series of 1s if a saturated run is found.
      - Returns array/sequence/Series of 0s otherwise.

    Raises
    ------
    ValueError
        If either input is not 1-dimensional or if their lengths do not match.

    Note
    ----
    In previous version, default values for the parameters were:

    * min_time_threshold =  48.0
    * shortest_run = 4
    """
    saturated = at == dpt

    # Label contiguous runs of saturation
    labeled_array, num_features = label(saturated)

    # Initialize result array
    qc_flags = np.zeros_like(at, dtype=int)

    for run_id in range(1, num_features + 1):
        indices = np.where(labeled_array == run_id)[0]

        if len(indices) < shortest_run:
            continue

        i_start = indices[0]
        i_end = indices[-1]

        # Time difference in hours
        tdiff = time_difference(date[i_start], date[i_end])

        if tdiff >= min_time_threshold:
            qc_flags[indices] = 1

    return qc_flags


@post_format_return_type(["value"])
@inspect_arrays(["value"])
def find_multiple_rounded_values(
    value: SequenceFloatType, min_count: int, threshold: float
) -> SequenceIntType:
    """Find instances when more than "threshold" of the observations are
    whole numbers and set the 'round' flag. Used in the humidity QC
    where there are times when the values are rounded and this may
    have caused a bias.

    Parameters
    ----------
    value : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
      One-dimensional array of values.
      Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    min_count : int, default: 20
      Minimum number of rounded figures that will trigger the test.

    threshold : float, default: 0.5
      Minimum fraction of all observations that will trigger the test.

    Returns
    -------
    Same type as input, but with integer values, shape (n,)
      One-dimensional array, sequence, or pandas Series of integer QC flags.
      - Returns array/sequence/Series of 1s if the value is a whole number.
      - Returns array/sequence/Series of 0s otherwise.

    Note
    ----
    Previous versions had default values for the parameters of:

    * min_count = 20
    * threshold = 0.5
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"Invalid threshold: {threshold}. Must be between 0.0 and 1.0."
        )

    number_of_obs = len(value)

    if number_of_obs == 0:
        return [passed] * number_of_obs

    rounded = np.asarray([passed] * number_of_obs)  # type: np.ndarray

    valid_indices = isvalid(value)
    allcount = np.count_nonzero(valid_indices)
    if allcount <= min_count:
        return rounded

    # Find rounded values by checking where value mod 1 equals zero and set to failed if they exceed threshold
    rounded_values = np.equal(np.mod(value[valid_indices], 1), 0)
    cutoff = allcount * threshold
    if np.count_nonzero(rounded_values) > cutoff:
        rounded[valid_indices & rounded_values] = failed

    return rounded


@post_format_return_type(["value"])
@inspect_arrays(["value"])
def find_repeated_values(
    value: SequenceFloatType, min_count: int, threshold: float
) -> SequenceIntType:
    """Find cases where more than a given proportion of SSTs have the same value

    This function goes through a voyage and finds any cases where more than a threshold fraction of
    the observations have the same values for a specified variable.


    Parameters
    ----------
    value : sequence of float, 1D np.ndarray of float, or pd.Series of float, shape (n,)
      One-dimensional array of values.
      Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    min_count : int, default: 20
      Minimum number of repeated values that will trigger the test.

    threshold : float, default: 0.7
      Smallest fraction of all observations that will trigger the test.

    Returns
    -------
    Same type as input, but with integer values, shape (n,)
      One-dimensional array, sequence, or pandas Series of integer QC flags.
      - Returns array/sequence/Series of 1s if the value is repeated.
      - Returns array/sequence/Series of 0s otherwise.

    Note
    ----
    Previous versions had default values for the parameters of:

    * min_count = 20
    * threshold = 0.7
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"Invalid threshold: {threshold}. Must be between 0.0 and 1.0."
        )

    number_of_obs = len(value)

    if number_of_obs == 0:
        return [passed] * number_of_obs

    rep = np.asarray([passed] * number_of_obs)  # type: np.ndarray

    valid_indices = isvalid(value)

    allcount = np.count_nonzero(valid_indices)
    if allcount <= min_count:
        return rep

    _, unique_inverse, counts = np.unique(
        value[valid_indices], return_inverse=True, return_counts=True
    )
    cutoff = threshold * allcount
    exceedances = counts > cutoff
    exceedances = np.where(
        exceedances, failed, passed
    )
    pass_fail = exceedances[
        unique_inverse
    ]
    rep[valid_indices] = pass_fail

    return rep


@post_format_return_type(["lat"])
@inspect_arrays(["lat", "lon", "date"], sortby="date")
@convert_units(lat="degrees", lon="degrees")
def do_iquam_track_check(
    lat: SequenceFloatType,
    lon: SequenceFloatType,
    date: SequenceDatetimeType,
    speed_limit: float,
    delta_d: float,
    delta_t: float,
    n_neighbours: int,
) -> SequenceIntType:
    """Perform the IQUAM track check as detailed in Xu and Ignatov 2013

    The track check calculates speeds between pairs of observations and
    counts how many exceed a threshold speed. The ob with the most
    violations of this limit is flagged as bad and removed from the
    calculation. Then the next worst is found and removed until no
    violations remain.

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

    speed_limit : float
      Speed limit of platform in kilometers per hour.
      Typically, 60.0 for ships and 15.0 for drifting buoys.

    delta_d : float
      Latitude tolerance in degrees.

    delta_t : float
      Time tolerance in hundredths of an hour.

    n_neighbours : int
      Number of neighbouring points considered in the analysis.

    Returns
    -------
    Same type as input, but with integer values, shape (n,)
      One-dimensional array, sequence, or pandas Series of integer QC flags.
      - Returns array/sequence/Series of 1s if the IQUAM QC fails.
      - Returns array/sequence/Series of 0s otherwise.

    Raises
    ------
    ValueError
        If either input is not 1-dimensional or if their lengths do not match.

    Note
    ----
    Previous versions had default values for the parameters of:

    * speed_limit = 60.0 for ships and 15.0 for drifting buoys
    * delta_d = 1.11
    * delta_t = 0.01
    * n_neighbours = 5
    """
    number_of_obs = len(lat)

    if number_of_obs == 0:
        return [passed] * number_of_obs

    speed_violations = []
    count_speed_violations = []

    iquam_track = np.asarray([passed] * number_of_obs)  # type: np.ndarray

    for t1 in range(0, number_of_obs):
        violations_for_this_report = []
        count_violations_this_report = 0.0

        lo = max(0, t1 - n_neighbours)
        hi = min(number_of_obs, t1 + n_neighbours + 1)

        for t2 in range(lo, hi):

            _, distance, _, timediff = calculate_course_parameters(
                lat_later=lat[t2],
                lat_earlier=lat[t1],
                lon_later=lon[t2],
                lon_earlier=lon[t1],
                date_later=date[t2],
                date_earlier=date[t1],
            )

            iquam_condition = max([abs(distance) - delta_d, 0.0]) / (
                abs(timediff) + delta_t
            )

            if iquam_condition > speed_limit:
                violations_for_this_report.append(t2)
                count_violations_this_report += 1.0

        speed_violations.append(violations_for_this_report)
        count_speed_violations.append(count_violations_this_report)

    while np.sum(count_speed_violations) > 0.0:
        most_fails = int(np.argmax(count_speed_violations))
        iquam_track[most_fails] = failed

        for index in speed_violations[most_fails]:
            if most_fails in speed_violations[index]:
                speed_violations[index].remove(most_fails)
                count_speed_violations[index] -= 1.0

        count_speed_violations[most_fails] = passed

    return iquam_track
