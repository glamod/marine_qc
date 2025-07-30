"""
The New Track Check QC module provides the functions needed to perform the
track check. The main routine is mds_full_track_check which takes a
list of class`.MarineReport` from a single ship and runs the track check on them.
This is an update of the MDS system track check in that it assumes the Earth is a
sphere. In practice, it gives similar results to the cylindrical earth formerly
assumed.
"""

from __future__ import annotations

import math

import numpy as np

from . import spherical_geometry as sph
from .auxiliary import convert_to, isvalid


def modal_speed(awork: list) -> float:
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
    awork : list
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
    ikmode = -32768
    acint = []
    ifreq = []
    for i in range(1, 13):
        acint.append(i * 3.0)
        ifreq.append(0.0)

    ntime = len(awork)
    atmode = 0
    icmode = 0
    amode = np.nan
    awork = convert_to(awork, "km/h", "knots")
    if ntime > 1:
        for i in range(1, ntime):
            # fixed so that indexing starts at zero
            index = int(math.floor(awork[i] / 3.0))
            if index < 0:
                index = 0
            elif index > 11:
                index = 11
            ifreq[index] = ifreq[index] + 1

        for index in range(0, 12):
            if ifreq[index] > ikmode:
                ikmode = ifreq[index]
                icmode = 1
                atmode = acint[index] - 1.50

        amode = atmode / icmode
        amode = max(amode, 8.5)

    return convert_to(amode, "knots", "km/h")


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
