"""Marine QC tracking module."""

# noqa: S101

from __future__ import annotations

import math
import warnings
from datetime import datetime
from typing import Sequence

import numpy as np

from .astronomical_geometry import sunangle
from .auxiliary import failed, isvalid, passed, untestable, untested
from .qc_sequential_reports import do_iquam_track_check, inspect_arrays
from .spherical_geometry import sphere_distance
from .statistics import trim_mean, trim_std
from .time_control import convert_date_to_hours, dayinyear

"""
The trackqc module contains a set of functions for performing the tracking QC
first described in Atkinson et al. [2013]. The general procedures described
in Atkinson et al. [2013] were later revised and improved for the SST CCI 2 project.
Documentation and IDL code for the revised (and original) procedures can be found
in the CMA FCM code repository. The code in this module represents a port of the
revised IDL code into the python marine QC suite. New versions of the aground
and speed checks have also been added.

These functions perform tracking QC checks on a :py:class:`ex.Voyage`.

References:

Atkinson, C.P., N.A. Rayner, J. Roberts-Jones, R.O. Smith, 2013:
Assessing the quality of sea surface temperature observations from
drifting buoys and ships on a platform-by-platform basis (doi:10.1002/jgrc.20257).
"""


def track_day_test(
    year: int,
    month: int,
    day: int,
    hour: float,
    lat: float,
    lon: float,
    elevdlim: float = -2.5,
) -> bool:
    """Given date, time, lat and lon calculate if the sun elevation is > elevdlim.
    If so return daytime is True

    This is the "day" test used by tracking QC to decide whether an SST measurement is night or day.
    This is important because daytime diurnal heating can affect comparison with an SST background.
    It uses the function sunangle to calculate the elevation of the sun. A default solar_zenith angle
    of 92.5 degrees (elevation of -2.5 degrees) delimits night from day.

    Parameters
    ----------
    year: int
        Year
    month: int
        Month
    day: int
        Day
    hour: float
        Hour expressed as decimal fraction (e.g. 20.75 = 20:45 pm)
    lat: float
        Latitude in degrees
    lon: float
        Longitude in degrees
    elevdlim: float, default: -2.5
        Elevation day/night delimiter in degrees above horizon

    Returns
    -------
    bool
        True if daytime, else False.

    Raises
    ------
    ValueError
        If either year, month, day, hour, lat or lon is numerically invalid or None
        of if either month, day, hour or lat is not in valid range.
    """
    if not isvalid(year):
        raise ValueError("year is missing")
    if not isvalid(month):
        raise ValueError("month is missing")
    if not isvalid(day):
        raise ValueError("day is missing")
    if not isvalid(hour):
        raise ValueError("hour is missing")
    if not isvalid(lat):
        raise ValueError("lat is missing")
    if not isvalid(lon):
        raise ValueError("lon is missing")
    if not (1 <= month <= 12):
        raise ValueError("Month not in range 1-12")
    if not (1 <= day <= 31):
        raise ValueError("Day not in range 1-31")
    if not (0 <= hour <= 24):
        raise ValueError("Hour not in range 0-24")
    if not (90 >= lat >= -90):
        raise ValueError("Latitude not in range -90 to 90")

    daytime = False

    year2 = year
    day2 = dayinyear(year, month, day)
    hour2 = math.floor(hour)
    minute2 = (hour - math.floor(hour)) * 60.0
    lat2 = lat
    lon2 = lon
    if lat == 0:
        lat2 = 0.0001
    if lon == 0:
        lon2 = 0.0001

    _, elevation, _, _, _, _ = sunangle(
        year2, day2, hour2, minute2, 0, 0, 0, lat2, lon2
    )

    if elevation > elevdlim:
        daytime = True

    return daytime


def is_monotonic(inarr: Sequence[float]) -> bool:
    """
    Tests if elements in an array are increasing monotonically. i.e. each element
    is greater than or equal to the preceding element.

    Parameters
    ----------
    inarr: array-like of datetime, shape (n,)
        1-dimensional date array.

    Returns
    -------
    bool
        True if array is increasing monotonically, False otherwise
    """
    for i in range(1, len(inarr)):
        if inarr[i] < inarr[i - 1]:
            return False
    return True


class SpeedChecker:
    """Check to see whether a drifter has been picked up by a ship (out of water) based on 1/100th degree
    precision positions. A flag 'drf_spd' is set for each input report: flag=1 for reports deemed picked up,
    else flag=0.

    A drifter is deemed picked up if it is moving faster than might be expected for a fast ocean current
    (a few m/s). Unreasonably fast movement is detected when speed of travel between report-pairs exceeds
    the chosen 'speed_limit' (speed is estimated as distance between reports divided by time separation -
    this 'straight line' speed between the two points is a minimum speed estimate given a less-direct
    path may have been followed). Positional errors introduced by lon/lat 'jitter' and data precision
    can be of order several km's. Reports must be separated by a suitably long period of time (the 'min_win_period')
    to minimise the effect of these errors when calculating speed e.g. for reports separated by 24 hours
    errors of several cm/s would result which are two orders of magnitude less than a fast ocean current
    which seems reasonable. Conversely, the period of time chosen should not be too long so as to resolve
    short-lived burst of speed on manoeuvring ships. Larger positional errors may also trigger the check.
    Because temporal sampling can be erratic the time period over which this assessment is made is specified
    as a range (bound by 'min_win_period' and 'max_win_period') - assessment uses the longest time separation
    available within this range.

    IMPORTANT - for optimal performance, drifter records with observations failing this check should be
    subsequently manually reviewed. Ships move around in all sorts of complicated ways that can readily
    confuse such a simple check (e.g. pausing at sea, crisscrossing its own path) and once some erroneous
    movement is detected it is likely a human operator can then better pick out the actual bad data. False
    fails caused by positional errors (particularly in fast ocean currents) will also need reinstating.
    """

    def __init__(
        self,
        lons: Sequence[float],
        lats: Sequence[float],
        dates: Sequence[datetime],
        speed_limit: float,
        min_win_period: float,
        max_win_period: float,
    ):
        """
        Create an object for performing the Speed Check.

        Parameters
        ----------
        lons: array-like of float, shape (n,)
            1-dimensional longitude array in degrees.
        lats: array-like of float, shape (n,)
            1-dimensional latitude array in degrees.
        dates: array-like of datetime, shape (n,)
            1-dimensional date array.
        speed_limit: float
            maximum allowable speed for an in situ drifting buoy (metres per second)
        min_win_period: float
            minimum period of time in days over which position is assessed for speed estimates (see
            description)
        max_win_period: float
            maximum period of time in days over which position is assessed for speed estimates
            (this should be greater than min_win_period and allow for some erratic temporal sampling e.g.
            min_win_period + 0.2 to allow for gaps of up to 0.2 - days in sampling).
        """
        self.lon = lons
        self.lat = lats
        self.nreps = len(lons)
        self.hrs = convert_date_to_hours(dates)

        self.speed_limit = speed_limit
        self.min_win_period = min_win_period
        self.max_win_period = max_win_period

        # Initialise QC outcomes with untested
        self.qc_outcomes = np.zeros(self.nreps) + untested

    def get_qc_outcomes(self):
        """Return the QC outcomes"""
        return self.qc_outcomes

    def valid_parameters(self) -> bool:
        """Check the parameters are valid. Raises a warning and returns False if not valid"""
        valid = True

        try:
            assert self.speed_limit >= 0, "speed_limit must be >= 0"
            assert self.min_win_period >= 0, "min_win_period must be >= 0"
            assert self.max_win_period >= 0, "max_win_period must be >= 0"
            assert (
                self.max_win_period >= self.min_win_period
            ), "max_win_period must be >= min_win_period"
        except AssertionError as error:
            warnings.warn(UserWarning("invalid input parameter: " + str(error)))
            valid = False

        return valid

    def valid_arrays(self) -> bool:
        """Check the input arrays are valid. Raises a warning and returns False if not valid"""
        valid = True
        try:
            assert not any(np.isnan(self.lon)), "Nan(s) found in longitude"
            assert not any(np.isnan(self.lat)), "Nan(s) found in latitude"
            assert not any(np.isnan(self.hrs)), "Nan(s) found in time differences"
            assert is_monotonic(self.hrs), "times are not sorted"
        except AssertionError as error:
            warnings.warn(UserWarning("problem with report values: " + str(error)))
            valid = False
        return valid

    def do_speed_check(self):
        """Perform the actual speed check"""
        nrep = self.nreps
        min_win_period_hours = self.min_win_period * 24.0
        max_win_period_hours = self.max_win_period * 24.0

        if not self.valid_arrays() or not self.valid_parameters():
            self.qc_outcomes = np.zeros(self.nreps) + untestable
            return

        # Default to pass
        self.qc_outcomes = np.zeros(nrep) + passed

        # loop through timeseries to see if drifter is moving too fast
        # and flag any occurrences
        index_arr = np.array(range(0, nrep))  # type: np.ndarray
        i = 0
        time_to_end = self.hrs[-1] - self.hrs[i]
        while time_to_end >= min_win_period_hours:
            # Find all time points before current time plus the max window period
            f_win = self.hrs <= self.hrs[i] + max_win_period_hours
            # Window length is the difference between the latest time point in
            # the window and the current time
            win_len = self.hrs[f_win][-1] - self.hrs[i]
            # If the actual window length is shorter than the minimum window period
            # then go to the next time step
            if win_len < min_win_period_hours:
                i += 1
                time_to_end = self.hrs[-1] - self.hrs[i]
                continue

            # If the actual window length is long enough then calculate the speed
            # based on the first and last points in the window
            displace = sphere_distance(
                self.lat[i], self.lon[i], self.lat[f_win][-1], self.lon[f_win][-1]
            )
            speed = displace / win_len  # km per hr
            speed = speed * 1000.0 / (60.0 * 60)  # metres per sec

            # If the average speed during the window is too high then set all
            # flags in the window to failed.
            if speed > self.speed_limit:
                self.qc_outcomes[i : index_arr[f_win][-1] + 1] = failed

            i += 1
            time_to_end = self.hrs[-1] - self.hrs[i]

        return


class NewSpeedChecker:
    """Check to see whether a drifter has been picked up by a ship (out of water) based on 1/100th degree
    precision positions. A flag 'drf_spd' is set for each input report: flag=1 for reports deemed picked up,
    else flag=0.

    A drifter is deemed picked up if it is moving faster than might be expected for a fast ocean current
    (a few m/s). Unreasonably fast movement is detected when speed of travel between report-pairs exceeds
    the chosen 'speed_limit' (speed is estimated as distance between reports divided by time separation -
    this 'straight line' speed between the two points is a minimum speed estimate given a less-direct
    path may have been followed). Positional errors introduced by lon/lat 'jitter' and data precision
    can be of order several km's. Reports must be separated by a suitably long period of time (the 'min_win_period')
    to minimise the effect of these errors when calculating speed e.g. for reports separated by 9 hours
    errors of order 10 cm/s would result which are a few percent of fast ocean current speed. Conversley,
    the period of time chosen should not be too long so as to resolve short-lived burst of speed on
    manouvering ships. Larger positional errors may also trigger the check.

    For each report, speed is assessed over the shortest available period that exceeds 'min_win_period'.

    Prior to assessment the drifter record is screened for positional errors using the iQuam track check
    method (from :py:class:`ex.Voyage`). When running the iQuam check the record is treated as a ship (not a
    drifter) so as to avoid accidentally filtering out observations made aboard a ship (which is what we
    are trying to detect). This iQuam track check does not overwrite any existing iQuam track check flags.

    IMPORTANT - for optimal performance, drifter records with observations failing this check should be
    subsequently manually reviewed. Ships move around in all sorts of complicated ways that can readily
    confuse such a simple check (e.g. pausing at sea, crisscrossing its own path) and once some erroneous
    movement is detected it is likely a human operator can then better pick out the actual bad data. False
    fails caused by positional errors (particularly in fast ocean currents) will also need reinstating.

    The class has the following class attributes which can be modified using the set_parameters method.

    iquam_parameters: Parameter dictionary for Voyage.iquam_track_check() function.
    speed_limit: maximum allowable speed for an in situ drifting buoy (metres per second)
    min_win_period: minimum period of time in days over which position is assessed for speed estimates (see
    description)
    """

    def __init__(
        self,
        lons: Sequence[float],
        lats: Sequence[float],
        dates: Sequence[datetime],
        speed_limit: float,
        min_win_period: float,
        ship_speed_limit: float,
        delta_d: float,
        delta_t: float,
        n_neighbours: int,
    ):
        """
        Object used to perform the new speed check

        Parameters
        ----------
        lons: array-like of float, shape (n,)
            1-dimensional longitude array in degrees.
        lats: array-like of float, shape (n,)
            1-dimensional latitude array in degrees.
        dates: array-like of datetime, shape (n,)
            1-dimensional date array.
        speed_limit: float
            maximum allowable speed for an in situ drifting buoy (metres per second)
        min_win_period: float
            minimum period of time in days over which position is assessed for speed estimates (see description)
        ship_speed_limit: float
            Ship speed limit for the IQUAM track check
        delta_d: float
            The smallest increment in distance that can be resolved. For 0.01 degrees of lat-lon this is 1.11 km. Used
            in the IQUAM track check
        delta_t: float
            The smallest increment in time that can be resolved. For hourly data expressed as a float this is 0.01
            hours. Used in the IQUAM track check
        n_neighbours: int
            Number of neighbours considered in the IQUAM track check
        """
        self.lon = lons
        self.lat = lats
        self.nreps = len(lons)
        self.dates = dates
        self.hrs = convert_date_to_hours(dates)

        self.speed_limit = speed_limit
        self.min_win_period = min_win_period

        self.ship_speed_limit = ship_speed_limit
        self.delta_d = delta_d
        self.delta_t = delta_t
        self.n_neighbours = n_neighbours

        self.iquam_track_ship = None

        # Initialise QC outcomes with untested
        self.qc_outcomes = np.zeros(self.nreps) + untested

    def get_qc_outcomes(self):
        """Return the QC outcomes"""
        return self.qc_outcomes

    def valid_parameters(self) -> bool:
        """Check the parameters are valid. Raises a warning and returns False if not valid"""
        valid = True
        try:
            assert self.speed_limit >= 0, "speed_limit must be >= 0"
            assert self.min_win_period >= 0, "min_win_period must be >= 0"
        except AssertionError as error:
            valid = False
            warnings.warn(UserWarning(f"invalid input parameter: {error}"))

        return valid

    def valid_arrays(self) -> bool:
        """Check the input arrays are valid. Raises a warning and returns False if not valid"""
        valid = True
        try:
            assert not any(np.isnan(self.lon)), "Nan(s) found in longitude"
            assert not any(np.isnan(self.lat)), "Nan(s) found in latitude"
            assert not any(np.isnan(self.hrs)), "Nan(s) found in time differences"
            assert is_monotonic(self.hrs), "times are not sorted"
        except AssertionError as error:
            warnings.warn(UserWarning("problem with report values: " + str(error)))
            valid = False
        return valid

    def perform_iquam_track_check(self):
        """Perform iQuam track check as if a ship a deep copy of reps is made so metadata can be safely modified
        ahead of iQuam check an array of qc flags (iquam_track_ship) is the result
        """
        self.iquam_track_ship = do_iquam_track_check(
            self.lat,
            self.lon,
            self.dates,
            self.ship_speed_limit,
            self.delta_d,
            self.delta_t,
            self.n_neighbours,
        )

    def do_new_speed_check(self) -> None:
        """Perform the actual new speed check"""
        nrep = self.nreps
        min_win_period_hours = self.min_win_period * 24.0

        if not self.valid_arrays() or not self.valid_parameters():
            self.qc_outcomes = np.zeros(self.nreps) + untestable
            return

        self.perform_iquam_track_check()

        # Initialise
        self.qc_outcomes = np.zeros(nrep) + passed

        # loop through timeseries to see if drifter is moving too fast and flag any occurrences
        index_arr = np.array(range(0, nrep))  # type: np.ndarray
        i = 0
        time_to_end = self.hrs[-1] - self.hrs[i]
        while time_to_end >= min_win_period_hours:
            if self.iquam_track_ship[i] == failed:
                i += 1
                time_to_end = self.hrs[-1] - self.hrs[i]
                continue
            f_win = (self.hrs >= self.hrs[i] + min_win_period_hours) & (
                self.iquam_track_ship == passed
            )
            if not any(f_win):
                i += 1
                time_to_end = self.hrs[-1] - self.hrs[i]
                continue

            win_len = self.hrs[f_win][0] - self.hrs[i]
            displace = sphere_distance(
                self.lat[i], self.lon[i], self.lat[f_win][0], self.lon[f_win][0]
            )
            speed = displace / win_len  # km per hr
            speed = speed * 1000.0 / (60.0 * 60)  # metres per sec

            if speed > self.speed_limit:
                self.qc_outcomes[i : index_arr[f_win][0] + 1] = failed

            i += 1
            time_to_end = self.hrs[-1] - self.hrs[i]


class AgroundChecker:
    """
    Check to see whether a drifter has run aground based on 1/100th degree precision positions.
    A flag 'drf_agr' is set for each input report: flag=1 for reports deemed aground, else flag=0.

    Positional errors introduced by lon/lat 'jitter' and data precision can be of order several km's.
    Longitude and latitude timeseries are smoothed prior to assessment to reduce position 'jitter'.
    Some post-smoothing position 'jitter' may remain and its expected magnitude is set within the
    function by the 'tolerance' parameter. A drifter is deemed aground when, after a period of time,
    the distance between reports is less than the 'tolerance'. The minimum period of time over which this
    assessment is made is set by 'min_win_period'. This period must be long enough such that slow moving
    drifters are not falsely flagged as aground given errors in position (e.g. a buoy drifting at around
    1 cm/s will travel around 1 km/day; given 'tolerance' and precision errors of a few km's the 'min_win_period'
    needs to be several days to ensure distance-travelled exceeds the error so that motion is reliably
    detected and the buoy is not falsely flagged as aground). However, min_win_period should not be longer
    than necessary as buoys that run aground for less than min_win_period will not be detected.

    Because temporal sampling can be erratic the time period over which an assessment is made is specified
    as a range (bound by 'min_win_period' and 'max_win_period') - assessment uses the longest time separation
    available within this range. If a drifter is deemed aground and subsequently starts moving (e.g. if a drifter
    has moved very slowly for a prolonged period) incorrectly flagged reports will be reinstated.

    smooth_win: length of window (odd number) in datapoints used for smoothing lon/lat
    min_win_period: minimum period of time in days over which position is assessed for no movement (see description)
    max_win_period: maximum period of time in days over which position is assessed for no movement (this should be
    greater than min_win_period and allow for erratic temporal sampling e.g. min_win_period+2 to allow for gaps of
    up to 2-days in sampling).
    """

    # displacement resulting from 1/100th deg 'position-jitter' at the equator (km)
    tolerance = sphere_distance(0, 0, 0.01, 0.01)

    def __init__(
        self,
        lons: Sequence[float],
        lats: Sequence[float],
        dates: Sequence[datetime],
        smooth_win: int,
        min_win_period: int,
        max_win_period: int | None,
    ):
        """
        Create on object for performing the Aground and New Aground checks.

        Parameters
        ----------
        lons: array-like of float, shape (n,)
            1-dimensional longitude array in degrees.
        lats: array-like of float, shape (n,)
            1-dimensional latitude array in degrees.
        dates: array-like of datetime, shape (n,)
            1-dimensional date array.
        smooth_win: int
            length of window (odd number) in datapoints used for smoothing lon/lat
        min_win_period: int
            minimum period of time in days over which position is assessed for no movement (see description)
        max_win_period: int or None
            maximum period of time in days over which position is assessed for no movement (this should be greater
            than min_win_period and allow for erratic temporal sampling e.g. min_win_period+2 to allow for gaps of
            up to 2-days in sampling).
        """
        self.lon = lons
        self.lat = lats
        self.nreps = len(lons)
        self.hrs = convert_date_to_hours(dates)

        self.smooth_win = smooth_win
        self.min_win_period = min_win_period
        self.max_win_period = max_win_period

        self.lon_smooth = None
        self.lat_smooth = None
        self.hrs_smooth = None

        # Initialise QC outcomes with untested
        self.qc_outcomes = np.zeros(self.nreps) + untested

    def get_qc_outcomes(self) -> Sequence[int]:
        """Return the QC outcomes"""
        return self.qc_outcomes

    def valid_parameters(self) -> bool:
        """Check the parameters are valid. Raises a warning and returns False if not valid"""
        valid = True
        try:
            assert self.smooth_win >= 1, "smooth_win must be >= 1"
            assert self.smooth_win % 2 != 0, "smooth_win must be an odd number"
            assert self.min_win_period >= 1, "min_win_period must be >= 1"
            if self.max_win_period is not None:
                assert self.max_win_period >= 1, "max_win_period must be >= 1"
                assert (
                    self.max_win_period >= self.min_win_period
                ), "max_win_period must be >= min_win_period"
        except AssertionError as error:
            warnings.warn(UserWarning("invalid input parameter: " + str(error)))
            valid = False
        return valid

    def valid_arrays(self) -> bool:
        """Check the input arrays are valid. Raises a warning and returns False if not valid"""
        valid = True
        try:
            assert not any(np.isnan(self.lon)), "Nan(s) found in longitude"
            assert not any(np.isnan(self.lat)), "Nan(s) found in latitude"
            assert not any(np.isnan(self.hrs)), "Nan(s) found in time differences"
            assert is_monotonic(self.hrs), "times are not sorted"
        except AssertionError as error:
            warnings.warn(UserWarning("problem with report values: " + str(error)))
            valid = False
        return valid

    def smooth_arrays(self):
        """Perform the preprocessing of the lat lon and time arrays"""
        half_win = int((self.smooth_win - 1) / 2)
        # create smoothed lon/lat timeseries  # length of series after smoothing
        nrep_smooth = self.nreps - self.smooth_win + 1
        lon_smooth = np.empty(nrep_smooth)  # type: np.ndarray
        lon_smooth[:] = np.nan
        lat_smooth = np.empty(nrep_smooth)  # type: np.ndarray
        lat_smooth[:] = np.nan
        hrs_smooth = np.empty(nrep_smooth)  # type: np.ndarray
        hrs_smooth[:] = np.nan

        for i in range(0, nrep_smooth):
            lon_smooth[i] = np.median(self.lon[i : i + self.smooth_win])
            lat_smooth[i] = np.median(self.lat[i : i + self.smooth_win])
            hrs_smooth[i] = self.hrs[i + half_win]

        self.lon_smooth = lon_smooth
        self.lat_smooth = lat_smooth
        self.hrs_smooth = hrs_smooth

    def do_aground_check(self):
        """Perform the actual aground check"""
        half_win = (self.smooth_win - 1) / 2
        min_win_period_hours = self.min_win_period * 24.0
        if self.max_win_period is not None:
            max_win_period_hours = self.max_win_period * 24.0
        else:
            max_win_period_hours = None

        if not self.valid_parameters() or not self.valid_arrays():
            self.qc_outcomes[:] = untestable
            return

        # Default to pass
        if self.nreps <= self.smooth_win:
            self.qc_outcomes[:] = passed
            return

        self.smooth_arrays()

        # loop through smoothed timeseries to see if drifter has run aground
        i = 0
        is_aground = False  # keeps track of whether drifter is deemed aground
        i_aground = np.nan  # keeps track of index when drifter first ran aground
        time_to_end = self.hrs_smooth[-1] - self.hrs_smooth[i]
        while time_to_end >= min_win_period_hours:

            if self.max_win_period is not None:
                f_win = self.hrs_smooth <= self.hrs_smooth[i] + max_win_period_hours
                win_len = self.hrs_smooth[f_win][-1] - self.hrs_smooth[i]
                if win_len < min_win_period_hours:
                    i += 1
                    time_to_end = self.hrs_smooth[-1] - self.hrs_smooth[i]
                    continue

                displace = sphere_distance(
                    self.lat_smooth[i],
                    self.lon_smooth[i],
                    self.lat_smooth[f_win][-1],
                    self.lon_smooth[f_win][-1],
                )
            else:
                displace = sphere_distance(
                    self.lat_smooth[i],
                    self.lon_smooth[i],
                    self.lat_smooth[-1],
                    self.lon_smooth[-1],
                )

            if displace <= AgroundChecker.tolerance:
                if not is_aground:
                    is_aground = True
                    i_aground = i
            else:
                is_aground = False
                i_aground = np.nan

            i += 1
            time_to_end = self.hrs_smooth[-1] - self.hrs_smooth[i]

        # set flags
        if is_aground and i_aground > 0:
            i_aground += half_win
        # this gets the first index the drifter is deemed aground for the original (un-smoothed) timeseries
        # n.b. if i_aground=0 then the entire drifter record is deemed aground and flagged as such
        self.qc_outcomes[:] = passed
        if is_aground:
            self.qc_outcomes[int(i_aground) :] = failed


class SSTTailChecker:
    """Check to see whether there is erroneous sea surface temperature data at the beginning or end of a drifter record
    (referred to as 'tails'). The flags 'drf_tail1' and 'drf_tail2' are set for each input report: flag=1 for reports
    with erroneous data, else flag=0, 'drf_tail1' is used for bad data at the beginning of a record, 'drf_tail2' is
    used for bad data at the end of a record.

    The tail check makes an assessment of the quality of data at the start and end of a drifting buoy record by
    comparing to a background reference field. Data found to be unacceptably biased or noisy relative to the
    background are flagged by the check. When making the comparison an allowance is made for background error
    variance and also normal drifter error (both bias and random measurement error). The correlation of the
    background error is treated as unknown and takes on a value which maximises background error dependent on the
    assessment being made. A background error variance limit is also specified, beyond which the background is deemed
    unreliable. Observations made during the day, in icy regions or where the background value is missing are
    excluded from the comparison.

    The check proceeds in two steps; a 'long tail-check' followed by a 'short tail-check'. The idea is that the short
    tail-check has finer resolution but lower sensitivity than the long tail-check and may pick off noisy data not
    picked up by the long tail check. Only observations that pass the long tail-check are passed to the short
    tail-check. Both of these tail checks proceed by moving a window over the data and assessing the data in each
    window. Once good data are found the check stops and any bad data preceding this are flagged. If unreliable
    background data are encountered the check stops. The checks are run forwards and backwards over the record so as
    to assess data at the start and end of the record. If the whole record fails no observations are flagged as there
    are then no 'tails' in the data (this is left for other checks). The long tail check looks for groups of
    observations that are too biased or noisy as a whole. The short tail check looks for individual observations
    exceeding a noise limit within the window.

    Parameters
    ----------
    long_win_len: int
        Length of window (in data-points) over which to make long tail-check (must be an odd number)
    long_err_std_n: float
        Number of standard deviations of combined background and drifter bias error, beyond which
        data fail bias check
    short_win_len: int
        Length of window (in data-points) over which to make the short tail-check
    short_err_std_n: float
        Number of standard deviations of combined background and drifter error, beyond which data
        are deemed suspicious
    short_win_n_bad: int
        Minimum number of suspicious data points required for failure of short check window
    drif_inter: float
        spread of biases expected in drifter data (standard deviation, degC)
    drif_intra: float
        Maximum random measurement uncertainty reasonably expected in drifter data (standard deviation, degC)
    background_err_lim: float
        Background error variance beyond which the SST background is deemed unreliable (degC squared)
    """

    def __init__(
        self,
        lat: Sequence[float],
        lon: Sequence[float],
        sst: Sequence[float],
        ostia: Sequence[float],
        ice: Sequence[float],
        bgvar: Sequence[float],
        dates: Sequence[datetime],
        long_win_len: int,
        long_err_std_n: float,
        short_win_len: int,
        short_err_std_n: float,
        short_win_n_bad: int,
        drif_inter: float,
        drif_intra: float,
        background_err_lim: float,
    ):
        """
        Create SSTTailChecker object to perform the SST Tail QC Check

        Parameters
        ----------
        lat: array-like of float, shape (n,)
            1-dimensional latitude array in degrees.
        lon: array-like of float, shape (n,)
            1-dimensional longitude array in degrees.
        sst: array-like of float, shape (n,)
            1-dimensional array of sea surface temperatures in K.
        ostia: array-like of float, shape (n,)
            1-dimensional array of background field sea surface temperatures in K
        ice: array-like of float, shape (n,)
            1-dimensional array of ice concentrations in the range 0.0 to 1.0
        bgvar: array-like of float, shape (n,)
            1-dimensional array of background sea surface temperature fields variances in K^2
        dates: array-like of datetime, shape (n,)
            1-dimensional date array.
        long_win_len: int
            length of window (in data-points) over which to make long tail-check (must be an odd number)
        long_err_std_n: float
            number of standard deviations of combined background and drifter bias error, beyond which
            data fail bias check
        short_win_len: int
            length of window (in data-points) over which to make the short tail-check
        short_err_std_n: float
            number of standard deviations of combined background and drifter error, beyond which data
            are deemed suspicious
        short_win_n_bad: int
            minimum number of suspicious data points required for failure of short check window
        drif_inter: float
            spread of biases expected in drifter data (standard deviation, degC or K)
        drif_intra: float
            maximum random measurement uncertainty reasonably expected in drifter data (standard deviation,
            degC or K)
        background_err_lim: background error variance beyond which the SST background is deemed unreliable (degC
            squared)
        """
        self.nreps = len(sst)

        self.lat = lat
        self.lon = lon
        self.sst = sst
        self.ostia = ostia
        self.ice = ice
        self.bgvar = bgvar
        self.dates = dates
        self.hrs = convert_date_to_hours(dates)

        self.reps_ind = None
        self.sst_anom = None
        self.bgerr = None

        self.start_tail_ind = None
        self.end_tail_ind = None

        self.qc_outcomes = np.zeros(self.nreps) + untested

        self.long_win_len = long_win_len
        self.long_err_std_n = long_err_std_n
        self.short_win_len = short_win_len
        self.short_err_std_n = short_err_std_n
        self.short_win_n_bad = short_win_n_bad
        self.drif_inter = drif_inter
        self.drif_intra = drif_intra
        self.background_err_lim = background_err_lim

    def get_qc_outcomes(self):
        """Return the QC outcomes"""
        return self.qc_outcomes

    def valid_parameters(self):
        """Check the parameters are valid. Raises a warning and returns False if not valid"""
        valid = True
        try:
            assert self.long_win_len >= 1, "long_win_len must be >= 1"
            assert self.long_win_len % 2 != 0, "long_win_len must be an odd number"
            assert self.long_err_std_n >= 0, "long_err_std_n must be >= 0"
            assert self.short_win_len >= 1, "short_win_len must be >= 1"
            assert self.short_err_std_n >= 0, "short_err_std_n must be >= 0"
            assert self.short_win_n_bad >= 1, "short_win_n_bad must be >= 1"
            assert self.drif_inter >= 0, "drif_inter must be >= 0"
            assert self.drif_intra >= 0, "drif_intra must be >= 0"
            assert self.background_err_lim >= 0, "background_err_lim must be >= 0"
        except AssertionError as error:
            warnings.warn(UserWarning("invalid input parameter: " + str(error)))
            valid = False
        return valid

    def do_sst_tail_check(self, start_tail: bool):
        """Perform the actual SST tail check"""
        if not self.valid_parameters():
            self.qc_outcomes[:] = untestable
            return

        if not is_monotonic(self.hrs):
            warnings.warn(UserWarning("Times do not increase monotonically"))
            self.qc_outcomes[:] = untestable
            return

        invalid_series = self._preprocess_reps()
        if invalid_series:
            self.qc_outcomes[:] = untestable
            return

        if len(self.sst_anom) == 0:
            self.qc_outcomes[:] = passed
            return

        self.qc_outcomes[:] = passed

        nrep = len(self.sst_anom)
        self.start_tail_ind = -1  # keeps track of index where start tail stops
        self.end_tail_ind = nrep  # keeps track of index where end tail starts

        # do long tail check - records shorter than long-window length aren't evaluated
        if not (nrep < self.long_win_len):
            # run forwards then backwards over timeseries
            self._do_long_tail_check(forward=True)
            self._do_long_tail_check(forward=False)

        # do short tail check on records that pass long tail check - whole record already failed long tail check
        if not (self.start_tail_ind >= self.end_tail_ind):
            first_pass_ind = (
                self.start_tail_ind + 1
            )  # first index passing long tail check
            last_pass_ind = self.end_tail_ind - 1  # last index passing long tail check
            self._do_short_tail_check(first_pass_ind, last_pass_ind, forward=True)
            self._do_short_tail_check(first_pass_ind, last_pass_ind, forward=False)

        # now flag reps - whole record failed tail checks, don't flag
        if self.start_tail_ind >= self.end_tail_ind:
            self.start_tail_ind = -1
            self.end_tail_ind = nrep

        if not self.start_tail_ind == -1 and start_tail:
            self.qc_outcomes[0 : self.reps_ind[self.start_tail_ind] + 1] = failed

        if not self.end_tail_ind == nrep and not start_tail:
            self.qc_outcomes[self.reps_ind[self.end_tail_ind] :] = failed

    @staticmethod
    def _parse_rep(lat, lon, ostia, ice, bgvar, dates) -> (float, float, float, bool):
        """Process a report

        Parameters
        ----------
        lat: float
            Latitude
        lon: float
            Longitude
        ostia: float
            OSTIA value matched to this observation
        ice: float
            Ice concentration value matched to this observation
        bgvar: float
            Background variance value matched to this observation
        dates: np.datetime
            Date and time of the observation

        Returns
        -------
        (float, float, float, bool)
            Background value, ice concentration, background variance, and a boolean variable indicating whether the
            report is "good"
        """
        invalid_ob = False

        bg_val = ostia

        if ice is None or np.isnan(ice):
            ice = 0.0
        if ice < 0.0 or ice > 1.0:
            warnings.warn(UserWarning("Invalid ice value"))
            invalid_ob = True

        try:
            daytime = track_day_test(
                dates.year,
                dates.month,
                dates.day,
                dates.hour + dates.minute / 60,
                lat,
                lon,
                -2.5,
            )
        except ValueError as error:
            warnings.warn(f"Daytime check failed with {error}")
            daytime = True
            invalid_ob = True

        land_match = bg_val is None
        ice_match = ice > 0.15

        good_match = not (daytime or land_match or ice_match)

        return bg_val, ice, bgvar, good_match, invalid_ob

    def _preprocess_reps(self) -> bool:
        """Process the reps and calculate the values used in the QC check"""
        invalid_series = False
        # test and filter out obs with unsuitable background matches
        reps_ind = []  # type: list
        sst_anom = []  # type: list
        bgvar = []  # type: list
        for ind in range(self.nreps):
            bg_val, _ice_val, bgvar_val, good_match, invalid_ob = self._parse_rep(
                self.lat[ind],
                self.lon[ind],
                self.ostia[ind],
                self.ice[ind],
                self.bgvar[ind],
                self.dates[ind],
            )
            if invalid_ob:
                invalid_series = True

            if good_match:
                if bg_val < -5.0 or bg_val > 45.0:
                    warnings.warn(UserWarning("Background value is invalid"))
                    invalid_series = True
                if bgvar_val < 0 or bgvar_val > 10.0:
                    warnings.warn(UserWarning("Background variance is invalid"))
                    invalid_series = True

                reps_ind.append(ind)
                sst_anom.append(self.sst[ind] - bg_val)
                bgvar.append(bgvar_val)

        # prepare numpy arrays and variables needed for tail checks
        # indices of obs suitable for assessment
        self.reps_ind = np.array(reps_ind)  # type: np.ndarray
        # ob-background differences
        self.sst_anom = np.array(sst_anom)  # type: np.ndarray
        # standard deviation of background error
        bgvar = np.array(bgvar)
        bgvar[bgvar < 0] = np.nan
        self.bgerr = np.sqrt(bgvar)  # type: np.ndarray

        return invalid_series

    def _do_long_tail_check(self, forward: bool = True) -> None:
        """Perform the long tail check

        Parameters
        ----------
        forward: bool
            Flag to set for a forward (True) or backward (False) pass of the long tail check

        Returns
        -------
        None
        """
        nrep = len(self.sst_anom)
        mid_win_ind = int((self.long_win_len - 1) / 2)

        if forward:
            sst_anom_temp = self.sst_anom
            bgerr_temp = self.bgerr
        else:
            sst_anom_temp = np.flipud(self.sst_anom)
            bgerr_temp = np.flipud(self.bgerr)

        # this is the long tail check
        for ix in range(0, nrep - self.long_win_len + 1):
            sst_anom_winvals = sst_anom_temp[ix : ix + self.long_win_len]
            bgerr_winvals = bgerr_temp[ix : ix + self.long_win_len]
            if np.any(bgerr_winvals > np.sqrt(self.background_err_lim)):
                break
            sst_anom_avg = trim_mean(sst_anom_winvals, 100)
            sst_anom_stdev = trim_std(sst_anom_winvals, 100)
            bgerr_avg = np.mean(bgerr_winvals)
            bgerr_rms = np.sqrt(np.mean(bgerr_winvals**2))
            if (
                abs(sst_anom_avg)
                > self.long_err_std_n * np.sqrt(self.drif_inter**2 + bgerr_avg**2)
            ) or (sst_anom_stdev > np.sqrt(self.drif_intra**2 + bgerr_rms**2)):
                if forward:
                    self.start_tail_ind = ix + mid_win_ind
                else:
                    self.end_tail_ind = (nrep - 1) - ix - mid_win_ind
            else:
                break

    def _do_short_tail_check(self, first_pass_ind, last_pass_ind, forward=True):
        """Perform the short tail check.

        Parameters
        ----------
        first_pass_ind: int
            Index
        last_pass_ind: int
            Index
        forward: bool
            Flag to set for a forward (True) or backward (False) pass of the short tail check

        Returns
        -------
        None
        """
        npass = last_pass_ind - first_pass_ind + 1
        assert npass > 0, "short tail check: npass not > 0"

        # records shorter than short-window length aren't evaluated
        if npass < self.short_win_len:
            return

        if forward:
            sst_anom_temp = self.sst_anom[first_pass_ind : last_pass_ind + 1]
            bgerr_temp = self.bgerr[first_pass_ind : last_pass_ind + 1]
        else:
            sst_anom_temp = np.flipud(self.sst_anom[first_pass_ind : last_pass_ind + 1])
            bgerr_temp = np.flipud(self.bgerr[first_pass_ind : last_pass_ind + 1])

        # this is the short tail check
        for ix in range(0, npass - self.short_win_len + 1):
            sst_anom_winvals = sst_anom_temp[ix : ix + self.short_win_len]
            bgerr_winvals = bgerr_temp[ix : ix + self.short_win_len]
            if np.any(bgerr_winvals > np.sqrt(self.background_err_lim)):
                break
            limit = self.short_err_std_n * np.sqrt(
                bgerr_winvals**2 + self.drif_inter**2 + self.drif_intra**2
            )
            exceed_limit = np.logical_or(
                sst_anom_winvals > limit, sst_anom_winvals < -limit
            )
            if np.sum(exceed_limit) >= self.short_win_n_bad:
                if forward:
                    # if all windows have failed, flag everything
                    if ix == npass - self.short_win_len:
                        self.start_tail_ind += self.short_win_len
                    else:
                        self.start_tail_ind += 1
                else:
                    # if all windows have failed, flag everything
                    if ix == npass - self.short_win_len:
                        self.end_tail_ind -= self.short_win_len
                    else:
                        self.end_tail_ind -= 1
            else:
                break


class SSTBiasedNoisyChecker:
    """Check to see whether a drifter sea surface temperature record is unacceptably biased or noisy as a whole.

    The check makes an assessment of the quality of data in a drifting buoy record by comparing to a background
    reference field. If the record is found to be unacceptably biased or noisy relative to the background all
    observations are flagged by the check. For longer records the flags 'drf_bias' and 'drf_noise' are set for each
    input report: flag=1 for records with erroneous data, else flag=0. For shorter records 'drf_short' is set for
    each input report: flag=1 for reports with erroneous data, else flag=0.

    When making the comparison an allowance is made for background error variance and also normal drifter error (both
    bias and random measurement error). A background error variance limit is also specified, beyond which the
    background is deemed unreliable and is excluded from comparison. Observations made during the day, in icy regions
    or where the background value is missing are also excluded from the comparison.

    The check has two separate streams; a 'long-record check' and a 'short-record check'. Records with at least
    n_eval observations are passed to the long-record check, else they are passed to the short-record check. The
    long-record check looks for records that are too biased or noisy as a whole. The short record check looks for
    individual observations exceeding a noise limit within a record. The purpose of n_eval is to ensure records with
    too few observations for their bias and noise to be reliably estimated are handled separately by the short-record
    check.

    The correlation of the background error is treated as unknown and handled differently for each assessment. For
    the long-record noise-check and the short-record check the background error is treated as uncorrelated,
    which maximises the possible impact of background error on these assessments. For the long-record bias-check a
    limit (bias_lim) is specified beyond which the record is considered biased. The default value for this limit was
    chosen based on histograms of drifter-background bias. An alternative approach would be to treat the background
    error as entirely correlated across a long-record, which maximises its possible impact on the bias assessment. In
    this case the histogram approach was used as the limit could be tuned to give better results.
    """

    def __init__(
        self,
        lat: Sequence[float],
        lon: Sequence[float],
        dates: Sequence[datetime],
        sst: Sequence[float],
        ostia: Sequence[float],
        bgvar: Sequence[float],
        ice: Sequence[float],
        n_eval: int,
        bias_lim: float,
        drif_intra: float,
        drif_inter: float,
        err_std_n: float,
        n_bad: int,
        background_err_lim: float,
    ):
        """
        Create an object for performing the SST Biased, Noisy and Short Checks.

        Parameters
        ----------
        lat: array-like of float, shape (n,)
            1-dimensional latitude array in degrees.
        lon: array-like of float, shape (n,)
            1-dimensional longitude array in degrees.
        dates: array-like of datetime, shape (n,)
            1-dimensional date array.
        sst: array-like of float, shape (n,)
            1-dimensional sea surface temperature array in K
        ostia: array-like of float, shape (n,)
            1-dimensional background field sea surface temperature array in K.
        bgvar: array-like of float, shape (n,)
            1-dimensional background variance array in K^2
        ice: array-like of float, shape (n,)
            1-dimensional ice concentration array in range 0,1
        n_eval: int
            the minimum number of drifter observations required to be assessed by the long-record check
        bias_lim: float
            maximum allowable drifter-background bias, beyond which a record is considered biased (degC or K)
        drif_intra: float
            maximum random measurement uncertainty reasonably expected in drifter data (standard deviation, degC or K)
        drif_inter: float
            spread of biases expected in drifter data (standard deviation, degC or K)
        err_std_n: float
            number of standard deviations of combined background and drifter error, beyond which
            short-record data are deemed suspicious
        n_bad: int
            minimum number of suspicious data points required for failure of short-record check
        background_err_lim: float
            background error variance beyond which the SST background is deemed unreliable (degC squared or K squared)
        """
        self.lat = lat
        self.lon = lon
        self.dates = dates
        self.sst = sst
        self.ostia = ostia
        self.bgvar = bgvar
        self.ice = ice

        self.nreps = len(lat)
        self.hrs = convert_date_to_hours(dates)

        self.n_eval = n_eval
        self.bias_lim = bias_lim
        self.drif_intra = drif_intra
        self.drif_inter = drif_inter
        self.err_std_n = err_std_n
        self.n_bad = n_bad
        self.background_err_lim = background_err_lim

        self.sst_anom = None
        self.bgerr = None
        self.bgvar_is_masked = None

        self.qc_outcomes_bias = np.zeros(self.nreps) + untested
        self.qc_outcomes_noise = np.zeros(self.nreps) + untested
        self.qc_outcomes_short = np.zeros(self.nreps) + untested

    def valid_parameters(self) -> bool:
        """Check the parameters are valid. Raises a warning and returns False if not valid"""
        valid = True
        try:
            assert self.n_eval > 0, "n_eval must be > 0"
            assert self.bias_lim >= 0, "bias_lim must be >= 0"
            assert self.drif_intra >= 0, "drif_intra must be >= 0"
            assert self.drif_inter >= 0, "drif_inter must be >= 0"
            assert self.err_std_n >= 0, "err_std_n must be >= 0"
            assert self.n_bad >= 1, "n_bad must be >= 1"
            assert self.background_err_lim >= 0, "background_err_lim must be >= 0"
        except AssertionError as error:
            valid = False
            warnings.warn(UserWarning(f"Invalid parameters {error}"))
        return valid

    def get_qc_outcomes_bias(self):
        """Return the QC outcomes for the bias check"""
        return self.qc_outcomes_bias

    def get_qc_outcomes_noise(self):
        """Return the QC outcomes for the noisy check"""
        return self.qc_outcomes_noise

    def get_qc_outcomes_short(self):
        """Return the QC outcomes for the short check"""
        return self.qc_outcomes_short

    def set_all_qc_outcomes_to(self, input_state):
        """Set all the QC outcomes to the specified input_state"""
        assert input_state in [
            passed,
            failed,
            untested,
            untestable,
        ], f"Unknown input_state {input_state}"
        self.qc_outcomes_short[:] = input_state
        self.qc_outcomes_noise[:] = input_state
        self.qc_outcomes_bias[:] = input_state

    def do_sst_biased_noisy_check(self):
        """Perform the bias/noise check QC"""
        if not self.valid_parameters():
            self.set_all_qc_outcomes_to(untestable)
            return

        invalid_series = self._preprocess_reps()
        if invalid_series:
            self.set_all_qc_outcomes_to(untestable)
            return

        long_record = not (len(self.sst_anom) < self.n_eval)

        self.set_all_qc_outcomes_to(passed)

        if long_record:
            self._long_record_qc()
        else:
            if not self.bgvar_is_masked:
                self._short_record_qc()

    @staticmethod
    def _parse_rep(
        lat, lon, ostia, ice, bgvar, dates, background_err_lim
    ) -> (float, float, float, bool, bool, bool):
        """
        Extract QC-relevant variables from a marine report and

        Parameters
        ----------
        lat: float
            Latitude of the observation to be parsed
        lon: float
            Longitude of the observation to be parsed
        ostia: float
            Background SST field value
        ice: float
            Ice concentration field value
        bgvar: float
            Background variance field value
        dates: datetime
            Date and time of the observation to be parsed.
        background_err_lim: float
            background error variance beyond which the SST background is deemed unreliable (degC squared or K squared)

        Returns
        -------
        float, float, float, bool, bool, bool
            Returns the background SST value, ice value, background SST variance, a flag that indicates a good match,
            and a flag that indicates if the background variance is valid, and a flag that indicates if the observation
            is valid overall
        """
        invalid_ob = False
        bg_val = ostia

        if ice is None or np.isnan(ice):
            ice = 0.0
        if ice < 0.0 or ice > 1.0:
            warnings.warn(UserWarning("Invalid ice value"))
            invalid_ob = True

        try:
            daytime = track_day_test(
                dates.year,
                dates.month,
                dates.day,
                dates.hour + dates.minute / 60,
                lat,
                lon,
                -2.5,
            )
        except ValueError as error:
            warnings.warn(f"Daytime check failed with {error}")
            daytime = True
            invalid_ob = True

        land_match = bg_val is None
        ice_match = ice > 0.15
        bgvar_mask = bgvar is not None and bgvar > background_err_lim

        good_match = not (daytime or land_match or ice_match or bgvar_mask)

        return bg_val, ice, bgvar, good_match, bgvar_mask, invalid_ob

    def _preprocess_reps(self) -> bool:
        """
        Fill SST anomalies and background errors used in the QC checks, as well as a flag
        indicating missing or invalid background values.
        """
        invalid_series = False
        # test and filter out obs with unsuitable background matches
        sst_anom = []
        bgvar = []
        bgvar_is_masked = False

        for ind in range(self.nreps):
            bg_val, _ice_val, bgvar_val, good_match, bgvar_mask, invalid_ob = (
                SSTBiasedNoisyChecker._parse_rep(
                    self.lat[ind],
                    self.lon[ind],
                    self.ostia[ind],
                    self.ice[ind],
                    self.bgvar[ind],
                    self.dates[ind],
                    self.background_err_lim,
                )
            )
            if invalid_ob:
                invalid_series = True

            if bgvar_mask:
                bgvar_is_masked = True

            if good_match:
                if bg_val < -5.0 or bg_val > 45.0:
                    warnings.warn(UserWarning("Background value is invalid"))
                    invalid_series = True
                if bgvar_val < 0 or bgvar_val > 10.0:
                    warnings.warn(UserWarning("Background variance is invalid"))
                    invalid_series = True

                sst_anom.append(self.sst[ind] - bg_val)
                bgvar.append(bgvar_val)

        # prepare numpy arrays and variables needed for checks
        self.sst_anom = np.array(sst_anom)  # ob-background differences
        bgvar = np.array(bgvar)
        bgvar[bgvar < 0] = np.nan
        self.bgerr = np.sqrt(np.array(bgvar))  # standard deviation of background error

        self.bgvar_is_masked = bgvar_is_masked

        if not is_monotonic(self.hrs):
            warnings.warn(UserWarning("Not sorted in time order"))
            invalid_series = True

        return invalid_series

    def _long_record_qc(self) -> None:
        """Perform the long record check"""
        sst_anom_avg = np.mean(self.sst_anom)
        sst_anom_stdev = np.std(self.sst_anom)
        bgerr_rms = np.sqrt(np.mean(self.bgerr**2))

        self.qc_outcomes_bias[:] = passed
        if abs(sst_anom_avg) > self.bias_lim:
            self.qc_outcomes_bias[:] = failed

        self.qc_outcomes_noise[:] = passed
        if sst_anom_stdev > np.sqrt(self.drif_intra**2 + bgerr_rms**2):
            self.qc_outcomes_noise[:] = failed

    def _short_record_qc(self) -> None:
        """Perform the short record check"""
        # Calculate the limit based on the combined uncertainties (background error, drifter inter and drifter intra
        # error) and then multiply by the err_std_n
        limit = self.err_std_n * np.sqrt(
            self.bgerr**2 + self.drif_inter**2 + self.drif_intra**2
        )

        # If the number of obs outside the limit exceed n_bad then flag them all as bad
        self.qc_outcomes_short[:] = passed
        exceed_limit = np.logical_or(self.sst_anom > limit, self.sst_anom < -limit)
        if np.sum(exceed_limit) >= self.n_bad:
            self.qc_outcomes_short[:] = failed


@inspect_arrays(["lons", "lats", "dates"])
def do_speed_check(
    lons: Sequence[float],
    lats: Sequence[float],
    dates: Sequence[datetime],
    speed_limit: float,
    min_win_period: float,
    max_win_period: float,
) -> Sequence[int]:
    """
    Perform the Track QC speed check

    Parameters
    ----------
    lons: array-like of float, shape (n,)
        1-dimensional longitude array in degrees.
    lats: array-like of float, shape (n,)
        1-dimensional latitude array in degrees.
    dates: array-like of datetime, shape (n,)
        1-dimensional date array.
    speed_limit: float
        maximum allowable speed for an in situ drifting buoy (metres per second)
    min_win_period: float
        minimum period of time in days over which position is assessed for speed estimates (see
        description)
    max_win_period: float
        maximum period of time in days over which position is assessed for speed estimates
        (this should be greater than min_win_period and allow for some erratic temporal sampling e.g.
        min_win_period + 0.2 to allow for gaps of up to 0.2 - days in sampling).

    Returns
    -------
    array-like of int, shape (n,)
        1-dimensional array containing QC flags.
        1 if speed check fails, 0 otherwise.

    Note
    ----
    In previous versions, default values for the parameters were:
    * speed_limit = 2.5
    * min_win_period = 0.8
    * max_win_perido = 1.8
    """
    checker = SpeedChecker(
        lons, lats, dates, speed_limit, min_win_period, max_win_period
    )
    checker.do_speed_check()
    return checker.get_qc_outcomes()


@inspect_arrays(["lons", "lats", "dates"])
def do_new_speed_check(
    lons: Sequence[float],
    lats: Sequence[float],
    dates: Sequence[datetime],
    speed_limit: float,
    min_win_period: float,
    ship_speed_limit: float,
    delta_d: float,
    delta_t: float,
    n_neighbours: int,
):
    """
    Perform the new speed check

    Parameters
    ----------
    lons: array-like of float, shape (n,)
        1-dimensional longitude array in degrees.
    lats: array-like of float, shape (n,)
        1-dimensional latitude array in degrees.
    dates: array-like of datetime, shape (n,)
        1-dimensional date array.
    speed_limit: float
        maximum allowable speed for an in situ drifting buoy (metres per second)
    min_win_period: float
        minimum period of time in days over which position is assessed for speed estimates (see description)
    ship_speed_limit: float
        Ship speed limit for the IQUAM track check
    delta_d: float
        The smallest increment in distance that can be resolved. For 0.01 degrees of lat-lon this is 1.11 km. Used
        in the IQUAM track check
    delta_t: float
        The smallest increment in time that can be resolved. For hourly data expressed as a float this is 0.01 hours.
        Used in the IQUAM track check
    n_neighbours: int
        Number of neighbours considered in the IQUAM track check

    Returns
    -------
    array-like of int, shape (n,)
        Array containing the QC outcomes for the new speed check

    Note
    ----
    In previous versions, default values for the parameters were:
    * speed_limit = 3.0
    * min_win_period = 0.375

    And, for the IQUAM-specific parameters:
    * ship_speed_limit = 60.0
    * delta_d = 1.11
    * delta_t = 0.01
    * n_neighbours = 5
    """
    checker = NewSpeedChecker(
        lons,
        lats,
        dates,
        speed_limit,
        min_win_period,
        ship_speed_limit,
        delta_d,
        delta_t,
        n_neighbours,
    )
    checker.do_new_speed_check()
    return checker.get_qc_outcomes()


@inspect_arrays(["lons", "lats", "dates"])
def do_aground_check(
    lons: Sequence[float],
    lats: Sequence[float],
    dates: Sequence[datetime],
    smooth_win: int,
    min_win_period: int,
    max_win_period: int,
):
    """
    Perform the aground check

    Parameters
    ----------
    lons: array-like of float, shape (n,)
        1-dimensional longitude array in degrees.
    lats: array-like of float, shape (n,)
        1-dimensional latitude array in degrees.
    dates: array-like of datetime, shape (n,)
        1-dimensional date array.
    smooth_win: int
        length of window (odd number) in datapoints used for smoothing lon/lat
    min_win_period: int
        minimum period of time in days over which position is assessed for no movement (see description)
    max_win_period: int or None
        maximum period of time in days over which position is assessed for no movement (this should be greater
        than min_win_period and allow for erratic temporal sampling e.g. min_win_period+2 to allow for gaps of
        up to 2-days in sampling).

    Returns
    -------
    array-like of int, shape (n,)
        1-dimensional array containing QC flags.
        1 if aground check fails, 0 otherwise.

    Note
    ----
    In previous versions, default values for the parameters were:
    * smooth_win = 41
    * min_win_period = 8
    * max_win_period = 10
    """
    checker = AgroundChecker(
        lons, lats, dates, smooth_win, min_win_period, max_win_period
    )
    checker.do_aground_check()
    return checker.get_qc_outcomes()


@inspect_arrays(["lons", "lats", "dates"])
def do_new_aground_check(
    lons: Sequence[float],
    lats: Sequence[float],
    dates: Sequence[datetime],
    smooth_win: int,
    min_win_period: int,
):
    """
    Perform the new aground check

    Parameters
    ----------
    lons: array-like of float, shape (n,)
        1-dimensional longitude array in degrees.
    lats: array-like of float, shape (n,)
        1-dimensional latitude array in degrees.
    dates: array-like of datetime, shape (n,)
        1-dimensional date array.
    smooth_win: int
        length of window (odd number) in datapoints used for smoothing lon/lat
    min_win_period: int
        minimum period of time in days over which position is assessed for no movement (see description)

    Returns
    -------
    array-like of int, shape (n,)
        1-dimensional array containing QC flags.
        1 if new aground check fails, 0 otherwise.

    Note
    ----
    In previous versions, default values for the parameters were:
    * smooth_win = 41
    * min_win_period = 8
    """
    checker = AgroundChecker(lons, lats, dates, smooth_win, min_win_period, None)
    checker.do_aground_check()
    return checker.get_qc_outcomes()


@inspect_arrays(["lats", "lons", "sst", "ostia", "ice", "bgvar", "dates"])
def do_sst_start_tail_check(
    lons: Sequence[float],
    lats: Sequence[float],
    dates: Sequence[datetime],
    sst: Sequence[float],
    ostia: Sequence[float],
    ice: Sequence[float],
    bgvar: Sequence[float],
    long_win_len: int,
    long_err_std_n: float,
    short_win_len: int,
    short_err_std_n: float,
    short_win_n_bad: int,
    drif_inter: float,
    drif_intra: float,
    background_err_lim: float,
):
    """
    Perform the SST Start Tail Check

    Parameters
    ----------
    lats: array-like of float, shape (n,)
        1-dimensional latitude array in degrees.
    lons: array-like of float, shape (n,)
        1-dimensional longitude array in degrees.
    sst: array-like of float, shape (n,)
        1-dimensional array of sea surface temperatures in K.
    ostia: array-like of float, shape (n,)
        1-dimensional array of background field sea surface temperatures in K
    ice: array-like of float, shape (n,)
        1-dimensional array of ice concentrations in the range 0.0 to 1.0
    bgvar: array-like of float, shape (n,)
        1-dimensional array of background sea surface temperature fields variances in K^2
    dates: array-like of datetime, shape (n,)
        1-dimensional date array.
    long_win_len: int
        length of window (in data-points) over which to make long tail-check (must be an odd number)
    long_err_std_n: float
        number of standard deviations of combined background and drifter bias error, beyond which
        data fail bias check
    short_win_len: int
        length of window (in data-points) over which to make the short tail-check
    short_err_std_n: float
        number of standard deviations of combined background and drifter error, beyond which data
        are deemed suspicious
    short_win_n_bad: int
        minimum number of suspicious data points required for failure of short check window
    drif_inter: float
        spread of biases expected in drifter data (standard deviation, degC or K)
    drif_intra: float
        maximum random measurement uncertainty reasonably expected in drifter data (standard deviation,
        degC or K)
    background_err_lim: background error variance beyond which the SST background is deemed unreliable (degC
        squared)

    Returns
    -------
    array-like of int, shape (n,)
    1-dimensional array containing QC flags.
    1 if SST start tail check fails, 0 otherwise.

    Note
    ----
    In previous versions, default values for the parameters were:
    * long_win_len = 121
    * long_err_std_n = 3.0
    * short_win_len = 30
    * short_err_std_n = 3.0
    * short_win_n_bad = 2
    * drif_inter = 0.29
    * drif_intra = 1.00
    * background_err_lim = 0.3
    """
    checker = SSTTailChecker(
        lats,
        lons,
        sst,
        ostia,
        ice,
        bgvar,
        dates,
        long_win_len,
        long_err_std_n,
        short_win_len,
        short_err_std_n,
        short_win_n_bad,
        drif_inter,
        drif_intra,
        background_err_lim,
    )
    checker.do_sst_tail_check(True)
    return checker.get_qc_outcomes()


@inspect_arrays(["lats", "lons", "sst", "ostia", "ice", "bgvar", "dates"])
def do_sst_end_tail_check(
    lons: Sequence[float],
    lats: Sequence[float],
    dates: Sequence[datetime],
    sst: Sequence[float],
    ostia: Sequence[float],
    ice: Sequence[float],
    bgvar: Sequence[float],
    long_win_len: int,
    long_err_std_n: float,
    short_win_len: int,
    short_err_std_n: float,
    short_win_n_bad: int,
    drif_inter: float,
    drif_intra: float,
    background_err_lim: float,
):
    """
    Perform the SST Start Tail Check

    Parameters
    ----------
    lats: array-like of float, shape (n,)
        1-dimensional latitude array in degrees.
    lons: array-like of float, shape (n,)
        1-dimensional longitude array in degrees.
    sst: array-like of float, shape (n,)
        1-dimensional array of sea surface temperatures in K.
    ostia: array-like of float, shape (n,)
        1-dimensional array of background field sea surface temperatures in K
    ice: array-like of float, shape (n,)
        1-dimensional array of ice concentrations in the range 0.0 to 1.0
    bgvar: array-like of float, shape (n,)
        1-dimensional array of background sea surface temperature fields variances in K^2
    dates: array-like of datetime, shape (n,)
        1-dimensional date array.
    long_win_len: int
        length of window (in data-points) over which to make long tail-check (must be an odd number)
    long_err_std_n: float
        number of standard deviations of combined background and drifter bias error, beyond which
        data fail bias check
    short_win_len: int
        length of window (in data-points) over which to make the short tail-check
    short_err_std_n: float
        number of standard deviations of combined background and drifter error, beyond which data
        are deemed suspicious
    short_win_n_bad: int
        minimum number of suspicious data points required for failure of short check window
    drif_inter: float
        spread of biases expected in drifter data (standard deviation, degC or K)
    drif_intra: float
        maximum random measurement uncertainty reasonably expected in drifter data (standard deviation,
        degC or K)
    background_err_lim: background error variance beyond which the SST background is deemed unreliable (degC
        squared)

    Returns
    -------
    array-like of int, shape (n,)
    1-dimensional array containing QC flags.
    1 if SST start tail check fails, 0 otherwise.

    Note
    ----
    In previous versions, default values for the parameters were:
    * long_win_len = 121
    * long_err_std_n = 3.0
    * short_win_len = 30
    * short_err_std_n = 3.0
    * short_win_n_bad = 2
    * drif_inter = 0.29
    * drif_intra = 1.00
    * background_err_lim = 0.3
    """
    checker = SSTTailChecker(
        lats,
        lons,
        sst,
        ostia,
        ice,
        bgvar,
        dates,
        long_win_len,
        long_err_std_n,
        short_win_len,
        short_err_std_n,
        short_win_n_bad,
        drif_inter,
        drif_intra,
        background_err_lim,
    )
    checker.do_sst_tail_check(False)
    return checker.get_qc_outcomes()


@inspect_arrays(["lats", "lons", "dates", "sst", "ostia", "bgvar", "ice"])
def do_sst_biased_check(
    lons: Sequence[float],
    lats: Sequence[float],
    dates: Sequence[datetime],
    sst: Sequence[float],
    ostia: Sequence[float],
    ice: Sequence[float],
    bgvar: Sequence[float],
    n_eval: int,
    bias_lim: float,
    drif_intra: float,
    drif_inter: float,
    err_std_n: float,
    n_bad: int,
    background_err_lim: float,
):
    """
    Perform the SST bias check

    Parameters
    ----------
    lats: array-like of float, shape (n,)
        1-dimensional latitude array in degrees.
    lons: array-like of float, shape (n,)
        1-dimensional longitude array in degrees.
    dates: array-like of datetime, shape (n,)
        1-dimensional date array.
    sst: array-like of float, shape (n,)
        1-dimensional sea surface temperature array in K.
    ostia: array-like of float, shape (n,)
        1-dimensional background sea surface temperature array in K.
    bgvar: array-like of float, shape (n,)
        1-dimensional background sea surface temperature variance array in degrees squared or K squared.
    ice: array-like of float, shape (n,)
        1-dimensional sea ice concentration array in range 0.0 to 1.0.
    n_eval: int
        the minimum number of drifter observations required to be assessed by the long-record check
    bias_lim: float
        maximum allowable drifter-background bias, beyond which a record is considered biased (degC or K)
    drif_intra: float
        maximum random measurement uncertainty reasonably expected in drifter data (standard deviation, degC or K)
    drif_inter: float
        spread of biases expected in drifter data (standard deviation, degC or K)
    err_std_n: float
        number of standard deviations of combined background and drifter error, beyond which short-record data are
        deemed suspicious
    n_bad: int
        minimum number of suspicious data points required for failure of short-record check.
    background_err_lim: float
        background error variance beyond which the SST background is deemed unreliable (degC squared or K squared)

    Returns
    -------
    array-like of int, shape (n,)
        1-dimensional array containing QC flags.
        1 if SST bias check fails, 0 otherwise.

    Note
    ----
    In previous versions, default values for the parameters were:
    * n_eval = 30
    * bias_lim = 1.10
    * drif_intra = 1.0
    * drif_inter = 0.29
    * err_std_n = 3.0
    * n_bad = 2
    * background_err_lim = 0.3
    """
    checker = SSTBiasedNoisyChecker(
        lats,
        lons,
        dates,
        sst,
        ostia,
        bgvar,
        ice,
        n_eval,
        bias_lim,
        drif_intra,
        drif_inter,
        err_std_n,
        n_bad,
        background_err_lim,
    )
    checker.do_sst_biased_noisy_check()
    return checker.get_qc_outcomes_bias()


@inspect_arrays(["lats", "lons", "dates", "sst", "ostia", "bgvar", "ice"])
def do_sst_noisy_check(
    lons: Sequence[float],
    lats: Sequence[float],
    dates: Sequence[datetime],
    sst: Sequence[float],
    ostia: Sequence[float],
    ice: Sequence[float],
    bgvar: Sequence[float],
    n_eval: int,
    bias_lim: float,
    drif_intra: float,
    drif_inter: float,
    err_std_n: float,
    n_bad: int,
    background_err_lim: float,
):
    """
    Perform the SST noise check

    Parameters
    ----------
    lats: array-like of float, shape (n,)
        1-dimensional latitude array in degrees.
    lons: array-like of float, shape (n,)
        1-dimensional longitude array in degrees.
    dates: array-like of datetime, shape (n,)
        1-dimensional date array.
    sst: array-like of float, shape (n,)
        1-dimensional sea surface temperature array in K.
    ostia: array-like of float, shape (n,)
        1-dimensional background sea surface temperature array in K.
    bgvar: array-like of float, shape (n,)
        1-dimensional background sea surface temperature variance array in degrees squared or K squared.
    ice: array-like of float, shape (n,)
        1-dimensional sea ice concentration array in range 0.0 to 1.0.
    n_eval: int
        the minimum number of drifter observations required to be assessed by the long-record check
    bias_lim: float
        maximum allowable drifter-background bias, beyond which a record is considered biased (degC or K)
    drif_intra: float
        maximum random measurement uncertainty reasonably expected in drifter data (standard deviation, degC or K)
    drif_inter: float
        spread of biases expected in drifter data (standard deviation, degC or K)
    err_std_n: float
        number of standard deviations of combined background and drifter error, beyond which short-record data are
        deemed suspicious
    n_bad: int
        minimum number of suspicious data points required for failure of short-record check.
    background_err_lim: float
        background error variance beyond which the SST background is deemed unreliable (degC squared or K squared)

    Returns
    -------
    array-like of int, shape (n,)
        1-dimensional array containing QC flags.
        1 if SST noise check fails, 0 otherwise.

    Note
    ----
    In previous versions, default values for the parameters were:
    * n_eval = 30
    * bias_lim = 1.10
    * drif_intra = 1.0
    * drif_inter = 0.29
    * err_std_n = 3.0
    * n_bad = 2
    * background_err_lim = 0.3
    """
    checker = SSTBiasedNoisyChecker(
        lats,
        lons,
        dates,
        sst,
        ostia,
        bgvar,
        ice,
        n_eval,
        bias_lim,
        drif_intra,
        drif_inter,
        err_std_n,
        n_bad,
        background_err_lim,
    )
    checker.do_sst_biased_noisy_check()
    return checker.get_qc_outcomes_noise()


@inspect_arrays(["lats", "lons", "dates", "sst", "ostia", "bgvar", "ice"])
def do_sst_biased_noisy_short_check(
    lons: Sequence[float],
    lats: Sequence[float],
    dates: Sequence[datetime],
    sst: Sequence[float],
    ostia: Sequence[float],
    ice: Sequence[float],
    bgvar: Sequence[float],
    n_eval: int,
    bias_lim: float,
    drif_intra: float,
    drif_inter: float,
    err_std_n: float,
    n_bad: int,
    background_err_lim: float,
):
    """
    Perform the SST short check

    Parameters
    ----------
    lats: array-like of float, shape (n,)
        1-dimensional latitude array in degrees.
    lons: array-like of float, shape (n,)
        1-dimensional longitude array in degrees.
    dates: array-like of datetime, shape (n,)
        1-dimensional date array.
    sst: array-like of float, shape (n,)
        1-dimensional sea surface temperature array in K.
    ostia: array-like of float, shape (n,)
        1-dimensional background sea surface temperature array in K.
    bgvar: array-like of float, shape (n,)
        1-dimensional background sea surface temperature variance array in degrees squared or K squared.
    ice: array-like of float, shape (n,)
        1-dimensional sea ice concentration array in range 0.0 to 1.0.
    n_eval: int
        the minimum number of drifter observations required to be assessed by the long-record check
    bias_lim: float
        maximum allowable drifter-background bias, beyond which a record is considered biased (degC or K)
    drif_intra: float
        maximum random measurement uncertainty reasonably expected in drifter data (standard deviation, degC or K)
    drif_inter: float
        spread of biases expected in drifter data (standard deviation, degC or K)
    err_std_n: float
        number of standard deviations of combined background and drifter error, beyond which short-record data are
        deemed suspicious
    n_bad: int
        minimum number of suspicious data points required for failure of short-record check.
    background_err_lim: float
        background error variance beyond which the SST background is deemed unreliable (degC squared or K squared)

    Returns
    -------
    array-like of int, shape (n,)
        1-dimensional array containing QC flags.
        1 if SST short check fails, 0 otherwise.

    Note
    ----
    In previous versions, default values for the parameters were:
    * n_eval = 30
    * bias_lim = 1.10
    * drif_intra = 1.0
    * drif_inter = 0.29
    * err_std_n = 3.0
    * n_bad = 2
    * background_err_lim = 0.3
    """
    checker = SSTBiasedNoisyChecker(
        lats,
        lons,
        dates,
        sst,
        ostia,
        bgvar,
        ice,
        n_eval,
        bias_lim,
        drif_intra,
        drif_inter,
        err_std_n,
        n_bad,
        background_err_lim,
    )
    checker.do_sst_biased_noisy_check()
    return checker.get_qc_outcomes_short()
