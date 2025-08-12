"""Module containing QC functions for deck level QC checks which could be applied on a DataBundle."""

from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np
import pandas as pd

from .auxiliary import (
    SequenceDatetimeType,
    SequenceFloatType,
    convert_units,
    failed,
    inspect_arrays,
    passed,
    post_format_return_type,
    untestable,
    untested,
)
from .buoy_tracking_qc import is_monotonic
from .external_clim import (
    Climatology,
    ClimFloatType,
    inspect_climatology,
)
from .location_control import (
    mds_lat_to_yindex,
    mds_lon_to_xindex,
)
from .statistics import p_gross
from .time_control import (
    pentad_to_month_day,
    which_pentad,
)


def get_threshold_multiplier(
    total_nobs: int, nob_limits: list[int], multiplier_values: list[float]
) -> float:
    """Find the highest value of i such that total_nobs is greater than nob_limits[i] and return multiplier_values[i]

    This routine is used by the buddy check. It's a bit niche.

    Parameters
    ----------
    total_nobs : int
        total number of neighbour observations

    nob_limits : list[int]
        list containing the limiting numbers of observations in ascending order first element must be zero

    multiplier_values : list[float]
        list containing the multiplier values associated.

    Returns
    -------
    float
        the multiplier value
    """
    if len(nob_limits) != len(multiplier_values):
        raise ValueError(
            f"Length mismatch: nob_limits has {len(nob_limits)}, but multiplier_values has{len(multiplier_values)}."
        )

    if min(multiplier_values) <= 0:
        raise ValueError(
            "Invalid minimum multiplier_value: {min(multiplier_values)}. Must be greater than zero."
        )
    if nob_limits[0] != 0:
        raise ValueError(f"Invalid first nob_limits: {nob_limits[0]}. Must be zero.")
    if min(nob_limits) != 0:
        raise ValueError(f"Invalid minimum nob_limit: {min(nob_limits)}. Must be zero.")
    if not is_monotonic(nob_limits):
        raise ValueError(
            "Invalid nob_limits: {nob_limits}. Must be in ascending order."
        )

    multiplier = -1
    if total_nobs == 0:
        multiplier = 4.0

    for i, limit in enumerate(nob_limits):
        if total_nobs > limit:
            multiplier = multiplier_values[i]

    if multiplier <= 0:
        raise ValueError(f"Invalid multiplier: {multiplier}. Must be positive.")

    return multiplier


class SuperObsGrid:
    """Class for gridding data in buddy check, based on numpy arrays."""

    def __init__(self):
        """Initialise empty grid"""
        self.grid = np.zeros((360, 180, 73))  # type: np.ndarray
        self.buddy_mean = np.zeros((360, 180, 73))  # type: np.ndarray
        self.buddy_stdev = np.zeros((360, 180, 73))  # type: np.ndarray
        self.nobs = np.zeros((360, 180, 73))  # type: np.ndarray

    @inspect_arrays(["lats", "lons", "values"])
    def add_multiple_observations(
        self,
        lats: SequenceFloatType,
        lons: SequenceFloatType,
        dates: SequenceDatetimeType,
        values: SequenceFloatType,
    ) -> None:
        """Add a series of observations to the grid and take the grid average

        Parameters
        ----------
        lats : array-like of float, shape (n,)
            1-dimensional latitude array.

        lons : array-like of float, shape (n,)
            1-dimensional longitude array.

        dates : array-like of datetime, shape (n,)
            1-dimensional date array.

        values : array-like of float, shape (n,)
            1-dimensional anomaly array.
        """
        n_obs = len(lats)
        for i in range(n_obs):
            date = pd.Timestamp(dates[i])
            self.add_single_observation(
                lats[i], lons[i], date.month, date.day, values[i]
            )
        self.take_average()

    def add_single_observation(
        self, lat: float, lon: float, month: int, day: int, anom: float
    ) -> None:
        """Add an anomaly to the grid from specified lat lon and date.

        Parameters
        ----------
        lat : float
            Latitude of the observation in degrees

        lon : float
            Longitude of the observation in degrees

        month : int
            Month of the observation

        day : int
            Day of the observation

        anom : float
            Value to be added to the grid

        Returns
        -------
        None
        """
        xindex = mds_lon_to_xindex(lon, res=1.0)
        yindex = mds_lat_to_yindex(lat, res=1.0)
        pindex = which_pentad(month, day) - 1

        if not (0 <= xindex < 360):
            raise ValueError(f"Invalid xindex: {xindex}. Must be between 0 and 360.")
        if not (0 <= yindex < 180):
            raise ValueError(f"Invalid yindex: {yindex}. Must be between 0 and 180.")
        if not (0 <= pindex < 73):
            raise ValueError(f"Invalid pindex: {pindex}. Must be between 0 and 72.")

        if anom is not None:
            self.grid[xindex, yindex, pindex] += anom
            self.nobs[xindex, yindex, pindex] += 1

    def take_average(self) -> None:
        """Take the average of a grid to which reps have been added using add_rep"""
        nonmiss = np.nonzero(self.nobs)
        self.grid[nonmiss] = self.grid[nonmiss] / self.nobs[nonmiss]

    def get_neighbour_anomalies(
        self, search_radius: list, xindex: int, yindex: int, pindex: int
    ) -> (list[float], list[float]):
        """Search within a specified search radius of the given point and extract the neighbours for buddy check

        Parameters
        ----------
        search_radius : list
            three element array search radius in which to look lon, lat, time

        xindex : int
            the xindex of the gridcell to start from

        yindex : int
            the yindex of the gridcell to start from

        pindex : int
            the pindex of the gridcell to start from

        Returns
        -------
        list[float]
            anomalies and numbers of observations in two lists
        """
        if len(search_radius) != 3:
            raise ValueError(
                f"Length mismatch (search_radius): {len(search_radius)}. Must be 3."
            )

        radcon = np.pi / 180.0

        latitude_approx = 89.5 - yindex

        xspan = search_radius[0]
        full_xspan = int(xspan / math.cos(latitude_approx * radcon))
        yspan = search_radius[1]
        pspan = search_radius[2]

        temp_anom = []
        temp_nobs = []

        for xpt, ypt, ppt in itertools.product(
            range(-1 * full_xspan, full_xspan + 1),
            range(-1 * yspan, yspan + 1),
            range(-1 * pspan, pspan + 1),
        ):
            # Skip the central grid cell, as we don't want to buddy check it against itself
            if xpt == 0 and ypt == 0 and ppt == 0:
                continue

            # Wrap at grid boundaries
            thisxx = (xindex + xpt) % 360
            thisyy = (yindex + ypt) % 180
            thispp = (pindex + ppt) % 73

            if self.nobs[thisxx, thisyy, thispp] != 0:
                temp_anom.append(self.grid[thisxx, thisyy, thispp])
                temp_nobs.append(self.nobs[thisxx, thisyy, thispp])

        return temp_anom, temp_nobs

    def get_buddy_limits_with_parameters(
        self,
        pentad_stdev: Climatology,
        limits: list[list[int]],
        number_of_obs_thresholds: list[list[int]],
        multipliers: list[list[float]],
    ) -> None:
        """Get buddy limits with parameters.

        Parameters
        ----------
        pentad_stdev : Climatology
            Climatology object containing the 3-dimensional latitude array containing the standard deviations.

        limits : list[list[int]]
            list of the limits

        number_of_obs_thresholds : list[list[int]]
            list containing the number of obs thresholds

        multipliers : list[list[float]]
            list containing the multipliers to be applied

        Returns
        -------
        None
        """
        nonmiss = np.nonzero(self.nobs)
        for i in range(len(nonmiss[0])):
            xindex = nonmiss[0][i]
            yindex = nonmiss[1][i]
            pindex = nonmiss[2][i]
            m, d = pentad_to_month_day(pindex + 1)

            # Originally get_value_mds_style - note: might be a mismatch
            stdev = pentad_stdev.get_value(
                lat=89.5 - yindex, lon=-179.5 + xindex, month=m, day=d
            )

            if stdev is None or stdev < 0.0 or np.isnan(stdev):
                stdev = 1.0

            match_not_found = True

            for j, limit in enumerate(limits):
                temp_anom, temp_nobs = self.get_neighbour_anomalies(
                    limit, xindex, yindex, pindex
                )

                if len(temp_anom) > 0 and match_not_found:
                    self.buddy_mean[xindex, yindex, pindex] = np.mean(temp_anom)
                    total_nobs = int(np.sum(temp_nobs))

                    self.buddy_stdev[xindex, yindex, pindex] = (
                        get_threshold_multiplier(
                            total_nobs, number_of_obs_thresholds[j], multipliers[j]
                        )
                        * stdev
                    )

                    match_not_found = False

            if match_not_found:
                self.buddy_mean[xindex, yindex, pindex] = 0.0
                self.buddy_stdev[xindex, yindex, pindex] = 500.0

    def get_new_buddy_limits(
        self,
        stdev1: Climatology,
        stdev2: Climatology,
        stdev3: Climatology,
        limits: list[int, int, int],
        sigma_m: float,
        noise_scaling: float,
    ) -> None:
        """Get buddy limits for new bayesian buddy check.

        Parameters
        ----------
        stdev1 : np.ndarray
            Field of standard deviations representing standard deviation of difference between target
            gridcell and complete neighbour average (grid area to neighbourhood difference)

        stdev2 : np.ndarray
            Field of standard deviations representing standard deviation of difference between a single
            observation and the target gridcell average (point to grid area difference)

        stdev3 : np.ndarray
            Field of standard deviations representing standard deviation of difference between random
            neighbour gridcell and full neighbour average (uncertainty in neighbour average)

        limits : list[int, int, int]
            three membered list of number of degrees in latitude and longitude and number of pentads

        sigma_m : float
            Estimated measurement error uncertainty

        noise_scaling : float
            scale noise by a factor of noise_scaling used to match observed variability

        Returns
        -------
        None

        Notes
        -----
        The original default values for limits, sigma_m, and noise_scaling originally defaulted to:
        * limits = (2, 2, 4)
        * sigma_m = 1.0
        * noise_scaling = 3.0
        """
        nonmiss = np.nonzero(self.nobs)

        for i in range(len(nonmiss[0])):
            xindex = nonmiss[0][i]
            yindex = nonmiss[1][i]
            pindex = nonmiss[2][i]

            m, d = pentad_to_month_day(pindex + 1)

            stdev1_ex = stdev1.get_value(89.5 - yindex, -179.5 + xindex, month=m, day=d)
            stdev2_ex = stdev2.get_value(89.5 - yindex, -179.5 + xindex, month=m, day=d)
            stdev3_ex = stdev3.get_value(89.5 - yindex, -179.5 + xindex, month=m, day=d)

            if stdev1_ex is None or stdev1_ex < 0.0 or np.isnan(stdev1_ex):
                stdev1_ex = 1.0
            if stdev2_ex is None or stdev2_ex < 0.0 or np.isnan(stdev2_ex):
                stdev2_ex = 1.0
            if stdev3_ex is None or stdev3_ex < 0.0 or np.isnan(stdev3_ex):
                stdev3_ex = 1.0

            # if there is neighbour in that range then calculate a mean
            temp_anom, temp_nobs = self.get_neighbour_anomalies(
                limits, xindex, yindex, pindex
            )

            if len(temp_anom) > 0:
                self.buddy_mean[xindex, yindex, pindex] = np.mean(temp_anom)

                tot = 0.0
                ntot = 0.0
                for n_obs in temp_nobs:
                    # measurement error for each 1x1x5day cell
                    tot += sigma_m**2.0 / n_obs
                    # sampling error for each 1x1x5day cell
                    # multiply by three to match observed stdev
                    tot += noise_scaling * (stdev2_ex**2.0 / n_obs)
                    ntot += 1.0

                sigma_buddy = tot / (ntot**2.0)
                sigma_buddy += stdev3_ex**2.0 / ntot

                self.buddy_stdev[xindex, yindex, pindex] = math.sqrt(
                    sigma_m**2.0
                    + stdev1_ex**2.0
                    + noise_scaling * stdev2_ex**2.0
                    + sigma_buddy
                )

            else:
                self.buddy_mean[xindex, yindex, pindex] = 0.0
                self.buddy_stdev[xindex, yindex, pindex] = 500.0

    def get_buddy_mean(self, lat: float, lon: float, month: int, day: int) -> float:
        """Get the buddy mean from the grid for a specified time and place

        Parameters
        ----------
        lat : float
            latitude of the location for which the buddy mean is desired

        lon : float
            longitude of the location for which the buddy mean is desired

        month : int
            month for which the buddy mean is desired

        day : int
            day for which the buddy mean is desired

        Returns
        -------
        float
            Buddy mean at the specified location
        """
        xindex = mds_lon_to_xindex(lon, res=1.0)
        yindex = mds_lat_to_yindex(lat, res=1.0)
        pindex = which_pentad(month, day) - 1
        return self.buddy_mean[xindex, yindex, pindex]

    def get_buddy_stdev(self, lat: float, lon: float, month: int, day: int) -> float:
        """Get the buddy standard deviation from the grid for a specified time and place

        Parameters
        ----------
        lat : float
            latitude of the location for which the buddy standard deviation is desired

        lon : float
            longitude of the location for which the buddy standard deviation is desired

        month : int
            month for which the buddy standard deviation is desired

        day : int
            day for which the buddy standard deviation is desired

        Returns
        -------
        float
            Buddy standard deviation at the specified location
        """
        xindex = mds_lon_to_xindex(lon, res=1.0)
        yindex = mds_lat_to_yindex(lat, res=1.0)
        pindex = which_pentad(month, day) - 1
        return self.buddy_stdev[xindex, yindex, pindex]


@post_format_return_type(["value"])
@inspect_arrays(["lat", "lon", "date", "value"])
@convert_units(lat="degrees", lon="degrees")
@inspect_climatology("climatology")
def do_mds_buddy_check(
    lat: SequenceFloatType,
    lon: SequenceFloatType,
    date: SequenceDatetimeType,
    value: SequenceFloatType,
    climatology: ClimFloatType,
    standard_deviation: Climatology,
    limits: list[list[int]],
    number_of_obs_thresholds: list[list[int]],
    multipliers: list[list[float]],
):
    """Do the old style buddy check. The buddy check compares an observation to the average of its near neighbours
    (called the buddy mean). Depending on how many neighbours there are and their proximity to the observation being
    tested a multiplier is set. If the difference between the observation and the buddy mean is larger than the
    multiplier times the standard deviation then the observation fails the buddy check. If no buddy observations are
    found within the specified limits, then the limits are expanded until the check runs out of specified limits or
    observations are found within the limits.

    Parameters
    ----------
    lat : array-like of float, shape (n,)
        1-dimensional latitude array.

    lon : array-like of float, shape (n,)
        1-dimensional longitude array.

    date : array-like of datetime, shape (n,)
        1-dimensional date array.

    value : array-like of float, shape (n,)
        1-dimensional anomaly array.

    climatology : float, None, sequence of float or None, 1D np.ndarray of float, pd.Series of float or Climatology
        The climatological average(s) used to calculate anomalies.
        Can be a scalar, a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    standard_deviation : Climatology
        Field of standard deviations of 1x1xpentad standard deviations

    limits : list[list]
        limits a list of lists. Each list member is a three-membered list specifying the longitudinal, latitudinal,
        and time range within which buddies are sought at each level of search.

    number_of_obs_thresholds : list[list]
        number of observations corresponding to each multiplier in `multipliers`. The initial list should be
        the same length as the limits list.

    multipliers : list[list]
        multiplier, x, used for buddy check mu +- x * sigma. The list should have the same structure as
        `number_of_obs_threshold`.

    Returns
    -------
    array-like of int, shape (n,)
        1-dimensional array containing QC flags.
        1 if buddy check fails, 0 otherwise.

    Notes
    -----
    The limits, number_of_obs_thresholds, and multipliers parameters are rather complex. The buddy check basically
    looks within a lat-lon-time range specified by the first element in limits. If there are more than zero
    observations in the search range then a multiplier is chosen based on how many observations there are.

    If the first element of limits is [1,1,2] then we first look within a distance equivalent to 1 degree
    latitude and longitude at the equator and 2 pentads in time. If there are more than zero observations then we
    calculate the buddy mean, and we consult the number_of_obs_threshold. If, for example, this is [0, 5, 15, 100]
    then we look for the first entry where the number of obs is greater than that threshold. We then look up the
    multiplier in the appropriate list (say [4, 3.5, 3.0, 2.5]). If the difference between an observation and the
    buddy mean is greater than the multiplier times the standard deviation at that point then it fails the buddy
    check. So, if there were 10 observations then the multiplier would be 3.5.
    """
    anoms = value - climatology

    if len(limits) != len(number_of_obs_thresholds) and len(limits) != len(multipliers):
        raise ValueError("Input parameter lists are not equal length")

    for i, thresholds in enumerate(number_of_obs_thresholds):
        if len(thresholds) != len(multipliers[i]):
            raise ValueError(
                "Number of obs thresholds and multipliers have different shapes"
            )

    # calculate superob averages and numbers of observations
    grid = SuperObsGrid()
    grid.add_multiple_observations(lat, lon, date, anoms)
    grid.get_buddy_limits_with_parameters(
        standard_deviation, limits, number_of_obs_thresholds, multipliers
    )

    numobs = len(lat)
    qc_outcomes = np.zeros(numobs) + untested

    # finally loop over all reports and update buddy QC
    for i in range(numobs):
        lat_ = lat[i]
        lon_ = lon[i]
        mon = pd.Timestamp(date[i]).month
        day = pd.Timestamp(date[i]).day

        # if the SST anomaly differs from the neighbour average by more than the calculated range then reject
        x = anoms[i]
        bm = grid.get_buddy_mean(lat_, lon_, mon, day)
        bsd = grid.get_buddy_stdev(lat_, lon_, mon, day)

        qc_outcomes[i] = passed
        if bsd == 500.0 or np.isnan(bsd):
            qc_outcomes[i] = untestable
        elif abs(x - bm) >= bsd:
            qc_outcomes[i] = failed

    del grid

    return qc_outcomes


@post_format_return_type(["value"])
@inspect_arrays(["lat", "lon", "date", "value"])
@convert_units(lat="degrees", lon="degrees")
@inspect_climatology("climatology")
def do_bayesian_buddy_check(
    lat: SequenceFloatType,
    lon: SequenceFloatType,
    date: SequenceDatetimeType,
    value: SequenceFloatType,
    climatology: ClimFloatType,
    stdev1: Climatology,
    stdev2: Climatology,
    stdev3: Climatology,
    prior_probability_of_gross_error: float,
    quantization_interval: float,
    one_sigma_measurement_uncertainty: float,
    limits: list[int],
    noise_scaling: float,
    maximum_anomaly: float,
    fail_probability: float,
) -> Sequence[int]:
    """Do the Bayesian buddy check. The bayesian buddy check assigns a
    probability of gross error to each observation, which is rounded down to the
    tenth and then multiplied by 10 to yield a flag between 0 and 9.

    Parameters
    ----------
    lat : array-like of float, shape (n,)
        1-dimensional latitude array.

    lon : array-like of float, shape (n,)
        1-dimensional longitude array.

    date : array-like of datetime, shape (n,)
        1-dimensional date array.

    value : array-like of float, shape (n,)
        1-dimensional anomaly array.

    climatology : float, None, sequence of float or None, 1D np.ndarray of float, pd.Series of float or Climatology
        The climatological average(s) used to calculate anomalies.
        Can be a scalar, a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.

    stdev1 : Climatology
        Field of standard deviations representing standard deviation of difference between
        target gridcell and complete neighbour average (grid area to neighbourhood difference)

    stdev2 : Climatology
        Field of standard deviations representing standard deviation of difference between
        a single observation and the target gridcell average (point to grid area difference)

    stdev3 : Climatology
        Field of standard deviations representing standard deviation of difference between
        random neighbour gridcell and full neighbour average (uncertainty in neighbour average)

    prior_probability_of_gross_error : float
        Prior probability of gross error, which is the background rate of gross errors.

    quantization_interval : float
        Smallest possible increment in the input values.

    one_sigma_measurement_uncertainty : float
        Estimated one sigma measurement uncertainty

    limits : list[int]
        List with three members which specify the search range for the buddy check

    noise_scaling : float
        Tuning parameter used to multiply stdev2. This was determined to be approximately 3.0 by comparison with
        observed point data. stdev2 was estimated from OSTIA data and typically underestimates the point to
        area-average difference by this factor.

    maximum_anomaly : float
        Largest absolute anomaly, assumes that the maximum and minimum anomalies have the same magnitude

    fail_probability : float
        Probability of gross error that corresponds to a failed test. Anything with a probability of gross error
        greater than fail_probability will be considered failing.

    Returns
    -------
    array-like of int, shape (n,)
        1-dimensional array containing passed, failed or untestable flags. Untestable flags will be set if there
        are no buddies in the specified limits.

    Notes
    -----
    In previous versions the default values for the parameters were
    * prior_probability_of_gross_error = 0.05
    * quantization_interval = 0.1
    * limits = [2, 2, 4]
    * noise_scaling = 3.0
    * one_sigma_measurement_uncertainty = 1.0
    * maximum_anomaly = 8.0
    """
    anoms = value - climatology

    p0 = prior_probability_of_gross_error
    q = quantization_interval
    sigma_m = one_sigma_measurement_uncertainty

    # previous upper QC limits set. Ideally, this should be set based on any previous QC checks.
    # The original default was 8 because the climatology check had a range of +-8C. However, a
    # climatology plus standard deviation check might narrow that range, and it might also be
    # spatially varying. There is currently no means of expressing that here.
    r_hi = maximum_anomaly
    r_lo = -1.0 * r_hi  # previous lower QC limit set

    grid = SuperObsGrid()
    grid.add_multiple_observations(lat, lon, date, anoms)
    grid.get_new_buddy_limits(stdev1, stdev2, stdev3, limits, sigma_m, noise_scaling)

    numobs = len(lat)
    qc_outcomes = np.zeros(numobs) + untested

    for i in range(numobs):
        lat_ = lat[i]
        lon_ = lon[i]
        mon = pd.Timestamp(date[i]).month
        day = pd.Timestamp(date[i]).day

        # Calculate the probability of gross error given the set-up
        buddy_stdev = grid.get_buddy_stdev(lat_, lon_, mon, day)
        ppp = p_gross(
            p0,
            q,
            r_hi,
            r_lo,
            anoms[i],
            grid.get_buddy_mean(lat_, lon_, mon, day),
            buddy_stdev,
        )

        # QC outcome is based on the probability of gross error. If the probability of gross error is larger than
        # the fail_probability then it is considered a fail.
        qc_outcomes[i] = passed
        if ppp > fail_probability:
            qc_outcomes[i] = failed
        if buddy_stdev == 500.0:
            qc_outcomes[i] = untestable

    del grid

    return qc_outcomes
