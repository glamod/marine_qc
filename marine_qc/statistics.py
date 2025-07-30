"""Some generally helpful statistical functions for base QC."""

from __future__ import annotations

import copy
import math
from typing import Sequence

import numpy as np


def p_data_given_good(
    x: float, q: float, r_hi: float, r_lo: float, mu: float, sigma: float
) -> float:
    """
    Calculate the probability of an observed value x given a normal distribution with mean mu
    standard deviation of sigma, where x is constrained to fall between R_hi and R_lo
    and is known only to an integer multiple of Q, the quantization level.

    Parameters
    ----------
    x: float
        Observed value for which probability is required.
    q: float
        Quantization of x, i.e. x is an integer multiple of Q.
    r_hi: float
        The upper limit on x imposed by previous QC choices.
    r_lo: float
        The lower limit on x imposed by previous QC choices.
    mu: float
        The mean of the distribution.
    sigma: float
        The standard deviation of the distribution.

    Returns
    -------
    float
        probability of the observed value given the specified distribution.

    Raises
    ------
    ValueError
        When inputs are incorrectly specified: q<=0, sigma<=0, r_lo > r_hi, x < r_lo or x > r_hi
    """
    if q <= 0.0:
        raise ValueError(f"Value is not positive: q = {q}.")
    if sigma <= 0.0:
        raise ValueError(f"Value is not positive: sigma = {sigma}.")
    if r_lo >= r_hi:
        raise ValueError(
            f"Lower limit is greater than upper limit: r_hi = {r_hi}, r_lo = {r_lo}."
        )
    if x < r_lo:
        raise ValueError(f"Lower limit is greater than x: r_lo = {r_lo}, x = {x}.")
    if x > r_hi:
        raise ValueError(f"Upper limit is less than x: r_hi = {r_hi}, x = {x}.")

    upper_x = min([x + 0.5 * q, r_hi + 0.5 * q])
    lower_x = max([x - 0.5 * q, r_lo - 0.5 * q])

    normalizer = 0.5 * (
        math.erf((r_hi + 0.5 * q - mu) / (sigma * math.sqrt(2)))
        - math.erf((r_lo - 0.5 * q - mu) / (sigma * math.sqrt(2)))
    )

    return (
        0.5
        * (
            math.erf((upper_x - mu) / (sigma * math.sqrt(2)))
            - math.erf((lower_x - mu) / (sigma * math.sqrt(2)))
        )
        / normalizer
    )


def p_data_given_gross(q: float, r_hi: float, r_lo: float) -> float:
    """
    Calculate the probability of the data given a gross error
    assuming gross errors are uniformly distributed between
    R_low and R_high and that the quantization, rounding level is Q

    q: float
        Quantization of x, i.e. x is an integer multiple of Q.
    r_hi: float
        The upper limit on x imposed by previous QC choices.
    r_lo: float
        The lower limit on x imposed by previous QC choices.

    Returns
    -------
    float
        probability of the observed value given that it is a gross error.

    Raises
    ------
    ValueError
        When limits are not ascending or q<=0
    """
    if r_hi < r_lo:
        raise ValueError(
            f"Lower limit is greater than upper limit: r_hi = {r_hi}, r_lo = {r_lo}."
        )
    if q <= 0.0:
        raise ValueError(f"Value is not positive: q = {q}.")

    r = r_hi - r_lo
    return 1.0 / (1.0 + (r / q))


def p_gross(
    p0: float, q: float, r_hi: float, r_lo: float, x: float, mu: float, sigma: float
) -> float:
    """
    Calculate the posterior probability of a gross error given the prior probability p0,
    the quantization level of the observed value, Q, previous limits on the observed value,
    R_hi and R_lo, the observed value, x, and the mean (mu) and standard deviation (sigma) of the
    distribution of good observations assuming they are normally distributed. Gross errors are
    assumed to be uniformly distributed between R_lo and R_hi.

    Parameters
    ----------
    p0: float
        Prior probability of gross error.
    q: float
        Quantization of x, i.e. x is an integer multiple of Q.
    r_hi: float
        The upper limit on x imposed by previous QC choices.
    r_lo: float
        The lower limit on x imposed by previous QC choices.
    x: float
        Observed value for which probability is required.
    mu: float
        The mean of the distribution of good obs.
    sigma: float
        The standard deviation of the distribution of good obs.

    Returns
    -------
    float
        probability of gross error given an observed value

    Raises
    ------
    ValueError
        When inputs are incorrectly specified: p0 < 0, p0 > 1, q <= 0, r_hi < r_lo, x < r_lo, x > r_hi, sigma <= 0
    """
    if p0 < 0.0:
        raise ValueError(f"Value is not positive: p0 = {p0}.")
    if p0 > 1.0:
        raise ValueError(f"Value is greater than 1.0: p0 = {p0}.")
    if q <= 0.0:
        raise ValueError(f"Value is not positive: q = {q}.")
    if r_hi < r_lo:
        raise ValueError(
            f"Lower limit is greater than upper limit: r_hi = {r_hi}, r_lo = {r_lo}."
        )
    if x < r_lo:
        raise ValueError(f"Lower limit is greater than x: r_lo = {r_lo}, x = {x}.")
    if x > r_hi:
        raise ValueError(f"Upper limit is less than x: r_hi = {r_hi}, x = {x}.")
    if sigma <= 0.0:
        raise ValueError(f"Value is not positive: sigma = {sigma}.")

    pgross = (
        p0
        * p_data_given_gross(q, r_hi, r_lo)
        / (
            p0 * p_data_given_gross(q, r_hi, r_lo)
            + (1 - p0) * p_data_given_good(x, q, r_hi, r_lo, mu, sigma)
        )
    )

    assert pgross >= 0.0, pgross
    assert pgross <= 1.0, pgross

    return pgross


def winsorised_mean(inarr: list[float]) -> float:
    """The winsorised mean is a resistant way of calculating an average.

    Parameters
    ----------
    inarr: list of float
        input array to be averaged

    Returns
    -------
    float
        The winsorised mean of the input array with a 25% trimming.

    Note
    ----
    The winsorised mean is that which you get if you set the first quarter of
    the sorted input array to the 1st quartile value and the last quarter
    to the 3rd quartile and then take the mean. This is quite a heavy trimming of
    the distribution. It makes it very resistant - about half the obs can be egregiously
    bad without affecting the mean strongly - but it will be less accurate if
    there are lots of observations, or the quality of the obs is higher.
    """
    length = len(inarr)

    total = 0
    lower = 0
    upper = length - 1

    inarr.sort()

    if length >= 4:
        lower = int(length / 4)
        upper = upper - lower
        total = total + (inarr[lower] + inarr[upper]) * lower

    for j in range(lower, upper + 1):
        total += inarr[j]

    return total / length


def missing_mean(inarr: list[float]) -> float | None:
    """Return mean of input array

    Parameters
    ----------
    inarr : list of float
        List of values for which mean is required. Missing values represented by None in list

    Returns
    -------
    float or None
         Mean of non-missing values or None
    """
    result = 0.0
    num = 0.0
    for val in inarr:
        if val is not None:
            result += val
            num += 1.0
    if num == 0.0:
        return None
    return result / num


def _trim_stat(inarr: Sequence[float], trim: int, stat: str) -> float:
    """Calculate a resistant (aka robust) statistics of an input array given a trimming criteria."""
    arr = copy.deepcopy(inarr)
    stat_func = getattr(np, stat)
    if trim == 0:
        return float(stat_func(arr))

    length = len(arr)
    arr.sort()

    index1 = int(length / trim)

    return float(stat_func(arr[index1 : length - index1]))


def trim_mean(inarr: Sequence[float], trim: int) -> float:
    """Calculate a resistant (aka robust) mean of an input array given a trimming criteria.

    Parameters
    ----------
    inarr: array-like of float, shape (n,)
        1-dimensional value array.
    trim: int
        trimming criteria. A value of 10 trims one tenth of the values off each end of the sorted array
        before calculating the mean.

    Returns
    -------
    float
        Trimmed mean
    """
    return _trim_stat(inarr, trim, "mean")


def trim_std(inarr: Sequence[float], trim: int) -> float:
    """Calculate a resistant (aka robust) standard deviation of an input array given a trimming criteria.

    Parameters
    ----------
    inarr: array-like of float, shape (n,)
        1-dimensional value array.
    trim: int
        trimming criteria. A value of 10 trims one tenth of the values off each end of the sorted array before
        calculating the standard deviation.

    Returns
    -------
    float
        Returns trimmed standard deviation
    """
    return _trim_stat(inarr, trim, "std")
