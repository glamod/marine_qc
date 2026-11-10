"""Some generally helpful time control functions for base QC."""

from __future__ import annotations
import calendar
import warnings
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any, overload

import numpy as np
import pandas as pd
import xarray as xr

from .auxiliary import (
    DECORATOR_KWARGS,
    SequenceDatetimeType,
    SequenceFloatType,
    ensure_arrays,
    generic_decorator,
    inspect_arrays,
    is_scalar_like,
    isvalid,
    post_format_return_type,
)


__all__ = [
    "convert_date",
    "convert_date_to_hours",
    "day_in_year",
    "get_month_lengths",
    "pentad_to_month_day",
    "split_date",
    "time_difference",
    "valid_month_day",
    "which_pentad",
]


def convert_date(*params: str) -> Callable[..., Any]:
    r"""
    Decorator to extract date components and inject them as function parameters.

    This decorator intercepts the 'date' argument from the function call, splits it into
    its components (e.g., year, month, day), and assigns those components to specified
    parameters in the wrapped function. It supports scalar or sequence inputs for 'date'.

    Parameters
    ----------
    \*params : str
        List of parameter names corresponding to date components to be extracted and
        passed to the decorated function.

    Returns
    -------
    Callable[..., Any]
        A decorator that wraps a function, extracting date components before calling it.

    Notes
    -----
    - The decorator expects the wrapped function to accept the parameters listed in
      `params`. If a parameter is missing, it raises a `ValueError`.
    - If the 'date' argument is None, the original function is called without modification.
    - Supports scalar-like 'date' values as well as iterable sequences.
    - Assumes a helper function `split_date` exists that splits a date into components
      and returns a dictionary mapping parameter names to their values.
    """

    def pre_handler(arguments: dict[str, Any], **meta_kwargs: Any) -> None:
        r"""
        Extract date components from the 'date' argument and inject them into function arguments.

        Parameters
        ----------
        arguments : dict
            Dictionary of function arguments that may include a ``date`` entry.
        \**meta_kwargs : dict
            Additional keyword arguments passed by the decorator framework (unused).
        """
        date = arguments.get("date")

        if date is None:
            return

        if isinstance(date, xr.DataArray):
            date = date.values

        extracted_dict: dict[str, float]
        if is_scalar_like(date):
            scalar = True
            extracted_dict = split_date(date)
        else:
            scalar = False
            extracted_list: list[dict[str, float]] = [split_date(d) for d in date]

        for param in params:
            if param not in arguments:
                raise ValueError(f"Parameter '{param}' is not a valid parameter.")

            if scalar:
                arguments[param] = extracted_dict[param]
            else:
                arguments[param] = [e[param] for e in extracted_list]

    DECORATOR_KWARGS[pre_handler] = set()

    return generic_decorator(pre_handler=pre_handler)


def split_date(date: datetime) -> dict[str, float]:
    """
    Split datetime date into year, month, day and hour.

    Parameters
    ----------
    date : datetime
        Date to split.

    Returns
    -------
    dict
        Dictionary containing year, month, day and hour.
    """
    try:
        date = pd.to_datetime(date)
    except (TypeError, ValueError) as e:
        warnings.warn(
            f"Could not convert {date!r} to datetime: {e}",
            stacklevel=2,
        )

    try:
        year: int | float = int(date.year)
    except (AttributeError, ValueError):
        year = np.nan
    try:
        month: int | float = int(date.month)
    except (AttributeError, ValueError):
        month = np.nan
    try:
        day: int | float = int(date.day)
    except (AttributeError, ValueError):
        day = np.nan
    try:
        hour: float = date.hour + date.minute / 60.0 + date.second / 3600.0
    except (AttributeError, ValueError):
        hour = np.nan

    return {"year": year, "month": month, "day": day, "hour": hour}


def pentad_to_month_day(p: int) -> tuple[int, int]:
    """
    Given a pentad number, return the month and day of the first day in the pentad.

    Parameters
    ----------
    p : int
        Pentad number from 1 to 73.

    Returns
    -------
    tuple of int
        A tuple of two ints representing month and day of the first day of the pentad.
    """
    if not (0 < p < 74):
        raise ValueError(f"Invalid p: {p}. Must be between 1 and 73")
    start = date(2001, 1, 1) + timedelta(days=int((p - 1) * 5))
    return start.month, start.day


def valid_month_day(year: int | None = None, month: int = 1, day: int = 1) -> bool:
    """
    Return True if month and day combination are allowed, False otherwise. Assumes that Feb 29th is valid.

    Parameters
    ----------
    year : int, optional, default: None
        Year to be tested.
        If none, set year to default leap year.
    month : int, default: 1
        Month to be tested.
    day : int, default: 1
        Day to be tested.

    Returns
    -------
    bool
        True if month and day (or year month and day) are a valid combination (e.g. 12th March) and False if not
        (e.g. 30th February).

    Notes
    -----
    Assumes that February 29th is a valid date.
    """
    try:
        date(2004 if year is None else year, month, day)
    except ValueError:
        return False
    return True


@overload
def day_in_year(year: int | np.ndarray | None, month: int, day: int | np.ndarray) -> int: ...  # numpydoc ignore=GL08
@overload
def day_in_year(year: int | np.ndarray | None, month: np.ndarray, day: int | np.ndarray) -> np.ndarray: ...  # numpydoc ignore=GL08
@post_format_return_type("month", dtype=int)
def day_in_year(year: int | np.ndarray | None = None, month: int | np.ndarray = 1, day: int | np.ndarray = 1) -> int | np.ndarray:
    """
    Get the day in year from 1 to 365 or 366 (leap).

    Parameters
    ----------
    year : :py:obj:`~marine_qc.ValueIntType`, optional, default: None
        Year to be tested.
        If None, set year to default leap year (2004).
    month : :py:obj:`~marine_qc.ValueIntType`, default: 1
        Month to be tested.
    day : :py:obj:`~marine_qc.ValueIntType`, default: 1
        Day to be tested.

    Returns
    -------
    :py:obj:`~marine_qc.ValueIntType`
        Day in year. If year is not specified then the year is treated as a leap year.
    """

    def _day_in_year(year: int, month: int, day: int) -> int:
        """
        Get the day in year from 1 to 365 or 366 (leap).

        Parameters
        ----------
        year : int
          Year to be tested.
        month : int
            Month to be tested.
        day : int
            Day to be tested.

        Returns
        -------
        int
            Day in year.
        """
        _date = date(int(year), int(month), int(day))
        return _date.timetuple().tm_yday

    _day_in_year_vectorized = np.vectorize(_day_in_year, otypes=[int])

    month_arr: np.ndarray = np.atleast_1d(month)
    day_arr: np.ndarray = np.atleast_1d(day)
    if year is None:
        year_arr = np.full(month_arr.shape, np.nan)
    else:
        year_arr = np.atleast_1d(year)

    if len(year_arr) == 1 and len(month_arr) > 1:
        year_arr = np.full(month_arr.shape, year_arr)

    year_invalid = np.logical_not(isvalid(year_arr))
    year_arr[year_invalid] = 2004

    return _day_in_year_vectorized(year_arr, month_arr, day_arr)  # type: ignore[no-any-return]


@overload
def which_pentad(year: int | np.ndarray | None, month: int, day: int | np.ndarray) -> int: ...  # numpydoc ignore=GL08
@overload
def which_pentad(year: int | np.ndarray | None, month: np.ndarray, day: int | np.ndarray) -> np.ndarray: ...  # numpydoc ignore=GL08
def which_pentad(year: int | np.ndarray | None = None, month: int | np.ndarray = 1, day: int | np.ndarray = 1) -> int | np.ndarray:
    """
    Take month and day as inputs and return pentad in range 1-73.

    Parameters
    ----------
    year : :py:obj:`~marine_qc.ValueIntType`, optional, default: None
        Year to be tested.
        If None, set year to default leap year (2004).
    month : :py:obj:`~marine_qc.ValueIntType`, default: 1
        Month to be tested.
    day : :py:obj:`~marine_qc.ValueIntType`, default: 1
        Day to be tested.

    Returns
    -------
    :py:obj:`~marine_qc.ValueIntType`
        Pentad (5-day period) containing input day, from 1 (1 Jan-5 Jan) to 73 (27-31 Dec).
    """
    return (day_in_year(year, month, day) - 1) // 5 + 1


def get_month_lengths(year: int) -> list[int]:
    """
    Return a list holding the lengths of the months in a given year.

    Parameters
    ----------
    year : int
        Year for which you want month lengths.

    Returns
    -------
    list of int
        List of month lengths.
    """
    return [calendar.monthrange(year, month)[1] for month in range(1, 13)]


def convert_date_to_hours(dates: Sequence[datetime]) -> Sequence[float]:
    """
    Convert an array of datetimes to an array of hours since the first element.

    Parameters
    ----------
    dates : array-like of datetime, shape (n,)
        1-dimensional date array.

    Returns
    -------
    array-like of float, shape (n,)
        1- dimensional array containing hours since the first element in the array.
    """
    if not dates:
        return []

    start = dates[0]
    return [(dt - start).total_seconds() / 3600 for dt in dates]


def get_utc_time(
    year: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
    zone: int,
    dasvtm: int,
) -> datetime:
    """
    Calculate UTC time.

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
    second : int
        Second.
    zone : int
        The local international time zone, counted westward from Greenwich.
    dasvtm : int
        1 if daylight saving time is in effect, otherwise 0.

    Returns
    -------
    datetime
        UTC time.
    """
    local_time = datetime(year, 1, 1) + timedelta(days=day - 1, hours=hour, minutes=minute, seconds=second)
    utc_offset_hours = -zone + dasvtm
    utc_time = (local_time - timedelta(hours=utc_offset_hours)).replace(tzinfo=UTC)
    return utc_time


@post_format_return_type("times1", "times2", dtype=float)
@inspect_arrays("times1", "times2")
def time_difference(times1: SequenceDatetimeType, times2: SequenceDatetimeType) -> SequenceFloatType:
    """
    Convert two arrays of datetimes to the difference in hours.

    Parameters
    ----------
    times1 : :py:obj:`~marine_qc.SequenceDatetimeType`
        1-dimensional array of reference time points.
    times2 : :py:obj:`~marine_qc.SequenceDatetimeType`
        1-dimensional array of time points to compare against ``times1``.

    Returns
    -------
    :py:obj:`~marine_qc.SequenceFloatType`
        1-dimensional array containing the time difference in hours
        computed as ``times2 - times1``.

    Raises
    ------
    TypeError
        If `inspect_arrays` does not return numpy ndarrays.
    """
    times1_arr, times2_arr = ensure_arrays(times1=times1, times2=times2)  # pylint: disable=unbalanced-tuple-unpacking

    times1_arr = pd.to_datetime(times1_arr, errors="coerce").values
    times2_arr = pd.to_datetime(times2_arr, errors="coerce").values

    valid = isvalid(times1_arr) & isvalid(times2_arr)

    result: np.ndarray = np.full(times1_arr.shape, np.nan, dtype=float)
    result[valid] = (times2_arr[valid] - times1_arr[valid]) / np.timedelta64(1, "h")

    return result
