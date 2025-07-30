"""Module to read external climatology files."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from datetime import datetime
from typing import Literal, Sequence, TypeAlias

import cf_xarray  # noqa
import numpy as np
import pandas as pd
import xarray as xr
from numpy import ndarray
from xclim.core.units import convert_units_to

from .auxiliary import ValueFloatType, generic_decorator, isvalid
from .time_control import convert_date, day_in_year, get_month_lengths, which_pentad


def inspect_climatology(
    *climatology_keys: str, optional: str | Sequence[str] = None
) -> Callable:
    """
    A decorator factory to preprocess function arguments that may be Climatology objects.

    This decorator inspects the specified function arguments and, if any are instances of
    `Climatology`, attempts to resolve them to concrete values using their `.get_value(**kwargs)` method.

    Parameters
    ----------
    climatology_keys : str
        Names of required function arguments to be inspected. These should be arguments that may be
        either a float or a `Climatology` object. If a `Climatology` object is detected, it will be
        replaced with the resolved value.

    optional : str or sequence of str, optional
        Argument names that should be treated as optional. If they are explicitly passed when the
        decorated function is called, they will be treated the same way as `climatology_keys`.

    Returns
    -------
    Callable
        A decorator that wraps the target function, processing specified arguments before the function is called.

    Notes
    -----
    - If a `Climatology` object is found, it will be resolved using its `.get_value(**kwargs)` method.
    - If required keys for `.get_value()` are missing from the function's `**kwargs`, a warning will be issued.
    - If resolution fails, the value will be replaced with `np.nan`.
    """
    if isinstance(optional, str):
        optional = [optional]
    elif optional is None:
        optional = []

    def pre_handler(arguments: dict, **meta_kwargs):
        active_keys = list(climatology_keys)
        for opt in optional:
            if opt in arguments:
                active_keys.append(opt)
        for clim_key in active_keys:
            if clim_key not in arguments:
                raise TypeError(
                    f"Missing expected argument '{clim_key}' in function '{pre_handler.__funcname__}'. "
                    "The decorator requires this argument to be present."
                )
            climatology = arguments[clim_key]
            if isinstance(climatology, Climatology):
                get_value_sig = inspect.signature(climatology.get_value)
                required_keys = {
                    name
                    for name, param in get_value_sig.parameters.items()
                    if param.default is param.empty
                    and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
                }
                missing_in_kwargs = required_keys - meta_kwargs.keys()
                if missing_in_kwargs:
                    warnings.warn(
                        f"The following required key-word arguments for 'Climatology.get_value' are missing "
                        f"in function '{pre_handler.__funcname__}': {missing_in_kwargs}. "
                        f"Ensure all required arguments are passed via **kwargs."
                    )
                try:
                    climatology = climatology.get_value(**meta_kwargs)
                except (TypeError, ValueError):
                    climatology = np.nan

            arguments[clim_key] = climatology

    pre_handler._decorator_kwargs = {"lat", "lon", "date", "month", "day"}

    return generic_decorator(pre_handler=pre_handler)


def open_xrdataset(
    files: str | list,
    use_cftime: bool = True,
    decode_cf: bool = False,
    decode_times: bool = False,
    parallel: bool = False,
    data_vars: Literal["all", "minimal", "different"] = "minimal",
    chunks: int | dict | Literal["auto", "default"] | None = "default",
    coords: Literal["all", "minimal", "different"] | None = "minimal",
    compat: Literal[
        "identical", "equals", "broadcast_equals", "no_conflicts", "override", "minimal"
    ] = "override",
    combine: Literal["by_coords", "nested"] | None = "by_coords",
    **kwargs,
) -> xr.Dataset:
    """Optimized function for opening large cf datasets.

    based on [open_xrdataset]_.
    decode_timedelta=False is added to leave variables and
    coordinates with time units in
    {"days", "hours", "minutes", "seconds", "milliseconds", "microseconds"}
    encoded as numbers.

    Parameters
    ----------
    files: str or list
        See [open_mfdataset]_
    use_cftime: bool, default: True
        See [decode_cf]_
    decode_cf: bool, default: True
        See [decode_cf]_
    decode_times: bool, default: False
        See [decode_cf]_
    parallel: bool, default: False
        See [open_mfdataset]_
    data_vars: {"minimal", "different", "all"} or list of str, default: "minimal"
        See [open_mfdataset]
    chunks: int, dict, "auto" or None, optional, default: "default"
        If chunks is "default", set chunks to {"time": 1}
        See [open_mfdataset]
    coords: {"minimal", "different", "all"} or list of str, optional, default: "minimal"
        See [open_mfdataset]
    compat: {"identical", "equals", "broadcast_equals", "no_conflicts", "override", "minimal"}, default: "override"
        See [open_mfdataset]
    combine: {"by_coords", "nested"}, optional, default: "by_coords"
        See [open_mfdataset]_

    Returns
    -------
    xarray.Dataset

    References
    ----------
    .. [open_xrdataset] https://github.com/pydata/xarray/issues/1385#issuecomment-561920115
    .. [open_mfdataset] https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html
    .. [decode_cf] https://docs.xarray.dev/en/stable/generated/xarray.decode_cf.html

    """

    def drop_all_coords(ds):
        return ds.reset_coords(drop=True)

    if chunks == "default":
        chunks = {"time": 1}

    ds = xr.open_mfdataset(
        files,
        parallel=parallel,
        decode_times=decode_times,
        combine=combine,
        preprocess=drop_all_coords,
        decode_cf=decode_cf,
        chunks=chunks,
        data_vars=data_vars,
        coords=coords,
        compat=compat,
        **kwargs,
    )
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=use_cftime)
    return xr.decode_cf(ds, decode_times=time_coder, decode_timedelta=False)


class Climatology:
    """Class for dealing with climatologies, reading, extracting values etc.
    Automatically detects if this is a single field, pentad or daily climatology.

    Parameters
    ----------
    data: xr.DataArray
        Climatology data
    time_axis: str, optional
        Name of time axis.
        Set if time axis in `data` is not CF compatible.
    lat_axis: str, optional
        Name of latitude axis.
        Set if latitude axis in `data` is not CF compatible.
    lon_axis: str, optional
        Name of longitude axis.
        Set if longitude axis in `data` is not CF compatible.
    source_units: str, optional
        Name of units in `data`.
        Set if units are not defined in `data`.
    target_units: str, optional
        Name of target units to which units must conform.
    valid_ntime: int or list, default: [1, 73, 365]
        Number of valid time steps:
        1: single field climatology
        73: pentad climatology
        365: daily climatology
    """

    def __init__(
        self,
        data: xr.DataArray,
        time_axis: str | None = None,
        lat_axis: str | None = None,
        lon_axis: str | None = None,
        source_units: str | None = None,
        target_units: str | None = None,
        valid_ntime: int | list = [1, 73, 365],
    ):
        self.data = data
        self.convert_units_to(target_units, source_units=source_units)
        if time_axis is None:
            self.time_axis = data.cf.coordinates["time"][0]
        else:
            self.time_axis = time_axis
        if lat_axis is None:
            self.lat_axis = data.cf.coordinates["latitude"][0]
        else:
            self.lat_axis = lat_axis
        if lon_axis is None:
            self.lon_axis = data.cf.coordinates["longitude"][0]
        else:
            self.lon_axis = lon_axis
        if not isinstance(valid_ntime, list):
            valid_ntime = [valid_ntime]
        self.ntime = len(data[self.time_axis])
        assert self.ntime in valid_ntime, "weird shaped field"

    @classmethod
    def open_netcdf_file(cls, file_name, clim_name, **kwargs) -> Climatology:
        """Open filename with xarray."""
        ds = open_xrdataset(file_name)
        da = ds[clim_name]
        return cls(da, **kwargs)

    def convert_units_to(self, target_units, source_units=None) -> None:
        """Convert units to user-specific units.

        Parameters
        ----------
        target_units : str
            Target units to which units must conform.

        source_units : str, optional
            Source units if not specified in :py:class:`Climatology`.

        Note
        ----
        For more information see: :py:func:`xclim.core.units.convert_units_to`
        """
        if target_units is None:
            return
        if source_units is not None:
            self.data.attrs["units"] = source_units
        self.data = convert_units_to(self.data, target_units)

    @convert_date(["month", "day"])
    def get_value(
        self,
        lat: float | Sequence[float] | np.ndarray,
        lon: float | Sequence[float] | np.ndarray,
        date: datetime | None | Sequence[datetime | None] | np.ndarray = None,
        month: int | None | Sequence[int | None] | np.ndarray = None,
        day: int | None | Sequence[int | None] | np.ndarray = None,
    ) -> ndarray | pd.Series:
        """Get the value from a climatology at the give position and time.

        Parameters
        ----------
        lat: float, optional
            Latitude of location to extract value from in degrees.
        lon: float, optional
            Longitude of location to extract value from in degrees.
        date: datetime-like, optional
            Date for which the value is required.
        month: int, optional
            Month for which the value is required.
        day: int, optional
            Day for which the value is required.

        Returns
        -------
        ndarray or pd.Series
            Climatology value at specified location and time.

        Note
        ----
        Use only exact matches for selecting time and nearest valid index value for selecting location.
        """
        lat_arr = np.atleast_1d(lat)  # type: np.ndarray
        lon_arr = np.atleast_1d(lon)  # type: np.ndarray
        month_arr = np.atleast_1d(month)  # type: np.ndarray
        day_arr = np.atleast_1d(day)  # type: np.ndarray
        valid_indices = isvalid(lat) & isvalid(lon) & isvalid(month) & isvalid(day)
        result = np.full(lat_arr.shape, None, dtype=float)  # type: np.ndarray

        if isinstance(valid_indices, (bool, np.bool)):
            valid_indices = [valid_indices]

        for i in range(np.size(result)):
            if not valid_indices[i]:
                continue

            mon_i = int(month_arr[i])
            ml = get_month_lengths(2004)
            day_i = int(day_arr[i])
            lat_i = lat_arr[i]
            lon_i = lon_arr[i]
            if (
                mon_i < 1
                or mon_i > 12
                or day_i < 1
                or day_i > ml[mon_i - 1]
                or lat_i < -180
                or lat_i > 180
                or lon_i < -90
                or lon_i > 90
            ):
                continue

            tindex = self.get_tindex(mon_i, day_i)
            data = self.data.isel(**{self.time_axis: tindex})
            data = data.sel(**{self.lat_axis: lat_i}, method="nearest")
            data = data.sel(**{self.lon_axis: lon_i}, method="nearest")
            result[i] = data.values

        if np.isscalar(lat):
            return result[0]

        if isinstance(lat, pd.Series):
            return pd.Series(result, index=lat.index)

        return result

    def get_tindex(self, month: int, day: int) -> int:
        """Get the time index of the input month and day.

        Parameters
        ----------
        month: int
            Month for which the time index is required.
        day: int
            Day for which the time index is required.

        Returns
        -------
        int
            Time index for specified month and day.
        """
        if self.ntime == 1:
            return 0
        if self.ntime == 73:
            return which_pentad(month, day) - 1
        return day_in_year(month, day) - 1


@inspect_climatology("climatology")
def get_climatological_value(climatology: Climatology, **kwargs) -> ndarray:
    """Get the value from a climatology.

    Parameters
    ----------
    climatology: Climatology
        Climatology class
    kwargs: dict
        Pass keyword-arguments to :py:class:~Climatology.get_value`

    Returns
    -------
    ndarray
            Climatology value at specified location and time.
    """
    return climatology


ClimFloatType: TypeAlias = ValueFloatType | Climatology
