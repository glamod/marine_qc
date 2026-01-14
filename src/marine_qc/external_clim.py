"""Module to read external climatology files."""

from __future__ import annotations
import inspect
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Literal, TypeAlias

import cf_xarray  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from numpy import ndarray
from xclim.core.units import convert_units_to

from .auxiliary import (
    ValueFloatType,
    SequenceFloatType,
    generic_decorator,
    isvalid,
    post_format_return_type,
)
from .time_control import (
    convert_date,
    day_in_year,
    day_in_year_array,
    get_month_lengths,
    which_pentad,
    which_pentad_array,
)


def _format_output(
    result: np.ndarray, 
    lat: ValueFloatType,
) -> ValueFloatType:
    """
    Format output to match the input latitude type.
    
    Parameters
    ----------
    result : np.ndarray
        Array of result values.
    lat : ValueFloatType
        Latitude input used to determine output format.

    Returns
    -------
    ValueFloatType
        Result formatted to match the type of `lat`.
    """
    if np.isscalar(lat):
        return result[0]
    if isinstance(lat, pd.Series):
        return pd.Series(result, index=lat.index)
    return result


def _select_point(
    i: int, 
    da_slice: xr.DataArray, 
    lat_arr: SequenceFloatType,
    lon_arr: SequenceFloatType,
    lat_axis: str, 
    lon_axis: str,
) -> Tuple[int, float]:
    """
    Select nearest grid point value for a single lat/lon pair.
    
    Parameters
    ----------
    i : int
        Index of the latitude/longitude pair.
    da_slice : xr.DataArray
        DataArray slice to sample from.
    lat_arr : SequenceFloatType
        Array of latitude values.
    lon_arr : SequenceFloatType
        Array of longitude values.
    lat_axis : str 
        Name of the latitude dimension in `da_slice`.
    lon_axis : str
        Name of the longitude dimension in `da_slice`.

    Returns
    -------
    tuple of (int, float)
        Index i and the selected grid-point value.
    """
    sel = da_slice.sel({lat_axis: lat_arr[i], lon_axis: lon_arr[i]}, method="nearest")
    return i, float(sel.values)


def _empty_dataarray():
    """
    Create an empty 3D DataArray with latitude, time, and longitude dimensions.
    
    Returns
    -------
    xr.DataArray
        Empty 3D DataArray with latitude, time, and longitude dimensions.
    """
    lat = xr.DataArray(
        [],
        dims="latitude",
        coords={"latitude": []},
        attrs={"standard_name": "latitude", "units": "degrees_north"},
    )
    time = xr.DataArray([], dims="time", coords={"time": []}, attrs={"standard_name": "time"})
    lon = xr.DataArray(
        [],
        dims="longitude",
        coords={"longitude": []},
        attrs={"standard_name": "longitude", "units": "degrees_east"},
    )

    return xr.DataArray(
        data=np.empty((0, 0, 0)),
        coords={"latitude": lat, "pentad_time": time, "longitude": lon},
        dims=["latitude", "time", "longitude"],
    )


def inspect_climatology(*climatology_keys: str, optional: str | Sequence[str] | None = None) -> Callable:
    """
    A decorator factory to preprocess function arguments that may be Climatology objects.

    This decorator inspects the specified function arguments and, if any are instances of
    `Climatology`, attempts to resolve them to concrete values using their `.get_value(**kwargs)` method.

    Parameters
    ----------
    *climatology_keys : str
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
        """
        Preprocess specified arguments, resoling Climatology objects to concrete values.
        
        Parameters
        ----------
        arguments : dict
            Function arguments as a dictionary.
        **meta_kwargs
            Additional keyword arguments to pass to `Climatology.get_value_fast()`.
            
        Returns
        -------
        None
            Updates `arguments` in place.
        """
        active_keys = list(climatology_keys)
        active_keys.extend(opt for opt in optional if opt in arguments)
        for clim_key in active_keys:
            if clim_key not in arguments:
                raise TypeError(
                    f"Missing expected argument '{clim_key}' in function '{pre_handler.__funcname__}'. "
                    "The decorator requires this argument to be present."
                )
            climatology = arguments[clim_key]
            if isinstance(climatology, Climatology):
                get_value_sig = inspect.signature(climatology.get_value_fast)
                required_keys = {
                    name
                    for name, param in get_value_sig.parameters.items()
                    if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
                }
                missing_in_kwargs = required_keys - meta_kwargs.keys()
                if missing_in_kwargs:
                    warnings.warn(
                        f"The following required key-word arguments for 'Climatology.get_value' are missing "
                        f"in function '{pre_handler.__funcname__}': {missing_in_kwargs}. "
                        f"Ensure all required arguments are passed via **kwargs.",
                        stacklevel=2,
                    )

                try:
                    climatology = climatology.get_value_fast(**meta_kwargs)
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
    compat: Literal["identical", "equals", "broadcast_equals", "no_conflicts", "override", "minimal"] = "override",
    combine: Literal["by_coords", "nested"] | None = "by_coords",
    **kwargs,
) -> xr.Dataset:
    """
    Optimized function for opening large CF-compliant datasets with xarray.

    based on [open_xrdataset]_.
    decode_timedelta=False is added to leave variables and
    coordinates with time units in
    {"days", "hours", "minutes", "seconds", "milliseconds", "microseconds"}
    encoded as numbers.

    Parameters
    ----------
    files : str or list
        See [open_mfdataset].
    use_cftime : bool, default: True
        See [decode_cf].
    decode_cf : bool, default: True
        See [decode_cf].
    decode_times : bool, default: False
        See [decode_cf].
    parallel : bool, default: False
        See [open_mfdataset].
    data_vars : {"minimal", "different", "all"} or list of str, default: "minimal"
        See [open_mfdataset].
    chunks : int, dict, "auto" or None, optional, default: "default"
        If chunks is "default", set chunks to {"time": 1}
        See [open_mfdataset].
    coords : {"minimal", "different", "all"} or list of str, optional, default: "minimal"
        See [open_mfdataset].
    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override", "minimal"}, default: "override"
        See [open_mfdataset].
    combine : {"by_coords", "nested"}, optional, default: "by_coords"
        See [open_mfdataset].
    **kwargs
        Additional keyword arguments passed to xarray.open_mfdataset.

    Returns
    -------
    xarray.Dataset
        Opened xarray Dataset, optimized for large CF datasets.

    References
    ----------
    .. [open_xrdataset] https://github.com/pydata/xarray/issues/1385#issuecomment-561920115
    .. [open_mfdataset] https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html
    .. [decode_cf] https://docs.xarray.dev/en/stable/generated/xarray.decode_cf.html
    """

    def drop_all_coords(ds):
        """
        Drop all non-dimension coordinates from an xarray Dataset.

        Parameters
        ----------
        ds : xr.Dataset
          Input Dataset from which all coordinates will be removed.

        Returns
        -------
        xr.Dataset
            Dataset with all coordinates dropped.
    """
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
    """
    Class for dealing with climatologies, reading, extracting values etc.
    
    Automatically detects if this is a single field, pentad or daily climatology.

    Parameters
    ----------
    data : xr.DataArray
        Climatology data.
    time_axis : str, optional
        Name of time axis.
        Set if time axis in `data` is not CF compatible.
    lat_axis : str, optional
        Name of latitude axis.
        Set if latitude axis in `data` is not CF compatible.
    lon_axis : str, optional
        Name of longitude axis.
        Set if longitude axis in `data` is not CF compatible.
    source_units : str, optional
        Name of units in `data`.
        Set if units are not defined in `data`.
    target_units : str, optional
        Name of target units to which units must conform.
    valid_ntime : int or list, default: [1, 73, 365]
        Number of valid time steps:
        - 1: single field climatology
        - 73: pentad climatology
        - 365: daily climatology
    """
    def __init__(
        self,
        data: xr.DataArray,
        time_axis: str | None = None,
        lat_axis: str | None = None,
        lon_axis: str | None = None,
        source_units: str | None = None,
        target_units: str | None = None,
        valid_ntime: int | list | None = None,
    ):
        """
        Initialize a Climatology object from an xarray DataArray.

        The climatology type (single-field, pentad, or daily) is inferred from
        the length of the time dimension.  
              
        Parameters
        ----------
        data : xr.DataArray
          Climatology data.
        time_axis : str, optional
          Name of time axis.
          Set if time axis in `data` is not CF compatible.
        lat_axis : str, optional
          Name of latitude axis.
          Set if latitude axis in `data` is not CF compatible.
        lon_axis : str, optional
          Name of longitude axis.
          Set if longitude axis in `data` is not CF compatible.
        source_units : str, optional
          Name of units in `data`.
          Set if units are not defined in `data`.
        target_units : str, optional
          Name of target units to which units must conform.
        valid_ntime : int or list, default: [1, 73, 365]
          Number of valid time steps:
          - 1: single field climatology
          - 73: pentad climatology
          - 365: daily climatology
        
        Raises
        ------
        ValueError
          If the length of the time dimension does not match one of the
          allowed values in ``valid_ntime``.
        """
        if valid_ntime is None:
            valid_ntime = [0, 1, 73, 365]
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
        if self.ntime not in valid_ntime:
            raise ValueError(f"Weird shaped field {self.ntime}. Use one of {valid_ntime}.")

    @classmethod
    def open_netcdf_file(cls, file_name, clim_name, **kwargs) -> Climatology:
        """
        Open a NetCDF climatology file and construct a Climatology instance.
        
        Parameters
        ----------
        file_name : str or path-like
            Path to the NetCDF file to open.
        clim_name : str
            Name of the climatology variable within the NetCDF file.
        **kwargs
            Additional keyword arguments passed to the Climatology constructor.

        Returns
        -------
        Climatology
          A Climatology instance constructed from the specified variable in
          the NetCDF file. If the file cannot be opened, an empty climatology
          object is returned.
        """
        try:
            ds = open_xrdataset(file_name)
            da = ds[clim_name]
            return cls(da, **kwargs)
        except OSError:
            warnings.warn(f"Could not open: {file_name}.", stacklevel=2)
        return cls(_empty_dataarray(), **kwargs)

    def convert_units_to(self, target_units, source_units=None) -> None:
        """
        Convert units to user-specific units.

        Parameters
        ----------
        target_units : str
            Target units to which units must conform.

        source_units : str, optional
            Source units if not specified in :py:class:`Climatology`.

        Notes
        -----
        For more information see: :py:func:`xclim.core.units.convert_units_to`
        """
        if target_units is None:
            return
        if source_units is not None:
            self.data.attrs["units"] = source_units
        self.data = convert_units_to(self.data, target_units)

    @post_format_return_type(["lat"], dtype=float)
    @convert_date(["month", "day"])
    def get_value_fast(
        self,
        lat: float | Sequence[float] | np.ndarray,
        lon: float | Sequence[float] | np.ndarray,
        date: datetime | None | Sequence[datetime | None] | np.ndarray = None,
        month: int | None | Sequence[int | None] | np.ndarray = None,
        day: int | None | Sequence[int | None] | np.ndarray = None,
    ) -> ndarray | pd.Series:
        """
        Get the value from a climatology at the give position and time.

        Parameters
        ----------
        lat : float, optional
            Latitude of location to extract value from in degrees.
        lon : float, optional
            Longitude of location to extract value from in degrees.
        date : datetime-like, optional
            Date for which the value is required.
        month : int, optional
            Month for which the value is required.
        day : int, optional
            Day for which the value is required.

        Returns
        -------
        ndarray or pd.Series
            Climatology value at specified location and time.

        Notes
        -----
        Assumes that the grid is a regular latitude longitude grid. The alternative method `get_value`
        works with non-regular grids.
        """
        lat_arr = np.atleast_1d(lat)  # type: np.ndarray
        lat_arr = np.where(lat_arr is None, np.nan, lat_arr).astype(float)

        lon_arr = np.atleast_1d(lon)  # type: np.ndarray
        lon_arr = np.where(lon_arr is None, np.nan, lon_arr).astype(float)

        month_arr = np.atleast_1d(month)  # type: np.ndarray
        month_arr = np.where(month_arr is None, np.nan, month_arr).astype(float)
        month_arr = np.where(np.isnan(month_arr), -1, month_arr).astype(int)

        day_arr = np.atleast_1d(day)  # type: np.ndarray
        day_arr = np.where(day_arr is None, np.nan, day_arr).astype(float)
        day_arr = np.where(np.isnan(day_arr), -1, day_arr).astype(int)

        valid = isvalid(lat) & isvalid(lon) & isvalid(month) & isvalid(day)
        valid &= (month_arr >= 1) & (month_arr <= 12)

        ml = np.array(get_month_lengths(2004))
        month_lengths = np.zeros_like(month_arr)
        month_lengths[valid] = ml[month_arr[valid] - 1]

        valid &= (day_arr >= 1) & (day_arr <= month_lengths)
        valid &= (lon_arr >= -180) & (lon_arr <= 180)
        valid &= (lat_arr >= -90) & (lat_arr <= 90)

        result = np.full(lat_arr.shape, np.nan, dtype=float)  # type: np.ndarray

        lat_axis = self.data.coords[self.lat_axis].data
        lon_axis = self.data.coords[self.lon_axis].data
        if lat_axis.size == 0 or lon_axis.size == 0:
            return result

        lat_indices = Climatology.get_y_index(lat_arr[valid], lat_axis)
        lon_indices = Climatology.get_x_index(lon_arr[valid], lon_axis)
        time_indices = Climatology.get_t_index(month_arr[valid], day_arr[valid], self.ntime)

        lat_indices = np.array(lat_indices, dtype=int)
        lon_indices = np.array(lon_indices, dtype=int)
        time_indices = np.array(time_indices, dtype=int)

        lat_mask = np.isin(lat_indices, np.arange(len(self.data[self.lat_axis])))
        lon_mask = np.isin(lon_indices, np.arange(len(self.data[self.lon_axis])))
        time_mask = np.isin(time_indices, np.arange(len(self.data[self.time_axis])))

        coord_mask = lat_mask & lon_mask & time_mask
        valid_indices = np.where(valid)[0]
        valid[valid_indices] &= coord_mask

        result[valid] = self.data.values[time_indices[coord_mask], lat_indices[coord_mask], lon_indices[coord_mask]]

        return result

    @staticmethod
    def get_y_index(lat_arr, lat_axis):
        """
        Convert an array of latitudes to an array of indices for the grid.

        Parameters
        ----------
        lat_arr :  ndarray
            Array of latitudes.
        lat_axis : ndarray
            Array containing the latitude axis.

        Returns
        -------
        ndarray
            Array of indices.
        """
        lat_axis_0 = lat_axis[0]
        lat_axis_delta = lat_axis[1] - lat_axis[0]

        # Need to know if grid cells are defined by centres or by lower edges...
        if lat_axis_0 not in [-90.0, 90.0]:
            if lat_axis_0 == -90 + lat_axis_delta / 2.0:
                lat_axis_0 = -90.0
            elif lat_axis_0 == 90 + lat_axis_delta / 2.0:
                lat_axis_0 = 90.0
            else:
                raise RuntimeError("I can't work this grid out grid box boundaries are not at +-90 or +-(90-delta/2)")

        y_index = ((lat_arr - lat_axis_0) / lat_axis_delta).astype(int)

        y_index[y_index >= len(lat_axis)] = len(lat_axis) - 1

        return y_index

    @staticmethod
    def get_x_index(lon_arr, lon_axis):
        """
        Convert an array of longitudes to an array of indices for the grid.

        Parameters
        ----------
        lon_arr : ndarray
            Array of longitudes.
        lon_axis : ndarray
            Array containing the longitude axis.

        Returns
        -------
        ndarray
            Array of indices.
        """
        lon_axis_0 = lon_axis[0]
        lon_axis_delta = lon_axis[1] - lon_axis[0]

        # Need to know if grid cells are defined by centres or by lower edges...
        if lon_axis_0 not in [-180.0, 180.0]:
            if lon_axis_0 == -180 + lon_axis_delta / 2.0:
                lon_axis_0 = -180.0
            elif lon_axis_0 == 180 + lon_axis_delta / 2.0:
                lon_axis_0 = 180.0
            else:
                raise RuntimeError("I can't work this grid out grid box boundaries are not at +-180 or +-(180-delta/2)")

        x_index = ((lon_arr - lon_axis_0) / lon_axis_delta).astype(int)

        x_index[x_index >= len(lon_axis)] = len(lon_axis) - 1

        return x_index

    @staticmethod
    def get_t_index(month, day, ntime):
        """
        Convert arrays of months and days to an array of indices for the grid.

        Parameters
        ----------
        month : ndarray
            Array of months.
        day : ndarray
            Array of days.
        ntime : int
            Number of time points in the grid, valid values are 1, 73 (pentad resolution) and
            365 (daily resolution).

        Returns
        -------
        ndarray
            Array of indices.
        """
        n_points = len(month)
        t_index = np.zeros(n_points)

        if ntime == 1:
            return t_index
        elif ntime == 73:
            return which_pentad_array(month, day) - 1
        elif ntime == 365:
            return day_in_year_array(month=month, day=day) - 1
        return t_index - 1

    @convert_date(["month", "day"])
    def get_value(
        self,
        lat: float | Sequence[float] | np.ndarray,
        lon: float | Sequence[float] | np.ndarray,
        date: datetime | None | Sequence[datetime | None] | np.ndarray = None,
        month: int | None | Sequence[int | None] | np.ndarray = None,
        day: int | None | Sequence[int | None] | np.ndarray = None,
    ) -> ndarray | pd.Series:
        """
        Get the value from a climatology at the give position and time.

        Parameters
        ----------
        lat : float, optional
            Latitude of location to extract value from in degrees.
        lon : float, optional
            Longitude of location to extract value from in degrees.
        date : datetime-like, optional
            Date for which the value is required.
        month : int, optional
            Month for which the value is required.
        day : int, optional
            Day for which the value is required.

        Returns
        -------
        ndarray or pd.Series
            Climatology value at specified location and time.

        Notes
        -----
        Use only exact matches for selecting time and nearest valid index value for selecting location.
        """
        lat_arr = np.atleast_1d(lat)  # type: np.ndarray
        lat_arr = np.where(lat_arr is None, np.nan, lat_arr).astype(float)
        lon_arr = np.atleast_1d(lon)  # type: np.ndarray
        lon_arr = np.where(lon_arr is None, np.nan, lon_arr).astype(float)
        month_arr = np.atleast_1d(month)  # type: np.ndarray
        month_arr = np.where(month_arr is None, np.nan, month_arr).astype(float)
        month_arr = np.where(np.isnan(month_arr), -1, month_arr).astype(int)
        day_arr = np.atleast_1d(day)  # type: np.ndarray
        day_arr = np.where(day_arr is None, np.nan, day_arr).astype(float)
        day_arr = np.where(np.isnan(day_arr), -1, day_arr).astype(int)

        ml = get_month_lengths(2004)
        month_lengths = np.array([ml[m - 1] if 1 <= m <= 12 else 0 for m in month_arr])

        valid = isvalid(lat) & isvalid(lon) & isvalid(month) & isvalid(day)
        valid &= (month_arr >= 1) & (month_arr <= 12)
        valid &= (day_arr >= 1) & (day_arr <= month_lengths)
        valid &= (lon_arr >= -180) & (lon_arr <= 180)
        valid &= (lat_arr >= -90) & (lat_arr <= 90)

        result = np.full(lat_arr.shape, np.nan, dtype=float)  # type: np.ndarray

        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            return _format_output(result, lat)

        grouped = defaultdict(list)
        for i in valid_idx:
            tindex = self.get_tindex(month_arr[i], day_arr[i])
            grouped[tindex].append(i)

        data = self.data.load()

        for tindex, indices in grouped.items():
            da_slice = data.isel({self.time_axis: tindex})

            results = Parallel(n_jobs=-1)(delayed(_select_point)(i, da_slice, lat_arr, lon_arr, self.lat_axis, self.lon_axis) for i in indices)
            for idx, value in results:
                result[idx] = value

        return _format_output(result, lat)

    def get_tindex(self, month: int, day: int) -> int:
        """
        Get the time index of the input month and day.

        Parameters
        ----------
        month : int
            Month for which the time index is required.
        day : int
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
        return day_in_year(month=month, day=day) - 1


@inspect_climatology("climatology")
def get_climatological_value(climatology: Climatology, **kwargs) -> ndarray:
    """
    Get the value from a climatology.

    Parameters
    ----------
    climatology : Climatology
        Climatology class.
    **kwargs : dict
        Pass keyword-arguments to :py:class:~Climatology.get_value`.

    Returns
    -------
    ndarray
            Climatology value at specified location and time.
    """
    return climatology


ClimFloatType: TypeAlias = ValueFloatType | Climatology
