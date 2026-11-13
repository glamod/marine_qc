"""Module to read external climatology files."""

from __future__ import annotations
import inspect
import os
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias

import cf_xarray  # noqa: F401 pylint: disable=unused-import
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from xclim.core.units import convert_units_to

from .auxiliary import (
    DECORATOR_KWARGS,
    DECORATOR_NAMES,
    SequenceDatetimeType,
    SequenceIntType,
    SequenceNumberType,
    ValueFloatType,
    ValueIntType,
    ValueNumberType,
    generic_decorator,
    isvalid,
    post_format_return_type,
)
from .location_control import get_xindex, get_yindex
from .time_control import (
    convert_date,
    get_month_lengths,
    get_tindex,
)


def _select_point(
    i: int,
    da_slice: xr.DataArray,
    lat_arr: SequenceNumberType,
    lon_arr: SequenceNumberType,
    lat_axis: str,
    lon_axis: str,
) -> tuple[int, float]:
    """
    Select nearest grid point value for a single lat/lon pair.

    Parameters
    ----------
    i : int
        Index of the latitude/longitude pair.
    da_slice : xarray.DataArray
        DataArray slice to sample from.
    lat_arr : :py:obj:`~marine_qc.SequenceNumberType`
        Array of latitude values.
    lon_arr : :py:obj:`~marine_qc.SequenceNumberType`
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


def _empty_dataarray() -> xr.DataArray:
    """
    Create an empty 3D DataArray with latitude, time, and longitude dimensions.

    Returns
    -------
    xarray.DataArray
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


def inspect_climatology(
    *climatology_keys: str,
    optional: str | Sequence[str] | None = None,
) -> Callable[..., Any]:
    r"""
    A decorator factory to preprocess function arguments that may be Climatology objects.

    This decorator inspects the specified function arguments and normalizes them to concrete
    numerical values before the decorated function is executed. Supported input types include
    raw numeric values, xarray objects, file paths, and :py:class:`~marine_qc.Climatology`
    instances.

    Parameters
    ----------
    \*climatology_keys : str
        Names of required function arguments to be inspected. These should be arguments that may be:

        - a numeric value
        - a xarray.DataArray
        - a xarray.Dataset
        - a string or path-like object pointing to a valid NetCDF file on disk
        - a :py:class:`~marine_qc.Climatology` instance

        If a :py:class:`~marine_qc.Climatology` object (or an object convertible to one) is detected,
        it will be resolved to a concrete value using its `.get_value(**kwargs)` method.

    optional : str or sequence of str, optional
        Argument names that should be treated as optional. If they are explicitly passed when the
        decorated function is called, they will be treated the same way as `climatology_keys`.

    Returns
    -------
    Callable[..., Any]
        A decorator that wraps the target function, processing specified arguments before the function is called.

    Raises
    ------
    TypeError
        If a required climatology argument is missing from the decorated function call
    ValueError
        If an xarray.Dataset is provided without specifying `clim_name`, or
        if a string/Path input does not point to a valid file on disk.

    Warns
    -----
    UserWarning
        Issued if required keyword arguments for :py:func:`~marine_qc.Climatology.get_value()`
        are missing. This warning does not stop execution; missing values
        are replaced with numpy.nan.

    Notes
    -----
    - xarray.Dataset inputs require the keyword argument `clim_name` to select
      the relevant data variable.
    - xarray.DataArray inputs are automatically wrapped in a :py:class:`~marine_qc.Climatology` object.
    - String or path-like inputs must point to an existing file and are
      opened via :py:func:`~marine_qc.Climatology.open_netcdf_file`.
    - If a :py:class:`~marine_qc.Climatology` object is processed, it is resolved using
      `.get_value(**kwargs)`.
    - If required keyword arguments for `.get_value()` are missing,
      a warning is issued.
    - If resolution fails due to `TypeError` or `ValueError`,
      the value is replaced with numpy.nan.
    """
    if isinstance(optional, str):
        optional = [optional]
    elif optional is None:
        optional = []

    def pre_handler(arguments: dict[str, Any], **meta_kwargs: Any) -> None:
        r"""
        Preprocess specified arguments, resoling :py:class:`~marine_qc.Climatology` objects to concrete values.

        Parameters
        ----------
        arguments : dict
            Function arguments as a dictionary.
        \**meta_kwargs : dict
            Additional keyword arguments to pass to :py:func:`~marine_qc.Climatology.get_value()`,
            :py:class:`~marine_qc.Climatology` and/or :py:func:`~marine_qc.Climatology.open_netcdf_file()`.

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
                    f"Missing expected argument '{clim_key}' in function '{DECORATOR_NAMES[pre_handler]}'. "
                    "The decorator requires this argument to be present."
                )
            climatology = arguments[clim_key]

            if isinstance(climatology, xr.Dataset):
                data_var = meta_kwargs.pop("clim_name", None)
                if data_var is None:
                    raise ValueError("No data variable to select is specified in climatology.Use parameter 'clim_name'.")
                climatology = climatology[data_var]

            if isinstance(climatology, xr.DataArray):
                clim_kwargs = {
                    param: meta_kwargs.pop(param, None)
                    for param in ["time_axis", "lat_axis", "lon_axis", "source_units", "target_units", "valid_ntime"]
                }
                climatology = Climatology(climatology, **clim_kwargs)

            elif isinstance(climatology, (str, os.PathLike)) and climatology != "default":
                path = Path(climatology)
                if not path.is_file():
                    raise FileNotFoundError(f"{climatology} is not a valid file on disk.")
                clim_kwargs = {
                    param: meta_kwargs.pop(param, None)
                    for param in ["clim_name", "time_axis", "lat_axis", "lon_axis", "source_units", "target_units", "valid_ntime"]
                }
                climatology = Climatology.open_netcdf_file(climatology, **clim_kwargs)

            if isinstance(climatology, Climatology):
                get_value_sig = inspect.signature(climatology.get_value)
                required_keys = {
                    name
                    for name, param in get_value_sig.parameters.items()
                    if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
                }
                missing_in_kwargs = required_keys - meta_kwargs.keys()
                if missing_in_kwargs:
                    warnings.warn(
                        f"The following required key-word arguments for 'Climatology.get_value' are missing "
                        f"in function '{DECORATOR_NAMES[pre_handler]}': {missing_in_kwargs}. "
                        f"Ensure all required arguments are passed via **kwargs.",
                        stacklevel=2,
                    )

                try:
                    climatology = climatology.get_value(**meta_kwargs)
                except (TypeError, ValueError):
                    climatology = np.nan

            arguments[clim_key] = climatology

    DECORATOR_KWARGS[pre_handler] = {
        "lat",
        "lon",
        "date",
        "month",
        "day",
        "clim_name",
        "time_axis",
        "lat_axis",
        "lon_axis",
        "source_units",
        "target_units",
        "valid_ntime",
    }

    return generic_decorator(pre_handler=pre_handler)


def open_xrdataset(
    files: str | list[str] | os.PathLike[str],
    use_cftime: bool = True,
    decode_cf: bool = False,
    decode_times: bool = False,
    parallel: bool = False,
    data_vars: Literal["all", "minimal", "different"] = "minimal",
    chunks: int | dict[str, Any] | Literal["auto", "default"] | None = "default",
    coords: Literal["all", "minimal", "different"] | None = "minimal",
    compat: Literal["identical", "equals", "broadcast_equals", "no_conflicts", "override", "minimal"] = "override",
    combine: Literal["by_coords", "nested"] = "by_coords",
    **kwargs: Any,
) -> xr.Dataset:
    r"""
    Optimized function for opening large CF-compliant datasets with xarray.

    This implementation follows guidance from:
    https://github.com/pydata/xarray/issues/1385#issuecomment-561920115

    ``decode_timedelta=False`` is added to leave variables and coordinates
    with time units in
    {"days", "hours", "minutes", "seconds", "milliseconds", "microseconds"}
    encoded as numbers.

    Parameters
    ----------
    files : str or list of str or path-like
      See the documentation for :py:func:`xarray.open_mfdataset`.
    use_cftime : bool, default: True
      See the documentation for :py:func:`xarray.decode_cf`.
    decode_cf : bool, default: True
      See the documentation for :py:func:`xarray.decode_cf`.
    decode_times : bool, default: False
      See the documentation for :py:func:`xarray.decode_cf`.
    parallel : bool, default: False
      See the documentation for :py:func:`xarray.open_mfdataset`.
    data_vars : {"minimal", "different", "all"} or list of str, default: "minimal"
      See the documentation for :py:func:`xarray.open_mfdataset`.
    chunks : int, dict, "auto" or None, optional, default: "default"
      If ``chunks`` is ``"default"``, chunks are set to ``{"time": 1}``.
      See the documentation for :py:func:`xarray.open_mfdataset`.
    coords : {"minimal", "different", "all"} or list of str, optional, default: "minimal"
      See the documentation for :py:func:`xarray.open_mfdataset`.
    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override", "minimal"},
      default: "override"
      See the documentation for :py:func:`xarray.open_mfdataset`.
    combine : {"by_coords", "nested"}, default: "by_coords"
      See the documentation for :py:func:`xarray.open_mfdataset`.
    \**kwargs : dict
      Additional keyword arguments passed to :py:func:`xarray.open_mfdataset`.

    Returns
    -------
    xarray.Dataset
      Opened xarray Dataset, optimized for large CF datasets.
    """

    def drop_all_coords(ds: xr.Dataset) -> xr.Dataset:
        """
        Drop all non-dimension coordinates from an xarray Dataset.

        Parameters
        ----------
        ds : xarray.Dataset
          Input Dataset from which all coordinates will be removed.

        Returns
        -------
        xarray.Dataset
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
    data : xarray.DataArray
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
        valid_ntime: int | list[int] | None = None,
    ) -> None:
        """
        Initialize a Climatology object from an xarray DataArray.

        The climatology type (single-field, pentad, or daily) is inferred from
        the length of the time dimension.

        Parameters
        ----------
        data : xarray.DataArray
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
        self.data = data
        self.convert_units_to(target_units, source_units=source_units)

        self.time_axis = time_axis or data.cf.coordinates["time"][0]
        self.lat_axis = lat_axis or data.cf.coordinates["latitude"][0]
        self.lon_axis = lon_axis or data.cf.coordinates["longitude"][0]

        valid_ntime = valid_ntime or [0, 1, 73, 365]
        if not isinstance(valid_ntime, (tuple, list)):
            valid_ntime = [valid_ntime]
        self.valid_ntime = valid_ntime
        self.ntime = len(data[self.time_axis])

        if self.ntime not in valid_ntime:
            raise ValueError(f"Weird shaped field {self.ntime}. Use one of {valid_ntime}.")

    @classmethod
    def open_netcdf_file(cls, file_name: str | os.PathLike[str], clim_name: str, **kwargs: Any) -> Climatology:
        r"""
        Open a NetCDF climatology file and construct a Climatology instance.

        Parameters
        ----------
        file_name : str or path-like
            Path to the NetCDF file to open.
        clim_name : str
            Name of the climatology variable within the NetCDF file.
        \**kwargs : dict
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

    def convert_units_to(self, target_units: str | None, source_units: str | None = None) -> None:
        """
        Convert units to user-specific units.

        Parameters
        ----------
        target_units : str
            Target units to which units must conform.

        source_units : str, optional
            Source units if not specified in :py:class:`~marine_qc.Climatology`.

        Notes
        -----
        For more information see: :py:func:`xclim.core.units.convert_units_to`
        """
        if target_units is None:
            return
        if source_units is not None:
            self.data.attrs["units"] = source_units
        self.data = convert_units_to(self.data, target_units)

    def get_tindex(self, month: int | np.ndarray, day: int | np.ndarray) -> int | np.ndarray:
        """
        Convert arrays of months and days to an array of indices for the grid.

        Parameters
        ----------
        month : int or np.ndarray
            Month(s) to select.
        day : int or np.ndarray
            Day(s) to select.

        Returns
        -------
        np.ndarray
            Array of indices.
        """
        return get_tindex(month, day, self.ntime)

    def get_xindex(self, lon: float | np.ndarray) -> int | np.ndarray:
        """
        Convert an array of longitudes to an array of indices for the grid.

        Parameters
        ----------
        lon : float np.ndarray
            Longitude(s) to select.

        Returns
        -------
        np.ndarray
            Array of indices.
        """
        return get_xindex(lon, self.data.coords[self.lon_axis].data)

    def get_yindex(self, lat: float | np.ndarray) -> int | np.ndarray:
        """
        Convert an array of latitudes to an array of indices for the grid.

        Parameters
        ----------
        lat : float or numpy.ndarray
            Latitude(s) to select.

        Returns
        -------
        numpy.ndarray
            Array of indices.
        """
        return get_yindex(lat, self.data.coords[self.lat_axis].data)

    @post_format_return_type("lat", dtype=float)
    @convert_date("month", "day")
    def get_value(
        self,
        lat: SequenceNumberType,
        lon: SequenceNumberType,
        date: SequenceDatetimeType | None = None,
        month: SequenceIntType | None = None,
        day: SequenceIntType | None = None,
        regular: bool = True,
    ) -> SequenceNumberType:
        """
        Get the value from a climatology at the give position and time.

        Parameters
        ----------
        lat : :py:obj:`~marine_qc.SequenceNumberType`, optional
            Latitude of location to extract value from in degrees.
        lon : :py:obj:`~marine_qc.SequenceNumberType`, optional
            Longitude of location to extract value from in degrees.
        date : :py:obj:`~marine_qc.SequenceDatetimeType`, optional
            Date for which the value is required.
        month : :py:obj:`~marine_qc.SequenceIntType`, optional
            Month for which the value is required.
        day : :py:obj:`~marine_qc.SequenceIntType`, optional
            Day for which the value is required.
        regular : bool, default: True
            Set True for regular spatial grids and False for irregular spatial grids.

        Returns
        -------
        :py:obj:`~marine_qc.SequenceNumberType`
            Climatology value at specified location and time.

        Raises
        ------
        TypeError
            If decorator `inspect_arrays` does not return numpy ndarrays.
        ValueError
            If `month` is None, `day` is None and `time_axis` is greater than 1.
        """
        if month is None and day is None:
            if self.ntime > 1:
                raise ValueError("No date information given: {self.ntime} needed")

            month = self.data[self.time_axis].dt.month.values
            day = self.data[self.time_axis].dt.day.values

        lat = np.array(lat, dtype=float)
        lat_arr = np.atleast_1d(lat)
        lon = np.array(lon, dtype=float)
        lon_arr = np.atleast_1d(lon)
        month = np.array(month, dtype=float)
        month_arr = np.atleast_1d(month)
        day = np.array(day, dtype=float)
        day_arr = np.atleast_1d(day)

        month_arr = np.where(np.isnan(month_arr), -1, month_arr).astype(int)
        day_arr = np.where(np.isnan(day_arr), -1, day_arr).astype(int)

        if len(month_arr) == 1 and len(lat_arr) > 1:
            month_arr = np.repeat(month_arr, len(lat_arr))

        if len(day_arr) == 1 and len(lat_arr) > 1:
            day_arr = np.repeat(day_arr, len(lat_arr))

        ml = get_month_lengths(2004)
        month_lengths = np.array([ml[m - 1] if 1 <= m <= 12 else 0 for m in month_arr])

        valid = (
            isvalid(lat_arr)
            & isvalid(lon_arr)
            & isvalid(month_arr)
            & isvalid(day_arr)
            & (month_arr >= 1)
            & (month_arr <= 12)
            & (day_arr >= 1)
            & (day_arr <= month_lengths)
            & (lon_arr >= -180)
            & (lon_arr <= 180)
            & (lat_arr >= -90)
            & (lat_arr <= 90)
        )

        result = np.full(lat_arr.shape, np.nan, dtype=float)

        if not np.any(valid):
            return result

        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            return result

        func = self._get_value_regular if regular else self._get_value_irregular

        result[valid] = func(lat_arr[valid], lon_arr[valid], month_arr[valid], day_arr[valid])

        return result

    def _get_value_regular(  # numpydoc ignore=PR01,RT01
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        month: np.ndarray,
        day: np.ndarray,
    ) -> np.ndarray:
        """Extract values at the specified geographic positions and times, assuming a regular grid."""
        result = np.full(len(lat), np.nan)

        lat_axis = self.data.coords[self.lat_axis].data
        lon_axis = self.data.coords[self.lon_axis].data

        if lat_axis.size == 0 or lon_axis.size == 0:
            return result

        lat_indices = self.get_yindex(lat)
        lon_indices = self.get_xindex(lon)
        time_indices = self.get_tindex(month, day)

        lat_indices = np.atleast_1d(lat_indices).astype(int)
        lon_indices = np.atleast_1d(lon_indices).astype(int)
        time_indices = np.atleast_1d(time_indices).astype(int)

        valid = (
            (lat_indices >= 0)
            & (lat_indices < len(lat_axis))
            & (lon_indices >= 0)
            & (lon_indices < len(lon_axis))
            & (time_indices >= 0)
            & (time_indices < self.ntime)
        )

        result[valid] = self.data.values[time_indices[valid], lat_indices[valid], lon_indices[valid]]
        return result

    def _get_value_irregular(  # numpydoc ignore=PR01,RT01
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        month: np.ndarray,
        day: np.ndarray,
    ) -> np.ndarray:
        """Extract values at the specified geographic positions and times for regular and irregular grids (slow!)."""
        result = np.full(len(lat), np.nan)

        grouped = defaultdict(list)
        tindexes = self.get_tindex(month, day)

        tindexes = np.atleast_1d(tindexes)

        for i, tindex in enumerate(tindexes):
            grouped[tindex].append(i)

        data = self.data.load()

        for tindex, indices in grouped.items():
            da_slice = data.isel({self.time_axis: tindex})

            results = Parallel(n_jobs=-1)(delayed(_select_point)(i, da_slice, lat, lon, self.lat_axis, self.lon_axis) for i in indices)
            for idx, value in results:
                result[idx] = value

        return result


@inspect_climatology("climatology")
def get_climatological_value(climatology: Climatology, **kwargs: Any) -> np.ndarray:
    r"""
    Get the value from a climatology.

    Parameters
    ----------
    climatology : :py:class:`~marine_qc.Climatology`
        Climatology class.
    \**kwargs : dict
        Pass keyword-arguments to :py:func:~Climatology.get_value`.

    Returns
    -------
    np.ndarray
            Climatology value at specified location and time.
    """
    return np.asarray(climatology, dtype=float)


ClimIntType: TypeAlias = ValueIntType | Climatology
ClimFloatType: TypeAlias = ValueFloatType | Climatology
ClimNumberType: TypeAlias = ValueNumberType | Climatology
ClimInputType: TypeAlias = str | os.PathLike[str] | xr.DataArray | xr.Dataset
ClimArgType: TypeAlias = ClimNumberType | ClimInputType
