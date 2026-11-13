from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from marine_qc import Climatology
from marine_qc.helpers.external_clim import (
    _empty_dataarray,
    _select_point,
    get_climatological_value,
    inspect_climatology,
)

from .helpers.load_file import load_file


@pytest.fixture(scope="session")
def external_clim():
    kwargs = {
        "cache_dir": ".pytest_cache/external_clim",
        "within_drs": False,
    }
    clim_dict = {}
    clim_dict["AT"] = load_file(
        "external_files/AT_pentad_climatology.nc",
        **kwargs,
    )
    clim_dict["SST"] = load_file(
        "external_files/SST_daily_climatology_january.nc",
        **kwargs,
    )
    return clim_dict


@pytest.fixture(scope="session")
def external_ds_at(external_clim):
    return xr.open_dataset(external_clim["AT"])


@pytest.fixture(scope="session")
def external_da_at(external_clim):
    return xr.open_dataset(external_clim["AT"])["at"]


@pytest.fixture(scope="session")
def external_at(external_clim):
    return Climatology.open_netcdf_file(
        external_clim["AT"],
        "at",
        time_axis="pentad_time",
    )


@pytest.fixture
def external_at_no_lat(external_at):
    data = external_at.data.copy()
    no_lat = data.isel(latitude=slice(0, 0))
    return Climatology(data=no_lat, time_axis="pentad_time")


@pytest.fixture(scope="session")
def external_sst_day(external_clim):
    da = xr.open_dataset(external_clim["SST"])["sst"]
    data = da.isel(time=slice(0, 1))
    return Climatology(data=data)


@pytest.fixture(scope="session")
def external_sst_year(external_sst_day):
    first_day = external_sst_day.data.copy()
    repeated = xr.concat([first_day] * 365, dim="time")
    repeated = repeated.assign_coords(time=pd.date_range(start="1961-01-01", periods=365))
    repeated.attrs = first_day.attrs.copy()
    for coord in first_day.coords:
        repeated.coords[coord].attrs = first_day.coords[coord].attrs.copy()
    return Climatology(data=repeated)


@inspect_climatology("climatology")
def _inspect_climatology(climatology, **kwargs):
    return climatology


@inspect_climatology("climatology2")
def _inspect_climatology2(climatology, **kwargs):
    return climatology


@pytest.mark.parametrize(
    "lat_coords, lon_coords, data, lat_arr, lon_arr, i, lat_axis, lon_axis, expected",
    [
        # Exact match
        (
            [0.0, 1.0],
            [10.0, 20.0],
            [[100, 200], [300, 400]],
            [0.0],
            [20.0],
            0,
            "lat",
            "lon",
            200.0,
        ),
        # Nearest match
        (
            [0.0, 1.0, 2.0],
            [10.0, 20.0, 30.0],
            [[0, 1, 2], [10, 11, 12], [20, 21, 22]],
            [1.1],  # nearest 1.0
            [19.0],  # nearest 20.0
            0,
            "lat",
            "lon",
            11.0,
        ),
        # Custom axis names
        (
            [0.0, 1.0],
            [10.0, 20.0],
            [[5, 6], [7, 8]],
            [1.0],
            [10.0],
            0,
            "y",
            "x",
            7.0,
        ),
        # Multiple index case (i=1)
        (
            [0.0, 1.0],
            [10.0, 20.0],
            [[1, 2], [3, 4]],
            [0.0, 1.0],
            [10.0, 20.0],
            1,
            "lat",
            "lon",
            4.0,
        ),
    ],
)
def test_select_point(
    lat_coords,
    lon_coords,
    data,
    lat_arr,
    lon_arr,
    i,
    lat_axis,
    lon_axis,
    expected,
):
    # Build DataArray
    da = xr.DataArray(
        np.array(data),
        coords={lat_axis: lat_coords, lon_axis: lon_coords},
        dims=(lat_axis, lon_axis),
    )

    idx, value = _select_point(
        i=i,
        da_slice=da,
        lat_arr=np.array(lat_arr),
        lon_arr=np.array(lon_arr),
        lat_axis=lat_axis,
        lon_axis=lon_axis,
    )

    assert idx == i
    assert value == expected


def test_empty_dataarray():
    da = _empty_dataarray()

    assert da.shape == (0, 0, 0)
    assert da.dims == ("latitude", "time", "longitude")
    assert "latitude" in da.coords
    assert "pentad_time" in da.coords
    assert "longitude" in da.coords
    assert len(da.coords["latitude"]) == 0
    assert len(da.coords["pentad_time"]) == 0
    assert len(da.coords["longitude"]) == 0

    lat = da.coords["latitude"]
    lon = da.coords["longitude"]

    assert lat.attrs["standard_name"] == "latitude"
    assert lat.attrs["units"] == "degrees_north"

    assert lon.attrs["standard_name"] == "longitude"
    assert lon.attrs["units"] == "degrees_east"

    time = da.coords["pentad_time"]
    assert time.attrs["standard_name"] == "time"

    assert isinstance(da.values, np.ndarray)
    assert da.size == 0
    assert da.ndim == 3


def test_climatology_init_clear(external_da_at):
    data = external_da_at.copy()
    data["pentad_time"].attrs["standard_name"] = "time"
    data["pentad_time"].attrs["axis"] = "T"

    climatology = Climatology(
        data=data,
    )

    assert hasattr(climatology, "data")
    assert isinstance(climatology.data, xr.DataArray)
    assert hasattr(climatology, "time_axis")
    assert isinstance(climatology.time_axis, str)
    assert climatology.time_axis == "pentad_time"
    assert hasattr(climatology, "lat_axis")
    assert isinstance(climatology.lat_axis, str)
    assert climatology.lat_axis == "latitude"
    assert hasattr(climatology, "lon_axis")
    assert isinstance(climatology.lon_axis, str)
    assert climatology.lon_axis == "longitude"
    assert hasattr(climatology, "ntime")
    assert isinstance(climatology.ntime, int)
    assert climatology.ntime == 73


def test_climatology_init_set(external_da_at):
    climatology = Climatology(
        data=external_da_at,
        time_axis="pentad_time",
        lat_axis="latitude",
        lon_axis="longitude",
    )

    assert hasattr(climatology, "data")
    assert isinstance(climatology.data, xr.DataArray)
    assert hasattr(climatology, "time_axis")
    assert isinstance(climatology.time_axis, str)
    assert climatology.time_axis == "pentad_time"
    assert hasattr(climatology, "lat_axis")
    assert isinstance(climatology.lat_axis, str)
    assert climatology.lat_axis == "latitude"
    assert hasattr(climatology, "lon_axis")
    assert isinstance(climatology.lon_axis, str)
    assert climatology.lon_axis == "longitude"
    assert hasattr(climatology, "ntime")
    assert isinstance(climatology.ntime, int)
    assert climatology.ntime == 73


def test_climatology_init_convert(external_da_at):
    climatology = Climatology(
        data=external_da_at,
        time_axis="pentad_time",
        lat_axis="latitude",
        lon_axis="longitude",
        source_units="degC",
        target_units="K",
    )

    assert hasattr(climatology, "data")
    assert isinstance(climatology.data, xr.DataArray)
    assert climatology.data.attrs["units"] == "K"
    assert hasattr(climatology, "time_axis")
    assert isinstance(climatology.time_axis, str)
    assert climatology.time_axis == "pentad_time"
    assert hasattr(climatology, "lat_axis")
    assert isinstance(climatology.lat_axis, str)
    assert climatology.lat_axis == "latitude"
    assert hasattr(climatology, "lon_axis")
    assert isinstance(climatology.lon_axis, str)
    assert climatology.lon_axis == "longitude"
    assert hasattr(climatology, "ntime")
    assert isinstance(climatology.ntime, int)
    assert climatology.ntime == 73


def test_climatology_raises(external_da_at):
    with pytest.raises(ValueError, match="Weird shaped field"):
        Climatology(
            data=external_da_at,
            time_axis="pentad_time",
            valid_ntime=25,
        )


def test_climatology_netcdf(external_clim):
    climatology = Climatology.open_netcdf_file(
        external_clim["AT"],
        clim_name="at",
        time_axis="pentad_time",
    )

    assert hasattr(climatology, "data")
    assert isinstance(climatology.data, xr.DataArray)
    assert hasattr(climatology, "time_axis")
    assert isinstance(climatology.time_axis, str)
    assert climatology.time_axis == "pentad_time"
    assert hasattr(climatology, "lat_axis")
    assert isinstance(climatology.lat_axis, str)
    assert climatology.lat_axis == "latitude"
    assert hasattr(climatology, "lon_axis")
    assert isinstance(climatology.lon_axis, str)
    assert climatology.lon_axis == "longitude"
    assert hasattr(climatology, "ntime")
    assert isinstance(climatology.ntime, int)
    assert climatology.ntime == 73


def test_climatology_empty():
    with pytest.warns(UserWarning, match="Could not open"):
        climatology = Climatology.open_netcdf_file("no_data", clim_name="at")

    assert hasattr(climatology, "data")
    assert isinstance(climatology.data, xr.DataArray)
    assert hasattr(climatology, "time_axis")
    assert isinstance(climatology.time_axis, str)
    assert climatology.time_axis == "pentad_time"
    assert hasattr(climatology, "lat_axis")
    assert isinstance(climatology.lat_axis, str)
    assert climatology.lat_axis == "latitude"
    assert hasattr(climatology, "lon_axis")
    assert isinstance(climatology.lon_axis, str)
    assert climatology.lon_axis == "longitude"
    assert hasattr(climatology, "ntime")
    assert isinstance(climatology.ntime, int)
    assert climatology.ntime == 0


def test_get_t_index_day_scalar(external_sst_day):
    assert external_sst_day.get_t_index(7, 4) == 0


def test_get_t_index_year_scalar(external_sst_year):
    assert external_sst_year.get_t_index(1, 1) == 0
    assert external_sst_year.get_t_index(12, 31) == 364


def test_get_t_index_year_list(external_sst_year):
    result = external_sst_year.get_t_index([1, 2, 3], [1, 11, 22])
    assert len(result) == 3
    assert result[0] == 0
    assert result[1] == 41
    assert result[2] == 80


def test_get_t_index_year_array(external_sst_year):
    result = external_sst_year.get_t_index(np.array([1, 2, 3]), np.array([1, 11, 22]))
    assert np.all(result == [0, 41, 80])


def test_get_x_index_day_scalar(external_sst_day):
    assert external_sst_day.get_x_index(10.0) == 190


def test_get_x_index_year_scalar(external_sst_year):
    assert external_sst_year.get_x_index(10.0) == 190
    assert external_sst_year.get_x_index(6.14) == 186


def test_get_x_index_year_list(external_sst_year):
    result = external_sst_year.get_x_index([10.0, 6.14, -73.59])
    assert len(result) == 3
    assert result[0] == 190
    assert result[1] == 186
    assert result[2] == 106


def test_get_x_index_year_array(external_sst_year):
    result = external_sst_year.get_x_index(np.array([10.0, 6.14, -73.59]))
    assert np.all(result == [190, 186, 106])


def test_get_y_index_day_scalar(external_sst_day):
    assert external_sst_day.get_y_index(53.55) == 36


def test_get_y_index_year_scalar(external_sst_year):
    assert external_sst_year.get_y_index(53.55) == 36
    assert external_sst_year.get_y_index(46.20) == 43


def test_get_y_index_year_list(external_sst_year):
    result = external_sst_year.get_y_index([53.55, 46.20, 45.51])
    assert len(result) == 3
    assert result[0] == 36
    assert result[1] == 43
    assert result[2] == 44


def test_get_y_index_year_array(external_sst_year):
    result = external_sst_year.get_y_index(np.array([53.55, 46.20, 45.51]))
    assert np.all(result == [36, 43, 44])


@pytest.mark.parametrize(
    "lat, lon, month, day, expected",
    [
        (53.5, 10.0, 7, 4, 17.317652),
        (42.5, 1.4, 2, 16, 3.7523544),
        (57.5, 9.4, 6, 1, 13.330604),
        (-68.4, -52.3, 11, 21, -4.2039094),
        (-190.0, 10.0, 7, 4, np.nan),
        (42.5, 95.0, 2, 16, -6.6292677),
        (57.5, 9.4, 13, 1, np.nan),
        (-68.4, -52.3, 11, 42, np.nan),
        (None, 10.0, 7, 4, np.nan),
        (42.5, None, 2, 16, np.nan),
        (57.5, 9.4, None, 1, np.nan),
        (-68.4, -52.3, 11, None, np.nan),
    ],
)
@pytest.mark.parametrize("regular", [False, True])
def test_get_value_scalar(external_at, lat, lon, month, day, expected, regular):
    kwargs = {
        "lat": lat,
        "lon": lon,
        "month": month,
        "day": day,
    }

    result = external_at.get_value(**kwargs, regular=regular)

    assert np.allclose(result, expected, equal_nan=True)


def test_get_value_list(external_at):
    result = external_at.get_value(
        lat=[53.5, 42.5, 57.5, -68.4, -190.0, 42.5, 57.5, -68.4, None, 42.5, 57.5, -68.4],
        lon=[10.0, 1.4, 9.4, -52.3, 10.0, 95.0, 9.4, -52.3, 10.0, None, 9.4, -52.3],
        month=[7, 2, 6, 11, 7, 2, 13, 11, 7, 2, None, 11],
        day=[4, 16, 1, 21, 4, 16, 1, 42, 4, 16, 1, None],
    )
    expected = [17.317652, 3.7523544, 13.330604, -4.2039094, np.nan, -6.6292677, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    assert np.allclose(result, expected, equal_nan=True)


def test_get_value_array(external_at):
    result = external_at.get_value(
        lat=np.array([53.5, 42.5, 57.5, -68.4, -190.0, 42.5, 57.5, -68.4, None, 42.5, 57.5, -68.4]),
        lon=np.array([10.0, 1.4, 9.4, -52.3, 10.0, 95.0, 9.4, -52.3, 10.0, None, 9.4, -52.3]),
        month=np.array([7, 2, 6, 11, 7, 2, 13, 11, 7, 2, None, 11]),
        day=np.array([4, 16, 1, 21, 4, 16, 1, 42, 4, 16, 1, None]),
    )
    expected = [17.317652, 3.7523544, 13.330604, -4.2039094, np.nan, -6.6292677, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    assert np.allclose(result, expected, equal_nan=True)


def test_get_value_raise(external_at):
    with pytest.raises(ValueError, match="No date information given"):
        external_at.get_value(lat=50.0, lon=10.0)


def test_get_value_no_time(external_sst_day):
    result = external_sst_day.get_value(lat=50.0, lon=-20.0)
    assert result == 285.1153564453125


def test_get_value_repeat_time(external_at):
    lat = [50.0, 51.0, 52.0]
    lon = [10.0, 11.0, 12.0]
    month = 7
    day = 1

    result = external_at.get_value(
        lat=lat,
        lon=lon,
        month=month,
        day=day,
    )

    assert len(result) == 3

    assert result[0] == 18.194162368774414
    assert result[1] == 17.43874740600586
    assert result[2] == 18.03187370300293


def test_get_value_empty_axes(external_at_no_lat):
    result = external_at_no_lat.get_value(
        lat=50.0,
        lon=10.0,
        month=7,
        day=4,
    )

    assert np.isnan(result).all()


@pytest.mark.parametrize(
    "lat, lon, month, day, expected",
    [
        [53.5, 10.0, 7, 4, 17.317651748657227],
        [42.5, 1.4, 2, 16, 3.752354383468628],
        [57.5, 9.4, 6, 1, 13.33060359954834],
        [-68.4, -52.3, 11, 21, -4.203909397125244],
    ],
)
def test_inspect_climatology_value(external_at, lat, lon, month, day, expected):
    result = _inspect_climatology(external_at, lat=lat, lon=lon, month=month, day=day)
    assert result == expected

    date = pd.to_datetime(f"2002-{month}-{day}")
    result = _inspect_climatology(external_at, lat=lat, lon=lon, date=date)
    assert result == expected


@pytest.mark.parametrize(
    "lat, lon, month, day",
    [
        [-190.0, 10.0, 7, 4],
        [57.5, 9.4, 13, 1],
        [-68.4, -52.3, 11, 42],
        [None, 10.0, 7, 4],
        [42.5, None, 2, 16],
        [57.5, 9.4, None, 1],
        [-68.4, -52.3, 11, None],
    ],
)
def test_inspect_climatology_nan(external_at, lat, lon, month, day):
    result = _inspect_climatology(external_at, lat=lat, lon=lon, month=month, day=day)
    assert np.isnan(result)


def test_inspect_climatology_raise(external_at):
    with pytest.raises(
        TypeError,
        match="Missing expected argument 'climatology2' in function '_inspect_climatology2'. The decorator requires this argument to be present.",
    ):
        _inspect_climatology2(external_at, lat=53.5, lon=10.0, month=7, day=4)


def test_inspect_climatology_warns(external_at):
    with pytest.warns(UserWarning):
        _inspect_climatology(external_at, lat=53.5)


def test_inspect_climatology_ds_pass(external_ds_at):
    result = _inspect_climatology(external_ds_at, lat=53.5, lon=10.0, month=7, day=4, clim_name="at", time_axis="pentad_time")
    assert result == 17.317651748657227


def test_inspect_climatology_ds_raise(external_ds_at):
    with pytest.raises(
        ValueError,
        match="No data variable to select is specified in climatology.",
    ):
        _inspect_climatology(external_ds_at, lat=-190.0, lon=10.0, month=7, day=4)


def test_inspect_climatology_da_pass(external_da_at):
    result = _inspect_climatology(external_da_at, lat=53.5, lon=10.0, month=7, day=4, time_axis="pentad_time")
    assert result == 17.317651748657227


def test_inspect_climatology_str_pass(external_clim):
    result = _inspect_climatology(external_clim["AT"], lat=53.5, lon=10.0, month=7, day=4, time_axis="pentad_time", clim_name="at")
    assert result == 17.317651748657227


def test_inspect_climatology_str_raises(external_clim):
    with pytest.raises(
        KeyError,
        match="No variable named",
    ):
        _inspect_climatology(external_clim["AT"], lat=53.5, lon=10.0, month=7, day=4, time_axis="pentad_time")


def test_inspect_climatology_path_pass(external_clim):
    filepath = Path(external_clim["AT"])
    result = _inspect_climatology(filepath, lat=53.5, lon=10.0, month=7, day=4, time_axis="pentad_time", clim_name="at")
    assert result == 17.317651748657227


def test_inspect_climatology_path_raise():
    with pytest.raises(
        FileNotFoundError,
        match="is not a valid file on disk",
    ):
        _inspect_climatology(Path("invalid_path"), lat=53.5, lon=10.0, month=7, day=4, time_axis="pentad_time")


@pytest.mark.parametrize(
    "lat, lon, month, day, expected",
    [
        [53.5, 10.0, 7, 4, 17.317651748657227],
        [42.5, 1.4, 2, 16, 3.752354383468628],
        [57.5, 9.4, 6, 1, 13.33060359954834],
        [-68.4, -52.3, 11, 21, -4.203909397125244],
    ],
)
def test_get_climatological_value(external_at, lat, lon, month, day, expected):
    result = get_climatological_value(external_at, lat=lat, lon=lon, month=month, day=day)
    assert result == expected
