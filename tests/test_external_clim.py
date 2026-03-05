from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from cdm_reader_mapper.common.getting_files import load_file

from marine_qc.external_clim import (
    Climatology,
    _empty_dataarray,
    _select_point,
    get_climatological_value,
    inspect_climatology,
)


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


def test_empty_dataarray_structure():
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


def test_get_tindex_single_time(external_sst_day):
    assert external_sst_day.get_tindex(7, 4) == 0


def test_get_tindex_daily(external_sst_year):
    assert external_sst_year.get_tindex(1, 1) == 0
    assert external_sst_year.get_tindex(12, 31) == 364


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
@pytest.mark.parametrize("fast", [False, True])
def test_get_value_with_external_at(external_at, lat, lon, month, day, expected, fast):
    kwargs = {
        "lat": lat,
        "lon": lon,
        "month": month,
        "day": day,
    }

    if fast is True:
        result = external_at.get_value_fast(**kwargs)
    else:
        result = external_at.get_value(**kwargs)

    assert np.allclose(result, expected, equal_nan=True)


def test_get_value_fast_raise(external_at):
    with pytest.raises(ValueError, match="No date information given"):
        external_at.get_value_fast(lat=50.0, lon=10.0)


def test_get_value_fast_day(external_sst_day):
    result = external_sst_day.get_value_fast(lat=50.0, lon=-20.0)
    assert result == 285.1153564453125


def test_get_value_fast_repeat(external_at):
    lat = [50.0, 51.0, 52.0]
    lon = [10.0, 11.0, 12.0]
    month = 7
    day = 1

    result = external_at.get_value_fast(
        lat=lat,
        lon=lon,
        month=month,
        day=day,
    )

    assert len(result) == 3


def test_get_value_fast_empty_axes(external_at_no_lat):
    result = external_at_no_lat.get_value_fast(
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
    "lats, lat0, delta, expected",
    [
        ([-89.9, 89.9], -90, 1, [0, 179]),
        ([-89.9, 89.9], -89.5, 1, [0, 179]),
        ([-89.9, 89.9], 90, -1, [179, 0]),
        ([-89.9, 89.9], 89.5, -1, [179, 0]),
        ([-89.9, 89.9], 90, -5, [35, 0]),
        ([-89.9, 89.9], 87.5, -5, [35, 0]),
        ([-89.9, 89.9], -90, 5, [0, 35]),
        ([-89.9, 89.9], -87.5, 5, [0, 35]),
    ],
)
def test_get_y_index_pass(lats, lat0, delta, expected):
    n_lat_axis = int(180 / abs(delta))
    lat_axis = np.arange(n_lat_axis) * delta + lat0
    lats = np.array(lats)
    expected = np.array(expected)

    result = Climatology.get_y_index(lats, lat_axis)

    assert np.all(expected == result)


def test_get_y_index_raise():
    lats = np.array([-10.0, 10.0])
    lat_axis = np.arange(180) + (-88)

    with pytest.raises(RuntimeError, match="I can't work this grid out grid box boundaries"):
        Climatology.get_y_index(lats, lat_axis)


@pytest.mark.parametrize(
    "lats, lat0, delta, expected",
    [
        ([-179.9, 179.9], -180, 1, [0, 359]),
        ([-179.9, 179.9], -179.5, 1, [0, 359]),
        ([-179.9, 179.9], 180, -1, [359, 0]),
        ([-179.9, 179.9], 179.5, -1, [359, 0]),
        ([-179.9, 179.9], 180, -5, [71, 0]),
        ([-179.9, 179.9], 177.5, -5, [71, 0]),
        ([-179.9, 179.9], -180, 5, [0, 71]),
        ([-179.9, 179.9], -177.5, 5, [0, 71]),
        ([-180, 180], -180, 1, [0, 359]),
        ([-180, 180], -179.5, 1, [0, 359]),
        ([-180, 180], 180, -1, [359, 0]),
        ([-180, 180], 179.5, -1, [359, 0]),
        ([-180, 180], 180, -5, [71, 0]),
        ([-180, 180], 177.5, -5, [71, 0]),
        ([-180, 180], -180, 5, [0, 71]),
        ([-180, 180], -177.5, 5, [0, 71]),
    ],
)
def test_get_x_index_pass(lats, lat0, delta, expected):
    n_lat_axis = int(360 / abs(delta))
    lat_axis = np.arange(n_lat_axis) * delta + lat0
    lats = np.array(lats)
    expected = np.array(expected)

    result = Climatology.get_x_index(lats, lat_axis)

    assert np.all(expected == result)


def test_get_x_index_raise():
    lons = np.array([-10.0, 10.0])
    lon_axis = np.arange(360) + (-178)

    with pytest.raises(RuntimeError, match="I can't work this grid out grid box boundaries"):
        Climatology.get_x_index(lons, lon_axis)


def test_get_t_index():
    month = np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12])
    day = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 31])

    result = Climatology.get_t_index(month, day, 365)
    assert np.all(result == np.array([0, 1, 33, 62, 94, 125, 157, 188, 220, 252, 283, 315, 346, 364]))

    result = Climatology.get_t_index(month, day, 73)
    assert np.all(result == np.array([0, 0, 6, 12, 18, 25, 31, 37, 44, 50, 56, 63, 69, 72]))

    result = Climatology.get_t_index(month, day, 1)
    assert np.all(result == np.zeros(len(result)))

    result = Climatology.get_t_index(month, day, 10)
    assert np.all(result == -1)


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
