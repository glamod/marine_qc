from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cdm_reader_mapper.common.getting_files import load_file

from marine_qc.Climatology import (
    Climatology as Climatology_exp,
)
from marine_qc.external_clim import (
    Climatology,
    inspect_climatology,
)


@pytest.fixture(scope="session")
def external_clim():
    kwargs = {
        "cache_dir": ".pytest_cache/external_clim",
        "within_drs": False,
    }
    clim_dict = {}
    clim_dict["AT"] = {
        "mean": load_file(
            "metoffice_qc/external_files/AT_pentad_climatology.nc",
            **kwargs,
        ),
    }
    return clim_dict


@pytest.fixture(scope="session")
def external_at(external_clim):
    return Climatology.open_netcdf_file(
        external_clim["AT"]["mean"],
        "at",
        time_axis="pentad_time",
    )


@pytest.fixture(scope="session")
def expected_at(external_clim):
    return Climatology_exp.from_filename(
        external_clim["AT"]["mean"],
        "at",
    )


@inspect_climatology("climatology")
def _inspect_climatology(climatology, **kwargs):
    return climatology


@inspect_climatology("climatology2")
def _inspect_climatology2(climatology, **kwargs):
    return climatology


@pytest.mark.parametrize(
    "lat, lon, month, day",
    [
        [53.5, 10.0, 7, 4],
        [42.5, 1.4, 2, 16],
        [57.5, 9.4, 6, 1],
        [-68.4, -52.3, 11, 21],
        [-190.0, 10.0, 7, 4],
        [42.5, 95.0, 2, 16],
        [57.5, 9.4, 13, 1],
        [-68.4, -52.3, 11, 42],
        [None, 10.0, 7, 4],
        [42.5, None, 2, 16],
        [57.5, 9.4, None, 1],
        [-68.4, -52.3, 11, None],
    ],
)
def test_get_value(external_at, expected_at, lat, lon, month, day):
    kwargs = {
        "lat": lat,
        "lon": lon,
        "month": month,
        "day": day,
    }
    result = external_at.get_value(**kwargs)
    expected = expected_at.get_value(**kwargs)
    expected = np.float64(np.nan if expected is None else expected)
    assert np.allclose(result, expected, equal_nan=True)


@pytest.mark.parametrize(
    "lat, lon, month, day, expected",
    [
        [53.5, 10.0, 7, 4, 17.317651748657227],
        [42.5, 1.4, 2, 16, 3.752354383468628],
        [57.5, 9.4, 6, 1, 13.33060359954834],
        [-68.4, -52.3, 11, 21, -4.203909397125244],
    ],
)
def test_inspect_climatology(external_at, lat, lon, month, day, expected):
    result = _inspect_climatology(external_at, lat=lat, lon=lon, month=month, day=day)
    assert result == expected


@pytest.mark.parametrize(
    "lat, lon, month, day, expected",
    [
        [53.5, 10.0, 7, 4, 17.317651748657227],
        [42.5, 1.4, 2, 16, 3.752354383468628],
        [57.5, 9.4, 6, 1, 13.33060359954834],
        [-68.4, -52.3, 11, 21, -4.203909397125244],
    ],
)
def test_inspect_climatology_date(external_at, lat, lon, month, day, expected):
    date = pd.to_datetime(f"2002-{month}-{day}")
    result = _inspect_climatology(external_at, lat=lat, lon=lon, date=date)
    assert result == expected


@pytest.mark.parametrize(
    "lat, lon, month, day",
    [
        [-190.0, 10.0, 7, 4],
        [42.5, 95.0, 2, 16],
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
def test_get_y_index(lats, lat0, delta, expected):

    n_lat_axis = int(180 / abs(delta))
    lat_axis = np.arange(n_lat_axis) * delta + lat0
    lats = np.array(lats)
    expected = np.array(expected)

    result = Climatology.get_y_index(lats, lat_axis)

    assert np.all(expected == result)


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
def test_get_x_index(lats, lat0, delta, expected):

    n_lat_axis = int(360 / abs(delta))
    lat_axis = np.arange(n_lat_axis) * delta + lat0
    lats = np.array(lats)
    expected = np.array(expected)

    result = Climatology.get_x_index(lats, lat_axis)

    assert np.all(expected == result)


def test_get_t_index():

    month = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    day = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

    result = Climatology.get_t_index(month, day, 365)
    assert np.all(
        result == np.array([2, 34, 63, 95, 126, 158, 189, 221, 253, 284, 316, 347])
    )

    result = Climatology.get_t_index(month, day, 73)
    assert np.all(result == np.array([1, 7, 13, 19, 26, 32, 38, 45, 51, 57, 64, 70]))

    result = Climatology.get_t_index(month, day, 1)
    assert np.all(result == np.zeros(len(result)))


def test_get_value_fast(external_at):

    lat = np.arange(12) * 15 - 90.0 + 0.1
    lon = np.arange(12) * 30 - 180.0 + 0.1
    month = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    day = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

    result = external_at.get_value_fast(lat, lon, month=month, day=day)
    expected = np.array(
        [
            -24.606144,
            -1.6097184,
            3.8155653,
            12.610888,
            17.65739,
            25.619099,
            24.483362,
            30.759687,
            27.863735,
            6.997858,
            -19.355358,
            -25.576801,
        ]
    )

    assert np.all(result.astype(np.float16) == expected.astype(np.float16))

    lat = np.random.uniform(-90, 90, [1000])
    lon = np.random.uniform(-180, 180, [1000])
    month = np.random.uniform(1, 12, [1000]).astype(int)
    day = np.random.uniform(1, 28, [1000]).astype(int)

    import time

    start = time.time_ns()
    result = external_at.get_value_fast(lat, lon, month=month, day=day)
    mid = time.time_ns()
    result_slow = external_at.get_value(lat, lon, month=month, day=day)
    end = time.time_ns()

    print(result[0:10])
    print(result_slow[0:10])

    print(f"get_value_fast is {(end-mid)/(mid-start)} times faster than get_value\n")
