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
    ]
)
def test_get_y_index(lats, lat0, delta, expected):

    n_lat_axis = int(180 / abs(delta))
    lat_axis = np.arange((n_lat_axis)) * delta + lat0
    lats = np.array(lats)
    expected = np.array(expected)

    result = Climatology.get_y_index(lats, lat_axis)

    assert np.all(expected == result)