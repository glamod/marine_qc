from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

from cdm_reader_mapper.common.getting_files import load_file

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
        )
    }    
    clim_dict["DPT"] = {
        "mean": load_file(
            "metoffice_qc/external_files/DPT_pentad_climatology.nc",
            **kwargs,
        )
    }  
    clim_dict["SLP"] = {
        "mean": load_file(
            "metoffice_qc/external_files/SLP_pentad_climatology.nc",
            **kwargs,
        )
    }        
    clim_dict["SST"] = {
        "mean": load_file(
            "metoffice_qc/external_files/SST_daily_climatology_january.nc",
            **kwargs,
        )
    }    
    clim_dict["SST2"] = {
        "mean": load_file(
            "metoffice_qc/external_files/HadSST2_pentad_climatology.nc",
            **kwargs,
        )
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
def external_dpt(external_clim):
    return Climatology.open_netcdf_file(
        external_clim["DPT"]["mean"],
        "dpt",
        time_axis="pentad_time",
    )  
       

@pytest.fixture(scope="session")
def external_slp(external_clim):
    return Climatology.open_netcdf_file(
        external_clim["SLP"]["mean"],
        "slp",
    )


@pytest.fixture(scope="session")
def external_sst(external_clim):
    return Climatology.open_netcdf_file(
        external_clim["SST"]["mean"],
        "sst",
        valid_ntime=31,        
    )
    
    
@pytest.fixture(scope="session")
def external_sst2(external_clim):
    return Climatology.open_netcdf_file(
        external_clim["SST2"]["mean"],
        "sst",
    )


@inspect_climatology("climatology")
def _inspect_climatology(climatology, **kwargs):
    return climatology


@inspect_climatology("climatology2")
def _inspect_climatology2(climatology, **kwargs):
    return climatology

def _get_value(external, lat, lon, month, day, expected):
      kwargs = {
        "lat": lat,
        "lon": lon,
        "month": month,
        "day": day,
      }
      result = external.get_value(**kwargs)
      assert np.allclose(result, expected, equal_nan=True)
      
def _get_value_fast(external, lat, lon, month, day, expected):        
    result = external.get_value_fast(lat, lon, month=month, day=day)
    assert np.allclose(result, expected, equal_nan=True)

@pytest.mark.parametrize(
    "lat, lon, month, day, expected",
    [
          [53.5, 10.0, 7, 4, 17.317652],
          [42.5, 1.4, 2, 16, 3.7523544],
          [57.5, 9.4, 6, 1, 13.330604],
          [-68.4, -52.3, 11, 21, -4.2039094],
          [-190.0, 10.0, 7, 4, np.nan],
          [42.5, 95.0, 2, 16, -6.6292677],
          [57.5, 9.4, 13, 1, np.nan],
          [-68.4, -52.3, 11, 42, np.nan],
          [None, 10.0, 7, 4, np.nan],
          [42.5, None, 2, 16, np.nan],
          [57.5, 9.4, None, 1, np.nan],
          [-68.4, -52.3, 11, None, np.nan],
    ]     
)
def test_get_value_with_external_at(external_at, lat, lon, month, day, expected):
    _get_value(external_at, lat, lon, month, day, expected)


@pytest.mark.parametrize(
    "lat, lon, month, day, expected",
    [
          [53.5, 10.0, 7, 4, 12.243161],
          [42.5, 1.4, 2, 16, -0.2412281],
          [57.5, 9.4, 6, 1, 9.056429],
          [-68.4, -52.3, 11, 21, -6.0470867],
          [-190.0, 10.0, 7, 4, np.nan],
          [42.5, 95.0, 2, 16, -17.384878],
          [57.5, 9.4, 13, 1, np.nan],
          [-68.4, -52.3, 11, 42, np.nan],
          [None, 10.0, 7, 4, np.nan],
          [42.5, None, 2, 16, np.nan],
          [57.5, 9.4, None, 1, np.nan],
          [-68.4, -52.3, 11, None, np.nan],
    ]     
)
def test_get_value_with_external_dpt(external_dpt, lat, lon, month, day, expected):
    _get_value(external_dpt, lat, lon, month, day, expected)


@pytest.mark.parametrize(
    "lat, lon, month, day, expected",
    [
          [53.5, 10.0, 7, 4, 1015.102783],
          [42.5, 1.4, 2, 16, 1017.175170],
          [57.5, 9.4, 6, 1, 1014.887268],
          [-68.4, -52.3, 11, 21, 982.609802],
          [-190.0, 10.0, 7, 4, np.nan],
          [42.5, 95.0, 2, 16, 1029.125366],
          [57.5, 9.4, 13, 1, np.nan],
          [-68.4, -52.3, 11, 42, np.nan],
          [None, 10.0, 7, 4, np.nan],
          [42.5, None, 2, 16, np.nan],
          [57.5, 9.4, None, 1, np.nan],
          [-68.4, -52.3, 11, None, np.nan],
    ]     
)
def test_get_value_with_external_slp(external_slp, lat, lon, month, day, expected):
    _get_value(external_slp, lat, lon, month, day, expected)

@pytest.mark.parametrize(
    "lat, lon, month, day, expected",
    [
          [53.5, 10.0, 1, 4, np.nan],
          [42.5, 1.4, 1, 16, np.nan],
          [57.5, 9.4, 1, 1, 278.65952],
          [-68.4, -52.3, 1, 21, 271.35],
          [-190.0, 10.0, 7, 4, np.nan],
          [42.5, 95.0, 1, 16, np.nan],
          [57.5, 9.4, 13, 1, np.nan],
          [-68.4, -52.3, 11, 42, np.nan],
          [None, 10.0, 7, 4, np.nan],
          [42.5, None, 2, 16, np.nan],
          [57.5, 9.4, None, 1, np.nan],
          [-68.4, -52.3, 11, None, np.nan],
    ]     
)
def test_get_value_with_external_sst(external_sst, lat, lon, month, day, expected):
    _get_value(external_sst, lat, lon, month, day, expected)
    
@pytest.mark.parametrize(
    "lat, lon, month, day, expected",
    [
          [53.5, 10.0, 7, 4, np.nan],
          [42.5, 1.4, 2, 16, np.nan],
          [57.5, 9.4, 6, 1, 11.798330],
          [-68.4, -52.3, 11, 21, -1.799999],
          [-190.0, 10.0, 7, 4, np.nan],
          [42.5, 95.0, 2, 16, np.nan],
          [57.5, 9.4, 13, 1, np.nan],
          [-68.4, -52.3, 11, 42, np.nan],
          [None, 10.0, 7, 4, np.nan],
          [42.5, None, 2, 16, np.nan],
          [57.5, 9.4, None, 1, np.nan],
          [-68.4, -52.3, 11, None, np.nan],
    ]     
)
def test_get_value_with_external_sst2(external_sst2, lat, lon, month, day, expected):
    _get_value(external_sst2, lat, lon, month, day, expected)  


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


def test_get_value_fast_at(external_at):
    lat = np.arange(12) * 15 - 90.0 + 0.1
    lon = np.arange(12) * 30 - 180.0 + 0.1
    month = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    day = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
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
    _get_value_fast(external_at, lat, lon, month, day, expected)


def test_get_value_fast_dpt(external_dpt):
    lat = np.arange(12) * 15 - 90.0 + 0.1
    lon = np.arange(12) * 30 - 180.0 + 0.1
    month = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    day = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    expected = np.array(
        [
            -27.990265,
            -3.525132,
            1.3157192,
            9.453145,
            13.400107,
            20.802454,
            20.59039,
            17.97907,
            -1.6715549,
            -3.3224113,
            -22.183725,
            -28.067549,
        ]
    )
    _get_value_fast(external_dpt, lat, lon, month, day, expected)
    
def test_get_value_fast_slp(external_slp):
    lat = np.arange(12) * 15 - 90.0 + 0.1
    lon = np.arange(12) * 30 - 180.0 + 0.1
    month = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    day = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    expected = np.array(
        [
            994.6687,
            986.797,
            990.4024,
            1014.28455,
            1014.8676,
            1016.58136,
            1014.6026,
            1007.42126,
            1005.2642,
            1021.6543,
            1021.2122,
            1017.6197,
        ]
    )
    _get_value_fast(external_slp, lat, lon, month, day, expected)
    
def test_get_value_fast_sst(external_sst):
    lat = np.arange(12) * 15 - 90.0 + 0.1
    lon = np.arange(12) * 30 - 180.0 + 0.1
    month = np.ones(12, dtype=int)
    day = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    expected = np.array(
        [
            np.nan,
            271.75177,
            277.03308,
            285.70743,
            np.nan,
            300.02383,
            300.77612,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            271.35,
        ]
    )
    _get_value_fast(external_sst, lat, lon, month, day, expected) 
    
def test_get_value_fast_sst2(external_sst2):
    lat = np.arange(12) * 15 - 90.0 + 0.1
    lon = np.arange(12) * 30 - 180.0 + 0.1
    month = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    day = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    expected = np.array(
        [
            np.nan,
            -1.1828293,
            4.2954683,
            12.715132,
            np.nan,
            26.310646,
            25.364834,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            -1.8,
        ]
    )
    _get_value_fast(external_sst2, lat, lon, month, day, expected)         