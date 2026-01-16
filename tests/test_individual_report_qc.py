from __future__ import annotations
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from marine_qc import (
    do_climatology_check,
    do_date_check,
    do_day_check,
    do_hard_limit_check,
    do_missing_value_check,
    do_missing_value_clim_check,
    do_night_check,
    do_position_check,
    do_sst_freeze_check,
    do_supersaturation_check,
    do_time_check,
    do_wind_consistency_check,
)
from marine_qc.auxiliary import (
    convert_to,
    failed,
    isvalid,
    passed,
    untestable,
)
from marine_qc.qc_individual_reports import value_check


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, False),
        (5.7, True),
        (np.nan, False),
    ],
)
def test_isvalid_check(value, expected):
    assert isvalid(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, failed),
        (5.7, passed),
        (np.nan, failed),
        (pd.NA, failed),
    ],
)
def test_value_check(value, expected):
    assert value_check(value) == expected


def test_value_check_numpy():
    result = value_check(np.array([1.0, np.nan, pd.NA]))
    expected = [passed, failed, failed]
    np.testing.assert_allclose(result, expected)


def test_value_check_pandas():
    result = value_check(pd.Series([1.0, None, 2.0]))
    expected = pd.Series([passed, failed, passed])
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "latitude, longitude, expected",
    [
        [0.0, 0.0, passed],
        [45.0, 125.0, passed],
        [91.0, 0.0, failed],
        [-91.0, 0.0, failed],
        [0.0, -180.1, failed],
        [0.0, 360.1, failed],
        [None, 0.0, untestable],
        [0.0, None, untestable],
    ],
)
def test_do_position_check(latitude, longitude, expected):
    assert do_position_check(latitude, longitude) == expected

    latitude = convert_to(latitude, "degrees", "rad")
    units = {
        "lat": "rad",
    }
    assert do_position_check(latitude, longitude, units=units) == expected


def _test_do_position_check_raises_value_error():
    # Make sure that an exception is raised if latitude or longitude is missing
    with pytest.raises(ValueError):
        _ = do_position_check(None, 0.0)
    with pytest.raises(ValueError):
        _ = do_position_check(0.0, None)


@pytest.mark.parametrize(
    "year, month, day, expected",
    [
        (2023, 1, 1, passed),  # 1st January 2023 PASS
        (2023, 2, 29, failed),  # 29th February 2023 FAIL
        (2023, 1, 31, passed),  # 31st January 2023 PASS
        (0, 0, 0, failed),  # 0th of 0 0 FAIL
        (2024, 2, 29, passed),  # 29th February 2024 PASS
        (2000, 2, 29, passed),  # 29th February 2000 PASS
        (1900, 2, 29, failed),  # 29th February 1900 FAIL
        (1899, 3, None, untestable),  # Missing day UNTESTABLE
        (None, 1, 1, untestable),  # Missing year UNTESTABLE
        (1850, None, 1, untestable),  # Missing month UNTESTABLE
    ],
)
def test_do_date_check(year, month, day, expected):
    result = do_date_check(year=year, month=month, day=day)
    assert result == expected


@pytest.mark.parametrize(
    "year, month, day, expected",
    [
        (2023, 1, 1, passed),  # 1st January 2023 PASS
        (2023, 1, 31, passed),  # 31st January 2023 PASS
        (2024, 2, 29, passed),  # 29th February 2024 PASS
        (2000, 2, 29, passed),  # 29th February 2000 PASS
    ],
)
def test_do_date_check_using_date(year, month, day, expected):
    result = do_date_check(date=datetime(year, month, day, 0))
    assert result == expected


def _test_do_date_check_raises_value_error():
    # Make sure that an exception is raised if year or month is set to None
    with pytest.raises(ValueError):
        _ = do_date_check(year=None, month=1, day=1)
    with pytest.raises(ValueError):
        _ = do_date_check(year=1850, month=None, day=1)


@pytest.mark.parametrize(
    "hour, expected",
    [
        (-1.0, failed),  # no negative hours
        (0.0, passed),
        (23.99, passed),
        (24.0, failed),  # 24 hours not allowed
        (29.2, failed),  # nothing over 24 either
        (6.34451, passed),  # check floats
        (None, untestable),
    ],
)
def test_do_time_check(hour, expected):
    assert do_time_check(hour=hour) == expected


@pytest.mark.parametrize(
    "date, expected",
    [
        (datetime(2005, 7, 19, 3, 52), passed),
    ],
)
def test_do_time_check_datetime(date, expected):
    assert do_time_check(date=date) == expected


@pytest.mark.parametrize(
    "year, month, day, hour, latitude, longitude, time, expected",
    [
        (
            2015,
            10,
            15,
            7.8000,
            50.7365,
            -3.5344,
            1.0,
            passed,
        ),  # Known values from direct observation (day); should trigger pass
        (
            2018,
            9,
            25,
            11.5000,
            50.7365,
            -3.5344,
            1.0,
            passed,
        ),  # Known values from direct observation (day); should trigger pass
        (
            2015,
            10,
            15,
            7.5000,
            50.7365,
            -3.5344,
            1.0,
            failed,
        ),  # Known values from direct observation (night); should trigger fail
        (
            2025,
            4,
            17,
            16.04,
            49.160383,
            5.383146,
            1.0,
            passed,
        ),  # Known values from direct observation: should trigger pass
        (
            2015,
            0,
            15,
            7.5000,
            50.7365,
            -3.5344,
            1.0,
            failed,
        ),  # bad month value should trigger fail
        (
            2015,
            10,
            0,
            7.5000,
            50.7365,
            -3.5344,
            1.0,
            failed,
        ),  # bad day value should trigger fail
        (
            2015,
            10,
            15,
            -7.5000,
            50.7365,
            -3.5344,
            1.0,
            failed,
        ),  # bad hour value should trigger fail
        (
            2015,
            1,
            1,
            0.5,
            0.0,
            0.0,
            1,
            failed,
        ),  # 0 lat 0 lon near midnight should trigger fail
        (2015, 1, 1, None, 0.0, 0.0, 1, untestable),  # missing hour should trigger fail
    ],
)
def test_do_day_check(year, month, day, hour, latitude, longitude, time, expected):
    result = do_day_check(
        year=year,
        month=month,
        day=day,
        hour=hour,
        lat=latitude,
        lon=longitude,
        time_since_sun_above_horizon=time,
    )
    assert result == expected

    latitude = convert_to(latitude, "degrees", "rad")
    units = {"lat": "rad"}
    result = do_day_check(
        year=year,
        month=month,
        day=day,
        hour=hour,
        lat=latitude,
        lon=longitude,
        time_since_sun_above_horizon=time,
        units=units,
    )
    assert result == expected


@pytest.mark.parametrize(
    "year, month, day, hour, latitude, longitude, time, expected",
    [
        (
            2015,
            10,
            15,
            7.8000,
            50.7365,
            -3.5344,
            1.0,
            failed,
        ),  # Known values from direct observation (day); should trigger fail
        (
            2018,
            9,
            25,
            11.5000,
            50.7365,
            -3.5344,
            1.0,
            failed,
        ),  # Known values from direct observation (day); should trigger fail
        (
            2015,
            10,
            15,
            7.5000,
            50.7365,
            -3.5344,
            1.0,
            passed,
        ),  # Known values from direct observation (night); should trigger pass
        (
            2025,
            4,
            17,
            16.04,
            49.160383,
            5.383146,
            1.0,
            failed,
        ),  # Known values from direct observation: should trigger fail
        (
            2015,
            0,
            15,
            7.5000,
            50.7365,
            -3.5344,
            1.0,
            failed,
        ),  # bad month value should trigger fail
        (
            2015,
            10,
            0,
            7.5000,
            50.7365,
            -3.5344,
            1.0,
            failed,
        ),  # bad day value should trigger fail
        (
            2015,
            10,
            15,
            -7.5000,
            50.7365,
            -3.5344,
            1.0,
            failed,
        ),  # bad hour value should trigger fail
        (
            2015,
            1,
            1,
            0.5,
            0.0,
            0.0,
            1,
            passed,
        ),  # 0 lat 0 lon near midnight should trigger pass
        (2015, 1, 1, None, 0.0, 0.0, 1, untestable),  # missing hour should trigger fail
    ],
)
def test_do_night_check(year, month, day, hour, latitude, longitude, time, expected):
    result = do_night_check(
        year=year,
        month=month,
        day=day,
        hour=hour,
        lat=latitude,
        lon=longitude,
        time_since_sun_above_horizon=time,
    )
    assert result == expected

    latitude = convert_to(latitude, "degrees", "rad")
    units = {"lat": "rad"}
    result = do_night_check(
        year=year,
        month=month,
        day=day,
        hour=hour,
        lat=latitude,
        lon=longitude,
        time_since_sun_above_horizon=time,
        units=units,
    )
    assert result == expected


@pytest.mark.parametrize("at, expected", [(5.6, passed), (None, failed), (np.nan, failed)])  # not sure if np.nan should trigger FAIL
def test_do_air_temperature_missing_value_check(at, expected):
    assert do_missing_value_check(at) == expected


@pytest.mark.parametrize(
    "at_climatology, expected",
    [(5.5, passed), (None, failed), (np.nan, failed), (56, passed)],
)  # not sure if np.nan should trigger FAIL
def test_do_air_temperature_missing_value_clim_check(at_climatology, expected):
    assert do_missing_value_clim_check(at_climatology) == expected


@pytest.mark.parametrize("sst, expected", [(5.6, passed), (None, failed), (np.nan, failed)])  # not sure if np.nan should trigger FAIL
def test_do_sst_missing_value_check(sst, expected):
    assert do_missing_value_check(sst) == expected


# fmt: off
@pytest.mark.parametrize(
    "value, climate_normal, standard_deviation, stdev_limits, limit, lowbar, expected",
    [
        (8.0, 0.0, "default", None, 8.0, None, passed),  # pass at limit
        (9.0, 0.0, "default", None, 8.0, None, failed),  # fail with anomaly exceeding limit
        (0.0, 9.0, "default", None, 8.0, None, failed),  # fail with same anomaly but negative
        (9.0, 0.0, "default", None, 11.0, None, passed),  # pass with higher limit
        (0.0, 9.0, "default", None, 11.0, None, passed),  # same with negative anomaly
        (None, 0.0, "default", None, 8.0, None, untestable),  # untestable with Nones as inputs
        (9.0, None, "default", None, 8.0, None, untestable),  # untestable with Nones as inputs
        (9.0, 0.0, "default", None, None, None, untestable),  # untestable with Nones as inputs
        (None, 0.0, 1.0, None, 3.0, 0.5, untestable),  # check None returns untestable
        (1.0, None, 1.0, None, 3.0, 0.5, untestable),
        (1.0, 0.0, None, None, 3.0, 0.5, untestable),
        (1.0, 0.0, 2.0, None, 3.0, 0.1, passed),  # Check simple pass 1.0 anomaly with 6.0 limits
        (7.0, 0.0, 2.0, None, 3.0, 0.1, failed),  # Check fail with 7.0 anomaly and 6.0 limits
        (0.4, 0.0, 0.1, None, 3.0, 0.5, passed),  # Anomaly outside std limits but < lowbar
        (0.4, 0.0, 0.1, None, -3.0, 0.5, untestable),  # Anomaly outside std limits but < lowbar
        (None, 0.0, 0.5, [0.0, 1.0], 5.0, None, untestable),  # untestable with None
        (2.0, None, 0.5, [0.0, 1.0], 5.0, None, untestable),  # untestable with None
        (2.0, 0.0, None, [0.0, 1.0], 5.0, None, untestable),  # untestable with None
        (2.0, 0.0, 0.5, [0.0, 1.0], 5.0, None, passed),  # simple pass
        (2.0, 0.0, 0.5, [0.0, 1.0], 3.0, None, failed),  # simple fail
        (3.0, 0.0, 1.5, [0.0, 1.0], 2.0, None, failed),  # fail with limited stdev
        (1.0, 0.0, 0.1, [0.5, 1.0], 5.0, None, passed),  # pass with limited stdev
        (1.0, 0.0, 0.5, [1.0, 0.0], 5.0, None, untestable),  # untestable with limited stdev
        (1.0, 0.0, 0.5, [0.0, 1.0], -1, None, untestable),  # untestable with limited stdev
        (5.6, 2.2, "default", None, 10.0, None, passed),
        (None, 2.2, "default", None, 10.0, None, untestable),
        (np.nan, 2.2, "default", None, 10.0, None, untestable),
        (5.6, 2.2, 3.3, [1.0, 10.0], 2.0, None, passed),
        (15.6, 0.6, 5.0, [1.0, 10.0], 2.0, None, failed),
        (1.0, 0.0, 0.1, [1.0, 10.0], 2.0, None, passed),
        (15.0, 0.0, 25.0, [1.0, 4.0], 2.0, None, failed),
        (None, 2.2, 3.3, [1.0, 10.0], 2.0, None, untestable),
        (np.nan, 2.2, 3.3, [1.0, 10.0], 2.0, None, untestable),
    ],
)
def test_climatology_check(
    value, climate_normal, standard_deviation, stdev_limits, limit, lowbar, expected
):
    assert (
        do_climatology_check(
            value,
            climate_normal,
            limit,
            standard_deviation=standard_deviation,
            standard_deviation_limits=stdev_limits,
            lowbar=lowbar,
        )
        == expected
    )
# fmt: on


def _test_climatology_plus_stdev_check_raises():
    with pytest.raises(ValueError):
        do_climatology_check(1.0, 0.0, 0.5, [1.0, 0.0], 5.0)
    with pytest.raises(ValueError):
        do_climatology_check(1.0, 0.0, 0.5, [0.0, 1.0], -1)


@pytest.mark.parametrize(
    "value, limits, expected",
    [
        (5.0, [-20.0, 20.0], passed),
        (25.0, [-20.0, 20.0], failed),
        (-10.0, [-30, 15.0], passed),
        (5.6, [-10.0, 10.0], passed),
        (15.6, [-10.0, 10.0], failed),
        (None, [-10.0, 10.0], untestable),
        (np.nan, [-10.0, 10.0], untestable),
        (56, [-100, 100], passed),
        (156, [-100, 100], failed),
        (56, [100, -100], untestable),
    ],
)
def test_do_hard_limit_check(value, limits, expected):
    # assert do_hard_limit_check(value, limits) == expected

    value = convert_to(value, "degC", "K")
    units = {"limits": "degC"}
    assert (
        do_hard_limit_check(
            value,
            limits,
            units=units,
        )
        == expected
    )
    # exit()


def test_do_supersaturation_check_array():
    dpt = [3.6, 5.6, 15.6, None, 12.0]
    at2 = [5.56, 5.6, 13.6, 12.0, np.nan]
    expected = [passed, passed, failed, untestable, untestable]
    results = do_supersaturation_check(dpt, at2)
    np.testing.assert_array_equal(results, expected)


@pytest.mark.parametrize(
    "sst, sst_uncertainty, freezing_point, n_sigma, expected",
    [
        (15.0, 0.0, -1.8, 2.0, passed),
        (-15.0, 0.0, -1.8, 2.0, failed),
        (-2.0, 0.0, -2.0, 2.0, passed),
        (-2.0, 0.5, -1.8, 2.0, passed),
        (-1.7, "default", -1.8, "default", passed),
        (-2.0, "default", -1.8, "default", failed),
        (-5.0, 0.5, -1.8, 2.0, failed),
        (0.0, None, -1.8, 2.0, untestable),
        (0.0, 0.0, None, 2.0, untestable),
    ],
)
def test_do_sst_freeze_check(sst, sst_uncertainty, freezing_point, n_sigma, expected):
    assert (
        do_sst_freeze_check(
            sst,
            freezing_point,
            freeze_check_n_sigma=n_sigma,
            sst_uncertainty=sst_uncertainty,
        )
        == expected
    )
    sst = convert_to(sst, "degC", "K")
    units = {"freezing_point": "degC"}
    assert (
        do_sst_freeze_check(
            sst,
            freezing_point,
            freeze_check_n_sigma=n_sigma,
            sst_uncertainty=sst_uncertainty,
            units=units,
        )
        == expected
    )


@pytest.mark.parametrize(
    "sst, freezing_point, n_sigma, expected",
    [
        (5.6, -1.8, 2.0, passed),
        (-5.6, -1.8, 2.0, failed),
        (0.0, -1.8, 2.0, passed),
        (5.6, 11.8, 2.0, failed),
    ],
)
def test_do_sst_freeze_check_nouncertainty(sst, freezing_point, n_sigma, expected):
    assert do_sst_freeze_check(sst, freezing_point, freeze_check_n_sigma=n_sigma) == expected


def test_do_sst_freeze_check_defaults():
    params = {"sst_uncertainty": 0, "freezing_point": -1.8, "freeze_check_n_sigma": 2.0}
    assert do_sst_freeze_check(0.0, **params) == passed
    assert do_sst_freeze_check(-1.8, **params) == passed
    assert do_sst_freeze_check(-2.0, **params) == failed


def _test_sst_freeze_check_raises():
    with pytest.raises(ValueError):
        do_sst_freeze_check(0.0, None, -1.8, 2.0)
    with pytest.raises(ValueError):
        do_sst_freeze_check(0.0, 0.0, None, 2.0)


@pytest.mark.parametrize(
    "dpt, at, expected",
    [
        (3.6, 5.6, passed),  # clearly unsaturated
        (5.6, 5.6, passed),  # 100% saturation
        (15.6, 13.6, failed),  # clearly supersaturated
        (None, 12.0, untestable),  # missing dpt FAIL
        (12.0, None, untestable),  # missing at FAIL
    ],
)
def test_do_supersaturation_check(dpt, at, expected):
    assert do_supersaturation_check(dpt, at) == expected


@pytest.mark.parametrize(
    "wind_speed, wind_direction, expected",
    [
        (None, 4, untestable),  # missing wind speed; failed
        (4, None, untestable),  # missing wind directory; failed
        (0, 0, passed),
        (0, 120, failed),
        (5.0, 0, failed),
        (5, 361, passed),  # do not test hard limits; passed
        (12.0, 362, passed),  # do not test hard limits; passed
        (5, 165, passed),
        (12.0, 73, passed),
    ],
)
def test_do_wind_consistency_check(wind_speed, wind_direction, expected):
    assert (
        do_wind_consistency_check(
            wind_speed,
            wind_direction,
        )
        == expected
    )


def test_do_wind_consistency_check_array():
    wind_speed = [None, 4, 0, 0, 5.0, 5, 12.0, 5, 12.0]
    wind_direction = [4, None, 0, 120, 0, 361, 362, 165, 73]
    expected = [
        untestable,
        untestable,
        passed,
        failed,
        failed,
        passed,
        passed,
        passed,
        passed,
    ]
    results = do_wind_consistency_check(
        wind_speed,
        wind_direction,
    )
    np.testing.assert_array_equal(results, expected)
