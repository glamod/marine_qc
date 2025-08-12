from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import marine_qc.Climatology as clim
from marine_qc.auxiliary import (
    failed,
    passed,
    untestable,
)
from marine_qc.qc_grouped_reports import (
    SuperObsGrid,
    do_bayesian_buddy_check,
    do_mds_buddy_check,
    get_threshold_multiplier,
)


@pytest.fixture
def reps():
    reps = {
        "ID": [
            "AAAAAAAAA",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
        ],
        "DATE": [
            "2003-01-01T02:00:00.000000000",
            "2002-12-31T02:00:00.000000000",
            "2002-12-30T02:00:00.000000000",
            "2002-12-29T02:00:00.000000000",
            "2002-12-28T02:00:00.000000000",
            "2003-01-06T02:00:00.000000000",
            "2003-01-07T02:00:00.000000000",
            "2003-01-08T02:00:00.000000000",
            "2003-01-09T02:00:00.000000000",
        ],
        "LAT": [0.5, 1.5, 1.5, 1.5, 0.5, 0.5, -0.5, -0.5, -0.5],
        "LON": [20.5, 20.5, 21.5, 19.5, 19.5, 21.5, 20.5, 21.5, 19.5],
        "SST": [5.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        "SST_CLIM": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    }

    for key in reps:
        reps[key] = np.array(reps[key])

    reps["DATE"] = pd.to_datetime(reps["DATE"]).tolist()

    return reps


@pytest.fixture
def reps2():
    reps = {
        "ID": [
            "AAAAAAAAA",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
        ],
        "LAT": [0.5, 1.5, 1.5, 1.5, 0.5, 0.5, -0.5, -0.5, -0.5],
        "LON": [20.5, 20.5, 21.5, 19.5, 19.5, 21.5, 20.5, 21.5, 19.5],
        "SST": [5.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        "DATE": [
            "2003-01-01T02:00:00.000000000",
            "2002-12-21T02:00:00.000000000",
            "2002-12-20T02:00:00.000000000",
            "2002-12-19T02:00:00.000000000",
            "2002-12-18T02:00:00.000000000",
            "2003-01-16T02:00:00.000000000",
            "2003-01-17T02:00:00.000000000",
            "2003-01-18T02:00:00.000000000",
            "2003-01-19T02:00:00.000000000",
        ],
        "SST_CLIM": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    }

    for key in reps:
        reps[key] = np.array(reps[key])

    reps["DATE"] = pd.to_datetime(reps["DATE"]).tolist()

    return reps


@pytest.fixture
def reps3():
    reps = {
        "ID": [
            "AAAAAAAAA",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
        ],
        "LAT": [0.5, 2.5, 2.5, 2.5, 0.5, 0.5, -1.5, -1.5, -1.5],
        "LON": [20.5, 20.5, 22.5, 18.5, 18.5, 22.5, 20.5, 22.5, 18.5],
        "SST": [5.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        "DATE": [
            "2003-01-01T02:00:00.000000000",
            "2002-12-21T02:00:00.000000000",
            "2002-12-20T02:00:00.000000000",
            "2002-12-19T02:00:00.000000000",
            "2002-12-18T02:00:00.000000000",
            "2003-01-16T02:00:00.000000000",
            "2003-01-17T02:00:00.000000000",
            "2003-01-18T02:00:00.000000000",
            "2003-01-19T02:00:00.000000000",
        ],
        "SST_CLIM": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    }

    for key in reps:
        reps[key] = np.array(reps[key])

    reps["DATE"] = pd.to_datetime(reps["DATE"]).tolist()

    return reps


@pytest.fixture
def dummy_pentad_stdev():
    return clim.Climatology(np.full([73, 180, 360], 1.5))


@pytest.fixture
def dummy_pentad_stdev_empty():
    return clim.Climatology(np.full([73, 180, 360], np.nan))


@pytest.fixture
def reps4():
    reps = {
        "ID": [
            "AAAAAAAAA",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
        ],
        "LAT": [0.5, 3.5, 3.5, 3.5, 0.5, 0.5, -2.5, -2.5, -2.5],
        "LON": [20.5, 20.5, 23.5, 17.5, 17.5, 23.5, 20.5, 23.5, 17.5],
        "SST": [5.0, 1.9, 1.7, 2.0, 2.0, 2.0, 2.0, 2.3, 2.1],
        "DATE": [
            "2003-01-01T02:00:00.000000000",
            "2002-12-21T02:00:00.000000000",
            "2002-12-20T02:00:00.000000000",
            "2002-12-19T02:00:00.000000000",
            "2002-12-18T02:00:00.000000000",
            "2003-01-16T02:00:00.000000000",
            "2003-01-17T02:00:00.000000000",
            "2003-01-18T02:00:00.000000000",
            "2003-01-19T02:00:00.000000000",
        ],
        "SST_CLIM": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    }

    for key in reps:
        reps[key] = np.array(reps[key])  # type: np.ndarray

    reps["DATE"] = pd.to_datetime(reps["DATE"]).tolist()

    return reps


def test_eight_near_neighbours_missing_stdev_defaults_to_one(
    dummy_pentad_stdev_empty, reps
):
    g = SuperObsGrid()
    g.add_multiple_observations(
        reps["LAT"], reps["LON"], reps["DATE"], reps["SST"] - reps["SST_CLIM"]
    )
    g.get_buddy_limits_with_parameters(
        dummy_pentad_stdev_empty,
        [[1, 1, 2], [2, 2, 2], [1, 1, 4], [2, 2, 4]],
        [[0, 5, 15, 100], [0], [0, 5, 15, 100], [0]],
        [[4.0, 3.5, 3.0, 2.5], [4.0], [4.0, 3.5, 3.0, 2.5], [4.0]],
    )
    mn = g.get_buddy_mean(0.5, 20.5, 1, 1)
    sd = g.get_buddy_stdev(0.5, 20.5, 1, 1)

    assert mn == 1.5
    assert sd == 3.5 * 1.0


def test_eight_near_neighbours(dummy_pentad_stdev, reps):
    g = SuperObsGrid()
    g.add_multiple_observations(
        reps["LAT"], reps["LON"], reps["DATE"], reps["SST"] - reps["SST_CLIM"]
    )
    g.get_buddy_limits_with_parameters(
        dummy_pentad_stdev,
        [[1, 1, 2], [2, 2, 2], [1, 1, 4], [2, 2, 4]],
        [[0, 5, 15, 100], [0], [0, 5, 15, 100], [0]],
        [[4.0, 3.5, 3.0, 2.5], [4.0], [4.0, 3.5, 3.0, 2.5], [4.0]],
    )
    mn = g.get_buddy_mean(0.5, 20.5, 1, 1)
    sd = g.get_buddy_stdev(0.5, 20.5, 1, 1)

    assert mn == 1.5
    assert sd == 3.5 * 1.5


def test_eight_distant_near_neighbours(dummy_pentad_stdev, reps2):
    g = SuperObsGrid()
    g.add_multiple_observations(
        reps2["LAT"], reps2["LON"], reps2["DATE"], reps2["SST"] - reps2["SST_CLIM"]
    )
    g.get_buddy_limits_with_parameters(
        dummy_pentad_stdev,
        [[1, 1, 2], [2, 2, 2], [1, 1, 4], [2, 2, 4]],
        [[0, 5, 15, 100], [0], [0, 5, 15, 100], [0]],
        [[4.0, 3.5, 3.0, 2.5], [4.0], [4.0, 3.5, 3.0, 2.5], [4.0]],
    )
    mn = g.get_buddy_mean(0.5, 20.5, 1, 1)
    sd = g.get_buddy_stdev(0.5, 20.5, 1, 1)

    assert mn == 1.5
    assert sd == 3.5 * 1.5


def test_eight_even_more_distant_near_neighbours(dummy_pentad_stdev, reps3):
    g = SuperObsGrid()
    g.add_multiple_observations(
        reps3["LAT"], reps3["LON"], reps3["DATE"], reps3["SST"] - reps3["SST_CLIM"]
    )
    g.get_buddy_limits_with_parameters(
        dummy_pentad_stdev,
        [[1, 1, 2], [2, 2, 2], [1, 1, 4], [2, 2, 4]],
        [[0, 5, 15, 100], [0], [0, 5, 15, 100], [0]],
        [[4.0, 3.5, 3.0, 2.5], [4.0], [4.0, 3.5, 3.0, 2.5], [4.0]],
    )
    mn = g.get_buddy_mean(0.5, 20.5, 1, 1)
    sd = g.get_buddy_stdev(0.5, 20.5, 1, 1)
    assert mn == 1.5
    assert sd == 4.0 * 1.5


def test_eight_too_distant_neighbours(dummy_pentad_stdev, reps4):
    g = SuperObsGrid()
    g.add_multiple_observations(
        reps4["LAT"], reps4["LON"], reps4["DATE"], reps4["SST"] - reps4["SST_CLIM"]
    )
    g.get_buddy_limits_with_parameters(
        dummy_pentad_stdev,
        [[1, 1, 2], [2, 2, 2], [1, 1, 4], [2, 2, 4]],
        [[0, 5, 15, 100], [0], [0, 5, 15, 100], [0]],
        [[4.0, 3.5, 3.0, 2.5], [4.0], [4.0, 3.5, 3.0, 2.5], [4.0]],
    )
    mn = g.get_buddy_mean(0.5, 20.5, 1, 1)
    sd = g.get_buddy_stdev(0.5, 20.5, 1, 1)
    assert mn == 0.0
    assert sd == 500.0


def test_nobs_limits_not_ascending():
    with pytest.raises(ValueError):
        _ = get_threshold_multiplier(0, [10, 5, 0], [4, 3, 2])


def test_lowest_nobs_limit_not_zero():
    with pytest.raises(ValueError):
        _ = get_threshold_multiplier(1, [1, 5, 10], [4, 3, 2])


def test_simple():
    for n in range(0, 20):
        multiplier = get_threshold_multiplier(n, [0, 5, 10], [4, 3, 2])
        if 0 <= n <= 5:
            assert multiplier == 4.0
        elif 5 < n <= 10:
            assert multiplier == 3.0
        else:
            assert multiplier == 2.0


def test_get_threshold_multipliers_different_len_arrays_raises():
    with pytest.raises(ValueError):
        get_threshold_multiplier(50, [0, 5, 10], [1, 2, 3, 4])


def test_get_threshold_multipliers_zero_multiplier_raises():
    with pytest.raises(ValueError):
        get_threshold_multiplier(50, [0, 5, 10], [0, 2, 3])
    with pytest.raises(ValueError):
        get_threshold_multiplier(50, [0, 5, 10], [-1, 2, 3])


def test_get_threshold_multiplier_nob_limits_raises():
    with pytest.raises(ValueError):
        get_threshold_multiplier(50, [1, 2, 3], [3, 6, 9])
    with pytest.raises(ValueError):
        get_threshold_multiplier(50, [0, -2, 3], [3, 6, 9])
    with pytest.raises(ValueError):
        get_threshold_multiplier(50, [0, 4, 2], [3, 6, 9])


def test_get_threshold_multiplier_negative_nobs_raises():
    with pytest.raises(ValueError):
        get_threshold_multiplier(-1, [0, 5, 10], [1, 5, 10])


@pytest.fixture
def reps_():
    reps = {
        "ID": ["AAAAAAAAA", "BBBBBBBBB", "BBBBBBBBB", "BBBBBBBBB"],
        "LAT": [0.5, 0.5, 0.5, 1.5],
        "LON": [0.5, 0.5, 0.5, 0.5],
        "SST": [5.0, 5.0, 5.0, 5.0],
        "DATE": [
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T10:00:00.000000000",
            "2003-12-01T10:00:00.000000000",
        ],
        "SST_CLIM": [0.5, 0.5, 0.5, 0.5],
    }

    for key in reps:
        reps[key] = np.array(reps[key])

    reps["DATE"] = pd.to_datetime(reps["DATE"]).tolist()

    return reps


@pytest.fixture
def reps2_():
    reps = {
        "ID": ["AAAAAAAAA", "BBBBBBBBB", "BBBBBBBBB", "BBBBBBBBB", "BBBBBBBBB"],
        "LAT": [0.5, 0.5, 0.5, 0.5, 1.5],
        "LON": [0.5, 0.5, 0.5, 0.5, 0.5],
        "SST": [5.0, 5.0, 5.0, 2.0, 5.0],
        "DATE": [
            "2003-01-01T00:00:00.000000000",
            "2003-01-01T00:00:00.000000000",
            "2003-01-01T10:00:00.000000000",
            "2003-12-31T10:00:00.000000000",
            "2003-01-01T10:00:00.000000000",
        ],
        "SST_CLIM": [0.5, 0.5, 0.5, 0.5, 0.5],
    }

    for key in reps:
        reps[key] = np.array(reps[key])

    reps["DATE"] = pd.to_datetime(reps["DATE"]).tolist()

    return reps


@pytest.fixture
def buddy_reps():
    reps = {
        "ID": [
            "AAAAAAAAA",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
        ],
        "LAT": [
            0.5,
            1.5,
            1.5,
            1.5,
            0.5,
            0.5,
            -0.5,
            -0.5,
            -0.5,
        ],
        "LON": [0.5, -0.5, 0.5, 1.5, -0.5, 1.5, -0.5, 0.5, 1.5],
        "SST": [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "DATE": [
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
        ],
        "SST_CLIM": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }

    for key in reps:
        reps[key] = np.array(reps[key])

    reps["DATE"] = pd.to_datetime(reps["DATE"]).tolist()

    return reps


@pytest.fixture
def buddy_reps_time(second_date):
    reps = {
        "ID": [
            "AAAAAAAAA",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
        ],
        "LAT": [
            0.5,
            1.5,
            1.5,
            1.5,
            0.5,
            0.5,
            -0.5,
            -0.5,
            -0.5,
        ],
        "LON": [0.5, -0.5, 0.5, 1.5, -0.5, 1.5, -0.5, 0.5, 1.5],
        "SST": [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "DATE": [
            "2003-12-01T00:00:00.000000000",
            second_date,
            second_date,
            second_date,
            second_date,
            second_date,
            second_date,
            second_date,
            second_date,
        ],
        "SST_CLIM": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }

    for key in reps:
        reps[key] = np.array(reps[key])

    reps["DATE"] = pd.to_datetime(reps["DATE"]).tolist()

    return reps


@pytest.fixture
def buddy_reps_spread():
    reps = {
        "ID": [
            "AAAAAAAAA",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
            "BBBBBBBBB",
        ],
        "LAT": [
            0.5,
            2.5,
            2.5,
            2.5,
            0.5,
            0.5,
            -1.5,
            -1.5,
            -1.5,
        ],
        "LON": [0.5, -1.5, 0.5, 2.5, -1.5, 2.5, -1.5, 0.5, 2.5],
        "SST": [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "DATE": [
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
            "2003-12-01T00:00:00.000000000",
        ],
        "SST_CLIM": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }

    for key in reps:
        reps[key] = np.array(reps[key])

    reps["DATE"] = pd.to_datetime(reps["DATE"]).tolist()

    return reps


@pytest.fixture
def buddy_reps_singleton():
    reps = {
        "ID": [
            "AAAAAAAAA",
        ],
        "LAT": [0.5],
        "LON": [0.5],
        "SST": [5.0],
        "DATE": ["2003-12-01T00:00:00.000000000"],
        "SST_CLIM": [0.0],
    }

    for key in reps:
        reps[key] = np.array(reps[key])

    reps["DATE"] = pd.to_datetime(reps["DATE"]).tolist()

    return reps


@pytest.fixture
def dummy_pentad_stdev_():
    return clim.Climatology(np.full([73, 180, 360], 1.0))


def test_get_neighbour_anomalies(reps2_):
    g = SuperObsGrid()
    g.add_multiple_observations(
        reps2_["LAT"], reps2_["LON"], reps2_["DATE"], reps2_["SST"] - reps2_["SST_CLIM"]
    )
    temp_anom, temp_nobs = g.get_neighbour_anomalies([2, 2, 2], 180, 89, 0)

    assert len(temp_anom) == 2
    assert len(temp_nobs) == 2

    assert 4.5 in temp_anom
    assert 1.5 in temp_anom

    assert temp_nobs[0] == 1
    assert temp_nobs[1] == 1


def test_get_neighbour_anomalies_raises(reps2_):
    g = SuperObsGrid()
    g.add_multiple_observations(
        reps2_["LAT"], reps2_["LON"], reps2_["DATE"], reps2_["SST"] - reps2_["SST_CLIM"]
    )

    with pytest.raises(ValueError):
        _temp_anom, _temp_nobs = g.get_neighbour_anomalies([2, 2], 180, 89, 0)


def test_add_one_maxes_limits(reps_, dummy_pentad_stdev_):
    g = SuperObsGrid()
    g.add_single_observation(
        reps_["LAT"][0],
        reps_["LON"][0],
        reps_["DATE"][0].month,
        reps_["DATE"][0].day,
        reps_["SST"][0] - reps_["SST_CLIM"][0],
    )
    g.take_average()
    g.get_new_buddy_limits(
        dummy_pentad_stdev_,
        dummy_pentad_stdev_,
        dummy_pentad_stdev_,
        limits=(2, 2, 4),
        sigma_m=1.0,
        noise_scaling=3.0,
    )

    mn = g.get_buddy_mean(
        reps_["LAT"][0], reps_["LON"][0], reps_["DATE"][0].month, reps_["DATE"][0].day
    )
    sd = g.get_buddy_stdev(
        reps_["LAT"][0], reps_["LON"][0], reps_["DATE"][0].month, reps_["DATE"][0].day
    )

    assert pytest.approx(sd, 0.0001) == 500
    assert mn == 0.0


def test_add_one_maxes_limits_with_missing_stdevs(reps_, dummy_pentad_stdev_empty):
    g = SuperObsGrid()
    g.add_single_observation(
        reps_["LAT"][0],
        reps_["LON"][0],
        reps_["DATE"][0].month,
        reps_["DATE"][0].day,
        reps_["SST"][0] - reps_["SST_CLIM"][0],
    )
    g.take_average()
    g.get_new_buddy_limits(
        dummy_pentad_stdev_empty,
        dummy_pentad_stdev_empty,
        dummy_pentad_stdev_empty,
        limits=(2, 2, 4),
        sigma_m=1.0,
        noise_scaling=3.0,
    )

    mn = g.get_buddy_mean(
        reps_["LAT"][0], reps_["LON"][0], reps_["DATE"][0].month, reps_["DATE"][0].day
    )
    sd = g.get_buddy_stdev(
        reps_["LAT"][0], reps_["LON"][0], reps_["DATE"][0].month, reps_["DATE"][0].day
    )

    assert pytest.approx(sd, 0.0001) == 500
    assert mn == 0.0


def test_add_single_observation_raises():
    g = SuperObsGrid()
    with pytest.raises(ValueError):
        g.add_single_observation(0.0, 500.0, 1, 1, 0.5)
    with pytest.raises(ValueError):
        g.add_single_observation(-100.0, 0.0, 1, 1, 0.5)


def test_add_multiple(reps_, dummy_pentad_stdev_):
    g = SuperObsGrid()
    g.add_multiple_observations(
        reps_["LAT"], reps_["LON"], reps_["DATE"], reps_["SST"] - reps_["SST_CLIM"]
    )
    g.get_new_buddy_limits(
        dummy_pentad_stdev_,
        dummy_pentad_stdev_,
        dummy_pentad_stdev_,
        limits=(2, 2, 4),
        sigma_m=1.0,
        noise_scaling=3.0,
    )

    mn = g.get_buddy_mean(
        reps_["LAT"][0], reps_["LON"][0], reps_["DATE"][0].month, reps_["DATE"][0].day
    )
    sd = g.get_buddy_stdev(
        reps_["LAT"][0], reps_["LON"][0], reps_["DATE"][0].month, reps_["DATE"][0].day
    )

    assert pytest.approx(sd, 0.0001) == math.sqrt(10.0)
    assert mn == 4.5


def test_creation():
    grid = SuperObsGrid()
    assert isinstance(grid, SuperObsGrid)


def test_buddy_check(reps_, dummy_pentad_stdev_):
    limits = [[1, 1, 2], [2, 2, 2], [1, 1, 4], [2, 2, 4]]
    number_of_obs_thresholds = [[0, 5, 15, 100], [0], [0, 5, 15, 100], [0]]
    multipliers = [[4.0, 3.5, 3.0, 2.5], [4.0], [4.0, 3.5, 3.0, 2.5], [4.0]]

    result = do_mds_buddy_check(
        reps_["LAT"],
        reps_["LON"],
        reps_["DATE"],
        reps_["SST"],
        reps_["SST_CLIM"],
        dummy_pentad_stdev_,
        limits,
        number_of_obs_thresholds,
        multipliers,
    )

    assert np.all(result == [passed, passed, passed, passed])


def test_buddy_check_single_ob_flagged_untestable(
    buddy_reps_singleton, dummy_pentad_stdev_
):
    limits = [[1, 1, 2], [2, 2, 2], [1, 1, 4], [2, 2, 4]]
    number_of_obs_thresholds = [[0, 5, 15, 100], [0], [0, 5, 15, 100], [0]]
    multipliers = [[4.0, 3.5, 3.0, 2.5], [4.0], [4.0, 3.5, 3.0, 2.5], [4.0]]

    result = do_mds_buddy_check(
        buddy_reps_singleton["LAT"],
        buddy_reps_singleton["LON"],
        buddy_reps_singleton["DATE"],
        buddy_reps_singleton["SST"],
        buddy_reps_singleton["SST_CLIM"],
        dummy_pentad_stdev_,
        limits,
        number_of_obs_thresholds,
        multipliers,
    )

    assert result[0] == untestable


def test_buddy_check_designed_to_fail(buddy_reps, dummy_pentad_stdev_):
    # One observation with 8 neighbours that has a disparate anomaly
    limits = [[1, 1, 2], [2, 2, 2], [1, 1, 4], [2, 2, 4]]
    number_of_obs_thresholds = [[0, 5, 15, 100], [0], [0, 5, 15, 100], [0]]
    multipliers = [[4.0, 3.5, 3.0, 2.5], [4.0], [4.0, 3.5, 3.0, 2.5], [4.0]]

    result = do_mds_buddy_check(
        buddy_reps["LAT"],
        buddy_reps["LON"],
        buddy_reps["DATE"],
        buddy_reps["SST"],
        buddy_reps["SST_CLIM"],
        dummy_pentad_stdev_,
        limits,
        number_of_obs_thresholds,
        multipliers,
    )

    for i, flag in enumerate(result):
        if i == 0:
            assert flag == 1
        else:
            assert flag == 0


@pytest.mark.parametrize(
    ["second_date", "expected"],
    [
        ["2003-12-11T00:00:00.000000000", failed],
        ["2003-12-21T00:00:00.000000000", failed],
        ["2003-11-21T00:00:00.000000000", failed],
        ["2003-12-25T00:00:00.000000000", untestable],
    ],
)
def test_buddy_check_designed_to_fail_time(
    buddy_reps_time, dummy_pentad_stdev_, expected
):
    # One observation with 8 neighbours that has a disparate anomaly
    limits = [[1, 1, 2], [2, 2, 2], [1, 1, 4], [2, 2, 4]]
    number_of_obs_thresholds = [[0, 5, 15, 100], [0], [0, 5, 15, 100], [0]]
    multipliers = [[4.0, 3.5, 3.0, 2.5], [4.0], [4.0, 3.5, 3.0, 2.5], [4.0]]

    result = do_mds_buddy_check(
        buddy_reps_time["LAT"],
        buddy_reps_time["LON"],
        buddy_reps_time["DATE"],
        buddy_reps_time["SST"],
        buddy_reps_time["SST_CLIM"],
        dummy_pentad_stdev_,
        limits,
        number_of_obs_thresholds,
        multipliers,
    )

    for i, flag in enumerate(result):
        if i == 0:
            assert flag == expected
        else:
            assert flag == 0


def test_buddy_check_designed_to_fail_2(buddy_reps_spread, dummy_pentad_stdev_):
    # One observation with 8 neighbours that has a disparate anomaly, but the neighbours
    # are spread further apart
    limits = [[1, 1, 2], [2, 2, 2], [1, 1, 4], [2, 2, 4]]
    number_of_obs_thresholds = [[0, 5, 15, 100], [0], [0, 5, 15, 100], [0]]
    multipliers = [[4.0, 3.5, 3.0, 2.5], [4.0], [4.0, 3.5, 3.0, 2.5], [4.0]]

    result = do_mds_buddy_check(
        buddy_reps_spread["LAT"],
        buddy_reps_spread["LON"],
        buddy_reps_spread["DATE"],
        buddy_reps_spread["SST"],
        buddy_reps_spread["SST_CLIM"],
        dummy_pentad_stdev_,
        limits,
        number_of_obs_thresholds,
        multipliers,
    )

    for i, flag in enumerate(result):
        if i == 0:
            assert flag == 1
        else:
            assert flag == 0


def test_buddy_check_raises(reps_, dummy_pentad_stdev_):
    # Parameter lists have different numbers of members
    limits = [[1, 1, 2], [2, 2, 2], [1, 1, 4]]
    number_of_obs_thresholds = [[0, 5, 15, 100], [0], [0, 5, 15, 100], [0]]
    multipliers = [[4.0, 3.5, 3.0, 2.5], [4.0], [4.0, 3.5, 3.0, 2.5], [4.0]]

    with pytest.raises(ValueError):
        _ = do_mds_buddy_check(
            reps_["LAT"],
            reps_["LON"],
            reps_["DATE"],
            reps_["SST"],
            reps_["SST_CLIM"],
            dummy_pentad_stdev_,
            limits,
            number_of_obs_thresholds,
            multipliers,
        )

    # number of obs thresholds and multipliers are differently structured
    limits = [[1, 1, 2], [2, 2, 2], [1, 1, 4], [2, 2, 4]]
    number_of_obs_thresholds = [[0, 5, 15, 100], [0], [0, 5, 15, 100], [0]]
    multipliers = [[4.0, 3.5, 3.0, 2.5], [4.0], [4.0, 3.5, 2.5], [4.0]]

    with pytest.raises(ValueError):
        _ = do_mds_buddy_check(
            reps_["LAT"],
            reps_["LON"],
            reps_["DATE"],
            reps_["SST"],
            reps_["SST_CLIM"],
            dummy_pentad_stdev_,
            limits,
            number_of_obs_thresholds,
            multipliers,
        )


def test_bayesian_buddy_check(reps_, dummy_pentad_stdev_):
    result = do_bayesian_buddy_check(
        reps_["LAT"],
        reps_["LON"],
        reps_["DATE"],
        reps_["SST"],
        reps_["SST_CLIM"],
        dummy_pentad_stdev_,
        dummy_pentad_stdev_,
        dummy_pentad_stdev_,
        0.05,
        0.1,
        1.0,
        [2, 2, 4],
        3.0,
        8.0,
        0.3,
    )

    assert np.all(result == [passed, passed, passed, passed])


def test_bayesian_buddy_check_again(buddy_reps, dummy_pentad_stdev_):
    result = do_bayesian_buddy_check(
        buddy_reps["LAT"],
        buddy_reps["LON"],
        buddy_reps["DATE"],
        buddy_reps["SST"],
        buddy_reps["SST_CLIM"],
        dummy_pentad_stdev_,
        dummy_pentad_stdev_,
        dummy_pentad_stdev_,
        0.05,
        0.1,
        1.0,
        [2, 2, 4],
        3.0,
        8.0,
        0.1,
    )

    assert np.all(
        result
        == [failed, passed, passed, passed, passed, passed, passed, passed, passed]
    )


@pytest.mark.parametrize(
    ["second_date", "expected"],
    [
        ["2003-12-11T00:00:00.000000000", failed],
        ["2003-12-21T00:00:00.000000000", failed],
        ["2003-11-21T00:00:00.000000000", failed],
        ["2003-12-25T00:00:00.000000000", untestable],
    ],
)
def test_bayesian_buddy_check_designed_to_fail_time(
    buddy_reps_time, dummy_pentad_stdev_, expected
):
    result = do_bayesian_buddy_check(
        buddy_reps_time["LAT"],
        buddy_reps_time["LON"],
        buddy_reps_time["DATE"],
        buddy_reps_time["SST"],
        buddy_reps_time["SST_CLIM"],
        dummy_pentad_stdev_,
        dummy_pentad_stdev_,
        dummy_pentad_stdev_,
        0.05,
        0.1,
        1.0,
        [2, 2, 4],
        3.0,
        8.0,
        0.1,
    )

    for i, flag in enumerate(result):
        if i == 0:
            assert flag == expected
        else:
            assert flag == 0
