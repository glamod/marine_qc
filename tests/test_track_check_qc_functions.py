from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marine_qc import (
    do_few_check,
    do_iquam_track_check,
    do_spike_check,
    do_track_check,
    find_multiple_rounded_values,
    find_repeated_values,
    find_saturated_runs,
)
from marine_qc.auxiliary import failed, passed
from marine_qc.track_check_utils import (
    calculate_course_parameters,
    calculate_speed_course_distance_time_difference,
    forward_discrepancy,
    backward_discrepancy,
    calculate_midpoint,
)
from marine_qc.spherical_geometry import (
    sphere_distance,
    intermediate_point,
    course_between_points,
    lat_lon_from_course_and_distance,
)
from marine_qc.time_control import time_difference


def generic_frame(in_pt):
    pt = [in_pt for _ in range(30)]
    lat = [-5.0 + i * 0.1 for i in range(30)]
    lon = [0.0 for _ in range(30)]
    sst = [22.0 for _ in range(30)]
    sst[15] = 33

    vsi = [11.11951 for _ in range(30)]  # km/h
    dsi = [0.0 for _ in range(30)]
    dck = [193 for _ in range(30)]

    date = pd.date_range(start="1850-01-01", freq="1h", periods=len(pt))

    df = pd.DataFrame(
        {
            "sst": sst,
            "date": date,
            "lat": lat,
            "lon": lon,
            "pt": pt,
            "vsi": vsi,
            "dsi": dsi,
            "dck": dck,
        }
    )

    id = ["GOODTHING" for _ in range(30)]

    df["id"] = id

    return df


@pytest.fixture
def ship_frame():
    frame = generic_frame(1)
    frame.attrs["delta_t"] = 2.0
    return frame


@pytest.fixture
def buoy_frame():
    frame = generic_frame(6)
    frame.attrs["delta_t"] = 1.0
    return frame


def test_do_spike_check(ship_frame, buoy_frame):
    for frame in [ship_frame, buoy_frame]:
        result = do_spike_check(
            value=frame.sst,
            lat=frame.lat,
            lon=frame.lon,
            date=frame.date,
            max_gradient_space=0.5,
            max_gradient_time=1.0,
            delta_t=frame.attrs["delta_t"],
            n_neighbours=5,
        )
        for i in range(30):
            row = result[i]
            if i == 15:
                assert row == failed
            else:
                assert row == passed


def test_do_spike_check_missing_ob(ship_frame):
    ship_frame.loc[[0], "sst"] = np.nan
    result = do_spike_check(
        value=ship_frame.sst,
        lat=ship_frame.lat,
        lon=ship_frame.lon,
        date=ship_frame.date,
        max_gradient_space=0.5,
        max_gradient_time=1.0,
        delta_t=ship_frame.attrs["delta_t"],
        n_neighbours=5,
    )
    for i in range(30):
        row = result[i]
        if i == 15:
            assert row == failed
        else:
            assert row == passed


@pytest.mark.parametrize("key", ["sst", "lat", "lon", "date"])
def test_do_spike_check_raises(ship_frame, key):
    series = ship_frame[key]
    series.loc[len(series)] = 1
    kwargs = {}
    for k in ["sst", "lat", "lon", "date"]:
        if k == "sst":
            k_ = "value"
        else:
            k_ = k
        if k == key:
            kwargs[k_] = series
        else:
            kwargs[k_] = ship_frame[k]
    with pytest.raises(ValueError):
        do_spike_check(
            max_gradient_space=0.5,
            max_gradient_time=1.0,
            delta_t=2.0,
            n_neighbours=5,
            **kwargs,
        )


def test_calculate_course_parameters(ship_frame):
    earlier = ship_frame.iloc[0]
    later = ship_frame.iloc[1]

    speed, distance, course, timediff = calculate_course_parameters(
        lat_later=later.lat,
        lat_earlier=earlier.lat,
        lon_later=later.lon,
        lon_earlier=earlier.lon,
        date_later=later.date,
        date_earlier=earlier.date,
    )

    assert pytest.approx(speed, 0.00001) == 11.119508064776555
    assert pytest.approx(distance, 0.00001) == 11.119508064776555
    assert course == 0.0
    assert pytest.approx(timediff, 0.0000001) == 1.0


def test_do_track_check_very_few_obs(ship_frame):
    ship_frame = ship_frame.loc[[0, 1]]
    trk = do_track_check(
        lat=ship_frame.lat,
        lon=ship_frame.lon,
        date=ship_frame.date,
        vsi=ship_frame.vsi,
        dsi=ship_frame.dsi,
        max_direction_change=60.0,
        max_speed_change=10.0,
        max_absolute_speed=40.0,
        max_midpoint_discrepancy=150.0,
    )
    for i in range(len(trk)):
        assert trk[i] == passed


def test_do_track_check_no_obs(ship_frame):
    ship_frame = ship_frame.loc[[]]
    trk = do_track_check(
        lat=[],
        lon=[],
        date=[],
        vsi=[],
        dsi=[],
        max_direction_change=60.0,
        max_speed_change=10.0,
        max_absolute_speed=40.0,
        max_midpoint_discrepancy=150.0,
    )
    assert len(trk) == 0


def test_do_track_check_passed(ship_frame):
    trk = do_track_check(
        lat=ship_frame.lat,
        lon=ship_frame.lon,
        date=ship_frame.date,
        vsi=ship_frame.vsi,
        dsi=ship_frame.dsi,
        max_direction_change=60.0,
        max_speed_change=10.0,
        max_absolute_speed=40.0,
        max_midpoint_discrepancy=150.0,
    )
    for i in range(len(trk)):
        assert trk[i] == passed


def test_do_track_check_mixed(ship_frame):
    lon = ship_frame.lon.array
    lon[15] = 30.0
    ship_frame["lon"] = lon
    trk = do_track_check(
        lat=ship_frame.lat,
        lon=ship_frame.lon,
        date=ship_frame.date,
        vsi=ship_frame.vsi,
        dsi=ship_frame.dsi,
        max_direction_change=60.0,
        max_speed_change=10.0,
        max_absolute_speed=40.0,
        max_midpoint_discrepancy=150.0,
    )
    for i in range(len(trk)):
        if i == 15:
            assert trk[i] == failed
        else:
            assert trk[i] == passed


def test_do_track_check_testdata():
    vsi = [np.nan] * 10
    dsi = [np.nan] * 10
    lat = [46.53, 46.31, 46.09, 45.87, 45.88, 46.53, 46.31, 46.09, 45.87, 45.88]
    lon = [
        -13.17,
        -12.99,
        -12.81,
        -12.62,
        -12.57,
        -13.17,
        -12.99,
        -12.81,
        -12.62,
        -12.57,
    ]
    date = np.array(
        [
            "1873-01-01T01:00:00.000000000",
            "1873-01-01T05:00:00.000000000",
            "1873-01-01T09:00:00.000000000",
            "1873-01-01T13:00:00.000000000",
            "1873-01-01T17:00:00.000000000",
            "1875-01-01T01:00:00.000000000",
            "1875-01-01T05:00:00.000000000",
            "1875-01-01T09:00:00.000000000",
            "1875-01-01T13:00:00.000000000",
            "1875-01-01T17:00:00.000000000",
        ]
    )
    date = pd.to_datetime(date).tolist()

    results = do_track_check(
        vsi=vsi,
        dsi=dsi,
        lat=lat,
        lon=lon,
        date=date,
        max_direction_change=60.0,
        max_speed_change=10.0,
        max_absolute_speed=40.0,
        max_midpoint_discrepancy=150.0,
    )
    expected = [
        passed,
        passed,
        passed,
        passed,
        passed,
        passed,
        passed,
        passed,
        passed,
        passed,
    ]
    np.testing.assert_array_equal(results, expected)


def test_do_track_check_array_very_few_obs(ship_frame):
    ship_frame = ship_frame.loc[[0, 1]]
    trk = do_track_check(
        lat=ship_frame.lat,
        lon=ship_frame.lon,
        date=ship_frame.date,
        vsi=ship_frame.vsi,
        dsi=ship_frame.dsi,
        max_direction_change=60.0,
        max_speed_change=10.0,
        max_absolute_speed=40.0,
        max_midpoint_discrepancy=150.0,
    )
    for i in range(len(trk)):
        assert trk[i] == passed


def test_do_track_check_array_no_obs(ship_frame):
    ship_frame = ship_frame.loc[[]]
    trk = do_track_check(
        lat=[],
        lon=[],
        date=[],
        vsi=[],
        dsi=[],
        max_direction_change=60.0,
        max_speed_change=10.0,
        max_absolute_speed=40.0,
        max_midpoint_discrepancy=150.0,
    )
    assert len(trk) == 0


def test_do_track_check_array_passed(ship_frame):
    trk = do_track_check(
        lat=ship_frame.lat,
        lon=ship_frame.lon,
        date=ship_frame.date,
        vsi=ship_frame.vsi,
        dsi=ship_frame.dsi,
        max_direction_change=60.0,
        max_speed_change=10.0,
        max_absolute_speed=40.0,
        max_midpoint_discrepancy=150.0,
    )
    for i in range(len(trk)):
        assert trk[i] == passed


def test_do_track_check_array_mixed(ship_frame):
    lon = ship_frame.lon.array
    lon[15] = 30.0
    ship_frame["lon"] = lon
    trk = do_track_check(
        lat=ship_frame.lat,
        lon=ship_frame.lon,
        date=ship_frame.date,
        vsi=ship_frame.vsi,
        dsi=ship_frame.dsi,
        max_direction_change=60.0,
        max_speed_change=10.0,
        max_absolute_speed=40.0,
        max_midpoint_discrepancy=150.0,
    )
    for i in range(len(trk)):
        if i == 15:
            assert trk[i] == failed
        else:
            assert trk[i] == passed


def test_do_track_check_array_testdata():
    vsi = [np.nan] * 10
    dsi = [np.nan] * 10
    lat = [46.53, 46.31, 46.09, 45.87, 45.88, 46.53, 46.31, 46.09, 45.87, 45.88]
    lon = [
        -13.17,
        -12.99,
        -12.81,
        -12.62,
        -12.57,
        -13.17,
        -12.99,
        -12.81,
        -12.62,
        -12.57,
    ]
    date = np.array(
        [
            "1873-01-01T01:00:00.000000000",
            "1873-01-01T05:00:00.000000000",
            "1873-01-01T09:00:00.000000000",
            "1873-01-01T13:00:00.000000000",
            "1873-01-01T17:00:00.000000000",
            "1875-01-01T01:00:00.000000000",
            "1875-01-01T05:00:00.000000000",
            "1875-01-01T09:00:00.000000000",
            "1875-01-01T13:00:00.000000000",
            "1875-01-01T17:00:00.000000000",
        ]
    )
    date = pd.to_datetime(date).tolist()

    results = do_track_check(
        vsi=vsi,
        dsi=dsi,
        lat=lat,
        lon=lon,
        date=date,
        max_direction_change=60.0,
        max_speed_change=10.0,
        max_absolute_speed=40.0,
        max_midpoint_discrepancy=150.0,
    )
    expected = [
        passed,
        passed,
        passed,
        passed,
        passed,
        passed,
        passed,
        passed,
        passed,
        passed,
    ]
    np.testing.assert_array_equal(results, expected)


def test_backward_discrepancy_array(ship_frame):
    result = backward_discrepancy(
        vsi=ship_frame["vsi"],
        dsi=ship_frame["dsi"],
        lat=ship_frame["lat"],
        lon=ship_frame["lon"],
        date=ship_frame["date"],
    )
    for i in range(len(result) - 1):
        assert pytest.approx(result[i], abs=0.0001) == 0.0
    assert np.isnan(result.values[-1])


def test_forward_discrepancy_array(ship_frame):
    result = forward_discrepancy(
        vsi=ship_frame["vsi"],
        dsi=ship_frame["dsi"],
        lat=ship_frame["lat"],
        lon=ship_frame["lon"],
        date=ship_frame["date"],
    )
    for i in range(1, len(result)):
        assert pytest.approx(result[i], abs=0.00001) == 0.0
    assert np.isnan(result[0])


def test_calc_alternate_speeds(ship_frame):
    speed, distance, course, timediff = calculate_speed_course_distance_time_difference(
        ship_frame.lat, ship_frame.lon, ship_frame.date, alternating=True
    )
    # for column in ['alt_speed', 'alt_course', 'alt_distance', 'alt_time_diff']:
    #     assert column in result

    for i in range(1, len(speed) - 1):
        # Reports are spaced by 1 hour and each hour the ship goes 0.1 degrees of latitude which is 11.11951 km
        # So with alternating reports, the speed is 11.11951 km/hour, the course is due north (0/360) the distance
        # between alternate reports is twice the hourly distance 22.23902 and the time difference is 2 hours
        assert pytest.approx(speed[i], abs=0.0001) == 11.11951
        assert (
            pytest.approx(course[i], abs=0.0001) == 0.0
            or pytest.approx(course[i], abs=0.0001) == 360.0
        )
        assert pytest.approx(distance[i], abs=0.0001) == 22.23902
        assert pytest.approx(timediff[i], abs=0.0001) == 2.0


def test_calc_alternate_speeds_array(ship_frame):
    speed, distance, course, timediff = calculate_speed_course_distance_time_difference(
        ship_frame.lat, ship_frame.lon, ship_frame.date, alternating=True
    )

    for i in range(1, len(speed) - 1):
        # Reports are spaced by 1 hour and each hour the ship goes 0.1 degrees of latitude which is 11.11951 km
        # So with alternating reports, the speed is 11.11951 km/hour, the course is due north (0/360) the distance
        # between alternate reports is twice the hourly distance 22.23902 and the time difference is 2 hours
        assert pytest.approx(speed[i], abs=0.0001) == 11.11951
        assert (
            pytest.approx(course[i], abs=0.0001) == 0.0
            or pytest.approx(course[i], abs=0.0001) == 360.0
        )
        assert pytest.approx(distance[i], abs=0.0001) == 22.23902
        assert pytest.approx(timediff[i], abs=0.0001) == 2.0


@pytest.mark.parametrize("key", ["lat", "lon", "date", "vsi", "dsi"])
def test_do_track_check_raises(ship_frame, key):
    series = ship_frame[key]
    series.loc[len(series)] = 1
    kwargs = {}
    for k in ["lat", "lon", "date", "vsi", "dsi"]:
        if k == key:
            kwargs[k] = series
        else:
            kwargs[k] = ship_frame[k]
    with pytest.raises(ValueError):
        do_track_check(
            max_direction_change=60.0,
            max_speed_change=10.0,
            max_absolute_speed=40.0,
            max_midpoint_discrepancy=150.0,
            **kwargs,
        )


def test_do_few_check_no_obs():
    few = do_few_check(value=np.array([]))
    assert len(few) == 0


def test_do_few_check_passed(ship_frame):
    few = do_few_check(
        value=ship_frame["lat"],
    )
    for i in range(len(few)):
        assert few[i] == passed


def test_do_few_check_failed(ship_frame):
    few = do_few_check(
        value=ship_frame["lat"][:2],
    )
    for i in range(len(few)):
        assert few[i] == failed


def test_calculate_speed_course_distance_time_difference(ship_frame):
    speed, distance, course, timediff = calculate_speed_course_distance_time_difference(
        lat=ship_frame.lat,
        lon=ship_frame.lon,
        date=ship_frame.date,
    )
    numobs = len(speed)
    for i in range(numobs):
        if i > 0:
            assert pytest.approx(speed[i], 0.00001) == 11.119508064776555
            assert pytest.approx(distance[i], 0.00001) == 11.119508064776555
            assert (
                pytest.approx(course[i], 0.00001) == 0
                or pytest.approx(course[i], 0.00001) == 360.0
            )
            assert pytest.approx(timediff[i], 0.0000001) == 1.0
        else:
            assert np.isnan(speed[i])


def test_calculate_speed_course_distance_time_difference_array(ship_frame):
    speed, distance, course, timediff = calculate_speed_course_distance_time_difference(
        lat=ship_frame.lat,
        lon=ship_frame.lon,
        date=ship_frame.date,
    )
    numobs = len(speed)
    for i in range(numobs):
        if i > 0:
            assert pytest.approx(speed[i], 0.00001) == 11.119508064776555
            assert pytest.approx(distance[i], 0.00001) == 11.119508064776555
            assert (
                pytest.approx(course[i], 0.00001) == 0
                or pytest.approx(course[i], 0.00001) == 360.0
            )
            assert pytest.approx(timediff[i], 0.0000001) == 1.0
        else:
            assert np.isnan(speed[i])


def test_calculate_speed_course_distance_time_difference_one_ob(ship_frame):
    ship_frame = ship_frame.loc[[0]]
    speed, distance, course, timediff = calculate_speed_course_distance_time_difference(
        lat=ship_frame.lat,
        lon=ship_frame.lon,
        date=ship_frame.date,
    )

    assert len(speed) == 1
    assert len(distance) == 1
    assert len(course) == 1
    assert len(timediff) == 1

    assert np.isnan(speed[0])
    assert np.isnan(distance[0])
    assert np.isnan(course[0])
    assert np.isnan(timediff[0])


@pytest.fixture
def long_frame():
    lat = [-5.0 + i * 0.1 for i in range(30)]
    lon = [0 for _ in range(30)]
    at = [15.0 for i in range(30)]
    dpt = [15.0 for i in range(30)]
    id = ["GOODTHING" for _ in range(30)]
    date = pd.date_range(start="1850-01-01", freq="1h", periods=len(lat))
    df = pd.DataFrame(
        {"date": date, "lat": lat, "lon": lon, "at": at, "dpt": dpt, "id": id}
    )
    return df


@pytest.fixture
def longer_frame():
    lat = [-5.0 + i * 0.1 for i in range(50)]
    lon = [0 for _ in range(50)]
    at = [15.0 for i in range(50)]
    dpt = [15.0 for i in range(50)]
    id = ["GOODTHING" for _ in range(50)]
    date = pd.date_range(start="1850-01-01", freq="1h", periods=len(lat))
    df = pd.DataFrame(
        {"date": date, "lat": lat, "lon": lon, "at": at, "dpt": dpt, "id": id}
    )
    return df


@pytest.fixture
def longer_frame_last_passes():
    lat = [-5.0 + i * 0.1 for i in range(50)]
    lon = [0 for _ in range(50)]
    at = [15.0 for i in range(50)]
    dpt = [15.0 for i in range(50)]
    dpt[49] = 10.0
    id = ["GOODTHING" for _ in range(50)]
    date = pd.date_range(start="1850-01-01", freq="1h", periods=len(lat))
    df = pd.DataFrame(
        {"date": date, "lat": lat, "lon": lon, "at": at, "dpt": dpt, "id": id}
    )
    return df


@pytest.fixture
def longer_frame_broken_run():
    lat = [-5.0 + i * 0.1 for i in range(50)]
    lon = [0 for _ in range(50)]
    at = [15.0 for i in range(50)]
    dpt = [15.0 for i in range(50)]
    dpt[25] = 10.0
    id = ["GOODTHING" for _ in range(50)]
    date = pd.date_range(start="1850-01-01", freq="1h", periods=len(lat))
    df = pd.DataFrame(
        {"date": date, "lat": lat, "lon": lon, "at": at, "dpt": dpt, "id": id}
    )
    return df


@pytest.fixture
def longer_frame_early_broken_run():
    lat = [-5.0 + i * 0.1 for i in range(50)]
    lon = [0 for _ in range(50)]
    at = [15.0 for i in range(50)]
    dpt = [15.0 for i in range(50)]
    dpt[3] = 10.0
    id = ["GOODTHING" for _ in range(50)]
    date = pd.date_range(start="1850-01-01", freq="1h", periods=len(lat))
    df = pd.DataFrame(
        {"date": date, "lat": lat, "lon": lon, "at": at, "dpt": dpt, "id": id}
    )
    return df


def test_find_saturated_runs_long_frame(long_frame):
    repsat = find_saturated_runs(
        lat=long_frame["lat"],
        lon=long_frame["lon"],
        at=long_frame["at"],
        dpt=long_frame["dpt"],
        date=long_frame["date"],
        min_time_threshold=48.0,
        shortest_run=4,
    )
    for i in range(len(repsat)):
        assert repsat[i] == passed


def test_find_saturated_runs_longer_frame(longer_frame):
    repsat = find_saturated_runs(
        lat=longer_frame["lat"],
        lon=longer_frame["lon"],
        at=longer_frame["at"],
        dpt=longer_frame["dpt"],
        date=longer_frame["date"],
        min_time_threshold=48.0,
        shortest_run=4,
    )
    for i in range(len(repsat)):
        assert repsat[i] == failed


def test_find_saturated_runs_longer_frame_last_passes(longer_frame_last_passes):
    repsat = find_saturated_runs(
        lat=longer_frame_last_passes["lat"],
        lon=longer_frame_last_passes["lon"],
        at=longer_frame_last_passes["at"],
        dpt=longer_frame_last_passes["dpt"],
        date=longer_frame_last_passes["date"],
        min_time_threshold=48.0,
        shortest_run=4,
    )
    for i in range(len(repsat) - 1):
        assert repsat[i] == failed
    assert repsat[49] == passed


def test_find_saturated_runs_longer_frame_broken_run(longer_frame_broken_run):
    repsat = find_saturated_runs(
        lat=longer_frame_broken_run["lat"],
        lon=longer_frame_broken_run["lon"],
        at=longer_frame_broken_run["at"],
        dpt=longer_frame_broken_run["dpt"],
        date=longer_frame_broken_run["date"],
        min_time_threshold=48.0,
        shortest_run=4,
    )
    for i in range(len(repsat)):
        assert repsat[i] == passed


def test_find_saturated_runs_longer_frame_early_broken_run(
    longer_frame_early_broken_run,
):
    repsat = find_saturated_runs(
        lat=longer_frame_early_broken_run["lat"],
        lon=longer_frame_early_broken_run["lon"],
        at=longer_frame_early_broken_run["at"],
        dpt=longer_frame_early_broken_run["dpt"],
        date=longer_frame_early_broken_run["date"],
        min_time_threshold=48.0,
        shortest_run=4,
    )
    for i in range(len(repsat)):
        assert repsat[i] == passed


@pytest.fixture
def unrounded_data():
    lat = [-5.0 + i * 0.1 for i in range(50)]
    lon = [0 for _ in range(50)]
    at = [15.0 + (i * 0.2) for i in range(50)]
    id = ["GOODTHING" for _ in range(50)]
    date = pd.date_range(start="1850-01-01", freq="1h", periods=len(lat))
    df = pd.DataFrame({"date": date, "lat": lat, "lon": lon, "at": at, "id": id})
    return df


@pytest.fixture
def rounded_data():
    lat = [-5.0 + i * 0.1 for i in range(50)]
    lon = [0 for _ in range(50)]
    at = [round(15.0 + (i * 0.2)) for i in range(50)]
    id = ["GOODTHING" for _ in range(50)]
    date = pd.date_range(start="1850-01-01", freq="1h", periods=len(lat))
    df = pd.DataFrame({"date": date, "lat": lat, "lon": lon, "at": at, "id": id})
    return df


def test_find_multiple_rounded_values(rounded_data, unrounded_data):
    rounded = find_multiple_rounded_values(unrounded_data["at"], 20, 0.5)
    for i in range(len(rounded)):
        assert rounded[i] == passed

    rounded = find_multiple_rounded_values(rounded_data["at"], 20, 0.5)
    for i in range(len(rounded)):
        assert rounded[i] == failed


def test_find_multiple_rounded_values_no_obs():
    rounded = find_multiple_rounded_values(np.array([]), 20, 0.8)
    assert len(rounded) == 0


def test_find_multiple_rounded_values_too_few_obs():
    # All values are rounded in this example, but while there are fewer than the min_count (20) everything will pass
    for i in range(1, 50):
        values = np.arange(i)
        rounded = find_multiple_rounded_values(values, 20, 0.5)
        if i <= 20:
            assert np.all(rounded == passed)
        else:
            assert np.all(rounded == failed)


def test_find_multiple_rounded_values_raises(rounded_data):
    with pytest.raises(ValueError):
        find_multiple_rounded_values(rounded_data["at"], 20, 1.5)
    with pytest.raises(ValueError):
        find_multiple_rounded_values(rounded_data["at"], 20, -1.5)


@pytest.fixture
def repeated_data():
    lat = [-5.0 + i * 0.1 for i in range(50)]
    lon = [0 for _ in range(50)]
    at = [22.3 for i in range(50)]
    at[49] = 22.5
    id = ["GOODTHING" for _ in range(50)]
    date = pd.date_range(start="1850-01-01", freq="1h", periods=len(lat))
    df = pd.DataFrame({"date": date, "lat": lat, "lon": lon, "at": at, "id": id})
    return df


@pytest.fixture
def almost_repeated_data():
    lat = [-5.0 + i * 0.1 for i in range(50)]
    lon = [0 for _ in range(50)]
    at = [22.3 for i in range(20)]
    for i in range(20, 50):
        at.append(22.5 + (i - 20) * 0.3)
    id = ["GOODTHING" for _ in range(50)]
    date = pd.date_range(start="1850-01-01", freq="1h", periods=len(lat))
    df = pd.DataFrame({"date": date, "lat": lat, "lon": lon, "at": at, "id": id})
    return df


@pytest.fixture
def almost_repeated_data_with_nan():
    lat = [-5.0 + i * 0.1 for i in range(50)]
    lon = [0 for _ in range(50)]
    at = [22.3 for i in range(19)]
    at.append(np.nan)
    for i in range(20, 50):
        at.append(22.5 + (i - 20) * 0.3)
    id = ["GOODTHING" for _ in range(50)]
    date = pd.date_range(start="1850-01-01", freq="1h", periods=len(lat))
    df = pd.DataFrame({"date": date, "lat": lat, "lon": lon, "at": at, "id": id})
    return df


def test_find_repeated_values(
    repeated_data, almost_repeated_data, almost_repeated_data_with_nan
):
    repeated = find_repeated_values(repeated_data["at"], 20, 0.7)
    for i in range(len(repeated) - 1):
        assert repeated[i] == failed
    assert repeated[49] == passed

    repeated = find_repeated_values(almost_repeated_data["at"], 20, 0.7)
    for i in range(len(repeated)):
        assert repeated[i] == passed

    # np.nan counts as a pass
    repeated = find_repeated_values(almost_repeated_data_with_nan["at"], 20, 0.7)
    for i in range(len(repeated)):
        assert repeated[i] == passed


def test_find_repeated_values_raises(repeated_data):
    with pytest.raises(ValueError):
        find_repeated_values(repeated_data["at"], 20, 1.1)
    with pytest.raises(ValueError):
        find_repeated_values(repeated_data["at"], 20, -0.1)


def test_find_repeated_values_no_obs():
    result = find_repeated_values(np.array([]), 20, 0.7)
    assert len(result) == 0


def test_find_repeated_values_too_few_obs():
    # All values are rounded in this example, but while there are fewer than the min_count (20) everything will pass
    for i in range(1, 50):
        values = np.array([1.0] * i)
        rounded = find_repeated_values(values, 20, 0.5)
        if i <= 20:
            assert np.all(rounded == passed)
        else:
            assert np.all(rounded == failed)


def iquam_frame(in_pt):
    pt = [in_pt for _ in range(30)]
    lat = [-5.0 + i * 0.1 for i in range(30)]
    lon = [0 for _ in range(30)]
    id = ["GOODTHING" for _ in range(30)]
    date = pd.date_range(start="1850-01-01", freq="1h", periods=len(pt))

    df = pd.DataFrame(
        {
            "date": date,
            "lat": lat,
            "lon": lon,
            "pt": pt,
            "id": id,
        }
    )

    return df


@pytest.fixture
def iquam_drifter():
    return iquam_frame(6)


@pytest.fixture
def iquam_ship():
    return iquam_frame(1)


def test_do_iquam_track_check_no_obs():
    iquam_track_check = do_iquam_track_check(
        lat=np.array([]),
        lon=np.array([]),
        date=np.array([]),
        speed_limit=15.0,
        delta_d=1.11,
        delta_t=0.01,
        n_neighbours=5,
    )
    assert len(iquam_track_check) == 0


def test_do_iquam_track_check_drifter(iquam_drifter):
    iquam_track = do_iquam_track_check(
        lat=iquam_drifter.lat,
        lon=iquam_drifter.lon,
        date=iquam_drifter.date,
        speed_limit=15.0,
        delta_d=1.11,
        delta_t=0.01,
        n_neighbours=5,
    )
    for i in range(len(iquam_track)):
        assert iquam_track[i] == passed


def test_do_iquam_track_check_ship(iquam_ship):
    iquam_track = do_iquam_track_check(
        lat=iquam_ship.lat,
        lon=iquam_ship.lon,
        date=iquam_ship.date,
        speed_limit=15.0,
        delta_d=1.11,
        delta_t=0.01,
        n_neighbours=5,
    )
    for i in range(len(iquam_track)):
        assert iquam_track[i] == passed


def test_do_iquam_track_check_ship_lon(iquam_ship):
    lon = iquam_ship.lon.array
    lon[15] = 30.0
    iquam_ship["lon"] = lon
    iquam_track = do_iquam_track_check(
        lat=iquam_ship.lat,
        lon=iquam_ship.lon,
        date=iquam_ship.date,
        speed_limit=15.0,
        delta_d=1.11,
        delta_t=0.01,
        n_neighbours=5,
    )
    for i in range(len(iquam_track)):
        if i == 15:
            assert iquam_track[i] == failed
        else:
            assert iquam_track[i] == passed


def test_do_iquam_track_check_drifter_speed_limit(iquam_drifter):
    iquam_track = do_iquam_track_check(
        lat=iquam_drifter.lat,
        lon=iquam_drifter.lon,
        date=iquam_drifter.date,
        speed_limit=10.8,
        delta_d=1.11,
        delta_t=0.01,
        n_neighbours=5,
    )
    for i in range(len(iquam_track)):
        if i in [4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25]:
            assert iquam_track[i] == failed
        else:
            assert iquam_track[i] == passed


@pytest.mark.parametrize(
    "lats, lons, timediffs, expected",
    [
        ([0, 1, 2], [0, 0, 0], [1, 1, 1], [np.nan, 0.0, np.nan]),
        ([0, 0, 0], [0, 0, 0], [0, 0, 0], [np.nan, 0.0, np.nan]),
        ([0, 0, 0], [0, 0, 0], [None, None, None], [np.nan, 0.0, np.nan]),
    ],
)
def test_calculate_midpoint(lats, lons, timediffs, expected):
    result = calculate_midpoint(np.array(lats), np.array(lons), np.array(timediffs))
    expected = np.array(expected)
    assert np.array_equal(result, expected, equal_nan=True)


@pytest.mark.parametrize(
    "lats, lons, timediffs, expected",
    [
        ([0, 1, 2], [0, 0, 0], [1, 1, 1], [np.nan, 0.0, np.nan]),
        ([0, 0, 0], [0, 0, 0], [0, 0, 0], [np.nan, 0.0, np.nan]),
        ([0, 0, 0], [0, 0, 0], [np.nan, np.nan, np.nan], [np.nan, 0.0, np.nan]),
    ],
)
def test_calculate_midpoint_array(lats, lons, timediffs, expected):
    result = calculate_midpoint(np.array(lats), np.array(lons), np.array(timediffs))
    expected = np.array(expected)
    assert np.array_equal(result, expected, equal_nan=True)


def test_time_differences_array():
    dates = pd.date_range(start="1850-01-01", freq="1h", periods=11)
    in1 = dates[0:10]
    in2 = dates[1:11]
    result = time_difference(in1, in2)
    assert np.all(result == 1)


def test_sphere_distance_array():
    lat1 = np.arange(-90.0, 90.0, 1.0)
    lat2 = np.arange(-89.0, 91.0, 1.0)
    lon1 = np.zeros(len(lat1))
    lon2 = np.zeros(len(lat2))

    result = sphere_distance(lat1, lon1, lat2, lon2)
    assert np.allclose(result, 6371.0088 * np.pi / 180.0)


def test_sphere_distance_array_works_with_nans():
    lat1 = np.arange(-90.0, 90.0, 1.0)
    lat2 = np.arange(-89.0, 91.0, 1.0)
    lon1 = np.zeros(len(lat1))
    lon2 = np.zeros(len(lat2))
    lat1[0] = np.nan

    result = sphere_distance(lat1, lon1, lat2, lon2)
    expected = np.zeros(len(lat1)) + 6371.0088 * np.pi / 180.0
    expected[0] = np.nan
    assert np.allclose(result, expected, equal_nan=True)


def test_intermediate_point_array():

    # 1. equator_is_halfway_from_pole_to_pole
    # 2. 5deg_is_one_72th_of_equator
    # 3. any_fraction_of_no_move_is_nothing
    lat1 = np.array([-89.0, 0.0, 10.0])
    lon1 = np.array([0.0, 0.0, 152.0])
    lat2 = np.array([89, 0.0, 10.0])
    lon2 = np.array([0.0, 90.0, 152.0])
    f = np.array([0.5, 1.0 / 18.0, 0.75])

    lat, lon = intermediate_point(lat1, lon1, lat2, lon2, f)

    assert np.allclose(lat, np.array([0.0, 0.0, 10.0]))
    assert np.allclose(lon, np.array([0.0, 5.0, 152.0]))


def test_course_between_point_array():
    lat1 = np.arange(-90.0, 90.0, 1.0)
    lat2 = np.arange(-89.0, 91.0, 1.0)
    lon1 = np.zeros(len(lat1))
    lon2 = np.zeros(len(lat2))

    # heading due north
    result = course_between_points(lat1, lon1, lat2, lon2)
    assert np.all(result == 0.0)
    # or, inverted, south
    result = course_between_points(lat2, lon2, lat1, lon1)
    assert np.all(result == 180.0)

    lon1 = np.arange(-180.0, 180.0, 1.0)
    lon2 = np.arange(-179.0, 181.0, 1.0)
    lat1 = np.zeros(len(lon1))
    lat2 = np.zeros(len(lon2))

    # heading east
    result = course_between_points(lat1, lon1, lat2, lon2)
    assert np.all(result == 90.0)
    # or, inverted, west
    result = course_between_points(lat2, lon2, lat1, lon1)
    assert np.all(result == -90.0)


def test_lat_lon_from_course_and_distance_array():

    earths_radius = 6371.0088
    npiearth = np.pi * earths_radius

    lat = np.array([0.0, -90.0, -90.0, 0.0, 0.0])
    lon = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    tc = np.array([0.0, 0.0, 4.7, 90.0, 45.0])
    d = np.array([0.0, npiearth, npiearth, 2 * npiearth, npiearth])

    expected_lat = np.array([0.0, 90.0, 90.0, 0.0, 0.0])
    expected_lon = np.array([0.0, 0.0, 90.0, 0.0, -180.0])
    result_lat, result_lon = lat_lon_from_course_and_distance(lat, lon, tc, d)

    assert np.allclose(result_lat, expected_lat)
    assert np.allclose(result_lon, expected_lon)
