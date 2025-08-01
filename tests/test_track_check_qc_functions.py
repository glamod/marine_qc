from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marine_qc.auxiliary import failed, passed
from marine_qc.qc_sequential_reports import (
    backward_discrepancy,
    calculate_course_parameters,
    calculate_speed_course_distance_time_difference,
    do_few_check,
    do_iquam_track_check,
    do_spike_check,
    do_track_check,
    find_multiple_rounded_values,
    find_repeated_values,
    find_saturated_runs,
    forward_discrepancy,
)


def generic_frame(in_pt):
    pt = [in_pt for _ in range(30)]
    lat = [-5.0 + i * 0.1 for i in range(30)]
    lon = [0 for _ in range(30)]
    sst = [22 for _ in range(30)]
    sst[15] = 33

    vsi = [11.11951 for _ in range(30)]  # km/h
    dsi = [0 for _ in range(30)]
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


def test_backward_discrepancy(ship_frame):
    result = backward_discrepancy(
        vsi=ship_frame["vsi"],
        dsi=ship_frame["dsi"],
        lat=ship_frame["lat"],
        lon=ship_frame["lon"],
        date=ship_frame["date"],
    )
    for i in range(len(result) - 1):
        assert pytest.approx(result[i], abs=0.00001) == 0.0
    assert np.isnan(result[-1])


def test_forward_discrepancy(ship_frame):
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


def test_find_repeated_values(repeated_data, almost_repeated_data):
    repeated = find_repeated_values(repeated_data["at"], 20, 0.7)
    for i in range(len(repeated) - 1):
        assert repeated[i] == failed
    assert repeated[49] == passed

    repeated = find_repeated_values(almost_repeated_data["at"], 20, 0.7)
    for i in range(len(repeated)):
        assert repeated[i] == passed


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
