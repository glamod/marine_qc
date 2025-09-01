from __future__ import annotations

import numpy as np
import pytest

import marine_qc.spherical_geometry as sg
from marine_qc.auxiliary import convert_to
from marine_qc.track_check_utils import (
    check_distance_from_estimate,
    direction_continuity,
    direction_continuity_array,
    increment_position,
    modal_speed,
    set_speed_limits,
    speed_continuity,
    speed_continuity_array,
)


def test_ship_heading_north_at_60knots_goes1degree_in_1hour():
    """A ship travelling north at 60 knots will go 1 degree in 1 hour"""
    for lat in range(-90, 90):
        alat1 = lat
        alon1 = lat
        avs = convert_to(60.0, "knots", "km/h")
        ads = 0.0
        timdif = 2.0
        alat2, alon2 = increment_position(alat1, alon1, avs, ads, timdif)
        assert pytest.approx(alon2, 0.001) == 0
        assert pytest.approx(alat2, 0.001) == 1
        alat2, alon2 = increment_position(alat1, alon1, avs, ads, None)
        assert np.isnan(alat2)
        assert np.isnan(alon2)


def test_ship_heading_east_at_60knots_goes1degree_in_1hour():
    """A ship at the equator travelling east at 60 knots will go 1 degree in 1 hour"""
    alat1 = 0.0
    alat2 = 0.0
    avs = convert_to(60.0, "knots", "km/h")
    ads = 90.0
    timdif = 2.0
    aud1, avd1 = increment_position(alat1, alat2, avs, ads, timdif)
    assert pytest.approx(avd1, 0.001) == 1
    assert pytest.approx(aud1, 0.001) == 0


def test_ship_heading_east_at_60knots_at_latitude60_goes2degrees_in_1hour():
    """A ship travelling east at 60 knots will go 2 degrees in 1 hour at 60N"""
    km_to_nm = 0.539957
    alat = 60.0
    alon = 0.0
    avs = convert_to(60.0, "knots", "km/h")
    ads = 90.0
    timdif = 2.0
    dlat, dlon = increment_position(alat, alon, avs, ads, timdif)
    distance = sg.sphere_distance(alat, alon, alat + dlat, alon + dlon) * km_to_nm
    assert pytest.approx(distance, 0.0001) == 60.0


def test_ship_goes_southwest():
    alat1 = 0.0
    alat2 = 0.0
    avs = convert_to(60.0, "knots", "km/h")
    ads = 225.0
    timdif = 2.0
    aud1, avd1 = increment_position(alat1, alat2, avs, ads, timdif)
    assert pytest.approx(avd1, 0.001) == -1.0 / np.sqrt(2)
    assert pytest.approx(aud1, 0.001) == -1.0 / np.sqrt(2)


def test_noinput():
    m = modal_speed([])
    assert np.isnan(m)


def test_one_input():
    m = modal_speed([17.0])
    assert np.isnan(m)


def test_zero_index_input():
    m = modal_speed([-17.0, -17.0])
    assert m == convert_to(8.5, "knots", "km/h")


@pytest.mark.parametrize(
    "base_speed, expected",
    [
        (20.0, 19.5),
        (2.0, 8.5),
        (200.0, 34.5),
    ],
)
def test_modesp_single_speed_input_over8point5(base_speed, expected):
    speeds = [convert_to(base_speed, "knots", "km/h") for _ in range(8)]
    m = modal_speed(speeds)
    assert m == convert_to(expected, "knots", "km/h")


def test_one_of_each_speed_input_min_under8point5():
    speeds = [convert_to(i, "knots", "km/h") for i in range(1, 20)]
    m = modal_speed(speeds)
    assert m == convert_to(8.5, "knots", "km/h")


@pytest.mark.parametrize(
    "amode, expected",
    [
        (None, (15.00, 20.00, 0.00)),
        (5.5, (15.00, 20.00, 0.00)),
        (
            9.5,
            (9.5 * 1.25, 30.00, 9.5 * 0.75),
        ),
    ],
)
def test_set_speed_limits(amode, expected):
    amode = convert_to(amode, "knots", "km/h")
    expected = convert_to(expected, "knots", "km/h")
    result = set_speed_limits(amode)
    np.allclose(np.array(result), np.array(expected))


@pytest.mark.parametrize("angle", [0, 45, 90, 135, 180, 225, 270, 315, 360])
def test_just_pass_and_just_fail(angle):
    assert 10 == direction_continuity(angle, angle, angle + 60.1)
    assert 0 == direction_continuity(angle, angle, angle + 59.9)


@pytest.mark.parametrize("angle", [0, 45, 90, 135, 180, 225, 270, 315, 360])
def test_direction_continuity_array(angle):
    dsi = np.zeros(10) + angle
    ship_directions = np.zeros(10) + angle + 60.1
    result = direction_continuity_array(dsi, ship_directions)
    assert np.all(result == 10.0)

    dsi = np.zeros(10) + angle
    ship_directions = np.zeros(10) + angle + 59.9
    result = direction_continuity_array(dsi, ship_directions)
    assert np.all(result == 0.0)


def test_direction_continuity():
    with pytest.raises(ValueError):
        direction_continuity(1, 0, 0 + 60.1)
    with pytest.raises(ValueError):
        direction_continuity(0, 1, 0 + 60.1)


@pytest.mark.parametrize(
    "vsi, vsi_previous, speeds, expected",
    [
        (12, 12, 12, 0),
        (12, 12, 12 + 10.01, 10),
        (12, 12, 12 + 9.99, 0),
        (12, 12, None, 0),
    ],
)
def test_speed_continuity(vsi, vsi_previous, speeds, expected):
    assert speed_continuity(vsi, vsi_previous, speeds) == expected


def test_speed_continuity_array():
    vsi = np.array([12, 12, 12, 12])
    speeds = np.array([12, 12 + 10.01, 12 + 9.9, np.nan])
    expected = np.array([0, 10, 0, 0])

    result = speed_continuity_array(vsi, speeds)
    assert np.all(result == expected)


@pytest.mark.parametrize(
    "vsi, vsi_previous, time_differences, fwd_diff_from_estimated, rev_diff_from_estimated, expected",
    [
        (None, 2.0, 5.0, 5.0, 5.0, 0.0),
        (2.0, None, 5.0, 5.0, 5.0, 0.0),
        (2.0, 2.0, None, 5.0, 5.0, 0.0),
        (2.0, 2.0, 5.0, None, 5.0, 0.0),
        (2.0, 2.0, 5.0, 5.0, None, 0.0),
        (2.0, 2.0, 5.0, 5.0, 5.0, 0.0),
        (2.0, 2.0, 1.0, 20.0, 20.0, 10.0),
    ],
)
def test_check_distance_from_estimate(
    vsi,
    vsi_previous,
    time_differences,
    fwd_diff_from_estimated,
    rev_diff_from_estimated,
    expected,
):
    result = check_distance_from_estimate(
        vsi,
        vsi_previous,
        time_differences,
        fwd_diff_from_estimated,
        rev_diff_from_estimated,
    )
    assert result == expected
