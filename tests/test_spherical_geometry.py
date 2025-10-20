from __future__ import annotations

import numpy as np
import pytest

import marine_qc.spherical_geometry as sg
from marine_qc.auxiliary import earths_radius


def test_same_start_and_end_have_zero_distance():
    """For a range of lats and longs test that identical start and end points map to zero distance"""
    for lat1 in range(-90, 90, 5):
        for lon1 in range(-180, 180, 5):
            a = sg.sphere_distance(lat1, lon1, lat1, lon1)
            assert a == 0.0


def test_pole_to_pole_a():
    """Make sure pole to pole distance is pi*r"""
    a = sg.sphere_distance(-90.0, 0.0, 90.0, 0.0)
    assert earths_radius / 1000.0 * np.pi == a


def test_round_equator_a():
    """Make sure half the equator is pi*r"""
    a = sg.sphere_distance(0.0, 0.0, 0.0, 180.0)
    assert earths_radius / 1000.0 * np.pi == a


def test_small_steps_latitude():
    """Test that 5degree lat increment is one 36th of pi*r to better than 1cm"""
    a = sg.sphere_distance(-90.0, 0.0, -85.0, 0.0)
    assert pytest.approx(a, 0.00000001) == earths_radius / 1000.0 * np.pi / 36


def test_small_steps_longitude():
    """Test that 5degree long increment is one 36th of pi*r to better than 1cm"""
    a = sg.sphere_distance(0.0, 22.0, 0.0, 27.0)
    assert pytest.approx(a, 0.00000001) == earths_radius / 1000.0 * np.pi / 36


def test_same_start_and_end_have_zero_angular_distance():
    """For a range of lats and longs test that identical start and end points map to zero distance"""
    for lat1 in range(-90, 90, 5):
        for lon1 in range(-180, 180, 5):
            a = sg.angular_distance(lat1, lon1, lat1, lon1)
            assert a == 0.0


def test_pole_to_pole():
    """Make sure pole to pole distance is pi"""
    a = sg.angular_distance(-90.0, 0.0, 90.0, 0.0)
    assert pytest.approx(a, 0.00000001) == np.pi


def test_round_equator():
    """Make sure half the equator is pi"""
    a = sg.angular_distance(0.0, 0.0, 0.0, 180.0)
    assert pytest.approx(a, 0.00000001) == np.pi


def test_round_the_world_is_zero():
    """Test all the way round the equator 0 - 360 gives zero angular distance"""
    a = sg.angular_distance(0.0, 0.0, 0.0, 360.0)
    assert pytest.approx(a, 0.00000001) == 0.0


@pytest.mark.parametrize(
    "lat1, lon1, lat2, lon2, expected",
    [
        (None, 0.0, 0.0, 0.0, ValueError),
        (0.0, None, 0.0, 0.0, ValueError),
        (0.0, 0.0, None, 0.0, ValueError),
        (0.0, 0.0, 0.0, None, ValueError),
    ],
)
def test_angular_distance_nan(lat1, lon1, lat2, lon2, expected):
    assert np.isnan(sg.angular_distance(lat1, lon1, lat2, lon2))


def test_going_nowhere():
    """If ship heads north from 0,0 and travels 0 distance, should end up at same place"""
    lat, lon = sg.lat_lon_from_course_and_distance(0.0, 0.0, 0.0, 0.0)
    assert lat == 0.0
    assert lon == 0.0


def test_heading_north_from_pole_to_pole():
    """Heading north from the southpole for an angular distance of pi takes you to the north pole"""
    lat, lon = sg.lat_lon_from_course_and_distance(-90.0, 0.0, 0.0, np.pi * earths_radius / 1000.0)
    assert lat == 90.0
    assert lon == 0.0


def test_heading_north_from_pole_to_pole_on_different_headings():
    """Heading north from the southpole for an angular distance of pi takes you to the north pole"""
    for i in range(0, 100):
        lat, _lon = sg.lat_lon_from_course_and_distance(-90.0, 0.0, i * 360.0 / 100.0, np.pi * earths_radius / 1000.0)
        assert lat == 90.0


def test_heading_east_round_equator():
    lat, lon = sg.lat_lon_from_course_and_distance(0.0, 0.0, 90.0, 2 * np.pi * earths_radius / 1000.0)
    assert pytest.approx(lat, 0.00000001) == 0.0
    assert pytest.approx(lon, 0.00000001) == 0.0


def test_heading_eastish_round_equator():
    lat, lon = sg.lat_lon_from_course_and_distance(0.0, 0.0, 45.0, np.pi * earths_radius / 1000.0)
    assert pytest.approx(lat, 0.00000001) == 0.0
    assert lon in [-180, 180]


def test_heading_just_east_of_north():
    """Just east of north heading should be zero"""
    heading = sg.course_between_points(0.0, 0.0, 1.3, 0.0000001)
    assert pytest.approx(heading, 1) == 0.0


def test_heading_cos_lat_near_zero():
    """Just east of north heading should be zero"""
    heading = sg.course_between_points(90.0, 0.0, 1.3, 0.0000001)
    assert pytest.approx(heading, 1) == 0.0

    heading = sg.course_between_points(-90.0, 0.0, 1.3, 0.0000001)
    assert pytest.approx(heading, 1) == 0.0


def test_heading_just_west_of_north():
    """Just west of north heading should be 360.0"""
    heading = sg.course_between_points(0.0, 0.0, 1.3, -0.0000001)
    assert pytest.approx(heading, abs=0.0001) == 360.0 or pytest.approx(heading, abs=0.0001) == 0.0


def test_heading_east():
    """Heading due east should be course of 90degrees"""
    heading = sg.course_between_points(0.0, 0.0, 0.0, 23.2)
    assert pytest.approx(heading, 0.0000001) == 90.0


def test_heading_west():
    """Heading due west should be course of 90degrees"""
    heading = sg.course_between_points(0.0, 0.0, 0.0, -23.2)
    assert pytest.approx(heading, 0.0000001) == 270.0 or pytest.approx(heading, 0.0000001) == -90.0


def test_heading_south():
    """Heading due south should be course of 90degrees"""
    heading = sg.course_between_points(0.0, 0.0, -11.1111, 0.0)
    assert pytest.approx(heading, 0.0000001) == 180.0


def test_heading_south2():
    """Test added in response to bug where heading due south or due north occasionally returned nan"""
    heading = sg.course_between_points(4.0, 92.0, 3.0, 92.0)
    assert pytest.approx(heading, 0.000001) == 180.0


def test_equator_is_halfway_from_pole_to_pole():
    """Make sure equator is halfway between (almost) the poles"""
    # don't do 90S to 90N as the great circle is not uniquely defined
    lat, lon = sg.intermediate_point(-89, 0, 89, 0, 0.5)
    assert pytest.approx(lat, 0.0000001) == 0.0
    assert pytest.approx(lon, 0.0000001) == 0.0


def test_that_5deg_is_one_72th_of_equator():
    """Test that 5degrees is one eighteenth of 90degrees around the equator"""
    lat, lon = sg.intermediate_point(0.0, 0.0, 0.0, 90.0, 1.0 / 18.0)
    assert lat == 0.0
    assert pytest.approx(lon, 0.0000001) == 5.0


def test_any_fraction_of_no_move_is_nothing():
    lat, lon = sg.intermediate_point(10.0, 152.0, 10.0, 152.0, 0.75)
    assert lon == 152.0
    assert lat == 10.0


@pytest.mark.parametrize("f", [1.75, -0.25])
def test_intermediate_point_nan(f):
    lat, lon = sg.intermediate_point(10.0, 152.0, 10.0, 152.0, f)
    assert np.isnan(lat)
    assert np.isnan(lon)
