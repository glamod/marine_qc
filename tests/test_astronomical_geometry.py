from __future__ import annotations
import math

import numpy as np
import pytest

from marine_qc.astronomical_geometry import (
    calculate_azimuth,
    calculate_sun_parameters,
    convert_degrees,
    elliptic_angle,
    mean_earth_anomaly,
    sin_of_elevation,
    sun_ascension,
    sun_azimuth,
    sun_declination,
    sun_hour_angle,
    sun_longitude,
    sun_position,
    sunangle,
    to_local_siderial_time,
    to_siderial_time,
)


degrad = np.pi / 180.0


@pytest.mark.parametrize(
    "time, expected_deg",
    [
        (0, 0),
        (365.25, 360),
        (182.625, 180),
        (-1, -360.0 / 365.25),
        (730.5, 720),
    ],
)
def test_sun_position(time, expected_deg):
    expected_rad = expected_deg * degrad
    result = sun_position(time)

    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected_rad, rtol=1e-12)


@pytest.mark.parametrize(
    "time, theta, expected",
    [
        (0, 0, -0.031271),
        (0, np.pi / 2, np.pi / 2 - 0.031271),
        (1e6, 0, -0.031271 - 4.5396e-7 * 1e6),
        (-1e6, 0, -0.031271 - 4.5396e-7 * (-1e6)),
        (365.25, np.pi, np.pi - 0.031271 - 4.5396e-7 * 365.25),
    ],
)
def test_mean_earth_anomaly(time, theta, expected):
    result = mean_earth_anomaly(time, theta)
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "time, expected",
    [
        (0, 4.899901),
        (182.625, 8.043654),
        (365.25, 11.183215),
        (-1, 4.882111),
        (730.5, 17.466529),
    ],
)
def test_sun_longitude(time, expected):
    result = sun_longitude(time)

    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "time, expected",
    [
        (0, 0.409140),
        (182.625, 0.40913886736),
        (1000000, 0.4029251),
        (-1000000, 0.4153549),
    ],
)
def test_elliptic_angle(time, expected):
    result = elliptic_angle(time)
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=1e-8)


@pytest.mark.parametrize(
    "long_of_sun, sin_long_of_sun, angle_of_elliptic, expected",
    [
        (0, 0, 0, 0.0),
        (np.pi / 2, 1, 0, np.pi / 2),
        (np.pi, 0.5, 0, 2.677945),
        (np.pi / 4, np.sqrt(2) / 2, 0.409140, 0.74238),
    ],
)
def test_sun_ascension(long_of_sun, sin_long_of_sun, angle_of_elliptic, expected):
    result = sun_ascension(long_of_sun, sin_long_of_sun, angle_of_elliptic)
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "sin_long_of_sun, angle_of_elliptic, expected",
    [
        (0, 0.409140, 0.0),
        (1, 0.409140, 0.40914),
        (-1, 0.409140, -0.40914),
        (0.5, 0.409140, 0.200246),
    ],
)
def test_sun_declination(sin_long_of_sun, angle_of_elliptic, expected):
    result = sun_declination(sin_long_of_sun, angle_of_elliptic)
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "time, expected_ra, expected_dec",
    [
        (0, 4.916324, -0.401552),
        (182.625, 1.777071, 0.401376),
        (365.25, 4.916463, -0.401539),
        (-1, 4.897049, -0.402918),
        (730.5, 4.916602, -0.401527),
    ],
)
def test_calculate_sun_parameters(time, expected_ra, expected_dec):
    rta, dec = calculate_sun_parameters(time)
    assert isinstance(rta, float)
    assert isinstance(dec, float)
    np.testing.assert_allclose(rta, expected_ra, rtol=1e-5)
    np.testing.assert_allclose(dec, expected_dec, rtol=1e-5)


@pytest.mark.parametrize(
    "time, delyear, expected",
    [
        (0, 0, 1.759335),
        (182.625, 0, 4.900996),
        (365.25, 1, 1.75947),
        (-1, 0, 1.742132),
    ],
)
def test_to_siderial_time(time, delyear, expected):
    sid = to_siderial_time(time, delyear)
    assert isinstance(sid, float)
    assert 0 <= sid < 2 * np.pi
    np.testing.assert_allclose(sid, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "time, time_in_hours, delyear, lon, expected",
    [
        (0, 0, 0, 0, 1.759335),
        (182.625, 12, 0, 0, 1.759402),
        (365.25, 0, 1, 0, 1.75947),
        (0, 6, 0, 90, 4.900928),
    ],
)
def test_to_local_siderial_time(time, time_in_hours, delyear, lon, expected):
    lsid = to_local_siderial_time(time, time_in_hours, delyear, lon)
    assert isinstance(lsid, float)
    assert 0 <= lsid < 2 * np.pi
    np.testing.assert_allclose(lsid, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "local_siderial_time, right_ascension, expected",
    [
        (3.0, 1.0, 2.0),
        (1.0, 3.0, 4.283185),
        (2.5, 2.5, 0.0),
        (0.0, np.pi, np.pi),
    ],
)
def test_sun_hour_angle(local_siderial_time, right_ascension, expected):
    hra = sun_hour_angle(local_siderial_time, right_ascension)
    assert isinstance(hra, float)
    assert 0 <= hra < 2 * np.pi
    np.testing.assert_allclose(hra, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "phi, declination, hour_angle, expected",
    [
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, np.pi, -1.0),
        (np.pi / 2, 0.0, 0.0, 0.0),
        (0.0, 0.409140, 0.0, 0.917463),
        (0.0, 0.409140, np.pi, -0.917463),
    ],
)
def test_sin_of_elevation(phi, declination, hour_angle, expected):
    result = sin_of_elevation(phi, declination, hour_angle)
    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "phi, declination, expected",
    [
        (1.0, 0.5, 0),
        (0.5, 0.5, 180),
        (0.0, 0.5, 180),
        (-0.5, -1.0, 0),
        (-1.0, -0.5, 180),
    ],
)
def test_sun_azimuth(phi, declination, expected):
    az = sun_azimuth(phi, declination)
    assert isinstance(az, int)
    assert az in (0, 180)
    assert az == expected


@pytest.mark.parametrize(
    "deg, expected",
    [
        (30.0, 30.0),
        (0.0, 0.0),
        (-30.0, 330.0),
        (-180.0, 180.0),
        (-360.0, 0.0),
        (359.0, 359.0),
        (-1.0, 359.0),
        (1.0, 1.0),
    ],
)
def test_convert_degrees(deg, expected):
    result = convert_degrees(deg)
    assert isinstance(result, float)
    assert result == expected


@pytest.mark.parametrize(
    "decl, ha, elevation, phi, expected",
    [
        (0.0, 0.0, 0.0, math.pi / 4, 180.0),
        (0.0, math.pi / 2, 0.0, math.pi / 4, 270.0),
        (0.0, -math.pi / 2, 0.0, math.pi / 4, 90.0),
    ],
)
def test_calculate_azimuth(decl, ha, elevation, phi, expected):
    az = calculate_azimuth(decl, ha, elevation, phi)
    assert isinstance(az, float)
    assert 0 <= az <= 360
    pytest.approx(az, rel=1e-6)


def test_sunangle_passes():
    az, el, rta, hra, sid, dec = sunangle(year=2024, day=150, hour=12, minute=0, sec=0, zone=0, dasvtm=0, lat=45.0, lon=8.0)

    np.testing.assert_allclose(az, 199.773743, rtol=1e-2)
    np.testing.assert_allclose(el, 65.681384, rtol=1e-2)
    np.testing.assert_allclose(rta, 67.4, rtol=1e-2)
    np.testing.assert_allclose(hra, 8.625919, rtol=1e-2)
    np.testing.assert_allclose(sid, 75.51126, rtol=1e-2)
    np.testing.assert_allclose(dec, 21.8, rtol=1e-2)


@pytest.mark.parametrize(
    ["day", "hour", "minute", "sec", "lat"],
    [
        [-1, 12, 12, 12, 55],
        [367, 12, 12, 12, 45],
        [12, 25, 12, 12, 55],
        [12, 12, 65, 12, 55],
        [12, 12, 12, -1, 55],
        [12, 12, 12, 12, 100],
    ],
)
def test_sunangle_raises(day, hour, minute, sec, lat):
    with pytest.raises(ValueError):
        sunangle(
            year=2000,
            day=day,
            hour=hour,
            minute=minute,
            sec=sec,
            zone=0,
            dasvtm=0,
            lat=lat,
            lon=8,
        )
