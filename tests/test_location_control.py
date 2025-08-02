from __future__ import annotations

import math

import pytest  # noqa

from marine_qc.location_control import (
    fill_missing_vals,
    get_four_surrounding_points,
    lat_to_yindex,
    lon_to_xindex,
    mds_lat_to_yindex,
    mds_lon_to_xindex,
    xindex_to_lon,
    yindex_to_lat,
)


@pytest.mark.parametrize(
    "yindex, resolution, expected",
    [
        (0, 1, 89.5),
        (179, 1, -89.5),
        (35, 5, -87.5),
        (0, 5, 87.5),
    ],
)
def test_0_is_89point5(yindex, resolution, expected):
    assert yindex_to_lat(yindex, res=resolution) == expected


def test_yindex_to_lat_raises():
    with pytest.raises(ValueError):
        yindex_to_lat(-1, 1)
    with pytest.raises(ValueError):
        yindex_to_lat(180, 1)


@pytest.mark.parametrize(
    "lat, res, expected",
    [
        (90.0, 5.0, 0),
        (88.0, 5.0, 0),
        (85.0, 5.0, 0),
        (-85.0, 5.0, 35),
        (-88.4, 5.0, 35),
        (-90.0, 5.0, 35),
        (0.0, 5.0, 18),
    ],
)
def test_lats_with_res(lat, res, expected):
    assert mds_lat_to_yindex(lat, res=res) == expected


@pytest.mark.parametrize(
    "lon, res, expected",
    [
        (-180.0, 5.0, 0),
        (-178.0, 5.0, 0),
        (-175.0, 5.0, 0),
        (175.0, 5.0, 71),
        (178.4, 5.0, 71),
        (180.0, 5.0, 71),
        (0.0, 5.0, 35),
    ],
)
def test_lons_with_res(lon, res, expected):
    assert mds_lon_to_xindex(lon, res=res) == expected


@pytest.mark.parametrize(
    "lat, expected",
    [
        (90.0, 0),
        (89.0, 0),
        (88.0, 1),
        (87.0, 2),
        (88.7, 1),
        (-90.0, 179),
        (-89.0, 179),
        (-88.0, 178),
        (-87.0, 177),
        (-88.7, 178),
        (0.0, 90),
        (0.5, 89),
        (1.0, 88),
        (-0.5, 90),
        (-1.0, 91),
    ],
)
def test_lats(lat, expected):
    assert mds_lat_to_yindex(lat, res=1) == expected


@pytest.mark.parametrize(
    "lon, expected",
    [
        (-180.0, 0),
        (-179.0, 0),
        (-178.0, 1),
        (180.0, 359),
        (179.0, 359),
        (178.0, 358),
        (-1.0, 178),
        (0.0, 179),
        (1.0, 181),
    ],
)
def test_lons(lon, expected):
    assert mds_lon_to_xindex(lon, res=1) == expected


@pytest.mark.parametrize(
    "lon, res, expected",
    [
        (-180, 1, 0),
        (180, 1, 0),
        (-180.5, 1, 359),
        (-181.5, 1, 358),
        (-74.0, 1, 106),
        (-179.5, 1, 0),
        (180.5, 1, 0),
        (359.5, 1, 179),
        (179.5, 5, 71),
        (179.9, 0.25, 1439),
        (-179.9 + 0.25, 0.25, 1),
        (720.0, 1, 180),
        (720.0, 5, 36),
        (-360.0, 1, 180),
        (-360.0, 5, 36),
    ],
)
def test_lon_to_xindex(lon, res, expected):
    assert lon_to_xindex(lon, res=res) == expected


def test_lons_ge_180():
    """Test to make sure wrapping works"""
    res = 1
    assert 180 == lon_to_xindex(360.0, res=res)
    assert 5 == lon_to_xindex(185.1, res=res)
    for i in range(0, 520):
        assert math.fmod(i, 360) == lon_to_xindex(-179.5 + float(i), res=res)
    # And at different resolutions
    for i in range(0, 520):
        assert math.fmod(int(i / 5), 72) == lon_to_xindex(-179.5 + float(i), res=5.0)


@pytest.mark.parametrize(
    "lat, res, expected",
    [
        (99.2, 5, 0),
        (-99.2, 5, 35),
        (99.2, 1, 0),
        (37.0, 1, 53),
        (35.0, 5, 11),
        (-199.3, 1, 179),
        (87.52, 0.25, 9),
        (-2.5, 5, 18),
        (89.9, 0.25, 0),
        (89.9, 0.5, 0),
        (89.9, 1.0, 0),
        (89.9, 2.0, 0),
        (89.9, 5.0, 0),
        (-89.9, 0.25, 719),
        (-89.9, 0.5, 359),
        (-89.9, 1.0, 179),
        (-89.9, 2.0, 89),
        (-89.9, 5.0, 35),
    ],
)
def test_lat_to_yindex(lat, res, expected):
    assert lat_to_yindex(lat, res=res) == expected


def test_borderline():
    for i in range(0, 180):
        assert i == lat_to_yindex(90 - i, res=1)


def test_gridcentres():
    for i in range(0, 180):
        assert i == lat_to_yindex(90 - i - 0.5, res=1)


@pytest.mark.parametrize(
    "xindex, res, lon",
    [
        (0, 1, -179.5),
        (359, 1, 179.5),
        (179, 1, -0.5),
        (180, 1, 0.5),
    ],
)
def test_xindex_to_lon(xindex, res, lon):
    assert xindex_to_lon(xindex, res) == lon


@pytest.mark.parametrize(
    "q11, q12, q21, q22, expected",
    [
        (None, None, None, None, (None, None, None, None)),
        (1.0, 2.0, 3.0, None, (1.0, 2.0, 3.0, 2.5)),
        (None, 2.0, 3.0, None, (2.5, 2.0, 3.0, 2.5)),
        (None, None, 3.0, None, (3.0, 3.0, 3.0, 3.0)),
    ],
)
def test_fill_missing_values(q11, q12, q21, q22, expected):
    assert fill_missing_vals(q11, q12, q21, q22) == expected


@pytest.mark.parametrize(
    "lat, lon, max90, expected",
    [
        (0.4, 322.2 - 360, True, (-38.5, -37.5, -0.5, 0.5)),
        (89.9, 0.1, True, (-0.5, 0.5, 89.5, 89.5)),
        (89.9, 0.1, False, (-0.5, 0.5, 89.5, 90.5)),
        (0.1, 0.1, True, (-0.5, 0.5, -0.5, 0.5)),
        (0.1, 0.1, None, (-0.5, 0.5, -0.5, 0.5)),
        (0.0, 0.0, True, (-0.5, 0.5, -0.5, 0.5)),
        (0.0, 0.0, None, (-0.5, 0.5, -0.5, 0.5)),
        (0.0, 179.9, True, (179.5, 180.5, -0.5, 0.5)),
        (0.0, 179.9, None, (179.5, 180.5, -0.5, 0.5)),
        (0.0, -179.9, True, (-180.5, -179.5, -0.5, 0.5)),
        (0.0, -179.9, None, (-180.5, -179.5, -0.5, 0.5)),
        (-89.9, 0.1, True, (-0.5, 0.5, -89.5, -89.5)),
        (-89.9, 0.1, False, (-0.5, 0.5, -90.5, -89.5)),
        (-89.9, 0.1, None, (-0.5, 0.5, -90.5, -89.5)),
    ],
)
def test_get_four_surrounding_points(lat, lon, max90, expected):
    assert get_four_surrounding_points(lat, lon, res=1.0, max90=max90) == expected
