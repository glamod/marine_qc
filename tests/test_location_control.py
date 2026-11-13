from __future__ import annotations

import numpy as np
import pytest

from marine_qc.helpers.location_control import (
    get_x_index,
    get_y_index,
    mds_lat_to_yindex,
    mds_lon_to_xindex,
)


@pytest.mark.parametrize(
    "lat, res, expected",
    [
        (90.0, 1.0, 0),
        (89.0, 1.0, 0),
        (88.0, 1.0, 1),
        (87.0, 1.0, 2),
        (88.7, 1.0, 1),
        (-90.0, 1.0, 179),
        (-89.0, 1.0, 179),
        (-88.0, 1.0, 178),
        (-87.0, 1.0, 177),
        (-88.7, 1.0, 178),
        (0.0, 1.0, 90),
        (0.5, 1.0, 89),
        (1.0, 1.0, 88),
        (-0.5, 1.0, 90),
        (-1.0, 1.0, 91),
        (90.0, 5.0, 0),
        (88.0, 5.0, 0),
        (85.0, 5.0, 0),
        (-85.0, 5.0, 35),
        (-88.4, 5.0, 35),
        (-90.0, 5.0, 35),
        (0.0, 5.0, 18),
        (
            np.array([90.0, 88.0, 85.0, -85.0, -88.4, -90.0, 0.0]),
            5.0,
            np.array([0, 0, 0, 35, 35, 35, 18]),
        ),
        (
            np.array([90, 89, 88, 87, 88.7, -90, -89, -88, -87, -88.7, 0, 0.5, 1.0, -0.5, -1.0]),
            1.0,
            np.array([0, 0, 1, 2, 1, 179, 179, 178, 177, 178, 90, 89, 88, 90, 91]),
        ),
    ],
)
def test_mds_lat_to_yindex(lat, res, expected):
    result = mds_lat_to_yindex(lat, res=res)
    if isinstance(lat, np.ndarray):
        assert np.all(result == expected)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "lon, res, expected",
    [
        (-180.0, 1.0, 0),
        (-179.0, 1.0, 0),
        (-178.0, 1.0, 1),
        (180.0, 1.0, 359),
        (179.0, 1.0, 359),
        (178.0, 1.0, 358),
        (-1.0, 1.0, 178),
        (0.0, 1.0, 179),
        (1.0, 1.0, 181),
        (-180.0, 5.0, 0),
        (-178.0, 5.0, 0),
        (-175.0, 5.0, 0),
        (175.0, 5.0, 71),
        (178.4, 5.0, 71),
        (180.0, 5.0, 71),
        (0.0, 5.0, 35),
        (
            np.array([-180, -178, -175, 175, 178.4, 180, 0]),
            5.0,
            np.array([0, 0, 0, 71, 71, 71, 35]),
        ),
        (
            np.array([-180, -179, -178, 180, 179, 178, -1, 0, 1]),
            1.0,
            np.array([0, 0, 1, 359, 359, 358, 178, 179, 181]),
        ),
    ],
)
def test_mds_lon_to_xindex(lon, res, expected):
    result = mds_lon_to_xindex(lon, res=res)
    if isinstance(lon, np.ndarray):
        assert np.all(result == expected)
    else:
        assert result == expected


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
def test_get_x_index_pass(lats, lat0, delta, expected):
    n_lat_axis = int(360 / abs(delta))
    lat_axis = np.arange(n_lat_axis) * delta + lat0
    lats = np.array(lats)
    expected = np.array(expected)

    result = get_x_index(lats, lat_axis)

    assert np.all(expected == result)


def test_get_x_index_raise():
    lons = np.array([-10.0, 10.0])
    lon_axis = np.arange(360) + (-178)

    with pytest.raises(RuntimeError, match="I can't work this grid out, grid box boundaries"):
        get_x_index(lons, lon_axis)


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
def test_get_y_index_pass(lats, lat0, delta, expected):
    n_lat_axis = int(180 / abs(delta))
    lat_axis = np.arange(n_lat_axis) * delta + lat0
    lats = np.array(lats)
    expected = np.array(expected)

    result = get_y_index(lats, lat_axis)

    assert np.all(expected == result)


def test_get_y_index_raise():
    lats = np.array([-10.0, 10.0])
    lat_axis = np.arange(180) + (-88)

    with pytest.raises(RuntimeError, match="I can't work this grid out, grid box boundaries"):
        get_y_index(lats, lat_axis)
