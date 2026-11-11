from __future__ import annotations

import numpy as np
import pytest

from marine_qc.helpers.location_control import (
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
def test_lats_with_res(lat, res, expected):
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
def test_lons_with_res(lon, res, expected):
    result = mds_lon_to_xindex(lon, res=res)
    if isinstance(lon, np.ndarray):
        assert np.all(result == expected)
    else:
        assert result == expected
