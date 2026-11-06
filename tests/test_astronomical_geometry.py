from __future__ import annotations

import numpy as np
import pytest

from marine_qc.helpers.astronomical_geometry import elevation_of_sun


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        (6, 23.654956),
        (9, 47.983963),
        (12, 55.792305),
        (15, 37.182847),
        (18, 11.867064),
    ],
)
def test_elevation_of_sun_passes(hour, expected):
    el = elevation_of_sun(
        year=2024,
        day=150,
        hour=hour,
        minute=0,
        sec=0,
        zone=0,
        dasvtm=0,
        lat=55.0,
        lon=10.0,
    )
    print(el, expected)
    np.testing.assert_allclose(el, expected, rtol=1e-2)


@pytest.mark.parametrize(
    ["day", "hour", "minute", "sec", "lat", "lon", "dasvtm"],
    [
        [-1, 12, 12, 12, 55, 10, 0],
        [367, 12, 12, 12, 55, 10, 0],
        [12, 25, 12, 12, 55, 10, 0],
        [12, 12, 65, 12, 55, 10, 0],
        [12, 12, 12, -1, 55, 10, 0],
        [12, 12, 12, 12, 100, 10, 0],
        [12, 12, 12, 12, 55, 200, 0],
        [12, 12, 12, 12, 55, 10, 2],
    ],
)
def test_elevation_of_sun_raises(day, hour, minute, sec, lat, lon, dasvtm):
    with pytest.raises(ValueError):
        elevation_of_sun(
            year=2000,
            day=day,
            hour=hour,
            minute=minute,
            sec=sec,
            zone=0,
            dasvtm=dasvtm,
            lat=lat,
            lon=lon,
        )
