from __future__ import annotations

import pytest

from marine_qc.astronomical_geometry import (
    convert_degrees,
    sunangle,
)


def test_convert_degrees():
    assert convert_degrees(-1.0) == 359.0
    assert convert_degrees(1.0) == 1.0


@pytest.mark.parametrize(
    ["day", "hour", "minute", "sec", "lat"],
    [
        [-1, 12, 12, 12, 55],
        [12, 25, 12, 12, 55],
        [12, 12, 65, 12, 55],
        [12, 12, 12, -1, 55],
        [12, 12, 12, 12, 100],
    ],
)
def test_sunangle(day, hour, minute, sec, lat):
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
