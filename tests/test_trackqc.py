# flake8: noqa: E501

from __future__ import annotations
from datetime import datetime

import numpy as np
import pytest

import marine_qc.buoy_tracking_qc as tqc
from marine_qc.auxiliary import untestable


@pytest.mark.parametrize(
    "year, month, day, hour, lat, lon, elevdlim, expected",
    [
        (2019, 6, 21, 12.0, 50.7, -3.5, -2.5, True),
        (2019, 6, 21, 0.0, 50.7, -3.5, -2.5, False),
        (2019, 6, 21, 0.0, 50.7, -3.5, 89.0, False),
        (2022, 5, 3, 12.0, 0.0, 0.0, -2.5, True),
        (2025, 5, 26, 4.0 + 55.0 / 60.0, -20.00, 23.42, -2.5, True),
        (2025, 5, 26, 4.0 + 35.0 / 60.0, -20.00, 23.42, -2.5, False),
    ],
)
def test_daytime(year, month, day, hour, lat, lon, elevdlim, expected):
    datetime = tqc.track_day_test(year, month, day, hour, lat, lon, elevdlim=elevdlim)
    assert datetime == expected


@pytest.mark.parametrize(
    "year, month, day, hour, lat, lon",
    [
        (2019, 13, 21, 0.0, 50.7, -3.5),
        (None, 12, 21, 0.0, 50.7, -3.5),
        (2019, None, 21, 0.0, 50.7, -3.5),
        (2019, 12, None, 0.0, 50.7, -3.5),
        (2019, 12, 21, None, 50.7, -3.5),
        (2019, 12, 21, 0.0, None, -3.5),
        (2019, 12, 21, 0.0, 50.7, None),
        (2019, 12, 43, 0.0, 50.7, -3.5),
        (2019, 12, 21, -5.0, 50.7, -3.5),
        (2019, 12, 21, 0.0, -99.0, -3.5),
    ],
)
def test_daytime_error_invalid_parameter(year, month, day, hour, lat, lon):
    with pytest.raises(ValueError):
        assert tqc.track_day_test(year, month, day, hour, lat, lon, elevdlim=-2.5)


@pytest.mark.parametrize(
    "inarr, expected",
    [
        ([2, 3, 4], True),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], True),
        ([2, 3, 4, 5, 4], False),
    ],
)
def test_is_monotonic(inarr, expected):
    assert tqc.is_monotonic(inarr) == expected


def aground_check_test_data(selector):
    # stationary drifter
    # fmt: off
    vals1 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
    ]
    # stationary drifter (artificial 'jitter' spikes)
    vals2 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 1.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.0, "LON": 1.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
    ]
    # stationary drifter (artificial 'jitter' which won't be fully smoothed and outside tolerance)
    vals3 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 1.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 1.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
    ]
    # stationary drifter (artificial 'jitter' which won't be fully smoothed and within tolerance)
    vals4 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 0.01, "LON": 0.01, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.01, "LON": 0.01, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
    ]
    # moving drifter (going west)
    vals5 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": -0.02, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 0.0, "LON": -0.04, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.0, "LON": -0.06, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.0, "LON": -0.08, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.0, "LON": -0.10, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": -0.12, "SST": 5.0, },
    ]
    # moving drifter (going north)
    vals6 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.02, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 0.04, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.06, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.08, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.10, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.12, "LON": 0.0, "SST": 5.0, },
    ]
    # runs aground (drifter going north then stops)
    vals7 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.02, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 0.04, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.06, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.08, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.08, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.08, "LON": 0.0, "SST": 5.0, },
    ]
    # stationary drifter (high frequency sampling prevents detection)
    vals8 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
    ]
    # stationary drifter (low frequency sampling prevents detection)
    vals9 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 10, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 13, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 16, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 19, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
    ]
    # stationary drifter (mid frequency sampling enables detection)
    vals10 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 9, "HR": 0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 10, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
    ]
    # stationary drifter (changed sampling prevents early detection)
    vals11 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 8, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 9, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 10, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 11, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
    ]
    # moving drifter (going northwest at equator but going slowly and within tolerance)
    vals12 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.005, "LON": -0.005, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 0.01, "LON": -0.01, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.015, "LON": -0.015, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.02, "LON": -0.02, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.025, "LON": -0.025, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.03, "LON": -0.03, "SST": 5.0, },
    ]
    # moving drifter (going west in high Arctic but going slower than tolerance set at equator)
    vals13 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 85.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 85.0, "LON": -0.02, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 85.0, "LON": -0.04, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 85.0, "LON": -0.06, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 85.0, "LON": -0.08, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 85.0, "LON": -0.10, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 85.0, "LON": -0.12, "SST": 5.0, },
    ]
    # stationary then moves
    vals14 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.02, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.04, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.06, "LON": 0.0, "SST": 5.0, },
    ]
    # too short for QC
    vals15 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": -0.02, "SST": 5.0, },
    ]
    # assertion error - bad input parameter
    vals16 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": -0.02, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 0.0, "LON": -0.04, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.0, "LON": -0.06, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.0, "LON": -0.08, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.0, "LON": -0.10, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": -0.12, "SST": 5.0, },
    ]
    # assertion error - missing observation
    vals17 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": -0.02, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 0.0, "LON": -0.04, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.0, "LON": -0.06, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.0, "LON": -0.08, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.0, "LON": -0.10, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": -0.12, "SST": 5.0, },
    ]
    # assertion error - times not sorted
    vals18 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": -0.02, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 0.0, "LON": -0.04, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": -0.06, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.0, "LON": -0.08, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.0, "LON": -0.10, "SST": 5.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": -0.12, "SST": 5.0, },
    ]
    # fmt: on
    obs = locals()[f"vals{selector}"]
    reps = {}
    for key in obs[0]:
        reps[key] = []
    reps["DATE"] = []

    for v in obs:
        for key in reps:
            if key != "DATE":
                reps[key].append(v[key])

        hour = int(v["HR"])
        minute = 60 * (v["HR"] - hour)
        date = datetime(v["YR"], v["MO"], v["DY"], hour, minute)
        reps["DATE"].append(date)

    for key in reps:
        reps[key] = np.array(reps[key])

    if selector == 17:
        reps["LON"][1] = np.nan

    return reps["LAT"], reps["LON"], reps["DATE"]


@pytest.mark.parametrize(
    # fmt: off
    "selector, smooth_win, min_win_period, max_win_period, expected, warns",
    [
        (1, 3, 1, 2, [1, 1, 1, 1, 1, 1, 1], False),  # test_stationary
        (2, 3, 1, 2, [1, 1, 1, 1, 1, 1, 1], False),  # test_stationary jitter spikes
        (3, 3, 1, 2, [0, 0, 0, 0, 1, 1, 1], False),  # test stationary big remaining jitter
        (4, 3, 1, 2, [1, 1, 1, 1, 1, 1, 1], False),  # test_stationary_small_remaining_jitter
        (5, 3, 1, 2, [0, 0, 0, 0, 0, 0, 0], False),  # test_moving_west
        (6, 3, 1, 2, [0, 0, 0, 0, 0, 0, 0], False),  # test_moving_north
        (7, 3, 1, 2, [0, 0, 0, 0, 1, 1, 1], False),  # test_moving_north_then_stop
        (8, 3, 1, 2, [0, 0, 0, 0, 0, 0, 0], False),  # test_stationary_high_freq_sampling
        (9, 3, 1, 2, [0, 0, 0, 0, 0, 0, 0], False),  # test_stationary_low_freq_sampling
        (10, 3, 1, 2, [1, 1, 1, 1, 1, 1, 1], False),  # test_stationary_mid_freq_sampling
        (11, 3, 1, 2, [0, 0, 1, 1, 1, 1, 1], False),  # test_stationary_low_to_mid_freq_sampling
        (12, 3, 1, 2, [1, 1, 1, 1, 1, 1, 1], False),  # test_moving_slowly_northwest
        (13, 3, 1, 2, [1, 1, 1, 1, 1, 1, 1], False),  # test_moving_slowly_west_in_arctic
        (14, 3, 1, 2, [0, 0, 0, 0, 0, 0, 0], False),  # test_stop_then_moving_north
        (15, 3, 1, 2, [0, 0], False),  # test_too_short_for_qc
        (16, 0, 1, 2, [untestable for x in range(7)], True),  # test_error_bad_input_parameter
        (17, 3, 1, 2, [untestable for x in range(7)], True),  # test_error_missing_observation
        (18, 3, 1, 2, [untestable for x in range(7)], True),  # test_error_not_time_sorted
        # fmt: off
    ],
)
def test_generic_aground(selector, smooth_win, min_win_period, max_win_period, expected, warns):
    lats, lons, dates = aground_check_test_data(selector)
    if warns:
        with pytest.warns(UserWarning):
            qc_outcomes = tqc.do_aground_check(lons, lats, dates, smooth_win, min_win_period, max_win_period)
    else:
        qc_outcomes = tqc.do_aground_check(lons, lats, dates, smooth_win, min_win_period, max_win_period)
    for i in range(len(lons)):
        assert qc_outcomes[i] == expected[i]


@pytest.mark.parametrize(  # fmt: off
    "selector, smooth_win, min_win_period, max_win_period, expected, warns",
    [
        (1, 3, 1, 2, [1, 1, 1, 1, 1, 1, 1], False),  # test_stationary
        (2, 3, 1, 2, [1, 1, 1, 1, 1, 1, 1], False),  # test_stationary jitter spikes
        (
            3,
            3,
            1,
            2,
            [0, 0, 0, 0, 1, 1, 1],
            False,
        ),  # test stationary big remaining jitter
        (
            4,
            3,
            1,
            2,
            [1, 1, 1, 1, 1, 1, 1],
            False,
        ),  # test_stationary_small_remaining_jitter
        (5, 3, 1, 2, [0, 0, 0, 0, 0, 0, 0], False),  # test_moving_west
        (6, 3, 1, 2, [0, 0, 0, 0, 0, 0, 0], False),  # test_moving_north
        (7, 3, 1, 2, [0, 0, 0, 0, 1, 1, 1], False),  # test_moving_north_then_stop
        (
            8,
            3,
            1,
            2,
            [0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_stationary_high_freq_sampling
        (9, 3, 1, 2, [1, 1, 1, 1, 1, 1, 1], False),  # test_stationary_low_freq_sampling
        (
            10,
            3,
            1,
            2,
            [1, 1, 1, 1, 1, 1, 1],
            False,
        ),  # test_stationary_mid_freq_sampling
        (
            11,
            3,
            1,
            2,
            [1, 1, 1, 1, 1, 1, 1],
            False,
        ),  # test_stationary_low_to_mid_freq_sampling
        (12, 3, 1, 2, [0, 0, 0, 1, 1, 1, 1], False),  # test_moving_slowly_northwest
        (
            13,
            3,
            1,
            2,
            [1, 1, 1, 1, 1, 1, 1],
            False,
        ),  # test_moving_slowly_west_in_arctic
        (14, 3, 1, 2, [0, 0, 0, 0, 0, 0, 0], False),  # test_stop_then_moving_north
        (15, 3, 1, 2, [0, 0], False),  # test_too_short_for_qc
        (
            16,
            0,
            1,
            2,
            [untestable for x in range(7)],
            True,
        ),  # test_error_bad_input_parameter
        (
            17,
            3,
            1,
            2,
            [untestable for x in range(7)],
            True,
        ),  # test_error_missing_observation
        (
            18,
            3,
            1,
            2,
            [untestable for x in range(7)],
            True,
        ),  # test_error_not_time_sorted
    ],  # fmt: off
)
def test_new_generic_aground(selector, smooth_win, min_win_period, max_win_period, expected, warns):
    lats, lons, dates = aground_check_test_data(selector)
    if warns:
        with pytest.warns(UserWarning):
            qc_outcomes = tqc.do_new_aground_check(lons, lats, dates, smooth_win, min_win_period)
    else:
        qc_outcomes = tqc.do_new_aground_check(lons, lats, dates, smooth_win, min_win_period)
    for i in range(len(lons)):
        assert qc_outcomes[i] == expected[i]


@pytest.fixture
def iquam_parameters():
    return {
        "number_of_neighbours": 5,
        "buoy_speed_limit": 15.0,
        "ship_speed_limit": 60.0,
        "delta_d": 1.11,
        "delta_t": 0.01,
    }


def speed_check_data(selector):
    # stationary drifter
    # fmt: off
    vals1 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
    ]
    # fast moving drifter
    vals2 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 3.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 6.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 9.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 12.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 15.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 18.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
    ]
    # slow moving drifter
    vals3 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": 1.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 0.0, "LON": 2.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.0, "LON": 3.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.0, "LON": 4.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.0, "LON": 5.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": 6.0, "SST": 5.0, "PT": 7, },
    ]
    # slow-fast-slow moving drifter
    vals4 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 1.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 2.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 5.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 8.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 9.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 10.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
    ]
    # fast moving drifter (high frequency sampling)
    vals5 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 1, "LAT": 3.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 2, "LAT": 6.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 3, "LAT": 9.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 4, "LAT": 12.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 5, "LAT": 15.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 6, "LAT": 18.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
    ]
    # fast moving drifter (low frequency sampling)
    vals6 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 13, "LAT": 5.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 14, "LAT": 10.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 15, "LAT": 15.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 16, "LAT": 20.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 17, "LAT": 25.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 18, "LAT": 30.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
    ]
    # slow-fast-slow moving drifter (mid frequency sampling)
    vals7 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.5, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 0, "LAT": 1.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 2.5, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 0, "LAT": 4.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 4.5, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 0, "LAT": 5.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
    ]
    # fast moving drifter (with irregular sampling)
    vals8 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 3.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 12.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 12.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 23, "LAT": 14.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 23, "LAT": 14.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 23, "LAT": 17.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
    ]
    # fast moving Arctic drifter
    vals9 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 85.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 85.0, "LON": 30.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 85.0, "LON": 60.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 85.0, "LON": 90.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 85.0, "LON": 120.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 85.0, "LON": 150.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 85.0, "LON": 180.0, "SST": 5.0, "PT": 7, },
    ]
    # stationary drifter (gross position errors)
    vals10 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 50.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 50.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
    ]
    # too short for QC
    vals11 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
    ]
    # assertion error - bad input parameter
    vals12 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
    ]
    # assertion error - missing observation
    vals13 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 4, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
    ]
    # assertion error - times not sorted
    vals14 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 3, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 2, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 5, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 6, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 7, "HR": 12, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "PT": 7, },
    ]
    # fmt: on
    obs = locals()[f"vals{selector}"]
    reps = {}
    for key in obs[0]:
        reps[key] = []
    reps["DATE"] = []

    for v in obs:
        for key in reps:
            if key != "DATE":
                reps[key].append(v[key])

        hour = int(v["HR"])
        minute = 60 * (v["HR"] - hour)
        date = datetime(v["YR"], v["MO"], v["DY"], hour, minute)
        reps["DATE"].append(date)

    for key in reps:
        reps[key] = np.array(reps[key])

    if selector == 13:
        reps["LON"][1] = np.nan

    return reps["LAT"], reps["LON"], reps["DATE"]


@pytest.mark.parametrize(  # fmt: off
    "selector, speed_limit, min_win_period, max_win_period, expected, warns",
    [
        (1, 2.5, 0.5, 1.0, [0, 0, 0, 0, 0, 0, 0], False),  # test stationary
        (2, 2.5, 0.5, 1.0, [1, 1, 1, 1, 1, 1, 1], False),  # test_fast_drifter
        (3, 2.5, 0.5, 1.0, [0, 0, 0, 0, 0, 0, 0], False),  # test_slow_drifter
        (4, 2.5, 0.5, 1.0, [0, 0, 1, 1, 1, 0, 0], False),  # test_slow_fast_slow_drifter
        (5, 2.5, 0.5, 1.0, [0, 0, 0, 0, 0, 0, 0], False),  # test_high_freqency_sampling
        (6, 2.5, 0.5, 1.0, [0, 0, 0, 0, 0, 0, 0], False),  # test_low_freqency_sampling
        (
            7,
            2.5,
            0.5,
            1.0,
            [0, 1, 1, 1, 1, 1, 0],
            False,
        ),  # test_slow_fast_slow_mid_freqency_sampling
        (8, 2.5, 0.5, 1.0, [1, 1, 0, 0, 0, 1, 1], False),  # test_irregular_sampling
        (9, 2.5, 0.5, 1.0, [1, 1, 1, 1, 1, 1, 1], False),  # test_fast_arctic_drifter
        (
            10,
            2.5,
            0.5,
            1.0,
            [0, 1, 1, 1, 1, 1, 1],
            False,
        ),  # test_stationary_gross_error
        (11, 2.5, 0.5, 1.0, [0, 0], False),  # test_too_short_for_qc_a
        (
            12,
            -2.5,
            0.5,
            1.0,
            [untestable for x in range(7)],
            True,
        ),  # test_error_bad_input_parameter_a
        (
            13,
            2.5,
            0.5,
            1.0,
            [untestable for x in range(7)],
            True,
        ),  # test_error_missing_observation_a
        (
            14,
            2.5,
            0.5,
            1.0,
            [untestable for x in range(7)],
            True,
        ),  # test_error_not_time_sorted_a
    ],  # fmt: off
)
def test_generic_speed_tests(selector, speed_limit, min_win_period, max_win_period, expected, warns):
    lats, lons, dates = speed_check_data(selector)
    if warns:
        with pytest.warns(UserWarning):
            qc_outcomes = tqc.do_speed_check(lons, lats, dates, speed_limit, min_win_period, max_win_period)
    else:
        qc_outcomes = tqc.do_speed_check(lons, lats, dates, speed_limit, min_win_period, max_win_period)
    for i in range(len(qc_outcomes)):
        assert qc_outcomes[i] == expected[i]


@pytest.mark.parametrize(
    # fmt: off
    "selector, speed_limit, min_win_period, ship_speed_limit, delta_d, delta_t, n_neighbours, expected, warns",
    [
        (1, 2.5, 0.5, 60.0, 1.11, 0.01, 5, [0, 0, 0, 0, 0, 0, 0], False),  # test_new_stationary_a
        (2, 2.5, 0.5, 60.0, 1.11, 0.01, 5, [1, 1, 1, 1, 1, 1, 1], False),  # test_new_fast_drifter
        (3, 2.5, 0.5, 60.0, 1.11, 0.01, 5, [0, 0, 0, 0, 0, 0, 0], False),  # test_new_slow_drifter
        (4, 2.5, 0.5, 60.0, 1.11, 0.01, 5, [0, 0, 1, 1, 1, 0, 0], False),  # test_new_slow_fast_slow_drifter
        (5, 2.5, 0.5, 60.0, 1.11, 0.01, 5, [0, 0, 0, 0, 0, 0, 0], False),  # test_new_high_freqency_sampling
        (6, 2.5, 0.5, 60.0, 1.11, 0.01, 5, [1, 1, 1, 1, 1, 1, 1], False),  # test_new_low_freqency_sampling
        (7, 2.5, 0.5, 60.0, 1.11, 0.01, 5, [0, 0, 1, 1, 1, 0, 0], False),  # test_new_slow_fast_slow_mid_freqency_sampling
        (8, 2.5, 0.5, 60.0, 1.11, 0.01, 5, [1, 1, 1, 0, 0, 1, 1], False),  # test_new_irregular_sampling
        (9, 2.5, 0.5, 60.0, 1.11, 0.01, 5, [1, 1, 1, 1, 1, 1, 1], False),  # test_new_fast_arctic_drifter
        (10, 2.5, 0.5, 60.0, 1.11, 0.01, 5, [0, 0, 0, 0, 0, 0, 0], False),  # test_new_stationary_gross_error
        (11, 2.5, 0.5, 60.0, 1.11, 0.01, 5, [0, 0], False),  # test_new_too_short_for_qc_a
        (12, -2.5, 0.5, 60.0, 1.11, 0.01, 5, [untestable for _ in range(7)], True),  # test_new_error_bad_input_parameter_a
        (13, 2.5, 0.5, 60.0, 1.11, 0.01, 5, [untestable for _ in range(7)], True),  # test_new_error_missing_observation_a
        (14, 2.5, 0.5, 60.0, 1.11, 0.01, 5, [untestable for _ in range(7)], True),  # test_new_error_not_time_sorted_a
    ],
    # fmt: on
)
def test_generic_new_speed_tests(
    selector,
    speed_limit,
    min_win_period,
    ship_speed_limit,
    delta_d,
    delta_t,
    n_neighbours,
    expected,
    warns,
):
    lats, lons, dates = speed_check_data(selector)
    if warns:
        with pytest.warns(UserWarning):
            qc_outcomes = tqc.do_new_speed_check(
                lons,
                lats,
                dates,
                speed_limit,
                min_win_period,
                ship_speed_limit,
                delta_d,
                delta_t,
                n_neighbours,
            )
    else:
        qc_outcomes = tqc.do_new_speed_check(
            lons,
            lats,
            dates,
            speed_limit,
            min_win_period,
            ship_speed_limit,
            delta_d,
            delta_t,
            n_neighbours,
        )
    for i in range(len(qc_outcomes)):
        assert qc_outcomes[i] == expected[i]


def tailcheck_vals(selector):
    # all daytime
    # fmt: off
    # @formatter:off
    vals1 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 12.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # all land-masked
    vals2 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # all ice
    vals3 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.2, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.2, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.2, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.2, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.2, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.2, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.2, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.2, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.2, },
    ]

    # one usable value
    vals4 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # start tail bias
    vals5 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # start tail negative bias
    vals6 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 4.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 4.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 4.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 4.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # start tail bias obs missing
    vals7 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # end tail bias
    vals8 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # end tail bias obs missing
    vals9 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": None, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # start tail noisy
    vals10 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 7.5, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # end tail noisy
    vals11 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 7.5, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # two tails
    vals12 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 7.5, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # all biased
    vals13 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # all noisy
    vals14 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 7.5, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 7.5, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 7.5, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # start tail bias with bgvar
    vals15 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.4, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # all biased with bgvar
    vals16 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.4, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # short start tail
    vals17 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # short end tail
    vals18 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # short two tails
    vals19 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # short all fail
    vals20 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # short start tail with bgvar
    vals21 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.4, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # short all fail with bgvar
    vals22 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.4, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 7.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # long and short start tail
    vals23 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 6.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # long and short end tail
    vals24 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 6.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # long and short two tail
    vals25 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 6.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 6.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # one long and one short tail
    vals26 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 6.2, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # too short for short tail
    vals27 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # long and short all fail
    vals28 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.5, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.5, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # long and short start tail with bgvar
    vals29 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 7.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 7.0, "OSTIA": 5.0, "BGVAR": 0.4, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # long and short all fail with bgvar
    vals30 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 6.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.5, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.5, "OSTIA": 5.0, "BGVAR": 0.4, "ICE": 0.0, },
    ]

    # good data
    vals31 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.1, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.1, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.1, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.1, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.1, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.01, "ICE": 0.0, },
    ]

    # long and short start tail big bgvar
    vals32 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 6.1, "OSTIA": 5.0, "BGVAR": 0.3, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 6.1, "OSTIA": 5.0, "BGVAR": 0.3, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 6.1, "OSTIA": 5.0, "BGVAR": 0.3, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 6.1, "OSTIA": 5.0, "BGVAR": 0.3, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.3, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.3, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.3, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.3, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 0.3, "ICE": 0.0, },
    ]

    # start tail noisy big bgvar
    vals33 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 7.5, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
    ]

    # assertion error - bad input parameter
    vals34 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
    ]

    # assertion error - missing matched value
    vals35 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
    ]

    # assertion error - invalid ice value
    vals36 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 1.1, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
    ]

    # assertion error - missing observation value
    vals37 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
    ]

    # assertion error - times not sorted
    vals38 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
    ]

    # assertion error - invalid background sst
    vals39 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 46.0, "BGVAR": 1.0, "ICE": 0.0, },
    ]

    # assertion error - invalid background error variance
    vals40 = [
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.0, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.1, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.2, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.3, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": -1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.4, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.5, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.6, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.7, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
        {"ID": "AAAAAAAAA", "YR": 2003, "MO": 12, "DY": 1, "HR": 0.8, "LAT": 0.0, "LON": 0.0, "SST": 5.0, "OSTIA": 5.0, "BGVAR": 1.0, "ICE": 0.0, },
    ]
    # fmt: on
    # @formatter:on
    obs = locals()[f"vals{selector}"]
    reps = {}
    for key in obs[0]:
        reps[key] = []
    reps["DATE"] = []

    for v in obs:
        for key in reps:
            if key != "DATE":
                reps[key].append(v[key])

        hour = int(v["HR"])
        minute = int(60 * (v["HR"] - hour))
        date = datetime(int(v["YR"]), int(v["MO"]), int(v["DY"]), hour, minute)
        reps["DATE"].append(date)

    for key in reps:
        reps[key] = np.array(reps[key])

    if selector == 37:
        reps["LAT"][1] = np.nan
    if selector == 35:
        reps["OSTIA"][:] = np.nan

    return (
        reps["LAT"],
        reps["LON"],
        reps["DATE"],
        reps["SST"],
        reps["OSTIA"],
        reps["BGVAR"],
        reps["ICE"],
    )


@pytest.mark.parametrize(
    # fmt: off
    "selector, long_win_len, long_err_std_n, short_win_len, short_err_std_n, short_win_n_bad, drif_inter, drif_intra, background_err_lim, expected1, expected2, warns",
    [
        (1, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_all_daytime
        (2, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_all_land_masked
        (3, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_all_ice
        (4, 3, 3.0, 2, 3.0, 1, 0.29, 1.0, 0.3, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_one_usable_value
        (5, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_start_tail_bias
        (6, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_start_tail_negative_bias
        (7, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_start_tail_bias_obs_missing
        (8, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1], False),  # test_end_tail_bias
        (9, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1], False),  # test_end_tail_bias_obs_missing
        (10, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_start_tail_noisy
        (11, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1], False),  # test_end_tail_noisy
        (12, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1], False),  # test_two_tails
        (13, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_all_biased
        (14, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_all_noisy
        (15, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_start_tail_bias_with_bgvar
        (16, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1], False),  # test_all_biased_with_bgvar
        (17, 7, 3.0, 3, 2.0, 2, 0.29, 1.0, 0.3, [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_short_start_tail
        (18, 7, 3.0, 3, 2.0, 2, 0.29, 1.0, 0.3, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], False),  # test_short_end_tail
        (19, 7, 3.0, 3, 2.0, 2, 0.29, 1.0, 0.3, [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], False),  # test_short_two_tails
        (20, 7, 9.0, 3, 2.0, 2, 0.29, 1.0, 0.3, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_short_all_fail
        (21, 7, 3.0, 3, 2.0, 2, 0.29, 1.0, 0.3, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_short_start_tail_with_bgvar
        (22, 7, 9.0, 3, 2.0, 2, 0.29, 1.0, 0.3, [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1], False),  # test_short_all_fail_with_bgvar
        (23, 3, 3.0, 1, 1.0, 1, 0.29, 1.0, 0.3, [1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_long_and_short_start_tail
        (24, 3, 3.0, 1, 1.0, 1, 0.29, 1.0, 0.3, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1], False),  # test_long_and_short_end_tail
        (25, 3, 3.0, 1, 1.0, 1, 0.29, 1.0, 0.3, [1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1], False),  # test_long_and_short_two_tails
        (26, 3, 3.0, 1, 1.0, 1, 0.29, 1.0, 0.3, [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], False),  # test_one_long_and_one_short_tail
        (27, 3, 3.0, 3, 0.5, 1, 0.29, 1.0, 0.3, [1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_too_short_for_short_tail
        (28, 3, 3.0, 1, 0.25, 1, 0.29, 1.0, 0.3, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_long_and_short_all_fail
        (
            29,
            3,
            3.0,
            1,
            1.0,
            1,
            0.29,
            1.0,
            0.3,
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_long_and_short_start_tail_with_bgvar
        (
            30,
            3,
            3.0,
            1,
            0.25,
            1,
            0.29,
            1.0,
            0.3,
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_long_and_short_all_fail_with_bgvar
        (31, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_good_data
        (
            32,
            3,
            3.0,
            1,
            1.0,
            1,
            0.29,
            1.0,
            0.3,
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_long_and_short_start_tail_big_bgvar
        (33, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 2.0, [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # test_start_tail_noisy_big_bgvar
        (
            34,
            0,
            3.0,
            1,
            3.0,
            1,
            0.29,
            1.0,
            0.3,
            [untestable for x in range(9)],
            [untestable for x in range(9)],
            True,
        ),  # test_error_bad_input_parameter_tail_check
        (36, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [untestable for x in range(9)], [untestable for x in range(9)], True),  # test_error_invalid_ice_value
        (37, 3, 3.0, 1, 3.0, 1, 0.29, 1.0, 0.3, [untestable for x in range(9)], [untestable for x in range(9)], True),  # test_error_missing_ob_value
        (
            38,
            3,
            3.0,
            1,
            3.0,
            1,
            0.29,
            1.0,
            0.3,
            [untestable for x in range(9)],
            [untestable for x in range(9)],
            True,
        ),  # test_error_not_time_sorted_tail_check
        (
            39,
            3,
            3.0,
            1,
            3.0,
            1,
            0.29,
            1.0,
            0.3,
            [untestable for x in range(9)],
            [untestable for x in range(9)],
            True,
        ),  # test_error_invalid_background
        (
            40,
            3,
            3.0,
            1,
            3.0,
            1,
            0.29,
            1.0,
            0.3,
            [untestable for x in range(9)],
            [untestable for x in range(9)],
            True,
        ),  # test_error_invalid_background_error_variance
        # fmt: on
    ],
)
def test_generic_tailcheck(
    selector,
    long_win_len,
    long_err_std_n,
    short_win_len,
    short_err_std_n,
    short_win_n_bad,
    drif_inter,
    drif_intra,
    background_err_lim,
    expected1,
    expected2,
    warns,
):
    lat, lon, dates, sst, ostia, bgvar, ice = tailcheck_vals(selector)
    # First do the start tail check
    if warns:
        with pytest.warns(UserWarning):
            qc_outcomes = tqc.do_sst_start_tail_check(
                lon,
                lat,
                dates,
                sst,
                ostia,
                ice,
                bgvar,
                long_win_len,
                long_err_std_n,
                short_win_len,
                short_err_std_n,
                short_win_n_bad,
                drif_inter,
                drif_intra,
                background_err_lim,
            )
    else:
        qc_outcomes = tqc.do_sst_start_tail_check(
            lon,
            lat,
            dates,
            sst,
            ostia,
            ice,
            bgvar,
            long_win_len,
            long_err_std_n,
            short_win_len,
            short_err_std_n,
            short_win_n_bad,
            drif_inter,
            drif_intra,
            background_err_lim,
        )
    for i in range(len(qc_outcomes)):
        assert qc_outcomes[i] == expected1[i]

    # Then do the end tail check on the same data
    if warns:
        with pytest.warns(UserWarning):
            qc_outcomes = tqc.do_sst_end_tail_check(
                lon,
                lat,
                dates,
                sst,
                ostia,
                ice,
                bgvar,
                long_win_len,
                long_err_std_n,
                short_win_len,
                short_err_std_n,
                short_win_n_bad,
                drif_inter,
                drif_intra,
                background_err_lim,
            )
    else:
        qc_outcomes = tqc.do_sst_end_tail_check(
            lon,
            lat,
            dates,
            sst,
            ostia,
            ice,
            bgvar,
            long_win_len,
            long_err_std_n,
            short_win_len,
            short_err_std_n,
            short_win_n_bad,
            drif_inter,
            drif_intra,
            background_err_lim,
        )
    for i in range(len(qc_outcomes)):
        assert qc_outcomes[i] == expected2[i]


# tests summary
"""
- NO CHECK MADE
+ alldaytime
+ all OSTIA missing
+ all ice
+ record too short for either check
- LONG-TAIL ONLY
+ start tail bias
+ start tail negative bias
+ start tail bias first few obs missing
+ end tail bias
+ end tail bias last few obs missing
+ start tail noisy
+ end tail noisy
+ two tails
+ all record biased
+ all record noisy
+ background error short circuits start tail
+ background error short circuits all biased
- SHORT-TAIL ONLY
+ start tail
+ end tail
+ two tails
+ all record fail
+ background error short circuits start tail
+ background error short circuits all fail
- LONG-TAIL then SHORT-TAIL
+ long and short start tail
+ long and short end tail
+ long and short two tails
+ one long tail and one short tail
+ too short for short tail
+ long and short combined fail whole record
+ background error short circuits start tail
+ background error short circuits all fail
- NO-TAILS
+ no tails
- EXTRA
+ long and short start tail big bgvar
+ start tail noisy big bgvar
+ assertion error - bad input parameter
+ assertion error - missing matched value
+ assertion error - missing ob value
+ assertion error - invalid ice value
+ assertion error - data not time-sorted
+ assertion error - invalid background sst
+ assertion error - invalid background error
"""


def sst_biased_noisy_check_vals(selector):
    # fmt: off
    # @formatter:off
    # all daytime
    vals1 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 12.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 12.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 12.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 12.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 12.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 12.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 12.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 12.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 12.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0}]

    # all land-masked
    vals2 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0}]

    # all ice
    vals3 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.2},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.2},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.2},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.2},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.2},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.2},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.2},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.2},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.2}]

    # all bgvar exceeds limit
    vals4 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.4, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.4, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.4, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.4, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.4, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.4, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.4, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.4, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.4, 'ICE': 0.0}]

    # biased warm
    vals5 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0}]

    # biased cool
    vals6 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 3.8, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 3.8, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 3.8, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 3.8, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 3.8, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 3.8, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 3.8, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 3.8, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 3.8, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0}]

    # noisy
    vals7 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 7.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 3.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 7.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 3.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 7.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0}]

    # biased and noisy
    vals8 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 9.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 7.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 7.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 9.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 7.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 7.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 9.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0}]

    # biased warm obs missing
    vals9 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
             {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 6.2, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0}]

    # short record one bad
    vals10 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 9.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0}]

    # short record two bad
    vals11 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 9.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 9.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0}]

    # short record two bad obs missing
    vals12 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 9.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 9.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0}]

    # short record two bad obs missing with bgvar masked
    vals13 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.4, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 9.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 9.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0}]

    # good data
    vals14 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0}]

    # short record good data
    vals15 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0}]

    # short record obs missing good data
    vals16 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 0.01, 'ICE': 0.0}]

    # noisy big bgvar
    vals17 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 7.0, 'OSTIA': 5.0, 'BGVAR': 4.0, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 4.0, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 3.0, 'OSTIA': 5.0, 'BGVAR': 4.0, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 4.0, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 7.0, 'OSTIA': 5.0, 'BGVAR': 4.0, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 4.0, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 3.0, 'OSTIA': 5.0, 'BGVAR': 4.0, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 4.0, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 7.0, 'OSTIA': 5.0, 'BGVAR': 4.0, 'ICE': 0.0}]

    # short record two bad obs missing big bgvar
    vals18 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 4.0, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 4.0, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 9.0, 'OSTIA': 5.0, 'BGVAR': 4.0, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 4.0, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 9.0, 'OSTIA': 5.0, 'BGVAR': 4.0, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 4.0, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 4.0, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 4.0, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': None, 'BGVAR': 4.0, 'ICE': 0.0}]

    # good data
    vals19 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0}]

    # assertion error - bad input parameter
    vals20 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0}]

    # assertion error - missing matched value
    vals21 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0}]

    # assertion error - invalid ice value
    vals22 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 1.1},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0}]

    # assertion error - missing observation value
    vals23 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0}]

    # assertion error - times not sorted
    vals24 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0}]

    # assertion error - invalid background sst
    vals25 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 46.0, 'BGVAR': 0.01, 'ICE': 0.0}]

    # assertion error - invalid background error variance
    vals26 = [{'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.0, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.1, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.2, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.3, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': -0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.4, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.5, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.6, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.7, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0},
              {'ID': 'AAAAAAAAA', 'YR': 2003, 'MO': 12, 'DY': 1, 'HR': 0.8, 'LAT': 0.0, 'LON': 0.0, 'SST': 5.0, 'OSTIA': 5.0, 'BGVAR': 0.01, 'ICE': 0.0}]
    # fmt: on
    # @formatter:on
    obs = locals()[f"vals{selector}"]
    reps = {}
    for key in obs[0]:
        reps[key] = []
    reps["DATE"] = []

    for v in obs:
        for key in reps:
            if key != "DATE":
                reps[key].append(v[key])

        hour = int(v["HR"])
        minute = int(60 * (v["HR"] - hour))
        date = datetime(int(v["YR"]), int(v["MO"]), int(v["DY"]), hour, minute)
        reps["DATE"].append(date)

    for key in reps:
        reps[key] = np.array(reps[key])

    if selector == 23:
        reps["LAT"][1] = np.nan
    if selector == 21:
        reps["OSTIA"][:] = np.nan

    return (
        reps["LAT"],
        reps["LON"],
        reps["DATE"],
        reps["SST"],
        reps["OSTIA"],
        reps["BGVAR"],
        reps["ICE"],
    )


@pytest.mark.parametrize(
    # fmt: off
    "selector, n_eval, bias_lim, drif_intra, drif_inter, err_std_n, n_bad, background_err_lim, expected_bias, expected_noisy, expected_short, warns",
    [
        (
            1,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_all_daytime_bnc
        (
            2,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_all_land_masked_bnc
        (
            3,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_all_ice_bnc
        (
            4,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_all_bgvar_exceeds_limit_bnc
        (
            5,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_biased_warm_bnc
        (
            6,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_biased_cool_bnc
        (
            7,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_noisy_bnc
        (
            8,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_biased_and_noisy_bnc
        (
            9,
            5,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_biased_warm_obs_missing_bnc
        (10, 9, 1.10, 1.0, 0.29, 3.0, 2, 0.3, [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], False),  # test_short_record_one_bad_bnc
        (11, 9, 1.10, 1.0, 0.29, 3.0, 2, 0.3, [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], False),  # test_short_record_two_bad_bnc
        (
            12,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            False,
        ),  # test_short_record_two_bad_obs_missing_bnc
        (
            13,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_short_record_two_bad_obs_missing_with_bgvar_bnc
        (
            14,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_good_data_bnc_14
        (15, 9, 1.10, 1.0, 0.29, 3.0, 2, 0.3, [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], False),  # test_short_record_good_data_bnc
        (
            16,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_short_record_obs_missing_good_data_bnc
        (
            17,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            4.0,
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_noisy_big_bgvar_bnc
        (
            18,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            4.0,
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_short_record_two_bad_obs_missing_big_bgvar_bnc
        (
            19,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            False,
        ),  # test_good_data_bnc_19
        (
            20,
            0,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [untestable for _ in range(9)],
            [untestable for _ in range(9)],
            [untestable for _ in range(9)],
            True,
        ),  # test_error_bad_input_parameter_bnc
        # Missing on purpose - test is no longer relevant after refactoring
        (
            22,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [untestable for _ in range(9)],
            [untestable for _ in range(9)],
            [untestable for _ in range(9)],
            True,
        ),  # test_error_invalid_ice_value_bnc
        (
            23,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [untestable for _ in range(9)],
            [untestable for _ in range(9)],
            [untestable for _ in range(9)],
            True,
        ),  # test_error_missing_ob_value_bnc
        (
            24,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [untestable for _ in range(9)],
            [untestable for _ in range(9)],
            [untestable for _ in range(9)],
            True,
        ),  # test_error_not_time_sorted_bnc
        (
            25,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [untestable for _ in range(9)],
            [untestable for _ in range(9)],
            [untestable for _ in range(9)],
            True,
        ),  # test_error_invalid_background_bnc
        (
            26,
            9,
            1.10,
            1.0,
            0.29,
            3.0,
            2,
            0.3,
            [untestable for _ in range(9)],
            [untestable for _ in range(9)],
            [untestable for _ in range(9)],
            True,
        ),  # test_error_invalid_background_error_variance_bnc
        # fmt: on
    ],
)
def test_sst_biased_noisy_check_generic(
    selector,
    n_eval,
    bias_lim,
    drif_intra,
    drif_inter,
    err_std_n,
    n_bad,
    background_err_lim,
    expected_bias,
    expected_noisy,
    expected_short,
    warns,
):
    lat, lon, dates, sst, ostia, bgvar, ice = sst_biased_noisy_check_vals(selector)

    inputs = [
        lon,
        lat,
        dates,
        sst,
        ostia,
        ice,
        bgvar,
        n_eval,
        bias_lim,
        drif_intra,
        drif_inter,
        err_std_n,
        n_bad,
        background_err_lim,
    ]

    if warns:
        with pytest.warns(UserWarning):
            bias_qc_outcomes = tqc.do_sst_biased_check(*inputs)
        with pytest.warns(UserWarning):
            noise_qc_outcomes = tqc.do_sst_noisy_check(*inputs)
        with pytest.warns(UserWarning):
            short_qc_outcomes = tqc.do_sst_biased_noisy_short_check(*inputs)
    else:
        bias_qc_outcomes = tqc.do_sst_biased_check(*inputs)
        noise_qc_outcomes = tqc.do_sst_noisy_check(*inputs)
        short_qc_outcomes = tqc.do_sst_biased_noisy_short_check(*inputs)

    for i in range(len(bias_qc_outcomes)):
        assert bias_qc_outcomes[i] == expected_bias[i]
        assert noise_qc_outcomes[i] == expected_noisy[i]
        assert short_qc_outcomes[i] == expected_short[i]
