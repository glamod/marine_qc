from __future__ import annotations

from datetime import datetime

import pytest  # noqa

from marine_qc.time_control import (
    convert_date_to_hours,
    day_in_year,
    leap_year_correction,
    pentad_to_month_day,
    split_date,
    which_pentad,
)


@pytest.mark.parametrize(
    "date, expected_year, expected_month, expected_day, expected_hour",
    [
        (datetime(2002, 3, 27, 17, 30), 2002, 3, 27, 17.5),
    ],
)
def test_split_date(date, expected_year, expected_month, expected_day, expected_hour):
    result = split_date(date)
    expected = {
        "year": expected_year,
        "month": expected_month,
        "day": expected_day,
        "hour": expected_hour,
    }
    for key in expected:
        assert result[key] == expected[key]


def test_pentad_to_mont():
    for p in range(1, 74):
        m, d = pentad_to_month_day(p)
        assert p == which_pentad(m, d)


@pytest.mark.parametrize(
    "month, day, expected",
    [
        (1, 6, 2),
        (1, 21, 5),
        (12, 26, 72),
        (1, 1, 1),
        (12, 31, 73),
        (2, 29, 12),
        (3, 1, 12),
    ],
)
def test_which_pentad(month, day, expected):
    assert which_pentad(month, day) == expected


def test_which_pentad_raises_value_error():
    with pytest.raises(ValueError):
        which_pentad(13, 1)
    with pytest.raises(ValueError):
        which_pentad(1, 41)


def test_day_in_year_leap_year():
    assert day_in_year(2, 29) == day_in_year(3, 1)

    # Just test all days
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    count = 1
    for month in range(1, 13):
        for day in range(1, month_lengths[month - 1] + 1):
            assert day_in_year(month, day) == count
            count += 1


def test_day_in_year_leap_year_test():
    with pytest.raises(ValueError):
        day_in_year(13, 1)
    with pytest.raises(ValueError):
        day_in_year(0, 1)
    with pytest.raises(ValueError):
        day_in_year(2, 30)
    with pytest.raises(ValueError):
        day_in_year(2, 0)


def test_leap_year_correction():
    assert leap_year_correction(24, 1, 0) == 0
    assert leap_year_correction(24, 1, 4) == 1461
    assert leap_year_correction(24, 1, -3) == -1096


@pytest.mark.parametrize(
    "dates, expected",
    [
        ([datetime(2000, 1, 1, 0, 0), datetime(2000, 1, 1, 1, 0)], [0, 1]),
        ([datetime(1999, 12, 31, 23, 0), datetime(2000, 1, 1, 1, 0)], [0, 2]),
    ],
)
def test_convert_date_to_hour(dates, expected):
    assert (convert_date_to_hours(dates) == expected).all()
