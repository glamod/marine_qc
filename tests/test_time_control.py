from __future__ import annotations
from datetime import datetime

import numpy as np
import pytest

from marine_qc.helpers.time_control import (
    convert_date_to_hours,
    day_in_year,
    pentad_to_month_day,
    split_date,
    time_difference,
    valid_month_day,
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


class PseudoDatetime:
    def __init__(self, attributes):
        self.attribs = attributes
        for a in attributes:
            if a != "minute" and a != "second":
                self.__setattr__(a, 5.0)
            else:
                self.__setattr__(a, 0.0)

    def __getattr__(self, item):
        """Get attribute by item name."""
        if item in self.attribs:
            return self.__getattr__(item)
        else:
            raise AttributeError


def test_split_date_exceptions():
    for v in ["year", "month", "day", "hour"]:
        list_of_variables = ["year", "month", "day", "hour", "minute", "second"]
        list_of_variables.remove(v)
        psd = PseudoDatetime(list_of_variables)
        result = split_date(psd)
        for key in result:
            if key == v:
                assert np.isnan(result[key])
            else:
                assert result[key] == 5.0


@pytest.mark.parametrize(
    "month, day",
    [
        (1, 32),
        (0, 1),
        (1, 0),
        (2, 30),
        (4, 31),
        (6, 31),
        (9, 31),
        (11, 31),
        (13, 31),
    ],
)
def test_valid_month_day_fails(month, day):
    assert not valid_month_day(month=month, day=day)


def test_valid_month_day_all():
    month_lengths = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for m in range(1, 13):
        for d in range(1, month_lengths[m - 1] + 1):
            assert valid_month_day(month=m, day=d)


def test_pentad_to_month():
    for p in range(1, 74):
        m, d = pentad_to_month_day(p)
        assert p == which_pentad(2001, m, d)


@pytest.mark.parametrize("pentad", [-1, 0, 74])
def test_pentad_to_month_day_raises(pentad):
    with pytest.raises(ValueError):
        pentad_to_month_day(pentad)


@pytest.mark.parametrize(
    "month, day, expected",
    [
        (1, 6, 2),
        (1, 21, 5),
        (12, 26, 72),
        (1, 1, 1),
        (12, 31, 73),
        (3, 1, 12),
    ],
)
def test_which_pentad(month, day, expected):
    assert which_pentad(year=2001, month=month, day=day) == expected


def test_which_pentad_leap():
    assert which_pentad(year=2000, month=2, day=29) == 12


def test_which_pentad_raises_value_error():
    with pytest.raises(ValueError):
        which_pentad(month=13, day=1)
    with pytest.raises(ValueError):
        which_pentad(month=1, day=41)


@pytest.mark.parametrize(
    "year, month, day, expected_error",
    [
        (None, 1, 32, ValueError),
        (None, 2, 30, ValueError),
        (2003, 2, 29, ValueError),
    ],
)
def test_day_in_year_raises(year, month, day, expected_error):
    with pytest.raises(expected_error):
        day_in_year(year, month, day)


def test_day_in_year_leap_year():
    assert day_in_year(year=2000, month=2, day=29) == day_in_year(year=2001, month=3, day=1)

    # Just test all days
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    count = 1
    for month in range(1, 13):
        for day in range(1, month_lengths[month - 1] + 1):
            assert day_in_year(year=2001, month=month, day=day) == count
            count += 1


def test_day_in_year_leap_year_test():
    with pytest.raises(ValueError):
        day_in_year(month=13, day=1)
    with pytest.raises(ValueError):
        day_in_year(month=0, day=1)
    with pytest.raises(ValueError):
        day_in_year(month=2, day=30)
    with pytest.raises(ValueError):
        day_in_year(month=2, day=0)


@pytest.mark.parametrize(
    "year, month, day",
    [
        (2005, 1, 0),
        (2005, 0, 1),
        (2005, 1, 32),
        (2005, 2, 29),
        (2004, 2, 30),
        (2005, 4, 31),
    ],
)
def test_dayinyear_raises(year, month, day):
    with pytest.raises(ValueError):
        day_in_year(year, month, day)


def test_dayinyear_all():
    # First lets do non-leap years
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    count = 1
    for m in range(1, 13):
        for d in range(1, month_lengths[m - 1] + 1):
            assert day_in_year(2005, m, d) == count
            count += 1

    # First lets do non-leap years
    month_lengths = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    count = 1
    for m in range(1, 13):
        for d in range(1, month_lengths[m - 1] + 1):
            assert day_in_year(2004, m, d) == count
            count += 1


def test_dayinyear_array_all():
    # First lets do non-leap years
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    months = []
    days = []
    day_number = []

    count = 1
    for m in range(1, 13):
        for d in range(1, month_lengths[m - 1] + 1):
            months.append(m)
            days.append(d)
            day_number.append(count)

            count += 1

    months = np.array(months)
    days = np.array(days)
    day_number = np.array(day_number)

    result = day_in_year(year=2001, month=months, day=days)

    assert np.all(result == day_number)


def test_which_pentad_array_all():
    # First lets do non-leap years
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    months = []
    days = []
    day_number = []

    count = 1
    for m in range(1, 13):
        for d in range(1, month_lengths[m - 1] + 1):
            months.append(m)
            days.append(d)
            day_number.append(count)

            count += 1

    months = np.array(months)
    days = np.array(days)
    pentad_number = ((np.array(day_number) - 1) / 5).astype(int) + 1

    result = which_pentad(year=2001, month=months, day=days)

    print(result)
    print(pentad_number)

    assert np.all(result == pentad_number)


@pytest.mark.parametrize(
    "dates, expected",
    [
        ([datetime(2000, 1, 1, 0, 0), datetime(2000, 1, 1, 1, 0)], [0.0, 1.0]),
        ([datetime(1999, 12, 31, 23, 0), datetime(2000, 1, 1, 1, 0)], [0.0, 2.0]),
    ],
)
def test_convert_date_to_hour(dates, expected):
    assert convert_date_to_hours(dates) == expected


@pytest.mark.parametrize(
    "time1, time2, expected",
    [
        ("2003-01-01 01:00", "2003-01-01 01:01", 1.0 / 60.0),
        ("2003-01-01 01:00", "2003-01-01 01:30", 0.5),
        ("2003-01-01 01:00", "2003-01-01 02:00", 1.0),
        ("2003-01-01 01:00", "2003-13-01 02:00", np.nan),
        ("2003-01-01 01:00", "2003-01-01 25:00", np.nan),
        ("2003-01-01 01:00", "2003-01-32 02:00", np.nan),
        ("2003-01--01 01:00", "2003-01-30 02:00", np.nan),
    ],
)
def test_time_difference(time1, time2, expected):
    result = time_difference(time1, time2)

    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == expected
