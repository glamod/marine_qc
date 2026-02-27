# noqa: D105

from __future__ import annotations
import datetime

import numpy as np
import pandas as pd
import pytest

from marine_qc.auxiliary import (
    convert_to,
    is_scalar_like,
    isvalid,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, False),
        (5.7, True),
        (0.0, True),
        (-3.2, True),
        (np.nan, False),
        (np.float64(3.14), True),
        (np.float64(np.nan), False),
    ],
)
def test_isvalid_scalar(value, expected):
    assert isvalid(value) == expected


def test_isvalid_list():
    value = [1.0, np.nan, None, 4.5]
    result = isvalid(value)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([True, False, False, True]))


def test_isvalid_numpy():
    value = np.array([1.0, np.nan, 2.5])
    result = isvalid(value)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([True, False, True]))


def test_isvalid_object():
    value = np.array([1.0, None, np.nan], dtype=object)
    result = isvalid(value)

    np.testing.assert_array_equal(result, np.array([True, False, False]))


def test_isvalid_series():
    value = pd.Series([1.0, np.nan, None, 5.0])
    result = isvalid(value)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([True, False, False, True]))


def test_isvalid_single_element_list():
    value = [np.nan]
    result = isvalid(value)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([False]))


@pytest.mark.parametrize(
    "value",
    [
        1,
        3.14,
        True,
        False,
        None,
        "hello",
        b"bytes",
        np.int32(5),
        np.float64(3.14),
        np.datetime64("2024-01-01"),
        np.array(5),  # 0-d array
        pd.NA,
        pd.NaT,
        pd.Timestamp("2024-01-01"),
        pd.Timedelta("1D"),
        datetime.date(2024, 1, 1),
        datetime.datetime(2024, 1, 1, 12, 0),
        datetime.time(12, 0),
    ],
)
def test_is_scalar_like_true(value):
    assert is_scalar_like(value) is True


@pytest.mark.parametrize(
    "value",
    [
        [1, 2, 3],
        (1, 2),
        {"a": 1},
        {1, 2, 3},
        np.array([1, 2, 3]),  # 1-d array
        np.array([[1, 2], [3, 4]]),  # 2-d array
        pd.Series([1, 2, 3]),
        pd.DataFrame({"a": [1, 2]}),
    ],
)
def test_is_scalar_like_false(value):
    assert is_scalar_like(value) is False


class RaisesTypeErrorOnNdim:
    def __array__(self):
        """Raise TypeError when attempting to convert to a NumPy array."""
        raise TypeError("Cannot convert to array")


def test_is_scalar_like_invalid():
    obj = RaisesTypeErrorOnNdim()
    assert is_scalar_like(obj) is False


def test_is_scalar_like_zero_dim():
    arr = np.array("hello", dtype=object)
    assert arr.ndim == 0
    assert is_scalar_like(arr) is True


@pytest.mark.parametrize(
    "value, source_unit, target_unit, expected",
    [
        (5.0, "degF", "unknown", -15.0 + 273.15),
        (5.0, "degF", "K", -15.0 + 273.15),
        (5.0, "degC", "K", 5.0 + 273.15),
        (5.0, "degF", "degC", -15.0),
        (-15.0, "degC", "degF", 5.0),
        (1.0, "knots", "kph", 1.852),
    ],
)
def test_convert_to(value, source_unit, target_unit, expected):
    result = convert_to(value, source_unit, target_unit)
    assert pytest.approx(result) == expected
