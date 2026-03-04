from __future__ import annotations
import re

import numpy as np
import pandas as pd
import pytest
from pint.errors import DimensionalityError

from marine_qc.auxiliary import (
    convert_to,
    convert_units,
    ensure_arrays,
    format_return_type,
    inspect_arrays,
    is_scalar_like,
    post_format_return_type,
)
from marine_qc.time_control import convert_date


@convert_units(value="K")
def _convert_function(value):
    return value


@convert_units(value2="K")
def _convert_function2(value):
    return value


@inspect_arrays(["value1", "value2"])
def _array_function(value1, value2):
    return value1, value2


@inspect_arrays(["value1", "value3"])
def _array_function2(value1, value2):
    return value1, value2


@convert_date(["year", "month", "day"])
def _date_function(date, year=None, month=None, day=None):
    return year, month, day


@convert_date(["year2", "month", "day"])
def _date_function2(date, year=None, month=None, day=None):
    return year, month, day


@post_format_return_type(["value"])
def _format_function(value):
    return pd.Series(value) + 5


@pytest.mark.parametrize("units", [{"value": "degC"}, "degC"])
def test_convert_units(units):
    result = _convert_function(30.0, units=units)
    assert result == 30.0 + 273.15


def test_convert_units_no_conversion():
    result = _convert_function(30.0, units={"value2": "degC"})
    assert result == 30.0


def test_convert_units_raise():
    with pytest.raises(DimensionalityError):
        _convert_function(30.0, units="hPa")


def test_convert_units_valueerror():
    with pytest.raises(ValueError, match="Parameter 'value2' not found in function arguments."):
        _convert_function2(30.0, units={"value2": "degC"})


@pytest.mark.parametrize(
    "value1, value2",
    [
        [2, 3],
        [[5, 8, 9], [7, 2, 3]],
        [pd.Series([4, 6, 9]), [8, 6, 1]],
        [np.array([8, 9, 3]), pd.Series([9, 8, 3])],
        [np.array([8, 6, 9]), [8, 3, 7]],
    ],
)
def test_inspect_arrays(value1, value2):
    result1, result2 = _array_function(value1, value2)
    expected1 = np.atleast_1d(value1)
    expected2 = np.atleast_1d(value2)
    np.testing.assert_equal(result1, expected1)
    np.testing.assert_equal(result2, expected2)


def test_inspect_arrays_raise_dimension():
    with pytest.raises(ValueError, match="Input 'value1' must be one-dimensional."):
        _array_function(np.ndarray(shape=(2, 2), dtype=float, order="F"), [1, 2, 3])


def test_inspect_arrays_raise_length():
    error_msg = "Input ['value1', 'value2'] must all have the same length."
    escaped_msg = re.escape(error_msg)
    with pytest.raises(ValueError, match=escaped_msg):
        _array_function([1, 2, 3, 4], [1, 2, 3])


def test_inspect_arrays_raise_parameter():
    with pytest.raises(ValueError, match="Parameter 'value3' is not a valid parameter."):
        _array_function2(1, 2)


@pytest.mark.parametrize(
    "date, year, month, day",
    [["2019-9-27", 2019, 9, 27], ["2019-9", 2019, 9, 1], ["2019", 2019, 1, 1]],
)
def test_convert_date(date, year, month, day):
    yy, mm, dd = _date_function(pd.to_datetime(date))
    assert yy == year
    assert mm == month
    assert dd == day


def test_convert_date_raise():
    with pytest.raises(ValueError, match="Parameter 'year2' is not a valid parameter."):
        _date_function2(pd.to_datetime("2019-09-27"))


@pytest.mark.parametrize(
    "value, expected, array_type",
    [
        [[1, 2, 3, 4], [6, 7, 8, 9], "list"],
        [pd.Series([1, 2, 3, 4]), pd.Series([6, 7, 8, 9]), "series"],
        [np.array([1, 2, 3, 4]), np.array([6, 7, 8, 9]), "numpy"],
        [1, 6, "scalar"],
    ],
)
def test_post_format_return_type(value, expected, array_type):
    result = _format_function(value)
    if array_type == "list":
        np.testing.assert_equal(result, expected)
    elif array_type == "numpy":
        np.testing.assert_equal(result, expected)
    elif array_type == "series":
        pd.testing.assert_series_equal(result, expected)
    elif array_type == "scalar":
        assert result == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.0, True),
        (0, True),
        (True, True),
        ([0.0], False),
        (np.array(5), True),
        (np.array([5, 6]), False),
        ("a", True),
    ],
)
def test_is_scalar_like(value, expected):
    assert is_scalar_like(value) == expected


def test_ensure_arrays_passes():
    arr_orig = np.array([1, 2, 3])
    (arr,) = ensure_arrays(arr=arr_orig)
    np.testing.assert_array_equal(arr, arr_orig)


def test_ensure_arrays_raises():
    with pytest.raises(TypeError):
        ensure_arrays(arr=1)


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


@pytest.mark.parametrize(
    "value, expected, array_type",
    [
        [np.array([1, 2, 3, 4]), [1, 2, 3, 4], "list"],
        [np.array([1, 2, 3, 4]), pd.Series([1, 2, 3, 4]), "series"],
        [np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), "numpy"],
    ],
)
def test_format_return_type(value, expected, array_type):
    result = format_return_type(value, expected)
    if array_type == "list":
        np.testing.assert_equal(result, expected)
    elif array_type == "numpy":
        np.testing.assert_equal(result, expected)
    elif array_type == "series":
        pd.testing.assert_series_equal(result, expected)
