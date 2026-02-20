from __future__ import annotations
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Literal, get_type_hints

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from marine_qc import (
    do_hard_limit_check,
)
from marine_qc.validations import (
    is_func_param,
    is_in_data,
    validate_arg,
    validate_args,
    validate_dict,
    validate_type,
)


def sample_func_long(
    a: int,
    b: list[int],
    c: dict[str, int],
    d: tuple[int, str],
    e: tuple[int, ...],
    f: int | None,
    g: Literal["x", "y"],
    h: int | str,
    i: list[int | str],
    j: Annotated[int, "positive"],
    k: Callable[[int], str],
    m: set[float],
    n: frozenset[int],
    o: Sequence[int],
    p: Mapping[str, list[int]],
    q: dict[str, list[tuple[int, float | None]]],
    r: pd.DataFrame,
    s: pd.Series,
    t: np.ndarray,
    u: npt.NDArray[np.int64],
):
    return a, b, c, d, e, f, g, h, i, j, k, m, n, o, p, q, r, s, t, u


def sample_func_short(a: int):
    return a


def sample_func_kwargs(a: int, **kwargs):
    return {"a": a, **kwargs}


@pytest.fixture
def series_ind():
    return pd.Series([1, 2, 3, 4], name="value")


@pytest.fixture
def df_ind():
    return pd.DataFrame({"value1": [1, 2, 3, 4], "value2": [1, 1, 2, 2]})


PARAMETERS = inspect.signature(sample_func_long).parameters
TYPE_HINTS = get_type_hints(sample_func_long)


def test_is_func_param():
    assert not is_func_param(sample_func_short, "invalid_param")
    assert is_func_param(sample_func_short, "a")


def test_is_in_data_series(series_ind):
    assert is_in_data("value", series_ind)
    assert not is_in_data("value2", series_ind)


def test_is_in_data_df(df_ind):
    assert is_in_data("value1", df_ind)
    assert is_in_data("value2", df_ind)
    assert not is_in_data("value3", df_ind)


def test_is_in_data_raises():
    with pytest.raises(TypeError, match="Unsupported data type"):
        is_in_data("test_name", [1, 2, 3])


def test_validate_dict_passing():
    validate_dict({"test": {"value": 1}})


@pytest.mark.parametrize(
    "input_value",
    [
        1,
        1.0,
        "1",
        [1, 2],
        (1, 2),
        pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
        pd.Series([1, 2]),
    ],
)
def test_validate_dict_invalid_input(input_value):
    with pytest.raises(TypeError, match="must be a dictionary"):
        validate_dict(input_value)


@pytest.mark.parametrize(
    "input_dict",
    [
        {1: "test"},
        {1.0: "test"},
    ],
)
def test_validate_dict_invalid_keys(input_dict):
    with pytest.raises(TypeError, match="must be a string"):
        validate_dict(input_dict)


@pytest.mark.parametrize(
    "input_dict",
    [
        {"test": 1},
        {"test": 1.0},
        {"test": "1"},
        {"test": [1, 2]},
        {"test": (1, 2)},
        {"test": pd.DataFrame({"A": [1, 2], "B": [3, 4]})},
        {"test": pd.Series([1, 2])},
    ],
)
def test_validate_dict_invalid_values(input_dict):
    with pytest.raises(TypeError, match="must be a dictionary"):
        validate_dict(input_dict)


@pytest.mark.parametrize(
    "value,expected",
    [
        (123, Any),
        (5, int),
        (10, Annotated[int, "meta"]),
        ("x", Literal["x", "y"]),
        (5, int | str),
        ("hello", int | str),
        (lambda x: x, Callable[[int], int]),
        ({"a": 1}, dict[str, int]),
        ([1, 2, 3], list[int]),
        ({1.0, 2.0}, set[float]),
        (frozenset({1, 2}), frozenset[int]),
        ((1, "x"), tuple[int, str]),
        ((1, 2, 3), tuple[int, ...]),
        ([1, 2, 3], Sequence[int]),
        (np.array([1, 2, 3]), np.ndarray),
        (np.array([1, 2, 3], dtype=np.int64), npt.NDArray[np.int64]),
        (pd.DataFrame({"a": [1, 2]}), pd.DataFrame),
        (pd.Series([1, 2]), pd.Series),
    ],
)
def test_validate_type_valid(value, expected):
    assert validate_type(value, expected)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("5", int),
        ("z", Literal["x", "y"]),
        (5.5, int | str),
        ({1: "a"}, dict[str, int]),
        ({"a": "b"}, dict[str, int]),
        ([1, "x"], list[int]),
        ((1,), tuple[int, str]),
        ((1, 2), tuple[int, str]),
        ([1, "x"], Sequence[int]),
        (np.array([1.0, 2.0]), npt.NDArray[np.int64]),
        (pd.Series([1, 2]), pd.DataFrame),
    ],
)
def test_validate_type_invalid(value, expected):
    assert not validate_type(value, expected)


def test_validate_type_origin_none_not_type():
    assert not validate_type(5, 123)


def test_validate_type_ndarray_any_dtype():
    arr = np.array([1.0, 2.0])
    assert validate_type(arr, npt.NDArray[Any])


def test_validate_type_callable_invalid():
    assert not validate_type(123, Callable[[int], int])


def test_validate_type_sequence_rejects_string():
    assert not validate_type("abc", Sequence[int])


@pytest.mark.parametrize(
    "key,value",
    [
        ("a", 5),
        ("b", [1, 2, 3]),
        ("c", {"x": 1, "y": 2}),
        ("d", (1, "hello")),
        ("e", (1, 2, 3)),
        ("f", None),
        ("f", 10),
        ("g", "x"),
        ("h", 42),
        ("h", "hello"),
        ("i", [1, "a", 3]),
        ("j", 99),
        ("k", lambda x: str(x)),
        ("m", {1.0, 2.5}),
        ("n", frozenset({1, 2, 3})),
        ("o", [1, 2, 3]),
        ("p", {"a": [1, 2], "b": [3]}),
        ("q", {"a": [(1, 2.0), (3, None)]}),
        ("r", pd.DataFrame({"a": [1, 2, 3]})),
        ("s", pd.Series([1, 2, 3])),
        ("t", np.array([1, 2, 3])),
        ("u", np.array([1, 2, 3], dtype=np.int64)),
    ],
)
def test_validate_arg_valid(key, value):
    validate_arg(
        key=key,
        value=value,
        func_name="sample_func_long",
        parameters=PARAMETERS,
        type_hints=TYPE_HINTS,
        reserved_keys=set(),
        has_arguments=False,
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("a", "not int"),
        ("b", [1, "wrong"]),
        ("c", {"x": "wrong"}),
        ("d", (1, 2)),
        ("e", ("wrong",)),
        ("f", "not optional int"),
        ("g", "z"),
        ("h", 3.14),
        ("i", [1, 2.0]),
        ("g", "z"),
        ("j", "not int"),
        ("k", "not callable"),
        ("m", {1, 2}),
        ("n", {1, 2}),
        ("o", ["x", "y"]),
        ("p", {"a": ["wrong"]}),
        ("q", {"a": [(1, "wrong")]}),
        ("d", (1, 2)),
        ("e", ("a",)),
        ("r", {"a": [1, 2, 3]}),
        ("s", [1, 2, 3]),
        ("t", [1, 2, 3]),
        ("u", np.array([1, 2, 3], dtype=float)),
    ],
)
def test_validate_arg_invalid_type(key, value):
    with pytest.raises(TypeError):
        validate_arg(
            key=key,
            value=value,
            func_name="sample_func_long",
            parameters=PARAMETERS,
            type_hints=TYPE_HINTS,
            reserved_keys=set(),
            has_arguments=False,
        )


def test_validate_arg_invalid_parameter_name():
    with pytest.raises(ValueError):
        validate_arg(
            key="unknown",
            value=123,
            func_name="sample_func_long",
            parameters=PARAMETERS,
            type_hints=TYPE_HINTS,
            reserved_keys=set(),
            has_arguments=False,
        )


def test_reserved_key_skips_validation():
    validate_arg(
        key="reserved",
        value="anything",
        func_name="sample_func_long",
        parameters=PARAMETERS,
        type_hints=TYPE_HINTS,
        reserved_keys={"reserved"},
        has_arguments=False,
    )


def test_has_arguments_skips_validation():
    validate_arg(
        key="unknown",
        value="anything",
        func_name="sample_func_long",
        parameters=PARAMETERS,
        type_hints=TYPE_HINTS,
        reserved_keys=set(),
        has_arguments=True,
    )


def test_validate_args_passing_required_only():
    validate_args(sample_func_short, kwargs={"a": 2})


def test_validate_args_passing_with_extra_kwargs():
    validate_args(sample_func_kwargs, kwargs={"a": 2, "extra": 123})


def test_validate_args_passing_with_args_and_kwargs():
    validate_args(sample_func_kwargs, args=2, kwargs={"extra": 123})


def test_validate_args_invalid_param():
    with pytest.raises(ValueError, match="is not a valid parameter of function"):
        validate_args(sample_func_short, kwargs={"a": 2, "extra": 123})


def test_validate_args_missing_required_param():
    with pytest.raises(TypeError, match="is missing for function"):
        validate_args(sample_func_kwargs, kwargs={"extra": 123})


def test_validate_args_too_many_args():
    with pytest.raises(TypeError, match="Too many positional arguments for function"):
        validate_args(sample_func_short, args=(1, 2))


def test_validate_args_qc_function():
    validate_args(do_hard_limit_check, args=([1, 2, 3, 4, 5],), kwargs={"limits": [5, 6]})


def test_validate_args_passing_args_only():
    validate_args(
        sample_func_long,
        args=(
            5,
            [1, 2, 3],
            {"x": 1, "y": 2},
            (1, "hello"),
            (1, 2, 3),
            10,
            "x",
            42,
            [1, "a", 3],
            99,
            lambda x: str(x),
            {1.0, 2.5},
            frozenset({1, 2, 3}),
            [1, 2, 3],
            {"a": [1, 2], "b": [3]},
            {"a": [(1, 2.0), (3, None)]},
            pd.DataFrame({"a": [1, 2, 3]}),
            pd.Series([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3], dtype=np.int64),
        ),
    )


def test_validate_args_passing_kwargs_only():
    validate_args(
        sample_func_long,
        kwargs={
            "a": 5,
            "b": [1, 2, 3],
            "c": {"x": 1, "y": 2},
            "d": (1, "hello"),
            "e": (1, 2, 3),
            "f": 10,
            "g": "x",
            "h": 42,
            "i": [1, "a", 3],
            "j": 99,
            "k": lambda x: str(x),
            "m": {1.0, 2.5},
            "n": frozenset({1, 2, 3}),
            "o": [1, 2, 3],
            "p": {"a": [1, 2], "b": [3]},
            "q": {"a": [(1, 2.0), (3, None)]},
            "r": pd.DataFrame({"a": [1, 2, 3]}),
            "s": pd.Series([1, 2, 3]),
            "t": np.array([1, 2, 3]),
            "u": np.array([1, 2, 3], dtype=np.int64),
        },
    )
