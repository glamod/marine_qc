"""Auxiliary functions for QC."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from datetime import datetime
from functools import wraps
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from xclim.core.units import convert_units_to, units

passed = 0
failed = 1
untestable = 2
untested = 3

PandasNAType: TypeAlias = type(pd.NA)
PandasNaTType: TypeAlias = type(pd.NaT)

# --- Scalars ---
ScalarIntType: TypeAlias = int | np.integer | PandasNAType | None
ScalarFloatType: TypeAlias = float | np.floating | PandasNAType | None
ScalarDatetimeType: TypeAlias = (
    datetime | np.datetime64 | pd.Timestamp | PandasNaTType | None
)

# --- Sequences ---
SequenceIntType: TypeAlias = (
    Sequence[ScalarIntType]
    | npt.NDArray[np.integer]
    | pd.Series  # optionally: pd.Series[np.integer] or pd.Series[pd.Int64Dtype]
)

SequenceFloatType: TypeAlias = (
    Sequence[ScalarFloatType]
    | npt.NDArray[np.floating]
    | pd.Series  # optionally: pd.Series[np.floating] or pd.Series[pd.Float64Dtype]
)

SequenceDatetimeType: TypeAlias = (
    Sequence[ScalarDatetimeType]
    | npt.NDArray[np.datetime64]
    | pd.Series  # optionally: pd.Series[pd.DatetimeTZDtype] or similar
)

# --- Value Types (Scalar or Sequence) ---
ValueFloatType: TypeAlias = ScalarFloatType | SequenceFloatType
ValueIntType: TypeAlias = ScalarIntType | SequenceIntType
ValueDatetimeType: TypeAlias = ScalarDatetimeType | SequenceDatetimeType


def is_scalar_like(x: Any) -> bool:
    """
    Return True if the input is scalar-like (i.e., has no dimensions).

    A scalar-like value includes:
    - Python scalars: int, float, bool, None
    - NumPy scalars: np.int32, np.float64, np.datetime64, etc.
    - Zero-dimensional NumPy arrays: np.array(5)
    - Pandas scalars: pd.Timestamp, pd.Timedelta, pd.NA, pd.NaT
    - Strings and bytes (unless excluded)

    Parameters
    ----------
        x (Any): The value to check.

    Returns
    -------
        bool: True if `x` is scalar-like, False otherwise.
    """
    try:
        return np.ndim(x) == 0
    except TypeError:
        return True  # fallback: built-in scalars like int, float, pd.Timestamp


def isvalid(inval: ValueFloatType) -> bool | np.ndarray:
    """Check if a value(s) are numerically valid (not None or NaN).

    Parameters
    ----------
    inval : float, None, array-like of float or None
        Input value(s) to be tested

    Returns
    -------
    bool or np.ndarray of bool
        Returns False where the input is None or NaN, True otherwise.
        Returns a boolean scalar if input is scalar, else a boolean array.
    """
    result = np.logical_not(pd.isna(inval))
    if np.isscalar(inval):
        return bool(result)
    return result


def format_return_type(result_array: np.ndarray, *input_values: Any) -> Any:
    """
    Convert the result numpy array(s) to the same type as the input `value`.

    If result_array is a sequence of arrays, format each element recursively,
    preserving the container type.

    Parameters
    ----------
    result_array : np.ndarray
        The numpy array of results.
    input_values : scalar, sequence, np.ndarray, pd.Series or None
        One or more original input values to infer the desired return type from.

    Returns
    -------
    Same type as input(s)
        The result formatted to match the type of the first valid input value.
    """
    input_value = next((val for val in input_values if val is not None), None)

    if input_value is None or is_scalar_like(input_value):
        if hasattr(result_array, "ndim") and result_array.ndim > 0:
            result_array = result_array[0]
        return int(result_array)
    if isinstance(input_value, pd.Series):
        return pd.Series(result_array, index=input_value.index, dtype=int)
    if isinstance(input_value, (list, tuple)):
        return type(input_value)(result_array.tolist())
    if isinstance(input_value, np.ndarray) and isinstance(result_array, pd.Series):
        return result_array.to_numpy()
    return result_array  # np.ndarray or fallback


def convert_to(
    value: float | None | Sequence[float | None], source_units: str, target_units: str
):
    """Convert a float with source units into target units.

    Parameters
    ----------
    value: float or None or array-like of float or None
        A single float value, None, or a sequence (e.g., list, tuple, array-like)
        containing floats and/or None values. `None` values will be ignored or passed
        through unchanged.
    source_units: str
        The units of the input value(s) (e.g., 'degC', 'm/s').
    target_units: str
        The units to convert to (e.g., 'K', 'km/h').
        If set to "unknown", the value(s) will be converted to the base SI units
        of the source_units (e.g., 'degC' ? 'kelvin', 'km' ? 'meter').

    Returns
    -------
    float or None or array-like of float or None
        Converted value(s) with the same structure as input,
        where floats are converted and None values are preserved.
    """

    def _convert_to(value):
        if not isvalid(value):
            return value
        return convert_units_to(value * registry, target_units)

    registry = units(source_units)
    if target_units == "unknown":
        target_units = registry.to_base_units()

    if isinstance(value, np.ndarray):
        return np.array([_convert_to(v) for v in value])
    if isinstance(value, Sequence):
        return type(value)(_convert_to(v) for v in value)
    return _convert_to(value)


def generic_decorator(
    pre_handler: Callable[[dict], None] | None = None,
    post_handler: Callable[[any, dict], any] | None = None,
) -> Callable:
    """
    Creates a decorator that binds function arguments, allows inspection or modification
    of those arguments via a custom handler function, and then calls the original function.

    This base decorator manages argument binding and supports passing additional reserved
    keyword arguments to the handler through the decorated function's kwargs. Pre-handlers are
    applied before the function is called and post-handlers afterwards.

    Parameters
    ----------
    pre_handler : Callable[[dict], None]
        A function that takes a dictionary of bound arguments (`bound_args.arguments`)
        and optionally other keyword arguments, to inspect, mutate, or validate these
        arguments before the decorated function executes.
        The handler should accept the signature:
        `handler(arguments: dict, **meta_kwargs) -> None`
    post_handler : Callable[[dict], None]
        A function that takes a dictionary of bound arguments (`bound_args.arguments`)
        and optionally other keyword arguments, to inspect, mutate, or validate these
        arguments before the decorated function executes.
        The handler should accept the signature:
        `handler(arguments: dict, **meta_kwargs) -> None`

    Returns
    -------
    Callable
        A decorator that can be applied to any function. The decorated function will
        have its arguments bound and passed to the handler before execution.

    Notes
    -----
    - The handler can specify a `_decorator_kwargs` attribute (a set of reserved keyword
      argument names). These reserved kwargs will be extracted from the decorated function's
      call kwargs and passed to the handler, then removed before calling the original function.
    - The original function is called with the possibly modified bound arguments after
      handler processing.
    """
    if pre_handler:
        pre_handler._is_post_handler = False
    if post_handler:
        post_handler._is_post_handler = True

    def decorator(func):
        handlers = []
        if pre_handler:
            handlers.append(pre_handler)
        if post_handler:
            handlers.append(post_handler)

        @wraps(func)
        def wrapper(*args, **kwargs):
            reserved_keys = set()
            all_pre_handlers = []
            all_post_handlers = []
            current_func = wrapper
            visited = set()

            while (
                hasattr(current_func, "__wrapped__") and id(current_func) not in visited
            ):
                visited.add(id(current_func))
                for handler in getattr(current_func, "_decorator_handlers", []):
                    if not callable(handler):
                        continue
                    if hasattr(handler, "_decorator_kwargs"):
                        reserved_keys.update(handler._decorator_kwargs)
                    if getattr(handler, "_is_post_handler", False):
                        all_post_handlers.append(handler)
                    else:
                        all_pre_handlers.append(handler)

                current_func = current_func.__wrapped__

            sig = inspect.signature(func)
            meta_kwargs = {
                k: kwargs.pop(k) if k not in sig.parameters else kwargs[k]
                for k in reserved_keys
                if k in kwargs
            }

            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            original_call = bound_args.arguments.copy()

            for handler in reversed(all_pre_handlers):
                handler.__funcname__ = func.__name__
                handler(bound_args.arguments, **meta_kwargs)

            result = func(*bound_args.args, **bound_args.kwargs)

            for handler in reversed(all_post_handlers):
                handler.__funcname__ = func.__name__
                result = handler(result, bound_args.arguments, **original_call)

            return result

        setattr(wrapper, "_decorator_handlers", handlers)

        return wrapper

    return decorator


def post_format_return_type(params: list[str]) -> Callable:
    """
    Decorator to format a function's return value to match the type of its original input(s).

    This decorator ensures that the output of the decorated function is converted back
    to the same structure/type as the original input(s) specified by `params`.
    It uses a context object (`_ctx`) if available to retrieve the original inputs
    before any preprocessing was applied. If no context is found, it falls back to
    the current bound arguments.

    Parameters
    ----------
    params : list of str
        List of parameter names whose original input types should be used to
        format the return value.

    Returns
    -------
    Callable
        A decorator that modifies the decorated function's output to match the
        input types.

    Notes
    -----
    - Assumes a `TypeContext` object may be passed via `_ctx` keyword argument,
      storing original input values for accurate type formatting.
    - Falls back gracefully if no context is available, using current arguments.
    - Useful when function inputs are preprocessed (e.g., converted to arrays),
      and the output should match the original input types.
    """

    def post_handler(result, arguments: dict, **original_call):

        input_values = []
        for param in params:
            if param in original_call:
                input_values.append(original_call[param])
                continue
        return format_return_type(result, *input_values)

    return generic_decorator(post_handler=post_handler)


def inspect_arrays(params: list[str], sortby: str | None = None) -> Callable:
    """
    Create a decorator that inspects specified input parameters of a function,
    converts them to one-dimensional NumPy arrays, and validates their lengths.

    This decorator is useful to enforce that certain input arguments are sequence-like,
    convert them to NumPy arrays for consistent processing, and ensure they all have
    the same length.

    Parameters
    ----------
    params : list of str
        A list of parameter names to inspect in the decorated function's arguments.
        Each named parameter will be converted to a NumPy array and validated.
    sortby : str, optional
        The name of the parameter to sort by.

    Returns
    -------
    Callable
        A decorator function that can be applied to other functions. When applied,
        the specified parameters will be converted to 1D NumPy arrays, validated
        to ensure they exist and have matching lengths, and then passed to the
        decorated function.

    Raises
    ------
    ValueError
        If any specified parameter name is not found in the decorated function's
        arguments.
        If any of the specified parameters is not one-dimensional.
        If the lengths of the specified arrays do not all match.

    Examples
    --------
    >>> @inspect_arrays(["a", "b"])
    ... def add_arrays(a, b):
    ...     return a + b
    ...

    >>> add_arrays([1, 2, 3], [4, 5, 6])
    array([5, 7, 9])

    >>> add_arrays([1, 2], [3, 4, 5])
    Traceback (most recent call last):
        ...
    ValueError: Input ['a', 'b'] must all have the same length.
    """

    def pre_handler(arguments: dict, **meta_kwargs):
        arrays = []
        for param in params:
            if param not in arguments:
                raise ValueError(f"Parameter '{param}' is not a valid parameter.")

            value = arguments[param]
            arr = np.atleast_1d(arguments[param])
            if arr.ndim != 1:
                raise ValueError(f"Input '{param}' must be one-dimensional.")

            arguments[param] = arr
            if value is not None:
                arrays.append(arr)

        lengths = [len(arr) for arr in arrays]
        if any(length != lengths[0] for length in lengths):
            raise ValueError(f"Input {params} must all have the same length.")

        if sortby:
            unsorted_array = arguments[sortby]
            indices = np.argsort(unsorted_array)
            for param in params:
                arguments[param] = arguments[param][indices]

    def post_handler(result, arguments: dict, **original_call):
        if sortby is None:
            return result
        sort_indices = np.argsort(original_call[sortby])
        inverse_indices = np.argsort(sort_indices)
        return result[inverse_indices]

    return generic_decorator(pre_handler=pre_handler, post_handler=post_handler)


def convert_units(**units_by_name) -> Callable:
    """
    Create a decorator to automatically convert units of specified function parameters.

    This decorator uses a `converter_dict` keyword argument to specify which parameters
    should be converted, along with their source and target units. Before the decorated
    function runs, the specified parameters are converted using the `convert_to` function.

    Parameters
    ----------
    None

    Returns
    -------
    Callable
        A decorator function that, when applied to another function, automatically
        converts the units of specified parameters according to the provided
        `converter_dict`.

    Notes
    -----
    - The decorator expects the decorated function to be called with a keyword argument
      `converter_dict`, which is a dictionary mapping parameter names (str) to tuples
      of `(source_units: str, target_units: str)`.
    - Parameters not present in the function arguments or with a value of None are
      skipped.
    - Uses the `convert_to` function to perform the actual unit conversion.

    Examples
    --------
    >>> @convert_units()
    ... def temperature_in_K(temp):
    ...     print(f"Temperature in Kelvin: {temp}")
    ...

    >>> temperature_in_K(25.0, converter_dict={"temp": ("degC", "K")})
    Temperature in Kelvin: 298.15
    """

    def pre_handler(arguments: dict, **meta_kwargs):
        units_dict = meta_kwargs.get("units")
        if units_dict is None:
            return
        if isinstance(units_dict, str):
            units_str = units_dict
            units_dict = {param: units_str for param in arguments}

        for param, target_units in units_by_name.items():
            if param not in arguments:
                raise ValueError(
                    f"Parameter '{param}' not found in function arguments."
                )
            if param not in units_dict:
                continue

            value = arguments[param]
            if value is None:
                continue

            source_units = units_dict[param]

            converted = convert_to(value, source_units, target_units)

            arguments[param] = converted

    pre_handler._decorator_kwargs = {"units"}

    return generic_decorator(pre_handler=pre_handler)
