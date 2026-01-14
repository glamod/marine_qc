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
from pandas._libs.missing import NAType
from pandas._libs.tslibs.nattype import NaTType
from xclim.core.units import convert_units_to, units


passed: int = 0
failed: int = 1
untestable: int = 2
untested: int = 3

PandasNAType: TypeAlias = NAType
PandasNaTType: TypeAlias = NaTType

# --- Scalars ---
ScalarIntType: TypeAlias = int | np.integer | PandasNAType | None
ScalarFloatType: TypeAlias = float | np.floating | PandasNAType | None
ScalarDatetimeType: TypeAlias = datetime | np.datetime64 | pd.Timestamp | PandasNaTType | None

# --- Sequences ---
SequenceIntType: TypeAlias = (
    Sequence[ScalarIntType] | npt.NDArray[np.integer] | pd.Series | np.ndarray  # optionally: pd.Series[np.integer] or pd.Series[pd.Int64Dtype]
)

SequenceFloatType: TypeAlias = (
    Sequence[ScalarFloatType] | npt.NDArray[np.floating] | pd.Series | np.ndarray  # optionally: pd.Series[np.floating] or pd.Series[pd.Float64Dtype]
)

SequenceDatetimeType: TypeAlias = (
    Sequence[ScalarDatetimeType] | npt.NDArray[np.datetime64] | pd.Series | np.ndarray  # optionally: pd.Series[pd.DatetimeTZDtype] or similar
)

# --- Value Types (Scalar or Sequence) ---
ValueFloatType: TypeAlias = ScalarFloatType | SequenceFloatType
ValueIntType: TypeAlias = ScalarIntType | SequenceIntType
ValueDatetimeType: TypeAlias = ScalarDatetimeType | SequenceDatetimeType

earths_radius = 6371008.8  # m


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
    x : Any
        The value to check.

    Returns
    -------
    bool
        True if `x` is scalar-like, False otherwise.
    """
    try:
        return bool(np.ndim(x) == 0)
    except TypeError:
        return True  # fallback: built-in scalars like int, float, pd.Timestamp


def isvalid(inval: ValueFloatType) -> bool | np.ndarray[bool]:
    """
    Check if a value(s) are numerically valid (not None or NaN).

    Parameters
    ----------
    inval : float, None, array-like of float or None
        Input value(s) to be tested.

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


def format_return_type(result_array: np.ndarray, *input_values: Any, dtype: type = int) -> Any:
    r"""
    Convert the result numpy array(s) to the same type as the input `value`.

    If `result_array` is a sequence of arrays, format each element recursively,
    preserving the container type.

    Parameters
    ----------
    result_array : np.ndarray
        The numpy array of results.
    \*input_values : scalar, sequence, np.ndarray, pd.Series or None
        One or more original input values to infer the desired return type from.
    dtype : type, optional
        Desired data type of the result. Default is int.

    Returns
    -------
    Same type as input(s)
        The result formatted to match the type of the first valid input value.
    """
    input_value = next((val for val in input_values if val is not None), None)

    if input_value is None or is_scalar_like(input_value):
        if hasattr(result_array, "ndim") and result_array.ndim > 0:
            result_array = result_array[0]
        return dtype(result_array)
    if isinstance(input_value, pd.Series):
        return pd.Series(result_array, index=input_value.index, dtype=dtype)
    if isinstance(input_value, (list, tuple)):
        return type(input_value)(result_array.tolist())
    if isinstance(input_value, np.ndarray) and isinstance(result_array, pd.Series):
        return result_array.to_numpy()
    return result_array  # np.ndarray or fallback


def convert_to(value: SequenceFloatType, source_units: str, target_units: str) -> SequenceFloatType:
    """
    Convert a float or sequence from source units to target units.

    Parameters
    ----------
    value : float or None or array-like of float or None
        A single float value, None, or a sequence (e.g., list, tuple, array-like)
        containing floats and/or None values. `None` values are passed through unchanged.
    source_units : str
        The unit(s) of the input value(s), e.g., 'degC', 'km/h'.
    target_units : str
        The unit(s) to convert to, e.g., 'K', 'm/s'.
        If set to "unknown", the value(s) will be converted to the base SI units
        of the source_units, e.g., 'degC' to 'kelvin', 'km/h' to 'meter/s'.

    Returns
    -------
    float or None or array-like of float or None
        The converted value(s), preserving the input structure (scalar, list, tuple, array).
        None values remain unchanged.

    Examples
    --------
    >>> convert_to(100, "degC", "K")
    373.15

    >>> convert_to([0, 100], "degC", "K")
    [273.15, 373.15]

    >>> convert_to([None, 100], "degC", "K")
    [None, 373.15]

    >>> convert_to(5, "km", "unknown")  # Converts to base unit 'meter'
    5000.0
    """

    def _convert_to(value: Any) -> Any:
        """
        Convert units of value.

        Parameters
        ----------
        value : Any
            Value to be converted.

        Returns
        -------
        Any
            Converted value.
        """
        if not isvalid(value):
            return value
        return convert_units_to(value * registry, target_units)

    registry = units(source_units)
    if target_units == "unknown":
        target_units = registry.to_base_units()

    if isinstance(value, np.ndarray):
        return np.array([_convert_to(v) for v in value])
    if isinstance(value, Sequence):
        return type(value)([_convert_to(v) for v in value])
    return _convert_to(value)


def generic_decorator(
    pre_handler: Callable[[dict[str, Any]], None] | None = None,
    post_handler: Callable[[Any, dict[str, Any]], Any] | None = None,
) -> Callable[..., Any]:
    """
    Create a decorator that binds function arguments and applies pre- and post-processing handlers.

    This decorator factory allows you to inspect, modify, or validate function arguments before
    and after the original function is called. Reserved keyword arguments can be passed to the
    handlers via `_decorator_kwargs` and removed from the call to the original function.

    Parameters
    ----------
    pre_handler : Callable[[dict], None]
      A function that takes the bound arguments dictionary (`bound_args.arguments`) and
      optionally additional keyword arguments, to inspect or modify arguments before the
      decorated function executes. Signature:
      `handler(arguments: dict, **meta_kwargs) -> None`.
    post_handler : Callable[[dict], None]
      A function that takes the bound arguments dictionary (`bound_args.arguments`) and
      optionally additional keyword arguments, to inspect or modify arguments after the
      decorated function executes. Signature:
      `handler(arguments: dict, **meta_kwargs) -> None`.

    Returns
    -------
    Callable
      A decorator that wraps any function. When applied, the function's arguments are bound
      and passed to the handlers before execution.

    Notes
    -----
    - Handlers can define a `_decorator_kwargs` attribute (a set of reserved keyword argument names).
      These reserved kwargs will be extracted from the decorated function's call kwargs, passed to
      the handler, and removed before calling the original function.
    - The original function is called with the possibly modified bound arguments after handler processing.
    """
    if pre_handler:
        pre_handler._is_post_handler = False
    if post_handler:
        post_handler._is_post_handler = True

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator that binds function arguments and applies pre- and post-handlers.

        Parameters
        ----------
        func : Callable[..., Any]
            The function to be decorated. Its arguments will be bound and optionally modified
            by the pre- and post-handlers.

        Returns
        -------
        Callable[..., Any]
            The `wrapper` function that executes pre-handlers, calls the original function
            and then executes post-handlers.
        """
        handlers = []
        if pre_handler:
            handlers.append(pre_handler)
        if post_handler:
            handlers.append(post_handler)

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            r"""
            Wrapper function that executes pre-handlers, calls the original function, and executes post-handlers.

            This function is generated by the decorator returned from `generic_decorator`. It:
            1. Binds all positional and keyword arguments of the original function.
            2. Extracts reserved keyword arguments specified by `_decorator_kwargs` from the call.
            3. Executes all pre-handlers in reverse order, passing the bound arguments and reserved kwargs.
            4. Calls the original function with the possibly modified bound arguments.
            5. Executes all post-handlers in reverse order, passing the function result, current arguments,
            and the original arguments.

            Parameters
            ----------
            \*args : tuple
              Positional arguments to the decorated function.
            \**kwargs : dict
              Keyword arguments to the decorated function. Reserved kwargs in `_decorator_kwargs`
              are removed from this dictionary before calling the original function but passed
              to the handlers.

            Returns
            -------
            Any
              The return value from the original function, optionally modified by post-handlers.

            Notes
            -----
            - Handlers can inspect and modify bound arguments before and after the function call.
            - Pre-handlers receive the arguments dictionary (`bound_args.arguments`) and reserved kwargs.
            - Post-handlers receive the function result, the current arguments, and the original arguments.
            """
            reserved_keys = set()
            all_pre_handlers = []
            all_post_handlers = []
            current_func = wrapper
            visited = set()

            while hasattr(current_func, "__wrapped__") and id(current_func) not in visited:
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
            meta_kwargs = {k: kwargs.pop(k) if k not in sig.parameters else kwargs[k] for k in reserved_keys if k in kwargs}

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

        wrapper._decorator_handlers = handlers

        return wrapper

    return decorator


def post_format_return_type(params: list[str], dtype: type = int, multiple: bool = False) -> Callable[..., Any]:
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
    dtype : type, optional
        Desired data type of the result. Default is int.
    multiple : bool, optional
        If True, assumes the function returns a sequence of results (e.g., a tuple),
        and applies `format_return_type` to each element individually.
        If False (default), applies `format_return_type` once on the entire result.

    Returns
    -------
    Callable[..., Any]
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

    def post_handler(result: Any, arguments: dict[str, Any], **original_call: Any) -> Any:
        r"""
        Post-processing handler that formats a function's return value.

        Parameters
        ----------
        result : Any
            The output returned by the decorated function, which will be reformatted.
        arguments : dict
            The dictionary of bound arguments for the decorated function.
        \**original_call : dict
            Original values of the inputs (from the decorated function's call or context)
            used to determine the target type and structure for formatting.

        Returns
        -------
        Any
            The reformatted function result:
            - If `multiple=False` (default), the entire result is formatted as a single object.
            - If `multiple=True`, each element in the result sequence (e.g., tuple) is individually formatted.
        """
        input_values = [original_call[param] for param in params if param in original_call]

        if multiple:
            return tuple(format_return_type(r, *input_values, dtype=dtype) for r in result)
        else:
            return format_return_type(result, *input_values, dtype=dtype)

    return generic_decorator(post_handler=post_handler)


def inspect_arrays(params: list[str], sortby: str | None = None) -> Callable[..., Any]:
    """
    Decorator to convert and validate specified function input parameters as 1D NumPy arrays.

    This decorator ensures that specified input arguments are sequence-like, converts them
    to 1D NumPy arrays, validates that they are one-dimensional, and checks that all arrays
    have the same length. Optionally, the arrays can be sorted by another parameter and
    later restored to the original order.

    Parameters
    ----------
    params : list of str
        Names of parameters to inspect in the decorated function. Each specified parameter
        will be converted to a 1D NumPy array and validated.
    sortby : str, optional
        Name of a parameter to sort the arrays by, if desired. The result will be returned
        in the original order of this parameter.

    Returns
    -------
    Callable[..., Any]
        A decorator that, when applied, converts the specified parameters to 1D NumPy arrays,
        validates them, optionally sorts them, and passes them to the decorated function.

    Raises
    ------
    ValueError
        If a specified parameter is missing from the function arguments.
        If any specified parameter is not one-dimensional.
        If the lengths of the specified arrays do not all match.

    Notes
    -----
    - If `sortby` is specified, the result of the function is reordered to match the
      original order of `sortby` after the function executes.

    Examples
    --------
    >>> @inspect_arrays(["a", "b"])
    ... def add_arrays(a, b):
    ...     return a + b

    >>> add_arrays([1, 2, 3], [4, 5, 6])
    array([5, 7, 9])

    >>> add_arrays([1, 2], [3, 4, 5])
    Traceback (most recent call last):
        ...
    ValueError: Input ['a', 'b'] must all have the same length.
    """

    def pre_handler(arguments: dict[str, Any], **meta_kwargs: Any) -> None:
        r"""
        Pre-processing handler to convert inputs to 1D NumPy arrays and validate lengths.

        Parameters
        ----------
        arguments : dict
            Bound arguments of the decorated function.
        \**meta_kwargs : dict
            Additional reserved keyword arguments passed through the decorator framework.

        Raises
        ------
        ValueError
            If any parameter in `params` is missing, not 1D, or arrays do not all have the same length.
        """
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

    def post_handler(result: Any, arguments: dict[str, Any], **original_call: Any) -> Any:
        r"""
        Post-processing handler to restore the original order if `sortby` is used.

        Parameters
        ----------
        result : Any
            The output returned by the decorated function.
        arguments : dict
            The dictionary of bound arguments for the decorated function.
        \**original_call : dict
            Original values of the inputs (before preprocessing) used to restore the order.

        Returns
        -------
        Any
            The output reordered to match the original input order of `sortby` if specified;
            otherwise, returns the result unmodified.
        """
        if sortby is None:
            return result
        sort_indices = np.argsort(original_call[sortby])
        inverse_indices = np.argsort(sort_indices)
        if len(result) == 0:
            return result
        return result[inverse_indices]

    return generic_decorator(pre_handler=pre_handler, post_handler=post_handler)


def convert_units(**units_by_name: str) -> Callable[..., Any]:
    r"""
    Decorator to automatically convert specified function arguments to target units.

    This decorator allows a function to accept inputs in various units and automatically
    converts them to desired target units before the function executes. It is especially
    useful for scientific or engineering functions where users may provide inputs in
    different unit systems.

    Parameters
    ----------
    \**units_by_name : str
        Keyword arguments mapping function argument names to their target units.
        Special case: if a target unit is "unknown", it will be converted to the base SI
        unit for the given source unit (e.g., "degC" ? "K", "km/h" ? "m/s").

    Returns
    -------
    Callable[..., Any]
        A decorator that converts specified parameters to the target units prior to
        executing the decorated function.

    Notes
    -----
    - The decorated function must be called with a `units` keyword argument, which can be:
        - A dictionary mapping argument names to their source units, or
        - A single string unit applied to all arguments.
    - Parameters not listed in `units_by_name` are not converted.
    - Parameters with `None` values are skipped.
    - If a target unit is "unknown", the value is converted to the base SI unit.

    Examples
    --------
    >>> @convert_units(temperature="K")
    ... def func_single(temperature):
    ...     print(f"Temperature: {temperature:.2f} K")

    >>> func_single(25.0, units={"temperature": "degC"})
    Temperature: 298.15 K

    >>> @convert_units(speed="m/s", altitude="m")
    ... def func_multiple(speed, altitude):
    ...     print(f"Speed: {speed:.1f} m/s, Altitude: {altitude:.0f} m")

    >>> func_multiple(72.0, 0.5, units={"speed": "km/h", "altitude": "km"})
    Speed: 20.0 m/s, Altitude: 500 m

    >>> @convert_units(distance="unknown")
    ... def func_base(distance):
    ...     print(f"Distance in SI units: {distance} m")

    >>> func_base(1.2, units={"distance": "km"})
    Distance in SI units: 1200.0 m
    """

    def pre_handler(arguments: dict[str, Any], **meta_kwargs: Any) -> None:
        r"""
        Pre-processing handler that converts specified arguments to target units.

        Parameters
        ----------
        arguments : dict
            Bound arguments of the decorated function.
        \**meta_kwargs : dict
            Additional reserved keyword arguments passed to the handler.
            Must include 'units', which maps parameter names to their source units.

        Raises
        ------
        ValueError
            If a specified parameter is missing in `arguments`.
        """
        units_dict = meta_kwargs.get("units")
        if units_dict is None:
            return
        if isinstance(units_dict, str):
            units_str = units_dict
            units_dict = {param: units_str for param in arguments}

        for param, target_units in units_by_name.items():
            if param not in arguments:
                raise ValueError(f"Parameter '{param}' not found in function arguments.")
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
