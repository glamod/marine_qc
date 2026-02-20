"""Module containing base QC which call multiple QC functions and could be applied on a DataBundle."""

from __future__ import annotations
import collections.abc as abc
import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
import numpy.typing as npt
import pandas as pd

from .auxiliary import DECORATOR_HANDLERS, DECORATOR_KWARGS
from .external_clim import get_climatological_value  # noqa: F401
from .qc_grouped_reports import (  # noqa: F401
    do_bayesian_buddy_check,
    do_mds_buddy_check,
)
from .qc_individual_reports import (  # noqa: F401
    do_climatology_check,
    do_date_check,
    do_day_check,
    do_hard_limit_check,
    do_missing_value_check,
    do_missing_value_clim_check,
    do_night_check,
    do_position_check,
    do_sst_freeze_check,
    do_supersaturation_check,
    do_time_check,
    do_wind_consistency_check,
)
from .qc_sequential_reports import (  # noqa: F401
    do_few_check,
    do_iquam_track_check,
    do_spike_check,
    do_track_check,
    find_multiple_rounded_values,
    find_repeated_values,
    find_saturated_runs,
)


def _validate_non_generic(value: Any, expected: Any) -> bool:
    """
    Validate a non-generic type (str, int, float, etc.).

    Parameters
    ----------
    value : Any
        The value to validate.
    expected : Any
        The expected type.

    Returns
    -------
    bool
        True if `value` matches `expected`, False otherwise.
    """
    if isinstance(expected, type):
        return isinstance(value, expected)
    return False


def _validate_mapping(value: Mapping[Any, Any], origin: type, args: Tuple[Any, ...]) -> bool:
    """
    Validate a mapping type (dict, Mapping).

    Parameters
    ----------
    value : Mapping[Any, Any]
        The value to validate.
    origin : type
        The mapping type (e.g., dict).
    args : tuple[Any, ...]
        Expected key and value types.

    Returns
    -------
    bool
        True if `value` matches the mapping type and key/value types, False otherwise.
    """
    if not isinstance(value, origin):
        return False
    if not args:
        return True
    key_type, val_type = args
    return all(validate_type(k, key_type) and validate_type(v, val_type) for k, v in value.items())


def _validate_iterable(value: Iterable[Any], origin: type, args: Tuple[Any, ...]) -> bool:
    """
    Validate an iterable type (list, set, frozenset).

    Parameters
    ----------
    value : Any
        The value to validate.
    origin : type
        The iterable type.
    args : tuple[Any, ...]
        Expected element types.

    Returns
    -------
    bool
        True if all elements match the expected type, False otherwise.
    """
    if not isinstance(value, origin):
        return False
    if not args:
        return True
    elem_type = args[0]
    return all(validate_type(v, elem_type) for v in value)


def _validate_sequence(value: Any, args: Tuple[Any, ...]) -> bool:
    """
    Validate a generic sequence type (e.g., Sequence[int]).

    Parameters
    ----------
    value : Any
        The value to validate.
    args : tuple[Any, ...]
        Expected element types.

    Returns
    -------
    bool
        True if all elements match the expected type, False otherwise.
    """
    if not isinstance(value, abc.Sequence) or isinstance(value, (str, bytes)):
        return False
    if not args:
        return True
    elem_type = args[0]
    return all(validate_type(v, elem_type) for v in value)


def _validate_tuple(value: Any, args: Tuple[Any, ...]) -> bool:
    """
    Validate a tuple type (fixed-length or homogeneous).

    Parameters
    ----------
    value : Any
        The value to validate.
    args : tuple[Any, ...]
        Expected element types.

    Returns
    -------
    bool
        True if the tuple matches the expected types and length, False otherwise.
    """
    if not isinstance(value, abc.Sequence) or isinstance(value, (str, bytes)):
        return False
    if not args:
        return True
    if len(args) == 2 and args[1] is Ellipsis:
        return all(validate_type(v, args[0]) for v in value)
    if len(args) != len(value):
        return False
    return all(validate_type(v, t) for v, t in zip(value, args, strict=False))


def _validate_ndarray(value: Any, args: Tuple[Any, ...]) -> bool:
    """
    Validate a numpy ndarray type, optionally checking dtype.

    Parameters
    ----------
    value : Any
        The value to validate.
    args : tuple[Any, ...]
        Expected dtype (first argument may be `Any` or unspecified).

    Returns
    -------
    bool
        True if `value` is an ndarray and matches expected dtype, False otherwise.
    """
    if not isinstance(value, np.ndarray):
        return False

    if not args:
        return True

    if len(args) < 2:
        return True

    expected_dtype = args[1]

    inner = get_args(expected_dtype)
    if inner:
        expected_dtype = inner[0]

    if expected_dtype in (Any, None):
        return True

    try:
        return np.issubdtype(value.dtype, expected_dtype)
    except TypeError:
        return False


def _safe_isinstance(value: Any, origin: Any) -> bool:
    """
    Safely check if value is an instance of a type, avoiding TypeError for weird generics.

    Parameters
    ----------
    value : Any
        Value to check.
    origin : Any
        Type or generic to check against.

    Returns
    -------
    bool
        True if `value` is an instance of `origin`, False otherwise.
    """
    try:
        return isinstance(value, origin)
    except TypeError:
        return False


def validate_type(value: Any, expected: Any) -> bool:
    """
    Recursively validate that a value matches the expected type hint.

    Parameters
    ----------
    value : Any
        The value to validate.
    expected : Any
        The expected value type for validation.

    Returns
    -------
    bool
        - True if type of `value` does match `expected`.
        - False if type of `value` does not match `expected`.
    """
    if expected is Any:
        return True

    origin = get_origin(expected)
    args = get_args(expected)

    if origin is None:
        return _validate_non_generic(value, expected)

    if origin is Annotated:
        return validate_type(value, args[0])

    if origin is Literal:
        return value in args

    if origin in (Union, UnionType):
        return any(validate_type(value, t) for t in args)

    if origin is abc.Callable:
        return callable(value)

    if origin in (np.ndarray, npt.NDArray):
        return _validate_ndarray(value, args)

    if isinstance(expected, type) and issubclass(expected, (pd.DataFrame, pd.Series)):
        return isinstance(value, expected)

    if isinstance(origin, type) and issubclass(origin, abc.Mapping):
        return _validate_mapping(value, origin, args)

    if isinstance(origin, type) and issubclass(origin, (list, set, frozenset)):
        return _validate_iterable(value, origin, args)

    if origin is tuple:
        return _validate_tuple(value, args)

    if isinstance(origin, type) and issubclass(origin, abc.Sequence):
        return _validate_sequence(value, args)

    return _safe_isinstance(value, origin)


def validate_arg(
    key: str,
    value: Any,
    func_name: str,
    parameters: Mapping[str, inspect.Parameter],
    type_hints: Mapping[str, Any],
    reserved_keys: set[str],
    has_arguments: bool,
) -> None:
    """
    Validate argument against a function's signature, taking decorators into account.

    Parameters
    ----------
    key : str
        The name of the argument to validate.
    value : Any
        The value of the argument to validate.
    func_name : str
        The name of the function (used in error message).
    parameters : Mapping[str, inspect.Parameter]
        A mapping of parameter names to `inspect.Parameter` objects,
        typically from `inspect.signature(func).parameters`.
    type_hints : Mapping[str, type]
        A mapping of parameter names to expected types,
        typically from `typing.get_type_hints(func)`.
    reserved_keys : set[str]
        Argument names that are considered reserved and should nor raise errors.
    has_arguments : bool
        Whether the function accepts arbitrary arguments.
    """
    if has_arguments or key in reserved_keys:
        return

    if key not in parameters:
        raise ValueError(f"Parameter '{key}' is not a valid parameter of function '{func_name}'.")

    expected = type_hints.get(key)
    if not expected or expected is inspect._empty:
        return

    if not validate_type(value, expected):
        raise TypeError(f"Parameter '{key}' does not match expected type {expected!r}. Got value {value!r} of type {type(value).__name__}.")


def validate_args(
    func: Callable[..., Any],
    args: Sequence[Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> None:
    """
    Validate positional and keyword arguments against a function's signature, taking decorators into account.

    This function checks that:
    - All provided keyword arguments correspond to valid parameters of the given function.
    - All required parameters of the function (i.e., parameters without default values) are present in the provided keyword arguments.

    Parameters
    ----------
    func : Callable[..., Any]
        The function whose signature is used for validation.
    args : Sequence[Any], optional
        Sequence of arguments intended to be passed to `func`.
    kwargs : Mapping[str, Any], optional
        Dictionary of keyword arguments intended to be passed to `func`.

    Raises
    ------
    ValueError
        If `kwargs` contains a key that is not a parameter of `func`.
    TypeError
        If a required parameter of `func` is missing from `kwargs`.
    """

    def all_handlers(func: Callable[..., Any]) -> list[Callable[..., Any]]:
        """
        Collect all decorator handlers applied to a function.

        Parameters
        ----------
        func : Callable[..., Any]
            The function to inspect for applied decorator handlers.

        Returns
        -------
        List[Callable[..., Any]]
            A list of all decorator handlers associated with the function,
            including handlers from wrapped functions.
        """
        handlers: list[Callable[..., Any]] = []
        current: Callable[..., Any] = func
        while True:
            handlers.extend(DECORATOR_HANDLERS.get(current, []))
            if hasattr(current, "__wrapped__"):
                current = current.__wrapped__
            else:
                break
        return handlers

    args = args or ()
    if not isinstance(args, (list, tuple)):
        args = (args,)

    kwargs = kwargs or {}

    reserved_keys: set[str] = set()
    for handler in all_handlers(func):
        reserved_keys.update(DECORATOR_KWARGS.get(handler, set()))

    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    positional_params = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]

    has_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)

    if len(args) > len(positional_params) and not has_args:
        raise TypeError(f"Too many positional arguments for function '{func.__name__}'.")

    bound_args = [positional_params[i].name for i in range(min(len(args), len(positional_params)))]

    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

    type_hints = get_type_hints(func)

    for i, arg in enumerate(args):
        validate_arg(bound_args[i], arg, func.__name__, sig.parameters, type_hints, reserved_keys, has_args)

    for key, value in kwargs.items():
        validate_arg(key, value, func.__name__, sig.parameters, type_hints, reserved_keys, has_kwargs)

    for param in params:
        if (
            param.default is inspect.Parameter.empty
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and param.name not in kwargs
            and param.name not in bound_args
        ):
            raise TypeError(f"Required parameter '{param.name}' is missing for function '{func.__name__}'.")


def validate_dict(input_dict: Mapping[str, Mapping[str, Any]]) -> None:
    """
    Validate that the input is a dictionary with string keys and dictionary values.

    This function checks that:
    - `input_dict` is a dictionary.
    - All keys in the dictionary are strings.
    - All top-level values in the dictionary are themselves dictionaries.

    Parameters
    ----------
    input_dict : Mapping[str, Mapping[str, Any]]
        The object to validate.

    Raises
    ------
    TypeError
        If `input_dict` is not a dictionary, if any key is not a string,
        or if any value is not a dictionary.
    """
    if not isinstance(input_dict, Mapping):
        raise TypeError(f"input must be a dictionary, not {type(input_dict)}.")

    for k, v in input_dict.items():
        if not isinstance(k, str):
            raise TypeError(f"input key {k} must be a string, not {type(k).__name__}.")
        if not isinstance(v, Mapping):
            raise TypeError(f"value for key {k} must be a dictionary, not {type(v).__name__}.")


def is_in_data(name: str, data: pd.Series | pd.DataFrame) -> bool:
    """
    Return True if named column or variable, name, is in data.

    Parameters
    ----------
    name : str
        Name of variable.
    data : pd.Series or pd.DataFrame
        Pandas Series or DataFrame to be tested.

    Returns
    -------
    bool
        Returns True if name is one of the columns or variables in data, False otherwise.

    Raises
    ------
    TypeError
        If data type is not pd.Series or pd.DataFrame.
    """
    if isinstance(data, pd.Series):
        return bool(data.name == name)
    if isinstance(data, pd.DataFrame):
        return bool(name in data.columns)
    raise TypeError(f"Unsupported data type: {type(data)}")


def is_func_param(func: Callable[..., Any], param: str) -> bool:
    """
    Return True if param is the name of a parameter of function func.

    Parameters
    ----------
    func : Callable
        Function whose parameters are to be inspected.
    param : str
        Name of the parameter.

    Returns
    -------
    bool
        Returns True if param is one of the functions parameters or the function uses ``**kwargs``.
    """
    sig = inspect.signature(func)
    if "kwargs" in sig.parameters:
        return True
    return param in sig.parameters
