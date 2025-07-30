"""Module containing base QC which call multiple QC functions and could be applied on a DataBundle."""

from __future__ import annotations

import inspect
from typing import Callable, Literal

import pandas as pd

from .auxiliary import failed, passed, untested
from .external_clim import get_climatological_value  # noqa
from .qc_individual_reports import (  # noqa
    do_climatology_check,
    do_date_check,
    do_day_check,
    do_hard_limit_check,
    do_missing_value_check,
    do_missing_value_clim_check,
    do_position_check,
    do_sst_freeze_check,
    do_supersaturation_check,
    do_time_check,
    do_wind_consistency_check,
)


def _get_function(name: str) -> Callable:
    """Returns the function of a given name or raises a NameError

    Parameters
    ----------
    name : str
        Name of the function to be returned

    Returns
    -------
    Callable

    Raises
    ------
    NameError
        If function of that name does not exist
    """
    func = globals().get(name)
    if not callable(func):
        raise NameError(f"Function '{name}' is not defined.")
    return func


def _is_func_param(func: Callable, param: str) -> bool:
    """Returns True if param is the name of a parameter of function func.

    Parameters
    ----------
    func: Callable
        Function whose parameters are to be inspected.
    param: str
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


def _is_in_data(name: str, data: pd.Series | pd.DataFrame) -> bool:
    """Return True if named column or variable, name, is in data

    Parameters
    ----------
    name: str
        Name of variable.
    data: pd.Series or pd.DataFrame
        Pandas Series or DataFrame to be tested.

    Returns
    -------
    bool
        Returns True if name is one of the columns or variables in data, False otherwise

    Raises
    ------
    TypeError
        If data type is not pd.Series or pd.DataFrame
    """
    if isinstance(data, pd.Series):
        return name in data
    if isinstance(data, pd.DataFrame):
        return name in data.columns
    raise TypeError(f"Unsupported data type: {type(data)}")


def _get_requests_from_params(
    params: dict | None, func: Callable, data: pd.Series | pd.DataFrame
) -> dict:
    """Given a dictionary of key value pairs where the keys are parameters in the function, func, and the values
    are columns or variables in data, create a new dictionary in which the keys are the parameter names (as in the
    original dictionary) and the values are the numbers extracted from data.

    Parameters
    ----------
    params : dict or None
        Dictionary. Keys are parameter names for the function func, and values are the names of columns or variables
        in data
    func : Callable
        Function for which the parameters will be checked
    data : pd.Series or pd.DataFrame
        DataSeries or DataFrame containing the data to be extracted.

    Returns
    -------
    dict
        Dictionary containing the key value pairs where the keys are as in the input dictionary and the values are
        extracted from the corresponding columns of data.

    Raises
    ------
    ValueError
        If one of the dictionary keys from params is not a valid argument in func.
    NameError
        If one of the dictionary values from params is not a column or variable in data.
    """
    requests = {}
    if params is None:
        return requests
    for param, cname in params.items():
        if not _is_func_param(func, param):
            raise ValueError(
                f"Parameter '{param}' is not a valid parameter of function '{func.__name__}'"
            )
        if not _is_in_data(cname, data):
            raise NameError(
                f"Variable '{cname}' is not available in input data: {data}."
            )
        requests[param] = data[cname]
    return requests


def _get_preprocessed_args(arguments: dict, preprocessed: dict) -> dict:
    """Given a dictionary of key value pairs, if one of the values is equal to __preprocessed__ then replace
    the value with the value corresponding to that key in preprocessed.

    Parameters
    ----------
    arguments: dict
        Dictionary of key value pairs where the keys are variable names and the values are strings.
    preprocessed: dict
        Dictionary of key value pairs where the keys correspond to variable names.

    Returns
    -------
    dict
        Dictionary of key value pairs where values in arguments that were set to __preprocessed__ were replaced by
        values from the dictionary preprocessed.
    """
    args = {}
    for k, v in arguments.items():
        if v == "__preprocessed__":
            v = preprocessed[k]
        args[k] = v
    return args


def do_multiple_row_check(
    data: pd.Series | pd.DataFrame,
    qc_dict: dict | None = None,
    preproc_dict: dict | None = None,
    return_method: Literal["all", "passed", "failed"] = "all",
) -> pd.Series | pd.DataFrame:
    """Basic row-by-row QC by using multiple QC functions.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Hashable input data.
    qc_dict : dict, optional
        Nested QC dictionary.
        Keys represent arbitrary names of the check.
        The values are dictionaries which contain the keys "func" (name of the QC function),
        "names" (input data names as keyword arguments, that will be retrieved from `data`) and,
        if necessary, "arguments" (the corresponding keyword arguments).
        For more information see Examples.
    preproc_dict : dict, optional
        Nested pre-processing dictionary.
        Keys represent variable names that can be used by `qc_dict`.
        The values are dictionaries which contain the keys "func" (name of the pre-processing function),
        "names" (input data names as keyword arguments, that will be retrieved from `data`), and "inputs"
        (list of input-given variables).
        For more information see Examples.
    return_method: {"all", "passed", "failed"}, default: "all"
        If "all", return QC dictionary containing all requested QC check flags.
        If "passed": return QC dictionary containing all requested QC check flags until the first check passes.
        Other QC checks are flagged as unstested (3).
        If "failed": return QC dictionary containing all requested QC check flags until the first check fails.
        Other QC checks are flagged as unstested (3).

    Returns
    -------
    pd.Series
        Columns represent arbitrary names of the check (taken from `qc_dict.keys()`).
        Values representing corresponding QC flags.
        For information to QC flags see QC functions.

    Raises
    ------
    NameError
        If a function listed in `qc_dict` or `preproc_dict` is not defined.
        If columns listed in `qc_dict` or `preproc_dict` are not available in `data`.
    ValueError
        If `return_method` is not one of ["all", "passed", "failed"]
        If variable names listed in `qc_dict` or `preproc_dict` are not valid parameters of the QC function.

    Notes
    -----
    If a variable is pre-processed using `preproc_dict`, mark the variable name as "__preprocessed__" in `qc_dict`.
    E.g. `"climatology": "__preprocessed__"`.

    For more information, see Examples.

    Examples
    --------
    An example `qc_dict` for a hard limit test:

    .. code-block:: python

        qc_dict = {
            "hard_limit_check": {
                "func": "do_hard_limit_check",
                "names": "ATEMP",
                "arguments": {"limits": [193.15, 338.15]},
            }
        }

    An example `qc_dict` for a climatology test. Variable "climatology" was previously defined:

    .. code-block:: python

        qc_dict = {
            "climatology_check": {
                "func": "do_climatology_check",
                "names": {
                    "value": "observation_value",
                    "lat": "latitude",
                    "lon": "longitude",
                    "date": "date_time",
                },
                "arguments": {
                    "climatology": climatology,
                    "maximum_anomaly": 10.0,  # K
                },
            },
        }

    An example `preproc_dict` for extracting a climatological value:

    .. code-block:: python

        preproc_dict = {
            "func": "get_climatological_value",
            "names": {
                "lat": "latitude",
                "lon": "longitude",
                "date": "date_time",
            },
            "inputs": climatology,
        }

    Make use of both dictionaries:

    .. code-block:: python

        preproc_dict = {
            "func": "get_climatological_value",
            "names": {
                "lat": "latitude",
                "lon": "longitude",
                "date": "date_time",
            },
            "inputs": climatology,
        }

        qc_dict = {
            "climatology_check": {
                "func": "do_climatology_check",
                "names": {
                    "value": "observation_value",
                },
                "arguments": {
                    "climatology": "__preprocessed__",
                    "maximum_anomaly": 10.0,  # K
                },
            },
        }

    """
    if qc_dict is None:
        qc_dict = {}

    if preproc_dict is None:
        preproc_dict = {}

    if return_method not in ["all", "passed", "failed"]:
        raise ValueError(
            f"'return_method' has to be one of ['all', 'passed', 'failed']: {return_method}"
        )

    # Firstly, check if all functions are callable and all requested input variables are available!
    preprocessed = {}
    for var_name, preproc_params in preproc_dict.items():
        func_name = preproc_params.get("func")
        func = _get_function(func_name)

        requests = _get_requests_from_params(preproc_params.get("names"), func, data)

        inputs = preproc_params.get("inputs")
        if not isinstance(inputs, list):
            inputs = [inputs]

        preprocessed[var_name] = func(*inputs, **requests)

    qc_inputs = {}
    for qc_name, qc_params in qc_dict.items():
        func_name = qc_params.get("func")
        func = _get_function(func_name)
        requests = _get_requests_from_params(qc_params.get("names"), func, data)

        qc_inputs[qc_name] = {}
        qc_inputs[qc_name]["function"] = func
        qc_inputs[qc_name]["requests"] = requests
        qc_inputs[qc_name]["kwargs"] = {}

        if "arguments" in qc_params.keys():
            qc_inputs[qc_name]["kwargs"] = _get_preprocessed_args(
                qc_params["arguments"], preprocessed
            )

    is_series = isinstance(data, pd.Series)
    if is_series:
        data = pd.DataFrame([data.values], columns=data.index)

    mask = pd.Series(True, index=data.index)
    results = pd.DataFrame(untested, index=data.index, columns=qc_inputs.keys())

    for qc_name, qc_params in qc_inputs.items():
        if not mask.any():
            continue

        args = {
            k: (v[mask] if isinstance(v, pd.Series) else v)
            for k, v in qc_params["requests"].items()
        }
        kwargs = {
            k: (v[mask] if isinstance(v, pd.Series) else v)
            for k, v in qc_params["kwargs"].items()
        }

        partial_result = qc_params["function"](**args, **kwargs)
        full_result = pd.Series(untested, index=data.index)
        full_result.loc[mask] = partial_result
        results[qc_name] = full_result

        if return_method == "failed":
            mask &= full_result != failed
        elif return_method == "passed":
            mask &= full_result != passed

    if is_series is True:
        return results.iloc[0]
    return results
