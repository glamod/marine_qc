"""Module containing base QC which call multiple QC functions and could be applied on a DataBundle."""

from __future__ import annotations
import inspect
from collections.abc import Callable, Iterable
from typing import Literal

import pandas as pd

from .auxiliary import failed, passed, untested
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


def _get_function(name: str) -> Callable:
    """
    Returns the function of a given name or raises a NameError

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
    """
    Returns True if param is the name of a parameter of function func.

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
    """
    Returns True if named column or variable, name, is in data

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


def _get_requests_from_params(params: dict | None, func: Callable, data: pd.Series | pd.DataFrame) -> dict:
    """
    Given a dictionary of key value pairs where the keys are parameters in the function, func, and the values
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
            raise ValueError(f"Parameter '{param}' is not a valid parameter of function '{func.__name__}'")
        if not _is_in_data(cname, data):
            raise NameError(f"Variable '{cname}' is not available in input data: {data}.")
        requests[param] = data[cname]
    return requests


def _get_preprocessed_args(arguments: dict, preprocessed: dict) -> dict:
    """
    Given a dictionary of key value pairs, if one of the values is equal to __preprocessed__ then replace
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


def _prepare_preprocessed_vars(preproc_dict, data):
    """Run preprocessing functions and return a {var_name: preprocessed_var} dict."""
    preprocessed = {}

    for var_name, params in preproc_dict.items():
        func = _get_function(params["func"])
        requests = _get_requests_from_params(params.get("names"), func, data)

        inputs = params.get("inputs")
        if not isinstance(inputs, list):
            inputs = [inputs]

        preprocessed[var_name] = func(*inputs, **requests)

    return preprocessed


def _prepare_qc_functions(qc_dict, preprocessed, data):
    """Return a {qc_name: {function, requests, kwargs}} dictionary."""
    qc_inputs = {}

    for qc_name, params in qc_dict.items():
        func = _get_function(params["func"])
        requests = _get_requests_from_params(params.get("names"), func, data)
        kwargs = _get_preprocessed_args(params.get("arguments", {}), preprocessed)

        qc_inputs[qc_name] = {"function": func, "requests": requests, "kwargs": kwargs}

    return qc_inputs


def _apply_qc_to_masked_rows(qc_func, args, kwargs, data_index, mask):
    """
    Execute QC function, align its output to data_index, and
    return full_result (Series with correct shape).
    """
    partial = qc_func(**args, **kwargs)

    partial = pd.Series(partial, index=data_index)

    full = pd.Series(untested, index=data_index)

    full.loc[mask] = partial.loc[mask]

    return full


def _normalize_groupby(data, groupby):
    """Return iterable of (name, group_df) pairs, trimming invalid rows."""
    if groupby is None:
        return [(None, data)]

    if not isinstance(groupby, pd.core.groupby.generic.DataFrameGroupBy):
        return list(data.groupby(groupby, group_keys=False, sort=False))

    valid = data.index
    groups = []

    for name, group in groupby:
        idx = group.index.intersection(valid)
        if len(idx) > 0:
            groups.append((name, group.loc[idx]))

    return groups


def _validate_and_normalize_input(
    data: pd.DataFrame | pd.Series,
    return_method: Literal["all", "passed", "failed"],
) -> tuple[pd.DataFrame, bool]:
    """
    Validate the return method and convert a Series into a single-row DataFrame
    while keeping track of whether the original input was a Series.
    """
    if return_method not in ("all", "passed", "failed"):
        raise ValueError("'return_method' must be 'all','passed','failed'.")
    is_series = isinstance(data, pd.Series)
    if is_series:
        data = pd.DataFrame([data.values], columns=data.index)
    return data, is_series


def _prepare_all_inputs(
    data: pd.DataFrame,
    qc_dict: dict | None,
    preproc_dict: dict | None,
) -> tuple[dict, pd.Series, pd.DataFrame]:
    """
    Build all inputs required for QC execution, including preporcessed variables,
    resolved QC function arguments, an initial boolean mask, and an empty results table.
    """
    qc_dict = qc_dict or {}
    preproc_dict = preproc_dict or {}

    preprocessed = _prepare_preprocessed_vars(preproc_dict, data)
    qc_inputs = _prepare_qc_functions(qc_dict, preprocessed, data)

    mask = pd.Series(True, index=data.index)
    results = pd.DataFrame(untested, index=data.index, columns=qc_inputs.keys())

    return qc_inputs, mask, results


def _group_iterator(
    data: pd.DataFrame,
    groupby: str | Iterable[str] | pd.core.groupby.generic.DataFrameGroupBy | None,
):
    """
    Returns an iterator of (key, group DataFrame).
    If groupby is None, yields the whole DataFrame as a single group.
    Otherwise, yields each group according to _normalize_groupby.
    """
    if groupby is None:
        yield None, data
    else:
        yield from _normalize_groupby(data, groupby)


def _run_qc_engine(
    data: pd.DataFrame,
    qc_inputs: dict,
    groups: Iterable,
    return_method: Literal["all", "passed", "failed"],
) -> pd.DataFrame:
    """
    Execute all QC checks over the provided groups using shared mask-based
    pass/fail propagation and collect the resulting QC flags.
    """
    mask = pd.Series(True, index=data.index)
    results = pd.DataFrame(untested, index=data.index, columns=qc_inputs.keys())

    for _, gdf in groups:
        group_mask = mask.loc[gdf.index].copy()

        for qc_name, qc in qc_inputs.items():
            if not group_mask.any():
                break

            args = {k: (v.loc[gdf.index] if isinstance(v, pd.Series) else v) for k, v in qc["requests"].items()}
            kwa = {k: (v.loc[gdf.index] if isinstance(v, pd.Series) else v) for k, v in qc["kwargs"].items()}

            full = _apply_qc_to_masked_rows(
                qc_func=qc["function"],
                args=args,
                kwargs=kwa,
                data_index=gdf.index,
                mask=group_mask,
            )

            results.loc[gdf.index, qc_name] = full

            if return_method == "failed":
                group_mask &= full != failed
                mask.loc[gdf.index] &= full != failed
            elif return_method == "passed":
                group_mask &= full != passed
                mask.loc[gdf.index] &= full != passed

    return results


def _do_multiple_check(
    data: pd.DataFrame | pd.Series,
    groupby: str | Iterable[str] | pd.core.groupby.generic.DataFrameGroupBy | None = None,
    qc_dict: dict | None = None,
    preproc_dict: dict | None = None,
    return_method: Literal["all", "passed", "failed"] = "all",
) -> pd.DataFrame | pd.Series:
    """
    Unified internal entry point for performing individual, sequential, or grouped
    QC checks by preparing inputs, constructing groups, and running the QC engine.
    """
    data, is_series = _validate_and_normalize_input(data, return_method)
    qc_inputs, mask, results = _prepare_all_inputs(data, qc_dict, preproc_dict)
    groups = _group_iterator(data, groupby)
    results = _run_qc_engine(data, qc_inputs, groups, return_method)
    return results.iloc[0] if is_series else results


def do_multiple_individual_check(
    data: pd.DataFrame | pd.Series,
    qc_dict: dict | None = None,
    preproc_dict: dict | None = None,
    return_method: Literal["all", "passed", "failed"] = "all",
) -> pd.DataFrame | pd.Series:
    """
    Apply one or more quality-control (QC) functions independently to each row of
    a DataFrame or Series.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Hashable input data.
    qc_dict : dict, optional
        Nested QC dictionary.
        Keys represent arbitrary user-specified names for the checks.
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
    pd.DataFrame or pd.Series
        A DataFrame (or Series if the input was a Series) whose columns correspond
        to the QC names in ``qc_dict`` and whose values contain QC flags for each row.
        Flags depend on the QC functions used.

    Raises
    ------
    NameError
        If a function listed in `qc_dict` or `preproc_dict` is not defined.
        If columns listed in `qc_dict` or `preproc_dict` are not available in `data`.
    ValueError
        If `return_method` is not one of ["all", "passed", "failed"]
        If variable names listed in `qc_dict` or `preproc_dict` are not valid
        parameters of the QC function.

    Note
    ----
    If a variable is pre-processed using `preproc_dict`, mark the variable name as
    "__preprocessed__" in `qc_dict`. For example: `"climatology": "__preprocessed__"`.

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

    Finally, run the function:

    .. code-block:: python

        do_multiple_individual_check(
            data=df,
            qc_dict=qc_dict,
            preproc_dict=preproc_dict,
            return_method="failed",
        )

    """
    return _do_multiple_check(
        data=data,
        groupby=None,
        qc_dict=qc_dict,
        preproc_dict=preproc_dict,
        return_method=return_method,
    )


def do_multiple_sequential_check(
    data: pd.DataFrame | pd.Series,
    groupby: str | Iterable[str] | pd.core.groupby.generic.DataFrameGroupBy | None = None,
    qc_dict: dict | None = None,
    preproc_dict: dict | None = None,
    return_method: Literal["all", "passed", "failed"] = "all",
) -> pd.DataFrame | pd.Series:
    """
    Apply one or more sequential quality-control (QC) functions to groups of
    a DataFrame or Series, typically for time-ordered or track-based checks.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Hashable input data.
    qc_dict : dict, optional
        Nested QC dictionary.
        Keys represent arbitrary user-specified names for the checks.
        The values are dictionaries which contain the keys "func" (name of the QC function),
        "names" (input data names as keyword arguments, that will be retrieved from `data`) and,
        if necessary, "arguments" (the corresponding keyword arguments).
    groupby : str, iterable of str, or pandas GroupBy, optional
        Specifies how the data should be grouped before applying QC functions.
        If a string or iterable of strings, ``data.groupby`` is called on those keys.
        If a ``pandas.DataFrameGroupBy`` object is provided, its groups are used
        directly. Any groups that contain indices not present in ``data`` are
        automatically trimmed.
        If ``None``, the entire input ``data`` is treated as a single group.
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
    pd.DataFrame or pd.Series
        A DataFrame (or Series if the input was a Series) whose columns correspond
        to the QC names in ``qc_dict`` and whose values contain QC flags for each row.
        Flags depend on the QC functions used.

    Raises
    ------
    NameError
        If a function listed in `qc_dict` or `preproc_dict` is not defined.
        If columns listed in `qc_dict` or `preproc_dict` are not available in `data`.
    ValueError
        If `return_method` is not one of ["all", "passed", "failed"]
        If variable names listed in `qc_dict` or `preproc_dict` are not valid
        parameters of the QC function.

    Note
    ----
    If a variable is pre-processed using `preproc_dict`, mark the variable name as
    "__preprocessed__" in `qc_dict`. For example: `"climatology": "__preprocessed__"`.

    For more information, see `do_multiple_individual_checks`.
    """
    return _do_multiple_check(
        data=data,
        groupby=groupby,
        qc_dict=qc_dict,
        preproc_dict=preproc_dict,
        return_method=return_method,
    )


def do_multiple_grouped_check(
    data: pd.DataFrame,
    qc_dict: dict | None = None,
    preproc_dict: dict | None = None,
    return_method: Literal["all", "passed", "failed"] = "all",
) -> pd.DataFrame:
    """
    Apply one or more buddy-check quality-control (QC) functions to a DataFrame or Series,
    where QC functions may compare rows against each other.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Hashable input data.
    qc_dict : dict, optional
        Nested QC dictionary.
        Keys represent arbitrary user-specified names for the checks.
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
    pd.DataFrame or pd.Series
        A DataFrame (or Series if the input was a Series) whose columns correspond
        to the QC names in ``qc_dict`` and whose values contain QC flags for each row.
        Flags depend on the QC functions used.

    Raises
    ------
    NameError
        If a function listed in `qc_dict` or `preproc_dict` is not defined.
        If columns listed in `qc_dict` or `preproc_dict` are not available in `data`.
    ValueError
        If `return_method` is not one of ["all", "passed", "failed"]
        If variable names listed in `qc_dict` or `preproc_dict` are not valid
        parameters of the QC function.

    Note
    ----
    If a variable is pre-processed using `preproc_dict`, mark the variable name as
    "__preprocessed__" in `qc_dict`. For example: `"climatology": "__preprocessed__"`.

    For more information, see `do_multiple_individual_checks`.
    """
    return _do_multiple_check(
        data=data,
        groupby=None,
        qc_dict=qc_dict,
        preproc_dict=preproc_dict,
        return_method=return_method,
    )
