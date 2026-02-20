"""Module containing base QC which call multiple QC functions and could be applied on a DataBundle."""

from __future__ import annotations
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Literal, cast

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
from .validations import (
    is_func_param,
    is_in_data,
    validate_args,
    validate_dict,
)


def _apply_qc_to_masked_rows(
    qc_func: Callable[..., Any],
    args: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    data_index: pd.Index,
    mask: pd.Series,
) -> pd.Series:
    """
    Apply a QC function to masked rows and return a Series aligned to ``data_index``.

    Parameters
    ----------
    qc_func : Callable
        QC function to execute.
    args : Mapping[str, Any]
        Keyword arguments constructed from requests.
    kwargs : Mapping[str, Any]
        Additional keyword arguments, typically from preprocessed variables.
    data_index : pandas.Index
        Full index of the dataset for aligning the QC result.
    mask : pandas.Series
        Boolean mask indicating which rows the QC function applies to.

    Returns
    -------
    pd.Series
        A Series indexed by ``data_index`` containing QC results for masked rows
        and default values elsewhere.
    """
    partial = qc_func(**args, **kwargs)

    partial = pd.Series(partial, index=data_index)

    full = pd.Series(untested, index=data_index)

    full.loc[mask] = partial.loc[mask]

    return full


def _run_qc_engine(
    data: pd.DataFrame | pd.Series,
    qc_inputs: Mapping[str, Any],
    groups: Iterable[tuple[Any | None, pd.DataFrame | pd.Series]],
    return_method: Literal["all", "passed", "failed"],
) -> pd.DataFrame | pd.Series:
    """
    Execute QC checks on the provided data groups and collect the results.

    Each QC function is applied to the corresponding group, respecting a
    shared mask that propagates pass/fail status. The results are stored
    in a DataFrame aligned with the original data.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Hashable input data.
    qc_inputs : Mapping
        Dictionary of QC inputs, each containing:
        {"function": callable, "requests": dict, "kwargs": dict}.
    groups : Iterable
        Iterable of (group_name, group_df) pairs, as returned by `_group_iterator`.
    return_method : {"all", "passed", "failed"}, default: "all"
        If "all", return QC dictionary containing all requested QC check flags.
        If "passed": return QC dictionary containing all requested QC check flags until the first check passes.
        Other QC checks are flagged as unstested (3).
        If "failed": return QC dictionary containing all requested QC check flags until the first check fails.
        Other QC checks are flagged as unstested (3).

    Returns
    -------
    pd.DataFrame
        DataFrame of QC results with the same index as `data` and columns
        corresponding to QC names.
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


def _normalize_groupby(
    data: pd.DataFrame | pd.Series,
    groupby: str | pd.core.groupby.generic.DataFrameGroupBy | None,
) -> list[tuple[Any, pd.DataFrame]]:
    """
    Return iterable of (name, group_df) pairs, trimming invalid rows.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Hashable input data.
    groupby : DataFrameGroupBy or object
        A groupby object or column(s) to group by. If None, the full DataFrame is returned as a single group.

    Returns
    -------
    list[tuple[Any, pd.DataFrame]]
        A list of tuples containing the group name (or None) and the corresponding DataFrame slice.
    """
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


def _group_iterator(
    data: pd.DataFrame | pd.Series,
    groupby: str | Iterable[str] | pd.core.groupby.generic.DataFrameGroupBy | None,
) -> Iterator[tuple[Any | None, pd.DataFrame | pd.Series]]:
    """
    Yield groups of a DataFrame as (group_name, group_df) pairs.

    If `groupby` is None, yields the entire DataFrame as a single group.
    Otherwise, yields each group as returned by `_normalize_groupby`.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        The DataFrame to iterate over in groups.
    groupby : str, iterable of str, DataFrameGroupBy, or None
        Column(s) or a groupby object to split `data` into groups. If None,
        the full DataFrame is returned as a single group.

    Yields
    ------
    tuple of (Any, pd.DataFrame)
        Tuples containing the group key (or None) and the corresponding
        DataFrame for that group.
    """
    if groupby is None:
        yield None, data
    else:
        yield from _normalize_groupby(data, groupby)


def _get_requests_from_params(
    params: Mapping[str, str] | None,
    func: Callable[..., Any],
    data: pd.Series | pd.DataFrame,
) -> Mapping[str, pd.Series | Any]:
    """
    Get requests from `func` or `data` using `params`.

    Given a dictionary of key value pairs where the keys are parameters in the function, func, and the values
    are columns or variables in data, create a new dictionary in which the keys are the parameter names (as in the
    original dictionary) and the values are the numbers extracted from data.

    Parameters
    ----------
    params : Mapping or None
        Dictionary. Keys are parameter names for the function func,
        and values are the names of columns or variables in data.
    func : Callable
        Function for which the parameters will be checked.
    data : pd.Series or pd.DataFrame
        DataSeries or DataFrame containing the data to be extracted.

    Returns
    -------
    Mapping
        Dictionary containing the key value pairs where the keys are as in the input dictionary and the values are
        extracted from the corresponding columns of data.

    Raises
    ------
    ValueError
        If one of the dictionary keys from params is not a valid argument in func.
    NameError
        If one of the dictionary values from params is not a column or variable in data.
    """
    requests: dict[str, pd.Series | Any] = {}
    if params is None:
        return requests
    for param, cname in params.items():
        if not is_func_param(func, param):
            raise ValueError(f"Parameter '{param}' is not a valid parameter of function '{func.__name__}'")
        if not is_in_data(cname, data):
            raise NameError(f"Variable '{cname}' is not available in input data: {data}.")
        requests[param] = data[cname]
    return requests


def _get_preprocessed_args(arguments: Mapping[str, str], preprocessed: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Update `arguments` for values available in `preprocessed`.

    Given a dictionary of key value pairs, if one of the values is equal to __preprocessed__ then replace
    the value with the value corresponding to that key in preprocessed.

    Parameters
    ----------
    arguments : Mapping
        Dictionary of key value pairs where the keys are variable names and the values are strings.
    preprocessed : dict
        Dictionary of key value pairs where the keys correspond to variable names.

    Returns
    -------
    Mapping
        Dictionary of key value pairs where values in arguments that were set to __preprocessed__ were replaced by
        values from the dictionary preprocessed.
    """
    args = {}
    for k, v in arguments.items():
        if v == "__preprocessed__":
            v = preprocessed[k]
        args[k] = v
    return args


def _get_function(name: str) -> Callable[..., Any]:
    """
    Return the function of a given name or raises a NameError.

    Parameters
    ----------
    name : str
        Name of the function to be returned.

    Returns
    -------
    Callable[..., Any]
        Function of a given name.

    Raises
    ------
    NameError
        If a callable with the given name does not exist.
    """
    func = globals().get(name)
    if not callable(func):
        raise NameError(f"Function '{name}' is not defined.")
    return cast(Callable[..., Any], func)


def _prepare_functions(
    config: Mapping[str, Mapping[str, Any]],
    data: pd.DataFrame | pd.Series,
    preprocessed: Mapping[str, Any] | None = None,
    execute: bool = False,
) -> Mapping[str, Any]:
    """
    Prepare functions defined in a configuration dictionary.

    Parameters
    ----------
    config : Mapping[str, Mapping[str, Any]]
        Dictionary describing functions, their inputs, and arguments.
    data : pd.DataFrame or pd.Series
        Data used to extract requested parameters.
    preprocessed : Mapping[str, Any], optional
        Previously computed preprocessed variables (used for QC functions).
    execute : bool, default: False
        If True, execute the functions and return their results.
        If False, return function references and resolved arguments.

    Returns
    -------
    Mapping[str, Any]
        If `execute=True`, returns a dict mapping names to results.
        If `execute=False`, returns a dict mapping names to dicts:
        `{"function": callable, "requests": dict, "kwargs": dict}`.
    """
    validate_dict(config)

    results: dict[str, Any] = {}

    for name, params in config.items():
        if "func" not in params:
            raise ValueError(f"'func' is not specified in {params}.")

        func = _get_function(params["func"])

        args = params.get("inputs", [])
        if not isinstance(args, (list, tuple)):
            args = (args,)

        arguments = params.get("arguments", {})
        if preprocessed is not None:
            arguments = _get_preprocessed_args(arguments, preprocessed)

        requests = _get_requests_from_params(params.get("names"), func, data)

        kwargs = {**requests, **arguments}

        validate_args(func, args=args, kwargs=kwargs)

        if execute:
            results[name] = func(*args, **kwargs)
        else:
            results[name] = {"function": func, "requests": requests, "kwargs": arguments}

    return results


def _prepare_all_inputs(
    data: pd.DataFrame | pd.Series,
    qc_dict: Mapping[str, Any] | None,
    preproc_dict: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], pd.Series, pd.DataFrame]:
    """
    Build all inputs required for QC execution.

    This includes preporcessed variables, resolved QC function arguments, an initial boolean mask,
    and an empty results table.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Hashable input data.
    qc_dict : Mapping or None
        Dictionary defining QC functions and their arguments.
    preproc_dict : Mapping or None
        Dictionary defining preprocessing steps.

    Returns
    -------
    tuple of (Mapping, pd.Series, pd.DataFrame)
        - QC inputs dictionary: {qc_name: {function, requests, kwargs}}.
        - Initial boolean mask Series (all True).
        - Empty results DataFrame with shape (n_rows, n_qcs).
    """
    qc_dict = qc_dict or {}
    preproc_dict = preproc_dict or {}

    preprocessed = _prepare_functions(preproc_dict, data, execute=True)
    qc_inputs = _prepare_functions(qc_dict, data, preprocessed=preprocessed)

    mask = pd.Series(True, index=data.index)
    results = pd.DataFrame(untested, index=data.index, columns=qc_inputs.keys())

    return qc_inputs, mask, results


def _normalize_input(
    data: pd.DataFrame | pd.Series,
    return_method: Literal["all", "passed", "failed"],
) -> tuple[pd.DataFrame, bool]:
    """
    Validate the return method and ensure the input is a DataFrame.

    Converts a Series to a single-column DataFrame and tracks if the original
    input was a Series.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Hashable input data.
    return_method : {'all', 'passed', 'failed'}
        Specifies which rows to return; must be one of 'all', 'passed', or 'failed'.

    Returns
    -------
    tuple of (pd.DataFrame, bool)
        - Normalized DataFrame version of the input.
        - Boolean indicating if the original input was a Series.
    """
    if return_method not in ("all", "passed", "failed"):
        raise ValueError("'return_method' must be 'all','passed','failed'.")
    is_series = isinstance(data, pd.Series)
    if is_series:
        data = data.to_frame()
    return data, is_series


def _do_multiple_check(
    data: pd.DataFrame | pd.Series,
    groupby: str | Iterable[str] | pd.core.groupby.generic.DataFrameGroupBy | None = None,
    qc_dict: Mapping[str, Any] | None = None,
    preproc_dict: Mapping[str, Any] | None = None,
    return_method: Literal["all", "passed", "failed"] = "all",
) -> pd.DataFrame | pd.Series:
    """
    Internal entry point for performing QC checks on data.

    Prepares inputs, constructs groups, and executes the QC engine
    for individual, sequential, or grouped checks.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Hashable input data.
    groupby : str, iterable of str, or pandas GroupBy, optional
        Specifies how the data should be grouped before applying QC functions.
        If a string or iterable of strings, ``data.groupby`` is called on those keys.
        If a ``pandas.DataFrameGroupBy`` object is provided, its groups are used
        directly. Any groups that contain indices not present in ``data`` are
        automatically trimmed.
        If ``None``, the entire input ``data`` is treated as a single group.
    qc_dict : Mapping, optional
        Nested QC dictionary.
        Keys represent arbitrary user-specified names for the checks.
        The values are dictionaries which contain the keys "func" (name of the QC function),
        "names" (input data names as keyword arguments, that will be retrieved from `data`) and,
        if necessary, "arguments" (the corresponding keyword arguments).
        For more information see Examples.
    preproc_dict : Mapping, optional
        Nested pre-processing dictionary.
        Keys represent variable names that can be used by `qc_dict`.
        The values are dictionaries which contain the keys "func" (name of the pre-processing function),
        "names" (input data names as keyword arguments, that will be retrieved from `data`), and "inputs"
        (list of input-given variables).
        For more information see Examples.
    return_method : {"all", "passed", "failed"}, default: "all"
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
    """
    data, is_series = _normalize_input(data, return_method)
    qc_inputs, mask, results = _prepare_all_inputs(data, qc_dict, preproc_dict)
    groups = _group_iterator(data, groupby)
    results = _run_qc_engine(data, qc_inputs, groups, return_method)
    return results.iloc[0] if is_series else results


def do_multiple_individual_check(
    data: pd.DataFrame | pd.Series,
    qc_dict: Mapping[str, Any] | None = None,
    preproc_dict: Mapping[str, Any] | None = None,
    return_method: Literal["all", "passed", "failed"] = "all",
) -> pd.DataFrame | pd.Series:
    """
    Apply one or more quality-control (QC) functions independently to each row of a DataFrame or Series.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Hashable input data.
    qc_dict : Mapping, optional
        Nested QC dictionary.
        Keys represent arbitrary user-specified names for the checks.
        The values are dictionaries which contain the keys "func" (name of the QC function),
        "names" (input data names as keyword arguments, that will be retrieved from `data`) and,
        if necessary, "arguments" (the corresponding keyword arguments).
        For more information see Examples.
    preproc_dict : Mapping, optional
        Nested pre-processing dictionary.
        Keys represent variable names that can be used by `qc_dict`.
        The values are dictionaries which contain the keys "func" (name of the pre-processing function),
        "names" (input data names as keyword arguments, that will be retrieved from `data`), and "inputs"
        (list of input-given variables).
        For more information see Examples.
    return_method : {"all", "passed", "failed"}, default: "all"
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

    Notes
    -----
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
    qc_dict: Mapping[str, Any] | None = None,
    preproc_dict: Mapping[str, Any] | None = None,
    return_method: Literal["all", "passed", "failed"] = "all",
) -> pd.DataFrame | pd.Series:
    """
    Apply one or more sequential quality-control (QC) functions to groups of a DataFrame or Series.

    Typically for time-ordered or track-based checks.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Hashable input data.
    groupby : str, iterable of str, or pandas GroupBy, optional
        Specifies how the data should be grouped before applying QC functions.
        If a string or iterable of strings, ``data.groupby`` is called on those keys.
        If a ``pandas.DataFrameGroupBy`` object is provided, its groups are used
        directly. Any groups that contain indices not present in ``data`` are
        automatically trimmed.
        If ``None``, the entire input ``data`` is treated as a single group.
        For more information see Examples.
    qc_dict : Mapping, optional
        Nested QC dictionary.
        Keys represent arbitrary user-specified names for the checks.
        The values are dictionaries which contain the keys "func" (name of the QC function),
        "names" (input data names as keyword arguments, that will be retrieved from `data`) and,
        if necessary, "arguments" (the corresponding keyword arguments).
    preproc_dict : Mapping, optional
        Nested pre-processing dictionary.
        Keys represent variable names that can be used by `qc_dict`.
        The values are dictionaries which contain the keys "func" (name of the pre-processing function),
        "names" (input data names as keyword arguments, that will be retrieved from `data`), and "inputs"
        (list of input-given variables).
        For more information see Examples.
    return_method : {"all", "passed", "failed"}, default: "all"
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

    Notes
    -----
    If a variable is pre-processed using `preproc_dict`, mark the variable name as
    "__preprocessed__" in `qc_dict`. For example: `"climatology": "__preprocessed__"`.

    For more information, see :py:func:`do_multiple_individual_checks`.
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
    qc_dict: Mapping[str, Any] | None = None,
    preproc_dict: Mapping[str, Any] | None = None,
    return_method: Literal["all", "passed", "failed"] = "all",
) -> pd.DataFrame | pd.Series:
    """
    Apply one or more buddy-check quality-control (QC) functions to a DataFrame or Series.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Hashable input data.
    qc_dict : Mapping, optional
        Nested QC dictionary.
        Keys represent arbitrary user-specified names for the checks.
        The values are dictionaries which contain the keys "func" (name of the QC function),
        "names" (input data names as keyword arguments, that will be retrieved from `data`) and,
        if necessary, "arguments" (the corresponding keyword arguments).
        For more information see Examples.
    preproc_dict : Mapping, optional
        Nested pre-processing dictionary.
        Keys represent variable names that can be used by `qc_dict`.
        The values are dictionaries which contain the keys "func" (name of the pre-processing function),
        "names" (input data names as keyword arguments, that will be retrieved from `data`), and "inputs"
        (list of input-given variables).
        For more information see Examples.
    return_method : {"all", "passed", "failed"}, default: "all"
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

    Notes
    -----
    If a variable is pre-processed using `preproc_dict`, mark the variable name as
    "__preprocessed__" in `qc_dict`. For example: `"climatology": "__preprocessed__"`.

    For more information, see :py:func:`do_multiple_individual_checks`.
    """
    return _do_multiple_check(
        data=data,
        groupby=None,
        qc_dict=qc_dict,
        preproc_dict=preproc_dict,
        return_method=return_method,
    )
