"""Common Data Model (CDM) pandas duplicate check."""

from __future__ import annotations
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import recordlinkage as rl

from ..helpers.auxiliary import (
    SequenceDatetimeType,
    SequenceIntType,
    SequenceNumberType,
    SequenceStrType,
    is_scalar_like,
    post_format_return_type,
)
from ._duplicate_settings import Compare, _compare_kwargs, _method_kwargs


def convert_series(df: pd.DataFrame, conversion: dict[Any, Any]) -> pd.DataFrame:
    """
    Convert data types in Dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    conversion : dict
        Conversion dictionary conating columns and
        new data type as key-value pairs.

    Returns
    -------
    pd.DataFrame
        DataFrame with converted data types.
    """

    def convert_date_to_float(date: pd.Series | pd.DatetimeIndex) -> pd.Series:
        """
        Convert datetime values to float seconds relative to the minimum value.

        Parameters
        ----------
        date : pd.Series or pd.DatetimeIndex
            Datetime-like values to convert.

        Returns
        -------
        pd.Series
            Float values representing seconds since the minimum datetime in `date`.
        """
        date = date.astype("datetime64[ns]")
        return (date - date.min()) / np.timedelta64(1, "s")

    df = df.copy()
    for column, method in conversion.items():
        try:
            df[column] = df[column].astype(method)
        except TypeError:
            df[column] = locals()[method](df[column])

    df = df.infer_objects(copy=False).fillna(9999.0)
    return df


class DupDetect:
    """
    Class to detect, flag, and remove duplicate entries in a DataFrame using a comparison matrix from recordlinkage.

    Parameters
    ----------
    data : pd.DataFrame
        Original dataset.
    compared : pd.DataFrame
        Comparison matrix of the dataset.
    method : str
        Duplicate detection method used for recordlinkage indexing.
    method_kwargs : dict
        Keyword arguments for recordlinkage indexing method.
    compare_kwargs : dict
        Keyword arguments used for recordlinkage.Compare.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        compared: pd.DataFrame,
        method: str,
        method_kwargs: dict[Any, Any],
        compare_kwargs: dict[Any, Any],
    ) -> None:
        """
        Initialize a DupDetect instance.

        Parameters
        ----------
        data : pd.DataFrame
          Original dataset.
        compared : pd.DataFrame
          Comparison matrix of the dataset.
        method : str
          Duplicate detection method used for recordlinkage indexing.
        method_kwargs : dict
          Keyword arguments for recordlinkage indexing method.
        compare_kwargs : dict
          Keyword arguments used for recordlinkage.Compare.
        """
        self.data = data.copy()
        self.compared = compared
        self.method = method
        self.method_kwargs = method_kwargs
        self.compare_kwargs = compare_kwargs

    def _get_limit(self, limit: str | float | None) -> float:
        """
        Resolve the duplicate threshold limit.

        Parameters
        ----------
        limit : str or float
            'default', None, or a numeric limit.

        Returns
        -------
        float
            Threshold for total score to consider duplicates.
        """
        default_limit = 0.991
        if limit is None or limit == "default":
            return default_limit

        return float(limit)

    def _get_equal_musts(self) -> list[str]:
        """
        Identify columns that must be equal for duplicates.

        Returns
        -------
        list[str]
            Columns that must match exactly to consider duplicates.
        """
        equal_musts: list[str] = []
        for value in self.compare_kwargs.keys():
            if isinstance(value, str):
                value_lst = [value]
            else:
                value_lst = list(value)
            equal_musts.extend(v for v in value_lst if v in self.data.columns)
        return equal_musts

    def _total_score(self) -> None:
        """Compute total similarity score for each row in `self.compared`."""
        pcmax = self.compared.shape[1]
        self.score = 1 - (abs(self.compared.sum(axis=1) - pcmax) / pcmax)

    def get_duplicates(
        self,
        keep: str | int = "first",
        limit: str | float | None = "default",
        equal_musts: str | list[str] | None = None,
        overwrite: bool = True,
    ) -> pd.DataFrame:
        """
        Identify duplicate matches based on the comparison matrix.

        Parameters
        ----------
        keep : str or int
            Which entry to keep: 'first', 'last', or -1, 0.
        limit : str or float, optional, default: default
            Threshold of total similarity score to consider as duplicate.
        equal_musts : str or list[str], optional
            Columns that must exactly match.
        overwrite : bool, default: True
            Whether to recompute matches if already calculated.

        Returns
        -------
        pd.DataFrame
            DataFrame containing matched duplicates.
        """
        if keep not in ["first", "last", -1, 0]:
            raise ValueError("keep has to be one of 'first', 'last', -1 or 0.")

        if keep == "first":
            keep = -1
        elif keep == "last":
            keep = 0

        self.keep = keep
        if keep == 0:
            self.drop = -1
        elif keep == -1:
            self.drop = 0

        if overwrite is True:
            self._total_score()
            self.limit = self._get_limit(limit)
            cond = self.score >= self.limit
            if equal_musts is None:
                equal_musts = self._get_equal_musts()
            if isinstance(equal_musts, str):
                equal_musts = [equal_musts]
            for must in equal_musts:
                cond = cond & (self.compared[must])
            self.matches = self.compared[cond]
        return self.matches

    def flag_duplicates(
        self,
        keep: str | int = "first",
        limit: str | float | None = "default",
        equal_musts: str | list[str] | None = None,
    ) -> tuple[pd.Series, pd.Series]:
        r"""
        Get result dataset with flagged duplicates.

        Parameters
        ----------
        keep : str or int, default: first
            Which entry should be kept in result dataset.
        limit : str, int or float, optional
            Limit of total score that as to be exceeded to be declared as a duplicate.
            Defaults to .991.
        equal_musts : str or list, optional
            Hashable of column name(s) that must totally be equal to be declared as a duplicate.
            Default: All column names found in method_kwargs.

        Returns
        -------
        tuple of pd.Series
            Tuple containing duplicate flags and indexes of corresponding duplicates.

        References
        ----------
        .. _duplicate_status: https://glamod.github.io/cdm-obs-documentation/tables/code_tables/duplicate_status/duplicate_status.html
        .. _quality_flag: https://glamod.github.io/cdm-obs-documentation/tables/code_tables/quality_flag/quality_flag.html
        """

        def _get_similars(drop_dict: dict[str | int, Any], keeps: Any) -> tuple[Any, Any]:
            """
            Get similar entries from a comparison dictionary.

            Parameters
            ----------
            drop_dict : dict
                Dictionary containing values under keys `drop_` and `keep_` used
                to determine similarity relationships.
            keeps : Any
                Reference collection used to determine whether a value in `drop` is
                considered a match.

            Returns
            -------
            tuple of Any and Any
              A tuple containing the matched `drop` and `keep` values converted
              to integers if possible. If the values are not convertible or no match
              is found, returns `(None, None)`.
            """
            if drop_dict[drop] in keeps:
                drops, keeps = drop_dict[drop], drop_dict[keep]
                try:
                    return int(drops), int(keeps)
                except ValueError:
                    return drops, keeps

            return None, None

        def _get_duplicates(x: pd.DataFrame, last: Any) -> pd.Series:
            """
            Extract unique duplicate values from a DataFrame column.

            Parameters
            ----------
            x : pd.DataFrame
                Input DataFrame containing the column to inspect.
            last : Any
                Column name used to extract values for duplicate detection.

            Returns
            -------
            pd.Series
                Series containing a single key "dups" with the list of unique
                duplicate values found in the specified column.
            """
            return pd.Series({"duplicates": list(sorted(set(x[last].values)))})

        def _delete_values_equal_keys(dictionary: dict[Any, Any]) -> tuple[dict[Any, Any], list[Any]]:
            """
            Remove entries where keys and values are identical.

            Parameters
            ----------
            dictionary : dict
                Input mapping of keys to values.

            Returns
            -------
            tuple of dict and list of Any
                A tuple containing:
                - A filtered dictionary with identical key-value pairs removed
                - A list of values that were removed because key == value
            """
            new_dictionary, drops = {}, []
            for k, v in dictionary.items():
                if k == v:
                    drops.append(v)
                else:
                    new_dictionary[k] = v
            return new_dictionary, drops

        def replace_keeps_and_drops(df: pd.DataFrame, keep: Any) -> pd.DataFrame:
            """
            Iteratively resolve and replace duplicate mappings in a DataFrame.

            Parameters
            ----------
            df : pd.DataFrame
                Input DataFrame containing values to be deduplicated.
            keep : Any
                Column name used to identify canonical ("keep") values.

            Returns
            -------
            pd.DataFrame
                Updated DataFrame with resolved duplicate mappings and cleaned
                keep-column values.
            """
            keeps = df[keep].values
            while True:
                df = df.sort_index()
                replaces = df.apply(lambda row, keeps=keeps: _get_similars(row, keeps), axis=1)
                replaces = {k: v for k, v in dict(replaces.values).items() if k is not None}
                replaces, drops = _delete_values_equal_keys(replaces)
                if drops:
                    df = df.drop(drops, axis="index")
                df[keep] = df[keep].replace(replaces)
                if not set(replaces.keys()).intersection(replaces.values()):
                    return df

        self.get_duplicates(keep=keep, limit=limit, equal_musts=equal_musts)

        if not hasattr(self, "matches"):
            self.get_duplicates(limit="default", equal_musts=equal_musts)

        indexes = self.matches.index
        indexes_df = indexes.to_frame()
        drop = indexes_df.columns[self.drop]
        keep = indexes_df.columns[self.keep]
        indexes_df = indexes_df.drop_duplicates(subset=[drop])
        indexes_df = replace_keeps_and_drops(indexes_df, keep)

        dup_keep = (
            indexes_df.groupby(indexes_df[keep])
            .apply(
                lambda x: _get_duplicates(x, drop),
                include_groups=False,
            )
            .iloc[:, 0]
        )
        dup_drop = (
            indexes_df.groupby(indexes_df[drop])
            .apply(
                lambda x: _get_duplicates(x, keep),
                include_groups=False,
            )
            .iloc[:, 0]
        )

        indexes_good = indexes_df[keep].values.tolist()
        indexes_bad = indexes_df[drop].values.tolist()

        flags = pd.Series([0] * len(self.data), index=self.data.index, name="duplicate_flags")
        flags.loc[indexes_good] = 1
        flags.loc[indexes_bad] = 3

        duplicates = pd.Series([np.nan] * len(self.data), index=self.data.index, name="duplicates", dtype="object")

        duplicates.loc[indexes_good] = dup_keep.loc[indexes_good]
        duplicates.loc[indexes_bad] = dup_drop.loc[indexes_bad]

        return flags, duplicates

    def remove_duplicates(
        self,
        keep: str | int = "first",
        limit: str | float | None = "default",
        equal_musts: str | list[str] | None = None,
    ) -> tuple[pd.Series, ...]:
        """
        Remove duplicate entries from the dataset.

        Parameters
        ----------
        keep : str or int
            Which entry to keep ('first' or 'last').
        limit : str or float, optional
            Minimum similarity score to declare duplicates.
        equal_musts : str or list[str], optional
            Columns that must exactly match.

        Returns
        -------
        tuple of pd.Series
            A tuple of pd.Series containing all original input data with the duplicates removed.
        """
        self.get_duplicates(keep=keep, limit=limit, equal_musts=equal_musts)
        result = self.data.drop(self.matches.index.get_level_values(self.drop))
        result = result.sort_index(ascending=True)
        return tuple(result[col] for col in result.columns)


def set_comparer(compare_dict: dict[Any, Any]) -> Compare:
    """
    Build a recordlinkage Compare object with optional conversion dictionary.

    Parameters
    ----------
    compare_dict : dict
        Dictionary of columns to compare,
        e.g. {"column_name": {"method": "exact" | "numeric" | "date2", "kwargs": {...}}}.

    Returns
    -------
    recordlinkage.Compare
        Compare object with added comparison methods and a 'conversion' attribute.
    """
    comparer = Compare()
    comparer.conversion = {}
    for column, c_dict in compare_dict.items():
        try:
            method = c_dict["method"]
        except KeyError as err:
            raise KeyError(
                "compare_kwargs must be hierarchically ordered: {<column_name>: {'method': <compare_method>}}. 'method' not found"
            ) from err
        try:
            kwargs = c_dict["kwargs"]
        except KeyError:
            kwargs = {}
        getattr(comparer, method)(
            column,
            column,
            label=column,
            **kwargs,
        )
        if method == "numeric":
            comparer.conversion[column] = float
        if method == "date":
            comparer.conversion[column] = "datetime64[ns]"
        if method == "date2":
            comparer.conversion[column] = "convert_date_to_float"

    return comparer


def remove_ignores(dic: dict[Any, Any], columns: str | list[str]) -> dict[Any, Any]:
    """
    Remove dictionary entries where keys or values match ignored columns.

    Parameters
    ----------
    dic : dict
        Original dictionary to filter.
    columns : str or list[str]
        Column(s) to ignore.

    Returns
    -------
    dict
        Filtered dictionary without the ignored columns.
    """
    new_dict = {}
    if isinstance(columns, str):
        columns = [columns]
    for k, v in dic.items():
        if k in columns:
            continue
        if v in columns:
            continue
        if isinstance(v, list):
            v2 = [v_ for v_ in v if v_ not in columns]
            if len(v2) == 0:
                continue
            v = v2
        new_dict[k] = v
    return new_dict


def change_offsets(dic: dict[Any, Any], dic_o: dict[Any, Any]) -> dict[Any, Any]:
    """
    Update the 'offset' value in compare dictionary kwargs.

    Parameters
    ----------
    dic : dict
        Original compare dictionary.
    dic_o : dict
        Dictionary mapping column names to new offsets.

    Returns
    -------
    dict
        Updated compare dictionary with modified offsets.
    """
    for key in dic.keys():
        if key not in dic_o.keys():
            continue
        dic[key]["kwargs"]["offset"] = dic_o[key]
    return dic


def reindex_nulls(df: pd.DataFrame, null_label: Any) -> pd.DataFrame:
    """
    Reindex a DataFrame in ascending order based on the number of 'null' strings in each row.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Cells with the string "null" are counted as nulls.
    null_label : Any
        Missing value representative.

    Returns
    -------
    pd.DataFrame
        DataFrame reindexed so that rows with fewer 'null' values appear first.
        Original row order is preserved for rows with the same null count.
    """

    def is_missing(x: Any) -> bool:
        """
        Determine whether a value is considered missing.

        This function supports scalar values as well as nested iterables
        (lists, tuples, numpy arrays). A value is considered missing if it is:
        - NaN (as defined by ``pandas.isna``)
        - Equal to ``null_label``
        - Any element inside an iterable is missing (recursively checked)

        Parameters
        ----------
        x : Any
            Value to check for missingness.

        Returns
        -------
        bool
          True if the value (or any nested value) is missing, otherwise False.
        """
        if isinstance(x, (list, tuple, np.ndarray)):
            return any(is_missing(v) for v in x)

        if pd.isna(x):
            return True

        if x == null_label:
            return True

        return False

    def count_nulls(row: pd.Series) -> int:
        """
        Count the number of missing values in a pandas Series.

        Parameters
        ----------
        row : pd.Series
            Input row or Series to evaluate.

        Returns
        -------
        int
            Number of missing values in the Series.
        """
        return sum(is_missing(x) for x in row)

    null_counts = df.apply(count_nulls, axis=1)

    if null_counts.empty:
        return df

    sorted_index = null_counts.sort_values(kind="stable").index
    return df.loc[sorted_index]


class Comparer:
    """
    Wrapper around recordlinkage.Compare to compute pairwise comparisons on a DataFrame.

    This class initializes a recordlinkage indexer and Compare object, optionally converting
    the data types before computing the comparisons.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to compare.
    method : str
        The indexing method from `recordlinkage.index`, e.g., 'SortedNeighbourhood'.
    method_kwargs : dict
        Keyword arguments to pass to the indexing method.
    compare_kwargs : dict
        Dictionary specifying columns and comparison methods for recordlinkage.Compare.
    pairs_df : list[pd.DataFrame], optional
        Optional pre-split DataFrames to pass to the indexer. Defaults to `[data]`.
    convert_data : bool, default False
        Whether to convert data using `compare_kwargs` conversion dictionary.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        method: str,
        method_kwargs: dict[Any, Any],
        compare_kwargs: dict[Any, Any],
        pairs_df: list[pd.DataFrame] | None = None,
        convert_data: bool = False,
    ):
        """
        Initialize a Comparer instance.

        Parameters
        ----------
        data : pd.DataFrame
          The dataset to compare.
        method : str
          The indexing method from `recordlinkage.index`, e.g., 'SortedNeighbourhood'.
        method_kwargs : dict
          Keyword arguments to pass to the indexing method.
        compare_kwargs : dict
          Dictionary specifying columns and comparison methods for recordlinkage.Compare.
        pairs_df : list[pd.DataFrame], optional
          Optional pre-split DataFrames to pass to the indexer. Defaults to `[data]`.
        convert_data : bool, default False
          Whether to convert data using `compare_kwargs` conversion dictionary.
        """
        compare_kwargs = {k: v for k, v in compare_kwargs.items() if k in data.columns}

        indexer = getattr(rl.index, method)(**method_kwargs)
        comparer = set_comparer(compare_kwargs)
        if convert_data is True:
            data_cp = convert_series(data, comparer.conversion)
        else:
            data_cp = data.copy()

        if pairs_df is None:
            pairs_df = [data_cp]
        pairs = indexer.index(*pairs_df)
        self.compared = comparer.compute(pairs, data_cp)
        self.data = data_cp


def duplicate_check(
    station_id: SequenceStrType,
    lat: SequenceNumberType,
    lon: SequenceNumberType,
    date: SequenceDatetimeType,
    vsi: SequenceNumberType,
    dsi: SequenceNumberType,
    obs: SequenceNumberType | list[SequenceNumberType] | None = None,
    method: str = "SortedNeighbourhood",
    method_kwargs: dict[Any, Any] | None = None,
    compare_kwargs: dict[Any, Any] | None = None,
    ignore_columns: str | None = None,
    ignore_entries: dict[str, Any] | None = None,
    offsets: dict[str, Any] | None = None,
    reindex_by_null: bool = True,
    null_label: Any = "null",
    **kwargs: dict[str, Any],
) -> DupDetect:
    r"""
    Run a duplicate check on a dataset using recordlinkage.

    Returns a DupDetect object.

    Parameters
    ----------
    station_id : :py:obj:`~marine_qc.SequenceStrType`
        One-dimensional array of station IDs.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    lat : :py:obj:`~marine_qc.SequenceNumberType`
        One-dimensional array of latitudes in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    lon : :py:obj:`~marine_qc.SequenceNumberType`
        One-dimensional array of longitudes in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    date : :py:obj:`~marine_qc.SequenceDatetimeType`
        One-dimensional array of datetime values.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    vsi : :py:obj:`~marine_qc.SequenceNumberType`
        One-dimensional reported speed array in km/h.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    dsi : :py:obj:`~marine_qc.SequenceNumberType`
        One-dimensional reported heading array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    obs : :py:obj:`~marine_qc.SequenceNumberType` or list of :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional reported observation value.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    method : str, default: SortedNeighbourhood
        Duplicate check method for recordlinkage.
    method_kwargs : dict, optional
        Keyword arguments for recordlinkage duplicate check.
        Defaults to _method_kwargs.
    compare_kwargs : dict, optional
        Keyword arguments for recordlinkage.Compare object.
        Defaults to _compare_kwargs.
    ignore_columns : str or list, optional
        Name of data columns to be ignored for duplicate check.
    ignore_entries : dict, optional
        Key: Column name.
        Value: value to be ignored.
        E.g. offsets={"station_speed": null}.
    offsets : dict, optional
        Change offsets for recordlinkage Compare object.
        Key: Column name.
        Value: new offset.
        E.g. offsets={"latitude": 0.1}.
    reindex_by_null : bool, optional
        If True data is re-indexed in ascending order according to the number of nulls in each row.
    null_label : str, optional
        Null label which is used if `reindex_by_null` is True.
    \**kwargs : dict
        Additional data to be used for checking duplicates.

    Returns
    -------
    cdm_reader_mapper.DupDetect
        A DupDetect instance.
    """
    if not method_kwargs:
        method_kwargs = deepcopy(_method_kwargs)
    if not compare_kwargs:
        compare_kwargs = deepcopy(_compare_kwargs)

    data = pd.DataFrame(
        {
            "station_id": station_id,
            "lat": lat,
            "lon": lon,
            "date": date,
            "vsi": vsi,
            "dsi": dsi,
        }
    )
    if obs is not None:
        if is_scalar_like(obs[0]):
            data["obs"] = obs
        else:
            for i in range(len(obs)):
                data[f"obs_{i + 1}"] = obs[i]
                compare_kwargs[f"obs_{i + 1}"] = deepcopy(compare_kwargs["obs"])
            del compare_kwargs["obs"]

    data = data.assign(**kwargs)
    index = data.index

    if reindex_by_null is True:
        data = reindex_nulls(data, null_label=null_label)

    data.reset_index(drop=True)

    dtypes = data.dtypes

    if ignore_columns:
        method_kwargs = remove_ignores(method_kwargs, ignore_columns)
        compare_kwargs = remove_ignores(compare_kwargs, ignore_columns)
    if offsets:
        compare_kwargs = change_offsets(compare_kwargs, offsets)

    comparer = Comparer(
        data=data,
        method=method,
        method_kwargs=method_kwargs,
        compare_kwargs=compare_kwargs,
        convert_data=True,
    )
    compared = comparer.compared

    if ignore_entries is None:
        data.set_index(index, inplace=True)
        return DupDetect(data, compared, method, method_kwargs, compare_kwargs)

    compared = [compared]
    data_ = comparer.data
    for column_, entry_ in ignore_entries.items():
        if not isinstance(entry_, list):
            entry_ = [entry_]
        entries = data[column_].isin(entry_)

        d1 = data.mask(entries).dropna(how="all")
        d2 = data.where(entries).dropna(how="all")
        if d1.empty:
            continue
        if d2.empty:
            continue

        method_kwargs_ = remove_ignores(method_kwargs, column_)
        compare_kwargs_ = remove_ignores(compare_kwargs, column_)

        compared_ = Comparer(
            data=data_,
            method=method,
            method_kwargs=method_kwargs_,
            compare_kwargs=compare_kwargs_,
            pairs_df=[d2, d1],
        ).compared
        compared_[list(ignore_entries.keys())] = 1
        compared.append(compared_)

    compared = pd.concat(compared)
    data.set_index(index, inplace=True)
    data = data.astype(dtypes)
    return DupDetect(data, compared, method, method_kwargs, compare_kwargs)


@post_format_return_type(["detected", "station_id", "lat", "lon", "date", "vsi", "dsi"], multiple=True, dtype=None, keep_index=True)
def remove_duplicates(
    station_id: SequenceStrType | None = None,
    lat: SequenceNumberType | None = None,
    lon: SequenceNumberType | None = None,
    date: SequenceDatetimeType | None = None,
    vsi: SequenceNumberType | None = None,
    dsi: SequenceNumberType | None = None,
    obs: SequenceNumberType | list[SequenceNumberType] | None = None,
    detected: DupDetect | None = None,
    keep: str | int = "first",
    limit: str | float | None = "default",
    equal_musts: str | list[str] | None = None,
    **kwargs: Any,
) -> tuple[
    SequenceStrType,
    SequenceNumberType,
    SequenceNumberType,
    SequenceDatetimeType,
    SequenceNumberType,
    SequenceNumberType,
    *tuple[SequenceNumberType, ...],
]:
    r"""
    Remove duplicates from inputs.

    Parameters
    ----------
    station_id : :py:obj:`~marine_qc.SequenceStrType`
        One-dimensional array of station IDs.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    lat : :py:obj:`~marine_qc.SequenceNumberType`
        One-dimensional array of latitudes in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    lon : :py:obj:`~marine_qc.SequenceNumberType`
        One-dimensional array of longitudes in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    date : :py:obj:`~marine_qc.SequenceDatetimeType`
        One-dimensional array of datetime values.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    vsi : :py:obj:`~marine_qc.SequenceNumberType`
        One-dimensional reported speed array in km/h.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    dsi : :py:obj:`~marine_qc.SequenceNumberType`
        One-dimensional reported heading array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    obs : :py:obj:`~marine_qc.SequenceNumberType` or list of :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional reported observation value.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    detected : :py:obj:`DupDetect`
        A `DupDetect` instance that already contains detected duplicates to flag.
    keep : str or int, default: first
        Which entry should be kept in result dataset.
    limit : str, int or float, optional
        Limit of total score that as to be exceeded to be declared as a duplicate.
        Defaults to .991.
    equal_musts : str or list, optional
        Hashable of column name(s) that must totally be equal to be declared as a duplicate.
    \**kwargs : Any
        Any keyword-arguments passed to `duplicate_check`.

    Returns
    -------
    tuple of Any
        All inputs with removed duplicates.

    Raises
    ------
    ValueError
        If none of `detected`, `station_id`, `lat`, `lon`, `date`, `vsi` and `dsi` is set.

    Notes
    -----
    If `detected` is set, `station_id`, `lat`, `lon`, `date`, `vsi` and `dsi` are ignored.
    If `detected` is set, the function always returns a tuple of pd.Series.
    """
    if all(x is None for x in (detected, station_id, lat, lon, date, vsi, dsi)):
        raise ValueError(
            "None of `detected`, `station_id`, `lat`, `lon`, `date`, `vsi` and `dsi` is set."
            "Set `dupdetect` or at least one of `station_id`, `lat`, `lon`, `date`, `vsi` and `dsi`."
        )

    if detected is None:
        detected = duplicate_check(station_id, lat, lon, date, vsi, dsi, obs, **kwargs)

    return detected.remove_duplicates(keep=keep, limit=limit, equal_musts=equal_musts)


@post_format_return_type(["detected", "station_id", "lat", "lon", "date", "vsi", "dsi"], multiple=True, dtype=[int, object])
def flag_duplicates(
    station_id: SequenceStrType | None = None,
    lat: SequenceNumberType | None = None,
    lon: SequenceNumberType | None = None,
    date: SequenceDatetimeType | None = None,
    vsi: SequenceNumberType | None = None,
    dsi: SequenceNumberType | None = None,
    obs: SequenceNumberType | list[SequenceNumberType] | None = None,
    detected: DupDetect | None = None,
    keep: str | int = "first",
    limit: str | float | None = "default",
    equal_musts: str | list[str] | None = None,
    **kwargs: Any,
) -> tuple[SequenceIntType, SequenceNumberType]:
    r"""
    Flag duplicates from inputs.

    Parameters
    ----------
    station_id : :py:obj:`~marine_qc.SequenceStrType`, optional
        One-dimensional array of station IDs.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    lat : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional array of latitudes in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    lon : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional array of longitudes in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    date : :py:obj:`~marine_qc.SequenceDatetimeType`, optional
        One-dimensional array of datetime values.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    vsi : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional reported speed array in km/h.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    dsi : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional reported heading array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    obs : :py:obj:`~marine_qc.SequenceNumberType` or list of :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional reported observation value.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
    detected : :py:obj:`DupDetect`, optional
        A `DupDetect` instance that already contains detected duplicates to flag.
    keep : str or int, default: first
        Which entry should be kept in result dataset.
    limit : str, int or float, optional
        Limit of total score that as to be exceeded to be declared as a duplicate.
        Defaults to .991.
    equal_musts : str or list, optional
        Hashable of column name(s) that must totally be equal to be declared as a duplicate.
        Default: All column names found in method_kwargs.
    \**kwargs : Any
        Any keyword-arguments passed to `duplicate_check`.

    Returns
    -------
    tuple of list of int and list of Any
        A tuple containing:

          - list of duplicate flags
          - list of detected duplicates per row

    Raises
    ------
    ValueError
        If none of `detected`, `station_id`, `lat`, `lon`, `date`, `vsi` and `dsi` is set.

    Notes
    -----
    If `detected` is set, `station_id`, `lat`, `lon`, `date`, `vsi` and `dsi` are ignored.
    If `detected` is set, the function always returns a tuple of pd.Series.
    """
    if all(x is None for x in (detected, station_id, lat, lon, date, vsi, dsi)):
        raise ValueError(
            "None of `detected`, `station_id`, `lat`, `lon`, `date`, `vsi` and `dsi` is set."
            "Set `dupdetect` or at least one of `station_id`, `lat`, `lon`, `date`, `vsi` and `dsi`."
        )

    if detected is None:
        detected = duplicate_check(station_id, lat, lon, date, vsi, dsi, obs, **kwargs)

    return detected.flag_duplicates(keep=keep, limit=limit, equal_musts=equal_musts)
