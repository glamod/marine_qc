"""Common Data Model (CDM) pandas duplicate check."""

from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker
from splink import comparison_level_library as cll

from ..helpers.auxiliary import (
    SequenceDatetimeType,
    SequenceIntType,
    SequenceNumberType,
    SequenceStrType,
    post_format_return_type,
)


general_settings = {
    "link_type": "dedupe_only",
    "probability_two_random_records_match": 0.01,
    "retain_matching_columns": True,
    "retain_intermediate_calculation_columns": True,
}
exact_match = ["station_id"]
absolute_difference: dict[str, dict[str, Any]] = {
    "lat": {"difference_threshold": 0.11},
    "lon": {"difference_threshold": 0.11},
    "vsi": {"difference_threshold": 0.09},
    "dsi": {"difference_threshold": 0.9},
    "date": {"threshold": 60, "input_is_string": False, "metric": "second"},
}
general_comparison = {
    "col_name": cll.NullLevel,
    "nan_name": cll.ElseLevel,
}


class DupDetect:
    """
    Class to detect, flag, and remove duplicate entries in a DataFrame using a comparison matrix from splink.

    Parameters
    ----------
    groups : list of list of Any
        Groups of index pairs of potentially duplicated observations.
    settings : dict
        Settings dict used for duplicate detection.
    data : pandas.DataFrame
        Original dataset.
    """

    def __init__(
        self,
        groups: list[list[Any]],
        settings: dict[str, Any],
        data: pd.DataFrame,
    ) -> None:
        """
        Initialize a DupDetect instance.

        Parameters
        ----------
        groups : list of list of Any
            Groups of index pairs of potentially duplicated observations.
        settings : dict
            Settings dict used for duplicate detection.
        data : pandas.DataFrame
            Original dataset.
        """
        self.groups = groups
        self.settings = settings
        self.data = data.copy()

    def get_duplicates(
        self,
        keep: str | int = "first",
        overwrite: bool = True,
    ) -> pd.Series:
        """
        Identify duplicate matches based on the comparison matrix.

        Parameters
        ----------
        keep : str or int, default: first
            Determines which duplicate entry should be retained.

            - ``"first"`` keeps the first occurrence.
            - ``"last"`` keeps the last occurrence.
            - Integer values keep the specified positional match.
        overwrite : bool, default: True
            Whether to recompute matches if already calculated.

        Returns
        -------
        pandas.Series
            A pandas Series containing the indexes of the corresponding duplicate(s).
        """
        if keep not in ["first", "last", -1, 0]:
            raise ValueError("keep has to be one of 'first', 'last', -1 or 0.")

        if keep == "first":
            keep = 0
        elif keep == "last":
            keep = -1

        self.keep = keep

        if overwrite is True:
            dups = {}
            best_duplicates = []
            worst_duplicates = []

            for group in self.groups:
                group = list(group)
                if self.keep == -1:
                    group = list(reversed(group))
                best = group[0]
                worst = group[1:]
                dups[best] = worst if len(worst) > 1 else worst[0]
                for w in worst:
                    dups[w] = best
                    worst_duplicates.append(w)

                best_duplicates.append(best)

            self._best_duplicates = best_duplicates
            self._worst_duplicates = worst_duplicates

            duplicates = []
            for idx in self.data.index:
                if idx in dups:
                    duplicates.append(dups[idx])
                else:
                    duplicates.append(np.nan)

            duplicates = pd.Series(duplicates, index=self.data.index, dtype=object, name="duplicates")

            self.duplicates = duplicates
        return self.duplicates

    def flag_duplicates(
        self,
        keep: str | int = "first",
    ) -> pd.Series:
        r"""
        Get result dataset with flagged duplicates.

        Parameters
        ----------
        keep : str or int, default: first
            Determines which duplicate entry should be retained.

            - ``"first"`` keeps the first occurrence.
            - ``"last"`` keeps the last occurrence.
            - Integer values keep the specified positional match.

        Returns
        -------
        pandas.Series
            Series containing duplicate flags for the detected observations.

        References
        ----------
        .. _duplicate_status: https://glamod.github.io/cdm-obs-documentation/tables/code_tables/duplicate_status/duplicate_status.html
        """
        if not hasattr(self, "_best_duplicates") or not hasattr(self, "_worst_duplicates"):
            self.get_duplicates(keep=keep)

        flags = pd.Series(0, index=self.data.index, name="duplicate_flags")
        flags.loc[self._best_duplicates] = 1
        flags.loc[self._worst_duplicates] = 3
        return flags

    def remove_duplicates(
        self,
        keep: str | int = "first",
    ) -> tuple[pd.Series, ...]:
        """
        Remove duplicate entries from the dataset.

        Parameters
        ----------
        keep : str or int
            Determines which duplicate entry should be retained.

            - ``"first"`` keeps the first occurrence.
            - ``"last"`` keeps the last occurrence.
            - Integer values keep the specified positional match.

        Returns
        -------
        tuple of pandas.Series
            A tuple of pandas Series containing all original input data with the duplicates removed.
        """
        if not hasattr(self, "_best_duplicates") or not hasattr(self, "_worst_duplicates"):
            self.get_duplicates(keep=keep)

        result = self.data.drop(index=self._worst_duplicates)
        return tuple(result[col] for col in result.columns)


def reindex_nulls(df: pd.DataFrame, null_label: Any) -> pd.DataFrame:
    """
    Reindex a DataFrame in ascending order based on the number of 'null' strings in each row.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame. Cells with the string "null" are counted as nulls.
    null_label : Any
        Missing value representative.

    Returns
    -------
    pandas.DataFrame
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
        row : pandas.Series
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


def build_dataframe(
    station_id: SequenceStrType,
    lat: SequenceNumberType,
    lon: SequenceNumberType,
    date: SequenceDatetimeType,
    vsi: SequenceNumberType,
    dsi: SequenceNumberType,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build a pandas DataFrame from the supplied columns.

    Parameters
    ----------
    station_id : :py:obj:`~marine_qc.SequenceStrType`, optional
        One-dimensional array of station IDs.
    lat : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional array of latitudes in degrees.
    lon : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional array of longitudes in degrees.
    date : :py:obj:`~marine_qc.SequenceDatetimeType`, optional
        One-dimensional array of datetime values.
    vsi : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional reported speed array in km/h.
    dsi : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional reported heading array in degrees.
    extra : dict, optional
        Additional column-value pairs.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame from the supplied columns.
    """
    extra = extra or {}
    return pd.DataFrame(
        {
            "station_id": station_id,
            "lat": lat,
            "lon": lon,
            "date": date,
            "vsi": vsi,
            "dsi": dsi,
            **extra,
        }
    )


def prepare_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a pandas DataFrame for detecting duplicates.

    Parameters
    ----------
    data : pandas.DataFrame
        A pandas DataFrame that should be prepared for detecting duplicates.

    Returns
    -------
    pandas.DataFrame
        A prepared pandas DataFrame for detecting duplicates.
    """
    data["unique_id"] = data.index

    for column, dtype in data.dtypes.items():
        if dtype == "str":
            data[column] = data[column].astype(object)

    return data


def prepare_nan_handling(nan_handling: str | list[str] | bool | None, columns: pd.Index) -> list[str]:
    """
    Resolve which DataFrame columns should be considered when handling NaN values or duplicate detection.

    Parameters
    ----------
    nan_handling : str, list of str, bool or None
        Specifies how NaN values are treated for duplicate detection on the
        selected columns.

        * ``True``  - treat **all** columns in ``columns`` as NaN-sensitive.
        * ``False`` - do not apply any NaN-handling (return an empty list).
        * ``None`` - do not apply any NaN-handling (return an empty list).
        * ``str``   - a single column name to which the NaN rule should be
          applied.
        * ``list[str]`` - an explicit list of column names to which the NaN
          rule should be applied.
    columns : pandas.Index
        The complete index of column names present in the DataFrame.

    Returns
    -------
    list of str
        A list of column names that should be used for NaN-aware duplicate
        comparison, based on the value of ``nan_handling``:

        * If ``nan_handling`` is ``True``, the function returns a list of **all**
          column names.
        * If ``nan_handling`` is a string, it returns a one-element list
          containing that column.
        * If ``nan_handling`` is an empty list, ``False`` or ``None``, it returns an empty
          list (no NaN handling).
        * If ``nan_handling`` is already a list of strings, it returns that list
          unchanged.
    """
    if nan_handling is True:
        return list(columns)
    if isinstance(nan_handling, str):
        return [nan_handling]
    if not nan_handling:
        return []
    return nan_handling


def make_comparison(
    column: str,
    compare_level_libraries: dict[str, Any],
    offsets: dict[str, float],
    ignore_entries: dict[str, Any],
    ignore_nan_both: list[str],
    ignore_nan_either: list[str],
) -> cl.CustomComparison:
    """
    Build a ``cl.CustomComparison`` for *column*.

    Parameters
    ----------
    column : str
        Name of the data column.
    compare_level_libraries : dict
        Override comparison levels for selected columns.
        This modifies the comparison level used during comparison.
    offsets : dict
        Override comparison offsets for selected columns.
        This modifies the tolerance used during comparison.
    ignore_entries : dict
        Ignore specific values for selected columns during comparison.
    ignore_nan_both : list of str
        For selected columns, consider two observations as duplicates if both values being compared are NaN.
    ignore_nan_either : list of str
        For selected columns, consider two observations as duplicates if either value being compared is NaN.

    Returns
    -------
    cl.CustomComparison
        A splink comparison library instance.
    """
    clevel = None
    cll_args: dict[str, Any] = {"col_name": column}

    if compare_level_libraries == {}:
        compare_level_libraries = dict()

    clevel = (
        getattr(cll, compare_level_libraries.get(column, ""), None)
        or (cll.ExactMatchLevel if column in exact_match else None)
        or (cll.AbsoluteDifferenceLevel if column in absolute_difference and "difference_threshold" in absolute_difference[column] else None)
        or (cll.AbsoluteTimeDifferenceLevel if column in absolute_difference and "metric" in absolute_difference[column] else None)
    )

    if clevel is None:
        return None

    if clevel in (cll.AbsoluteDifferenceLevel, cll.AbsoluteTimeDifferenceLevel):
        cll_diff_args: dict[str, Any]
        if column in offsets:
            cll_diff_args = (
                {"difference_threshold": offsets[column]}
                if clevel is cll.AbsoluteDifferenceLevel
                else {"threshold": offsets[column], "input_is_string": False, "metric": "second"}
            )
        elif column in absolute_difference:
            cll_diff_args = absolute_difference[column]
        else:
            raise ValueError(
                f"No offset or absolute-difference configuration found for column '{column}'. Provide an entry in `offsets` or `absolute_difference`."
            )
        cll_args = {**cll_args, **cll_diff_args}

    sub_levels: list[Any] = [clevel(**cll_args)]

    if column in ignore_entries:
        entries = ignore_entries[column]
        if not isinstance(entries, list):
            entries = [entries]
        sub_levels.extend(cll.CustomLevel(f"{column}_l = '{e}' OR {column}_r = '{e}'") for e in entries)

    if column in ignore_nan_either:
        sub_levels.append(cll.CustomLevel(f"{column}_l is NULL OR {column}_r is NULL"))
    if column in ignore_nan_both:
        sub_levels.append(cll.CustomLevel(f"{column}_l is NULL AND {column}_r is NULL"))

    main_level = sub_levels[0] if len(sub_levels) == 1 else cll.Or(*sub_levels)

    comparison_levels = [
        main_level,
        general_comparison["col_name"](column),
        general_comparison["nan_name"](),
    ]

    return cl.CustomComparison(output_column_name=column, comparison_levels=comparison_levels)


def group_matches(matches: pd.DataFrame, order_map: dict[Any, Any]) -> list[list[Any]]:
    """
    Re-order and group matched entity pairs according to a supplied ordering.

    Parameters
    ----------
    matches : pandas.DataFrame
        A pandas DataFrame that must contain the columns ``unique_id_l`` and ``unique_id_r``.
        Each row represents a candidate match between a left-hand identifier
        (``unique_id_l``) and a right-hand identifier (``unique_id_r``).
    order_map : dict[Any, Any]
        Mapping from an identifier to a sortable rank (e.g. integer).  The
        function uses this map to:

        * decide whether to swap the left/right columns so that the left side
          always has the smaller rank,
        * sort the matches first by the left-hand rank and then by the
          right-hand rank.

    Returns
    -------
    list[list[Any]]
        A list of groups, where each group is a list of identifiers that are
        transitively connected through the matches.  For example, if the input
        contains pairs ``(A, B)`` and ``(B, C)``, the result will contain a
        single group ``[A, B, C]``.  Unconnected pairs appear as separate
        two-element groups.
    """
    matches = matches.copy()
    swap = matches["unique_id_l"].map(order_map) > matches["unique_id_r"].map(order_map)
    matches.loc[swap, ["unique_id_l", "unique_id_r"]] = matches.loc[swap, ["unique_id_r", "unique_id_l"]].to_numpy(copy=True)

    matches["rank_l"] = matches["unique_id_l"].map(order_map)
    matches = matches.sort_values("rank_l").drop(columns="rank_l").reset_index(drop=True)
    matches["rank_r"] = matches["unique_id_r"].map(order_map)
    matches = matches.groupby("unique_id_l", sort=False).apply(lambda g: g.sort_values("rank_r")).reset_index(drop=False)

    pairs = list(zip(matches["unique_id_l"], matches["unique_id_r"], strict=True))
    groups: list[list[Any]] = []
    for a, b in pairs:
        placed = False
        for group in groups:
            if a in group or b in group:
                if a not in group:
                    group.append(a)
                if b not in group:
                    group.append(b)
                placed = True
                break

        if not placed:
            groups.append([a, b])
    return groups


def duplicate_check(
    station_id: SequenceStrType | None = None,
    lat: SequenceNumberType | None = None,
    lon: SequenceNumberType | None = None,
    date: SequenceDatetimeType | None = None,
    vsi: SequenceNumberType | None = None,
    dsi: SequenceNumberType | None = None,
    data: pd.DataFrame | None = None,
    ignore_columns: str | list[str] | None = None,
    ignore_entries: dict[str, Any] | None = None,
    ignore_nan_both: str | list[str] | bool = True,
    ignore_nan_either: str | list[str] | bool | None = None,
    offsets: dict[str, Any] | None = None,
    compare_level_libraries: dict[str, Any] | None = None,
    reindex_by_null: bool = True,
    null_label: Any = "null",
    **kwargs: dict[str, Any],
) -> DupDetect:
    r"""
    Detect potentially duplicated observations using `Python SPLINK Toolkit <https://moj-analytical-services.github.io/splink/index.html>`_.

    This function builds a pandas DataFrame from the provided observation metadata and compares records using configurable
    record linkage rules.

    Candidate record pairs are generated using the SPLINK framework, after which only pairs satisfying
    all configured comparison conditions are retained as duplicates.

    The result is returned as a :py:class:`~marine_qc.duplicate_checker.duplicates.DupDetect` object
    containing the processed input data, detected duplicate groups, and comparison configuration information.

    Parameters
    ----------
    station_id : :py:obj:`~marine_qc.SequenceStrType`, optional
        One-dimensional array of station IDs.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``data`` is provided.
    lat : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional array of latitudes in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``data`` is provided.
    lon : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional array of longitudes in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``data`` is provided.
    date : :py:obj:`~marine_qc.SequenceDatetimeType`, optional
        One-dimensional array of datetime values.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``data`` is provided.
    vsi : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional reported speed array in km/h.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``data`` is provided.
    dsi : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional reported heading array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``data`` is provided.
    data : pandas.DataFrame, optional
        A pandas DataFrame containing the relevant input data.
        If provided, all input data arguments (``station_id``, ``lat``, ``lon``, ``date``, ``vsi`` and ``dsi``)
        are ignored.
    ignore_columns : str or list, optional
        Column names to exclude entirely from duplicate detection.
        For rows containing ignored values, the corresponding comparison is treated as a match for that column.
    ignore_entries : dict, optional
        Ignore specific values for selected columns during comparison.

        This is useful when placeholder or invalid values should not prevent
        duplicate matches.

        Keys correspond to column names and values correspond to entries
        to ignore.

        Ignore missing station IDs::

            ignore_entries = {
                "station_id": "UNKNOWN",
            }

        Ignore multiple values::

            ignore_entries = {
                "station_id": ["UNKNOWN", "MISSING"],
            }
    ignore_nan_both : str, list of str or bool, default: True
        For selected columns, consider two observations as duplicates whenever both compared value are NaN.
        If True, all columns are affected.
    ignore_nan_either : str, list of str or bool, optional
        For selected columns, consider two observations as duplicates whenever either compared value is NaN.
        If True, all columns are affected.
    offsets : dict, optional
        Override comparison offsets for selected columns.
        This modifies the tolerance used during comparison.

        Example::

            offsets = {
                "lat": 0.05,
                "lon": 0.05,
            }
    compare_level_libraries : dict, optional
        Override comparison levels for selected columns.
        This modifies the comparison level used during comparison.

        Example::

            compare_level_libraries = {
                "lat: "ExactMatchLevel",
            }
    reindex_by_null : bool, optional
        If True, rows are reordered according to the number of missing values.
        Rows may be reordered based on the distribution of missing values.
        This can improve duplicate matching performance and consistency when null values are present.
    null_label : str, optional
        Placeholder value used internally when ``reindex_by_null=True``.
    \**kwargs : dict
        Additional columns to include in duplicate detection.

        Extra keyword arguments are added directly to the internal DataFrame.

        Example::

           duplicate_check(
               station_id,
               lat,
               lon,
               date,
               vsi,
               dsi,
               platform_type=platform_type,
               source=source,
           )

    Returns
    -------
    :py:class:`~marine_qc.duplicate_checker.duplicates.DupDetect`
        Duplicate detection result object.

        The returned object contains:

        - The processed input data.
        - Detected duplicate groups.
        - Comparison configuration information.

    Warnings
    --------
    If `ignore_nan_either` is set, this can lead to misleading duplicate chains.

    In this example, we focus on only two input variables:

    .. code-block:: python

        df = pd.DataFrame(
            {
                "station_id": ["A", "A", "B", "A", None],
                "lon": [29.7, -29.7, 29.7, np.nan, 29.7],
            }
        )
        print(df)

    .. code-block:: text

          station_id   lon
        0          A  29.7
        1          A -29.7
        2          B  29.7
        3          A   NaN
        4       None  29.7

    This produces the following duplicate pairs:

    .. code-block:: python

        detected = duplicate_check(data=df, ignore_nan_either=True)

    * (0,3): (["A", 29.7  ], ["A" , np.nan])
    * (0,4): (["A", 29.7  ], [None, 29.7])
    * (1,3): (["A", -29.7 ], ["A" , np.nan])
    * (2,4): (["B", 29.7  ], [None, 29.7])
    * (3,4): (["A", np.nan], [None, 29.7])

    All of these duplicates pairs are reasonable if two observations are considered as duplicates whenever either compared value is NaN.

    However, this also produces the following misleading duplicate chains, even though the connected observations are clearly not duplicates:

    * 0 -> 3 -> 1
    * 0 -> 4 -> 2

    Since (3,4) is also considered a duplicate pair, all entries are connected, resulting in:

    .. code-block:: python

        print(detected.groups)
        >>> [[0, 3, 4, 1, 2]]

    Examples
    --------
    Basic usage:

    >>> dup = duplicate_check(
    ...     station_id=station_id,
    ...     lat=lat,
    ...     lon=lon,
    ...     date=date,
    ...     vsi=vsi,
    ...     dsi=dsi,
    ... )

    Using additional observations:

    >>> dup = duplicate_check(
    ...     station_id=station_id,
    ...     lat=lat,
    ...     lon=lon,
    ...     date=date,
    ...     vsi=vsi,
    ...     dsi=dsi,
    ...     temperature=temperature,
    ...     salinity=salinity,
    ... )

    Ignoring placeholder station IDs:

    >>> dup = duplicate_check(
    ...     station_id=station_id,
    ...     lat=lat,
    ...     lon=lon,
    ...     date=date,
    ...     vsi=vsi,
    ...     dsi=dsi,
    ...     ignore_entries={"station_id": "UNKNOWN"},
    ... )

    Increasing spatial tolerances:

    >>> dup = duplicate_check(
    ...     station_id=station_id,
    ...     lat=lat,
    ...     lon=lon,
    ...     date=date,
    ...     vsi=vsi,
    ...     dsi=dsi,
    ...     offsets={
    ...         "lat": 1.0,
    ...         "lon": 1.0,
    ...     },
    ... )
    """
    if data is None:
        data = build_dataframe(station_id, lat, lon, date, vsi, dsi, kwargs)

    data_orig = data.copy()

    if reindex_by_null is True:
        data = reindex_nulls(data, null_label=null_label)

    data = prepare_dataframe(data)

    columns = data.columns

    ignore_entries = ignore_entries or {}
    ignore_columns = ignore_columns or []
    ignore_columns = [ignore_columns] if isinstance(ignore_columns, str) else ignore_columns

    ignore_nan_both = prepare_nan_handling(ignore_nan_both, columns)
    ignore_nan_either = prepare_nan_handling(ignore_nan_either, columns)

    offsets = offsets or {}
    compare_level_libraries = compare_level_libraries or {}

    comparisons = []
    for column in columns:
        if column in ignore_columns:
            continue

        comp = make_comparison(
            column,
            compare_level_libraries,
            offsets,
            ignore_entries,
            ignore_nan_both,
            ignore_nan_either,
        )

        if comp is None:
            continue

        comparisons.append(comp)

    settings = {
        **general_settings,
        "comparisons": comparisons,
    }

    linker = Linker(data, settings, db_api=DuckDBAPI())
    predictions = linker.inference.predict()
    results = predictions.as_pandas_dataframe()

    cond = (results.filter(regex=r"^gamma_") == 1).all(axis=1)
    matches = results[cond]

    order_map = {uid: i for i, uid in enumerate(data.index)}

    groups = group_matches(matches, order_map)

    return DupDetect(groups, settings, data_orig)


@post_format_return_type(["station_id", "lat", "lon", "date", "vsi", "dsi", "data", "detected"], multiple=True, dtype=None, keep_index=True)
def remove_duplicates(
    station_id: SequenceStrType | None = None,
    lat: SequenceNumberType | None = None,
    lon: SequenceNumberType | None = None,
    date: SequenceDatetimeType | None = None,
    vsi: SequenceNumberType | None = None,
    dsi: SequenceNumberType | None = None,
    data: pd.DataFrame | None = None,
    detected: DupDetect | None = None,
    keep: str | int = "first",
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
    Remove potentially duplicated observations using `Python SPLINK Toolkit <https://moj-analytical-services.github.io/splink/index.html>`_.

    This function identifies duplicate observations either from a precomputed
    :py:class:`~marine_qc.duplicate_checker.duplicates.DupDetect` instance or by internally calling
    :py:func:`~marine_qc.duplicate_checker.duplicates.duplicate_check`.

    Candidate record pairs are generated using the SPLINK framework, after which only pairs satisfying
    all configured comparison conditions are retained as duplicates.

    The function removes duplicate observations according to the selected ``keep`` strategy and returns the filtered input data.

    Parameters
    ----------
    station_id : :py:obj:`~marine_qc.SequenceStrType`, optional
        One-dimensional array of station IDs.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    lat : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional array of latitudes in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    lon : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional array of longitudes in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    date : :py:obj:`~marine_qc.SequenceDatetimeType`, optional
        One-dimensional array of datetime values.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    vsi : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional reported speed array in km/h.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    dsi : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional reported heading array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    data : pandas.DataFrame, optional
        A pandas.DataFrame containing the relevant input data.
        Ignored if ``detected`` is provided.
        If provided, all input data arguments (``station_id``, ``lat``, ``lon``, ``date``, ``vsi`` and ``dsi``)
        are ignored.
    detected : :py:class:`~marine_qc.duplicate_checker.duplicates.DupDetect`, optional
        A :py:class:`~marine_qc.duplicate_checker.duplicates.DupDetect` instance that already contains
        detected duplicates to flag.
        If provided, duplicate detection is not rerun and all input data arguments
        (``station_id``, ``lat``, ``lon``, ``date``, ``vsi``, ``dsi``, and ``data``) are ignored.
    keep : str or int, default: first
        Determines which duplicate entry should be retained.

        - ``"first"`` keeps the first occurrence.
        - ``"last"`` keeps the last occurrence.
        - Integer values keep the specified positional match.
    \**kwargs : Any
        Additional keyword arguments passed to :py:func:`~marine_qc.duplicate_checker.duplicates.duplicate_check`
        when ``detected`` is not provided.
        Additionally, that could be extra input data as well.

    Returns
    -------
    tuple of result arrays.
        Same type as input, but with integer values
        A tuple of all input data without the removed duplcited rows.

    Raises
    ------
    ValueError
        If none of ``detected``, ``data``, ``station_id``, ``lat``, ``lon``, ``date``, ``vsi`` and ``dsi`` is set.

    Notes
    -----
    If ``detected`` is set, ``station_id``, ``lat``, ``lon``, ``date``, ``vsi``, ``dsi`` and ``data`` are ignored.
    If ``detected`` is set, the function always returns pandas.Series.
    If ``data`` is set, ``station_id``, ``lat``, ``lon``, ``date``, ``vsi`` and ``dsi`` are ignored.

    Examples
    --------
    Remove duplicates directly from raw observations:

    >>> results = remove_duplicates(
    ...     station_id=station_id,
    ...     lat=lat,
    ...     lon=lon,
    ...     date=date,
    ...     vsi=vsi,
    ...     dsi=dsi,
    ... )

    Use a precomputed duplicate detection result:
    >>> detected = duplicate_check(
    ...     station_id=station_id,
    ...     lat=lat,
    ...     lon=lon,
    ...     date=date,
    ...     vsi=vsi,
    ...     dsi=dsi,
    ... )
    ... results = remove_duplicates(detected=detected)
    """
    if all(x is None for x in (detected, station_id, lat, lon, date, vsi, dsi, data)):
        raise ValueError(
            "None of `detected`, `station_id`, `lat`, `lon`, `date`, `vsi`, `dsi` and `data` is set."
            "Set `dupdetect` or `data` or at least one of `station_id`, `lat`, `lon`, `date`, `vsi` and `dsi`."
        )

    if detected is None:
        detected = duplicate_check(station_id, lat, lon, date, vsi, dsi, data, **kwargs)

    return detected.remove_duplicates(keep=keep)


@post_format_return_type(["station_id", "lat", "lon", "date", "vsi", "dsi", "data", "detected"])
def flag_duplicates(
    station_id: SequenceStrType | None = None,
    lat: SequenceNumberType | None = None,
    lon: SequenceNumberType | None = None,
    date: SequenceDatetimeType | None = None,
    vsi: SequenceNumberType | None = None,
    dsi: SequenceNumberType | None = None,
    data: pd.DataFrame | None = None,
    detected: DupDetect | None = None,
    keep: str | int = "first",
    **kwargs: Any,
) -> SequenceIntType:
    r"""
    Flag potentially duplicated observations using `Python SPLINK Toolkit <https://moj-analytical-services.github.io/splink/index.html>`_.

    This function identifies duplicate observations either from a precomputed
    :py:class:`~marine_qc.duplicate_checker.duplicates.DupDetect` instance or by internally calling
    :py:func:`~marine_qc.duplicate_checker.duplicates.duplicate_check`.

    Candidate record pairs are generated using the SPLINK framework, after which only pairs satisfying
    all configured comparison conditions are retained as duplicates.

    The function returns duplicate flags for the detected observations according to the selected ``keep`` strategy.

    Parameters
    ----------
    station_id : :py:obj:`~marine_qc.SequenceStrType`, optional
        One-dimensional array of station IDs.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    lat : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional array of latitudes in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    lon : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional array of longitudes in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    date : :py:obj:`~marine_qc.SequenceDatetimeType`, optional
        One-dimensional array of datetime values.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    vsi : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional reported speed array in km/h.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    dsi : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional reported heading array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    data : pandas.DataFrame, optional
        A pandas.DataFrame containing the relevant input data.
        Ignored if ``detected`` is provided.
        If provided, all input data arguments (``station_id``, ``lat``, ``lon``, ``date``, ``vsi`` and ``dsi``)
        are ignored.
    detected : :py:class:`~marine_qc.duplicate_checker.duplicates.DupDetect`, optional
        A :py:class:`~marine_qc.duplicate_checker.duplicates.DupDetect` instance that already contains
        detected duplicates to flag.
        If provided, duplicate detection is not rerun and all input data arguments
        (``station_id``, ``lat``, ``lon``, ``date``, ``vsi``, ``dsi``, and ``data``) are ignored.
    keep : str or int, default: first
        Determines which duplicate entry should be retained.

        - ``"first"`` keeps the first occurrence.
        - ``"last"`` keeps the last occurrence.
        - Integer values keep the specified positional match.
    \**kwargs : Any
        Additional keyword arguments passed to :py:func:`~marine_qc.duplicate_checker.duplicates.duplicate_check`
        when ``detected`` is not provided.
        Additionally, that could be extra input data as well.

    Returns
    -------
    :py:obj:`~marine_qc.SequenceIntType`
        Same type as input, but with integer values

          - Returns 0 (or array/sequence/Series of 1s) for unique observation(s)
          - Returns 1 (or array/sequence/Series of 1s) for best duplicate(s)
          - Returns 3 (or array/sequence/Series of 1s) for worst duplicate(s)

    Raises
    ------
    ValueError
        If none of ``detected``, ``data``, ``station_id``, ``lat``, ``lon``, ``date``, ``vsi`` and ``dsi`` is set.

    Notes
    -----
    If ``detected`` is set, ``station_id``, ``lat``, ``lon``, ``date``, ``vsi``, ``dsi`` and ``data`` are ignored.
    If ``detected`` is set, the function always returns pandas.Series.
    If ``data`` is set, ``station_id``, ``lat``, ``lon``, ``date``, ``vsi`` and ``dsi`` are ignored.

    Examples
    --------
    Flag duplicates directly from raw observations:

    >>> flags = flag_duplicates(
    ...     station_id=station_id,
    ...     lat=lat,
    ...     lon=lon,
    ...     date=date,
    ...     vsi=vsi,
    ...     dsi=dsi,
    ... )

    Use a precomputed duplicate detection result:
    >>> detected = duplicate_check(
    ...     station_id=station_id,
    ...     lat=lat,
    ...     lon=lon,
    ...     date=date,
    ...     vsi=vsi,
    ...     dsi=dsi,
    ... )
    ... flags = flag_duplicates(detected=detected)
    """
    if all(x is None for x in (detected, station_id, lat, lon, date, vsi, dsi, data)):
        raise ValueError(
            "None of `detected`, `station_id`, `lat`, `lon`, `date`, `vsi`, `dsi` and `data` is set."
            "Set `dupdetect` or `data` or at least one of `station_id`, `lat`, `lon`, `date`, `vsi` and `dsi`."
        )

    if detected is None:
        detected = duplicate_check(station_id, lat, lon, date, vsi, dsi, data, **kwargs)

    return detected.flag_duplicates(keep=keep)


@post_format_return_type(["station_id", "lat", "lon", "date", "vsi", "dsi", "data", "detected"])
def get_duplicates(
    station_id: SequenceStrType | None = None,
    lat: SequenceNumberType | None = None,
    lon: SequenceNumberType | None = None,
    date: SequenceDatetimeType | None = None,
    vsi: SequenceNumberType | None = None,
    dsi: SequenceNumberType | None = None,
    data: pd.DataFrame | None = None,
    detected: DupDetect | None = None,
    keep: str | int = "first",
    **kwargs: Any,
) -> SequenceIntType:
    r"""
    Get potentially duplicated observations using `Python SPLINK Toolkit <https://moj-analytical-services.github.io/splink/index.html>`_.

    This function identifies duplicate observations either from a precomputed
    :py:class:`~marine_qc.duplicate_checker.duplicates.DupDetect` instance or by internally calling
    :py:func:`~marine_qc.duplicate_checker.duplicates.duplicate_check`.

    Candidate record pairs are generated using the SPLINK framework, after which only pairs satisfying
    all configured comparison conditions are retained as duplicates.

    The function returns the indices of the detected duplicate observations according to the selected ``keep`` strategy.

    Parameters
    ----------
    station_id : :py:obj:`~marine_qc.SequenceStrType`, optional
        One-dimensional array of station IDs.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    lat : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional array of latitudes in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    lon : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional array of longitudes in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    date : :py:obj:`~marine_qc.SequenceDatetimeType`, optional
        One-dimensional array of datetime values.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    vsi : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional reported speed array in km/h.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    dsi : :py:obj:`~marine_qc.SequenceNumberType`, optional
        One-dimensional reported heading array in degrees.
        Can be a sequence (e.g., list or tuple), a one-dimensional NumPy array, or a pandas Series.
        Ignored if ``detected`` or ``data`` is provided.
    data : pandas.DataFrame, optional
        A pandas.DataFrame containing the relevant input data.
        Ignored if ``detected`` is provided.
        If provided, all input data arguments (``station_id``, ``lat``, ``lon``, ``date``, ``vsi`` and ``dsi``)
        are ignored.
    detected : :py:class:`~marine_qc.duplicate_checker.duplicates.DupDetect`, optional
        A :py:class:`~marine_qc.duplicate_checker.duplicates.DupDetect` instance that already contains
        detected duplicates to flag.
        If provided, duplicate detection is not rerun and all input data arguments
        (``station_id``, ``lat``, ``lon``, ``date``, ``vsi``, ``dsi``, and ``data``) are ignored.
    keep : str or int, default: first
        Determines which duplicate entry should be retained.

        - ``"first"`` keeps the first occurrence.
        - ``"last"`` keeps the last occurrence.
        - Integer values keep the specified positional match.
    \**kwargs : Any
        Additional keyword arguments passed to :py:func:`~marine_qc.duplicate_checker.duplicates.duplicate_check`
        when ``detected`` is not provided.
        Additionally, that could be extra input data as well.

    Returns
    -------
    :py:obj:`~marine_qc.SequenceIntType`
        Same type as input, but with integer values

          Returns the indexes of the corresponding duplicate(s).

    Raises
    ------
    ValueError
        If none of ``detected``, ``data``, ``station_id``, ``lat``, ``lon``, ``date``, ``vsi`` and ``dsi`` is set.

    Notes
    -----
    If ``detected`` is set, ``station_id``, ``lat``, ``lon``, ``date``, ``vsi``, ``dsi`` and ``data`` are ignored.
    If ``detected`` is set, the function always returns pandas.Series.
    If ``data`` is set, ``station_id``, ``lat``, ``lon``, ``date``, ``vsi`` and ``dsi`` are ignored.

    Examples
    --------
    Get duplicates directly from raw observations:

    >>> duplicates = get_duplicates(
    ...     station_id=station_id,
    ...     lat=lat,
    ...     lon=lon,
    ...     date=date,
    ...     vsi=vsi,
    ...     dsi=dsi,
    ... )

    Use a precomputed duplicate detection result:
    >>> detected = duplicate_check(
    ...     station_id=station_id,
    ...     lat=lat,
    ...     lon=lon,
    ...     date=date,
    ...     vsi=vsi,
    ...     dsi=dsi,
    ... )
    ... duplicates = get_duplicates(detected=detected)
    """
    if all(x is None for x in (detected, station_id, lat, lon, date, vsi, dsi, data)):
        raise ValueError(
            "None of `detected`, `station_id`, `lat`, `lon`, `date`, `vsi`, `dsi` and `data` is set."
            "Set `dupdetect` or `data` or at least one of `station_id`, `lat`, `lon`, `date`, `vsi` and `dsi`."
        )

    if detected is None:
        detected = duplicate_check(station_id, lat, lon, date, vsi, dsi, data, **kwargs)

    return detected.get_duplicates(keep=keep)
