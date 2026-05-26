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


class DupDetect:
    """
    Class to detect, flag, and remove duplicate entries in a DataFrame using a comparison matrix from recordlinkage.

    Parameters
    ----------
    groups : list of list of Any
        Groups of index pairs of potentially duplicated observations.
    settings : dict
        Settings dict used for duplicate detection.
    data : pd.DataFrame
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
        data : pd.DataFrame
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
        pd.Series
            Series containing the indexes of the corresponding duplicate(s).
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
        pd.Series
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
        tuple of pd.Series
            A tuple of pd.Series containing all original input data with the duplicates removed.
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
    data : pd.DataFrame, optional
        A pd.DataFrame containing the relevant input data.
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
        For selected columns, consider two observations as duplicates if both values being compared are NaN.
        If True, all columns are affected.
    ignore_nan_either : str, list of str or bool, optional
        For selected columns, consider two observations as duplicates if either value being compared is NaN.
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
        data = pd.DataFrame(
            {
                "station_id": station_id,
                "lat": lat,
                "lon": lon,
                "date": date,
                "vsi": vsi,
                "dsi": dsi,
                **kwargs,
            }
        )

    data_orig = data.copy()

    data["unique_id"] = data.index

    if reindex_by_null is True:
        data = reindex_nulls(data, null_label=null_label)

    data.reset_index(drop=True)

    for column, dtype in data.dtypes.items():
        if dtype == "str":
            data[column] = data[column].astype(object)

    if ignore_entries is None:
        ignore_entries = {}

    if ignore_nan_both is True:
        ignore_nan_both = list(data.columns)
    elif isinstance(ignore_nan_both, str):
        ignore_nan_both = [ignore_nan_both]
    elif ignore_nan_both is False:
        ignore_nan_both = []

    if ignore_nan_either is True:
        ignore_nan_either = list(data.columns)
    elif isinstance(ignore_nan_either, str):
        ignore_nan_either = [ignore_nan_either]
    elif ignore_nan_either is False:
        ignore_nan_either = []
    elif ignore_nan_either is None:
        ignore_nan_either = []

    if ignore_columns is None:
        ignore_columns = []
    if not isinstance(ignore_columns, list):
        ignore_columns = [ignore_columns]

    if offsets is None:
        offsets = {}

    if compare_level_libraries is None:
        compare_level_libraries = {}

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

    comparisons = []
    for column in data.columns:
        if column in ignore_columns:
            continue

        comparison_levels = []
        comparison_level = []

        clevel = None

        cll_args = {"col_name": column}

        if column in compare_level_libraries:
            clevel = getattr(cll, compare_level_libraries[column])
        else:
            if column in exact_match:
                clevel = cll.ExactMatchLevel

            if column in absolute_difference:
                if "difference_threshold" in absolute_difference[column]:
                    clevel = cll.AbsoluteDifferenceLevel
                elif "metric" in absolute_difference[column]:
                    clevel = cll.AbsoluteTimeDifferenceLevel
                else:
                    raise ValueError("sdafgsdag")

        if clevel is None:
            continue

        if clevel == cll.AbsoluteDifferenceLevel:
            if column in offsets:
                cll_diff_args = {"difference_threshold": offsets[column]}
            elif column in absolute_difference:
                cll_diff_args = absolute_difference[column]
            else:
                raise ValueError("sgdasfdjhfg")
            cll_args = {**cll_args, **cll_diff_args}

        if clevel == cll.AbsoluteTimeDifferenceLevel:
            if column in offsets:
                cll_diff_args = {"threshold": offsets[column], "input_is_string": False, "metric": "second"}
            elif column in absolute_difference:
                cll_diff_args = absolute_difference[column]
            else:
                raise ValueError("sgdasfdjhfg")
            cll_args = {**cll_args, **cll_diff_args}

        comparison_level.append(clevel(**cll_args))

        if column in ignore_entries:
            entries = ignore_entries[column]
            if not isinstance(entries, list):
                entries = [entries]
            for entry in entries:
                ignored_entry = cll.CustomLevel(f"{column}_l = '{entry}' OR {column}_r = '{entry}'")
                comparison_level.append(ignored_entry)

        if column in ignore_nan_either:
            addition = cll.CustomLevel(f"{column}_l is NULL OR {column}_r is NULL")
            comparison_level.append(addition)
        if column in ignore_nan_both:
            addition = cll.CustomLevel(f"{column}_l is NULL AND {column}_r is NULL")
            comparison_level.append(addition)

        if len(comparison_level) == 0:
            continue

        if len(comparison_level) == 1:
            comparison_level = comparison_level[0]
        else:
            comparison_level = cll.Or(*comparison_level)

        comparison_levels = [
            comparison_level,
            general_comparison["col_name"](column),
            general_comparison["nan_name"](),
        ]

        comparisons.append(
            cl.CustomComparison(
                output_column_name=column,
                comparison_levels=comparison_levels,
            )
        )

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
    :py:func:`~marine_qc.duplicate_check`.

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
    data : pd.DataFrame, optional
        A pd.DataFrame containing the relevant input data.
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
        Additional keyword arguments passed to :py:func:`~.marine_qc.duplicate_check` when ``detected`` is not provided.
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
    If ``detected`` is set, the function always returns pd.Series.
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
    :py:func:`~marine_qc.duplicate_check`.

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
    data : pd.DataFrame, optional
        A pd.DataFrame containing the relevant input data.
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
        Additional keyword arguments passed to :py:func:`~.marine_qc.duplicate_check` when ``detected`` is not provided.
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
    If ``detected`` is set, the function always returns pd.Series.
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
    :py:func:`~marine_qc.duplicate_check`.

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
    data : pd.DataFrame, optional
        A pd.DataFrame containing the relevant input data.
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
        Additional keyword arguments passed to :py:func:`~.marine_qc.duplicate_check` when ``detected`` is not provided.
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
    If ``detected`` is set, the function always returns pd.Series.
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
