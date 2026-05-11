from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marine_qc import flag_duplicates, remove_duplicates
from marine_qc.duplicate_checker._duplicate_settings import (
    Compare,
    _compare_kwargs,
    _method_kwargs,
)
from marine_qc.duplicate_checker.duplicates import (
    Comparer,
    DupDetect,
    change_offsets,
    convert_series,
    duplicate_check,
    reindex_nulls,
    remove_ignores,
    set_comparer,
)


@pytest.fixture
def dummy_data():
    return pd.DataFrame(
        {
            "station_id": ["S1", "S1", "S2", "S2", "S1", "S1"],
            "lon": [0.1, 0.1, 0.2, 0.1, 0.1, 0.1],
            "lat": [51.0, 51.2, 52.0, 51.0, 51.0, 51.0],
            "date": pd.to_datetime(
                [
                    "2023-01-01 00:00",
                    "2023-01-01 00:00",
                    "2023-01-01 00:00",
                    "2023-01-01 00:00",
                    "2023-01-01 00:00",
                    "2023-01-01 00:00",
                ]
            ),
            "vsi": [10.0, 10.0, 8.0, 10.0, 8.0, 10.0],
            "dsi": [90, 90, 180, 90, 60, 90],
            "flag": 2,
        },
        index=["A", "B", "C", "D", "E", "F"],
    )


@pytest.fixture
def expert_data():
    index = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U"]
    station_id = ["AA", "AA", "AA", "AA", "AA", "AA", "AA", "AA", "AA", "AA", "AA", "AA", "AB", "AA", "BB", "AA", "AA", "AA", "AA", "AA", "AA"]
    lon = [22.3, 29.7, 32.0, 8.1, -21.2, -21.2, 21.2, 29.7, 29.7, 32.0, 8.5, 8.15, 8.1, 8.05, 8.1, -21.4, -21.1, 29.7, 29.7, 29.7, 29.7]
    lat = [71.3, 71.3, 71.2, 66.0, 65.8, 65.8, -65.8, 71.3, 71.3, 71.2, 66.0, 66.05, 66.0, 65.95, 66.0, 65.6, 65.9, 71.3, 71.3, 71.3, 71.3]
    qc1 = [2, 0, 2, 0, 2, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0]
    vsi = [0.0, 4.11552, np.nan, 0.0, 0.0, 0.0, 0.0, 4.11552, 4.11552, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.11552, 4.0, 4.11552, np.nan]
    dsi = [0.0, 315.0, np.nan, 0.0, 0.0, 0.0, 0.0, 315.0, 315.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 316.0, 315.0, np.nan, 315.0]
    date = pd.to_datetime(
        [
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:01:00",
            "2022-02-02 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
        ]
    )
    qc2 = [1, 1, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1]
    qc3 = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    return pd.DataFrame(
        {
            "station_id": station_id,
            "lon": lon,
            "lat": lat,
            "date": date,
            "qc1": qc1,
            "vsi": vsi,
            "dsi": dsi,
            "qc2": qc2,
            "qc3": qc3,
            "index": index,
        },
        index=index,
    )


exp1 = {
    "duplicate_status": [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 0, 0, 0],
    "report_quality": [1, 1, 0, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1],
    "duplicates": [
        None,
        "{ICOADS-302-N688DT}",
        "{ICOADS-302-N688DW}",
        "{ICOADS-302-N688EC,ICOADS-302-N688EE}",
        "{ICOADS-302-N688EW,ICOADS-302-N688EY}",
        "{ICOADS-302-N688EI}",
        None,
        "{ICOADS-302-N688DS}",
        None,
        "{ICOADS-302-N688DV}",
        None,
        "{ICOADS-302-N688EH}",
        None,
        "{ICOADS-302-N688EH}",
        None,
        None,
        "{ICOADS-302-N688EI}",
        None,
        None,
        None,
        None,
    ],
}


def test_convert_series_basic():
    df = pd.DataFrame({"a": ["1", "2", "3"], "b": ["10.5", "20.5", "30.5"]})
    conversion = {"a": "int", "b": "float"}

    expected = pd.DataFrame({"a": [1, 2, 3], "b": [10.5, 20.5, 30.5]})

    result = convert_series(df, conversion)
    pd.testing.assert_frame_equal(result, expected)


def test_convert_series_null_replacement():
    df = pd.DataFrame({"a": ["1", None, "3"], "b": [None, "2.5", None]})
    conversion = {"a": "float", "b": "float"}

    expected = pd.DataFrame({"a": [1.0, 9999.0, 3.0], "b": [9999.0, 2.5, 9999.0]})

    result = convert_series(df, conversion)
    pd.testing.assert_frame_equal(result, expected)


def test_convert_series_date_to_float():
    df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02", "2023-01-03"]})
    conversion = {"date": "convert_date_to_float"}

    result = convert_series(df, conversion)
    expected = pd.DataFrame({"date": [0.0, 86400.0, 172800.0]})

    pd.testing.assert_frame_equal(result, expected)


def test_convert_series_mixed():
    df = pd.DataFrame(
        {
            "num": ["1", None, "3"],
            "val": ["10.5", "20.5", None],
            "date": ["2023-01-01", None, "2023-01-03"],
        }
    )
    conversion = {"num": "Int64", "val": "float", "date": "convert_date_to_float"}

    result = convert_series(df, conversion)
    expected = pd.DataFrame(
        {
            "num": [1, 9999, 3],
            "val": [10.5, 20.5, 9999.0],
            "date": [0.0, 9999.0, 172800.0],
        }
    )

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_set_comparer():
    compare_dict = {
        "col1": {"method": "exact"},
        "col2": {"method": "numeric", "kwargs": {"method": "step", "offset": 0.1}},
        "col3": {"method": "date2"},
    }
    comparer = set_comparer(compare_dict)
    assert isinstance(comparer, Compare)
    assert comparer.conversion["col2"] is float
    assert comparer.conversion["col3"] == "convert_date_to_float"


def test_remove_ignores():
    dic = {"a": 1, "b": ["x", "y"], "c": "z"}
    filtered = remove_ignores(dic, ["b", "c"])
    assert "b" not in filtered
    assert "c" not in filtered
    assert "a" in filtered


def test_change_offsets():
    dic = {"col1": {"kwargs": {"offset": 0.1}}, "col2": {"kwargs": {"offset": 0.2}}}
    new_offsets = {"col1": 0.5}
    updated = change_offsets(dic, new_offsets)
    assert updated["col1"]["kwargs"]["offset"] == 0.5
    assert updated["col2"]["kwargs"]["offset"] == 0.2


def test_reindex_nulls_orders_by_null_count():
    df = pd.DataFrame({"a": ["null", 1, "null", 2], "b": ["null", 2, 3, "null"]})
    result = reindex_nulls(df, null_label="null")

    expected_order = [1, 2, 3, 0]
    assert list(result.index) == expected_order


def test_reindex_nulls_empty_df():
    df = pd.DataFrame()
    result = reindex_nulls(df, null_label="null")
    assert result.equals(df)


def test_comparer_basic():
    df = pd.DataFrame(
        {
            "station_id": ["S1", "S1", "S2"],
            "lon": [0.1, 0.15, 0.2],
            "lat": [51.0, 51.01, 52.0],
            "date": pd.to_datetime(["2023-01-01 00:00", "2023-01-01 00:01", "2023-01-02 00:00"]),
            "vsi": [10.0, 12.0, 8.0],
            "dsi": [90, 180, 270],
        },
        index=[
            "A",
            "B",
            "C",
        ],
    )

    comp = Comparer(
        data=df,
        method="SortedNeighbourhood",
        method_kwargs=_method_kwargs,
        compare_kwargs=_compare_kwargs,
        convert_data=True,
    )

    assert isinstance(comp.data, pd.DataFrame)
    exp_data = pd.DataFrame(
        {
            "station_id": ["S1", "S1", "S2"],
            "lon": [0.1, 0.15, 0.2],
            "lat": [51.0, 51.01, 52.0],
            "date": [0.0, 60.0, 86400.0],
            "vsi": [10.0, 12.0, 8.0],
            "dsi": [90.0, 180.0, 270.0],
        },
        index=[
            "A",
            "B",
            "C",
        ],
    )
    pd.testing.assert_frame_equal(comp.data, exp_data)

    assert isinstance(comp.compared, pd.DataFrame)
    exp_comp = pd.DataFrame(
        {
            "station_id": [1],
            "lon": [1.0],
            "lat": [1.0],
            "date": [1.0],
            "vsi": [0.0],
            "dsi": [0.0],
        },
        index=pd.MultiIndex.from_tuples([("B", "A")]),
    )
    pd.testing.assert_frame_equal(comp.compared, exp_comp)


def test_duplicate_check_basic():
    station_id = ["S1", "S1", "S2"]
    lon = [0.1, 0.15, 0.2]
    lat = [51.0, 51.01, 52.0]
    date = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 00:01", "2023-01-02 00:00"])
    vsi = [10.0, 12.0, 8.0]
    dsi = [90, 180, 270]
    detector = duplicate_check(
        station_id=station_id,
        lon=lon,
        lat=lat,
        date=date,
        vsi=vsi,
        dsi=dsi,
    )

    assert isinstance(detector, DupDetect)
    exp_data = pd.DataFrame(
        {
            "station_id": station_id,
            "lat": lat,
            "lon": lon,
            "date": date,
            "vsi": vsi,
            "dsi": dsi,
        },
    )
    pd.testing.assert_frame_equal(detector.data, exp_data)

    exp_comp = pd.DataFrame(
        {
            "station_id": [1],
            "lon": [1.0],
            "lat": [1.0],
            "date": [1.0],
            "vsi": [0.0],
            "dsi": [0.0],
        },
        index=pd.MultiIndex.from_tuples([(1, 0)]),
    )
    pd.testing.assert_frame_equal(detector.compared, exp_comp)

    assert detector.method == "SortedNeighbourhood"
    assert detector.method_kwargs == _method_kwargs
    assert detector.compare_kwargs == _compare_kwargs


def test_duplicate_check_reindex(dummy_data):
    dd = duplicate_check(**dummy_data.to_dict(), reindex_by_null=False)

    assert hasattr(dd, "compared")

    result = dd.compared

    exp_idx = pd.MultiIndex.from_tuples([("B", "A"), ("D", "C"), ("E", "A"), ("E", "B"), ("F", "A"), ("F", "B"), ("F", "E")])

    pd.testing.assert_index_equal(dd.compared.index, exp_idx)

    assert list(result.columns) == [
        "station_id",
        "lon",
        "lat",
        "date",
        "vsi",
        "dsi",
    ]


def test_get_total_score(dummy_data):
    dd = duplicate_check(**dummy_data.to_dict())
    dd._total_score()

    assert hasattr(dd, "score")

    expected = pd.Series(
        [5.0 / 6.0, 0.5, 2.0 / 3.0, 0.5, 1.0, 5.0 / 6.0, 2.0 / 3.0],
        index=pd.MultiIndex.from_tuples([("B", "A"), ("D", "C"), ("E", "A"), ("E", "B"), ("F", "A"), ("F", "B"), ("F", "E")]),
    )
    pd.testing.assert_series_equal(dd.score, expected)


@pytest.mark.parametrize(
    "kwargs, exp_ids",
    [
        ({}, [("F", "A")]),
        ({"offsets": {"lat": 0.22}}, [("B", "A"), ("F", "A"), ("F", "B")]),
        (
            {"ignore_columns": ["vsi", "dsi"]},
            [("E", "A"), ("F", "A"), ("F", "E")],
        ),
        ({"ignore_entries": {"station_id": "S2"}}, [("F", "A"), ("D", "A"), ("D", "F")]),
        ({"ignore_entries": {"station_id": ["S2"]}}, [("F", "A"), ("D", "A"), ("D", "F")]),
    ],
)
def test_get_duplicates_kwargs(dummy_data, kwargs, exp_ids):
    dd = duplicate_check(**dummy_data.to_dict(), **kwargs)

    assert hasattr(dd, "compared")

    dd.get_duplicates()

    assert hasattr(dd, "matches")

    pd.testing.assert_index_equal(dd.matches.index, pd.MultiIndex.from_tuples(exp_ids))


def test_get_duplicates_limit_and_equal_musts(dummy_data):
    dd = duplicate_check(**dummy_data.to_dict())

    matches_default = dd.get_duplicates(keep="first", limit=0.5)
    expected_indexes = pd.MultiIndex.from_tuples([("F", "A")])
    pd.testing.assert_index_equal(matches_default.index, expected_indexes)

    matches_eq_str = dd.get_duplicates(keep="first", equal_musts="station_id")
    expected_indexes = pd.MultiIndex.from_tuples([("F", "A")])
    pd.testing.assert_index_equal(matches_eq_str.index, expected_indexes)

    matches_eq_list = dd.get_duplicates(keep="first", equal_musts=["station_id", "lon"])
    expected_indexes = pd.MultiIndex.from_tuples([("F", "A")])
    pd.testing.assert_index_equal(matches_eq_list.index, expected_indexes)


def test_get_duplicates_raises(dummy_data):
    dd = duplicate_check(
        **dummy_data.to_dict(),
    )
    with pytest.raises(ValueError):
        dd.get_duplicates(keep=1)


@pytest.mark.parametrize("directly", [True, False])
@pytest.mark.parametrize(
    "keep, exp_flag",
    [
        ("first", [1, 0, 0, 0, 0, 3]),
        ("last", [3, 0, 0, 0, 0, 1]),
        (0, [3, 0, 0, 0, 0, 1]),
        (-1, [1, 0, 0, 0, 0, 3]),
    ],
)
def test_flag_duplicates_basic(directly, dummy_data, keep, exp_flag):
    if directly is True:
        result = flag_duplicates(**dummy_data.to_dict(), keep=keep)
    elif directly is False:
        dd = duplicate_check(**dummy_data.to_dict())
        result = dd.flag_duplicates(keep=keep)

    assert isinstance(result, tuple)
    assert len(result) == 2
    for i in [0, 1]:
        assert isinstance(result[i], pd.Series)

    exp_flag = pd.Series(exp_flag, index=["A", "B", "C", "D", "E", "F"], name="duplicate_flags")
    exp_dups = pd.Series([["F"], np.nan, np.nan, np.nan, np.nan, ["A"]], index=["A", "B", "C", "D", "E", "F"], name="duplicates")

    pd.testing.assert_series_equal(result[0], exp_flag)
    pd.testing.assert_series_equal(result[1], exp_dups)


def test_flag_duplicates_detected(dummy_data):
    detected = duplicate_check(**dummy_data.to_dict())
    result = flag_duplicates(detected=detected)

    print(result[0])

    assert isinstance(result, tuple)
    assert len(result) == 2
    for i in [0, 1]:
        assert isinstance(result[i], pd.Series)

    exp_flag = pd.Series([1, 0, 0, 0, 0, 3], index=["A", "B", "C", "D", "E", "F"], name="duplicate_flags")
    exp_dups = pd.Series([["F"], np.nan, np.nan, np.nan, np.nan, ["A"]], index=["A", "B", "C", "D", "E", "F"], name="duplicates")

    pd.testing.assert_series_equal(result[0], exp_flag)
    pd.testing.assert_series_equal(result[1], exp_dups)


def test_flag_duplicates_series(dummy_data):
    result = flag_duplicates(
        station_id=dummy_data["station_id"],
        lat=dummy_data["lat"],
        lon=dummy_data["lon"],
        date=dummy_data["date"],
        vsi=dummy_data["vsi"],
        dsi=dummy_data["dsi"],
        flag=dummy_data["flag"],
    )

    assert isinstance(result, tuple)
    assert len(result) == 2
    for i in [0, 1]:
        assert isinstance(result[i], pd.Series)

    exp_flag = pd.Series([1, 0, 0, 0, 0, 3], index=["A", "B", "C", "D", "E", "F"], name="duplicate_flags")
    exp_dups = pd.Series([["F"], np.nan, np.nan, np.nan, np.nan, ["A"]], index=["A", "B", "C", "D", "E", "F"], name="duplicates")

    pd.testing.assert_series_equal(result[0], exp_flag)
    pd.testing.assert_series_equal(result[1], exp_dups)


@pytest.mark.parametrize("directly", [True, False])
@pytest.mark.parametrize(
    "keep, exp_idx",
    [
        ("first", [0, 1, 2, 3, 4]),
        ("last", [1, 2, 3, 4, 5]),
        (0, [1, 2, 3, 4, 5]),
        (-1, [0, 1, 2, 3, 4]),
    ],
)
def test_remove_duplicates_basic(directly, dummy_data, keep, exp_idx):
    if directly is True:
        result = remove_duplicates(**dummy_data.to_dict(), keep=keep)
    elif directly is False:
        dd = duplicate_check(**dummy_data.to_dict())
        result = dd.remove_duplicates(keep=keep)

    assert isinstance(result, tuple)
    assert len(result) == 7
    for i in [0, 1, 2, 3, 4, 5, 6]:
        assert isinstance(result[i], pd.Series)

    pd.testing.assert_series_equal(result[0], dummy_data["station_id"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[1], dummy_data["lat"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[2], dummy_data["lon"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[3], dummy_data["date"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[4], dummy_data["vsi"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[5], dummy_data["dsi"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[6], dummy_data["flag"].iloc[exp_idx])


def test_remove_duplicates_detected(dummy_data):
    detected = duplicate_check(**dummy_data.to_dict())
    result = remove_duplicates(detected=detected)

    assert isinstance(result, tuple)
    assert len(result) == 7
    for i in [0, 1, 2, 3, 4, 5, 6]:
        assert isinstance(result[i], pd.Series)

    exp_idx = [0, 1, 2, 3, 4]
    pd.testing.assert_series_equal(result[0], dummy_data["station_id"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[1], dummy_data["lat"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[2], dummy_data["lon"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[3], dummy_data["date"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[4], dummy_data["vsi"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[5], dummy_data["dsi"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[6], dummy_data["flag"].iloc[exp_idx])


def test_remove_duplicates_series(dummy_data):
    result = remove_duplicates(
        station_id=dummy_data["station_id"],
        lat=dummy_data["lat"],
        lon=dummy_data["lon"],
        date=dummy_data["date"],
        vsi=dummy_data["vsi"],
        dsi=dummy_data["dsi"],
        flag=dummy_data["flag"],
    )
    assert isinstance(result, tuple)
    assert len(result) == 7
    for i in [0, 1, 2, 3, 4, 5, 6]:
        assert isinstance(result[i], pd.Series)

    exp_idx = [0, 1, 2, 3, 4]
    pd.testing.assert_series_equal(result[0], dummy_data["station_id"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[1], dummy_data["lat"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[2], dummy_data["lon"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[3], dummy_data["date"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[4], dummy_data["vsi"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[5], dummy_data["dsi"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[6], dummy_data["flag"].iloc[exp_idx])


@pytest.mark.parametrize(
    "kwargs, flags, duplicates",
    [
        (
            {},
            [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 0, 0, 0],
            [
                np.nan,
                ["H"],
                ["J"],
                ["L", "N"],
                ["F", "Q"],
                ["E"],
                np.nan,
                ["B"],
                np.nan,
                ["C"],
                np.nan,
                ["D"],
                np.nan,
                ["D"],
                np.nan,
                np.nan,
                ["E"],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        ),
        (
            {"ignore_entries": {"station_id": ["AA", "BB"]}},
            [0, 1, 1, 3, 1, 3, 0, 3, 0, 3, 0, 3, 1, 3, 3, 0, 3, 0, 0, 0, 0],
            [
                np.nan,
                ["H"],
                ["J"],
                ["M"],
                ["F", "Q"],
                ["E"],
                np.nan,
                ["B"],
                np.nan,
                ["C"],
                np.nan,
                ["M"],
                ["D", "L", "N", "O"],
                ["M"],
                ["M"],
                np.nan,
                ["E"],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        ),
        (
            {"ignore_entries": {"vsi": np.nan, "dsi": np.nan}},
            [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 0, 3, 3],
            [
                np.nan,
                ["H", "T", "U"],
                ["J"],
                ["L", "N"],
                ["F", "Q"],
                ["E"],
                np.nan,
                ["B"],
                np.nan,
                ["C"],
                np.nan,
                ["D"],
                np.nan,
                ["D"],
                np.nan,
                np.nan,
                ["E"],
                np.nan,
                np.nan,
                ["B"],
                ["B"],
            ],
        ),
        (
            {
                "ignore_entries": {
                    "station_id": ["AA", "BB"],
                    "vsi": np.nan,
                    "dsi": np.nan,
                }
            },
            [0, 1, 1, 3, 1, 3, 0, 3, 0, 3, 0, 3, 1, 3, 3, 0, 3, 0, 0, 3, 3],
            [
                np.nan,
                ["H", "T", "U"],
                ["J"],
                ["M"],
                ["F", "Q"],
                ["E"],
                np.nan,
                ["B"],
                np.nan,
                ["C"],
                np.nan,
                ["M"],
                ["D", "L", "N", "O"],
                ["M"],
                ["M"],
                np.nan,
                ["E"],
                np.nan,
                np.nan,
                ["B"],
                ["B"],
            ],
        ),
        (
            {"method_kwargs": {"left_on": "date", "window": 7, "block_on": ["station_id"]}},
            [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 0, 0, 0],
            [
                np.nan,
                ["H"],
                ["J"],
                ["L", "N"],
                ["F", "Q"],
                ["E"],
                np.nan,
                ["B"],
                np.nan,
                ["C"],
                np.nan,
                ["D"],
                np.nan,
                ["D"],
                np.nan,
                np.nan,
                ["E"],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        ),
        (
            {"compare_kwargs": {"station_id": {"method": "exact"}, "date": {"method": "date2", "kwargs": {"method": "gauss", "offset": 60.0}}}},
            [1, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3],
            [
                ["B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "N", "P", "Q", "R", "S", "T", "U"],
                ["A"],
                ["A"],
                ["A"],
                ["A"],
                ["A"],
                ["A"],
                ["A"],
                np.nan,
                ["A"],
                ["A"],
                ["A"],
                np.nan,
                ["A"],
                np.nan,
                ["A"],
                ["A"],
                ["A"],
                ["A"],
                ["A"],
                ["A"],
            ],
        ),
        (
            {"ignore_columns": ["station_id"]},
            [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 0, 3, 3, 3, 3, 0, 3, 0, 0, 0, 0],
            [
                np.nan,
                ["H"],
                ["J"],
                ["L", "M", "N", "O"],
                ["F", "Q"],
                ["E"],
                np.nan,
                ["B"],
                np.nan,
                ["C"],
                np.nan,
                ["D"],
                ["D"],
                ["D"],
                ["D"],
                np.nan,
                ["E"],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        ),
        (
            {"offsets": {"lat": 1.0, "lon": 1.0, "date": 360}},
            [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 3, 3, 0, 3, 0, 3, 3, 0, 0, 0, 0],
            [
                np.nan,
                ["H"],
                ["J"],
                ["K", "L", "N"],
                ["F", "P", "Q"],
                ["E"],
                np.nan,
                ["B"],
                np.nan,
                ["C"],
                ["D"],
                ["D"],
                np.nan,
                ["D"],
                np.nan,
                ["E"],
                ["E"],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        ),
        (
            {"method": "Block", "method_kwargs": {"left_on": "date"}},
            [0, 0, 1, 1, 1, 3, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 0, 0, 0],
            [
                np.nan,
                np.nan,
                ["J"],
                ["L", "N"],
                ["F", "Q"],
                ["E"],
                np.nan,
                np.nan,
                np.nan,
                ["C"],
                np.nan,
                ["D"],
                np.nan,
                ["D"],
                np.nan,
                np.nan,
                ["E"],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        ),
    ],
)
def test_flag_duplicates_expert(expert_data, kwargs, flags, duplicates):
    result = flag_duplicates(
        **expert_data.to_dict(),
        **kwargs,
    )

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], pd.Series)
    assert isinstance(result[1], pd.Series)
    assert len(result[0]) == len(flags)
    assert len(result[1]) == len(duplicates)

    exp_flags = pd.Series(flags, index=expert_data.index, name="duplicate_flags")
    exp_duplicates = pd.Series(duplicates, index=expert_data.index, name="duplicates")

    pd.testing.assert_series_equal(result[0], exp_flags)
    pd.testing.assert_series_equal(result[1], exp_duplicates)
