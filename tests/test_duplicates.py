from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import splink.comparison_library as cl

from marine_qc import flag_duplicates, get_duplicates, remove_duplicates
from marine_qc.duplicate_checker.duplicates import (
    DupDetect,
    build_dataframe,
    duplicate_check,
    group_matches,
    make_comparison,
    prepare_dataframe,
    prepare_nan_handling,
    reindex_nulls,
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
            "flag": [None, 2, 2, 2, 2, 2],
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
    sst = [
        300.0,
        302.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
    ]
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
            "sst": sst,
            "index": index,
        },
        index=index,
    )


def test_reindex_nulls_orders_single():
    df = pd.DataFrame({"a": ["null", 1, "null", 2], "b": ["null", 2, 3, "null"]})
    result = reindex_nulls(df, null_label="null")

    expected_order = [1, 2, 3, 0]
    assert list(result.index) == expected_order


def test_reindex_nulls_handles_nested():
    df = pd.DataFrame(
        {
            "a": [
                [1, 2],
                [1, "null"],
                [1, 2, 3],
            ]
        }
    )

    result = reindex_nulls(df, null_label="null")

    expected_order = [0, 2, 1]
    assert list(result.index) == expected_order


def test_reindex_nulls_empty_df():
    df = pd.DataFrame()
    result = reindex_nulls(df, null_label="null")
    assert result.equals(df)


def test_build_dataframe(dummy_data):
    df = build_dataframe(
        station_id=dummy_data["station_id"],
        lat=dummy_data["lat"],
        lon=dummy_data["lon"],
        date=dummy_data["date"],
        vsi=dummy_data["vsi"],
        dsi=dummy_data["dsi"],
    )
    pd.testing.assert_frame_equal(df, dummy_data[["station_id", "lat", "lon", "date", "vsi", "dsi"]])

    df = build_dataframe(
        station_id=dummy_data["station_id"],
        lat=dummy_data["lat"],
        lon=dummy_data["lon"],
        date=dummy_data["date"],
        vsi=dummy_data["vsi"],
        dsi=dummy_data["dsi"],
        extra={"flag": dummy_data["flag"]},
    )
    pd.testing.assert_frame_equal(df, dummy_data[["station_id", "lat", "lon", "date", "vsi", "dsi", "flag"]])


def test_prepare_dataframe(dummy_data):
    df = prepare_dataframe(dummy_data)
    df_exp = dummy_data.copy()
    df_exp["unique_id"] = dummy_data.index
    df_exp["station_id"] = df_exp["station_id"].astype(object)
    df_exp["unique_id"] = df_exp["unique_id"].astype(object)

    pd.testing.assert_frame_equal(df, df_exp)


def test_prepare_nan_handling(dummy_data):
    columns = dummy_data.columns
    assert prepare_nan_handling(True, columns) == list(columns)
    assert prepare_nan_handling("station_id", columns) == ["station_id"]
    assert prepare_nan_handling(False, columns) == []
    assert prepare_nan_handling(None, columns) == []
    assert prepare_nan_handling(["lat", "lon"], columns) == ["lat", "lon"]


def test_make_comparison_returns_unknown_column():
    result = make_comparison(
        column="unknown",
        compare_level_libraries={},
        offsets={},
        ignore_entries={},
        ignore_nan_both=[],
        ignore_nan_either=[],
    )
    assert result is None


def test_make_comparison_exact_match():
    result = make_comparison(
        column="station_id",
        compare_level_libraries={},
        offsets={},
        ignore_entries={},
        ignore_nan_both=[],
        ignore_nan_either=[],
    )

    assert isinstance(result, cl.CustomComparison)
    assert result.__dict__["_output_column_name"] == "station_id"
    assert "_comparison_levels" in result.__dict__
    assert result.__dict__["_description"] is None


def test_make_comparison_absolute_difference():
    result = make_comparison(
        column="lat",
        compare_level_libraries={},
        offsets={"lat": 0.5},
        ignore_entries={},
        ignore_nan_both=[],
        ignore_nan_either=[],
    )

    assert isinstance(result, cl.CustomComparison)
    assert result.__dict__["_output_column_name"] == "lat"
    assert "_comparison_levels" in result.__dict__
    assert result.__dict__["_description"] is None


def test_make_comparison_absolute_time_difference():
    result = make_comparison(
        column="date",
        compare_level_libraries={},
        offsets={"date": 30},
        ignore_entries={},
        ignore_nan_both=[],
        ignore_nan_either=[],
    )

    assert isinstance(result, cl.CustomComparison)
    assert result.__dict__["_output_column_name"] == "date"
    assert "_comparison_levels" in result.__dict__
    assert result.__dict__["_description"] is None


def test_make_comparison_ignore_single_entry():
    result = make_comparison(
        column="station_id",
        compare_level_libraries={},
        offsets={},
        ignore_entries={"station_id": "UNKNOWN"},
        ignore_nan_both=[],
        ignore_nan_either=[],
    )

    assert isinstance(result, cl.CustomComparison)
    assert result.__dict__["_output_column_name"] == "station_id"
    assert "_comparison_levels" in result.__dict__
    assert result.__dict__["_description"] is None


def test_make_comparison_ignore_multiple_entries():
    result = make_comparison(
        column="station_id",
        compare_level_libraries={},
        offsets={},
        ignore_entries={"station_id": ["UNKNOWN", "MISSING"]},
        ignore_nan_both=[],
        ignore_nan_either=[],
    )

    assert isinstance(result, cl.CustomComparison)
    assert result.__dict__["_output_column_name"] == "station_id"
    assert "_comparison_levels" in result.__dict__
    assert result.__dict__["_description"] is None


def test_make_comparison_ignore_nan_either():
    result = make_comparison(
        column="station_id",
        compare_level_libraries={},
        offsets={},
        ignore_entries={},
        ignore_nan_both=[],
        ignore_nan_either=["station_id"],
    )

    assert isinstance(result, cl.CustomComparison)
    assert result.__dict__["_output_column_name"] == "station_id"
    assert "_comparison_levels" in result.__dict__
    assert result.__dict__["_description"] is None


def test_make_comparison_ignore_nan_both():
    result = make_comparison(
        column="station_id",
        compare_level_libraries={},
        offsets={},
        ignore_entries={},
        ignore_nan_both=["station_id"],
        ignore_nan_either=[],
    )

    assert isinstance(result, cl.CustomComparison)
    assert result.__dict__["_output_column_name"] == "station_id"
    assert "_comparison_levels" in result.__dict__
    assert result.__dict__["_description"] is None


def test_make_comparison_ignore_both_nan():
    result = make_comparison(
        column="station_id",
        compare_level_libraries={},
        offsets={},
        ignore_entries={},
        ignore_nan_both=["station_id"],
        ignore_nan_either=["station_id"],
    )

    assert isinstance(result, cl.CustomComparison)
    assert result.__dict__["_output_column_name"] == "station_id"
    assert "_comparison_levels" in result.__dict__
    assert result.__dict__["_description"] is None


def test_make_comparison_raises():
    with pytest.raises(ValueError, match="No offset or absolute-difference configuration found for column"):
        make_comparison(
            column="extra",
            compare_level_libraries={"extra": "AbsoluteDifferenceLevel"},
            offsets={},
            ignore_entries={},
            ignore_nan_both=[],
            ignore_nan_either=[],
        )


def test_group_matches_one_pair():
    df = pd.DataFrame(
        {
            "unique_id_l": ["B", "A"],
            "unique_id_r": ["A", "B"],
        }
    )

    order_map = {"A": 0, "B": 1}

    result = group_matches(df.copy(), order_map)

    assert result == [["A", "B"]]


def test_group_matches_two_pairs():
    df = pd.DataFrame(
        {
            "unique_id_l": ["A", "C"],
            "unique_id_r": ["B", "D"],
        }
    )

    order_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    result = group_matches(df, order_map)

    assert result == [["A", "B"], ["C", "D"]]


def test_group_matches_chain_direct():
    df = pd.DataFrame(
        {
            "unique_id_l": ["A", "B"],
            "unique_id_r": ["B", "C"],
        }
    )

    order_map = {"A": 0, "B": 1, "C": 2}

    result = group_matches(df, order_map)

    assert result == [["A", "B", "C"]]


def test_group_matches_chain_indirect():
    df = pd.DataFrame(
        {
            "unique_id_l": ["A", "C"],
            "unique_id_r": ["B", "B"],
        }
    )

    order_map = {"A": 0, "B": 1, "C": 2}

    result = group_matches(df, order_map)

    assert result == [["A", "B", "C"]]


def test_group_matches_empty():
    df = pd.DataFrame(columns=["unique_id_l", "unique_id_r"])
    result = group_matches(df, {})
    assert result == []


def test_group_matches_groupby_sorting():
    df = pd.DataFrame(
        {
            "unique_id_l": ["A", "A"],
            "unique_id_r": ["C", "B"],
        }
    )

    order_map = {"A": 0, "B": 1, "C": 2}

    result = group_matches(df.copy(), order_map)

    assert result[0][0] == "A"


def test_duplicate_check_basic():
    station_id = ["S1", "S1", "S1", "S1", "S2", "S2"]
    lon = [0.1, 0.15, 0.15, 0.2, 0.1, 0.2]
    lat = [51.0, 51.01, 51.01, 51.0, 51.0, 52.0]
    date = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 00:01", "2023-01-01 00:01", "2023-01-01 00:00", "2023-01-01 00:00", "2023-01-02 00:00"])
    vsi = [10.0, 12.0, 10.0, 10.0, 10.0, 8.0]
    dsi = [90, 180, 90, 90, 90, 270]

    detector = duplicate_check(
        station_id=station_id,
        lon=lon,
        lat=lat,
        date=date,
        vsi=vsi,
        dsi=dsi,
    )

    assert isinstance(detector, DupDetect)
    assert hasattr(detector, "data")
    assert isinstance(detector.data, pd.DataFrame)
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

    assert hasattr(detector, "groups")
    assert isinstance(detector.groups, list)
    assert detector.groups == [[0, 2, 3]]


def test_duplicate_check_reindex(dummy_data):
    detector = duplicate_check(**dummy_data.to_dict(), reindex_by_null=True)

    assert hasattr(detector, "groups")
    assert isinstance(detector.groups, list)
    assert detector.groups == [["F", "A"]]

    detector = duplicate_check(**dummy_data.to_dict(), reindex_by_null=False)

    assert hasattr(detector, "groups")
    assert isinstance(detector.groups, list)
    assert detector.groups == [["A", "F"]]


def test_get_duplicates_basic(dummy_data):
    dups_res = get_duplicates(data=dummy_data)
    dups_exp = pd.Series(["F", np.nan, np.nan, np.nan, np.nan, "A"], dtype=object, index=dummy_data.index, name="duplicates")
    pd.testing.assert_series_equal(dups_res, dups_exp)


def test_get_duplicates_detector(dummy_data):
    detector = duplicate_check(data=dummy_data)
    detector.get_duplicates()

    assert hasattr(detector, "_best_duplicates")
    assert detector._best_duplicates == ["F"]

    assert hasattr(detector, "_worst_duplicates")
    assert detector._worst_duplicates == ["A"]

    assert hasattr(detector, "duplicates")
    dups_exp = pd.Series(["F", np.nan, np.nan, np.nan, np.nan, "A"], dtype=object, index=dummy_data.index, name="duplicates")
    pd.testing.assert_series_equal(detector.duplicates, dups_exp)

    detector.get_duplicates(keep="last")

    assert hasattr(detector, "_best_duplicates")
    assert detector._best_duplicates == ["A"]

    assert hasattr(detector, "_worst_duplicates")
    assert detector._worst_duplicates == ["F"]

    assert hasattr(detector, "duplicates")
    dups_exp = pd.Series(["F", np.nan, np.nan, np.nan, np.nan, "A"], dtype=object, index=dummy_data.index, name="duplicates")
    pd.testing.assert_series_equal(detector.duplicates, dups_exp)


def test_get_duplicates_invalid_keep(dummy_data):
    dd = duplicate_check(
        **dummy_data.to_dict(),
    )
    with pytest.raises(ValueError):
        dd.get_duplicates(keep=1)


def test_get_duplicates_raises():
    with pytest.raises(ValueError, match="None of `detected`, `station_id`, `lat`, `lon`, `date`, `vsi`, `dsi` and `data` is set."):
        get_duplicates()


@pytest.mark.parametrize("directly", [True, False])
@pytest.mark.parametrize(
    "keep, exp_flag",
    [
        ("first", [3, 0, 0, 0, 0, 1]),
        ("last", [1, 0, 0, 0, 0, 3]),
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

    flags_exp = pd.Series(exp_flag, index=dummy_data.index, name="duplicate_flags")
    pd.testing.assert_series_equal(result, flags_exp)


def test_flag_duplicates_detected(dummy_data):
    detected = duplicate_check(**dummy_data.to_dict())
    result = flag_duplicates(detected=detected)

    flags_exp = pd.Series([3, 0, 0, 0, 0, 1], index=dummy_data.index, name="duplicate_flags")
    pd.testing.assert_series_equal(result, flags_exp)


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

    flags_exp = pd.Series([3, 0, 0, 0, 0, 1], index=dummy_data.index, name="duplicate_flags")
    pd.testing.assert_series_equal(result, flags_exp)


def test_flag_duplicates_array(dummy_data):
    result = flag_duplicates(
        station_id=np.array(dummy_data["station_id"]),
        lat=np.array(dummy_data["lat"]),
        lon=np.array(dummy_data["lon"]),
        date=np.array(dummy_data["date"]),
        vsi=np.array(dummy_data["vsi"]),
        dsi=np.array(dummy_data["dsi"]),
        flag=np.array(dummy_data["flag"]),
    )
    np.testing.assert_array_equal(result, [3, 0, 0, 0, 0, 1])


def test_flag_duplicates_list(dummy_data):
    result = flag_duplicates(
        station_id=np.array(dummy_data["station_id"]),
        lat=dummy_data["lat"].tolist(),
        lon=dummy_data["lon"].tolist(),
        date=dummy_data["date"].tolist(),
        vsi=dummy_data["vsi"].tolist(),
        dsi=dummy_data["dsi"].tolist(),
        flag=dummy_data["flag"].tolist(),
    )
    np.testing.assert_array_equal(result, [3, 0, 0, 0, 0, 1])


def test_flag_duplicates_obs_single(dummy_data):
    data = dummy_data.copy()
    data["obs"] = [1, 1, 1, 1, 1, 2]
    result = flag_duplicates(**data.to_dict(), compare_level_libraries={"obs": "AbsoluteDifferenceLevel"}, offsets={"obs": 0.9})

    flags_exp = pd.Series([0, 0, 0, 0, 0, 0], index=dummy_data.index, name="duplicate_flags")
    pd.testing.assert_series_equal(result, flags_exp)


def test_flag_duplicates_obs_multiple(dummy_data):
    data = dummy_data.copy()
    data["obs1"] = [1, 1, 1, 1, 1, 1]
    data["obs2"] = [2, 3, 3, 3, 3, 3]
    result = flag_duplicates(
        **data.to_dict(),
        compare_level_libraries={"obs1": "AbsoluteDifferenceLevel", "obs2": "AbsoluteDifferenceLevel"},
        offsets={"obs1": 0.9, "obs2": 0.9},
    )

    flags_exp = pd.Series([0, 0, 0, 0, 0, 0], index=dummy_data.index, name="duplicate_flags")
    pd.testing.assert_series_equal(result, flags_exp)


def test_flag_duplicates_obs_offsets(dummy_data):
    data = dummy_data.copy()
    data["obs1"] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.4]
    data["obs2"] = [2, 3, 3, 3, 3, 3]
    result = flag_duplicates(
        **data.to_dict(),
        compare_level_libraries={"obs1": "AbsoluteDifferenceLevel", "obs2": "AbsoluteDifferenceLevel"},
        offsets={"obs1": 0.5, "obs2": 1.0},
    )

    flags_exp = pd.Series([3, 0, 0, 0, 0, 1], index=dummy_data.index, name="duplicate_flags")
    pd.testing.assert_series_equal(result, flags_exp)


def test_fla_duplicates_raises():
    with pytest.raises(ValueError, match="None of `detected`, `station_id`, `lat`, `lon`, `date`, `vsi`, `dsi` and `data` is set."):
        flag_duplicates()


@pytest.mark.parametrize("directly", [True, False])
@pytest.mark.parametrize(
    "keep, exp_idx",
    [
        ("first", [1, 2, 3, 4, 5]),
        ("last", [0, 1, 2, 3, 4]),
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
    for r in result:
        assert isinstance(r, pd.Series)

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
    for r in result:
        assert isinstance(r, pd.Series)

    exp_idx = [1, 2, 3, 4, 5]
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
    for r in result:
        assert isinstance(r, pd.Series)

    exp_idx = [1, 2, 3, 4, 5]
    pd.testing.assert_series_equal(result[0], dummy_data["station_id"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[1], dummy_data["lat"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[2], dummy_data["lon"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[3], dummy_data["date"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[4], dummy_data["vsi"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[5], dummy_data["dsi"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[6], dummy_data["flag"].iloc[exp_idx])


def test_remove_duplicates_obs_single(dummy_data):
    data = dummy_data.copy()
    data["obs"] = [1, 1, 1, 1, 1, 2]
    result = remove_duplicates(**data.to_dict(), compare_level_libraries={"obs": "AbsoluteDifferenceLevel"}, offsets={"obs": 0.9})

    assert isinstance(result, tuple)
    assert len(result) == 8
    for r in result:
        assert isinstance(r, pd.Series)

    pd.testing.assert_series_equal(result[0], data["station_id"])
    pd.testing.assert_series_equal(result[1], data["lat"])
    pd.testing.assert_series_equal(result[2], data["lon"])
    pd.testing.assert_series_equal(result[3], data["date"])
    pd.testing.assert_series_equal(result[4], data["vsi"])
    pd.testing.assert_series_equal(result[5], data["dsi"])
    pd.testing.assert_series_equal(result[6], data["flag"])
    pd.testing.assert_series_equal(result[7], data["obs"])


def test_remove_duplicates_obs_multiple(dummy_data):
    data = dummy_data.copy()
    data["obs1"] = [1, 1, 1, 1, 1, 1]
    data["obs2"] = [2, 3, 3, 3, 3, 3]
    result = remove_duplicates(
        **data.to_dict(),
        compare_level_libraries={"obs1": "AbsoluteDifferenceLevel", "obs2": "AbsoluteDifferenceLevel"},
        offsets={"obs1": 0.9, "obs2": 0.9},
    )

    assert isinstance(result, tuple)
    assert len(result) == 9
    for r in result:
        assert isinstance(r, pd.Series)

    pd.testing.assert_series_equal(result[0], data["station_id"])
    pd.testing.assert_series_equal(result[1], data["lat"])
    pd.testing.assert_series_equal(result[2], data["lon"])
    pd.testing.assert_series_equal(result[3], data["date"])
    pd.testing.assert_series_equal(result[4], data["vsi"])
    pd.testing.assert_series_equal(result[5], data["dsi"])
    pd.testing.assert_series_equal(result[6], data["flag"])
    pd.testing.assert_series_equal(result[7], data["obs1"])
    pd.testing.assert_series_equal(result[8], data["obs2"])


def test_remove_duplicates_obs_offsets(dummy_data):
    data = dummy_data.copy()
    data["obs1"] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.4]
    data["obs2"] = [2, 3, 3, 3, 3, 3]
    result = remove_duplicates(
        **data.to_dict(),
        compare_level_libraries={"obs1": "AbsoluteDifferenceLevel", "obs2": "AbsoluteDifferenceLevel"},
        offsets={"obs1": 0.5, "obs2": 1.0},
    )
    assert isinstance(result, tuple)
    assert len(result) == 9
    for r in result:
        assert isinstance(r, pd.Series)

    exp_idx = [1, 2, 3, 4, 5]
    pd.testing.assert_series_equal(result[0], data["station_id"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[1], data["lat"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[2], data["lon"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[3], data["date"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[4], data["vsi"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[5], data["dsi"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[6], data["flag"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[7], data["obs1"].iloc[exp_idx])
    pd.testing.assert_series_equal(result[8], data["obs2"].iloc[exp_idx])


def test_remove_duplicates_raises():
    with pytest.raises(ValueError, match="None of `detected`, `station_id`, `lat`, `lon`, `date`, `vsi`, `dsi` and `data` is set."):
        remove_duplicates()


@pytest.mark.parametrize(
    "kwargs, flags, duplicates",
    [
        (
            {},
            [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 0, 0, 0],
            [
                np.nan,
                "H",
                "J",
                ["L", "N"],
                ["F", "Q"],
                "E",
                np.nan,
                "B",
                np.nan,
                "C",
                np.nan,
                "D",
                np.nan,
                "D",
                np.nan,
                np.nan,
                "E",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        ),
        (
            {"ignore_entries": {"station_id": ["AA", "BB"]}},
            [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 0, 3, 3, 3, 3, 0, 3, 0, 0, 0, 0],
            [
                np.nan,
                "H",
                "J",
                ["L", "M", "N", "O"],
                ["F", "Q"],
                "E",
                np.nan,
                "B",
                np.nan,
                "C",
                np.nan,
                "D",
                "D",
                "D",
                "D",
                np.nan,
                "E",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        ),
        (
            {"ignore_nan_both": "vsi"},
            [0, 1, 0, 1, 1, 3, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 3, 0, 0, 0, 0],
            [
                np.nan,
                "H",
                np.nan,
                ["L", "N"],
                ["F", "Q"],
                "E",
                np.nan,
                "B",
                np.nan,
                np.nan,
                np.nan,
                "D",
                np.nan,
                "D",
                np.nan,
                np.nan,
                "E",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        ),
        (
            {"ignore_nan_either": ["vsi", "dsi"]},
            # [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 0, 3, 3],
            [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 3, 3, 3, 3],
            [
                np.nan,
                # ["H", "T", "U"],
                ["H", "T", "U", "R", "S"],
                "J",
                ["L", "N"],
                ["F", "Q"],
                "E",
                np.nan,
                "B",
                np.nan,
                "C",
                np.nan,
                "D",
                np.nan,
                "D",
                np.nan,
                np.nan,
                "E",
                # np.nan,
                "B",
                # np.nan,
                "B",
                "B",
                "B",
            ],
        ),  # not solved yet
        (
            {
                "ignore_entries": {"station_id": ["AA", "BB"]},
                "ignore_nan_either": ["vsi", "dsi"],
            },
            # [0, 1, 1, 3, 1, 3, 0, 3, 0, 3, 0, 3, 1, 3, 3, 0, 3, 0, 0, 3, 3],
            [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 0, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3],
            [
                np.nan,
                # ["H", "T", "U"],
                ["H", "T", "U", "R", "S"],
                "J",
                ["L", "M", "N", "O"],
                ["F", "Q"],
                "E",
                np.nan,
                "B",
                np.nan,
                "C",
                np.nan,
                "D",
                "D",
                "D",
                "D",
                np.nan,
                "E",
                # np.nan,
                "B",
                # np.nan,
                "B",
                "B",
                "B",
            ],
        ),  # not solved yet (see above)
        (
            {"ignore_columns": ["station_id"]},
            [0, 1, 1, 1, 1, 3, 0, 3, 0, 3, 0, 3, 3, 3, 3, 0, 3, 0, 0, 0, 0],
            [
                np.nan,
                "H",
                "J",
                ["L", "M", "N", "O"],
                ["F", "Q"],
                "E",
                np.nan,
                "B",
                np.nan,
                "C",
                np.nan,
                "D",
                "D",
                "D",
                "D",
                np.nan,
                "E",
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
                "H",
                "J",
                ["K", "L", "N"],
                ["F", "P", "Q"],
                "E",
                np.nan,
                "B",
                np.nan,
                "C",
                "D",
                "D",
                np.nan,
                "D",
                np.nan,
                "E",
                "E",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        ),
        (
            {"compare_level_libraries": {"sst": "AbsoluteDifferenceLevel"}, "offsets": {"sst": 1.0}},
            [0, 0, 1, 1, 1, 3, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 0, 0, 0],
            [
                np.nan,
                np.nan,
                "J",
                ["L", "N"],
                ["F", "Q"],
                "E",
                np.nan,
                np.nan,
                np.nan,
                "C",
                np.nan,
                "D",
                np.nan,
                "D",
                np.nan,
                np.nan,
                "E",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        ),
    ],
)
def test_duplicates_expert(expert_data, kwargs, flags, duplicates):
    detected = duplicate_check(data=expert_data, **kwargs)

    result_duplicates = get_duplicates(detected=detected)
    exp_duplicates = pd.Series(duplicates, index=expert_data.index, name="duplicates")
    pd.testing.assert_series_equal(result_duplicates, exp_duplicates)

    result_flags = flag_duplicates(detected=detected)
    exp_flags = pd.Series(flags, index=expert_data.index, name="duplicate_flags")
    pd.testing.assert_series_equal(result_flags, exp_flags)

    result_removed = remove_duplicates(detected=detected)
    exp_removed = expert_data[np.array(flags) != 3]
    for series in result_removed:
        pd.testing.assert_series_equal(series, exp_removed[series.name])
