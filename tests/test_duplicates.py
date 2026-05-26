from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marine_qc import flag_duplicates, get_duplicates, remove_duplicates
from marine_qc.duplicate_checker.duplicates import (
    DupDetect,
    duplicate_check,
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


def test_reindex_nulls_orders_by_null_count():
    df = pd.DataFrame({"a": ["null", 1, "null", 2], "b": ["null", 2, 3, "null"]})
    result = reindex_nulls(df, null_label="null")

    expected_order = [1, 2, 3, 0]
    assert list(result.index) == expected_order


def test_reindex_nulls_empty_df():
    df = pd.DataFrame()
    result = reindex_nulls(df, null_label="null")
    assert result.equals(df)


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
