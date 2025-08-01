from __future__ import annotations

import pandas as pd
import pytest  # noqa
from cdm_reader_mapper import DataBundle, read_tables
from cdm_reader_mapper.common.getting_files import load_file

from marine_qc.auxiliary import (
    failed,
    passed,
    untestable,
    untested,
)
from marine_qc.external_clim import Climatology
from marine_qc.multiple_row_checks import (
    do_multiple_row_check,
)
from marine_qc.qc_grouped_reports import (
    do_bayesian_buddy_check,
    do_mds_buddy_check,
)
from marine_qc.qc_individual_reports import (
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
from marine_qc.qc_sequential_reports import (
    do_few_check,
    do_iquam_track_check,
    do_spike_check,
    do_track_check,
    find_multiple_rounded_values,
    find_repeated_values,
    find_saturated_runs,
)


def _get_parameters(dataset="", release="", deck="", trange=""):
    cache_dir = f".pytest_cache/marine_qc/{dataset}/{deck}"
    input_dir = f"icoads/{release}/{deck}"
    cdm_name = f"icoads_{release}_{deck}_{trange}_subset"
    return cache_dir, input_dir, cdm_name


@pytest.fixture(scope="session")
def testdata():
    cache_dir, input_dir, cdm_name = _get_parameters(
        dataset="ICOADS_R3.0.2T", release="r302", deck="d992", trange="2022-01-01"
    )
    tables = [
        "header",
        "observations-at",
        "observations-dpt",
        "observations-slp",
        "observations-sst",
        "observations-wd",
        "observations-ws",
    ]
    for table in tables:
        load_file(
            f"{input_dir}/cdm_tables/{table}-{cdm_name}.psv",
            cache_dir=cache_dir,
            within_drs=False,
        )

    data_dict = {}
    db_tables = read_tables(cache_dir)

    for table in tables:
        db_table = DataBundle()
        db_table.data = db_tables[table].copy()
        if table == "header":
            db_table.data["platform_type"] = db_table["platform_type"].astype(int)
            db_table.data["latitude"] = db_table["latitude"].astype(float)
            db_table.data["longitude"] = db_table["longitude"].astype(float)
            db_table.data["report_timestamp"] = pd.to_datetime(
                db_table["report_timestamp"],
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )
        else:
            db_table.data["observation_value"] = db_table["observation_value"].astype(
                float
            )
            db_table.data["latitude"] = db_table["latitude"].astype(float)
            db_table.data["longitude"] = db_table["longitude"].astype(float)
            db_table.data["date_time"] = pd.to_datetime(
                db_table["date_time"],
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )
        if table == "observations-slp":
            db_table.data["observation_value"] = db_table.data["observation_value"]

        data_dict[table] = db_table

    return data_dict


@pytest.fixture(scope="session")
def climdata_buddy():
    kwargs = {
        "cache_dir": ".pytest_cache/marine_qc/external_files",
        "within_drs": False,
    }
    buddy_data = {
        "stdev": load_file(
            "metoffice_qc/external_files/HadSST2_pentad_stdev_climatology.nc",
            **kwargs,
        ),
        "mean": load_file(
            "metoffice_qc/external_files/HadSST2_pentad_climatology.nc",
            **kwargs,
        ),
    }
    return buddy_data


@pytest.fixture(scope="session")
def climdata_bayesian():
    kwargs = {
        "cache_dir": ".pytest_cache/marine_qc/external_files",
        "within_drs": False,
    }
    buddy_data = {
        "ostia1": load_file(
            "metoffice_qc/external_files/OSTIA_buddy_range_sampling_error.nc",
            **kwargs,
        ),
        "ostia2": load_file(
            "metoffice_qc/external_files/OSTIA_compare_1x1x5box_to_buddy_average.nc",
            **kwargs,
        ),
        "ostia3": load_file(
            "metoffice_qc/external_files/OSTIA_compare_one_ob_to_1x1x5box.nc",
            **kwargs,
        ),
        "mean": load_file(
            "metoffice_qc/external_files/HadSST2_pentad_climatology.nc",
            **kwargs,
        ),
    }
    return buddy_data


@pytest.fixture(scope="session")
def testdata_track():
    cache_dir, input_dir, cdm_name = _get_parameters(
        dataset="ICOADS_R3.0.2T", release="r302", deck="PT2", trange="2016-04-11"
    )
    tables = [
        "header",
        "observations-at",
        "observations-dpt",
        "observations-sst",
    ]
    for table in tables:
        load_file(
            f"{input_dir}/cdm_tables/{table}-{cdm_name}.psv",
            cache_dir=cache_dir,
            within_drs=False,
        )

    db_tables = read_tables(cache_dir)
    db_tables.data = db_tables.replace("null", None)
    for table in tables:
        db_tables.data[(table, "latitude")] = db_tables[(table, "latitude")].astype(
            float
        )
        db_tables.data[(table, "longitude")] = db_tables[(table, "longitude")].astype(
            float
        )
        if table == "header":
            db_tables.data[(table, "report_timestamp")] = pd.to_datetime(
                db_tables[(table, "report_timestamp")],
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )
            db_tables.data[(table, "station_speed")] = db_tables[
                (table, "station_speed")
            ].astype(float)
            db_tables.data[(table, "station_course")] = db_tables[
                (table, "station_course")
            ].astype(float)
        else:
            db_tables.data[(table, "observation_value")] = db_tables[
                (table, "observation_value")
            ].astype(float)
            db_tables.data[(table, "date_time")] = pd.to_datetime(
                db_tables[(table, "date_time")],
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )
    return db_tables


@pytest.fixture(scope="session")
def climdata():
    kwargs = {
        "cache_dir": ".pytest_cache/marine_qc/external_files",
        "within_drs": False,
    }
    clim_dict = {}
    clim_dict["AT"] = {
        "mean": load_file(
            "metoffice_qc/external_files/AT_pentad_climatology.nc",
            **kwargs,
        ),
        "stdev": load_file(
            "metoffice_qc/external_files/AT_pentad_stdev_climatology.nc",
            **kwargs,
        ),
    }
    clim_dict["DPT"] = {
        "mean": load_file(
            "metoffice_qc/external_files/DPT_pentad_climatology.nc",
            **kwargs,
        ),
        "stdev": load_file(
            "metoffice_qc/external_files/DPT_pentad_stdev_climatology.nc",
            **kwargs,
        ),
    }
    clim_dict["SLP"] = {
        "mean": load_file(
            "metoffice_qc/external_files/SLP_pentad_climatology.nc",
            **kwargs,
        ),
        "stdev": load_file(
            "metoffice_qc/external_files/SLP_pentad_stdev_climatology.nc",
            **kwargs,
        ),
    }
    clim_dict["SST"] = {
        "mean": load_file(
            "metoffice_qc/external_files/SST_daily_climatology_january.nc",
            **kwargs,
        ),
    }
    return clim_dict


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_position_check(testdata, apply_func):
    db_ = testdata["header"].copy()
    if apply_func is True:
        results = db_.apply(
            lambda row: do_position_check(lat=row["latitude"], lon=row["longitude"]),
            axis=1,
        )
    else:
        results = do_position_check(lat=db_["latitude"], lon=db_["longitude"])
    expected = pd.Series([passed] * 13)  # all positions are valid
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_date_check(testdata, apply_func):
    db_ = testdata["header"].copy()
    if apply_func is True:
        results = db_.apply(
            lambda row: do_date_check(date=row["report_timestamp"]), axis=1
        )
    else:
        results = do_date_check(date=db_["report_timestamp"])
    expected = pd.Series(
        [
            untestable,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )  # first entry is null
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_time_check(testdata, apply_func):
    db_ = testdata["header"].copy()
    if apply_func is True:
        results = db_.apply(
            lambda row: do_time_check(date=row["report_timestamp"]), axis=1
        )
    else:
        results = do_time_check(date=db_["report_timestamp"])
    expected = pd.Series([untestable] + [passed] * 12)  # first entry is null
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_day_check(testdata, apply_func):
    db_ = testdata["header"].copy()
    if apply_func is True:
        results = db_.apply(
            lambda row: do_day_check(
                date=row["report_timestamp"],
                lat=row["latitude"],
                lon=row["longitude"],
            ),
            axis=1,
        )
    else:
        results = do_day_check(
            date=db_["report_timestamp"],
            lat=db_["latitude"],
            lon=db_["longitude"],
        )
    expected = pd.Series([untestable] + [failed] * 12)  # observations are at night
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_at_missing_value_check(testdata, apply_func):
    db_ = testdata["observations-at"].copy()
    if apply_func is True:
        results = db_.apply(
            lambda row: do_missing_value_check(value=row["observation_value"]),
            axis=1,
        )
    else:
        results = do_missing_value_check(value=db_["observation_value"])
    expected = pd.Series(
        [
            passed,
            passed,
            failed,
            passed,
            failed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_at_hard_limit_check(testdata, apply_func):
    db_ = testdata["observations-at"].copy()
    params = {
        "limits": [-80, 65],
        "units": {"limits": "degC"},
    }
    if apply_func is True:
        results = db_.apply(
            lambda row: do_hard_limit_check(
                value=row["observation_value"],
                **params,
            ),
            axis=1,
        )
    else:
        results = do_hard_limit_check(
            value=db_["observation_value"],
            **params,
        )
    expected = pd.Series(
        [
            passed,
            passed,
            untestable,
            passed,
            untestable,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_at_missing_value_clim_check(testdata, climdata, apply_func):
    db_ = testdata["observations-at"].copy()
    climatology = Climatology.open_netcdf_file(
        climdata["AT"]["mean"],
        "at",
        time_axis="pentad_time",
    )
    if apply_func is True:
        results = db_.apply(
            lambda row: do_missing_value_clim_check(
                climatology=climatology,
                lat=row["latitude"],
                lon=row["longitude"],
                date=row["date_time"],
            ),
            axis=1,
        )
    else:
        results = do_missing_value_clim_check(
            climatology=climatology,
            lat=db_["latitude"],
            lon=db_["longitude"],
            date=db_["date_time"],
        )
    expected = pd.Series(
        [
            failed,
            passed,
            failed,
            passed,
            failed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_at_climatology_check(testdata, climdata, apply_func):
    db_ = testdata["observations-at"].copy()
    climatology = Climatology.open_netcdf_file(
        climdata["AT"]["mean"],
        "at",
        time_axis="pentad_time",
        target_units="K",
        source_units="degC",
    )
    if apply_func is True:
        results = db_.apply(
            lambda row: do_climatology_check(
                value=row["observation_value"],
                climatology=climatology,
                maximum_anomaly=10.0,  # K
                lat=row["latitude"],
                lon=row["longitude"],
                date=row["date_time"],
            ),
            axis=1,
        )
    else:
        results = do_climatology_check(
            value=db_["observation_value"],
            climatology=climatology,
            maximum_anomaly=10.0,
            lat=db_["latitude"],
            lon=db_["longitude"],
            date=db_["date_time"],
        )
    expected = pd.Series(
        [
            untestable,
            failed,
            untestable,
            passed,
            untestable,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_at_climatology_plus_stdev_check(testdata, climdata, apply_func):
    db_ = testdata["observations-at"].copy()
    climatology = Climatology.open_netcdf_file(
        climdata["AT"]["mean"],
        "at",
        time_axis="pentad_time",
        target_units="K",
        source_units="degC",
    )
    stdev = Climatology.open_netcdf_file(
        climdata["AT"]["stdev"],
        "at",
        time_axis="pentad_time",
    )
    if apply_func is True:
        results = db_.apply(
            lambda row: do_climatology_check(
                value=row["observation_value"],
                climatology=climatology,
                standard_deviation=stdev,
                standard_deviation_limits=[1.0, 4.0],  # K
                maximum_anomaly=5.5,  # K
                lat=row["latitude"],
                lon=row["longitude"],
                date=row["date_time"],
            ),
            axis=1,
        )
    else:
        results = do_climatology_check(
            value=db_["observation_value"],
            climatology=climatology,
            standard_deviation=stdev,
            standard_deviation_limits=[1.0, 4.0],  # K
            maximum_anomaly=5.5,  # K
            lat=db_["latitude"],
            lon=db_["longitude"],
            date=db_["date_time"],
        )
    expected = pd.Series(
        [
            untestable,
            passed,
            untestable,
            passed,
            untestable,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_slp_missing_value_check(testdata, apply_func):
    db_ = testdata["observations-slp"].copy()
    if apply_func is True:
        results = db_.apply(
            lambda row: do_missing_value_check(value=row["observation_value"]),
            axis=1,
        )
    else:
        results = do_missing_value_check(value=db_["observation_value"])
    expected = pd.Series(
        [
            passed,
            passed,
            failed,
            passed,
            failed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_slp_missing_value_clim_check(testdata, climdata, apply_func):
    db_ = testdata["observations-slp"].copy()
    climatology = Climatology.open_netcdf_file(climdata["SLP"]["mean"], "slp")
    if apply_func is True:
        results = db_.apply(
            lambda row: do_missing_value_clim_check(
                climatology=climatology,
                lat=row["latitude"],
                lon=row["longitude"],
                date=row["date_time"],
            ),
            axis=1,
        )
    else:
        results = do_missing_value_clim_check(
            climatology=climatology,
            lat=db_["latitude"],
            lon=db_["longitude"],
            date=db_["date_time"],
        )
    expected = pd.Series(
        [
            failed,
            passed,
            failed,
            passed,
            failed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_slp_climatology_plus_stdev_with_lowbar_check(
    testdata, climdata, apply_func
):
    db_ = testdata["observations-slp"].copy()
    climatology = Climatology.open_netcdf_file(
        climdata["SLP"]["mean"],
        "slp",
        target_units="Pa",
        source_units="hPa",
    )
    stdev = Climatology.open_netcdf_file(
        climdata["SLP"]["stdev"],
        "slp",
        target_units="Pa",
        source_units="hPa",
    )
    if apply_func is True:
        results = db_.apply(
            lambda row: do_climatology_check(
                value=row["observation_value"],
                climatology=climatology,
                standard_deviation=stdev,
                maximum_anomaly=300,  # Pa
                lowbar=1000,  # Pa
                lat=row["latitude"],
                lon=row["longitude"],
                date=row["date_time"],
            ),
            axis=1,
        )
    else:
        results = do_climatology_check(
            value=db_["observation_value"],
            climatology=climatology,
            standard_deviation=stdev,
            maximum_anomaly=300,  # Pa
            lowbar=1000,  # Pa
            lat=db_["latitude"],
            lon=db_["longitude"],
            date=db_["date_time"],
        )
    expected = pd.Series(
        [
            untestable,
            passed,
            untestable,
            passed,
            untestable,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_dpt_missing_value_check(testdata, apply_func):
    db_ = testdata["observations-dpt"].copy()
    if apply_func is True:
        results = db_.apply(
            lambda row: do_missing_value_check(value=row["observation_value"]),
            axis=1,
        )
    else:
        results = do_missing_value_check(value=db_["observation_value"])
    expected = pd.Series(
        [
            failed,
            passed,
            failed,
            passed,
            failed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_dpt_hard_limit_check(testdata, apply_func):
    db_ = testdata["observations-dpt"].copy()
    params = {"limits": [-80, 65], "units": {"limits": "degC"}}
    if apply_func is True:
        results = db_.apply(
            lambda row: do_hard_limit_check(
                value=row["observation_value"],
                **params,
            ),
            axis=1,
        )
    else:
        results = do_hard_limit_check(
            value=db_["observation_value"],
            **params,
        )
    expected = pd.Series(
        [
            untestable,
            passed,
            untestable,
            passed,
            untestable,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_dpt_missing_value_clim_check(testdata, climdata, apply_func):
    db_ = testdata["observations-dpt"].copy()
    climatology = Climatology.open_netcdf_file(
        climdata["DPT"]["mean"],
        "dpt",
        time_axis="pentad_time",
    )
    if apply_func is True:
        results = db_.apply(
            lambda row: do_missing_value_clim_check(
                climatology=climatology,
                lat=row["latitude"],
                lon=row["longitude"],
                date=row["date_time"],
            ),
            axis=1,
        )
    else:
        results = do_missing_value_clim_check(
            climatology=climatology,
            lat=db_["latitude"],
            lon=db_["longitude"],
            date=db_["date_time"],
        )

    expected = pd.Series(
        [
            failed,  # This should be untesable since lat is not available
            passed,
            failed,  # This should be untesable since lat is not available
            passed,
            failed,  # This should be untesable since lat is not available
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_dpt_climatology_plus_stdev_check(testdata, climdata, apply_func):
    db_ = testdata["observations-dpt"].copy()
    climatology = Climatology.open_netcdf_file(
        climdata["DPT"]["mean"],
        "dpt",
        time_axis="pentad_time",
        target_units="K",
        source_units="degC",
    )
    stdev = Climatology.open_netcdf_file(
        climdata["DPT"]["stdev"],
        "dpt",
        time_axis="pentad_time",
    )
    if apply_func is True:
        results = db_.apply(
            lambda row: do_climatology_check(
                value=row["observation_value"],
                climatology=climatology,
                standard_deviation=stdev,
                standard_deviation_limits=[1.0, 4.0],  # K
                maximum_anomaly=5.5,  # K
                lat=row["latitude"],
                lon=row["longitude"],
                date=row["date_time"],
            ),
            axis=1,
        )
    else:
        results = do_climatology_check(
            value=db_["observation_value"],
            climatology=climatology,
            standard_deviation=stdev,
            standard_deviation_limits=[1.0, 4.0],  # K
            maximum_anomaly=5.5,  # K
            lat=db_["latitude"],
            lon=db_["longitude"],
            date=db_["date_time"],
        )
    expected = pd.Series(
        [
            untestable,
            passed,
            untestable,
            passed,
            untestable,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_supersaturation_check(testdata, apply_func):
    db_ = testdata["observations-at"].copy()
    db2_ = testdata["observations-dpt"].copy()
    if apply_func is True:
        db_.data["observation_value_dpt"] = db2_["observation_value"]
        results = db_.apply(
            lambda row: do_supersaturation_check(
                dpt=row["observation_value_dpt"],
                at2=row["observation_value"],
            ),
            axis=1,
        )
    else:
        results = do_supersaturation_check(
            dpt=db2_["observation_value"],
            at2=db_["observation_value"],
        )
    expected = pd.Series(
        [
            untestable,
            passed,
            untestable,
            passed,
            untestable,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_sst_missing_value_check(testdata, apply_func):
    db_ = testdata["observations-sst"].copy()
    if apply_func is True:
        results = db_.apply(
            lambda row: do_missing_value_check(value=row["observation_value"]),
            axis=1,
        )
    else:
        results = do_missing_value_check(value=db_["observation_value"])
    expected = pd.Series(
        [
            passed,
            passed,
            failed,
            failed,
            failed,
            failed,
            failed,
            failed,
            failed,
            failed,
            failed,
            failed,
            failed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_sst_freeze_check(testdata, apply_func):
    db_ = testdata["observations-sst"].copy()
    params = {
        "freezing_point": -1.8,
        "freeze_check_n_sigma": 2.0,
        "units": {"freezing_point": "degC"},
    }
    if apply_func is True:
        results = db_.apply(
            lambda row: do_sst_freeze_check(
                sst=row["observation_value"],
                **params,
            ),
            axis=1,
        )
    else:
        results = do_sst_freeze_check(
            sst=db_["observation_value"],
            **params,
        )
    expected = pd.Series(
        [
            passed,
            passed,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_sst_hard_limit_check(testdata, apply_func):
    db_ = testdata["observations-sst"].copy()
    params = {
        "limits": [-5, 65],
        "units": {"limits": "degC"},
    }
    if apply_func is True:
        results = db_.apply(
            lambda row: do_hard_limit_check(
                value=row["observation_value"],
                **params,
            ),
            axis=1,
        )
    else:
        results = do_hard_limit_check(
            value=db_["observation_value"],
            **params,
        )
    expected = pd.Series(
        [
            passed,
            passed,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_sst_missing_value_clim_check(testdata, climdata, apply_func):
    db_ = testdata["observations-sst"].copy()
    climatology = Climatology.open_netcdf_file(
        climdata["SST"]["mean"],
        "sst",
        valid_ntime=31,
    )
    if apply_func is True:
        results = db_.apply(
            lambda row: do_missing_value_clim_check(
                climatology=climatology,
                lat=row["latitude"],
                lon=row["longitude"],
                date=row["date_time"],
            ),
            axis=1,
        )
    else:
        results = do_missing_value_clim_check(
            climatology=climatology,
            lat=db_["latitude"],
            lon=db_["longitude"],
            date=db_["date_time"],
        )
    expected = pd.Series(
        [
            failed,
            passed,
            failed,
            failed,
            failed,
            failed,
            failed,
            failed,
            failed,
            failed,
            failed,
            failed,
            failed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_sst_climatology_check(testdata, climdata, apply_func):
    db_ = testdata["observations-sst"].copy()
    climatology = Climatology.open_netcdf_file(
        climdata["SST"]["mean"],
        "sst",
        valid_ntime=31,
    )
    if apply_func is True:
        results = db_.apply(
            lambda row: do_climatology_check(
                value=row["observation_value"],
                climatology=climatology,
                maximum_anomaly=1.0,
                lat=row["latitude"],
                lon=row["longitude"],
                date=row["date_time"],
            ),
            axis=1,
        )
    else:
        results = do_climatology_check(
            value=db_["observation_value"],
            climatology=climatology,
            maximum_anomaly=1.0,
            lat=db_["latitude"],
            lon=db_["longitude"],
            date=db_["date_time"],
        )
    expected = pd.Series(
        [
            untestable,
            passed,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
            untestable,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_wind_speed_missing_value_check(testdata, apply_func):
    db_ = testdata["observations-ws"].copy()
    if apply_func is True:
        results = db_.apply(
            lambda row: do_missing_value_check(value=row["observation_value"]),
            axis=1,
        )
    else:
        results = do_missing_value_check(value=db_["observation_value"])
    expected = pd.Series(
        [
            passed,
            passed,
            failed,
            passed,
            failed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_wind_speed_hard_limit_check(testdata, apply_func):
    db_ = testdata["observations-ws"].copy()
    if apply_func is True:
        results = db_.apply(
            lambda row: do_hard_limit_check(
                value=row["observation_value"],
                limits=[0, 13],
            ),
            axis=1,
        )
    else:
        results = do_hard_limit_check(
            value=db_["observation_value"],
            limits=[0, 13],
        )
    expected = pd.Series(
        [
            passed,
            passed,
            untestable,
            passed,
            untestable,
            failed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_wind_direction_missing_value_check(testdata, apply_func):
    db_ = testdata["observations-wd"].copy()
    if apply_func is True:
        results = db_.apply(
            lambda row: do_missing_value_check(value=row["observation_value"]),
            axis=1,
        )
    else:
        results = do_missing_value_check(value=db_["observation_value"])
    expected = pd.Series(
        [
            passed,
            passed,
            failed,
            passed,
            failed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_wind_direction_hard_limit_check(testdata, apply_func):
    db_ = testdata["observations-wd"].copy()
    if apply_func is True:
        results = db_.apply(
            lambda row: do_hard_limit_check(
                value=row["observation_value"], limits=[0, 360]
            ),
            axis=1,
        )
    else:
        results = do_hard_limit_check(value=db_["observation_value"], limits=[0, 360])
    expected = pd.Series(
        [
            passed,
            passed,
            untestable,
            passed,
            untestable,
            passed,
            failed,
            failed,
            passed,
            passed,
            passed,
            passed,
            passed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


@pytest.mark.parametrize("apply_func", [False, True])
def test_do_wind_consistency_check(testdata, apply_func):
    db_ = testdata["observations-ws"].copy()
    db2_ = testdata["observations-wd"].copy()
    if apply_func is True:
        db_.data["observation_value_wd"] = db2_["observation_value"]
        results = db_.apply(
            lambda row: do_wind_consistency_check(
                wind_speed=row["observation_value"],
                wind_direction=row["observation_value_wd"],
            ),
            axis=1,
        )
    else:
        results = do_wind_consistency_check(
            db_["observation_value"],
            db2_["observation_value"],
        )
    expected = pd.Series(
        [
            passed,
            passed,
            untestable,
            passed,
            untestable,
            passed,
            passed,
            passed,
            failed,
            failed,
            failed,
            failed,
            failed,
        ]
    )
    pd.testing.assert_series_equal(results, expected)


def test_do_spike_check(testdata_track):
    db_ = testdata_track.copy()
    db_.data.loc[152, ("observations-at", "observation_value")] = 1000.0
    db_.data.loc[162, ("observations-at", "observation_value")] = 1000.0
    db_.data.loc[174, ("observations-at", "observation_value")] = 1000.0
    db_.data.loc[198, ("observations-at", "observation_value")] = 1000.0
    groups = db_.groupby(
        [("header", "primary_station_id")], group_keys=False, sort=False
    )
    results = groups.apply(
        lambda track: do_spike_check(
            value=track[("observations-at", "observation_value")],
            lat=track[("observations-at", "latitude")],
            lon=track[("observations-at", "longitude")],
            date=track[("observations-at", "date_time")],
            max_gradient_space=0.5,
            max_gradient_time=1.0,
            delta_t=1.0,
            n_neighbours=5,
        ),
        include_groups=False,
    )
    expected = pd.Series([passed] * len(results), index=results.index)
    expected.iloc[152] = 1
    expected.iloc[162] = 1
    expected.iloc[174] = 1
    expected.iloc[198] = 1
    pd.testing.assert_series_equal(results, expected, check_names=False)


def test_do_track_check(testdata_track):
    db_ = testdata_track.copy()
    db_.data.loc[2, ("header", "latitude")] = -23.0
    db_.data.loc[12, ("header", "latitude")] = -23.0
    db_.data.loc[24, ("header", "latitude")] = -23.0
    db_.data.loc[48, ("header", "latitude")] = -23.0

    groups = db_.groupby(
        [("header", "primary_station_id")], group_keys=False, sort=False
    )
    results = groups.apply(
        lambda track: do_track_check(
            vsi=track[("header", "station_speed")],
            dsi=track[("header", "station_course")],
            lat=track[("header", "latitude")],
            lon=track[("header", "longitude")],
            date=track[("header", "report_timestamp")],
            max_direction_change=60.0,
            max_speed_change=10.0,
            max_absolute_speed=40.0,
            max_midpoint_discrepancy=150.0,
        ),
        include_groups=False,
    ).squeeze()

    expected = pd.Series([passed] * len(results))
    expected.iloc[2] = 1
    expected.iloc[12] = 1
    expected.iloc[24] = 1
    expected.iloc[48] = 1
    pd.testing.assert_series_equal(results, expected, check_names=False)


def test_do_few_check_passed(testdata_track):
    db_ = testdata_track.copy()
    groups = db_.groupby(
        [("header", "primary_station_id")], group_keys=False, sort=False
    )
    results = groups.apply(
        lambda track: do_few_check(
            value=track[("header", "latitude")],
        ),
        include_groups=False,
    )
    expected = pd.Series([passed] * len(results))
    pd.testing.assert_series_equal(results, expected, check_names=False)


def test_do_few_check_failed(testdata_track):
    db_ = testdata_track.copy()
    db_.data = db_.data[:2]
    groups = db_.groupby(
        [("header", "primary_station_id")], group_keys=False, sort=False
    )
    results = groups.apply(
        lambda track: do_few_check(
            value=track[("header", "latitude")],
        ),
        include_groups=False,
    )
    results = results.iloc[0, :]
    expected = pd.Series([failed] * len(results))
    pd.testing.assert_series_equal(results, expected, check_names=False)


def test_do_iquam_track_check(testdata_track):
    db_ = testdata_track.copy()
    groups = db_.groupby(
        [("header", "primary_station_id")], group_keys=False, sort=False
    )
    results = groups.apply(
        lambda track: do_iquam_track_check(
            lat=track[("header", "latitude")],
            lon=track[("header", "longitude")],
            date=track[("header", "report_timestamp")],
            speed_limit=60.0,
            delta_d=1.11,
            delta_t=0.01,
            n_neighbours=5,
        ),
        include_groups=False,
    )
    expected = pd.Series([passed] * len(results))
    pd.testing.assert_series_equal(results, expected, check_names=False)


def test_find_repeated_values(testdata_track):
    db_ = testdata_track.copy()
    repeated = db_.data.loc[160, ("observations-at", "observation_value")]
    for i in range(161, 201):
        db_.data.loc[i, ("observations-at", "observation_value")] = repeated
    groups = db_.groupby(
        [("header", "primary_station_id")], group_keys=False, sort=False
    )
    results = groups.apply(
        lambda track: find_repeated_values(
            value=track[("observations-at", "observation_value")],
            min_count=20,
            threshold=0.7,
        ),
        include_groups=False,
    )
    expected = pd.Series([passed] * len(results), index=results.index)
    for i in range(160, 201):
        expected.iloc[i] = 1
    pd.testing.assert_series_equal(results, expected, check_names=False)


def test_find_saturated_runs(testdata_track):
    db_ = testdata_track.copy()
    for i in range(161, 201):
        db_.data.loc[i, ("observations-at", "observation_value")] = 300.0
        db_.data.loc[i, ("observations-dpt", "observation_value")] = 300.0
    groups = db_.groupby(
        [("header", "primary_station_id")], group_keys=False, sort=False
    )
    results = groups.apply(
        lambda track: find_saturated_runs(
            at=track[("observations-at", "observation_value")],
            dpt=track[("observations-dpt", "observation_value")],
            lat=track[("header", "latitude")],
            lon=track[("header", "longitude")],
            date=track[("header", "report_timestamp")],
            min_time_threshold=38.0,
            shortest_run=4,
        ),
        include_groups=False,
    )
    expected = pd.Series([passed] * len(results), index=results.index)
    for i in range(161, 201):
        expected.iloc[i] = 1
    pd.testing.assert_series_equal(results, expected, check_names=False)


def test_find_multiple_rounded_values(testdata_track):
    db_ = testdata_track.copy()
    db_.data[("observations-at", "observation_value")] -= 273.15
    for i in range(160, 200):
        db_.data.loc[i, ("observations-at", "observation_value")] = 30.0
    groups = db_.groupby(
        [("header", "primary_station_id")], group_keys=False, sort=False
    )
    results = groups.apply(
        lambda track: find_multiple_rounded_values(
            value=track[("observations-at", "observation_value")],
            min_count=20,
            threshold=0.5,
        ),
        include_groups=False,
    )
    expected = pd.Series([passed] * len(results))
    for i in range(160, 200):
        expected.iloc[i] = failed
    pd.testing.assert_series_equal(results, expected, check_names=False)


@pytest.mark.parametrize(
    "return_method, expected",
    [
        (
            "all",
            {
                "MISSVAL": [
                    passed,
                    passed,
                    failed,
                    passed,
                    failed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                ],
                "MISSCLIM": [
                    failed,
                    passed,
                    failed,
                    passed,
                    failed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                ],
                "HLIMITS": [
                    passed,
                    passed,
                    untestable,
                    passed,
                    untestable,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                ],
                "CLIM1": [
                    untestable,
                    failed,
                    untestable,
                    passed,
                    untestable,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                ],
                "CLIM2": [
                    untestable,
                    passed,
                    untestable,
                    passed,
                    untestable,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                ],
            },
        ),
        (
            "passed",
            {
                "MISSVAL": [
                    passed,
                    passed,
                    failed,
                    passed,
                    failed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                ],
                "MISSCLIM": [
                    untested,
                    untested,
                    failed,
                    untested,
                    failed,
                    untested,
                    untested,
                    untested,
                    untested,
                    untested,
                    untested,
                    untested,
                    untested,
                ],
                "HLIMITS": [
                    untested,
                    untested,
                    untestable,
                    untested,
                    untestable,
                    untested,
                    untested,
                    untested,
                    untested,
                    untested,
                    untested,
                    untested,
                    untested,
                ],
                "CLIM1": [
                    untested,
                    untested,
                    untestable,
                    untested,
                    untestable,
                    untested,
                    untested,
                    untested,
                    untested,
                    untested,
                    untested,
                    untested,
                    untested,
                ],
                "CLIM2": [
                    untested,
                    untested,
                    untestable,
                    untested,
                    untestable,
                    untested,
                    untested,
                    untested,
                    untested,
                    untested,
                    untested,
                    untested,
                    untested,
                ],
            },
        ),
        (
            "failed",
            {
                "MISSVAL": [
                    passed,
                    passed,
                    failed,
                    passed,
                    failed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                ],
                "MISSCLIM": [
                    failed,
                    passed,
                    untested,
                    passed,
                    untested,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                ],
                "HLIMITS": [
                    untested,
                    passed,
                    untested,
                    passed,
                    untested,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                ],
                "CLIM1": [
                    untested,
                    failed,
                    untested,
                    passed,
                    untested,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                ],
                "CLIM2": [
                    untested,
                    untested,
                    untested,
                    passed,
                    untested,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                    passed,
                ],
            },
        ),
    ],
)
@pytest.mark.parametrize("apply_func", [False, True])
def test_multiple_row_check(testdata, climdata, return_method, expected, apply_func):
    db_ = testdata["observations-at"].copy()
    climatology = Climatology.open_netcdf_file(
        climdata["AT"]["mean"],
        "at",
        time_axis="pentad_time",
        target_units="K",
        source_units="degC",
    )
    stdev = Climatology.open_netcdf_file(
        climdata["AT"]["stdev"],
        "at",
        time_axis="pentad_time",
    )
    preproc_dict = {
        "climatology": {
            "func": "get_climatological_value",
            "names": {
                "lat": "latitude",
                "lon": "longitude",
                "date": "date_time",
            },
            "inputs": climatology,
        },
        "standard_deviation": {
            "func": "get_climatological_value",
            "names": {
                "lat": "latitude",
                "lon": "longitude",
                "date": "date_time",
            },
            "inputs": stdev,
        },
    }
    qc_dict = {
        "MISSVAL": {
            "func": "do_missing_value_check",
            "names": {"value": "observation_value"},
        },
        "MISSCLIM": {
            "func": "do_missing_value_clim_check",
            "arguments": {"climatology": "__preprocessed__"},
        },
        "HLIMITS": {
            "func": "do_hard_limit_check",
            "names": {"value": "observation_value"},
            "arguments": {
                "limits": [-80, 65],  # degC
                "units": {"limits": "degC"},
            },
        },
        "CLIM1": {
            "func": "do_climatology_check",
            "names": {
                "value": "observation_value",
            },
            "arguments": {
                "climatology": "__preprocessed__",
                "maximum_anomaly": 10.0,  # K
            },
        },
        "CLIM2": {
            "func": "do_climatology_check",
            "names": {
                "value": "observation_value",
            },
            "arguments": {
                "climatology": "__preprocessed__",
                "standard_deviation": "__preprocessed__",
                "standard_deviation_limits": [1.0, 4.0],  # K
                "maximum_anomaly": 5.5,  # K
            },
        },
    }
    if apply_func is True:
        results = db_.apply(
            lambda row: do_multiple_row_check(
                data=row,
                qc_dict=qc_dict,
                preproc_dict=preproc_dict,
                return_method=return_method,
            ),
            axis=1,
        )
    else:
        results = do_multiple_row_check(
            data=db_.data,
            qc_dict=qc_dict,
            preproc_dict=preproc_dict,
            return_method=return_method,
        )
    expected = pd.DataFrame(expected)
    pd.testing.assert_frame_equal(results, expected)


def test_buddy_check(climdata_buddy, testdata_track):

    sst_climatology = Climatology.open_netcdf_file(
        climdata_buddy["mean"],
        "sst",
        time_axis="time",
        source_units="degC",
        target_units="K",
    )
    stdev_climatology = Climatology.open_netcdf_file(
        climdata_buddy["stdev"], "sst", time_axis="time"
    )

    limits = [[1, 1, 2], [2, 2, 2], [1, 1, 4], [2, 2, 4]]
    number_of_obs_thresholds = [[0, 5, 15, 100], [0], [0, 5, 15, 100], [0]]
    multipliers = [[4.0, 3.5, 3.0, 2.5], [4.0], [4.0, 3.5, 3.0, 2.5], [4.0]]

    db_ = testdata_track["observations-sst"].copy()
    db_.dropna(subset=["observation_value"], inplace=True, ignore_index=True)

    result = do_mds_buddy_check(
        lat=db_["latitude"],
        lon=db_["longitude"],
        date=db_["date_time"],
        value=db_["observation_value"],
        climatology=sst_climatology,
        standard_deviation=stdev_climatology,
        limits=limits,
        number_of_obs_thresholds=number_of_obs_thresholds,
        multipliers=multipliers,
    )

    for i, flag in enumerate(result):
        if i in [7, 8, 9, 10, 12, 13, 14, 15, 45]:
            assert flag == failed
        else:
            assert flag == passed


def test_bayesian_buddy_check(climdata_bayesian, testdata_track):

    sst_climatology = Climatology.open_netcdf_file(
        climdata_bayesian["mean"],
        "sst",
        time_axis="time",
        source_units="degC",
        target_units="K",
    )
    ostia1_climatology = Climatology.open_netcdf_file(
        climdata_bayesian["ostia1"], "sst", time_axis="time"
    )
    ostia2_climatology = Climatology.open_netcdf_file(
        climdata_bayesian["ostia2"], "sst", time_axis="time"
    )
    ostia3_climatology = Climatology.open_netcdf_file(
        climdata_bayesian["ostia3"], "sst", time_axis="time"
    )

    db_ = testdata_track["observations-sst"].copy()
    db_.dropna(subset=["observation_value"], inplace=True, ignore_index=True)

    result = do_bayesian_buddy_check(
        lat=db_["latitude"],
        lon=db_["longitude"],
        date=db_["date_time"],
        value=db_["observation_value"],
        climatology=sst_climatology,
        stdev1=ostia1_climatology,
        stdev2=ostia2_climatology,
        stdev3=ostia3_climatology,
        prior_probability_of_gross_error=0.05,
        quantization_interval=0.1,
        one_sigma_measurement_uncertainty=1.0,
        limits=[2, 2, 4],
        noise_scaling=3.0,
        maximum_anomaly=8.0,
        fail_probability=0.3,
    )

    for i, flag in enumerate(result):
        if i in [7, 8, 9, 10, 11]:
            assert flag == failed
        else:
            assert flag == passed
