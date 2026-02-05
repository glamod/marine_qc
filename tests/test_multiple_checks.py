from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marine_qc import (
    do_hard_limit_check,
    do_multiple_individual_check,
    do_multiple_sequential_check,
)
from marine_qc.auxiliary import failed, passed, untested
from marine_qc.multiple_checks import (
    _apply_qc_to_masked_rows,
    _do_multiple_check,
    _get_function,
    _get_preprocessed_args,
    _get_requests_from_params,
    _group_iterator,
    _is_func_param,
    _is_in_data,
    _normalize_groupby,
    _prepare_all_inputs,
    _prepare_preprocessed_vars,
    _prepare_qc_functions,
    _run_qc_engine,
    _validate_and_normalize_input,
    _validate_args,
    _validate_dict,
)


def simple_test_function(in_param, **kwargs):
    return in_param * 2


def simple_test_function_no_kwargs(in_param):
    return in_param * 2


@pytest.fixture
def series_ind():
    return pd.Series([1, 2, 3, 4], name="value")


@pytest.fixture
def df_ind():
    return pd.DataFrame({"value1": [1, 2, 3, 4], "value2": [1, 1, 2, 2]})


@pytest.fixture
def df_seq():
    return pd.DataFrame(
        {
            "lat": np.arange(10) + 50,
            "lon": np.arange(10) + 50,
            "date": pd.date_range(start="1850-01-01", freq="1h", periods=10),
            "value": [5, 100, 5, 5, 5, 5, 5, 100, 5, 5],
            "name": ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A"],
        }
    )


@pytest.fixture
def qc_dict():
    return {
        "test1": {
            "func": "do_hard_limit_check",
            "names": {"value": "value1"},
            "arguments": {"limits": [2, 3]},
        },
        "test2": {
            "func": "do_hard_limit_check",
            "names": {"value": "value2"},
            "arguments": {"limits": [2, 3]},
        },
    }


@pytest.fixture
def qc_dict_seq():
    return {
        "test1": {
            "func": "do_spike_check",
            "names": {
                "value": "value",
                "lat": "lat",
                "lon": "lon",
                "date": "date",
            },
            "arguments": {
                "max_gradient_space": 0.5,
                "max_gradient_time": 1.0,
                "delta_t": 2.0,
                "n_neighbours": 5,
            },
        },
        "test2": {
            "func": "do_spike_check",
            "names": {
                "value": "value",
                "lat": "lat",
                "lon": "lon",
                "date": "date",
            },
            "arguments": {
                "max_gradient_space": 0.5,
                "max_gradient_time": 1.0,
                "delta_t": 2.0,
                "n_neighbours": 5,
            },
        },
    }


def test_get_function():
    result = _get_function("do_day_check")
    assert callable(result)
    assert result.__name__ == "do_day_check"


def test_get_function_raises():
    with pytest.raises(NameError, match="Function 'BAD_NAME' is not defined."):
        _get_function("BAD_NAME")


def test_is_func_param():
    assert not _is_func_param(_is_func_param, "Non existent parameter")
    assert _is_func_param(_is_func_param, "param")
    assert _is_func_param(simple_test_function, "non existent parameter")


def test_is_in_data_series(series_ind):
    assert _is_in_data("value", series_ind)
    assert not _is_in_data("value2", series_ind)


def test_is_in_data_df(df_ind):
    assert _is_in_data("value1", df_ind)
    assert _is_in_data("value2", df_ind)
    assert not _is_in_data("value3", df_ind)


def test_is_in_data_raises():
    with pytest.raises(TypeError, match="Unsupported data type"):
        _is_in_data("test_name", [1, 2, 3])


def test_validate_dict_passing():
    _validate_dict({"test": {"value": 1}})


@pytest.mark.parametrize(
    "input_value",
    [
        1,
        1.0,
        "1",
        [1, 2],
        (1, 2),
        pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
        pd.Series([1, 2]),
    ],
)
def test_validate_dict_invalid_input(input_value):
    with pytest.raises(TypeError, match="must be a dictionary"):
        _validate_dict(input_value)


@pytest.mark.parametrize(
    "input_dict",
    [
        {1: "test"},
        {1.0: "test"},
    ],
)
def test_validate_dict_invalid_keys(input_dict):
    with pytest.raises(TypeError, match="must be a string"):
        _validate_dict(input_dict)


@pytest.mark.parametrize(
    "input_dict",
    [
        {"test": 1},
        {"test": 1.0},
        {"test": "1"},
        {"test": [1, 2]},
        {"test": (1, 2)},
        {"test": pd.DataFrame({"A": [1, 2], "B": [3, 4]})},
        {"test": pd.Series([1, 2])},
    ],
)
def test_validate_dict_invalid_values(input_dict):
    with pytest.raises(TypeError, match="must be a dictionary"):
        _validate_dict(input_dict)


def test_validate_args_passing_required_only():
    _validate_args(simple_test_function, {"in_param": 2})


def test_validate_args_passing_with_extra_kwargs():
    _validate_args(
        simple_test_function,
        {"in_param": 2, "extra": 123, "another": "value"},
    )


def test_validate_args_passing_with_kwargs_only():
    _validate_args(
        simple_test_function,
        {"in_param": 2, "unexpected_param": 42},
    )


def test_validate_args_passing_with_args_and_kwargs():
    _validate_args(simple_test_function, 2, {"extra": 123})


def test_validate_args_invalid_param():
    with pytest.raises(ValueError, match="is not a valid parameter of function"):
        _validate_args(simple_test_function_no_kwargs, {"in_param": 2, "extra": 123})


def test_validate_args_missing_required_param():
    with pytest.raises(TypeError, match="is missing for function"):
        _validate_args(simple_test_function, {"extra": 123})


def test_validate_args_too_many_args():
    with pytest.raises(TypeError, match="Too many positional arguments for function"):
        _validate_args(simple_test_function_no_kwargs, (1, 2))


def test_get_requests_from_params(df_ind):
    test_params = {"in_param": "value1"}
    result = _get_requests_from_params(test_params, simple_test_function, df_ind)
    assert "in_param" in result
    assert np.all(result["in_param"] == df_ind["value1"])

    test_params = {"in_param": "value1", "second_param": "value2"}
    result = _get_requests_from_params(test_params, simple_test_function, df_ind)
    assert "in_param" in result
    assert "second_param" in result
    assert np.all(result["in_param"] == df_ind["value1"])
    assert np.all(result["second_param"] == df_ind["value2"])


def test_get_requests_from_params_raises(df_ind):
    test_params = {"wrong_param": "value1"}
    with pytest.raises(ValueError, match="is not a valid parameter"):
        _get_requests_from_params(test_params, simple_test_function_no_kwargs, df_ind)

    test_params = {"in_param": "wrong_name"}
    with pytest.raises(NameError, match="is not available in input data"):
        _get_requests_from_params(test_params, simple_test_function_no_kwargs, df_ind)


def test_get_preprocessed_args():
    test_arguments = {"var1": "filename", "var2": "__preprocessed__"}

    test_preprocessed = {"var2": 99}

    result = _get_preprocessed_args(test_arguments, test_preprocessed)

    assert result["var1"] == "filename"
    assert result["var2"] == 99


def test_prepare_preprocessed_vars_basic(df_ind):
    preproc_dict = {
        "test": {
            "func": "do_hard_limit_check",
            "names": {"value": "value1"},
            "arguments": {"limits": [2, 3]},
        },
    }
    result = _prepare_preprocessed_vars(preproc_dict, df_ind)

    expected = pd.Series([failed, passed, passed, failed])
    pd.testing.assert_series_equal(result["test"], expected)


def test_prepare_preprocessed_vars_error(df_ind):
    preproc_dict = {
        "test": {
            "names": {"value": "value1"},
            "arguments": {"limits": [2, 3]},
        },
    }
    with pytest.raises(ValueError, match="'func' is not specified"):
        _prepare_preprocessed_vars(preproc_dict, df_ind)


def test_prepare_qc_functions_basic(df_ind, qc_dict):
    preprocessed = {}
    result = _prepare_qc_functions(qc_dict, preprocessed, df_ind)

    for i in ["1", "2"]:
        function = result[f"test{i}"]["function"]
        assert callable(function)
        assert function.__name__ == "do_hard_limit_check"
        requests = result[f"test{i}"]["requests"]

        pd.testing.assert_series_equal(requests["value"], df_ind[f"value{i}"])
        assert result[f"test{i}"]["kwargs"] == qc_dict[f"test{i}"]["arguments"]


def test_prepare_qc_functions_error(df_ind):
    preprocessed = {}
    qc_dict = {
        "test": {
            "names": {"value": "value1"},
            "arguments": {"limits": [2, 3]},
        }
    }
    with pytest.raises(ValueError, match="'func' is not specified"):
        _prepare_qc_functions(qc_dict, preprocessed, df_ind)


def test_apply_qc_to_masked_rows():
    result = _apply_qc_to_masked_rows(
        do_hard_limit_check, {"value": pd.Series([1, 2, 3, 4])}, {"limits": [2, 3]}, [0, 1, 2, 3], [True, False, True, True]
    )
    expected = pd.Series([failed, untested, passed, failed])
    pd.testing.assert_series_equal(result, expected)


def test_normalize_groupby(df_ind):
    result1 = _normalize_groupby(df_ind, None)

    assert isinstance(result1, list)
    assert len(result1) == 1
    assert isinstance(result1[0], tuple)

    assert result1[0][0] is None
    pd.testing.assert_frame_equal(result1[0][1], df_ind)

    result2 = _normalize_groupby(df_ind, "value2")

    assert isinstance(result2, list)
    assert len(result2) == 2
    assert isinstance(result2[0], tuple)

    assert result2[0][0] == 1
    expected = pd.DataFrame({"value1": [1, 2], "value2": [1, 1]})
    pd.testing.assert_frame_equal(result2[0][1], expected)

    assert isinstance(result2[1], tuple)

    assert result2[1][0] == 2
    expected = pd.DataFrame({"value1": [3, 4], "value2": [2, 2]}, index=[2, 3])
    pd.testing.assert_frame_equal(result2[1][1], expected)

    groupby = df_ind.groupby("value2")
    result3 = _normalize_groupby(df_ind, groupby)

    assert isinstance(result3, list)
    assert len(result3) == 2
    assert isinstance(result3[0], tuple)

    assert result3[0][0] == 1
    expected = pd.DataFrame({"value1": [1, 2], "value2": [1, 1]})
    pd.testing.assert_frame_equal(result3[0][1], expected)

    assert isinstance(result3[1], tuple)

    assert result3[1][0] == 2
    expected = pd.DataFrame({"value1": [3, 4], "value2": [2, 2]}, index=[2, 3])
    pd.testing.assert_frame_equal(result3[1][1], expected)


@pytest.mark.parametrize(
    "data, return_method, is_series",
    [
        (pd.DataFrame({"value": [1, 2, 3, 4]}), "all", False),
        (pd.DataFrame({"value": [1, 2, 3, 4]}), "passed", False),
        (pd.DataFrame({"value": [1, 2, 3, 4]}), "failed", False),
        (pd.Series([1, 2, 3, 4], name="value"), "all", True),
        (pd.Series([1, 2, 3, 4], name="value"), "passed", True),
        (pd.Series([1, 2, 3, 4], name="value"), "failed", True),
    ],
)
def test_validate_and_normalize_input(data, return_method, is_series):
    result = _validate_and_normalize_input(data, return_method)

    assert result[1] is is_series
    pd.testing.assert_frame_equal(result[0], pd.DataFrame({"value": [1, 2, 3, 4]}))


def test_validate_and_normalize_input_raise():
    with pytest.raises(ValueError, match="'return_method' must be 'all','passed','failed'."):
        _validate_and_normalize_input(pd.DataFrame(), "invalid")


def test_prepare_all_inputs(df_ind, qc_dict):
    preproc_dict = {}
    result = _prepare_all_inputs(df_ind, qc_dict, preproc_dict)

    assert isinstance(result, tuple)
    assert isinstance(result[0], dict)
    assert isinstance(result[1], pd.Series)
    assert isinstance(result[2], pd.DataFrame)

    for i in ["1", "2"]:
        function = result[0][f"test{i}"]["function"]
        assert callable(function)
        assert function.__name__ == qc_dict[f"test{i}"]["func"]

        requests = result[0][f"test{i}"]["requests"]
        assert isinstance(requests, dict)
        assert "value" in requests
        assert isinstance(requests["value"], pd.Series)
        pd.testing.assert_series_equal(requests["value"], df_ind[f"value{i}"])

        kwargs = result[0][f"test{i}"]["kwargs"]
        assert isinstance(kwargs, dict)
        assert kwargs == qc_dict[f"test{i}"]["arguments"]

    pd.testing.assert_series_equal(result[1], pd.Series([True, True, True, True]))

    pd.testing.assert_frame_equal(
        result[2], pd.DataFrame({"test1": [untested, untested, untested, untested], "test2": [untested, untested, untested, untested]})
    )


def test_group_iterator(df_ind):
    result1 = _group_iterator(df_ind, None)

    r1 = next(result1)
    assert isinstance(r1, tuple)
    assert len(r1) == 2

    assert r1[0] is None
    pd.testing.assert_frame_equal(r1[1], df_ind)

    result2 = _group_iterator(df_ind, "value2")

    r2 = next(result2)
    assert isinstance(r2, tuple)
    assert len(r2) == 2

    assert r2[0] == 1
    expected = pd.DataFrame({"value1": [1, 2], "value2": [1, 1]})
    pd.testing.assert_frame_equal(r2[1], expected)

    r2 = next(result2)
    assert isinstance(r2, tuple)
    assert len(r2) == 2

    assert r2[0] == 2
    expected = pd.DataFrame({"value1": [3, 4], "value2": [2, 2]}, index=[2, 3])
    pd.testing.assert_frame_equal(r2[1], expected)

    groupby = df_ind.groupby("value2")
    result3 = _group_iterator(df_ind, groupby)

    r3 = next(result3)

    assert isinstance(r3, tuple)
    assert len(r3) == 2

    assert r3[0] == 1
    expected = pd.DataFrame({"value1": [1, 2], "value2": [1, 1]})
    pd.testing.assert_frame_equal(r3[1], expected)

    r3 = next(result3)
    assert isinstance(r3, tuple)
    assert len(r3) == 2

    assert r3[0] == 2
    expected = pd.DataFrame({"value1": [3, 4], "value2": [2, 2]}, index=[2, 3])
    pd.testing.assert_frame_equal(r3[1], expected)


@pytest.mark.parametrize(
    "return_method, exp",
    [
        ("all", {"test1": [failed, passed, passed, failed], "test2": [failed, failed, passed, passed]}),
        ("passed", {"test1": [failed, passed, passed, failed], "test2": [failed, untested, untested, passed]}),
        ("failed", {"test1": [failed, passed, passed, failed], "test2": [untested, failed, passed, untested]}),
    ],
)
def test_run_qc_engine(df_ind, return_method, exp):
    qc_inputs = {
        "test1": {
            "function": do_hard_limit_check,
            "requests": {"value": df_ind["value1"]},
            "kwargs": {"limits": [2, 3]},
        },
        "test2": {
            "function": do_hard_limit_check,
            "requests": {"value": df_ind["value2"]},
            "kwargs": {"limits": [2, 3]},
        },
    }
    groups = _group_iterator(df_ind, None)
    result = _run_qc_engine(df_ind, qc_inputs, groups, return_method)

    pd.testing.assert_frame_equal(result, pd.DataFrame(exp))

    groups = _group_iterator(df_ind, "value2")
    result = _run_qc_engine(df_ind, qc_inputs, groups, return_method)

    pd.testing.assert_frame_equal(result, pd.DataFrame(exp))


@pytest.mark.parametrize(
    "return_method, exp",
    [
        ("all", {"test1": [failed, passed, passed, failed], "test2": [failed, failed, passed, passed]}),
        ("passed", {"test1": [failed, passed, passed, failed], "test2": [failed, untested, untested, passed]}),
        ("failed", {"test1": [failed, passed, passed, failed], "test2": [untested, failed, passed, untested]}),
    ],
)
def test_do_multiple_check_basic(df_ind, qc_dict, return_method, exp):
    result = _do_multiple_check(df_ind, qc_dict=qc_dict, return_method=return_method)

    pd.testing.assert_frame_equal(result, pd.DataFrame(exp))


@pytest.mark.parametrize("param", ["qc_dict", "preproc_dict"])
def test_do_multiple_check_invalid_dicts(df_ind, param):
    kwargs = "invalid_input"
    with pytest.raises(TypeError, match="must be a dictionary"):
        _do_multiple_check(df_ind, **{param: kwargs})


@pytest.mark.parametrize("param", ["qc_dict", "preproc_dict"])
def test_do_multiple_check_invalid_keys(df_ind, param):
    kwargs = {1: {"func": "no_valid_check"}}
    with pytest.raises(TypeError, match="must be a string"):
        _do_multiple_check(df_ind, **{param: kwargs})


@pytest.mark.parametrize("param", ["qc_dict", "preproc_dict"])
def test_do_multiple_check_invalid_values(df_ind, param):
    kwargs = {"qc_dict": "no_valid_check"}
    with pytest.raises(TypeError, match="must be a dictionary"):
        _do_multiple_check(df_ind, **{param: kwargs})


@pytest.mark.parametrize("param", ["qc_dict", "preproc_dict"])
def test_do_multiple_check_invalid_func(df_ind, param):
    kwargs = {"test": {"func": "no_valid_check"}}
    with pytest.raises(NameError, match="is not defined"):
        _do_multiple_check(df_ind, **{param: kwargs})


@pytest.mark.parametrize("param", ["qc_dict", "preproc_dict"])
def test_do_multiple_check_invalid_names(df_ind, param):
    kwargs = {
        "test": {
            "func": "do_hard_limit_check",
            "names": {"value": "invalid_value"},
        }
    }
    with pytest.raises(NameError, match="is not available in input data"):
        _do_multiple_check(df_ind, **{param: kwargs})


@pytest.mark.parametrize("param", ["qc_dict", "preproc_dict"])
def test_do_multiple_check_invalid_params(df_ind, param):
    kwargs = {
        "test": {
            "func": "do_hard_limit_check",
            "names": {"invalid_param": "value2"},
        },
    }
    with pytest.raises(ValueError, match="is not a valid parameter of function"):
        _do_multiple_check(df_ind, **{param: kwargs})


@pytest.mark.parametrize("param", ["qc_dict", "preproc_dict"])
def test_do_multiple_check_invalid_args(df_ind, param):
    kwargs = {
        "test": {
            "func": "do_hard_limit_check",
            "names": {"value": "value2"},
            "arguments": {"invalid_args": [2, 3]},
        },
    }
    with pytest.raises(ValueError, match="is not a valid parameter of function"):
        _do_multiple_check(df_ind, **{param: kwargs})


@pytest.mark.parametrize("param", ["qc_dict", "preproc_dict"])
def test_do_multiple_check_missing_func(df_ind, param):
    kwargs = {"test": {}}
    with pytest.raises(ValueError, match="'func' is not specified"):
        _do_multiple_check(df_ind, **{param: kwargs})


@pytest.mark.parametrize("param", ["qc_dict", "preproc_dict"])
def test_do_multiple_check_missing_names(df_ind, param):
    kwargs = {
        "test": {
            "func": "do_hard_limit_check",
        },
    }
    with pytest.raises(TypeError, match="is missing for function"):
        _do_multiple_check(df_ind, **{param: kwargs})


@pytest.mark.parametrize("param", ["qc_dict", "preproc_dict"])
def test_do_multiple_check_missing_args(df_ind, param):
    kwargs = {
        "test": {
            "func": "do_hard_limit_check",
            "names": {"value": "value2"},
        },
    }
    with pytest.raises(TypeError, match="is missing for function"):
        _do_multiple_check(df_ind, **{param: kwargs})


@pytest.mark.parametrize(
    "return_method, exp",
    [
        ("all", {"test1": [failed, passed, passed, failed], "test2": [failed, failed, passed, passed]}),
        ("passed", {"test1": [failed, passed, passed, failed], "test2": [failed, untested, untested, passed]}),
        ("failed", {"test1": [failed, passed, passed, failed], "test2": [untested, failed, passed, untested]}),
    ],
)
def test_do_multiple_individual_check(df_ind, qc_dict, return_method, exp):
    result = do_multiple_individual_check(
        data=df_ind,
        qc_dict=qc_dict,
        return_method=return_method,
    )

    pd.testing.assert_frame_equal(result, pd.DataFrame(exp))


def test_multiple_individual_check_raises_return_method():
    with pytest.raises(ValueError, match="'return_method' must be 'all','passed','failed'."):
        do_multiple_individual_check(
            data=pd.Series(),
            qc_dict=None,
            return_method="false",
        )


def test_multiple_individual_check_raises_func():
    with pytest.raises(NameError, match="is not defined"):
        do_multiple_individual_check(
            data=pd.Series(),
            qc_dict={"test_QC": {"func": "do_test_qc"}},
        )


def test_multiple_individual_check_raises_not_in_data():
    with pytest.raises(NameError, match="is not available in input data"):
        do_multiple_individual_check(
            data=pd.Series(),
            qc_dict={
                "MISSVAL": {
                    "func": "do_missing_value_check",
                    "names": {"value": "observation_value"},
                }
            },
        )


def test_multiple_individual_check_raises_not_in_func():
    with pytest.raises(ValueError, match="is not a valid parameter of function"):
        do_multiple_individual_check(
            data=pd.Series(),
            qc_dict={
                "MISSVAL": {
                    "func": "do_missing_value_check",
                    "names": {"value2": "observation_value"},
                }
            },
        )


@pytest.mark.parametrize(
    "return_method, exp",
    [
        (
            "all",
            {
                "test1": [passed, failed, passed, passed, passed, passed, passed, failed, passed, passed],
                "test2": [passed, failed, passed, passed, passed, passed, passed, failed, passed, passed],
            },
        ),
        (
            "passed",
            {
                "test1": [passed, failed, passed, passed, passed, passed, passed, failed, passed, passed],
                "test2": [untested, failed, untested, untested, untested, untested, untested, failed, untested, untested],
            },
        ),
        (
            "failed",
            {
                "test1": [passed, failed, passed, passed, passed, passed, passed, failed, passed, passed],
                "test2": [passed, untested, passed, passed, passed, passed, passed, untested, passed, passed],
            },
        ),
    ],
)
def test_multiple_sequential_check(df_seq, qc_dict_seq, return_method, exp):
    result = do_multiple_sequential_check(
        data=df_seq,
        groupby="name",
        qc_dict=qc_dict_seq,
        return_method=return_method,
    )
    pd.testing.assert_frame_equal(result, pd.DataFrame(exp))
