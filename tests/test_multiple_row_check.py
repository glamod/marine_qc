from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marine_qc import do_multiple_row_check
from marine_qc.multiple_row_checks import (
    _get_function,
    _get_preprocessed_args,
    _get_requests_from_params,
    _is_func_param,
    _is_in_data,
)


def simple_test_function(in_param, **kwargs):
    return in_param * 2


def simple_test_function_no_kwargs(in_param):
    return in_param * 2


def test_get_function():
    result = _get_function("_get_function")
    assert callable(result)
    assert result.__name__ == "_get_function"


def test_get_function_raises():
    with pytest.raises(NameError):
        _get_function("BAD_NAME")


def test_is_func_param():
    assert not _is_func_param(_is_func_param, "Non existent parameter")
    assert _is_func_param(_is_func_param, "param")

    # A function with kwargs always returns True
    assert _is_func_param(simple_test_function, "non existent parameter")


def test_is_in_data():
    data_series = pd.Series({"test_name": [1, 3, 5, 7]})
    assert _is_in_data("test_name", data_series)
    assert not _is_in_data("wrong_test_name", data_series)

    data_series = pd.DataFrame(
        {"test_name": [1, 3, 5, 7], "different_name": [2, 4, 6, 8]}
    )
    assert _is_in_data("test_name", data_series)
    assert _is_in_data("different_name", data_series)
    assert not _is_in_data("wrong_test_name", data_series)


def test_is_in_data_raises():
    with pytest.raises(TypeError):
        _is_in_data("test_name", [1, 2, 3])


def test_get_requests_from_params():
    test_params = {"in_param": "test_name"}
    data_series = pd.DataFrame(
        {"test_name": [1, 3, 5, 7], "different_name": [2, 4, 6, 8]}
    )
    result = _get_requests_from_params(test_params, simple_test_function, data_series)
    assert "in_param" in result
    assert np.all(result["in_param"] == data_series["test_name"])

    test_params = {"in_param": "test_name", "second_param": "different_name"}
    data_series = pd.DataFrame(
        {"test_name": [1, 3, 5, 7], "different_name": [2, 4, 6, 8]}
    )
    result = _get_requests_from_params(test_params, simple_test_function, data_series)
    assert "in_param" in result
    assert "second_param" in result
    assert np.all(result["in_param"] == data_series["test_name"])
    assert np.all(result["second_param"] == data_series["different_name"])


def test_get_requests_from_params_raises():
    test_params = {"wrong_param": "test_name"}
    data_series = pd.DataFrame(
        {"test_name": [1, 3, 5, 7], "different_name": [2, 4, 6, 8]}
    )
    with pytest.raises(ValueError):
        _get_requests_from_params(
            test_params, simple_test_function_no_kwargs, data_series
        )

    test_params = {"in_param": "wrong_name"}
    data_series = pd.DataFrame(
        {"test_name": [1, 3, 5, 7], "different_name": [2, 4, 6, 8]}
    )
    with pytest.raises(NameError):
        _get_requests_from_params(
            test_params, simple_test_function_no_kwargs, data_series
        )


def test_get_preprocessed_args():

    test_arguments = {"var1": "filename", "var2": "__preprocessed__"}

    test_preprocessed = {"var2": 99}

    result = _get_preprocessed_args(test_arguments, test_preprocessed)

    assert result["var1"] == "filename"
    assert result["var2"] == 99


def test_multiple_row_check_raises_return_method():
    with pytest.raises(ValueError):
        do_multiple_row_check(
            data=pd.Series(),
            qc_dict=None,
            return_method="false",
        )


def test_multiple_row_check_raises_func():
    with pytest.raises(NameError):
        do_multiple_row_check(
            data=pd.Series(),
            qc_dict={"test_QC": {"func": "do_test_qc"}},
        )


def test_multiple_row_check_raises_3():
    with pytest.raises(NameError):
        do_multiple_row_check(
            data=pd.Series(),
            qc_dict={
                "MISSVAL": {
                    "func": "do_missing_value_check",
                    "names": {"value": "observation_value"},
                }
            },
        )


def test_multiple_row_check_raises_4():
    with pytest.raises(ValueError):
        do_multiple_row_check(
            data=pd.Series(),
            qc_dict={
                "MISSVAL": {
                    "func": "do_missing_value_check",
                    "names": {"value2": "observation_value"},
                }
            },
        )
