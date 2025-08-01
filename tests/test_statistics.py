from __future__ import annotations

import copy

import numpy as np
import pytest  # noqa

from marine_qc.statistics import (
    missing_mean,
    p_data_given_good,
    p_data_given_gross,
    p_gross,
    trim_mean,
    trim_std,
    winsorised_mean,
)


@pytest.mark.parametrize(
    "inarr, trim, expected",
    [
        ([0, 0, 0, 0, 0, 0, 0, 0, 0], 10, 0.0),  # test_all_zeroes
        (
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 10.0],
            4,
            0.0,
        ),  # test_all_zeroes_one_outlier_trimmed
        (
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 10.0],
            50,
            1.0,
        ),  # test_all_zeroes_one_outlier_not_trimmed
        ([1.3, 0.7, 4.0], 0, 2.0),  # test_trim_zero
        ([10.0, 4.0, 3.0, 2.0, 1.0], 0, 4.0),
        ([10.0, 4.0, 3.0, 2.0, 1.0], 5, 3.0),
    ],
)
def test_trimmed_mean(inarr, trim, expected):
    original_array = copy.deepcopy(inarr)
    assert trim_mean(inarr, trim) == expected
    # This checks the array is not modified by the function
    assert np.all(inarr == original_array)


@pytest.mark.parametrize(
    "inarr, trimming, expected",
    [
        ([6.0, 1.0, 1.0, 1.0, 1.0], 0, 2.0),
        ([6.0, 1.0, 1.0, 1.0, 1.0], 5, 0.0),
    ],
)
def test_trim_std(inarr, trimming, expected):
    original_array = copy.deepcopy(inarr)
    assert trim_std(inarr, trimming) == expected
    # This checks the array is not modified by the function
    assert np.all(inarr == original_array)


@pytest.mark.parametrize(
    "inarr, expected",
    [
        ([None, None], None),  # test_all_missing
        ([None, 7.3], 7.3),  # test_one_non_missing
        ([994.2, None], 994.2),
        ([6.0, 1.0], 3.5),  # test_all_non_missing
    ],
)
def test_missing_mean(inarr, expected):
    assert missing_mean(inarr) == expected


@pytest.mark.parametrize(
    "inarr, expected",
    [
        ([0, 0, 0, 0, 0, 0, 0, 0, 0], 0.0),  # test_all_zeroes
        ([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0], 4.0),  # test_all_fours
        ([4.0, 4.0, 4.0], 4.0),  # test_three_fours
        ([3.0, 4.0, 5.0], 4.0),  # test_ascending
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 4.0),  # test_longer_ascending_run
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 700.0],
            4.0,
        ),  # test_longer_ascending_run_large_outlier
        (
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                99.0,
                1000.0,
            ],
            8.5,
        ),
        # test_longer_ascending_run_two_outliers
    ],
)
def test_winsorised_mean(inarr, expected):
    assert winsorised_mean(inarr) == expected


@pytest.mark.parametrize(
    "x, q, r_hi, r_lo, mu, sigma",
    [
        (1.0, -0.1, 8.0, -8.0, 0.5, 0.5),
        (1.0, 0.1, 8.0, -8.0, 0.5, -0.5),
        (1.0, 0.1, -8.0, 8.0, 0.5, 0.5),
        (-9.0, 0.1, 8.0, -8.0, 0.5, 0.5),
        (9.0, 0.1, 8.0, -8.0, 0.5, 0.5),
    ],
)
def test_p_data_given_good_raises(x, q, r_hi, r_lo, mu, sigma):
    with pytest.raises(ValueError):
        p_data_given_good(x, q, r_hi, r_lo, mu, sigma)


@pytest.mark.parametrize(
    "x, q, r_hi, r_lo, mu, sigma, expected",
    [
        (
            1.0,
            0.1,
            8.0,
            -8.0,
            1.0,
            0.00001,
            1.0,
        ),  # very narrow standard deviation gives unity
    ],
)
def test_p_data_given_good(x, q, r_hi, r_lo, mu, sigma, expected):
    assert p_data_given_good(x, q, r_hi, r_lo, mu, sigma) == expected


@pytest.mark.parametrize(
    "q, r_hi, r_lo",
    [
        (-0.1, 8.0, -8.0),
        (1.0, -8.0, 8.0),
    ],
)
def test_p_data_given_gross_raises(q, r_hi, r_lo):
    with pytest.raises(ValueError):
        p_data_given_gross(q, r_hi, r_lo)


@pytest.mark.parametrize("q, r_hi, r_lo, expected", [(0.1, 1.0, -1.0, 1 / 21.0)])
def test_p_data_given_gross(q, r_hi, r_lo, expected):
    assert p_data_given_gross(q, r_hi, r_lo) == expected


@pytest.mark.parametrize(
    "p0, q, r_hi, r_lo, x, mu, sigma",
    [
        (-0.04, 0.1, 8.0, -8.0, 1.0, 0.5, 0.25),
        (1.04, 0.1, 8.0, -8.0, 1.0, 0.5, 0.25),
        (0.04, -0.1, 8.0, -8.0, 1.0, 0.5, 0.25),
        (0.04, 0.1, -8.0, 8.0, 1.0, 0.5, 0.25),
        (0.04, 0.1, 8.0, -8.0, -9.0, 0.5, 0.25),
        (0.04, 0.1, 8.0, -8.0, 9.0, 0.5, 0.25),
        (0.04, 0.1, 8.0, -8.0, 1.0, 0.5, -0.25),
    ],
)
def test_p_gross_raises(p0, q, r_hi, r_lo, x, mu, sigma):
    with pytest.raises(ValueError):
        p_gross(p0, q, r_hi, r_lo, x, mu, sigma)


@pytest.mark.parametrize(
    "p0, q, r_hi, r_lo, x, mu, sigma, expected",
    [
        (
            0.04,
            0.1,
            8.0,
            -8.0,
            -7.0,
            7.5,
            0.01,
            1.0,
        ),  # very very high probability of gross error
        (
            0.04,
            0.1,
            8.0,
            -8.0,
            0.0,
            0.0,
            0.00001,
            ((1.0 / 161.0) * 0.04) / (((1.0 / 161.0) * 0.04) + (1 * 0.96)),
        ),
        (
            0.01,
            1,
            1.0,
            -1.0,
            0.0,
            0.0,
            0.00001,
            ((1.0 / 3.0) * 0.01) / (((1.0 / 3.0) * 0.01) + (1 * 0.99)),
        ),
    ],
)
def test_p_gross(p0, q, r_hi, r_lo, x, mu, sigma, expected):
    assert pytest.approx(p_gross(p0, q, r_hi, r_lo, x, mu, sigma), 0.00001) == expected
