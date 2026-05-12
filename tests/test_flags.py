from __future__ import annotations

from marine_qc.helpers.auxiliary import (
    best,
    duplicate,
    failed,
    passed,
    unique,
    untestable,
    untested,
    worst,
)


def test_passed_flag():
    assert passed == 0


def test_failed_flag():
    assert failed == 1


def test_untestable_flag():
    assert untestable == 2


def test_untested_flag():
    assert untested == 3


def test_unique_flag():
    assert unique == 0


def test_best_flag():
    assert best == 1


def test_any_flag():
    assert duplicate == 2


def test_worst_flag():
    assert worst == 3
