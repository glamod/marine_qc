from __future__ import annotations

from marine_qc import Flags


def test_passed_flag():
    assert Flags.passed == 0


def test_failed_flag():
    assert Flags.failed == 1


def test_untestable_flag():
    assert Flags.untestable == 2


def test_untested_flag():
    assert Flags.untested == 3


def test_unique_flag():
    assert Flags.unique == 0


def test_best_flag():
    assert Flags.best == 1


def test_any_flag():
    assert Flags.duplicate == 2


def test_worst_flag():
    assert Flags.worst == 3
