from __future__ import annotations

from marine_qc.auxiliary import (
    failed,
    passed,
    untestable,
    untested,
)


def test_passed_flag():
    assert passed == 0


def test_failed_flag():
    assert failed == 1


def test_untestable_flag():
    assert untestable == 2


def test_untested_flag():
    assert untested == 3
