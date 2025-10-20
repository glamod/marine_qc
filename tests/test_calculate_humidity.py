from __future__ import annotations

import numpy as np

import marine_qc.calculate_humidity as ch


def test_vap_from_example():
    """Test from Kate's example in the file."""
    assert ch.vap(10.0, 15.0, 1013) == 12.3
    assert ch.vap(-15.0, -10.0, 1013) == 1.7
    assert np.isnan(ch.vap(None, 15.0, 1013))


def test_vap_from_sh():
    """Test from Kate's example in the file."""
    assert ch.vap_from_sh(7.6, 1013.0) == 12.3


def test_sh():
    """Test from Kate's example in the file."""
    assert ch.sh(10.0, 15.0, 1013.0) == 7.6
    assert np.isnan(ch.sh(None, 15.0, 1013.0))
    assert ch.sh(-15.0, -10.0, 1013.0) == 1.0


def test_sh_from_vap():
    """Test from Kate's example in the file."""
    assert ch.sh_from_vap(12.3, 1013.0, 1013.0) == 7.6


def test_rh():
    """Test from Kate's example in the file."""
    assert ch.rh(10.0, 15.0, 1013.0) == 72.0
    assert ch.rh(-15.0, -10.0, 1013.0) == 63.6
    assert np.isnan(ch.rh(None, 15.0, 1013.0))


def test_wb():
    """Test from Kate's example in the file."""
    assert ch.wb(10.0, 15.0, 1013) == 12.2
    assert ch.wb(-15.0, -10.0, 1013) == -10.9
    assert np.isnan(ch.wb(None, 15.0, 1013))


def test_dpd():
    """Test from Kate's example in the file."""
    assert ch.dpd(10.0, 15.0) == 5.0
    assert np.isnan(ch.dpd(None, 15.0))


def test_td_from_vap():
    """Tests from Kate's examples in the file."""
    assert ch.td_from_vap(12.3, 1013.0, 15.0) == 10.0
    assert ch.td_from_vap(12.3, 1013.0, -15.0) == 8.7
