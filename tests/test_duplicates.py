from __future__ import annotations
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Literal, get_type_hints

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from marine_qc import (
    do_hard_limit_check,
)
from marine_qc.duplicate_checker.duplicates import remove_duplicates, flag_duplicates

def test_remove_duplicates():
    station_id = ["A", "B", "A"]
    lat = [50, 60, 50]
    lon = [10, 20, 10]
    date = ["2010-07-12 12:00:00", "2010-07-12 12:00:00", "2010-07-12 12:00:00"]
    vsi = [25, 25, 25]
    dsi = [45, 45, 45]
    
    result = remove_duplicates(
        station_id=station_id,
        lat=lat,
        lon=lon,
        date=date,
        vsi=vsi,
        dsi=dsi,
        test=1,
    )
    
    assert isinstance(result, tuple)
    assert len(result) == 7
    assert result[0] == ["A", "B"]
    assert result[1] == [50, 60]
    assert result[2] == [10, 20]
    assert result[3] == ["2010-07-12 12:00:00", "2010-07-12 12:00:00"]
    assert result[4] == [25, 25]
    assert result[5] == [45, 45]
    assert result[6] == [1, 1]


def test_flag_duplicates():
    station_id = ["A", "B", "A"]
    lat = [50, 60, 50]
    lon = [10, 20, 10]
    date = ["2010-07-12 12:00:00", "2010-07-12 12:00:00", "2010-07-12 12:00:00"]
    vsi = [25, 25, 25]
    dsi = [45, 45, 45]
    result = flag_duplicates(
        station_id=station_id,
        lat=lat,
        lon=lon,
        date=date,
        vsi=vsi,
        dsi=dsi,
        test=1,
    )
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == [1, 0, 3]
    assert result[1] == [np.int64(2), pd.NA, np.int64(0)]
