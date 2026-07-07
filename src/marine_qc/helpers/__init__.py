"""Marine Quality Control helpers package."""

from __future__ import annotations

from .auxiliary import best, duplicate, failed, passed, unique, untestable, untested, worst
from .external_clim import Climatology, get_climatological_value, open_xrdataset


class Flags:
    """Constants used to annotate quality control (QC) and duplicate-check results."""

    passed = passed
    failed = failed
    untestable = untestable
    untested = untested
    unique = unique
    best = best
    duplicate = duplicate
    worst = worst


__all__ = [
    "Climatology",
    "Flags",
    "get_climatological_value",
    "open_xrdataset",
]
