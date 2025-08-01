"""Marine Quality Control package."""

from __future__ import annotations

from .multiple_row_checks import do_multiple_row_check  # noqa
from .qc_grouped_reports import do_mds_buddy_check, do_bayesian_buddy_check  # noqa
from .qc_individual_reports import (
    do_position_check,
    do_date_check,
    do_time_check,
    do_day_check,
    do_missing_value_check,
    do_missing_value_clim_check,
    do_hard_limit_check,
    do_climatology_check,
    do_supersaturation_check,
    do_sst_freeze_check,
    do_wind_consistency_check,
)  # noqa
from .qc_sequential_reports import (
    do_spike_check,
    do_track_check,
    do_few_check,
    find_saturated_runs,
    find_multiple_rounded_values,
    find_repeated_values,
    do_iquam_track_check,
)  # noqa

__author__ = """Ludwig Lierhammer"""
__email__ = "ludwiglierhammer@dwd.de"
__version__ = "0.0.1"
