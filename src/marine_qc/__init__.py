"""Marine Quality Control package."""

from __future__ import annotations

from .buoy_tracking_qc import (
    do_aground_check,
    do_new_aground_check,
    do_new_speed_check,
    do_speed_check,
    do_sst_biased_check,
    do_sst_biased_noisy_short_check,
    do_sst_end_tail_check,
    do_sst_noisy_check,
    do_sst_start_tail_check,
)
from .multiple_row_checks import do_multiple_row_check
from .qc_grouped_reports import do_bayesian_buddy_check, do_mds_buddy_check
from .qc_individual_reports import (
    do_climatology_check,
    do_date_check,
    do_day_check,
    do_hard_limit_check,
    do_missing_value_check,
    do_missing_value_clim_check,
    do_night_check,
    do_position_check,
    do_sst_freeze_check,
    do_supersaturation_check,
    do_time_check,
    do_wind_consistency_check,
)
from .qc_sequential_reports import (
    do_few_check,
    do_iquam_track_check,
    do_spike_check,
    do_track_check,
    find_multiple_rounded_values,
    find_repeated_values,
    find_saturated_runs,
)


__author__ = """Ludwig Lierhammer"""
__email__ = "ludwig.lierhammer@dwd.de"
__version__ = "0.1.0"
