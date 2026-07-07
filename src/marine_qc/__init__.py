"""Marine Quality Control package."""

from __future__ import annotations

from .duplicate_checker import duplicate_check, flag_duplicates, get_duplicates, remove_duplicates
from .helpers import Climatology, Flags
from .helpers.auxiliary import (
    PandasNAType,
    PandasNaTType,
    ScalarDatetimeType,
    ScalarFloatType,
    ScalarIntType,
    ScalarNumberType,
    ScalarStrType,
    SequenceDatetimeType,
    SequenceFloatType,
    SequenceIntType,
    SequenceNumberType,
    SequenceStrType,
    ValueDatetimeType,
    ValueFloatType,
    ValueIntType,
    ValueNumberType,
    ValueStrType,
)
from .helpers.external_clim import (
    ClimArgType,
    ClimFloatType,
    ClimInputType,
    ClimIntType,
    ClimNumberType,
)
from .quality_control import (
    combine_qc_results,
    do_aground_check,
    do_bayesian_buddy_check,
    do_climatology_check,
    do_date_check,
    do_day_check,
    do_few_check,
    do_hard_limit_check,
    do_iquam_track_check,
    do_landlocked_check,
    do_maritime_check,
    do_mds_buddy_check,
    do_missing_value_check,
    do_missing_value_clim_check,
    do_multiple_grouped_check,
    do_multiple_individual_check,
    do_multiple_sequential_check,
    do_new_aground_check,
    do_new_speed_check,
    do_night_check,
    do_position_check,
    do_speed_check,
    do_spike_check,
    do_sst_biased_check,
    do_sst_biased_noisy_short_check,
    do_sst_end_tail_check,
    do_sst_freeze_check,
    do_sst_noisy_check,
    do_sst_start_tail_check,
    do_supersaturation_check,
    do_time_check,
    do_track_check,
    do_valid_value_check,
    do_valid_value_clim_check,
    do_wind_consistency_check,
    find_multiple_rounded_values,
    find_repeated_values,
    find_saturated_runs,
)
from .visualization import (
    plot_latitude_longitude,
    plot_latitude_variable,
    plot_variable_longitude,
)


__author__ = """Ludwig Lierhammer"""
__email__ = "ludwig.lierhammer@dwd.de"
__version__ = "0.3.3-dev.7"
