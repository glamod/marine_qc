.. currentmodule:: marine_qc

.. _api:

"""""""""""""
API reference
"""""""""""""

This page provides an auto-generated summary of the ``marine_qc`` API.

QC checks on individual reports
===============================

.. autosummary::
   :toctree: generated/

    do_position_check
    do_date_check
    do_time_check
    do_day_check
    do_missing_value_check
    do_missing_value_clim_check
    do_hard_limit_check
    do_climatology_check
    do_supersaturation_check
    do_sst_freeze_check
    do_wind_consistency_check
    do_multiple_row_check

QC checks on sequential reports
===============================

.. autosummary::
   :toctree: generated/

    do_few_check
    do_spike_check
    do_track_check
    do_iquam_track_check
    find_saturated_runs
    find_multiple_rounded_values
    find_repeated_values


QC checks on grouped reports
============================

.. autosummary::
   :toctree: generated/

   do_mds_buddy_check
   do_bayesian_buddy_check
