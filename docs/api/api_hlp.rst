.. currentmodule:: marine_qc

Helpers in QC checks for individual reports module
--------------------------------------------------

.. autosummary::
   :toctree: generated/

   qc_individual_reports._do_daytime_check

Helpers in multiple checks module
---------------------------------

.. autosummary::
   :toctree: generated/

   multiple_checks._get_function
   multiple_checks._is_func_param
   multiple_checks._is_in_data
   multiple_checks._get_requests_from_params
   multiple_checks._get_preprocessed_args
   multiple_checks._prepare_preprocessed_vars
   multiple_checks._prepare_qc_functions
   multiple_checks._apply_qc_to_masked_rows
   multiple_checks._normalize_groupby
   multiple_checks._validate_and_normalize_input
   multiple_checks._prepare_all_inputs
   multiple_checks._group_iterator
   multiple_checks._run_qc_engine
   multiple_checks._do_multiple_check


Helpers in external climatology module
--------------------------------------

.. autosummary::
   :toctree: generated/

   external_clim._format_output
   external_clim._select_point
   external_clim._empty_dataarray

Helpers in spherical geometry module
------------------------------------

.. autosummary::
   :toctree: generated/

   spherical_geometry._geod_inv

Helpers in statistical functions module
---------------------------------------

.. autosummary::
   :toctree: generated/

   statistics._trim_stat

Helpers in plotting module
--------------------------

.. autosummary::
   :toctree: generated/

   plot_qc_outcomes._get_colours_labels
   plot_qc_outcomes._make_plot

Static methods of buoy tracking QC classes
------------------------------------------

.. autosummary::
   :toctree: generated/

   buoy_tracking_qc.SSTTailChecker._parse_rep
   buoy_tracking_qc.SSTTailChecker._preprocess_reps
   buoy_tracking_qc.SSTTailChecker._do_long_tail_check
   buoy_tracking_qc.SSTTailChecker._do_short_tail_check
   buoy_tracking_qc.SSTBiasedNoisyChecker._parse_rep
   buoy_tracking_qc.SSTBiasedNoisyChecker._preprocess_reps
   buoy_tracking_qc.SSTBiasedNoisyChecker._long_record_qc
   buoy_tracking_qc.SSTBiasedNoisyChecker._short_record_qc
