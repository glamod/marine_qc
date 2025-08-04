.. currentmodule:: marine_qc

Buoy tracking classes
---------------------

.. autosummary::
   :toctree: generated/
   
   buoy_tracking_qc.SpeedChecker
   buoy_tracking_qc.SpeedChecker.do_speed_check
   buoy_tracking_qc.SpeedChecker.get_qc_outcomes
   buoy_tracking_qc.SpeedChecker.valid_arrays
   buoy_tracking_qc.SpeedChecker.valid_parameters   
   buoy_tracking_qc.NewSpeedChecker
   buoy_tracking_qc.NewSpeedChecker.do_new_speed_check
   buoy_tracking_qc.NewSpeedChecker.get_qc_outcomes
   buoy_tracking_qc.NewSpeedChecker.perform_iquam_track_check
   buoy_tracking_qc.NewSpeedChecker.valid_arrays
   buoy_tracking_qc.NewSpeedChecker.valid_parameters
   buoy_tracking_qc.AgroundChecker
   buoy_tracking_qc.AgroundChecker.do_aground_check
   buoy_tracking_qc.AgroundChecker.get_qc_outcomes
   buoy_tracking_qc.AgroundChecker.smooth_arrays
   buoy_tracking_qc.AgroundChecker.valid_arrays
   buoy_tracking_qc.AgroundChecker.valid_parameters
   buoy_tracking_qc.SSTTailChecker
   buoy_tracking_qc.SSTTailChecker.do_sst_tail_check
   buoy_tracking_qc.SSTTailChecker.get_qc_outcomes
   buoy_tracking_qc.SSTTailChecker.valid_parameters
   buoy_tracking_qc.SSTBiasedNoisyChecker
   buoy_tracking_qc.SSTBiasedNoisyChecker.do_sst_biased_noisy_check
   buoy_tracking_qc.SSTBiasedNoisyChecker.get_qc_outcomes_bias
   buoy_tracking_qc.SSTBiasedNoisyChecker.get_qc_outcomes_noise
   buoy_tracking_qc.SSTBiasedNoisyChecker.get_qc_outcomes_short
   buoy_tracking_qc.SSTBiasedNoisyChecker.set_all_qc_outcomes_to
   buoy_tracking_qc.SSTBiasedNoisyChecker.valid_parameters
      

Buoy tracking functions
-----------------------

.. autosummary::
   :toctree: generated/

   buoy_tracking_qc.track_day_test
   buoy_tracking_qc.is_monotonic
