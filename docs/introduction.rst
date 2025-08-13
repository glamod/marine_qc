.. marine QC documentation master file

-----------
Basic Guide
-----------

.. include:: ./description.rst

The `MarineQC` package comprises quality control tests of three kinds:

1. tests that are performed on individual reports and only use information from a single report, such that they can
   be performed individually.
2. tests that are performed on sequences of reports from a single ship or platform.
3. tests that are performed on a group of reports for a specified area and period, potentially comprising reports
   from many platforms and platform types.

The following sections describe the quality control (QC) tests in more detail. The tests are based on
those developed in:

Kennedy, J. J., Rayner, N. A., Atkinson, C. P., & Killick, R. E. (2019). An Ensemble Data Set of Sea
Surface Temperature Change from 1850: The Met Office Hadley Centre HadSST.4.0.0.0 Data Set. Journal
of Geophysical Research: Atmospheres, 124. https://doi.org/10.1029/2018JD029867

Willett, K. M., Dunn, R. J. H., Kennedy, J. J., and Berry, D. I.: Development of the HadISDH.marine
humidity climate monitoring dataset, Earth Syst. Sci. Data, 12, 2853-2880,
https://doi.org/10.5194/essd-12-2853-2020, 2020.

Xu, F., and A. Ignatov, 2014: In situ SST Quality Monitor (iQuam). J. Atmos. Oceanic Technol., 31,
164-180, https://doi.org/10.1175/JTECH-D-13-00121.1.

Atkinson, C. P., N. A. Rayner, J. Roberts-Jones, and R. O. Smith (2013), Assessing the quality of sea
surface temperature observations from drifting buoys and ships on a platform-by-platform basis, J.
Geophys. Res. Oceans, 118, 3507-3529,  https://doi.org/10.1002/jgrc.20257

QC Flags
--------

The QC checks output QC flags that indicate the status of each observation. There are four numbered
flags:

* 0 Passed - the observation has passed this particular quality control check
* 1 Failed - the observation has failed this particular quality control check
* 2 Untestable - the observation cannot be tested using this quality control check, usually because one or
  more pieces of information are missing. For example, a climatology check with a missing climatology value.
* 3 Untested - the observation has not been tested for this quality control check

Running the QC Checks
---------------------

The QC checks can be run simply. Each one takes one or more input values, which can be a float, list, 1-d numpy array
or Pandas DataSeries, along with zero or more parameters.

So, for example, one can run a hard limit check like so::


  input_values = np.array([-15.0, 0.0, 20.0, 55.0])
  result = do_hard_limit_check(input_values, [-10., 40.])

Additionally, some checks use climatological averages which can be provided like the other
inputs, or passed as a Climatology object. For example, the climatology check can be run like so::

  input_ssts = np.array([15.0, 17.3, 21.3, 32.0])
  climatological_averages = np.array([14.0, 15.8, 19.1, 20.3])
  result = do_climatology_check(input_ssts, climatological_averages, 8.0)

Alternatively, the climatological values can be specified using a Climatology and providing the datetime and location
of the reports as keyword arguments::

  input_ssts = np.array([15.0, 17.3, 21.3, 32.0])
  latitudes = np.array([33.0, 28.0, 22.0, 15.0])
  longitudes = np.array([-30.3, -29.9, -31.8, -31.7])
  dates = np.array(['2003-01-01T02:00:00.00', '2003-01-01T08:00:00.00', '2003-01-01T14:00:00.00', '2003-01-01T20:00:00.00'])
  climatological_averages = Climatology.open_netcdf_file('climatology_file.nc')
  result = do_climatology_check(input_ssts, climatological_averages, 8.0, lat=latitude, lon=longitudes, date=dates)

This will automatically extract the climatological values at the specified times and locations.

Unit Conversions
----------------

The QC checks written using SI (and derived) units. Inputs can be converted when a QC function is called using the
`units` keyword argument::

  temperature_in_K(25.0, units={"value": "degC"})

As an example.


QC of Individual Reports
------------------------

The tests in `qc_individual_reports.py` work on individual marine reports, either singly or in arrays (it doesn't
change the outcome). These include simple checks of whether the location, time and date of the observation are
valid as well as more complex checks involving comparison to climatologies.

:func:`.do_position_check`
==========================

Checks whether the latitude is in the range -90 to 90 degrees and that the longitude is in the range -180 to 360
degrees.

:func:`.do_date_check`
======================

Checks whether the date specified either as a Datetime object or by year, month, and day, is a valid date. If any
component of the input is numerically invalid (Nan, None or similar) then the flag is set to 2, i.e. untestable

:func:`.do_time_check`
======================

Checks that the time of the report is valid. If the input Datetime or hour is not numerically valid (Nan, None, or the
like) then the flag is set to 2, i.e. untestable.

:func:`.do_day_check`
=====================

Checks whether an observation was made during the day (flag set to 1, fail) or night (flag set to 0, pass). The
definition of day is between a specified amount of time after sunrise and the same amount of time after sunset. If
any of the inputs are numerically invalid, the flag is set to 2, untestable.

:func:`.do_missing_value_check`
===============================

Checks whether a value is None or numerically invalid. If the report is numerically invalid the flag is set to 1, fail,
otherwise it is set to 0, pass.

:func:`.do_missing_value_clim_check`
====================================

Checks whether a value in a report was made at a location with a valid climatological average. If the climatological
value is valid, the flag is set to 0, pass otherwise it is set to 1, fail.

:func:`.do_hard_limit_check`
============================

Checks whether a value is between specified limits or not. If the value is between the specified upper and lower limits
or equal to either one then the flag is set to 0, pass, otherwise the flag is set to 1, fail.

:func:`.do_climatology_check`
=============================

Checks whether a value from a report is close (in some sense) to the climatological average at that location. "Close"
can be defined using four parameters:

1. Maximum anomaly. If this is set then the flag is set to 1, fail if the absolute difference between the value and
   the climatological average at that point is greater than the maximum anomaly, otherwise it is set to 0, pass.
2. If standard_deviation is set then the value is converted to a standardised anomaly. the flag is set to 1, fail if
   the absolute standardised anomaly is greater than the maximum anomaly, otherwise it is set to 0, pass.
3. If standard_deviation_limits is set then the input standard deviation is constrained to lie between the upper and
   lower limits thus specified before the calculation of the standardised anomalies.
4. If lowbar is set then the absolute anomaly must be greater than the lowbar to fail regardless of the standard
   deviation.

These allow for a great deal of flexibility in the check depending what information is available.

:func:`.do_supersaturation_check`
=================================

Check whether the dewpoint temperature is greater than the air temperature. If the dew point is greater than the
air temperature then the conditions are supersaturated and the flag is set to 1, fail. If the dewpoint is less than
or equal to the air temperature then the flag is set to 0, pass. If either of the inputs is numerically invalid then
the flag is set to 2, untestable.

:func:`.do_sst_freeze_check`
============================

Check whether the sea-surface temperature is above a specified freezing point (generally sea water freezes at -1.8C).
There are optional inputs, which allow you to specify an observational uncertainty and a multiplier. If these are not
supplied then the uncertainty is set to zero. If the sea-surface temperature is more than the multiplier times the
uncertainty below the freezing point then the flag is set to 1, fail, otherwise it is set to 0, pass. If any of the
inputs is numerically invalid (Nan, None or something of that kind) then the flag is set to 2, untestable.

:func:`.do_wind_consistency_check`
==================================

Compares the wind speed and wind direction to check for consistency. If the windspeed is zero, the direction should
be set to zero also. If the wind speed is greater than zero then the wind directions should not equal zero. If either
of these constraints is violated then the flag is set to 1, fail, otherwise it is set to 0. If either of the inputs
is numerically valid then the flag is set to 2, untestable.

Running Multiple Individual Report Checks
-----------------------------------------

Multiple indvidual report checks can be run simultaneously using the :func:`.do_multiple_row_check` function. Aside from the
input dataframe, two additional arguments can be specified: `qc_dict` and `preproc_dict`. The `qc_dict` is a
dictionary that specifies the names of the qc function to be run, the variables used as input and the values of the
arguments. The `preproc_dict` is a dictionary that specifies any pre-processing functions such as a function to
extract the climatological values corresponding to the input reports.

Currently, the following QC checks can be used:

* :func:`.do_climatology_check`,
* :func:`.do_date_check`,
* :func:`.do_day_check`,
* :func:`.do_hard_limit_check`,
* :func:`.do_missing_value_check`,
* :func:`.do_missing_value_clim_check`,
* :func:`.do_night_check`,
* :func:`.do_position_check`,
* :func:`.do_sst_freeze_check`,
* :func:`.do_supersaturation_check`,
* :func:`.do_time_check`,
* :func:`.do_wind_consistency_check`

And the following preprocessing functions:

* :func:`.get_climatological_value`

The function is called like so:

.. code-block:: python

    result = do_multiple_row_check(data, qc_dict, preproc_dict)

An example `qc_dict` for a hard limit test:

.. code-block:: python

    qc_dict = {
        "hard_limit_check": {
            "func": "do_hard_limit_check",
            "names": "ATEMP",
            "arguments": {"limits": [193.15, 338.15]},
        }
    }

An example `qc_dict` for a climatology test. Variable "climatology" was previously defined:

.. code-block:: python

    qc_dict = {
        "climatology_check": {
            "func": "do_climatology_check",
            "names": {
                "value": "observation_value",
                "lat": "latitude",
                "lon": "longitude",
                "date": "date_time",
            },
            "arguments": {
                "climatology": climatology,
                "maximum_anomaly": 10.0,  # K
            },
        },
    }

An example `preproc_dict` for extracting a climatological value:

.. code-block:: python

    preproc_dict = {
        "func": "get_climatological_value",
        "names": {
            "lat": "latitude",
            "lon": "longitude",
            "date": "date_time",
        },
        "inputs": climatology,
    }

Make use of both dictionaries:

.. code-block:: python

    preproc_dict = {
        "func": "get_climatological_value",
        "names": {
            "lat": "latitude",
            "lon": "longitude",
            "date": "date_time",
        },
        "inputs": climatology,
    }

    qc_dict = {
        "climatology_check": {
            "func": "do_climatology_check",
            "names": {
                "value": "observation_value",
            },
            "arguments": {
                "climatology": "__preprocessed__",
                "maximum_anomaly": 10.0,  # K
            },
        },
    }


QC of Sequential Reports
------------------------

Some test work on sequences of reports from a single ship, drifter or other platform. They include tests that
compare values at different times and locations to assess data quality.

:func:`.do_track_check`
=======================

The track check uses the location and datetime information from the reports as well as the ship speed and direction
information, if available, to determine if any of the reported locations and times are likely to be erroneous.

For a detailed description see :doc:`track_check`

:func:`.do_few_check`
=====================

If there are three or fewer reports then the flags for all reports are set to 1, fail. If there are four or more,
the flags are all set to 0, pass.

:func:`.do_iquam_track_check`
=============================

The IQUAM track check is based on the track check implemented by NOAA's IQUAM system. It verifies that consecutive
locations of a platform are consistent with the times of the report, assuming that the platform can't move faster
than a certain speed. To avoid problems with the rounding of locations and times, a minimum separation is specified
in time and space. The report with the most speed violations is flagged and excluded and the process is repeated
till no more violations are detected.

Details are in the `IQUAM paper`_.

.. _IQUAM paper: https://doi.org/10.1175/JTECH-D-13-00121.1

:func:`.do_spike_check`
=======================

The spike checks looks for large changes in input value between reports. It is based on the spike check implemented
by NOAA's IQUAM system. It uses the locations and datetimes of the reports to calculate space and time gradients
which are then compared to maximum allowed gradients. For the report being tested, gradients are calculated for a
specified number of observations before and after the target observation. The number of calculated gradients that
exceed the specified maximums are used to decide which reports pass (flag set to 0) or fail (flag set to 1) the
spike check.

Details are in the `IQUAM paper`_.

.. _IQUAM paper: https://doi.org/10.1175/JTECH-D-13-00121.1

:func:`.find_saturated_runs`
============================

A sequence of reports is checked for runs where conditions are saturated i.e. the reported air temperature and dewpoint
temperature are the same. This can happen when the reservoir of water for the wetbulb thermometer dries out, or loses
contact with the thermometer bulb. If a run of saturated reports is longer than a specified number of reports and
cover a period longer than a specified threshold then the run of saturated values is flagged as 1 (fail) otherwise the
reports are flagged as 0, pass.

:func:`.find_multiple_rounded_values`
=====================================

A sequence of reports is checked for values which are given to a whole number. If more than a specified fraction of
observations are given to a whole number and the total number of whole numbers exceeds a specified threshold then
all the flags for all the rounded numbers are set to 1, fail. The flags for all other reports are set to 0, pass.

:func:`.find_repeated_values`
=============================

A sequence of reports is checked for values which are repeated many times. If more than a specified fraction of
reports have the same value and the total number of reports of that value exceeds a specified threshold then
all the flags for all reports with that value are set to 1, fail. The flags for all other reports are set to 0, pass.

QC of Grouped Reports
---------------------

The final type of tests are those performed on a group of reports, potentially comprising reports from many platforms
and platform types. The reports can cover large areas and multiple months. The tests currently include so-called
"buddy" checks in which the values for each report are compared to those of their neighbours.

:func:`.do_mds_buddy_check`
===========================

The buddy check compares the observed value from each report expressed as an anomaly to the average of that variable
from other nearby reports (the buddies in the buddy check, also converted to anomalies). Depending how many neighbours
there are and how close they are, an adaptive multiplier is used. The difference between the observed value for the
report and the "buddy" mean must be less than the multiplier times the standard deviation of the variable at that
location taken from a climatology. If the difference is less the flag for that report is set to 0, pass otherwise it
is set to 1, failed.

For a detailed description see :doc:`buddy_check`

:func:`.do_bayesian_buddy_check`
================================

The bayesian buddy check works in a similar way to `do_mds_buddy_check`. The principle is the same -  a report is
compared to the average of nearby reports - but the determination of whether it is too far away is based on an
explicit estimate of the probability of gross error.

For a detailed description see :doc:`bayesian_buddy_check`


Tracking QC
-----------

There are additional routines that are intended for the QC of measurements of sea surface temperature
from drifting buoys specifically. These checks are based on Atkinson et al. 2013. The checks are designed to be
run on whole drifting buoy records from when a drifter is first deployed to when it ceases to report.

Atkinson, C. P., N. A. Rayner, J. Roberts-Jones, and R. O. Smith (2013), Assessing the quality of sea
surface temperature observations from drifting buoys and ships on a platform-by-platform basis, J.
Geophys. Res. Oceans, 118, 3507–3529, https://doi.org/10.1002/jgrc.20257

:func:`.do_speed_check`
=======================

The speed check aims to flag reports from drifting buoys that have been picked up by a ship (and are
therefore likely to be out of the water). Reports are flagged if the mean velocity over a specified period
is above a threshold (2.5 m/s) and the reports cover a period of time longer than a specified minimum.

:func:`.do_new_speed_check`
===========================

The new speed check behaves similarly to the speed check, but observations are prescreened using the
IQUAM track check. Speed is assessed over the shortest available period that exceeds the specified
minimum window period. To avoid problems with the discretization of time and location variables (for
example, latitude and longitude are often given to the nearest tenth of a degree), which can lead to large
apparent speeds, a minimum increment can be specified.

:func:`.do_aground_check`
=========================

The aground check aims to flag reports from drifting buoys that have fetched up on land. A drifter is
deemed aground when, after a minimum specified period of time, the distance between reports is less than
a specified 'tolerance'. Sometimes a drifting buoy will return to the sea, so a maximum period is also
specified to avoid missing short lived groundings.

:func:`.do_new_aground_check`
=============================

The new aground check is the same as the aground check but there is no upper window limit.

:func:`.do_sst_start_tail_check` and :func:`.do_sst_end_tail_check`
===================================================================

The tail checks (see also the end tail check) aim to flag reports at the start (or end) of a record that are
biased or noisy based on comparisons with a spatially complete background or reference SST field. There are two steps
identifying longer and shorter-lived tails of low quality reports at the ends of the record. Biased and noisy
reports are detected using a moving window.

The long-tail check is first and uses as 120 report window (the lengths of windows and multipliers are user defined,
here we give the original default values). The mean and standard deviation of the difference
between the reported sea-surface temperature and the background value are calculated. If the mean difference is
more than 3 times the means background standard deviation, the reports in the window are flagged
as biased. If the standard deviation of the differences is more than 3 times the root mean square
of the background standard deviations. The window is moved along the sequence of reports until a set of reports
passes the test.

The short-tail check uses a 30 report window (again, these parameters are user-defined) and if one or more of
report-background differences exceeds 9 times the the standard deviation of the background standard deviation for
that report, the whole window fails the QC. The window is moved along the sequence of reports until a set of reports
passes the test.

The combination of the longer, more sensitive test and the shorter, less sensitive test helps to detect a wider range
of tail behaviours.

The end tail check works in the same way as the start tail check, but runs through the reports in reverse
time order.

:func:`.do_sst_biased_check`, :func:`.do_sst_noisy_check`, and :func:`.`do_sst_biased_noisy_short_check`
========================================================================================================

This group of checks flags reports from drifters that are persistently biased or noisy. The biased and noisy checks
are only applied to drifting buoys which made more than 30 reports.

For the bias check, if the mean bias relative to the background is larger than the bias limit then the reports are
flagged 1, failed. Otherwise they pass

For the noise check, if the standard deviation of the report-background differences is larger than the mean
background standard deviation added in quadrature to the specified uncertainty in the drifting buoy SST reports.

For the short record check (fewer than 30 reports), the whole record is flagged as failed (1) if more than a
specified number of reports have a report-background difference larger than 3 times the combined standard deviation.
The combined standard deviation is the square root of the sum of squared contributions from the background
uncertainty, inter-drifter uncertainty and intra-drifter uncertainty. Otherwise the reports are flagged as passes (0).
