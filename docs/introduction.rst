Introduction
------------

This Python package provides a set of tools for quality control of marine meteorological reports. Marine
meteorological reports typically comprise latitude, longitude, time, and date as well as one or more
marine meteorological variables often including, but not limited to sea-surface temperature, air temperature,
dew point temperature, sea level pressure, wind speed and wind direction.

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

QC of Individual Reports
------------------------

The tests in `qc_individual_reports.py` work on individual marine reports, either singly or in arrays (it doesn't
change the outcome). These include simple checks of whether the location, time and date of the observation are
valid as well as more complex checks involving comparison to climatologies.

do_position_check
=================

Checks whether the latitude is in the range -90 to 90 degrees and that the longitude is in the range -180 to 360
degrees.

do_date_check
=============

Checks whether the date specified either as a Datetime object or by year, month, and day, is a valid date. If any
component of the input is numerically invalid (Nan, None or similar) then the flag is set to 2, i.e. untestable

do_time_check
=============

Checks that the time of the report is valid. If the input Datetime or hour is not numerically valid (Nan, None, or the
like) then the flag is set to 2, i.e. untestable.

do_day_check
============

Checks whether an observation was made during the day (flag set to 1, fail) or night (flag set to 0, pass). The
definition of day is between a specified amount of time after sunrise and the same amount of time after sunset. If
any of the inputs are numerically invalid, the flag is set to 2, untestable.

do_missing_value_check
======================

Checks whether a value is None or numerically invalid. If the report is numerically invalid the flag is set to 1, fail,
otherwise it is set to 0, pass.

do_missing_value_clim_check
===========================

Checks whether a value in a report was made at a location with a valid climatological average. If the climatological
value is valid, the flag is set to 0, pass otherwise it is set to 1, fail.

do_hard_limit_check
===================

Checks whether a value is between specified limits or not. If the value is between the specified upper and lower limits
or equal to either one then the flag is set to 0, pass, otherwise the flag is set to 1, fail.

do_climatology_check
====================

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

do_supersaturation_check
========================

Check whether the dewpoint temperature is greater than the air temperature. If the dew point is greater than the
air temperature then the conditions are supersaturated and the flag is set to 1, fail. If the dewpoint is less than
or equal to the air temperature then the flag is set to 0, pass. If either of the inputs is numerically invalide then
the flag is set to 2, untestable.

do_sst_freeze_check
===================

Check whether the sea-surface temperature is above a specified freezing point (generally sea water freezes at -1.8C).
There are optional inputs, which allow you to specify an observational uncertainty and a multiplier. If these are not
supplied then the uncertainty is set to zero. If the sea-surface temperature is more than the multiplier times the
uncertainty below the freezing point then the flag is set to 1, fail, otherwise it is set to 0, pass. If any of the
inputs is numerically invalid (Nan, None or something of that kind) then the flag is set to 2, untestable.

do_wind_consistency_check
=========================

Compares the wind speed and wind direction to check for consistency. If the windspeed is zero, the direction should
be set to zero also. If the wind speed is greater than zero then the wind directions should not equal zero. If either
of these constraints is violated then the flag is set to 1, fail, otherwise it is set to 0. If either of the inputs
is numerically valid then the flag is set to 2, untestable.