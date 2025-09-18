.. marine QC documentation master file

---------------------------------------------------
Overview of QC functions for individual reports
---------------------------------------------------

This page gives a brief overview of each of the QC functions currently implemented. For more detailed documentation
please see the API. Titles of individual sections below link to the relevant pages in the API.

The tests in `qc_individual_reports.py` work on individual marine reports, either singly or in arrays (it doesn't
change the outcome). These include simple checks of whether the location, time and date of the observation are
valid as well as more complex checks involving comparison to climatologies.

:func:`.do_position_check`
==========================

Tests whether the latitiude-longitude position of the report is valid.

Checks whether the latitude is in the range -90 to 90 degrees and that the longitude is in the range -180 to 360
degrees.

:func:`.do_date_check`
======================

Tests whether the date of the report is valid.

Checks whether the date specified either as a Datetime object or by year, month, and day, is a valid date. If any
component of the input is numerically invalid (Nan, None or similar) then the flag is set to 2, i.e. untestable

:func:`.do_time_check`
======================

Tests whether the time of the report is valid.

Checks that the time of the report is valid. If the input Datetime or hour is not numerically valid (Nan, None, or the
like) then the flag is set to 2, i.e. untestable.

:func:`.do_day_check`
=====================

Tests whether the report was during the day.

Checks whether an observation was made during the day (flag set to 1, fail) or night (flag set to 0, pass). The
definition of day is between a specified amount of time after sunrise and the same amount of time after sunset. If
any of the inputs are numerically invalid, the flag is set to 2, untestable.

:func:`.do_missing_value_check`
===============================

Tests that the value is present.

Checks whether a value is None or numerically invalid. If the report is numerically invalid the flag is set to 1, fail,
otherwise it is set to 0, pass.

:func:`.do_missing_value_clim_check`
====================================

Tests whether there is a valid climatological value at the report location.

Checks whether a value in a report was made at a location with a valid climatological average. If the climatological
value is valid, the flag is set to 0, pass otherwise it is set to 1, fail.

:func:`.do_hard_limit_check`
============================

Tests whether the value of report is within a specified range

Checks whether a value is between specified limits or not. If the value is between the specified upper and lower limits
or equal to either one then the flag is set to 0, pass, otherwise the flag is set to 1, fail.

:func:`.do_climatology_check`
=============================

Tests whether the value of a report is within an acceptable range around the climatological mean.

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

Tests whether a report represents supersaturated conditions.

Check whether the dewpoint temperature is greater than the air temperature. If the dew point is greater than the
air temperature then the conditions are supersaturated and the flag is set to 1, fail. If the dewpoint is less than
or equal to the air temperature then the flag is set to 0, pass. If either of the inputs is numerically invalid then
the flag is set to 2, untestable.

:func:`.do_sst_freeze_check`
============================

Tests whether the sea-surface temperature is above freezing.

Check whether the sea-surface temperature is above a specified freezing point (generally sea water freezes at -1.8C).
There are optional inputs, which allow you to specify an observational uncertainty and a multiplier. If these are not
supplied then the uncertainty is set to zero. If the sea-surface temperature is more than the multiplier times the
uncertainty below the freezing point then the flag is set to 1, fail, otherwise it is set to 0, pass. If any of the
inputs is numerically invalid (Nan, None or something of that kind) then the flag is set to 2, untestable.

:func:`.do_wind_consistency_check`
==================================

Tests that wind speed and direction are consistent.

Compares the wind speed and wind direction to check for consistency. If the windspeed is zero, the direction should
be set to zero also. If the wind speed is greater than zero then the wind directions should not equal zero. If either
of these constraints is violated then the flag is set to 1, fail, otherwise it is set to 0. If either of the inputs
is numerically valid then the flag is set to 2, untestable.

Running Multiple Individual Report Checks
-----------------------------------------

Multiple individual report checks can be run simultaneously using the :func:`.do_multiple_row_check` function. Aside from the
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
