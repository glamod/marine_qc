.. marine QC documentation master file

.. _overview_dup:

--------------------------------------------------------------
Overview of QC functions for possibly duplicated observations
--------------------------------------------------------------

This page gives a brief overview of each of the QC functions currently implemented. For more detailed documentation
please see the API. Titles of individual sections below link to the relevant pages in the API.

The tests work on the whole bunch of marine reports to detect possibly duplicated observations.
The test are based on the `Python SPLINK Toolkit <https://moj-analytical-services.github.io/splink/index.html>`_.

By default the duplicate check comprises seven kinds of observational values:

   * "station_id": Any kind of idnetifier of the observing station (e.g. ship name, WIGOS Station Identifier).
   * "lat": Latitude of the observing station in degrees.
   * "lon": Longitude of the observing station in degrees.
   * "date": Time stamp of the obersing station (datetime object or datetime-formatted string).
   * "vsi": Speed of the observing station in kilometers per hour.
   * "dsi": Course of the observing station in degrees.

The default settings to detect possibly duplicated observations are listed below but can be adjusted individually
when using the duplicate detection functions below. These settings are used to detect duplicates by default:

   * "station_id": values must match exactly for observations to be considered duplicates.
   * "lat": values may differ by up to 0.11 degrees for observations to be considered duplicates.
   * "lon": values may differ by up to 0.11 degrees for observations to be considered duplicates.
   * "date": values may differ by up to 1 minute for observations to be considered duplicates.
   * "vsi": values may differ by up to 0.09 kilometers per hour for observations to be considered duplicates.
   * "dsi": values may differ by up to 0.9 degrees for observations to be considered duplicates.

Two observations are not considered to be duplicates if only one of these conditions is not fullfied.

For more details how to manipulate the duplicate checker settings see :func:`.duplicate_check`.

:func:`.duplicate_check`
========================

Detects possibly duplicated observations.

Checks whether observations are potential duplicates. It returns a :class:`.DupDetect` instance that contains
both the original data and the duplicate comparison scores.

:func:`.get_duplicates`
=======================

Get potentially duplicated observations and returns the corresponding detected duplicate matches for each observation,
either based on a precomputed detection result or by running duplicate detection internally.

:func:`.flag_duplicates`
========================

Flags potentially duplicated observations and returns the duplicate flags for each observation, either based
on a precomputed detection result or by running duplicate detection internally.
For more information about the flagging schema see :ref:`flags`.

:func:`.remove_duplicates`
==========================

Removes potentially duplicated observations and returns the input data with duplicates excluded, either based
on a precomputed detection result or by running duplicate detection internally.
