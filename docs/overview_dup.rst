.. marine QC documentation master file

.. _overview_dup:

----------------------------------------------------------------
Overview of QC functions for potentially duplicated observations
----------------------------------------------------------------

This page gives a brief overview of each of the duplicate checker functions currently implemented.
For more detailed documentation please see the API. Titles of individual sections below link to the relevant pages in the API.
The titles of the individual sections below link directly to the corresponding API pages.

The tests operate on collections of marine reports to detect potentially duplicated observations.
The duplicate detection routines are based on the `Python SPLINK Toolkit <https://moj-analytical-services.github.io/splink/index.html>`_.

By default, the duplicate check comprises six kinds of observational values:

   * "station_id": Any kind of idnetifier of the observing station (e.g. ship name, WIGOS Station Identifier).
   * "lat": Latitude of the observing station in degrees.
   * "lon": Longitude of the observing station in degrees.
   * "date": Time stamp of the obersing station (datetime object or datetime-formatted string).
   * "vsi": Speed of the observing station in kilometers per hour.
   * "dsi": Course of the observing station in degrees.

The default settings to detect potentially duplicated observations are listed below. These settings can be adjusted individually
when using the duplicate detection functions below.

By default, duplicate observations are detected according to the following criteria:

   * "station_id": values must match exactly for observations to be considered duplicates.
   * "lat": values may differ by up to 0.11 degrees for observations to be considered duplicates.
   * "lon": values may differ by up to 0.11 degrees for observations to be considered duplicates.
   * "date": values may differ by up to 1 minute for observations to be considered duplicates.
   * "vsi": values may differ by up to 0.09 kilometers per hour for observations to be considered duplicates.
   * "dsi": values may differ by up to 0.9 degrees for observations to be considered duplicates.

Two observations are not considered duplicates if any of these conditions is not fullfied.

For more details how to customize the duplicate checker settings see :func:`.duplicate_check`.

:func:`.duplicate_check`
========================

Detects possibly duplicated observations.

Checks whether observations are potential duplicates. It returns a :class:`.DupDetect` instance containing
both the original data and the duplicate comparison results.

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
