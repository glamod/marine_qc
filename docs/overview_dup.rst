.. marine QC documentation master file

.. _overview_dup:

--------------------------------------------------------------
Overview of QC functions for possibly duplicated observations
--------------------------------------------------------------

This page gives a brief overview of each of the QC functions currently implemented. For more detailed documentation
please see the API. Titles of individual sections below link to the relevant pages in the API.

The tests work on the whole bunch of marine reports to detect possibly duplicated observations.
The test are based on the `Python Record Linkage Toolkit <https://recordlinkage.readthedocs.io/en/latest/>`_.

By default the duplicate check comprises seven kinds of observational values:

   * "station_id": Any kind of idnetifier of the observing station (e.g. ship name, WIGOS Station Identifier).
   * "lat": Latitude of the observing station in degrees.
   * "lon": Longitude of the observing station in degrees.
   * "date": Time stamp of the obersing station (datetime object or datetime-formatted string).
   * "vsi": Speed of the observing station in kilometers per hour.
   * "dsi": Course of the observing station in degrees.
   * "obs": Any number of observational values from the observing station (e.g. temperature, pressure, wind speed).

The default settings to detect possibly duplicated observations are listed below but can we adjusted individually
when using the QC functions. These settings are used to set a
`recordlinkage.Compare instance <https://recordlinkage.readthedocs.io/en/latest/ref-compare.html#recordlinkage.Compare>`_:

* default settings to set `recordlinkage.Compare instance <https://recordlinkage.readthedocs.io/en/latest/ref-compare.html#recordlinkage.Compare>`_ instance:

   * "station_id": values must match exactly for observations to be considered duplicates.
   * "lat": values may differ by up to 0.11 degrees for observations to be considered duplicates.
   * "lon": values may differ by up to 0.11 degrees for observations to be considered duplicates.
   * "date": values may differ by up to 1 minute for observations to be considered duplicates.
   * "vsi": values may differ by up to 0.09 kilometers per hour for observations to be considered duplicates.
   * "dsi": values may differ by up to 0.9 degrees for observations to be considered duplicates.
   * "obs": values may differ by up to 0.9 whatever units for observations to be considered duplicates.

* default indexing settings to detect possibly duplicated observations:

   * Candidate pairs are generated using the "date" column as the sorting key.
   * A window size of `5` is used.
   * Additional standard blocking is applied on the "station_id" column.

For more details how to manipulate the duplicate checker settings see :func:`.duplicate_check`.

Executing these settings produces pairwise comparison scores used to identify potential duplicates.

   * "keep": Determines which duplicate observation should be retained.
   * "limit": Minimum comparison score required for observations to be considered duplicates.
   * "equal_musts": Column names whose values must match exactly for observations to be considered duplicates.

For more details how to manipulate these settings see :func:`.flag_duplicates` or :func:`.remove_duplicates`.

:func:`.duplicate_check`
===============================

Detects possibly duplicated observations.

Checks whether observations are potential duplicates. It returns a :class:`.DupDetect` instance that contains
both the processed data and the duplicate comparison scores.

:func:`.flag_duplicates`
====================================

Flags potentially duplicated observations and returns both duplicate flags and the corresponding detected duplicate
matches for each observation, either based on a precomputed detection result or by running duplicate detection internally.
For more information about the flagging schema see :ref:`flags`.

:func:`.remove_duplicates`
===============================

Removes potentially duplicated observations and returns the input data with duplicates excluded, either based
on a precomputed detection result or by running duplicate detection internally.
