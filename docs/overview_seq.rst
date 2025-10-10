.. marine QC documentation master file

---------------------------------------------------
Overview of QC functions for sequential reports
---------------------------------------------------

This page gives a brief overview of each of the QC functions currently implemented. For more detailed documentation
please see the API. Titles of individual sections below link to the relevant pages in the API.

Some test work on sequences of reports from a single ship, drifter or other platform. They include tests that
compare values at different times and locations to assess data quality.

:func:`.do_track_check`
=======================

Tests that the locations of a series of reports form a plausible ship track.

The track check uses the location and datetime information from the reports as well as the ship speed and direction
information, if available, to determine if any of the reported locations and times are likely to be erroneous.

For a detailed description see :doc:`track_check`

:func:`.do_few_check`
=====================

Tests whether there are more than a few reports associated with a ship. Ships making few reports are typically
unreliable.

If there are three or fewer reports then the flags for all reports are set to 1, fail. If there are four or more,
the flags are all set to 0, pass.

:func:`.do_iquam_track_check`
=============================

Tests that the locations of a series of reports form a plausible ship track.

The IQUAM track check is based on the track check implemented by NOAA's IQUAM system. It verifies that consecutive
locations of a platform are consistent with the times of the report, assuming that the platform can't move faster
than a certain speed. To avoid problems with the rounding of locations and times, a minimum separation is specified
in time and space. The report with the most speed violations is flagged and excluded and the process is repeated
till no more violations are detected.

Details are in the `IQUAM paper`_.

.. _IQUAM paper: https://doi.org/10.1175/JTECH-D-13-00121.1

:func:`.do_spike_check`
=======================

Tests that a sequence of values has unlikely "spikes" in it.

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

Tests whether there are implausibly long runs of reports with super-saturated conditions.

A sequence of reports is checked for runs where conditions are saturated i.e. the reported air temperature and dewpoint
temperature are the same. This can happen when the reservoir of water for the wetbulb thermometer dries out, or loses
contact with the thermometer bulb. If a run of saturated reports is longer than a specified number of reports and
cover a period longer than a specified threshold then the run of saturated values is flagged as 1 (fail) otherwise the
reports are flagged as 0, pass.

:func:`.find_multiple_rounded_values`
=====================================

Tests whether there are large numbers of rounded values in a sequence of reports.

A sequence of reports is checked for values which are given to a whole number. If more than a specified fraction of
observations are given to a whole number and the total number of whole numbers exceeds a specified threshold then
all the flags for all the rounded numbers are set to 1, fail. The flags for all other reports are set to 0, pass.

:func:`.find_repeated_values`
=============================

Tests whether there are implausibly large number of repeated values in a sequence of reports.

A sequence of reports is checked for values which are repeated many times. If more than a specified fraction of
reports have the same value and the total number of reports of that value exceeds a specified threshold then
all the flags for all reports with that value are set to 1, fail. The flags for all other reports are set to 0, pass.
