.. marine QC documentation master file

.. _overview_grp:

------------------------------------------------
Overview of QC functions for grouped reports
------------------------------------------------

This page gives a brief overview of each of the QC functions currently implemented. For more detailed documentation
please see the API. Titles of individual sections below link to the relevant pages in the API.

The final type of tests are those performed on a group of reports, potentially comprising reports from many platforms
and platform types. The reports can cover large areas and multiple months. The tests currently include so-called
"buddy" checks in which the values for each report are compared to those of their neighbours.

:func:`.do_mds_buddy_check`
===========================

Tests that each observation is reasonably close to the average of its near neighbours in time and space.

The buddy check compares the observed value from each report expressed as an anomaly to the average of that variable
from other nearby reports (the buddies in the buddy check, also converted to anomalies). Depending how many neighbours
there are and how close they are, an adaptive multiplier is used. The difference between the observed value for the
report and the "buddy" mean must be less than the multiplier times the standard deviation of the variable at that
location taken from a climatology. If the difference is less the flag for that report is set to 0, pass otherwise it
is set to 1, failed.

For a detailed description see :ref:`buddy_check`

:func:`.do_bayesian_buddy_check`
================================

Tests that each observation is reasonably close to the average of its near neighbours in time and space.

The bayesian buddy check works in a similar way to `do_mds_buddy_check`. The principle is the same -  a report is
compared to the average of nearby reports - but the determination of whether it is too far away is based on an
explicit estimate of the probability of gross error.

For a detailed description see :ref:`bayesian_buddy_check`

:func:`.do_multiple_grouped_check`
==================================

Multiple sequential report checks can be run simultaneously using the :func:`.do_multiple_grouped_check` function.
For more information how to use the function see :ref:`running_multiple_ind`.

Currently, the following QC checks can be used:

    * :func:`.do_bayesian_buddy_check`
    * :func:`.do_mds_buddy_check`
