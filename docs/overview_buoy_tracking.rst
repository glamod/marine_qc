.. marine QC documentation master file

---------------------------------------------------
Overview of QC functions for additional buoy QC
---------------------------------------------------

This page gives a brief overview of each of the buoy-specific QC functions currently implemented. These routines
were specifically designed for the QC of sea-surface temperatures from drifting buoys and are intended to be run
on whole drifting buoy records from when a drifter is first deployed to when it ceases to report.

Some of the functions require additional information. Here "matched" means that the values at the location of the
report have been extracted, usually from a gridded field.

* matched background SSTs from a spatially complete analysis. OSTIA was used in the original paper.
* matched sea ice concentrations from a spatially complete analysis. OSTIA was used in the original paper.
* matched background sea surface temperature field variances.

For more detailed documentation please see the API and the paper Atkinson et al. (2013). Titles of individual
sections below link to the relevant pages in the API.

Atkinson, C. P., N. A. Rayner, J. Roberts-Jones, and R. O. Smith (2013), Assessing the quality of sea
surface temperature observations from drifting buoys and ships on a platform-by-platform basis, J.
Geophys. Res. Oceans, 118, 3507-3529,  https://doi.org/10.1002/jgrc.20257


:func:`.do_speed_check`
=======================

Tests that speeds inferred from a sequence of reports are not implausible for a drifting buoy.

The speed check aims to flag reports from drifting buoys that have been picked up by a ship (and are
therefore likely to be out of the water). Reports are flagged if the mean velocity over a specified period
is above a threshold (2.5 m/s) and the reports cover a period of time longer than a specified minimum.

:func:`.do_new_speed_check`
===========================

Tests that speeds inferred from a sequence of reports are not implausible for a drifting buoy

The new speed check behaves similarly to the speed check, but observations are prescreened using the
IQUAM track check. Speed is assessed over the shortest available period that exceeds the specified
minimum window period. To avoid problems with the discretization of time and location variables (for
example, latitude and longitude are often given to the nearest tenth of a degree), which can lead to large
apparent speeds, a minimum increment can be specified.

:func:`.do_aground_check`
=========================

Tests whether reports from a drifting buoy suggest it has run aground and stopped moving.

The aground check aims to flag reports from drifting buoys that have fetched up on land. A drifter is
deemed aground when, after a minimum specified period of time, the distance between reports is less than
a specified 'tolerance'. Sometimes a drifting buoy will return to the sea, so a maximum period is also
specified to avoid missing short lived groundings.

:func:`.do_new_aground_check`
=============================

Tests whether reports from a drifting buoy suggest it has run aground and stopped moving.

The new aground check is the same as the aground check but there is no upper window limit.

:func:`.do_sst_start_tail_check` and :func:`.do_sst_end_tail_check`
===================================================================

Tests for strange behaviour (bias or noise) at the start or end of a sequence of reports from a drifting buoy.

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

:func:`.do_sst_biased_check`, :func:`.do_sst_noisy_check`, and :func:`.do_sst_biased_noisy_short_check`
========================================================================================================

Tests for sequences of reports from a drifting buoy that are biased or noisy with a version that works on shorter
records.

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
