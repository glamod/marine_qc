Introduction
------------

This Python package provides a set of tools for quality control of marine meteorological reports. Marine
meteorological reports typically comprise latitude, longitude, time, and date as well as one or more
marine meteorological variables often including, but not limited to sea-surface temperature, air temperature,
dew point temperature, sea level pressure, wind speed and wind direction.

The `MarineQC` package comprises quality control tests of three kinds:

1. that are performed on individual reports,
2. tests that are performed on sequences of reports from a single ship or platform
3. tests that are performed on all reports for a specified period, potentially comprising reports
   from many platforms

The following sections describe the quality control (QC) tests in more detail. The tests are based on
those developed in:

Kennedy, J. J., Rayner, N. A., Atkinson, C. P., & Killick, R. E. (2019). An Ensemble Data Set of Sea
Surface Temperature Change from 1850: The Met Office Hadley Centre HadSST.4.0.0.0 Data Set. Journal
of Geophysical Research: Atmospheres, 124. https://doi.org/10.1029/2018JD029867

Willett, K. M., Dunn, R. J. H., Kennedy, J. J., and Berry, D. I.: Development of the HadISDH.marine
humidity climate monitoring dataset, Earth Syst. Sci. Data, 12, 2853–2880,
https://doi.org/10.5194/essd-12-2853-2020, 2020.

Xu, F., and A. Ignatov, 2014: In situ SST Quality Monitor (iQuam). J. Atmos. Oceanic Technol., 31,
164–180, https://doi.org/10.1175/JTECH-D-13-00121.1.

Atkinson, C. P., N. A. Rayner, J. Roberts-Jones, and R. O. Smith (2013), Assessing the quality of sea
surface temperature observations from drifting buoys and ships on a platform-by-platform basis, J.
Geophys. Res. Oceans, 118, 3507–3529,  https://doi.org/10.1002/jgrc.20257

QC Flags
--------

The QC checks output QC flags that indicate the status of each observation. There are four numbered
flags:

* 0 Passed - the observation has passed this particular quality control check
* 1 Failed - the observation has failed this particular quality control check
* 2 Untested - the observation has not been tested for this quality control check
* 3 Untestable - the observation cannot be tested using this quality control check, usually because one or
  more pieces of information are missing. For example, a climatology check with a missing climatology value.

dd