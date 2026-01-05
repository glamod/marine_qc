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

* 0 Passed - the report has passed this particular quality control check.
* 1 Failed - the report has failed this particular quality control check.
* 2 Untestable - the report cannot be tested using this quality control check, usually because one or
  more pieces of information are missing. For example, a climatology check with a missing climatology value.
* 3 Untested - the report has not been tested for this quality control check.

Running the QC Checks
---------------------

The QC checks can be run simply. Each one takes one or more input values, which can be a float, list, 1-d numpy array
or Pandas DataSeries, along with zero or more parameters depending on the function.

So, for example, one can run a hard limit check like so::


  input_values = np.array([-15.0, 0.0, 20.0, 55.0])
  result = do_hard_limit_check(input_values, [-10., 40.])

Additionally, some checks use climatological averages which can be provided like the other
inputs, or passed as a :class:`.Climatology` object. For example, the climatology check can be run like so::

  input_ssts = np.array([15.0, 17.3, 21.3, 32.0])
  climatological_averages = np.array([14.0, 15.8, 19.1, 20.3])
  result = do_climatology_check(input_ssts, climatological_averages, 8.0)

Alternatively, the climatological values can be specified using a :class:`.Climatology` and providing the datetime and location
of the reports as keyword arguments::

  input_ssts = np.array([15.0, 17.3, 21.3, 32.0])
  latitudes = np.array([33.0, 28.0, 22.0, 15.0])
  longitudes = np.array([-30.3, -29.9, -31.8, -31.7])
  dates = np.array(['2003-01-01T02:00:00.00', '2003-01-01T08:00:00.00', '2003-01-01T14:00:00.00', '2003-01-01T20:00:00.00'])
  climatological_averages = Climatology.open_netcdf_file('climatology_file.nc')
  result = do_climatology_check(input_ssts, climatological_averages, 8.0, lat=latitude, lon=longitudes, date=dates)

This will automatically extract the climatological values at the specified times and locations.

We have four different types of QC checks:

* QC functions for individual reports

  * for an overview see :ref:`overview_ind`
  * for API references see :ref:`api_qc_ind`

* QC functions for sequential reports (e.g. track checks)

  * for an overview see :ref:`overview_seq`
  * for API references see :ref:`api_qc_seq`

* QC functions for grouped reports (e.g. buoy checks)

  * for an overview see :ref:`overview_grp`
  * for API references see :ref:`api_qc_grp`

* QC functions for sequential buoy reports:

  * for an overview see :ref:`overview_buoy`
  * for API references see :ref:`api_qc_buoy`

Unit Conversions
----------------

The QC checks written using SI (and derived) units. Inputs can be converted when a QC function is called using the
`units` keyword argument::

  temperature_in_K(25.0, units={"value": "degC"})

.. _running_multiple_ind
