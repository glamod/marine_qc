=========
Changelog
=========

0.1.0 (2025-10-21)
------------------
Contributors to this version: Ludwig Lierhammer (:user:`ludwiglierhammer`), John Kennedy (:user:`jjk-code-otter`) and Trevor James Smith (:user:`Zeitsperre`).

Announcements
^^^^^^^^^^^^^
* This marine QC repository is a copy of the results of https://github.com/glamod/glamod-marine-processing/pull/117. This repository replaces `glamod_marine_proccesing.qc_suite.modules` as an independent package (:pull:`6`).

License and Legal
^^^^^^^^^^^^^^^^^
* Update copyright statements in LICENSE (:pull:`23`, :pull:`43`)
* Update author list for publishing on zendodo and readthedocs (:issue:`20`, :issue:`25`, :pull:`23`, :pull:`43`)

CI changes
^^^^^^^^^^
* Use `cruft` and cookicutter template `https://github.com/Ouranosinc/cookiecutter-pypackage` (:issue:`13`, :pull:`30`, :pull:`55`)

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* ``do_night_check``: reverse ``do_day_check`` (:pull:`21`)
* Added documentation (:issue:`4`, :pull:`11`, :pull:`12`)
* Added ``get_value_fast`` for extracting values from climatologies (:pull:`24`)
* Added ``mds_lat_to_yindex_fast`` for extracting values from climatologies (:pull:`24`)
* Added ``mds_lon_to_xindex_fast`` for extracting values from climatologies (:pull:`24`)
* Implement plotting routines for QC outcomes (:pull:`24`):

  * `marine_qc.plot_qc_outcomes.latitude_variable_plot`: Plot a graph of points showing the latitude and value of a set of observations coloured according to the QC outcomes.
  * `marine_qc.plot_qc_outcomes.latitude_longitude_plot`: Plot a graph of points showing the latitude and longitude of a set of observations coloured according to the QC outcomes.

* decorator `post_formt_return_type` has new parameters (:pull:`24`):

  * dtype: Desired data type of the result. Default is int.
  * multiple: If True, assumes the function returns a sequence of results (e.g., a tuple), and applies `format_return_type` to each element individually.

* Both `do_bayesian_buddy_check` and `do_mds_buddy_check` allow a list of row numbers to be skipped (`ignore_index`) (:pull:`24`).

Internal changes
^^^^^^^^^^^^^^^^
* Remove both jupyter notebook specific (nbqa-pyupgrade, nbqa-black, nbqa-isort, nbstripout) and json-related (pretty-format-json) pre-commit hooks (:pull:`7`)
* Replace assert statements with if statement raising error messages (:pull:`7`)
* Split some try statements into single if statements giving warnings (:pull:`7`)
* Fixing some typos in docstrings and comments (:pull:`7`)
* Improved unit test coverage (:pull:`9`)
* combine `time_control.day_in_year` and `time_control.dayinyear` to `time_control.day_in_year` (:pull:`9`)
* new function `time_control.valid_month_day` to validate month and day information (:pull:`9`)
* extract daytime check from `do_day_check` and `do_night_check` (:pull:`21`)
* vectorised many of the QC checks to speed up processing on large datasets (:pull:`24`)
* moved to using pyproj for spherical geometry calculations (:pull:`24`)
* removed dependence on old Climatology class (:pull:`24`)
* utility functions moved from qc_sequential_reports to track_check_utils (:pull:`24`)
