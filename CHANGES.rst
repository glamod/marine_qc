
=========
Changelog
=========

0.0.1 (unreleased)
------------------
Contributors to this version: Ludwig Lierhammer (:user:`ludwiglierhammer`) and John F Kennedy (:user:`jjk-code-otter`)

Announcements
^^^^^^^^^^^^^
* This marine QC repository is a copy of the results of https://github.com/glamod/glamod-marine-processing/pull/117. This repository replaces `glamod_marine_proccesing.qc_suite.modules` as an independent package (:pull:`6`).

New features and enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* ``do_night_check``: reverse ``do_day_check`` (:pull:`21`)

Internal changes
^^^^^^^^^^^^^^^^
* Remove both jupyter notebook specific (nbqa-pyupgrade, nbqa-black, nbqa-isort, nbstripout) and json-related (pretty-format-json) pre-commit hooks (:pull:`7`)
* Replace assert statements with if statement raising error messages (:pull:`7`)
* Split some try statements into single if statements giving warnings (:pull:`7`)
* Fixing some typos in docstrings and comments (:pull:`7`)
* extract daytime check from `do_day_check` and `do_night_check` (:pull:`21`)
