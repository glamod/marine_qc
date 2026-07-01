.. marine_qc documentation master file

``Marine Quality Control`` documentation
----------------------------------------

.. include:: ./description.rst

It was developed from the quality control system developed at the Met Office and used in the production of
the `HadSST4`_, `HadISDH`_, and `HadNMAT2`_ datasets and is used in the Copernicus Climate Change Service for
the quality control of in situ marine observations.

The quality control routines have been extracted individually so that they can be used and reused in a more
modular way via a standardised interface.

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents:

   About <readme>
   installation
   usage
   introduction
   overview_ind
   overview_seq
   overview_grp
   overview_buoy_tracking
   overview_dup
   theoreticals
   notebooks/index
   contributing
   releasing
   dev_guide
   authors
   changelog

.. toctree::
   :maxdepth: 2
   :caption: User API

   api

.. toctree::
   :maxdepth: 1
   :caption: All Modules

   apidoc/modules

.. toctree::
   :caption: GitHub Repository

   glamod/Marine-Quality-Control <https://github.com/glamod/marine_qc>

About
-----

:Version: |pypi|

:Citation: |zenodo|

:License: |license|

|newline|

.. raw:: html

   <div style="text-align: right;">
       <img src="https://raw.githubusercontent.com/glamod/marine_qc/main/docs/_static/images/copernicus.png" width="150">
       <img src="https://raw.githubusercontent.com/glamod/marine_qc/main/docs/_static/images/logo_c3s-392x154.png" width="150">
   </div>


.. raw:: html

   <div style="text-align: right;">
       <img src="https://raw.githubusercontent.com/glamod/marine_qc/main/docs/_static/images/logo_DWD_RGB.png" width="150">
       <img src="https://raw.githubusercontent.com/glamod/marine_qc/main/docs/_static/images/logo_metoffice_rbg.png" width="150">
       <img src="https://raw.githubusercontent.com/glamod/marine_qc/main/docs/_static/images/LOGO_2025_NOC.png" width="150">
   </div>


.. include:: hyperlinks.rst

.. include:: ../README.rst
    :start-after: hyperlinks

.. |newline| raw:: html

   <br>

.. |tab| raw:: html

   &emsp;  <!-- HTML em-space -->
