==================================================================
Marine Quality Control: ``marine_qc`` toolbox
==================================================================

+----------------------------+----------------------------------------------------------------+
| Versions                   | |pypi| |versions|                                              |
+----------------------------+----------------------------------------------------------------+
| Documentation and Support  | |docs|                                                         |
+----------------------------+----------------------------------------------------------------+
| Open Source                | |license| |zenodo|                                             |
+----------------------------+----------------------------------------------------------------+
|                            | |fair-software| |ossf|                                         |
+----------------------------+----------------------------------------------------------------+
| Coding Standards           | |black| |ruff| |pre-commit| |codefactor|                       |
+----------------------------+----------------------------------------------------------------+
|                            | |security| |fossa|                                             |
+----------------------------+----------------------------------------------------------------+
| Development Status         | |status| |build| |coveralls|                                   |
+----------------------------+----------------------------------------------------------------+
| Funding                    | |c3s|                                                          |
+----------------------------+----------------------------------------------------------------+
|                            | |noc| |ukmcas|                                                 |
+----------------------------+----------------------------------------------------------------+


Installation
============

Installation using pip
----------------------

You can install the package directly from pip:

.. code-block:: console

    pip install marine_qc

If you want to contribute, we recommend cloning the repository and installing the package in development mode, e.g.

.. code-block:: console

    git clone https://github.com/glamod/marine_qc
    cd marine_qc
    pip install -e .

This will install the package but you can still edit it and you don't need the package in your :code:`PYTHONPATH`

Installation using uv
---------------------

You can install the package using `uv`_ package manager, this will add the library to your active environment:

.. code-block:: console

    uv add marine_qc

To develop the package using uv, the following will create a virtual environment, uv defaults to ``.venv``:

.. code-block:: console

    git clone https://github.com/glamod/marine_qc
    cd marine_qc
    uv venv --python 3.12      # Create an environment with the recommended python version
    source .venv/bin/activate  # Load the virtual environment (for bash or zsh)
    uv sync


Contributing to marine_qc
=========================

If you're interested in participating in the development of ``marine_qc`` by suggesting new features, new indices or report bugs, please leave us a message on the `issue tracker`_.

If you would like to contribute code or documentation (which is greatly appreciated!), check out the `Contributing Guidelines`_ before you begin!

How to cite this library
========================

If you wish to cite `marine_qc` in a research publication, we kindly ask that you refer to Zenodo: .

License
=======

This is free software: you can redistribute it and/or modify it under the terms of the `Apache License 2.0`_. A copy of this license is provided in the code repository (`LICENSE`_).

Credits
=======

``marine_qc`` development is funded through Copernicus Climate Change Service (C3S_).

This package was created with Cookiecutter_ and the `audreyfeldroy/cookiecutter-pypackage`_ project template.

.. hyperlinks

.. _Apache License 2.0: https://opensource.org/license/apache-2-0/

.. _audreyfeldroy/cookiecutter-pypackage: https://github.com/audreyfeldroy/cookiecutter-pypackage/

.. _C3S: https://climate.copernicus.eu/

.. _Contributing Guidelines: https://github.com/glamod/marine_qc/blob/main/CONTRIBUTING.rst

.. _Cookiecutter: https://github.com/cookiecutter/cookiecutter/

.. _LICENSE: https://github.com/glamod/marine_qc/blob/main/LICENSE

.. _issue tracker: https://github.com/glamod/marine_qc/issues

.. _uv: https://docs.astral.sh/uv/

.. |build| image:: https://github.com/glamod/marine_qc/actions/workflows/testing-suite.yml/badge.svg
        :target: https://github.com/glamod/marine_qc/actions/workflows/testing-suite.yml
        :alt: Build Status

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/psf/black
        :alt: Python Black

.. |c3s| image:: https://img.shields.io/badge/Powered%20by-Copernicus%20Climate%20Change%20Service-blue.svg
        :target: https://climate.copernicus.eu/
        :alt: Funding

.. |codefactor| image:: https://www.codefactor.io/repository/github/glamod/marine_qc/badge
		    :target: https://www.codefactor.io/repository/github/glamod/marine_qc
		    :alt: CodeFactor

.. |coveralls| image:: https://codecov.io/gh/glamod/marine_qc/branch/main/graph/badge.svg
	      :target: https://codecov.io/gh/glamod/marine_qc
	      :alt: Coveralls

.. |docs| image:: https://readthedocs.org/projects/marine_qc/badge/?version=latest
        :target: https://marine-qc.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. |fair-software| image:: https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green
   	    :target: https://fair-software.eu
	      :alt: FAIR-software

.. |fossa| image:: https://app.fossa.com/api/projects/custom%2B41576%2Fgithub.com%2Fglamod%2Fmarine_qc.svg?type=shield
        :target: https://app.fossa.com/projects/custom%2B41576%2Fgithub.com%2Fglamod%2Fmarine_qc?ref=badge_shield
        :alt: FOSSA

.. |license| image:: https://img.shields.io/github/license/glamod/marine_qc.svg
        :target: https://github.com/glamod/marine_qc/blob/main/LICENSE
        :alt: License

.. |ossf| image:: https://api.securityscorecards.dev/projects/github.com/glamod/marine_qc/badge
        :target: https://securityscorecards.dev/viewer/?uri=github.com/glamod/marine_qc
        :alt: OpenSSF Scorecard

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/glamod/marine_qc/main.svg
        :target: https://results.pre-commit.ci/latest/github/glamod/marine_qc/main
        :alt: pre-commit.ci status

.. |pypi| image:: https://img.shields.io/pypi/v/marine_qc.svg
        :target: https://pypi.python.org/pypi/marine_qc
        :alt: Python Package Index Build

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
        :target: https://github.com/astral-sh/ruff
        :alt: Ruff

.. |security| image:: https://bestpractices.coreinfrastructure.org/projects/./badge
	      :target: https://bestpractices.coreinfrastructure.org/projects/
	      :alt: OpenSSf Best Practices

.. |status| image:: https://www.repostatus.org/badges/latest/active.svg
        :target: https://www.repostatus.org/#concept
        :alt: Project Status: Concept – Minimal or no implementation has been done yet, or the repository is only intended to be a limited example, demo, or proof-of-concept.

.. |versions| image:: https://img.shields.io/pypi/pyversions/marine_qc.svg
        :target: https://pypi.python.org/pypi/marine_qc
        :alt: Supported Python Versions

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo..svg
        :target: https://doi.org/10.5281/zenodo.
 	:alt: DOI
