.. marine QC documentation master file

Installation
============

The **marine_qc**  toolbox is a pure Python package, but it has a few dependencies that rely in a specific python and module version.

If you don't have pip_ installed, this `Python installation guide`_ can guide you through the process.

.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

Stable release
~~~~~~~~~~~~~~

To install the **marine_qc** toolbox in your current environment, run this command in your terminal:

.. code-block:: console

  python -m pip install marine_qc

This is the preferred method to install the **marine_qc** toolbox, as it will always install the most recent stable release.

Alternatively, it can be installed using the `uv`_ package manager:

.. code-block:: console

    python -m uv add marine_qc

.. include:: hyperlinks.rst

From source
~~~~~~~~~~~

.. warning:: It is not guaranteed that the version on source will run stably. Therefore, we highly recommend to use the ``Stable release`` installation.

The source for the **marine_qc** can be downloaded from the `GitHub repository`_ via git_.

#. Download the source code from the `Github repository`_ using one of the following methods:

    * Clone the public repository:

        .. code-block:: console

            git clone git@github.com:ludwiglierhammer/Marine_Quality_Control.git

    * Download the `tarball <https://github.com/ludwiglierhammer/Marine-Quality-Control/tarball/main>`_:

        .. code-block:: console

            curl -OJL https://github.com/ludwiglierhammer/Marine-Quality-Control/tarball/main

#. Once you have a copy of the source, you can install it with pip_:

.. code-block:: console

   python -m pip install -e .

Or using the `uv`_ package manager to install marine_qc:

.. code-block:: console

    python -m uv add .

Development mode
~~~~~~~~~~~~~~~~

If you're interested in participating in the development of the **marine_qc** toolbox, you can install the package in development mode after cloning the repository from source:

.. code-block:: console

    python -m pip install -e .[dev]      # Install optional development dependencies in addition
    python -m pip install -e .[docs]     # Install optional dependencies for the documentation in addition
    python -m pip install -e .[all]      # Install all the above for complete dependency version

Alternatively, you can use the uv package manager:

.. code-block:: console

    uv sync       # Install in development mode and create a virtual environment

You can specify optional dependency groups with the `--extra` option.

Creating a Conda Environment
----------------------------

To create a conda environment including **marine_qc**'s dependencies and and development dependencies, run the following command from within your cloned repo:

.. code-block:: console

    $ conda env create -n my_qc_env python=3.12 --file=environment.yml
    $ conda activate my_qc_env
    (my_qc_env) $ make dev

.. include:: ../README.rst
    :start-after: hyperlinks
