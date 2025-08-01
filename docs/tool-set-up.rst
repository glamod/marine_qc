.. marine QC documentation master file

Installation
============

The **marine_qc**  toolbox is a pure Python package, but it has a few dependencies that rely in a specific python and module version.

Stable release
~~~~~~~~~~~~~~

To install the **marine_qc** toolbox in your current environment, run this command in your terminal:

.. code-block:: console

  pip install marine_qc

This is the preferred method to install the **marine_qc** toolbox, as it will always install the most recent stable release.

Alternatively, it can be installed using the `uv`_ package manager:

.. code-block:: console

    uv add marine_qc

.. include:: hyperlinks.rst

From source
~~~~~~~~~~~

.. warning:: It is not guaranteed that the version on source will run stably. Therefore, we highly recommend to use the ``Stable release`` installation.

The source for the **marine_qc** can be downloaded from the `GitHub repository`_ via git_.

You can either clone the public repository:

.. code-block:: console

    git clone https://github.com/glamod/marine_qc

or download th tarball_:

.. code-block:: console

   curl -OJL https://github.com/glamod/marine_qc/tarball/master

Once you have a copy of the source, you can install it with pip_:

.. code-block:: console

   pip install -e .

Or using the `uv`_ package manager to install marine_qc:

.. code-block:: console

    uv add .

Development mode
~~~~~~~~~~~~~~~~

If you're interested in participating in the development of the **marine_qc** toolbox, you can install the package in development mode after cloning the repository from source:

.. code-block:: console

    pip install -e .[dev]      # Install optional development dependencies in addition
    pip install -e .[docs]     # Install optional dependencies for the documentation in addition
    pip install -e .[all]      # Install all the above for complete dependency version

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
    (my_qc_env) $ python -m pip install -e --no-deps .

.. include:: ../README.rst
    :start-after: hyperlinks
