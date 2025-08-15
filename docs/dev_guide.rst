.. marine QC documentation master file

-----------------
Developer's Guide
-----------------

The QC functions implemented in this package make use of decorators to maintain flexibility for the user. Three
key decorators are:

* :func:`.inspect_arrays` - Decorator that inspects specified input parameters of a function,
  converts them to one-dimensional NumPy arrays, and validates their lengths. This allows the user to run the
  functions using lists, numpy arrays or Pandas DataSeries according to their need.
* :func:`.inspect_climatology` - Decorator used to automatically extract values from a climatology at the locations
  of the reports if one is provided.
* :func:`.convert_units` - Decorator to automatically convert specified function arguments to desired units.
* :func:`.post_format_return_type` - Decorator to format a function's return value to match the type of its original
  input(s).
* :func:`.convert_date` - Decorator to extract date components and inject them as function parameters.

An example use case for four of these can be seen in the :func:`.do_climatology_check` function

.. code-block:: python

  @post_format_return_type(["value"])
  @inspect_arrays(["value", "climatology"])
  @convert_units(value="unknown", climatology="unknown")
  @inspect_climatology("climatology", optional="standard_deviation")
  def do_climatology_check(
      value: ValueFloatType,
      climatology: ClimFloatType,
      maximum_anomaly: float,
      standard_deviation: ValueFloatType = "default",
      standard_deviation_limits: tuple[float, float] | None = None,
      lowbar: float | None = None,
  )

This can be called in a large number of different ways. For example, it can be called specifying all inputs as
arrays. For example

.. code-block:: python

  values = np.array([274.1, 276.2, 280.0])
  climatology = np.array([274.3, 274.7, 275.3])
  result = do_climatology_check(values, climatology, 8.0)

Alternatively, we can provide a :class:`.Climatology` object and the locations of the reports like so:

.. code-block:: python

  values = np.array([274.1, 276.2, 280.0])
  climatology = Climatology.open_netcdf_file('climatology_file.nc', 'variable_name')
  latitudes = np.array([76.0, 76.2, 76.2])
  longitudes = np.array([-5.1, -5.0, -4.8])
  dates = pd.date_range(start="1850-01-01", freq="1h", periods=3)
  result = do_climatology_check(values, climatology, 8.0, lat=latitudes, lon=longitudes, date=dates)

This will automatically extract the climatological values from the :class:`.Climatology` at the location of the
reports and pass them to the function.

We can also convert the units from degrees Celsius to Kelvin (the preferred SI unit).

.. code-block:: python

  values = np.array([1.15, 3.05, 6.85])
  climatology = np.array([274.3, 274.7, 275.3])
  result = do_climatology_check(values, climatology, 8.0, units={"value": "degC"})
