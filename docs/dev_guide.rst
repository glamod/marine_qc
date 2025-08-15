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




