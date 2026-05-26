"""Settings for duplicate check."""

from __future__ import annotations

# from recordlinkage import Compare
# from recordlinkage.compare import Numeric


# __all__ = ["Compare"]

# _method_kwargs = {
#    "left_on": "date",
#    "window": 5,
#    "block_on": ["station_id"],
# }

# _compare_kwargs = {
#    "station_id": {"method": "exact"},
#    "lon": {
#        "method": "numeric",
#        "kwargs": {"method": "step", "offset": 0.11},
#    },
#    "lat": {
#        "method": "numeric",
#        "kwargs": {"method": "step", "offset": 0.11},
#    },
#    "date": {
#        "method": "date2",
#        "kwargs": {"method": "gauss", "offset": 60.0},
#    },
#    "vsi": {
#        "method": "numeric",
#        "kwargs": {"method": "step", "offset": 0.09},
#    },
#    "dsi": {
#        "method": "numeric",
#        "kwargs": {"method": "step", "offset": 0.9},
#    },
# }


# class Date2(Numeric):  # type: ignore[misc]
#    """Copy of ``rl.compare.Numeric`` class."""

#    pass


# def date2(object: Compare, *args: Any, **kwargs: Any) -> Compare:
#    r"""
#    New method for ``rl.Compare`` object using ``Date2`` object.

#    Parameters
#    ----------
#    object : Compare
#        Object to with the new method should be added.
#    \*args : Any
#        Positional argument for `Date2`.
#    \**kwargs : Any
#        Keyword-arguments for `Date2`.

#    Returns
#    -------
#    Compare
#        Compare object with new method.
#    """
#    compare = Date2(*args, **kwargs)
#    object.add(compare)
#    return object


# Compare.date2 = date2
