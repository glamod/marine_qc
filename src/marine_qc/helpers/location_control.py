"""Some generally helpful location control functions for base QC."""

from __future__ import annotations

import numpy as np

from .auxiliary import is_scalar_like


__all__ = [
    "get_xindex",
    "get_yindex",
]


def _get_coordinate_index(  # numpydoc ignore=PR01,RT01
    coord: float | np.ndarray,
    axis: np.ndarray | None,
    res: float | None,
    lower_bound: float,
    upper_bound: float,
    offset: int,
) -> int | np.ndarray:
    """Calculate indices for a regular global coordinate grid."""
    if axis is None and res is None:
        raise ValueError("One of axis and res must be provided.")
    if axis is not None and res is not None:
        raise ValueError("Only one of axis and res must be provided.")

    coord_arr = np.atleast_1d(coord).astype(float)

    if res is not None:
        coord_arr[coord_arr == lower_bound] += 0.001
        coord_arr[coord_arr == upper_bound] -= 0.001

        index = (upper_bound - lower_bound) / (2 * res) + offset * (coord_arr / res).astype(int) - (coord_arr <= 0 if offset == 1 else coord_arr > 0)

    elif axis is not None:
        axis_0 = axis[0]
        delta = axis[1] - axis[0]

        # Determine whether grid cells are defined by centres or boundaries
        if axis_0 not in [lower_bound, upper_bound]:
            if axis_0 == lower_bound + delta / 2:
                axis_0 = lower_bound
            elif axis_0 == upper_bound + delta / 2:
                axis_0 = upper_bound
            else:
                raise RuntimeError(
                    f"I can't work this grid out, grid box boundaries are not at {lower_bound} or {upper_bound} or their half-cell offsets."
                )

        index = (coord_arr - axis_0) / delta
        index[index >= len(axis)] = len(axis) - 1

    if is_scalar_like(coord):
        return int(index[0])

    return index.astype(int)


def get_xindex(lon: float | np.ndarray, lon_axis: np.ndarray | None = None, res: float | None = None) -> int | np.ndarray:
    """
    Convert longitude(s) to x-indices of a global longitude grid.

    Parameters
    ----------
    lon : float or np.ndarray
        Longitude(s) in degrees to select.
    lon_axis : np.ndarray, optional
        Longitude coordinate axis of the grid. If provided, indices are
        calculated based on this axis. The axis must describe a regular
        global grid with boundaries at -180 and 180.
    res : float, optional
        Grid resolution in degrees. If provided, indices are calculated
        using the MDS2/3 indexing convention for a regular global grid
        extending from -180 to 180.

    Returns
    -------
    int or np.ndarray
        Grid index or indices corresponding to the input longitude(s).
        Returns an integer for scalar input and a numpy.ndarray for
        array input.

    Raises
    ------
    ValueError
        If none or both of `lon_axis` and `res` are provided.
    RuntimeError
        If the grid boundaries cannot be determined from `lon_axis`.
    """
    return _get_coordinate_index(
        lon,
        lon_axis,
        res,
        lower_bound=-180.0,
        upper_bound=180.0,
        offset=1,
    )


def get_yindex(lat: float | np.ndarray, lat_axis: np.ndarray | None = None, res: float | None = None) -> int | np.ndarray:
    """
    Convert latitude(s) to y-indices of a global latitude grid.

    Parameters
    ----------
    lat : float or np.ndarray
        Latitude(s) in degrees to select.
    lat_axis : np.ndarray, optional
        Latitude coordinate axis of the grid. If provided, indices are
        calculated based on this axis. The axis must describe a regular
        global grid with boundaries at -90 and 90.
    res : float, optional
        Grid resolution in degrees. If provided, indices are calculated
        using the MDS2/3 indexing convention for a regular global grid
        extending from -90 to 90.

    Returns
    -------
    int or np.ndarray
        Grid index or indices corresponding to the input latitude(s).
        Returns an integer for scalar input and a numpy.ndarray for
        array input.

    Raises
    ------
    ValueError
        If none or both of `lat_axis` and `res` are provided.
    RuntimeError
        If the grid boundaries cannot be determined from `lat_axis`.
    """
    return _get_coordinate_index(
        lat,
        lat_axis,
        res,
        lower_bound=-90.0,
        upper_bound=90.0,
        offset=-1,
    )
