"""
Plot QC outcomes
================

Some plotting routines for QC outcomes
"""

import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def _get_colours_labels(qc_outcomes):
    colours = []
    colour_passed = "#55ff55"
    colour_failed = "#ff5555"
    colour_other = "#808080"
    passed = 0
    failed = 0
    other = 0
    for outcome in qc_outcomes:
        if outcome == 0:
            colours.append(colour_passed)
            passed += 1
        elif outcome == 1:
            colours.append(colour_failed)
            failed += 1
        else:
            colours.append(colour_other)
            other += 1

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"0: {passed}",
            markerfacecolor=colour_passed,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"1: {failed}",
            markerfacecolor=colour_failed,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"other: {other}",
            markerfacecolor=colour_other,
        ),
    ]
    return colours, legend_elements


def latitude_variable_plot(
    lat: np.ndarray, value: np.ndarray, qc_outcomes: np.ndarray, filename: str = None
):
    """
    Plot a graph of points showing the latitude and value of a set of observations coloured according to
    whether or not they pass qc. The graph is output to the temporary directory.

    Parameters
    ----------
    lat: np.ndarray
        Array of latitude values in degrees
    value: np.ndarray
        Array of observed values for the variable
    qc_outcomes: np.ndarray
        Array containing the QC outcomes, with 0 meaning pass and non-zero entries indicating failure
    filename: str or None
        Filename to save the figure to. If None, the figure is saved with a standard name

    Returns
    -------
    None
    """
    colours, legend_elements = _get_colours_labels(qc_outcomes)

    plt.scatter(value, lat, c=colours)
    plt.ylim(-90.0, 90.0)

    plt.xlabel("Variable")
    plt.ylabel("Latitude")

    plt.legend(handles=legend_elements)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    plt.close()


def latitude_longitude_plot(
    lat: np.ndarray, lon: np.ndarray, qc_outcomes: np.ndarray, filename: str = None
) -> None:
    """
    Plot a graph of points showing the latitude and longitude of a set of observations coloured according to
    the QC outcomes.

    Parameters
    ----------
    lat: np.ndarray
        array of latitude values in degrees
    lon: np.ndarray
        array of longitude values in degrees
    qc_outcomes: np.ndarray
        array containing the QC outcomes, with 0 meaning pass and non-zero entries indicating failure
    filename: str or None
        Filename to save the figure to. If None, the figure is saved with a standard name

    Returns
    -------
    None
    """
    colours, legend_elements = _get_colours_labels(qc_outcomes)

    plt.scatter(lon, lat, c=colours)
    plt.xlim(-180.0, 180.0)
    plt.ylim(-90.0, 90.0)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.legend(handles=legend_elements)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    plt.close()
