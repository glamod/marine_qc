import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


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
    colours = []
    for outcome in qc_outcomes:
        if outcome == 0:
            colours.append("#555555")
        else:
            colours.append("#ff5555")

    plt.plot(value, lat, c=colours)
    plt.set_ylim(-90.0, 90.0)

    plt.xlabel("Variable")
    plt.ylabel("Latitude")

    dir = Path(tempfile.mkdtemp())

    if filename is None:
        plt.savefig(dir / "latitude_variable_plot.png")
    else:
        plt.savefig(dir / filename)

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
    colours = []
    for outcome in qc_outcomes:
        if outcome == 0:
            colours.append("#555555")
        else:
            colours.append("#ff5555")

    plt.plot(lon, lat, c=colours)
    plt.set_xlim(-180.0, 180.0)
    plt.set_ylim(-90.0, 90.0)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    dir = Path(tempfile.mkdtemp())

    if filename is None:
        plt.savefig(dir / "latitude_longitude_plot.png")
    else:
        plt.savefig(dir / filename)

    plt.close()
