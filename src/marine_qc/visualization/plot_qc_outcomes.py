"""
Plot QC outcomes.

Some plotting routines for QC outcomes
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure, lines


def _get_colours_labels(qc_outcomes: np.ndarray) -> tuple[np.ndarray, list[lines.Line2D]]:
    """
    Get color lebels.

    Parameters
    ----------
    qc_outcomes : numpy.ndarray
        Array containing the QC outcomes, with 0 meaning pass and non-zero entries indicating failure.

    Returns
    -------
    tuple of (numpy.ndarray, list of lines.Line2D)
        Color names and legend elements.
    """
    colour_passed = "#55ff55"
    colour_failed = "#ff5555"
    colour_other = "#808080"

    passed = 0
    failed = 0
    other = 0

    colours_list = []
    for outcome in qc_outcomes:
        if outcome == 0:
            colours_list.append(colour_passed)
            passed += 1
        elif outcome == 1:
            colours_list.append(colour_failed)
            failed += 1
        else:
            colours_list.append(colour_other)
            other += 1

    colours = np.array(colours_list, dtype=str)

    legend_elements = [
        lines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"0: {passed}",
            markerfacecolor=colour_passed,
        ),
        lines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"1: {failed}",
            markerfacecolor=colour_failed,
        ),
        lines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"other: {other}",
            markerfacecolor=colour_other,
        ),
    ]
    return colours, legend_elements


def _make_plot(
    xvalue: np.ndarray,
    yvalue: np.ndarray,
    flags: np.ndarray,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    xlabel: str,
    ylabel: str,
    marker_size: int | None,
    filename: str | None,
) -> figure.Figure:
    """
    Make plot.

    Parameters
    ----------
    xvalue : numpy.ndarray
        Array of x values.
    yvalue : numpy.ndarray
        Array of y values.
    flags : numpy.ndarray
        Array containing the QC outcomes, with 0 meaning pass and non-zero entries indicating failure.
    xlim : tuple of float and float or None
        If not None: set xlim for plotting.
    ylim : tuple of float and float or None
        If not None: set ylim for plotting.
    xlabel : str
        Name of the x axis.
    ylabel : str
        Name of the y axis.
    marker_size : int
        Marker size in points.
    filename : str or None
        Filename to save the figure to. If None, the figure is not saved nut shown.

    Returns
    -------
    Figure
        The main figure obkect created by `plt.subplots()`.
    """
    colours, legend_elements = _get_colours_labels(flags)

    mask_passed = flags == 0
    mask_failed = flags == 1
    mask_other = (flags != 0) & (flags != 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    titles = ["QC == 0 (Passed)", "QC == 1 (Failed)", "QC == Other", "All Points"]

    masks = [mask_passed, mask_failed, mask_other, np.ones_like(flags, dtype=bool)]

    marker_size = marker_size or 1

    for i in range(4):
        ax = axes[i]
        ax.scatter(xvalue[masks[i]], yvalue[masks[i]], c=colours[masks[i]], s=marker_size)
        ax.set_title(titles[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)

    fig.legend(
        handles=legend_elements,
        loc="center",
        ncol=len(legend_elements),
        bbox_to_anchor=(0.5, 0.53),
    )

    plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))

    if filename is None:
        plt.show(block=False)
    else:
        plt.savefig(filename)

    return fig


def plot_variable_longitude(
    lon: np.ndarray,
    value: np.ndarray,
    qc_outcomes: np.ndarray,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    marker_size: int | None = None,
    filename: str | None = None,
) -> figure.Figure:
    """
    Plot a graph of points showing the value and the longitude of a set of observations coloured according to flagged outcomes.

    Parameters
    ----------
    lon : numpy.ndarray
        Array of longitude values in degrees.
    value : numpy.ndarray
        Array of observed values for the variable.
    qc_outcomes : numpy.ndarray
        Array containing the QC outcomes, with 0 meaning pass and non-zero entries indicating failure.
    xlim : tuple of float and float, optional
        Limits of the current x axis. If None, set to (-180.0, 180.0).
    ylim : tuple of float and float, optional
        Limits of the current y axis.
    marker_size : int, optional
        Marker size in points. If None, it is set to 1.
    filename : str, optional
        Filename to save the figure to. If None, the figure is not saved but shown.

    Returns
    -------
    Figure
        The main figure object created by `plt.subplots()`.
    """
    if xlim is None:
        xlim = (-180.0, 180.0)
    return _make_plot(
        xvalue=lon,
        yvalue=value,
        flags=qc_outcomes,
        xlim=None,
        ylim=None,
        xlabel="Longitude",
        ylabel="Variable",
        marker_size=marker_size,
        filename=filename,
    )


def plot_latitude_variable(
    lat: np.ndarray,
    value: np.ndarray,
    qc_outcomes: np.ndarray,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    marker_size: int | None = None,
    filename: str | None = None,
) -> figure.Figure:
    """
    Plot a graph of points showing the latitude and the value of a set of observations coloured according to flagged outcomes.

    Parameters
    ----------
    lat : numpy.ndarray
        Array of latitude values in degrees.
    value : numpy.ndarray
        Array of observed values for the variable.
    qc_outcomes : numpy.ndarray
        Array containing the QC outcomes, with 0 meaning pass and non-zero entries indicating failure.
    xlim : tuple of float and float, optional
        Limits of the current x axis.
    ylim : tuple of float and float, optional
        Limits of the current y axis. If None, set to (-90.0, 90.0).
    marker_size : int, optional
        Marker size in points. If None, it is set to 1.
    filename : str, optional
        Filename to save the figure to. If None, the figure is not saved but shown.

    Returns
    -------
    Figure
        The main figure object created by `plt.subplots()`.
    """
    if ylim is None:
        ylim = (-90.0, 90.0)
    return _make_plot(
        xvalue=value,
        yvalue=lat,
        flags=qc_outcomes,
        xlim=None,
        ylim=ylim,
        xlabel="Variable",
        ylabel="Latitude",
        marker_size=marker_size,
        filename=filename,
    )


def plot_latitude_longitude(
    lat: np.ndarray,
    lon: np.ndarray,
    qc_outcomes: np.ndarray,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    marker_size: int | None = None,
    filename: str | None = None,
) -> figure.Figure:
    """
    Plot a graph of points showing the latitude and the longitude of a set of observations coloured according to flagged outcomes.

    Parameters
    ----------
    lat : numpy.ndarray
        Array of latitude values in degrees.
    lon : numpy.ndarray
        Array of longitude values in degrees.
    qc_outcomes : numpy.ndarray
        Array containing the QC outcomes, with 0 meaning pass and non-zero entries indicating failure.
    xlim : tuple of float and float, optional
        Limits of the current x axis. If None, set to (-180.0, 180.0).
    ylim : tuple of float and float, optional
        Limits of the current y axis. If None, set to (-90.0, 90.0).
    marker_size : int, optional
        Marker size in points. If None, it is set to 1.
    filename : str, optional
        Filename to save the figure to. If None, the figure is not saved but shown.

    Returns
    -------
    Figure
        The main figure object created by `plt.subplots()`.
    """
    if xlim is None:
        xlim = (-180.0, 180.0)
    if ylim is None:
        ylim = (-90.0, 90.0)
    return _make_plot(
        xvalue=lon,
        yvalue=lat,
        flags=qc_outcomes,
        xlim=xlim,
        ylim=ylim,
        xlabel="Longitude",
        ylabel="Latitude",
        marker_size=marker_size,
        filename=filename,
    )


plot_latitude_longitude.__module__ = "marine_qc.visualization"
plot_latitude_variable.__module__ = "marine_qc.visualization"
plot_variable_longitude.__module__ = "marine_qc.visualization"
