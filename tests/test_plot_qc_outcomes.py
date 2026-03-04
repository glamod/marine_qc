from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from marine_qc.plot_qc_outcomes import (
    _get_colours_labels,
    _make_plot,
    latitude_longitude_plot,
    latitude_variable_plot,
)


def test_get_colours_labels():
    colours, labels = _get_colours_labels(np.array([0, 1, 2]))

    np.testing.assert_array_equal(colours, ["#55ff55", "#ff5555", "#808080"])

    assert isinstance(labels, list)
    assert len(labels) == 3
    for label in labels:
        assert isinstance(label, Line2D)


def test_make_plot(tmp_path):
    plot_kwargs = {
        "xvalue": np.array([-10, 0, 10]),
        "yvalue": np.array([-10, 0, 10]),
        "flags": np.array([0, 1, 2]),
        "xlim": None,
        "ylim": None,
        "xlabel": "longitude",
        "ylabel": "latitude",
    }
    fig = _make_plot(
        **plot_kwargs,
        filename=None,
    )
    plt.close(fig)

    filename = tmp_path / "make_plot.png"
    _make_plot(
        **plot_kwargs,
        filename=filename,
    )
    assert filename.exists()
    assert filename.is_file()
    filename.unlink()


def test_latitude_variable_plot():
    fig = latitude_variable_plot(
        lat=np.array([-10, 0, 10]),
        value=np.array([5, 6, 7]),
        qc_outcomes=np.array([0, 1, 2]),
    )
    plt.close(fig)


def test_latitude_longitude_plot():
    fig = latitude_longitude_plot(
        lat=np.array([-10, 0, 10]),
        lon=np.array([-10, 0, 10]),
        qc_outcomes=np.array([0, 1, 2]),
    )
    plt.close(fig)
