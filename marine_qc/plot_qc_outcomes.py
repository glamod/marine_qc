import tempfile
from pathlib import Path
import matplotlib.pyplot as plt


def latitude_variable_plot(lat, value, qc_outcomes, filename=None):

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

def latitude_longitude_plot(lat, lon, qc_outcomes, filename=None):
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
