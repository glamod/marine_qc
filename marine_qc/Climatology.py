"""Quality control climatology module."""

from __future__ import annotations

import numpy as np
from netCDF4 import Dataset

from .location_control import (
    lat_to_yindex,
    lon_to_xindex,
    mds_lat_to_yindex,
    mds_lon_to_xindex,
)
from .time_control import day_in_year, get_month_lengths, which_pentad


class Climatology:
    """
    Class for dealing with climatologies, reading, extracting values etc.
    Automatically detects if this is a single field, pentad or daily climatology
    """

    def __init__(self, infield):
        """
        Read in the climatology for variable var from infile

        :param infield: numpy array containing the climatology
        :type infield: numpy array
        """
        self.field = infield
        self.n = self.field.shape[0]
        assert self.n in [1, 73, 365], "weird shaped field"
        self.res = 180.0 / self.field.shape[1]

    @classmethod
    def from_filename(cls, infile, var):
        """
        Read in the climatology for variable var from infile

        :param infile: filename of a netcdf file
        :param var: the variable name to be extracted from the netcdf file
        :type infile: string
        :type var: string
        """
        latitudes = None
        longitudes = None
        if infile is not None:
            climatology = Dataset(infile)
            field = climatology.variables[var][:]

            lat_synonyms = ["lat", "lats", "latitude", "latitudes"]
            found_lat = False
            for lat in lat_synonyms:
                if lat in climatology.variables:
                    latitudes = climatology.variables[lat][:]
                    found_lat = True
            assert found_lat, (
                "no readable latitude information in NetCDF file: " + infile
            )

            lon_synonyms = ["lon", "lons", "long", "longs", "longitude", "longitudes"]
            found_lon = False
            for lon in lon_synonyms:
                if lon in climatology.variables:
                    longitudes = climatology.variables[lon][:]
                    found_lon = True
            assert found_lon, (
                "no readable longitude information in NetCDF file: " + infile
            )

            climatology.close()

            # transpose the fields if the second axis is longitude
            if field.shape[0] == 1 and field.shape[1] == 360 and field.shape[2] == 180:
                field = field.transpose(0, 2, 1)

            # added an exception here as the OSTIA background variances are specified from S to N so first
            # element in latitudes is negative
            if latitudes[0] < 0:
                field = np.flip(field, 1)

            # if the longitudes start near zero then roll the array along its longitude axis
            if 0.0 < longitudes[0] < 1.0:
                lon_len = field.shape[2]
                field = np.roll(field, lon_len // 2, axis=2)

            if field.ndim == 4:
                field = field[:, 0, :, :]

        else:
            field = np.ma.array(np.zeros((1, 180 * 20, 360 * 20)), mask=True)

        return cls(field)

    def get_tindex(self, month, day):
        """Get the time index of the input month and day.

        :param month: month for which the time index is required
        :param day: day for which the time index is required
        :type month: integer
        :type day: integer
        :return: time index for specified month and day.
        :rtype: integer
        """
        tindex = None

        if self.n == 1:
            tindex = 0
        if self.n == 73:
            tindex = which_pentad(month, day) - 1
        if self.n == 365:
            tindex = day_in_year(month, day) - 1

        return tindex

    def get_value_ostia(self, lat, lon):
        """
        :param lat: latitude of location to extract value from in degrees of arc
        :param lon: longitude of location to extract value from in degrees of arc
        :return: SST at that location or None

        :type lat: float
        :type lon: float
        :rtype: float
        """
        yindex = mds_lat_to_yindex(lat, res=0.05)
        xindex = mds_lon_to_xindex(lon, res=0.05)
        tindex = 0

        result = self.field[tindex, yindex, xindex]

        if isinstance(result, np.float64) or isinstance(result, np.float32):
            pass
        else:
            if result.mask:
                result = None
            else:
                result = result.data[0]

        return result

    def get_value_mds_style(self, lat, lon, month, day):
        """
        Get the value from the climatology at the give position and time using the MDS
        method for deciding which grid cell borderline cases fall into

        :param lat: latitude of location to extract value from in degrees
        :param lon: longitude of location to extract value from in degrees
        :param month: month for which the value is required
        :param day: day for which the value is required
        :type lat: float
        :type lon: float
        :type month: integer
        :type day: integer
        :return: climatology value at specified location and time.
        :rtype: float
        """
        if month is None:
            return
        if day is None:
            return
        if month < 1 or month > 12:
            return
        ml = get_month_lengths(2004)
        if day < 1 or day > ml[month - 1]:
            return

        yindex = mds_lat_to_yindex(lat, res=1.0)
        xindex = mds_lon_to_xindex(lon, res=1.0)
        tindex = self.get_tindex(month, day)

        result = self.field[tindex, yindex, xindex]

        if isinstance(result, np.float64) or isinstance(result, np.float32):
            pass
        else:
            if result.mask:
                result = None
            else:
                result = result.data[0]

        return result

    def get_value(self, lat, lon, month, day):
        """
        Get the value from the climatology at the give position and time

        :param lat: latitude of location to extract value from in degrees
        :param lon: longitude of location to extract value from in degrees
        :param month: month for which the value is required
        :param day: day for which the value is required
        :type lat: float
        :type lon: float
        :type month: integer
        :type day: integer
        :return: climatology value at specified location and time.
        :rtype: float
        """
        if month is None:
            return
        if day is None:
            return
        if month < 1 or month > 12:
            return None
        ml = get_month_lengths(2004)
        if day < 1 or day > ml[month - 1]:
            return None
        if lat is None:
            return
        if lat < -180 or lat > 180:
            return
        if lon is None:
            return
        if lon < -80 or lon > 90:
            return

        yindex = lat_to_yindex(lat, self.res)
        xindex = lon_to_xindex(lon, self.res)
        tindex = self.get_tindex(month, day)

        result = self.field[tindex, yindex, xindex]

        if isinstance(result, np.float64) or isinstance(result, np.float32):
            pass
        else:
            if result.mask:
                result = None
            else:
                result = result.data[0]

        return result
