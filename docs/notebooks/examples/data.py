import pandas as pd
import numpy as np
import xarray as xr

import regionmask

from datetime import datetime

def get_individual_data():
  return pd.DataFrame({
    "location": [
        "Mediterranean Sea",
        "North Sea",
        "South Pacific Ocean",
        "New York, USA",
        "Paris, France",
        "Tokyo, Japan",
        "Sydney, Australia",
        "Gulf of Mexico",
        "Equatorial Atlantic",
        "Norwegian Sea"        
    ],
    "lat": [
        36.0,
        54.5,
        -15.0,
        40.7,
        48.9,
        35.7,
        -33.9,
        25.0,
        0.0,
        60.0,       
    ],
    "lon": [
        18.0,
        3.0,
        -140.0,
        -74.0,
        2.3,
        139.7,
        151.2,
        -90.0,
        -30.0,
        5.0,    
    ],
    "date": pd.to_datetime([
        "2025-06-01 06:00:00",
        "2025-06-01 12:00:00",
        "2025-06-01 18:00:00",
        "2025-06-02 09:15:00",
        "2025-06-02 14:30:00",
        "2025-06-03 08:45:00",
        "2025-06-03 20:10:00",
        "2025-06-04 11:00:00",
        "2025-06-04 16:20:00",
        "2025-06-05 07:50:00"
    ]),
    "sea_surface_temperature": [
        22.8,
        13.6,
        27.4,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        29.1,
        28.3,
        8.5,        
    ],
    "wind_speed": [
        5.2,
        11.4,
        7.8,
        4.1,
        3.5,
        6.2,
        8.1,
        10.5,
        5.9,
        14.3        
    ],
    "wind_direction": [
        135,
        270,
        90,
        180,
        45,
        225,
        160,
        315,
        110,
        290,        
    ],
  })


def get_climatology_dataset():
    lat = np.arange(-90, 90, 1)
    lon = np.arange(-180, 180, 1)
    time = xr.DataArray(
      pd.to_datetime(["2026-07-01T12:00:00"]),
      dims="time",
      attrs={
        "standard_name": "time",
        "axis": "T",
      },
    )
    
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(lon, lat)
    mask_2d = xr.where(np.isnan(land), 0, 1).astype("int8")
    mask_3d = np.broadcast_to(mask_2d.values, (1, mask_2d.shape[0], mask_2d.shape[1]))

    ds = xr.Dataset(
        data_vars={
            "land_sea_mask": (
                ("time", "latitude", "longitude"),
                mask_3d,
                {
                    "long_name": "Land-sea mask",
                    "flag_values": np.array([0, 1], dtype="int8"),
                    "flag_meanings": "land sea",
                    "coordinates": "latitude longitude time",
                },
            )
        },
        coords={
            "time": time,
            "latitude": (
                "latitude",
                lat,
                {
                    "standard_name": "latitude",
                    "units": "degrees_north",
                    "axis": "Y",
                },
            ),
            "longitude": (
                "longitude",
                lon,
                {
                    "standard_name": "longitude",
                    "units": "degrees_east",
                    "axis": "X",
                },
            ),
        },
    )

    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    sst_2d = 28.0 * np.cos(np.deg2rad(lat2d)) ** 2
    sst_2d += 2.0 * np.sin(np.deg2rad(lon2d / 2.0)) * np.cos(np.deg2rad(lat2d))
    sst_2d = np.clip(sst_2d, -1.8, None)

    sst_2d = xr.where(ds.land_sea_mask.isel(time=0) == 0, sst_2d, np.nan)
    sst_3d = np.broadcast_to(sst_2d.values, (1, sst_2d.shape[0], sst_2d.shape[1]))

    ds["sea_surface_temperature"] = (
        ("time", "latitude", "longitude"),
        sst_3d,
        {
            "standard_name": "sea_surface_temperature",
            "long_name": "Sea surface temperature",
            "units": "degC",
            "coordinates": "time latitude longitude",
            "grid_mapping": "crs",
        },
    )

    ds["crs"] = xr.DataArray(
        0,
        attrs={
            "grid_mapping_name": "latitude_longitude",
            "epsg_code": "EPSG:4326",
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        },
    )

    return ds
