import cartopy.io.shapereader as shpreader
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point
from shapely.ops import unary_union


def get_individual_data():
    return pd.DataFrame(
        {
            "location": [
                "Mediterranean Sea",
                "North Sea",
                "South Pacific Ocean",
                "Paris, France",
                "Tokyo, Japan",
                "Sydney, Australia",
                "Gulf of Mexico",
                "Equatorial Atlantic",
                "Norwegian Sea",
            ],
            "lat": [
                36.0,
                54.5,
                -140.0,
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
                -15.0,
                2.3,
                139.7,
                151.2,
                -90.0,
                -30.0,
                5.0,
            ],
            "date": pd.to_datetime(
                [
                    "2025-06-01 06:00:00",
                    "2025-06-01 12:00:00",
                    "2025-06-01 18:00:00",
                    "2025-06-02 14:30:00",
                    "2025-06-03 08:45:00",
                    "2025-06-03 20:10:00",
                    None,
                    "2025-06-04 16:20:00",
                    "2025-06-05 07:50:00",
                ]
            ),
            "sea_surface_temperature": [
                22.8,
                13.6,
                27.4,
                np.nan,
                np.nan,
                np.nan,
                29.1,
                28.3,
                8.5,
            ],
            "wind_speed": [5.2, 0.0, 7.8, 3.5, 6.2, 8.1, 10.5, 5.9, 14.3],
            "wind_direction": [
                135,
                270,
                90,
                45,
                225,
                160,
                315,
                110,
                290,
            ],
        }
    )


def get_sequential_data():
    n = 24
    datetime = pd.date_range("2026-07-01 00:00:00", periods=n, freq="1h")
    lat0, lon0 = 45.0, -30.0

    base_speed = 5.0
    amp_speed = 1.0
    speed_ms = base_speed + amp_speed * np.sin(2 * np.pi * np.arange(n) / 24)

    base_dir = 120.0
    amp_dir = 30.0
    heading_deg = base_dir + amp_dir * np.sin(2 * np.pi * np.arange(n) / 24)

    earth_radius = 6371000

    dlat = (speed_ms * 3600 / earth_radius) * (180 / np.pi) * np.cos(np.deg2rad(heading_deg))
    dlon = (speed_ms * 3600 / earth_radius) * (180 / np.pi) / np.cos(np.deg2rad(lat0))

    lat = lat0 + np.cumsum(dlat)
    lon = lon0 + np.cumsum(dlon)

    sst_base = 24 - 0.25 * (lat - 30)
    diurnal = 0.4 * np.sin(2 * np.pi * np.arange(n) / 24)
    sst = sst_base + diurnal

    sst[1] += 15
    sst[16] += 15
    sst[19] += 15

    return pd.DataFrame(
        {
            "ship_id": "ship_1",
            "date": datetime,
            "lat": lat,
            "lon": lon,
            "sst": sst,
        }
    )


def get_grouped_data():
    rng = np.random.default_rng(42)

    start = pd.Timestamp("2026-07-01 12:00")

    platforms = [
        ("ship_1", 45.000, -30.000),
        ("ship_2", 45.018, -29.982),
        ("ship_3", 44.992, -30.015),
        ("ship_4", 45.010, -30.008),
        ("ship_5", 45.006, -29.995),
    ]

    rows = []

    for _, (name, lat0, lon0) in enumerate(platforms):
        for hour in range(6):
            lat = lat0 + rng.normal(0, 0.003)
            lon = lon0 + rng.normal(0, 0.003)

            sst = 19.5 + 0.2 * np.sin(hour / 6 * 2 * np.pi) + rng.normal(0, 0.15)

            rows.append(
                dict(
                    platform=name,
                    date=start + pd.Timedelta(hours=hour),
                    lat=lat,
                    lon=lon,
                    sst=sst,
                )
            )

    df = pd.DataFrame(rows)

    # Add obvious bad observation
    df.loc[(df.platform == "ship_1") & (df.date == start + pd.Timedelta(hours=1)), "sst"] += 4.5
    df.loc[(df.platform == "ship_2") & (df.date == start + pd.Timedelta(hours=2)), "sst"] += 4.5
    df.loc[(df.platform == "ship_3") & (df.date == start + pd.Timedelta(hours=3)), "sst"] += 4.5
    df.loc[(df.platform == "ship_4") & (df.date == start + pd.Timedelta(hours=4)), "sst"] += 4.5
    df.loc[(df.platform == "ship_5") & (df.date == start + pd.Timedelta(hours=5)), "sst"] += 4.5

    return df


def get_buoy_data():
    buoy_id = "buoy_1"
    year = 2003
    month = 12
    day = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    hour = 0
    lat = [0.0, 1.0, 2.0, 5.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 19.0, 19.0, 19.0]
    lon = 0.0
    sst = [22.0, 21.6, 21.2, 20.8, 20.4, 20.0, 21.6, 21.2, 20.8, 20.4, 20.0, 19.6, 19.2, 18.8, 18.4, 16.0, 16.0, 16.0, 16.0]

    date = pd.DataFrame(
        {
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
        }
    )

    return pd.DataFrame(
        {
            "buoy_id": buoy_id,
            "date": pd.to_datetime(date),
            "lat": lat,
            "lon": lon,
            "sst": sst,
        }
    )


def get_climatology_data():
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

    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    # --- Land-sea mask ---

    shp = shpreader.natural_earth(
        resolution="110m",
        category="physical",
        name="land",
    )
    land = unary_union(list(shpreader.Reader(shp).geometries()))

    mask_2d = np.array(
        [land.contains(Point(x, y)) for x, y in zip(lon2d.ravel(), lat2d.ravel(), strict=False)],
        dtype="int8",
    ).reshape(lat.size, lon.size)

    mask_3d = mask_2d[np.newaxis, :, :]

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

    # --- Sea-surface temperature ---

    sst_2d = 28.0 * np.cos(np.deg2rad(lat2d)) ** 2
    sst_2d += 2.0 * np.sin(np.deg2rad(lon2d / 2.0)) * np.cos(np.deg2rad(lat2d))
    sst_2d = np.clip(sst_2d, -1.8, None)

    sst_2d = xr.where(ds.land_sea_mask.isel(time=0) == 0, sst_2d, np.nan)
    sst_3d = np.broadcast_to(sst_2d.values, (1, sst_2d.shape[0], sst_2d.shape[1]))

    ds["sst"] = (
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

    lat_factor = np.sin(np.deg2rad(np.abs(lat2d)))
    lon_factor = 0.5 * (1 + np.sin(np.deg2rad(lon2d)))

    # --- Standard deviations ---

    stdev1_2d = 0.25 + 0.20 * lat_factor + 0.05 * lon_factor

    stdev2_2d = 0.55 + 0.25 * lat_factor + 0.08 * lon_factor

    stdev3_2d = 0.12 + 0.10 * lat_factor + 0.03 * lon_factor

    stdev1_2d = xr.where(ds.land_sea_mask.isel(time=0) == 0, stdev1_2d, np.nan)
    stdev2_2d = xr.where(ds.land_sea_mask.isel(time=0) == 0, stdev2_2d, np.nan)
    stdev3_2d = xr.where(ds.land_sea_mask.isel(time=0) == 0, stdev3_2d, np.nan)

    stdev1_3d = np.broadcast_to(stdev1_2d.values, (1, *stdev1_2d.shape))
    stdev2_3d = np.broadcast_to(stdev2_2d.values, (1, *stdev2_2d.shape))
    stdev3_3d = np.broadcast_to(stdev3_2d.values, (1, *stdev3_2d.shape))

    for name, data, long_name in [
        (
            "sst_stdev1",
            stdev1_3d,
            "Standard deviation of grid cell minus neighbourhood mean",
        ),
        (
            "sst_stdev2",
            stdev2_3d,
            "Standard deviation of point observation minus grid cell mean",
        ),
        (
            "sst_stdev3",
            stdev3_3d,
            "Standard deviation of neighbourhood mean uncertainty",
        ),
    ]:
        ds[name] = (
            ("time", "latitude", "longitude"),
            data,
            {
                "units": "degC",
                "coordinates": "time latitude longitude",
                "grid_mapping": "crs",
                "long_name": long_name,
            },
        )

    # --- Information ---

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


def get_advanced_data():
    index = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U"]
    station_id = ["AA", "AA", "AA", "AA", "AA", "AA", "AA", "AA", "AA", "AA", "AA", "AA", "AB", "AA", "BB", "AA", "AA", "AA", "AA", "AA", "AA"]
    lon = [22.3, 29.7, 32.0, 8.1, -21.2, -21.2, 21.2, 29.7, 29.7, 32.0, 8.5, 8.15, 8.1, 8.05, 8.1, -21.4, -21.1, 29.7, 29.7, 29.7, 29.7]
    lat = [71.3, 71.3, 71.2, 66.0, 65.8, 65.8, -65.8, 71.3, 71.3, 71.2, 66.0, 66.05, 66.0, 65.95, 66.0, 65.6, 65.9, 71.3, 71.3, 71.3, 71.3]
    vsi = [0.0, 4.11552, np.nan, 0.0, 0.0, 0.0, 0.0, 4.11552, 4.11552, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.11552, 4.0, 4.11552, np.nan]
    dsi = [0.0, 315.0, np.nan, 0.0, 0.0, 0.0, 0.0, 315.0, 315.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 316.0, 315.0, np.nan, 315.0]
    date = pd.to_datetime(
        [
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:01:00",
            "2022-02-02 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
            "2022-02-01 00:00:00",
        ]
    )
    sst = [
        300.0,
        302.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
    ]
    return pd.DataFrame(
        {
            "station_id": station_id,
            "lon": lon,
            "lat": lat,
            "date": date,
            "vsi": vsi,
            "dsi": dsi,
            "sst": sst,
        },
        index=index,
    )
