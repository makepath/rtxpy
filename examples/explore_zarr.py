"""Explore a massive zarr DEM interactively by loading only a windowed region.

Opens the USGS 10 m continental DEM stored as a zarr archive and loads a
small window around a chosen center point.  The window is subsampled,
pushed to GPU, and launched in the rtxpy interactive viewer.

The zarr store is never fully read — only the chunks that overlap the
requested window are fetched from disk.

For zarr v3 stores with GPU-compatible codecs (e.g. ZstdCodec), chunks
are decompressed directly on the GPU via ``zarr.config.enable_gpu()``.
For zarr v2 stores (blosc/numcodecs), chunks are read on CPU and then
transferred to GPU with ``cupy.asarray()``.

Usage:
    # Default: centers on Birmingham, Alabama
    python explore_zarr.py

    # Custom center (lon, lat) and window size in degrees
    python explore_zarr.py --lon -118.25 --lat 34.08 --size 0.25

    # Adjust subsample factor (higher = coarser but faster)
    python explore_zarr.py --subsample 8

Requirements:
    pip install rtxpy[all] xarray zarr cupy
"""

import argparse
import time

import cupy as cp
import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr
import zarr

import rtxpy


ZARR_PATH = "/home/brendan/elevation/usgs10m_dem_c6.zarr"


def _open_zarr_array(zarr_path):
    """Open the zarr store, enabling GPU-direct reads when possible.

    Returns (zarr.Array, zarr_format_version, GeoTransform tuple, CF encoding, CRS WKT).
    """
    root = zarr.open_group(zarr_path, mode="r")
    z = root["usgs10m_dem"]

    # Parse GeoTransform: "x_origin dx rot_x y_origin rot_y dy"
    gt = root["spatial_ref"].attrs["GeoTransform"]
    parts = [float(v) for v in gt.split()]
    x_origin, dx, _, y_origin, _, dy = parts

    # Read CRS for attaching to the DataArray
    crs_wkt = root["spatial_ref"].attrs.get("crs_wkt", None)

    # CF encoding: int32 centimeters → float metres
    scale_factor = float(z.attrs.get("scale_factor", 1.0))
    add_offset = float(z.attrs.get("add_offset", 0.0))
    fill_value = z.fill_value

    fmt = z.metadata.zarr_format if hasattr(z.metadata, "zarr_format") else 2

    # Enable GPU-direct reads for zarr v3 stores with GPU-native codecs
    if fmt >= 3:
        zarr.config.enable_gpu()
        # Re-open so the GPU prototype is active
        root = zarr.open_group(zarr_path, mode="r")
        z = root["usgs10m_dem"]
        print(f"Zarr v{fmt} — GPU-direct reads enabled")
    else:
        print(f"Zarr v{fmt} (blosc) — CPU read + GPU transfer")

    return z, fmt, (x_origin, dx, y_origin, dy), (scale_factor, add_offset, fill_value), crs_wkt


def _utm_epsg(lon, lat):
    """Return the UTM EPSG code for a given lon/lat."""
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def load_window(zarr_path, center_lon, center_lat, size_deg, subsample):
    """Load a subsampled elevation window from the zarr DEM.

    Reads the geographic-CRS chunk from disk, applies CF decoding, then
    reprojects to the local UTM zone so coordinates are in metres.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr store.
    center_lon, center_lat : float
        Center of the window in WGS84 degrees.
    size_deg : float
        Half-width of the window in degrees (window is 2*size_deg on a side).
    subsample : int
        Take every Nth pixel to reduce resolution.

    Returns
    -------
    xarray.DataArray
        Elevation window on GPU (cupy-backed), with projected metric coords.
    """
    z, fmt, (x_origin, dx, y_origin, dy), (scale, offset, fill), crs_wkt = \
        _open_zarr_array(zarr_path)

    H, W = z.shape
    lon_min = center_lon - size_deg
    lon_max = center_lon + size_deg
    lat_min = center_lat - size_deg
    lat_max = center_lat + size_deg

    # Pixel indices from GeoTransform (x ascending, y descending)
    xi0 = max(int((lon_min - x_origin) / dx), 0)
    xi1 = min(int((lon_max - x_origin) / dx) + 1, W)
    yi0 = max(int((lat_max - y_origin) / dy), 0)   # dy is negative
    yi1 = min(int((lat_min - y_origin) / dy) + 1, H)

    raw_w = xi1 - xi0
    raw_h = yi1 - yi0
    out_w = raw_w // subsample
    out_h = raw_h // subsample
    print(f"Window: lon [{lon_min:.3f}, {lon_max:.3f}], "
          f"lat [{lat_min:.3f}, {lat_max:.3f}]")
    print(f"Pixel window: {raw_w:,} x {raw_h:,} "
          f"(subsample {subsample}x -> {out_w:,} x {out_h:,})")

    # Read only the chunks that overlap the window
    t0 = time.time()
    data = z[yi0:yi1:subsample, xi0:xi1:subsample]
    dt_read = time.time() - t0

    # Ensure numpy for CF decoding + reprojection
    if not isinstance(data, np.ndarray):
        data = cp.asnumpy(data)
    data = data.astype(np.float32)
    data[data == fill] = np.nan
    data = data * scale + offset
    print(f"Read: {dt_read:.2f}s")

    # Build geographic DataArray with source CRS
    x_coords = np.arange(xi0, xi1, subsample) * dx + x_origin
    y_coords = np.arange(yi0, yi1, subsample) * dy + y_origin

    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords},
    )
    if crs_wkt is not None:
        da = da.rio.write_crs(crs_wkt)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    # Reproject to local UTM so coordinates are in metres
    target_epsg = _utm_epsg(center_lon, center_lat)
    t1 = time.time()
    da = da.rio.reproject(f"EPSG:{target_epsg}")
    dt_proj = time.time() - t1
    print(f"Reprojected to EPSG:{target_epsg} in {dt_proj:.2f}s")

    # Transfer to GPU
    cpu_data = np.ascontiguousarray(da.values)
    gpu_data = cp.asarray(cpu_data)
    out = da.copy(data=gpu_data)

    elev_min = float(cp.nanmin(gpu_data))
    elev_max = float(cp.nanmax(gpu_data))
    print(f"Shape: {out.shape}, {gpu_data.nbytes / 1e6:.1f} MB, "
          f"elevation: {elev_min:.0f}m to {elev_max:.0f}m")

    return out


def make_terrain_loader(zarr_path, size_deg, subsample, center_lon, center_lat):
    """Create a terrain loader callback for dynamic chunk streaming.

    The viewer passes the camera position in the DataArray's CRS (UTM
    metres after reprojection).  This closure inverse-projects back to
    geographic lon/lat before fetching a new window from the zarr store.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr store.
    size_deg : float
        Half-width of the window in degrees.
    subsample : int
        Subsample factor.
    center_lon, center_lat : float
        Initial center in WGS84 degrees (used to pick the UTM zone for
        the inverse projection).

    Returns
    -------
    callable
        ``loader(utm_x, utm_y) -> xr.DataArray | None``
    """
    from pyproj import Transformer

    target_epsg = _utm_epsg(center_lon, center_lat)
    to_lonlat = Transformer.from_crs(
        f"EPSG:{target_epsg}", "EPSG:4326", always_xy=True,
    )

    def loader(cam_x, cam_y):
        try:
            lon, lat = to_lonlat.transform(cam_x, cam_y)
            return load_window(zarr_path, lon, lat, size_deg, subsample)
        except Exception as e:
            print(f"Terrain loader error: {e}")
            return None

    return loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore a large zarr DEM interactively."
    )
    parser.add_argument("--lon", type=float, default=-86.8,
                        help="Center longitude (default: -86.8, Birmingham AL)")
    parser.add_argument("--lat", type=float, default=33.5,
                        help="Center latitude (default: 33.5, Birmingham AL)")
    parser.add_argument("--size", type=float, default=0.25,
                        help="Half-width of window in degrees (default: 0.25)")
    parser.add_argument("--subsample", type=int, default=4,
                        help="Subsample factor (default: 4)")
    parser.add_argument("--zarr", type=str, default=ZARR_PATH,
                        help="Path to zarr store")
    args = parser.parse_args()

    terrain = load_window(
        args.zarr, args.lon, args.lat, args.size, args.subsample,
    )

    loader = make_terrain_loader(args.zarr, args.size, args.subsample, args.lon, args.lat)

    print(f"\nLaunching explore...\n")
    terrain.rtx.explore(
        width=2048,
        height=1600,
        render_scale=0.5,
        color_stretch='cbrt',
        terrain_loader=loader,
        repl=True,
    )

    print("Done")
