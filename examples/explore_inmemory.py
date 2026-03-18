"""In-memory terrain exploration — the classic DataArray path.

Demonstrates the simplest way to use rtxpy: load a DEM as an xarray
DataArray and call ``dem.rtx.explore()``.  The terrain is held in GPU
memory as a single raster; the LOD system tiles it internally using
an InMemoryChunkSource (automatic, no zarr required).

This is the right path when:
- Your DEM fits comfortably in GPU memory (up to ~8K x 8K)
- You're working with a DataArray from rioxarray, xarray-spatial, etc.
- You want to place geometry interactively with place_buildings(), etc.

Usage:
    python explore_inmemory.py
    python explore_inmemory.py --dem copernicus --subsample 2

Requirements:
    pip install rtxpy[all]
"""

import argparse

import numpy as np
import xarray as xr  # noqa
import xrspatial  # noqa

import rtxpy  # noqa

# Birmingham, Alabama — rolling Appalachian foothills
BOUNDS = (-86.9, 33.4, -86.7, 33.6)


def _auto_utm(bounds):
    """Pick the correct UTM zone for a lon/lat bounding box."""
    west, south, east, north = bounds
    lon = (west + east) / 2
    lat = (south + north) / 2
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f'EPSG:{epsg}'


def main():
    parser = argparse.ArgumentParser(
        description="Explore terrain from an in-memory DataArray.")
    parser.add_argument("--dem", type=str, default="copernicus",
                        choices=["copernicus", "srtm", "usgs_10m"],
                        help="DEM source (default: copernicus)")
    parser.add_argument("--subsample", type=int, default=1,
                        help="Subsample factor (default: 1)")
    parser.add_argument("--bounds", type=float, nargs=4,
                        metavar=("W", "S", "E", "N"),
                        default=list(BOUNDS),
                        help="Bounding box in lon/lat degrees")
    args = parser.parse_args()

    bounds = tuple(args.bounds)
    crs = _auto_utm(bounds)

    # ---- Load DEM into memory as a DataArray ----------------------------
    print(f"Fetching {args.dem} DEM for {bounds} (CRS: {crs})...")
    dem = rtxpy.fetch_dem(
        bounds=bounds,
        output_path=f"explore_inmemory_{args.dem}.zarr",
        source=args.dem,
        crs=crs,
    )

    if args.subsample > 1:
        dem = dem[::args.subsample, ::args.subsample]

    # Ensure contiguous numpy then push to GPU
    dem.data = np.ascontiguousarray(dem.data)
    dem = dem.rtx.to_cupy()

    import cupy as cp
    emin = float(cp.nanmin(dem.data))
    emax = float(cp.nanmax(dem.data))
    print(f"Terrain: {dem.shape}, elevation {emin:.0f}–{emax:.0f} m")

    # ---- Explore --------------------------------------------------------
    # This is the classic path: a single DataArray in GPU memory.
    # Internally, the LOD system wraps it in an InMemoryChunkSource and
    # tiles it automatically.  No zarr, no chunk source, no streaming —
    # just a raster and a viewer.
    dem.rtx.explore(
        width=1600,
        height=1200,
        render_scale=0.5,
        color_stretch='cbrt',
        repl=True,
    )

    print("Done")


if __name__ == "__main__":
    main()
