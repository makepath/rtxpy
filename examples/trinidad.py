"""Interactive playground for the island of Trinidad.

Explore the terrain of Trinidad using GPU-accelerated ray tracing.
Elevation data is sourced from the Copernicus GLO-30 DEM (30 m).

Supports two modes:
  python trinidad.py           # single-layer elevation explorer
  python trinidad.py --dataset # multi-layer Dataset with landcover overlay

In Dataset mode, press G to cycle between elevation and landcover coloring.

Requirements:
    pip install rtxpy[analysis] matplotlib xarray rioxarray requests
"""

import numpy as np
import requests
import xarray as xr

from xrspatial import slope, aspect, quantile, curvature
from pathlib import Path

# Import rtxpy to register the .rtx accessor
import rtxpy


def download_trinidad_dem(output_path):
    """Download Copernicus GLO-30 DEM tiles for Trinidad and merge.

    Downloads 30 m resolution tiles from the Copernicus DEM hosted on
    AWS S3 (no authentication required) and clips to the island extent.

    Parameters
    ----------
    output_path : Path
        Path to save the merged/clipped DEM file.

    Returns
    -------
    xarray.DataArray
        The elevation data as an xarray DataArray.
    """
    import rioxarray as rxr

    # Trinidad bounding box (WGS84)
    # Northern Range peaks at ~940 m (El Cerro del Aripo)
    bounds = (-61.95, 10.04, -60.85, 10.85)
    west, south, east, north = bounds

    # Copernicus GLO-30 tiles needed (1x1 degree, named by SW corner)
    # Trinidad spans two tiles: W062 (62-61°W) and W061 (61-60°W)
    tiles_needed = [
        ("N10", "W062"),
        ("N10", "W061"),
    ]

    base_url = "https://copernicus-dem-30m.s3.amazonaws.com"
    tile_paths = []

    print("Downloading Trinidad elevation data from Copernicus GLO-30 DEM...")

    for ns, ew in tiles_needed:
        tile_name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
        url = f"{base_url}/{tile_name}/{tile_name}.tif"
        tile_path = output_path.parent / f"{tile_name}.tif"

        if not tile_path.exists():
            print(f"  Downloading {ns}_{ew}...")
            try:
                resp = requests.get(url, timeout=120, stream=True)
                resp.raise_for_status()
                with open(tile_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            except requests.RequestException as e:
                print(f"  Warning: Failed to download {ns}_{ew}: {e}")
                continue
        else:
            print(f"  Using cached {ns}_{ew}")

        tile_paths.append(tile_path)

    if not tile_paths:
        raise RuntimeError("Failed to download any elevation tiles")

    # Open all tiles
    tiles = [rxr.open_rasterio(str(p), masked=True).squeeze() for p in tile_paths]

    if len(tiles) == 1:
        merged = tiles[0]
    else:
        from rioxarray.merge import merge_arrays
        merged = merge_arrays(tiles)

    # Clip to Trinidad bounds
    merged = merged.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north)

    # Reproject to UTM Zone 20N (appropriate for Trinidad at ~61°W)
    merged = merged.rio.reproject("EPSG:32620")

    # Save clipped result
    merged.rio.to_raster(str(output_path))
    print(f"  Saved clipped DEM to {output_path}")

    return merged


def load_terrain():
    """Load Trinidad terrain data, downloading if necessary."""
    dem_path = Path(__file__).parent / "trinidad_dem.tif"

    if not dem_path.exists():
        print(f"DEM file not found at {dem_path}")
        terrain = download_trinidad_dem(dem_path)
    else:
        print(f"Loading existing DEM: {dem_path}")
        import rioxarray as rxr
        terrain = rxr.open_rasterio(str(dem_path), masked=True).squeeze()

    # Subsample for faster interactive performance
    terrain = terrain[::4, ::4]

    # Mask ocean / water pixels so they are not rendered
    terrain = terrain.where(terrain > 0)

    # Scale down elevation for visualization (optional)
    terrain.data = terrain.data * 0.025

    # Ensure contiguous array before GPU transfer
    terrain.data = np.ascontiguousarray(terrain.data)

    # Get stats before GPU transfer (nanmin/nanmax to skip NaN ocean pixels)
    elev_min = float(np.nanmin(terrain.data))
    elev_max = float(np.nanmax(terrain.data))

    # Convert to cupy for GPU processing using the accessor
    terrain = terrain.rtx.to_cupy()

    print(f"Terrain loaded: {terrain.shape}, elevation range: "
          f"{elev_min:.0f}m to {elev_max:.0f}m (scaled)")

    return terrain


def make_landcover(terrain):
    """Derive a synthetic tropical landcover classification from elevation.

    Classes (by Z-code):
        100  Water / coast        — lowest 3 % of elevation
        400  Mangrove / wetland   — low-lying flat areas (low elev, gentle slope)
        800  Tropical lowland     — low-to-mid elevation, moderate slope
       1200  Tropical forest      — mid-to-upper elevation, moderate slope
       1800  Cloud forest         — highest 5 % of elevation

    Parameters
    ----------
    terrain : xarray.DataArray
        Elevation data (numpy or cupy).

    Returns
    -------
    numpy.ndarray
        2-D float32 array of landcover class codes, same shape as terrain.
    """
    data = terrain.data
    if hasattr(data, 'get'):
        data = data.get()
    else:
        data = np.asarray(data)

    dy, dx = np.gradient(data)
    slope = np.sqrt(dx**2 + dy**2)

    p3  = np.nanpercentile(data, 3)
    p15 = np.nanpercentile(data, 15)
    p50 = np.nanpercentile(data, 50)
    p95 = np.nanpercentile(data, 95)
    sp30 = np.nanpercentile(slope, 30)

    lc = np.full(data.shape, 800.0, dtype=np.float32)    # tropical lowland
    lc[data <= p3]                       = 100.0   # water / coast
    lc[(data > p3) & (data <= p15) &
       (slope <= sp30)]                  = 400.0   # mangrove / wetland
    lc[(data > p50) & (data < p95)]      = 1200.0  # tropical forest
    lc[data >= p95]                      = 1800.0  # cloud forest

    unique, counts = np.unique(lc, return_counts=True)
    names = {100: 'Water/coast', 400: 'Mangrove', 800: 'Tropical low',
             1200: 'Trop. forest', 1800: 'Cloud forest'}
    total = lc.size
    print("Landcover classes:")
    for val, cnt in zip(unique, counts):
        print(f"  {names.get(val, val):14s} ({val:5.0f}): {100*cnt/total:5.1f}%")

    return lc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="rtxpy Trinidad terrain explorer")
    parser.add_argument("--dataset", action="store_true",
                        help="Build an xr.Dataset with landcover overlay (G to cycle)")
    args = parser.parse_args()

    # Load terrain data (downloads if needed)
    terrain = load_terrain()

    print("\nControls:")
    print("  W/S/A/D or Arrow keys: Move camera")
    print("  Q/E or Page Up/Down: Move up/down")
    print("  I/J/K/L: Look around")
    print("  +/-: Adjust movement speed")
    print("  G: Cycle overlay layers" if args.dataset else "  G: Cycle geometry layers")
    print("  O: Place observer (for viewshed)")
    print("  V: Toggle viewshed (teal glow)")
    print("  [/]: Adjust observer height")
    print("  T: Toggle shadows")
    print("  C: Cycle colormap")
    print("  F: Screenshot")
    print("  H: Toggle help overlay")
    print("  X: Exit\n")

    # Camera: south edge looking north
    H, W = terrain.shape
    elev_data = terrain.data
    if hasattr(elev_data, 'get'):
        elev_np = elev_data.get()
    else:
        elev_np = np.asarray(elev_data)
    elev_max = float(np.nanmax(elev_np))
    elev_mean = float(np.nanmean(elev_np))
    diag = np.sqrt(H**2 + W**2)
    start_pos = (W / 2, H * 1.05, elev_max + diag * 0.08)
    look_target = (W / 2, H / 2, elev_mean)

    if args.dataset:
        import cupy as cp

        print("Building Dataset with landcover overlay...")
        lc = make_landcover(terrain)

        coords = {d: terrain.coords[d] for d in terrain.dims}

        ds = xr.Dataset({
            'elevation': terrain.rename(None),
            'landcover': xr.DataArray(cp.asarray(lc), dims=terrain.dims, coords=coords),
            'slope':  slope(terrain),
            'aspect': aspect(terrain),
            'quantile':  quantile(terrain),
        })
        print(ds)
        print("\nLaunching Dataset explore (press G to cycle elevation <-> landcover)...\n")
        ds.rtx.explore(
            z='elevation',
            width=1024,
            height=768,
            render_scale=0.5,
            color_stretch='cbrt',
            start_position=start_pos,
            look_at=look_target,
        )
    else:
        print("Launching explore mode...\n")
        terrain.rtx.explore(
            width=1024,
            height=768,
            render_scale=0.5,
            color_stretch='cbrt',
            start_position=start_pos,
            look_at=look_target,
        )

    print("Done")
