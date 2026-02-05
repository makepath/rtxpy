"""Interactive playground for rtxpy terrain exploration.

This example demonstrates real-time terrain exploration using
GPU-accelerated ray tracing via the xarray accessor's explore mode.

Requirements:
    pip install rtxpy[analysis] matplotlib xarray rioxarray requests
"""

import numpy as np
import requests
from pathlib import Path

# Import rtxpy to register the .rtx accessor
import rtxpy


def download_crater_lake_dem(output_path):
    """Download SRTM elevation data for Crater Lake National Park.

    Downloads 1-arc-second (~30m) SRTM tiles from the USGS National Map
    and clips to the Crater Lake area.

    Parameters
    ----------
    output_path : Path
        Path to save the downloaded/processed DEM file.

    Returns
    -------
    xarray.DataArray
        The elevation data as an xarray DataArray.
    """
    import rioxarray as rxr

    # Crater Lake National Park bounds (WGS84)
    # The park is centered around 42.94°N, 122.10°W
    # Note: north bound limited to 43.0 to match n42 SRTM tile coverage
    bounds = (-122.3, 42.8, -121.9, 43.0)
    west, south, east, north = bounds

    # SRTM tiles needed (1x1 degree tiles, named by northern latitude boundary)
    # For Crater Lake at ~42.94°N: n43 tiles cover lat 42-43
    tiles_needed = [
        ("n43", "w123"),
        ("n43", "w122"),
    ]

    base_url = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/current"
    tile_paths = []

    print("Downloading Crater Lake elevation data from USGS...")

    for ns, ew in tiles_needed:
        tile_name = f"{ns}{ew}"
        url = f"{base_url}/{tile_name}/USGS_1_{tile_name}.tif"
        tile_path = output_path.parent / f"USGS_1_{tile_name}.tif"

        if not tile_path.exists():
            print(f"  Downloading {tile_name}...")
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                tile_path.write_bytes(resp.content)
            except requests.RequestException as e:
                print(f"  Warning: Failed to download {tile_name}: {e}")
                continue
        else:
            print(f"  Using cached {tile_name}")

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

    # Clip to Crater Lake bounds
    merged = merged.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north)

    # Reproject to EPSG:5070 (Conus Albers Equal Area) for proper metric units
    merged = merged.rio.reproject("EPSG:5070")

    # Save clipped result
    merged.rio.to_raster(str(output_path))
    print(f"  Saved clipped DEM to {output_path}")

    # Clean up individual tiles (optional - keep them for caching)
    # for p in tile_paths:
    #     if p != output_path and p.exists():
    #         p.unlink()

    return merged


def load_terrain():
    """Load Crater Lake terrain data, downloading if necessary."""
    dem_path = Path(__file__).parent / "crater_lake_national_park.tif"

    if not dem_path.exists():
        print(f"DEM file not found at {dem_path}")
        terrain = download_crater_lake_dem(dem_path)
    else:
        print(f"Loading existing DEM: {dem_path}")
        import rioxarray as rxr
        terrain = rxr.open_rasterio(str(dem_path), masked=True).squeeze()

    # Subsample for faster interactive performance
    terrain = terrain[::2, ::2]

    # Scale down elevation for visualization (optional)
    terrain.data = terrain.data * 0.1

    # Ensure contiguous array before GPU transfer
    terrain.data = np.ascontiguousarray(terrain.data)

    # Get stats before GPU transfer
    elev_min = float(terrain.min())
    elev_max = float(terrain.max())

    # Convert to cupy for GPU processing using the accessor
    terrain = terrain.rtx.to_cupy()

    print(f"Terrain loaded: {terrain.shape}, elevation range: "
          f"{elev_min:.0f}m to {elev_max:.0f}m (scaled)")

    return terrain


if __name__ == "__main__":
    # Load terrain data (downloads if needed)
    terrain = load_terrain()

    print("\nLaunching explore mode...")
    print("Controls:")
    print("  W/S/A/D or Arrow keys: Move camera")
    print("  Q/E or Page Up/Down: Move up/down")
    print("  I/J/K/L: Look around")
    print("  +/-: Adjust movement speed")
    print("  O: Place observer (for viewshed)")
    print("  V: Toggle viewshed (teal glow)")
    print("  [/]: Adjust observer height")
    print("  T: Toggle shadows")
    print("  C: Cycle colormap")
    print("  F: Screenshot")
    print("  H: Toggle help overlay")
    print("  X: Exit\n")

    # Launch interactive explore mode
    terrain.rtx.explore(
        mesh_type='voxelate',
        width=1024,
        height=768,
        render_scale=0.5
    )
    print("Done")
