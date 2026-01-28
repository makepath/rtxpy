"""Interactive playground for rtxpy viewshed and hillshade analysis.

This example demonstrates real-time viewshed and hillshade computation
using GPU-accelerated ray tracing. Click on the terrain to move the
viewshed origin point.

Requirements:
    pip install rtxpy[analysis] matplotlib xarray rioxarray requests
"""

import matplotlib.pyplot as plt
import numpy as np
import cupy
import xarray as xr
import requests
from pathlib import Path

from rtxpy import RTX, viewshed, hillshade


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
    bounds = (-122.3, 42.8, -121.9, 43.1)
    west, south, east, north = bounds

    # SRTM tiles needed (1x1 degree tiles named by SW corner)
    # For Crater Lake: n42w123 and n42w122
    tiles_needed = [
        ("n42", "w123"),
        ("n42", "w122"),
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
    terrain.data = terrain.data * 0.2

    # Ensure contiguous array before GPU transfer
    terrain.data = np.ascontiguousarray(terrain.data)

    # Get stats before GPU transfer
    elev_min = float(terrain.min())
    elev_max = float(terrain.max())

    # Convert to cupy for GPU processing
    terrain.data = cupy.asarray(terrain.data)

    print(f"Terrain loaded: {terrain.shape}, elevation range: "
          f"{elev_min:.0f}m to {elev_max:.0f}m (scaled)")

    return terrain


# Load terrain data (downloads if needed)
terrain = load_terrain()

# Initial sun azimuth for hillshade
azimuth = 225


def onclick(event):
    """Click handler for live adjustment of the viewshed origin."""
    ix, iy = event.xdata, event.ydata
    print(f'x = {ix}, y = {iy}')

    nix = ix / terrain.shape[1]
    niy = iy / terrain.shape[0]

    x_coords = terrain.indexes.get('x').values
    y_coords = terrain.indexes.get('y').values
    rangex = x_coords.max() - x_coords.min()
    rangey = y_coords.max() - y_coords.min()

    global vsw, vsh
    vsw = x_coords.min() + nix * rangex
    vsh = y_coords.max() - niy * rangey


def generate_hiking_path(x_coords, y_coords, num_points=360):
    """Generate a hiking path around Crater Lake (roughly circular).

    Creates a path that circles around the crater, simulating a hiker
    walking the rim trail.
    """
    # Center of the terrain
    cx = (x_coords.min() + x_coords.max()) / 2
    cy = (y_coords.min() + y_coords.max()) / 2

    # Radius for the hiking loop (about 1/3 of the terrain extent)
    rx = (x_coords.max() - x_coords.min()) * 0.25
    ry = (y_coords.max() - y_coords.min()) * 0.25

    # Generate circular path with some wobble for realism
    angles = np.linspace(0, 2 * np.pi, num_points)
    wobble = np.sin(angles * 8) * 0.1  # Small variations in radius

    path_x = cx + (rx + rx * wobble) * np.cos(angles)
    path_y = cy + (ry + ry * wobble) * np.sin(angles)

    return path_x, path_y


def coords_to_pixel(x, y, x_coords, y_coords):
    """Convert data coordinates to pixel coordinates for plotting."""
    # Find nearest pixel indices
    px = np.searchsorted(x_coords, x)
    py = np.searchsorted(-y_coords, -y)  # y_coords are typically descending
    return int(np.clip(px, 0, len(x_coords) - 1)), int(np.clip(py, 0, len(y_coords) - 1))


def draw_observer_marker(colors, px, py, radius=8):
    """Draw a bright marker at the observer's position."""
    H, W = colors.shape[:2]

    # Draw a bright yellow/white circle for the observer
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:
                ny, nx = py + dy, px + dx
                if 0 <= ny < H and 0 <= nx < W:
                    # Outer ring (yellow)
                    if dx*dx + dy*dy > (radius-2)*(radius-2):
                        colors[ny, nx] = [255, 255, 0]
                    # Inner circle (white)
                    else:
                        colors[ny, nx] = [255, 255, 255]


def run_playground():
    """Run the interactive viewshed/hillshade playground with hiking animation."""
    import time

    runs = 360
    H, W = terrain.data.shape

    # Create RTX instance for reuse across frames
    rtx = RTX()

    # Set up the figure with title
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.canvas.mpl_connect('button_press_event', onclick)

    colors = np.uint8(np.zeros((H, W, 3)))
    imgplot = ax.imshow(colors)
    ax.set_title("Crater Lake Viewshed - Hiker's View\n(Click to move observer)", fontsize=14)
    ax.axis('off')

    x_coords = terrain.indexes.get('x').values
    y_coords = terrain.indexes.get('y').values

    # Generate the hiking path
    path_x, path_y = generate_hiking_path(x_coords, y_coords, num_points=runs)

    global vsw, vsh
    global azimuth

    # Start at first point on hiking path
    vsw = path_x[0]
    vsh = path_y[0]

    print(f"Starting hiking simulation around Crater Lake...")
    print(f"  Terrain size: {H}x{W}")
    print(f"  Frames: {runs}")
    print(f"  Click anywhere to teleport the observer!\n")

    for i in range(runs):
        frame_start = time.time()

        # Update observer position along hiking path
        vsw = path_x[i]
        vsh = path_y[i]

        # Slowly rotate sun for dynamic lighting
        azimuth = (azimuth + 1) % 360

        # Compute hillshade and viewshed
        rt_start = time.time()
        hs = hillshade(terrain,
                       shadows=True,
                       azimuth=azimuth,
                       angle_altitude=25,
                       rtx=rtx)
        vs = viewshed(terrain,
                      x=vsw,
                      y=vsh,
                      observer_elev=2.0,  # 2 meter observer height (standing hiker)
                      rtx=rtx)
        rt_time = time.time() - rt_start

        # Convert hillshade to grayscale image
        hs_data = hs.data.get() if hasattr(hs.data, 'get') else hs.data
        vs_data = vs.data.get() if hasattr(vs.data, 'get') else vs.data

        # Track NaN pixels - these will be shown as black
        nan_mask = np.isnan(hs_data) | np.isnan(vs_data)

        hs_data = np.nan_to_num(hs_data, nan=0.0)
        gray = np.uint8(np.clip(hs_data * 200, 0, 255))

        # Create visibility mask
        visible_mask = vs_data > 0

        # Compose the final image:
        # - Base: grayscale hillshade
        # - Visible areas: tinted green
        # - Non-visible areas: tinted blue/darker
        colors[:, :, 0] = gray  # Red channel
        colors[:, :, 1] = gray  # Green channel
        colors[:, :, 2] = gray  # Blue channel

        # Tint visible areas green
        colors[visible_mask, 1] = np.minimum(255, colors[visible_mask, 1] + 60)

        # Tint non-visible areas slightly blue (shadows)
        colors[~visible_mask, 2] = np.minimum(255, colors[~visible_mask, 2] + 40)
        colors[~visible_mask, 0] = colors[~visible_mask, 0] // 2
        colors[~visible_mask, 1] = colors[~visible_mask, 1] // 2

        # Make NaN pixels black (no data)
        colors[nan_mask] = [0, 0, 0]

        # Draw observer marker
        px, py = coords_to_pixel(vsw, vsh, x_coords, y_coords)
        draw_observer_marker(colors, px, py, radius=6)

        # Update display
        imgplot.set_data(colors)
        ax.set_title(f"Crater Lake Viewshed - Hiker Position {i+1}/{runs}\n"
                     f"RT: {rt_time*1000:.1f}ms | Sun azimuth: {azimuth}°", fontsize=12)

        plt.pause(0.001)

        # Print progress every 30 frames
        if i % 30 == 0:
            fps = 1.0 / (time.time() - frame_start) if (time.time() - frame_start) > 0 else 0
            print(f"Frame {i:3d}/{runs} | RT: {rt_time*1000:5.1f}ms | ~{fps:.1f} FPS")

    print("\nHiking complete! Close the window to exit.")
    plt.show()


if __name__ == "__main__":
    run_playground()
    print("Done")
