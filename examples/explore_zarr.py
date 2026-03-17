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


def _build_lod_arrays(zarr_path, array_name, max_factor=64):
    """Build cached box-filter downsampled LOD arrays in the zarr store.

    Creates arrays named ``{array_name}_lod{factor}`` for each power-of-2
    factor from 2 up to ``max_factor``.  Each level is built from the
    previous one, so the full-res array is only read once.

    This is a one-time cost — subsequent runs skip existing arrays.
    """
    import warnings

    root = zarr.open_group(zarr_path, mode="r+")
    z = root[array_name]

    scale = float(z.attrs.get("scale_factor", 1.0))
    add_offset = float(z.attrs.get("add_offset", 0.0))
    fill_value = z.fill_value

    factor = 2
    prev_name = array_name
    while factor <= max_factor:
        lod_name = f"{array_name}_lod{factor}"
        if lod_name in root:
            prev_name = lod_name
            factor *= 2
            continue

        prev = root[prev_name]
        pH, pW = prev.shape
        oH, oW = pH // 2, pW // 2
        if oH < 2 or oW < 2:
            break

        chunk_h = min(512, oH)
        chunk_w = min(512, oW)
        out = root.create_array(
            lod_name,
            shape=(oH, oW),
            chunks=(chunk_h, chunk_w),
            dtype=np.float32,
            fill_value=np.nan,
            overwrite=True,
        )

        is_source = (prev_name == array_name)
        BLOCK = 512  # output rows per processing pass
        t0 = time.time()
        for out_r0 in range(0, oH, BLOCK):
            out_r1 = min(out_r0 + BLOCK, oH)
            in_r0 = out_r0 * 2
            in_r1 = min(out_r1 * 2, pH)

            block = np.array(prev[in_r0:in_r1, :], dtype=np.float32)
            if is_source:
                block[block == fill_value] = np.nan
                block = block * scale + add_offset

            # Box-filter 2×2
            hh = block.shape[0] // 2 * 2
            ww = block.shape[1] // 2 * 2
            if hh == 0 or ww == 0:
                break
            block2 = block[:hh, :ww].reshape(hh // 2, 2, ww // 2, 2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ds = np.nanmean(block2, axis=(1, 3)).astype(np.float32)
            out[out_r0:out_r0 + ds.shape[0], :ds.shape[1]] = ds

        dt = time.time() - t0
        print(f"  Built {lod_name}: {oH:,}×{oW:,} ({dt:.1f}s)")

        prev_name = lod_name
        factor *= 2

    return root


def _get_lod_array(root, array_name, subsample):
    """Find the best cached LOD array for a given subsample factor.

    Returns ``(zarr_array, remaining_stride, needs_cf_decode)`` where
    ``remaining_stride`` is the additional stride to apply after reading.
    """
    # Find the largest LOD factor ≤ subsample
    best_factor = 1
    best_name = array_name
    factor = 2
    while factor <= subsample:
        lod_name = f"{array_name}_lod{factor}"
        if lod_name in root:
            best_factor = factor
            best_name = lod_name
        factor *= 2

    remaining = subsample // best_factor
    needs_cf = (best_name == array_name)
    return root[best_name], remaining, needs_cf, best_factor


def _read_raw_window(z, yi0, yi1, xi0, xi1, subsample, scale, offset, fill,
                     lod_root=None, array_name=None):
    """Read a raw pixel window from the zarr array and CF-decode it.

    When ``lod_root`` is provided, reads from the best cached LOD array
    for the requested subsample factor instead of striding the full-res.
    """
    if lod_root is not None and array_name is not None and subsample > 1:
        lod_arr, stride, needs_cf, lod_factor = _get_lod_array(
            lod_root, array_name, subsample)
        # Map full-res pixel coords to LOD array coords
        ly0 = yi0 // lod_factor
        ly1 = max(ly0 + 1, yi1 // lod_factor)
        lx0 = xi0 // lod_factor
        lx1 = max(lx0 + 1, xi1 // lod_factor)
        # Clamp to array bounds
        lH, lW = lod_arr.shape
        ly0 = max(0, min(ly0, lH - 1))
        ly1 = min(ly1, lH)
        lx0 = max(0, min(lx0, lW - 1))
        lx1 = min(lx1, lW)
        data = lod_arr[ly0:ly1:stride, lx0:lx1:stride]
        if not isinstance(data, np.ndarray):
            data = cp.asnumpy(data)
        data = data.astype(np.float32)
        if needs_cf:
            data[data == fill] = np.nan
            data = data * scale + offset
        return data

    # Fallback: stride the full-res array directly
    data = z[yi0:yi1:subsample, xi0:xi1:subsample]
    if not isinstance(data, np.ndarray):
        data = cp.asnumpy(data)
    data = data.astype(np.float32)
    data[data == fill] = np.nan
    data = data * scale + offset
    return data


def load_window(zarr_path, center_lon, center_lat, size_deg, subsample,
                horizon_rings=0, horizon_multiplier=4):
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
    horizon_rings : int
        Number of concentric low-res rings to load around the main window.
        Each ring doubles the footprint at ``horizon_multiplier`` times the
        subsample of the previous ring.  0 = no horizon (original behavior).
    horizon_multiplier : int
        Subsample multiplier per ring (default 4: each ring is 4x coarser).

    Returns
    -------
    xarray.DataArray
        Elevation window on GPU (cupy-backed), with projected metric coords.
    """
    z, fmt, (x_origin, dx, y_origin, dy), (scale, offset, fill), crs_wkt = \
        _open_zarr_array(zarr_path)

    ARRAY_NAME = "usgs10m_dem"
    H, W = z.shape

    # Use cached LOD arrays for fast horizon reads when available.
    # If they don't exist, _read_raw_window falls back to strided reads.
    # Run with --build-lods to pre-build the full LOD cache explicitly.
    lod_root = None
    if horizon_rings > 0:
        lod_root = zarr.open_group(zarr_path, mode="r")
        # Check if any LOD arrays exist
        has_lods = any(f"{ARRAY_NAME}_lod{2**k}" in lod_root
                       for k in range(1, 7))
        if has_lods:
            print("Using cached LOD arrays for horizon reads")
        else:
            print("No LOD cache found — using strided reads "
                  "(run with --build-lods to pre-build)")

    if horizon_rings > 0:
        # Build a composite: coarse wide canvas with high-res center.
        # Work outward from the widest ring, then paste finer rings on top.
        s = subsample
        sz = size_deg
        for _ in range(horizon_rings):
            s *= horizon_multiplier
            sz *= 2
        # Outermost ring defines the canvas
        canvas_sub = s
        rings_spec = []
        cur_s, cur_sz = canvas_sub, sz
        while cur_s >= subsample:
            rings_spec.append((cur_sz, cur_s))
            cur_s //= horizon_multiplier
            cur_sz /= 2

        # Read outermost (coarsest) ring as the canvas
        outer_sz, outer_sub = rings_spec[0]
        lon_min_c = center_lon - outer_sz
        lon_max_c = center_lon + outer_sz
        lat_min_c = center_lat - outer_sz
        lat_max_c = center_lat + outer_sz
        xi0_c = max(int((lon_min_c - x_origin) / dx), 0)
        xi1_c = min(int((lon_max_c - x_origin) / dx) + 1, W)
        yi0_c = max(int((lat_max_c - y_origin) / dy), 0)
        yi1_c = min(int((lat_min_c - y_origin) / dy) + 1, H)

        print(f"Horizon: {len(rings_spec)} rings, "
              f"outermost {outer_sz:.3f}° at {outer_sub}x subsample")
        t0 = time.time()
        canvas = _read_raw_window(z, yi0_c, yi1_c, xi0_c, xi1_c,
                                  outer_sub, scale, offset, fill,
                                  lod_root=lod_root, array_name=ARRAY_NAME)
        # Canvas geo coords
        x_coords_c = np.arange(xi0_c, xi1_c, outer_sub) * dx + x_origin
        y_coords_c = np.arange(yi0_c, yi1_c, outer_sub) * dy + y_origin

        # Paste inner rings (each finer, smaller footprint)
        for ring_sz, ring_sub in rings_spec[1:]:
            lon_min_r = center_lon - ring_sz
            lon_max_r = center_lon + ring_sz
            lat_min_r = center_lat - ring_sz
            lat_max_r = center_lat + ring_sz
            xi0_r = max(int((lon_min_r - x_origin) / dx), 0)
            xi1_r = min(int((lon_max_r - x_origin) / dx) + 1, W)
            yi0_r = max(int((lat_max_r - y_origin) / dy), 0)
            yi1_r = min(int((lat_min_r - y_origin) / dy) + 1, H)

            ring_data = _read_raw_window(z, yi0_r, yi1_r, xi0_r, xi1_r,
                                         ring_sub, scale, offset, fill,
                                         lod_root=lod_root,
                                         array_name=ARRAY_NAME)
            ring_x = np.arange(xi0_r, xi1_r, ring_sub) * dx + x_origin
            ring_y = np.arange(yi0_r, yi1_r, ring_sub) * dy + y_origin

            # Upsample ring_data to canvas resolution via nearest-neighbor
            # and paste into the canvas where it overlaps.
            ratio = outer_sub // ring_sub
            cx0 = np.searchsorted(x_coords_c, ring_x[0])
            cy0 = np.searchsorted(-y_coords_c, -ring_y[0])  # y descending
            # Upsample: repeat each ring pixel ratio×ratio times
            upsampled = np.repeat(np.repeat(ring_data, ratio, axis=0),
                                  ratio, axis=1)
            # Clip to canvas bounds
            paste_h = min(upsampled.shape[0], canvas.shape[0] - cy0)
            paste_w = min(upsampled.shape[1], canvas.shape[1] - cx0)
            if paste_h > 0 and paste_w > 0:
                canvas[cy0:cy0 + paste_h, cx0:cx0 + paste_w] = \
                    upsampled[:paste_h, :paste_w]

        dt_read = time.time() - t0
        print(f"Horizon read: {dt_read:.2f}s, "
              f"canvas {canvas.shape[1]}x{canvas.shape[0]}")

        data = canvas
        x_coords = x_coords_c
        y_coords = y_coords_c
        out_h, out_w = data.shape
    else:
        lon_min = center_lon - size_deg
        lon_max = center_lon + size_deg
        lat_min = center_lat - size_deg
        lat_max = center_lat + size_deg

        xi0 = max(int((lon_min - x_origin) / dx), 0)
        xi1 = min(int((lon_max - x_origin) / dx) + 1, W)
        yi0 = max(int((lat_max - y_origin) / dy), 0)
        yi1 = min(int((lat_min - y_origin) / dy) + 1, H)

        raw_w = xi1 - xi0
        raw_h = yi1 - yi0
        out_w = raw_w // subsample
        out_h = raw_h // subsample
        print(f"Window: lon [{lon_min:.3f}, {lon_max:.3f}], "
              f"lat [{lat_min:.3f}, {lat_max:.3f}]")
        print(f"Pixel window: {raw_w:,} x {raw_h:,} "
              f"(subsample {subsample}x -> {out_w:,} x {out_h:,})")

        t0 = time.time()
        data = _read_raw_window(z, yi0, yi1, xi0, xi1,
                                subsample, scale, offset, fill)
        dt_read = time.time() - t0
        print(f"Read: {dt_read:.2f}s")

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


def make_terrain_loader(zarr_path, size_deg, subsample, center_lon, center_lat,
                        horizon_rings=0):
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
    horizon_rings : int
        Number of low-res horizon rings to include on reload.

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
            return load_window(zarr_path, lon, lat, size_deg, subsample,
                               horizon_rings=horizon_rings)
        except Exception as e:
            print(f"Terrain loader error: {e}")
            return None

    return loader


def make_tile_data_fn(zarr_path, center_lon, center_lat):
    """Create a tile data callback for streaming terrain from zarr.

    The callback converts UTM world-space bounds to geographic
    coordinates, reads the corresponding slice from the zarr store,
    CF-decodes the elevation data, and returns it as a float32 array.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr store.
    center_lon, center_lat : float
        Center of the initial view in WGS84 degrees (used to pick the
        UTM zone for the inverse projection).

    Returns
    -------
    callable
        ``fn(x_min, y_min, x_max, y_max, target_samples)``
        → ``np.ndarray`` of shape ``(H, W)`` (float32 elevations,
        row 0 = northernmost) or ``None``.
    """
    from pyproj import Transformer

    z, _fmt, (x_origin, dx, y_origin, dy), \
        (scale, offset, fill), _crs_wkt = _open_zarr_array(zarr_path)
    H, W = z.shape

    target_epsg = _utm_epsg(center_lon, center_lat)
    to_lonlat = Transformer.from_crs(
        f"EPSG:{target_epsg}", "EPSG:4326", always_xy=True)

    lod_root = zarr.open_group(zarr_path, mode="r")
    ARRAY_NAME = "usgs10m_dem"

    def tile_data_fn(x_min, y_min, x_max, y_max, target_samples):
        # Convert UTM corners to lon/lat
        lon_min, lat_min = to_lonlat.transform(x_min, y_min)
        lon_max, lat_max = to_lonlat.transform(x_max, y_max)

        # Zarr pixel indices (dy < 0: row 0 = northernmost latitude)
        xi0 = int((lon_min - x_origin) / dx)
        xi1 = int((lon_max - x_origin) / dx) + 1
        yi0 = int((lat_max - y_origin) / dy)   # north → smaller index
        yi1 = int((lat_min - y_origin) / dy) + 1  # south → larger index

        # Clamp to array bounds
        xi0 = max(0, xi0)
        xi1 = min(W, xi1)
        yi0 = max(0, yi0)
        yi1 = min(H, yi1)

        if xi1 <= xi0 or yi1 <= yi0:
            return None

        # Compute stride to achieve approximately target_samples per side
        raw_size = max(xi1 - xi0, yi1 - yi0)
        stride = max(1, raw_size // target_samples)

        data = _read_raw_window(z, yi0, yi1, xi0, xi1, stride,
                                scale, offset, fill,
                                lod_root=lod_root,
                                array_name=ARRAY_NAME)
        return data

    return tile_data_fn


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
    parser.add_argument("--horizon", type=int, default=2,
                        help="Number of low-res horizon rings (default: 2, 0=off)")
    parser.add_argument("--build-lods", action="store_true",
                        help="Pre-build LOD cache arrays and exit")
    parser.add_argument("--clear-lods", action="store_true",
                        help="Delete cached LOD arrays from the zarr store and exit")
    parser.add_argument("--chunk-source", action="store_true",
                        help="Use ZarrChunkSource (for UTM-projected zarr stores)")
    args = parser.parse_args()

    if args.clear_lods:
        root = zarr.open_group(args.zarr, mode="r+")
        removed = []
        for key in list(root.keys()):
            if "_lod" in key:
                del root[key]
                removed.append(key)
        if removed:
            print(f"Removed {len(removed)} LOD arrays: {', '.join(removed)}")
        else:
            print("No LOD arrays found.")
        raise SystemExit(0)

    if args.build_lods:
        max_sub = args.subsample
        for _ in range(max(args.horizon, 2)):
            max_sub *= 4
        print(f"Building LOD arrays up to {max_sub}x...")
        _build_lod_arrays(args.zarr, "usgs10m_dem", max_factor=max_sub)
        print("Done.")
        raise SystemExit(0)

    if args.chunk_source:
        # ---- Chunk-source path (for UTM-projected zarr stores) ----
        # When the zarr store is already in a projected CRS (e.g. a scene
        # zarr built by build_scene()), the ZarrChunkSource reads chunks
        # directly — no window loading or reprojection needed.
        from rtxpy.chunk_source import ZarrChunkSource

        z, _fmt, (x_origin, dx, y_origin, dy), \
            (scale, offset, fill), crs_wkt = _open_zarr_array(args.zarr)
        H, W = z.shape
        chunks = z.chunks if hasattr(z, 'chunks') else (256, 256)
        ch = int(chunks[0]) if isinstance(chunks, (list, tuple)) else int(chunks)

        source = ZarrChunkSource(
            args.zarr, "usgs10m_dem",
            pixel_spacing_x=abs(dx),
            pixel_spacing_y=abs(dy),
            crs_origin=(x_origin, y_origin),
            crs_pixel_spacing=(dx, dy),
            cf_decode=(scale, offset, fill),
        )

        print(f"ZarrChunkSource: {H}×{W}, chunks {ch}×{ch}, "
              f"{source.n_chunk_rows}×{source.n_chunk_cols} tiles")

        # Still need a small raster for the viewer (colormap, stats, etc.)
        terrain = load_window(
            args.zarr, args.lon, args.lat, args.size, args.subsample,
            horizon_rings=0,
        )

        hydro = {'enabled': False}

        print(f"\nLaunching explore with ZarrChunkSource...\n")
        terrain.rtx.explore(
            width=2048,
            height=1600,
            render_scale=0.5,
            color_stretch='cbrt',
            terrain_source=source,
            hydro_data=hydro,
            repl=True,
        )
    else:
        # ---- Legacy path (geographic-CRS zarr with UTM reprojection) ----
        # Tile streaming handles the horizon — no need for baked multi-res
        # rings in the initial window (they produce coarse pixels in LOD 0).
        terrain = load_window(
            args.zarr, args.lon, args.lat, args.size, args.subsample,
            horizon_rings=0,
        )

        tile_fn = make_tile_data_fn(args.zarr, args.lon, args.lat)

        # Hydro flow: computed lazily on GPU when first enabled (Shift+Y)
        hydro = {'enabled': False}

        print(f"\nLaunching explore (Shift+Y to toggle hydro)...\n")
        terrain.rtx.explore(
            width=2048,
            height=1600,
            render_scale=0.5,
            color_stretch='cbrt',
            tile_data_fn=tile_fn,
            hydro_data=hydro,
            repl=True,
        )

    print("Done")
