"""Explore procedural Perlin terrain with hydro flow — no data files needed.

Generates a synthetic island archipelago using xarray-spatial's ridged
multi-octave Perlin noise with domain warping, all on GPU via dask + cupy.
Hydraulic erosion carves realistic drainage channels and hydrological flow
direction, accumulation, stream order, and stream links are recomputed for
every window — including dynamically loaded ones.

The terrain is infinite: fly to the edge and a new Perlin window is
generated on the fly.  The fixed normalization (clip → scale) and fixed
sea level ensure adjacent windows tile seamlessly.

Controls:
    WASD / arrows  Move camera
    Shift+Y        Toggle hydro flow particle animation
    G              Cycle data layers (elevation, slope, stream_link, ...)
    R / Shift+R    Increase / decrease resolution
    Y              Cycle color stretch

Usage:
    python explore_perlin.py
    python explore_perlin.py --size 2048 --chunks 512 --seed 42
    python explore_perlin.py --noise fbm --warp 0.5
    python explore_perlin.py --erosion-iters 500000

Requirements:
    pip install rtxpy[all] xarray-spatial dask cupy scipy
"""

import argparse
import time

import cupy as cp
import dask.array as da
import numpy as np
import xarray as xr
from xrspatial import (
    generate_terrain,
    slope,
    aspect,
    fill as xrs_fill,
    flow_direction,
    flow_accumulation,
    stream_order,
    stream_link,
)
from xrspatial.erosion import erode
from scipy.ndimage import uniform_filter

import rtxpy  # noqa: F401 — registers .rtx accessor

WINDOW_HALF = 10_000  # half-width of each window in metres (20 km total)
ZFACTOR = 4000  # raw elevation multiplier

# full_extent == window size → each window covers 1.0 noise-space units,
# giving proper multi-octave detail.  Windows beyond this range still work
# because Perlin noise is defined for all coordinates.
FULL_EXTENT = (-WINDOW_HALF, -WINDOW_HALF, WINDOW_HALF, WINDOW_HALF)

# Sea level as a fraction of ZFACTOR.  Ridged noise after fixed
# normalisation sits in roughly [0.5·zf, zf].  Subtracting 0.55·zf
# puts the coastline where the noise dips, creating an archipelago.
SEA_LEVEL_FRAC = 0.55

# Noise parameters (set from CLI args at startup, shared with loader).
# NOTE: worley_blend is 0 because the dask+cupy path normalises Worley
# per-chunk (min/max), creating visible grid seams.  Domain warping is
# seamless (uses deterministic Perlin displacement).
NOISE_PARAMS: dict = {}


def _apply_sea_level(terrain):
    """Subtract a fixed sea level and zero out ocean pixels.

    Because the sea level is an absolute constant (not per-window), the
    transform is identical for every window, preserving tile coherence.
    """
    sea = ZFACTOR * SEA_LEVEL_FRAC
    terrain.data[:] = cp.maximum(terrain.data - sea, 0)
    return terrain


def generate_window(size, seed, x_range, y_range, chunks=None):
    """Generate a cupy-backed elevation DataArray from Perlin noise.

    When *chunks* is given, the template is dask+cupy so noise is generated
    in parallel across GPU chunks before being materialised.
    """
    if chunks is not None:
        data = da.zeros((size, size), dtype=np.float32,
                        chunks=(chunks, chunks))
        data = data.map_blocks(cp.asarray, dtype=np.float32,
                               meta=cp.array((), dtype=np.float32))
        template = xr.DataArray(data, dims=['y', 'x'])
        print(f"Dask template: {size}x{size}, "
              f"chunks={chunks}x{chunks} "
              f"({(size // chunks) ** 2} tasks)")
    else:
        template = xr.DataArray(
            cp.zeros((size, size), dtype=cp.float32), dims=['y', 'x'],
        )

    t0 = time.time()
    terrain = generate_terrain(
        template,
        x_range=x_range,
        y_range=y_range,
        seed=seed,
        zfactor=ZFACTOR,
        full_extent=FULL_EXTENT,
        **NOISE_PARAMS,
    )

    # Materialise dask graph → cupy
    if isinstance(terrain.data, da.Array):
        print("Computing dask graph on GPU...")
        terrain = terrain.compute()
    if not hasattr(terrain.data, 'device'):
        terrain = terrain.copy(data=cp.asarray(terrain.data))

    # Fixed sea-level cutoff (same for every window → seamless tiling)
    terrain = _apply_sea_level(terrain)

    dt = time.time() - t0
    elev_min = float(cp.nanmin(terrain.data))
    elev_max = float(cp.nanmax(terrain.data))
    water_pct = float((terrain.data == 0).sum()) / terrain.data.size * 100
    print(f"Generated {terrain.shape[0]}x{terrain.shape[1]} terrain "
          f"in {dt:.2f}s  (elev {elev_min:.0f}–{elev_max:.0f} m, "
          f"{water_pct:.0f}% water)")
    return terrain


def erode_terrain(terrain, iterations, seed):
    """Apply hydraulic erosion to carve drainage channels."""
    print(f"Eroding terrain ({iterations:,} droplets)...")
    t0 = time.time()
    terrain = erode(terrain, iterations=iterations, seed=seed)
    if not hasattr(terrain.data, 'device'):
        terrain = terrain.copy(data=cp.asarray(terrain.data))
    # Re-clamp: erosion can push values slightly below 0
    terrain.data[:] = cp.maximum(terrain.data, 0)
    dt = time.time() - t0
    print(f"  Erosion done in {dt:.2f}s")
    return terrain


def compute_hydro(terrain):
    """Condition DEM and compute D8 hydro flow layers.

    Returns a hydro dict ready for explore(), plus a stream_link DataArray
    suitable for adding to a Dataset overlay.
    """
    print("Conditioning DEM for hydrological flow...")
    t0 = time.time()

    elev = cp.asnumpy(terrain.data).astype(np.float32)

    # Mark water (sea-level pixels are 0) as ocean sentinel
    ocean = (elev == 0.0) | np.isnan(elev)
    elev[ocean] = -100.0

    # Smooth to fill noise pits
    smoothed = uniform_filter(elev, size=15, mode='nearest')
    smoothed[ocean] = -100.0

    sm_gpu = cp.asarray(smoothed)
    filled = xrs_fill(terrain.copy(data=sm_gpu))

    # Resolve flats: small perturbation toward drainage
    delta = filled.data - sm_gpu
    resolved = filled.data + delta * 0.01
    cp.random.seed(0)
    resolved += cp.random.uniform(0, 0.001, resolved.shape, dtype=cp.float32)
    resolved[cp.asarray(ocean)] = -100.0

    conditioned = terrain.copy(data=resolved)
    fd = flow_direction(conditioned)
    fa = flow_accumulation(fd)
    so = stream_order(fd, fa, threshold=50)
    sl = stream_link(fd, fa, threshold=50)

    # Mask ocean back to NaN
    ocean_gpu = cp.asarray(ocean)
    fd.data[ocean_gpu] = cp.nan
    fa.data[ocean_gpu] = cp.nan
    so.data[ocean_gpu] = cp.nan
    sl.data[ocean_gpu] = cp.nan

    dt = time.time() - t0
    n_streams = int(cp.nanmax(sl.data))
    print(f"  Hydro computed in {dt:.2f}s — "
          f"{n_streams} stream segments found")

    # Clean stream_link for Dataset overlay (NaN → 0)
    sl_clean = cp.nan_to_num(sl.data, nan=0.0).astype(cp.float32)

    hydro = {
        'flow_dir': fd.data,
        'flow_accum': fa.data,
        'stream_order': so.data,
        'stream_link': sl.data,
        'accum_threshold': 50,
        'enabled': False,
    }
    return hydro, terrain.copy(data=sl_clean)


def make_terrain_loader(size, seed, chunks, erosion_iters=0, do_hydro=True):
    """Create a callback that generates new Perlin terrain at the camera.

    No clamping — Perlin noise is defined everywhere, so exploration is
    unlimited.  The full_extent mapping just sets the scale factor so each
    window covers 1.0 noise-space units.

    Each new window gets its own erosion and hydro computation so flow
    patterns are consistent with the local terrain.  Returns a
    ``(terrain_da, hydro_dict)`` tuple that the engine's terrain reload
    handler picks up to reinitialise hydro particles.
    """

    def loader(cam_x, cam_y):
        x_range = (cam_x - WINDOW_HALF, cam_x + WINDOW_HALF)
        y_range = (cam_y - WINDOW_HALF, cam_y + WINDOW_HALF)
        try:
            terrain = generate_window(size, seed, x_range, y_range,
                                      chunks=chunks)
            if erosion_iters > 0:
                terrain = erode_terrain(terrain, erosion_iters, seed)

            hydro = None
            if do_hydro:
                try:
                    hydro, _ = compute_hydro(terrain)
                except Exception as e:
                    print(f"Loader hydro error: {e}")

            return (terrain, hydro)
        except Exception as e:
            print(f"Terrain loader error: {e}")
            return None

    return loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore procedural Perlin terrain with hydro flow.",
    )
    parser.add_argument("--size", type=int, default=2048,
                        help="Grid size in pixels (default: 2048)")
    parser.add_argument("--chunks", type=int, default=512,
                        help="Dask chunk size (default: 512, 0=no dask)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Perlin noise seed (default: 42)")
    parser.add_argument("--noise", type=str, default='ridged',
                        choices=['fbm', 'ridged'],
                        help="Noise mode (default: ridged)")
    parser.add_argument("--warp", type=float, default=0.4,
                        help="Domain warp strength (default: 0.4)")
    parser.add_argument("--worley", type=float, default=0.0,
                        help="Worley cellular noise blend (default: 0.0, "
                             "causes chunk seams in dask+cupy path)")
    parser.add_argument("--octaves", type=int, default=6,
                        help="Noise octaves (default: 6)")
    parser.add_argument("--erosion-iters", type=int, default=200_000,
                        help="Hydraulic erosion droplets (default: 200000)")
    parser.add_argument("--no-erode", action="store_true",
                        help="Skip hydraulic erosion")
    parser.add_argument("--no-hydro", action="store_true",
                        help="Skip hydro flow computation")
    args = parser.parse_args()

    chunks = args.chunks if args.chunks > 0 else None

    # Noise params shared between initial window and terrain loader.
    # Erosion is handled separately (not via generate_terrain's erode param)
    # so it runs after sea-level subtraction.
    NOISE_PARAMS = {
        'noise_mode': args.noise,
        'warp_strength': args.warp,
        'worley_blend': args.worley,
        'octaves': args.octaves,
    }
    desc = [args.noise]
    if args.warp > 0:
        desc.append(f"warp={args.warp}")
    if args.worley > 0:
        desc.append(f"worley={args.worley}")
    if not args.no_erode:
        desc.append(f"erode({args.erosion_iters:,})")
    print(f"Noise: {', '.join(desc)}")

    # ---- Generate initial terrain from Perlin noise ----------------------
    x_range = (-WINDOW_HALF, WINDOW_HALF)
    y_range = (-WINDOW_HALF, WINDOW_HALF)
    terrain = generate_window(args.size, args.seed, x_range, y_range,
                              chunks=chunks)

    # ---- Hydraulic erosion ------------------------------------------------
    if not args.no_erode:
        terrain = erode_terrain(terrain, args.erosion_iters, args.seed)

    # ---- Build Dataset with analysis layers ------------------------------
    print("Computing terrain analysis layers...")
    ds = xr.Dataset({
        'elevation': terrain.rename(None),
        'slope': slope(terrain),
        'aspect': aspect(terrain),
    })

    # ---- Hydro flow ------------------------------------------------------
    hydro = None
    if not args.no_hydro:
        try:
            hydro, sl_layer = compute_hydro(terrain)
            ds['stream_link'] = sl_layer.rename(None)
        except Exception as e:
            print(f"Skipping hydro: {e}")

    # ---- Terrain loader for infinite Perlin exploration ------------------
    # The loader runs in a background thread (engine does not block the
    # render loop).  Cap erosion at 50k for reloaded windows so the load
    # completes in a few seconds instead of 30–60s.
    loader_erosion = 0 if args.no_erode else min(args.erosion_iters, 50_000)
    loader = make_terrain_loader(
        args.size, args.seed, chunks,
        erosion_iters=loader_erosion,
        do_hydro=not args.no_hydro,
    )

    # ---- Launch ----------------------------------------------------------
    print(f"\nLaunching explore "
          f"(G = cycle layers, Shift+Y = hydro flow)...\n")
    ds.rtx.explore(
        z='elevation',
        width=2048,
        height=1600,
        render_scale=0.5,
        color_stretch='cbrt',
        terrain_loader=loader,
        hydro_data=hydro,
        repl=True,
    )
    print("Done")
