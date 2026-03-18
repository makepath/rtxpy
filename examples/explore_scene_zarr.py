"""Zarr-chunk-driven scene exploration — the new chunk source path.

Demonstrates the zarr-native approach: build a scene zarr with
``build_scene()`` and explore it with ``explore_scene()``.  The
elevation array's zarr chunks directly define the LOD tile grid —
one chunk = one tile.  Placed geometry (buildings, roads, water)
is loaded per-chunk in sync with terrain tile visibility.

No full raster is loaded into memory.  The ZarrChunkSource reads
elevation chunks on demand, and the SceneMeshManager loads placed
geometry as tiles come into view.

This is the right path when:
- You want chunk-driven, memory-efficient terrain streaming
- You're working with pre-built scene zarr stores
- Your scene has placed geometry that should load/unload with tiles

Usage:
    # Build + explore in one shot
    python explore_scene_zarr.py

    # Just explore an existing scene
    python explore_scene_zarr.py --scene my_scene.zarr

    # Custom location
    python explore_scene_zarr.py --bounds -122.5 37.7 -122.3 37.85

Requirements:
    pip install rtxpy[all]
"""

import argparse
from pathlib import Path

import rtxpy


# Default: downtown San Francisco — dense buildings + grid streets
BOUNDS = (-122.43, 37.76, -122.38, 37.80)


def main():
    parser = argparse.ArgumentParser(
        description="Explore a zarr scene with chunk-driven LOD.")
    parser.add_argument("--scene", type=str, default=None,
                        help="Path to an existing scene zarr "
                             "(skip build if provided)")
    parser.add_argument("--bounds", type=float, nargs=4,
                        metavar=("W", "S", "E", "N"),
                        default=list(BOUNDS),
                        help="Bounding box in lon/lat degrees")
    parser.add_argument("--dem", type=str, default="copernicus",
                        choices=["copernicus", "srtm", "usgs_10m"],
                        help="DEM source (default: copernicus)")
    parser.add_argument("--tile-size", type=int, default=256,
                        help="Zarr chunk / LOD tile size in pixels "
                             "(default: 256)")
    parser.add_argument("--no-buildings", action="store_true",
                        help="Skip building placement")
    parser.add_argument("--no-roads", action="store_true",
                        help="Skip road placement")
    parser.add_argument("--no-water", action="store_true",
                        help="Skip water feature placement")
    args = parser.parse_args()

    bounds = tuple(args.bounds)

    if args.scene is not None:
        scene_path = args.scene
    else:
        scene_path = "explore_scene.zarr"

        # ---- Build the scene zarr ---------------------------------------
        # build_scene() fetches the DEM, places geometry, and writes
        # everything to a single zarr store.  The tile_size parameter
        # controls zarr chunk sizing — this directly determines the LOD
        # tile grid (one chunk = one tile).
        print(f"Building scene for {bounds}...")
        print(f"  DEM source: {args.dem}")
        print(f"  Tile size:  {args.tile_size}px")
        print(f"  Buildings:  {not args.no_buildings}")
        print(f"  Roads:      {not args.no_roads}")
        print(f"  Water:      {not args.no_water}")
        print()

        rtxpy.build_scene(
            bounds,
            scene_path,
            dem_source=args.dem,
            buildings=not args.no_buildings,
            roads=not args.no_roads,
            water=not args.no_water,
            wind=False,
            weather=False,
            tile_size=args.tile_size,
        )

    # ---- Explore the scene zarr -----------------------------------------
    # explore_scene() auto-constructs a ZarrChunkSource from the elevation
    # array when the chunk size is tile-friendly.  The LOD manager reads
    # terrain from the zarr source, and its SceneMeshManager loads placed
    # geometry (buildings, roads, water) per-chunk as tiles come into view.
    #
    # Key differences from the in-memory path:
    #   - Terrain is read per-chunk from zarr, not held in full in memory
    #   - Placed geometry loads/unloads in sync with terrain tiles
    #   - A single LOD computation drives both terrain and geometry
    print(f"\nExploring {scene_path}...")
    rtxpy.explore_scene(
        scene_path,
        width=1600,
        height=1200,
        render_scale=0.5,
        color_stretch='cbrt',
        repl=True,
    )

    print("Done")


if __name__ == "__main__":
    main()
