"""New Orleans scene benchmark: CUDA-GL interop vs CPU path.

Builds a zarr scene with 1m USGS DEM (int32 CF-encoded, 256x256 chunks),
buildings, roads, water, and wind. Uses the zarr-chunk-driven LOD explore
path where one zarr chunk = one LOD tile, streamed on demand.

Then runs an A/B comparison of interop vs CPU display path.
"""
import json
import os
import time
import numpy as np
import rtxpy

# New Orleans metro area — large extent for LOD tile streaming benchmark
# Copernicus 30m DEM (USGS 1m preferred but API unstable), 256x256 zarr chunks
BOUNDS = (-90.25, 29.85, -89.90, 30.05)
SCENE_PATH = "nola_scene.zarr"
BENCH_SECONDS = 15
WARMUP_FRAMES = 30


def build_scene():
    """Build the New Orleans scene zarr (skips if already exists)."""
    if os.path.exists(SCENE_PATH):
        print(f"Scene already exists: {SCENE_PATH}")
        print("  Delete it to rebuild.\n")
        return

    print("Building New Orleans scene...")
    print(f"  Bounds: {BOUNDS}")
    print(f"  DEM: Copernicus 30m, int16 CF-encoded, 256x256 zarr chunks")
    print(f"  Layers: elevation, buildings, roads, water")
    print()

    rtxpy.build_scene(
        BOUNDS,
        SCENE_PATH,
        dem_source="copernicus",
        buildings=True,
        roads=True,
        water=True,
        hydro=False,
        wind=False,
        weather=False,
        fires=False,
        tile_size=256,
    )
    print(f"\nScene built: {SCENE_PATH}\n")


def run_bench(label, interop_enabled):
    """Run benchmark on the scene, return dict of results."""
    from rtxpy.engine import InteractiveViewer
    original_tick = InteractiveViewer._tick

    frame_times = []
    frame_count = [0]
    bench_start = [None]

    def patched_tick(self):
        now = time.perf_counter()
        frame_count[0] += 1

        if frame_count[0] == WARMUP_FRAMES + 1:
            bench_start[0] = now
            print(f"  Warmup done ({WARMUP_FRAMES} frames). "
                  f"Benchmarking for {BENCH_SECONDS}s...")

        if frame_count[0] > WARMUP_FRAMES and hasattr(self, '_bench_last_tick'):
            frame_times.append(now - self._bench_last_tick)

        if bench_start[0] and (now - bench_start[0]) > BENCH_SECONDS:
            import glfw
            if self._glfw_window:
                glfw.set_window_should_close(self._glfw_window, True)

        self._bench_last_tick = time.perf_counter()
        original_tick(self)

        # Gentle camera drift for continuous rendering + LOD updates
        # At 1m resolution, drift slowly to trigger LOD tile streaming
        if hasattr(self, 'position') and hasattr(self, '_forward'):
            self.position[0] += self._forward[0] * 0.5
            self.position[1] += self._forward[1] * 0.5
            self._render_needed = True

    InteractiveViewer._tick = patched_tick

    if not interop_enabled:
        os.environ['RTXPY_NO_GL_INTEROP'] = '1'
    else:
        os.environ.pop('RTXPY_NO_GL_INTEROP', None)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Interop: {'ENABLED' if interop_enabled else 'DISABLED'}")
    print(f"{'='*60}")

    # explore_scene auto-constructs a ZarrChunkSource when chunk sizes
    # are tile-friendly. This is the zarr-chunk-driven LOD path where
    # one zarr chunk = one LOD tile, streamed on demand.
    #
    # Start zoomed into the center of the DEM so only a handful of
    # LOD tiles are visible — this is how the LOD system is meant to
    # be used on large terrains.
    rtxpy.explore_scene(
        SCENE_PATH,
        width=1920,
        height=1080,
        render_scale=1.0,
        color_stretch='cbrt',
        ao_samples=0,
        denoise=False,
        repl=False,
        start_position=(600, 350, 150),
    )

    InteractiveViewer._tick = original_tick
    os.environ.pop('RTXPY_NO_GL_INTEROP', None)

    if not frame_times:
        print("  No frames recorded!")
        return None

    ft = np.array(frame_times) * 1000
    results = {
        'label': label,
        'interop': interop_enabled,
        'total_frames': len(frame_times),
        'duration_s': sum(frame_times),
        'fps_mean': len(frame_times) / sum(frame_times),
        'frame_ms_mean': float(np.mean(ft)),
        'frame_ms_median': float(np.median(ft)),
        'frame_ms_p95': float(np.percentile(ft, 95)),
        'frame_ms_p99': float(np.percentile(ft, 99)),
        'frame_ms_min': float(np.min(ft)),
        'frame_ms_max': float(np.max(ft)),
    }

    print(f"\n  Results ({label}):")
    print(f"    Frames:     {results['total_frames']}")
    print(f"    Duration:   {results['duration_s']:.1f}s")
    print(f"    FPS:        {results['fps_mean']:.1f}")
    print(f"    Frame time: {results['frame_ms_mean']:.2f}ms mean, "
          f"{results['frame_ms_median']:.2f}ms median")
    print(f"    P95/P99:    {results['frame_ms_p95']:.2f}ms / "
          f"{results['frame_ms_p99']:.2f}ms")
    print(f"    Min/Max:    {results['frame_ms_min']:.2f}ms / "
          f"{results['frame_ms_max']:.2f}ms")

    return results


if __name__ == '__main__':
    build_scene()

    all_results = []

    r1 = run_bench("Interop (zero-copy) + NOLA 1m scene", interop_enabled=True)
    if r1:
        all_results.append(r1)

    time.sleep(2)

    r2 = run_bench("CPU path + NOLA 1m scene", interop_enabled=False)
    if r2:
        all_results.append(r2)

    if len(all_results) == 2:
        a, b = all_results
        print(f"\n{'='*60}")
        print(f"  NEW ORLEANS 1m SCENE BENCHMARK SUMMARY")
        print(f"  Copernicus 30m DEM (int16) + Buildings + Roads + Water")
        print(f"  Zarr-chunk LOD (256x256 tiles, streamed on demand)")
        print(f"{'='*60}")
        print(f"  {'Metric':<20} {'Interop':>12} {'CPU Path':>12} {'Speedup':>10}")
        print(f"  {'-'*54}")
        print(f"  {'FPS':<20} {a['fps_mean']:>12.1f} {b['fps_mean']:>12.1f} "
              f"{a['fps_mean']/b['fps_mean']:>9.2f}x")
        print(f"  {'Frame ms (mean)':<20} {a['frame_ms_mean']:>11.2f}ms "
              f"{b['frame_ms_mean']:>11.2f}ms")
        print(f"  {'Frame ms (median)':<20} {a['frame_ms_median']:>11.2f}ms "
              f"{b['frame_ms_median']:>11.2f}ms")
        print(f"  {'Frame ms (p95)':<20} {a['frame_ms_p95']:>11.2f}ms "
              f"{b['frame_ms_p95']:>11.2f}ms")
        print(f"  {'Frame ms (p99)':<20} {a['frame_ms_p99']:>11.2f}ms "
              f"{b['frame_ms_p99']:>11.2f}ms")
        saved = b['frame_ms_mean'] - a['frame_ms_mean']
        pct = saved / b['frame_ms_mean'] * 100 if b['frame_ms_mean'] > 0 else 0
        print(f"\n  Interop saves {saved:.2f}ms/frame ({pct:.1f}% of frame time)")

    with open('bench_results_nola.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to bench_results_nola.json")
