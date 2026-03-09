"""Tests for validate_scene() against the scene zarr specification."""

import numpy as np
import pytest
import zarr

from rtxpy.mesh_store import validate_scene, SCENE_VERSION


@pytest.fixture
def scene_dir(tmp_path):
    """Return a path for a temporary zarr store."""
    return str(tmp_path / "test_88_scene.zarr")


def _make_minimal_scene(path):
    """Create a minimal valid scene zarr (elevation + spatial_ref)."""
    store = zarr.open(path, mode='w')
    store.attrs['rtxpy_scene_version'] = SCENE_VERSION

    elev = np.random.rand(64, 64).astype(np.float32) * 100
    store.create_array('elevation', data=elev, chunks=(64, 64))
    store['elevation'].attrs['scale_factor'] = 0.1
    store['elevation'].attrs['add_offset'] = 0.0
    store['elevation'].attrs['_FillValue'] = -9999

    sr = store.create_array('spatial_ref', data=np.int32(0))
    sr.attrs['crs_wkt'] = 'GEOGCS["WGS 84"]'
    sr.attrs['GeoTransform'] = "0 30 0 0 0 -30"

    return store


def test_minimal_valid_scene(scene_dir):
    _make_minimal_scene(scene_dir)
    issues = validate_scene(scene_dir)
    errors = [msg for lvl, msg in issues if lvl == "error"]
    assert errors == [], f"Unexpected errors: {errors}"


def test_missing_elevation(scene_dir):
    store = zarr.open(scene_dir, mode='w')
    store.attrs['rtxpy_scene_version'] = SCENE_VERSION
    sr = store.create_array('spatial_ref', data=np.int32(0))
    sr.attrs['crs_wkt'] = 'GEOGCS["WGS 84"]'
    sr.attrs['GeoTransform'] = "0 30 0 0 0 -30"

    issues = validate_scene(scene_dir)
    errors = [msg for lvl, msg in issues if lvl == "error"]
    assert any("elevation" in e for e in errors)


def test_missing_spatial_ref(scene_dir):
    store = zarr.open(scene_dir, mode='w')
    store.attrs['rtxpy_scene_version'] = SCENE_VERSION
    elev = np.zeros((32, 32), dtype=np.float32)
    store.create_array('elevation', data=elev, chunks=(32, 32))
    store['elevation'].attrs['scale_factor'] = 0.1
    store['elevation'].attrs['add_offset'] = 0.0
    store['elevation'].attrs['_FillValue'] = -9999

    issues = validate_scene(scene_dir)
    errors = [msg for lvl, msg in issues if lvl == "error"]
    assert any("spatial_ref" in e for e in errors)


def test_missing_version_warns(scene_dir):
    store = zarr.open(scene_dir, mode='w')
    elev = np.zeros((32, 32), dtype=np.float32)
    store.create_array('elevation', data=elev, chunks=(32, 32))
    store['elevation'].attrs['scale_factor'] = 0.1
    store['elevation'].attrs['add_offset'] = 0.0
    store['elevation'].attrs['_FillValue'] = -9999
    sr = store.create_array('spatial_ref', data=np.int32(0))
    sr.attrs['crs_wkt'] = 'GEOGCS["WGS 84"]'
    sr.attrs['GeoTransform'] = "0 30 0 0 0 -30"

    issues = validate_scene(scene_dir)
    warnings = [msg for lvl, msg in issues if lvl == "warning"]
    assert any("rtxpy_scene_version" in w for w in warnings)


def test_missing_cf_attrs_warns(scene_dir):
    store = zarr.open(scene_dir, mode='w')
    store.attrs['rtxpy_scene_version'] = SCENE_VERSION
    elev = np.zeros((32, 32), dtype=np.float32)
    store.create_array('elevation', data=elev, chunks=(32, 32))
    # No CF attrs set
    sr = store.create_array('spatial_ref', data=np.int32(0))
    sr.attrs['crs_wkt'] = 'GEOGCS["WGS 84"]'
    sr.attrs['GeoTransform'] = "0 30 0 0 0 -30"

    issues = validate_scene(scene_dir)
    warnings = [msg for lvl, msg in issues if lvl == "warning"]
    assert any("scale_factor" in w for w in warnings)


def test_meshes_missing_color_warns(scene_dir):
    store = _make_minimal_scene(scene_dir)
    mg = store.create_group('meshes')
    mg.attrs['pixel_spacing'] = [30.0, 30.0]
    mg.attrs['elevation_shape'] = [64, 64]
    mg.attrs['elevation_chunks'] = [64, 64]

    gg = mg.create_group('building')
    # No color attr set
    cg = gg.create_group('0_0')
    cg.create_array('vertices', data=np.zeros(9, dtype=np.float32))
    cg.create_array('indices', data=np.array([0, 1, 2], dtype=np.int32))

    issues = validate_scene(scene_dir)
    warnings = [msg for lvl, msg in issues if lvl == "warning"]
    assert any("color" in w for w in warnings)


def test_meshes_valid_triangles(scene_dir):
    store = _make_minimal_scene(scene_dir)
    mg = store.create_group('meshes')
    mg.attrs['pixel_spacing'] = [30.0, 30.0]
    mg.attrs['elevation_shape'] = [64, 64]
    mg.attrs['elevation_chunks'] = [64, 64]

    gg = mg.create_group('building')
    gg.attrs['color'] = [0.6, 0.6, 0.6, 1.0]
    cg = gg.create_group('0_0')
    cg.create_array('vertices', data=np.zeros(9, dtype=np.float32))
    cg.create_array('indices', data=np.array([0, 1, 2], dtype=np.int32))

    issues = validate_scene(scene_dir)
    errors = [msg for lvl, msg in issues if lvl == "error"]
    assert errors == []


def test_meshes_curve_missing_widths_warns(scene_dir):
    store = _make_minimal_scene(scene_dir)
    mg = store.create_group('meshes')
    mg.attrs['pixel_spacing'] = [30.0, 30.0]
    mg.attrs['elevation_shape'] = [64, 64]
    mg.attrs['elevation_chunks'] = [64, 64]

    gg = mg.create_group('road')
    gg.attrs['color'] = [0.3, 0.3, 0.3, 1.0]
    gg.attrs['type'] = 'curve'
    cg = gg.create_group('0_0')
    cg.create_array('vertices', data=np.zeros(9, dtype=np.float32))
    cg.create_array('indices', data=np.array([0], dtype=np.int32))
    # Missing 'widths'

    issues = validate_scene(scene_dir)
    warnings = [msg for lvl, msg in issues if lvl == "warning"]
    assert any("widths" in w for w in warnings)


def test_overlay_shape_mismatch_warns(scene_dir):
    store = _make_minimal_scene(scene_dir)
    og = store.create_group('overlays')
    slope_grp = og.create_group('slope')
    # Elevation is 64x64, overlay is 32x32 — mismatch
    slope_grp.create_array('data', data=np.zeros((32, 32), dtype=np.float32))

    issues = validate_scene(scene_dir)
    warnings = [msg for lvl, msg in issues if lvl == "warning"]
    assert any("shape" in w and "slope" in w for w in warnings)


def test_overlay_matching_shape_ok(scene_dir):
    store = _make_minimal_scene(scene_dir)
    og = store.create_group('overlays')
    slope_grp = og.create_group('slope')
    slope_grp.create_array('data', data=np.zeros((64, 64), dtype=np.float32))

    issues = validate_scene(scene_dir)
    overlay_warnings = [msg for lvl, msg in issues
                        if lvl == "warning" and "overlay" in msg]
    assert overlay_warnings == []


def test_wind_missing_arrays_warns(scene_dir):
    store = _make_minimal_scene(scene_dir)
    wg = store.create_group('wind')
    wg.attrs['grid_bounds'] = {'x0': 0, 'y0': 0, 'x1': 100, 'y1': 100}
    # No u or v arrays

    issues = validate_scene(scene_dir)
    warnings = [msg for lvl, msg in issues if lvl == "warning"]
    assert any("'u'" in w for w in warnings)
    assert any("'v'" in w for w in warnings)


def test_wind_valid(scene_dir):
    store = _make_minimal_scene(scene_dir)
    wg = store.create_group('wind')
    wg.attrs['grid_bounds'] = {'x0': 0, 'y0': 0, 'x1': 100, 'y1': 100}
    wg.create_array('u', data=np.zeros((20, 20), dtype=np.float32))
    wg.create_array('v', data=np.zeros((20, 20), dtype=np.float32))

    issues = validate_scene(scene_dir)
    wind_issues = [msg for lvl, msg in issues if "wind" in msg]
    assert wind_issues == []


def test_hydro_missing_arrays_warns(scene_dir):
    store = _make_minimal_scene(scene_dir)
    store.create_group('hydro')

    issues = validate_scene(scene_dir)
    warnings = [msg for lvl, msg in issues if lvl == "warning"]
    assert any("flow_accum" in w for w in warnings)
    assert any("flow_dir_mfd" in w for w in warnings)


def test_tour_mismatched_lengths_warns(scene_dir):
    store = _make_minimal_scene(scene_dir)
    tg = store.create_group('tour')
    tg.create_array('time', data=np.array([0, 5, 10], dtype=np.float32))
    tg.create_array('position', data=np.zeros((3, 3), dtype=np.float32))
    tg.create_array('yaw', data=np.array([0, 90], dtype=np.float32))  # wrong length
    tg.create_array('pitch', data=np.array([0, -20, -30], dtype=np.float32))

    issues = validate_scene(scene_dir)
    warnings = [msg for lvl, msg in issues if lvl == "warning"]
    assert any("mismatched" in w for w in warnings)


def test_tour_valid(scene_dir):
    store = _make_minimal_scene(scene_dir)
    tg = store.create_group('tour')
    tg.attrs['fps'] = 30
    tg.attrs['loop'] = False
    tg.create_array('time', data=np.array([0, 5, 10], dtype=np.float32))
    tg.create_array('position', data=np.zeros((3, 3), dtype=np.float32))
    tg.create_array('yaw', data=np.array([0, 90, 180], dtype=np.float32))
    tg.create_array('pitch', data=np.array([-20, -30, -25], dtype=np.float32))

    issues = validate_scene(scene_dir)
    tour_issues = [msg for lvl, msg in issues if "tour" in msg]
    assert tour_issues == []


def test_nonexistent_path():
    issues = validate_scene("/nonexistent/path/88.zarr")
    assert len(issues) == 1
    assert issues[0][0] == "error"
    assert "Cannot open" in issues[0][1]


def test_full_scene_no_errors(scene_dir):
    """A scene with all optional groups filled in should have zero errors."""
    store = _make_minimal_scene(scene_dir)

    # meshes
    mg = store.create_group('meshes')
    mg.attrs['pixel_spacing'] = [30.0, 30.0]
    mg.attrs['elevation_shape'] = [64, 64]
    mg.attrs['elevation_chunks'] = [64, 64]
    gg = mg.create_group('building')
    gg.attrs['color'] = [0.6, 0.6, 0.6, 1.0]
    cg = gg.create_group('0_0')
    cg.create_array('vertices', data=np.zeros(9, dtype=np.float32))
    cg.create_array('indices', data=np.array([0, 1, 2], dtype=np.int32))

    # overlays
    og = store.create_group('overlays')
    sg = og.create_group('slope')
    sg.create_array('data', data=np.zeros((64, 64), dtype=np.float32))

    # wind
    wg = store.create_group('wind')
    wg.attrs['grid_bounds'] = {'x0': 0, 'y0': 0, 'x1': 1920, 'y1': 1920}
    wg.create_array('u', data=np.zeros((20, 20), dtype=np.float32))
    wg.create_array('v', data=np.zeros((20, 20), dtype=np.float32))

    # hydro
    hg = store.create_group('hydro')
    hg.create_array('flow_accum', data=np.zeros((64, 64), dtype=np.float32))
    hg.create_array('flow_dir_mfd',
                    data=np.zeros((8, 64, 64), dtype=np.float32))

    # tour
    tg = store.create_group('tour')
    tg.attrs['fps'] = 30
    tg.create_array('time', data=np.array([0, 5], dtype=np.float32))
    tg.create_array('position', data=np.zeros((2, 3), dtype=np.float32))
    tg.create_array('yaw', data=np.array([0, 90], dtype=np.float32))
    tg.create_array('pitch', data=np.array([-20, -30], dtype=np.float32))

    issues = validate_scene(scene_dir)
    errors = [msg for lvl, msg in issues if lvl == "error"]
    assert errors == [], f"Unexpected errors: {errors}"
