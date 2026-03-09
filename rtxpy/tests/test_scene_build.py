"""Tests for rtxpy.scene — build_scene zarr writers and CLI parsing."""

import numpy as np
import pytest
import zarr

from rtxpy.scene import (
    _save_wind,
    _save_weather,
    _save_hydro,
    _load_wind,
    _load_weather,
    _load_hydro,
    main,
)
from rtxpy.mesh_store import validate_scene


@pytest.fixture
def store(tmp_path):
    """Create a minimal zarr store and return (store, path)."""
    path = str(tmp_path / "test_88_build.zarr")
    s = zarr.open(path, mode='w')
    s.attrs['rtxpy_scene_version'] = '1.0'
    elev = np.random.rand(32, 32).astype(np.float32) * 100
    s.create_array('elevation', data=elev, chunks=(32, 32))
    s['elevation'].attrs['scale_factor'] = 0.1
    s['elevation'].attrs['add_offset'] = 0.0
    s['elevation'].attrs['_FillValue'] = -9999
    sr = s.create_array('spatial_ref', data=np.int32(0))
    sr.attrs['crs_wkt'] = 'GEOGCS["WGS 84"]'
    sr.attrs['GeoTransform'] = "0 30 0 0 0 -30"
    return s, path


# -------------------------------------------------------------------
# Wind round-trip
# -------------------------------------------------------------------

def test_save_load_wind(store):
    s, path = store
    bounds = (-112.2, 36.0, -112.0, 36.2)
    wind_data = {
        'u': np.random.randn(10, 10).astype(np.float32),
        'v': np.random.randn(10, 10).astype(np.float32),
        'lats': np.linspace(36.0, 36.2, 10).astype(np.float32),
        'lons': np.linspace(-112.2, -112.0, 10).astype(np.float32),
    }
    _save_wind(s, wind_data, bounds)

    # Reload and verify
    s2 = zarr.open(path, mode='r')
    loaded = _load_wind(s2)
    assert loaded is not None
    np.testing.assert_array_almost_equal(loaded['u'], wind_data['u'])
    np.testing.assert_array_almost_equal(loaded['v'], wind_data['v'])
    np.testing.assert_array_almost_equal(loaded['lats'], wind_data['lats'])
    np.testing.assert_array_almost_equal(loaded['lons'], wind_data['lons'])

    # Validate passes
    issues = validate_scene(path)
    wind_issues = [msg for lvl, msg in issues if 'wind' in msg]
    assert wind_issues == []


def test_load_wind_missing(store):
    s, _ = store
    assert _load_wind(s) is None


# -------------------------------------------------------------------
# Weather round-trip
# -------------------------------------------------------------------

def test_save_load_weather(store):
    s, path = store
    bounds = (-112.2, 36.0, -112.0, 36.2)
    weather_data = {
        'cloud_cover': np.random.rand(8, 8).astype(np.float32),
        'temperature': (np.random.rand(8, 8) * 30 + 270).astype(np.float32),
        'lats': np.linspace(36.0, 36.2, 8).astype(np.float32),
        'lons': np.linspace(-112.2, -112.0, 8).astype(np.float32),
    }
    _save_weather(s, weather_data, bounds)

    s2 = zarr.open(path, mode='r')
    loaded = _load_weather(s2)
    assert loaded is not None
    np.testing.assert_array_almost_equal(loaded['cloud_cover'],
                                         weather_data['cloud_cover'])
    np.testing.assert_array_almost_equal(loaded['temperature'],
                                         weather_data['temperature'])
    assert 'humidity' not in loaded  # not in input


def test_load_weather_missing(store):
    s, _ = store
    assert _load_weather(s) is None


# -------------------------------------------------------------------
# Hydro round-trip
# -------------------------------------------------------------------

def test_save_load_hydro(store):
    s, path = store
    hydro_data = {
        'flow_accum': np.random.rand(32, 32).astype(np.float32),
        'flow_dir_mfd': np.random.rand(8, 32, 32).astype(np.float32),
        'stream_order': np.random.randint(0, 5, (32, 32)).astype(np.int32),
        'n_particles': 8000,
        'speed': 0.5,
        'color': [0.2, 0.5, 1.0],
    }
    _save_hydro(s, hydro_data)

    s2 = zarr.open(path, mode='r')
    loaded = _load_hydro(s2)
    assert loaded is not None
    np.testing.assert_array_almost_equal(loaded['flow_accum'],
                                         hydro_data['flow_accum'])
    assert loaded['n_particles'] == 8000
    assert loaded['speed'] == 0.5

    issues = validate_scene(path)
    hydro_issues = [msg for lvl, msg in issues
                    if 'hydro' in msg and lvl == 'error']
    assert hydro_issues == []


def test_load_hydro_missing(store):
    s, _ = store
    assert _load_hydro(s) is None


# -------------------------------------------------------------------
# CLI argument parsing
# -------------------------------------------------------------------

def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Build a scene zarr" in captured.out


def test_cli_parses_bounds():
    """Verify argparse accepts the expected positional args."""
    import argparse
    # We can't run build_scene without network, but we can verify
    # the parser accepts the right args by catching the ImportError
    # that would come from fetch_dem.
    try:
        main(["-112.2", "36.0", "-112.0", "36.2", "/tmp/test.zarr",
              "--no-buildings", "--no-water", "--no-wind", "--no-weather"])
    except (ImportError, Exception):
        pass  # expected — we just want to verify arg parsing works


# -------------------------------------------------------------------
# Overwrite safety — _save_wind clears existing group
# -------------------------------------------------------------------

def test_save_wind_overwrites(store):
    s, path = store
    bounds = (-112.2, 36.0, -112.0, 36.2)
    wind1 = {
        'u': np.ones((5, 5), dtype=np.float32),
        'v': np.zeros((5, 5), dtype=np.float32),
        'lats': np.linspace(36.0, 36.2, 5).astype(np.float32),
        'lons': np.linspace(-112.2, -112.0, 5).astype(np.float32),
    }
    _save_wind(s, wind1, bounds)

    wind2 = {
        'u': np.zeros((8, 8), dtype=np.float32),
        'v': np.ones((8, 8), dtype=np.float32),
        'lats': np.linspace(36.0, 36.2, 8).astype(np.float32),
        'lons': np.linspace(-112.2, -112.0, 8).astype(np.float32),
    }
    _save_wind(s, wind2, bounds)

    s2 = zarr.open(path, mode='r')
    loaded = _load_wind(s2)
    assert loaded['u'].shape == (8, 8)
    np.testing.assert_array_almost_equal(loaded['u'], 0.0)
