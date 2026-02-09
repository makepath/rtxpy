"""Tests for GeoJSON parsing, coordinate conversion, and mesh generation."""

import json
import warnings

import numpy as np
import pytest
import xarray as xr

from rtxpy.geojson import (
    _build_transformer,
    _flatten_multi,
    _geojson_to_world_coords,
    _linestring_to_wall_mesh,
    _load_geojson,
    _make_marker_cube,
    _points_in_polygon,
    _polygon_to_mesh,
    _sanitize_label,
    _scatter_in_polygon,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raster(H=10, W=10, x0=0.0, x1=9.0, y0=0.0, y1=9.0):
    """Create a simple xarray DataArray with pixel-space coordinates (no CRS)."""
    data = np.arange(H * W, dtype=np.float32).reshape(H, W)
    xs = np.linspace(x0, x1, W)
    ys = np.linspace(y0, y1, H)
    return xr.DataArray(data, dims=("y", "x"), coords={"y": ys, "x": xs})


# ---------------------------------------------------------------------------
# _load_geojson
# ---------------------------------------------------------------------------

class TestLoadGeojson:
    def test_feature_collection(self):
        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 2]}, "properties": {"name": "a"}},
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [3, 4]}, "properties": {"name": "b"}},
            ],
        }
        result = _load_geojson(fc)
        assert len(result) == 2
        assert result[0][0]["type"] == "Point"
        assert result[0][1]["name"] == "a"

    def test_single_feature(self):
        feat = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [5, 6]}, "properties": {"k": "v"}}
        result = _load_geojson(feat)
        assert len(result) == 1
        assert result[0][0]["coordinates"] == [5, 6]
        assert result[0][1]["k"] == "v"

    def test_bare_geometry(self):
        geom = {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}
        result = _load_geojson(geom)
        assert len(result) == 1
        assert result[0][0]["type"] == "LineString"
        assert result[0][1] == {}

    def test_geometry_collection(self):
        gc = {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Point", "coordinates": [1, 2]},
                {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
            ],
        }
        result = _load_geojson(gc)
        assert len(result) == 2
        assert result[0][1] == {}  # no properties for GeometryCollection children

    def test_file_path(self, tmp_path):
        geojson_data = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [10, 20]},
            "properties": {},
        }
        path = tmp_path / "test.geojson"
        path.write_text(json.dumps(geojson_data))
        result = _load_geojson(str(path))
        assert len(result) == 1
        assert result[0][0]["coordinates"] == [10, 20]

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _load_geojson("/no/such/file.geojson")

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            _load_geojson([1, 2, 3])

    def test_unsupported_geojson_type(self):
        with pytest.raises(ValueError, match="Unsupported GeoJSON type"):
            _load_geojson({"type": "Topology"})

    def test_empty_feature_collection(self):
        fc = {"type": "FeatureCollection", "features": []}
        assert _load_geojson(fc) == []

    def test_feature_with_null_geometry(self):
        feat = {"type": "Feature", "geometry": None, "properties": {"a": 1}}
        assert _load_geojson(feat) == []

    def test_feature_missing_properties(self):
        feat = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}, "properties": None}
        result = _load_geojson(feat)
        assert result[0][1] == {}


# ---------------------------------------------------------------------------
# _flatten_multi
# ---------------------------------------------------------------------------

class TestFlattenMulti:
    def test_multi_point(self):
        geom = {"type": "MultiPoint", "coordinates": [[1, 2], [3, 4]]}
        result = _flatten_multi(geom)
        assert len(result) == 2
        assert result[0] == {"type": "Point", "coordinates": [1, 2]}
        assert result[1] == {"type": "Point", "coordinates": [3, 4]}

    def test_multi_linestring(self):
        geom = {"type": "MultiLineString", "coordinates": [[[0, 0], [1, 1]], [[2, 2], [3, 3]]]}
        result = _flatten_multi(geom)
        assert len(result) == 2
        assert result[0]["type"] == "LineString"

    def test_multi_polygon(self):
        ring = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
        geom = {"type": "MultiPolygon", "coordinates": [[ring], [ring]]}
        result = _flatten_multi(geom)
        assert len(result) == 2
        assert result[0]["type"] == "Polygon"

    def test_already_primitive(self):
        geom = {"type": "Point", "coordinates": [5, 6]}
        result = _flatten_multi(geom)
        assert len(result) == 1
        assert result[0] is geom

    def test_empty_multi(self):
        geom = {"type": "MultiPoint", "coordinates": []}
        assert _flatten_multi(geom) == []


# ---------------------------------------------------------------------------
# _sanitize_label
# ---------------------------------------------------------------------------

class TestSanitizeLabel:
    def test_none(self):
        assert _sanitize_label(None) == "unknown"

    def test_simple_string(self):
        assert _sanitize_label("hello") == "hello"

    def test_spaces_and_special(self):
        assert _sanitize_label("Tower #1 (north)") == "Tower--1--north"

    def test_numeric(self):
        assert _sanitize_label(42) == "42"

    def test_empty_string(self):
        assert _sanitize_label("") == "unknown"

    def test_all_special(self):
        assert _sanitize_label("@!#$") == "unknown"

    def test_leading_trailing_special(self):
        assert _sanitize_label("--hello--") == "hello"


# ---------------------------------------------------------------------------
# _geojson_to_world_coords
# ---------------------------------------------------------------------------

class TestGeojsonToWorldCoords:
    def test_basic_conversion(self):
        """Coordinates matching raster grid → pixel-space world coords."""
        raster = _make_raster(10, 10)
        terrain = np.arange(100, dtype=np.float32).reshape(10, 10)
        # coords at pixel (2, 3) → lon=2.0, lat=3.0 on this identity raster
        result = _geojson_to_world_coords(
            [[2.0, 3.0]], raster, terrain, psx=1.0, psy=1.0,
        )
        assert result.shape == (1, 3)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result[0, 0], 2.0, atol=1e-5)
        np.testing.assert_allclose(result[0, 1], 3.0, atol=1e-5)
        # z = terrain[3, 2] = 3*10+2 = 32
        np.testing.assert_allclose(result[0, 2], 32.0, atol=1e-5)

    def test_pixel_spacing(self):
        raster = _make_raster(10, 10)
        terrain = np.zeros((10, 10), dtype=np.float32)
        result = _geojson_to_world_coords(
            [[5.0, 5.0]], raster, terrain, psx=25.0, psy=25.0,
        )
        np.testing.assert_allclose(result[0, 0], 5.0 * 25.0, atol=1e-3)
        np.testing.assert_allclose(result[0, 1], 5.0 * 25.0, atol=1e-3)

    def test_empty_input(self):
        raster = _make_raster(10, 10)
        terrain = np.zeros((10, 10), dtype=np.float32)
        result = _geojson_to_world_coords(
            np.empty((0, 2)), raster, terrain, psx=1.0, psy=1.0,
        )
        assert result.shape == (0, 3)
        assert result.dtype == np.float32

    def test_empty_input_with_return_pixel_coords(self):
        raster = _make_raster(10, 10)
        terrain = np.zeros((10, 10), dtype=np.float32)
        world, pixel = _geojson_to_world_coords(
            [], raster, terrain, psx=1.0, psy=1.0, return_pixel_coords=True,
        )
        assert world.shape == (0, 3)
        assert pixel.shape == (0, 2)

    def test_out_of_bounds_warning(self):
        raster = _make_raster(10, 10)
        terrain = np.zeros((10, 10), dtype=np.float32)
        oob_counter = [0]
        _geojson_to_world_coords(
            [[100.0, 100.0]], raster, terrain, psx=1.0, psy=1.0,
            oob_counter=oob_counter,
        )
        assert oob_counter[0] >= 1

    def test_no_crs_warning(self):
        raster = _make_raster(10, 10)
        terrain = np.zeros((10, 10), dtype=np.float32)
        # No CRS → should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _geojson_to_world_coords(
                [[5.0, 5.0]], raster, terrain, psx=1.0, psy=1.0,
            )
            crs_warns = [x for x in w if "no CRS metadata" in str(x.message)]
            assert len(crs_warns) >= 1

    def test_return_pixel_coords(self):
        raster = _make_raster(10, 10)
        terrain = np.zeros((10, 10), dtype=np.float32)
        world, pixel = _geojson_to_world_coords(
            [[4.0, 6.0]], raster, terrain, psx=1.0, psy=1.0,
            return_pixel_coords=True,
        )
        assert world.shape == (1, 3)
        assert pixel.shape == (1, 2)
        np.testing.assert_allclose(pixel[0, 0], 4.0, atol=1e-5)
        np.testing.assert_allclose(pixel[0, 1], 6.0, atol=1e-5)

    def test_nan_terrain_replaced_with_zero(self):
        raster = _make_raster(10, 10)
        terrain = np.full((10, 10), np.nan, dtype=np.float32)
        result = _geojson_to_world_coords(
            [[5.0, 5.0]], raster, terrain, psx=1.0, psy=1.0,
        )
        assert result[0, 2] == 0.0

    def test_1d_input(self):
        """Single [lon, lat] pair (not nested) should be reshaped."""
        raster = _make_raster(10, 10)
        terrain = np.zeros((10, 10), dtype=np.float32)
        result = _geojson_to_world_coords(
            [3.0, 4.0], raster, terrain, psx=1.0, psy=1.0,
        )
        assert result.shape == (1, 3)


# ---------------------------------------------------------------------------
# _linestring_to_wall_mesh
# ---------------------------------------------------------------------------

class TestLinestringToWallMesh:
    def test_basic_wall(self):
        pts = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], dtype=np.float32)
        v, idx = _linestring_to_wall_mesh(pts, height=5.0)
        # 3 points → 6 verts, 2 segments → 8 tris
        assert len(v) == 3 * 2 * 3  # 6 verts × 3 components
        assert len(idx) == 2 * 4 * 3  # 2 segments × 4 tris × 3 indices

    def test_single_point(self):
        pts = np.array([[0, 0, 0]], dtype=np.float32)
        v, idx = _linestring_to_wall_mesh(pts, height=5.0)
        assert len(v) == 0
        assert len(idx) == 0

    def test_closed_mode(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32)
        v, idx = _linestring_to_wall_mesh(pts, height=2.0, closed=True)
        # 3 pts + 1 closing → 4 pts, 8 verts, 3 segments → 12 tris
        assert len(v) == 4 * 2 * 3
        assert len(idx) == 3 * 4 * 3

    def test_height_affects_top_z(self):
        pts = np.array([[0, 0, 10], [5, 0, 10]], dtype=np.float32)
        v, _ = _linestring_to_wall_mesh(pts, height=7.0)
        verts = v.reshape(-1, 3)
        # Top vertices should be at z = 10 + 7 = 17
        top_z = verts[1::2, 2]  # odd indices are top vertices
        np.testing.assert_allclose(top_z, 17.0)

    def test_two_points(self):
        pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        v, idx = _linestring_to_wall_mesh(pts, height=1.0)
        assert len(v) == 2 * 2 * 3
        assert len(idx) == 1 * 4 * 3


# ---------------------------------------------------------------------------
# _polygon_to_mesh
# ---------------------------------------------------------------------------

class TestPolygonToMesh:
    def test_exterior_ring(self):
        ring = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=np.float32)
        v, idx = _polygon_to_mesh([ring], height=5.0)
        assert len(v) > 0
        assert len(idx) > 0

    def test_exterior_and_hole(self):
        exterior = np.array([[0, 0, 0], [20, 0, 0], [20, 20, 0], [0, 20, 0]], dtype=np.float32)
        hole = np.array([[5, 5, 0], [15, 5, 0], [15, 15, 0], [5, 15, 0]], dtype=np.float32)
        v, idx = _polygon_to_mesh([exterior, hole], height=3.0)
        # Both rings produce walls
        v_ext, _ = _linestring_to_wall_mesh(exterior, 3.0, closed=True)
        v_hole, _ = _linestring_to_wall_mesh(hole, 3.0, closed=True)
        assert len(v) == len(v_ext) + len(v_hole)

    def test_empty_rings(self):
        ring = np.array([[0, 0, 0]], dtype=np.float32)  # too few points
        v, idx = _polygon_to_mesh([ring], height=5.0)
        assert len(v) == 0
        assert len(idx) == 0

    def test_all_empty_rings(self):
        v, idx = _polygon_to_mesh([], height=5.0)
        assert len(v) == 0
        assert len(idx) == 0


# ---------------------------------------------------------------------------
# _points_in_polygon
# ---------------------------------------------------------------------------

class TestPointsInPolygon:
    def _square_ring(self):
        return np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)

    def test_inside(self):
        ring = self._square_ring()
        pts = np.array([[5, 5]], dtype=np.float64)
        mask = _points_in_polygon(pts, ring)
        assert mask[0]

    def test_outside(self):
        ring = self._square_ring()
        pts = np.array([[15, 5]], dtype=np.float64)
        mask = _points_in_polygon(pts, ring)
        assert not mask[0]

    def test_on_edge(self):
        """Points exactly on edges are implementation-defined; just check no crash."""
        ring = self._square_ring()
        pts = np.array([[0, 5], [5, 0]], dtype=np.float64)
        mask = _points_in_polygon(pts, ring)
        assert mask.shape == (2,)

    def test_multiple_points(self):
        ring = self._square_ring()
        pts = np.array([[5, 5], [15, 5], [5, 15]], dtype=np.float64)
        mask = _points_in_polygon(pts, ring)
        assert mask[0]       # inside
        assert not mask[1]   # outside
        assert not mask[2]   # outside

    def test_triangle(self):
        ring = np.array([[0, 0], [10, 0], [5, 10]], dtype=np.float64)
        pts = np.array([[5, 3]], dtype=np.float64)
        mask = _points_in_polygon(pts, ring)
        assert mask[0]

    def test_empty_points(self):
        ring = self._square_ring()
        pts = np.empty((0, 2), dtype=np.float64)
        mask = _points_in_polygon(pts, ring)
        assert len(mask) == 0


# ---------------------------------------------------------------------------
# _scatter_in_polygon
# ---------------------------------------------------------------------------

class TestScatterInPolygon:
    def test_basic_grid(self):
        ring = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
        result = _scatter_in_polygon(ring, spacing=5.0, H=20, W=20)
        # Should have some points inside the 10×10 square with spacing 5
        assert result.shape[1] == 2
        assert len(result) > 0

    def test_empty_polygon(self):
        """Polygon smaller than spacing → no fill points."""
        ring = np.array([[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]], dtype=np.float64)
        result = _scatter_in_polygon(ring, spacing=5.0, H=100, W=100)
        assert len(result) == 0

    def test_large_polygon_warning(self):
        """Large polygon with tiny spacing → warning about many instances."""
        ring = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.float64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _scatter_in_polygon(ring, spacing=1.0, H=2000, W=2000)
            fills = [x for x in w if "instances" in str(x.message)]
            if len(result) > 10000:
                assert len(fills) >= 1

    def test_clipped_to_raster_bounds(self):
        """Polygon extends beyond raster → clipped to H, W."""
        ring = np.array([[-10, -10], [100, -10], [100, 100], [-10, 100]], dtype=np.float64)
        result = _scatter_in_polygon(ring, spacing=5.0, H=20, W=20)
        # All points should be within [0, W-1] x [0, H-1]
        if len(result) > 0:
            assert result[:, 0].min() >= 0
            assert result[:, 0].max() <= 19
            assert result[:, 1].min() >= 0
            assert result[:, 1].max() <= 19


# ---------------------------------------------------------------------------
# _make_marker_cube
# ---------------------------------------------------------------------------

class TestMakeMarkerCube:
    def test_default_size(self):
        v, idx = _make_marker_cube()
        assert v.dtype == np.float32
        assert idx.dtype == np.int32
        # 8 vertices × 3 = 24 floats
        assert len(v) == 24
        # 12 triangles × 3 = 36 indices
        assert len(idx) == 36

    def test_custom_size(self):
        v, idx = _make_marker_cube(size=5.0)
        verts = v.reshape(-1, 3)
        # Half-size = 2.5
        np.testing.assert_allclose(verts[:, 0].min(), -2.5)
        np.testing.assert_allclose(verts[:, 0].max(), 2.5)
        # Base at Z=0, top at Z=5
        np.testing.assert_allclose(verts[:, 2].min(), 0.0)
        np.testing.assert_allclose(verts[:, 2].max(), 5.0)

    def test_indices_in_range(self):
        v, idx = _make_marker_cube(size=2.0)
        n_verts = len(v) // 3
        assert idx.min() >= 0
        assert idx.max() < n_verts


# ---------------------------------------------------------------------------
# _build_transformer
# ---------------------------------------------------------------------------

class TestBuildTransformer:
    def test_no_crs(self):
        raster = _make_raster()
        assert _build_transformer(raster) is None

    def test_no_rio_attribute(self):
        raster = _make_raster()
        # Ensure no rio accessor mishap
        assert _build_transformer(raster) is None
