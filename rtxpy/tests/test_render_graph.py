"""Tests for the render graph framework (issue #73)."""

import numpy as np
import pytest

from rtxpy.render_graph import (
    AllocationPlan,
    BufferDesc,
    CompiledGraph,
    GraphValidationError,
    RenderGraph,
    RenderPass,
)


# ---------------------------------------------------------------------------
# Helpers — concrete pass implementations for testing
# ---------------------------------------------------------------------------


class StubPass(RenderPass):
    """Minimal concrete pass that records execute calls."""

    def __init__(self, name, inputs=None, outputs=None, **kwargs):
        super().__init__(name, inputs, outputs, **kwargs)
        self.executed = False
        self.exec_order = -1

    def execute(self, buffers):
        self.executed = True


class WritingPass(StubPass):
    """Writes a constant into its output buffer for verification."""

    def __init__(self, name, inputs=None, outputs=None, value=1.0, **kwargs):
        super().__init__(name, inputs, outputs, **kwargs)
        self.value = value

    def execute(self, buffers):
        super().execute(buffers)
        for buf_name in self.outputs:
            buffers[buf_name][:] = self.value


class SummingPass(StubPass):
    """Sums all input buffers into the output for verification."""

    def __init__(self, name, inputs=None, outputs=None, **kwargs):
        super().__init__(name, inputs, outputs, **kwargs)

    def execute(self, buffers):
        super().execute(buffers)
        out_name = list(self.outputs)[0]
        buffers[out_name][:] = 0
        for buf_name in self.inputs:
            buffers[out_name] += buffers[buf_name]


# ---------------------------------------------------------------------------
# BufferDesc tests
# ---------------------------------------------------------------------------


class TestBufferDesc:
    def test_per_pixel_rgb(self):
        desc = BufferDesc(dtype="float32", channels=3, per_pixel=True)
        assert desc.shape(1920, 1080) == (1080, 1920, 3)

    def test_per_pixel_scalar(self):
        desc = BufferDesc(dtype="float32", channels=1, per_pixel=True)
        assert desc.shape(1920, 1080) == (1080, 1920)

    def test_per_ray_multi_channel(self):
        desc = BufferDesc(dtype="float32", channels=4, per_pixel=False)
        assert desc.shape(1920, 1080) == (1920 * 1080, 4)

    def test_per_ray_scalar(self):
        desc = BufferDesc(dtype="int32", channels=1, per_pixel=False)
        assert desc.shape(100, 50) == (5000,)

    def test_frozen(self):
        desc = BufferDesc()
        with pytest.raises(AttributeError):
            desc.dtype = "int32"

    def test_equality(self):
        a = BufferDesc(dtype="float32", channels=3)
        b = BufferDesc(dtype="float32", channels=3)
        assert a == b

    def test_inequality(self):
        a = BufferDesc(dtype="float32", channels=3)
        b = BufferDesc(dtype="float32", channels=4)
        assert a != b


# ---------------------------------------------------------------------------
# Pass management
# ---------------------------------------------------------------------------


class TestPassManagement:
    def test_add_and_list(self):
        g = RenderGraph(width=64, height=64)
        g.add_pass(StubPass("a"))
        g.add_pass(StubPass("b"))
        assert [p.name for p in g.passes] == ["a", "b"]

    def test_duplicate_name_raises(self):
        g = RenderGraph()
        g.add_pass(StubPass("a"))
        with pytest.raises(ValueError, match="already exists"):
            g.add_pass(StubPass("a"))

    def test_remove_pass(self):
        g = RenderGraph()
        g.add_pass(StubPass("a"))
        g.add_pass(StubPass("b"))
        g.remove_pass("a")
        assert [p.name for p in g.passes] == ["b"]

    def test_remove_missing_raises(self):
        g = RenderGraph()
        with pytest.raises(KeyError):
            g.remove_pass("nonexistent")

    def test_get_pass(self):
        g = RenderGraph()
        p = StubPass("a")
        g.add_pass(p)
        assert g.get_pass("a") is p


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def test_linear_chain(self):
        """A -> B -> C should execute in that order."""
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=8, height=8)
        g.add_pass(StubPass("A", outputs={"color": rgb}))
        g.add_pass(StubPass("B", inputs={"color": rgb}, outputs={"denoised": rgb}))
        g.add_pass(StubPass("C", inputs={"denoised": rgb}, outputs={"final": rgb}))
        compiled = g.compile()
        names = [p.name for p in compiled.ordered_passes]
        assert names == ["A", "B", "C"]

    def test_diamond_dependency(self):
        """A -> B, A -> C, B+C -> D."""
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=8, height=8)
        g.add_pass(StubPass("A", outputs={"color": rgb, "normal": rgb}))
        g.add_pass(StubPass("B", inputs={"color": rgb}, outputs={"ao": rgb}))
        g.add_pass(StubPass("C", inputs={"normal": rgb}, outputs={"shadow": rgb}))
        g.add_pass(
            StubPass("D", inputs={"ao": rgb, "shadow": rgb}, outputs={"final": rgb})
        )
        compiled = g.compile()
        names = [p.name for p in compiled.ordered_passes]
        assert names.index("A") < names.index("B")
        assert names.index("A") < names.index("C")
        assert names.index("B") < names.index("D")
        assert names.index("C") < names.index("D")

    def test_independent_passes_preserve_insertion_order(self):
        """Passes with no dependencies keep their insertion order."""
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=8, height=8)
        g.add_pass(StubPass("X", outputs={"x": rgb}))
        g.add_pass(StubPass("Y", outputs={"y": rgb}))
        g.add_pass(StubPass("Z", outputs={"z": rgb}))
        compiled = g.compile()
        names = [p.name for p in compiled.ordered_passes]
        assert names == ["X", "Y", "Z"]

    def test_cycle_detection(self):
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=8, height=8)
        g.add_pass(StubPass("A", inputs={"b_out": rgb}, outputs={"a_out": rgb}))
        g.add_pass(StubPass("B", inputs={"a_out": rgb}, outputs={"b_out": rgb}))
        with pytest.raises(GraphValidationError, match="Cycle"):
            g.compile()


# ---------------------------------------------------------------------------
# Capability gating
# ---------------------------------------------------------------------------


class TestCapabilityGating:
    def test_pass_disabled_when_capability_missing(self):
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=8, height=8)
        g.add_pass(StubPass("A", outputs={"color": rgb}))
        g.add_pass(
            StubPass(
                "denoiser",
                inputs={"color": rgb},
                outputs={"denoised": rgb},
                requires=["optix_denoiser"],
            )
        )
        # No capabilities -> denoiser skipped
        compiled = g.compile(capabilities={})
        names = [p.name for p in compiled.ordered_passes]
        assert "denoiser" not in names

    def test_pass_enabled_when_capability_present(self):
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=8, height=8)
        g.add_pass(StubPass("A", outputs={"color": rgb}))
        g.add_pass(
            StubPass(
                "denoiser",
                inputs={"color": rgb},
                outputs={"denoised": rgb},
                requires=["optix_denoiser"],
            )
        )
        compiled = g.compile(capabilities={"optix_denoiser": True})
        names = [p.name for p in compiled.ordered_passes]
        assert "denoiser" in names

    def test_manually_disabled_pass(self):
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=8, height=8)
        g.add_pass(StubPass("A", outputs={"color": rgb}, enabled=False))
        compiled = g.compile()
        assert len(compiled.ordered_passes) == 0


# ---------------------------------------------------------------------------
# Fallback wiring
# ---------------------------------------------------------------------------


class TestFallbackWiring:
    def test_fallback_when_producer_disabled(self):
        """When denoiser is skipped, 'denoised' falls back to 'color'."""
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=8, height=8)
        g.add_pass(WritingPass("shade", outputs={"color": rgb}, value=42.0))
        g.add_pass(
            StubPass(
                "denoise",
                inputs={"color": rgb},
                outputs={"denoised": rgb},
                requires=["optix_denoiser"],
            )
        )
        g.add_pass(StubPass("tonemap", inputs={"denoised": rgb}, outputs={"final": rgb}))
        g.set_fallback("denoised", "color")

        compiled = g.compile(capabilities={})
        names = [p.name for p in compiled.ordered_passes]
        assert "denoise" not in names
        assert "tonemap" in names

        result = compiled.execute(
            allocator=lambda shape, dtype: np.zeros(shape, dtype=dtype)
        )
        # 'denoised' should map to the same array as 'color'
        assert result["denoised"] is result["color"]

    def test_chained_fallbacks(self):
        """A -> B -> C fallback chain resolves correctly."""
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=4, height=4)
        g.add_pass(WritingPass("src", outputs={"raw": rgb}, value=1.0))
        g.add_pass(
            StubPass("mid", inputs={"raw": rgb}, outputs={"enhanced": rgb}, enabled=False)
        )
        g.add_pass(
            StubPass(
                "final_proc",
                inputs={"enhanced": rgb},
                outputs={"polished": rgb},
                enabled=False,
            )
        )
        g.add_pass(StubPass("out", inputs={"polished": rgb}, outputs={"display": rgb}))
        g.set_fallback("polished", "enhanced")
        g.set_fallback("enhanced", "raw")

        compiled = g.compile()
        result = compiled.execute(
            allocator=lambda shape, dtype: np.zeros(shape, dtype=dtype)
        )
        assert result["polished"] is result["raw"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_input_raises(self):
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=8, height=8)
        g.add_pass(StubPass("A", inputs={"nonexistent": rgb}, outputs={"out": rgb}))
        with pytest.raises(GraphValidationError, match="nonexistent"):
            g.compile()

    def test_missing_input_with_fallback_ok(self):
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=8, height=8)
        g.add_pass(WritingPass("src", outputs={"raw": rgb}))
        g.add_pass(StubPass("consumer", inputs={"missing": rgb}, outputs={"out": rgb}))
        g.set_fallback("missing", "raw")
        # Should not raise
        compiled = g.compile()
        assert len(compiled.ordered_passes) == 2


# ---------------------------------------------------------------------------
# Buffer lifetime analysis & allocation
# ---------------------------------------------------------------------------


class TestLifetimeAnalysis:
    def test_non_overlapping_buffers_share_slot(self):
        """Two buffers with non-overlapping lifetimes and same desc share a slot."""
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=8, height=8)
        g.add_pass(WritingPass("A", outputs={"buf_a": rgb}))
        g.add_pass(
            WritingPass("B", inputs={"buf_a": rgb}, outputs={"buf_b": rgb})
        )
        # buf_a is last read by B (idx 1), buf_b is first written by B (idx 1).
        # buf_a lifetime: [0, 1], buf_b lifetime: [1, end]
        # They overlap at index 1, so they should NOT share.
        compiled = g.compile()
        plan = compiled.allocation_plan
        assert plan.slots["buf_a"] != plan.slots["buf_b"]

    def test_truly_non_overlapping_share(self):
        """A produces X, B consumes X and produces Y, C consumes Y and produces Z.
        X and Z don't overlap -> should share a slot if same desc."""
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=8, height=8)
        g.add_pass(WritingPass("A", outputs={"X": rgb}))
        g.add_pass(WritingPass("B", inputs={"X": rgb}, outputs={"Y": rgb}))
        g.add_pass(WritingPass("C", inputs={"Y": rgb}, outputs={"Z": rgb}))
        compiled = g.compile()
        plan = compiled.allocation_plan
        # X: written at 0, last read at 1
        # Y: written at 1, last read at 2
        # Z: written at 2, last read at end (3)
        # X and Z don't overlap -> can share
        assert plan.slots["X"] == plan.slots["Z"]
        assert plan.slots["X"] != plan.slots["Y"]

    def test_different_descs_dont_share(self):
        """Buffers with different descriptors never share a slot."""
        rgb = BufferDesc(dtype="float32", channels=3)
        scalar = BufferDesc(dtype="int32", channels=1)
        g = RenderGraph(width=8, height=8)
        g.add_pass(WritingPass("A", outputs={"X": rgb}))
        g.add_pass(WritingPass("B", inputs={"X": rgb}, outputs={"Y": scalar}))
        g.add_pass(WritingPass("C", inputs={"Y": scalar}, outputs={"Z": rgb}))
        compiled = g.compile()
        plan = compiled.allocation_plan
        # Even though X and Z don't overlap, Z is rgb and could share with X
        # but Y is scalar, so it gets its own slot
        assert plan.slots["Y"] != plan.slots["X"]
        assert plan.slots["Y"] != plan.slots["Z"]

    def test_allocation_plan_num_slots(self):
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=8, height=8)
        g.add_pass(WritingPass("A", outputs={"a": rgb}))
        g.add_pass(WritingPass("B", outputs={"b": rgb}))
        compiled = g.compile()
        # Two independent buffers, both alive until end -> 2 slots
        assert compiled.allocation_plan.num_slots == 2


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class TestExecution:
    def test_simple_pipeline(self):
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=4, height=4)
        g.add_pass(WritingPass("producer", outputs={"color": rgb}, value=7.0))
        g.add_pass(
            SummingPass("consumer", inputs={"color": rgb}, outputs={"result": rgb})
        )

        compiled = g.compile()
        result = compiled.execute(
            allocator=lambda shape, dtype: np.zeros(shape, dtype=dtype)
        )
        np.testing.assert_allclose(result["result"], 7.0)

    def test_external_buffers(self):
        """External buffers are available to passes without allocation."""
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=4, height=4)

        class ReaderPass(StubPass):
            def execute(self, buffers):
                super().execute(buffers)
                for buf_name in self.outputs:
                    buffers[buf_name][:] = buffers["scene_handle"] * 2

        g.add_pass(ReaderPass("shade", inputs={"scene_handle": rgb}, outputs={"color": rgb}))

        # Inject an external buffer (simulating an OptiX scene handle)
        ext = np.full((4, 4, 3), 3.0, dtype=np.float32)
        compiled = g.compile(validate=False)  # scene_handle has no producer
        result = compiled.execute(
            external_buffers={"scene_handle": ext},
            allocator=lambda shape, dtype: np.zeros(shape, dtype=dtype),
        )
        np.testing.assert_allclose(result["color"], 6.0)

    def test_all_passes_execute(self):
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=4, height=4)
        p1 = StubPass("A", outputs={"x": rgb})
        p2 = StubPass("B", inputs={"x": rgb}, outputs={"y": rgb})
        g.add_pass(p1)
        g.add_pass(p2)
        compiled = g.compile()
        compiled.execute(
            allocator=lambda shape, dtype: np.zeros(shape, dtype=dtype)
        )
        assert p1.executed
        assert p2.executed


# ---------------------------------------------------------------------------
# RenderPass interface
# ---------------------------------------------------------------------------


class TestRenderPassInterface:
    def test_repr(self):
        p = StubPass("test", inputs={"a": BufferDesc()}, outputs={"b": BufferDesc()})
        r = repr(p)
        assert "test" in r
        assert "inputs" in r

    def test_setup_and_teardown_called(self):
        class TrackedPass(StubPass):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.setup_called = False
                self.teardown_called = False

            def setup(self, graph):
                self.setup_called = True

            def teardown(self):
                self.teardown_called = True

        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=4, height=4)
        p = TrackedPass("t", outputs={"out": rgb})
        g.add_pass(p)
        compiled = g.compile()
        assert p.setup_called
        # teardown is user-called
        p.teardown()
        assert p.teardown_called


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_graph(self):
        g = RenderGraph(width=8, height=8)
        compiled = g.compile()
        assert compiled.ordered_passes == []
        result = compiled.execute(
            allocator=lambda shape, dtype: np.zeros(shape, dtype=dtype)
        )
        assert result == {}

    def test_single_pass_no_deps(self):
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=4, height=4)
        g.add_pass(WritingPass("only", outputs={"out": rgb}, value=99.0))
        compiled = g.compile()
        result = compiled.execute(
            allocator=lambda shape, dtype: np.zeros(shape, dtype=dtype)
        )
        np.testing.assert_allclose(result["out"], 99.0)

    def test_pass_self_loop_is_not_cycle(self):
        """A pass that reads and writes the same buffer (in-place) is fine."""
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=4, height=4)
        g.add_pass(WritingPass("src", outputs={"buf": rgb}))
        g.add_pass(StubPass("inplace", inputs={"buf": rgb}, outputs={"buf": rgb}))
        compiled = g.compile()
        assert len(compiled.ordered_passes) == 2

    def test_wide_fan_out(self):
        """One producer, many consumers."""
        rgb = BufferDesc(channels=3)
        g = RenderGraph(width=4, height=4)
        g.add_pass(WritingPass("src", outputs={"shared": rgb}, value=1.0))
        for i in range(10):
            g.add_pass(
                SummingPass(f"consumer_{i}", inputs={"shared": rgb}, outputs={f"out_{i}": rgb})
            )
        compiled = g.compile()
        result = compiled.execute(
            allocator=lambda shape, dtype: np.zeros(shape, dtype=dtype)
        )
        for i in range(10):
            np.testing.assert_allclose(result[f"out_{i}"], 1.0)

    def test_allocation_plan_empty_graph(self):
        plan = AllocationPlan()
        assert plan.num_slots == 0


# ---------------------------------------------------------------------------
# Integration-style: realistic pipeline shape
# ---------------------------------------------------------------------------


class TestRealisticPipeline:
    def test_full_pipeline_shape(self):
        """Mirror the proposed pass structure from issue #73."""
        rgb = BufferDesc(dtype="float32", channels=3, per_pixel=True)
        scalar = BufferDesc(dtype="float32", channels=1, per_pixel=True)
        vec4 = BufferDesc(dtype="float32", channels=4, per_pixel=False)
        mat_id = BufferDesc(dtype="int32", channels=1, per_pixel=False)

        g = RenderGraph(width=64, height=64)

        g.add_pass(
            WritingPass(
                "gbuffer",
                outputs={
                    "albedo": rgb,
                    "normal": rgb,
                    "depth": scalar,
                    "position": rgb,
                    "material_id": mat_id,
                },
                value=0.5,
            )
        )
        g.add_pass(
            WritingPass(
                "shadow",
                inputs={"position": rgb, "normal": rgb},
                outputs={"shadow_mask": scalar},
                value=1.0,
            )
        )
        g.add_pass(
            WritingPass(
                "ao",
                inputs={"position": rgb, "normal": rgb, "depth": scalar},
                outputs={"ao_map": scalar},
                value=0.9,
            )
        )
        g.add_pass(
            WritingPass(
                "gi",
                inputs={
                    "position": rgb,
                    "normal": rgb,
                    "albedo": rgb,
                    "shadow_mask": scalar,
                },
                outputs={"indirect_light": rgb},
                value=0.2,
            )
        )
        g.add_pass(
            WritingPass(
                "denoise",
                inputs={"color": rgb, "albedo": rgb, "normal": rgb},
                outputs={"denoised_color": rgb},
                requires=["optix_denoiser"],
                value=0.6,
            )
        )
        g.add_pass(
            WritingPass(
                "tonemap",
                inputs={"denoised_color": rgb},
                outputs={"ldr_color": rgb},
                value=0.8,
            )
        )
        g.add_pass(
            StubPass(
                "composite",
                inputs={"ldr_color": rgb},
                outputs={"final": rgb},
            )
        )

        # The shade pass produces 'color' consumed by denoise
        g.add_pass(
            WritingPass(
                "shade",
                inputs={
                    "albedo": rgb,
                    "shadow_mask": scalar,
                    "ao_map": scalar,
                    "indirect_light": rgb,
                },
                outputs={"color": rgb},
                value=0.7,
            )
        )

        g.set_fallback("denoised_color", "color")

        # Compile without denoiser capability
        compiled = g.compile(capabilities={})
        names = [p.name for p in compiled.ordered_passes]
        assert "denoise" not in names
        assert "gbuffer" in names
        assert "tonemap" in names

        # gbuffer must come before shadow, ao
        assert names.index("gbuffer") < names.index("shadow")
        assert names.index("gbuffer") < names.index("ao")
        # shade must come after shadow, ao, gi
        assert names.index("shade") > names.index("shadow")
        assert names.index("shade") > names.index("ao")
        assert names.index("shade") > names.index("gi")
        # tonemap after shade (via fallback denoised_color -> color)
        assert names.index("tonemap") > names.index("shade")

        result = compiled.execute(
            allocator=lambda shape, dtype: np.zeros(shape, dtype=dtype)
        )
        # denoised_color falls back to color (which shade wrote as 0.7)
        assert result["denoised_color"] is result["color"]
        np.testing.assert_allclose(result["color"], 0.7)

        # Now compile WITH denoiser
        compiled2 = g.compile(capabilities={"optix_denoiser": True})
        names2 = [p.name for p in compiled2.ordered_passes]
        assert "denoise" in names2
        assert names2.index("shade") < names2.index("denoise")
        assert names2.index("denoise") < names2.index("tonemap")
