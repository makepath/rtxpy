"""Render graph demo — build a multi-pass pipeline with capability gating.

Shows how to define custom render passes, wire them into a graph, and let the
graph handle execution order and fallback wiring when optional passes are
unavailable.

This example runs on CPU (numpy) and doesn't need a GPU.
"""

import numpy as np

from rtxpy import BufferDesc, RenderGraph, RenderPass

# Buffer descriptors for the pipeline
RGB = BufferDesc(dtype="float32", channels=3, per_pixel=True)
SCALAR = BufferDesc(dtype="float32", channels=1, per_pixel=True)


# --- Pass definitions --------------------------------------------------------


class GBufferPass(RenderPass):
    """Simulate a GBuffer pass that produces albedo, normals, and depth."""

    def __init__(self):
        super().__init__(
            "gbuffer",
            outputs={"albedo": RGB, "normal": RGB, "depth": SCALAR},
        )

    def execute(self, buffers):
        h, w, _ = buffers["albedo"].shape
        # Checkerboard albedo
        yy, xx = np.mgrid[:h, :w]
        checker = ((xx // 8 + yy // 8) % 2).astype(np.float32)
        buffers["albedo"][:, :, 0] = 0.2 + 0.6 * checker
        buffers["albedo"][:, :, 1] = 0.3 + 0.3 * checker
        buffers["albedo"][:, :, 2] = 0.1 + 0.2 * (1 - checker)

        # Upward-facing normals
        buffers["normal"][:] = [0.0, 0.0, 1.0]

        # Linear depth gradient
        buffers["depth"][:, :] = np.linspace(0.0, 1.0, w, dtype=np.float32)


class ShadowPass(RenderPass):
    """Compute a simple shadow mask from depth."""

    def __init__(self):
        super().__init__(
            "shadow",
            inputs={"depth": SCALAR},
            outputs={"shadow_mask": SCALAR},
        )

    def execute(self, buffers):
        # Fake shadow: darker where depth > 0.5
        buffers["shadow_mask"][:] = np.where(
            buffers["depth"] > 0.5, 0.4, 1.0
        ).astype(np.float32)


class AOPass(RenderPass):
    """Fake ambient occlusion from depth edges."""

    def __init__(self):
        super().__init__(
            "ao",
            inputs={"depth": SCALAR},
            outputs={"ao_map": SCALAR},
        )

    def execute(self, buffers):
        depth = buffers["depth"]
        # Approximate AO by depth variance in a 3x3 window
        padded = np.pad(depth, ((1, 1), (1, 1)), mode="edge")
        ao = np.ones_like(depth)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ao -= 0.02 * np.abs(
                    padded[1 + dy : depth.shape[0] + 1 + dy,
                           1 + dx : depth.shape[1] + 1 + dx]
                    - depth
                )
        buffers["ao_map"][:] = np.clip(ao, 0.3, 1.0)


class ShadePass(RenderPass):
    """Combine albedo, shadow, and AO into a lit color buffer."""

    def __init__(self):
        super().__init__(
            "shade",
            inputs={"albedo": RGB, "shadow_mask": SCALAR, "ao_map": SCALAR},
            outputs={"color": RGB},
        )

    def execute(self, buffers):
        albedo = buffers["albedo"]
        shadow = buffers["shadow_mask"][:, :, np.newaxis]
        ao = buffers["ao_map"][:, :, np.newaxis]
        buffers["color"][:] = albedo * shadow * ao


class DenoisePass(RenderPass):
    """Placeholder denoiser — requires 'optix_denoiser' capability."""

    def __init__(self):
        super().__init__(
            "denoise",
            inputs={"color": RGB, "albedo": RGB, "normal": RGB},
            outputs={"denoised_color": RGB},
            requires=["optix_denoiser"],
        )

    def execute(self, buffers):
        # Real implementation would call OptiX denoiser
        buffers["denoised_color"][:] = buffers["color"]


class TonemapPass(RenderPass):
    """Simple Reinhard tone mapping."""

    def __init__(self):
        super().__init__(
            "tonemap",
            inputs={"denoised_color": RGB},
            outputs={"ldr_color": RGB},
        )

    def execute(self, buffers):
        hdr = buffers["denoised_color"]
        buffers["ldr_color"][:] = hdr / (1.0 + hdr)


# --- Build and run the graph ------------------------------------------------


def main():
    width, height = 128, 96

    graph = RenderGraph(width=width, height=height)
    graph.add_pass(GBufferPass())
    graph.add_pass(ShadowPass())
    graph.add_pass(AOPass())
    graph.add_pass(ShadePass())
    graph.add_pass(DenoisePass())
    graph.add_pass(TonemapPass())

    # If denoiser is unavailable, tonemap reads 'color' directly
    graph.set_fallback("denoised_color", "color")

    # --- Run without denoiser ---
    print("Compiling graph WITHOUT denoiser capability...")
    compiled = graph.compile(capabilities={})
    print(f"  Active passes: {[p.name for p in compiled.ordered_passes]}")
    print(f"  Buffer pool slots: {compiled.allocation_plan.num_slots}")

    result = compiled.execute(
        allocator=lambda shape, dtype: np.zeros(shape, dtype=dtype)
    )
    ldr = result["ldr_color"]
    print(f"  Output shape: {ldr.shape}, range: [{ldr.min():.3f}, {ldr.max():.3f}]")

    # --- Run with denoiser ---
    print("\nCompiling graph WITH denoiser capability...")
    compiled2 = graph.compile(capabilities={"optix_denoiser": True})
    print(f"  Active passes: {[p.name for p in compiled2.ordered_passes]}")

    result2 = compiled2.execute(
        allocator=lambda shape, dtype: np.zeros(shape, dtype=dtype)
    )
    ldr2 = result2["ldr_color"]
    print(f"  Output shape: {ldr2.shape}, range: [{ldr2.min():.3f}, {ldr2.max():.3f}]")

    # Save to PNG if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(np.clip(result["ldr_color"], 0, 1))
        axes[0].set_title("Without denoiser")
        axes[0].axis("off")
        axes[1].imshow(np.clip(result2["ldr_color"], 0, 1))
        axes[1].set_title("With denoiser")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig("render_graph_demo.png", dpi=150)
        print("\nSaved render_graph_demo.png")
    except ImportError:
        print("\nmatplotlib not available, skipping image save")


if __name__ == "__main__":
    main()
