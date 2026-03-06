"""Render graph: a lightweight DAG of render passes with automatic
dependency resolution, buffer lifetime analysis, and capability gating.

Usage::

    graph = RenderGraph(width=1920, height=1080)
    graph.add_pass(GBufferPass(...))
    graph.add_pass(ShadowPass(...))
    graph.add_pass(DenoisePass(...))
    graph.add_pass(TonemapPass(...))

    compiled = graph.compile(capabilities=get_capabilities())
    result = compiled.execute()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class BufferDesc:
    """Describes a GPU buffer's shape and data type.

    Parameters
    ----------
    dtype : str
        NumPy-compatible dtype string (e.g. ``'float32'``, ``'int32'``).
    channels : int
        Number of channels per element. For a per-pixel RGB buffer this is 3;
        for a scalar per-ray buffer this is 1.
    per_pixel : bool
        If True the buffer shape is ``(height, width, channels)`` (image-like).
        If False the shape is ``(width * height, channels)`` (per-ray flat).
    """

    dtype: str = "float32"
    channels: int = 3
    per_pixel: bool = True

    def shape(self, width: int, height: int) -> tuple[int, ...]:
        if self.per_pixel:
            if self.channels == 1:
                return (height, width)
            return (height, width, self.channels)
        n = width * height
        if self.channels == 1:
            return (n,)
        return (n, self.channels)


class RenderPass(ABC):
    """Abstract base for a single render pass.

    Subclasses declare their buffer *inputs* and *outputs* so the graph can
    resolve execution order and manage GPU memory.

    Parameters
    ----------
    name : str
        Unique name for this pass (e.g. ``"gbuffer"``, ``"shadow"``).
    inputs : dict[str, BufferDesc]
        Buffers this pass reads.  Keys are buffer names (globally unique within
        the graph).
    outputs : dict[str, BufferDesc]
        Buffers this pass writes.
    enabled : bool
        If False the pass is skipped during compilation.
    requires : list[str]
        Capability keys (from :func:`rtxpy.get_capabilities`) that must be
        truthy for this pass to run.  The graph disables the pass automatically
        when requirements aren't met.
    """

    def __init__(
        self,
        name: str,
        inputs: dict[str, BufferDesc] | None = None,
        outputs: dict[str, BufferDesc] | None = None,
        *,
        enabled: bool = True,
        requires: list[str] | None = None,
    ):
        self.name = name
        self.inputs: dict[str, BufferDesc] = inputs or {}
        self.outputs: dict[str, BufferDesc] = outputs or {}
        self.enabled = enabled
        self.requires: list[str] = requires or []

    def setup(self, graph: RenderGraph) -> None:
        """Called once when the graph is compiled. Override to allocate
        one-time resources."""

    @abstractmethod
    def execute(self, buffers: dict[str, Any]) -> None:
        """Run this pass.

        *buffers* maps buffer names (both inputs and outputs) to allocated GPU
        arrays.  The pass should read its declared inputs and write its
        declared outputs.
        """

    def teardown(self) -> None:
        """Called when the graph is torn down. Override to free resources."""

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"inputs={list(self.inputs)}, outputs={list(self.outputs)}, "
            f"enabled={self.enabled})"
        )


class GraphValidationError(Exception):
    """Raised when the render graph fails validation."""


@dataclass
class _BufferLifetime:
    """Tracks first-write and last-read pass indices for a buffer."""

    first_write: int = -1
    last_read: int = -1


@dataclass
class AllocationPlan:
    """The result of buffer lifetime analysis.

    Maps each buffer name to a *pool slot* index.  Buffers assigned to the same
    slot have non-overlapping lifetimes and can share the same GPU allocation.
    """

    slots: dict[str, int] = field(default_factory=dict)
    slot_descs: dict[int, BufferDesc] = field(default_factory=dict)

    @property
    def num_slots(self) -> int:
        if not self.slot_descs:
            return 0
        return max(self.slot_descs) + 1


@dataclass
class CompiledGraph:
    """A compiled, ready-to-execute render graph.

    Produced by :meth:`RenderGraph.compile`.
    """

    ordered_passes: list[RenderPass]
    allocation_plan: AllocationPlan
    buffer_descs: dict[str, BufferDesc]
    fallback_map: dict[str, str]
    width: int
    height: int

    def execute(
        self,
        external_buffers: dict[str, Any] | None = None,
        allocator: Callable[[tuple[int, ...], str], Any] | None = None,
    ) -> dict[str, Any]:
        """Execute all passes in topological order.

        Parameters
        ----------
        external_buffers : dict, optional
            Pre-allocated buffers to inject (e.g. the OptiX scene handle).
            These are available to passes but are not managed by the graph.
        allocator : callable, optional
            ``allocator(shape, dtype) -> array``.  Defaults to
            ``cupy.zeros`` if CuPy is available, else ``numpy.zeros``.

        Returns
        -------
        dict[str, array]
            All live buffers after execution completes.
        """
        if allocator is None:
            allocator = _default_allocator()

        buffers: dict[str, Any] = dict(external_buffers or {})

        # Allocate managed buffers grouped by pool slot
        pool: dict[int, Any] = {}
        for buf_name, slot_idx in self.allocation_plan.slots.items():
            if buf_name in buffers:
                continue  # external
            if slot_idx not in pool:
                desc = self.allocation_plan.slot_descs[slot_idx]
                pool[slot_idx] = allocator(
                    desc.shape(self.width, self.height), desc.dtype
                )
            buffers[buf_name] = pool[slot_idx]

        # Also allocate any buffers not in the allocation plan (outputs of
        # active passes that somehow escaped lifetime analysis — shouldn't
        # happen, but defensive).
        for buf_name, desc in self.buffer_descs.items():
            if buf_name not in buffers:
                buffers[buf_name] = allocator(
                    desc.shape(self.width, self.height), desc.dtype
                )

        # Apply fallback wiring: if a buffer is absent (producer was disabled),
        # point it at the fallback buffer.
        for buf_name, fallback in self.fallback_map.items():
            if buf_name not in buffers and fallback in buffers:
                buffers[buf_name] = buffers[fallback]

        for pass_ in self.ordered_passes:
            pass_.execute(buffers)

        return buffers


class RenderGraph:
    """A configurable DAG of render passes.

    Passes declare typed buffer inputs/outputs.  The graph resolves execution
    order via topological sort, detects cycles, performs buffer lifetime
    analysis for memory reuse, and gates passes on runtime GPU capabilities.
    """

    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self._passes: dict[str, RenderPass] = {}
        self._insertion_order: list[str] = []
        self._fallbacks: dict[str, str] = {}

    # -- Pass management ---------------------------------------------------

    def add_pass(self, pass_: RenderPass) -> None:
        """Add a render pass to the graph.

        Raises
        ------
        ValueError
            If a pass with the same name already exists.
        """
        if pass_.name in self._passes:
            raise ValueError(f"Pass {pass_.name!r} already exists in the graph")
        self._passes[pass_.name] = pass_
        self._insertion_order.append(pass_.name)

    def remove_pass(self, name: str) -> None:
        """Remove a render pass by name.

        Raises
        ------
        KeyError
            If no pass with that name exists.
        """
        if name not in self._passes:
            raise KeyError(f"No pass named {name!r}")
        del self._passes[name]
        self._insertion_order.remove(name)

    def get_pass(self, name: str) -> RenderPass:
        return self._passes[name]

    @property
    def passes(self) -> list[RenderPass]:
        return [self._passes[n] for n in self._insertion_order]

    def set_fallback(self, buffer: str, fallback: str) -> None:
        """Register a fallback: if *buffer*'s producer is disabled, read
        *fallback* instead.

        Example: ``graph.set_fallback("denoised_color", "color")`` — if the
        denoise pass is skipped, downstream passes reading ``denoised_color``
        transparently receive ``color``.
        """
        self._fallbacks[buffer] = fallback

    # -- Compilation -------------------------------------------------------

    def compile(
        self,
        capabilities: dict[str, Any] | None = None,
        validate: bool = True,
    ) -> CompiledGraph:
        """Compile the graph into a ready-to-execute form.

        1. Capability-gate passes (disable those whose requirements aren't met).
        2. Resolve fallback wiring for disabled pass outputs.
        3. Topological sort by buffer dependencies.
        4. Buffer lifetime analysis and allocation planning.
        5. Validation (missing inputs, cycles).

        Parameters
        ----------
        capabilities : dict, optional
            Runtime capabilities from :func:`rtxpy.get_capabilities`.
        validate : bool
            If True, raise :class:`GraphValidationError` on problems.

        Returns
        -------
        CompiledGraph
        """
        caps = capabilities or {}
        active = self._gate_passes(caps)
        active_names = {p.name for p in active}

        # Collect all buffer descriptors (outputs define the canonical desc)
        buffer_descs: dict[str, BufferDesc] = {}
        for p in active:
            for buf_name, desc in p.outputs.items():
                buffer_descs[buf_name] = desc

        # Resolve fallbacks: find buffers consumed but not produced
        fallback_map = self._resolve_fallbacks(active, buffer_descs)

        # Build dependency edges
        producer: dict[str, str] = {}  # buffer_name -> pass_name
        for p in active:
            for buf_name in p.outputs:
                producer[buf_name] = p.name

        adj: dict[str, list[str]] = defaultdict(list)  # pass -> [deps]
        for p in active:
            for buf_name in p.inputs:
                resolved = fallback_map.get(buf_name, buf_name)
                if resolved in producer:
                    dep_pass = producer[resolved]
                    if dep_pass != p.name:
                        adj[p.name].append(dep_pass)

        # Topological sort (Kahn's algorithm)
        ordered = self._topological_sort(active, adj)

        if validate:
            self._validate(ordered, buffer_descs, fallback_map, producer)

        # Buffer lifetime analysis
        allocation_plan = self._lifetime_analysis(ordered, fallback_map)

        # Call setup on each pass
        for p in ordered:
            p.setup(self)

        return CompiledGraph(
            ordered_passes=ordered,
            allocation_plan=allocation_plan,
            buffer_descs=buffer_descs,
            fallback_map=fallback_map,
            width=self.width,
            height=self.height,
        )

    # -- Internal helpers --------------------------------------------------

    def _gate_passes(self, capabilities: dict[str, Any]) -> list[RenderPass]:
        """Return only passes that are enabled and whose capability
        requirements are met."""
        active: list[RenderPass] = []
        for name in self._insertion_order:
            p = self._passes[name]
            if not p.enabled:
                continue
            if p.requires and not all(capabilities.get(r) for r in p.requires):
                continue
            active.append(p)
        return active

    def _resolve_fallbacks(
        self,
        active: list[RenderPass],
        buffer_descs: dict[str, BufferDesc],
    ) -> dict[str, str]:
        """Build the fallback map for buffers whose producers are inactive."""
        produced = set(buffer_descs)
        consumed: set[str] = set()
        for p in active:
            consumed.update(p.inputs)

        fallback_map: dict[str, str] = {}
        for buf in consumed - produced:
            fb = buf
            visited: set[str] = set()
            while fb in self._fallbacks and fb not in produced:
                if fb in visited:
                    break  # avoid cycles in fallback chain
                visited.add(fb)
                fb = self._fallbacks[fb]
            if fb != buf:
                fallback_map[buf] = fb
        return fallback_map

    def _topological_sort(
        self,
        active: list[RenderPass],
        adj: dict[str, list[str]],
    ) -> list[RenderPass]:
        """Kahn's algorithm.  Stable: breaks ties by insertion order."""
        in_degree: dict[str, int] = {p.name: 0 for p in active}
        reverse_adj: dict[str, list[str]] = defaultdict(list)

        for node, deps in adj.items():
            for dep in deps:
                if dep in in_degree:
                    reverse_adj[dep].append(node)
                    in_degree[node] = in_degree.get(node, 0) + 1

        # Seed queue with zero-in-degree passes, ordered by insertion order
        order_idx = {name: i for i, name in enumerate(self._insertion_order)}
        queue = sorted(
            [p.name for p in active if in_degree.get(p.name, 0) == 0],
            key=lambda n: order_idx.get(n, 0),
        )

        result: list[str] = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in sorted(
                reverse_adj.get(node, []),
                key=lambda n: order_idx.get(n, 0),
            ):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(active):
            in_cycle = {p.name for p in active} - set(result)
            raise GraphValidationError(
                f"Cycle detected among passes: {in_cycle}"
            )

        pass_map = {p.name: p for p in active}
        return [pass_map[n] for n in result]

    def _lifetime_analysis(
        self,
        ordered: list[RenderPass],
        fallback_map: dict[str, str],
    ) -> AllocationPlan:
        """Compute buffer lifetimes and assign pool slots for memory reuse."""
        lifetimes: dict[str, _BufferLifetime] = {}
        descs: dict[str, BufferDesc] = {}

        for idx, p in enumerate(ordered):
            for buf_name, desc in p.outputs.items():
                if buf_name not in lifetimes:
                    lifetimes[buf_name] = _BufferLifetime()
                lifetimes[buf_name].first_write = idx
                descs[buf_name] = desc

            for buf_name in p.inputs:
                resolved = fallback_map.get(buf_name, buf_name)
                if resolved not in lifetimes:
                    lifetimes[resolved] = _BufferLifetime()
                lifetimes[resolved].last_read = max(
                    lifetimes[resolved].last_read, idx
                )

        # Extend last_read for buffers never explicitly read (keep alive
        # until end so they appear in the final output dict).
        num_passes = len(ordered)
        for buf, lt in lifetimes.items():
            if lt.last_read < lt.first_write:
                lt.last_read = num_passes

        # Greedy interval-colouring: assign buffers to pool slots such that
        # overlapping lifetimes get different slots.  Buffers can only share
        # a slot if they have the same shape & dtype.
        slots: dict[str, int] = {}
        slot_descs: dict[int, BufferDesc] = {}
        # slot_end[slot_idx] = last_read of current occupant
        slot_end: dict[int, int] = {}
        next_slot = 0

        # Sort buffers by first_write for deterministic allocation
        sorted_bufs = sorted(lifetimes, key=lambda b: lifetimes[b].first_write)

        for buf in sorted_bufs:
            lt = lifetimes[buf]
            desc = descs.get(buf)
            if desc is None:
                continue  # external buffer, skip

            # Try to reuse an existing slot whose occupant has expired
            reused = False
            for sid in sorted(slot_end):
                if (
                    slot_end[sid] < lt.first_write
                    and slot_descs[sid] == desc
                ):
                    slots[buf] = sid
                    slot_end[sid] = lt.last_read
                    reused = True
                    break

            if not reused:
                slots[buf] = next_slot
                slot_descs[next_slot] = desc
                slot_end[next_slot] = lt.last_read
                next_slot += 1

        return AllocationPlan(slots=slots, slot_descs=slot_descs)

    def _validate(
        self,
        ordered: list[RenderPass],
        buffer_descs: dict[str, BufferDesc],
        fallback_map: dict[str, str],
        producer: dict[str, str],
    ) -> None:
        """Check for missing inputs and warn on unused outputs."""
        produced = set(buffer_descs)
        produced.update(fallback_map.values())

        errors: list[str] = []
        for p in ordered:
            for buf_name in p.inputs:
                resolved = fallback_map.get(buf_name, buf_name)
                if resolved not in produced:
                    errors.append(
                        f"Pass {p.name!r} requires input {buf_name!r} "
                        f"but no active pass produces it"
                    )

        if errors:
            raise GraphValidationError("\n".join(errors))


def _default_allocator():
    """Return a zero-allocator: tries CuPy first, falls back to NumPy."""
    try:
        import cupy

        def _alloc(shape, dtype):
            return cupy.zeros(shape, dtype=dtype)

        return _alloc
    except ImportError:
        import numpy as np

        def _alloc(shape, dtype):
            return np.zeros(shape, dtype=dtype)

        return _alloc
