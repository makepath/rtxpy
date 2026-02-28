# rtxpy Roadmap

Evolution path from GPU ray-tracing geospatial engine to a real-time digital twin platform.

Tracked in [#57](https://github.com/makepath/rtxpy/issues/57).

---

## Current State

rtxpy is a geospatial visualization engine built on NVIDIA OptiX with:
- Hardware-accelerated ray tracing (triangles, B-spline curves, heightfields) with OptiX 9.1 cluster acceleration
- Interactive GLFW + Jupyter viewers with 30+ keyboard controls
- Multi-GAS scene architecture with per-geometry transforms and visibility
- GPU-accelerated GIS analysis (viewshed, hillshade, slope, aspect)
- Data integration: Overture Maps, OSM, GTFS, NASA FIRMS, NOAA wind, DEMs (Copernicus, USGS 3DEP)
- Keyframe animation, flyover tours, GIF/MP4 export

---

## Phase 1 — Architecture

Decompose the monolithic `InteractiveViewer` (6400+ lines, ~80 instance variables) into composed, testable subsystems. Approach: plain Python composition — no ECS, no scene graph trees, no event buses. Complexity is added only where profiling justifies it.

### Viewer Decomposition via Composition ([#61](https://github.com/makepath/rtxpy/issues/61))

Extract subsystem classes from the monolithic `InteractiveViewer`:

- **`CameraState`** — position, yaw, pitch, fov, speeds, `apply_input()`, `view_matrix()`
- **`TerrainState`** — raster data, mesh cache, resolution management, basemap cycling
- **`WindState`** — particle arrays, GPU buffers, `step()` simulation
- **`OverlayManager`** — ordered overlay layers, compositing, alpha control
- **`ObserverManager`** — observer slots, drone modes, viewshed state
- **`GeometryLayerManager`** — geometry grouping, visibility cycling, chunk loading (see #62)
- **`RenderSettings`** — shadows, AO, DOF, denoise, colormap, VE, sun position
- **`InputState`** — held keys, mouse state, modifiers
- **`HUDState`** — help overlay, minimap, FPS counter, title text

`InteractiveViewer` holds these as composed objects. `_tick()` delegates to subsystem methods. Selective `@njit` acceleration applied only to proven CPU hotspots (wind particle updates, bilinear Z-sampling).

### Geometry Layer Manager ([#62](https://github.com/makepath/rtxpy/issues/62))

Replace scattered geometry management variables with a `GeometryLayerManager`:

- Flat layer grouping with ordered visibility cycling — matches OptiX's flat IAS model
- `GeometryLayer` groups related GAS entries with shared visibility toggle
- Chunk managers owned by their respective layers
- Generic `cycle_index()` helper replaces 4 duplicated cycling implementations
- No scene graph tree — OptiX IAS is inherently flat, hardware BVH already handles frustum culling

### Declarative Key-Binding Map ([#63](https://github.com/makepath/rtxpy/issues/63))

Replace the 280-line if/elif chain in `_handle_key_press()`:

- Dict-based `(key, shift) → action` dispatch table
- Direct method calls to subsystem objects — no pub/sub event bus
- REPL command queue stays as-is (already correct)
- Help text auto-generation from binding table (stretch goal)

### Main Loop Cleanup ([#64](https://github.com/makepath/rtxpy/issues/64))

Fix phase ordering and separate concerns in the main loop:

- Move `glfw.poll_events()` before `_tick()` — fixes one-frame input lag
- Split `_tick()` into clear input / simulation / render sections
- Extract `_drain_command_queue()` and `_present_if_dirty()` methods
- Separate render (ray trace + post-process) from present (readback + composite) in `_update_frame()`
- No fixed-timestep accumulator (no physics), no Platform ABC (two backends don't justify it)

---

## Phase 2 — Rendering

Modern material and effects pipeline.

### PBR Materials
- Metallic-roughness workflow (glTF PBR model)
- Per-geometry material assignment
- Material library: concrete, glass, vegetation, water, asphalt, terrain variants

### Texture & UV Support
- GLB/glTF texture loading (baseColor, normal, metallic-roughness)
- UV coordinates via barycentric interpolation in closest-hit
- Terrain texture splatting, satellite imagery draping

### Render Graph
- Configurable DAG of render passes (GBuffer → Shadow → AO → GI → Denoise → Tonemap → Composite)
- Easy to add/remove passes per capability

### Level of Detail
- Distance-based terrain LOD (subsample far tiles)
- Instanced geometry LOD, billboard imposters
- Hybrid: cluster GAS close, simplified GAS far

### Particle Systems
- GPU particle simulation (CUDA compute kernels)
- Emitters: point, line, area, volume
- Applications: smoke/fire, rain/snow, traffic flow, pollution dispersion

---

## Phase 3 — Simulation

Time-evolving behavior.

### Physics
- Rigid body dynamics, fluid simulation on terrain
- Structural load visualization
- Integration: cuPhysics or bullet3 via CUDA interop
- Fixed-timestep accumulator pattern added here when physics demands it

### Animation System
- Property animation: any component value over time
- Skeletal animation (glTF), procedural animation (flag waving, tree sway)
- Timeline editor

### AI & Pathfinding
- Navigation mesh generation from terrain + buildings
- A*/Dijkstra pathfinding, agent simulation (pedestrians, vehicles, drones)
- Crowd simulation, GTFS integration for transit agents

### Environmental Simulation
- Solar exposure accumulation, shadow analysis (hours of sun per location)
- Noise propagation via ray tracing
- Line-of-sight network analysis, flood inundation modeling

---

## Phase 4 — Platform

Accessibility beyond a single Python process.

### Web Viewer
- Server-side rendering + frame streaming (WebRTC / WebSocket + H.264)
- Progressive resolution, thin browser client
- Long-term: WebGPU client with scene serialization

### Multi-User
- Shared scene state, per-user camera/selection
- Collaborative annotation, role-based access

### API Layer
- REST/GraphQL for scene management
- Python client library (same interface as local accessor)
- Webhook/event streams, batch rendering API

### IoT & Sensor Integration
- MQTT/AMQP subscriber → entity binding → visual update
- Time-series storage and playback, alert rules

---

## Phase 5 — Digital Twin

Domain-specific capabilities.

### BIM & CityGML
- IFC, CityGML LOD1–3, 3D Tiles / Cesium ion import
- Indoor/outdoor seamless navigation

### Temporal Dimension
- Time slider with historical scene states
- Version-controlled snapshots, change detection

### What-If Scenarios
- Fork scenes, modify parameters, compare outcomes
- Side-by-side comparison views, scenario libraries

### Domain Verticals
- **Urban Planning**: zoning, density, transit accessibility
- **Disaster Response**: flood modeling, fire spread, evacuation routing
- **Telecommunications**: RF propagation via ray tracing, tower placement
- **Renewable Energy**: solar panel placement, wind farm siting
- **Environmental Monitoring**: air quality, noise mapping, vegetation health
- **Infrastructure Management**: utility networks, maintenance scheduling

---

## Key Advantages

- **Ray tracing as rendering primitive** — viewshed, shadows, AO, GI all share the OptiX pipeline; new analyses = new launch kernel
- **GIS-native coordinates** — real DEMs with real CRS metadata, GeoJSON/Overture/OSM data flows directly in
- **xarray accessor** — `dem.rtx.explore()` zero-boilerplate interface
- **GPU-native data path** — xarray → CuPy → OptiX device buffers, no CPU round-trips
- **Real-time data patterns** — GTFS-RT, FIRMS, NOAA wind already demonstrate fetch → transform → place → update
- **Zarr persistence** — spatially chunked mesh store, foundation for LOD streaming and temporal versioning
