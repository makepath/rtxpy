"""Reusable particle subsystem for the interactive viewer.

Provides a data-oriented ``ParticleSystem`` container and shared
projection/splatting utilities.  Type-specific advection and rendering
live in ``wind``, ``rain``, and ``hydro`` modules as free functions.

Architecture
------------
``ParticleSystem`` is a pure data class — positions, ages, lifetimes,
optional trail ring buffer, and a cached spawn distribution.  It owns
the particle lifecycle (spawn, age, respawn, trail shift) but has no
knowledge of advection physics or rendering.

Each particle type provides free functions that operate on a
``ParticleSystem`` instance:

    wind.advect_wind(system, ...)        — bilinear wind-field sampling
    wind.render_wind_cpu(system, ...)    — CPU trail splatting
    rain.advect_rain(system, ...)        — vertical z_frac descent
    rain.render_rain_cpu(system, ...)    — vertical streak rendering
    hydro.render_hydro_cpu(system, ...)  — per-particle color/radius splatting

Shared projection math lives in ``project.py`` and is used by all renderers.
"""

from .system import ParticleSystem
from . import project, wind, rain, hydro

__all__ = ['ParticleSystem', 'project', 'wind', 'rain', 'hydro']
