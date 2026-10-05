"""How the powered model samples a leg's lateral distribution (QGIS-free).

Both powered categories cast parallel rays across the *first* normal
component of a direction's lateral distribution (``mean{d}_1`` /
``std{d}_1``, the figures the ray caster in
``geometries/get_powered_overlap.py`` reads).  This module owns the two
numbers that define that sampling, so the caster, the report and the
tangent-line markers on the canvas can never disagree:

* :data:`RAY_SPAN_SIGMA` -- the rays run from ``mean - k*sigma`` to
  ``mean + k*sigma``.  Chosen 2026-10-02 as 6 (was 4): the mass of a
  normal beyond 4 sigma is 6.3e-5 per side, which with 10,000 ships a
  year and a causation factor of 1.9e-4 is 1e-4 accidents a year, more
  than a whole Category II total on the ESCOW project.
* :data:`N_RAYS` -- 1000 rays (was 500), so the spacing stays finer than
  before (0.012 sigma against 0.016) although the span is wider.  The
  cost is linear in the ray count; powered grounding on the ESCOW
  project went from 117 s to about 200 s.

They are constants, not project settings: every results Markdown prints
them (``ray_sampling`` on the powered reports), so a run always says how
it was sampled.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    'RAY_SPAN_SIGMA', 'N_RAYS', 'span_bounds', 'ray_offsets',
    'ray_sampling_settings', 'describe_ray_sampling',
]

RAY_SPAN_SIGMA: float = 6.0
N_RAYS: int = 1000


def span_bounds(mean: float, sigma: float, span_sigma: float = RAY_SPAN_SIGMA) -> tuple[float, float]:
    """``(low, high)`` lateral offsets the rays cover, in the distribution's unit."""
    return float(mean) - span_sigma * float(sigma), float(mean) + span_sigma * float(sigma)


def ray_offsets(mean: float, sigma: float, n_rays: int = N_RAYS, span_sigma: float = RAY_SPAN_SIGMA) -> np.ndarray:
    """Evenly spaced ray offsets across :func:`span_bounds`."""
    lo, hi = span_bounds(mean, sigma, span_sigma)
    return np.linspace(lo, hi, int(n_rays))


def ray_sampling_settings() -> dict[str, Any]:
    """``{'span_sigma': 6.0, 'n_rays': 1000}`` for reports and sidecars."""
    return {'span_sigma': float(RAY_SPAN_SIGMA), 'n_rays': int(N_RAYS)}


def describe_ray_sampling(settings: dict[str, Any] | None = None) -> str:
    """One line for the results Markdown."""
    s = settings or ray_sampling_settings()
    return f"+/- {float(s.get('span_sigma', RAY_SPAN_SIGMA)):g} sigma, {int(s.get('n_rays', N_RAYS))} rays"
