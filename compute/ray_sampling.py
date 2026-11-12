"""How the powered model samples a leg's lateral distribution (QGIS-free).

Both powered categories cast parallel rays (ship tracks) across a
direction's *whole* lateral distribution -- every normal component with
its weight and the uniform part, the mixture
``geometries.distribution_curves.mixture`` draws on the tangent line --
and give each ray the share of ships the mixture puts there
(:func:`mixture_rays`).  Until 2026-11-09 only the first normal component
(``mean{d}_1`` / ``std{d}_1``) was cast, with 100 % of the ships, so ships
of a second lane were counted in the first one (KNOWN_ISSUES 25).  A leg
with a single normal component is sampled exactly as before.

This module owns the numbers that define the sampling, so the caster, the
report and the tangent-line markers on the canvas can never disagree:

* :data:`RAY_SPAN_SIGMA` -- the rays cover ``mean +/- k*sigma`` of every
  component (and the uniform bounds).  Chosen 2026-10-02 as 6 (was 4): the
  mass of a normal beyond 4 sigma is 6.3e-5 per side, which with 10,000
  ships a year and a causation factor of 1.9e-4 is 1e-4 accidents a year,
  more than a whole Category II total on a wind farm case study.
* :data:`N_RAYS` -- 1000 rays for one component (0.012 sigma apart).  A
  wider mixture span gets more rays, so the spacing stays 0.012 x the
  narrowest component's sigma, up to :data:`MAX_RAYS`.  The cost is linear
  in the ray count; powered grounding on that case study took about 200 s
  with 1000 rays.

They are constants, not project settings: every results Markdown prints
them (``ray_sampling`` on the powered reports), so a run always says how
it was sampled.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

__all__ = [
    'RAY_SPAN_SIGMA', 'N_RAYS', 'MAX_RAYS', 'span_bounds', 'ray_offsets',
    'mixture_span', 'mixture_pdf', 'mixture_rays',
    'ray_sampling_settings', 'describe_ray_sampling',
]

RAY_SPAN_SIGMA: float = 6.0
N_RAYS: int = 1000
MAX_RAYS: int = 3 * N_RAYS

Component = tuple[str, float, float, float]   # ('n', mean, std, w) | ('u', low, high, w)


def span_bounds(mean: float, sigma: float, span_sigma: float = RAY_SPAN_SIGMA) -> tuple[float, float]:
    """``(low, high)`` lateral offsets the rays cover, in the distribution's unit."""
    return float(mean) - span_sigma * float(sigma), float(mean) + span_sigma * float(sigma)


def ray_offsets(mean: float, sigma: float, n_rays: int = N_RAYS, span_sigma: float = RAY_SPAN_SIGMA) -> np.ndarray:
    """Evenly spaced ray offsets across :func:`span_bounds`."""
    lo, hi = span_bounds(mean, sigma, span_sigma)
    return np.linspace(lo, hi, int(n_rays))


def _usable(components: Sequence[Component] | None) -> list[Component]:
    out = []
    for kind, a, b, w in (components or []):
        try:
            a, b, w = float(a), float(b), float(w)
        except (TypeError, ValueError):
            continue
        if w <= 0 or not (math.isfinite(a) and math.isfinite(b)):
            continue
        if (kind == 'n' and b > 0) or (kind == 'u' and b > a):
            out.append((kind, a, b, w))
    return out


def mixture_span(components: Sequence[Component], span_sigma: float = RAY_SPAN_SIGMA) -> tuple[float, float] | None:
    """``(low, high)`` covering every normal's ``mean +/- k*sigma`` and the uniform bounds."""
    lows, highs = [], []
    for kind, a, b, _w in _usable(components):
        lo, hi = span_bounds(a, b, span_sigma) if kind == 'n' else (a, b)
        lows.append(lo)
        highs.append(hi)
    if not lows:
        return None
    return min(lows), max(highs)


def mixture_pdf(components: Sequence[Component], x: np.ndarray) -> np.ndarray:
    """Mixture density at ``x`` (1/m); weights are used as given."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for kind, a, b, w in _usable(components):
        if kind == 'n':
            z = (x - a) / b
            out += w * np.exp(-0.5 * z * z) / (b * math.sqrt(2.0 * math.pi))
        else:
            out += np.where((x >= a) & (x <= b), w / (b - a), 0.0)
    return out


def mixture_rays(
    components: Sequence[Component], n_rays: int = N_RAYS, span_sigma: float = RAY_SPAN_SIGMA,
    max_rays: int = MAX_RAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(offsets, pdf_values, masses)`` of the rays for one direction.

    The rays are evenly spaced over :func:`mixture_span`; their count keeps
    the spacing of a single component (``2 k sigma / (n_rays - 1)``) for the
    narrowest component, between ``n_rays`` and ``max_rays``.  ``masses`` =
    density x spacing, i.e. the share of ships on each ray.  One normal
    component gives exactly :func:`ray_offsets` and its normal masses.
    """
    comps = _usable(components)
    span = mixture_span(comps, span_sigma)
    if span is None:
        raise ValueError("no usable lateral distribution component")
    lo, hi = span
    sigmas = [b for kind, _a, b, _w in comps if kind == 'n']
    n = int(n_rays)
    if sigmas:
        target = 2.0 * span_sigma * min(sigmas) / max(n_rays - 1, 1)
        n = int(math.ceil((hi - lo) / target - 1e-6)) + 1
        n = max(int(n_rays), min(int(max_rays), n))
    offsets = np.linspace(lo, hi, n)
    dx = offsets[1] - offsets[0] if n > 1 else 1.0
    pdf_vals = mixture_pdf(comps, offsets)
    return offsets, pdf_vals, pdf_vals * dx


def ray_sampling_settings() -> dict[str, Any]:
    """``{'span_sigma': 6.0, 'n_rays': 1000, 'max_rays': 3000, 'mixture': True}`` for reports and sidecars."""
    return {'span_sigma': float(RAY_SPAN_SIGMA), 'n_rays': int(N_RAYS), 'max_rays': int(MAX_RAYS), 'mixture': True}


def describe_ray_sampling(settings: dict[str, Any] | None = None) -> str:
    """One line for the results Markdown."""
    s = settings or ray_sampling_settings()
    text = f"+/- {float(s.get('span_sigma', RAY_SPAN_SIGMA)):g} sigma, {int(s.get('n_rays', N_RAYS))} rays"
    if s.get('mixture'):
        text = (f"whole lateral distribution, +/- {float(s.get('span_sigma', RAY_SPAN_SIGMA)):g} sigma of "
                f"every component, {int(s.get('n_rays', N_RAYS))}-{int(s.get('max_rays', MAX_RAYS))} rays")
    return text
