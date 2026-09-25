"""Lateral distribution curves drawn along the tangent line, as in IWRAP.

Pure Python (no Qt / QGIS) so the standalone test suite covers it.  The
QGIS side (:meth:`geometries.handle_qgis_iface.HandleQGISIface.refresh_distribution_curves`)
turns the points into features of the *Tangent Line* layer.

Geometry
--------
The tangent line is the leg's cross-section: it runs through the point at
``Tangent_Pos`` along the leg, perpendicular to it, over the leg width.
Its lateral axis ``x`` is the one every distribution uses: metres from
the leg centreline, *negative on the starboard side of the drawn
direction* (the tangent / AIS passage line starts on the bearing+90 side,
see CLAUDE.md "Leg direction convention").  Both directions share it.

Each direction's fitted distribution (the normal + uniform mixture the
models use: ``mean{d}_{i}`` / ``std{d}_{i}`` / ``weight{d}_{i}`` and
``u_min{d}`` / ``u_max{d}`` / ``u_p{d}``, weights normalised by their sum)
is drawn in true scale along ``x`` and bulges *towards the side its ships
sail to*: direction 1 (``Dirs[0]``, drawn direction) towards ``End_Point``,
direction 2 towards ``Start_Point``.  Heights share one scale per leg: the
taller of the two peaks is :data:`HEIGHT_FRACTION` of the leg width, so the
two shapes can be compared.  A curve starts and ends on the tangent line
(the tails beyond the leg width are clipped).
"""
from __future__ import annotations

from math import exp, isfinite, pi, sqrt
from typing import Any

#: The taller peak of a leg's two curves is this fraction of the leg width.
HEIGHT_FRACTION = 0.25
#: Evenly spaced samples across the width (more are added around each peak).
N_GRID = 121

Component = tuple[str, float, float, float]   # ('n', mean, std, w) | ('u', low, high, w)


def _f(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if isfinite(v) else None


def mixture(seg_d: dict[str, Any], direction: int) -> list[Component]:
    """The weighted components of direction ``direction`` (0 or 1), with
    weights normalised to sum to 1.  Empty when nothing is defined."""
    if not isinstance(seg_d, dict):
        return []
    d = direction + 1
    comps: list[Component] = []
    for i in (1, 2, 3):
        w, m, s = (_f(seg_d.get(f'{k}{d}_{i}')) for k in ('weight', 'mean', 'std'))
        if w is not None and w > 0 and m is not None and s is not None and s > 0:
            comps.append(('n', m, s, w))
    w_u, lo, hi = _f(seg_d.get(f'u_p{d}')), _f(seg_d.get(f'u_min{d}')), _f(seg_d.get(f'u_max{d}'))
    if w_u is not None and w_u > 0 and lo is not None and hi is not None and hi > lo:
        comps.append(('u', lo, hi, w_u))
    total = sum(c[3] for c in comps)
    if total <= 0:
        return []
    return [(kind, a, b, w / total) for kind, a, b, w in comps]


def pdf(comps: list[Component], x: float) -> float:
    """Mixture density at ``x`` (1/m)."""
    out = 0.0
    for kind, a, b, w in comps:
        if kind == 'n':
            z = (x - a) / b
            out += w * exp(-0.5 * z * z) / (b * sqrt(2.0 * pi))
        elif a <= x <= b:
            out += w / (b - a)
    return out


def sample_xs(comps: list[Component], half_width: float) -> list[float]:
    """Sample positions across ``[-half_width, half_width]``: an even grid
    plus points around every peak and uniform edge, so a narrow peak on a
    wide leg keeps its shape."""
    xs = {-half_width + 2.0 * half_width * i / (N_GRID - 1) for i in range(N_GRID)}
    for kind, a, b, _w in comps:
        if kind == 'n':
            xs.update(a + b * k / 4.0 for k in range(-16, 17))       # mean +/- 4 std
        else:
            eps = max(half_width * 1e-4, 1e-6)
            xs.update((a - eps, a + eps, b - eps, b + eps))
    return sorted(x for x in xs if -half_width <= x <= half_width)


def curve_profiles(seg_d: dict[str, Any], width: float) -> dict[int, list[tuple[float, float]]]:
    """``{direction: [(x, height), ...]}`` in metres for the leg's two
    directions, on a shared height scale.  Directions without a
    distribution are left out; ``{}`` when neither has one."""
    half = _f(width)
    if half is None or half <= 0:
        return {}
    half /= 2.0
    comps = {d: mixture(seg_d, d) for d in (0, 1)}
    raw: dict[int, list[tuple[float, float]]] = {}
    for d, cs in comps.items():
        if cs:
            raw[d] = [(x, pdf(cs, x)) for x in sample_xs(cs, half)]
    peak = max((p for pts in raw.values() for _x, p in pts), default=0.0)
    if peak <= 0:
        return {}
    scale = HEIGHT_FRACTION * 2.0 * half / peak
    out: dict[int, list[tuple[float, float]]] = {}
    for d, pts in raw.items():
        prof = [(x, p * scale) for x, p in pts]
        # Start and end on the tangent line even when a tail is clipped.
        prof = [(-half, 0.0)] + prof + [(half, 0.0)]
        out[d] = prof
    return out


def curve_points(
    mid: tuple[float, float],
    leg_unit: tuple[float, float],
    profile: list[tuple[float, float]],
    direction: int,
) -> list[tuple[float, float]]:
    """Map a profile to projected coordinates.

    ``mid`` is the tangent point on the leg, ``leg_unit`` the unit vector
    Start -> End (both in a metric CRS, x east / y north).  The lateral
    axis is the leg direction turned 90 degrees anticlockwise (port of the
    drawn direction), so ``x < 0`` lands on the starboard side exactly like
    the tangent line and the AIS passage line.  Direction 0 bulges towards
    ``End_Point``, direction 1 towards ``Start_Point``.
    """
    ux, uy = leg_unit
    px, py = -uy, ux
    sign = 1.0 if direction == 0 else -1.0
    return [(mid[0] + px * x + sign * ux * h, mid[1] + py * x + sign * uy * h) for x, h in profile]


def signature(seg_d: dict[str, Any], width: Any, tangent_pos: Any, ends: Any) -> tuple:
    """Everything a leg's curves depend on (to skip needless redraws)."""
    keys = [f'{k}{d}_{i}' for d in (1, 2) for i in (1, 2, 3) for k in ('mean', 'std', 'weight')]
    keys += [f'{k}{d}' for d in (1, 2) for k in ('u_min', 'u_max', 'u_p')]
    seg_d = seg_d if isinstance(seg_d, dict) else {}
    return tuple(_f(seg_d.get(k)) for k in keys) + (_f(width), _f(tangent_pos), ends)
