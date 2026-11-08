"""How far a disabled ship drifts before it is repaired (pure; no QGIS).

``repair_time_quantile_h`` reads the repair-time distribution of the drift
settings (Weibull, normal, lognormal or a user CDF expression) and returns
the time by which a share ``q`` of the ships is repaired.  The drift model
uses ``q = 0.99`` as its corridor length (:data:`REACH_QUANTILE`); the depth
simplification tool uses a larger share for the zone around structures.
"""
from __future__ import annotations

from typing import Any

from scipy import stats

REACH_QUANTILE = 0.99
"""Share repaired at the end of the drift model's corridors."""


def repair_time_quantile_h(repair: dict[str, Any] | None, q: float) -> float | None:
    """Hours by which a share ``q`` of disabled ships is repaired, or None."""
    rep = repair or {}
    dist_type = rep.get('dist_type', '')
    t_h: float | None = None
    if dist_type == 'weibull':
        t_h = float(stats.weibull_min(c=float(rep.get('wb_shape', 1.0)), loc=float(rep.get('wb_loc', 0.0)),
                                      scale=float(rep.get('wb_scale', 1.0))).ppf(q))
    elif dist_type == 'normal':
        t_h = float(stats.norm(loc=float(rep.get('norm_mean', 0.0)), scale=float(rep.get('norm_std', 1.0))).ppf(q))
    elif rep.get('use_lognormal', False):
        t_h = float(stats.lognorm(float(rep.get('std', 0.0)), loc=float(rep.get('loc', 0.0)),
                                  scale=float(rep.get('scale', 1.0))).ppf(q))
    elif rep.get('func'):
        # User-defined repair CDF: bisection on the compiled expression.
        from compute.basic_equations import _safe_compile, _safe_eval
        code = _safe_compile(str(rep['func']))
        lo, hi = 0.0, 200.0
        if _safe_eval(code, hi) >= q:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if _safe_eval(code, mid) >= q:
                    hi = mid
                else:
                    lo = mid
            t_h = hi
    if t_h is None or not t_h > 0:
        return None
    return t_h


def drift_reach_m(drift: dict[str, Any] | None, q: float = REACH_QUANTILE) -> float | None:
    """Metres drifted (at the drift speed) before a share ``q`` is repaired, or None."""
    drift = drift or {}
    t_h = repair_time_quantile_h(drift.get('repair', {}), q)
    speed_kn = float(drift.get('speed', 0.0) or 0.0)
    if t_h is None or speed_kn <= 0:
        return None
    return speed_kn * 1852.0 * t_h
