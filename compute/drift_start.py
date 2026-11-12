"""Where a drifting ship starts: the leg centreline or the centre of its lateral distribution.

**Settings > Drift settings > Start drifting from** (``drift['start_from']``):

* ``'leg_center'`` (default) -- the drift distance to an obstacle is measured
  from the leg line;
* ``'distribution_center'`` -- it is measured from a line parallel to the
  leg, offset by the mean of the direction's lateral distribution (normals
  and uniform part, by weight).  One line per traffic direction, so a lane on
  the starboard side of the leg starts closer to the obstacles on that side
  and further from the others.

Only the distance changes (and with it the probability of not being
repaired before the obstacle); which ships reach an obstacle still follows
the whole lateral distribution (the probability holes).  QGIS-free.
"""
from __future__ import annotations

import math
from typing import Any

LEG_CENTER = 'leg_center'
DISTRIBUTION_CENTER = 'distribution_center'
START_FROM_KEY = 'start_from'


def uses_distribution_center(drift: dict | None) -> bool:
    """True when the project measures drift distances from the distribution centre."""
    if not isinstance(drift, dict):
        return False
    return str(drift.get(START_FROM_KEY, LEG_CENTER)).strip().lower() == DISTRIBUTION_CENTER


def mixture_mean(distributions: list[Any], weights: list[float]) -> float:
    """Weighted mean of the frozen scipy distributions (``0.0`` when undefined).

    Components with a weight <= 0 or a non-finite mean are left out, so the
    placeholder ``norm(0, 1)`` / ``uniform(0, 1)`` entries of
    :func:`compute.data_preparation.get_distribution` do not count.
    """
    total = 0.0
    acc = 0.0
    for dist, w in zip(distributions or [], weights or []):
        try:
            w = float(w)
            m = float(dist.mean())
        except (TypeError, ValueError, AttributeError):
            continue
        if w > 0 and math.isfinite(w) and math.isfinite(m):
            acc += w * m
            total += w
    return acc / total if total > 0 else 0.0
