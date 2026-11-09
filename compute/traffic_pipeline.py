"""The traffic steps every model run takes on its private data copy (pure).

:func:`prepare_traffic_for_run` runs, in this order,

1. :func:`compute.traffic_gaps.fill_traffic_gaps` -- missing speeds and
   draughts filled (``FILL_RULE``);
2. :func:`compute.traffic_scaling.apply_traffic_scaling` -- ``Scaling (%)``;
3. :func:`compute.traffic_redirect.apply_traffic_redirects` -- suppressed
   legs' ships moved, the legs dropped;
4. :func:`compute.speed_restriction.apply_speed_restrictions` -- speeds
   capped in the speed restriction zones (needs the filled speeds and the
   moved traffic, so it comes last).

``CalculationTask.run``, ``sensitivity_task.run_phases_on_stub`` and
``compare_explain.run_collision_model`` all call it, so a new step is added
here once.  The traffic report repeats steps 2-4 itself (it shows the
stored values next to the calculated ones and must not fill gaps).
"""
from __future__ import annotations

from typing import Any

from compute.speed_restriction import apply_speed_restrictions
from compute.traffic_gaps import fill_traffic_gaps
from compute.traffic_redirect import apply_traffic_redirects
from compute.traffic_scaling import apply_traffic_scaling

__all__ = ['prepare_traffic_for_run']


def prepare_traffic_for_run(data: dict[str, Any]) -> dict[str, Any]:
    """Apply the four steps to ``data`` in place and return their summaries.

    ``{'gaps': [...fill summaries...], 'redirect': {...}, 'speed': {...}}``.
    Raises :class:`compute.speed_restriction.SpeedRestrictionError` for
    overlapping or invalid enabled speed restriction zones.
    """
    gaps = fill_traffic_gaps(data)
    apply_traffic_scaling(data)
    redirect = apply_traffic_redirects(data)
    speed = apply_speed_restrictions(data)
    return {'gaps': gaps, 'redirect': redirect, 'speed': speed}
