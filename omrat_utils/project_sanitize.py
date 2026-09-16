"""Make an in-memory project pass ``RootModelSchema`` before it is
written or after it is read.

Two things in a freshly drawn project are legitimately "not a number
yet" and would otherwise fail validation:

* ``Speed / Draught / Ship heights / Ship Beam`` cells start life as
  empty lists that the AIS pass appends observations to
  (``Traffic.create_empty_dict``).  ``AisUpdateTask._collapse_td_block``
  turns them into the mean, or ``inf`` when nothing was observed; the
  same rule is applied here so a project saved *before* any AIS pass
  looks exactly like one saved after an AIS pass over empty water.
* The lateral-distribution fields (``mean1_1`` ...) are seeded the first
  time a leg is shown in the Distributions tab
  (``Distributions.change_dist_segment``).  Legs never shown there have
  none, so the same defaults are seeded here.

Both the writer (``GatherData.get_all_for_save``) and the reader
(``Storage._normalize_legacy_to_schema``) call ``sanitize_project`` so a
file OMRAT writes is always a file OMRAT can open.  Pure Python.
"""
from __future__ import annotations

from typing import Any

#: Same values ``Distributions.change_dist_segment`` seeds on first view.
DISTRIBUTION_DEFAULTS: dict[str, float | int] = {
    'mean1_1': 0, 'std1_1': 0, 'weight1_1': 100,
    'mean2_1': 0, 'std2_1': 0, 'weight2_1': 100,
    'mean1_2': 0, 'std1_2': 0, 'weight1_2': 0,
    'mean1_3': 0, 'std1_3': 0, 'weight1_3': 0,
    'mean2_2': 0, 'std2_2': 0, 'weight2_2': 0,
    'mean2_3': 0, 'std2_3': 0, 'weight2_3': 0,
    'u_min1': 0, 'u_max1': 0, 'u_p1': 0, 'ai1': 180,
    'u_min2': 0, 'u_max2': 0, 'u_p2': 0, 'ai2': 180,
}

#: Value of an observation cell that received no AIS pings -- the same
#: sentinel ``AisUpdateTask._collapse_td_block`` writes.
NO_OBSERVATION = float('inf')


def seed_distribution_defaults(segment_data: dict[str, Any] | None) -> int:
    """Add missing lateral-distribution fields to every leg, in place.
    Returns the number of legs that were missing at least one field."""
    touched = 0
    for seg in (segment_data or {}).values():
        if not isinstance(seg, dict):
            continue
        missing = [k for k in DISTRIBUTION_DEFAULTS if k not in seg]
        if not missing:
            continue
        for k in missing:
            seg[k] = DISTRIBUTION_DEFAULTS[k]
        touched += 1
    return touched


def collapse_sample_cells(traffic_data: dict[str, Any] | None) -> int:
    """Replace every list-valued matrix cell with its mean (``inf`` when
    the list is empty), in place.  Returns the number of cells changed.

    Frequency and Scaling cells are numbers already and pass through;
    anything that is not a list is left alone.
    """
    changed = 0
    for leg in (traffic_data or {}).values():
        if not isinstance(leg, dict):
            continue
        for block in leg.values():
            if not isinstance(block, dict):
                continue
            for matrix in block.values():
                if not isinstance(matrix, list):
                    continue
                for row in matrix:
                    if not isinstance(row, list):
                        continue
                    for c, val in enumerate(row):
                        if isinstance(val, list):
                            row[c] = (sum(float(x) for x in val) / len(val)) if val else NO_OBSERVATION
                            changed += 1
    return changed


def sanitize_project(data: dict[str, Any]) -> dict[str, Any]:
    """Apply both fixes to ``data`` in place and return it."""
    seed_distribution_defaults(data.get('segment_data'))
    collapse_sample_cells(data.get('traffic_data'))
    return data
