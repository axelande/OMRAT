"""Apply the Traffic tab's ``Scaling (%)`` matrices to the frequencies (pure; no QGIS).

``Q_effective = Q * Scaling / 100`` per cell, in place on a private copy
of the data.  :func:`apply_traffic_scaling` runs at the start of every
calculation (``CalculationTask.run``, ``sensitivity_task.run_phases_on_stub``,
``compare_explain.run_collision_model``) and in the Traffic report
(``omrat_utils.traffic_summary``), so the report's "in calculation"
column uses the same rule.  ``compute.data_preparation`` re-exports it.
"""
from typing import Any

import numpy as np

SCALING_KEY = 'Scaling (%)'
FREQ_KEY = 'Frequency (ships/year)'


def _scale_direction_freq(var: dict) -> None:
    freq = var.get(FREQ_KEY)
    if freq is None:
        return
    scaling = var.get(SCALING_KEY)
    for i, row in enumerate(freq):
        if not hasattr(row, '__iter__'):
            continue
        for j, q in enumerate(row):
            factor = 1.0
            if scaling is not None and i < len(scaling):
                s_row = scaling[i]
                if hasattr(s_row, '__iter__') and j < len(s_row):
                    try:
                        factor = float(s_row[j]) / 100.0
                    except (TypeError, ValueError):
                        factor = 1.0
            if factor == 1.0:
                continue
            if q == '' or q is None:
                continue
            try:
                q_val = float(q)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(q_val):
                continue
            row[j] = q_val * factor


def apply_traffic_scaling(data: dict[str, Any]) -> None:
    """Multiply every ``Frequency (ships/year)`` cell by its ``Scaling (%)`` / 100.

    Mutates ``data['traffic_data']`` in place.  Callers must pass a deep-copy
    of the live UI state so the user's stored Q values stay untouched.
    """
    traffic = data.get('traffic_data') or {}
    if not isinstance(traffic, dict):
        return
    for _seg, dirs in traffic.items():
        if not isinstance(dirs, dict):
            continue
        for _di, var in dirs.items():
            if isinstance(var, dict):
                _scale_direction_freq(var)
