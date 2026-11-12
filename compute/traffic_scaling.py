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


def _scale_direction_freq(var: dict) -> int:
    """Scale one direction block in place; returns the number of cells changed."""
    freq = var.get(FREQ_KEY)
    if freq is None:
        return 0
    scaling = var.get(SCALING_KEY)
    changed = 0
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
            changed += 1
    return changed


def _direction_blocks(data: dict[str, Any]):
    traffic = data.get('traffic_data') or {}
    if not isinstance(traffic, dict):
        return
    for dirs in traffic.values():
        if not isinstance(dirs, dict):
            continue
        for var in dirs.values():
            if isinstance(var, dict):
                yield var


def apply_traffic_scaling(data: dict[str, Any]) -> None:
    """Multiply every ``Frequency (ships/year)`` cell by its ``Scaling (%)`` / 100.

    Mutates ``data['traffic_data']`` in place.  Callers must pass a deep-copy
    of the live UI state so the user's stored Q values stay untouched.
    """
    for var in _direction_blocks(data):
        _scale_direction_freq(var)


def bake_traffic_scaling(data: dict[str, Any]) -> int:
    """Apply ``Scaling (%)`` to the frequencies and set every Scaling cell to 100.

    For a copy that leaves OMRAT (the IWRAP export, which has no scaling of
    its own): the frequencies are written as the calculation uses them, and
    a second call changes nothing.  Mutates ``data`` in place; returns the
    number of frequency cells that were scaled.
    """
    changed = 0
    for var in _direction_blocks(data):
        changed += _scale_direction_freq(var)
        scaling = var.get(SCALING_KEY)
        if isinstance(scaling, np.ndarray):
            scaling.fill(100.0)
        elif isinstance(scaling, list):
            for i, row in enumerate(scaling):
                if isinstance(row, np.ndarray):
                    row.fill(100.0)
                elif isinstance(row, list):
                    scaling[i] = [100.0] * len(row)
    block = data.get('traffic_scaling')
    if isinstance(block, dict) and 'global_percent' in block:
        block['global_percent'] = 100.0
    return changed
