"""IWRAP-compatible default values for OMRAT inputs.

QGIS-free constants module: importable from both UI handlers (which
also touch ``qgis.PyQt``) and the standalone audit-report renderer.
The single source of truth for "what the default is" — change here
and every consumer follows.
"""
from __future__ import annotations

from typing import Any


# Causation factors.  Mirrors ``omrat_utils/causation_factors.py``.
IWRAP_PC_DEFAULTS: dict[str, float] = {
    'p_pc': 1.6e-4,
    'd_pc': 1,
    'headon': 4.9e-5,
    'overtaking': 1.1e-4,
    'crossing': 1.3e-4,
    'bend': 1.3e-4,
    'grounding': 1.6e-4,
    'allision': 1.9e-4,
}


# 8-direction wind rose with equal weights (compass bearings).
IWRAP_ROSE_DEFAULT: dict[str, float] = {
    '0': 0.125, '45': 0.125, '90': 0.125, '135': 0.125,
    '180': 0.125, '225': 0.125, '270': 0.125, '315': 0.125,
}


# Repair-time lognormal parameters (drift cascade).
IWRAP_REPAIR_DEFAULT: dict[str, Any] = {
    'func': '',
    'std': 0.95,
    'loc': 0.2,
    'scale': 0.85,
    'use_lognormal': True,
}


# Drift cascade parameters.  Mirrors ``omrat_utils/handle_settings.py``.
# ``blackout_by_ship_type`` is filled in by callers via
# ``compute.basic_equations.default_blackout_by_ship_type`` because it
# depends on ``SHIP_TYPE_NAMES``.
IWRAP_DRIFT_DEFAULTS: dict[str, Any] = {
    'drift_p': 1,
    'anchor_p': 0.70,
    'anchor_d': 7,
    'speed': 1.0,
    'start_from': 'leg_center',
    'squat_mode': 'average_speed',
    'rose': dict(IWRAP_ROSE_DEFAULT),
    'repair': dict(IWRAP_REPAIR_DEFAULT),
}


def default_drift_values() -> dict[str, Any]:
    """Fresh dict (rose / repair copied) suitable for live mutation."""
    return {
        **{k: v for k, v in IWRAP_DRIFT_DEFAULTS.items()
           if k not in ('rose', 'repair')},
        'rose': dict(IWRAP_ROSE_DEFAULT),
        'repair': dict(IWRAP_REPAIR_DEFAULT),
    }


def default_pc_values() -> dict[str, float]:
    """Fresh copy of the IWRAP causation-factor defaults."""
    return dict(IWRAP_PC_DEFAULTS)
