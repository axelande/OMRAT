"""Oil-spill consequence calculation.

Combines per-cell accident frequencies emitted by the four model mixins
with the project-level consequence inputs (oil onboard, conditional spill
probability, spill fraction, catastrophe levels) to produce the annual
exceedance frequency for each user-defined catastrophe level.

Pure-Python and QGIS-free so it can run inside ``CalculationTask`` and be
unit-tested standalone.
"""
from __future__ import annotations

from typing import Any

from omrat_utils.consequence_defaults import ACCIDENT_KEYS, ACCIDENT_TYPES


def _by_cell_for_accident(
    accident_key: str,
    drifting_report: dict[str, Any] | None,
    powered_grounding_report: dict[str, Any] | None,
    powered_allision_report: dict[str, Any] | None,
    collision_report: dict[str, Any] | None,
) -> dict[str, float]:
    """Return the per-cell breakdown dict for one accident category.

    Cell keys are ``"{ship_type_idx}_{length_idx}"`` strings, values are
    annual frequencies.  Returns an empty dict if the corresponding model
    didn't run.
    """
    if accident_key == 'drifting_allision':
        return dict((drifting_report or {}).get('by_cell_allision', {}) or {})
    if accident_key == 'drifting_grounding':
        return dict((drifting_report or {}).get('by_cell_grounding', {}) or {})
    if accident_key == 'powered_allision':
        return dict((powered_allision_report or {}).get('by_cell', {}) or {})
    if accident_key == 'powered_grounding':
        return dict((powered_grounding_report or {}).get('by_cell', {}) or {})
    if collision_report is not None:
        bc = collision_report.get('by_cell', {}) or {}
        if accident_key == 'overtaking':
            return dict(bc.get('overtaking', {}) or {})
        if accident_key == 'head_on':
            return dict(bc.get('head_on', {}) or {})
        if accident_key == 'crossing':
            return dict(bc.get('crossing', {}) or {})
        if accident_key == 'merging':
            # Merging is the small-angle subset of crossing in the
            # collision model.  When the user wants pure bend (waypoint
            # turn-failure), it's tracked separately under 'bend'.
            merging = dict(bc.get('merging', {}) or {})
            bend = bc.get('bend', {}) or {}
            for k, v in bend.items():
                merging[k] = merging.get(k, 0.0) + float(v)
            return merging
    return {}


def _parse_cell_key(cell_key: str) -> tuple[int, int] | None:
    parts = cell_key.split('_')
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _lookup_oil(
    oil_onboard: list[list[float]],
    ship_type_idx: int,
    length_idx: int,
) -> float:
    if 0 <= ship_type_idx < len(oil_onboard):
        row = oil_onboard[ship_type_idx]
        if isinstance(row, (list, tuple)) and 0 <= length_idx < len(row):
            try:
                return float(row[length_idx])
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def compute_catastrophe_exceedance(
    consequence: dict[str, Any],
    *,
    drifting_report: dict[str, Any] | None = None,
    powered_grounding_report: dict[str, Any] | None = None,
    powered_allision_report: dict[str, Any] | None = None,
    collision_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute annual exceedance frequencies per catastrophe level.

    Parameters
    ----------
    consequence
        Project-level consequence block: ``oil_onboard`` (2D), ``spill_probability``
        (8 x 4 in percent), ``spill_fraction`` (8 x 4 in percent), and
        ``catastrophe_levels`` (list of ``{name, quantity}``).  Same shape as
        ``Consequence.load_from_dict`` produces.
    drifting_report, powered_grounding_report, powered_allision_report,
    collision_report
        The four model reports written by ``Calculation.*`` runs.  Each is
        looked up for its ``by_cell`` (or ``by_cell_allision`` /
        ``by_cell_grounding`` for drifting) breakdown.

    Returns
    -------
    dict
        ``{
            'levels':   list of {'name': str, 'quantity': float, 'exceedance': float},
            'total_spill_frequency': float,    # sum across levels (level 0 only)
            'mean_spill_volume_per_year': float,
            'by_accident': {accident_label: {'frequency': float, 'spill_m3': float}},
        }``
    """
    oil_onboard: list[list[float]] = consequence.get('oil_onboard', []) or []
    spill_prob: list[list[float]] = consequence.get('spill_probability', []) or []
    spill_frac: list[list[float]] = consequence.get('spill_fraction', []) or []
    levels = consequence.get('catastrophe_levels', []) or []

    # Sort thresholds for the result so callers can rely on increasing order.
    sorted_levels = sorted(
        ({
            'name': str(lvl.get('name', '')),
            'quantity': float(lvl.get('quantity', 0.0)),
        } for lvl in levels),
        key=lambda d: d['quantity'],
    )

    results = {
        lvl['name']: {
            'name': lvl['name'],
            'quantity': lvl['quantity'],
            'exceedance': 0.0,
        }
        for lvl in sorted_levels
    }
    by_accident: dict[str, dict[str, float]] = {
        label: {'frequency': 0.0, 'spill_m3': 0.0}
        for label in ACCIDENT_TYPES
    }
    mean_spill_volume_per_year = 0.0
    total_spill_frequency = 0.0

    for accident_idx, accident_key in enumerate(ACCIDENT_KEYS):
        accident_label = ACCIDENT_TYPES[accident_idx]
        cell_breakdown = _by_cell_for_accident(
            accident_key,
            drifting_report,
            powered_grounding_report,
            powered_allision_report,
            collision_report,
        )
        if not cell_breakdown:
            continue

        # Per-row probability and fraction vectors (length 4, in percent).
        prob_row = (
            spill_prob[accident_idx]
            if accident_idx < len(spill_prob)
            else [0.0, 0.0, 0.0, 0.0]
        )
        frac_row = (
            spill_frac[accident_idx]
            if accident_idx < len(spill_frac)
            else [0.0, 0.0, 0.0, 0.0]
        )

        for cell_key, freq_value in cell_breakdown.items():
            cell = _parse_cell_key(cell_key)
            if cell is None:
                continue
            ship_type_idx, length_idx = cell
            try:
                freq = float(freq_value)
            except (TypeError, ValueError):
                continue
            if freq <= 0.0:
                continue
            oil = _lookup_oil(oil_onboard, ship_type_idx, length_idx)
            by_accident[accident_label]['frequency'] += freq

            # Each spill level contributes ``freq * P(level | accident)``
            # events per year, with spill volume ``oil * fraction(level)``.
            for lvl_idx, prob_pct in enumerate(prob_row):
                try:
                    p = float(prob_pct) / 100.0
                except (TypeError, ValueError):
                    continue
                if p <= 0.0:
                    continue
                level_freq = freq * p
                try:
                    frac_pct = float(frac_row[lvl_idx])
                except (TypeError, ValueError, IndexError):
                    frac_pct = 0.0
                spill_volume = oil * frac_pct / 100.0
                # "No spill" contributes zero spill volume but still
                # carries probability mass; track it for completeness in
                # ``total_spill_frequency`` only when volume > 0.
                if spill_volume > 0.0:
                    total_spill_frequency += level_freq
                    mean_spill_volume_per_year += level_freq * spill_volume
                    by_accident[accident_label]['spill_m3'] += (
                        level_freq * spill_volume
                    )
                # Catastrophe exceedance: this (accident, level) contributes
                # to *every* catastrophe threshold whose quantity it crosses.
                for lvl in sorted_levels:
                    if spill_volume > lvl['quantity']:
                        results[lvl['name']]['exceedance'] += level_freq

    return {
        'levels': [
            results[lvl['name']] for lvl in sorted_levels
        ],
        'total_spill_frequency': float(total_spill_frequency),
        'mean_spill_volume_per_year': float(mean_spill_volume_per_year),
        'by_accident': by_accident,
    }
