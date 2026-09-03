"""Pure row builder for the per-leg / per-leg-pair collision breakdown
dialog (``compute.visualization.run_collision_breakdown_dialog``).

Kept free of Qt so the table content -- including the "% of total"
column -- can be unit-tested without a QGIS environment.
"""
from __future__ import annotations

from typing import Any

PAIR_TYPES = ('crossing', 'merging')


def _f(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _collect(report: dict[str, Any], encounter_type: str) -> tuple[list[str], list[tuple[list[str], float]]]:
    """Return ``(headers_without_probability, [(text_cells, probability), ...])``."""
    rows: list[tuple[list[str], float]] = []
    if encounter_type in PAIR_TYPES:
        headers = ['Leg pair', 'Waypoint (lon lat)', 'Angle°']
        for label, rec in (report.get('by_leg_pair', {}) or {}).items():
            if not isinstance(rec, dict):
                continue
            v = _f(rec.get(encounter_type))
            if v <= 0.0:
                continue
            rows.append((
                [str(label), str(rec.get('waypoint', '')), f"{_f(rec.get('angle_deg')):.1f}"],
                v,
            ))
    elif encounter_type == 'bend':
        headers = ['Leg pair', 'Waypoint (lon lat)']
        for label, rec in (report.get('bend_by_pair', {}) or {}).items():
            if not isinstance(rec, dict):
                continue
            v = _f(rec.get('bend'))
            if v <= 0.0:
                continue
            rows.append(([str(label), str(rec.get('waypoint', ''))], v))
    else:
        headers = ['Leg']
        for leg_id, leg_vals in (report.get('by_leg', {}) or {}).items():
            if not isinstance(leg_vals, dict):
                continue
            v = _f(leg_vals.get(encounter_type))
            if v <= 0.0:
                continue
            rows.append(([str(leg_id)], v))
    return headers, rows


def build_breakdown_rows(
    report: dict[str, Any] | None, encounter_type: str,
) -> tuple[list[str], list[list[str]]]:
    """Return ``(headers, rows)`` for the breakdown table.

    The last two columns are ``Probability`` and ``% of total``.  The
    total is the model's own figure (``report['totals'][type]``) when
    it is present and positive, otherwise the sum of the listed rows,
    so the column always reads as "share of this accident type".
    Rows are sorted by descending probability.
    """
    report = report or {}
    headers, collected = _collect(report, encounter_type)
    if not collected:
        return headers + ['Probability', '% of total'], []
    collected.sort(key=lambda item: -item[1])
    total = _f((report.get('totals', {}) or {}).get(encounter_type))
    if total <= 0.0:
        total = sum(v for _cells, v in collected)
    rows: list[list[str]] = []
    for cells, v in collected:
        share = (v / total * 100.0) if total > 0.0 else 0.0
        rows.append(cells + [f"{v:.3e}", f"{share:.1f}%"])
    return headers + ['Probability', '% of total'], rows
