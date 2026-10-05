"""Per-(leg, direction) rows for the powered **View** dialog (QGIS-free).

The powered allision / grounding View lists every (leg, direction) the
ray caster analysed, with the leg named as in the route table
(``compute/leg_labels.py``), its annual frequency and its **% of total**
-- the same share the collision breakdown shows.

* The frequency of a leg-direction is the sum of
  ``by_obstacle_leg[obs]["<seg>:<dir_idx>"]`` over all obstacles of the
  model report; the total is ``totals[<mode>]``.
* Powered reports are not stored in the run's GeoPackage, so
  ``RunHistoryMixin`` writes :func:`sidecar_payload` next to it as
  ``<stem>.powered.json`` (one block per mode, already reduced to
  ``by_leg_direction``).
* A run saved before that sidecar existed has no powered report.  The
  allision one is rebuilt exactly from the View's own ray casts by
  :func:`allision_report_from_computations` (the same code path as
  ``run_powered_allision_model``).  Grounding is ray cast once per
  draught bin in the model, which would make the View slow, so it is not
  rebuilt: its frequency and share columns show ``--``.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from compute.leg_labels import direction_name, leg_name

__all__ = [
    'SIDECAR_SUFFIX', 'MODES', 'leg_title', 'geometric_probability',
    'leg_direction_frequencies', 'report_total', 'summarize_report',
    'sidecar_payload', 'allision_report_from_computations',
    'build_sidebar_rows', 'format_frequency', 'format_share',
]

SIDECAR_SUFFIX = '.powered.json'
MODES: tuple[str, str] = ('allision', 'grounding')


def _f(value: Any) -> float:
    try:
        v = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def leg_title(segment_data: Mapping[str, Any] | None, seg_id: Any, fallback: Any = None) -> str:
    """Route-table name of a leg (``"LEG_6_1_b"``); ``fallback`` or ``"Leg <id>"`` when unnamed."""
    name = leg_name(segment_data, seg_id)
    if name:
        return name
    text = str(fallback or '').strip()
    return text or f'Leg {seg_id}'


def geometric_probability(comp: Mapping[str, Any]) -> float:
    """Per-ship hit probability of one computation: Cat II ``p_approx`` plus Cat I mass.

    No traffic, causation factor or ship-speed weighting -- it is what the
    View can show when the run has no powered report.
    """
    total = 0.0
    blocks = [comp.get('summaries') or {}, (comp.get('cat1') or {}).get('summaries') or {}]
    for block in blocks:
        for s in block.values():
            total += _f(s.get('p_approx'))
    return total


def leg_direction_frequencies(report: Mapping[str, Any] | None) -> dict[str, float]:
    """``{"<seg>:<dir_idx>": annual frequency}`` from a powered report.

    A model report is summed from ``by_obstacle_leg``; the sidecar form
    already carries ``by_leg_direction``.
    """
    if not report:
        return {}
    stored = report.get('by_leg_direction')
    if isinstance(stored, Mapping):
        return {str(k): _f(v) for k, v in stored.items()}
    out: dict[str, float] = {}
    for leg_map in (report.get('by_obstacle_leg') or {}).values():
        if not isinstance(leg_map, Mapping):
            continue
        for dir_key, value in leg_map.items():
            out[str(dir_key)] = out.get(str(dir_key), 0.0) + _f(value)
    return out


def report_total(report: Mapping[str, Any] | None, mode: str) -> float:
    """``totals[mode]``, or the sum of the leg-direction frequencies when that is missing or zero."""
    total = _f(((report or {}).get('totals') or {}).get(mode))
    if total <= 0.0:
        total = sum(leg_direction_frequencies(report).values())
    return total


def summarize_report(report: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The part of a powered report the View needs, small enough for a sidecar."""
    if not isinstance(report, Mapping) or not report.get('totals'):
        return None
    return {
        'totals': {str(k): _f(v) for k, v in (report.get('totals') or {}).items()},
        'by_leg_direction': leg_direction_frequencies(report),
        'by_obstacle': {str(k): _f(v) for k, v in (report.get('by_obstacle') or {}).items()},
        'causation_factor': report.get('causation_factor'),
        'causation_factor_cat1': report.get('causation_factor_cat1'),
    }


def sidecar_payload(
    allision_report: Mapping[str, Any] | None, grounding_report: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """``{"allision": {...}, "grounding": {...}}`` for ``<stem>.powered.json``; ``None`` when both are empty."""
    payload: dict[str, Any] = {}
    for mode, rep in (('allision', allision_report), ('grounding', grounding_report)):
        block = summarize_report(rep)
        if block is not None:
            payload[mode] = block
    return payload or None


def allision_report_from_computations(
    data: Mapping[str, Any], computations: list[dict], object_geoms: list[dict],
) -> dict[str, Any]:
    """Powered-allision report rebuilt from the View's ray casts.

    ``computations`` must come from ``_run_all_computations`` with the
    project's Cat I start zone, ``object_geoms`` from
    ``_build_legs_and_obstacles`` (split ids + heights) -- then this is
    the figure ``run_powered_allision_model`` produced.
    """
    from compute.powered_model import PoweredModelMixin, _pc_pair
    pc_vals = data.get('pc') if isinstance(data.get('pc'), dict) else {}
    pc_cat2, pc_cat1 = _pc_pair(pc_vals, 'allision')
    heights = {str(o.get('id')): _f(o.get('height')) for o in object_geoms or []}
    acc = PoweredModelMixin()._sum_allision_contribs(
        computations, data.get('traffic_data') or {}, heights, pc_cat2, pc_cat1,
    )
    return acc.report('allision', pc_cat2, pc_cat1)


def build_sidebar_rows(
    computations: list[dict], report: Mapping[str, Any] | None,
    segment_data: Mapping[str, Any] | None, mode: str,
) -> list[dict[str, Any]]:
    """One row per computation, most important first.

    Each row: ``comp_idx`` (index into ``computations``), ``key``
    (``"<seg>:<dir_idx>"``), ``leg``, ``direction``, ``frequency`` and
    ``share`` (percent of the mode's total; both ``None`` without a
    report) and ``p_geom`` (:func:`geometric_probability`).
    """
    freqs = leg_direction_frequencies(report) if report else None
    total = report_total(report, mode) if report else 0.0
    rows: list[dict[str, Any]] = []
    for idx, comp in enumerate(computations):
        seg = str(comp.get('seg_id', ''))
        dir_idx = int(comp.get('dir_idx', 0) or 0)
        key = f'{seg}:{dir_idx}'
        dir_text = (comp.get('dir_info') or {}).get('name') or direction_name(segment_data, seg, dir_idx) or ''
        freq = None if freqs is None else freqs.get(key, 0.0)
        share = (freq / total * 100.0) if (freq is not None and total > 0.0) else None
        rows.append({
            'comp_idx': idx, 'key': key,
            'leg': leg_title(segment_data, seg, (comp.get('leg') or {}).get('name')),
            'direction': str(dir_text),
            'frequency': freq, 'share': share,
            'p_geom': geometric_probability(comp),
        })
    if freqs is None:
        rows.sort(key=lambda r: -r['p_geom'])
    else:
        rows.sort(key=lambda r: (-r['frequency'], -r['p_geom']))
    return rows


def format_frequency(value: float | None) -> str:
    """Table text for a frequency: ``--`` unknown, ``0`` for none, else ``1.234e-05``."""
    if value is None:
        return '--'
    if value <= 0.0:
        return '0'
    return f'{value:.3e}'


def format_share(value: float | None) -> str:
    """Table text for a share in percent, formatted like the collision breakdown (``12.3%``)."""
    if value is None:
        return '--'
    return f'{value:.1f}%'
