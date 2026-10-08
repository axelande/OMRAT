"""Where two runs differ: per-leg / waypoint / object result breakdown (QGIS-free).

The accident table of the Compare tab answers *how much* two runs
differ.  This module answers *where*: it joins the breakdowns each run
writes next to its snapshot (``<stem>.collision.json``,
``.drifting.json``, ``.powered.json``, ``.consequence.json``, read with
:func:`omrat_utils.compare.load_run_outputs`) item by item:

=================================  =============================================
Category                           Source
=================================  =============================================
Collisions per leg                 ``collision.by_leg`` (head-on, overtaking, bend)
Collisions per waypoint            ``collision.by_waypoint`` (crossing, merging, bend)
Collisions per leg pair            ``collision.by_leg_pair`` (crossing, merging)
Drifting per leg                   ``drifting.by_leg_direction`` summed per leg
Drifting per object                ``drifting.by_object``
Powered per leg and direction      ``powered.<mode>.by_leg_direction``
Powered per object                 ``powered.<mode>.by_obstacle``
Oil spill per accident type        ``consequence.by_accident[...].spill_m3``
=================================  =============================================

Legs are joined on id **and** name (see :mod:`omrat_utils.compare_common`),
waypoints on their six-decimal coordinates and objects on their names.
An item present in one run only counts as zero on the other side for
the change.

:func:`build_breakdown_rows` returns ``[category, item, a, b, delta,
share_pct]`` with ``a`` / ``b`` ``None`` when the item is absent in that
run and ``share_pct`` the item's share of the summed absolute change in
its category.  Rows are sorted by category, then by absolute change.
"""
from __future__ import annotations

from typing import Any, Mapping

from .compare_common import finite, leg_identity, leg_title, parse_xy

__all__ = ['CATEGORIES', 'build_breakdown_rows', 'top_movers']

CATEGORIES: tuple[str, ...] = (
    'Collisions per leg',
    'Collisions per waypoint',
    'Collisions per leg pair',
    'Drifting per leg',
    'Drifting per object',
    'Powered per leg and direction',
    'Powered per object',
    'Oil spill per accident type',
)

_TYPE_LABEL = {
    'head_on': 'head-on', 'overtaking': 'overtaking', 'bend': 'bend',
    'crossing': 'crossing', 'merging': 'merging',
}
_ZERO = 1e-300

Items = dict[tuple, tuple[str, float]]


def _add(items: Items, key: tuple, label: str, value: Any) -> None:
    v = finite(value)
    if v is None:
        return
    if key in items:
        items[key] = (items[key][0], items[key][1] + v)
    else:
        items[key] = (label, v)


def _segs(snap: Mapping[str, Any] | None) -> dict[str, Any]:
    seg = (snap or {}).get('segment_data') or {}
    return {str(k): v for k, v in seg.items() if isinstance(v, Mapping)}


def _dir_name(segs: Mapping[str, Any], seg_id: str, idx: Any) -> str:
    dirs = (segs.get(seg_id) or {}).get('Dirs')
    try:
        i = int(idx)
    except (TypeError, ValueError):
        return str(idx)
    if isinstance(dirs, (list, tuple)) and 0 <= i < len(dirs):
        return str(dirs[i])
    return f'direction {i + 1}'


def _waypoint_names(segs: Mapping[str, Any]) -> dict[str, list[str]]:
    """``{"lon lat" (6 decimals): [leg names ending there]}``.

    Built from both runs, so a waypoint label names the legs of A and B.
    """
    out: dict[str, list[str]] = {}
    for k, s in segs.items():
        for field in ('Start_Point', 'End_Point'):
            xy = parse_xy(s.get(field))
            if xy is None:
                continue
            name = str(s.get('Leg_name') or f'Leg {k}')
            out.setdefault(f'{xy[0]:.6f} {xy[1]:.6f}', []).append(name)
    return out


# ---------------------------------------------------------------------------
# Per-category extractors: ``{match key: (label, value)}`` for one run
# ---------------------------------------------------------------------------

def _collision_leg(out: Mapping[str, Any], segs, _names) -> Items:
    items: Items = {}
    for seg_id, rec in ((out.get('collision') or {}).get('by_leg') or {}).items():
        if not isinstance(rec, Mapping):
            continue
        for t in ('head_on', 'overtaking', 'bend'):
            _add(items, ('leg', leg_identity(segs, seg_id), t),
                 f'{leg_title(segs, seg_id)} – {_TYPE_LABEL[t]}', rec.get(t))
    return items


def _collision_waypoint(out: Mapping[str, Any], _segs, names) -> Items:
    items: Items = {}
    for wp, rec in ((out.get('collision') or {}).get('by_waypoint') or {}).items():
        if not isinstance(rec, Mapping):
            continue
        xy = parse_xy(wp)
        key_wp = f'{xy[0]:.6f} {xy[1]:.6f}' if xy else str(wp)
        legs = sorted(set(names.get(key_wp, [])))
        where = f'{xy[0]:.4f} {xy[1]:.4f}' if xy else str(wp)
        suffix = f' [{", ".join(legs)}]' if legs else ''
        for t in ('crossing', 'merging', 'bend'):
            _add(items, ('wp', key_wp, t), f'{where}{suffix} – {_TYPE_LABEL[t]}', rec.get(t))
    return items


def _collision_pair(out: Mapping[str, Any], segs, _names) -> Items:
    items: Items = {}
    for pair, rec in ((out.get('collision') or {}).get('by_leg_pair') or {}).items():
        if not isinstance(rec, Mapping):
            continue
        a, _, b = str(pair).partition('->')
        if not a or not b:
            continue
        ident = frozenset({leg_identity(segs, a), leg_identity(segs, b)})
        angle = finite(rec.get('angle_deg'))
        ang = f', {angle:.0f}°' if angle is not None else ''
        for t in ('crossing', 'merging'):
            _add(items, ('pair', ident, t),
                 f'{leg_title(segs, a)} x {leg_title(segs, b)} – {_TYPE_LABEL[t]}{ang}', rec.get(t))
    return items


def _drifting_leg(out: Mapping[str, Any], segs, _names) -> Items:
    items: Items = {}
    for key, rec in ((out.get('drifting') or {}).get('by_leg_direction') or {}).items():
        if not isinstance(rec, Mapping):
            continue
        seg_id = str(key).split(':')[0].strip()
        for mode in ('allision', 'grounding'):
            _add(items, ('dleg', leg_identity(segs, seg_id), mode),
                 f'{leg_title(segs, seg_id)} – drifting {mode}', rec.get(f'contrib_{mode}'))
    return items


def _drifting_object(out: Mapping[str, Any], _segs, _names) -> Items:
    items: Items = {}
    for name, rec in ((out.get('drifting') or {}).get('by_object') or {}).items():
        if not isinstance(rec, Mapping):
            continue
        for mode in ('allision', 'grounding'):
            _add(items, ('dobj', str(name), mode), f'{name} – drifting {mode}', rec.get(mode))
    return items


def _powered_legdir(out: Mapping[str, Any], segs, _names) -> Items:
    items: Items = {}
    for mode in ('allision', 'grounding'):
        block = (out.get('powered') or {}).get(mode) or {}
        for key, value in (block.get('by_leg_direction') or {}).items():
            seg_id, _, idx = str(key).partition(':')
            _add(items, ('pleg', leg_identity(segs, seg_id), idx, mode),
                 f'{leg_title(segs, seg_id)} {_dir_name(segs, seg_id, idx)} – powered {mode}', value)
    return items


def _powered_object(out: Mapping[str, Any], _segs, _names) -> Items:
    items: Items = {}
    for mode in ('allision', 'grounding'):
        block = (out.get('powered') or {}).get(mode) or {}
        for obs, value in (block.get('by_obstacle') or {}).items():
            _add(items, ('pobj', str(obs), mode), f'{obs} – powered {mode}', value)
    return items


def _oil(out: Mapping[str, Any], _segs, _names) -> Items:
    items: Items = {}
    for acc, rec in ((out.get('consequence') or {}).get('by_accident') or {}).items():
        if isinstance(rec, Mapping):
            _add(items, ('oil', str(acc)), f'{acc} – spill (m³/year)', rec.get('spill_m3'))
    return items


_EXTRACTORS = (
    ('Collisions per leg', _collision_leg),
    ('Collisions per waypoint', _collision_waypoint),
    ('Collisions per leg pair', _collision_pair),
    ('Drifting per leg', _drifting_leg),
    ('Drifting per object', _drifting_object),
    ('Powered per leg and direction', _powered_legdir),
    ('Powered per object', _powered_object),
    ('Oil spill per accident type', _oil),
)


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def build_breakdown_rows(
    snap_a: Mapping[str, Any] | None,
    snap_b: Mapping[str, Any] | None,
    outputs_a: Mapping[str, Any] | None,
    outputs_b: Mapping[str, Any] | None,
) -> list[list[Any]]:
    """Rows ``[category, item, a, b, delta, share_pct]``.

    Items that are zero (or absent) in both runs are dropped.  ``delta``
    is ``b - a`` with an absent side counted as zero; ``share_pct`` is
    ``|delta|`` as a percentage of the summed ``|delta|`` of the category.
    """
    segs_a, segs_b = _segs(snap_a), _segs(snap_b)
    out_a, out_b = outputs_a or {}, outputs_b or {}
    names = _waypoint_names(segs_a)
    for wp, legs in _waypoint_names(segs_b).items():
        names.setdefault(wp, []).extend(legs)
    rows: list[list[Any]] = []
    for category, extract in _EXTRACTORS:
        items_a = extract(out_a, segs_a, names)
        items_b = extract(out_b, segs_b, names)
        cat_rows = []
        for key in set(items_a) | set(items_b):
            la, va = items_a.get(key, (None, None))
            lb, vb = items_b.get(key, (None, None))
            if abs(va or 0.0) < _ZERO and abs(vb or 0.0) < _ZERO:
                continue
            delta = (vb or 0.0) - (va or 0.0)
            cat_rows.append([category, la or lb, va, vb, delta])
        total = sum(abs(r[4]) for r in cat_rows)
        for r in cat_rows:
            r.append(100.0 * abs(r[4]) / total if total > 0 else 0.0)
        cat_rows.sort(key=lambda r: (-abs(r[4]), str(r[1])))
        rows.extend(cat_rows)
    return rows


# Categories that each count every accident once and say *where* it is;
# the leg-pair and object views repeat the same events.
_LOCATION_CATEGORIES = (
    'Collisions per leg', 'Collisions per waypoint',
    'Drifting per leg', 'Powered per leg and direction',
)


def top_movers(rows: list[list[Any]], n: int = 1) -> list[list[Any]]:
    """The ``n`` rows with the largest absolute change in events/year.

    Only the location categories are ranked (leg, waypoint, leg-direction),
    so an event is not counted twice and oil (m³) is not mixed with
    frequencies.
    """
    events = [r for r in rows if r[0] in _LOCATION_CATEGORIES and abs(r[4]) > 0]
    return sorted(events, key=lambda r: -abs(r[4]))[:n]
