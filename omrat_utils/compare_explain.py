"""Why did the ship-ship collisions change between two runs? (QGIS-free)

The Compare tab says *how much* (Totals) and *where* (Breakdown) two
runs differ.  This module says *why*: it splits the change of every
collision type over the groups of inputs that differ between the two
snapshots (:data:`GROUPS`).

Method
------
The ship-ship model is analytic (a couple of seconds per run, unlike
drifting / powered), so it is simply re-run on *mixed* inputs: run A's
data with some groups taken from run B (:func:`hybrid_data`).  With the
value of every coalition of groups known, each group gets its Shapley
value -- its marginal effect averaged over every order in which the
groups could be switched from A to B.  The values do not depend on an
arbitrary order and add up to ``B - A`` exactly; an interaction (two
groups that only matter together) is shared between them.  ``k``
differing groups cost ``2**k`` runs (``MAX_EXACT_GROUPS``); beyond that
the average runs over ``2k`` fixed orders, which still sums exactly.

Run B itself is computed too: anything the groups do not cover shows up
as :data:`RESIDUAL` instead of silently disappearing.

The same runs give the split per leg (head-on, overtaking, bend) and
per leg pair (crossing, merging), and :func:`build_findings` turns the
differences into plain-language findings (suppressed legs and where
their ships went, crossing <-> merging re-classification, narrower
lanes, junction matrices that lost their AIS counts, pairs on the 30
degree merging limit, inputs that differ without touching collisions).

Legs are matched on id **and** name (:func:`compare_common.match_legs`);
a leg present in one run only belongs to the ``legs`` group.
"""
from __future__ import annotations

import copy
import itertools
import math
import re
import time
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .compare_common import finite, leg_traffic, match_legs, parse_xy

__all__ = [
    'TYPES', 'TOTAL', 'TYPE_LABELS', 'GROUPS', 'GROUP_LABELS', 'RESIDUAL',
    'MAX_EXACT_GROUPS', 'MERGING_LIMIT_DEG', 'ItemRow', 'Finding',
    'CollisionExplanation', 'segment_field_group', 'traffic_field_group',
    'hybrid_data', 'model_data', 'active_groups', 'group_differences',
    'run_collision_model', 'shapley_values', 'explain_collisions',
    'merge_split_legs', 'route_name', 'items_for', 'build_findings',
    'findings_for', 'to_markdown', 'type_for_label', 'leg_endpoints',
    'lateral_change', 'short_type',
]

TYPES: tuple[str, ...] = ('head_on', 'overtaking', 'crossing', 'merging', 'bend')
TOTAL = 'total'
TYPE_LABELS: dict[str, str] = {
    'head_on': 'Head-on collision',
    'overtaking': 'Overtaking collision',
    'crossing': 'Crossing collision',
    'merging': 'Merging collision',
    'bend': 'Bend collision',
    TOTAL: 'All collisions',
}

GROUPS: tuple[str, ...] = (
    'routing', 'geometry', 'lateral', 'traffic', 'ships',
    'junctions', 'causation', 'legs', 'other',
)
GROUP_LABELS: dict[str, str] = {
    'routing': 'Routing (suppressed legs / moved traffic)',
    'geometry': 'Leg geometry (waypoints, lengths, angles)',
    'lateral': 'Lateral distributions',
    'traffic': 'Traffic volume',
    'ships': 'Ship properties (speed, beam, length classes)',
    'junctions': 'Junction transition matrices',
    'causation': 'Causation factors',
    'legs': 'Legs added / removed',
    'other': 'Other leg inputs',
}
RESIDUAL = 'unattributed'
GROUP_LABELS[RESIDUAL] = 'Not attributed to an input group'

#: Up to this many differing groups every coalition is run (2**k runs).
MAX_EXACT_GROUPS = 6

#: ``ShipCollisionModelMixin.MERGING_ANGLE_DEG`` (kept here so this module
#: needs no QGIS import; ``tests/test_compare_explain.py`` pins them equal).
MERGING_LIMIT_DEG = 30.0
_LIMIT_BAND_DEG = 1.0

# Top-level blocks the collision run reads.
_MODEL_KEYS: tuple[str, ...] = (
    'segment_data', 'traffic_data', 'junctions', 'pc', 'ship_categories', 'traffic_scaling',
)
_TOP_GROUP = {'junctions': 'junctions', 'ship_categories': 'ships', 'traffic_scaling': 'traffic'}
_PC_KEYS = {'headon': 'head-on', 'overtaking': 'overtaking', 'crossing': 'crossing',
            'merging': 'merging', 'bend': 'bend'}

_GEOMETRY_FIELDS = frozenset({'Start_Point', 'End_Point', 'line_length', 'start_wp', 'end_wp', 'bend_angle'})
_ROUTING_FIELDS = frozenset({'suppressed', 'traffic_redirect', 'suppressed_with'})
# Direction names must stay with the traffic blocks they name.
_TRAFFIC_SEG_FIELDS = frozenset({'Dirs'})
_LATERAL_PREFIXES = ('mean', 'std', 'weight', 'u_min', 'u_max', 'u_p', 'dist')
#: Leg fields the collision model never reads, with the reason shown to the user.
NO_EFFECT_FIELDS: dict[str, str] = {
    'ai1': 'powered grounding / allision only',
    'ai2': 'powered grounding / allision only',
    'Width': 'drawing and AIS sample only',
    'Tangent_Pos': 'AIS passage line only',
    'traffic_locked': 'AIS update lock',
    'traffic_source': 'copy provenance',
    'Route_Id': 'label',
    'Segment_Id': 'label',
    'Leg_name': 'label',
}
_SHIP_FIELDS = frozenset({'Speed (knots)', 'Ship Beam (meters)', 'Draught (meters)', 'Ship heights (meters)'})
_OTHER_BLOCKS = {
    'depths': 'Depth areas', 'objects': 'Structures', 'drift': 'Drifting settings',
    'consequence': 'Oil-spill consequence inputs',
}
_ZERO = 1e-300


# ---------------------------------------------------------------------------
# Field classification and mixed inputs
# ---------------------------------------------------------------------------

def segment_field_group(name: str) -> str | None:
    """Input group of a ``segment_data`` field; ``None`` = not read by the collision model."""
    if name in _GEOMETRY_FIELDS:
        return 'geometry'
    if name in _ROUTING_FIELDS:
        return 'routing'
    if name in _TRAFFIC_SEG_FIELDS:
        return 'traffic'
    if name in NO_EFFECT_FIELDS:
        return None
    if name.startswith(_LATERAL_PREFIXES):
        return 'lateral'
    return 'other'


def traffic_field_group(name: str) -> str:
    """Input group of a field of a ``traffic_data[seg][dir]`` block."""
    return 'ships' if name in _SHIP_FIELDS else 'traffic'


def _segs(snap: Mapping[str, Any] | None) -> dict[str, Any]:
    seg = (snap or {}).get('segment_data') or {}
    return {str(k): v for k, v in seg.items() if isinstance(v, Mapping)}


def _traffic(snap: Mapping[str, Any] | None) -> dict[str, Any]:
    td = (snap or {}).get('traffic_data') or {}
    return {str(k): v for k, v in td.items()} if isinstance(td, Mapping) else {}


def model_data(snap: Mapping[str, Any]) -> dict[str, Any]:
    """Private copy of the blocks the collision run reads (keys as strings)."""
    out = {k: copy.deepcopy(snap[k]) for k in _MODEL_KEYS if k in snap}
    out['segment_data'] = copy.deepcopy(_segs(snap))
    out['traffic_data'] = copy.deepcopy(_traffic(snap))
    return out


def _copy_field(dst: dict, src: Mapping, name: str) -> None:
    if name in src:
        dst[name] = copy.deepcopy(src[name])
    else:
        dst.pop(name, None)


def _mix_traffic(traffic: dict[str, Any], td_b: Mapping[str, Any], seg: str, groups: set[str]) -> None:
    want_traffic, want_ships = 'traffic' in groups, 'ships' in groups
    if not (want_traffic or want_ships):
        return
    blk_a, blk_b = traffic.get(seg), td_b.get(seg)
    if not isinstance(blk_a, dict) or not isinstance(blk_b, Mapping) or set(blk_a) != set(blk_b):
        # Different directions (or one side without traffic): the whole
        # block travels with the traffic group.
        if want_traffic:
            _copy_field(traffic, td_b, seg)
        return
    for d, var_b in blk_b.items():
        var_a = blk_a.get(d)
        if not isinstance(var_a, dict) or not isinstance(var_b, Mapping):
            if want_traffic:
                blk_a[d] = copy.deepcopy(var_b)
            continue
        for name in set(var_a) | set(var_b):
            if traffic_field_group(name) in groups:
                _copy_field(var_a, var_b, name)


def hybrid_data(
    snap_a: Mapping[str, Any], snap_b: Mapping[str, Any], groups: Iterable[str],
) -> dict[str, Any]:
    """Run A's model inputs with every input of ``groups`` taken from run B.

    ``hybrid_data(a, b, ())`` is run A, ``hybrid_data(a, b, GROUPS)`` is
    run B as far as the collision model can tell.
    """
    groups = set(groups)
    out = model_data(snap_a)
    segs, traffic = out['segment_data'], out['traffic_data']
    seg_b, td_b = _segs(snap_b), _traffic(snap_b)
    matched, only_a, only_b = match_legs(_segs(snap_a), seg_b)
    for seg in matched:
        sa, sb = segs[seg], seg_b[seg]
        for name in set(sa) | set(sb):
            if segment_field_group(name) in groups:
                _copy_field(sa, sb, name)
        _mix_traffic(traffic, td_b, seg, groups)
    if 'legs' in groups:
        for seg in only_a:
            segs.pop(seg, None)
            traffic.pop(seg, None)
        for seg in only_b:
            segs[seg] = copy.deepcopy(seg_b[seg])
            if seg in td_b:
                traffic[seg] = copy.deepcopy(td_b[seg])
    for key, group in _TOP_GROUP.items():
        if group in groups:
            _copy_field(out, snap_b, key)
    if 'causation' in groups:
        # Only the collision factors: the others never reach this model.
        pc = dict(out.get('pc') or {})
        for key in _PC_KEYS:
            _copy_field(pc, snap_b.get('pc') or {}, key)
        if pc != (out.get('pc') or {}):
            out['pc'] = pc
    return out


def active_groups(snap_a: Mapping[str, Any], snap_b: Mapping[str, Any]) -> list[str]:
    """The groups whose inputs differ, i.e. that change the model data when taken from B."""
    base = hybrid_data(snap_a, snap_b, ())
    return [g for g in GROUPS if hybrid_data(snap_a, snap_b, (g,)) != base]


# ---------------------------------------------------------------------------
# Describing the differences
# ---------------------------------------------------------------------------

def _name(segs: Mapping[str, Any], seg: str) -> str:
    return str((segs.get(seg) or {}).get('Leg_name') or f'Leg {seg}')


def _mixture(seg: Mapping[str, Any], d: int) -> tuple[float, float] | None:
    """Weighted mean / std of direction ``d`` (1 or 2) as the model collapses it."""
    parts: list[tuple[float, float, float]] = []
    for i in (1, 2, 3):
        w, mu, sd = (finite(seg.get(f'{k}{d}_{i}')) for k in ('weight', 'mean', 'std'))
        if w and w > 0 and mu is not None and sd and sd > 0:
            parts.append((w, mu, sd * sd))
    up = finite(seg.get(f'u_p{d}'))
    lo, hi = finite(seg.get(f'u_min{d}')), finite(seg.get(f'u_max{d}'))
    if up and up > 0 and lo is not None and hi is not None:
        parts.append((up, (lo + hi) / 2.0, (hi - lo) ** 2 / 12.0))
    total = sum(p[0] for p in parts)
    if total <= 0:
        return None
    mean = sum(w * mu for w, mu, _ in parts) / total
    var = sum(w * (v + mu * mu) for w, mu, v in parts) / total - mean * mean
    return mean, math.sqrt(max(var, 0.0))


def _fmt_pair(a: tuple[float, float] | None, b: tuple[float, float] | None, idx: int) -> str:
    return '/'.join('-' if x is None else f'{x[idx]:.0f}' for x in (a, b))


def lateral_change(seg_a: Mapping[str, Any], seg_b: Mapping[str, Any]) -> str:
    """``"sigma 1421/1344 -> 500/500 m; mean -58/290 -> 0/500 m"`` (dir 1 / dir 2)."""
    a1, a2 = _mixture(seg_a, 1), _mixture(seg_a, 2)
    b1, b2 = _mixture(seg_b, 1), _mixture(seg_b, 2)
    parts = []
    sig_a, sig_b = _fmt_pair(a1, a2, 1), _fmt_pair(b1, b2, 1)
    mu_a, mu_b = _fmt_pair(a1, a2, 0), _fmt_pair(b1, b2, 0)
    if sig_a != sig_b:
        parts.append(f'sigma {sig_a} -> {sig_b} m (dir 1/dir 2)')
    if mu_a != mu_b:
        parts.append(f'mean {mu_a} -> {mu_b} m')
    return '; '.join(parts) or 'distribution components changed (same mean and sigma)'


def _fields_by_group(sa: Mapping[str, Any], sb: Mapping[str, Any]) -> dict[str | None, list[str]]:
    out: dict[str | None, list[str]] = {}
    for name in sorted(set(sa) | set(sb)):
        if sa.get(name) != sb.get(name) or (name in sa) != (name in sb):
            out.setdefault(segment_field_group(name), []).append(name)
    return out


def _km(value: Any) -> str:
    v = finite(value)
    return '-' if v is None else f'{v / 1000.0:.1f} km'


def group_differences(snap_a: Mapping[str, Any], snap_b: Mapping[str, Any]) -> dict[str, list[str]]:
    """One readable line per difference, keyed by group (``None`` key: no effect on collisions)."""
    seg_a, seg_b = _segs(snap_a), _segs(snap_b)
    td_a, td_b = _traffic(snap_a), _traffic(snap_b)
    matched, only_a, only_b = match_legs(seg_a, seg_b)
    out: dict[str, list[str]] = {}
    no_effect: dict[str, list[str]] = {}
    for seg in matched:
        sa, sb, name = seg_a[seg], seg_b[seg], _name(seg_a, seg)
        by_group = _fields_by_group(sa, sb)
        if 'geometry' in by_group:
            la, lb = sa.get('line_length'), sb.get('line_length')
            text = f'{_km(la)} -> {_km(lb)}' if la != lb else 'end points moved'
            out.setdefault('geometry', []).append(f'{name}: {text}')
        if 'lateral' in by_group:
            out.setdefault('lateral', []).append(f'{name}: {lateral_change(sa, sb)}')
        if 'routing' in by_group:
            sup_a, sup_b = bool(sa.get('suppressed')), bool(sb.get('suppressed'))
            if sup_a != sup_b:
                text = 'suppressed in B' if sup_b else 'suppressed in A only'
            else:
                text = 'redirect changed'
            out.setdefault('routing', []).append(f'{name}: {text}')
        if 'traffic' in by_group:
            out.setdefault('traffic', []).append(f'{name}: direction names differ')
        if 'other' in by_group:
            out.setdefault('other', []).append(f'{name}: {", ".join(by_group["other"])}')
        for f in by_group.get(None, []):
            if f != 'Leg_name':
                no_effect.setdefault(f, []).append(name)
        qa, qb = leg_traffic(td_a, seg), leg_traffic(td_b, seg)
        if abs(qa - qb) > 1e-9 * max(1.0, abs(qa)):
            out.setdefault('traffic', []).append(f'{name}: {qa:.0f} -> {qb:.0f} ships/year')
        blk_a, blk_b = td_a.get(seg) or {}, td_b.get(seg) or {}
        if isinstance(blk_a, Mapping) and isinstance(blk_b, Mapping):
            ships = sorted({f for d in set(blk_a) & set(blk_b)
                            for f in _SHIP_FIELDS
                            if (blk_a.get(d) or {}).get(f) != (blk_b.get(d) or {}).get(f)})
            if ships:
                out.setdefault('ships', []).append(f'{name}: {", ".join(s.split(" (")[0] for s in ships)}')
    if only_a:
        out.setdefault('legs', []).append('Only in A: ' + ', '.join(_name(seg_a, s) for s in only_a))
    if only_b:
        out.setdefault('legs', []).append('Only in B: ' + ', '.join(_name(seg_b, s) for s in only_b))
    if (snap_a.get('traffic_scaling') or {}) != (snap_b.get('traffic_scaling') or {}):
        out.setdefault('traffic', []).append('Traffic scaling settings differ')
    if (snap_a.get('ship_categories') or {}) != (snap_b.get('ship_categories') or {}):
        out.setdefault('ships', []).append('Ship categories (length classes) differ')
    pc_a, pc_b = snap_a.get('pc') or {}, snap_b.get('pc') or {}
    for key, label in _PC_KEYS.items():
        if pc_a.get(key) != pc_b.get(key):
            out.setdefault('causation', []).append(f'{label}: {pc_a.get(key)} -> {pc_b.get(key)}')
    out.update(_junction_differences(snap_a, snap_b))
    for f, legs in sorted(no_effect.items()):
        why = NO_EFFECT_FIELDS.get(f, '')
        out.setdefault('_no_effect', []).append(
            f'{f} ({why}) differs on {len(legs)} leg(s): {", ".join(legs[:6])}{" ..." if len(legs) > 6 else ""}')
    for key, label in _OTHER_BLOCKS.items():
        if snap_a.get(key) != snap_b.get(key):
            out.setdefault('_no_effect', []).append(f'{label} differ')
    if any(pc_a.get(k) != pc_b.get(k) for k in set(pc_a) | set(pc_b) if k not in _PC_KEYS):
        out.setdefault('_no_effect', []).append('Causation factors of the other accident types differ')
    return out


def _junction_label(jn: Mapping[str, Any], seg_a: Mapping, seg_b: Mapping) -> str:
    names = []
    for leg in sorted((jn.get('legs') or {}), key=str):
        names.append(_name(seg_b, str(leg)) if str(leg) in seg_b else _name(seg_a, str(leg)))
    pt = jn.get('point')
    where = f' ({pt[0]:.4f} {pt[1]:.4f})' if isinstance(pt, (list, tuple)) and len(pt) >= 2 else ''
    return ' / '.join(names) + where


def _junction_differences(snap_a: Mapping[str, Any], snap_b: Mapping[str, Any]) -> dict[str, list[str]]:
    ja, jb = snap_a.get('junctions') or {}, snap_b.get('junctions') or {}
    if not isinstance(ja, Mapping) or not isinstance(jb, Mapping):
        return {}
    seg_a, seg_b = _segs(snap_a), _segs(snap_b)
    lines: list[str] = []
    for jid in sorted(set(ja) | set(jb)):
        a, b = ja.get(jid), jb.get(jid)
        if a == b:
            continue
        if not isinstance(a, Mapping) or not isinstance(b, Mapping):
            lines.append(f'Junction {_junction_label(a or b or {}, seg_a, seg_b)}: only in {"B" if a is None else "A"}')
            continue
        src_a, src_b = a.get('source'), b.get('source')
        what = f'source {src_a} -> {src_b}' if src_a != src_b else 'matrix edited'
        if a.get('legs') != b.get('legs'):
            what += ', legs differ'
        lines.append(f'Junction {_junction_label(b, seg_a, seg_b)}: {what}')
    return {'junctions': lines} if lines else {}


# ---------------------------------------------------------------------------
# Running the model
# ---------------------------------------------------------------------------

def _collision_host():
    from compute.ship_collision_model import ShipCollisionModelMixin

    class _Host(ShipCollisionModelMixin):
        """Bare host: no UI, no progress, no result layers."""

        def __init__(self) -> None:
            self.p = types.SimpleNamespace(main_widget=None)
            self.collision_report: dict[str, Any] = {}
            self.ship_collision_prob = 0.0

        def _report_progress(self, *_args: Any) -> bool:
            return True

        def _finalize_collision_layers(self, *_args: Any) -> None:
            return None

    return _Host()


def run_collision_model(data: dict[str, Any]) -> dict[str, Any]:
    """Collision report for ``data`` exactly as Run model computes it.

    Applies traffic scaling and the suppressed-leg redirects first, like
    ``CalculationTask.run``.  ``data`` is consumed (pass a private copy).
    """
    from compute.data_preparation import apply_traffic_scaling
    from compute.traffic_redirect import apply_traffic_redirects
    apply_traffic_scaling(data)
    apply_traffic_redirects(data)
    host = _collision_host()
    host.run_ship_collision_model(data)
    return host.collision_report or {}


# ---------------------------------------------------------------------------
# Shapley attribution
# ---------------------------------------------------------------------------

def _orders(players: list[str]) -> list[list[str]]:
    """``2k`` fixed orders (every rotation, forwards and backwards)."""
    k = len(players)
    rot = [players[i:] + players[:i] for i in range(k)]
    return rot + [list(reversed(r)) for r in rot]


def _coalitions(players: list[str]) -> list[frozenset]:
    if len(players) <= MAX_EXACT_GROUPS:
        return [frozenset(c) for r in range(len(players) + 1)
                for c in itertools.combinations(players, r)]
    seen: dict[frozenset, None] = {}
    for order in _orders(players):
        for i in range(len(order) + 1):
            seen[frozenset(order[:i])] = None
    return list(seen)


def shapley_values(players: list[str], value: Callable[[frozenset], float]) -> dict[str, float]:
    """Shapley value of every player; ``sum == value(all) - value(none)``.

    Exact up to :data:`MAX_EXACT_GROUPS` players, otherwise averaged over
    :func:`_orders` (each order sums exactly, so the average does too).
    """
    n = len(players)
    phi = {p: 0.0 for p in players}
    if n == 0:
        return phi
    if n > MAX_EXACT_GROUPS:
        orders = _orders(players)
        for order in orders:
            prev = frozenset()
            for p in order:
                cur = prev | {p}
                phi[p] += (value(cur) - value(prev)) / len(orders)
                prev = cur
        return phi
    for p in players:
        others = [q for q in players if q != p]
        for r in range(n):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for sub in itertools.combinations(others, r):
                s = frozenset(sub)
                phi[p] += w * (value(s | {p}) - value(s))
    return phi


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass
class ItemRow:
    """One leg (head-on / overtaking / bend) or leg pair (crossing / merging)."""
    kind: str                       # 'leg' | 'pair'
    acc_type: str
    label: str
    legs: tuple[tuple[str, str], ...]   # (segment id, leg name) for the map
    a: float
    b: float
    by_group: dict[str, float] = field(default_factory=dict)
    angle_a: float | None = None
    angle_b: float | None = None

    @property
    def delta(self) -> float:
        return self.b - self.a


@dataclass
class Finding:
    level: str                      # 'warning' | 'info'
    title: str
    details: list[str] = field(default_factory=list)
    types: tuple[str, ...] = ()     # accident types concerned; () = general


@dataclass
class CollisionExplanation:
    groups: list[str]
    differences: dict[str, list[str]]
    totals_a: dict[str, float]
    totals_b: dict[str, float]
    by_group: dict[str, dict[str, float]]
    items: list[ItemRow]
    findings: list[Finding] = field(default_factory=list)
    runs: int = 0
    seconds: float = 0.0
    exact: bool = True
    stored_a: dict[str, float] = field(default_factory=dict)
    stored_b: dict[str, float] = field(default_factory=dict)

    def delta(self, acc_type: str) -> float:
        return self.totals_b.get(acc_type, 0.0) - self.totals_a.get(acc_type, 0.0)

    def identical_groups(self) -> list[str]:
        return [g for g in GROUPS if g not in self.groups]

    def columns(self) -> list[str]:
        """The attributed groups in :data:`GROUPS` order, plus :data:`RESIDUAL` when any type has one."""
        resid = any(RESIDUAL in rec for rec in self.by_group.values())
        return [g for g in GROUPS if g in self.groups] + ([RESIDUAL] if resid else [])

    def contribution_rows(self, acc_type: str) -> list[tuple[str, float]]:
        """``[(group, contribution)]`` for :meth:`columns`."""
        rec = self.by_group.get(acc_type, {})
        return [(g, rec.get(g, 0.0)) for g in self.columns()]


def _totals(report: Mapping[str, Any]) -> dict[str, float]:
    t = report.get('totals') or {}
    out = {k: float(finite(t.get(k)) or 0.0) for k in TYPES}
    out[TOTAL] = sum(out[k] for k in TYPES)
    return out


def _report_items(report: Mapping[str, Any], names: Mapping[str, str]) -> tuple[dict, dict]:
    """``({key: value}, {key: angle_deg})`` for one run's collision report."""
    values: dict[tuple, float] = {}
    angles: dict[tuple, float] = {}
    for seg, rec in (report.get('by_leg') or {}).items():
        if not isinstance(rec, Mapping):
            continue
        ident = (str(seg), names.get(str(seg), ''))
        for t in ('head_on', 'overtaking', 'bend'):
            v = finite(rec.get(t))
            if v:
                values[('leg', ident, t)] = values.get(('leg', ident, t), 0.0) + v
    for pair, rec in (report.get('by_leg_pair') or {}).items():
        if not isinstance(rec, Mapping):
            continue
        a, _, b = str(pair).partition('->')
        legs = frozenset({(a, names.get(a, '')), (b, names.get(b, ''))})
        ang = finite(rec.get('angle_deg'))
        for t in ('crossing', 'merging'):
            key = ('pair', legs, t)
            v = finite(rec.get(t))
            if v:
                values[key] = values.get(key, 0.0) + v
            if ang is not None:
                angles[key] = ang
    return values, angles


def _leg_label(ident: tuple[str, str]) -> str:
    return ident[1] or f'Leg {ident[0]}'


def _item_label(kind: str, legs: Any) -> tuple[str, tuple[tuple[str, str], ...]]:
    if kind == 'leg':
        return _leg_label(legs), (legs,)
    ordered = tuple(sorted(legs, key=lambda x: (x[1], x[0])))
    return ' x '.join(_leg_label(x) for x in ordered), ordered


def _stored_totals(outputs: Mapping[str, Any] | None) -> dict[str, float]:
    t = ((outputs or {}).get('collision') or {}).get('totals') or {}
    out = {k: v for k in TYPES if (v := finite(t.get(k))) is not None}
    if out:
        out[TOTAL] = sum(out.values())
    return out


def explain_collisions(
    snap_a: Mapping[str, Any],
    snap_b: Mapping[str, Any],
    outputs_a: Mapping[str, Any] | None = None,
    outputs_b: Mapping[str, Any] | None = None,
    run: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> CollisionExplanation | None:
    """Attribute the change of every collision type to the input groups.

    ``run`` defaults to :func:`run_collision_model`; ``outputs_*`` are the
    runs' sidecars (``compare.load_run_outputs``), only used to check that
    the recomputed values match the stored ones.  Returns ``None`` when
    cancelled.
    """
    run = run or run_collision_model
    cancelled = is_cancelled or (lambda: False)
    t0 = time.monotonic()
    groups = active_groups(snap_a, snap_b)
    coalitions = _coalitions(groups)
    n_runs = len(coalitions) + 1
    cache: dict[frozenset, tuple[dict, dict, dict]] = {}

    def _one(data: dict[str, Any]) -> tuple[dict, dict, dict]:
        names = {str(k): str((v or {}).get('Leg_name') or '') for k, v in data['segment_data'].items()}
        report = run(data)
        values, angles = _report_items(report, names)
        return _totals(report), values, angles

    for i, coal in enumerate(coalitions):
        if cancelled():
            return None
        what = ' + '.join(g for g in groups if g in coal) or 'run A'
        if on_progress is not None:
            on_progress(i, n_runs, f'Model run {i + 1}/{n_runs}: {what}')
        cache[coal] = _one(hybrid_data(snap_a, snap_b, coal))
    if cancelled():
        return None
    if on_progress is not None:
        on_progress(n_runs - 1, n_runs, f'Model run {n_runs}/{n_runs}: run B')
    tot_b, val_b, ang_b = _one(model_data(snap_b))
    empty, full = frozenset(), frozenset(groups)
    tot_a, val_a, ang_a = cache[empty]
    tot_full, val_full, _ = cache[full]

    by_group: dict[str, dict[str, float]] = {}
    for t in TYPES + (TOTAL,):
        rec = shapley_values(groups, lambda c, t=t: cache[c][0][t])
        resid = tot_b[t] - tot_full[t]
        if abs(resid) > 1e-12 * max(abs(tot_a[t]), abs(tot_b[t]), _ZERO):
            rec[RESIDUAL] = resid
        by_group[t] = rec

    keys = set(val_b)
    for cached in cache.values():
        keys |= set(cached[1])
    items: list[ItemRow] = []
    for key in keys:
        kind, legs, t = key
        a, b = val_a.get(key, 0.0), val_b.get(key, 0.0)
        rec = shapley_values(groups, lambda c, key=key: cache[c][1].get(key, 0.0))
        resid = b - val_full.get(key, 0.0)
        if abs(resid) > 1e-12 * max(abs(a), abs(b), _ZERO):
            rec[RESIDUAL] = resid
        if abs(a) < _ZERO and abs(b) < _ZERO:
            continue  # absent in both runs; any group values are interactions that cancel
        label, ordered = _item_label(kind, legs)
        items.append(ItemRow(kind, t, label, ordered, a, b, rec, ang_a.get(key), ang_b.get(key)))
    items.sort(key=lambda r: (-abs(r.delta), r.label))

    expl = CollisionExplanation(
        groups=groups, differences=group_differences(snap_a, snap_b),
        totals_a=tot_a, totals_b=tot_b, by_group=by_group, items=items,
        runs=n_runs, exact=len(groups) <= MAX_EXACT_GROUPS,
        stored_a=_stored_totals(outputs_a), stored_b=_stored_totals(outputs_b),
    )
    expl.findings = build_findings(snap_a, snap_b, expl)
    expl.seconds = time.monotonic() - t0
    if on_progress is not None:
        on_progress(n_runs, n_runs, f'Done: {n_runs} model runs')
    return expl


# ---------------------------------------------------------------------------
# Items: filter, merge split legs
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r'^(.*\d)_([a-z]{1,2})$')


def route_name(leg_name: str) -> str:
    """``"LEG_1_2_a"`` -> ``"LEG_1_2"``; names without a split suffix unchanged."""
    m = _SPLIT_RE.match(leg_name or '')
    return m.group(1) if m else leg_name


def merge_split_legs(rows: Iterable[ItemRow]) -> list[ItemRow]:
    """Sum the sub-legs of a split route (``LEG_1_2_a`` + ``LEG_1_2_b``).

    Moving a split point moves length from one sub-leg to the other; per
    sub-leg that looks like two large changes that cancel.
    """
    merged: dict[tuple, ItemRow] = {}
    counts: dict[tuple, int] = {}
    for r in rows:
        routes = tuple(sorted({route_name(_leg_label(x)) for x in r.legs}))
        if r.kind == 'pair' and len(routes) == 1:
            label = f'{routes[0]} (between its sub-legs)'
        else:
            label = ' x '.join(routes)
        key = (r.kind, r.acc_type, routes)
        if key not in merged:
            merged[key] = ItemRow(r.kind, r.acc_type, label, tuple(r.legs), r.a, r.b,
                                  dict(r.by_group), r.angle_a, r.angle_b)
            counts[key] = 1
            continue
        m = merged[key]
        m.a += r.a
        m.b += r.b
        m.legs = tuple(dict.fromkeys(m.legs + tuple(r.legs)))
        for g, v in r.by_group.items():
            m.by_group[g] = m.by_group.get(g, 0.0) + v
        m.angle_a = m.angle_b = None
        counts[key] += 1
    for key, m in merged.items():
        if counts[key] > 1 and m.kind == 'leg':
            m.label = f'{m.label} ({counts[key]} sub-legs)'
    out = list(merged.values())
    out.sort(key=lambda r: (-abs(r.delta), r.label))
    return out


def items_for(expl: CollisionExplanation, acc_type: str, merge_split: bool = False) -> list[ItemRow]:
    """Item rows of one accident type (``TOTAL``: all types), largest change first."""
    rows = [r for r in expl.items if acc_type == TOTAL or r.acc_type == acc_type]
    if merge_split:
        rows = merge_split_legs(rows)
    return sorted(rows, key=lambda r: (-abs(r.delta), r.label))


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def _e(v: float) -> str:
    return '0' if abs(v) < _ZERO else f'{v:+.2e}'


def short_type(acc_type: str) -> str:
    """``"head_on"`` -> ``"head-on"``."""
    return TYPE_LABELS[acc_type].split()[0].lower()


def _effects(expl: CollisionExplanation, group: str, min_share: float = 0.02) -> tuple[str, tuple[str, ...]]:
    """``("crossing -8.55e-04, merging +5.08e-04", types)`` for one group.

    Types where the group moves less than ``min_share`` of the type's
    change are left out (they would only clutter the title).
    """
    parts, concerned = [], []
    for t in TYPES:
        v = expl.by_group.get(t, {}).get(group, 0.0)
        d = abs(expl.delta(t))
        if abs(v) <= _ZERO or (d > 0 and abs(v) < min_share * d):
            continue
        parts.append(f'{short_type(t)} {_e(v)}')
        concerned.append(t)
    return ', '.join(parts), tuple(concerned)


def _group_items(expl: CollisionExplanation, group: str, types: Iterable[str] | None = None) -> list[ItemRow]:
    wanted = set(types or TYPES)
    rows = [r for r in expl.items if r.acc_type in wanted and abs(r.by_group.get(group, 0.0)) > _ZERO]
    return sorted(rows, key=lambda r: -abs(r.by_group.get(group, 0.0)))


def _angle_text(r: ItemRow, run: str) -> str:
    ang = r.angle_b if run == 'b' else r.angle_a
    return '' if ang is None else f' at {ang:.1f} deg'


def _finding_consistency(expl: CollisionExplanation) -> list[Finding]:
    out = []
    for run, stored, now in (('A', expl.stored_a, expl.totals_a), ('B', expl.stored_b, expl.totals_b)):
        off = [t for t in TYPES if t in stored
               and abs(stored[t] - now[t]) > 1e-6 * max(abs(stored[t]), abs(now[t]), 1e-15)]
        if off:
            t = max(off, key=lambda k: abs(stored[k] - now[k]))
            out.append(Finding('warning', f'Run {run} recomputes differently from its stored result', [
                f'{TYPE_LABELS[t]}: stored {stored[t]:.3e}, recomputed {now[t]:.3e}.',
                'The run was probably made with another OMRAT version.  The explanation uses the '
                'recomputed values for both runs, so it stays consistent.']))
    return out


def _redirect_moves(snap: Mapping[str, Any]) -> tuple[dict, dict[str, float]]:
    """``(summary, ships per leg before the move)`` of a run's suppressed-leg redirects."""
    try:
        from compute.data_preparation import apply_traffic_scaling
        from compute.traffic_redirect import apply_traffic_redirects
    except Exception:  # noqa: BLE001 - descriptive extra only
        return {'moves': []}, {}
    data = model_data(snap)
    apply_traffic_scaling(data)
    ships = {seg: leg_traffic(data['traffic_data'], seg) for seg in data['traffic_data']}
    return apply_traffic_redirects(data), ships


def _finding_routing(snap_a, snap_b, expl: CollisionExplanation) -> list[Finding]:
    if 'routing' not in expl.groups:
        return []
    seg_a, seg_b = _segs(snap_a), _segs(snap_b)
    effects, concerned = _effects(expl, 'routing')
    details = []
    for run, snap, segs, other in (('B', snap_b, seg_b, seg_a), ('A', snap_a, seg_a, seg_b)):
        summary, ships = _redirect_moves(snap)
        by_src: dict[str, dict[str, float]] = {}
        for mv in summary.get('moves') or []:
            by_src.setdefault(str(mv['src']), {})
            dst = str(mv['dst'])
            by_src[str(mv['src'])][dst] = by_src[str(mv['src'])].get(dst, 0.0) + float(mv.get('ships') or 0.0)
        for seg, sd in sorted(segs.items()):
            if not sd.get('suppressed'):
                continue
            od = other.get(seg) or {}
            if all(od.get(f) == sd.get(f) for f in _ROUTING_FIELDS):
                continue  # suppressed the same way in both runs
            lead = sd.get('suppressed_with')
            if lead:
                details.append(f'{_name(segs, seg)} is suppressed in {run} together with {_name(segs, str(lead))}.')
                continue
            dsts = by_src.get(seg) or {}
            moved = ', '.join(f'{_name(segs, d)} ({v:.0f})' for d, v in sorted(dsts.items(), key=lambda x: -x[1]))
            text = f'{_name(segs, seg)} ({ships.get(seg, 0.0):.0f} ships/year) is suppressed in {run}'
            details.append(text + (f'; its ships move to {moved}.' if moved else '; its ships are not moved.'))
    tops = _group_items(expl, 'routing')[:3]
    if tops:
        details.append('Largest effects: ' + '; '.join(
            f'{r.label} {short_type(r.acc_type)} {_e(r.by_group["routing"])}' for r in tops))
    return [Finding('info', f'Routing differs: {effects or "no effect on collisions"}', details, concerned)]


def _finding_reclassification(expl: CollisionExplanation) -> list[Finding]:
    out = []
    dc, dm = expl.delta('crossing'), expl.delta('merging')
    for g in expl.groups:
        pc = expl.by_group.get('crossing', {}).get(g, 0.0)
        pm = expl.by_group.get('merging', {}).get(g, 0.0)
        if pc * pm >= 0 or abs(pc) < 0.25 * abs(dc) or abs(pm) < 0.25 * abs(dm):
            continue
        if min(abs(pc), abs(pm)) < 0.1 * max(abs(pc), abs(pm)):
            continue
        details = ['Leg pairs meeting at 30 deg or less count as merging, steeper ones as crossing.  '
                   'When traffic is moved, the same ships can meet another flow at a new angle '
                   'and move from one type to the other.']
        for t, sign in (('crossing', pc), ('merging', pm)):
            rows = [r for r in _group_items(expl, g, (t,)) if r.by_group[g] * sign > 0]
            if rows:
                r = rows[0]
                run = 'b' if r.by_group[g] > 0 else 'a'
                details.append(f'Largest {t} change: {r.label}{_angle_text(r, run)}: {_e(r.by_group[g])}')
        out.append(Finding(
            'info', f'Crossing ({_e(pc)}) and merging ({_e(pm)}) move in opposite directions through '
                    f'{GROUP_LABELS[g].split(" (")[0].lower()}', details, ('crossing', 'merging')))
    return out


def _leg_effect(expl: CollisionExplanation, group: str, seg: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in expl.items:
        if r.kind == 'leg' and r.legs and r.legs[0][0] == seg:
            v = r.by_group.get(group, 0.0)
            if abs(v) > _ZERO:
                out[r.acc_type] = out.get(r.acc_type, 0.0) + v
    return out


def _effect_text(eff: Mapping[str, float]) -> str:
    return ', '.join(f'{short_type(t)} {_e(v)}' for t, v in sorted(eff.items(), key=lambda x: -abs(x[1])))


def _finding_lateral(snap_a, snap_b, expl: CollisionExplanation) -> list[Finding]:
    if 'lateral' not in expl.groups:
        return []
    seg_a, seg_b = _segs(snap_a), _segs(snap_b)
    matched, _, _ = match_legs(seg_a, seg_b)
    legs = []
    for seg in matched:
        if 'lateral' not in _fields_by_group(seg_a[seg], seg_b[seg]):
            continue
        eff = _leg_effect(expl, 'lateral', seg)
        legs.append((sum(abs(v) for v in eff.values()), seg, eff))
    legs.sort(key=lambda x: -x[0])
    effects, concerned = _effects(expl, 'lateral')
    details = [f'{_name(seg_a, seg)}: {lateral_change(seg_a[seg], seg_b[seg])}'
               + (f'; effect {_effect_text(eff)}' if eff else '') for _, seg, eff in legs[:6]]
    if len(legs) > 6:
        details.append(f'... and {len(legs) - 6} more leg(s).')
    details.append('A narrower distribution packs the traffic closer together, which raises head-on and '
                   'overtaking; a wider gap between the two directions lowers head-on.')
    return [Finding('info', f'Lateral distributions differ on {len(legs)} leg(s): {effects or "no effect"}',
                    details, concerned)]


def _finding_geometry(snap_a, snap_b, expl: CollisionExplanation) -> list[Finding]:
    if 'geometry' not in expl.groups:
        return []
    seg_a, seg_b = _segs(snap_a), _segs(snap_b)
    matched, _, _ = match_legs(seg_a, seg_b)
    legs = []
    for seg in matched:
        if 'geometry' not in _fields_by_group(seg_a[seg], seg_b[seg]):
            continue
        eff = _leg_effect(expl, 'geometry', seg)
        legs.append((sum(abs(v) for v in eff.values()), seg, eff))
    legs.sort(key=lambda x: -x[0])
    effects, concerned = _effects(expl, 'geometry')
    details = []
    for _, seg, eff in legs[:6]:
        la, lb = seg_a[seg].get('line_length'), seg_b[seg].get('line_length')
        moved = f'{_km(la)} -> {_km(lb)}' if la != lb else 'end points moved'
        details.append(f'{_name(seg_a, seg)}: {moved}' + (f'; effect {_effect_text(eff)}' if eff else ''))
    if len(legs) > 6:
        details.append(f'... and {len(legs) - 6} more leg(s).')
    routes: dict[str, list[float]] = {}
    for _, seg, eff in legs:
        base = route_name(_name(seg_a, seg))
        if base != _name(seg_a, seg):
            routes.setdefault(base, []).append(sum(eff.values()))
    largest = legs[0][0] if legs else 0.0
    for base, vals in sorted(routes.items()):
        gross = sum(abs(v) for v in vals)
        if len(vals) > 1 and gross >= 0.1 * largest > 0 and abs(sum(vals)) < 0.5 * gross:
            details.append(f'{base}: the split point moved; the sub-legs change by {gross:.2e} in total '
                           f'but net only {_e(sum(vals))} (tick "Merge split legs").')
    details.append('Head-on and overtaking scale with leg length; crossing and merging change with the '
                   'angle at which two legs meet.')
    return [Finding('info', f'Leg geometry differs on {len(legs)} leg(s): {effects or "no effect"}',
                    details, concerned)]


def _finding_junctions(snap_a, snap_b, expl: CollisionExplanation) -> list[Finding]:
    lines = expl.differences.get('junctions') or []
    if 'junctions' not in expl.groups and not lines:
        return []
    effects, concerned = _effects(expl, 'junctions')
    lost_ais = [ln for ln in lines if 'source ais -> geometry' in ln]
    details = list(lines[:8])
    if len(lines) > 8:
        details.append(f'... and {len(lines) - 8} more.')
    tops = _group_items(expl, 'junctions')[:2]
    if tops:
        details.append('Largest effects: ' + '; '.join(
            f'{r.label} {short_type(r.acc_type)} {_e(r.by_group["junctions"])}' for r in tops))
    if lost_ais:
        details.append('A junction falls back to the geometric default when a leg at it is edited (waypoint, '
                       'width or tangent) - usually not part of the scenario.  If it is not meant, re-count the '
                       'junctions in B (Update all distributions; it also refreshes traffic and distributions) '
                       'or enter the rows under Settings > Junction transition matrix.')
        title = (f'{len(lost_ais)} junction(s) lost their AIS counts in B: '
                 f'{effects or "no effect on collisions"}')
        return [Finding('warning', title, details, concerned)]
    return [Finding('info', f'Junction transition matrices differ: {effects or "no effect on collisions"}',
                    details, concerned)]


def _finding_angles(expl: CollisionExplanation) -> list[Finding]:
    """Pairs whose angle moved onto (or across) the merging / crossing limit.

    Only pairs present in both runs: a pair that exists in one run only
    is explained by the routing or the legs, not by its angle.
    """
    pairs: dict[str, tuple[float, float]] = {}
    for r in expl.items:
        if r.kind == 'pair' and r.angle_a is not None and r.angle_b is not None and r.a > 0 and r.b > 0:
            pairs.setdefault(r.label, (r.angle_a, r.angle_b))
    switched, near = [], []
    for label, (a, b) in sorted(pairs.items()):
        if (a <= MERGING_LIMIT_DEG) != (b <= MERGING_LIMIT_DEG):
            switched.append(f'{label}: {a:.1f} deg in A, {b:.1f} deg in B')
        elif abs(a - b) > 0.05 and abs(b - MERGING_LIMIT_DEG) <= _LIMIT_BAND_DEG:
            near.append(f'{label}: {b:.1f} deg in B ({a:.1f} deg in A)')
    why = ('Pairs meeting at 30 deg or less are merging, steeper ones crossing: different formulas and '
           'causation factors, so a small waypoint move there changes the result in a step.')
    out = []
    if switched:
        out.append(Finding('warning', f'{len(switched)} leg pair(s) switched between merging and crossing',
                           switched + [why], ('crossing', 'merging')))
    if near:
        out.append(Finding('warning', f'{len(near)} leg pair(s) moved onto the {MERGING_LIMIT_DEG:.0f} deg '
                           'merging / crossing limit', near + [why], ('crossing', 'merging')))
    return out


def _finding_simple(expl: CollisionExplanation, group: str, title: str) -> list[Finding]:
    lines = expl.differences.get(group) or []
    if group not in expl.groups:
        return []
    effects, concerned = _effects(expl, group)
    details = lines[:8] + ([f'... and {len(lines) - 8} more.'] if len(lines) > 8 else [])
    return [Finding('info', f'{title}: {effects or "no effect on collisions"}', details, concerned)]


def _finding_no_effect(expl: CollisionExplanation) -> list[Finding]:
    lines = expl.differences.get('_no_effect') or []
    if not lines:
        return []
    return [Finding('info', 'Inputs that differ but do not enter the ship-ship collision model', lines + [
        'Structures and depth areas act on collisions only through the route, traffic and distribution '
        'changes listed above.'])]


def _finding_residual(expl: CollisionExplanation) -> list[Finding]:
    off = [t for t in TYPES if RESIDUAL in expl.by_group.get(t, {})]
    if not off:
        return []
    return [Finding('warning', 'Part of the change could not be attributed to an input group', [
        ', '.join(f'{TYPE_LABELS[t]} {_e(expl.by_group[t][RESIDUAL])}' for t in off),
        'Run B differs from run A with every listed group switched to B.  This points at an input this '
        'explanation does not know about yet.'], tuple(off))]


def build_findings(
    snap_a: Mapping[str, Any], snap_b: Mapping[str, Any], expl: CollisionExplanation,
) -> list[Finding]:
    """Plain-language findings, warnings first."""
    found: list[Finding] = []
    found += _finding_consistency(expl)
    found += _finding_residual(expl)
    found += _finding_routing(snap_a, snap_b, expl)
    found += _finding_reclassification(expl)
    found += _finding_lateral(snap_a, snap_b, expl)
    found += _finding_geometry(snap_a, snap_b, expl)
    found += _finding_junctions(snap_a, snap_b, expl)
    found += _finding_angles(expl)
    found += _finding_simple(expl, 'traffic', 'Traffic volume differs')
    found += _finding_simple(expl, 'ships', 'Ship properties differ')
    found += _finding_simple(expl, 'causation', 'Collision causation factors differ')
    found += _finding_simple(expl, 'legs', 'Legs differ')
    found += _finding_simple(expl, 'other', 'Other leg inputs differ')
    found += _finding_no_effect(expl)
    if not expl.groups:
        found.append(Finding('info', 'No input of the ship-ship collision model differs between A and B'))
    found.sort(key=lambda f: 0 if f.level == 'warning' else 1)
    return found


def findings_for(expl: CollisionExplanation, acc_type: str) -> list[Finding]:
    """Findings that concern ``acc_type`` plus the general ones (``TOTAL``: all)."""
    return [f for f in expl.findings if acc_type == TOTAL or not f.types or acc_type in f.types]


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _pct(delta: float, base: float) -> str:
    return f'{100.0 * delta / base:+.1f} %' if base else '-'


def to_markdown(expl: CollisionExplanation, name_a: str = 'A', name_b: str = 'B', top: int = 10) -> str:
    """The whole explanation (every type) as a Markdown report."""
    lines = ['# Why did the collisions change?\n',
             f'- Run A: `{name_a}`', f'- Run B: `{name_b}`',
             f'- {expl.runs} collision-model runs, {expl.seconds:.0f} s'
             + ('' if expl.exact else ' (Shapley values averaged over fixed orders)'), '']
    ident = [GROUP_LABELS[g] for g in expl.identical_groups()]
    if ident:
        lines += [f'Identical in A and B: {", ".join(ident)}.', '']
    cols = expl.columns()
    lines += ['## Summary\n', '| Accident type | Run A | Run B | Delta | Delta % | '
              + ' | '.join(GROUP_LABELS[g] for g in cols) + ' |',
              '|---|---|---|---|---|' + '---|' * len(cols)]
    for t in TYPES + (TOTAL,):
        d = expl.delta(t)
        lines.append(f'| {TYPE_LABELS[t]} | {expl.totals_a[t]:.3e} | {expl.totals_b[t]:.3e} | {_e(d)} | '
                     f'{_pct(d, expl.totals_a[t])} | '
                     + ' | '.join(_e(v) for _, v in expl.contribution_rows(t)) + ' |')
    lines += ['', '## Findings\n']
    for f in expl.findings:
        mark = 'Warning: ' if f.level == 'warning' else ''
        lines.append(f'- **{mark}{f.title}**')
        lines += [f'  - {d}' for d in f.details]
    for t in TYPES:
        rows = items_for(expl, t)[:top]
        if not rows:
            continue
        lines += ['', f'## {TYPE_LABELS[t]}: largest items\n',
                  '| Item | Run A | Run B | Delta | ' + ' | '.join(cols) + ' |',
                  '|---|---|---|---|' + '---|' * len(cols)]
        for r in rows:
            lines.append(f'| {r.label} | {r.a:.3e} | {r.b:.3e} | {_e(r.delta)} | '
                         + ' | '.join(_e(r.by_group.get(g, 0.0)) for g in cols) + ' |')
    return '\n'.join(lines) + '\n'


def type_for_label(label: str) -> str:
    """Accident type of a Compare Totals row label (``"All collisions"`` -> ``TOTAL``)."""
    for key, text in TYPE_LABELS.items():
        if text == (label or '').strip():
            return key
    return TOTAL


def leg_endpoints(snap_a: Mapping[str, Any], snap_b: Mapping[str, Any],
                  legs: Iterable[tuple[str, str]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Start / end (lon, lat) of ``legs`` -- from run B when it has the leg, else run A."""
    out = []
    for seg, name in legs:
        for segs in (_segs(snap_b), _segs(snap_a)):
            sd = segs.get(str(seg))
            if sd is None or str(sd.get('Leg_name') or '') != name:
                continue
            p, q = parse_xy(sd.get('Start_Point')), parse_xy(sd.get('End_Point'))
            if p and q:
                out.append((p, q))
            break
    return out
