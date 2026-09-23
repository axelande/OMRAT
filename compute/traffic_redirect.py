"""Suppress a leg and move its traffic onto other legs (scenario analysis).

Pure Python (no Qt / QGIS) so the standalone test suite covers it.

Why
---
A planned wind farm (or any new obstacle) often sits on a lane that the
AIS data shows is in use today.  Once it is built, those ships take a
detour.  Deleting the leg would drop its ships from the model and
copying its traffic would duplicate them; *suppressing* the leg keeps it
in the project (the baseline can be restored) while the compute moves
its traffic onto the detour legs, so the number of ships is conserved.

Data model
----------
``segment_data[seg]['suppressed']``        bool, default ``False``.
``segment_data[seg]['traffic_redirect']``  list of
``{'from_dir': 0|1, 'leg': '<id>', 'dir': 0|1, 'share': <percent>}``.

``from_dir`` / ``dir`` are direction indices (``Dirs[0]`` is the drawn
direction, see CLAUDE.md "Leg direction convention").  ``share`` is the
percentage of the suppressed leg's ``from_dir`` traffic that sails
``leg``.  The legs of one detour in series each get the full share
(e.g. 100 % on every leg); alternative routes split it (80 % / 20 %).
Shares are *not* required to sum to 100.

The redirect is kept on the leg when the leg is restored, so ticking
"suppressed" again brings the scenario back.  It is only applied while
the leg is suppressed.

``segment_data[seg]['suppressed_with']``  id of the *lead* leg.  A whole
route is suppressed by giving one leg (the lead, usually the one with the
cleanest AIS sample) the redirect and suppressing the other legs of the
route *with* it: they carry the same ships, so they are left out without
moving anything again (a redirect on every leg would move the ships once
per leg).  A member's own ``traffic_redirect`` is ignored.  Restoring the
lead restores its members.

Compute
-------
:func:`apply_traffic_redirects` runs on the private data copy right
after :func:`compute.data_preparation.apply_traffic_scaling` (the moved
traffic therefore carries the source leg's scaling).  It adds the moved
frequencies to the target cells, merges speed / draught / height / beam
as a frequency-weighted mean, keeps the target's own lateral
distribution, and then removes every suppressed leg from
``segment_data``, ``traffic_data`` and the junction matrices.  The live
project is never touched.
"""
from __future__ import annotations

import copy
from math import atan2, cos, degrees, isfinite, radians
from typing import Any

SUPPRESS_KEY = 'suppressed'
REDIRECT_KEY = 'traffic_redirect'
GROUP_KEY = 'suppressed_with'

FREQ_KEY = 'Frequency (ships/year)'
SCALING_KEY = 'Scaling (%)'
# Per-cell attributes merged as a frequency-weighted mean.
ATTR_KEYS = ('Speed (knots)', 'Draught (meters)', 'Ship heights (meters)', 'Ship Beam (meters)')


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

def is_suppressed(segment_data: dict[str, Any], seg: str) -> bool:
    seg_d = (segment_data or {}).get(str(seg))
    return isinstance(seg_d, dict) and seg_d.get(SUPPRESS_KEY) is True


def suppressed_legs(segment_data: dict[str, Any]) -> list[str]:
    return [str(k) for k, v in (segment_data or {}).items() if isinstance(v, dict) and v.get(SUPPRESS_KEY) is True]


def group_lead(segment_data: dict[str, Any], seg: str) -> str | None:
    """The lead leg ``seg`` is suppressed with, or ``None``."""
    seg_d = (segment_data or {}).get(str(seg))
    if not isinstance(seg_d, dict) or seg_d.get(SUPPRESS_KEY) is not True:
        return None
    lead = seg_d.get(GROUP_KEY)
    return str(lead) if lead not in (None, '') else None


def group_members(segment_data: dict[str, Any], lead: str) -> list[str]:
    """Legs suppressed together with ``lead``."""
    return [str(k) for k in (segment_data or {}) if group_lead(segment_data, str(k)) == str(lead)]


def set_group(segment_data: dict[str, Any], lead: str, members: list[str]) -> list[str]:
    """Suppress ``members`` with ``lead`` (and un-suppress former members
    that are no longer listed).  Returns every leg whose flag changed."""
    lead = str(lead)
    wanted = [str(m) for m in members if str(m) != lead and str(m) in (segment_data or {})]
    changed: list[str] = []
    for old in group_members(segment_data, lead):
        if old not in wanted:
            seg_d = segment_data[old]
            seg_d[SUPPRESS_KEY] = False
            seg_d.pop(GROUP_KEY, None)
            changed.append(old)
    for m in wanted:
        seg_d = segment_data[m]
        seg_d[SUPPRESS_KEY] = True
        seg_d[GROUP_KEY] = lead
        changed.append(m)
    lead_d = (segment_data or {}).get(lead)
    if isinstance(lead_d, dict):
        lead_d.pop(GROUP_KEY, None)   # a lead is never a member itself
    return changed


def restore_group(segment_data: dict[str, Any], lead: str) -> list[str]:
    """Un-suppress ``lead`` and its members; returns the legs changed.
    Redirects are kept for a later re-suppress."""
    lead = str(lead)
    changed = []
    for seg in [lead] + group_members(segment_data, lead):
        seg_d = (segment_data or {}).get(seg)
        if isinstance(seg_d, dict):
            seg_d[SUPPRESS_KEY] = False
            seg_d.pop(GROUP_KEY, None)
            changed.append(seg)
    return changed


def get_redirect(segment_data: dict[str, Any], seg: str) -> list[dict[str, Any]]:
    seg_d = (segment_data or {}).get(str(seg))
    if not isinstance(seg_d, dict):
        return []
    return normalise_redirect(seg_d.get(REDIRECT_KEY))


def set_suppressed(
    segment_data: dict[str, Any],
    seg: str,
    suppressed: bool,
    redirect: list[dict[str, Any]] | None = None,
) -> bool:
    """Set the flag (and the redirect when given); ``False`` for an unknown leg."""
    seg_d = (segment_data or {}).get(str(seg))
    if not isinstance(seg_d, dict):
        return False
    seg_d[SUPPRESS_KEY] = bool(suppressed)
    if redirect is not None:
        seg_d[REDIRECT_KEY] = normalise_redirect(redirect)
    return True


def normalise_redirect(entries: Any) -> list[dict[str, Any]]:
    """Coerce a stored redirect list; drops malformed rows."""
    out: list[dict[str, Any]] = []
    if not isinstance(entries, (list, tuple)):
        return out
    for e in entries:
        if not isinstance(e, dict):
            continue
        try:
            from_dir = int(e.get('from_dir', 0))
            to_dir = int(e.get('dir', 0))
            share = float(e.get('share', 0.0))
        except (TypeError, ValueError):
            continue
        leg = e.get('leg')
        if leg is None or from_dir not in (0, 1) or to_dir not in (0, 1) or not isfinite(share):
            continue
        out.append({'from_dir': from_dir, 'leg': str(leg), 'dir': to_dir, 'share': share})
    return out


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------

def _point(text: Any) -> tuple[float, float] | None:
    from geometries.route_validation import parse_wkt_point
    return parse_wkt_point(text)


def direction_bearing(seg_d: dict[str, Any], dir_idx: int) -> float | None:
    """Compass bearing (deg) of direction ``dir_idx`` (0 = drawn direction)."""
    if not isinstance(seg_d, dict):
        return None
    sp, ep = _point(seg_d.get('Start_Point')), _point(seg_d.get('End_Point'))
    if sp is None or ep is None:
        return None
    dx = (ep[0] - sp[0]) * cos(radians((sp[1] + ep[1]) / 2.0))
    dy = ep[1] - sp[1]
    if dx == 0 and dy == 0:
        return None
    b = degrees(atan2(dx, dy)) % 360.0
    return b if dir_idx == 0 else (b + 180.0) % 360.0


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def auto_target_dir(segment_data: dict[str, Any], src: str, from_dir: int, dst: str) -> int:
    """Direction of ``dst`` whose bearing is closest to ``src``'s ``from_dir``."""
    b_src = direction_bearing((segment_data or {}).get(str(src)), from_dir)
    dst_d = (segment_data or {}).get(str(dst))
    b0, b1 = direction_bearing(dst_d, 0), direction_bearing(dst_d, 1)
    if b_src is None or b0 is None or b1 is None:
        return from_dir
    return 0 if _angle_diff(b_src, b0) <= _angle_diff(b_src, b1) else 1


def direction_label(
    seg: str, dir_idx: int, segment_data: dict[str, Any], traffic_data: dict[str, Any] | None = None,
) -> str:
    """``'North going'`` style label of a direction index."""
    seg_d = (segment_data or {}).get(str(seg)) or {}
    dirs = seg_d.get('Dirs') if isinstance(seg_d, dict) else None
    if isinstance(dirs, (list, tuple)) and len(dirs) > dir_idx:
        return str(dirs[dir_idx])
    block = (traffic_data or {}).get(str(seg))
    if isinstance(block, dict) and len(block) > dir_idx:
        return list(block.keys())[dir_idx]
    return f"Direction {dir_idx + 1}"


def dir_key(block: dict[str, Any], seg_d: dict[str, Any], dir_idx: int) -> str | None:
    """Key of direction ``dir_idx`` in a traffic block (``Dirs`` order wins)."""
    dirs = seg_d.get('Dirs') if isinstance(seg_d, dict) else None
    if isinstance(dirs, (list, tuple)) and len(dirs) > dir_idx and str(dirs[dir_idx]) in block:
        return str(dirs[dir_idx])
    keys = list(block.keys())
    return keys[dir_idx] if len(keys) > dir_idx else None


# ---------------------------------------------------------------------------
# Cell merging
# ---------------------------------------------------------------------------

def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if isfinite(v) else None


def total_frequency(var: dict[str, Any] | None) -> float:
    """Sum of the finite, positive frequency cells of one direction."""
    total = 0.0
    for row in (var or {}).get(FREQ_KEY) or []:
        if not hasattr(row, '__iter__'):
            continue
        for q in row:
            v = _num(q)
            if v is not None and v > 0:
                total += v
    return total


def _empty_like(var: dict[str, Any]) -> dict[str, Any]:
    """A direction block shaped like ``var`` with zero frequency."""
    out: dict[str, Any] = {}
    for key, mat in var.items():
        if key == SCALING_KEY or (not isinstance(mat, (list, tuple)) and not hasattr(mat, 'tolist')):
            continue
        rows = mat.tolist() if hasattr(mat, 'tolist') else mat
        out[key] = [[0.0 for _ in row] if hasattr(row, '__iter__') else 0.0 for row in rows]
    return out


def merge_direction(target: dict[str, Any], source: dict[str, Any], share_pct: float) -> float:
    """Add ``share_pct`` % of ``source``'s ships into ``target`` in place.

    Frequencies add; the per-cell attributes become the frequency-weighted
    mean (a cell empty on one side takes the other side's value).  Returns
    the number of ships per year moved.
    """
    factor = float(share_pct) / 100.0
    if factor <= 0:
        return 0.0
    f_src = source.get(FREQ_KEY) or []
    if FREQ_KEY not in target:
        target.update(_empty_like(source))
    f_dst = target[FREQ_KEY]
    for key in ATTR_KEYS:
        if key in source and key not in target:
            target[key] = copy.deepcopy(source[key])
    moved = 0.0
    for i, row in enumerate(f_src):
        if not hasattr(row, '__iter__') or i >= len(f_dst) or not hasattr(f_dst[i], '__iter__'):
            continue
        for j, q in enumerate(row):
            q_m = _num(q)
            if q_m is None or q_m <= 0 or j >= len(f_dst[i]):
                continue
            q_m *= factor
            q_t = _num(f_dst[i][j])
            q_t = q_t if q_t is not None and q_t > 0 else 0.0
            for key in ATTR_KEYS:
                if key not in source or key not in target:
                    continue
                try:
                    a_s = _num(source[key][i][j])
                    a_t = _num(target[key][i][j])
                except (IndexError, TypeError):
                    continue
                if a_s is None:
                    continue
                if a_t is None or q_t <= 0:
                    target[key][i][j] = a_s
                else:
                    target[key][i][j] = (a_t * q_t + a_s * q_m) / (q_t + q_m)
            f_dst[i][j] = q_t + q_m
            moved += q_m
    return moved


# ---------------------------------------------------------------------------
# Junctions
# ---------------------------------------------------------------------------

def drop_legs_from_junctions(
    junctions_payload: dict[str, Any] | None,
    removed: set[str],
    segment_data: dict[str, Any],
) -> dict[str, Any] | None:
    """Serialised junctions without the ``removed`` legs.

    Rows and columns of a removed leg are dropped and the remaining rows
    renormalised.  A row whose whole share went to a removed leg falls
    back to the geometric default of the reduced junction; a junction
    left with fewer than two legs disappears.  ``segment_data`` must be
    the reduced one.
    """
    if not isinstance(junctions_payload, dict) or not removed:
        return junctions_payload
    from geometries.junctions import (
        compute_geometric_transition_matrix, deserialize_junctions, serialize_junctions,
    )
    registry = deserialize_junctions(junctions_payload)
    out = {}
    for jid, j in registry.items():
        if not removed.intersection(j.legs):
            out[jid] = j
            continue
        j.legs = {k: v for k, v in j.legs.items() if k not in removed}
        if len(j.legs) < 2:
            continue
        geo = compute_geometric_transition_matrix(j, segment_data)
        trans: dict[str, dict[str, float]] = {}
        for in_leg in j.legs:
            row = {o: s for o, s in (j.transitions.get(in_leg) or {}).items()
                   if o not in removed and o in j.legs}
            total = sum(v for v in row.values() if v > 0)
            if total > 1e-12:
                trans[in_leg] = {o: max(s, 0.0) / total for o, s in row.items()}
            elif in_leg in geo:
                trans[in_leg] = dict(geo[in_leg])
        j.transitions = trans
        out[jid] = j
    return serialize_junctions(out)


# ---------------------------------------------------------------------------
# Compute entry point
# ---------------------------------------------------------------------------

def apply_traffic_redirects(data: dict[str, Any]) -> dict[str, list]:
    """Move suppressed legs' traffic and drop them from ``data`` in place.

    ``data`` must be a private copy (as in ``CalculationTask.run``).
    Returns ``{'moves': [...], 'removed': [...], 'warnings': [...]}`` for
    logging; each move is ``{src, from_dir, dst, dir, share, ships}``.
    """
    summary: dict[str, list] = {'moves': [], 'removed': [], 'warnings': []}
    segs = data.get('segment_data') or {}
    traffic = data.get('traffic_data')
    if not isinstance(segs, dict):
        return summary
    if not isinstance(traffic, dict):
        traffic = {}
    removed = set(suppressed_legs(segs))
    if not removed:
        return summary

    for src in sorted(removed):
        src_block = traffic.get(src)
        if group_lead(segs, src) is not None:
            # Same ships as its lead leg, which moves them.
            continue
        entries = get_redirect(segs, src)
        if not isinstance(src_block, dict) or not src_block:
            if entries:
                summary['warnings'].append(f"leg {src} is suppressed but has no traffic to move")
            continue
        for e in entries:
            dst = e['leg']
            if dst == src or dst not in segs:
                summary['warnings'].append(f"leg {src}: target leg {dst} does not exist, skipped")
                continue
            if dst in removed:
                summary['warnings'].append(f"leg {src}: target leg {dst} is itself suppressed, skipped")
                continue
            src_key = dir_key(src_block, segs.get(src), e['from_dir'])
            if src_key is None:
                continue
            dst_block = traffic.get(dst)
            if not isinstance(dst_block, dict) or not dst_block:
                # Target without traffic of its own: give it both
                # directions (every leg has exactly two), empty.
                dst_block = {direction_label(dst, d, segs): _empty_like(src_block[src_key]) for d in (0, 1)}
                traffic[dst] = dst_block
            dst_key = dir_key(dst_block, segs.get(dst), e['dir'])
            if dst_key is None:
                dst_key = direction_label(dst, e['dir'], segs)
                dst_block[dst_key] = _empty_like(src_block[src_key])
            ships = merge_direction(dst_block[dst_key], src_block[src_key], e['share'])
            summary['moves'].append({
                'src': src, 'from_dir': e['from_dir'], 'dst': dst,
                'dir': e['dir'], 'share': e['share'], 'ships': ships,
            })

    for seg in removed:
        segs.pop(seg, None)
        traffic.pop(seg, None)
    summary['removed'] = sorted(removed)
    if 'junctions' in data:
        data['junctions'] = drop_legs_from_junctions(data.get('junctions'), removed, segs)
    return summary


# ---------------------------------------------------------------------------
# Keeping redirects valid when legs are split
# ---------------------------------------------------------------------------

def after_split(segment_data: dict[str, Any], parent: str, sub_ids: list[str]) -> None:
    """Fix redirects after ``parent`` was split into ``sub_ids`` (in place).

    Every sub-leg inherits the parent's full traffic.  If the parent was
    suppressed, only the first sub-leg keeps the redirect -- the others
    are suppressed *with* it (``suppressed_with``), otherwise the same
    ships would be moved once per sub-leg.  A redirect that targeted the parent now
    targets every sub-leg with the same share (a detour in series).
    """
    parent, subs = str(parent), [str(s) for s in sub_ids]
    if len(subs) < 2:
        return
    parent_d = segment_data.get(parent)
    lead_split = (isinstance(parent_d, dict) and parent_d.get(SUPPRESS_KEY) is True
                  and group_lead(segment_data, parent) is None)
    for sid in subs[1:]:
        seg_d = segment_data.get(sid)
        if not isinstance(seg_d, dict):
            continue
        if REDIRECT_KEY in seg_d:
            seg_d[REDIRECT_KEY] = []
        if lead_split:
            # The other sub-legs carry the same ships as the first one.
            seg_d[GROUP_KEY] = parent
    for seg_id, seg_d in segment_data.items():
        if not isinstance(seg_d, dict) or not seg_d.get(REDIRECT_KEY) or str(seg_id) in subs:
            continue
        entries = normalise_redirect(seg_d[REDIRECT_KEY])
        if not any(e['leg'] == parent for e in entries):
            continue
        new: list[dict[str, Any]] = []
        for e in entries:
            if e['leg'] != parent:
                new.append(e)
                continue
            new.extend(dict(e, leg=s) for s in subs)
        seg_d[REDIRECT_KEY] = new


# ---------------------------------------------------------------------------
# Export (IWRAP has no suppressed legs)
# ---------------------------------------------------------------------------

def prepare_export_data(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list]]:
    """``(copy, summary)``: a deep copy of ``data`` with the redirects
    applied and the suppressed legs removed, exactly as the calculation
    sees it.  ``data`` itself is left untouched."""
    out = copy.deepcopy(data)
    summary = apply_traffic_redirects(out)
    return out, summary


def _leg_name(seg: str, segment_data: dict[str, Any]) -> str:
    seg_d = (segment_data or {}).get(str(seg)) or {}
    name = seg_d.get('Leg_name') if isinstance(seg_d, dict) else None
    return f"{name} ({seg})" if name else str(seg)


def describe_redirect_summary(
    summary: dict[str, list], segment_data: dict[str, Any], max_lines: int = 15,
) -> str:
    """Human-readable lines for a popup.  ``segment_data`` is the project's
    (still holding the suppressed legs) so their names can be shown."""
    lines = [f"Suppressed: {', '.join(_leg_name(s, segment_data) for s in summary.get('removed', []))}"]
    moves = summary.get('moves', [])
    for mv in moves[:max_lines]:
        lines.append(
            f"  {_leg_name(mv['src'], segment_data)} {direction_label(mv['src'], mv['from_dir'], segment_data)}"
            f" -> {_leg_name(mv['dst'], segment_data)} {direction_label(mv['dst'], mv['dir'], segment_data)}:"
            f" {mv['share']:g} % ({mv['ships']:,.0f} ships/year)"
        )
    if len(moves) > max_lines:
        lines.append(f"  ... and {len(moves) - max_lines} more")
    if not moves:
        lines.append("  No traffic is moved (the suppressed legs' ships are left out).")
    for msg in summary.get('warnings', []):
        lines.append(f"  Warning: {msg}")
    return "\n".join(lines)
