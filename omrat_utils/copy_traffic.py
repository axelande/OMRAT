"""Copy traffic and lateral distributions from one leg to another, and the
per-leg *AIS lock* that keeps copied data safe from the next AIS refresh.

Pure Python (no Qt / QGIS) so the standalone test suite covers it.

Why
---
When routes split each other into sub-legs (three routes crossing give
a dozen of them) some sub-legs sit where the AIS sample is polluted by
the crossing traffic.  The analyst then wants to say "leg d carries the
same traffic as leg a", and wants that statement to *stick* when
"Update all distributions" is pressed again.

Data model
----------
``segment_data[seg]['traffic_locked']``  bool, default ``False``.  Locked
legs are skipped by :meth:`omrat_utils.handle_ais.AIS.update_legs` (both
the bulk and the per-leg button).
``segment_data[seg]['traffic_source']``  the leg id the data was copied
from.  The link describes the *data*, so it goes when the copy stops
being true: unticking the AIS lock releases the copy
(:func:`release_copy`, the next AIS update replaces it) and an AIS update
that writes the leg clears it (``AisUpdateTask._apply_leg_result``).
:func:`geometries.junctions.linked_partners` reads it: at a
junction shared by the source and the target the transition matrix
defaults to 100 % continuation between them (the two sub-legs are one
route again).

Direction mapping
-----------------
Traffic is stored per direction label (``'East going'`` ...).  Two legs
of the same original route share the drawing direction, so direction
index ``i`` of the source maps to index ``i`` of the target.  When the
target leg was drawn the *opposite* way pass ``swap_dirs=True``: the two
directions are exchanged and the lateral axis is mirrored (means, the
raw offset samples and the uniform bounds change sign) because "left of
the leg" flips with the drawing direction.

The direction *names* are not compared: they come from 90-degree sectors,
so legs at 44 and 46 degrees are "North going" and "East going" although
they run the same way.  :func:`copy_angle` compares the bearings instead,
and :func:`misaligned_targets` flags a target whose receiving direction
heads more than :data:`MAX_ALIGNED_DEG` away from the source's -- the
dialog warns and can swap those legs only (:func:`aligned_swaps`), which
is what a selection of legs drawn both ways needs.
"""
from __future__ import annotations

import copy
from typing import Any, Iterable

import numpy as np

from compute.traffic_redirect import direction_bearing

LOCK_KEY = 'traffic_locked'
SOURCE_KEY = 'traffic_source'

# A copy whose direction 1 lands on a target direction heading more than
# this away from the source's direction 1 sends the ships the wrong way.
MAX_ALIGNED_DEG = 90.0

# Per-direction distribution keys; ``{d}`` is the direction index 1 / 2.
_MIRROR_KEYS = ('mean{d}_1', 'mean{d}_2', 'mean{d}_3')
_PLAIN_KEYS = (
    'std{d}_1', 'std{d}_2', 'std{d}_3',
    'weight{d}_1', 'weight{d}_2', 'weight{d}_3',
    'u_p{d}', 'ai{d}',
)


def is_locked(segment_data: dict[str, Any], seg: str) -> bool:
    seg_d = segment_data.get(str(seg))
    return isinstance(seg_d, dict) and seg_d.get(LOCK_KEY) is True


def set_locked(segment_data: dict[str, Any], seg: str, locked: bool) -> bool:
    """Set the lock flag; returns ``False`` when the leg is unknown."""
    seg_d = segment_data.get(str(seg))
    if not isinstance(seg_d, dict):
        return False
    seg_d[LOCK_KEY] = bool(locked)
    return True


def release_copy(segment_data: dict[str, Any], seg: str) -> str | None:
    """Unlock ``seg`` and drop its copy link.

    Returns the leg the traffic had been copied from (``None`` when the
    leg was not a copy).  Called when the user unticks **AIS lock**: an
    unlocked copy is replaced by the leg's own AIS traffic on the next
    update, so keeping the link would leave the labels, the Traffic links
    arrows and the junction continuation rows claiming a copy that is
    about to be gone.
    """
    seg_d = segment_data.get(str(seg))
    if not isinstance(seg_d, dict):
        return None
    seg_d[LOCK_KEY] = False
    src = seg_d.pop(SOURCE_KEY, None)
    return str(src) if src not in (None, '') else None


def locked_legs(segment_data: dict[str, Any]) -> list[str]:
    return [k for k, v in segment_data.items() if isinstance(v, dict) and v.get(LOCK_KEY) is True]


def split_locked(legs: dict[str, Any], segment_data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """``(unlocked_legs, skipped_ids)`` for an AIS update request."""
    keep: dict[str, Any] = {}
    skipped: list[str] = []
    for key, val in legs.items():
        if is_locked(segment_data, key):
            skipped.append(key)
        else:
            keep[key] = val
    return keep, skipped


def _dirs_of(seg_key: str, traffic_data: dict[str, Any], segment_data: dict[str, Any]) -> list[str]:
    block = traffic_data.get(seg_key)
    if isinstance(block, dict) and block:
        return list(block.keys())
    seg_d = segment_data.get(seg_key)
    if isinstance(seg_d, dict):
        dirs = seg_d.get('Dirs')
        if isinstance(dirs, (list, tuple)) and dirs:
            return [str(d) for d in dirs]
    return []


def angle_between(a: float, b: float) -> float:
    """Smallest angle (0..180 deg) between two compass bearings."""
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def copy_angle(
    segment_data: dict[str, Any], src: str, dst: str, swap_dirs: bool = False,
) -> float | None:
    """Angle (deg, 0..180) between the source's direction 1 and the target
    direction that receives its traffic: direction 1 of ``dst``, or
    direction 2 with ``swap_dirs``.  ``None`` when either leg lacks a
    usable Start_Point / End_Point."""
    b_src = direction_bearing(segment_data.get(str(src)), 0)
    b_dst = direction_bearing(segment_data.get(str(dst)), 1 if swap_dirs else 0)
    if b_src is None or b_dst is None:
        return None
    return angle_between(b_src, b_dst)


def misaligned_targets(
    segment_data: dict[str, Any], src: str, targets: Iterable[str], swap_dirs: bool = False,
) -> list[tuple[str, float]]:
    """``[(target, angle)]`` for the targets where the copy would put the
    source's traffic on a direction heading more than
    :data:`MAX_ALIGNED_DEG` away.  Legs without geometry are not flagged."""
    out: list[tuple[str, float]] = []
    for dst in targets:
        if str(dst) == str(src):
            continue
        ang = copy_angle(segment_data, src, dst, swap_dirs)
        if ang is not None and ang > MAX_ALIGNED_DEG:
            out.append((str(dst), ang))
    return out


def aligned_swaps(
    segment_data: dict[str, Any], src: str, targets: Iterable[str], swap_dirs: bool = False,
) -> dict[str, bool]:
    """Per-target ``swap_dirs``: ``swap_dirs`` itself, flipped for every
    target :func:`misaligned_targets` flags."""
    flagged = {t for t, _ in misaligned_targets(segment_data, src, targets, swap_dirs)}
    return {str(t): (not swap_dirs) if str(t) in flagged else bool(swap_dirs) for t in targets}


def describe_misaligned(
    segment_data: dict[str, Any], src: str, flagged: list[tuple[str, float]], n_targets: int,
) -> str:
    """Body of the dialog's direction warning."""
    lines = [
        f"Direction 1 of {describe_targets([src], segment_data)} would be copied onto a "
        f"direction heading more than {MAX_ALIGNED_DEG:.0f} deg away on:",
        "",
    ]
    lines += [f"    {describe_targets([t], segment_data)}: {ang:.0f} deg apart" for t, ang in flagged]
    lines += [
        "",
        "The ships would sail the wrong way along those legs and their lateral "
        "distribution would be mirrored (Cat I / Cat II, drifting and collisions "
        "all depend on it).",
    ]
    if len(flagged) < n_targets:
        lines += ["", "The other selected legs are fine as chosen."]
    return "\n".join(lines)


def _copy_samples(value: Any, mirror: bool) -> Any:
    """Deep-copy a ``dist1`` / ``dist2`` sample array, negating if mirrored."""
    if value is None:
        return np.array([])
    arr = np.array(value, dtype=float).copy() if not isinstance(value, np.ndarray) else value.copy()
    return -arr if mirror else arr


def _neg(value: Any) -> Any:
    try:
        return -float(value)
    except (TypeError, ValueError):
        return value


def copy_leg_traffic(
    traffic_data: dict[str, Any],
    segment_data: dict[str, Any],
    src: str,
    dst: str,
    *,
    swap_dirs: bool = False,
    copy_distributions: bool = True,
    lock: bool = True,
) -> list[str]:
    """Copy ``src``'s traffic block (and optionally its lateral
    distributions) onto ``dst`` in place.  Returns the target's direction
    labels in the order they were filled.

    Raises ``KeyError`` when ``src`` has no traffic block and
    ``ValueError`` when ``src == dst``.
    """
    src, dst = str(src), str(dst)
    if src == dst:
        raise ValueError("source and target leg are the same")
    src_block = traffic_data.get(src)
    if not isinstance(src_block, dict) or not src_block:
        raise KeyError(f"leg {src} has no traffic data to copy")

    src_dirs = list(src_block.keys())
    dst_dirs = _dirs_of(dst, traffic_data, segment_data) or list(src_dirs)
    ordered_src = list(reversed(src_dirs)) if swap_dirs else src_dirs

    new_block: dict[str, Any] = {}
    for dst_dir, src_dir in zip(dst_dirs, ordered_src):
        new_block[dst_dir] = copy.deepcopy(src_block[src_dir])
    traffic_data[dst] = new_block

    dst_seg = segment_data.get(dst)
    if not isinstance(dst_seg, dict):
        dst_seg = {}
        segment_data[dst] = dst_seg

    if copy_distributions:
        src_seg = segment_data.get(src) or {}
        _copy_distributions(src_seg, dst_seg, swap_dirs)

    dst_seg[SOURCE_KEY] = src
    if lock:
        dst_seg[LOCK_KEY] = True
    return list(new_block.keys())


def _copy_distributions(src_seg: dict[str, Any], dst_seg: dict[str, Any], swap: bool) -> None:
    for d in (1, 2):
        s = (3 - d) if swap else d
        for tmpl in _PLAIN_KEYS:
            key_s, key_d = tmpl.format(d=s), tmpl.format(d=d)
            if key_s in src_seg:
                dst_seg[key_d] = copy.deepcopy(src_seg[key_s])
        for tmpl in _MIRROR_KEYS:
            key_s, key_d = tmpl.format(d=s), tmpl.format(d=d)
            if key_s in src_seg:
                dst_seg[key_d] = _neg(src_seg[key_s]) if swap else copy.deepcopy(src_seg[key_s])
        # Uniform bounds: mirroring negates and exchanges min / max.
        u_min_s, u_max_s = f'u_min{s}', f'u_max{s}'
        if u_min_s in src_seg or u_max_s in src_seg:
            lo = src_seg.get(u_min_s, 0)
            hi = src_seg.get(u_max_s, 0)
            if swap:
                lo, hi = _neg(hi), _neg(lo)
            dst_seg[f'u_min{d}'] = copy.deepcopy(lo)
            dst_seg[f'u_max{d}'] = copy.deepcopy(hi)
        dist_s = f'dist{s}'
        if dist_s in src_seg:
            dst_seg[f'dist{d}'] = _copy_samples(src_seg[dist_s], swap)


def describe_targets(ids: Iterable[str], segment_data: dict[str, Any]) -> str:
    """``"LEG_5_12_d (12), LEG_5_12_c (11)"`` for messages."""
    parts = []
    for k in ids:
        seg_d = segment_data.get(str(k)) or {}
        name = seg_d.get('Leg_name') if isinstance(seg_d, dict) else None
        parts.append(f"{name} ({k})" if name else str(k))
    return ", ".join(parts)
