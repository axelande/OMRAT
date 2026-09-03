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
from -- provenance only, nothing reads it.

Direction mapping
-----------------
Traffic is stored per direction label (``'East going'`` ...).  Two legs
of the same original route share the drawing direction, so direction
index ``i`` of the source maps to index ``i`` of the target.  When the
target leg was drawn the *opposite* way pass ``swap_dirs=True``: the two
directions are exchanged and the lateral axis is mirrored (means, the
raw offset samples and the uniform bounds change sign) because "left of
the leg" flips with the drawing direction.
"""
from __future__ import annotations

import copy
from typing import Any, Iterable

import numpy as np

LOCK_KEY = 'traffic_locked'
SOURCE_KEY = 'traffic_source'

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
