"""Leg names for the result outputs (QGIS-free).

Compute keys legs by their global ``Segment_Id`` (``"9"``), a leg and
direction by ``"9:0"`` (direction *index*: 0 = drawn direction) and the
drifting model by ``"9:North going:270"`` (direction *name* and drift
compass angle).  Those keys are stable and stay in the reports and
sidecars.  Everything a person reads -- result-layer columns, the
Markdown report, the View breakdown -- goes through this module so the
legs carry the names of the route table (``LEG_1_2_b East going``)
instead of ``leg_9_0``.

Every function falls back to the id form when ``segment_data`` lacks the
leg, so a report written for legs that have since been deleted still
renders.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

__all__ = [
    'leg_name', 'direction_name', 'leg_label', 'leg_field_name',
    'leg_display', 'parse_leg_key', 'leg_field_map', 'pair_label',
]

_FIELD_SAFE = re.compile(r'[^A-Za-z0-9_]+')


def parse_leg_key(key: Any) -> tuple[str, str | None, str | None]:
    """``(seg_id, direction, angle)`` from any of the three key forms.

    ``direction`` is the index text (``"0"`` / ``"1"``) or the direction
    name, whichever the key carried; ``angle`` is the drift compass angle
    text or ``None``.
    """
    parts = str(key).split(':')
    seg = parts[0].strip()
    if len(parts) == 1:
        return seg, None, None
    if len(parts) == 2:
        return seg, parts[1].strip(), None
    return seg, parts[1].strip(), parts[-1].strip()


def leg_name(segment_data: Mapping[str, Any] | None, seg_id: Any) -> str | None:
    """The route table's ``Leg_name`` for ``seg_id``, or ``None``."""
    if not segment_data:
        return None
    seg = segment_data.get(str(seg_id))
    if seg is None:
        seg = segment_data.get(seg_id)
    if not isinstance(seg, Mapping):
        return None
    name = seg.get('Leg_name')
    if name is None or not str(name).strip():
        return None
    return str(name).strip()


def direction_name(segment_data: Mapping[str, Any] | None, seg_id: Any, direction: Any) -> str | None:
    """Direction text for a leg: an index (``0`` / ``"1"``) is looked up in
    ``Dirs``; any other text is taken as the name itself."""
    if direction is None:
        return None
    text = str(direction).strip()
    if text in ('0', '1'):
        seg = (segment_data or {}).get(str(seg_id)) if segment_data else None
        dirs = seg.get('Dirs') if isinstance(seg, Mapping) else None
        if isinstance(dirs, (list, tuple)) and len(dirs) > int(text):
            return str(dirs[int(text)])
        return f'dir {text}'
    return text


def leg_label(segment_data: Mapping[str, Any] | None, key: Any) -> str:
    """Human label: ``"LEG_1_2_b East going"``, ``"LEG_1_2_b"`` or
    ``"LEG_1_2_b North going 270 deg"``.  Falls back to the key itself."""
    seg, direction, angle = parse_leg_key(key)
    name = leg_name(segment_data, seg)
    if name is None:
        return str(key)
    parts = [name]
    dname = direction_name(segment_data, seg, direction)
    if dname:
        parts.append(dname)
    if angle:
        parts.append(f'{angle} deg')
    return ' '.join(parts)


def leg_display(segment_data: Mapping[str, Any] | None, key: Any) -> str:
    """Label plus the id in brackets for reports: ``"LEG_1_2_b East going (9:0)"``."""
    label = leg_label(segment_data, key)
    return label if label == str(key) else f'{label} ({key})'


def leg_field_name(segment_data: Mapping[str, Any] | None, key: Any) -> str:
    """Attribute-table column for a leg key.

    ``"LEG_1_2_b_East_going"`` when the leg is known, else the historical
    ``"leg_9_0"``.  Only letters, digits and underscores survive, so the
    name is safe for GeoPackage, expressions and shapefile-ish tools.
    """
    label = leg_label(segment_data, key)
    if label == str(key):
        safe = _FIELD_SAFE.sub('_', str(key)).strip('_')
        return f'leg_{safe}'
    safe = _FIELD_SAFE.sub('_', label).strip('_')
    return safe or f'leg_{_FIELD_SAFE.sub("_", str(key))}'


def leg_field_map(segment_data: Mapping[str, Any] | None, keys: Any) -> dict[str, str]:
    """``{key: unique field name}`` for a set of leg keys.

    Two legs that share a name (it happens after manual renames) get the
    id appended so no column is lost.
    """
    out: dict[str, str] = {}
    used: dict[str, str] = {}

    def _order(k: Any) -> tuple:
        seg, direction, angle = parse_leg_key(k)
        return (0, int(seg)) if seg.isdigit() else (1, seg), str(direction or ''), str(angle or '')

    for key in sorted(keys, key=_order):
        field = leg_field_name(segment_data, key)
        if field in used and used[field] != str(key):
            seg, direction, _angle = parse_leg_key(key)
            suffix = f'_id{seg}' + (f'_{direction}' if direction is not None else '')
            field = f'{field}{_FIELD_SAFE.sub("_", suffix)}'
        used[field] = str(key)
        out[str(key)] = field
    return out


_PAIR_TEXT = re.compile(r"^\(\s*'([^']*)'\s*,\s*'([^']*)'\s*\)$")


def pair_label(segment_data: Mapping[str, Any] | None, label: Any) -> str:
    """Label for a leg pair: a ``(leg1, leg2)`` tuple or its ``str()`` form
    (as it comes back from a JSON sidecar) -> ``"LEG_1_2_b (9) x LEG_3_7_b (8)"``."""
    if isinstance(label, (tuple, list)) and len(label) == 2:
        a, b = str(label[0]), str(label[1])
    else:
        m = _PAIR_TEXT.match(str(label).strip())
        if not m:
            return str(label)
        a, b = m.group(1), m.group(2)
    return f'{leg_display(segment_data, a)} x {leg_display(segment_data, b)}'
