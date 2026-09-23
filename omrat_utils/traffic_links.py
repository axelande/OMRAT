"""Which leg's traffic refers to which: labels and map links.

Pure Python (no Qt / QGIS) so the standalone test suite covers it.  The
map layer (``HandleQGISIface.show_traffic_links``) and every leg label
(Copy traffic / Suppress leg dialogs, Traffic tab selector) are built
from here, so they always tell the same story.

Three kinds of link, drawn from the leg whose data is used to the leg
that uses it:

``copy``      ``traffic_source`` -> the leg holding a copy of its traffic
              (:mod:`omrat_utils.copy_traffic`).
``redirect``  a suppressed leg -> each target of its ``traffic_redirect``
              (:mod:`compute.traffic_redirect`); the label gives the
              shares per direction.
``with``      a leg suppressed *with* a lead -> the lead.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians
from typing import Any

from compute.traffic_redirect import (
    direction_label, get_redirect, group_lead, group_members, is_suppressed,
)
from omrat_utils.copy_traffic import LOCK_KEY, SOURCE_KEY

KINDS = ('copy', 'redirect', 'with')


@dataclass(frozen=True)
class Link:
    kind: str
    src: str
    dst: str
    label: str


def leg_name(seg: str, segment_data: dict[str, Any]) -> str:
    seg_d = (segment_data or {}).get(str(seg))
    name = seg_d.get('Leg_name') if isinstance(seg_d, dict) else None
    return str(name) if name else f"LEG_{seg}"


def _short_dir(label: str) -> str:
    """``'North going'`` -> ``'N'``; anything else unchanged."""
    word = label.split()[0] if label else label
    return word[0].upper() if word in ('North', 'South', 'East', 'West') else label


def _copy_source(seg: str, segment_data: dict[str, Any]) -> str | None:
    seg_d = (segment_data or {}).get(str(seg))
    src = seg_d.get(SOURCE_KEY) if isinstance(seg_d, dict) else None
    return str(src) if src not in (None, '') else None


def status_suffix(seg: str, segment_data: dict[str, Any]) -> str:
    """``'  [locked, copy of LEG_2_3_a]'``-style suffix, ``''`` for a plain leg."""
    seg = str(seg)
    seg_d = (segment_data or {}).get(seg)
    if not isinstance(seg_d, dict):
        return ''
    parts: list[str] = []
    lead = group_lead(segment_data, seg)
    if lead is not None:
        parts.append(f"suppressed with {leg_name(lead, segment_data)}")
    elif is_suppressed(segment_data, seg):
        # In the order the user entered them (first target named).
        targets = list(dict.fromkeys(e['leg'] for e in get_redirect(segment_data, seg) if e['leg'] in segment_data))
        if targets:
            more = f" +{len(targets) - 1}" if len(targets) > 1 else ''
            parts.append(f"suppressed -> {leg_name(targets[0], segment_data)}{more}")
        else:
            parts.append("suppressed")
        n_with = len(group_members(segment_data, seg))
        if n_with:
            parts.append(f"+{n_with} leg(s) with it")
    if seg_d.get(LOCK_KEY) is True:
        parts.append("locked")
    src = _copy_source(seg, segment_data)
    if src is not None:
        parts.append(f"copy of {leg_name(src, segment_data)}")
    return f"  [{', '.join(parts)}]" if parts else ''


def leg_label(seg: str, segment_data: dict[str, Any]) -> str:
    """``'LEG_2_3_b  (id 16)  [locked, copy of LEG_2_3_a]'``."""
    return f"{leg_name(seg, segment_data)}  (id {seg}){status_suffix(seg, segment_data)}"


def build_links(segment_data: dict[str, Any]) -> list[Link]:
    """Every link between existing legs, in a stable order."""
    segs = segment_data or {}
    links: list[Link] = []
    for seg in segs:
        seg = str(seg)
        src = _copy_source(seg, segs)
        if src is not None and src in segs and src != seg:
            seg_d = segs[seg]
            links.append(Link('copy', src, seg, 'copy (locked)' if seg_d.get(LOCK_KEY) is True else 'copy'))
        lead = group_lead(segs, seg)
        if lead is not None:
            if lead in segs:
                links.append(Link('with', seg, lead, 'suppressed with'))
            continue
        if not is_suppressed(segs, seg):
            continue
        shares: dict[str, list[str]] = {}
        for e in get_redirect(segs, seg):
            if e['leg'] in segs and e['leg'] != seg:
                d = _short_dir(direction_label(seg, e['from_dir'], segs))
                shares.setdefault(e['leg'], []).append(f"{d} {e['share']:g} %")
        for dst, parts in shares.items():
            links.append(Link('redirect', seg, dst, ', '.join(parts)))
    return links


def related_legs(seg: str, segment_data: dict[str, Any]) -> set[str]:
    """Legs linked to ``seg`` in either direction (not ``seg`` itself)."""
    seg = str(seg)
    out: set[str] = set()
    for link in build_links(segment_data):
        if link.src == seg:
            out.add(link.dst)
        elif link.dst == seg:
            out.add(link.src)
    return out


# ---------------------------------------------------------------------------
# Geometry of a link arrow
# ---------------------------------------------------------------------------

def leg_midpoint(seg_d: dict[str, Any]) -> tuple[float, float] | None:
    from geometries.route_validation import parse_wkt_point
    if not isinstance(seg_d, dict):
        return None
    sp, ep = parse_wkt_point(seg_d.get('Start_Point')), parse_wkt_point(seg_d.get('End_Point'))
    if sp is None or ep is None:
        return None
    return ((sp[0] + ep[0]) / 2.0, (sp[1] + ep[1]) / 2.0)


def curve_points(
    p0: tuple[float, float], p1: tuple[float, float], bend: float = 0.2, n: int = 16,
) -> list[tuple[float, float]]:
    """Quadratic Bezier from ``p0`` to ``p1`` (lon, lat), bowed to the left
    of the travel direction by ``bend`` x the distance, so a pair of
    opposite links does not overlap.  ``n`` segments -> ``n + 1`` points."""
    kx = cos(radians((p0[1] + p1[1]) / 2.0)) or 1e-9   # metres-ish in x
    dx, dy = (p1[0] - p0[0]) * kx, p1[1] - p0[1]
    mx, my = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
    cx, cy = mx + (-dy * bend) / kx, my + dx * bend
    pts = []
    for i in range(n + 1):
        t = i / n
        a, b, c = (1 - t) ** 2, 2 * (1 - t) * t, t ** 2
        pts.append((a * p0[0] + b * cx + c * p1[0], a * p0[1] + b * cy + c * p1[1]))
    return pts


def link_geometries(segment_data: dict[str, Any]) -> list[tuple[Link, list[tuple[float, float]]]]:
    """``(link, points)`` for every link whose legs both have coordinates."""
    out = []
    for link in build_links(segment_data):
        a = leg_midpoint((segment_data or {}).get(link.src))
        b = leg_midpoint((segment_data or {}).get(link.dst))
        if a is None or b is None or a == b:
            continue
        out.append((link, curve_points(a, b)))
    return out
