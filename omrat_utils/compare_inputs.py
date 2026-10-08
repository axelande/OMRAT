"""Input differences between two ``.omrat`` snapshots (QGIS-free).

The settings table of the Compare tab (:func:`omrat_utils.compare.build_settings_table`)
covers the model *parameters*.  This module covers the model *inputs*
that decide where and how often accidents happen, which is where two
layouts of the same area usually differ:

* **Legs** -- legs present in one run only, moved start / end points,
  renamed directions and the Cat II check interval ``ai``.
* **Traffic** -- ships per year per leg and direction, and which other
  traffic fields (speed, draught, scaling ...) differ.
* **Lateral distribution** -- mean / std / weight of the three normal
  components and the uniform part.
* **Routing** -- suppressed legs, suppression groups and traffic
  redirects.
* **Junctions** -- junctions present in one run only, the legs that
  meet there, the matrix source and the largest change in a
  transition share.
* **Depths** / **Structures** -- feature counts and, per depth class or
  structure, polygon count, area and height.

Legs are matched on id *and* name (:func:`omrat_utils.compare_common.match_legs`),
so comparing two different layouts lists their legs as one-sided
instead of diffing unrelated legs that happen to share an id.

:func:`build_input_table` returns rows ``[group, item, run_a, run_b]``
as text, ready for a ``QTableWidget``.
"""
from __future__ import annotations

from typing import Any, Mapping

from .compare_common import (
    finite, fmt_xy, haversine_m, leg_identity, leg_title, local_projector,
    match_legs, parse_xy, direction_frequency,
)

__all__ = ['build_input_table', 'describe', 'INPUT_GROUPS']

INPUT_GROUPS: tuple[str, ...] = (
    'Project', 'Legs', 'Traffic', 'Lateral distribution', 'Routing',
    'Junctions', 'Depths', 'Structures',
)

_MAX_LISTED = 10          # names listed in a one-sided summary row
_SHARE_TOL = 0.005        # transition-share change worth reporting
_POINT_TOL_DEG = 1e-6     # endpoints are written with six decimals


def _close(a: Any, b: Any, rel: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    fa, fb = finite(a), finite(b)
    if fa is None or fb is None:
        return a == b
    return abs(fa - fb) <= max(abs_tol, rel * max(abs(fa), abs(fb)))


def _fmt(v: Any) -> str:
    if v is None or v == '':
        return '—'
    f = finite(v)
    if f is None:
        return str(v)
    return f'{f:.6g}'


def _names(segment_data: Mapping[str, Any], keys: list[str]) -> str:
    titles = [leg_title(segment_data, k) for k in keys]
    if len(titles) > _MAX_LISTED:
        return ', '.join(titles[:_MAX_LISTED]) + f', ... (+{len(titles) - _MAX_LISTED})'
    return ', '.join(titles)


def _dir_name(seg: Mapping[str, Any], idx: int) -> str:
    dirs = seg.get('Dirs') if isinstance(seg, Mapping) else None
    if isinstance(dirs, (list, tuple)) and len(dirs) > idx:
        return str(dirs[idx])
    return f'direction {idx + 1}'


# ---------------------------------------------------------------------------
# Project + legs
# ---------------------------------------------------------------------------

def _project_rows(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[list[str]]:
    rows = []
    for key, label in (('omrat_version', 'OMRAT version'), ('project_name', 'Project name')):
        va, vb = a.get(key), b.get(key)
        if (va or vb) and va != vb:
            rows.append(['Project', label, _fmt(va), _fmt(vb)])
    return rows


def _leg_rows(seg_a, seg_b, matched, only_a, only_b) -> list[list[str]]:
    rows = []
    if only_a:
        rows.append(['Legs', f'Legs only in A ({len(only_a)})', _names(seg_a, only_a), '—'])
    if only_b:
        rows.append(['Legs', f'Legs only in B ({len(only_b)})', '—', _names(seg_b, only_b)])
    for k in matched:
        sa, sb = seg_a[k], seg_b[k]
        title = leg_title(seg_a, k)
        for field, label in (('Start_Point', 'start point'), ('End_Point', 'end point')):
            pa, pb = parse_xy(sa.get(field)), parse_xy(sb.get(field))
            if pa is None and pb is None:
                continue
            if pa is not None and pb is not None and max(
                    abs(pa[0] - pb[0]), abs(pa[1] - pb[1])) <= _POINT_TOL_DEG:
                continue
            moved = f' (moved {haversine_m(pa, pb):,.0f} m)' if pa and pb else ''
            rows.append(['Legs', f'{title}: {label}', fmt_xy(pa), fmt_xy(pb) + moved])
        if list(sa.get('Dirs') or []) != list(sb.get('Dirs') or []):
            rows.append(['Legs', f'{title}: direction names',
                         ', '.join(map(str, sa.get('Dirs') or [])) or '—',
                         ', '.join(map(str, sb.get('Dirs') or [])) or '—'])
        for idx in (0, 1):
            field = f'ai{idx + 1}'
            if not _close(sa.get(field), sb.get(field)):
                rows.append(['Legs', f'{title}: Cat II check interval ai, {_dir_name(sa, idx)} (s)',
                             _fmt(sa.get(field)), _fmt(sb.get(field))])
    return rows


# ---------------------------------------------------------------------------
# Traffic + lateral distribution
# ---------------------------------------------------------------------------

def _flat(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for v in value:
            out.extend(_flat(v))
        return out
    return [value]


def _block_fields_differ(block_a: Mapping[str, Any], block_b: Mapping[str, Any]) -> list[str]:
    fields = []
    for field in sorted(set(block_a) | set(block_b)):
        fa, fb = _flat(block_a.get(field)), _flat(block_b.get(field))
        if len(fa) != len(fb) or not all(_close(x, y) for x, y in zip(fa, fb)):
            fields.append(str(field))
    return fields


def _traffic_rows(snap_a, snap_b, matched) -> list[list[str]]:
    rows = []
    seg_a = snap_a.get('segment_data') or {}
    td_a = snap_a.get('traffic_data') or {}
    td_b = snap_b.get('traffic_data') or {}
    for k in matched:
        dirs_a = td_a.get(k) or {}
        dirs_b = td_b.get(k) or {}
        title = leg_title(seg_a, k)
        for d in list(dirs_a) + [d for d in dirs_b if d not in dirs_a]:
            ba, bb = dirs_a.get(d) or {}, dirs_b.get(d) or {}
            qa, qb = direction_frequency(ba), direction_frequency(bb)
            if not _close(qa, qb):
                rows.append(['Traffic', f'{title} {d}: ships/year', f'{qa:,.0f}', f'{qb:,.0f}'])
                continue
            fields = _block_fields_differ(ba, bb)
            if fields:
                rows.append(['Traffic', f'{title} {d}: fields that differ', '—', ', '.join(fields)])
    return rows


_COMPONENT_FIELDS = (('mean', 'mean (m)'), ('std', 'std (m)'), ('weight', 'weight'))


def _distribution_rows(seg_a, seg_b, matched) -> list[list[str]]:
    rows = []
    for k in matched:
        sa, sb = seg_a[k], seg_b[k]
        title = leg_title(seg_a, k)
        for d in (1, 2):
            dname = _dir_name(sa, d - 1)
            fields = [(f'{p}{d}_{i}', f'{label}, component {i}')
                      for i in (1, 2, 3) for p, label in _COMPONENT_FIELDS]
            fields += [(f'u_min{d}', 'uniform min (m)'), (f'u_max{d}', 'uniform max (m)'),
                       (f'u_p{d}', 'uniform weight')]
            for field, label in fields:
                if field not in sa and field not in sb:
                    continue
                if not _close(sa.get(field), sb.get(field)):
                    rows.append(['Lateral distribution', f'{title} {dname}: {label}',
                                 _fmt(sa.get(field)), _fmt(sb.get(field))])
    return rows


# ---------------------------------------------------------------------------
# Routing (suppression / redirects)
# ---------------------------------------------------------------------------

def _redirect_text(seg: Mapping[str, Any], segment_data: Mapping[str, Any]) -> str:
    parts = []
    for r in seg.get('traffic_redirect') or []:
        if not isinstance(r, Mapping):
            continue
        target = segment_data.get(str(r.get('leg'))) or {}
        try:
            to_dir = _dir_name(target, int(r.get('dir', 0)))
            from_dir = _dir_name(seg, int(r.get('from_dir', 0)))
        except (TypeError, ValueError):
            to_dir = from_dir = '?'
        parts.append(f"{from_dir} -> {leg_title(segment_data, r.get('leg'))} {to_dir} "
                     f"{_fmt(r.get('share'))} %")
    return '; '.join(parts)


def _routing_rows(seg_a, seg_b, matched) -> list[list[str]]:
    rows = []
    for k in matched:
        sa, sb = seg_a[k], seg_b[k]
        title = leg_title(seg_a, k)
        if bool(sa.get('suppressed')) != bool(sb.get('suppressed')):
            rows.append(['Routing', f'{title}: suppressed',
                         'yes' if sa.get('suppressed') else 'no',
                         'yes' if sb.get('suppressed') else 'no'])
        wa, wb = sa.get('suppressed_with'), sb.get('suppressed_with')
        if (wa or wb) and str(wa) != str(wb):
            rows.append(['Routing', f'{title}: suppressed together with',
                         leg_title(seg_a, wa) if wa else '—', leg_title(seg_b, wb) if wb else '—'])
        ra, rb = _redirect_text(sa, seg_a), _redirect_text(sb, seg_b)
        if ra != rb:
            rows.append(['Routing', f'{title}: traffic moved to', ra or '—', rb or '—'])
    return rows


# ---------------------------------------------------------------------------
# Junctions
# ---------------------------------------------------------------------------

def _junction_label(jid: str, j: Mapping[str, Any], segment_data: Mapping[str, Any]) -> str:
    pt = j.get('point') if isinstance(j, Mapping) else None
    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
        where = f'{float(pt[0]):.4f} {float(pt[1]):.4f}'
    else:
        where = jid
    legs = [leg_title(segment_data, k) for k in sorted((j.get('legs') or {}) if isinstance(j, Mapping) else {})]
    return f'Junction {where}' + (f' [{", ".join(legs)}]' if legs else '')


def _junction_rows(snap_a, snap_b) -> list[list[str]]:
    rows = []
    ja = snap_a.get('junctions') or {}
    jb = snap_b.get('junctions') or {}
    seg_a = snap_a.get('segment_data') or {}
    seg_b = snap_b.get('segment_data') or {}
    only_a = [j for j in ja if j not in jb]
    only_b = [j for j in jb if j not in ja]
    if only_a:
        rows.append(['Junctions', f'Junctions only in A ({len(only_a)})',
                     '; '.join(_junction_label(j, ja[j], seg_a) for j in only_a[:_MAX_LISTED]), '—'])
    if only_b:
        rows.append(['Junctions', f'Junctions only in B ({len(only_b)})', '—',
                     '; '.join(_junction_label(j, jb[j], seg_b) for j in only_b[:_MAX_LISTED])])
    for jid in [j for j in ja if j in jb]:
        a, b = ja[jid] or {}, jb[jid] or {}
        label = _junction_label(jid, a, seg_a)
        legs_a, legs_b = a.get('legs') or {}, b.get('legs') or {}
        same_legs = set(legs_a) == set(legs_b) and all(
            leg_identity(seg_a, k) == leg_identity(seg_b, k) for k in legs_a)
        if not same_legs:
            rows.append(['Junctions', f'{label}: legs',
                         ', '.join(leg_title(seg_a, k) for k in sorted(legs_a)) or '—',
                         ', '.join(leg_title(seg_b, k) for k in sorted(legs_b)) or '—'])
            continue
        if a.get('source') != b.get('source'):
            rows.append(['Junctions', f'{label}: matrix source', _fmt(a.get('source')), _fmt(b.get('source'))])
        ta, tb = a.get('transitions') or {}, b.get('transitions') or {}
        changes = []
        for src in set(ta) | set(tb):
            for dst in set(ta.get(src) or {}) | set(tb.get(src) or {}):
                va = finite((ta.get(src) or {}).get(dst)) or 0.0
                vb = finite((tb.get(src) or {}).get(dst)) or 0.0
                if abs(va - vb) > _SHARE_TOL:
                    changes.append((abs(va - vb), src, dst, va, vb))
        if changes:
            changes.sort(reverse=True)
            _d, src, dst, va, vb = changes[0]
            more = f' (+{len(changes) - 1} more)' if len(changes) > 1 else ''
            rows.append(['Junctions',
                         f'{label}: share {leg_title(seg_a, src)} -> {leg_title(seg_a, dst)}{more}',
                         f'{va:.2f}', f'{vb:.2f}'])
    return rows


# ---------------------------------------------------------------------------
# Depths + structures
# ---------------------------------------------------------------------------

def _area_km2(*wkt_texts: str) -> float | None:
    """Area of the union of the polygons (so duplicates are not counted twice)."""
    try:
        from shapely import wkt as _wkt
        from shapely.ops import transform, unary_union
        geom = unary_union([_wkt.loads(w) for w in wkt_texts])
        if geom.is_empty:
            return 0.0
        return transform(local_projector(geom.centroid.y), geom).area / 1e6
    except Exception:  # nosec B110 B112 -- area is a nice-to-have
        return None


def _fmt_area(area: float | None) -> str:
    return '?' if area is None else f'{area:,.1f}'


def _depth_classes(depths: Any) -> dict[float, list[str]]:
    out: dict[float, list[str]] = {}
    for row in depths or []:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        v = finite(row[1])
        if v is None:
            continue
        out.setdefault(v, []).append(str(row[2]))
    return out


def _depth_rows(snap_a, snap_b) -> list[list[str]]:
    ca = _depth_classes(snap_a.get('depths'))
    cb = _depth_classes(snap_b.get('depths'))
    if {k: sorted(v) for k, v in ca.items()} == {k: sorted(v) for k, v in cb.items()}:
        return []
    na = sum(len(v) for v in ca.values())
    nb = sum(len(v) for v in cb.values())
    rows = [['Depths', 'Depth polygons', str(na), str(nb)]]
    for depth in sorted(set(ca) | set(cb)):
        pa, pb = ca.get(depth, []), cb.get(depth, [])
        if sorted(pa) == sorted(pb):
            continue
        area_a = _area_km2(*pa) if pa else 0.0
        area_b = _area_km2(*pb) if pb else 0.0
        rows.append(['Depths', f'Depth class {depth:g} m: polygons / area (km²)',
                     f'{len(pa)} / {_fmt_area(area_a)}', f'{len(pb)} / {_fmt_area(area_b)}'])
    return rows


def _objects(snap: Mapping[str, Any]) -> dict[str, tuple[Any, str]]:
    out = {}
    for row in snap.get('objects') or []:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            out[str(row[0])] = (row[1], str(row[2]))
    return out


def _structure_rows(snap_a, snap_b) -> list[list[str]]:
    oa, ob = _objects(snap_a), _objects(snap_b)
    rows = []

    def _desc(obj: tuple[Any, str]) -> str:
        area = _area_km2(obj[1])
        return f"height {_fmt(obj[0])} m" + (f', {area:,.1f} km²' if area is not None else '')

    for oid in sorted(set(oa) | set(ob)):
        if oid not in ob:
            rows.append(['Structures', f'Structure {oid}', _desc(oa[oid]), '—'])
        elif oid not in oa:
            rows.append(['Structures', f'Structure {oid}', '—', _desc(ob[oid])])
        else:
            (ha, wa), (hb, wb) = oa[oid], ob[oid]
            if not _close(ha, hb):
                rows.append(['Structures', f'Structure {oid}: height (m)', _fmt(ha), _fmt(hb)])
            if wa != wb:
                aa, ab = _area_km2(wa), _area_km2(wb)
                rows.append(['Structures', f'Structure {oid}: geometry, area (km²)',
                             _fmt(None if aa is None else round(aa, 3)),
                             _fmt(None if ab is None else round(ab, 3))])
    return rows


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def build_input_table(snap_a: Any, snap_b: Any) -> list[list[str]]:
    """Rows ``[group, item, run_a, run_b]`` for every input that differs.

    Groups come in :data:`INPUT_GROUPS` order.  Identical inputs give
    no rows.
    """
    a = snap_a if isinstance(snap_a, dict) else {}
    b = snap_b if isinstance(snap_b, dict) else {}
    seg_a = a.get('segment_data') or {}
    seg_b = b.get('segment_data') or {}
    seg_a = {str(k): v for k, v in seg_a.items() if isinstance(v, Mapping)}
    seg_b = {str(k): v for k, v in seg_b.items() if isinstance(v, Mapping)}
    matched, only_a, only_b = match_legs(seg_a, seg_b)
    rows: list[list[str]] = []
    rows += _project_rows(a, b)
    rows += _leg_rows(seg_a, seg_b, matched, only_a, only_b)
    rows += _traffic_rows({**a, 'segment_data': seg_a}, b, matched)
    rows += _distribution_rows(seg_a, seg_b, matched)
    rows += _routing_rows(seg_a, seg_b, matched)
    rows += _junction_rows({**a, 'segment_data': seg_a}, {**b, 'segment_data': seg_b})
    rows += _depth_rows(a, b)
    rows += _structure_rows(a, b)
    return rows


def describe(rows: list[list[str]]) -> str:
    """One-line count per group, for the summary label."""
    counts: dict[str, int] = {}
    for r in rows:
        counts[r[0]] = counts.get(r[0], 0) + 1
    return ', '.join(f'{g}: {counts[g]}' for g in INPUT_GROUPS if g in counts) or 'none'
