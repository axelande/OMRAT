"""What traffic each leg carries, and whether it looks right (pure; no Qt / QGIS).

The data behind the Traffic report (:mod:`omrat_utils.traffic_report_html`,
opened from **Traffic report...** on the Traffic Data tab and File menu).
:func:`build_traffic_summary` takes the project dict
(``GatherData.get_all_for_save``, the same shape as a ``.omrat`` file) and
returns plain dicts and lists:

* per leg and direction the ships per year *as stored* (what the Traffic
  tab shows) and *in the calculation*.  The latter repeats Run model's own
  steps on a private copy: :func:`compute.traffic_scaling.apply_traffic_scaling`,
  then :func:`compute.traffic_redirect.apply_traffic_redirects` (suppressed
  legs leave, their ships join the target legs), then
  :func:`compute.speed_restriction.apply_speed_restrictions` (lower speeds
  in speed restriction zones).  Keep it in step with
  ``compute/traffic_pipeline.py`` (gap filling is left out on purpose: the
  report shows the gaps);
* where a leg's traffic comes from (own data, copy, suppressed, none);
* the ship-type mix, mean speed / draught and the type x length matrix;
* the automatic checks (:data:`CHECK_TEXT` lists them), each a
  ``warning`` (probably wrong) or a ``note`` (check that it is expected).

Ships are counted at each leg's own passage line, so one ship that sails
three legs is counted on each of them; a sum over legs is not a number of
ships, and the report never shows one.
"""
from __future__ import annotations

import copy
import math
from datetime import datetime
from typing import Any

from compute.speed_restriction import (
    ZONES_KEY, SpeedRestrictionError, apply_speed_restrictions, zone_label,
)
from compute.traffic_gaps import DRAUGHT_KEY, FILL_RULE, FREQ_KEY, SPEED_KEY, cell_gaps
from compute.traffic_redirect import (
    apply_traffic_redirects, dir_key, direction_bearing, direction_label, get_redirect,
    group_lead, group_members, is_suppressed,
)
from compute.traffic_scaling import apply_traffic_scaling
from geometries.route_validation import parse_wkt_point
from omrat_utils.copy_traffic import LOCK_KEY, SOURCE_KEY

JUMP_RATIO = 1.5
"""Two legs meeting alone at a waypoint carry the same ships; a ratio above
this between their totals is reported (``jump``)."""
JUMP_MIN_SHIPS = 100.0
"""...when the difference is at least this many ships per year."""
GAP_WARN_SHARE = 0.05
"""Ships without speed / draught are a warning (not a note) when they are
more than this share of some leg's traffic."""

KINDS = ('own', 'copy', 'received', 'suppressed', 'none')
"""Where a leg's traffic comes from; ``received`` = gets ships from a
suppressed leg (calculation view only)."""

CHECK_TEXT = {
    'no_traffic': "Legs in the calculation without any ships",
    'one_direction': "Ships in one direction only",
    'copy_stale': "Copies that no longer match their source leg",
    'copy_source_missing': "Copies whose source leg was removed",
    'jump': "Neighbouring legs with very different traffic",
    'gaps': "Ships without speed or draught",
    'scaling': "Traffic scaling other than 100 %",
    'suppressed': "Suppressed legs and where their ships go",
    'suppressed_lost': "Suppressed ships without a target",
    'redirect': "Problems moving suppressed traffic",
    'locked': "Locked legs (Update all distributions skips them)",
    'speed_zone': "Legs in speed restriction zones (lower speed in the calculation)",
    'speed_zone_error': "Speed restriction zones that stop Run model",
}
"""Every check the report runs, for its "what was checked" list."""

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _positive(value: Any) -> float:
    v = _num(value)
    return v if v is not None and v > 0 else 0.0


def _cell(matrix: Any, i: int, j: int) -> Any:
    try:
        return matrix[i][j]
    except (TypeError, IndexError, KeyError):
        return None


def _shape(block: Any) -> tuple[int, int]:
    freq = block.get(FREQ_KEY) if isinstance(block, dict) else None
    if not isinstance(freq, list) or not freq:
        return 0, 0
    return len(freq), max((len(r) for r in freq if isinstance(r, list)), default=0)


def block_total(block: Any) -> float:
    """Ships per year in one direction block (finite, positive cells)."""
    rows, cols = _shape(block)
    freq = block.get(FREQ_KEY) if rows else None
    return sum(_positive(_cell(freq, i, j)) for i in range(rows) for j in range(cols))


def _weighted_mean(block: Any, field: str, rows: range, cols: range) -> float | None:
    """Ship-weighted mean of ``field`` over the cells with a valid value."""
    if not isinstance(block, dict):
        return None
    freq, values = block.get(FREQ_KEY), block.get(field)
    num = den = 0.0
    for i in rows:
        for j in cols:
            q = _positive(_cell(freq, i, j))
            v = _positive(_cell(values, i, j))
            if q > 0 and v > 0:
                num += q * v
                den += q
    return num / den if den > 0 else None


def _matrices_equal(a: Any, b: Any) -> bool:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if not isinstance(ra, list) or not isinstance(rb, list) or len(ra) != len(rb):
            return False
        for x, y in zip(ra, rb):
            vx, vy = _positive(x), _positive(y)
            if abs(vx - vy) > 1e-6 * max(1.0, abs(vx), abs(vy)):
                return False
    return True


def haversine_m(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Great-circle distance in metres between two ``(lon, lat)`` pairs."""
    lon1, lat1, lon2, lat2 = map(math.radians, (p1[0], p1[1], p2[0], p2[1]))
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6_371_000.0 * math.asin(math.sqrt(a))


def short_type(label: str) -> str:
    """``'Cargo, all ships of this type'`` -> ``'Cargo'``."""
    return str(label).split(',')[0].strip() or str(label)


def fmt_ships(q: float | None) -> str:
    """``1,234`` ships per year; one decimal below 10 (``2.5``, but ``3``),
    ``0`` for none."""
    if q is None:
        return '-'
    if q <= 0:
        return '0'
    if q >= 10 or abs(q - round(q)) < 0.05:
        return f"{q:,.0f}"
    return f"{q:,.1f}"


def _pct_change(new: float, old: float) -> str:
    if old <= 0:
        return 'new'
    pct = (new - old) / old * 100.0
    return f"{'+' if pct >= 0 else '-'}{abs(pct):.0f} %"


def _labels(data: dict[str, Any], rows: int, cols: int) -> tuple[list[str], list[str]]:
    cats = data.get('ship_categories') if isinstance(data.get('ship_categories'), dict) else {}
    types = [str(t) for t in (cats.get('types') or [])]
    lengths = []
    for iv in cats.get('length_intervals') or []:
        if isinstance(iv, dict):
            label = iv.get('label') or f"{iv.get('min', '')} - {iv.get('max', '')}"
            lengths.append(str(label))
    types += [f"Type {i + 1}" for i in range(len(types), rows)]
    lengths += [f"Length class {j + 1}" for j in range(len(lengths), cols)]
    return types, lengths


# ---------------------------------------------------------------------------
# Leg names / provenance
# ---------------------------------------------------------------------------

def _name(seg: str, segs: dict[str, Any]) -> str:
    seg_d = segs.get(str(seg))
    name = seg_d.get('Leg_name') if isinstance(seg_d, dict) else None
    return str(name) if name else f"leg {seg}"


def _copy_source(seg_d: dict[str, Any]) -> str | None:
    src = seg_d.get(SOURCE_KEY)
    return str(src) if src not in (None, '') else None


def _dir_short(label: str) -> str:
    word = str(label).split()[0] if label else ''
    return word if word in ('North', 'South', 'East', 'West') else str(label)


def _redirect_text(seg: str, segs: dict[str, Any], traffic: dict[str, Any]) -> str:
    parts = []
    for e in get_redirect(segs, seg):
        if e['leg'] not in segs:
            continue
        frm = _dir_short(direction_label(seg, e['from_dir'], segs, traffic))
        parts.append(f"{frm} {e['share']:g} % -> {_name(e['leg'], segs)}")
    return ', '.join(parts)


# ---------------------------------------------------------------------------
# Building the summary
# ---------------------------------------------------------------------------

def _direction_entry(seg: str, d: int, segs: dict[str, Any], traffic: dict[str, Any],
                     rows: int, cols: int) -> dict[str, Any]:
    seg_d = segs.get(seg) or {}
    block = traffic.get(seg) if isinstance(traffic.get(seg), dict) else {}
    key = dir_key(block, seg_d, d) if block else None
    var = block.get(key) if key is not None else None
    var = var if isinstance(var, dict) else {}
    freq = var.get(FREQ_KEY)
    matrix = [[_positive(_cell(freq, i, j)) for j in range(cols)] for i in range(rows)]
    no_speed = no_draught = 0.0
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] <= 0:
                continue
            missing = cell_gaps(var, i, j)
            if 'speed' in missing:
                no_speed += matrix[i][j]
            if 'draught' in missing:
                no_draught += matrix[i][j]
    all_cols = range(cols)
    return {
        'index': d,
        'key': key,
        'label': direction_label(seg, d, segs, traffic),
        'bearing': direction_bearing(seg_d, d),
        'ships': sum(map(sum, matrix)),
        'matrix': matrix,
        'mean_speed': _weighted_mean(var, SPEED_KEY, range(rows), all_cols),
        'mean_draught': _weighted_mean(var, DRAUGHT_KEY, range(rows), all_cols),
        'type_speed': [_weighted_mean(var, SPEED_KEY, range(i, i + 1), all_cols) for i in range(rows)],
        'type_draught': [_weighted_mean(var, DRAUGHT_KEY, range(i, i + 1), all_cols) for i in range(rows)],
        'no_speed': no_speed,
        'no_draught': no_draught,
        'scaled': 0.0,
        'calc': None,
        'received': 0.0,
    }


def _calculation_view(data: dict[str, Any]) -> tuple[dict, dict, dict[str, Any], dict[str, Any], list[str]]:
    """``(scaled traffic, calculated traffic, redirect summary, speed zone
    summary, speed zone problems)`` -- Run model's own pre-processing on a
    private copy (lateral samples left out).  Zones that would stop the run
    give no speed change and their problems instead."""
    segs = {k: {f: v for f, v in d.items() if f not in ('dist1', 'dist2')} if isinstance(d, dict) else d
            for k, d in (data.get('segment_data') or {}).items()}
    work = {'segment_data': copy.deepcopy(segs),
            'traffic_data': copy.deepcopy(data.get('traffic_data') or {}),
            'ship_categories': copy.deepcopy(data.get('ship_categories') or {}),
            ZONES_KEY: copy.deepcopy(data.get(ZONES_KEY) or [])}
    apply_traffic_scaling(work)
    scaled = copy.deepcopy(work['traffic_data'])
    summary = apply_traffic_redirects(work)
    try:
        speed, speed_problems = apply_speed_restrictions(work), []
    except SpeedRestrictionError as exc:
        speed, speed_problems = {'zones': [], 'legs': []}, [p for p in str(exc).splitlines() if p]
    calc = {'segment_data': work['segment_data'], 'traffic_data': work['traffic_data']}
    return scaled, calc, summary, speed, speed_problems


def _speed_zone_text(row: dict[str, Any], zones: dict[int, dict[str, Any]]) -> str:
    """``Zone 1 "Anholt" 43 %`` (or ``... (end point only)``) for one leg."""
    parts = [f"{zone_label(zones.get(zid, {'id': zid}))} {100.0 * share:.0f} %"
             for zid, share in sorted((row.get('shares') or {}).items())]
    for key, end in (('start_zone', 'start'), ('end_zone', 'end')):
        zid = row.get(key)
        if zid is not None and zid not in (row.get('shares') or {}):
            parts.append(f"{zone_label(zones.get(zid, {'id': zid}))} ({end} point only)")
    return ', '.join(parts)


def _node_key(text: Any) -> tuple[float, float] | None:
    p = parse_wkt_point(text)
    return (round(p[0], 6), round(p[1], 6)) if p is not None else None


def build_traffic_summary(
    data: dict[str, Any], *, project_path: str = '', generated: datetime | None = None,
) -> dict[str, Any]:
    """Everything the Traffic report shows, from the project dict ``data``
    (left unchanged)."""
    segs: dict[str, Any] = {str(k): v for k, v in (data.get('segment_data') or {}).items() if isinstance(v, dict)}
    traffic: dict[str, Any] = {str(k): v for k, v in (data.get('traffic_data') or {}).items()
                               if isinstance(v, dict)}
    rows = cols = 0
    for dirs in traffic.values():
        for var in dirs.values():
            r, c = _shape(var)
            rows, cols = max(rows, r), max(cols, c)
    types, lengths = _labels(data, rows, cols)
    rows, cols = max(rows, len(types)), max(cols, len(lengths))

    scaled_traffic, calc, redirect_summary, speed_summary, speed_problems = _calculation_view(data)
    calc_segs, calc_traffic = calc['segment_data'], calc['traffic_data']
    speed_zones = {z['id']: z for z in speed_summary.get('zones') or []}
    speed_rows = {str(r['leg']): r for r in speed_summary.get('legs') or []}

    legs: list[dict[str, Any]] = []
    for seg, seg_d in segs.items():
        sp, ep = parse_wkt_point(seg_d.get('Start_Point')), parse_wkt_point(seg_d.get('End_Point'))
        length = _num(seg_d.get('line_length'))
        if (length is None or length <= 0) and sp is not None and ep is not None:
            length = haversine_m(sp, ep)
        dirs = [_direction_entry(seg, d, segs, traffic, rows, cols) for d in (0, 1)]
        for entry in dirs:
            s_block = scaled_traffic.get(seg)
            s_key = dir_key(s_block, seg_d, entry['index']) if isinstance(s_block, dict) and s_block else None
            entry['scaled'] = block_total(s_block.get(s_key)) if s_key is not None else 0.0
            if seg in calc_segs:
                c_block = calc_traffic.get(seg)
                c_key = (dir_key(c_block, calc_segs[seg], entry['index'])
                         if isinstance(c_block, dict) and c_block else None)
                entry['calc'] = block_total(c_block.get(c_key)) if c_key is not None else 0.0
        by_type = [sum(d['matrix'][i][j] for d in dirs for j in range(cols)) for i in range(rows)]
        total = sum(by_type)
        top = sorted(((i, q) for i, q in enumerate(by_type) if q > 0), key=lambda t: -t[1])[:3]
        all_rows, all_cols = range(rows), range(cols)
        block = traffic.get(seg) or {}
        both = [block.get(d['key']) for d in dirs if d['key'] is not None]
        suppressed = is_suppressed(segs, seg)
        speed_zone = None
        if seg in speed_rows and seg in calc_segs:
            c_block = calc_traffic.get(seg) or {}
            speed_zone = {
                'zones': _speed_zone_text(speed_rows[seg], speed_zones),
                'calc_speed': _combined_mean(list(c_block.values()), SPEED_KEY, range(rows), range(cols)),
            }
        legs.append({
            'id': seg,
            'name': _name(seg, segs),
            'route': str(seg_d.get('Route_Id', '') or ''),
            'length_m': length,
            'start': sp,
            'end': ep,
            'dirs': dirs,
            'ships': total,
            'scaled': sum(d['scaled'] for d in dirs),
            'calc': None if suppressed else sum((d['calc'] or 0.0) for d in dirs),
            'by_type': by_type,
            'top_types': [(i, q, q / total) for i, q in top],
            'mean_speed': _combined_mean(both, SPEED_KEY, all_rows, all_cols),
            'mean_draught': _combined_mean(both, DRAUGHT_KEY, all_rows, all_cols),
            'no_speed': sum(d['no_speed'] for d in dirs),
            'no_draught': sum(d['no_draught'] for d in dirs),
            'locked': seg_d.get(LOCK_KEY) is True,
            'copy_of': _copy_source(seg_d),
            'suppressed': suppressed,
            'suppressed_with': group_lead(segs, seg),
            'speed_zone': speed_zone,
            'received_moves': [],
            'checks': [],
        })
    by_id = {leg['id']: leg for leg in legs}

    for mv in redirect_summary.get('moves', []):
        dst = by_id.get(str(mv['dst']))
        if dst is None:
            continue
        dst['received_moves'].append({
            'src': str(mv['src']), 'src_name': _name(mv['src'], segs),
            'dst': dst['id'], 'dst_name': dst['name'],
            'from_label': direction_label(mv['src'], mv['from_dir'], segs, traffic),
            'dir_label': direction_label(mv['dst'], mv['dir'], segs, traffic),
            'share': mv['share'], 'ships': mv['ships'],
        })
        dst['dirs'][mv['dir']]['received'] += mv['ships']

    for leg in legs:
        leg['kind'], leg['calc_kind'] = _kinds(leg)
        leg['source_text'] = _source_text(leg, segs, traffic)

    checks = _run_checks(legs, by_id, segs, traffic, redirect_summary)
    checks.extend(_speed_zone_checks(legs, speed_problems))
    for idx, check in enumerate(checks):
        for seg in check['legs']:
            if seg in by_id:
                by_id[seg]['checks'].append(idx)

    type_idx = [i for i in range(rows) if any(leg['by_type'][i] > 0 for leg in legs)]
    calc_differs = any(
        abs((d['calc'] if d['calc'] is not None else 0.0) - d['ships']) > 1e-6 or leg['suppressed']
        for leg in legs for d in leg['dirs']
    )
    in_calc = [leg for leg in legs if not leg['suppressed']]
    busiest = max(legs, key=lambda leg: leg['ships'], default=None)
    generated = generated or datetime.now()
    figures = {
        'legs': len(legs),
        'in_calc': len(in_calc),
        'suppressed': len(legs) - len(in_calc),
        'busiest': (busiest['id'], busiest['name'], busiest['ships']) if busiest and busiest['ships'] > 0 else None,
        'no_traffic': sum(1 for leg in in_calc if (leg['calc'] or 0.0) <= 0),
        'copies': sum(1 for leg in legs if leg['copy_of'] is not None),
        'locked': sum(1 for leg in legs if leg['locked']),
        'warnings': sum(1 for c in checks if c['level'] == 'warning'),
        'notes': sum(1 for c in checks if c['level'] == 'note'),
        'types_with_ships': len(type_idx),
    }
    return {
        'project': {
            'name': str(data.get('project_name') or ''),
            'file': str(project_path or ''),
            'omrat_version': str(data.get('omrat_version') or ''),
            'generated': generated.strftime('%Y-%m-%d %H:%M'),
        },
        'types': types,
        'lengths': lengths,
        'legs': legs,
        'checks': checks,
        'figures': figures,
        'type_index': type_idx,
        'calc_differs': calc_differs,
        'scaling': dict(data.get('traffic_scaling') or {}),
    }


def _combined_mean(blocks: list[Any], field: str, rows: range, cols: range) -> float | None:
    num = den = 0.0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        freq, values = block.get(FREQ_KEY), block.get(field)
        for i in rows:
            for j in cols:
                q, v = _positive(_cell(freq, i, j)), _positive(_cell(values, i, j))
                if q > 0 and v > 0:
                    num += q * v
                    den += q
    return num / den if den > 0 else None


def _kinds(leg: dict[str, Any]) -> tuple[str, str]:
    """``(kind as stored, kind in the calculation)``."""
    if leg['suppressed']:
        return 'suppressed', 'suppressed'
    stored = 'none' if leg['ships'] <= 0 else ('copy' if leg['copy_of'] is not None else 'own')
    if (leg['calc'] or 0.0) <= 0:
        return stored, 'none'
    if leg['received_moves']:
        return stored, 'received'
    return stored, ('own' if stored == 'none' else stored)


def _source_text(leg: dict[str, Any], segs: dict[str, Any], traffic: dict[str, Any]) -> str:
    seg = leg['id']
    if leg['suppressed_with'] is not None:
        return f"Suppressed together with {_name(leg['suppressed_with'], segs)} (not in the calculation)"
    if leg['suppressed']:
        targets = _redirect_text(seg, segs, traffic)
        n_with = len(group_members(segs, seg))
        with_txt = f" (+{n_with} leg(s) with it)" if n_with else ''
        return (f"Suppressed{with_txt}, not in the calculation; ships moved: {targets}" if targets
                else f"Suppressed{with_txt}, not in the calculation; no target given")
    if leg['ships'] <= 0:
        text = 'No traffic'
    elif leg['copy_of'] is not None:
        text = f"Copy of {_name(leg['copy_of'], segs)}" + (' (locked)' if leg['locked'] else '')
    else:
        text = 'Own data' + (' (locked)' if leg['locked'] else '')
    if leg['received_moves']:
        srcs = list(dict.fromkeys(mv['src_name'] for mv in leg['received_moves']))
        text += f"; receives ships from suppressed {', '.join(srcs)}"
    return text


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check(level: str, code: str, legs: list[str], text: str) -> dict[str, Any]:
    return {'level': level, 'code': code, 'legs': [str(s) for s in legs], 'text': text}


def _copy_matches(dst: dict[str, Any], src: dict[str, Any], dst_seg: dict, src_seg: dict) -> bool:
    """``dst`` holds ``src``'s frequencies, straight or with the directions swapped."""
    def freq(block: dict, seg_d: dict, d: int) -> Any:
        key = dir_key(block, seg_d, d) if block else None
        var = block.get(key) if key is not None else None
        return var.get(FREQ_KEY) if isinstance(var, dict) else None
    d0, d1 = freq(dst, dst_seg, 0), freq(dst, dst_seg, 1)
    s0, s1 = freq(src, src_seg, 0), freq(src, src_seg, 1)
    straight = _matrices_equal(d0, s0) and _matrices_equal(d1, s1)
    return straight or (_matrices_equal(d0, s1) and _matrices_equal(d1, s0))


def _run_checks(legs: list[dict], by_id: dict[str, dict], segs: dict[str, Any], traffic: dict[str, Any],
                redirect_summary: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    active = [leg for leg in legs if not leg['suppressed']]

    for leg in active:
        if (leg['calc'] or 0.0) <= 0:
            text = (f"{leg['name']} has no ships." if leg['ships'] <= 0 else
                    f"{leg['name']} has no ships in the calculation: traffic scaling sets its "
                    f"{fmt_ships(leg['ships'])} ships/year to 0.")
            checks.append(_check('warning', 'no_traffic', [leg['id']], text))

    for leg in active:
        d0, d1 = leg['dirs']
        if leg['ships'] > 0 and (d0['ships'] <= 0 or d1['ships'] <= 0):
            full, empty = (d0, d1) if d0['ships'] > 0 else (d1, d0)
            checks.append(_check(
                'note', 'one_direction', [leg['id']],
                f"{leg['name']} has ships {full['label']} only ({fmt_ships(full['ships'])} ships/year); "
                f"{empty['label']} has none. Fine for a one-way lane; otherwise check the passage line."))

    for leg in legs:
        src = leg['copy_of']
        if src is None:
            continue
        if src not in segs:
            checks.append(_check('warning', 'copy_source_missing', [leg['id']],
                                 f"{leg['name']} was copied from leg id {src}, which no longer exists."))
            continue
        if not _copy_matches(traffic.get(leg['id']) or {}, traffic.get(src) or {}, segs[leg['id']], segs[src]):
            checks.append(_check(
                'warning', 'copy_stale', [leg['id'], src],
                f"{leg['name']} is a copy of {by_id[src]['name']}, but the two no longer carry the same "
                f"traffic ({fmt_ships(leg['ships'])} and {fmt_ships(by_id[src]['ships'])} ships/year). "
                f"One of them changed after the copy; run Copy traffic again if the copy should follow it."))

    checks.extend(_jump_checks(active, segs))

    checks.extend(_gap_checks(legs))

    scaled = [leg for leg in legs if abs(leg['scaled'] - leg['ships']) > 1e-6 * max(1.0, leg['ships'])]
    if scaled:
        before = sum(leg['ships'] for leg in scaled)
        after = sum(leg['scaled'] for leg in scaled)
        shown = ', '.join(f"{leg['name']} {_pct_change(leg['scaled'], leg['ships'])}" for leg in scaled[:8])
        more = f" and {len(scaled) - 8} more" if len(scaled) > 8 else ''
        checks.append(_check(
            'note', 'scaling', [leg['id'] for leg in scaled],
            f"Traffic scaling changes {len(scaled)} leg(s) in the calculation "
            f"({_pct_change(after, before)} on those legs): {shown}{more}."))

    for leg in legs:
        if not leg['suppressed'] or leg['suppressed_with'] is not None:
            continue
        moved = [mv for other in legs for mv in other['received_moves'] if mv['src'] == leg['id']]
        members = group_members(segs, leg['id'])
        with_txt = (f" together with {', '.join(_name(m, segs) for m in members)}" if members else '')
        if moved:
            targets = '; '.join(f"{mv['from_label']} {mv['share']:g} % -> {mv['dst_name']} "
                                f"{mv['dir_label']} ({fmt_ships(mv['ships'])} ships/year)" for mv in moved)
            text = f"{leg['name']} is suppressed{with_txt}. Its ships go to: {targets}."
        else:
            text = f"{leg['name']} is suppressed{with_txt} and moves no ships."
        checks.append(_check('note', 'suppressed', [leg['id']] + members, text))
        targeted = {e['from_dir'] for e in get_redirect(segs, leg['id'])
                    if e['leg'] in segs and e['share'] > 0}
        for d in leg['dirs']:
            if d['scaled'] > 0 and d['index'] not in targeted:
                checks.append(_check(
                    'warning', 'suppressed_lost', [leg['id']],
                    f"{leg['name']} is suppressed, but its {d['label']} ships ({fmt_ships(d['scaled'])} "
                    f"ships/year) have no target: they are left out of the calculation."))

    for msg in redirect_summary.get('warnings', []):
        checks.append(_check('warning', 'redirect', [], f"Suppressed traffic: {msg}."))

    locked = [leg for leg in legs if leg['locked']]
    if locked:
        names = ', '.join(leg['name'] for leg in locked[:12])
        more = f" and {len(locked) - 12} more" if len(locked) > 12 else ''
        checks.append(_check(
            'note', 'locked', [leg['id'] for leg in locked],
            f"{len(locked)} leg(s) are locked, so Update all distributions leaves their traffic as it is: "
            f"{names}{more}."))

    order = {'warning': 0, 'note': 1}
    checks.sort(key=lambda c: order.get(c['level'], 2))
    return checks


def _speed_zone_checks(legs: list[dict], problems: list[str]) -> list[dict[str, Any]]:
    """Legs whose speed the zones lower (note) and zones that stop the run (warning)."""
    checks: list[dict[str, Any]] = []
    if problems:
        checks.append(_check('warning', 'speed_zone_error', [], (
            "Speed restriction zones stop Run model: " + ' '.join(problems))))
    in_zone = [leg for leg in legs if leg.get('speed_zone')]
    if in_zone:
        names = ', '.join(f"{leg['name']} ({leg['speed_zone']['zones']})" for leg in in_zone)
        checks.append(_check('note', 'speed_zone', [leg['id'] for leg in in_zone], (
            f"{len(in_zone)} leg(s) pass through speed restriction zones; the calculation uses a lower "
            f"speed than the table shows (the time-equivalent speed along the leg, the zone speed at a "
            f"waypoint inside a zone): {names}.")))
    return checks


def _gap_checks(legs: list[dict]) -> list[dict[str, Any]]:
    """One line per quantity: a few ships without a draught is normal for an
    AIS sample, so it is a note unless :data:`GAP_WARN_SHARE` of a leg's
    ships lack it."""
    out: list[dict[str, Any]] = []
    for quantity, key in (('speed', 'no_speed'), ('draught', 'no_draught')):
        affected = sorted(((leg, leg[key]) for leg in legs if leg[key] > 0), key=lambda t: -t[1])
        if not affected:
            continue
        worst = max(q / leg['ships'] for leg, q in affected if leg['ships'] > 0)
        shown = ', '.join(f"{leg['name']} {fmt_ships(q)}" for leg, q in affected[:8])
        more = f" and {len(affected) - 8} more" if len(affected) > 8 else ''
        out.append(_check(
            'warning' if worst > GAP_WARN_SHARE else 'note', 'gaps', [leg['id'] for leg, _q in affected],
            f"{len(affected)} leg(s) have ships without a {quantity}, "
            f"{fmt_ships(sum(q for _leg, q in affected))} ships/year in all ({shown}{more}). {FILL_RULE}"))
    return out


def _jump_checks(active: list[dict], segs: dict[str, Any]) -> list[dict[str, Any]]:
    """Two legs that meet alone at a waypoint carry the same ships."""
    nodes: dict[tuple[float, float], list[dict]] = {}
    for leg in active:
        seg_d = segs.get(leg['id']) or {}
        for end in ('Start_Point', 'End_Point'):
            key = _node_key(seg_d.get(end))
            if key is not None:
                nodes.setdefault(key, []).append(leg)
    out: list[dict[str, Any]] = []
    seen: set[frozenset] = set()
    for incident in nodes.values():
        if len(incident) != 2 or incident[0] is incident[1]:
            continue
        a, b = incident
        pair = frozenset((a['id'], b['id']))
        if pair in seen:
            continue
        seen.add(pair)
        lo, hi = sorted((a, b), key=lambda leg: leg['ships'])
        if lo['ships'] <= 0 or hi['ships'] - lo['ships'] < JUMP_MIN_SHIPS:
            continue
        ratio = hi['ships'] / lo['ships']
        if ratio <= JUMP_RATIO:
            continue
        out.append(_check(
            'note', 'jump', [a['id'], b['id']],
            f"{hi['name']} ({fmt_ships(hi['ships'])} ships/year) and {lo['name']} ({fmt_ships(lo['ships'])}) "
            f"meet at a waypoint with no other leg, yet {hi['name']} carries {ratio:.1f} times as many ships. "
            f"The same ships sail both legs: check their passage lines, or copy the traffic of the better leg."))
    return out


# ---------------------------------------------------------------------------
# CSV (Excel) tables
# ---------------------------------------------------------------------------

def _csv_number(value: float | None, decimal: str, digits: int = 1) -> str:
    if value is None:
        return ''
    text = f"{value:.{digits}f}"
    return text.replace('.', decimal) if decimal != '.' else text


def _csv_text(value: Any, sep: str) -> str:
    text = str(value)
    if any(ch in text for ch in (sep, '"', '\n', '\r')):
        text = '"' + text.replace('"', '""') + '"'
    return text


def legs_csv(summary: dict[str, Any], decimal: str = '.') -> str:
    """The leg table as CSV.  A comma decimal point gets ``;`` between the
    fields, which is what Excel expects in those locales."""
    sep = ';' if decimal == ',' else ','
    head = ['Leg', 'Id', 'Route', 'Length (km)', 'Direction 1', 'Ships/year dir 1', 'Direction 2',
            'Ships/year dir 2', 'Ships/year total', 'In calculation', 'Source', 'Mean speed (kn)',
            'Mean draught (m)', 'Warnings', 'Notes']
    lines = [sep.join(head)]
    checks = summary['checks']
    for leg in summary['legs']:
        d0, d1 = leg['dirs']
        n_warn = sum(1 for i in leg['checks'] if checks[i]['level'] == 'warning')
        n_note = sum(1 for i in leg['checks'] if checks[i]['level'] == 'note')
        length_km = leg['length_m'] / 1000.0 if leg['length_m'] else None
        row = [
            _csv_text(leg['name'], sep), _csv_text(leg['id'], sep), _csv_text(leg['route'], sep),
            _csv_number(length_km, decimal, 2),
            _csv_text(d0['label'], sep), _csv_number(d0['ships'], decimal),
            _csv_text(d1['label'], sep), _csv_number(d1['ships'], decimal),
            _csv_number(leg['ships'], decimal), _csv_number(leg['calc'], decimal),
            _csv_text(leg['source_text'], sep),
            _csv_number(leg['mean_speed'], decimal), _csv_number(leg['mean_draught'], decimal),
            str(n_warn), str(n_note),
        ]
        lines.append(sep.join(row))
    return '\r\n'.join(lines) + '\r\n'


def types_csv(summary: dict[str, Any], decimal: str = '.') -> str:
    """Ships per year per ship type (columns) and leg (rows), both directions."""
    sep = ';' if decimal == ',' else ','
    idx = summary['type_index']
    lines = [sep.join(['Leg', 'Id'] + [_csv_text(summary['types'][i], sep) for i in idx] + ['Total'])]
    for leg in summary['legs']:
        cells = [_csv_number(leg['by_type'][i], decimal) for i in idx]
        lines.append(sep.join([_csv_text(leg['name'], sep), _csv_text(leg['id'], sep)] + cells
                              + [_csv_number(leg['ships'], decimal)]))
    return '\r\n'.join(lines) + '\r\n'
