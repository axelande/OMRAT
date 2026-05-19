"""Audit/transparency section for OMRAT result reports.

Renders a "Choices & Deltas" Markdown section that diffs the user's
project inputs against IWRAP defaults, so reviewers can see at a glance
which knobs were turned and which were left alone.  QGIS-free: operates
on the same dict shape that ``storage.py`` writes to ``.omrat``.

Scope (intentionally narrow):
    - Causation factors vs IWRAP defaults
    - Drift parameters vs defaults (speed, wind rose, anchor, blackout)
    - Input completeness (depths / objects / traffic per leg)
    - Junction matrices and their ``source`` provenance
    - Route geometry vs the write-once ``segments_imported`` baseline
    - Consequence inputs: per-matrix "default vs modified" bit

Out of scope (open work, see CLAUDE.md):
    - Per-cell consequence deltas (the matrices are large; one bit per
      matrix is enough to surface that the analyst changed something)
    - Tamper-resistance against text-edited .omrat files (this is an
      honest-actor tool; if anyone hand-edits both ``segment_data`` and
      ``segments_imported`` to match, no diff can detect it)
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

from compute.iwrap_defaults import (
    IWRAP_PC_DEFAULTS,
    IWRAP_DRIFT_DEFAULTS,
    IWRAP_ROSE_DEFAULT,
    IWRAP_REPAIR_DEFAULT,
)

__all__ = [
    "build_choices_and_deltas_markdown",
    "parse_point_wkt",
    "haversine_m",
]


def parse_point_wkt(wkt: str) -> tuple[float, float] | None:
    """Pull ``(lon, lat)`` out of a ``POINT (...)`` / ``POINT Z (...)`` WKT."""
    if not isinstance(wkt, str):
        return None
    m = re.search(r'\(([^)]+)\)', wkt)
    if not m:
        return None
    parts = m.group(1).split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def haversine_m(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Great-circle distance in metres between two ``(lon, lat)`` pairs."""
    lon1, lat1 = p1
    lon2, lat2 = p2
    r = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _fmt_sig(x: Any, sig: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v == 0:
        return "0"
    if abs(v) < 1e-3 or abs(v) >= 1e4:
        return format(v, f".{sig - 1}e")
    return format(v, f".{sig}g")


def _pct_delta(current: float, default: float) -> str:
    if default == 0:
        return "—" if current == 0 else "n/a"
    pct = (current - default) / default * 100.0
    if abs(pct) < 0.5:
        return "—"
    sign = "+" if pct > 0 else "−"
    return f"{sign}{abs(pct):.0f}%"


def _is_close(a: float, b: float, rel: float = 1e-3, absv: float = 1e-9) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=absv)
    except (TypeError, ValueError):
        return False


def _pc_section(pc: dict[str, Any]) -> list[str]:
    lines = ["### Causation factors (pc)"]
    lines.append("| Factor | This run | IWRAP default | Δ |")
    lines.append("|---|---:|---:|---:|")
    order = ['headon', 'overtaking', 'crossing', 'bend', 'grounding', 'allision', 'p_pc', 'd_pc']
    any_delta = False
    for key in order:
        cur = pc.get(key, IWRAP_PC_DEFAULTS.get(key))
        dflt = IWRAP_PC_DEFAULTS.get(key)
        if cur is None or dflt is None:
            continue
        try:
            cur_f = float(cur)
            dflt_f = float(dflt)
        except (TypeError, ValueError):
            continue
        if _is_close(cur_f, dflt_f):
            delta = "—"
        else:
            delta = "⚠ " + _pct_delta(cur_f, dflt_f)
            any_delta = True
        lines.append(
            f"| pc.{key} | {_fmt_sig(cur_f)} | {_fmt_sig(dflt_f)} | {delta} |"
        )
    if not any_delta:
        lines.append("")
        lines.append("_All causation factors match IWRAP defaults._")
    lines.append("")
    return lines


def _drift_section(drift: dict[str, Any]) -> list[str]:
    lines = ["### Drift parameters"]
    lines.append("| Parameter | This run | Default | Δ |")
    lines.append("|---|---:|---:|---:|")

    scalar_specs: list[tuple[str, str]] = [
        ('speed', 'Drift speed (kn)'),
        ('anchor_p', 'Anchor probability'),
        ('anchor_d', 'Anchor distance factor'),
        ('drift_p', 'Drift probability'),
        ('start_from', 'Start from'),
        ('squat_mode', 'Squat mode'),
    ]
    for key, label in scalar_specs:
        cur = drift.get(key)
        dflt = IWRAP_DRIFT_DEFAULTS.get(key)
        if cur is None and dflt is None:
            continue
        if isinstance(dflt, (int, float)) and isinstance(cur, (int, float)):
            same = _is_close(cur, dflt)
            delta = "—" if same else "⚠ " + _pct_delta(float(cur), float(dflt))
            cur_s, dflt_s = _fmt_sig(cur), _fmt_sig(dflt)
        else:
            same = str(cur) == str(dflt)
            delta = "—" if same else "⚠ changed"
            cur_s, dflt_s = str(cur), str(dflt)
        lines.append(f"| {label} | {cur_s} | {dflt_s} | {delta} |")

    rose = drift.get('rose') or {}
    if isinstance(rose, dict) and rose:
        is_uniform = all(
            _is_close(float(rose.get(k, 0.0)), IWRAP_ROSE_DEFAULT[k])
            for k in IWRAP_ROSE_DEFAULT
        )
        if is_uniform:
            lines.append("| Wind rose | uniform 0.125 | uniform 0.125 | — |")
        else:
            ranked = sorted(
                ((k, float(v)) for k, v in rose.items()),
                key=lambda kv: -kv[1],
            )
            top_two = ", ".join(f"{k}°={v:.3f}" for k, v in ranked[:2])
            lines.append(
                f"| Wind rose | non-uniform ({top_two}, …) | uniform 0.125 | ⚠ skewed |"
            )

    repair = drift.get('repair') or {}
    if isinstance(repair, dict):
        rep_diffs: list[str] = []
        for k in ('std', 'loc', 'scale', 'use_lognormal'):
            cur = repair.get(k)
            dflt = IWRAP_REPAIR_DEFAULT.get(k)
            if cur is None:
                continue
            if isinstance(cur, bool) or isinstance(dflt, bool):
                if bool(cur) != bool(dflt):
                    rep_diffs.append(f"{k}={cur}")
            else:
                try:
                    if not _is_close(float(cur), float(dflt)):
                        rep_diffs.append(f"{k}={cur} (default {dflt})")
                except (TypeError, ValueError):
                    pass
        if rep_diffs:
            lines.append(
                f"| Repair lognormal | {'; '.join(rep_diffs)} | "
                f"std=0.95, loc=0.2, scale=0.85 | ⚠ changed |"
            )
        else:
            lines.append("| Repair lognormal | default | default | — |")

    blackout = drift.get('blackout_by_ship_type') or {}
    if isinstance(blackout, dict) and blackout:
        non_default_count = 0
        for k, v in blackout.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            expected = 0.1 if idx in (8, 9, 10, 11) else 1.0
            try:
                if not _is_close(float(v), expected):
                    non_default_count += 1
            except (TypeError, ValueError):
                pass
        if non_default_count:
            lines.append(
                f"| Blackout per ship type | {non_default_count} type(s) overridden "
                f"| 1.0 (Ro-ro/Passenger=0.1) | ⚠ changed |"
            )
        else:
            lines.append("| Blackout per ship type | IWRAP defaults | IWRAP defaults | — |")

    lines.append("")
    return lines


def _completeness_section(data: dict[str, Any]) -> list[str]:
    lines = ["### Input completeness"]
    lines.append("| Input | Status |")
    lines.append("|---|---|")

    seg_data = data.get('segment_data') or {}
    n_legs = len(seg_data)

    depths = data.get('depths') or []
    n_depths = len(depths)
    lines.append(
        f"| Depths layer | {n_depths} feature{'s' if n_depths != 1 else ''} "
        f"{'✓' if n_depths > 0 else '⚠ empty (drifting/powered grounding will be 0)'} |"
    )

    objects = data.get('objects') or []
    n_obj = len(objects)
    lines.append(
        f"| Objects layer | {n_obj} feature{'s' if n_obj != 1 else ''} "
        f"{'✓' if n_obj > 0 else '⚠ empty (powered allision will be 0)'} |"
    )

    td = data.get('traffic_data') or {}
    legs_with_traffic = 0
    for leg_id in seg_data.keys():
        if leg_id in td and td[leg_id]:
            legs_with_traffic += 1
    status = '✓' if legs_with_traffic == n_legs and n_legs > 0 else f'⚠ {legs_with_traffic}/{n_legs}'
    lines.append(
        f"| Traffic per leg | {legs_with_traffic}/{n_legs} {status} |"
    )

    junctions = data.get('junctions') or {}
    n_junc = len(junctions) if isinstance(junctions, dict) else 0
    sources: dict[str, int] = {}
    if isinstance(junctions, dict):
        for j in junctions.values():
            if isinstance(j, dict):
                src = str(j.get('source', 'geometry'))
                sources[src] = sources.get(src, 0) + 1
    if n_junc:
        src_str = ", ".join(f"{n} {s}" for s, n in sorted(sources.items()))
        lines.append(f"| Junction matrices | {n_junc} ({src_str}) |")
    else:
        lines.append("| Junction matrices | none |")
    lines.append("")
    return lines


def _geometry_section(data: dict[str, Any], move_threshold_m: float = 25.0) -> list[str]:
    seg_data = data.get('segment_data') or {}
    imported = data.get('segments_imported') or {}

    lines = ["### Route geometry vs imported baseline"]

    if not imported:
        lines.append(
            "_No imported-geometry baseline on file._  This project pre-dates "
            "the audit feature, or was created without IWRAP import / canvas "
            "digitize.  The next save will stamp the current geometry as the "
            "baseline so future runs can flag waypoint moves."
        )
        lines.append("")
        return lines

    lines.append("| Leg | Endpoint | Imported | Current | Moved |")
    lines.append("|---|---|---|---|---:|")

    any_move = False
    for sid in sorted(seg_data.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k)):
        seg = seg_data.get(sid, {}) or {}
        imp = imported.get(str(sid), {}) or {}
        for endpoint in ('Start_Point', 'End_Point'):
            cur = parse_point_wkt(seg.get(endpoint, ''))
            ref = parse_point_wkt(imp.get(endpoint, ''))
            if not cur or not ref:
                continue
            d = haversine_m(cur, ref)
            if d < move_threshold_m:
                continue
            any_move = True
            ref_s = f"({ref[0]:.6f}, {ref[1]:.6f})"
            cur_s = f"({cur[0]:.6f}, {cur[1]:.6f})"
            lines.append(
                f"| {sid} | {endpoint.replace('_', ' ')} | {ref_s} | {cur_s} | ⚠ {d:.0f} m |"
            )

    if not any_move:
        lines.append("| _all endpoints_ | — | match baseline | match baseline | — |")
    else:
        lines.append("")
        lines.append(
            f"_Endpoints within {move_threshold_m:.0f} m of their imported "
            "baseline are not listed._"
        )
    lines.append("")
    return lines


def _consequence_section(data: dict[str, Any]) -> list[str]:
    lines = ["### Consequence inputs"]
    cons = data.get('consequence')
    if not isinstance(cons, dict) or not cons:
        lines.append("_No consequence block in this project._")
        lines.append("")
        return lines

    lines.append("| Matrix | Cells | Status |")
    lines.append("|---|---:|---|")

    def _cell_count(mx: Any) -> int:
        if not isinstance(mx, list):
            return 0
        return sum(len(row) if isinstance(row, list) else 0 for row in mx)

    for key, label, default_hint in (
        ('oil_onboard', 'Oil onboard (m³)', '80 × avg_length for tankers, 100 for others'),
        ('spill_probability', 'Spill probability (%)', 'rows sum to 100; drift [98,2,0,0], others [97,1,1,1]'),
        ('spill_fraction', 'Spill fraction (%)', '[0, 10, 30, 100] per accident type'),
    ):
        mx = cons.get(key) or []
        n = _cell_count(mx)
        lines.append(f"| {label} | {n} | _see appendix for full table_ — default: {default_hint} |")

    levels = cons.get('catastrophe_levels') or []
    if isinstance(levels, list) and levels:
        default_levels = [('Minor', 50), ('Major', 500), ('Catastrophic', 5000)]
        match = (
            len(levels) == 3
            and all(
                isinstance(lv, dict)
                and lv.get('name') == name
                and _is_close(float(lv.get('quantity', lv.get('quantity_m3', 0))), q)
                for lv, (name, q) in zip(levels, default_levels)
            )
        )
        status = 'default' if match else f'⚠ {len(levels)} custom level(s)'
        summary = ', '.join(
            f"{lv.get('name', '?')}={lv.get('quantity', lv.get('quantity_m3', '?'))} m³"
            for lv in levels
            if isinstance(lv, dict)
        )
        lines.append(f"| Catastrophe levels | {len(levels)} | {status} ({summary}) |")
    lines.append("")
    return lines


def build_choices_and_deltas_markdown(data: dict[str, Any]) -> str:
    """Build the audit/transparency section as Markdown.

    ``data`` is the dict shape persisted to ``.omrat`` (the same one
    handed to :func:`omrat_utils.full_report.build_full_report_markdown`
    via its ``data`` argument).
    """
    md: list[str] = []
    md.append("## Choices & Deltas from IWRAP defaults")
    md.append("")
    md.append(
        "_This section auto-derives every project knob that differs from "
        "IWRAP defaults so reviewers don't have to read the input file._"
    )
    md.append("")

    pc = data.get('pc') or {}
    md.extend(_pc_section(pc))

    drift = data.get('drift') or {}
    md.extend(_drift_section(drift))

    md.extend(_completeness_section(data))
    md.extend(_geometry_section(data))
    md.extend(_consequence_section(data))

    return "\n".join(md)
