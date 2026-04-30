"""Combined Markdown report covering every OMRAT accident type.

Reuses :meth:`compute.drifting_report.DriftingReportMixin.generate_drifting_report_markdown`
for the drifting section (the canonical, well-tested layout the user
already knows from ``drifting_report.md``) and adds equivalent sections
for powered grounding/allision and ship-ship collisions so a single
``<name>_results_<timestamp>.md`` captures everything a run produced.

The functions are written to be QGIS-soft: they take plain dicts (the
calc reports) and the input-snapshot ``data`` dict, so they can be
unit-tested without a QGIS instance.
"""
from __future__ import annotations

from typing import Any, Iterable

__all__ = ["build_full_report_markdown"]


def _fmt(x: Any, fmt: str = ".3e") -> str:
    try:
        return format(float(x), fmt)
    except (TypeError, ValueError):
        return "n/a"


def _powered_section(
    title: str,
    report: dict[str, Any] | None,
    obstacle_records: Iterable[dict[str, Any]] | None,
    value_label: str,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"## {title}")
    if not report:
        lines.append("_No powered model report available._\n")
        return lines

    totals = report.get("totals", {}) or {}
    by_obstacle = report.get("by_obstacle", {}) or {}
    by_obstacle_leg = report.get("by_obstacle_leg", {}) or {}

    lines.append(
        f"- Total: {_fmt(next(iter(totals.values()), 0.0))}"
    )
    cf = report.get("causation_factor")
    if cf is not None:
        lines.append(f"- Causation factor: {_fmt(cf, '.4f')}")
    lines.append("")

    # Per-obstacle breakdown.
    obs_meta: dict[str, dict[str, Any]] = {
        str(rec.get("id", "")): rec for rec in (obstacle_records or [])
    }
    if by_obstacle:
        lines.append(f"### Per obstacle ({value_label})")
        lines.append(f"| Obstacle | {value_label} | Probability |")
        lines.append("|---|---:|---:|")
        for obs_id, prob in sorted(
            by_obstacle.items(), key=lambda kv: -float(kv[1] or 0.0)
        ):
            meta = obs_meta.get(str(obs_id), {})
            val_key = "depth" if value_label.lower().startswith("depth") else "height"
            val = meta.get(val_key, "")
            lines.append(f"| {obs_id} | {val} | {_fmt(prob)} |")
        lines.append("")

    # Per-leg breakdown (inverted view of by_obstacle_leg).
    if by_obstacle_leg:
        by_leg: dict[str, float] = {}
        by_leg_obs: dict[str, dict[str, float]] = {}
        for obs_id, leg_map in by_obstacle_leg.items():
            for leg_id, contrib in (leg_map or {}).items():
                by_leg[leg_id] = by_leg.get(leg_id, 0.0) + float(contrib or 0.0)
                by_leg_obs.setdefault(leg_id, {})[str(obs_id)] = float(contrib or 0.0)
        if by_leg:
            lines.append("### Per leg (summed across obstacles)")
            lines.append("| Leg | Total | Top contributor |")
            lines.append("|---|---:|---|")
            for leg_id, total in sorted(
                by_leg.items(),
                key=lambda kv: -float(kv[1] or 0.0),
            ):
                contribs = by_leg_obs.get(leg_id, {})
                if contribs:
                    top_obs, top_val = max(contribs.items(), key=lambda kv: kv[1])
                    top_text = f"obs {top_obs} ({_fmt(top_val)})"
                else:
                    top_text = "—"
                lines.append(f"| {leg_id} | {_fmt(total)} | {top_text} |")
            lines.append("")
    return lines


def _ship_collision_section(report: dict[str, Any] | None) -> list[str]:
    lines: list[str] = []
    lines.append("## Ship-ship collisions")
    if not report:
        lines.append("_No collision report available._\n")
        return lines

    totals = report.get("totals", {}) or {}
    lines.append("| Type | Probability |")
    lines.append("|---|---:|")
    for key in ("head_on", "overtaking", "crossing", "merging", "bend"):
        if key in totals:
            label = key.replace('_', '-').title()
            lines.append(f"| {label} | {_fmt(totals.get(key, 0.0))} |")
    if "total" in totals:
        lines.append(f"| **Combined** | {_fmt(totals.get('total', 0.0))} |")
    lines.append("")

    by_leg = report.get("by_leg", {}) or {}
    if by_leg:
        lines.append("### Per-leg breakdown")
        # Discover the column set from the first non-empty leg entry.
        first = next(
            (v for v in by_leg.values() if isinstance(v, dict) and v),
            {},
        )
        cols = [k for k in first.keys() if k not in ('leg_id',)]
        if cols:
            header = "| Leg | " + " | ".join(c.title() for c in cols) + " |"
            sep = "|---|" + "|".join(["---:"] * len(cols)) + "|"
            lines.append(header)
            lines.append(sep)
            for leg_id, vals in sorted(
                by_leg.items(),
                key=lambda kv: int(str(kv[0])) if str(kv[0]).isdigit() else str(kv[0]),
            ):
                row = [str(leg_id)] + [
                    _fmt(vals.get(c, 0.0)) for c in cols
                ]
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return lines


def build_full_report_markdown(
    *,
    run_name: str,
    timestamp: str,
    data: dict[str, Any],
    drifting_md: str | None,
    powered_grounding_report: dict[str, Any] | None,
    powered_allision_report: dict[str, Any] | None,
    collision_report: dict[str, Any] | None,
    structures_meta: Iterable[dict[str, Any]] | None = None,
    depths_meta: Iterable[dict[str, Any]] | None = None,
    totals_summary: dict[str, float] | None = None,
) -> str:
    """Compose the combined Markdown report for one OMRAT run.

    ``drifting_md`` is the already-rendered drifting section (the
    output of :meth:`generate_drifting_report_markdown`).  The other
    ``*_report`` arguments are the raw dicts captured on the calc
    object.  Everything else is metadata used for the header and the
    overall summary table.
    """
    md: list[str] = []
    md.append(f"# OMRAT results - {run_name}")
    md.append("")
    md.append(f"_Generated: {timestamp}_")
    md.append("")

    # ------------------------------------------------------------------
    # 1. Top-level summary across every accident type.
    # ------------------------------------------------------------------
    md.append("## Summary")
    md.append("| Accident type | Probability |")
    md.append("|---|---:|")
    summary = dict(totals_summary or {})
    if not summary:
        if drifting_md:
            # Pull the two drift totals out of the rendered drifting markdown
            # so we don't duplicate calculation logic.
            for line in drifting_md.splitlines():
                low = line.lower()
                if low.startswith('- total allision:'):
                    summary['drift_allision'] = line.split(':', 1)[1].strip()
                elif low.startswith('- total grounding:'):
                    summary['drift_grounding'] = line.split(':', 1)[1].strip()
        if powered_grounding_report:
            t = (powered_grounding_report.get('totals') or {}).get('grounding')
            if t is not None:
                summary['powered_grounding'] = _fmt(t)
        if powered_allision_report:
            t = (powered_allision_report.get('totals') or {}).get('allision')
            if t is not None:
                summary['powered_allision'] = _fmt(t)
        if collision_report:
            ctotals = collision_report.get('totals') or {}
            for k in ('head_on', 'overtaking', 'crossing', 'merging', 'bend'):
                if k in ctotals:
                    summary[f'ship_{k}'] = _fmt(ctotals[k])
    pretty = {
        'drift_allision': 'Drifting allision',
        'drift_grounding': 'Drifting grounding',
        'powered_allision': 'Powered allision',
        'powered_grounding': 'Powered grounding',
        'ship_head_on': 'Head-on collision',
        'ship_overtaking': 'Overtaking collision',
        'ship_crossing': 'Crossing collision',
        'ship_merging': 'Merging collision',
        'ship_bend': 'Bend collision',
    }
    for key, label in pretty.items():
        if key in summary:
            md.append(f"| {label} | {summary[key]} |")
    md.append("")

    # ------------------------------------------------------------------
    # 2. Drifting (verbatim from generate_drifting_report_markdown).
    # ------------------------------------------------------------------
    if drifting_md:
        md.append("---")
        md.append("")
        md.append("# Drifting model")
        md.append("")
        md.append(drifting_md)
        md.append("")

    # ------------------------------------------------------------------
    # 3. Powered grounding + allision.
    # ------------------------------------------------------------------
    md.append("---")
    md.append("")
    md.append("# Powered model (Cat. II)")
    md.append("")
    md.extend(_powered_section(
        "Powered grounding",
        powered_grounding_report,
        depths_meta,
        "Depth (m)",
    ))
    md.extend(_powered_section(
        "Powered allision",
        powered_allision_report,
        structures_meta,
        "Height (m)",
    ))

    # ------------------------------------------------------------------
    # 4. Ship-ship collisions.
    # ------------------------------------------------------------------
    md.append("---")
    md.append("")
    md.extend(_ship_collision_section(collision_report))

    md.append("")
    md.append(
        "_Generated by OMRAT.  Values are annual probabilities unless "
        "otherwise stated._"
    )
    return "\n".join(md)
