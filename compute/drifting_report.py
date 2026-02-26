"""
Drifting report generation mixin.

Extracted from run_calculations.py to keep the report-formatting logic
in its own module.  The mixin expects the host class to provide:

    self.drifting_report : dict[str, Any] | None
        Populated by the drifting calculation pipeline.
    self.p.main_widget
        Reference to the QGIS plugin widget (used elsewhere; not
        directly needed by these three methods but present on the
        same class).
"""
from __future__ import annotations

from typing import Any


class DriftingReportMixin:
    """Mixin supplying drifting-report helpers for RunCalculations."""

    # -- public API ----------------------------------------------------------

    def get_drifting_report(self) -> dict[str, Any] | None:
        return self.drifting_report

    def generate_drifting_report_markdown(self, data: dict[str, Any] | None = None) -> str:
        """
        Build a human-readable Markdown appendix report from the last drifting run.

        Includes:
        - Parameter summary (drift, anchoring, repair)
        - Overall totals (allision, grounding)
        - Directional aggregates
        - Per leg-direction highlights
        - Ship category breakdown
        """
        rep = self.drifting_report or {}
        totals = rep.get('totals', {})
        bld: dict[str, Any] = rep.get('by_leg_direction', {})
        by_obj: dict[str, Any] = rep.get('by_object', {})
        by_struct_legdir: dict[str, Any] = rep.get('by_structure_legdir', {})

        # Parameter summary
        drift = {} if data is None else data.get('drift', {})
        repair = drift.get('repair', {}) if isinstance(drift, dict) else {}
        rose = drift.get('rose', {}) if isinstance(drift, dict) else {}

        # Aggregates
        total_base_hours = 0.0
        dir_agg: dict[str, dict[str, float]] = {}
        ship_cat_totals: dict[str, dict[str, float]] = {}
        leg_rows: list[tuple[str, float, float, float, float, float]] = []
        # seg:dir key => metrics
        for key, rec in bld.items():
            total_base_hours += float(rec.get('base_hours', 0.0))
            try:
                angle = key.split(':')[-1]
            except Exception:
                angle = '0'
            da = dir_agg.setdefault(angle, {
                'allision': 0.0,
                'grounding': 0.0,
                'base_hours': 0.0,
                'anchor_factor_sum': 0.0,
                'not_repaired_sum': 0.0,
                'overlap_sum': 0.0,
                'weight_sum': 0.0,
            })
            a = float(rec.get('contrib_allision', 0.0))
            g = float(rec.get('contrib_grounding', 0.0))
            da['allision'] += a
            da['grounding'] += g
            da['base_hours'] += float(rec.get('base_hours', 0.0))
            da['anchor_factor_sum'] += float(rec.get('anchor_factor_sum', 0.0))
            da['not_repaired_sum'] += float(rec.get('not_repaired_sum', 0.0))
            da['overlap_sum'] += float(rec.get('overlap_sum', 0.0))
            da['weight_sum'] += float(rec.get('weight_sum', 0.0))

            # Per leg-direction highlight row
            af = float(rec.get('anchor_factor_sum', 0.0))
            wf = float(rec.get('weight_sum', 0.0))
            nrs = float(rec.get('not_repaired_sum', 0.0))
            ovs = float(rec.get('overlap_sum', 0.0))
            avg_anchor = (af / wf) if wf > 0 else 0.0
            avg_not_rep = (nrs / wf) if wf > 0 else 0.0
            avg_overlap = (ovs / wf) if wf > 0 else 0.0
            # store: key, allision, grounding, avg_anchor, avg_not_rep, avg_overlap
            leg_rows.append((key, a, g, avg_anchor, avg_not_rep, avg_overlap))

            # Ship categories
            for cat, vals in rec.get('ship_categories', {}).items():
                sct = ship_cat_totals.setdefault(cat, {'allision': 0.0, 'grounding': 0.0, 'freq': 0.0})
                sct['allision'] += float(vals.get('allision', 0.0))
                sct['grounding'] += float(vals.get('grounding', 0.0))
                sct['freq'] += float(vals.get('freq', 0.0))

        # Sort per-leg-direction rows by segment then direction for readability
        def _parse_key(k: str) -> tuple[int, str, int]:
            try:
                parts = k.split(':')
                if len(parts) == 3:
                    seg, legdir, ang = parts
                    return int(str(seg)), str(legdir), int(str(ang))
                elif len(parts) == 2:
                    seg, ang = parts
                    return int(str(seg)), '', int(str(ang))
                else:
                    return 0, '', 0
            except Exception:
                return (0, '', 0)
        leg_rows_sorted = sorted(leg_rows, key=lambda r: _parse_key(r[0]))

        # Directional table rows
        def dir_row(angle: str, d: dict[str, float]) -> str:
            w = d.get('weight_sum', 0.0)
            avg_anchor = (d.get('anchor_factor_sum', 0.0) / w) if w > 0 else 0.0
            avg_not_rep = (d.get('not_repaired_sum', 0.0) / w) if w > 0 else 0.0
            avg_overlap = (d.get('overlap_sum', 0.0) / w) if w > 0 else 0.0
            return f"| {angle}\u00b0 | {d.get('base_hours', 0.0):.2f} | {d.get('allision', 0.0):.3e} | {d.get('grounding', 0.0):.3e} | {avg_anchor:.3f} | {avg_not_rep:.3f} | {avg_overlap:.3f} |"

        md_lines: list[str] = []
        # Apply reduction factors to match GUI display
        pc_vals = data.get('pc', {}) if isinstance(data, dict) and isinstance(data.get('pc', {}), dict) else {}
        allision_rf = float(pc_vals.get('allision_drifting_rf', 1.0))
        grounding_rf = float(pc_vals.get('grounding_drifting_rf', 1.0))

        final_allision = float(totals.get('allision', 0.0)) * allision_rf
        final_grounding = float(totals.get('grounding', 0.0)) * grounding_rf

        md_lines.append("# Drifting Model Appendix Report")
        md_lines.append("")
        md_lines.append("## Summary")
        md_lines.append(f"- Total allision: {final_allision:.3e}")
        md_lines.append(f"- Total grounding: {final_grounding:.3e}")
        md_lines.append(f"- Total allision (before RF): {float(totals.get('allision', 0.0)):.3e}")
        md_lines.append(f"- Total grounding (before RF): {float(totals.get('grounding', 0.0)):.3e}")
        md_lines.append(f"- Allision reduction factor: {allision_rf:.3f}")
        md_lines.append(f"- Grounding reduction factor: {grounding_rf:.3f}")
        md_lines.append(f"- Aggregated ship-hours on legs used in model: {total_base_hours:.2f}")
        md_lines.append("")
        md_lines.append("## Parameters")
        drift_speed_kts = float(drift.get('speed', 0.0)) if isinstance(drift, dict) else 0.0
        md_lines.append(f"- Drift speed: {drift_speed_kts} knots ({drift_speed_kts * 1852.0 / 3600.0:.3f} m/s)")
        md_lines.append(f"- Blackout prob per ship-year: {float(drift.get('drift_p', 0.0)) if isinstance(drift, dict) else 0.0}")
        md_lines.append(f"- Anchor prob: {float(drift.get('anchor_p', 0.0)) if isinstance(drift, dict) else 0.0}, distance factor: {float(drift.get('anchor_d', 0.0)) if isinstance(drift, dict) else 0.0}")
        if isinstance(repair, dict) and repair.get('use_lognormal', False):
            md_lines.append(f"- Repair lognormal (std={float(repair.get('std', 0.0))}, loc={float(repair.get('loc', 0.0))}, scale={float(repair.get('scale', 1.0))})")
        else:
            md_lines.append("- Repair model: not specified/lognormal disabled")
        if isinstance(rose, dict) and rose:
            md_lines.append("- Wind rose: " + ", ".join([f"{k}\u00b0={v}" for k, v in rose.items()]))
        md_lines.append("")

        # Per-leg details: lat/lon and distributions
        try:
            seg_data = {} if data is None else (data.get('segment_data', {}) or {})
            if isinstance(seg_data, dict) and seg_data:
                md_lines.append("## Leg Details")
                for seg_id, sd in seg_data.items():
                    sp = str(sd.get('Start_Point', '')).strip()
                    ep = str(sd.get('End_Point', '')).strip()
                    def _fmt_point(pt: str) -> str:
                        try:
                            s = pt.strip().lstrip('(').rstrip(')')
                            parts = [p for p in s.replace(',', ' ').split() if p]
                            if len(parts) >= 2:
                                lon = float(parts[0])
                                lat = float(parts[1])
                                return f"({lon:.6f}, {lat:.6f})"
                        except Exception:
                            pass
                        return pt
                    start_txt = _fmt_point(sp)
                    end_txt = _fmt_point(ep)
                    length = float(sd.get('line_length', 0.0))
                    md_lines.append(f"### Leg {seg_id}")
                    md_lines.append(f"- Start: {start_txt}")
                    md_lines.append(f"- End: {end_txt}")
                    md_lines.append(f"- Length (m): {length:.2f}")
                    # Directions and distributions
                    dirs = list(sd.get('Dirs', []) or [])
                    for d_idx in (1, 2):
                        label = str(dirs[d_idx-1]) if 0 <= (d_idx-1) < len(dirs) else str(d_idx)
                        # Gather normal components
                        comps: list[str] = []
                        for i in range(1, 4):
                            w = float(sd.get(f'weight{d_idx}_{i}', 0.0) or 0.0)
                            m = float(sd.get(f'mean{d_idx}_{i}', 0.0) or 0.0)
                            sdev = float(sd.get(f'std{d_idx}_{i}', 0.0) or 0.0)
                            if w > 0.0:
                                comps.append(f"w={w:.2f}: N({m:.2f}, {sdev:.2f})")
                        # Uniform component
                        up = float(sd.get(f'u_p{d_idx}', 0.0) or 0.0)
                        if up > 0.0:
                            umin = float(sd.get(f'u_min{d_idx}', 0.0) or 0.0)
                            umax = float(sd.get(f'u_max{d_idx}', 0.0) or 0.0)
                            comps.append(f"u_p={up:.2f}: U[{umin:.2f}, {umax:.2f}]")
                        comp_txt = ", ".join(comps) if comps else "(no active distributions)"
                        md_lines.append(f"- Dir {label}: {comp_txt}")
                    md_lines.append("")
                md_lines.append("")
        except Exception:
            # Do not block report generation if segment details are malformed
            pass

        md_lines.append("## Directional Aggregates")
        md_lines.append("| Direction | Base hours | Allision | Grounding | Avg anchor | Avg not-repaired | Avg overlap |")
        md_lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for ang in sorted(dir_agg.keys(), key=lambda x: int(x)):
            md_lines.append(dir_row(ang, dir_agg[ang]))
        md_lines.append("")

        md_lines.append("## Directional Aggregates per Leg-Direction")
        md_lines.append("| Leg:Dir:Angle | Allision | Grounding | Avg anchor | Avg not-repaired | Avg overlap |")
        md_lines.append("|---|---:|---:|---:|---:|---:|")
        for key, a, g, avg_anchor, avg_not_rep, avg_overlap in leg_rows_sorted:
            md_lines.append(f"| {key} | {a:.3e} | {g:.3e} | {avg_anchor:.3f} | {avg_not_rep:.3f} | {avg_overlap:.3f} |")
        md_lines.append("")

        # Per-structure, per leg-direction contributions (allision only)
        if by_struct_legdir:
            md_lines.append("## Per Structure: Directional Aggregates per Leg-Direction")
            for skey in sorted(by_struct_legdir.keys()):
                md_lines.append(f"### {skey}")
                md_lines.append("| Leg:Dir:Angle | Allision |")
                md_lines.append("|---|---:|")
                s_map = by_struct_legdir[skey] or {}
                for k in sorted(s_map.keys(), key=lambda x: _parse_key(x)):
                    md_lines.append(f"| {k} | {float(s_map[k]):.3e} |")
                md_lines.append("")

        # Per object totals
        if by_obj:
            md_lines.append("## Per Object")
            md_lines.append("| Object | Allision | Grounding |")
            md_lines.append("|---|---:|---:|")
            # Only include structures; omit depths as requested
            for okey in sorted([k for k in by_obj.keys() if str(k).startswith('Structure - ')]):
                ob = by_obj[okey]
                md_lines.append(f"| {okey} | {float(ob.get('allision', 0.0)):.3e} | {float(ob.get('grounding', 0.0)):.3e} |")
            md_lines.append("")

        # Ship Category Breakdown with names
        md_lines.append("## Ship Category Breakdown (by Leg:Dir:Angle)")
        md_lines.append("| Leg:Dir:Angle | Type-Size | Annual Frequency | Allision | Grounding |")
        md_lines.append("|---|---|---:|---:|---:|")
        # Build mapping from indices to labels if provided
        type_labels: list[str] = []
        size_labels: list[str] = []
        if isinstance(data, dict):
            try:
                sc = data.get('ship_categories', {})
                if isinstance(sc, dict):
                    type_labels = list(sc.get('types', []) or [])
                    size_labels = [str(x.get('label', '')) for x in (sc.get('length_intervals', []) or [])]
            except Exception:
                pass
        # Emit by walking leg-direction keys to include granularity
        for legdir_key in sorted(bld.keys(), key=lambda x: _parse_key(x)):
            rec = bld[legdir_key]
            cats = rec.get('ship_categories', {}) or {}
            for cat in sorted(cats.keys()):
                vals = cats[cat]
                disp = cat
                try:
                    s_type, s_size = cat.split('-')
                    ti = int(s_type)
                    si = int(s_size)
                    tname = type_labels[ti] if 0 <= ti < len(type_labels) else s_type
                    sname = size_labels[si] if 0 <= si < len(size_labels) else s_size
                    disp = f"{tname} - {sname}"
                except Exception:
                    pass
                md_lines.append(
                    f"| {legdir_key} | {disp} | {float(vals.get('freq', 0.0)):.2f} | {float(vals.get('allision', 0.0)):.3e} | {float(vals.get('grounding', 0.0)):.3e} |"
                )

        md_lines.append("")

        # Summary of legs
        md_lines.append("## Leg Summary")
        md_lines.append("| Leg | Allision | Grounding |")
        md_lines.append("|---:|---:|---:|")
        leg_sums: dict[str, dict[str, float]] = {}
        for key, rec in bld.items():
            try:
                leg = key.split(':')[0]
            except Exception:
                leg = key
            agg = leg_sums.setdefault(leg, {'allision': 0.0, 'grounding': 0.0})
            agg['allision'] += float(rec.get('contrib_allision', 0.0))
            agg['grounding'] += float(rec.get('contrib_grounding', 0.0))
        for leg in sorted(leg_sums.keys(), key=lambda x: int(str(x)) if str(x).isdigit() else str(x)):
            ls = leg_sums[leg]
            md_lines.append(f"| {leg} | {ls['allision']:.3e} | {ls['grounding']:.3e} |")
        md_lines.append("")

        md_lines.append("")
        md_lines.append("_Generated by OMRA Tool drift model. Values are annual probabilities unless otherwise stated._")
        return "\n".join(md_lines)

    def write_drifting_report_markdown(self, file_path: str, data: dict[str, Any] | None = None) -> str:
        """Generate and write the drifting Markdown report to file_path. Returns the content."""
        content = self.generate_drifting_report_markdown(data)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return content
