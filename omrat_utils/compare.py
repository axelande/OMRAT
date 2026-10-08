"""Side-by-side comparison of two OMRAT ``.omrat`` snapshots.

The user picks two snapshot files from the Compare tab; this module
returns three tables that the tab can drop straight into its
``QTableWidget``s:

* **Accident probabilities** -- per-accident-type probability for run A
  vs B with absolute and percent deltas, the three summary rows, the
  total, and the oil-spill result.  Values come from the run's own
  sidecars next to the snapshot (``<stem>.collision.json`` ...,
  :func:`load_run_outputs`); the run-history database fills in what
  the sidecars lack, so Compare still works for a run whose files are
  partly gone.
* **Settings differences** -- every scalar of the ``drift``, ``pc``,
  ``traffic_scaling``, ``consequence`` and ``ship_categories`` blocks
  (and per-leg ``Width``) that differs between A and B.  Equal fields
  are hidden so the table only shows the things that actually changed.
* **Route-distance per leg** -- length per matching leg id, in metres.
  Mismatching leg ids show in their own row with the missing side
  blank.

Leg geometry, traffic, distributions, junctions, depths and structures
are diffed by :mod:`omrat_utils.compare_inputs`; per-leg / waypoint /
object result differences by :mod:`omrat_utils.compare_breakdown` and
the common-area comparison by :mod:`omrat_utils.compare_zone`.

The module is QGIS-soft; everything here works on plain dicts and is
unit-testable without a QGIS instance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

__all__ = [
    "load_snapshot",
    "load_run_outputs",
    "totals_from_outputs",
    "build_accident_table",
    "build_settings_table",
    "build_leg_distance_table",
]

# ``<stem><suffix>`` sidecars written next to the snapshot by
# ``RunHistoryMixin._auto_save_run`` (one per model phase that ran).
SIDECAR_SUFFIXES: dict[str, str] = {
    "collision": ".collision.json",
    "drifting": ".drifting.json",
    "powered": ".powered.json",
    "consequence": ".consequence.json",
}


# ---------------------------------------------------------------------------
# Snapshot loading
# ---------------------------------------------------------------------------

def load_snapshot(path: str | Path) -> dict[str, Any]:
    """Return the JSON dict from a ``.omrat`` snapshot."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_run_outputs(snapshot_path: str | Path) -> dict[str, Any]:
    """The run's sidecars next to ``snapshot_path``, keyed like :data:`SIDECAR_SUFFIXES`.

    A sidecar that is missing or unreadable is simply left out (a
    partial run writes only the phases it calculated).
    """
    path = Path(snapshot_path)
    out: dict[str, Any] = {}
    for key, suffix in SIDECAR_SUFFIXES.items():
        side = path.with_name(path.stem + suffix)
        if not side.is_file():
            continue
        try:
            with side.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            out[key] = data
    return out


def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def totals_from_outputs(outputs: dict[str, Any] | None) -> dict[str, float]:
    """Accident totals (``ACCIDENT_TOTAL_KEYS`` names) from the run's sidecars.

    Only keys whose phase wrote a sidecar are returned, so a skipped
    phase stays *missing* -- never zero.
    """
    outputs = outputs or {}
    out: dict[str, float] = {}
    coll = (outputs.get("collision") or {}).get("totals") or {}
    for k in ("head_on", "overtaking", "crossing", "merging", "bend"):
        v = _num(coll.get(k))
        if v is not None:
            out[k] = v
    drift = (outputs.get("drifting") or {}).get("totals") or {}
    for src, dst in (("allision", "drift_allision"), ("grounding", "drift_grounding")):
        v = _num(drift.get(src))
        if v is not None:
            out[dst] = v
    powered = outputs.get("powered") or {}
    for mode, dst in (("allision", "powered_allision"), ("grounding", "powered_grounding")):
        v = _num(((powered.get(mode) or {}).get("totals") or {}).get(mode))
        if v is not None:
            out[dst] = v
    return out


def _match_run_history(snapshot_path: Path) -> dict[str, Any] | None:
    """Look up the run-history row of this snapshot's GeoPackage.

    A row whose ``output_dir`` + ``output_filename`` is exactly
    ``<stem>.gpkg`` next to the snapshot wins; otherwise the first row
    with that file name (the snapshot was moved).  Returns ``None``
    when no row is found or when the run-history module is unavailable
    (e.g. unit-test environment without sqlite).
    """
    try:
        from .run_history import RunHistory
    except Exception:  # nosec B110 B112
        return None
    try:
        history = RunHistory()
        runs = history.list_runs()
    except Exception:  # nosec B110 B112
        return None
    snapshot_path = Path(snapshot_path)
    target_gpkg = snapshot_path.stem + ".gpkg"
    try:
        exact = snapshot_path.with_suffix(".gpkg").resolve()
    except OSError:
        exact = None
    by_name = None
    for r in runs:
        try:
            if getattr(r, "output_filename", "") != target_gpkg:
                continue
            row = r.__dict__ if hasattr(r, "__dict__") else dict(r)
            gpkg = r.gpkg_path() if hasattr(r, "gpkg_path") else None
            if exact is not None and gpkg is not None and Path(gpkg).resolve() == exact:
                return row
            if by_name is None:
                by_name = row
        except Exception:  # nosec B110 B112
            continue
    return by_name


# ---------------------------------------------------------------------------
# Accident probabilities
# ---------------------------------------------------------------------------

_ACCIDENT_KEYS: tuple[tuple[str, str], ...] = (
    ("drift_allision", "Drifting allision"),
    ("drift_grounding", "Drifting grounding"),
    ("powered_allision", "Powered allision"),
    ("powered_grounding", "Powered grounding"),
    ("head_on", "Head-on collision"),
    ("overtaking", "Overtaking collision"),
    ("crossing", "Crossing collision"),
    ("merging", "Merging collision"),
    ("bend", "Bend collision"),
)


def _totals_from_run_row(row: dict[str, Any] | None) -> dict[str, float]:
    """Best-effort extraction of accident totals from a run-history row.

    The ``RunHistory.totals`` schema has changed shape across the
    project's life; we accept either a flat dict or a nested
    ``{'drift': {...}, 'collision': {...}}`` form.
    """
    if not row:
        return {}
    out: dict[str, float] = {}
    candidates = (
        row.get("totals"),
        row.get("totals_dict"),
        row,
    )
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        for k in (
            "drift_allision", "drift_grounding",
            "powered_allision", "powered_grounding",
            "head_on", "overtaking", "crossing", "merging", "bend",
        ):
            # ``None`` = the phase was not calculated in that run: leave
            # the key out so the table shows n/a instead of a false zero.
            v = _num(cand.get(k)) if k in cand else None
            if v is not None:
                out[k] = v
        if "drift" in cand and isinstance(cand["drift"], dict):
            d = cand["drift"]
            if "allision" in d:
                out["drift_allision"] = float(d.get("allision", 0.0) or 0.0)
            if "grounding" in d:
                out["drift_grounding"] = float(d.get("grounding", 0.0) or 0.0)
        if "collision" in cand and isinstance(cand["collision"], dict):
            c = cand["collision"]
            for k in ("head_on", "overtaking", "crossing", "merging", "bend"):
                if k in c:
                    try:
                        out[k] = float(c[k] or 0.0)
                    except (TypeError, ValueError):
                        pass
    return out


def _value_row(label: str, a: float | None, b: float | None, fmt: str = ".3e") -> list[str]:
    """``[label, a, b, delta_abs, delta_pct]`` as table text."""
    a_text = "n/a" if a is None else format(a, fmt)
    b_text = "n/a" if b is None else format(b, fmt)
    if a is None or b is None:
        return [label, a_text, b_text, "n/a", "n/a"]
    delta = b - a
    d_abs = format(delta, "+" + fmt)
    if a == 0.0:
        d_pct = "+inf%" if delta > 0 else ("0.00%" if delta == 0.0 else "-inf%")
    else:
        d_pct = f"{(delta / a) * 100:+.2f}%"
    return [label, a_text, b_text, d_abs, d_pct]


def _sum_present(totals: dict[str, float], keys: tuple[str, ...]) -> float | None:
    vals = [totals[k] for k in keys if k in totals]
    return sum(vals) if vals else None


def _consequence_values(cons: dict[str, Any] | None) -> dict[tuple, float]:
    """``{row key: value}`` for the oil-spill rows of one run."""
    if not isinstance(cons, dict):
        return {}
    out: dict[tuple, float] = {}
    v = _num(cons.get("mean_spill_volume_per_year"))
    if v is not None:
        out[("mean",)] = v
    v = _num(cons.get("total_spill_frequency"))
    if v is not None:
        out[("any",)] = v
    for lvl in cons.get("levels") or []:
        if not isinstance(lvl, dict):
            continue
        q = _num(lvl.get("quantity"))
        e = _num(lvl.get("exceedance"))
        if q is None or e is None:
            continue
        out[("level", str(lvl.get("name", "")), q)] = e
    return out


def build_accident_table(
    path_a: Path,
    path_b: Path,
    outputs_a: dict[str, Any] | None = None,
    outputs_b: dict[str, Any] | None = None,
) -> list[list[str]]:
    """Rows: ``[label, run_a, run_b, delta_abs, delta_pct]``.

    The nine accident types, the summary rows (all grounding / allision
    / collisions, all accidents) and -- when either run has a
    ``.consequence.json`` -- the oil-spill rows.  Values come from the
    run's sidecars (``outputs_*``, read with :func:`load_run_outputs`
    when not given); the run-history row fills in what the sidecars
    lack.  Returned values are pre-formatted strings so callers can
    drop them straight into a ``QTableWidget``.
    """
    from .accident_summary import SUMMARY_ROWS

    if outputs_a is None:
        outputs_a = load_run_outputs(path_a)
    if outputs_b is None:
        outputs_b = load_run_outputs(path_b)
    totals_a = _totals_from_run_row(_match_run_history(Path(path_a)))
    totals_b = _totals_from_run_row(_match_run_history(Path(path_b)))
    totals_a.update(totals_from_outputs(outputs_a))
    totals_b.update(totals_from_outputs(outputs_b))

    rows: list[list[str]] = []
    for key, label in _ACCIDENT_KEYS:
        if key not in totals_a and key not in totals_b:
            continue
        rows.append(_value_row(label, totals_a.get(key), totals_b.get(key)))
    if not rows:
        return rows
    all_keys = tuple(k for k, _ in _ACCIDENT_KEYS)
    for label, keys in SUMMARY_ROWS + (("All accidents", all_keys),):
        rows.append(_value_row(label, _sum_present(totals_a, keys), _sum_present(totals_b, keys)))

    cons_a = _consequence_values(outputs_a.get("consequence"))
    cons_b = _consequence_values(outputs_b.get("consequence"))
    if cons_a or cons_b:
        if ("mean",) in cons_a or ("mean",) in cons_b:
            rows.append(_value_row("Oil spill: expected volume (m³/year)",
                                   cons_a.get(("mean",)), cons_b.get(("mean",)), ".4g"))
        if ("any",) in cons_a or ("any",) in cons_b:
            rows.append(_value_row("Oil spill: any spill (per year)",
                                   cons_a.get(("any",)), cons_b.get(("any",))))
        levels = sorted({k for k in list(cons_a) + list(cons_b) if k[0] == "level"}, key=lambda k: k[2])
        for key in levels:
            label = f"Oil spill > {key[2]:g} m³ ({key[1]}) (per year)" if key[1] else \
                f"Oil spill > {key[2]:g} m³ (per year)"
            rows.append(_value_row(label, cons_a.get(key), cons_b.get(key)))
    return rows


# ---------------------------------------------------------------------------
# Settings differences
# ---------------------------------------------------------------------------

# Top-level ``.omrat`` blocks that hold model *settings* (as opposed to
# geometry / traffic).  Every scalar underneath is diffed.
_SETTINGS_BLOCKS: tuple[str, ...] = (
    "drift", "pc", "traffic_scaling", "consequence", "ship_categories",
)

# Human labels for the well-known paths.  Anything not listed falls back
# to a prettified dotted path so new settings are never silently hidden.
_SETTING_LABELS: dict[str, str] = {
    "drift.drift_p": "Drift (blackout) probability",
    "drift.anchor_p": "Anchor probability",
    "drift.anchor_d": "Anchor distance factor",
    "drift.speed": "Drift speed (knots)",
    "drift.start_from": "Drift start position",
    "drift.squat_mode": "Squat mode",
    "drift.repair.func": "Repair-time function",
    "drift.repair.std": "Repair-time std",
    "drift.repair.loc": "Repair-time loc",
    "drift.repair.scale": "Repair-time scale",
    "drift.repair.use_lognormal": "Use lognormal repair time",
    "pc.p_pc": "Causation factor (powered, legacy)",
    "pc.d_pc": "Causation factor (drifting, legacy)",
    "pc.headon": "Causation factor head-on",
    "pc.overtaking": "Causation factor overtaking",
    "pc.crossing": "Causation factor crossing",
    "pc.merging": "Causation factor merging",
    "pc.bend": "Causation factor bend",
    "pc.grounding": "Causation factor powered grounding",
    "pc.allision": "Causation factor powered allision",
    "pc.grounding_cat1": "Causation factor powered grounding (Cat I, in lane)",
    "pc.allision_cat1": "Causation factor powered allision (Cat I, in lane)",
    "traffic_scaling.global_percent": "Global traffic scaling (%)",
}

_BLOCK_TITLES: dict[str, str] = {
    "drift": "Drift",
    "pc": "Causation",
    "traffic_scaling": "Traffic scaling",
    "consequence": "Consequence",
    "ship_categories": "Ship categories",
}


def _dig(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _flatten(value: Any, prefix: str, out: dict[str, Any]) -> None:
    """Flatten nested dicts / lists into ``{dotted.path: scalar}``."""
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _flatten(v, f"{prefix}[{i}]", out)
    else:
        out[prefix] = value


def _label_for(path: str) -> str:
    if path in _SETTING_LABELS:
        return _SETTING_LABELS[path]
    if path.startswith("drift.rose."):
        return f"Wind rose {path[len('drift.rose.'):]}°"
    if path.startswith("drift.blackout_by_ship_type."):
        return f"Blackout probability: {path[len('drift.blackout_by_ship_type.'):]}"
    block, _, rest = path.partition(".")
    title = _BLOCK_TITLES.get(block, block)
    return f"{title}: {rest}" if rest else title


def _fmt_value(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def build_settings_table(
    snap_a: dict[str, Any],
    snap_b: dict[str, Any],
) -> list[list[str]]:
    """Return rows ``[label, value_a, value_b]`` for settings that differ.

    Every scalar under the ``drift``, ``pc``, ``traffic_scaling``,
    ``consequence`` and ``ship_categories`` blocks is compared, plus the
    per-leg ``Width`` of ``segment_data``.  A setting present on only
    one side shows an em-dash on the other.  Equal values are hidden.
    """
    snap_a = snap_a if isinstance(snap_a, dict) else {}
    snap_b = snap_b if isinstance(snap_b, dict) else {}
    rows: list[list[str]] = []
    for block in _SETTINGS_BLOCKS:
        flat_a: dict[str, Any] = {}
        flat_b: dict[str, Any] = {}
        _flatten(snap_a.get(block), block, flat_a)
        _flatten(snap_b.get(block), block, flat_b)
        # ``_flatten`` of a missing block yields {block: None}; drop that.
        flat_a.pop(block, None)
        flat_b.pop(block, None)
        for path in sorted(set(flat_a) | set(flat_b), key=_path_sort_key):
            a = flat_a.get(path)
            b = flat_b.get(path)
            if _eq(a, b):
                continue
            rows.append([_label_for(path), _fmt_value(a), _fmt_value(b)])

    seg_a = snap_a.get("segment_data") or {}
    seg_b = snap_b.get("segment_data") or {}
    for k in sorted(set(seg_a) | set(seg_b), key=lambda k: int(k) if str(k).isdigit() else 10**9):
        wa = (seg_a.get(k) or {}).get("Width") if isinstance(seg_a.get(k), dict) else None
        wb = (seg_b.get(k) or {}).get("Width") if isinstance(seg_b.get(k), dict) else None
        if wa is None and wb is None:
            continue
        if _eq(wa, wb):
            continue
        rows.append([f"Leg {k} width (m)", _fmt_value(wa), _fmt_value(wb)])
    return rows


def _path_sort_key(path: str) -> tuple:
    """Sort numerically where a path segment is an integer (wind rose)."""
    parts = []
    for seg in path.replace("[", ".").replace("]", "").split("."):
        parts.append((0, int(seg), "") if seg.isdigit() else (1, 0, seg))
    return tuple(parts)


# ---------------------------------------------------------------------------
# Per-leg route distance
# ---------------------------------------------------------------------------

def _leg_length_m(seg: dict[str, Any]) -> float | None:
    """Pick a reasonable distance value out of a segment dict.

    Snapshots written by recent OMRAT versions cache ``line_length``
    (UTM metres).  For older snapshots fall back to a great-circle
    estimate from the start/end points.
    """
    if "line_length" in seg:
        try:
            return float(seg["line_length"] or 0.0)
        except (TypeError, ValueError):
            pass
    sp = _parse_xy(seg.get("Start_Point") or seg.get("Start Point"))
    ep = _parse_xy(seg.get("End_Point") or seg.get("End Point"))
    if sp is None or ep is None:
        return None
    return _haversine_m(sp, ep)


def _parse_xy(text: Any) -> tuple[float, float] | None:
    if not isinstance(text, str):
        return None
    parts = text.replace(",", " ").split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    R = 6_371_000.0  # earth radius in metres
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def build_leg_distance_table(
    snap_a: dict[str, Any],
    snap_b: dict[str, Any],
) -> list[list[str]]:
    """Return rows ``[leg_id, length_a_m, length_b_m, delta_m, delta_pct]``.

    Legs that exist on only one side show their length on that side and
    a blank cell on the other.
    """
    seg_a = (snap_a.get("segment_data") or {}) if isinstance(snap_a, dict) else {}
    seg_b = (snap_b.get("segment_data") or {}) if isinstance(snap_b, dict) else {}
    keys = sorted(
        set(seg_a) | set(seg_b),
        key=lambda k: int(k) if str(k).isdigit() else str(k),
    )
    rows: list[list[str]] = []
    for k in keys:
        la = _leg_length_m(seg_a.get(k, {})) if k in seg_a else None
        lb = _leg_length_m(seg_b.get(k, {})) if k in seg_b else None
        if la is None and lb is None:
            continue
        a_text = f"{la:.1f}" if la is not None else "—"
        b_text = f"{lb:.1f}" if lb is not None else "—"
        if la is None or lb is None:
            rows.append([str(k), a_text, b_text, "—", "—"])
            continue
        d = lb - la
        d_abs = f"{d:+.1f}"
        d_pct = f"{(d / la) * 100:+.2f}%" if la else "+inf%"
        rows.append([str(k), a_text, b_text, d_abs, d_pct])
    return rows
