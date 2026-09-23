"""One-at-a-time (OAT) sensitivity analysis of the OMRAT inputs.

QGIS-free.  The module answers "which input moves the result most?" by
perturbing one parameter at a time by ``+/- delta`` (a fraction, default
0.2) around the project's current values and ranking the parameters by
the swing they produce in a chosen output (an accident total or an
aggregate such as *All grounding*).

A full OMRAT run on a real project takes from half an hour to a few
hours, so re-running the whole model for every parameter is not an
option.  Two things keep the analysis tractable:

* **Analytical parameters** never re-run anything.  Causation factors
  are pure multipliers on their accident total (Category I / II split
  for powered accidents), a uniform traffic-volume change scales the
  drifting / powered totals linearly and the ship-ship totals
  quadratically, and a per-ship-type volume change is first-order exact
  from the per-cell breakdowns every model emits.  These are evaluated
  from the baseline reports alone.
* **Computed parameters** carry the tuple of model phases they can
  influence (``ParameterSpec.phases``).  Only those phases are re-run;
  the other totals are copied from the baseline.

The QGIS side (``omrat_utils/sensitivity_task.py`` and
``omrat_utils/sensitivity_dialog.py``) supplies the ``run_phases``
callable and renders the ranking; everything in here works on plain
dicts and is unit-tested standalone.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

__all__ = [
    'ACCIDENT_KEYS', 'AGGREGATE_KEYS', 'OUTPUT_KEYS', 'OUTPUT_LABELS',
    'PHASES', 'PHASE_LABELS', 'DEFAULT_DELTA',
    'ParameterSpec', 'BaselineReports', 'ParameterResult',
    'SensitivityResult', 'RunPlanItem',
    'build_parameter_specs', 'apply_perturbation', 'baseline_value',
    'analytic_totals', 'output_value', 'aggregate_totals', 'build_run_plan',
    'run_sensitivity', 'rank_results', 'result_to_markdown',
    'result_to_json_dict', 'result_from_json_dict', 'reports_from_calc',
]

# Row order of the accident table; mirrors
# ``omrat_utils.accident_summary.ACCIDENT_TOTAL_KEYS`` (a test pins it).
ACCIDENT_KEYS: tuple[str, ...] = (
    'drift_allision', 'drift_grounding',
    'powered_allision', 'powered_grounding',
    'overtaking', 'head_on', 'crossing', 'merging', 'bend',
)

AGGREGATES: dict[str, tuple[str, ...]] = {
    'all_grounding': ('drift_grounding', 'powered_grounding'),
    'all_allision': ('drift_allision', 'powered_allision'),
    'all_collisions': ('overtaking', 'head_on', 'crossing', 'merging', 'bend'),
    'total': ACCIDENT_KEYS,
}
AGGREGATE_KEYS: tuple[str, ...] = tuple(AGGREGATES)
OUTPUT_KEYS: tuple[str, ...] = ('total',) + tuple(
    k for k in AGGREGATE_KEYS if k != 'total') + ACCIDENT_KEYS

OUTPUT_LABELS: dict[str, str] = {
    'total': 'All accidents',
    'all_grounding': 'All grounding',
    'all_allision': 'All allision',
    'all_collisions': 'All collisions',
    'drift_allision': 'Drifting allision',
    'drift_grounding': 'Drifting grounding',
    'powered_allision': 'Powered allision',
    'powered_grounding': 'Powered grounding',
    'overtaking': 'Overtaking collision',
    'head_on': 'Head-on collision',
    'crossing': 'Crossing collision',
    'merging': 'Merging collision',
    'bend': 'Bend collision',
}

# Model phases, in the order ``CalculationTask._run_all_phases`` runs them.
PHASES: tuple[str, ...] = (
    'drifting', 'collision', 'powered_grounding', 'powered_allision',
)
PHASE_LABELS: dict[str, str] = {
    'drifting': 'Drifting',
    'collision': 'Ship-ship collisions',
    'powered_grounding': 'Powered grounding',
    'powered_allision': 'Powered allision',
}
# Which accident totals each phase produces.
PHASE_TOTALS: dict[str, tuple[str, ...]] = {
    'drifting': ('drift_allision', 'drift_grounding'),
    'collision': ('overtaking', 'head_on', 'crossing', 'merging', 'bend'),
    'powered_grounding': ('powered_grounding',),
    'powered_allision': ('powered_allision',),
}
ALL_PHASES: tuple[str, ...] = PHASES

DEFAULT_DELTA: float = 0.2

# Traffic matrix variable names (``traffic_data[seg][dir][variable]``).
_FREQ = 'Frequency (ships/year)'
_SPEED = 'Speed (knots)'
_DRAUGHT = 'Draught (meters)'
_HEIGHT = 'Ship heights (meters)'
_BEAM = 'Ship Beam (meters)'

# Causation factor -> (accident total, powered category or None).
_PC_TARGETS: dict[str, tuple[str, str | None]] = {
    'headon': ('head_on', None),
    'overtaking': ('overtaking', None),
    'crossing': ('crossing', None),
    'merging': ('merging', None),
    'bend': ('bend', None),
    'grounding': ('powered_grounding', 'cat2'),
    'grounding_cat1': ('powered_grounding', 'cat1'),
    'allision': ('powered_allision', 'cat2'),
    'allision_cat1': ('powered_allision', 'cat1'),
}
_PC_LABELS: dict[str, str] = {
    'headon': 'Causation factor: head-on',
    'overtaking': 'Causation factor: overtaking',
    'crossing': 'Causation factor: crossing',
    'merging': 'Causation factor: merging',
    'bend': 'Causation factor: bend',
    'grounding': 'Causation factor: powered grounding (Cat II)',
    'grounding_cat1': 'Causation factor: powered grounding (Cat I)',
    'allision': 'Causation factor: powered allision (Cat II)',
    'allision_cat1': 'Causation factor: powered allision (Cat I)',
}

GROUP_PC = 'Causation factors'
GROUP_TRAFFIC = 'Traffic'
GROUP_DRIFT = 'Drift settings'
GROUP_LEGS = 'Legs'


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParameterSpec:
    """One perturbable input.

    ``phases`` is empty for analytical parameters (evaluated from the
    baseline reports); otherwise it lists the model phases that must be
    re-run when the parameter changes.  ``kind`` selects the perturbation
    / analytic rule and ``target`` carries the rule's argument (a ``pc``
    key, a traffic variable name, a drift key path or a ship-type index).
    """
    key: str
    label: str
    group: str
    kind: str
    target: Any = None
    phases: tuple[str, ...] = ()
    clamp01: bool = False
    unit: str = ''

    @property
    def analytic(self) -> bool:
        return not self.phases


@dataclass
class BaselineReports:
    """The parts of a finished run the analysis needs.

    ``totals`` holds one value per ``ACCIDENT_KEYS``; ``powered_cat`` the
    Category I / II split per powered total; ``by_cell`` a
    ``{cell_key: value}`` map per accident key (``cell_key`` is
    ``"<ship_type_idx>_<length_idx>"``).
    """
    totals: dict[str, float] = field(default_factory=dict)
    powered_cat: dict[str, dict[str, float]] = field(default_factory=dict)
    by_cell: dict[str, dict[str, float]] = field(default_factory=dict)

    def total(self, key: str) -> float:
        return float(self.totals.get(key, 0.0) or 0.0)

    def type_share(self, key: str, ship_type_idx: int) -> float:
        """Sum of the per-cell contributions of one ship type."""
        prefix = f'{int(ship_type_idx)}_'
        cells = self.by_cell.get(key) or {}
        return float(sum(
            float(v or 0.0) for k, v in cells.items()
            if str(k).startswith(prefix)))

    def has_cells(self) -> bool:
        return any(self.by_cell.get(k) for k in ACCIDENT_KEYS)


@dataclass
class ParameterResult:
    spec: ParameterSpec
    base_value: float | None
    totals_minus: dict[str, float]
    totals_plus: dict[str, float]
    seconds: float = 0.0
    computed: bool = True  # False when the analytic rule was used


@dataclass
class SensitivityResult:
    delta: float
    baseline: BaselineReports
    results: list[ParameterResult] = field(default_factory=list)
    cancelled: bool = False
    baseline_seconds: float = 0.0
    started: str = ''
    project_name: str = ''

    def completed_keys(self) -> set[str]:
        return {r.spec.key for r in self.results}


@dataclass(frozen=True)
class RunPlanItem:
    spec: ParameterSpec
    factor: float
    phases: tuple[str, ...]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _iter_traffic_blocks(data: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for seg_block in (data.get('traffic_data') or {}).values():
        if not isinstance(seg_block, dict):
            continue
        for dir_block in seg_block.values():
            if isinstance(dir_block, dict):
                yield dir_block


def _iter_matrix_cells(matrix: Any) -> Iterable[tuple[int, int, Any]]:
    if not isinstance(matrix, list):
        return
    for i, row in enumerate(matrix):
        if not isinstance(row, list):
            continue
        for j, cell in enumerate(row):
            yield i, j, cell


def _scale_matrix(matrix: Any, factor: float, rows: set[int] | None = None) -> None:
    for i, j, cell in _iter_matrix_cells(matrix):
        if rows is not None and i not in rows:
            continue
        val = _f(cell, float('nan'))
        if math.isnan(val):
            continue
        matrix[i][j] = val * factor


def _matrix_row_sum(matrix: Any, row: int) -> float:
    total = 0.0
    for i, _j, cell in _iter_matrix_cells(matrix):
        if i == row:
            total += _f(cell)
    return total


def _traffic_mean(data: Mapping[str, Any], variable: str, weighted: bool = True) -> float | None:
    """Traffic-weighted mean of one traffic variable (``None`` if empty)."""
    num = 0.0
    den = 0.0
    for block in _iter_traffic_blocks(data):
        values = block.get(variable)
        freqs = block.get(_FREQ) if weighted else None
        for i, j, cell in _iter_matrix_cells(values):
            v = _f(cell, float('nan'))
            if math.isnan(v):
                continue
            w = 1.0
            if freqs is not None:
                try:
                    w = _f(freqs[i][j])
                except (IndexError, TypeError):
                    w = 0.0
                if w <= 0:
                    continue
            num += v * w
            den += w
    if den <= 0:
        return None
    return num / den


def _ship_type_names(data: Mapping[str, Any], n: int) -> list[str]:
    names: list[str] = []
    types = (data.get('ship_categories') or {}).get('types') if isinstance(
        data.get('ship_categories'), dict) else None
    if isinstance(types, list):
        for t in types:
            if isinstance(t, dict):
                names.append(str(t.get('name') or t.get('label') or ''))
            else:
                names.append(str(t))
    if len(names) < n:
        try:
            from compute.basic_equations import SHIP_TYPE_NAMES
        except Exception:  # pragma: no cover - defensive
            SHIP_TYPE_NAMES = {}
        for idx in range(len(names), n):
            names.append(str(SHIP_TYPE_NAMES.get(idx, f'Ship type {idx}')))
    return [nm or f'Ship type {i}' for i, nm in enumerate(names)]


def _n_ship_types(data: Mapping[str, Any]) -> int:
    n = 0
    for block in _iter_traffic_blocks(data):
        freq = block.get(_FREQ)
        if isinstance(freq, list):
            n = max(n, len(freq))
    return n


def _type_traffic(data: Mapping[str, Any]) -> dict[int, float]:
    """Total ships/year per ship-type row, summed over legs and directions."""
    out: dict[int, float] = {}
    for block in _iter_traffic_blocks(data):
        for i, _j, cell in _iter_matrix_cells(block.get(_FREQ)):
            out[i] = out.get(i, 0.0) + _f(cell)
    return out


def _leg_values(data: Mapping[str, Any], prefixes: tuple[str, ...]) -> list[float]:
    vals: list[float] = []
    for seg in (data.get('segment_data') or {}).values():
        if not isinstance(seg, dict):
            continue
        for k, v in seg.items():
            if k.startswith(prefixes):
                x = _f(v, float('nan'))
                if not math.isnan(x) and x > 0:
                    vals.append(x)
    return vals


def _get_path(d: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, Mapping) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _set_path(d: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur: Any = d
    for p in path[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[path[-1]] = value


# ---------------------------------------------------------------------------
# Parameter registry
# ---------------------------------------------------------------------------

def build_parameter_specs(data: Mapping[str, Any]) -> list[ParameterSpec]:
    """Every parameter the project exposes to the analysis.

    Parameters that would be a no-op on this project (a traffic variable
    with no traffic, a ship type with zero ships, legs without a lateral
    spread) are left out so the list only offers things that can move
    the result.
    """
    specs: list[ParameterSpec] = []

    # Causation factors -- analytical.
    # Always offered: compute falls back to the IWRAP defaults for keys
    # the project has not set, and ``baseline_value`` does the same.
    for pc_key, label in _PC_LABELS.items():
        specs.append(ParameterSpec(
            key=f'pc.{pc_key}', label=label, group=GROUP_PC,
            kind='pc', target=pc_key, phases=()))

    # Traffic.
    has_traffic = any(_type_traffic(data).values())
    if has_traffic:
        specs.append(ParameterSpec(
            key='traffic.frequency', label='Traffic volume (all ships)',
            group=GROUP_TRAFFIC, kind='traffic_all', target=_FREQ, phases=(),
            unit='ships/year'))
        n_types = _n_ship_types(data)
        names = _ship_type_names(data, n_types)
        per_type = _type_traffic(data)
        for idx in range(n_types):
            if per_type.get(idx, 0.0) <= 0:
                continue
            specs.append(ParameterSpec(
                key=f'traffic.type.{idx}',
                label=f'Traffic volume: {names[idx]}',
                group=GROUP_TRAFFIC, kind='traffic_type', target=idx,
                phases=(), unit='ships/year'))
        specs.append(ParameterSpec(
            key='traffic.speed', label='Ship speed (all cells)',
            group=GROUP_TRAFFIC, kind='traffic_var', target=_SPEED,
            phases=ALL_PHASES, unit='knots'))
        specs.append(ParameterSpec(
            key='traffic.draught', label='Ship draught (all cells)',
            group=GROUP_TRAFFIC, kind='traffic_var', target=_DRAUGHT,
            phases=('drifting', 'powered_grounding'), unit='m'))
        specs.append(ParameterSpec(
            key='traffic.height', label='Ship height (all cells)',
            group=GROUP_TRAFFIC, kind='traffic_var', target=_HEIGHT,
            phases=('powered_allision',), unit='m'))
        specs.append(ParameterSpec(
            key='traffic.beam', label='Ship beam (all cells)',
            group=GROUP_TRAFFIC, kind='traffic_var', target=_BEAM,
            phases=('collision',), unit='m'))

    # Drift settings -- drifting phase only.
    drift = data.get('drift') or {}
    specs.append(ParameterSpec(
        key='drift.blackout', label='Blackout frequency (all ship types)',
        group=GROUP_DRIFT, kind='blackout', target=None,
        phases=('drifting',), unit='events/ship-year'))
    specs.append(ParameterSpec(
        key='drift.anchor_p', label='Anchoring success probability',
        group=GROUP_DRIFT, kind='drift_path', target=('anchor_p',),
        phases=('drifting',), clamp01=True))
    specs.append(ParameterSpec(
        key='drift.anchor_d', label='Anchoring depth limit',
        group=GROUP_DRIFT, kind='drift_path', target=('anchor_d',),
        phases=('drifting',), unit='m'))
    specs.append(ParameterSpec(
        key='drift.speed', label='Drift speed',
        group=GROUP_DRIFT, kind='drift_path', target=('speed',),
        phases=('drifting',), unit='knots'))
    repair = drift.get('repair') if isinstance(drift.get('repair'), dict) else {}
    specs.append(ParameterSpec(
        key='drift.repair.scale', label='Repair time: scale (median)',
        group=GROUP_DRIFT, kind='drift_path', target=('repair', 'scale'),
        phases=('drifting',), unit='h'))
    specs.append(ParameterSpec(
        key='drift.repair.std', label='Repair time: spread (sigma)',
        group=GROUP_DRIFT, kind='drift_path', target=('repair', 'std'),
        phases=('drifting',)))
    if _f(repair.get('loc'), 0.0) > 0:
        specs.append(ParameterSpec(
            key='drift.repair.loc', label='Repair time: minimum (loc)',
            group=GROUP_DRIFT, kind='drift_path', target=('repair', 'loc'),
            phases=('drifting',), unit='h'))

    # Legs.
    if _leg_values(data, ('std1_', 'std2_')):
        specs.append(ParameterSpec(
            key='legs.lateral_std', label='Lateral spread of traffic (all legs)',
            group=GROUP_LEGS, kind='leg_prefix', target=('std1_', 'std2_'),
            phases=ALL_PHASES, unit='m'))
    if _leg_values(data, ('ai1', 'ai2')):
        specs.append(ParameterSpec(
            key='legs.ai', label='Position check interval (all legs)',
            group=GROUP_LEGS, kind='leg_prefix', target=('ai1', 'ai2'),
            phases=('powered_grounding', 'powered_allision'), unit='s'))
    return specs


# ---------------------------------------------------------------------------
# Baseline values + perturbation
# ---------------------------------------------------------------------------

def baseline_value(data: Mapping[str, Any], spec: ParameterSpec) -> float | None:
    """Representative current value (a mean for matrix / per-leg inputs)."""
    if spec.kind == 'pc':
        pc = data.get('pc') or {}
        if spec.target in pc:
            return _f(pc[spec.target], 0.0)
        try:
            from compute.iwrap_defaults import IWRAP_PC_DEFAULTS
            return _f(IWRAP_PC_DEFAULTS.get(spec.target), 0.0)
        except Exception:  # pragma: no cover - defensive
            return None
    if spec.kind == 'traffic_all':
        return sum(_type_traffic(data).values())
    if spec.kind == 'traffic_type':
        return _type_traffic(data).get(int(spec.target), 0.0)
    if spec.kind == 'traffic_var':
        return _traffic_mean(data, str(spec.target))
    if spec.kind == 'blackout':
        drift = data.get('drift') or {}
        return _f(drift.get('drift_p', 1.0), 1.0)
    if spec.kind == 'drift_path':
        val = _get_path(data.get('drift') or {}, tuple(spec.target))
        return None if val is None else _f(val, 0.0)
    if spec.kind == 'leg_prefix':
        vals = _leg_values(data, tuple(spec.target))
        return (sum(vals) / len(vals)) if vals else None
    return None


def apply_perturbation(data: Mapping[str, Any], spec: ParameterSpec, factor: float) -> dict[str, Any]:
    """Deep copy of ``data`` with ``spec`` multiplied by ``factor``."""
    out = copy.deepcopy(dict(data))
    if factor == 1.0:
        return out
    if spec.kind == 'pc':
        pc = out.setdefault('pc', {})
        base = baseline_value(data, spec) or 0.0
        pc[spec.target] = base * factor
    elif spec.kind == 'traffic_all':
        for block in _iter_traffic_blocks(out):
            _scale_matrix(block.get(_FREQ), factor)
    elif spec.kind == 'traffic_type':
        for block in _iter_traffic_blocks(out):
            _scale_matrix(block.get(_FREQ), factor, rows={int(spec.target)})
    elif spec.kind == 'traffic_var':
        for block in _iter_traffic_blocks(out):
            _scale_matrix(block.get(str(spec.target)), factor)
    elif spec.kind == 'blackout':
        drift = out.setdefault('drift', {})
        drift['drift_p'] = _f(drift.get('drift_p', 1.0), 1.0) * factor
        by_type = drift.get('blackout_by_ship_type')
        if isinstance(by_type, dict):
            for k, v in list(by_type.items()):
                by_type[k] = _f(v, 1.0) * factor
    elif spec.kind == 'drift_path':
        drift = out.setdefault('drift', {})
        path = tuple(spec.target)
        cur = _get_path(drift, path)
        new = _f(cur, 0.0) * factor
        if spec.clamp01:
            new = min(1.0, max(0.0, new))
        _set_path(drift, path, new)
    elif spec.kind == 'leg_prefix':
        prefixes = tuple(spec.target)
        for seg in (out.get('segment_data') or {}).values():
            if not isinstance(seg, dict):
                continue
            for k in list(seg.keys()):
                if k.startswith(prefixes):
                    x = _f(seg[k], float('nan'))
                    if not math.isnan(x) and x > 0:
                        seg[k] = x * factor
    else:  # pragma: no cover - unknown kinds are programming errors
        raise ValueError(f'Unknown parameter kind {spec.kind!r}')
    return out


def effective_factor(data: Mapping[str, Any], spec: ParameterSpec, factor: float) -> float:
    """The factor actually applied once clamping is taken into account."""
    if not spec.clamp01:
        return factor
    base = baseline_value(data, spec)
    if base is None or base <= 0:
        return factor
    new = min(1.0, max(0.0, base * factor))
    return new / base


# ---------------------------------------------------------------------------
# Analytical rules
# ---------------------------------------------------------------------------

def analytic_totals(baseline: BaselineReports, spec: ParameterSpec, factor: float) -> dict[str, float]:
    """Totals after ``spec`` is scaled by ``factor``, from the baseline alone."""
    totals = {k: baseline.total(k) for k in ACCIDENT_KEYS}
    if spec.kind == 'pc':
        acc_key, cat = _PC_TARGETS[str(spec.target)]
        cats = baseline.powered_cat.get(acc_key) or {}
        if cat is None or not cats:
            totals[acc_key] = totals[acc_key] * factor
        else:
            other = 'cat1' if cat == 'cat2' else 'cat2'
            totals[acc_key] = _f(cats.get(cat)) * factor + _f(cats.get(other))
        return totals
    if spec.kind == 'traffic_all':
        for k in ACCIDENT_KEYS:
            power = 2.0 if k in AGGREGATES['all_collisions'] else 1.0
            totals[k] = totals[k] * (factor ** power)
        return totals
    if spec.kind == 'traffic_type':
        idx = int(spec.target)
        for k in ACCIDENT_KEYS:
            elasticity = 2.0 if k in AGGREGATES['all_collisions'] else 1.0
            share = baseline.type_share(k, idx)
            totals[k] = max(0.0, totals[k] + (factor - 1.0) * elasticity * share)
        return totals
    raise ValueError(f'No analytic rule for parameter kind {spec.kind!r}')


# ---------------------------------------------------------------------------
# Outputs and aggregation
# ---------------------------------------------------------------------------

def aggregate_totals(totals: Mapping[str, Any]) -> dict[str, float]:
    """Accident totals plus the four aggregates."""
    out = {k: _f(totals.get(k)) for k in ACCIDENT_KEYS}
    for agg, keys in AGGREGATES.items():
        out[agg] = float(sum(out[k] for k in keys))
    return out


def output_value(totals: Mapping[str, Any], output_key: str) -> float:
    if output_key in AGGREGATES:
        return float(sum(_f(totals.get(k)) for k in AGGREGATES[output_key]))
    return _f(totals.get(output_key))


# ---------------------------------------------------------------------------
# Run plan + execution
# ---------------------------------------------------------------------------

def build_run_plan(specs: Iterable[ParameterSpec], delta: float) -> list[RunPlanItem]:
    """The model runs needed for the computed (non-analytic) parameters."""
    plan: list[RunPlanItem] = []
    for spec in specs:
        if spec.analytic:
            continue
        for factor in (1.0 - delta, 1.0 + delta):
            plan.append(RunPlanItem(spec=spec, factor=factor, phases=tuple(spec.phases)))
    return plan


def phase_counts(plan: Iterable[RunPlanItem]) -> dict[str, int]:
    counts = {p: 0 for p in PHASES}
    for item in plan:
        for p in item.phases:
            counts[p] = counts.get(p, 0) + 1
    return counts


RunPhases = Callable[[dict[str, Any], tuple[str, ...]], BaselineReports]
"""``run_phases(data, phases)`` runs the listed phases on ``data`` and
returns their reports (totals for other phases may be missing)."""


def run_sensitivity(
    data: Mapping[str, Any],
    specs: Iterable[ParameterSpec],
    run_phases: RunPhases,
    delta: float = DEFAULT_DELTA,
    baseline: BaselineReports | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    on_result: Callable[[ParameterResult], None] | None = None,
) -> SensitivityResult:
    """Evaluate every spec at ``1 - delta`` and ``1 + delta``.

    The baseline is computed first (all phases) unless one is passed in.
    Analytical specs are evaluated instantly from it; the others go
    through ``run_phases`` for their phases only.  Cancelling returns the
    results gathered so far with ``cancelled=True``.
    """
    specs = list(specs)
    cancelled = is_cancelled or (lambda: False)
    progress = on_progress or (lambda done, total, msg: None)
    result = SensitivityResult(
        delta=float(delta), baseline=baseline or BaselineReports(),
        started=time.strftime('%Y-%m-%d %H:%M:%S'),
        project_name=str(data.get('project_name') or ''),
    )
    plan = build_run_plan(specs, delta)
    total_steps = len(plan) + (0 if baseline is not None else 1)
    done = 0

    if baseline is None:
        progress(done, total_steps, 'Baseline run (all phases)')
        t0 = time.monotonic()
        result.baseline = run_phases(copy.deepcopy(dict(data)), ALL_PHASES)
        result.baseline_seconds = time.monotonic() - t0
        done += 1
        progress(done, total_steps, 'Baseline done')
        if cancelled():
            result.cancelled = True
            return result
    base = result.baseline

    # Analytical parameters first: instant, and useful even if the user
    # cancels the long runs later.
    for spec in specs:
        if not spec.analytic:
            continue
        pr = ParameterResult(
            spec=spec, base_value=baseline_value(data, spec),
            totals_minus=analytic_totals(base, spec, 1.0 - delta),
            totals_plus=analytic_totals(base, spec, 1.0 + delta),
            seconds=0.0, computed=False,
        )
        result.results.append(pr)
        if on_result is not None:
            on_result(pr)

    # Computed parameters: two runs each, only the affected phases.
    pending: dict[str, dict[str, Any]] = {}
    for item in plan:
        if cancelled():
            result.cancelled = True
            break
        sign = '-' if item.factor < 1.0 else '+'
        progress(done, total_steps,
                 f'{item.spec.label} ({sign}{delta * 100:.0f} %): '
                 + ', '.join(PHASE_LABELS[p] for p in item.phases))
        perturbed = apply_perturbation(data, item.spec, item.factor)
        t0 = time.monotonic()
        partial = run_phases(perturbed, item.phases)
        seconds = time.monotonic() - t0
        totals = {k: base.total(k) for k in ACCIDENT_KEYS}
        for p in item.phases:
            for k in PHASE_TOTALS[p]:
                totals[k] = partial.total(k)
        slot = pending.setdefault(item.spec.key, {'seconds': 0.0})
        slot['minus' if item.factor < 1.0 else 'plus'] = totals
        slot['seconds'] += seconds
        done += 1
        progress(done, total_steps, f'{item.spec.label} ({sign}{delta * 100:.0f} %) done')
        if 'minus' in slot and 'plus' in slot:
            pr = ParameterResult(
                spec=item.spec, base_value=baseline_value(data, item.spec),
                totals_minus=slot['minus'], totals_plus=slot['plus'],
                seconds=slot['seconds'], computed=True,
            )
            result.results.append(pr)
            if on_result is not None:
                on_result(pr)
    return result


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

@dataclass
class RankedRow:
    key: str
    label: str
    group: str
    base_value: float | None
    unit: str
    base_output: float
    output_minus: float
    output_plus: float
    swing: float          # output_plus - output_minus (signed)
    swing_pct: float | None
    elasticity: float | None
    computed: bool
    phases: tuple[str, ...]

    @property
    def abs_swing(self) -> float:
        return abs(self.swing)


def rank_results(result: SensitivityResult, output_key: str = 'total') -> list[RankedRow]:
    """Rows sorted by the absolute swing in ``output_key`` (largest first).

    ``elasticity`` is the central-difference estimate
    ``((y+ - y-) / y0) / (2 * delta)``: 1 means the output moves in
    proportion to the input, 2 quadratically, 0 not at all.
    """
    base_out = output_value(result.baseline.totals, output_key)
    rows: list[RankedRow] = []
    for pr in result.results:
        y_minus = output_value(pr.totals_minus, output_key)
        y_plus = output_value(pr.totals_plus, output_key)
        swing = y_plus - y_minus
        if base_out > 0:
            swing_pct: float | None = swing / base_out * 100.0
            elasticity: float | None = (swing / base_out) / (2.0 * result.delta) if result.delta > 0 else None
        else:
            swing_pct = None
            elasticity = None
        rows.append(RankedRow(
            key=pr.spec.key, label=pr.spec.label, group=pr.spec.group,
            base_value=pr.base_value, unit=pr.spec.unit,
            base_output=base_out, output_minus=y_minus, output_plus=y_plus,
            swing=swing, swing_pct=swing_pct, elasticity=elasticity,
            computed=pr.computed, phases=tuple(pr.spec.phases),
        ))
    rows.sort(key=lambda r: (-r.abs_swing, r.label))
    return rows


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _fmt(value: float | None, spec: str = '.3e') -> str:
    if value is None:
        return '-'
    if value == 0:
        return '0'
    return format(value, spec)


def result_to_markdown(result: SensitivityResult, output_key: str = 'total') -> str:
    """Human-readable report: one ranked table per output plus a summary."""
    lines: list[str] = []
    pct = result.delta * 100.0
    lines.append('# OMRAT sensitivity analysis')
    lines.append('')
    if result.project_name:
        lines.append(f'Project: {result.project_name}  ')
    lines.append(f'Started: {result.started}  ')
    lines.append(f'Perturbation: +/- {pct:.0f} % one-at-a-time around the current values  ')
    lines.append(f'Parameters evaluated: {len(result.results)}'
                 + (' (cancelled before completion)' if result.cancelled else ''))
    lines.append('')
    lines.append('## Baseline totals (events/year)')
    lines.append('')
    lines.append('| Output | Value |')
    lines.append('|---|---|')
    agg = aggregate_totals(result.baseline.totals)
    for k in OUTPUT_KEYS:
        lines.append(f'| {OUTPUT_LABELS[k]} | {_fmt(agg.get(k))} |')
    lines.append('')

    def _table(key: str) -> None:
        rows = rank_results(result, key)
        lines.append(f'## Ranking by {OUTPUT_LABELS[key]}')
        lines.append('')
        base_out = rows[0].base_output if rows else agg.get(key)
        lines.append(f'Baseline {OUTPUT_LABELS[key]}: {_fmt(base_out)} events/year')
        lines.append('')
        lines.append(f'| # | Parameter | Group | Current value | At -{pct:.0f} % | At +{pct:.0f} % '
                     '| Swing | Swing % | Elasticity | Method |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|')
        for i, r in enumerate(rows, start=1):
            unit = f' {r.unit}' if r.unit else ''
            cur = '-' if r.base_value is None else f'{r.base_value:.4g}{unit}'
            swing_pct = '-' if r.swing_pct is None else f'{r.swing_pct:+.1f} %'
            elas = '-' if r.elasticity is None else f'{r.elasticity:+.2f}'
            method = 'analytical' if not r.computed else 're-run: ' + ', '.join(
                PHASE_LABELS[p] for p in r.phases)
            lines.append(
                f'| {i} | {r.label} | {r.group} | {cur} | {_fmt(r.output_minus)} | {_fmt(r.output_plus)} '
                f'| {_fmt(r.swing, "+.3e")} | {swing_pct} | {elas} | {method} |')
        lines.append('')

    _table(output_key)
    for key in OUTPUT_KEYS:
        if key != output_key and output_value(agg, key) > 0:
            _table(key)
    lines.append('Elasticity = relative change of the output divided by the relative change of the input '
                 '(central difference).  1 = proportional, 2 = quadratic, 0 = no effect.')
    lines.append('')
    return '\n'.join(lines)


def result_to_json_dict(result: SensitivityResult) -> dict[str, Any]:
    return {
        'format': 'omrat-sensitivity/1',
        'delta': result.delta,
        'started': result.started,
        'project_name': result.project_name,
        'cancelled': result.cancelled,
        'baseline_seconds': result.baseline_seconds,
        'baseline': {
            'totals': dict(result.baseline.totals),
            'powered_cat': {k: dict(v) for k, v in result.baseline.powered_cat.items()},
        },
        'results': [
            {
                'key': pr.spec.key, 'label': pr.spec.label, 'group': pr.spec.group,
                'kind': pr.spec.kind, 'target': pr.spec.target,
                'phases': list(pr.spec.phases), 'unit': pr.spec.unit,
                'clamp01': pr.spec.clamp01,
                'base_value': pr.base_value,
                'totals_minus': dict(pr.totals_minus),
                'totals_plus': dict(pr.totals_plus),
                'seconds': pr.seconds, 'computed': pr.computed,
            }
            for pr in result.results
        ],
    }


def result_from_json_dict(d: Mapping[str, Any]) -> SensitivityResult:
    base_block = d.get('baseline') or {}
    baseline = BaselineReports(
        totals={k: _f(v) for k, v in (base_block.get('totals') or {}).items()},
        powered_cat={k: {kk: _f(vv) for kk, vv in (v or {}).items()}
                     for k, v in (base_block.get('powered_cat') or {}).items()},
    )
    result = SensitivityResult(
        delta=_f(d.get('delta'), DEFAULT_DELTA), baseline=baseline,
        cancelled=bool(d.get('cancelled')), baseline_seconds=_f(d.get('baseline_seconds')),
        started=str(d.get('started') or ''), project_name=str(d.get('project_name') or ''),
    )
    for item in d.get('results') or []:
        target = item.get('target')
        if isinstance(target, list):
            target = tuple(target)
        spec = ParameterSpec(
            key=str(item.get('key')), label=str(item.get('label')), group=str(item.get('group')),
            kind=str(item.get('kind')), target=target, phases=tuple(item.get('phases') or ()),
            clamp01=bool(item.get('clamp01')), unit=str(item.get('unit') or ''),
        )
        bv = item.get('base_value')
        result.results.append(ParameterResult(
            spec=spec, base_value=None if bv is None else _f(bv),
            totals_minus={k: _f(v) for k, v in (item.get('totals_minus') or {}).items()},
            totals_plus={k: _f(v) for k, v in (item.get('totals_plus') or {}).items()},
            seconds=_f(item.get('seconds')), computed=bool(item.get('computed', True)),
        ))
    return result


def dump_json(result: SensitivityResult, path: Any) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(result_to_json_dict(result), fh, indent=2)


# ---------------------------------------------------------------------------
# Extracting a baseline from a finished ``Calculation``
# ---------------------------------------------------------------------------

def reports_from_calc(calc: Any) -> BaselineReports:
    """Pull totals, Cat I/II splits and per-cell maps off a Calculation.

    Works on any object exposing the four ``*_report`` attributes; missing
    reports contribute zeros, so it also serves the partial (single-phase)
    runs of the analysis.
    """
    drift = getattr(calc, 'drifting_report', None) or {}
    coll = getattr(calc, 'collision_report', None) or {}
    pg = getattr(calc, 'powered_grounding_report', None) or {}
    pa = getattr(calc, 'powered_allision_report', None) or {}
    d_tot = drift.get('totals') or {}
    c_tot = coll.get('totals') or {}
    pg_tot = pg.get('totals') or {}
    pa_tot = pa.get('totals') or {}
    totals = {
        'drift_allision': _f(d_tot.get('allision')),
        'drift_grounding': _f(d_tot.get('grounding')),
        'powered_grounding': _f(pg_tot.get('grounding')),
        'powered_allision': _f(pa_tot.get('allision')),
        'head_on': _f(c_tot.get('head_on')),
        'overtaking': _f(c_tot.get('overtaking')),
        'crossing': _f(c_tot.get('crossing')),
        'merging': _f(c_tot.get('merging')),
        'bend': _f(c_tot.get('bend')),
    }
    powered_cat: dict[str, dict[str, float]] = {}
    for acc_key, tot in (('powered_grounding', pg_tot), ('powered_allision', pa_tot)):
        if 'cat1' in tot or 'cat2' in tot:
            powered_cat[acc_key] = {'cat1': _f(tot.get('cat1')), 'cat2': _f(tot.get('cat2'))}
    by_cell: dict[str, dict[str, float]] = {}
    for acc_key, cells in (
        ('drift_allision', drift.get('by_cell_allision')),
        ('drift_grounding', drift.get('by_cell_grounding')),
        ('powered_grounding', pg.get('by_cell')),
        ('powered_allision', pa.get('by_cell')),
    ):
        if isinstance(cells, dict):
            by_cell[acc_key] = {str(k): _f(v) for k, v in cells.items()}
    coll_cells = coll.get('by_cell') or {}
    if isinstance(coll_cells, dict):
        for acc_key in AGGREGATES['all_collisions']:
            cells = coll_cells.get(acc_key)
            if isinstance(cells, dict):
                by_cell[acc_key] = {str(k): _f(v) for k, v in cells.items()}
    return BaselineReports(totals=totals, powered_cat=powered_cat, by_cell=by_cell)
