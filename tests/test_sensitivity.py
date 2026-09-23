"""Standalone tests for ``compute/sensitivity.py`` (no QGIS needed).

Run with ``-p no:qgis --noconftest``.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute import sensitivity as sens  # noqa: E402
from compute.sensitivity import (  # noqa: E402
    ACCIDENT_KEYS, ALL_PHASES, BaselineReports, ParameterSpec,
    analytic_totals, apply_perturbation, baseline_value,
    build_parameter_specs, build_run_plan, effective_factor, output_value,
    phase_counts, rank_results, reports_from_calc, result_from_json_dict,
    result_to_json_dict, result_to_markdown, run_sensitivity,
)

EXAMPLE = ROOT / 'tests' / 'example_data' / 'proj.omrat'


@pytest.fixture
def project() -> dict:
    with open(EXAMPLE, encoding='utf-8') as fh:
        data = json.load(fh)
    # The example predates the per-type causation factors / blackout map;
    # add them so every perturbation kind has something to bite on.
    data['pc'].update({'headon': 4.9e-5, 'grounding': 1.6e-4, 'grounding_cat1': 1.6e-4})
    data['drift']['blackout_by_ship_type'] = {'0': 1.0, '8': 0.1}
    return data


def _spec(specs, key) -> ParameterSpec:
    return next(s for s in specs if s.key == key)


def _baseline(**overrides) -> BaselineReports:
    totals = {k: 0.0 for k in ACCIDENT_KEYS}
    totals.update({
        'drift_allision': 1e-3, 'drift_grounding': 2e-3,
        'powered_allision': 3e-3, 'powered_grounding': 4e-3,
        'overtaking': 1e-4, 'head_on': 2e-4, 'crossing': 3e-4,
        'merging': 4e-4, 'bend': 5e-4,
    })
    totals.update(overrides)
    return BaselineReports(
        totals=totals,
        powered_cat={'powered_grounding': {'cat1': 1e-3, 'cat2': 3e-3},
                     'powered_allision': {'cat1': 2e-3, 'cat2': 1e-3}},
        by_cell={
            'drift_grounding': {'0_1': 1.5e-3, '1_0': 0.5e-3},
            'head_on': {'0_1': 1e-4, '1_0': 1e-4},
        },
    )


# ---------------------------------------------------------------------------
# Vocabulary stays in step with the accident table
# ---------------------------------------------------------------------------

def test_accident_keys_match_accident_table():
    from omrat_utils.accident_summary import ACCIDENT_TOTAL_KEYS, SUMMARY_ROWS
    assert ACCIDENT_KEYS == ACCIDENT_TOTAL_KEYS
    by_label = {label: keys for label, keys in SUMMARY_ROWS}
    assert sens.AGGREGATES['all_grounding'] == by_label['All grounding']
    assert sens.AGGREGATES['all_allision'] == by_label['All allision']
    assert sens.AGGREGATES['all_collisions'] == by_label['All collisions']


def test_phase_totals_cover_every_accident_once():
    seen = [k for keys in sens.PHASE_TOTALS.values() for k in keys]
    assert sorted(seen) == sorted(ACCIDENT_KEYS)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_on_example_project(project):
    specs = build_parameter_specs(project)
    keys = {s.key for s in specs}
    for pc_key in ('headon', 'overtaking', 'crossing', 'merging', 'bend',
                   'grounding', 'grounding_cat1', 'allision', 'allision_cat1'):
        assert f'pc.{pc_key}' in keys
    assert 'traffic.frequency' in keys
    assert any(k.startswith('traffic.type.') for k in keys)
    for k in ('traffic.speed', 'traffic.draught', 'traffic.height', 'traffic.beam',
              'drift.blackout', 'drift.anchor_p', 'drift.anchor_d', 'drift.speed',
              'drift.repair.scale', 'drift.repair.std', 'drift.repair.loc',
              'legs.lateral_std', 'legs.ai'):
        assert k in keys, k
    assert len(keys) == len(specs), 'keys must be unique'
    # Analytical vs computed split.
    assert all(s.analytic for s in specs if s.kind in ('pc', 'traffic_all', 'traffic_type'))
    assert _spec(specs, 'traffic.speed').phases == ALL_PHASES
    assert _spec(specs, 'traffic.height').phases == ('powered_allision',)
    assert _spec(specs, 'traffic.beam').phases == ('collision',)
    assert set(_spec(specs, 'traffic.draught').phases) == {'drifting', 'powered_grounding'}
    assert _spec(specs, 'legs.ai').phases == ('powered_grounding', 'powered_allision')
    assert all(s.phases == ('drifting',) for s in specs if s.group == sens.GROUP_DRIFT)


def test_registry_skips_ship_types_without_traffic(project):
    specs = build_parameter_specs(project)
    per_type = sens._type_traffic(project)
    offered = {int(s.target) for s in specs if s.kind == 'traffic_type'}
    assert offered == {i for i, q in per_type.items() if q > 0}
    assert offered, 'example project has traffic'


def test_registry_without_traffic_or_legs():
    data = {'pc': {}, 'drift': {}, 'traffic_data': {}, 'segment_data': {}}
    specs = build_parameter_specs(data)
    keys = {s.key for s in specs}
    assert 'traffic.frequency' not in keys
    assert 'legs.ai' not in keys
    assert 'pc.headon' in keys
    assert 'drift.anchor_p' in keys


# ---------------------------------------------------------------------------
# Baseline values + perturbation
# ---------------------------------------------------------------------------

def test_baseline_values(project):
    specs = build_parameter_specs(project)
    assert baseline_value(project, _spec(specs, 'pc.headon')) == pytest.approx(4.9e-5)
    # Missing pc key falls back to the IWRAP default rather than None.
    assert baseline_value(project, _spec(specs, 'pc.overtaking')) == pytest.approx(1.1e-4)
    assert baseline_value(project, _spec(specs, 'drift.anchor_p')) == pytest.approx(0.95)
    assert baseline_value(project, _spec(specs, 'drift.repair.scale')) == pytest.approx(0.85)
    assert baseline_value(project, _spec(specs, 'legs.ai')) == pytest.approx(180.0)
    q_total = baseline_value(project, _spec(specs, 'traffic.frequency'))
    assert q_total > 0
    speed = baseline_value(project, _spec(specs, 'traffic.speed'))
    assert speed is not None and speed > 0


def test_perturbation_is_a_deep_copy(project):
    specs = build_parameter_specs(project)
    before = copy.deepcopy(project)
    for spec in specs:
        apply_perturbation(project, spec, 1.2)
    assert project == before


def test_perturb_pc_and_traffic(project):
    specs = build_parameter_specs(project)
    out = apply_perturbation(project, _spec(specs, 'pc.headon'), 1.5)
    assert out['pc']['headon'] == pytest.approx(4.9e-5 * 1.5)
    # Missing key is created from the default.
    out = apply_perturbation(project, _spec(specs, 'pc.overtaking'), 2.0)
    assert out['pc']['overtaking'] == pytest.approx(2.2e-4)

    out = apply_perturbation(project, _spec(specs, 'traffic.frequency'), 1.3)
    assert sens._type_traffic(out) == pytest.approx(
        {k: v * 1.3 for k, v in sens._type_traffic(project).items()})

    idx = int(_spec(specs, next(s.key for s in specs if s.kind == 'traffic_type')).target)
    out = apply_perturbation(project, _spec(specs, f'traffic.type.{idx}'), 0.5)
    before = sens._type_traffic(project)
    after = sens._type_traffic(out)
    assert after[idx] == pytest.approx(before[idx] * 0.5)
    for other in before:
        if other != idx:
            assert after[other] == pytest.approx(before[other])

    out = apply_perturbation(project, _spec(specs, 'traffic.speed'), 1.1)
    assert baseline_value(out, _spec(specs, 'traffic.speed')) == pytest.approx(
        1.1 * baseline_value(project, _spec(specs, 'traffic.speed')))


def test_perturb_drift_and_legs(project):
    specs = build_parameter_specs(project)
    out = apply_perturbation(project, _spec(specs, 'drift.blackout'), 2.0)
    assert out['drift']['drift_p'] == pytest.approx(2.0)
    assert out['drift']['blackout_by_ship_type'] == pytest.approx({'0': 2.0, '8': 0.2})

    out = apply_perturbation(project, _spec(specs, 'drift.repair.scale'), 1.2)
    assert out['drift']['repair']['scale'] == pytest.approx(0.85 * 1.2)
    assert out['drift']['repair']['std'] == pytest.approx(0.95)

    # anchor_p is a probability: clamped at 1.
    spec_ap = _spec(specs, 'drift.anchor_p')
    out = apply_perturbation(project, spec_ap, 1.2)
    assert out['drift']['anchor_p'] == 1.0
    assert effective_factor(project, spec_ap, 1.2) == pytest.approx(1.0 / 0.95)
    assert effective_factor(project, spec_ap, 0.8) == pytest.approx(0.8)

    out = apply_perturbation(project, _spec(specs, 'legs.lateral_std'), 1.5)
    seg_in = project['segment_data']['1']
    seg_out = out['segment_data']['1']
    assert seg_out['std1_1'] == pytest.approx(seg_in['std1_1'] * 1.5)
    assert seg_out['std2_1'] == pytest.approx(seg_in['std2_1'] * 1.5)
    assert seg_out['std1_2'] == 0  # zero (unused) spreads stay zero
    assert seg_out['mean1_1'] == seg_in['mean1_1']

    out = apply_perturbation(project, _spec(specs, 'legs.ai'), 0.5)
    assert out['segment_data']['1']['ai1'] == pytest.approx(90.0)
    assert out['segment_data']['1']['ai2'] == pytest.approx(90.0)


def test_perturb_ai_zero_stays_zero(project):
    """``ai <= 0`` disables Cat II for that leg; the analysis must not re-enable it."""
    project['segment_data']['1']['ai2'] = 0
    specs = build_parameter_specs(project)
    out = apply_perturbation(project, _spec(specs, 'legs.ai'), 1.2)
    assert out['segment_data']['1']['ai2'] == 0
    assert out['segment_data']['1']['ai1'] == pytest.approx(216.0)


# ---------------------------------------------------------------------------
# Analytical rules
# ---------------------------------------------------------------------------

def test_analytic_pc_plain_multiplier():
    base = _baseline()
    spec = ParameterSpec('pc.headon', 'x', 'pc', 'pc', target='headon')
    out = analytic_totals(base, spec, 1.2)
    assert out['head_on'] == pytest.approx(2e-4 * 1.2)
    for k in ACCIDENT_KEYS:
        if k != 'head_on':
            assert out[k] == base.totals[k]


def test_analytic_pc_uses_powered_category_split():
    base = _baseline()
    cat2 = ParameterSpec('pc.grounding', 'x', 'pc', 'pc', target='grounding')
    cat1 = ParameterSpec('pc.grounding_cat1', 'x', 'pc', 'pc', target='grounding_cat1')
    assert analytic_totals(base, cat2, 1.5)['powered_grounding'] == pytest.approx(1e-3 + 3e-3 * 1.5)
    assert analytic_totals(base, cat1, 1.5)['powered_grounding'] == pytest.approx(1e-3 * 1.5 + 3e-3)
    # Without a split the whole total scales.
    base.powered_cat = {}
    assert analytic_totals(base, cat2, 1.5)['powered_grounding'] == pytest.approx(4e-3 * 1.5)


def test_analytic_traffic_all_linear_and_quadratic():
    base = _baseline()
    spec = ParameterSpec('traffic.frequency', 'x', 'traffic', 'traffic_all')
    out = analytic_totals(base, spec, 1.2)
    for k in ('drift_allision', 'drift_grounding', 'powered_allision', 'powered_grounding'):
        assert out[k] == pytest.approx(base.totals[k] * 1.2)
    for k in ('overtaking', 'head_on', 'crossing', 'merging', 'bend'):
        assert out[k] == pytest.approx(base.totals[k] * 1.44)


def test_analytic_traffic_type_uses_cells():
    base = _baseline()
    spec = ParameterSpec('traffic.type.0', 'x', 'traffic', 'traffic_type', target=0)
    out = analytic_totals(base, spec, 1.2)
    # Type 0 holds 1.5e-3 of the 2e-3 drifting grounding: elasticity 1.
    assert out['drift_grounding'] == pytest.approx(2e-3 + 0.2 * 1.5e-3)
    # Type 0 holds half the head-on cells: elasticity 2 on that share.
    assert out['head_on'] == pytest.approx(2e-4 + 0.2 * 2 * 1e-4)
    # No cells -> unchanged.
    assert out['crossing'] == base.totals['crossing']
    out = analytic_totals(base, spec, 0.8)
    assert out['drift_grounding'] == pytest.approx(2e-3 - 0.2 * 1.5e-3)


def test_analytic_rule_missing_raises():
    with pytest.raises(ValueError):
        analytic_totals(_baseline(), ParameterSpec('x', 'x', 'g', 'drift_path', target=('speed',),
                                                   phases=('drifting',)), 1.2)


# ---------------------------------------------------------------------------
# Run plan + execution
# ---------------------------------------------------------------------------

def test_run_plan_only_for_computed_specs(project):
    specs = build_parameter_specs(project)
    plan = build_run_plan(specs, 0.2)
    computed = [s for s in specs if not s.analytic]
    assert len(plan) == 2 * len(computed)
    factors = {(i.spec.key, round(i.factor, 6)) for i in plan}
    for s in computed:
        assert (s.key, 0.8) in factors and (s.key, 1.2) in factors
    counts = phase_counts(plan)
    assert counts['powered_allision'] >= 2  # traffic.height at least
    assert counts['drifting'] >= 2 * sum(1 for s in specs if s.group == sens.GROUP_DRIFT)


class _FakeModel:
    """Toy model: totals are simple functions of the inputs, phases recorded."""

    def __init__(self):
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, data, phases):
        self.calls.append(tuple(phases))
        drift = data['drift']
        seg = data['segment_data']['1']
        speed = sens._traffic_mean(data, 'Speed (knots)') or 1.0
        totals = {}
        if 'drifting' in phases:
            totals['drift_grounding'] = drift['anchor_d'] * 1e-3 / speed
            totals['drift_allision'] = drift['repair']['scale'] ** 2 * 1e-3
        if 'collision' in phases:
            totals['head_on'] = 1e-4 * speed
        if 'powered_grounding' in phases:
            totals['powered_grounding'] = 1e-3 * float(seg['ai1']) / 180.0
        if 'powered_allision' in phases:
            totals['powered_allision'] = 5e-4
        return BaselineReports(totals=totals)


def test_run_sensitivity_reruns_only_affected_phases(project):
    specs = build_parameter_specs(project)
    chosen = [_spec(specs, k) for k in ('pc.headon', 'drift.anchor_d', 'legs.ai', 'traffic.frequency')]
    model = _FakeModel()
    progress: list[tuple[int, int, str]] = []
    done: list[str] = []
    result = run_sensitivity(
        project, chosen, model, delta=0.2,
        on_progress=lambda d, t, m: progress.append((d, t, m)),
        on_result=lambda pr: done.append(pr.spec.key),
    )
    assert not result.cancelled
    # Baseline (all phases) + 2 runs for anchor_d (drifting) + 2 for ai (both powered).
    assert model.calls[0] == ALL_PHASES
    assert sorted(model.calls[1:]) == sorted([
        ('drifting',), ('drifting',),
        ('powered_grounding', 'powered_allision'), ('powered_grounding', 'powered_allision'),
    ])
    assert set(done) == {s.key for s in chosen}
    # Analytical ones are reported before the long runs.
    assert done[:2] == ['pc.headon', 'traffic.frequency'] or set(done[:2]) == {'pc.headon', 'traffic.frequency'}
    assert progress[-1][0] == progress[-1][1] == 5
    assert result.baseline_seconds >= 0

    by_key = {r.spec.key: r for r in result.results}
    base = result.baseline.totals
    # anchor_d: drifting totals re-run, everything else copied from baseline.
    r = by_key['drift.anchor_d']
    assert r.computed
    assert r.totals_plus['drift_grounding'] == pytest.approx(base['drift_grounding'] * 1.2)
    assert r.totals_minus['drift_grounding'] == pytest.approx(base['drift_grounding'] * 0.8)
    assert r.totals_plus['head_on'] == base['head_on']
    assert r.totals_plus['powered_grounding'] == base['powered_grounding']
    # ai: only the powered totals change.
    r = by_key['legs.ai']
    assert r.totals_plus['powered_grounding'] == pytest.approx(base['powered_grounding'] * 1.2)
    assert r.totals_plus['drift_grounding'] == base['drift_grounding']
    # Analytical head-on causation factor.
    r = by_key['pc.headon']
    assert not r.computed and r.seconds == 0
    assert r.totals_plus['head_on'] == pytest.approx(base['head_on'] * 1.2)


def test_run_sensitivity_with_given_baseline_skips_baseline_run(project):
    specs = build_parameter_specs(project)
    model = _FakeModel()
    baseline = model(copy.deepcopy(project), ALL_PHASES)
    model.calls.clear()
    result = run_sensitivity(project, [_spec(specs, 'drift.speed')], model, delta=0.1, baseline=baseline)
    assert model.calls == [('drifting',), ('drifting',)]
    assert result.baseline is baseline
    assert result.baseline_seconds == 0


def test_run_sensitivity_cancel_keeps_partial(project):
    specs = build_parameter_specs(project)
    chosen = [_spec(specs, k) for k in ('pc.headon', 'drift.anchor_d', 'legs.ai')]
    model = _FakeModel()
    state = {'n': 0}

    def cancelled() -> bool:
        state['n'] += 1
        return state['n'] > 2  # allow the baseline and the first perturbed run

    result = run_sensitivity(project, chosen, model, delta=0.2, is_cancelled=cancelled)
    assert result.cancelled
    keys = result.completed_keys()
    assert 'pc.headon' in keys           # analytical, always finished
    assert 'legs.ai' not in keys         # never reached
    assert len(model.calls) < 5


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _result_with(base: BaselineReports, specs):
    """Result with every spec evaluated at +/- 20 %.

    Analytical specs use their rule; computed ones get a toy response
    (drifting totals scale with the factor) so ranking has something to
    compare.
    """
    res = sens.SensitivityResult(delta=0.2, baseline=base)
    for spec in specs:
        if spec.analytic:
            minus, plus = analytic_totals(base, spec, 0.8), analytic_totals(base, spec, 1.2)
        else:
            minus = {k: base.total(k) * (0.8 if k.startswith('drift') else 1.0) for k in ACCIDENT_KEYS}
            plus = {k: base.total(k) * (1.2 if k.startswith('drift') else 1.0) for k in ACCIDENT_KEYS}
        res.results.append(sens.ParameterResult(
            spec=spec, base_value=1.0, totals_minus=minus, totals_plus=plus,
            computed=not spec.analytic))
    return res


def test_rank_results_orders_by_swing_and_reports_elasticity():
    base = _baseline()
    pc_bend = ParameterSpec('pc.bend', 'bend cf', 'pc', 'pc', target='bend')
    pc_headon = ParameterSpec('pc.headon', 'headon cf', 'pc', 'pc', target='headon')
    traffic = ParameterSpec('traffic.frequency', 'traffic', 'traffic', 'traffic_all')
    res = _result_with(base, [pc_headon, pc_bend, traffic])

    rows = rank_results(res, 'total')
    assert [r.key for r in rows] == ['traffic.frequency', 'pc.bend', 'pc.headon']
    total = output_value(base.totals, 'total')
    bend_row = rows[1]
    assert bend_row.swing == pytest.approx(0.4 * 5e-4)
    assert bend_row.elasticity == pytest.approx(5e-4 / total)
    assert bend_row.swing_pct == pytest.approx(0.4 * 5e-4 / total * 100)

    # Ranked by its own accident type a causation factor has elasticity 1;
    # traffic volume (quadratic on collisions) outranks it there.
    rows = rank_results(res, 'bend')
    assert [r.key for r in rows] == ['traffic.frequency', 'pc.bend', 'pc.headon']
    by_key = {r.key: r for r in rows}
    assert by_key['pc.bend'].elasticity == pytest.approx(1.0)
    assert by_key['traffic.frequency'].elasticity == pytest.approx(2.0)
    assert by_key['pc.headon'].swing == 0

    # Traffic volume: elasticity 2 on collisions, 1 on drifting.
    rows = {r.key: r for r in rank_results(res, 'all_collisions')}
    assert rows['traffic.frequency'].elasticity == pytest.approx(2.0)
    rows = {r.key: r for r in rank_results(res, 'all_grounding')}
    assert rows['traffic.frequency'].elasticity == pytest.approx(1.0)


def test_rank_results_zero_baseline_has_no_relative_numbers():
    base = _baseline()
    base.totals = {k: 0.0 for k in ACCIDENT_KEYS}
    base.powered_cat = {}
    res = _result_with(base, [ParameterSpec('pc.bend', 'bend cf', 'pc', 'pc', target='bend')])
    row = rank_results(res, 'total')[0]
    assert row.swing == 0 and row.swing_pct is None and row.elasticity is None


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_markdown_and_json_round_trip(tmp_path):
    base = _baseline()
    specs = [
        ParameterSpec('pc.bend', 'Causation factor: bend', 'Causation factors', 'pc', target='bend'),
        ParameterSpec('drift.speed', 'Drift speed', 'Drift settings', 'drift_path',
                      target=('speed',), phases=('drifting',), unit='knots'),
    ]
    res = _result_with(base, specs)
    assert res.results[1].computed
    res.results[1].seconds = 12.5
    res.project_name = 'demo'

    md = result_to_markdown(res, 'total')
    assert '# OMRAT sensitivity analysis' in md
    assert 'Ranking by All accidents' in md
    assert 'Causation factor: bend' in md
    assert 're-run: Drifting' in md
    assert 'analytical' in md

    d = result_to_json_dict(res)
    text = json.dumps(d)  # must be JSON serialisable (tuples -> lists)
    back = result_from_json_dict(json.loads(text))
    assert back.delta == res.delta
    assert back.project_name == 'demo'
    assert back.baseline.totals == pytest.approx(res.baseline.totals)
    assert [r.spec.key for r in back.results] == ['pc.bend', 'drift.speed']
    assert back.results[1].spec.target == ('speed',)
    assert back.results[1].spec.phases == ('drifting',)
    assert back.results[1].seconds == 12.5
    assert rank_results(back, 'total')[0].swing == pytest.approx(rank_results(res, 'total')[0].swing)

    path = tmp_path / 'r.json'
    sens.dump_json(res, path)
    assert json.loads(path.read_text(encoding='utf-8'))['format'] == 'omrat-sensitivity/1'


# ---------------------------------------------------------------------------
# Extracting reports from a calculation object
# ---------------------------------------------------------------------------

class _FakeCalc:
    drifting_report = {
        'totals': {'allision': 1e-3, 'grounding': 2e-3},
        'by_cell_allision': {'0_0': 1e-3}, 'by_cell_grounding': {'0_0': 2e-3},
    }
    collision_report = {
        'totals': {'head_on': 1e-4, 'overtaking': 2e-4, 'crossing': 3e-4, 'merging': 4e-4, 'bend': 5e-4},
        'by_cell': {'head_on': {'1_2': 1e-4}, 'bend': {'1_2': 5e-4}},
    }
    powered_grounding_report = {'totals': {'grounding': 4e-3, 'cat1': 1e-3, 'cat2': 3e-3}, 'by_cell': {'2_2': 4e-3}}
    powered_allision_report = None


def test_reports_from_calc():
    base = reports_from_calc(_FakeCalc())
    assert base.totals['drift_grounding'] == 2e-3
    assert base.totals['bend'] == 5e-4
    assert base.totals['powered_allision'] == 0.0
    assert base.powered_cat == {'powered_grounding': {'cat1': 1e-3, 'cat2': 3e-3}}
    assert base.type_share('head_on', 1) == 1e-4
    assert base.type_share('head_on', 0) == 0.0
    assert base.type_share('powered_grounding', 2) == 4e-3
    assert base.has_cells()
    # Missing reports -> zeros, so partial (single-phase) runs work too.
    empty = reports_from_calc(object())
    assert all(v == 0.0 for v in empty.totals.values())
    assert not empty.has_cells()
