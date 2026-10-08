"""Standalone tests for ``omrat_utils.compare_explain`` (Compare -> More info).

A small synthetic project (four legs meeting at one junction, one ship
cell per direction) keeps the real collision model fast enough to run
every coalition.  Run with ``--noconftest -p no:qgis``.
"""
from __future__ import annotations

import copy

import pytest

from geometries.junctions import apply_geometric_defaults, build_junctions, serialize_junctions
from omrat_utils import compare_explain as ce
from omrat_utils.compare_common import haversine_m

PTS = {
    'W': (14.0, 55.0), 'J': (14.5, 55.0), 'E': (15.0, 55.0),
    'N': (14.5, 55.4), 'NE': (15.0, 55.15), 'X': (16.0, 56.0),
}


def _xy(p: tuple[float, float]) -> str:
    return f'{p[0]:.6f} {p[1]:.6f}'


def _seg(seg_id: str, name: str, a: str, b: str, dirs=('East going', 'West going')) -> dict:
    return {
        'Segment_Id': seg_id, 'Leg_name': name, 'Route_Id': 1,
        'Start_Point': _xy(PTS[a]), 'End_Point': _xy(PTS[b]),
        'line_length': haversine_m(PTS[a], PTS[b]), 'Width': 5000, 'Dirs': list(dirs),
        'mean1_1': 0.0, 'std1_1': 500.0, 'weight1_1': 1.0,
        'mean2_1': 0.0, 'std2_1': 500.0, 'weight2_1': 1.0,
        'u_p1': 0, 'u_p2': 0, 'u_min1': 0, 'u_max1': 0, 'u_min2': 0, 'u_max2': 0,
        'ai1': 180.0, 'ai2': 180.0,
    }


def _dir(q: float, speed: float = 12.0) -> dict:
    return {
        'Frequency (ships/year)': [[q]], 'Speed (knots)': [[speed]],
        'Ship Beam (meters)': [[20.0]], 'Draught (meters)': [[6.0]],
        'Ship heights (meters)': [[20.0]], 'Scaling (%)': [[100.0]],
    }


def _project() -> dict:
    segs = {
        '1': _seg('1', 'LEG_1_1', 'W', 'J'),
        '2': _seg('2', 'LEG_1_2', 'J', 'E'),
        '3': _seg('3', 'LEG_2_1', 'J', 'N', dirs=('North going', 'South going')),
        '4': _seg('4', 'LEG_4_1', 'J', 'NE'),  # ~28 deg off LEG_1_2: merging
    }
    q = {'1': 2000.0, '2': 1500.0, '3': 500.0, '4': 800.0}
    traffic = {k: {d: _dir(q[k]) for d in v['Dirs']} for k, v in segs.items()}
    junctions = build_junctions(segs)
    apply_geometric_defaults(junctions, segs)
    return {
        'segment_data': segs, 'traffic_data': traffic,
        'junctions': serialize_junctions(junctions),
        'pc': {'headon': 4.9e-5, 'overtaking': 1.1e-4, 'crossing': 1.3e-4,
               'merging': 1.3e-4, 'bend': 1.3e-4, 'grounding': 1.6e-4},
        'ship_categories': {'types': ['Cargo'], 'length_intervals': [{'min': 100, 'max': 150}]},
        'depths': [], 'objects': [],
    }


def _assert_sums(expl: ce.CollisionExplanation) -> None:
    for t in ce.TYPES + (ce.TOTAL,):
        total = sum(expl.by_group[t].values())
        assert total == pytest.approx(expl.delta(t), rel=1e-9, abs=1e-18), t


# ---------------------------------------------------------------------------
# Classification and mixed inputs
# ---------------------------------------------------------------------------

def test_merging_limit_matches_model():
    from compute.ship_collision_model import ShipCollisionModelMixin
    assert ce.MERGING_LIMIT_DEG == ShipCollisionModelMixin.MERGING_ANGLE_DEG


def test_field_groups():
    assert ce.segment_field_group('End_Point') == 'geometry'
    assert ce.segment_field_group('line_length') == 'geometry'
    assert ce.segment_field_group('std1_2') == 'lateral'
    assert ce.segment_field_group('u_max2') == 'lateral'
    assert ce.segment_field_group('suppressed_with') == 'routing'
    assert ce.segment_field_group('Dirs') == 'traffic'
    assert ce.segment_field_group('ai1') is None
    assert ce.segment_field_group('Width') is None
    assert ce.segment_field_group('something_new') == 'other'
    assert ce.traffic_field_group('Speed (knots)') == 'ships'
    assert ce.traffic_field_group('Frequency (ships/year)') == 'traffic'
    assert ce.traffic_field_group('Scaling (%)') == 'traffic'


def test_hybrid_corners_are_a_and_b():
    a = _project()
    b = copy.deepcopy(a)
    b['segment_data']['1']['std1_1'] = 200.0                                   # lateral
    b['traffic_data']['2']['East going']['Frequency (ships/year)'] = [[3000.0]]  # traffic
    b['traffic_data']['3']['North going']['Speed (knots)'] = [[18.0]]          # ships
    b['segment_data']['4']['End_Point'] = _xy((15.0, 55.2))                    # geometry
    b['segment_data']['4']['line_length'] = 40000.0
    b['pc']['crossing'] = 2.0e-4                                               # causation
    b['segment_data']['5'] = _seg('5', 'LEG_5_1', 'E', 'X')                    # legs
    b['traffic_data']['5'] = {d: _dir(100.0) for d in b['segment_data']['5']['Dirs']}
    assert ce.hybrid_data(a, b, ()) == ce.model_data(a)
    assert ce.hybrid_data(a, b, ce.GROUPS) == ce.model_data(b)
    assert ce.active_groups(a, b) == ['geometry', 'lateral', 'traffic', 'ships', 'causation', 'legs']
    only_ships = ce.hybrid_data(a, b, ('ships',))
    assert only_ships['traffic_data']['3']['North going']['Speed (knots)'] == [[18.0]]
    assert only_ships['traffic_data']['2']['East going']['Frequency (ships/year)'] == [[1500.0]]
    assert '5' not in only_ships['segment_data']


def test_hybrid_swaps_whole_traffic_block_when_directions_differ():
    a = _project()
    b = copy.deepcopy(a)
    b['traffic_data']['1'] = {'Eastbound': _dir(10.0), 'Westbound': _dir(20.0)}
    assert set(ce.hybrid_data(a, b, ('ships',))['traffic_data']['1']) == {'East going', 'West going'}
    assert set(ce.hybrid_data(a, b, ('traffic',))['traffic_data']['1']) == {'Eastbound', 'Westbound'}


def test_no_effect_inputs_are_not_groups():
    a = _project()
    b = copy.deepcopy(a)
    b['segment_data']['1']['ai1'] = 0.0
    b['segment_data']['2']['Width'] = 9000
    b['objects'] = [['1', '10', 'POLYGON((0 0, 1 0, 1 1, 0 0))']]
    b['pc']['grounding'] = 9e-4
    assert ce.active_groups(a, b) == []
    lines = ce.group_differences(a, b)['_no_effect']
    assert any(line.startswith('ai1 (powered') and 'LEG_1_1' in line for line in lines)
    assert 'Structures differ' in lines
    assert 'Causation factors of the other accident types differ' in lines


# ---------------------------------------------------------------------------
# Shapley values
# ---------------------------------------------------------------------------

def _game(c: frozenset) -> float:
    v = 1.0 * ('x' in c) + 2.0 * ('y' in c) + 0.5 * ('z' in c)
    return v + 3.0 * ('x' in c and 'y' in c)       # x and y only matter together


def test_shapley_exact_shares_interaction():
    phi = ce.shapley_values(['x', 'y', 'z'], _game)
    assert sum(phi.values()) == pytest.approx(_game(frozenset('xyz')) - _game(frozenset()))
    assert phi['x'] == pytest.approx(1.0 + 1.5)
    assert phi['y'] == pytest.approx(2.0 + 1.5)
    assert phi['z'] == pytest.approx(0.5)


def test_shapley_over_orders_still_sums(monkeypatch):
    monkeypatch.setattr(ce, 'MAX_EXACT_GROUPS', 2)
    phi = ce.shapley_values(['x', 'y', 'z'], _game)
    assert sum(phi.values()) == pytest.approx(6.5)
    assert phi['z'] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Explanations with the real collision model
# ---------------------------------------------------------------------------

def test_explanation_sums_to_change_and_reproduces_both_runs():
    a = _project()
    b = copy.deepcopy(a)
    b['segment_data']['1']['std1_1'] = 200.0
    b['segment_data']['1']['std2_1'] = 200.0
    for d in b['traffic_data']['3'].values():
        d['Frequency (ships/year)'] = [[1500.0]]
    expl = ce.explain_collisions(a, b)
    assert expl.groups == ['lateral', 'traffic']
    assert expl.runs == 5 and expl.exact
    _assert_sums(expl)
    assert ce.RESIDUAL not in expl.columns()
    direct_a = ce.run_collision_model(ce.model_data(a))['totals']
    direct_b = ce.run_collision_model(ce.model_data(b))['totals']
    for t in ce.TYPES:
        assert expl.totals_a[t] == pytest.approx(float(direct_a[t]))
        assert expl.totals_b[t] == pytest.approx(float(direct_b[t]))
    leg1 = next(r for r in ce.items_for(expl, 'head_on') if r.label == 'LEG_1_1')
    assert leg1.delta > 0                       # narrower lane -> more head-on
    assert leg1.by_group['lateral'] == pytest.approx(leg1.delta)
    assert leg1.by_group['traffic'] == pytest.approx(0.0, abs=1e-18)
    for r in expl.items:
        assert sum(r.by_group.values()) == pytest.approx(r.delta, rel=1e-9, abs=1e-18)
    lateral = next(f for f in expl.findings if f.title.startswith('Lateral distributions differ'))
    assert any(d.startswith('LEG_1_1: sigma 500/500 -> 200/200 m') for d in lateral.details)


def test_suppressed_leg_moves_merging_to_crossing():
    a = _project()
    b = copy.deepcopy(a)
    b['segment_data']['4']['suppressed'] = True
    b['segment_data']['4']['traffic_redirect'] = [
        {'from_dir': 0, 'leg': '3', 'dir': 0, 'share': 100.0},
        {'from_dir': 1, 'leg': '3', 'dir': 1, 'share': 100.0},
    ]
    expl = ce.explain_collisions(a, b)
    assert expl.groups == ['routing']
    _assert_sums(expl)
    assert expl.delta('merging') < 0 < expl.delta('crossing')
    routing = next(f for f in expl.findings if f.title.startswith('Routing differs'))
    assert any('LEG_4_1 (1600 ships/year) is suppressed in B; its ships move to LEG_2_1 (1600)' in d
               for d in routing.details)
    reclass = [f for f in expl.findings if f.title.startswith('Crossing (')]
    assert reclass and 'through routing' in reclass[0].title
    gone = next(r for r in ce.items_for(expl, 'merging') if 'LEG_4_1' in r.label)
    assert gone.b == 0 and gone.angle_a is not None and gone.angle_a <= ce.MERGING_LIMIT_DEG


def test_leg_only_in_b():
    a = _project()
    b = copy.deepcopy(a)
    b['segment_data']['5'] = _seg('5', 'LEG_5_1', 'E', 'X')
    b['traffic_data']['5'] = {d: _dir(400.0) for d in b['segment_data']['5']['Dirs']}
    expl = ce.explain_collisions(a, b)
    assert expl.groups == ['legs']
    _assert_sums(expl)
    new = next(r for r in ce.items_for(expl, 'head_on') if r.label == 'LEG_5_1')
    assert new.a == 0 and new.b > 0 and new.by_group['legs'] == pytest.approx(new.b)
    assert any(f.title.startswith('Legs differ') for f in expl.findings)


def test_junction_losing_ais_counts_is_a_warning():
    a = _project()
    b = copy.deepcopy(a)
    jid = next(iter(a['junctions']))
    a['junctions'][jid]['source'] = 'ais'
    a['junctions'][jid]['transitions']['1'] = {'2': 0.2, '3': 0.0, '4': 0.8}
    expl = ce.explain_collisions(a, b)
    assert expl.groups == ['junctions']
    _assert_sums(expl)
    warn = next(f for f in expl.findings if 'lost their AIS counts' in f.title)
    assert warn.level == 'warning'
    assert any('source ais -> geometry' in d for d in warn.details)
    assert expl.findings[0].level == 'warning'


def test_ordered_attribution_when_many_groups(monkeypatch):
    monkeypatch.setattr(ce, 'MAX_EXACT_GROUPS', 1)
    a = _project()
    b = copy.deepcopy(a)
    b['segment_data']['1']['std1_1'] = 200.0
    b['pc']['headon'] = 1e-4
    expl = ce.explain_collisions(a, b)
    assert expl.groups == ['lateral', 'causation'] and not expl.exact
    _assert_sums(expl)


def test_stored_results_that_differ_give_a_warning():
    a = _project()
    b = copy.deepcopy(a)
    b['pc']['headon'] = 1e-4
    stored = {'collision': {'totals': {t: 1.0 for t in ce.TYPES}}}
    expl = ce.explain_collisions(a, b, outputs_a=stored)
    assert any(f.title == 'Run A recomputes differently from its stored result' for f in expl.findings)
    assert not any('Run B' in f.title for f in expl.findings)


def test_cancel_returns_none():
    a = _project()
    b = copy.deepcopy(a)
    b['pc']['headon'] = 1e-4
    assert ce.explain_collisions(a, b, is_cancelled=lambda: True) is None


def test_identical_runs():
    a = _project()
    expl = ce.explain_collisions(a, copy.deepcopy(a))
    assert expl.groups == [] and expl.runs == 2
    assert expl.delta(ce.TOTAL) == 0
    assert any(f.title.startswith('No input of the ship-ship collision model differs') for f in expl.findings)


# ---------------------------------------------------------------------------
# Items, findings filter, Markdown
# ---------------------------------------------------------------------------

def _row(label_legs, t, a, b, geo):
    legs = tuple((str(i), name) for i, name in label_legs)
    kind = 'leg' if len(legs) == 1 else 'pair'
    return ce.ItemRow(kind, t, ' x '.join(n for _, n in legs), legs, a, b, {'geometry': geo})


def test_route_name():
    assert ce.route_name('LEG_1_2_a') == 'LEG_1_2'
    assert ce.route_name('LEG_10_1_ab') == 'LEG_10_1'
    assert ce.route_name('LEG_8_1') == 'LEG_8_1'
    assert ce.route_name('Fairway') == 'Fairway'


def test_merge_split_legs_sums_sub_legs():
    rows = [
        _row([(2, 'LEG_1_2_a')], 'head_on', 9e-5, 7e-5, -2e-5),
        _row([(23, 'LEG_1_2_b')], 'head_on', 2e-4, 2.2e-4, 2e-5),
        _row([(5, 'LEG_4_1_a'), (20, 'LEG_4_1_b')], 'merging', 1e-5, 3e-5, 2e-5),
        _row([(1, 'LEG_1_1'), (2, 'LEG_1_2_a')], 'merging', 1e-4, 2e-4, 1e-4),
    ]
    merged = {r.label: r for r in ce.merge_split_legs(rows)}
    assert merged['LEG_1_2 (2 sub-legs)'].delta == pytest.approx(0.0, abs=1e-15)
    assert merged['LEG_1_2 (2 sub-legs)'].by_group['geometry'] == pytest.approx(0.0, abs=1e-15)
    assert len(merged['LEG_1_2 (2 sub-legs)'].legs) == 2
    assert 'LEG_4_1 (between its sub-legs)' in merged
    assert 'LEG_1_1 x LEG_1_2' in merged


def test_items_findings_and_markdown():
    a = _project()
    b = copy.deepcopy(a)
    b['segment_data']['1']['std1_1'] = 200.0
    b['pc']['crossing'] = 2.6e-4
    expl = ce.explain_collisions(a, b)
    assert {r.acc_type for r in ce.items_for(expl, ce.TOTAL)} >= {'head_on', 'crossing'}
    assert all(r.acc_type == 'crossing' for r in ce.items_for(expl, 'crossing'))
    crossing = ce.findings_for(expl, 'crossing')
    assert any(f.title.startswith('Collision causation factors differ') for f in crossing)
    assert not any(f.title.startswith('Lateral') for f in crossing)
    md = ce.to_markdown(expl, 'a.omrat', 'b.omrat')
    assert md.startswith('# Why did the collisions change?')
    assert '| Crossing collision |' in md and 'Lateral distributions' in md
    assert '## Head-on collision: largest items' in md
    assert ce.type_for_label('Merging collision') == 'merging'
    assert ce.type_for_label('All collisions') == ce.TOTAL
    assert ce.type_for_label('Drifting allision') == ce.TOTAL


def test_leg_endpoints_prefers_run_b():
    a = _project()
    b = copy.deepcopy(a)
    b['segment_data']['4']['End_Point'] = _xy((15.0, 55.2))
    pts = ce.leg_endpoints(a, b, [('4', 'LEG_4_1'), ('9', 'missing')])
    assert pts == [((14.5, 55.0), (15.0, 55.2))]
