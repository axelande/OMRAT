"""Drift cascade: what shields a structure (KNOWN_ISSUES 11, 12, 13).

One E-W leg (11.0-11.1 E at 56.0 N), a structure 3.3 km north of it and
drift towards north only; draught 10 m, so a ship anchors below 70 m and
grounds below 10 m.  The structure's allision is compared with and without
water / shoals / structures around it.  Needs the QGIS conftest (the run
builds result layers).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from shapely.geometry import box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent

STRUCTURE = box(11.045, 56.030, 11.055, 56.036)
ARM = box(11.085, 55.95, 11.11, 56.02)               # beside the structure's path, crosses the leg
BEHIND = box(10.98, 55.95, 11.12, 55.99)             # south of the leg: behind it for northward drift
FRONT = box(11.04, 56.010, 11.06, 56.020)            # between the leg and the structure, whole strip
FRONT_HALF = box(11.045, 56.010, 11.050, 56.014)     # same, left half of the structure's strip


def _mock_parent():
    mp = MagicMock()
    mp.main_widget = MagicMock()
    return mp


@pytest.fixture(scope='module')
def template():
    return json.loads((ROOT / 'tests' / 'example_data' / 'proj.omrat').read_text(encoding='utf-8'))


def _allision_of_structure_1(template, objects, depths):
    return _run(template, objects, depths)[0]


def _run(template, objects, depths, anchor_p=0.7):
    """``(allision of structure 1, total grounding)``."""
    from compute.basic_equations import default_blackout_by_ship_type
    from compute.run_calculations import Calculation
    k = list(template['segment_data'])[0]
    seg = copy.deepcopy(template['segment_data'][k])
    seg.update({'Start_Point': '11.000000 56.000000', 'End_Point': '11.100000 56.000000', 'Width': '5000',
                'mean1_1': 0.0, 'std1_1': 500.0, 'mean2_1': 0.0, 'std2_1': 500.0,
                'weight1_1': 100, 'weight2_1': 100})
    td = copy.deepcopy(template['traffic_data'][k])
    for blk in td.values():
        n, m = len(blk['Frequency (ships/year)']), len(blk['Frequency (ships/year)'][0])
        blk['Frequency (ships/year)'] = [[1000.0 if (i, j) == (5, 5) else 0.0 for j in range(m)] for i in range(n)]
        blk['Draught (meters)'] = [[10.0] * m for _ in range(n)]
        blk['Speed (knots)'] = [[12.0] * m for _ in range(n)]
    data = copy.deepcopy(template)
    data['segment_data'] = {k: seg}
    data['traffic_data'] = {k: td}
    data['junctions'] = {}
    data['objects'] = [[str(i + 1), '10', g.wkt] for i, g in enumerate(objects)]
    far_deep = box(12.0, 57.0, 12.01, 57.01)          # keeps the depth layer non-empty, out of reach
    data['depths'] = [['1', '30', far_deep.wkt]] + [[str(i + 2), str(v), g.wkt] for i, (v, g) in enumerate(depths)]
    data['drift'].update({
        'speed': 1.0, 'anchor_p': anchor_p, 'anchor_d': 7.0,
        'rose': {str(a): (1.0 if a == 0 else 0.0) for a in range(0, 360, 45)},
        'repair': {'func': '', 'std': 0.97, 'loc': 0.2, 'scale': 0.5, 'use_lognormal': True},
        'blackout_by_ship_type': default_blackout_by_ship_type(),
    })
    calc = Calculation(_mock_parent())
    calc.set_progress_callback(lambda completed, total, msg: True)
    _, grounding = calc.run_drifting_model(data)
    return calc.drifting_report['by_object'].get('Structure - 1', {}).get('allision', 0.0), grounding


@pytest.fixture(scope='module')
def alone(template):
    value = _allision_of_structure_1(template, [STRUCTURE], [])
    assert value > 0
    return value


def test_water_behind_the_leg_does_not_anchor_ships(template, alone):
    """KNOWN_ISSUES 11: the merged polygon also reaches into the corridor (the arm),
    but its part behind the leg must not shield the structure."""
    value = _allision_of_structure_1(template, [STRUCTURE], [(30, unary_union([ARM, BEHIND]))])
    assert value == pytest.approx(alone, rel=1e-6)


def test_water_in_front_anchors_anchor_p_of_the_ships(template, alone):
    value = _allision_of_structure_1(template, [STRUCTURE], [(30, unary_union([ARM, FRONT]))])
    assert value == pytest.approx(0.3 * alone, rel=1e-3)


def test_shoal_across_the_path_shields_the_structure(template):
    """KNOWN_ISSUES 12: ships that ground on the shoal cannot also allide."""
    assert _allision_of_structure_1(template, [STRUCTURE], [(5, FRONT)]) == 0.0


def test_shoal_over_half_the_path_halves_the_allision(template, alone):
    """KNOWN_ISSUES 12 + 13: half the strip shielded -> half the allision, not a quarter."""
    value = _allision_of_structure_1(template, [STRUCTURE], [(5, FRONT_HALF)])
    assert value == pytest.approx(0.5 * alone, rel=0.02)


def test_structure_over_half_the_path_halves_the_allision(template, alone):
    """KNOWN_ISSUES 13: the polygon-level shadow and the edge re-carve counted it twice (-75 %)."""
    near = box(11.045, 56.018, 11.050, 56.022)
    value = _allision_of_structure_1(template, [STRUCTURE, near], [])
    assert value == pytest.approx(0.5 * alone, rel=0.02)


def test_structure_over_the_whole_path_shields_the_structure(template):
    near = box(11.04, 56.018, 11.06, 56.022)
    assert _allision_of_structure_1(template, [STRUCTURE, near], []) == 0.0
