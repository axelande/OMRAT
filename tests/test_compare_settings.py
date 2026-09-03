"""Pure tests for the Compare tab's settings diff (``omrat_utils.compare``).

Before this rewrite ``build_settings_table`` looked up keys that no
``.omrat`` file has ever contained (``drift.speed_knots``,
``pc.cat_i`` ...), so the "Settings differences" table was always empty.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.compare import build_settings_table  # noqa: E402

EXAMPLE = ROOT / 'tests' / 'example_data' / 'proj.omrat'


def _snapshot():
    with EXAMPLE.open('r', encoding='utf-8') as f:
        return json.load(f)


def _labels(rows):
    return [r[0] for r in rows]


class TestIdenticalSnapshots:
    def test_no_rows(self):
        snap = _snapshot()
        assert build_settings_table(snap, copy.deepcopy(snap)) == []

    def test_empty_inputs(self):
        assert build_settings_table({}, {}) == []
        assert build_settings_table(None, {}) == []


class TestRealSchemaKeys:
    def test_drift_speed_and_anchor(self):
        a = _snapshot()
        b = copy.deepcopy(a)
        b['drift']['speed'] = a['drift']['speed'] * 2
        b['drift']['anchor_p'] = 0.5
        rows = build_settings_table(a, b)
        labels = _labels(rows)
        assert 'Drift speed (knots)' in labels
        assert 'Anchor probability' in labels
        anchor = next(r for r in rows if r[0] == 'Anchor probability')
        assert anchor[1] == '0.95'
        assert anchor[2] == '0.5'

    def test_repair_parameters(self):
        a = _snapshot()
        b = copy.deepcopy(a)
        b['drift']['repair']['use_lognormal'] = False
        b['drift']['repair']['scale'] = 1.2
        labels = _labels(build_settings_table(a, b))
        assert 'Use lognormal repair time' in labels
        assert 'Repair-time scale' in labels

    def test_causation_factors(self):
        a = _snapshot()
        a['pc'].update({'headon': 4.9e-5, 'grounding': 1.6e-4})
        b = copy.deepcopy(a)
        b['pc']['headon'] = 9.8e-5
        b['pc']['grounding'] = 3.2e-4
        rows = build_settings_table(a, b)
        labels = _labels(rows)
        assert 'Causation factor head-on' in labels
        assert 'Causation factor powered grounding' in labels
        assert 'Causation factor (powered, legacy)' not in labels  # unchanged

    def test_wind_rose_direction(self):
        a = _snapshot()
        b = copy.deepcopy(a)
        b['drift']['rose']['90'] = 0.5
        rows = build_settings_table(a, b)
        assert rows == [['Wind rose 90°', '0.125', '0.5']]

    def test_leg_width(self):
        a = _snapshot()
        b = copy.deepcopy(a)
        first = next(iter(b['segment_data']))
        b['segment_data'][first]['Width'] = 9999
        rows = build_settings_table(a, b)
        assert rows[-1][0] == f'Leg {first} width (m)'
        assert rows[-1][2] == '9999'

    def test_setting_only_on_one_side_shows_dash(self):
        a = _snapshot()
        b = copy.deepcopy(a)
        b['drift']['squat_mode'] = 'max_speed'
        rows = build_settings_table(a, b)
        assert rows == [['Squat mode', '—', 'max_speed']]

    def test_traffic_scaling_and_unknown_keys_fall_back_to_path_label(self):
        a = _snapshot()
        b = copy.deepcopy(a)
        a['traffic_scaling'] = {'global_percent': 100.0, 'follow_global': [True, True]}
        b['traffic_scaling'] = {'global_percent': 130.0, 'follow_global': [True, False]}
        rows = build_settings_table(a, b)
        labels = _labels(rows)
        assert 'Global traffic scaling (%)' in labels
        assert 'Traffic scaling: follow_global[1]' in labels

    def test_int_float_equality_is_not_a_difference(self):
        a = _snapshot()
        b = copy.deepcopy(a)
        b['drift']['drift_p'] = 1.0  # a has int 1
        assert build_settings_table(a, b) == []
