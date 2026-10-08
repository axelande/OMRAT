"""Pure tests for the Compare tab's result breakdown and run-output loading.

Covers ``omrat_utils.compare_breakdown`` and the sidecar side of
``omrat_utils.compare`` (``load_run_outputs``, ``totals_from_outputs``,
``build_accident_table``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils import compare  # noqa: E402
from omrat_utils.compare import (  # noqa: E402
    _totals_from_run_row, build_accident_table, load_run_outputs, totals_from_outputs,
)
from omrat_utils.compare_breakdown import CATEGORIES, build_breakdown_rows, top_movers  # noqa: E402


def _seg(name, start, end, dirs=('East going', 'West going')):
    return {'Leg_name': name, 'Start_Point': start, 'End_Point': end, 'Dirs': list(dirs)}


SNAP_A = {'segment_data': {
    '1': _seg('LEG_1_1', '16.000000 56.000000', '16.500000 56.000000'),
    '2': _seg('LEG_2_1', '16.500000 56.000000', '17.000000 56.200000'),
}}
SNAP_B = {'segment_data': {
    '1': _seg('LEG_1_1', '16.000000 56.000000', '16.500000 56.000000'),
    '2': _seg('LEG_7_7', '16.500000 56.000000', '17.000000 55.800000'),
}}


def _outputs(head_on_1, head_on_2, crossing_wp, pa_dir0, spill):
    return {
        'collision': {
            'totals': {'head_on': head_on_1 + head_on_2, 'overtaking': 0.0, 'crossing': crossing_wp,
                       'merging': 0.0, 'bend': 0.0},
            'by_leg': {'1': {'head_on': head_on_1, 'overtaking': 0.0, 'bend': 0.0},
                       '2': {'head_on': head_on_2, 'overtaking': 0.0, 'bend': 0.0}},
            'by_waypoint': {'16.500000 56.000000': {'crossing': crossing_wp, 'merging': 0.0, 'bend': 0.0}},
            'by_leg_pair': {'1->2': {'crossing': crossing_wp, 'merging': 0.0, 'angle_deg': 40.0}},
        },
        'drifting': {
            'totals': {'allision': 0.01, 'grounding': 0.002, 'anchoring': 1.0},
            'by_leg_direction': {'1:East going:90': {'contrib_allision': 0.006, 'contrib_grounding': 0.001},
                                 '1:West going:270': {'contrib_allision': 0.004, 'contrib_grounding': 0.001}},
            'by_object': {'Structure - 1_0': {'allision': 0.01, 'grounding': 0.0, 'anchoring': 0.0}},
        },
        'powered': {
            'allision': {'totals': {'allision': pa_dir0}, 'by_leg_direction': {'1:0': pa_dir0},
                         'by_obstacle': {'1': pa_dir0}},
            'grounding': {'totals': {'grounding': 0.0}, 'by_leg_direction': {}, 'by_obstacle': {}},
        },
        'consequence': {
            'levels': [{'name': 'Minor', 'quantity': 50.0, 'exceedance': 1e-3}],
            'total_spill_frequency': 2e-3, 'mean_spill_volume_per_year': spill,
            'by_accident': {'Crossing collision': {'frequency': crossing_wp, 'spill_m3': spill}},
        },
    }


OUT_A = _outputs(1e-3, 2e-3, 5e-3, 1e-4, 1.0)
OUT_B = _outputs(1.5e-3, 3e-3, 4e-3, 0.0, 1.2)


def _cat(rows, category):
    return [r for r in rows if r[0] == category]


class TestBreakdownRows:
    rows = build_breakdown_rows(SNAP_A, SNAP_B, OUT_A, OUT_B)

    def test_categories_in_order(self):
        seen = []
        for r in self.rows:
            if r[0] not in seen:
                seen.append(r[0])
        assert seen == [c for c in CATEGORIES if c in seen]

    def test_leg_matched_on_id_and_name(self):
        legs = _cat(self.rows, 'Collisions per leg')
        by_item = {r[1]: r for r in legs}
        same = by_item['LEG_1_1 (1) – head-on']
        assert same[2:5] == [pytest.approx(1e-3), pytest.approx(1.5e-3), pytest.approx(5e-4)]
        # Leg id 2 is a different leg in each run: two one-sided rows.
        assert by_item['LEG_2_1 (2) – head-on'][2:5] == [pytest.approx(2e-3), None, pytest.approx(-2e-3)]
        assert by_item['LEG_7_7 (2) – head-on'][2:5] == [None, pytest.approx(3e-3), pytest.approx(3e-3)]

    def test_sorted_by_absolute_change_and_shares_sum_to_100(self):
        for category in CATEGORIES:
            cat = _cat(self.rows, category)
            if not cat:
                continue
            deltas = [abs(r[4]) for r in cat]
            assert deltas == sorted(deltas, reverse=True)
            if sum(deltas) > 0:
                assert sum(r[5] for r in cat) == pytest.approx(100.0)

    def test_waypoint_label_names_the_legs(self):
        wp = _cat(self.rows, 'Collisions per waypoint')
        assert len(wp) == 1
        assert wp[0][1].startswith('16.5000 56.0000 [LEG_1_1, LEG_2_1, LEG_7_7] – crossing')
        assert wp[0][4] == pytest.approx(-1e-3)

    def test_drifting_summed_per_leg(self):
        d = {r[1]: r for r in _cat(self.rows, 'Drifting per leg')}
        assert d['LEG_1_1 (1) – drifting allision'][2] == pytest.approx(0.01)

    def test_powered_named_by_direction_and_zero_rows_dropped(self):
        p = _cat(self.rows, 'Powered per leg and direction')
        assert [r[1] for r in p] == ['LEG_1_1 (1) East going – powered allision']
        assert p[0][3] == 0.0 and p[0][4] == pytest.approx(-1e-4)
        assert not any('grounding' in r[1] for r in p)

    def test_oil_rows(self):
        oil = _cat(self.rows, 'Oil spill per accident type')
        assert oil[0][1] == 'Crossing collision – spill (m³/year)'
        assert oil[0][4] == pytest.approx(0.2)

    def test_top_movers_ignores_pairs_and_oil(self):
        top = top_movers(self.rows, 3)
        assert all(r[0] in ('Collisions per leg', 'Collisions per waypoint',
                            'Drifting per leg', 'Powered per leg and direction') for r in top)
        assert top[0][1] == 'LEG_7_7 (2) – head-on'

    def test_no_outputs_no_rows(self):
        assert build_breakdown_rows(SNAP_A, SNAP_B, {}, None) == []


class TestRunOutputs:
    def test_load_run_outputs_reads_sidecars_and_skips_bad_ones(self, tmp_path):
        snap = tmp_path / 'run_20261001_120000.omrat'
        snap.write_text('{}', encoding='utf-8')
        (tmp_path / 'run_20261001_120000.collision.json').write_text(json.dumps(OUT_A['collision']), 'utf-8')
        (tmp_path / 'run_20261001_120000.drifting.json').write_text('{not json', 'utf-8')
        out = load_run_outputs(snap)
        assert sorted(out) == ['collision']

    def test_totals_keep_skipped_phases_missing(self):
        totals = totals_from_outputs({'collision': OUT_A['collision']})
        assert totals['head_on'] == pytest.approx(3e-3)
        assert 'drift_allision' not in totals and 'powered_grounding' not in totals
        full = totals_from_outputs(OUT_A)
        assert full['drift_allision'] == pytest.approx(0.01)
        assert full['powered_allision'] == pytest.approx(1e-4)

    def test_history_none_is_not_zero(self):
        """A phase not calculated in a partial run must not show as 0."""
        out = _totals_from_run_row({'head_on': None, 'crossing': 2e-3})
        assert out == {'crossing': 2e-3}

    def test_accident_table_from_sidecars(self, monkeypatch, tmp_path):
        monkeypatch.setattr(compare, '_match_run_history', lambda _p: None)
        rows = build_accident_table(tmp_path / 'a.omrat', tmp_path / 'b.omrat', OUT_A, OUT_B)
        by = {r[0]: r for r in rows}
        assert by['Head-on collision'][1:3] == ['3.000e-03', '4.500e-03']
        assert by['All allision'][1] == f'{0.01 + 1e-4:.3e}'
        assert by['Oil spill: expected volume (m³/year)'][1:3] == ['1', '1.2']
        assert 'Oil spill > 50 m³ (Minor) (per year)' in by
        assert list(by)[-4:-3] == ['All accidents']

    def test_accident_table_history_fills_gaps(self, monkeypatch, tmp_path):
        monkeypatch.setattr(compare, '_match_run_history',
                            lambda p: {'merging': 7e-4} if Path(p).stem == 'a' else None)
        rows = build_accident_table(tmp_path / 'a.omrat', tmp_path / 'b.omrat',
                                    {'collision': OUT_A['collision']}, {})
        by = {r[0]: r for r in rows}
        assert by['Merging collision'][1] == '0.000e+00'   # sidecar wins over history
        assert by['Head-on collision'][2] == 'n/a'


class TestMatchRunHistory:
    def test_prefers_run_in_the_snapshot_folder(self, monkeypatch, tmp_path):
        import omrat_utils.run_history as rh

        class _Run:
            def __init__(self, folder, value):
                self.output_dir = str(folder)
                self.output_filename = 'run_1.gpkg'
                self.crossing = value

            def gpkg_path(self):
                return Path(self.output_dir) / self.output_filename

        class _History:
            def list_runs(self):
                return [_Run(tmp_path / 'other', 1.0), _Run(tmp_path / 'here', 2.0)]

        monkeypatch.setattr(rh, 'RunHistory', _History)
        (tmp_path / 'here').mkdir()
        row = compare._match_run_history(tmp_path / 'here' / 'run_1.omrat')
        assert row['crossing'] == 2.0
        moved = compare._match_run_history(tmp_path / 'elsewhere' / 'run_1.omrat')
        assert moved['crossing'] == 1.0
