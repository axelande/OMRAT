"""Standalone tests for ``compute.powered_breakdown`` and the drifting sidebar shares.

The powered View dialog lists each (leg, direction) with its route-table
name, annual frequency and % of the total; the drifting View gained the
same names and share column.  Everything tested here is pure data, so it
runs with ``--noconftest -p no:qgis``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compute.powered_breakdown import (  # noqa: E402
    build_sidebar_rows, format_frequency, format_share, geometric_probability,
    leg_direction_frequencies, leg_title, report_total, sidecar_payload, summarize_report,
)

SEGMENTS = {
    '9': {'Leg_name': 'LEG_6_1_a', 'Dirs': ['East going', 'West going']},
    '17': {'Leg_name': 'LEG_6_1_b', 'Dirs': ['East going', 'West going']},
}

# Shape of a model report (``_ContribAccumulator.report``).
REPORT = {
    'totals': {'allision': 1.0e-3, 'cat1': 0.9e-3, 'cat2': 0.1e-3},
    'by_obstacle': {'1_0': 0.8e-3, '2_0': 0.2e-3},
    'by_obstacle_leg': {
        '1_0': {'17:0': 0.6e-3, '17:1': 0.2e-3},
        '2_0': {'17:0': 0.1e-3, '9:0': 0.1e-3},
    },
    'by_obstacle_segment_legdir': {'1_0': {'seg_0': {'17:0': 0.6e-3}}},
    'causation_factor': 1.9e-4,
    'causation_factor_cat1': 1.9e-4,
}


def _comp(seg: str, dir_idx: int, name: str, cat2_p: float = 0.0, cat1_mass: float = 0.0) -> dict:
    cat2 = {('object', 'a'): {'p_approx': cat2_p}} if cat2_p else {}
    cat1 = {('object', 'b'): {'p_approx': cat1_mass}} if cat1_mass else {}
    return {'seg_id': seg, 'dir_idx': dir_idx, 'dir_info': {'name': name},
            'summaries': cat2, 'cat1': {'summaries': cat1}}


COMPS = [
    _comp('9', 0, 'East going', cat2_p=5e-6),
    _comp('17', 0, 'East going', cat1_mass=0.09),
    _comp('17', 1, 'West going', cat1_mass=0.04),
]


class TestFrequencies:
    def test_sums_by_obstacle_leg_per_leg_direction(self):
        freqs = leg_direction_frequencies(REPORT)
        assert freqs == pytest.approx({'17:0': 0.7e-3, '17:1': 0.2e-3, '9:0': 0.1e-3})

    def test_leg_directions_add_up_to_the_total(self):
        assert sum(leg_direction_frequencies(REPORT).values()) == pytest.approx(REPORT['totals']['allision'])

    def test_sidecar_form_is_read_as_is(self):
        assert leg_direction_frequencies({'by_leg_direction': {'9:0': '2e-6'}}) == {'9:0': 2e-6}

    def test_empty_and_bad_values(self):
        assert leg_direction_frequencies(None) == {}
        bad = {'by_obstacle_leg': {'x': {'1:0': 'nonsense', '1:1': float('inf')}, 'y': None}}
        assert leg_direction_frequencies(bad) == {'1:0': 0.0, '1:1': 0.0}

    def test_total_falls_back_to_the_leg_sum(self):
        assert report_total(REPORT, 'allision') == pytest.approx(1.0e-3)
        no_total = {'totals': {'allision': 0.0}, 'by_leg_direction': {'1:0': 1e-4, '1:1': 3e-4}}
        assert report_total(no_total, 'allision') == pytest.approx(4e-4)
        assert report_total(None, 'grounding') == 0.0


class TestSidecar:
    def test_summary_keeps_what_the_view_needs(self):
        block = summarize_report(REPORT)
        assert set(block) == {'totals', 'by_leg_direction', 'by_obstacle',
                              'causation_factor', 'causation_factor_cat1'}
        assert 'by_obstacle_segment_legdir' not in block
        assert block['by_leg_direction'] == pytest.approx(leg_direction_frequencies(REPORT))

    def test_payload_round_trips_through_json(self):
        payload = json.loads(json.dumps(sidecar_payload(REPORT, None)))
        assert set(payload) == {'allision'}
        rows = build_sidebar_rows(COMPS, payload['allision'], SEGMENTS, 'allision')
        direct = build_sidebar_rows(COMPS, REPORT, SEGMENTS, 'allision')
        assert [r['key'] for r in rows] == [r['key'] for r in direct]
        assert [r['share'] for r in rows] == pytest.approx([r['share'] for r in direct])

    def test_payload_is_none_without_reports(self):
        assert sidecar_payload(None, {}) is None
        assert sidecar_payload({'totals': {}}, None) is None


class TestSidebarRows:
    def test_rows_use_route_table_names(self):
        rows = build_sidebar_rows(COMPS, REPORT, SEGMENTS, 'allision')
        assert [(r['leg'], r['direction']) for r in rows] == [
            ('LEG_6_1_b', 'East going'), ('LEG_6_1_b', 'West going'), ('LEG_6_1_a', 'East going'),
        ]

    def test_rows_sorted_by_frequency_with_share_of_total(self):
        rows = build_sidebar_rows(COMPS, REPORT, SEGMENTS, 'allision')
        assert [r['key'] for r in rows] == ['17:0', '17:1', '9:0']
        assert [r['share'] for r in rows] == pytest.approx([70.0, 20.0, 10.0])
        assert [r['comp_idx'] for r in rows] == [1, 2, 0]

    def test_without_report_rank_by_geometric_probability(self):
        rows = build_sidebar_rows(COMPS, None, SEGMENTS, 'grounding')
        assert [r['key'] for r in rows] == ['17:0', '17:1', '9:0']
        assert all(r['frequency'] is None and r['share'] is None for r in rows)
        assert rows[0]['p_geom'] == pytest.approx(0.09)

    def test_leg_missing_from_report_gets_zero(self):
        comps = COMPS + [_comp('9', 1, 'West going', cat2_p=1e-9)]
        row = next(r for r in build_sidebar_rows(comps, REPORT, SEGMENTS, 'allision') if r['key'] == '9:1')
        assert row['frequency'] == 0.0 and row['share'] == 0.0

    def test_unnamed_leg_falls_back(self):
        assert leg_title(SEGMENTS, '9') == 'LEG_6_1_a'
        assert leg_title(SEGMENTS, '42', fallback='old name') == 'old name'
        assert leg_title({}, '42') == 'Leg 42'

    def test_geometric_probability_sums_both_categories(self):
        comp = _comp('1', 0, 'x', cat2_p=0.25, cat1_mass=0.5)
        assert geometric_probability(comp) == pytest.approx(0.75)
        assert geometric_probability({}) == 0.0

    def test_formatting_matches_collision_breakdown(self):
        assert format_frequency(None) == '--'
        assert format_frequency(0.0) == '0'
        assert format_frequency(1.234e-5) == '1.234e-05'
        assert format_share(None) == '--'
        assert format_share(42.04) == '42.0%'


class TestDriftSidebarShares:
    def test_display_leg_name_uses_route_table_name(self):
        from geometries.drift_overlap_sidebar import display_leg_name
        assert display_leg_name('Leg 9-East going', SEGMENTS) == 'LEG_6_1_a East going'
        assert display_leg_name('Leg 7-North going', SEGMENTS) == 'Leg 7-North going'
        assert display_leg_name('Leg 9-East going', None) == 'Leg 9-East going'

    def test_contribution_total_prefers_report_total(self):
        from geometries.drift_overlap_sidebar import contribution_total
        report = {
            'totals': {'allision': 4e-4, 'grounding': 0.0},
            'by_leg_direction': {
                '9:East going:90': {'contrib_allision': 1e-4, 'contrib_grounding': 2e-4},
                '9:East going:0': {'contrib_allision': 3e-4, 'contrib_grounding': 6e-4},
            },
        }
        assert contribution_total(report, 'allision') == pytest.approx(4e-4)
        # Zero total -> sum of the rows.
        assert contribution_total(report, 'grounding') == pytest.approx(8e-4)
        assert contribution_total(None, 'allision') == 0.0

    def test_share_text(self):
        from geometries.drift_overlap_sidebar import _format_share
        assert _format_share(1e-4, 4e-4) == ('25.0%', pytest.approx(25.0))
        assert _format_share(None, 4e-4) == ('--', -1.0)
        assert _format_share(1e-4, 0.0) == ('--', -1.0)
