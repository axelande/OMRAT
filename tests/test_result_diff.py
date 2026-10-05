"""Standalone tests for the pure join in ``omrat_utils/result_diff.py``.

Run with ``-p no:qgis --noconftest``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.result_diff import DIFF_SPECS, DiffSpec, diff_rows, summary_lines  # noqa: E402

EDGE = next(s for s in DIFF_SPECS if s.layer_name == 'drifting_grounding')
POINTS = next(s for s in DIFF_SPECS if s.layer_name == 'collision_points')


def test_specs_cover_every_result_layer():
    from omrat_utils.run_persistence import _LAYER_NAMES
    assert tuple(s.layer_name for s in DIFF_SPECS) == _LAYER_NAMES
    assert all(isinstance(s, DiffSpec) for s in DIFF_SPECS)


def test_join_on_keys_and_delta():
    a = [
        {'obstacle_id': 'd3', 'segment_idx': 1, 'total_edge_probability': 1e-6, 'object_probability': 3e-6,
         'geometry': 'LINESTRING(0 0, 1 1)'},
        {'obstacle_id': 'd3', 'segment_idx': 2, 'total_edge_probability': 2e-6, 'object_probability': 3e-6},
        {'obstacle_id': 'd6', 'segment_idx': 1.0, 'total_edge_probability': 5e-7, 'object_probability': 5e-7},
    ]
    b = [
        {'obstacle_id': 'd3', 'segment_idx': 1, 'total_edge_probability': 2e-6, 'object_probability': 4e-6},
        {'obstacle_id': 'd3', 'segment_idx': 2, 'total_edge_probability': 2e-6, 'object_probability': 4e-6},
        {'obstacle_id': 'd6', 'segment_idx': 1, 'total_edge_probability': 0.0, 'object_probability': 0.0,
         'geometry': 'LINESTRING(5 5, 6 6)'},
        {'obstacle_id': 'd9', 'segment_idx': 7, 'total_edge_probability': 1e-7, 'object_probability': 1e-7},
    ]
    s = diff_rows(a, b, EDGE)
    assert s.n_features == 4
    assert s.sum_a == pytest.approx(3.5e-6)
    assert s.sum_b == pytest.approx(4.1e-6)
    assert s.total_delta == pytest.approx(0.6e-6)
    assert s.n_changed == 3  # (d3,2) unchanged
    by = {(r['obstacle_id'], r['segment_idx']): r for r in s.rows}
    r = by[('d3', 1)]
    assert r['delta'] == pytest.approx(1e-6)
    assert r['ratio'] == pytest.approx(2.0)
    assert r['pct_change'] == pytest.approx(100.0)
    assert r['delta_share_pct'] == pytest.approx(1e-6 / 0.6e-6 * 100)
    assert r['object_probability_delta'] == pytest.approx(1e-6)
    assert r['geometry'] == 'LINESTRING(0 0, 1 1)'
    assert r['rank'] == 1
    # Geometry taken from B when A has none; integer-valued float keys match ints.
    r = by[('d6', 1)]
    assert r['geometry'] == 'LINESTRING(5 5, 6 6)'
    assert r['delta'] == pytest.approx(-5e-7) and r['pct_change'] == pytest.approx(-100.0)
    # Only in B: no ratio / pct (no base), flagged.
    r = by[('d9', 7)]
    assert r['in_a'] is False and r['in_b'] is True
    assert r['ratio'] is None and r['pct_change'] is None
    assert r['delta'] == pytest.approx(1e-7)
    # Ranks follow |delta| descending.
    assert [x['rank'] for x in s.rows] == [1, 2, 3, 4]
    assert [abs(x['delta']) for x in s.rows] == sorted((abs(x['delta']) for x in s.rows), reverse=True)


def test_negligible_features_do_not_dominate_share():
    """The -40 % artefact: thousands of tiny edges must not outrank the real mover."""
    a = [{'obstacle_id': 'x', 'segment_idx': i, 'total_edge_probability': 1e-12} for i in range(1000)]
    b = [{'obstacle_id': 'x', 'segment_idx': i, 'total_edge_probability': 0.6e-12} for i in range(1000)]
    a.append({'obstacle_id': 'x', 'segment_idx': 5000, 'total_edge_probability': 1e-6})
    b.append({'obstacle_id': 'x', 'segment_idx': 5000, 'total_edge_probability': 1.6e-6})
    s = diff_rows(a, b, EDGE)
    top = s.rows[0]
    assert top['segment_idx'] == 5000 and top['rank'] == 1
    assert top['delta_share_pct'] == pytest.approx(0.6e-6 / s.total_delta * 100, rel=1e-6)
    assert top['delta_share_pct'] > 99.9
    tiny = s.rows[1]
    assert tiny['pct_change'] == pytest.approx(-40.0)
    assert abs(tiny['delta_share_pct']) < 1e-3


def test_points_spec_and_summary_lines():
    a = [{'waypoint': 'wp1', 'combined': 1e-4, 'crossing': 1e-4, 'merging': 0.0, 'bend': 0.0}]
    b = [{'waypoint': 'wp1', 'combined': 2e-4, 'crossing': 1e-4, 'merging': 1e-4, 'bend': 0.0}]
    s = diff_rows(a, b, POINTS)
    r = s.rows[0]
    assert r['waypoint'] == 'wp1' and r['delta'] == pytest.approx(1e-4)
    assert r['merging_delta'] == pytest.approx(1e-4) and r['crossing_delta'] == 0.0
    lines = summary_lines([s, diff_rows([], [], EDGE)])
    assert len(lines) == 1
    assert lines[0].startswith('Collision Points: 1 of 1 features changed')
    assert '(+100.0 %)' in lines[0]


def test_empty_inputs():
    s = diff_rows([], [], EDGE)
    assert s.n_features == 0 and s.rows == [] and s.total_delta == 0.0
