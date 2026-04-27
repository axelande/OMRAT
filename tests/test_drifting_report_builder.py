"""Unit tests for ``compute.drifting_report_builder``.

``DriftingReportBuilderMixin`` is a pure-Python mixin -- the four
report-assembly helpers mutate a ``report`` dict in place and never touch
QGIS.  These tests exercise them directly against a trivial mixin
instance and synthetic shapely geometries.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from shapely.geometry import LineString, Polygon, box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.drifting_report_builder import DriftingReportBuilderMixin


@pytest.fixture
def rb():
    return DriftingReportBuilderMixin()


@pytest.fixture
def empty_report():
    """Report shape produced by _iterate_traffic_and_sum before the cascade runs."""
    return {
        'totals': {'allision': 0.0, 'grounding': 0.0, 'anchoring': 0.0},
        'by_leg_direction': {},
        'by_object': {},
        'by_structure_legdir': {},
        'by_depth_legdir': {},
        'by_anchoring_legdir': {},
        'by_structure_segment_legdir': {},
        'by_depth_segment_legdir': {},
        'by_anchoring_segment_legdir': {},
    }


# ---------------------------------------------------------------------------
# _add_direct_segment_contrib
# ---------------------------------------------------------------------------

class TestAddDirectSegmentContrib:
    def test_none_seg_idx_is_noop(self, rb, empty_report):
        rb._add_direct_segment_contrib(
            empty_report, 'by_structure_segment_legdir',
            'Structure - s1', None, '1:East:0', 1.0e-6,
        )
        # No keys added because seg_idx was None.
        assert empty_report['by_structure_segment_legdir'] == {}

    def test_writes_into_fresh_map(self, rb, empty_report):
        rb._add_direct_segment_contrib(
            empty_report, 'by_depth_segment_legdir',
            'Depth - d1', 3, '1:North:0', 2.5e-6,
        )
        assert empty_report['by_depth_segment_legdir'] == {
            'Depth - d1': {'seg_3': {'1:North:0': 2.5e-6}}
        }

    def test_accumulates_multiple_calls(self, rb, empty_report):
        rb._add_direct_segment_contrib(
            empty_report, 'by_structure_segment_legdir',
            'Structure - s1', 0, '1:East:0', 1.0e-6,
        )
        rb._add_direct_segment_contrib(
            empty_report, 'by_structure_segment_legdir',
            'Structure - s1', 0, '1:East:0', 2.0e-6,
        )
        assert empty_report['by_structure_segment_legdir'][
            'Structure - s1']['seg_0']['1:East:0'] == pytest.approx(3.0e-6)

    def test_multiple_legs_kept_separate(self, rb, empty_report):
        rb._add_direct_segment_contrib(
            empty_report, 'by_structure_segment_legdir',
            'Structure - s1', 0, '1:East:0', 1.0e-6,
        )
        rb._add_direct_segment_contrib(
            empty_report, 'by_structure_segment_legdir',
            'Structure - s1', 0, '2:West:180', 4.0e-6,
        )
        seg = empty_report['by_structure_segment_legdir']['Structure - s1']['seg_0']
        assert seg == {'1:East:0': 1.0e-6, '2:West:180': 4.0e-6}


# ---------------------------------------------------------------------------
# _update_report -- no drift_corridor (the simple path)
# ---------------------------------------------------------------------------

class TestUpdateReportNoCorridor:
    def test_allision_populates_structure_map(self, rb, empty_report):
        structs = [{'id': 's1', 'wkt': box(10, 10, 20, 20)}]
        cell = {'direction': 'East going'}
        rb._update_report(
            empty_report, 'allision', contrib=1e-6, idx=0,
            structures=structs, depths=[], seg_id='1',
            cell=cell, d_idx=2, dist=100.0,
            base=1.0, rp=0.5, anchor_factor=0.1,
            p_nr=0.7, ov_frac=0.4, freq=5.0,
            ship_type=1, ship_size=2,
        )
        key = '1:East going:90'
        assert empty_report['by_structure_legdir']['Structure - s1'][key] == 1e-6
        rec = empty_report['by_leg_direction'][key]
        assert rec['contrib_allision'] == 1e-6
        assert rec['min_distance_allision'] == 100.0
        assert rec['weight_sum'] == pytest.approx(0.5)
        scat = rec['ship_categories']['1-2']
        assert scat['allision'] == 1e-6
        assert scat['freq'] == 5.0
        # Per-segment map stays empty when drift_corridor is None.
        assert empty_report['by_structure_segment_legdir'] == {}

    def test_grounding_populates_depth_map(self, rb, empty_report):
        depths = [{'id': 'd1', 'wkt': box(10, 10, 20, 20), 'depth': 5.0}]
        cell = {'direction': 'North going'}
        rb._update_report(
            empty_report, 'grounding', contrib=2e-6, idx=0,
            structures=[], depths=depths, seg_id='1',
            cell=cell, d_idx=0, dist=50.0,
            base=1.0, rp=0.5, anchor_factor=0.0,
            p_nr=0.8, ov_frac=0.3, freq=2.0,
            ship_type=3, ship_size=0,
        )
        key = '1:North going:0'
        assert empty_report['by_depth_legdir']['Depth - d1'][key] == 2e-6
        rec = empty_report['by_leg_direction'][key]
        assert rec['contrib_grounding'] == 2e-6
        assert rec['min_distance_grounding'] == 50.0

    def test_min_distance_keeps_smallest(self, rb, empty_report):
        structs = [{'id': 's1', 'wkt': box(10, 10, 20, 20)}]
        cell = {'direction': 'East going'}
        for dist in (100.0, 50.0, 200.0):
            rb._update_report(
                empty_report, 'allision', contrib=1e-6, idx=0,
                structures=structs, depths=[], seg_id='1',
                cell=cell, d_idx=2, dist=dist,
                base=1.0, rp=0.5, anchor_factor=0.0,
                p_nr=0.5, ov_frac=0.3, freq=1.0,
                ship_type=1, ship_size=0,
            )
        key = '1:East going:90'
        assert empty_report['by_leg_direction'][key]['min_distance_allision'] == 50.0

    def test_missing_idx_swallowed(self, rb, empty_report):
        """idx out of bounds in the per-structure block is silently ignored."""
        cell = {'direction': 'East going'}
        rb._update_report(
            empty_report, 'allision', contrib=1e-6, idx=99,
            structures=[], depths=[], seg_id='1',
            cell=cell, d_idx=2, dist=100.0,
            base=1.0, rp=0.5, anchor_factor=0.0,
            p_nr=0.5, ov_frac=0.3, freq=1.0,
            ship_type=1, ship_size=0,
        )
        # by_leg_direction record is still written (the basic path runs before
        # the per-structure block).
        assert '1:East going:90' in empty_report['by_leg_direction']
        # Per-structure map untouched because the structures list was empty.
        assert empty_report['by_structure_legdir'] == {}


# ---------------------------------------------------------------------------
# _update_report -- with drift_corridor (routes to _update_segment_contributions)
# ---------------------------------------------------------------------------

class TestUpdateReportWithCorridor:
    @pytest.fixture
    def obstacle_in_corridor(self):
        """Small obstacle that sits fully inside a large drift corridor."""
        corridor = box(0, 0, 100, 100)
        obstacle = box(40, 40, 60, 60)
        return obstacle, corridor

    def test_allision_with_corridor_triggers_segment_update(
        self, rb, empty_report, obstacle_in_corridor
    ):
        obstacle, corridor = obstacle_in_corridor
        structs = [{'id': 's1', 'wkt': obstacle}]
        cell = {'direction': 'East going'}
        rb._update_report(
            empty_report, 'allision', contrib=4e-6, idx=0,
            structures=structs, depths=[], seg_id='1',
            cell=cell, d_idx=2, dist=50.0,
            base=1.0, rp=0.5, anchor_factor=0.0,
            p_nr=0.7, ov_frac=0.4, freq=5.0,
            ship_type=1, ship_size=0,
            drift_corridor=corridor,
            leg=None,  # no leg -> no direction filter
        )
        seg_map = empty_report['by_structure_segment_legdir']['Structure - s1']
        # All 4 edges intersect the corridor geometrically.
        assert len(seg_map) == 4
        # Total contribution is preserved across edges.
        key = '1:East going:90'
        total = sum(seg_data[key] for seg_data in seg_map.values())
        assert total == pytest.approx(4e-6, rel=1e-9)

    def test_grounding_with_corridor_triggers_segment_update(
        self, rb, empty_report, obstacle_in_corridor
    ):
        obstacle, corridor = obstacle_in_corridor
        depths = [{'id': 'd1', 'wkt': obstacle, 'depth': 3.0}]
        cell = {'direction': 'North going'}
        rb._update_report(
            empty_report, 'grounding', contrib=8e-6, idx=0,
            structures=[], depths=depths, seg_id='2',
            cell=cell, d_idx=0, dist=10.0,
            base=1.0, rp=0.5, anchor_factor=0.0,
            p_nr=0.7, ov_frac=0.4, freq=2.0,
            ship_type=1, ship_size=0,
            drift_corridor=corridor,
            leg=None,
        )
        seg_map = empty_report['by_depth_segment_legdir']['Depth - d1']
        assert len(seg_map) == 4
        key = '2:North going:0'
        total = sum(seg_data[key] for seg_data in seg_map.values())
        assert total == pytest.approx(8e-6, rel=1e-9)

    def test_missing_wkt_skips_segment_update(self, rb, empty_report):
        """If the structure has no 'wkt' key, the per-segment block is skipped."""
        structs = [{'id': 's1'}]  # no 'wkt'
        cell = {'direction': 'East going'}
        rb._update_report(
            empty_report, 'allision', contrib=1e-6, idx=0,
            structures=structs, depths=[], seg_id='1',
            cell=cell, d_idx=2, dist=10.0,
            base=1.0, rp=0.5, anchor_factor=0.0,
            p_nr=0.7, ov_frac=0.4, freq=1.0,
            ship_type=1, ship_size=0,
            drift_corridor=box(0, 0, 100, 100),
            leg=None,
        )
        # by_structure_legdir still populated.
        assert '1:East going:90' in empty_report['by_structure_legdir']['Structure - s1']
        # Per-segment map untouched.
        assert empty_report['by_structure_segment_legdir'] == {}


# ---------------------------------------------------------------------------
# _update_segment_contributions -- direct tests
# ---------------------------------------------------------------------------

class TestUpdateSegmentContributions:
    def test_empty_polygon_noop(self, rb, empty_report):
        rb._update_segment_contributions(
            empty_report, 'by_structure_segment_legdir',
            'Structure - s1', '1:East:0', 1e-6,
            obs_geom=Polygon(), drift_corridor=box(0, 0, 100, 100),
        )
        assert empty_report['by_structure_segment_legdir'] == {}

    def test_no_intersecting_segments_returns_early(self, rb, empty_report):
        """Obstacle entirely outside the corridor produces no segment entries."""
        rb._update_segment_contributions(
            empty_report, 'by_structure_segment_legdir',
            'Structure - s1', '1:East:0', 1e-6,
            obs_geom=box(1000, 1000, 1010, 1010),
            drift_corridor=box(0, 0, 100, 100),
        )
        assert empty_report['by_structure_segment_legdir'] == {}

    def test_all_edges_inside_corridor_split_equally_by_length(self, rb, empty_report):
        """Symmetric box -> all 4 edges have equal intersection length -> split 1/4 each."""
        rb._update_segment_contributions(
            empty_report, 'by_structure_segment_legdir',
            'Structure - s1', '1:East:0', 8e-6,
            obs_geom=box(40, 40, 60, 60),
            drift_corridor=box(0, 0, 100, 100),
        )
        seg_map = empty_report['by_structure_segment_legdir']['Structure - s1']
        assert len(seg_map) == 4
        # Equal length edges -> equal split.
        for seg_data in seg_map.values():
            assert seg_data['1:East:0'] == pytest.approx(2e-6, rel=1e-9)

    def test_contribution_weighted_by_intersection_length(self, rb, empty_report):
        """Partial-corridor overlap: longer-intersection edge gets more weight."""
        # Long narrow box: edges differ in length.
        rb._update_segment_contributions(
            empty_report, 'by_structure_segment_legdir',
            'Structure - s1', '1:East:0', 100.0,
            obs_geom=box(10, 10, 90, 20),  # long in x, short in y
            drift_corridor=box(0, 0, 100, 100),
        )
        seg_map = empty_report['by_structure_segment_legdir']['Structure - s1']
        # Sum of contribs must equal the total, modulo floating point.
        total = sum(sd['1:East:0'] for sd in seg_map.values())
        assert total == pytest.approx(100.0, rel=1e-9)
        # The long horizontal edges (length 80) must each receive more than
        # the short vertical edges (length 10).
        lengths = sorted(sd['1:East:0'] for sd in seg_map.values())
        # 2 short, 2 long.
        assert lengths[0] == pytest.approx(lengths[1], rel=1e-9)  # 2 short equal
        assert lengths[2] == pytest.approx(lengths[3], rel=1e-9)  # 2 long equal
        assert lengths[2] > lengths[0]

    def test_runtime_segment_hits_recorded(self, rb, empty_report):
        """Segment debug metadata is written under ``runtime_segment_hits``."""
        leg = LineString([(0, 5), (100, 5)])
        rb._update_segment_contributions(
            empty_report, 'by_structure_segment_legdir',
            'Structure - s1', '1:East:0', 4e-6,
            obs_geom=box(40, 40, 60, 60),
            drift_corridor=box(0, 0, 100, 100),
            leg=leg,
        )
        runtime = empty_report['runtime_segment_hits']['Structure - s1']
        assert len(runtime) == 4
        for seg_meta in runtime.values():
            assert 'segment_wkt_utm' in seg_meta
            assert seg_meta['segment_wkt_utm'].startswith('LINESTRING')
            assert seg_meta['segment_wkt_wgs84'] is None  # no converter
            hit = seg_meta['hits']['1:East:0']
            assert hit['count'] == 1
            assert hit['max_intersection_len_m'] > 0.0
            # leg runs at y=5, square at y=40-60 -> dist >= 35.
            assert hit['min_distance_to_leg_m'] >= 35.0

    def test_accumulates_hits_on_repeat(self, rb, empty_report):
        """Two calls with the same leg_dir_key bump count to 2 and add contribs."""
        obstacle = box(40, 40, 60, 60)
        corridor = box(0, 0, 100, 100)
        for _ in range(2):
            rb._update_segment_contributions(
                empty_report, 'by_structure_segment_legdir',
                'Structure - s1', '1:East:0', 4e-6,
                obs_geom=obstacle, drift_corridor=corridor,
            )
        runtime = empty_report['runtime_segment_hits']['Structure - s1']
        for seg_meta in runtime.values():
            hit = seg_meta['hits']['1:East:0']
            assert hit['count'] == 2
            assert hit['contrib_sum'] == pytest.approx(2 * 1e-6, rel=1e-9)

    def test_wgs84_converter_used_when_host_provides_it(self, empty_report):
        """If host exposes ``_segment_utm_to_wgs84`` the converted WKT is recorded."""
        class Host(DriftingReportBuilderMixin):
            def _segment_utm_to_wgs84(self, seg_line):
                # Return a trivial stub LineString to prove the hook was called.
                return LineString([(0, 0), (1, 1)])

        host = Host()
        host._update_segment_contributions(
            empty_report, 'by_structure_segment_legdir',
            'Structure - s1', '1:East:0', 4e-6,
            obs_geom=box(40, 40, 60, 60),
            drift_corridor=box(0, 0, 100, 100),
        )
        runtime = empty_report['runtime_segment_hits']['Structure - s1']
        for seg_meta in runtime.values():
            assert seg_meta['segment_wkt_wgs84'] == 'LINESTRING (0 0, 1 1)'

    def test_converter_exception_swallowed(self, empty_report):
        """Converter raising does not abort the rest of the block."""
        class BrokenHost(DriftingReportBuilderMixin):
            def _segment_utm_to_wgs84(self, seg_line):
                raise RuntimeError("boom")

        host = BrokenHost()
        host._update_segment_contributions(
            empty_report, 'by_structure_segment_legdir',
            'Structure - s1', '1:East:0', 4e-6,
            obs_geom=box(40, 40, 60, 60),
            drift_corridor=box(0, 0, 100, 100),
        )
        runtime = empty_report['runtime_segment_hits']['Structure - s1']
        assert len(runtime) == 4
        for seg_meta in runtime.values():
            assert seg_meta['segment_wkt_wgs84'] is None

    def test_top_level_exception_swallowed(self, rb, empty_report):
        """Passing a bogus obs_geom type does not raise."""
        # Passing a string in place of a geometry trips _extract_obstacle_segments
        # which is wrapped in the outer try/except.
        rb._update_segment_contributions(
            empty_report, 'by_structure_segment_legdir',
            'Structure - s1', '1:East:0', 1e-6,
            obs_geom="not a geometry",  # type: ignore[arg-type]
            drift_corridor=box(0, 0, 100, 100),
        )
        # Report remains unchanged.
        assert empty_report['by_structure_segment_legdir'] == {}


# ---------------------------------------------------------------------------
# _update_anchoring_report
# ---------------------------------------------------------------------------

class TestUpdateAnchoringReport:
    def test_updates_by_anchoring_legdir_without_corridor(self, rb, empty_report):
        depths = [{'id': 'd1', 'wkt': box(40, 40, 60, 60), 'depth': 6.0}]
        rb._update_anchoring_report(
            empty_report, anchor_contrib=3e-6, obs_idx=0,
            depths=depths, seg_id='1', d_idx=0,
            dist=10.0, hole_pct=0.8,
            drift_corridor=None, leg=LineString([(0, 0), (100, 0)]),
        )
        key = '1:N:0'
        assert empty_report['by_anchoring_legdir']['Anchoring - d1'][key] == 3e-6
        # No corridor -> no segment map populated.
        assert 'by_anchoring_segment_legdir' not in empty_report or \
               empty_report['by_anchoring_segment_legdir'] == {}

    def test_updates_segment_map_with_corridor(self, rb, empty_report):
        depths = [{'id': 'd1', 'wkt': box(40, 40, 60, 60), 'depth': 6.0}]
        rb._update_anchoring_report(
            empty_report, anchor_contrib=4e-6, obs_idx=0,
            depths=depths, seg_id='1', d_idx=2,
            dist=10.0, hole_pct=0.8,
            drift_corridor=box(0, 0, 100, 100),
            leg=None,  # no leg -> geometric-only intersection
        )
        key = '1:E:90'
        assert empty_report['by_anchoring_legdir']['Anchoring - d1'][key] == 4e-6
        seg_map = empty_report['by_anchoring_segment_legdir']['Anchoring - d1']
        assert len(seg_map) == 4
        total = sum(sd[key] for sd in seg_map.values())
        assert total == pytest.approx(4e-6, rel=1e-9)

    def test_direction_name_follows_d_idx(self, rb, empty_report):
        depths = [{'id': 'd1', 'wkt': box(0, 0, 1, 1), 'depth': 5.0}]
        for d_idx, dname, angle in [
            (0, 'N', 0), (1, 'NE', 45), (2, 'E', 90), (3, 'SE', 135),
            (4, 'S', 180), (5, 'SW', 225), (6, 'W', 270), (7, 'NW', 315),
        ]:
            rep = {'by_anchoring_legdir': {}}
            rb._update_anchoring_report(
                rep, anchor_contrib=1e-6, obs_idx=0,
                depths=depths, seg_id='1', d_idx=d_idx,
                dist=10.0, hole_pct=0.8,
                drift_corridor=None, leg=LineString([(0, 0), (1, 0)]),
            )
            assert f'1:{dname}:{angle}' in rep['by_anchoring_legdir']['Anchoring - d1']

    def test_bad_obs_idx_swallowed(self, rb, empty_report):
        """idx out of range of depths list is caught by the outer try/except."""
        rb._update_anchoring_report(
            empty_report, anchor_contrib=1e-6, obs_idx=99,
            depths=[],  # empty -> IndexError
            seg_id='1', d_idx=0,
            dist=10.0, hole_pct=0.8,
            drift_corridor=None, leg=LineString([(0, 0), (1, 0)]),
        )
        # Nothing recorded, nothing raised.
        assert empty_report['by_anchoring_legdir'] == {}
