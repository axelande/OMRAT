"""Unit tests for geometries/drift/clipping.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.drift.clipping import (
    clip_corridor_at_obstacles,
    split_corridor_by_anchor_zone,
    keep_reachable_part,
    _get_upwind_edge,
)


# ---------------------------------------------------------------------------
# clip_corridor_at_obstacles
# ---------------------------------------------------------------------------

class TestClipCorridorAtObstacles:
    def test_empty_obstacles_returns_corridor_unchanged(self):
        corr = box(0, 0, 1000, 100)
        out = clip_corridor_at_obstacles(corr, [], drift_angle_deg=0.0)
        assert out.equals(corr)

    def test_empty_corridor_returns_empty(self):
        out = clip_corridor_at_obstacles(Polygon(), [(box(0, 0, 1, 1), 0)], 0.0)
        assert out.is_empty

    def test_non_intersecting_obstacle_preserves_corridor(self):
        corr = box(0, 0, 100, 100)
        obs = [(box(500, 500, 600, 600), 0)]
        out = clip_corridor_at_obstacles(corr, obs, 0.0)
        assert out.area == pytest.approx(corr.area, rel=1e-9)

    def test_intersecting_obstacle_reduces_area(self):
        corr = box(0, 0, 100, 500)  # tall corridor going "north"
        obs = [(box(40, 100, 60, 120), 0)]
        out = clip_corridor_at_obstacles(corr, obs, drift_angle_deg=0.0)
        # Original area 50_000; clipped must be smaller.
        assert out.area < corr.area

    def test_empty_obstacle_polygon_skipped(self):
        corr = box(0, 0, 100, 100)
        out = clip_corridor_at_obstacles(corr, [(Polygon(), 0)], 0.0)
        assert out.equals(corr)


# ---------------------------------------------------------------------------
# split_corridor_by_anchor_zone
# ---------------------------------------------------------------------------

class TestSplitCorridorByAnchorZone:
    def test_empty_clipped_returns_empty_empty(self):
        b, g = split_corridor_by_anchor_zone(
            Polygon(), box(0, 0, 1, 1), 0.0, (-1, -1, 1, 1))
        assert b.is_empty and g.is_empty

    def test_no_anchor_overlap_all_green(self):
        clipped = box(0, 0, 100, 100)
        anchor = box(500, 500, 600, 600)
        b, g = split_corridor_by_anchor_zone(
            clipped, anchor, drift_angle_deg=0.0,
            corridor_bounds=(-1000, -1000, 1000, 1000))
        assert b.is_empty
        assert g.area == pytest.approx(clipped.area, rel=1e-9)

    def test_full_anchor_overlap_all_blue(self):
        clipped = box(0, 0, 100, 100)
        anchor = box(-10, -10, 110, 110)
        b, g = split_corridor_by_anchor_zone(
            clipped, anchor, 0.0, (-500, -500, 500, 500))
        assert g.is_empty
        assert b.area == pytest.approx(clipped.area, rel=1e-9)

    def test_partial_overlap_blue_plus_green_behind_becomes_blue(self):
        """Green areas behind blue (downwind) are converted to blue."""
        clipped = box(0, 0, 100, 500)  # 0..500 in "north"
        anchor = box(0, 100, 100, 200)  # anchor in the middle slice
        b, g = split_corridor_by_anchor_zone(
            clipped, anchor, drift_angle_deg=0.0,
            corridor_bounds=(-500, -500, 500, 1500),
        )
        # Blue includes the anchor zone plus everything "downwind"
        # (north of) the anchor zone.
        assert b.area > anchor.area


# ---------------------------------------------------------------------------
# keep_reachable_part
# ---------------------------------------------------------------------------

class TestKeepReachablePart:
    def test_empty_clipped_passthrough(self):
        out = keep_reachable_part(Polygon(), box(0, 0, 100, 100), 0.0)
        assert out.is_empty

    def test_single_part_returned_unchanged(self):
        corr = box(0, 0, 100, 100)
        out = keep_reachable_part(corr, corr, 0.0)
        assert out.equals(corr)

    def test_two_disjoint_parts_only_upwind_kept(self):
        """For drift angle 0 (north), upwind edge is the south boundary.
        Only the part touching the south edge survives."""
        original = box(0, 0, 100, 1000)
        # Two disjoint squares: upper one doesn't touch the south edge.
        south_part = box(10, 0, 50, 100)
        north_part = box(10, 500, 50, 600)
        clipped = MultiPolygon([south_part, north_part])
        out = keep_reachable_part(clipped, original, drift_angle_deg=0.0)
        # Only south part should remain (intersects upwind=south).
        assert out.intersects(south_part)
        # The north part should be excluded from the result.
        # (north_part and out may touch at a coord but should have
        # minimal intersection area.)
        inter_north = out.intersection(north_part)
        assert inter_north.area < 1e-6

    def test_no_parts_touch_upwind_fallback_to_largest(self):
        """When no part touches the upwind edge the helper falls back
        to returning the largest part."""
        original = box(0, 0, 100, 1000)
        # Both parts far from the south (upwind) edge.
        small = box(10, 500, 30, 550)
        large = box(40, 600, 90, 800)
        clipped = MultiPolygon([small, large])
        out = keep_reachable_part(clipped, original, drift_angle_deg=0.0)
        assert out.area == pytest.approx(large.area, rel=1e-9)


# ---------------------------------------------------------------------------
# _get_upwind_edge -- one case per cardinal
# ---------------------------------------------------------------------------

class TestGetUpwindEdge:
    bounds = (0.0, 0.0, 100.0, 100.0)

    @pytest.mark.parametrize("angle, side_check", [
        (0.0,   lambda e: e.bounds[1] < 1),      # N drift -> south edge (low Y)
        (90.0,  lambda e: e.bounds[2] > 99),     # W drift -> east edge (high X)
        (180.0, lambda e: e.bounds[3] > 99),     # S drift -> north edge
        (270.0, lambda e: e.bounds[0] < 1),      # E drift -> west edge
    ])
    def test_upwind_edge_for_cardinal_directions(self, angle, side_check):
        e = _get_upwind_edge(self.bounds, angle)
        assert side_check(e), f"edge {e.bounds} not on expected side for angle {angle}"

    def test_diagonal_angles_produce_corner_boxes(self):
        for angle in (45.0, 135.0, 225.0, 315.0):
            e = _get_upwind_edge(self.bounds, angle)
            assert isinstance(e, Polygon) and not e.is_empty
