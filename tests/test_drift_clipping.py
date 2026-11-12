"""Unit tests for geometries/drift/clipping.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.drift.clipping import (  # noqa: E402
    clip_corridor_at_obstacles,
    split_corridor_by_anchor_zone,
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
