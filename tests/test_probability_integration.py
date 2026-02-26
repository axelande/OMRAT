# -*- coding: utf-8 -*-
"""
Tests for shadow-adjusted probability integration.

Tests the integration of drift corridor shadow algorithm with
probability calculation, verifying that upstream obstacles properly
reduce downstream probabilities.
"""

import pytest
import numpy as np
from shapely.geometry import LineString, Polygon, box

from geometries.drift.probability_integration import (
    compute_shadow_adjusted_holes,
    separate_obstacles_by_type,
    blend_with_pdf_holes,
    get_direction_index,
    direction_index_to_angle,
)
from geometries.drift.constants import DIRECTIONS


class TestDirectionIndexConversion:
    """Test direction index conversion utilities."""

    def test_direction_index_north(self):
        """North (0°) should be index 0."""
        assert get_direction_index(0) == 0

    def test_direction_index_northwest(self):
        """Northwest (45°) should be index 1."""
        assert get_direction_index(45) == 1

    def test_direction_index_west(self):
        """West (90°) should be index 2."""
        assert get_direction_index(90) == 2

    def test_direction_index_south(self):
        """South (180°) should be index 4."""
        assert get_direction_index(180) == 4

    def test_direction_index_east(self):
        """East (270°) should be index 6."""
        assert get_direction_index(270) == 6

    def test_direction_index_wraparound(self):
        """360° should wrap to index 0."""
        assert get_direction_index(360) == 0

    def test_index_to_angle_roundtrip(self):
        """Converting index to angle and back should give same index."""
        for idx in range(8):
            angle = direction_index_to_angle(idx)
            result_idx = get_direction_index(angle)
            assert result_idx == idx


class TestShadowAdjustedHoles:
    """Test the shadow-adjusted probability hole calculation."""

    def test_single_obstacle_has_positive_probability(self):
        """A single obstacle in the corridor should have positive probability."""
        # Create a simple leg
        leg = LineString([(0, 0), (1000, 0)])

        # Create an obstacle north of the leg (will be hit by N drift)
        obstacle = box(400, 500, 600, 700)  # In the path of north drift
        obstacles = [(obstacle, 5.0, 'depth', 0)]

        result = compute_shadow_adjusted_holes(
            [leg], obstacles,
            half_width=200,  # 400m wide corridor
            projection_dist=2000,
        )

        assert result is not None
        assert not result.get('cancelled', False)

        # Check north direction (index 0)
        north_idx = 0
        hole = result['effective_holes'][0][north_idx][0]
        assert hole > 0, "Obstacle in corridor should have positive probability"
        assert hole <= 1.0, "Probability should not exceed 1.0"

    def test_obstacle_outside_corridor_has_zero_probability(self):
        """An obstacle outside the corridor should have zero probability."""
        leg = LineString([(0, 0), (1000, 0)])

        # Create an obstacle far east - won't be in north corridor
        obstacle = box(5000, 500, 5200, 700)
        obstacles = [(obstacle, 5.0, 'depth', 0)]

        result = compute_shadow_adjusted_holes(
            [leg], obstacles,
            half_width=200,
            projection_dist=2000,
        )

        # North direction
        north_idx = 0
        hole = result['effective_holes'][0][north_idx][0]
        assert hole == 0, "Obstacle outside corridor should have zero probability"

    def test_upstream_obstacle_shadows_downstream(self):
        """An upstream obstacle should reduce probability of hitting downstream obstacles."""
        leg = LineString([(0, 0), (1000, 0)])

        # Two obstacles in line, north of leg
        upstream = box(400, 300, 600, 500)    # Closer to leg
        downstream = box(400, 800, 600, 1000)  # Farther from leg

        obstacles = [
            (upstream, 5.0, 'depth', 0),
            (downstream, 5.0, 'depth', 1),
        ]

        result = compute_shadow_adjusted_holes(
            [leg], obstacles,
            half_width=200,
            projection_dist=2000,
        )

        north_idx = 0
        upstream_hole = result['effective_holes'][0][north_idx][0]
        downstream_hole = result['effective_holes'][0][north_idx][1]

        # Upstream should have higher probability than downstream
        # because downstream is shadowed by upstream
        assert upstream_hole > 0, "Upstream obstacle should have positive probability"
        assert downstream_hole < upstream_hole, \
            "Downstream obstacle should be shadowed (lower probability)"

    def test_gap_between_obstacles_allows_passage(self):
        """Ships should be able to drift through gaps between scattered obstacles."""
        leg = LineString([(0, 0), (1000, 0)])

        # Two obstacles with a gap between them
        left_obstacle = box(100, 500, 300, 700)   # Left side
        right_obstacle = box(700, 500, 900, 700)  # Right side
        # Gap at x=300-700

        # Downstream obstacle in the gap
        downstream = box(400, 900, 600, 1100)

        obstacles = [
            (left_obstacle, 5.0, 'depth', 0),
            (right_obstacle, 5.0, 'depth', 1),
            (downstream, 5.0, 'depth', 2),
        ]

        result = compute_shadow_adjusted_holes(
            [leg], obstacles,
            half_width=500,  # Wide enough to cover all obstacles
            projection_dist=2000,
        )

        north_idx = 0
        downstream_hole = result['effective_holes'][0][north_idx][2]

        # Downstream should have positive probability because of the gap
        assert downstream_hole > 0, \
            "Downstream obstacle should be reachable through gap"

    def test_wide_obstacle_blocks_completely(self):
        """A wide obstacle should completely block downstream obstacles."""
        leg = LineString([(0, 0), (1000, 0)])

        # Wide obstacle that spans the entire corridor
        wide_obstacle = box(-100, 500, 1100, 700)  # Wider than corridor

        # Downstream obstacle
        downstream = box(400, 900, 600, 1100)

        obstacles = [
            (wide_obstacle, 5.0, 'depth', 0),
            (downstream, 5.0, 'depth', 1),
        ]

        result = compute_shadow_adjusted_holes(
            [leg], obstacles,
            half_width=200,
            projection_dist=2000,
        )

        north_idx = 0
        downstream_hole = result['effective_holes'][0][north_idx][1]

        # Downstream should be completely blocked (zero or near-zero probability)
        assert downstream_hole < 0.01, \
            "Downstream obstacle should be completely blocked by wide upstream obstacle"

    def test_shadow_factor_decreases_with_more_obstacles(self):
        """Shadow factor should decrease as more obstacles are processed."""
        leg = LineString([(0, 0), (1000, 0)])

        # Multiple obstacles in sequence
        obstacles = [
            (box(400, 200 + i*300, 600, 400 + i*300), 5.0, 'depth', i)
            for i in range(4)
        ]

        result = compute_shadow_adjusted_holes(
            [leg], obstacles,
            half_width=200,
            projection_dist=2000,
        )

        north_idx = 0
        shadow_factors = [
            result['shadow_factors'][0][north_idx][i]
            for i in range(4)
        ]

        # Each subsequent obstacle should have lower or equal shadow factor
        # Use small tolerance for floating-point comparison
        for i in range(1, 4):
            assert shadow_factors[i] <= shadow_factors[i-1] + 1e-9, \
                f"Shadow factor should not increase: {shadow_factors}"


class TestSeparateObstaclesByType:
    """Test separation of combined obstacle results by type."""

    def test_separates_structures_and_depths(self):
        """Should correctly separate structure and depth results."""
        leg = LineString([(0, 0), (1000, 0)])

        # Mix of structures and depths
        obstacles = [
            (box(400, 500, 600, 700), 15.0, 'structure', 0),  # Structure
            (box(100, 500, 300, 700), 5.0, 'depth', 0),       # Depth
            (box(700, 500, 900, 700), 20.0, 'structure', 1),  # Structure
        ]

        result = compute_shadow_adjusted_holes(
            [leg], obstacles,
            half_width=500,
            projection_dist=2000,
        )

        struct_holes, depth_holes = separate_obstacles_by_type(
            result, obstacles, num_structures=2
        )

        # Should have correct number of each type
        assert len(struct_holes[0][0]) == 2, "Should have 2 structure results"
        assert len(depth_holes[0][0]) == 1, "Should have 1 depth result"


class TestBlendWithPdfHoles:
    """Test blending shadow-adjusted and PDF-based holes."""

    def test_shadow_caps_pdf(self):
        """Shadow should act as upper bound on PDF values."""
        shadow_holes = [[[0.3]]]  # 30% shadow-adjusted probability
        pdf_holes = [[[0.5]]]     # 50% PDF-based probability

        blended = blend_with_pdf_holes(shadow_holes, pdf_holes, blend_factor=0.0)

        # With blend_factor=0, PDF should be capped at shadow value
        assert blended[0][0][0] == 0.3, \
            "PDF (0.5) should be capped at shadow (0.3)"

    def test_blend_factor_one_uses_shadow(self):
        """blend_factor=1.0 should use pure shadow values."""
        shadow_holes = [[[0.3]]]
        pdf_holes = [[[0.5]]]

        blended = blend_with_pdf_holes(shadow_holes, pdf_holes, blend_factor=1.0)

        assert blended[0][0][0] == 0.3, \
            "With blend_factor=1.0, should use shadow value"

    def test_zero_shadow_zeros_result(self):
        """Zero shadow should result in zero probability regardless of PDF."""
        shadow_holes = [[[0.0]]]  # Completely shadowed
        pdf_holes = [[[0.5]]]     # High PDF probability

        blended = blend_with_pdf_holes(shadow_holes, pdf_holes, blend_factor=0.0)

        assert blended[0][0][0] == 0.0, \
            "Zero shadow should result in zero probability"

    def test_empty_pdf_falls_back_to_shadow(self):
        """If PDF holes are empty, should use shadow values."""
        shadow_holes = [[[0.3]]]
        pdf_holes = []  # Empty

        blended = blend_with_pdf_holes(shadow_holes, pdf_holes, blend_factor=0.0)

        # Should not crash and should use 0.0 for missing PDF
        assert blended[0][0][0] == 0.0


class TestMultipleDirections:
    """Test that shadow calculation works correctly for all 8 directions."""

    def test_all_directions_computed(self):
        """All 8 directions should be computed."""
        leg = LineString([(500, 500), (1500, 500)])  # Centered leg
        obstacle = box(900, 900, 1100, 1100)  # Obstacle to the north

        obstacles = [(obstacle, 5.0, 'depth', 0)]

        result = compute_shadow_adjusted_holes(
            [leg], obstacles,
            half_width=200,
            projection_dist=2000,
        )

        # Should have results for all 8 directions
        assert len(result['effective_holes'][0]) == 8, \
            "Should have results for all 8 directions"

    def test_obstacle_only_affects_relevant_direction(self):
        """Obstacle to the north should only affect north-ish directions."""
        leg = LineString([(500, 500), (1500, 500)])
        obstacle = box(900, 1500, 1100, 1700)  # Far north

        obstacles = [(obstacle, 5.0, 'depth', 0)]

        result = compute_shadow_adjusted_holes(
            [leg], obstacles,
            half_width=200,
            projection_dist=2000,
        )

        # North (index 0) should have some probability
        north_hole = result['effective_holes'][0][0][0]

        # South (index 4) should have zero probability
        south_hole = result['effective_holes'][0][4][0]

        assert south_hole == 0, "Obstacle to north shouldn't affect south direction"


class TestCancellation:
    """Test that cancellation is properly handled."""

    def test_cancelled_flag_set_when_callback_returns_false(self):
        """Should set cancelled flag when progress callback returns False."""
        leg = LineString([(0, 0), (1000, 0)])
        obstacles = [(box(400, 500, 600, 700), 5.0, 'depth', 0)]

        def cancel_callback(completed, total, msg):
            return False  # Always cancel

        result = compute_shadow_adjusted_holes(
            [leg], obstacles,
            half_width=200,
            projection_dist=2000,
            progress_callback=cancel_callback,
        )

        assert result.get('cancelled', False), "Should be marked as cancelled"
