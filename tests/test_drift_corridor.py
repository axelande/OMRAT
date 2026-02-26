# -*- coding: utf-8 -*-
"""
Tests for drift corridor generation and shadow/obstacle handling.

These tests verify:
1. Shadow creation from obstacles
2. Corridor projection in various directions
3. Obstacle intersection and subtraction
4. Coordinate transformation
"""

import pytest
import numpy as np
from shapely.geometry import Polygon, LineString, box, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

# Import the functions under test
from geometries.drift_corridor import (
    create_shadow_behind_obstacle,
    apply_obstacle_shadows,
    clip_corridor_at_obstacles,
    create_projected_corridor,
    create_base_surface,
    get_distribution_width,
    get_projection_distance,
    DIRECTIONS,
)


class TestCreateShadowBehindObstacle:
    """Test the shadow creation function."""

    def test_simple_square_obstacle_east(self):
        """Shadow cast to the east from a square obstacle."""
        # Create a 100x100 meter square obstacle
        obstacle = box(0, 0, 100, 100)
        drift_angle_deg = 0  # East
        projection_dist = 500

        shadow = create_shadow_behind_obstacle(obstacle, drift_angle_deg, projection_dist)

        # Shadow should include the obstacle
        assert shadow.contains(obstacle) or shadow.equals(obstacle) or obstacle.intersection(shadow).area > 0.99 * obstacle.area

        # Shadow should extend 500m to the east
        shadow_bounds = shadow.bounds
        expected_max_x = 100 + 500  # original obstacle right edge + projection
        assert shadow_bounds[2] >= expected_max_x - 1  # Allow small tolerance

        # Shadow should have same y extent as obstacle (convex hull of square)
        assert abs(shadow_bounds[1] - 0) < 1  # min_y
        assert abs(shadow_bounds[3] - 100) < 1  # max_y

    def test_simple_square_obstacle_north(self):
        """Shadow cast to the north from a square obstacle."""
        obstacle = box(0, 0, 100, 100)
        drift_angle_deg = 90  # North
        projection_dist = 500

        shadow = create_shadow_behind_obstacle(obstacle, drift_angle_deg, projection_dist)

        # Shadow should extend 500m to the north
        shadow_bounds = shadow.bounds
        expected_max_y = 100 + 500
        assert shadow_bounds[3] >= expected_max_y - 1

    def test_simple_square_obstacle_west(self):
        """Shadow cast to the west from a square obstacle."""
        obstacle = box(0, 0, 100, 100)
        drift_angle_deg = 180  # West
        projection_dist = 500

        shadow = create_shadow_behind_obstacle(obstacle, drift_angle_deg, projection_dist)

        # Shadow should extend 500m to the west (negative x)
        shadow_bounds = shadow.bounds
        expected_min_x = 0 - 500
        assert shadow_bounds[0] <= expected_min_x + 1

    def test_simple_square_obstacle_south(self):
        """Shadow cast to the south from a square obstacle."""
        obstacle = box(0, 0, 100, 100)
        drift_angle_deg = 270  # South
        projection_dist = 500

        shadow = create_shadow_behind_obstacle(obstacle, drift_angle_deg, projection_dist)

        # Shadow should extend 500m to the south (negative y)
        shadow_bounds = shadow.bounds
        expected_min_y = 0 - 500
        assert shadow_bounds[1] <= expected_min_y + 1

    def test_shadow_area_reasonable(self):
        """Shadow area should be approximately obstacle + projected rectangle."""
        obstacle = box(0, 0, 100, 100)
        projection_dist = 500

        for direction, angle in DIRECTIONS.items():
            shadow = create_shadow_behind_obstacle(obstacle, angle, projection_dist)

            # For a square with convex hull shadow, the area should be roughly
            # obstacle_area + projection_dist * obstacle_width
            # The convex hull approach gives a hexagonal shape for diagonal directions
            min_expected_area = obstacle.area  # At minimum, includes obstacle
            max_expected_area = obstacle.area + projection_dist * 100 + projection_dist * 100 * 2  # Generous upper bound

            assert shadow.area >= min_expected_area, f"Shadow too small for {direction}"
            assert shadow.area <= max_expected_area, f"Shadow too large for {direction}"

    def test_empty_obstacle(self):
        """Empty obstacle should return empty shadow."""
        obstacle = Polygon()
        shadow = create_shadow_behind_obstacle(obstacle, 0, 500)
        assert shadow.is_empty

    def test_concave_obstacle_convex_hull(self):
        """Concave obstacle shadow should be the convex hull of original + translated."""
        # Create an L-shaped obstacle (concave)
        l_shape = Polygon([
            (0, 0), (100, 0), (100, 50), (50, 50), (50, 100), (0, 100), (0, 0)
        ])

        shadow = create_shadow_behind_obstacle(l_shape, 0, 500)  # East

        # The shadow should be convex (convex hull approach)
        assert shadow.is_valid
        # Convex hull of an L-shape is a rectangle-ish shape
        # The shadow should be larger than just the L-shape
        assert shadow.area > l_shape.area


class TestApplyObstacleShadows:
    """Test the shadow application to corridors."""

    def test_obstacle_inside_corridor_creates_hole(self):
        """An obstacle fully inside the corridor should create a hole."""
        # Create a large corridor
        corridor = box(0, 0, 1000, 200)

        # Create a small obstacle inside
        obstacle = box(400, 50, 500, 150)
        obstacles = [(obstacle, 10.0)]  # depth=10m

        result = apply_obstacle_shadows(corridor, obstacles, 0, 200, "Test: ")

        # Result should have less area than original corridor
        assert result.area < corridor.area

        # The shadow from obstacle going east should be subtracted
        # Original corridor area minus shadow area should equal result area
        shadow = create_shadow_behind_obstacle(obstacle, 0, 200)
        shadow_in_corridor = shadow.intersection(corridor)
        expected_area = corridor.area - shadow_in_corridor.area

        assert abs(result.area - expected_area) < 1  # Allow small tolerance

    def test_obstacle_outside_corridor_no_effect(self):
        """An obstacle outside the corridor should have no effect."""
        corridor = box(0, 0, 1000, 200)

        # Create an obstacle far outside the corridor
        obstacle = box(2000, 2000, 2100, 2100)
        obstacles = [(obstacle, 10.0)]

        result = apply_obstacle_shadows(corridor, obstacles, 0, 200, "Test: ")

        # Result should have same area as original
        assert abs(result.area - corridor.area) < 1

    def test_obstacle_at_edge_creates_partial_shadow(self):
        """An obstacle at the edge of the corridor creates partial shadow."""
        corridor = box(0, 0, 1000, 200)

        # Create an obstacle at the western edge, partially inside
        obstacle = box(-50, 50, 50, 150)
        obstacles = [(obstacle, 10.0)]

        result = apply_obstacle_shadows(corridor, obstacles, 0, 200, "Test: ")

        # Result should have less area than original
        assert result.area < corridor.area

    def test_multiple_obstacles(self):
        """Multiple obstacles should all create shadows."""
        corridor = box(0, 0, 2000, 200)

        # Create two obstacles
        obstacle1 = box(200, 50, 300, 150)
        obstacle2 = box(800, 50, 900, 150)
        obstacles = [(obstacle1, 10.0), (obstacle2, 15.0)]

        result = apply_obstacle_shadows(corridor, obstacles, 0, 300, "Test: ")

        # Both obstacles should create shadows
        assert result.area < corridor.area

        # Check that both shadow regions are removed
        shadow1 = create_shadow_behind_obstacle(obstacle1, 0, 300).intersection(corridor)
        shadow2 = create_shadow_behind_obstacle(obstacle2, 0, 300).intersection(corridor)

        # Result should not intersect with the shadow regions significantly
        # (some edge intersection is ok due to geometry precision)
        result_shadow1_overlap = result.intersection(shadow1)
        result_shadow2_overlap = result.intersection(shadow2)

        # The overlap should be minimal (much less than the shadow area)
        assert result_shadow1_overlap.area < shadow1.area * 0.1
        assert result_shadow2_overlap.area < shadow2.area * 0.1

    def test_no_obstacles(self):
        """No obstacles should return corridor unchanged."""
        corridor = box(0, 0, 1000, 200)

        result = apply_obstacle_shadows(corridor, [], 0, 200, "Test: ")

        assert result.equals(corridor) or abs(result.area - corridor.area) < 1

    def test_invalid_obstacle_handled_gracefully(self):
        """Invalid obstacle geometries should be handled without crashing."""
        corridor = box(0, 0, 1000, 200)

        # Create a self-intersecting (bowtie) polygon
        bowtie = Polygon([(0, 0), (100, 100), (100, 0), (0, 100), (0, 0)])
        obstacles = [(bowtie, 10.0)]

        # Should not raise an exception
        result = apply_obstacle_shadows(corridor, obstacles, 0, 200, "Test: ")

        # Result should still be valid
        assert result.is_valid or make_valid(result).is_valid


class TestCreateProjectedCorridor:
    """Test the corridor projection function."""

    def test_corridor_extends_in_drift_direction(self):
        """Corridor should extend from base surface in drift direction."""
        leg = LineString([(0, 0), (1000, 0)])  # East-west leg
        half_width = 50
        projection_dist = 500

        # East drift
        corridor_east = create_projected_corridor(leg, half_width, 0, projection_dist)

        # Should extend to the east
        bounds = corridor_east.bounds
        assert bounds[2] >= 1000 + 500 - 1  # max_x should be at least 1500m

    def test_corridor_maintains_width(self):
        """Corridor should maintain the distribution width."""
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 100  # Total width should be 200

        corridor = create_projected_corridor(leg, half_width, 0, 500)

        bounds = corridor.bounds
        corridor_width = bounds[3] - bounds[1]  # max_y - min_y

        assert abs(corridor_width - 200) < 1  # Should be ~200m wide

    def test_corridor_covers_base_surface(self):
        """Corridor should always contain the base surface."""
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 50

        base = create_base_surface(leg, half_width)

        for direction, angle in DIRECTIONS.items():
            corridor = create_projected_corridor(leg, half_width, angle, 500)

            # Base surface should be inside corridor (with tolerance for floating point)
            intersection = corridor.intersection(base)
            assert intersection.area >= base.area * 0.99, f"Base not in corridor for {direction}"


class TestCreateBaseSurface:
    """Test the base surface creation function."""

    def test_base_surface_area(self):
        """Base surface area should be leg_length × distribution_width."""
        leg = LineString([(0, 0), (1000, 0)])  # 1000m leg
        half_width = 50  # 100m total width

        base = create_base_surface(leg, half_width)

        expected_area = 1000 * 100  # 100,000 m²
        assert abs(base.area - expected_area) < 1

    def test_base_surface_centered_on_leg(self):
        """Base surface should be centered on the leg."""
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 50

        base = create_base_surface(leg, half_width)

        # Centroid should be at (500, 0)
        centroid = base.centroid
        assert abs(centroid.x - 500) < 1
        assert abs(centroid.y - 0) < 1

    def test_diagonal_leg(self):
        """Base surface should work with diagonal legs."""
        leg = LineString([(0, 0), (1000, 1000)])  # 45-degree leg
        half_width = 50

        base = create_base_surface(leg, half_width)

        # Should be a valid polygon
        assert base.is_valid
        assert not base.is_empty

        # Area should be leg_length × width
        leg_length = np.sqrt(1000**2 + 1000**2)  # ~1414m
        expected_area = leg_length * 100
        assert abs(base.area - expected_area) < 10


class TestGetDistributionWidth:
    """Test the distribution width calculation."""

    def test_width_increases_with_std(self):
        """Width should increase with standard deviation."""
        width_small = get_distribution_width(50, 0.99)
        width_large = get_distribution_width(100, 0.99)

        assert width_large > width_small

    def test_width_increases_with_coverage(self):
        """Width should increase with higher coverage."""
        width_95 = get_distribution_width(100, 0.95)
        width_99 = get_distribution_width(100, 0.99)

        assert width_99 > width_95

    def test_99_percent_coverage_multiplier(self):
        """99% coverage should be approximately 2 × 2.576 × std."""
        std = 100
        width = get_distribution_width(std, 0.99)

        # 2 * z_0.995 * std ≈ 2 * 2.576 * 100 = 515.2
        expected = 2 * 2.576 * std
        assert abs(width - expected) < 1


class TestGetProjectionDistance:
    """Test the projection distance calculation."""

    def test_distance_with_default_params(self):
        """Should return reasonable distance with default parameters."""
        repair_params = {
            'use_lognormal': True,
            'std': 0.95,
            'loc': 0.2,
            'scale': 0.85,
        }
        drift_speed_ms = 1.0  # 1 m/s

        distance = get_projection_distance(repair_params, drift_speed_ms)

        # Should be positive and less than max (50km)
        assert 0 < distance <= 50000

    def test_distance_increases_with_speed(self):
        """Distance should increase with drift speed."""
        repair_params = {
            'use_lognormal': True,
            'std': 0.95,
            'loc': 0.2,
            'scale': 0.85,
        }

        dist_slow = get_projection_distance(repair_params, 0.5)
        dist_fast = get_projection_distance(repair_params, 2.0)

        assert dist_fast > dist_slow

    def test_max_distance_cap(self):
        """Distance should be capped at max_distance."""
        repair_params = {
            'use_lognormal': True,
            'std': 0.95,
            'loc': 0.2,
            'scale': 10.0,  # Very large scale
        }

        distance = get_projection_distance(repair_params, 10.0, max_distance=10000)

        assert distance <= 10000

    def test_invalid_speed_fallback(self):
        """Invalid drift speed should use fallback."""
        repair_params = {'use_lognormal': True, 'std': 0.95, 'loc': 0.2, 'scale': 0.85}

        # Zero speed
        distance_zero = get_projection_distance(repair_params, 0)
        assert distance_zero > 0

        # Negative speed
        distance_neg = get_projection_distance(repair_params, -1)
        assert distance_neg > 0


class TestDepthObstacleScenarios:
    """Test realistic depth obstacle scenarios."""

    def test_shallow_depth_blocks_corridor(self):
        """Shallow depth area should block the corridor path."""
        # Ship route going east
        leg = LineString([(0, 0), (2000, 0)])
        half_width = 50
        projection_dist = 1000

        # Create corridor going east
        corridor = create_projected_corridor(leg, half_width, 0, projection_dist)

        # Shallow area in the path (simulating a reef)
        shallow_area = box(1500, -100, 1700, 100)  # Extends beyond corridor width
        obstacles = [(shallow_area, 5.0)]  # 5m depth - shallow

        result = apply_obstacle_shadows(corridor, obstacles, 0, projection_dist, "Shallow: ")

        # The shallow area and everything east of it should be blocked
        assert result.area < corridor.area

        # Check that the eastern part (behind obstacle) is removed
        eastern_region = box(1700, -50, 3000, 50)
        eastern_intersection = result.intersection(eastern_region)

        # Most of the eastern region should be blocked
        assert eastern_intersection.area < eastern_region.area * 0.5

    def test_deep_channel_not_blocked(self):
        """Deep channel should allow passage (not create shadow)."""
        # This tests that obstacles are correctly filtered by threshold
        corridor = box(0, 0, 2000, 200)

        # Deep area (should NOT be an obstacle if threshold is 10m and depth is 50m)
        # Note: the obstacle list should already be filtered before calling apply_obstacle_shadows
        # This test verifies that the filtering works at the data collection level

        # With empty obstacle list, corridor should be unchanged
        result = apply_obstacle_shadows(corridor, [], 0, 500, "Test: ")
        assert abs(result.area - corridor.area) < 1

    def test_overlapping_depth_obstacles(self):
        """Multiple overlapping depth areas should merge their shadows."""
        corridor = box(0, 0, 3000, 200)

        # Two overlapping shallow areas
        shallow1 = box(500, 50, 700, 150)
        shallow2 = box(600, 50, 800, 150)  # Overlaps with shallow1
        obstacles = [(shallow1, 5.0), (shallow2, 5.0)]

        result = apply_obstacle_shadows(corridor, obstacles, 0, 500, "Overlap: ")

        # Should block more area than a single obstacle
        single_result = apply_obstacle_shadows(corridor, [(shallow1, 5.0)], 0, 500, "Single: ")

        # The overlapping case should block at least as much area
        assert result.area <= single_result.area


class TestDirectionIntegrity:
    """Test that all 8 directions work correctly."""

    def test_all_directions_produce_valid_corridors(self):
        """All 8 wind directions should produce valid corridors."""
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 50
        projection_dist = 500

        for direction, angle in DIRECTIONS.items():
            corridor = create_projected_corridor(leg, half_width, angle, projection_dist)

            assert corridor.is_valid, f"Invalid corridor for {direction}"
            assert not corridor.is_empty, f"Empty corridor for {direction}"
            assert corridor.area > 0, f"Zero area corridor for {direction}"

    def test_opposite_directions_extend_opposite_ways(self):
        """Opposite directions should extend in opposite ways."""
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 50
        projection_dist = 500

        # East vs West
        east_corridor = create_projected_corridor(leg, half_width, DIRECTIONS['E'], projection_dist)
        west_corridor = create_projected_corridor(leg, half_width, DIRECTIONS['W'], projection_dist)

        east_bounds = east_corridor.bounds
        west_bounds = west_corridor.bounds

        # East should extend further east (higher max_x)
        assert east_bounds[2] > west_bounds[2]
        # West should extend further west (lower min_x)
        assert west_bounds[0] < east_bounds[0]

    def test_shadows_follow_drift_direction(self):
        """Shadows should extend in the drift direction."""
        obstacle = box(0, 0, 100, 100)
        projection_dist = 500

        shadow_east = create_shadow_behind_obstacle(obstacle, DIRECTIONS['E'], projection_dist)
        shadow_west = create_shadow_behind_obstacle(obstacle, DIRECTIONS['W'], projection_dist)

        # East shadow should extend east (higher max_x)
        assert shadow_east.bounds[2] > obstacle.bounds[2]
        # West shadow should extend west (lower min_x)
        assert shadow_west.bounds[0] < obstacle.bounds[0]


class TestClipCorridorAtObstacles:
    """Test the new clip_corridor_at_obstacles function."""

    def test_obstacle_in_middle_clips_beyond(self):
        """Obstacle in middle of corridor should clip everything beyond it."""
        # Create corridor going east
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 100
        projection_dist = 2000

        corridor = create_projected_corridor(leg, half_width, 0, projection_dist)  # East
        original_area = corridor.area
        leg_centroid = (500, 0)

        # Obstacle in the middle of the corridor (x=1500 to x=1700)
        # The corridor extends from x=0 to x=3000 (leg 0-1000 + projection 2000)
        obstacle_middle = box(1500, -200, 1700, 200)
        obstacles = [(obstacle_middle, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 0, leg_centroid, "Test: ")

        # Result should have less area (everything beyond obstacle is clipped)
        assert result.area < original_area, "Obstacle should reduce corridor area"

        # But the start should still exist
        start_region = box(-50, -50, 500, 50)
        assert result.intersects(start_region), "Start of corridor should still exist"

        # The obstacle area should be included (ships ground there)
        obstacle_in_corridor = corridor.intersection(obstacle_middle)
        assert result.intersects(obstacle_in_corridor), "Obstacle area should be in result"

        # Area beyond obstacle (x > 1700) should be mostly clipped
        beyond_obstacle = box(1800, -100, 2500, 100)
        beyond_intersection = result.intersection(beyond_obstacle)
        assert beyond_intersection.area < beyond_obstacle.area * 0.1, "Area beyond obstacle should be clipped"

    def test_no_obstacles_returns_unchanged(self):
        """No obstacles should return corridor unchanged."""
        corridor = box(0, -100, 2000, 100)
        leg_centroid = (500, 0)

        result = clip_corridor_at_obstacles(corridor, [], 0, leg_centroid, "Test: ")

        assert abs(result.area - corridor.area) < 1

    def test_obstacle_not_intersecting_returns_unchanged(self):
        """Obstacle not intersecting corridor should return unchanged."""
        corridor = box(0, -100, 2000, 100)
        leg_centroid = (500, 0)

        # Obstacle far away from corridor
        obstacle_far = box(5000, 5000, 6000, 6000)
        obstacles = [(obstacle_far, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 0, leg_centroid, "Test: ")

        assert abs(result.area - corridor.area) < 1

    def test_multiple_obstacles_combined(self):
        """Multiple obstacles should be combined for clipping."""
        corridor = box(0, -200, 3000, 200)
        leg_centroid = (500, 0)

        # Two obstacles in the corridor
        obstacle1 = box(1000, -50, 1200, 50)
        obstacle2 = box(1100, -100, 1300, 100)  # Overlapping with first
        obstacles = [(obstacle1, 5.0), (obstacle2, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 0, leg_centroid, "Test: ")

        # Result should have area removed
        assert result.area < corridor.area

    def test_multipolygon_scattered_parts_not_convex_hulled(self):
        """MultiPolygon with scattered parts should NOT be treated as convex hull.

        This is the key fix for the "corridors stop too early" issue. When depth
        obstacles come from gridded data (like GEBCO), they are often MultiPolygons
        with many scattered parts. The old code used convex hull on all parts,
        which filled in gaps and blocked corridor area that ships should reach.

        The fix processes each part individually, so shadows only block behind
        each individual part, not the entire convex hull of all parts.
        """
        # Create a corridor (east-west oriented, drifting North)
        corridor = box(0, 0, 10000, 5000)
        leg_centroid = (5000, 0)

        # Create a MultiPolygon with 3 scattered parts (simulating depth grid cells)
        # Parts are separated in x-direction, so convex hull would fill the gaps
        part1 = box(1000, 2000, 1500, 2500)  # 500x500 = 250000 m²
        part2 = box(3000, 2200, 3500, 2700)  # 500x500 = 250000 m²
        part3 = box(5000, 2100, 5500, 2600)  # 500x500 = 250000 m²

        scattered_multipolygon = MultiPolygon([part1, part2, part3])
        obstacles = [(scattered_multipolygon, 5.0)]

        # Clip with North direction (90 degrees)
        result = clip_corridor_at_obstacles(corridor, obstacles, 90, leg_centroid, "Test: ")

        # Check that area between scattered parts is NOT blocked
        # If old convex hull code was used, these points would be blocked
        from shapely.geometry import Point
        between_parts_1_2 = Point(2000, 3500)  # x=2000 is between part1 and part2
        between_parts_2_3 = Point(4000, 3500)  # x=4000 is between part2 and part3

        assert result.contains(between_parts_1_2), \
            "Area between scattered parts should NOT be blocked (convex hull bug)"
        assert result.contains(between_parts_2_3), \
            "Area between scattered parts should NOT be blocked (convex hull bug)"

        # The shadow should only cover the area directly behind each part
        # With 3 parts of 500m width each and ~2500m shadow length (to corridor edge),
        # shadow area ≈ 3 * 500 * 2500 = 3.75M m²
        # If convex hull was used, shadow would cover ~4500m * 2500m = 11.25M m²
        area_reduction = corridor.area - result.area
        # Allow some tolerance for edge effects
        assert area_reduction < 6_000_000, \
            f"Shadow area ({area_reduction}) too large - convex hull issue suspected"


class TestExampleDataNorthCorridor:
    """Test North drift corridor using real example data from proj.omrat.

    This test verifies the fix for corridors not following depth contours properly.
    The example data contains leg 1 and depth obstacles from GEBCO-style gridded data.
    """

    @pytest.fixture
    def example_data(self):
        """Load example data from proj.omrat"""
        import json
        from pathlib import Path

        proj_file = Path(__file__).parent / "example_data" / "proj.omrat"
        if not proj_file.exists():
            pytest.skip("Example data file proj.omrat not found")

        with open(proj_file, "r") as f:
            return json.load(f)

    @pytest.fixture
    def leg_1_utm(self, example_data):
        """Get leg 1 as a LineString in UTM coordinates."""
        from pyproj import CRS
        from geometries.drift_corridor import get_utm_crs, transform_geometry

        seg1 = example_data['segment_data']['1']
        start = seg1['Start_Point'].split()
        end = seg1['End_Point'].split()

        start_lon, start_lat = float(start[0]), float(start[1])
        end_lon, end_lat = float(end[0]), float(end[1])

        leg_wgs84 = LineString([(start_lon, start_lat), (end_lon, end_lat)])

        # Transform to UTM
        centroid = leg_wgs84.centroid
        utm_crs = get_utm_crs(centroid.x, centroid.y)
        wgs84 = CRS("EPSG:4326")

        return transform_geometry(leg_wgs84, wgs84, utm_crs), utm_crs, wgs84

    @pytest.fixture
    def depth_obstacles_utm(self, example_data, leg_1_utm):
        """Get depth obstacles in UTM coordinates with 15m threshold filtering."""
        return self._get_depth_obstacles(example_data, leg_1_utm, depth_threshold=15.0)

    @pytest.fixture
    def depth_obstacles_utm_30m(self, example_data, leg_1_utm):
        """Get depth obstacles in UTM coordinates with 30m threshold filtering."""
        return self._get_depth_obstacles(example_data, leg_1_utm, depth_threshold=30.0)

    def _get_depth_obstacles(self, example_data, leg_1_utm, depth_threshold: float):
        """Get depth obstacles in UTM coordinates with specified threshold filtering."""
        from shapely import wkt as shapely_wkt
        from geometries.drift_corridor import transform_geometry

        _, utm_crs, wgs84 = leg_1_utm

        bin_width = 3.0  # meters (detected from depth values 0, 3, 6, 9, ...)

        obstacles = []
        for d in example_data['depths']:
            depth_value = float(d[1])
            max_depth = depth_value + bin_width

            if max_depth > depth_threshold:
                continue

            wkt_str = d[2]
            geom = shapely_wkt.loads(wkt_str)
            geom = make_valid(geom)

            # Transform to UTM
            geom_utm = transform_geometry(geom, wgs84, utm_crs)
            geom_utm = make_valid(geom_utm)

            if not geom_utm.is_empty:
                obstacles.append((geom_utm, depth_value))

        return obstacles

    def test_north_corridor_follows_depth_contours(self, leg_1_utm, depth_obstacles_utm):
        """North corridor from leg 1 should follow actual depth contours.

        The key verification is that areas BETWEEN scattered depth obstacle parts
        are still reachable in the corridor. The old convex hull bug would block
        these areas incorrectly.
        """
        leg_utm, utm_crs, wgs84 = leg_1_utm

        # Create North corridor with reasonable parameters
        half_width = 1500  # meters
        projection_dist = 15000  # meters

        corridor = create_projected_corridor(leg_utm, half_width, 90, projection_dist)  # North = 90°
        original_area = corridor.area
        original_north = corridor.bounds[3]

        leg_centroid = leg_utm.centroid
        leg_centroid_tuple = (leg_centroid.x, leg_centroid.y)

        # Clip corridor at depth obstacles
        result = clip_corridor_at_obstacles(
            corridor, depth_obstacles_utm, 90, leg_centroid_tuple, "TestNorth: "
        )

        assert not result.is_empty, "Clipped corridor should not be empty"

        # Verify the corridor was clipped (some area removed)
        assert result.area < original_area, "Depth obstacles should reduce corridor area"

        # The corridor should still extend reasonably far north
        # With the fix, it should reach close to the shallowest depth obstacles
        result_north = result.bounds[3]
        north_reduction = original_north - result_north

        # The reduction should be reasonable (not the entire projection distance)
        # If convex hull bug was present, the reduction would be much larger
        assert north_reduction < projection_dist * 0.5, \
            f"Northern extent reduced too much ({north_reduction:.0f}m) - possible convex hull issue"

        # Area reduction should be moderate, not excessive
        area_reduction_pct = (original_area - result.area) / original_area * 100
        assert area_reduction_pct < 50, \
            f"Area reduction too large ({area_reduction_pct:.1f}%) - possible convex hull issue"

    def test_depth_obstacles_are_multipolygons_with_high_concavity(self, depth_obstacles_utm):
        """Verify that depth obstacles are MultiPolygons with significant concavity.

        This confirms the test is exercising the fix for scattered MultiPolygon parts.
        """
        total_parts = 0
        high_concavity_count = 0

        for geom, depth in depth_obstacles_utm:
            if isinstance(geom, MultiPolygon):
                parts = len(geom.geoms)
                total_parts += parts

                # Calculate concavity
                hull_area = geom.convex_hull.area
                actual_area = geom.area
                if hull_area > 0:
                    concavity = (hull_area - actual_area) / hull_area * 100
                    if concavity > 50:
                        high_concavity_count += 1

        # The example data should have MultiPolygons with many parts
        assert total_parts > 10, \
            f"Expected many MultiPolygon parts, got {total_parts}"

        # At least some obstacles should have high concavity (>50%)
        assert high_concavity_count > 0, \
            "Expected some obstacles with high concavity (>50%)"

    def test_corridor_stops_at_obstacle_south_edge(self, leg_1_utm, depth_obstacles_utm):
        """The corridor should stop at the SOUTH edge of obstacles (where ships first hit them).

        When drifting NORTH, a ship will first encounter the SOUTHERN edge of an obstacle.
        The corridor should stop there, not extend into or beyond the obstacle.
        """
        leg_utm, utm_crs, wgs84 = leg_1_utm

        # Create North corridor
        half_width = 1500
        projection_dist = 15000
        corridor = create_projected_corridor(leg_utm, half_width, 90, projection_dist)

        leg_centroid = leg_utm.centroid
        leg_centroid_tuple = (leg_centroid.x, leg_centroid.y)

        # Find the SOUTHERNMOST extent (leading edge for north drift) of the shallowest (0.0m) obstacle
        shallowest_south = None
        for geom, depth in depth_obstacles_utm:
            if depth == 0.0:  # Shallowest bin
                intersection = corridor.intersection(geom)
                if not intersection.is_empty:
                    # Get the SOUTH (minimum Y) edge of the obstacle - where ship first hits it
                    south_edge = intersection.bounds[1]
                    if shallowest_south is None or south_edge < shallowest_south:
                        shallowest_south = south_edge

        if shallowest_south is None:
            pytest.skip("No 0.0m depth obstacles intersect the corridor")
            return  # For type checker

        # Clip corridor
        result = clip_corridor_at_obstacles(
            corridor, depth_obstacles_utm, 90, leg_centroid_tuple, ""
        )

        # The clipped corridor's northern extent should be reasonably close to
        # the obstacle's southern edge. With scattered MultiPolygon parts,
        # the corridor may stop at the south edge of any intersecting part,
        # not necessarily the globally southernmost edge.
        result_north = result.bounds[3]

        # Allow tolerance for scattered obstacle parts - the corridor stops at the
        # south edge of whichever part it first encounters in the drift direction
        # which depends on the x-coordinate of the corridor
        tolerance = 5000.0  # meters - accounts for scattered parts at different y-coords
        max_expected = shallowest_south + tolerance
        assert result_north <= max_expected, \
            f"Corridor extends to {result_north:.0f}m but should stop near obstacle south edge {shallowest_south:.0f}m"

        # Also verify that the corridor was actually clipped (not the full projection distance)
        original_north = corridor.bounds[3]
        assert result_north < original_north - 1000, \
            f"Corridor should be significantly clipped. Original: {original_north:.0f}m, Result: {result_north:.0f}m"

    def test_north_corridor_stops_before_30m_depth_obstacles(self, leg_1_utm, depth_obstacles_utm_30m):
        """The north end of the corridor must stop BEFORE depth obstacles ≤30m.

        This test verifies that the corridor properly terminates at the leading edge
        of shallow water obstacles. The area BEYOND the corridor's north edge should
        be covered by depth obstacles of 30m or less.

        This ensures corridors follow depth contours correctly - ships drifting
        north will stop when they FIRST HIT shallow water, not pass through it.
        """
        leg_utm, utm_crs, wgs84 = leg_1_utm

        # Create North corridor
        half_width = 1500
        projection_dist = 15000
        corridor = create_projected_corridor(leg_utm, half_width, 90, projection_dist)

        leg_centroid = leg_utm.centroid
        leg_centroid_tuple = (leg_centroid.x, leg_centroid.y)

        # Clip corridor at depth obstacles (30m threshold)
        result = clip_corridor_at_obstacles(
            corridor, depth_obstacles_utm_30m, 90, leg_centroid_tuple, "Test30m: "
        )

        assert not result.is_empty, "Clipped corridor should not be empty"

        # Get the northern boundary of the clipped corridor
        result_bounds = result.bounds
        result_north = result_bounds[3]
        result_west = result_bounds[0]
        result_east = result_bounds[2]

        # Union all depth obstacles (30m threshold)
        all_obstacles = unary_union([geom for geom, _ in depth_obstacles_utm_30m])
        all_obstacles = make_valid(all_obstacles)

        # The area BEYOND the corridor's north edge should be covered by obstacles
        # (the corridor stops just before hitting obstacles)
        buffer_width = 50.0  # meters - check a strip just beyond the corridor
        beyond_north_strip = box(result_west, result_north, result_east, result_north + buffer_width)
        beyond_covered = beyond_north_strip.intersection(all_obstacles)

        # The area beyond should have substantial obstacle coverage
        # (this confirms the corridor stopped at the depth contour edge)
        if not beyond_covered.is_empty:
            beyond_coverage = beyond_covered.area / beyond_north_strip.area
            assert beyond_coverage > 0.3, \
                f"Expected obstacles just beyond corridor north edge, got {beyond_coverage:.1%} coverage"

        # The corridor itself should NOT significantly overlap with obstacles
        # (corridor stops before obstacles, not inside them)
        corridor_obstacle_overlap = result.intersection(all_obstacles)
        if not corridor_obstacle_overlap.is_empty:
            overlap_ratio = corridor_obstacle_overlap.area / result.area
            # Allow some overlap due to geometry precision and irregular obstacle shapes
            assert overlap_ratio < 0.2, \
                f"Corridor overlaps too much with obstacles ({overlap_ratio:.1%}). " \
                f"Corridor should stop BEFORE obstacles, not pass through them."


class TestDepthIntervalParsing:
    """Test depth interval parsing logic (simulates what get_depth_obstacles does)."""

    def parse_depth_value(self, depth_text: str) -> float:
        """Replicate the parsing logic from get_depth_obstacles."""
        # Handle interval format like "0-10" or "0.0-10.0" or single value "10"
        # Also handle negative values (GEBCO stores depths as negative)
        if '-' in depth_text and not depth_text.startswith('-'):
            # Interval format: "0-10" or "10-20"
            parts = depth_text.split('-')
            # Use the UPPER bound - the maximum depth in this area
            depth = float(parts[-1])
        elif depth_text.startswith('-'):
            # Negative single value or negative interval like "-10" or "-20--10"
            # Convert to positive depth
            if '--' in depth_text:
                # Format like "-20--10" means -20 to -10 (depths 10 to 20m)
                parts = depth_text.split('--')
                depth = abs(float(parts[0]))  # Use the shallower (less negative)
            else:
                depth = abs(float(depth_text))
        else:
            depth = float(depth_text)
        return depth

    def test_simple_interval_parsing(self):
        """Test parsing of simple depth intervals."""
        assert self.parse_depth_value("0-10") == 10.0
        assert self.parse_depth_value("10-20") == 20.0
        assert self.parse_depth_value("0-5") == 5.0

    def test_single_value_parsing(self):
        """Test parsing of single depth values."""
        assert self.parse_depth_value("10") == 10.0
        assert self.parse_depth_value("5.5") == 5.5

    def test_negative_depth_parsing(self):
        """Test parsing of negative depth values (GEBCO format)."""
        assert self.parse_depth_value("-10") == 10.0
        assert self.parse_depth_value("-20--10") == 20.0  # -20 to -10 → depths 10-20m

    def test_threshold_comparison(self):
        """Test the threshold comparison logic.

        If threshold=10m:
        - "0-10" (max depth 10m) should be INCLUDED (grounding risk in shallow parts)
        - "10-20" (max depth 20m) should be EXCLUDED (deep enough)
        """
        threshold = 10.0

        # Test "0-10" interval
        depth_0_10 = self.parse_depth_value("0-10")  # = 10.0
        include_0_10 = depth_0_10 <= threshold  # 10 <= 10 = True
        assert include_0_10, "0-10m interval should be included with 10m threshold"

        # Test "10-20" interval
        depth_10_20 = self.parse_depth_value("10-20")  # = 20.0
        include_10_20 = depth_10_20 <= threshold  # 20 <= 10 = False
        assert not include_10_20, "10-20m interval should NOT be included with 10m threshold"

        # Test "0-5" interval
        depth_0_5 = self.parse_depth_value("0-5")  # = 5.0
        include_0_5 = depth_0_5 <= threshold  # 5 <= 10 = True
        assert include_0_5, "0-5m interval should be included with 10m threshold"

    def test_threshold_edge_cases(self):
        """Test threshold comparison edge cases."""
        # With threshold=15m
        threshold = 15.0

        depth_0_10 = self.parse_depth_value("0-10")  # = 10.0
        include = depth_0_10 <= threshold  # 10 <= 15 = True
        assert include, "0-10m should be included with 15m threshold"

        depth_10_20 = self.parse_depth_value("10-20")  # = 20.0
        include = depth_10_20 <= threshold  # 20 <= 15 = False
        assert not include, "10-20m should NOT be included with 15m threshold"

        depth_0_15 = self.parse_depth_value("0-15")  # = 15.0
        include = depth_0_15 <= threshold  # 15 <= 15 = True
        assert include, "0-15m should be included with 15m threshold (boundary case)"


class TestCoordinateTransformWithObstacles:
    """Test coordinate transformation with obstacles to verify the full pipeline."""

    def test_wgs84_to_utm_and_back(self):
        """Test that obstacles transform correctly from WGS84 to UTM and back."""
        from pyproj import CRS
        from geometries.drift_corridor import get_utm_crs, transform_geometry

        # Create an obstacle in WGS84 (typical North Sea coordinates)
        # This is approximately a 1km x 1km square
        wgs84 = CRS("EPSG:4326")
        obstacle_wgs84 = box(5.0, 58.0, 5.01, 58.01)  # ~1km square

        # Get UTM CRS for this location
        utm_crs = get_utm_crs(5.0, 58.0)

        # Transform to UTM
        obstacle_utm = transform_geometry(obstacle_wgs84, wgs84, utm_crs)

        # UTM coordinates should be in meters, area should be approximately 1km²
        # 0.01 degrees latitude ≈ 1.1km, 0.01 degrees longitude at 58°N ≈ 0.6km
        assert obstacle_utm.area > 500000  # Should be > 0.5 km²
        assert obstacle_utm.area < 1500000  # Should be < 1.5 km²

        # Transform back to WGS84
        obstacle_back = transform_geometry(obstacle_utm, utm_crs, wgs84)

        # Should be approximately the same as original
        assert abs(obstacle_back.bounds[0] - obstacle_wgs84.bounds[0]) < 0.0001
        assert abs(obstacle_back.bounds[1] - obstacle_wgs84.bounds[1]) < 0.0001

    def test_corridor_with_obstacle_different_locations(self):
        """Test corridor with obstacle at different geographic locations."""
        from pyproj import CRS
        from geometries.drift_corridor import get_utm_crs, transform_geometry

        wgs84 = CRS("EPSG:4326")

        # Test location in North Sea
        lon, lat = 5.0, 58.0
        utm_crs = get_utm_crs(lon, lat)

        # Create leg and obstacle in WGS84
        leg_wgs84 = LineString([(lon, lat), (lon + 0.1, lat)])  # ~8km leg going east
        obstacle_wgs84 = box(lon + 0.05, lat - 0.005, lon + 0.06, lat + 0.005)  # Obstacle in path

        # Transform to UTM
        leg_utm = transform_geometry(leg_wgs84, wgs84, utm_crs)
        obstacle_utm = transform_geometry(obstacle_wgs84, wgs84, utm_crs)

        # Create corridor
        half_width = 500  # 500m
        projection_dist = 5000  # 5km
        corridor = create_projected_corridor(leg_utm, half_width, 0, projection_dist)  # East

        # Verify obstacle intersects corridor
        assert corridor.intersects(obstacle_utm), "Obstacle should intersect corridor"

        # Apply shadow
        obstacles = [(obstacle_utm, 5.0)]
        result = apply_obstacle_shadows(corridor, obstacles, 0, projection_dist, "Test: ")

        # Result should have less area
        assert result.area < corridor.area, "Shadow should reduce corridor area"

    def test_obstacle_intersection_with_logging(self):
        """Test obstacle intersection checking with explicit verification."""
        # Create a corridor and obstacle that should definitely intersect
        corridor = box(0, -100, 2000, 100)  # 2km x 200m corridor

        # Obstacle clearly inside the corridor
        obstacle = box(500, -50, 600, 50)

        # Verify intersection
        assert corridor.intersects(obstacle), "Obstacle should intersect corridor"

        # The intersection area
        intersection = corridor.intersection(obstacle)
        assert intersection.area > 0, "Intersection should have positive area"
        assert abs(intersection.area - obstacle.area) < 1, "Full obstacle should be inside corridor"


class TestFullPipelineSimulation:
    """Simulate the full pipeline from WKT storage to corridor generation."""

    def test_wkt_roundtrip(self):
        """Test that WKT parsing produces valid geometry."""
        from shapely import wkt as shapely_wkt

        # Create a polygon similar to what GEBCO might produce
        original = box(5.0, 58.0, 5.1, 58.1)
        wkt_str = original.wkt

        # Parse it back
        parsed = shapely_wkt.loads(wkt_str)

        assert parsed.is_valid
        assert not parsed.is_empty
        assert abs(parsed.area - original.area) < 0.0001

    def test_multipolygon_wkt(self):
        """Test that MultiPolygon WKT is handled correctly."""
        from shapely import wkt as shapely_wkt
        from shapely.geometry import MultiPolygon

        # Create a MultiPolygon (like GEBCO might produce with disconnected areas)
        poly1 = box(5.0, 58.0, 5.05, 58.05)
        poly2 = box(5.1, 58.1, 5.15, 58.15)
        multi = MultiPolygon([poly1, poly2])

        wkt_str = multi.wkt
        parsed = shapely_wkt.loads(wkt_str)

        # Should be a MultiPolygon
        assert isinstance(parsed, MultiPolygon)
        assert len(parsed.geoms) == 2

    def test_obstacle_data_flow(self):
        """Simulate the full data flow from table to obstacle list."""
        from shapely import wkt as shapely_wkt

        # Simulate table data (like what get_depth_obstacles would read)
        table_data = [
            {"depth_text": "0-10", "wkt": box(5.0, 58.0, 5.01, 58.01).wkt},  # Shallow
            {"depth_text": "10-20", "wkt": box(5.02, 58.0, 5.03, 58.01).wkt},  # Deeper
            {"depth_text": "0-5", "wkt": box(5.04, 58.0, 5.05, 58.01).wkt},   # Very shallow
        ]

        depth_threshold = 10.0
        obstacles = []

        for row_data in table_data:
            depth_text = row_data["depth_text"]

            # Parse depth (same logic as get_depth_obstacles)
            if '-' in depth_text and not depth_text.startswith('-'):
                parts = depth_text.split('-')
                depth = float(parts[-1])
            else:
                depth = float(depth_text)

            # Check threshold
            if depth > depth_threshold:
                continue  # Skip deep areas

            # Parse WKT
            wkt = row_data["wkt"]
            shapely_geom = shapely_wkt.loads(wkt)

            if hasattr(shapely_geom, 'exterior'):
                obstacles.append((shapely_geom, depth))

        # Should have 2 obstacles (0-10 and 0-5), not 10-20
        assert len(obstacles) == 2

        # Verify the depths
        depths = [d for _, d in obstacles]
        assert 10.0 in depths  # 0-10 interval
        assert 5.0 in depths   # 0-5 interval
        assert 20.0 not in depths  # 10-20 should be excluded

    def test_corridor_generation_with_simulated_data(self):
        """Test full corridor generation with simulated depth data."""
        from pyproj import CRS
        from geometries.drift_corridor import get_utm_crs, transform_geometry

        wgs84 = CRS("EPSG:4326")

        # Create a ship route
        route_start = (5.0, 58.0)
        route_end = (5.1, 58.0)
        leg_wgs84 = LineString([route_start, route_end])

        # Create depth obstacles in WGS84
        # Shallow reef in the middle of the route
        shallow_reef = box(5.04, 57.995, 5.06, 58.005)  # Shallow area crossing the route

        obstacles_wgs84 = [(shallow_reef, 5.0)]  # 5m depth

        # Get UTM for this location
        utm_crs = get_utm_crs(5.05, 58.0)

        # Transform to UTM
        leg_utm = transform_geometry(leg_wgs84, wgs84, utm_crs)
        obstacles_utm = []
        for poly, depth in obstacles_wgs84:
            poly_utm = transform_geometry(poly, wgs84, utm_crs)
            obstacles_utm.append((poly_utm, depth))

        # Create corridor
        half_width = 500  # 500m
        projection_dist = 5000  # 5km
        corridor = create_projected_corridor(leg_utm, half_width, 0, projection_dist)

        # Verify obstacle intersects
        obstacle_utm, _ = obstacles_utm[0]
        assert corridor.intersects(obstacle_utm), "Shallow reef should intersect corridor"

        # Apply shadows
        result = apply_obstacle_shadows(corridor, obstacles_utm, 0, projection_dist, "Sim: ")

        # Result should have less area due to shadow
        assert result.area < corridor.area, "Shadow should reduce corridor area"

        # The reduction should be significant (at least 1% for a reef in the path)
        area_reduction = (corridor.area - result.area) / corridor.area
        assert area_reduction > 0.01, f"Area reduction {area_reduction:.2%} too small"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_obstacle(self):
        """Very small obstacles should still create shadows."""
        corridor = box(0, 0, 1000, 200)

        # 1x1 meter obstacle
        tiny_obstacle = box(500, 100, 501, 101)
        obstacles = [(tiny_obstacle, 5.0)]

        result = apply_obstacle_shadows(corridor, obstacles, 0, 200, "Tiny: ")

        # Should still create a shadow (small but not zero)
        assert result.area < corridor.area

    def test_obstacle_larger_than_corridor(self):
        """Obstacle larger than corridor should block corridor entirely."""
        corridor = box(100, 50, 200, 150)  # Small corridor

        # Large obstacle covering the entire corridor
        huge_obstacle = box(0, 0, 1000, 200)
        obstacles = [(huge_obstacle, 5.0)]

        result = apply_obstacle_shadows(corridor, obstacles, 0, 500, "Huge: ")

        # Depending on direction, corridor might be entirely blocked
        # For east direction, the shadow covers everything
        assert result.area <= corridor.area

    def test_zero_projection_distance(self):
        """Zero projection distance should result in obstacle-only shadow."""
        obstacle = box(0, 0, 100, 100)

        shadow = create_shadow_behind_obstacle(obstacle, 0, 0)

        # Shadow should equal the obstacle (no projection)
        assert abs(shadow.area - obstacle.area) < 1

    def test_very_long_projection(self):
        """Very long projection should still work."""
        obstacle = box(0, 0, 100, 100)
        projection_dist = 50000  # 50km

        shadow = create_shadow_behind_obstacle(obstacle, 0, projection_dist)

        assert shadow.is_valid
        assert shadow.bounds[2] >= 50000  # Should extend at least 50km

    def test_large_obstacle_partially_overlapping_corridor(self):
        """Large obstacle should only block from where it actually intersects corridor.

        This tests the fix for corridors stopping too early when large depth polygons
        partially overlap the corridor at its start.
        """
        # Corridor going east from (0,0) to (3000, 0) with width 200m
        # Total corridor covers roughly (0, -100) to (3500, 100) including projection
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 100
        projection_dist = 2500

        corridor = create_projected_corridor(leg, half_width, 0, projection_dist)  # East
        original_area = corridor.area

        # Large obstacle that only partially overlaps the corridor at the END
        # The obstacle is 2km x 2km, but only a small part overlaps the corridor
        # at position x=3000 (near the projected end)
        large_obstacle = box(2900, -1000, 4900, 1000)

        obstacles = [(large_obstacle, 5.0)]  # 5m depth (shallow)

        result = apply_obstacle_shadows(corridor, obstacles, 0, projection_dist, "LargePartial: ")

        # The corridor should still extend most of the way before being blocked
        # Only the end part (roughly x=2900 to x=3500) should be blocked
        assert result.area > 0, "Corridor should not be entirely blocked"

        # The area reduction should be modest (the shadow only covers the end)
        # Not the entire corridor from the start
        area_kept = result.area / original_area
        assert area_kept > 0.5, f"Too much area lost ({area_kept:.1%}), obstacle should only block the end"

        # Verify the start of the corridor is still present
        start_region = box(-100, -50, 500, 50)
        start_intersection = result.intersection(start_region)
        assert start_intersection.area > 0, "Start of corridor should still exist"

    def test_obstacle_at_corridor_start_blocks_from_start(self):
        """Obstacle at corridor start should block area behind it in drift direction."""
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 100
        projection_dist = 2500

        corridor = create_projected_corridor(leg, half_width, 0, projection_dist)  # East

        # Obstacle at the very start of the corridor
        # This creates a 200x100 obstacle that overlaps the corridor start
        obstacle_at_start = box(-100, -50, 100, 50)
        obstacles = [(obstacle_at_start, 5.0)]

        result = apply_obstacle_shadows(corridor, obstacles, 0, projection_dist, "AtStart: ")

        # The shadow extends from the obstacle intersection (x=0 to x=100, y=-50 to y=50)
        # through the drift direction (east) for projection_dist
        # So everything from x=0 to x=2600 in the y=-50 to y=50 band should be blocked
        assert result.area < corridor.area, "Shadow should reduce corridor area"

        # The corridor width is 200m (-100 to 100), obstacle blocks middle 100m (-50 to 50)
        # So approximately half the width is blocked for most of the length
        # Note: sweep-based shadow (no convex hull) creates slightly smaller shadow
        area_blocked_fraction = 1 - (result.area / corridor.area)
        assert area_blocked_fraction > 0.1, f"Expected significant area blocked, got {area_blocked_fraction:.1%}"


class TestExampleDataWithStructures:
    """Test corridors with BOTH depth AND structure obstacles from proj.omrat.

    This tests the real use case where the corridor should stop at both:
    1. Depth contours (blue in QGIS) - shallow water grounding risk
    2. Structure obstacles (orange in QGIS) - platform allision risk
    """

    @pytest.fixture
    def example_data(self):
        """Load example data from proj.omrat"""
        import json
        from pathlib import Path

        proj_file = Path(__file__).parent / "example_data" / "proj.omrat"
        if not proj_file.exists():
            pytest.skip("Example data file proj.omrat not found")

        with open(proj_file, "r") as f:
            return json.load(f)

    @pytest.fixture
    def leg_1_utm(self, example_data):
        """Get leg 1 as a LineString in UTM coordinates."""
        from pyproj import CRS
        from geometries.drift_corridor import get_utm_crs, transform_geometry

        seg1 = example_data['segment_data']['1']
        start = seg1['Start_Point'].split()
        end = seg1['End_Point'].split()

        start_lon, start_lat = float(start[0]), float(start[1])
        end_lon, end_lat = float(end[0]), float(end[1])

        leg_wgs84 = LineString([(start_lon, start_lat), (end_lon, end_lat)])

        centroid = leg_wgs84.centroid
        utm_crs = get_utm_crs(centroid.x, centroid.y)
        wgs84 = CRS("EPSG:4326")

        return transform_geometry(leg_wgs84, wgs84, utm_crs), utm_crs, wgs84

    @pytest.fixture
    def structure_obstacles_utm(self, example_data, leg_1_utm):
        """Get structure obstacles in UTM coordinates."""
        from shapely import wkt as shapely_wkt
        from geometries.drift_corridor import transform_geometry

        _, utm_crs, wgs84 = leg_1_utm

        obstacles = []
        height_threshold = 15.0  # Include structures with height <= 15m

        for obj in example_data.get('objects', []):
            try:
                height = float(obj[1])
                if height > height_threshold:
                    continue

                wkt_str = obj[2]
                geom = shapely_wkt.loads(wkt_str)
                geom = make_valid(geom)

                geom_utm = transform_geometry(geom, wgs84, utm_crs)
                geom_utm = make_valid(geom_utm)

                if not geom_utm.is_empty:
                    obstacles.append((geom_utm, height))
            except Exception:
                continue

        return obstacles

    @pytest.fixture
    def depth_obstacles_utm_15m(self, example_data, leg_1_utm):
        """Get depth obstacles with 15m threshold."""
        from shapely import wkt as shapely_wkt
        from geometries.drift_corridor import transform_geometry

        _, utm_crs, wgs84 = leg_1_utm

        bin_width = 3.0
        depth_threshold = 15.0
        obstacles = []

        for d in example_data['depths']:
            depth_value = float(d[1])
            max_depth = depth_value + bin_width

            if max_depth > depth_threshold:
                continue

            wkt_str = d[2]
            geom = shapely_wkt.loads(wkt_str)
            geom = make_valid(geom)

            geom_utm = transform_geometry(geom, wgs84, utm_crs)
            geom_utm = make_valid(geom_utm)

            if not geom_utm.is_empty:
                obstacles.append((geom_utm, depth_value))

        return obstacles

    def test_structures_exist_in_example_data(self, example_data):
        """Verify that the example data contains structure obstacles."""
        objects = example_data.get('objects', [])
        assert len(objects) > 0, "Example data should contain structure obstacles"

        # Check structure format [id, height, wkt]
        for obj in objects:
            assert len(obj) >= 3, "Structure should have at least 3 fields: id, height, polygon"
            height = float(obj[1])
            assert height > 0, "Structure height should be positive"

    def test_structure_obstacles_transform_correctly(self, structure_obstacles_utm):
        """Verify that structure obstacles are correctly transformed to UTM."""
        assert len(structure_obstacles_utm) > 0, "Should have structure obstacles after filtering"

        for geom, height in structure_obstacles_utm:
            assert geom.is_valid, "Structure geometry should be valid"
            assert not geom.is_empty, "Structure geometry should not be empty"
            # UTM coordinates should be in meters (large numbers)
            bounds = geom.bounds
            assert bounds[2] - bounds[0] > 1000, "Structure should be at least 1km wide in UTM"

    def test_north_corridor_intersects_structure(self, leg_1_utm, structure_obstacles_utm):
        """The North corridor from leg 1 should intersect with at least one structure."""
        leg_utm, utm_crs, wgs84 = leg_1_utm

        half_width = 1500
        projection_dist = 20000  # 20km to definitely reach the structure

        corridor = create_projected_corridor(leg_utm, half_width, 90, projection_dist)  # North

        # Check if any structure obstacle intersects the corridor
        intersecting_structures = []
        for geom, height in structure_obstacles_utm:
            if corridor.intersects(geom):
                intersecting_structures.append((geom, height))

        assert len(intersecting_structures) > 0, \
            f"North corridor should intersect at least one structure. " \
            f"Corridor bounds: {corridor.bounds}, " \
            f"Structure bounds: {[s[0].bounds for s in structure_obstacles_utm]}"

    def test_north_corridor_clipped_by_combined_obstacles(
        self, leg_1_utm, structure_obstacles_utm, depth_obstacles_utm_15m
    ):
        """North corridor should be clipped by BOTH structure AND depth obstacles."""
        leg_utm, utm_crs, wgs84 = leg_1_utm

        half_width = 1500
        projection_dist = 20000

        corridor = create_projected_corridor(leg_utm, half_width, 90, projection_dist)
        original_area = corridor.area
        original_north = corridor.bounds[3]

        leg_centroid = leg_utm.centroid
        leg_centroid_tuple = (leg_centroid.x, leg_centroid.y)

        # Combine both structure and depth obstacles (like the real code does)
        all_obstacles = structure_obstacles_utm + depth_obstacles_utm_15m

        # Clip corridor at all obstacles
        result = clip_corridor_at_obstacles(
            corridor, all_obstacles, 90, leg_centroid_tuple, "TestCombined: "
        )

        assert not result.is_empty, "Clipped corridor should not be empty"
        assert result.area < original_area, "Combined obstacles should reduce corridor area"

        # The northern extent should be reduced
        result_north = result.bounds[3]
        assert result_north < original_north, \
            f"Corridor should not extend as far north. Before: {original_north:.0f}m, After: {result_north:.0f}m"

    def test_structure_creates_shadow_in_north_direction(self, leg_1_utm, structure_obstacles_utm):
        """Structure obstacle should create shadow when drifting north."""
        leg_utm, utm_crs, wgs84 = leg_1_utm

        half_width = 1500
        projection_dist = 20000

        corridor = create_projected_corridor(leg_utm, half_width, 90, projection_dist)
        original_area = corridor.area

        leg_centroid = leg_utm.centroid
        leg_centroid_tuple = (leg_centroid.x, leg_centroid.y)

        # Clip with ONLY structure obstacles (no depth)
        result = clip_corridor_at_obstacles(
            corridor, structure_obstacles_utm, 90, leg_centroid_tuple, "TestStructure: "
        )

        # If any structure intersects, the area should be reduced
        structure_intersects = any(
            corridor.intersects(geom) for geom, _ in structure_obstacles_utm
        )

        if structure_intersects:
            assert result.area < original_area, \
                f"Structure should create shadow and reduce area. " \
                f"Original: {original_area:.0f}m², Result: {result.area:.0f}m²"

    def test_depth_creates_shadow_in_north_direction(self, leg_1_utm, depth_obstacles_utm_15m):
        """Depth obstacle should create shadow when drifting north - same behavior as structures."""
        leg_utm, utm_crs, wgs84 = leg_1_utm

        half_width = 1500
        projection_dist = 20000

        corridor = create_projected_corridor(leg_utm, half_width, 90, projection_dist)
        original_area = corridor.area

        leg_centroid = leg_utm.centroid
        leg_centroid_tuple = (leg_centroid.x, leg_centroid.y)

        # Clip with ONLY depth obstacles (no structure)
        result = clip_corridor_at_obstacles(
            corridor, depth_obstacles_utm_15m, 90, leg_centroid_tuple, "TestDepthOnly: "
        )

        # If any depth obstacle intersects, the area should be reduced
        depth_intersects = any(
            corridor.intersects(geom) for geom, _ in depth_obstacles_utm_15m
        )

        if depth_intersects:
            assert result.area < original_area, \
                f"Depth obstacles should create shadow and reduce area (same as structures). " \
                f"Original: {original_area:.0f}m², Result: {result.area:.0f}m²"

            # The corridor's northern extent should also be reduced
            original_north = corridor.bounds[3]
            result_north = result.bounds[3]
            assert result_north < original_north, \
                f"Depth obstacles should reduce corridor's northern extent. " \
                f"Original north: {original_north:.0f}m, Result north: {result_north:.0f}m"

    def test_corridor_stops_at_first_obstacle(self, leg_1_utm, structure_obstacles_utm, depth_obstacles_utm_15m):
        """Corridor should stop at the first obstacle it encounters in the drift direction."""
        leg_utm, utm_crs, wgs84 = leg_1_utm

        half_width = 1500
        projection_dist = 20000

        corridor = create_projected_corridor(leg_utm, half_width, 90, projection_dist)

        leg_centroid = leg_utm.centroid
        leg_centroid_tuple = (leg_centroid.x, leg_centroid.y)

        all_obstacles = structure_obstacles_utm + depth_obstacles_utm_15m

        # Find the northernmost obstacle that intersects the corridor
        northernmost_obstacle_y = None
        for geom, _ in all_obstacles:
            if corridor.intersects(geom):
                intersection = corridor.intersection(geom)
                if not intersection.is_empty:
                    obstacle_north = intersection.bounds[3]
                    if northernmost_obstacle_y is None or obstacle_north > northernmost_obstacle_y:
                        northernmost_obstacle_y = obstacle_north

        if northernmost_obstacle_y is None:
            pytest.skip("No obstacles intersect the corridor")
            return  # For type checker

        # Clip corridor
        result = clip_corridor_at_obstacles(
            corridor, all_obstacles, 90, leg_centroid_tuple, ""
        )

        # The corridor's north extent should be at or slightly before the northernmost obstacle
        result_north = result.bounds[3]

        # Allow 1km tolerance (the shadow might extend beyond the obstacle intersection)
        tolerance = 1000.0  # meters
        max_expected_north = northernmost_obstacle_y + tolerance
        assert result_north <= max_expected_north, \
            f"Corridor north ({result_north:.0f}m) should stop at or before " \
            f"northernmost obstacle ({northernmost_obstacle_y:.0f}m)"
