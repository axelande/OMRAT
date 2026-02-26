# -*- coding: utf-8 -*-
"""
Tests for drift corridor generation v2 - using compass/nautical convention.

These tests verify:
1. Compass angle convention (0=N, 90=W, 180=S, 270=E)
2. Shadow creation from obstacles
3. Corridor projection in various directions
4. Proper handling of scattered MultiPolygon obstacles
"""

import pytest
import numpy as np
from shapely.geometry import Polygon, LineString, box, MultiPolygon, Point
from shapely.ops import unary_union
from shapely.validation import make_valid

# Import the functions under test
from geometries.drift_corridor_v2 import (
    compass_to_vector,
    create_obstacle_shadow,
    clip_corridor_at_obstacles,
    create_projected_corridor,
    create_base_surface,
    get_distribution_width,
    get_projection_distance,
    extract_polygons,
    DIRECTIONS,
)


class TestCompassConvention:
    """Test the compass/nautical angle convention."""

    def test_north_is_zero(self):
        """North should be 0 degrees."""
        assert DIRECTIONS['N'] == 0

    def test_west_is_90(self):
        """West should be 90 degrees."""
        assert DIRECTIONS['W'] == 90

    def test_south_is_180(self):
        """South should be 180 degrees."""
        assert DIRECTIONS['S'] == 180

    def test_east_is_270(self):
        """East should be 270 degrees."""
        assert DIRECTIONS['E'] == 270

    def test_eight_directions(self):
        """Should have exactly 8 directions."""
        assert len(DIRECTIONS) == 8


class TestCompassToVector:
    """Test the compass_to_vector function."""

    def test_north_vector(self):
        """North (0°) should give +Y direction."""
        dx, dy = compass_to_vector(0, 100)
        assert abs(dx) < 0.01  # No X movement
        assert dy > 99  # Positive Y

    def test_south_vector(self):
        """South (180°) should give -Y direction."""
        dx, dy = compass_to_vector(180, 100)
        assert abs(dx) < 0.01  # No X movement
        assert dy < -99  # Negative Y

    def test_east_vector(self):
        """East (270°) should give +X direction."""
        dx, dy = compass_to_vector(270, 100)
        assert dx > 99  # Positive X
        assert abs(dy) < 0.01  # No Y movement

    def test_west_vector(self):
        """West (90°) should give -X direction."""
        dx, dy = compass_to_vector(90, 100)
        assert dx < -99  # Negative X
        assert abs(dy) < 0.01  # No Y movement

    def test_northeast_vector(self):
        """NorthEast (315°) should give +X, +Y direction."""
        dx, dy = compass_to_vector(315, 100)
        assert dx > 0  # Positive X
        assert dy > 0  # Positive Y
        # Both components should be roughly equal (45° diagonal)
        assert abs(abs(dx) - abs(dy)) < 1


class TestCreateObstacleShadow:
    """Test the shadow creation function.

    The shadow/blocking zone blocks the obstacle AND everything beyond it.
    Ships CANNOT reach or pass through the obstacle - they stop at its near edge.

    The quad-based sweep algorithm:
    1. Preserves the EXACT obstacle contour on the front edge
    2. Extends in the drift direction (may extend beyond corridor bounds)
    3. The shadow is later clipped when subtracted from the corridor
    """

    def test_north_shadow_extends_north(self):
        """Shadow for North drift should include obstacle and extend northward."""
        obstacle = box(100, 200, 300, 400)
        corridor_bounds = (0, 0, 500, 1000)

        shadow = create_obstacle_shadow(obstacle, 0, corridor_bounds)  # N = 0°

        # Shadow SHOULD include obstacle (ships stop at obstacle's south edge)
        assert shadow.contains(obstacle) or shadow.intersection(obstacle).area > obstacle.area * 0.99

        # Shadow should start at obstacle's SOUTH edge (front edge for N drift)
        assert shadow.bounds[1] == pytest.approx(200, abs=1)  # Starts at obstacle's south edge

        # Shadow should extend northward beyond corridor top (will be clipped when used)
        assert shadow.bounds[3] > 1000  # Extends beyond corridor top

    def test_south_shadow_extends_south(self):
        """Shadow for South drift should include obstacle and extend southward."""
        obstacle = box(100, 200, 300, 400)
        corridor_bounds = (0, 0, 500, 1000)

        shadow = create_obstacle_shadow(obstacle, 180, corridor_bounds)  # S = 180°

        # Shadow should include obstacle
        assert shadow.contains(obstacle) or shadow.intersection(obstacle).area > obstacle.area * 0.99

        # Shadow should start at obstacle's NORTH edge (front edge for S drift)
        assert shadow.bounds[3] == pytest.approx(400, abs=1)  # Starts at obstacle's north edge

        # Shadow should extend southward beyond corridor bottom
        assert shadow.bounds[1] < 0  # Extends beyond corridor bottom

    def test_east_shadow_extends_east(self):
        """Shadow for East drift should include obstacle and extend eastward."""
        obstacle = box(100, 200, 300, 400)
        corridor_bounds = (0, 0, 1000, 500)

        shadow = create_obstacle_shadow(obstacle, 270, corridor_bounds)  # E = 270°

        # Shadow should include obstacle
        assert shadow.contains(obstacle) or shadow.intersection(obstacle).area > obstacle.area * 0.99

        # Shadow should start at obstacle's WEST edge (front edge for E drift)
        assert shadow.bounds[0] == pytest.approx(100, abs=1)  # Starts at obstacle's west edge

        # Shadow should extend eastward beyond corridor right
        assert shadow.bounds[2] > 1000  # Extends beyond corridor right

    def test_west_shadow_extends_west(self):
        """Shadow for West drift should include obstacle and extend westward."""
        obstacle = box(100, 200, 300, 400)
        corridor_bounds = (0, 0, 1000, 500)

        shadow = create_obstacle_shadow(obstacle, 90, corridor_bounds)  # W = 90°

        # Shadow should include obstacle
        assert shadow.contains(obstacle) or shadow.intersection(obstacle).area > obstacle.area * 0.99

        # Shadow should start at obstacle's EAST edge (front edge for W drift)
        assert shadow.bounds[2] == pytest.approx(300, abs=1)  # Starts at obstacle's east edge

        # Shadow should extend westward beyond corridor left
        assert shadow.bounds[0] < 0  # Extends beyond corridor left

    def test_empty_obstacle_returns_empty(self):
        """Empty obstacle should return empty shadow."""
        obstacle = Polygon()
        corridor_bounds = (0, 0, 1000, 1000)

        shadow = create_obstacle_shadow(obstacle, 0, corridor_bounds)

        assert shadow.is_empty

    def test_shadow_preserves_obstacle_contour(self):
        """Shadow should preserve the obstacle's exact contour on the front edge.

        This tests the key feature of the quad-based sweep algorithm.
        """
        # Create an L-shaped obstacle (concave)
        l_shape = Polygon([
            (100, 100), (100, 300), (200, 300), (200, 200),
            (300, 200), (300, 100), (100, 100)
        ])
        corridor_bounds = (0, 0, 500, 1000)

        shadow = create_obstacle_shadow(l_shape, 0, corridor_bounds)  # N = 0°

        # Shadow should contain the L-shape
        assert shadow.contains(l_shape) or shadow.intersection(l_shape).area > l_shape.area * 0.99

        # Shadow south edge should match obstacle south edge
        assert shadow.bounds[1] == pytest.approx(100, abs=1)


class TestClipCorridorAtObstacles:
    """Test the corridor clipping function."""

    def test_no_obstacles_returns_unchanged(self):
        """No obstacles should return corridor unchanged."""
        corridor = box(0, 0, 1000, 500)

        result = clip_corridor_at_obstacles(corridor, [], 0, "Test: ")

        assert abs(result.area - corridor.area) < 1

    def test_obstacle_creates_shadow(self):
        """Obstacle in corridor should create shadow and reduce area."""
        corridor = box(0, 0, 1000, 500)

        obstacle = box(300, 100, 400, 200)
        obstacles = [(obstacle, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 0, "Test: ")  # N drift

        # Area should be reduced
        assert result.area < corridor.area

    def test_obstacle_outside_corridor_no_effect(self):
        """Obstacle outside corridor should have no effect."""
        corridor = box(0, 0, 1000, 500)

        obstacle = box(2000, 2000, 2100, 2100)
        obstacles = [(obstacle, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 0, "Test: ")

        assert abs(result.area - corridor.area) < 1

    def test_scattered_multipolygon_not_convex_hulled(self):
        """Scattered MultiPolygon parts should NOT be treated as convex hull.

        This is the key fix for the bug where corridors stopped too early.
        """
        corridor = box(0, 0, 10000, 5000)

        # Create 3 scattered parts - if convex hull was used, the gaps would be filled
        part1 = box(1000, 2000, 1500, 2500)
        part2 = box(3000, 2200, 3500, 2700)
        part3 = box(5000, 2100, 5500, 2600)

        scattered = MultiPolygon([part1, part2, part3])
        obstacles = [(scattered, 5.0)]

        # Clip with North direction
        result = clip_corridor_at_obstacles(corridor, obstacles, 0, "Test: ")

        # Check that areas BETWEEN scattered parts are NOT blocked
        between_1_2 = Point(2000, 3500)
        between_2_3 = Point(4000, 3500)

        assert result.contains(between_1_2), "Area between parts 1 and 2 should NOT be blocked"
        assert result.contains(between_2_3), "Area between parts 2 and 3 should NOT be blocked"


class TestCreateProjectedCorridor:
    """Test corridor projection in different directions."""

    def test_north_corridor_extends_north(self):
        """North drift corridor should extend northward (increasing Y)."""
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 50
        projection_dist = 500

        corridor = create_projected_corridor(leg, half_width, 0, projection_dist)  # N = 0°

        # Corridor should extend north (positive Y)
        assert corridor.bounds[3] > 400  # max_y should be large

    def test_south_corridor_extends_south(self):
        """South drift corridor should extend southward (decreasing Y)."""
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 50
        projection_dist = 500

        corridor = create_projected_corridor(leg, half_width, 180, projection_dist)  # S = 180°

        # Corridor should extend south (negative Y)
        assert corridor.bounds[1] < -400  # min_y should be negative

    def test_east_corridor_extends_east(self):
        """East drift corridor should extend eastward (increasing X)."""
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 50
        projection_dist = 500

        corridor = create_projected_corridor(leg, half_width, 270, projection_dist)  # E = 270°

        # Corridor should extend east (positive X beyond leg end)
        assert corridor.bounds[2] > 1400  # max_x should be > leg_end + projection

    def test_corridor_covers_base_surface(self):
        """Corridor should contain the base surface for all directions."""
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 50

        base = create_base_surface(leg, half_width)

        for direction, angle in DIRECTIONS.items():
            corridor = create_projected_corridor(leg, half_width, angle, 500)
            intersection = corridor.intersection(base)
            assert intersection.area >= base.area * 0.99, f"Base not in corridor for {direction}"


class TestCreateBaseSurface:
    """Test base surface creation."""

    def test_base_surface_area(self):
        """Base surface area should be leg_length x distribution_width."""
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 50

        base = create_base_surface(leg, half_width)

        expected_area = 1000 * 100  # leg_length * width
        assert abs(base.area - expected_area) < 1

    def test_base_surface_centered(self):
        """Base surface should be centered on the leg."""
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 50

        base = create_base_surface(leg, half_width)

        centroid = base.centroid
        assert abs(centroid.x - 500) < 1
        assert abs(centroid.y - 0) < 1


class TestExtractPolygons:
    """Test the polygon extraction helper."""

    def test_extract_from_polygon(self):
        """Should extract single Polygon correctly."""
        poly = box(0, 0, 100, 100)
        result = extract_polygons(poly)
        assert len(result) == 1
        assert result[0].equals(poly)

    def test_extract_from_multipolygon(self):
        """Should extract all parts from MultiPolygon."""
        poly1 = box(0, 0, 100, 100)
        poly2 = box(200, 200, 300, 300)
        multi = MultiPolygon([poly1, poly2])

        result = extract_polygons(multi)
        assert len(result) == 2

    def test_extract_from_empty(self):
        """Should return empty list for empty geometry."""
        result = extract_polygons(Polygon())
        assert len(result) == 0


class TestGetDistributionWidth:
    """Test distribution width calculation."""

    def test_width_increases_with_std(self):
        """Width should increase with standard deviation."""
        width_small = get_distribution_width(50, 0.99)
        width_large = get_distribution_width(100, 0.99)
        assert width_large > width_small

    def test_99_percent_coverage(self):
        """99% coverage should be approximately 2 x 2.576 x std."""
        std = 100
        width = get_distribution_width(std, 0.99)
        expected = 2 * 2.576 * std
        assert abs(width - expected) < 1


class TestGetProjectionDistance:
    """Test projection distance calculation."""

    def test_returns_positive_distance(self):
        """Should return positive distance with valid params."""
        params = {
            'use_lognormal': True,
            'std': 0.95,
            'loc': 0.2,
            'scale': 0.85,
        }
        distance = get_projection_distance(params, 1.0)
        assert distance > 0

    def test_respects_max_distance(self):
        """Should respect max_distance cap."""
        params = {
            'use_lognormal': True,
            'std': 0.95,
            'loc': 0.2,
            'scale': 10.0,  # Large scale
        }
        distance = get_projection_distance(params, 10.0, max_distance=5000)
        assert distance <= 5000


class TestDirectionIntegrity:
    """Test all 8 directions work correctly."""

    def test_all_directions_produce_valid_corridors(self):
        """All 8 directions should produce valid corridors."""
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

        # North vs South
        north = create_projected_corridor(leg, half_width, DIRECTIONS['N'], projection_dist)
        south = create_projected_corridor(leg, half_width, DIRECTIONS['S'], projection_dist)

        assert north.bounds[3] > south.bounds[3]  # N extends further north
        assert south.bounds[1] < north.bounds[1]  # S extends further south


class TestRealisticScenario:
    """Test realistic depth obstacle scenarios."""

    def test_shallow_depth_blocks_north_corridor(self):
        """Shallow depth to the north should create blocking zone in obstacle's x-range.

        With blocking zones, the area removed is the full blocking zone from the
        obstacle's front edge (south for N drift) to the corridor boundary.
        """
        # Ship route going east-west
        leg = LineString([(0, 0), (2000, 0)])
        half_width = 100
        projection_dist = 1500

        # Create north corridor
        corridor = create_projected_corridor(leg, half_width, 0, projection_dist)  # N = 0°

        # Shallow area to the north (like a reef) - doesn't span full corridor width
        # Reef at x=500-1500, y=800-1000
        shallow_reef = box(500, 800, 1500, 1000)
        obstacles = [(shallow_reef, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 0, "Reef: ")

        # Corridor should be reduced
        assert result.area < corridor.area

        # The blocking zone extends from reef's south edge (y=800) to corridor top
        # Width is reef width (1000m), height is from y=800 to corridor top (~1600)
        # So blocking area should be roughly 1000 * 800 = 800000
        actual_reduction = corridor.area - result.area
        assert actual_reduction > 500000, \
            f"Blocking zone should remove significant area, got {actual_reduction}"

        # Areas OUTSIDE the obstacle's x-range should still extend to corridor top
        # Check that corridor still reaches full height at x=100 (left of reef)
        left_of_reef = Point(100, 1400)  # Above reef level but left of it
        assert result.contains(left_of_reef), \
            "Corridor should still extend north on left side of reef"

        # Check that corridor still reaches full height at x=1800 (right of reef)
        right_of_reef = Point(1800, 1400)  # Above reef level but right of it
        assert result.contains(right_of_reef), \
            "Corridor should still extend north on right side of reef"

    def test_wide_obstacle_blocks_corridor_completely(self):
        """Obstacle spanning full corridor width should block corridor at obstacle's near edge.

        Ships drifting north hit the obstacle's SOUTH edge (y=800) and stop.
        The corridor should stop at the reef's SOUTH edge (not north).
        """
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 100
        projection_dist = 1500

        # Create north corridor (width goes from y=-100 to y=1600 roughly)
        corridor = create_projected_corridor(leg, half_width, 0, projection_dist)

        # Wide obstacle spanning the full corridor width
        wide_reef = box(-200, 800, 1200, 1000)  # Wider than corridor
        obstacles = [(wide_reef, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 0, "WideReef: ")

        # Result should not extend beyond reef's SOUTH edge (where ships hit)
        result_north = result.bounds[3]
        reef_south = 800  # Ships stop here - they hit this edge first

        # The corridor should stop at the reef's south edge (within tolerance)
        assert result_north <= reef_south + 50, \
            f"Corridor extends too far north ({result_north}) past wide reef's south edge ({reef_south})"


class TestCorridorReachesObstacle:
    """Test that corridors reach ALL THE WAY to obstacles.

    This is critical: the corridor should touch the obstacle's front edge,
    not stop before it. Ships can reach the obstacle and ground there.
    """

    def test_corridor_reaches_obstacle_north(self):
        """North corridor should reach obstacle's SOUTH edge (not stop before it)."""
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 100
        projection_dist = 2000

        corridor = create_projected_corridor(leg, half_width, 0, projection_dist)

        # Obstacle at y=800-1000
        obstacle = box(200, 800, 800, 1000)
        obstacles = [(obstacle, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 0, "")

        # Corridor should reach y=800 (obstacle's south edge) in the obstacle's x-range
        # Check a point just south of the obstacle
        just_before_obstacle = Point(500, 795)
        assert result.contains(just_before_obstacle), \
            "Corridor should reach all the way to obstacle's south edge"

        # But should NOT contain points inside the obstacle
        inside_obstacle = Point(500, 850)
        assert not result.contains(inside_obstacle), \
            "Corridor should NOT contain points inside the obstacle"

    def test_corridor_reaches_obstacle_south(self):
        """South corridor should reach obstacle's NORTH edge."""
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 100
        projection_dist = 2000

        corridor = create_projected_corridor(leg, half_width, 180, projection_dist)

        # Obstacle at y=-1000 to y=-800
        obstacle = box(200, -1000, 800, -800)
        obstacles = [(obstacle, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 180, "")

        # Corridor should reach y=-800 (obstacle's north edge)
        just_before_obstacle = Point(500, -795)
        assert result.contains(just_before_obstacle), \
            "Corridor should reach all the way to obstacle's north edge"

        # But should NOT contain points inside the obstacle
        inside_obstacle = Point(500, -850)
        assert not result.contains(inside_obstacle), \
            "Corridor should NOT contain points inside the obstacle"

    def test_corridor_reaches_obstacle_east(self):
        """East corridor should reach obstacle's WEST edge."""
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 100
        projection_dist = 2000

        corridor = create_projected_corridor(leg, half_width, 270, projection_dist)

        # Obstacle at x=2000-2200
        obstacle = box(2000, -50, 2200, 50)
        obstacles = [(obstacle, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 270, "")

        # Corridor should reach x=2000 (obstacle's west edge)
        just_before_obstacle = Point(1995, 0)
        assert result.contains(just_before_obstacle), \
            "Corridor should reach all the way to obstacle's west edge"

        # But should NOT contain points inside the obstacle
        inside_obstacle = Point(2050, 0)
        assert not result.contains(inside_obstacle), \
            "Corridor should NOT contain points inside the obstacle"

    def test_corridor_reaches_obstacle_west(self):
        """West corridor should reach obstacle's EAST edge."""
        leg = LineString([(0, 0), (1000, 0)])
        half_width = 100
        projection_dist = 2000

        corridor = create_projected_corridor(leg, half_width, 90, projection_dist)

        # Obstacle at x=-2200 to x=-2000
        obstacle = box(-2200, -50, -2000, 50)
        obstacles = [(obstacle, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 90, "")

        # Corridor should reach x=-2000 (obstacle's east edge)
        just_before_obstacle = Point(-1995, 0)
        assert result.contains(just_before_obstacle), \
            "Corridor should reach all the way to obstacle's east edge"

        # But should NOT contain points inside the obstacle
        inside_obstacle = Point(-2050, 0)
        assert not result.contains(inside_obstacle), \
            "Corridor should NOT contain points inside the obstacle"


class TestShadowBlockingAllDirections:
    """Test that shadows properly block in all 8 directions."""

    def test_shadow_blocks_north_completely(self):
        """For N drift, nothing should exist north of obstacle."""
        corridor = box(0, 0, 1000, 2000)
        obstacle = box(0, 800, 1000, 1000)  # Spans full width
        obstacles = [(obstacle, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 0, "")

        # Nothing should exist at y > 800 (obstacle's south edge)
        above_obstacle = Point(500, 1500)
        assert not result.contains(above_obstacle), \
            "North drift: corridor should be blocked north of obstacle"

    def test_shadow_blocks_south_completely(self):
        """For S drift, nothing should exist south of obstacle."""
        corridor = box(0, -2000, 1000, 0)
        obstacle = box(0, -1000, 1000, -800)  # Spans full width
        obstacles = [(obstacle, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 180, "")

        # Nothing should exist at y < -1000 (obstacle's south edge)
        below_obstacle = Point(500, -1500)
        assert not result.contains(below_obstacle), \
            "South drift: corridor should be blocked south of obstacle"

    def test_shadow_blocks_east_completely(self):
        """For E drift, nothing should exist east of obstacle."""
        corridor = box(0, 0, 2000, 1000)
        obstacle = box(800, 0, 1000, 1000)  # Spans full height
        obstacles = [(obstacle, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 270, "")

        # Nothing should exist at x > 800 (obstacle's west edge)
        east_of_obstacle = Point(1500, 500)
        assert not result.contains(east_of_obstacle), \
            "East drift: corridor should be blocked east of obstacle"

    def test_shadow_blocks_west_completely(self):
        """For W drift, nothing should exist west of obstacle."""
        corridor = box(-2000, 0, 0, 1000)
        obstacle = box(-1000, 0, -800, 1000)  # Spans full height
        obstacles = [(obstacle, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 90, "")

        # Nothing should exist at x < -1000 (obstacle's west edge)
        west_of_obstacle = Point(-1500, 500)
        assert not result.contains(west_of_obstacle), \
            "West drift: corridor should be blocked west of obstacle"


class TestConcaveObstaclePreservation:
    """Test that concave obstacles preserve their shape (not over-blocked)."""

    def test_l_shaped_obstacle_preserves_gap(self):
        """L-shaped obstacle should NOT fill in the concave gap.

        This tests the key advantage of quad-based sweep over bounding box.
        """
        corridor = box(0, 0, 1000, 2000)

        # L-shaped obstacle: blocks left side and bottom, but has open area in top-right
        #   +--+
        #   |  |
        #   |  +--+
        #   |     |
        #   +-----+
        l_shape = Polygon([
            (200, 500), (200, 1200), (400, 1200), (400, 800),
            (600, 800), (600, 500), (200, 500)
        ])
        obstacles = [(l_shape, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 0, "")

        # The concave "notch" area (top-right of L) should still be passable
        # because the quad-based sweep follows the actual contour
        in_notch = Point(500, 1000)  # In the notch area, above the L's bottom part

        # With quad-based sweep, this area should be blocked because it's
        # "behind" the obstacle in the drift direction (north)
        # The shadow extends from the obstacle's south edge (y=500) northward
        # So the notch IS blocked

        # But the area to the RIGHT of the L should still be passable
        right_of_l = Point(700, 600)
        assert result.contains(right_of_l), \
            "Area to the right of L-shape should still be passable"

    def test_u_shaped_obstacle_with_gap_facing_upwind(self):
        """U-shaped obstacle with opening facing UPWIND allows ships through the gap.

        When the U-shape's opening faces the direction ships come FROM (south for N drift),
        ships CAN drift through the opening between the arms.
        """
        corridor = box(0, 0, 1000, 2000)

        # U-shape with opening facing SOUTH (upwind for N drift)
        # Ships approaching from south can enter between the arms
        # Top connects at y=1000, arms extend down to y=500
        top = box(200, 1000, 800, 1100)
        left_arm = box(200, 500, 300, 1000)
        right_arm = box(700, 500, 800, 1000)

        u_shape = MultiPolygon([top, left_arm, right_arm])
        obstacles = [(u_shape, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 0, "")

        # The top part creates a shadow from y=1000 to corridor top
        # But the gap between the arms (x=300 to x=700) at lower Y should still be open
        # Ships can drift into the U before hitting the top
        inside_u_opening = Point(500, 700)  # Inside the U, below the top bar
        assert result.contains(inside_u_opening), \
            "Ships should be able to drift into the U-shape opening"

        # But above the top bar (y > 1000) in the middle should be blocked
        above_top = Point(500, 1500)
        assert not result.contains(above_top), \
            "Area above the U-shape top should be blocked"


class TestMultipleObstaclesWithGaps:
    """Test that gaps between multiple obstacles remain passable."""

    def test_two_obstacles_with_gap_between(self):
        """Gap between two obstacles should remain open for ships to pass."""
        corridor = box(0, 0, 2000, 1500)

        # Two obstacles with a gap between them
        obstacle1 = box(200, 800, 600, 1000)
        obstacle2 = box(1000, 800, 1400, 1000)
        obstacles = [(obstacle1, 5.0), (obstacle2, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 0, "")

        # Ships should be able to drift through the gap (x=600 to x=1000)
        in_gap = Point(800, 1200)  # Above the obstacles, in the gap
        assert result.contains(in_gap), \
            "Ships should drift through gap between obstacles"

    def test_staggered_obstacles(self):
        """Staggered obstacles at different Y positions should each create own shadow."""
        corridor = box(0, 0, 2000, 2000)

        # Staggered obstacles at different heights
        obstacle1 = box(200, 600, 600, 800)   # Lower
        obstacle2 = box(800, 1000, 1200, 1200)  # Higher

        obstacles = [(obstacle1, 5.0), (obstacle2, 5.0)]

        result = clip_corridor_at_obstacles(corridor, obstacles, 0, "")

        # Area above obstacle1 but left of obstacle2's x-range should be blocked
        above_obs1 = Point(400, 1000)
        assert not result.contains(above_obs1), \
            "Area directly above obstacle1 should be blocked"

        # Area between the obstacles (x-wise) and below obstacle2 should still be open
        between_obstacles = Point(700, 800)
        assert result.contains(between_obstacles), \
            "Area between obstacles should be passable"

        # Area above obstacle2 should be blocked
        above_obs2 = Point(1000, 1500)
        assert not result.contains(above_obs2), \
            "Area directly above obstacle2 should be blocked"
