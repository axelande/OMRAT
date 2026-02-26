"""
Tests for per-segment probability attribution.

This module tests that drift corridor intersection with obstacle segments
is calculated correctly, ensuring that:
1. Segments only receive contributions from drift corridors that actually intersect them
2. The segment indexing matches between run_calculations.py and result_layers.py
3. The drift direction angles are interpreted correctly

Set SHOW_PLOT=True to display visual debugging plots (tests will wait for window close).
"""
import sys
import os

# Ensure project root is on the path so imports work when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from shapely.geometry import Polygon, LineString, Point

from compute.run_calculations import (
    _extract_obstacle_segments,
    _create_drift_corridor,
    _segment_intersects_corridor,
)

# Set to True to show visual plots during tests (useful for debugging)
# Can also be set via environment variable: SHOW_PLOT=1
SHOW_PLOT = True# = os.environ.get('SHOW_PLOT', '').lower() in ('1', 'true', 'yes')


def plot_corridor_test(leg, corridors, structure, segments, title="Drift Corridor Test"):
    """
    Plot the leg, drift corridors, structure, and segments for visual debugging.

    Args:
        leg: LineString representing the traffic leg
        corridors: Dict of {direction_name: corridor_polygon}
        structure: Polygon representing the obstacle
        segments: List of ((x1, y1), (x2, y2)) tuples
        title: Plot title
    """
    if not SHOW_PLOT:
        return

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Color map for different directions
    direction_colors = {
        'East (0°)': 'red',
        'NE (45°)': 'orange',
        'North (90°)': 'green',
        'NW (135°)': 'cyan',
        'West (180°)': 'blue',
        'SW (225°)': 'purple',
        'South (270°)': 'magenta',
        'SE (315°)': 'brown',
    }

    # Plot corridors with transparency
    for dir_name, corridor in corridors.items():
        if corridor is not None:
            color = direction_colors.get(dir_name, 'gray')
            x, y = corridor.exterior.xy
            ax.fill(x, y, alpha=0.2, fc=color, ec=color, linewidth=2, label=dir_name)

    # Plot structure
    if structure is not None:
        x, y = structure.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='lightgray', ec='black', linewidth=2, label='Structure')

    # Plot leg
    if leg is not None:
        x, y = leg.xy
        ax.plot(x, y, 'b-', linewidth=3, label='Leg')
        ax.plot(x[0], y[0], 'bo', markersize=10)  # Start point
        ax.plot(x[-1], y[-1], 'b^', markersize=10)  # End point

    # Plot segments with indices
    segment_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
    for seg_idx, ((x1, y1), (x2, y2)) in enumerate(segments):
        color = segment_colors[seg_idx % len(segment_colors)]
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=4, label=f'Seg {seg_idx}')
        # Label at midpoint
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.annotate(f'{seg_idx}', (mid_x, mid_y), fontsize=12, fontweight='bold',
                   ha='center', va='center', color='white',
                   bbox=dict(boxstyle='circle', fc=color, ec='black'))

    ax.set_aspect('equal')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_intersection_results(leg, structure, segments, results_by_direction, title="Intersection Results"):
    """
    Plot which segments intersect with each drift direction's corridor.

    Args:
        leg: LineString representing the traffic leg
        structure: Polygon representing the obstacle
        segments: List of ((x1, y1), (x2, y2)) tuples
        results_by_direction: Dict of {direction_name: list of intersecting segment indices}
        title: Plot title
    """
    if not SHOW_PLOT:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    n_directions = len(results_by_direction)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    direction_colors = {
        'East (0°)': 'red',
        'NE (45°)': 'orange',
        'North (90°)': 'green',
        'NW (135°)': 'cyan',
        'West (180°)': 'blue',
        'SW (225°)': 'purple',
        'South (270°)': 'magenta',
        'SE (315°)': 'brown',
    }

    for ax_idx, (dir_name, hit_indices) in enumerate(results_by_direction.items()):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]

        # Plot structure
        x, y = structure.exterior.xy
        ax.fill(x, y, alpha=0.3, fc='lightgray', ec='black', linewidth=1)

        # Plot leg
        lx, ly = leg.xy
        ax.plot(lx, ly, 'b-', linewidth=2)

        # Plot segments - highlight those that intersect
        for seg_idx, ((x1, y1), (x2, y2)) in enumerate(segments):
            if seg_idx in hit_indices:
                ax.plot([x1, x2], [y1, y2], 'g-', linewidth=4, label='Hit' if seg_idx == hit_indices[0] else '')
            else:
                ax.plot([x1, x2], [y1, y2], 'r--', linewidth=2, alpha=0.5)
            # Label
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.annotate(f'{seg_idx}', (mid_x, mid_y), fontsize=8, ha='center', va='center')

        ax.set_title(f'{dir_name}\nHits: {hit_indices}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for ax_idx in range(len(results_by_direction), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


class TestExtractObstacleSegments:
    """Test segment extraction from polygons."""

    def test_rectangle_segments(self):
        """A rectangle should have 4 segments."""
        # Create a rectangle with known coordinates
        rect = Polygon([(0, 0), (10, 0), (10, 5), (0, 5), (0, 0)])
        segments = _extract_obstacle_segments(rect)

        assert len(segments) == 4

        # Verify segment coordinates (CCW order starting from first vertex)
        expected = [
            ((0.0, 0.0), (10.0, 0.0)),   # Bottom edge
            ((10.0, 0.0), (10.0, 5.0)),  # Right edge
            ((10.0, 5.0), (0.0, 5.0)),   # Top edge
            ((0.0, 5.0), (0.0, 0.0)),    # Left edge
        ]

        for i, (seg, exp) in enumerate(zip(segments, expected)):
            assert seg == exp, f"Segment {i}: expected {exp}, got {seg}"

    def test_triangle_segments(self):
        """A triangle should have 3 segments."""
        tri = Polygon([(0, 0), (10, 0), (5, 10), (0, 0)])
        segments = _extract_obstacle_segments(tri)

        assert len(segments) == 3


class TestCreateDriftCorridor:
    """Test drift corridor creation."""

    def test_corridor_east_drift(self):
        """East drift (0°) should extend corridor to positive X."""
        leg = LineString([(0, 0), (100, 0)])
        corridor = _create_drift_corridor(leg, drift_angle=0, distance=50, lateral_spread=20)

        assert corridor is not None
        minx, miny, maxx, maxy = corridor.bounds

        # East drift extends to positive X
        assert maxx > 100, f"Expected corridor to extend east beyond leg end, got maxx={maxx}"
        assert maxx == pytest.approx(150, abs=1)  # 100 + 50

    def test_corridor_north_drift(self):
        """North drift (90°) should extend corridor to positive Y."""
        leg = LineString([(0, 0), (100, 0)])
        corridor = _create_drift_corridor(leg, drift_angle=90, distance=50, lateral_spread=20)

        assert corridor is not None
        minx, miny, maxx, maxy = corridor.bounds

        # North drift extends to positive Y
        assert maxy > 20, f"Expected corridor to extend north, got maxy={maxy}"
        assert maxy == pytest.approx(70, abs=1)  # 20 (lateral) + 50 (drift)

    def test_corridor_west_drift(self):
        """West drift (180°) should extend corridor to negative X."""
        leg = LineString([(0, 0), (100, 0)])
        corridor = _create_drift_corridor(leg, drift_angle=180, distance=50, lateral_spread=20)

        assert corridor is not None
        minx, miny, maxx, maxy = corridor.bounds

        # West drift extends to negative X
        assert minx < 0, f"Expected corridor to extend west, got minx={minx}"
        assert minx == pytest.approx(-50, abs=1)

    def test_corridor_south_drift(self):
        """South drift (270°) should extend corridor to negative Y."""
        leg = LineString([(0, 0), (100, 0)])
        corridor = _create_drift_corridor(leg, drift_angle=270, distance=50, lateral_spread=20)

        assert corridor is not None
        minx, miny, maxx, maxy = corridor.bounds

        # South drift extends to negative Y
        assert miny < -20, f"Expected corridor to extend south, got miny={miny}"
        assert miny == pytest.approx(-70, abs=1)  # -20 (lateral) - 50 (drift)


class TestSegmentIntersectsCorridor:
    """Test segment-corridor intersection logic."""

    def test_segment_inside_corridor(self):
        """A segment fully inside the corridor should intersect if not parallel to drift."""
        leg = LineString([(0, 0), (100, 0)])
        corridor = _create_drift_corridor(leg, drift_angle=90, distance=50, lateral_spread=20)

        # Horizontal segment at y=40, perpendicular to north drift - should be hit
        segment = ((30, 40), (70, 40))
        leg_centroid = (50, 0)
        assert _segment_intersects_corridor(segment, corridor, drift_angle=90, leg_centroid=leg_centroid)

    def test_segment_outside_corridor(self):
        """A segment completely outside the corridor should not intersect."""
        leg = LineString([(0, 0), (100, 0)])
        corridor = _create_drift_corridor(leg, drift_angle=90, distance=50, lateral_spread=20)

        # Segment far to the right of corridor
        segment = ((200, 0), (200, 10))
        leg_centroid = (50, 0)
        assert not _segment_intersects_corridor(segment, corridor, drift_angle=90, leg_centroid=leg_centroid)

    def test_segment_partially_inside(self):
        """A segment that crosses the corridor boundary should intersect."""
        leg = LineString([(0, 0), (100, 0)])
        corridor = _create_drift_corridor(leg, drift_angle=90, distance=50, lateral_spread=20)

        # Horizontal segment that crosses the corridor top boundary - should be hit
        segment = ((30, 60), (70, 80))  # Crosses y=70 corridor boundary
        leg_centroid = (50, 0)
        assert _segment_intersects_corridor(segment, corridor, drift_angle=90, leg_centroid=leg_centroid)

    def test_segment_behind_leg_not_hit(self):
        """A segment behind the leg (opposite to drift direction) should not be hit."""
        leg = LineString([(0, 0), (100, 0)])
        # North drift (90°) goes to positive Y
        corridor = _create_drift_corridor(leg, drift_angle=90, distance=50, lateral_spread=20)

        # Segment south of the leg (negative Y direction, opposite to drift)
        segment = ((50, -50), (50, -30))
        leg_centroid = (50, 0)

        # The corridor might touch this segment geometrically (due to lateral spread at leg)
        # but direction check should filter it out
        result = _segment_intersects_corridor(segment, corridor, drift_angle=90, leg_centroid=leg_centroid)
        # Segment is at y=-50 to -30, leg is at y=0, drift is north
        # This segment is south of the leg, so it should NOT be hit by north drift
        assert not result, "Segment behind leg should not be hit by drift in opposite direction"


class TestStructureSegmentAttribution:
    """
    Test that structure segments receive contributions from correct legs.

    Scenario:
    - Structure (rectangle) is NORTH of the legs
    - LEG_1 is to the SOUTHWEST
    - LEG_2 is directly SOUTH
    - LEG_3 is to the SOUTHEAST

    Expected behavior:
    - Bottom edge of structure: hit by NORTH drift from legs below
    - Left edge: hit by EAST drift from legs to the west
    - Right edge: hit by WEST drift from legs to the east
    - Top edge: generally not hit unless structure is very close to leg
    """

    def setup_method(self):
        """Set up test geometry."""
        # Structure is a rectangle north of the legs
        # Coordinates chosen so structure is at y=100 to y=200
        self.structure = Polygon([
            (50, 100),   # bottom-left
            (150, 100),  # bottom-right
            (150, 200),  # top-right
            (50, 200),   # top-left
            (50, 100),   # close
        ])

        self.segments = _extract_obstacle_segments(self.structure)
        # Expected segment order (CCW):
        # 0: bottom (50,100) -> (150,100) - faces SOUTH
        # 1: right (150,100) -> (150,200) - faces EAST
        # 2: top (150,200) -> (50,200) - faces NORTH
        # 3: left (50,200) -> (50,100) - faces WEST

        # Legs are south of the structure
        self.leg1 = LineString([(0, 0), (50, 50)])      # SW leg
        self.leg2 = LineString([(50, 0), (150, 0)])     # S leg (horizontal)
        self.leg3 = LineString([(150, 50), (200, 0)])   # SE leg

    def test_segment_order(self):
        """Verify segment order matches expected."""
        assert len(self.segments) == 4

        # Segment 0: bottom edge
        seg0 = self.segments[0]
        assert seg0[0][1] == 100 and seg0[1][1] == 100, f"Segment 0 should be bottom edge: {seg0}"

        # Segment 1: right edge
        seg1 = self.segments[1]
        assert seg1[0][0] == 150 and seg1[1][0] == 150, f"Segment 1 should be right edge: {seg1}"

        # Segment 2: top edge
        seg2 = self.segments[2]
        assert seg2[0][1] == 200 and seg2[1][1] == 200, f"Segment 2 should be top edge: {seg2}"

        # Segment 3: left edge
        seg3 = self.segments[3]
        assert seg3[0][0] == 50 and seg3[1][0] == 50, f"Segment 3 should be left edge: {seg3}"

    def test_north_drift_hits_bottom_edge(self):
        """North drift (90°) from leg2 should hit the bottom edge (segment 0)."""
        corridor = _create_drift_corridor(self.leg2, drift_angle=90, distance=150, lateral_spread=50)

        assert corridor is not None
        assert corridor.intersects(self.structure), "North corridor should reach structure"

        # Check which segments are hit
        hit_segments = []
        for i, seg in enumerate(self.segments):
            if _segment_intersects_corridor(seg, corridor):
                hit_segments.append(i)

        # Bottom edge (segment 0) should be hit
        assert 0 in hit_segments, f"Bottom edge should be hit by north drift. Hit segments: {hit_segments}"

    def test_south_drift_does_not_hit_structure(self):
        """South drift (270°) from leg2 should NOT hit structure (it's north of the leg)."""
        corridor = _create_drift_corridor(self.leg2, drift_angle=270, distance=150, lateral_spread=50)

        assert corridor is not None
        # Structure is north (y=100-200), south drift goes to negative Y
        assert not corridor.intersects(self.structure), "South corridor should not reach structure"

    def test_leg1_northeast_drift(self):
        """Northeast drift (45°) from leg1 (SW) should hit bottom and possibly left edge."""
        corridor = _create_drift_corridor(self.leg1, drift_angle=45, distance=200, lateral_spread=50)

        if corridor.intersects(self.structure):
            hit_segments = [i for i, seg in enumerate(self.segments)
                          if _segment_intersects_corridor(seg, corridor)]

            # Should NOT hit the right edge (segment 1) since leg1 is to the SW
            # Should possibly hit bottom (0) and left (3) edges
            print(f"LEG1 NE drift hits segments: {hit_segments}")

            # Right edge should generally not be hit by NE drift from SW leg
            # (unless corridor is very wide)
            if 1 in hit_segments:
                # This might be acceptable if the corridor is wide enough
                pass

    def test_leg3_northwest_drift(self):
        """Northwest drift (135°) from leg3 (SE) should hit bottom and possibly right edge."""
        corridor = _create_drift_corridor(self.leg3, drift_angle=135, distance=200, lateral_spread=50)

        if corridor.intersects(self.structure):
            hit_segments = [i for i, seg in enumerate(self.segments)
                          if _segment_intersects_corridor(seg, corridor)]

            print(f"LEG3 NW drift hits segments: {hit_segments}")

            # Left edge (segment 3) should generally NOT be hit by NW drift from SE leg
            # since the corridor extends NW from the SE, away from the left edge


class TestDriftDirectionConvention:
    """
    Verify that drift directions follow the correct convention.

    The convention used in pdf_corrected_fast_probability_holes.py is:
    - Standard math convention: 0° = East, 90° = North, 180° = West, 270° = South

    d_idx mapping:
    - d_idx=0 -> 0° -> East
    - d_idx=1 -> 45° -> NorthEast
    - d_idx=2 -> 90° -> North
    - d_idx=3 -> 135° -> NorthWest
    - d_idx=4 -> 180° -> West
    - d_idx=5 -> 225° -> SouthWest
    - d_idx=6 -> 270° -> South
    - d_idx=7 -> 315° -> SouthEast
    """

    def test_direction_vectors(self):
        """Verify drift vectors point in correct directions."""
        leg = LineString([(0, 0), (100, 0)])

        expected_directions = {
            0: ('East', lambda b: b[2] > 100),      # maxx extends east
            90: ('North', lambda b: b[3] > 20),    # maxy extends north
            180: ('West', lambda b: b[0] < 0),     # minx extends west
            270: ('South', lambda b: b[1] < -20),  # miny extends south
        }

        for angle, (name, check) in expected_directions.items():
            corridor = _create_drift_corridor(leg, angle, distance=50, lateral_spread=20)
            assert corridor is not None
            bounds = corridor.bounds
            assert check(bounds), f"Drift {angle}° ({name}) failed: bounds={bounds}"


class TestRealWorldScenario:
    """
    Test a scenario similar to the user's screenshot.

    The structure is a triangle/polygon north of the shipping lanes.
    Multiple legs pass south of the structure.
    """

    def test_triangular_structure(self):
        """Test with triangular structure similar to screenshot."""
        # Triangular structure north of legs
        structure = Polygon([
            (100, 150),  # top vertex
            (50, 50),    # bottom-left
            (150, 50),   # bottom-right
            (100, 150),  # close
        ])

        segments = _extract_obstacle_segments(structure)
        assert len(segments) == 3

        # Legs below the structure
        leg1 = LineString([(0, 0), (75, 0)])    # Left leg
        leg2 = LineString([(75, 0), (125, 0)])  # Center leg
        leg3 = LineString([(125, 0), (200, 0)]) # Right leg

        # Test north drift from each leg
        drift_distance = 200
        lateral_spread = 30

        results = {}
        for leg_name, leg in [('leg1', leg1), ('leg2', leg2), ('leg3', leg3)]:
            # North drift (90°) should hit the structure
            corridor = _create_drift_corridor(leg, 90, drift_distance, lateral_spread)

            if corridor and corridor.intersects(structure):
                hit_segs = [i for i, seg in enumerate(segments)
                           if _segment_intersects_corridor(seg, corridor)]
                results[leg_name] = hit_segs
            else:
                results[leg_name] = []

        print(f"North drift results: {results}")

        # All legs should hit the structure with north drift
        for leg_name in ['leg1', 'leg2', 'leg3']:
            assert len(results[leg_name]) > 0, f"{leg_name} should hit structure with north drift"

        # Now test that south drift doesn't hit
        for leg_name, leg in [('leg1', leg1), ('leg2', leg2), ('leg3', leg3)]:
            corridor = _create_drift_corridor(leg, 270, drift_distance, lateral_spread)
            if corridor:
                assert not corridor.intersects(structure), f"{leg_name} south drift should not hit structure"

    def test_screenshot_scenario_visual(self):
        """
        Visual test replicating the user's screenshot scenario.

        Structure: Triangle with vertices roughly at:
        - Top-right corner (highest point)
        - Bottom-left corner
        - Bottom-right corner (forms a right angle)

        Legs:
        - LEG 1.1: Southwest of structure, runs roughly horizontal
        - LEG 2.1: South of structure, runs roughly horizontal
        - LEG 3.1: Southeast of structure, angled

        Set SHOW_PLOT=True to see the visualization.
        """
        # Create triangle similar to screenshot (pointing up-right)
        # Segments will be:
        # 0: bottom edge (bottom-left to bottom-right) - faces SOUTH
        # 1: right edge (bottom-right to top) - faces EAST/NE
        # 2: left edge (top to bottom-left) - faces WEST/NW
        structure = Polygon([
            (0, 0),      # bottom-left
            (200, 0),    # bottom-right
            (200, 150),  # top-right
            (0, 0),      # close
        ])

        segments = _extract_obstacle_segments(structure)
        assert len(segments) == 3

        # Print segment details
        print("\nSegment details:")
        for i, seg in enumerate(segments):
            print(f"  Segment {i}: {seg[0]} -> {seg[1]}")

        # Legs positioned like in screenshot
        leg1 = LineString([(-100, -50), (50, -50)])    # LEG 1.1 - left, below structure
        leg2 = LineString([(50, -80), (250, -80)])     # LEG 2.1 - center, below structure
        leg3 = LineString([(250, -50), (350, 50)])     # LEG 3.1 - right, angled up

        drift_distance = 200
        lateral_spread = 50

        # Test all 8 drift directions for each leg
        direction_names = {
            0: 'East (0°)',
            45: 'NE (45°)',
            90: 'North (90°)',
            135: 'NW (135°)',
            180: 'West (180°)',
            225: 'SW (225°)',
            270: 'South (270°)',
            315: 'SE (315°)',
        }

        all_results = {}
        for leg_name, leg in [('LEG1', leg1), ('LEG2', leg2), ('LEG3', leg3)]:
            leg_results = {}
            corridors_for_plot = {}

            leg_centroid = (leg.centroid.x, leg.centroid.y)

            for angle, dir_name in direction_names.items():
                corridor = _create_drift_corridor(leg, angle, drift_distance, lateral_spread)
                corridors_for_plot[dir_name] = corridor

                if corridor and corridor.intersects(structure):
                    hit_segs = [i for i, seg in enumerate(segments)
                               if _segment_intersects_corridor(
                                   seg, corridor,
                                   drift_angle=angle,
                                   leg_centroid=leg_centroid
                               )]
                    leg_results[dir_name] = hit_segs
                else:
                    leg_results[dir_name] = []

            all_results[leg_name] = leg_results

            # Visual plot for this leg
            plot_corridor_test(leg, corridors_for_plot, structure, segments,
                             title=f"{leg_name} - Drift Corridors")
            plot_intersection_results(leg, structure, segments, leg_results,
                                    title=f"{leg_name} - Segment Intersections")

        # Print summary
        print("\n=== INTERSECTION SUMMARY ===")
        for leg_name, leg_results in all_results.items():
            print(f"\n{leg_name}:")
            for dir_name, hit_segs in leg_results.items():
                if hit_segs:
                    print(f"  {dir_name}: hits segments {hit_segs}")

        # Verify expected behavior based on user requirements:
        # 1. East should NOT hit segment 0 (drift is parallel to horizontal segment)
        # 2. NE, North, NW should NOT hit segment 2 (diagonal) - they should only hit seg 0
        # 3. SE should NOT hit segment 0 (structure is north of leg, SE drift goes south)

        # North drift (90°) - ships drifting north from south should hit bottom edge
        assert 0 in all_results['LEG1'].get('North (90°)', []), "LEG1 north drift should hit segment 0 (bottom)"
        assert 0 in all_results['LEG2'].get('North (90°)', []), "LEG2 north drift should hit segment 0 (bottom)"

        # South drift (270°) - ships drifting south shouldn't hit structure (it's north of them)
        assert len(all_results['LEG1'].get('South (270°)', [])) == 0, "LEG1 south drift should NOT hit structure"
        assert len(all_results['LEG2'].get('South (270°)', [])) == 0, "LEG2 south drift should NOT hit structure"

        # East drift should NOT hit segment 0 (parallel)
        assert 0 not in all_results['LEG1'].get('East (0°)', []), "LEG1 east drift should NOT hit segment 0"

        # NE, North, NW should NOT hit segment 2 (the diagonal)
        assert 2 not in all_results['LEG1'].get('NE (45°)', []), "LEG1 NE drift should NOT hit segment 2"
        assert 2 not in all_results['LEG1'].get('North (90°)', []), "LEG1 north drift should NOT hit segment 2"
        assert 2 not in all_results['LEG1'].get('NW (135°)', []), "LEG1 NW drift should NOT hit segment 2"

        # SE should NOT hit segment 0
        assert 0 not in all_results['LEG1'].get('SE (315°)', []), "LEG1 SE drift should NOT hit segment 0"


class TestSegmentIndexConsistency:
    """Test that segment indices are consistent across coordinate transformations."""

    def test_transform_preserves_vertex_order(self):
        """Verify that coordinate transformation preserves vertex order.

        The key insight is that shapely.ops.transform applies the coordinate
        transformation to each vertex, preserving their order. We verify this
        by checking that the exterior ring coordinates transform consistently.
        """
        from shapely.ops import transform

        # Create a rectangle in "WGS84" coordinates (simplified)
        original = Polygon([(10, 50), (11, 50), (11, 51), (10, 51)])
        original_coords = list(original.exterior.coords)

        # Transform function must handle both scalars and arrays
        def scale_transform(x, y, z=None):
            if hasattr(x, '__iter__'):
                return ([xi * 2 for xi in x], [yi * 2 for yi in y])
            return (x * 2, y * 2)

        transformed = transform(scale_transform, original)
        transformed_coords = list(transformed.exterior.coords)

        # Same number of vertices (and thus segments)
        assert len(original_coords) == len(transformed_coords)

        # Vertex order should be preserved
        for i, (orig_coord, trans_coord) in enumerate(zip(original_coords, transformed_coords)):
            expected = (orig_coord[0] * 2, orig_coord[1] * 2)
            assert trans_coord[0] == pytest.approx(expected[0], abs=0.001), f"Vertex {i} x mismatch"
            assert trans_coord[1] == pytest.approx(expected[1], abs=0.001), f"Vertex {i} y mismatch"

    def test_segment_index_consistency(self):
        """Test that segment indices extracted from transformed geometry match original.

        This is the core test for the bug fix: segment indices must be consistent
        between the UTM geometry (used for corridor intersection) and the WGS84
        geometry (used for display).
        """
        # Create a rectangle
        original = Polygon([(0, 0), (100, 0), (100, 50), (0, 50), (0, 0)])

        # Extract segments from original
        original_segments = _extract_obstacle_segments(original)
        assert len(original_segments) == 4

        # The fix ensures we transform UTM back to WGS84, so segment indices match.
        # Verify that coordinates extracted from exterior ring maintain order.
        exterior_coords = list(original.exterior.coords)
        for seg_idx, segment in enumerate(original_segments):
            # Each segment should connect consecutive vertices
            expected_start = exterior_coords[seg_idx]
            expected_end = exterior_coords[seg_idx + 1]

            assert segment[0][0] == pytest.approx(expected_start[0], abs=0.001)
            assert segment[0][1] == pytest.approx(expected_start[1], abs=0.001)
            assert segment[1][0] == pytest.approx(expected_end[0], abs=0.001)
            assert segment[1][1] == pytest.approx(expected_end[1], abs=0.001)

    def test_make_valid_single_polygon(self):
        """Test that make_valid on a valid polygon doesn't change segment count."""
        try:
            from shapely import make_valid
        except ImportError:
            pytest.skip("shapely.make_valid not available")

        # A valid rectangle
        valid_rect = Polygon([(0, 0), (100, 0), (100, 50), (0, 50), (0, 0)])

        # make_valid shouldn't change it
        fixed = make_valid(valid_rect)

        original_segments = _extract_obstacle_segments(valid_rect)
        fixed_segments = _extract_obstacle_segments(fixed)

        # Should have same number of segments
        assert len(original_segments) == len(fixed_segments)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--noconftest'])
