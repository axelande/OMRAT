"""
Tests for Structure 1 segment attribution using actual proj.omrat data.

This test verifies that the per-segment probability calculations are logically
correct using the real geometry data from the OMRAT project file.

Structure 1 geometry (WGS84):
    Polygon ((13.89417 55.23217, 14.25963 55.25552, 14.29743 55.31652, 14.12416 55.30397, 13.89417 55.23217))

    The structure is roughly quadrilateral with vertices:
    - V0: (13.89417, 55.23217)  - Southwest corner
    - V1: (14.25963, 55.25552)  - Southeast corner
    - V2: (14.29743, 55.31652)  - Northeast corner
    - V3: (14.12416, 55.30397)  - Northwest corner

    Segments (CCW order):
    - Seg 0: V0 -> V1 (bottom edge, faces roughly SOUTH)
    - Seg 1: V1 -> V2 (right edge, faces roughly EAST)
    - Seg 2: V2 -> V3 (top edge, faces roughly NORTH)
    - Seg 3: V3 -> V0 (left edge, faces roughly WEST/SW)

Legs (WGS84):
    - Leg 1: (13.295152, 55.224254) -> (14.158810, 55.187230) - SW of structure, runs W-E
    - Leg 2: (14.15881, 55.18723) -> (14.29113, 55.17644)     - S of structure, runs W-E
    - Leg 3: (14.291130, 55.176440) -> (14.619750, 55.421273) - SE of structure, runs SW-NE
    - Leg 4: (14.619750, 55.421273) -> (15.091956, 55.560522) - E of structure, runs SW-NE

Expected logical behavior:
    - North drift (90°): Ships drift northward
        - Leg 1 (SW): Could hit bottom edge (seg 0) via corridor
        - Leg 2 (S): Should hit bottom edge (seg 0) - directly below
        - Leg 3 (SE): Could hit bottom/right edge
        - Leg 4 (E): Unlikely to hit - too far east

    - South drift (270°): Ships drift southward
        - Legs 1-4 are all south of structure, so south drift should NOT hit it

    - East drift (0°): Ships drift eastward
        - Leg 1 (SW): Could hit left edge (seg 3) if corridor reaches
        - Legs 2-4: Less likely - they're to the east/south already

    - West drift (180°): Ships drift westward
        - Leg 3/4 (E): Could hit right edge (seg 1)
        - Legs 1-2: Unlikely - west drift goes away from structure
"""
import sys
import os

# Ensure project root is on the path so imports work when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import transform as shapely_transform
from scipy.stats import norm
import geopandas as gpd

from compute.run_calculations import (
    _extract_obstacle_segments,
    _create_drift_corridor,
    _segment_intersects_corridor,
)
from geometries.calculate_probability_holes import compute_probability_holes
from compute.basic_equations import get_not_repaired

# Set to True to show visual plots during tests
SHOW_PLOT  = os.environ.get('SHOW_PLOT', '').lower() in ('1', 'true', 'yes')


# Default parameters matching actual QGIS/OMRAT proj.omrat settings
# These values come from tests/example_data/proj.omrat
# Per-leg total ship frequencies from proj.omrat (summed over all ship types and directions)
LEG_FREQUENCIES = {
    'Leg1': 3371,   # ships/year
    'Leg2': 6221,   # ships/year
    'Leg3': 8010,   # ships/year
    'Leg4': 10097,  # ships/year
}

DEFAULT_TRAFFIC_PARAMS = {
    'ship_frequency': 340,  # ships per year (fallback if leg not in LEG_FREQUENCIES)
    'ship_speed_kts': 12.0,  # knots (typical speed)
    'drift_p': 1.0,  # Annual blackout probability per ship (drift_p from proj.omrat)
}

DEFAULT_DRIFT_PARAMS = {
    'drift_speed': 1.0,  # m/s (1.9438 knots from proj.omrat)
    'repair': {
        'use_lognormal': 1,
        'std': 0.95,  # From proj.omrat
        'loc': 0.2,   # From proj.omrat
        'scale': 0.85,  # From proj.omrat (hours)
    },
}

# Wind rose - equal probability for all 8 directions (from proj.omrat)
DEFAULT_WIND_ROSE = {i * 45: 0.125 for i in range(8)}  # 0, 45, 90, ..., 315


def calculate_allision_probability(
    legs: dict,
    structure: Polygon,
    drift_distance: float,
    lateral_spread: float,
    traffic_params: dict | None = None,
    drift_params: dict | None = None,
    wind_rose: dict | None = None,
) -> dict:
    """
    Calculate allision probability using the same method as QGIS/OMRAT.

    This replicates the cascade calculation from compute/run_calculations.py:
        contrib = base * rp * remaining_prob * hole_pct * p_nr

    Where:
        - base = hours_present * blackout_per_hour
        - rp = wind rose probability for this direction
        - remaining_prob = cumulative probability ship reached this obstacle
        - hole_pct = probability_holes[leg][dir][structure] from geometric calculation
        - p_nr = probability not repaired before hitting (distance decay)

    Args:
        legs: Dict of leg LineStrings (in UTM coordinates)
        structure: Structure polygon (in UTM coordinates)
        drift_distance: Maximum drift distance in meters
        lateral_spread: Lateral spread (half-width) in meters
        traffic_params: Traffic parameters (frequency, speed, blackout prob)
        drift_params: Drift parameters (drift speed, repair distribution)
        wind_rose: Wind rose probabilities by direction (0-315 in 45° steps)

    Returns:
        dict: {leg_name: {angle: allision_probability}}
    """
    if traffic_params is None:
        traffic_params = DEFAULT_TRAFFIC_PARAMS
    if drift_params is None:
        drift_params = DEFAULT_DRIFT_PARAMS
    if wind_rose is None:
        wind_rose = DEFAULT_WIND_ROSE

    # Calculate std_dev from lateral_spread (lateral_spread is 5-sigma)
    std_dev = lateral_spread / 5.0

    # Prepare data for compute_probability_holes_smart_hybrid
    leg_list = list(legs.values())
    leg_names = list(legs.keys())

    # Create distributions and weights matching QGIS format
    distributions = []
    weights = []
    for _ in leg_list:
        distributions.append([norm(loc=0, scale=std_dev)])
        weights.append([1.0])

    # Create GeoDataFrame for structure
    structure_gdf = gpd.GeoDataFrame(geometry=[structure])

    # Calculate probability holes using accurate DBLQUAD integration
    # This uses scipy.dblquad for precise 2D integration with geometric ray intersection
    try:
        prob_holes = compute_probability_holes(
            leg_list,
            distributions,
            weights,
            [structure_gdf],
            distance=drift_distance
        )
    except Exception as e:
        print(f"Warning: compute_probability_holes failed: {e}")
        # Return zeros if calculation fails
        return {name: {angle: 0.0 for angle in range(0, 360, 45)} for name in leg_names}

    # Calculate allision probability for each leg/direction
    result = {}

    for leg_idx, leg_name in enumerate(leg_names):
        leg = leg_list[leg_idx]
        leg_length_m = leg.length
        leg_length_nm = leg_length_m / 1852.0  # Convert to nautical miles

        result[leg_name] = {}

        # Calculate base: hours_present * blackout_per_hour
        # This matches the QGIS formula in run_calculations.py
        # Use per-leg frequency if available, otherwise use default
        ship_frequency = LEG_FREQUENCIES.get(leg_name, traffic_params['ship_frequency'])
        hours_present = leg_length_nm / traffic_params['ship_speed_kts'] * ship_frequency
        drift_p = traffic_params.get('drift_p', 1.0)  # Annual blackout probability
        blackout_per_hour = drift_p / (365 * 24)
        base = hours_present * blackout_per_hour

        for dir_idx, angle in enumerate([0, 45, 90, 135, 180, 225, 270, 315]):
            # Get wind rose probability
            rp = wind_rose.get(angle, 0.125)

            # Get probability hole from geometric calculation
            hole_pct = prob_holes[leg_idx][dir_idx][0] if prob_holes else 0.0

            if hole_pct <= 0:
                result[leg_name][angle] = 0.0
                continue

            # Calculate distance from leg to structure for p_nr
            min_dist = leg.distance(structure)

            # Get probability not repaired (distance decay)
            p_nr = get_not_repaired(drift_params['repair'], drift_params['drift_speed'], min_dist)

            # For single structure, remaining_prob = 1.0 (no earlier obstacles)
            remaining_prob = 1.0

            # Calculate contribution using QGIS formula
            contrib = base * rp * remaining_prob * hole_pct * p_nr

            result[leg_name][angle] = contrib

            if contrib > 1e-15:
                print(f"    {leg_name} {angle:3}°: base={base:.4e}, rp={rp:.3f}, "
                      f"hole={hole_pct:.4e}, p_nr={p_nr:.4f}, contrib={contrib:.4e}")

    return result


def plot_structure1_test(structure, legs, corridors_by_leg, segments, title="Structure 1 Test"):
    """Plot structure, legs, and corridors for visual debugging."""
    if not SHOW_PLOT:
        return

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    direction_names = ['East (0°)', 'NE (45°)', 'North (90°)', 'NW (135°)',
                       'West (180°)', 'SW (225°)', 'South (270°)', 'SE (315°)']
    leg_colors = ['blue', 'green', 'orange', 'red']

    for ax_idx, dir_name in enumerate(direction_names):
        ax = axes[ax_idx]
        angle = ax_idx * 45

        # Plot structure
        x, y = structure.exterior.xy
        ax.fill(x, y, alpha=0.3, fc='lightgray', ec='black', linewidth=2)

        # Plot segments with colors
        seg_colors = ['red', 'green', 'blue', 'orange']
        for seg_idx, ((x1, y1), (x2, y2)) in enumerate(segments):
            color = seg_colors[seg_idx % len(seg_colors)]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=3)
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.annotate(f'{seg_idx}', (mid_x, mid_y), fontsize=8,
                       ha='center', va='center', color='white',
                       bbox=dict(boxstyle='circle,pad=0.1', fc=color))

        # Plot corridors for this direction
        for leg_idx, (leg_name, leg) in enumerate(legs.items()):
            color = leg_colors[leg_idx % len(leg_colors)]

            # Plot leg
            lx, ly = leg.xy
            ax.plot(lx, ly, color=color, linewidth=2, linestyle='-')

            # Plot corridor if it exists for this direction
            if leg_name in corridors_by_leg and angle in corridors_by_leg[leg_name]:
                corridor = corridors_by_leg[leg_name][angle]
                if corridor is not None:
                    cx, cy = corridor.exterior.xy
                    ax.fill(cx, cy, alpha=0.15, fc=color, ec=color, linewidth=1)

        ax.set_title(f'{dir_name}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_corridor_segment_hits(structure, legs, segments, results, title="Corridor-Segment Intersections",
                               allision_probs: dict | None = None):
    """
    Create a detailed plot showing which corridors from which legs hit which segments.

    Each subplot shows ONE leg with ALL its corridors that hit any segment.
    Segments that are hit are highlighted and annotated with the contributing directions
    and their calculated allision probability values (using the same method as QGIS).

    Args:
        structure: Structure polygon
        legs: Dict of leg LineStrings
        segments: List of segment tuples
        results: Dict of {leg_name: {angle: [hit_segment_indices]}}
        title: Plot title
        allision_probs: Optional dict of {leg_name: {angle: probability}} from calculate_allision_probability
    """
    if not SHOW_PLOT:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    # Create one subplot per leg
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()

    leg_colors = {'Leg1': 'blue', 'Leg2': 'green', 'Leg3': 'orange', 'Leg4': 'red'}
    seg_colors = ['#e41a1c', '#4daf4a', '#377eb8', '#ff7f00']  # red, green, blue, orange
    direction_names = {
        0: 'E', 45: 'NE', 90: 'N', 135: 'NW',
        180: 'W', 225: 'SW', 270: 'S', 315: 'SE'
    }

    for ax_idx, (leg_name, leg) in enumerate(legs.items()):
        ax = axes[ax_idx]
        leg_results = results.get(leg_name, {})
        leg_color = leg_colors.get(leg_name, 'gray')

        # Plot structure filled
        x, y = structure.exterior.xy
        ax.fill(x, y, alpha=0.2, fc='lightgray', ec='black', linewidth=2)

        # Collect segment hit info with allision probabilities
        hit_segments_info = {}  # seg_idx -> list of (direction_name, probability)
        for angle, hit_segs in leg_results.items():
            # Get allision probability for this leg/direction
            prob = 0.0
            if allision_probs and leg_name in allision_probs:
                prob = allision_probs[leg_name].get(angle, 0.0)

            for seg_idx in hit_segs:
                if seg_idx not in hit_segments_info:
                    hit_segments_info[seg_idx] = []
                # Divide probability among hit segments for this direction
                seg_prob = prob / len(hit_segs) if hit_segs else 0.0
                hit_segments_info[seg_idx].append((direction_names[angle], seg_prob, prob))

        for seg_idx, ((x1, y1), (x2, y2)) in enumerate(segments):
            color = seg_colors[seg_idx % len(seg_colors)]
            is_hit = seg_idx in hit_segments_info

            # Draw segment - thick if hit, thin if not
            linewidth = 6 if is_hit else 2
            alpha = 1.0 if is_hit else 0.4
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=alpha)

            # Label segment
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

            if is_hit:
                # Show which directions hit this segment with probabilities
                info = hit_segments_info[seg_idx]
                total_seg_prob = sum(sp for _, sp, _ in info)
                dirs = ', '.join([d for d, _, _ in info])
                if allision_probs:
                    label = f'Seg {seg_idx}\n({dirs})\nP={total_seg_prob:.2e}'
                else:
                    label = f'Seg {seg_idx}\n({dirs})'
                fontsize = 8
                fontweight = 'bold'
            else:
                label = f'{seg_idx}'
                fontsize = 8
                fontweight = 'normal'

            ax.annotate(label, (mid_x, mid_y), fontsize=fontsize, fontweight=fontweight,
                       ha='center', va='center', color='white',
                       bbox=dict(boxstyle='round,pad=0.3', fc=color, ec='black', alpha=0.9))

        # Plot the leg
        lx, ly = leg.xy
        ax.plot(lx, ly, color=leg_color, linewidth=4, linestyle='-', label=leg_name)
        # Mark start and end
        ax.plot(lx[0], ly[0], 'o', color=leg_color, markersize=10)
        ax.plot(lx[-1], ly[-1], '^', color=leg_color, markersize=10)

        # Create summary text with allision probabilities
        summary_lines = [f"{leg_name}:"]
        total_leg_prob = 0.0

        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            hit_segs = leg_results.get(angle, [])
            if hit_segs:
                dir_name = direction_names[angle]
                seg_str = ', '.join([f'Seg{s}' for s in hit_segs])

                # Get allision probability for this direction
                if allision_probs and leg_name in allision_probs:
                    prob = allision_probs[leg_name].get(angle, 0.0)
                    total_leg_prob += prob
                    summary_lines.append(f"  {dir_name}: {seg_str} (P={prob:.2e})")
                else:
                    summary_lines.append(f"  {dir_name}: {seg_str}")

        if not any(leg_results.get(a, []) for a in [0, 45, 90, 135, 180, 225, 270, 315]):
            summary_lines.append("  No hits")
        elif allision_probs:
            summary_lines.append(f"  TOTAL: P={total_leg_prob:.2e}")

        # Add summary text box
        summary_text = '\n'.join(summary_lines)
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_title(f'{leg_name} - Segment Hits', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Add legend for segments
    legend_text = 'Segment Colors: Seg0(red)=BOTTOM, Seg1(green)=RIGHT, Seg2(blue)=TOP, Seg3(orange)=LEFT | Bold = HIT'
    if allision_probs:
        legend_text += '\nP values = allision probability (same calculation as QGIS: base × wind_rose × hole_pct × p_nr)'
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.show()


def print_contribution_summary(results: dict, segments: list, allision_probs: dict | None = None):
    """
    Print a clear summary showing which leg/direction combinations contribute
    to which segments, with allision probability values.

    Args:
        results: Dict of {leg_name: {angle: [hit_segment_indices]}}
        segments: List of segment tuples
        allision_probs: Optional dict of {leg_name: {angle: probability}} from calculate_allision_probability
    """
    print("\n" + "=" * 90)
    print("CONTRIBUTION SUMMARY: Which Leg+Direction hits which Segment")
    if allision_probs:
        print("(Using QGIS allision calculation: base × wind_rose × hole_pct × p_nr)")
    print("=" * 90)

    direction_names = {
        0: 'East', 45: 'NE', 90: 'North', 135: 'NW',
        180: 'West', 225: 'SW', 270: 'South', 315: 'SE'
    }

    # Collect all contributions
    contributions = []  # List of (leg_name, direction, angle, seg_idx, prob)

    for leg_name, leg_results in results.items():
        for angle, hit_segs in leg_results.items():
            if hit_segs:
                dir_name = direction_names[angle]
                # Get allision probability if available
                prob = 0.0
                if allision_probs and leg_name in allision_probs:
                    prob = allision_probs[leg_name].get(angle, 0.0)

                for seg_idx in hit_segs:
                    # Divide probability among hit segments
                    seg_prob = prob / len(hit_segs)
                    contributions.append((leg_name, dir_name, angle, seg_idx, seg_prob, prob))

    if not contributions:
        print("\nNo corridor-segment intersections found.")
        return

    # Group by segment
    print("\n--- Grouped by SEGMENT ---")
    for seg_idx in range(len(segments)):
        seg_contribs = [(leg, d, a, sp, tp) for leg, d, a, s, sp, tp in contributions if s == seg_idx]
        if seg_contribs:
            edge_names = ["BOTTOM (faces S)", "RIGHT (faces E)", "TOP (faces N)", "LEFT (faces NW)"]
            edge_name = edge_names[seg_idx] if seg_idx < 4 else f"Edge {seg_idx}"
            total_seg_prob = sum(sp for _, _, _, sp, _ in seg_contribs)
            if allision_probs:
                print(f"\n  Segment {seg_idx} ({edge_name}): Total P = {total_seg_prob:.4e}")
            else:
                print(f"\n  Segment {seg_idx} ({edge_name}):")
            for leg_name, dir_name, angle, seg_prob, total_prob in seg_contribs:
                if allision_probs:
                    print(f"    ← {leg_name} + {dir_name:5} drift ({angle:3}°)  [P = {total_prob:.4e}]")
                else:
                    print(f"    ← {leg_name} + {dir_name:5} drift ({angle:3}°)")
        else:
            print(f"\n  Segment {seg_idx}: No contributions")

    # Group by leg
    print("\n--- Grouped by LEG ---")
    for leg_name in results.keys():
        leg_contribs = [(d, a, s, sp, tp) for leg, d, a, s, sp, tp in contributions if leg == leg_name]
        if leg_contribs:
            total_leg_prob = sum(tp for _, _, _, _, tp in leg_contribs) / len(leg_contribs) if leg_contribs else 0
            # Actually sum unique direction probs
            unique_dirs = set((a, tp) for _, a, _, _, tp in leg_contribs)
            total_leg_prob = sum(tp for _, tp in unique_dirs)
            if allision_probs:
                print(f"\n  {leg_name}: Total P = {total_leg_prob:.4e}")
            else:
                print(f"\n  {leg_name}:")
            for dir_name, angle, seg_idx, seg_prob, total_prob in leg_contribs:
                edge_names = ["BOTTOM", "RIGHT", "TOP", "LEFT"]
                edge = edge_names[seg_idx] if seg_idx < 4 else f"Edge{seg_idx}"
                if allision_probs:
                    print(f"    {dir_name:5} drift ({angle:3}°) → Seg {seg_idx} ({edge}) [P = {total_prob:.4e}]")
                else:
                    print(f"    {dir_name:5} drift ({angle:3}°) → Seg {seg_idx} ({edge})")
        else:
            print(f"\n  {leg_name}: No contributions")

    # Summary table
    print("\n--- ALLISION PROBABILITY TABLE ---")
    if allision_probs:
        print(f"{'Leg':<8} {'Direction':<10} {'Angle':<8} {'Allision P':<14} {'Hit Segs'}")
        print("-" * 60)
    else:
        print(f"{'Leg':<8} {'Direction':<10} {'Angle':<8} {'Hit Segs'}")
        print("-" * 40)

    for leg_name in results.keys():
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            hit_segs = results[leg_name].get(angle, [])
            prob = 0.0
            if allision_probs and leg_name in allision_probs:
                prob = allision_probs[leg_name].get(angle, 0.0)
            if prob > 0 or hit_segs:
                dir_name = direction_names[angle]
                seg_str = ','.join(map(str, hit_segs)) if hit_segs else '-'
                if allision_probs:
                    print(f"{leg_name:<8} {dir_name:<10} {angle:<8} {prob:<14.4e} {seg_str}")
                elif hit_segs:
                    print(f"{leg_name:<8} {dir_name:<10} {angle:<8} {seg_str}")

    print("\n" + "=" * 90)


def print_results_table(results: dict, segments: list):
    """Print a formatted table of which segments are hit by each leg/direction."""
    print("\n" + "=" * 80)
    print("SEGMENT HIT MATRIX")
    print("=" * 80)

    directions = ['E(0°)', 'NE(45°)', 'N(90°)', 'NW(135°)',
                  'W(180°)', 'SW(225°)', 'S(270°)', 'SE(315°)']

    # Print header
    header = f"{'Leg':<10}"
    for d in directions:
        header += f"{d:>10}"
    print(header)
    print("-" * 80)

    # Print each leg's results
    for leg_name, leg_results in results.items():
        row = f"{leg_name:<10}"
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            hit_segs = leg_results.get(angle, [])
            if hit_segs:
                row += f"{'['+','.join(map(str, hit_segs))+']':>10}"
            else:
                row += f"{'---':>10}"
        print(row)

    print("=" * 80)
    print("\nSegment faces (CCW polygon):")
    for i, seg in enumerate(segments):
        # Calculate segment direction and outward normal
        dx = seg[1][0] - seg[0][0]
        dy = seg[1][1] - seg[0][1]
        # Outward normal for CCW: rotate 90° clockwise = (dy, -dx)
        normal_angle = np.degrees(np.arctan2(-dx, dy))
        print(f"  Seg {i}: {seg[0]} -> {seg[1]} | outward normal ≈ {normal_angle:.0f}°")


def print_spatial_explanation(results: dict, segments: list, legs: dict, structure):
    """Print a clear spatial explanation of the segment hit results."""
    print("\n" + "=" * 100)
    print("SPATIAL EXPLANATION OF SEGMENT HITS")
    print("=" * 100)

    # Segment descriptions based on their outward normals
    seg_descriptions = []
    for i, seg in enumerate(segments):
        dx = seg[1][0] - seg[0][0]
        dy = seg[1][1] - seg[0][1]
        normal_angle = np.degrees(np.arctan2(-dx, dy))

        # Map angle to compass direction
        if -112.5 <= normal_angle < -67.5:
            facing = "SOUTH"
        elif -67.5 <= normal_angle < -22.5:
            facing = "SOUTH-EAST"
        elif -22.5 <= normal_angle < 22.5:
            facing = "EAST"
        elif 22.5 <= normal_angle < 67.5:
            facing = "NORTH-EAST"
        elif 67.5 <= normal_angle < 112.5:
            facing = "NORTH"
        elif 112.5 <= normal_angle < 157.5:
            facing = "NORTH-WEST"
        elif normal_angle >= 157.5 or normal_angle < -157.5:
            facing = "WEST"
        else:
            facing = "SOUTH-WEST"

        seg_descriptions.append(facing)

    print("\nSTRUCTURE 1 SEGMENTS (viewed from above, CCW order):")
    print("-" * 60)
    edge_names = ["BOTTOM", "RIGHT", "TOP", "LEFT"]
    for i, (seg, facing) in enumerate(zip(segments, seg_descriptions)):
        edge = edge_names[i] if i < 4 else f"Edge {i}"
        print(f"  Seg {i} ({edge:6}): faces {facing:12} | Ships must drift {_opposite_direction(facing)} to hit")

    print("\nLEG POSITIONS relative to structure:")
    print("-" * 60)
    struct_centroid = structure.centroid
    for name, leg in legs.items():
        leg_centroid = leg.centroid
        dx = leg_centroid.x - struct_centroid.x
        dy = leg_centroid.y - struct_centroid.y
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.degrees(np.arctan2(dy, dx))

        # Map to compass quadrant
        if -22.5 <= angle < 22.5:
            position = "EAST"
        elif 22.5 <= angle < 67.5:
            position = "NORTH-EAST"
        elif 67.5 <= angle < 112.5:
            position = "NORTH"
        elif 112.5 <= angle < 157.5:
            position = "NORTH-WEST"
        elif angle >= 157.5 or angle < -157.5:
            position = "WEST"
        elif -157.5 <= angle < -112.5:
            position = "SOUTH-WEST"
        elif -112.5 <= angle < -67.5:
            position = "SOUTH"
        else:
            position = "SOUTH-EAST"

        print(f"  {name}: {dist/1000:.1f} km to the {position} of structure")

    print("\nDRIFT DIRECTION EFFECTS:")
    print("-" * 60)
    direction_names = {
        0: "EAST",
        45: "NORTH-EAST",
        90: "NORTH",
        135: "NORTH-WEST",
        180: "WEST",
        225: "SOUTH-WEST",
        270: "SOUTH",
        315: "SOUTH-EAST"
    }

    for angle, dir_name in direction_names.items():
        print(f"\n  {dir_name} drift ({angle}°): Ships drift {dir_name}ward")

        # Which segments can be hit by this drift?
        hittable_segs = []
        for i, facing in enumerate(seg_descriptions):
            opposite = _opposite_direction(facing)
            if _directions_compatible(dir_name, opposite):
                hittable_segs.append(i)

        if hittable_segs:
            seg_names = [f"Seg {i} ({edge_names[i] if i < 4 else f'Edge {i}'})"
                        for i in hittable_segs]
            print(f"    → Can hit: {', '.join(seg_names)}")
        else:
            print("    → Cannot hit any segment (drift parallel or away from all faces)")

        # What legs actually hit?
        for leg_name, leg_results in results.items():
            hit_segs = leg_results.get(angle, [])
            if hit_segs:
                seg_names = [f"Seg {i}" for i in hit_segs]
                print(f"    → {leg_name} hits: {', '.join(seg_names)}")

    print("\n" + "=" * 100)


def _opposite_direction(direction: str) -> str:
    """Get the opposite compass direction."""
    opposites = {
        "NORTH": "SOUTH",
        "SOUTH": "NORTH",
        "EAST": "WEST",
        "WEST": "EAST",
        "NORTH-EAST": "SOUTH-WEST",
        "SOUTH-WEST": "NORTH-EAST",
        "NORTH-WEST": "SOUTH-EAST",
        "SOUTH-EAST": "NORTH-WEST",
    }
    return opposites.get(direction, direction)


def _directions_compatible(drift_dir: str, required_dir: str) -> bool:
    """Check if a drift direction can hit a segment requiring a certain approach direction."""
    # Normalize
    drift_dir = drift_dir.upper().replace(" ", "-")
    required_dir = required_dir.upper().replace(" ", "-")

    if drift_dir == required_dir:
        return True

    # Allow adjacent directions (45° tolerance)
    adjacent = {
        "NORTH": ["NORTH-EAST", "NORTH-WEST"],
        "SOUTH": ["SOUTH-EAST", "SOUTH-WEST"],
        "EAST": ["NORTH-EAST", "SOUTH-EAST"],
        "WEST": ["NORTH-WEST", "SOUTH-WEST"],
        "NORTH-EAST": ["NORTH", "EAST"],
        "NORTH-WEST": ["NORTH", "WEST"],
        "SOUTH-EAST": ["SOUTH", "EAST"],
        "SOUTH-WEST": ["SOUTH", "WEST"],
    }

    return required_dir in adjacent.get(drift_dir, [])


class TestStructure1Segments:
    """
    Test segment attribution for Structure 1 using actual proj.omrat geometry.

    All coordinates are transformed to UTM zone 33N for accurate distance calculations.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test geometry from proj.omrat."""
        # WGS84 to UTM transformer (zone 33N for Baltic Sea area)
        self.wgs84 = CRS.from_epsg(4326)
        self.utm = CRS.from_epsg(32633)  # UTM zone 33N
        self.to_utm = Transformer.from_crs(self.wgs84, self.utm, always_xy=True)
        self.to_wgs84 = Transformer.from_crs(self.utm, self.wgs84, always_xy=True)

        # Structure 1 in WGS84 (from proj.omrat)
        structure_wgs84 = Polygon([
            (13.89417, 55.23217),   # V0 - SW
            (14.25963, 55.25552),   # V1 - SE
            (14.29743, 55.31652),   # V2 - NE
            (14.12416, 55.30397),   # V3 - NW
            (13.89417, 55.23217),   # close
        ])

        # Transform to UTM
        self.structure = shapely_transform(self.to_utm.transform, structure_wgs84)
        self.segments = _extract_obstacle_segments(self.structure)

        # Legs from proj.omrat in WGS84
        self.legs_wgs84 = {
            'Leg1': LineString([(13.295152, 55.224254), (14.158810, 55.187230)]),
            'Leg2': LineString([(14.15881, 55.18723), (14.29113, 55.17644)]),
            'Leg3': LineString([(14.291130, 55.176440), (14.619750, 55.421273)]),
            'Leg4': LineString([(14.619750, 55.421273), (15.091956, 55.560522)]),
        }

        # Transform legs to UTM
        self.legs = {
            name: shapely_transform(self.to_utm.transform, leg)
            for name, leg in self.legs_wgs84.items()
        }

        # Parameters for drift corridor creation
        # QGIS calculates reach_distance from t99 of repair distribution (~28.6 km for proj.omrat)
        # We use 30km to ensure corridors reach the structure from all legs
        self.drift_distance = 30000  # 30 km drift distance (meters)
        self.lateral_spread = 2000   # 2 km half-width

    def test_structure_segments_order(self):
        """Verify structure has 4 segments in expected CCW order."""
        assert len(self.segments) == 4

        # Print segment info for debugging
        print("\nStructure 1 segments (UTM coords):")
        for i, seg in enumerate(self.segments):
            dx = seg[1][0] - seg[0][0]
            dy = seg[1][1] - seg[0][1]
            length = np.sqrt(dx**2 + dy**2)
            # Outward normal angle (CCW polygon)
            normal_angle = np.degrees(np.arctan2(-dx, dy))
            print(f"  Seg {i}: length={length:.0f}m, outward normal ≈ {normal_angle:.0f}° (math)")

    def test_leg_positions_relative_to_structure(self):
        """Verify legs are positioned as expected relative to structure."""
        struct_centroid = self.structure.centroid

        print("\nLeg positions relative to structure centroid:")
        for name, leg in self.legs.items():
            leg_centroid = leg.centroid
            dx = leg_centroid.x - struct_centroid.x
            dy = leg_centroid.y - struct_centroid.y
            dist = np.sqrt(dx**2 + dy**2)
            angle = np.degrees(np.arctan2(dy, dx))
            print(f"  {name}: {dist:.0f}m away at {angle:.0f}° (math convention)")

        # Basic sanity checks
        # Leg 1 should be roughly west-southwest of structure
        leg1_centroid = self.legs['Leg1'].centroid
        assert leg1_centroid.x < struct_centroid.x, "Leg1 should be west of structure"
        assert leg1_centroid.y < struct_centroid.y, "Leg1 should be south of structure"

        # Leg 2 should be roughly south of structure
        leg2_centroid = self.legs['Leg2'].centroid
        assert leg2_centroid.y < struct_centroid.y, "Leg2 should be south of structure"

    def test_all_directions_all_legs(self):
        """Test all 8 drift directions for all 4 legs and report results."""
        results = {}
        corridors_by_leg = {}

        for leg_name, leg in self.legs.items():
            leg_results = {}
            corridors_by_leg[leg_name] = {}
            leg_centroid = (leg.centroid.x, leg.centroid.y)

            for d_idx in range(8):
                angle = d_idx * 45  # Math convention: 0=E, 90=N, etc.

                corridor = _create_drift_corridor(
                    leg, angle, self.drift_distance, self.lateral_spread
                )
                corridors_by_leg[leg_name][angle] = corridor

                if corridor is None:
                    leg_results[angle] = []
                    continue

                # Check which segments are hit
                hit_segs = []
                for seg_idx, seg in enumerate(self.segments):
                    if _segment_intersects_corridor(
                        seg, corridor,
                        drift_angle=angle,
                        leg_centroid=leg_centroid
                    ):
                        hit_segs.append(seg_idx)

                leg_results[angle] = hit_segs

            results[leg_name] = leg_results

        # Calculate allision probability using the same method as QGIS
        print("\n" + "=" * 90)
        print("CALCULATING ALLISION PROBABILITY (same method as QGIS)...")
        print("Formula: contrib = base × wind_rose × hole_pct × p_nr")
        print("  base = hours_present × blackout_per_hour")
        print("  hole_pct = probability_holes from geometric calculation")
        print("  p_nr = probability not repaired (distance decay)")
        print("=" * 90)

        allision_probs = calculate_allision_probability(
            self.legs, self.structure, self.drift_distance, self.lateral_spread
        )

        # Print allision probability summary
        print("\nAllision probability by leg and direction:")
        total_allision = 0.0
        for leg_name in self.legs.keys():
            leg_total = 0.0
            print(f"\n  {leg_name}:")
            for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
                prob = allision_probs[leg_name].get(angle, 0.0)
                leg_total += prob
                dir_names = {0: 'E', 45: 'NE', 90: 'N', 135: 'NW', 180: 'W', 225: 'SW', 270: 'S', 315: 'SE'}
                print(f"    {dir_names[angle]:3} ({angle:3}°): {prob:.4e}")
            print(f"    TOTAL: {leg_total:.4e}")
            total_allision += leg_total
        print(f"\n  GRAND TOTAL ALLISION PROBABILITY: {total_allision:.4e}")

        # Print results table
        print_results_table(results, self.segments)

        # Print clear spatial explanation
        print_spatial_explanation(results, self.segments, self.legs, self.structure)

        # Print contribution summary showing which leg+direction hits which segment
        print_contribution_summary(results, self.segments, allision_probs)

        # Visualize if enabled - per-leg corridor-segment hit view with allision probabilities
        plot_corridor_segment_hits(
            self.structure, self.legs, self.segments, results,
            title="Structure 1 - Allision Probability (QGIS Method)",
            allision_probs=allision_probs
        )

        # Also show the original 8-direction view
        plot_structure1_test(
            self.structure, self.legs, corridors_by_leg, self.segments,
            title="Structure 1 - All Legs and Directions"
        )

        # Key logical assertions based on geometry:
        # 1. South drift (270°) should hit nothing (all legs are south of structure)
        for leg_name in results:
            assert results[leg_name].get(270, []) == [], \
                f"{leg_name} South drift should not hit any segment"

        # 2. Segment 2 (top, faces north) should NEVER be hit by N/NE/NW drift
        for leg_name in results:
            for angle in [45, 90, 135]:  # NE, N, NW
                assert 2 not in results[leg_name].get(angle, []), \
                    f"{leg_name} {angle}° drift should not hit seg 2 (north-facing)"

        # 3. Segment 0 (bottom, faces south) should NEVER be hit by S/SE/SW drift
        for leg_name in results:
            for angle in [225, 270, 315]:  # SW, S, SE
                assert 0 not in results[leg_name].get(angle, []), \
                    f"{leg_name} {angle}° drift should not hit seg 0 (south-facing)"

        # 4. Leg 4 is too far east to hit the structure in most directions
        # (this is OK if it doesn't hit anything)

        # 5. North drift (90°) from Leg 2 (directly south) should hit seg 0
        assert 0 in results['Leg2'].get(90, []), \
            "Leg2 North drift should hit bottom edge (seg 0)"

    def test_north_drift_from_leg2_hits_bottom(self):
        """
        North drift (90°) from Leg 2 should hit the bottom edge (segment 0).

        Leg 2 is directly south of the structure, so north drift should
        push ships into the bottom edge.
        """
        leg = self.legs['Leg2']
        leg_centroid = (leg.centroid.x, leg.centroid.y)

        corridor = _create_drift_corridor(leg, 90, self.drift_distance, self.lateral_spread)

        assert corridor is not None, "North corridor should be created"
        assert corridor.intersects(self.structure), "North corridor should reach structure"

        # Check segment hits
        hit_segs = [
            i for i, seg in enumerate(self.segments)
            if _segment_intersects_corridor(seg, corridor, drift_angle=90, leg_centroid=leg_centroid)
        ]

        print(f"\nLeg2 North drift (90°) hits segments: {hit_segs}")

        # Segment 0 (bottom edge) should be hit
        assert 0 in hit_segs, f"Bottom edge (seg 0) should be hit by north drift. Got: {hit_segs}"

        # Segment 2 (top edge) should NOT be hit - it faces north (away from drift)
        assert 2 not in hit_segs, f"Top edge (seg 2) should NOT be hit by north drift. Got: {hit_segs}"

    def test_south_drift_does_not_hit_structure(self):
        """
        South drift (270°) should NOT hit the structure from any leg.

        All legs are south of the structure, so south drift pushes ships
        away from the structure.
        """
        for leg_name, leg in self.legs.items():
            leg_centroid = (leg.centroid.x, leg.centroid.y)
            corridor = _create_drift_corridor(leg, 270, self.drift_distance, self.lateral_spread)

            if corridor is None:
                continue

            # South drift should not reach structure (structure is north of all legs)
            hit_segs = [
                i for i, seg in enumerate(self.segments)
                if _segment_intersects_corridor(seg, corridor, drift_angle=270, leg_centroid=leg_centroid)
            ]

            assert len(hit_segs) == 0, \
                f"{leg_name} south drift (270°) should NOT hit structure. Got: {hit_segs}"

    def test_north_drift_does_not_hit_segment2(self):
        """
        North drift (90°) should NOT hit segment 2 (top edge facing north).

        Segment 2 faces north (outward normal points north), so ships
        drifting north would exit through this face, not enter.
        """
        for leg_name, leg in self.legs.items():
            leg_centroid = (leg.centroid.x, leg.centroid.y)
            corridor = _create_drift_corridor(leg, 90, self.drift_distance, self.lateral_spread)

            if corridor is None:
                continue

            # Check if segment 2 is incorrectly hit
            seg2 = self.segments[2]
            seg2_hit = _segment_intersects_corridor(
                seg2, corridor, drift_angle=90, leg_centroid=leg_centroid
            )

            assert not seg2_hit, \
                f"{leg_name}: North drift should NOT hit segment 2 (top edge facing north)"

    def test_west_drift_from_leg3_could_hit_right_edge(self):
        """
        West drift (180°) from Leg 3 (SE of structure) could hit the right edge (seg 1).

        Leg 3 is southeast of the structure. West drift pushes ships westward,
        potentially into the right edge.
        """
        leg = self.legs['Leg3']
        leg_centroid = (leg.centroid.x, leg.centroid.y)

        corridor = _create_drift_corridor(leg, 180, self.drift_distance, self.lateral_spread)

        if corridor is None:
            pytest.skip("West corridor not created")

        hit_segs = [
            i for i, seg in enumerate(self.segments)
            if _segment_intersects_corridor(seg, corridor, drift_angle=180, leg_centroid=leg_centroid)
        ]

        print(f"\nLeg3 West drift (180°) hits segments: {hit_segs}")

        # If the corridor reaches the structure, it should hit segment 1 (right edge)
        if corridor.intersects(self.structure):
            # Right edge (seg 1) faces east, west drift goes into it
            assert 1 in hit_segs, f"Right edge (seg 1) should be hit by west drift. Got: {hit_segs}"
            # Left edge (seg 3) faces west/southwest, west drift would exit through it
            assert 3 not in hit_segs, f"Left edge (seg 3) should NOT be hit by west drift. Got: {hit_segs}"

    def test_segment_outward_normals(self):
        """
        Verify segment outward normals are calculated correctly.

        For a CCW polygon, outward normal = rotate segment vector 90° clockwise.
        """
        print("\nSegment outward normals:")
        for i, seg in enumerate(self.segments):
            p1, p2 = seg
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            seg_len = np.sqrt(dx**2 + dy**2)

            # Outward normal for CCW: (dy, -dx) normalized
            normal = np.array([dy, -dx]) / seg_len
            normal_angle = np.degrees(np.arctan2(normal[1], normal[0]))

            # Map to compass direction
            compass_dir = ""
            if -22.5 <= normal_angle < 22.5:
                compass_dir = "East"
            elif 22.5 <= normal_angle < 67.5:
                compass_dir = "NorthEast"
            elif 67.5 <= normal_angle < 112.5:
                compass_dir = "North"
            elif 112.5 <= normal_angle < 157.5:
                compass_dir = "NorthWest"
            elif normal_angle >= 157.5 or normal_angle < -157.5:
                compass_dir = "West"
            elif -157.5 <= normal_angle < -112.5:
                compass_dir = "SouthWest"
            elif -112.5 <= normal_angle < -67.5:
                compass_dir = "South"
            elif -67.5 <= normal_angle < -22.5:
                compass_dir = "SouthEast"

            print(f"  Seg {i}: normal angle = {normal_angle:.1f}° (math) -> faces {compass_dir}")

        # Basic checks based on structure shape
        # Segment 0 (bottom) should face roughly south
        seg0_normal = self._get_outward_normal_angle(self.segments[0])
        assert -135 < seg0_normal < -45, f"Segment 0 should face south, got {seg0_normal}°"

        # Segment 2 (top) should face roughly north
        seg2_normal = self._get_outward_normal_angle(self.segments[2])
        assert 45 < seg2_normal < 135, f"Segment 2 should face north, got {seg2_normal}°"

    def _get_outward_normal_angle(self, seg):
        """Calculate outward normal angle for a segment (CCW polygon)."""
        p1, p2 = seg
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.degrees(np.arctan2(-dx, dy))

    def test_compass_to_math_index_conversion(self):
        """
        Verify the compass-to-math index conversion is correct.

        This tests the _compass_idx_to_math_idx function that was added
        to fix the angle convention mismatch.
        """
        from compute.run_calculations import _compass_idx_to_math_idx

        # Expected mappings:
        # Compass d_idx=0 (North, 0°) -> math 90° -> math_idx=2
        # Compass d_idx=1 (NE, 45°) -> math 45° -> math_idx=1
        # Compass d_idx=2 (East, 90°) -> math 0° -> math_idx=0
        # Compass d_idx=3 (SE, 135°) -> math -45° = 315° -> math_idx=7
        # Compass d_idx=4 (South, 180°) -> math -90° = 270° -> math_idx=6
        # Compass d_idx=5 (SW, 225°) -> math -135° = 225° -> math_idx=5
        # Compass d_idx=6 (West, 270°) -> math -180° = 180° -> math_idx=4
        # Compass d_idx=7 (NW, 315°) -> math -225° = 135° -> math_idx=3

        expected = {
            0: 2,  # N -> North
            1: 1,  # NE -> NE
            2: 0,  # E -> East
            3: 7,  # SE -> SE
            4: 6,  # S -> South
            5: 5,  # SW -> SW
            6: 4,  # W -> West
            7: 3,  # NW -> NW
        }

        for compass_idx, expected_math_idx in expected.items():
            actual = _compass_idx_to_math_idx(compass_idx)
            assert actual == expected_math_idx, \
                f"compass_idx={compass_idx}: expected math_idx={expected_math_idx}, got {actual}"

        print("\nCompass-to-math index conversion verified:")
        compass_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        math_names = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
        for i in range(8):
            m = _compass_idx_to_math_idx(i)
            print(f"  compass_idx={i} ({compass_names[i]}) -> math_idx={m} ({math_names[m]})")


class TestDriftDirectionCheck:
    """
    Test the drift direction check logic in _segment_intersects_corridor.

    Key insight: For a ship to HIT a segment (enter the polygon), the drift
    direction must OPPOSE the segment's outward normal (negative dot product).

    If the dot product is positive, ships would EXIT through that face.
    """

    def test_drift_into_vs_out_of_segment(self):
        """
        Test that drift direction is correctly checked against segment normal.

        Setup:
        - Horizontal segment at y=100 from x=0 to x=100
        - Segment vector: (100, 0)
        - Outward normal (CCW): (0, -100) normalized = (0, -1) -> faces SOUTH

        Expected:
        - North drift (90°): drift_dir=(0,1), dot with (0,-1) = -1 < 0 -> HIT
        - South drift (270°): drift_dir=(0,-1), dot with (0,-1) = 1 > 0 -> MISS (exit face)
        """
        # Create a simple corridor that covers the segment
        leg = LineString([(50, 0), (50, 50)])  # Vertical leg below segment

        # Horizontal segment facing south (bottom edge of a CCW rectangle)
        segment = ((0, 100), (100, 100))

        # Create large corridor to ensure it covers the segment
        corridor_north = _create_drift_corridor(leg, 90, 200, 100)
        corridor_south = _create_drift_corridor(leg, 270, 200, 100)

        leg_centroid = (50, 25)

        # North drift should hit the south-facing segment
        north_hits = _segment_intersects_corridor(
            segment, corridor_north, drift_angle=90, leg_centroid=leg_centroid
        )

        # South drift should NOT hit the south-facing segment (would exit through it)
        south_hits = _segment_intersects_corridor(
            segment, corridor_south, drift_angle=270, leg_centroid=leg_centroid
        )

        print(f"\nHorizontal segment facing south:")
        print(f"  North drift (90°): {'HIT' if north_hits else 'MISS'}")
        print(f"  South drift (270°): {'HIT' if south_hits else 'MISS'}")

        assert north_hits, "North drift should HIT south-facing segment"
        assert not south_hits, "South drift should NOT hit south-facing segment (exit face)"

    def test_parallel_drift_does_not_hit(self):
        """
        Drift parallel to a segment should not hit it.

        Setup:
        - Horizontal segment (y=100)
        - East/West drift (0°/180°) is parallel to segment
        """
        leg = LineString([(50, 0), (50, 50)])
        segment = ((0, 100), (100, 100))

        corridor_east = _create_drift_corridor(leg, 0, 200, 100)
        corridor_west = _create_drift_corridor(leg, 180, 200, 100)

        leg_centroid = (50, 25)

        # Neither east nor west drift should hit horizontal segment
        east_hits = _segment_intersects_corridor(
            segment, corridor_east, drift_angle=0, leg_centroid=leg_centroid
        )
        west_hits = _segment_intersects_corridor(
            segment, corridor_west, drift_angle=180, leg_centroid=leg_centroid
        )

        print(f"\nHorizontal segment with parallel drift:")
        print(f"  East drift (0°): {'HIT' if east_hits else 'MISS'}")
        print(f"  West drift (180°): {'HIT' if west_hits else 'MISS'}")

        assert not east_hits, "East drift (parallel) should NOT hit horizontal segment"
        assert not west_hits, "West drift (parallel) should NOT hit horizontal segment"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '--noconftest'])
