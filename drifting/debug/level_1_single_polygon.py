"""
Level 1: Drifting Grounding - Single Polygon, Single Ship Category, Single Leg
===============================================================================

This example demonstrates the fundamental drifting grounding probability
calculation in OMRAT. A ship on a traffic leg suffers a blackout and drifts
in one direction toward a shallow polygon.

Setup:
    - One straight traffic leg (Leg 3 from proj_3_3, bearing ~41 deg NE)
    - One depth polygon (12m depth, a rectangular cell south of the leg)
    - One ship category: Oil tanker 225-250m, draught=14.27m
    - Drift direction: 315 deg compass (NW) -- toward the polygon
    - Uniform wind rose: 1/8 per direction

We show two distance modes:
    (a) start_from = 'leg_center': distance from the leg LINE (default)
    (b) start_from = 'distribution_center': distance from the mean-offset line

In both modes, distance is measured from each edge VERTEX back to the
reference line (leg or offset) along the drift direction using reverse rays.

All computations use OMRAT's actual functions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import LineString, Polygon, Point, box
from shapely.ops import transform
from scipy import stats
import pyproj

from drifting.engine import (
    LegState, ShipState, DriftConfig,
    compass_to_math_deg,
    directional_distance_to_point_from_offset_leg,
    directional_distance_m,
    corridor_width_m,
    build_directional_corridor,
    edge_average_distance_m,
)
from compute.basic_equations import get_not_repaired, get_drift_time
from geometries.analytical_probability import (
    compute_probability_analytical,
    _extract_polygon_rings,
)


# =============================================================================
# 1. GEOMETRY SETUP (from proj_3_3 in EPSG:4326, transformed to UTM)
# =============================================================================

# Leg 3 endpoints (from proj_3_3 model.xml)
# WP_2_End_Point -> WP_3_End_Point (SW to NE, bearing ~41 deg)
leg3_start_lonlat = (14.24187, 55.16728)  # SW endpoint (WP_2_End_Point)
leg3_end_lonlat = (14.59271, 55.39937)    # NE endpoint (WP_3_End_Point)

# Depth polygon BD5A4C46 vertices -- more natural shape with extra corners
poly_vertices_lonlat = [
    (14.20417, 55.30833),
    (14.20300, 55.30650),
    (14.20417, 55.30417),
    (14.20200, 55.30417),
    (14.20000, 55.30200),
    (14.20000, 55.30000),
    (14.20250, 55.30050),
    (14.20417, 55.30000),
    (14.20417, 55.30833),
]

# Transform to UTM zone 33N (EPSG:32633) for metric calculations
proj_wgs84 = pyproj.CRS("EPSG:4326")
proj_utm = pyproj.CRS("EPSG:32633")
transformer = pyproj.Transformer.from_crs(proj_wgs84, proj_utm, always_xy=True)
transformer_inv = pyproj.Transformer.from_crs(proj_utm, proj_wgs84, always_xy=True)

def to_utm(lon, lat):
    return transformer.transform(lon, lat)

def to_wgs84(x, y):
    return transformer_inv.transform(x, y)

# Transform leg to UTM
leg_start_utm = to_utm(*leg3_start_lonlat)
leg_end_utm = to_utm(*leg3_end_lonlat)
leg_line = LineString([leg_start_utm, leg_end_utm])

# Transform polygon to UTM
poly_utm_coords = [to_utm(lon, lat) for lon, lat in poly_vertices_lonlat]
depth_polygon = Polygon(poly_utm_coords)

print("=" * 80)
print("LEVEL 1: Single Polygon, Single Ship Category, Single Leg")
print("=" * 80)
print()
print("GEOMETRY (UTM Zone 33N, EPSG:32633):")
print(f"  Leg start:  ({leg_start_utm[0]:.1f}, {leg_start_utm[1]:.1f}) m")
print(f"  Leg end:    ({leg_end_utm[0]:.1f}, {leg_end_utm[1]:.1f}) m")
print(f"  Leg length: {leg_line.length:.1f} m")
print(f"  Leg bearing: ~41 deg (NE)")
print(f"  Polygon depth: 12 m")
print(f"  Polygon centroid (UTM): ({depth_polygon.centroid.x:.1f}, {depth_polygon.centroid.y:.1f})")
print()

# =============================================================================
# 2. SHIP AND DRIFT PARAMETERS
# =============================================================================

# Ship category: Oil products tanker 225-250m
ship_draught = 14.27  # meters (design draught)
ship_speed_kts = 12.5  # knots (service speed)
ship_freq = 610  # ships/year on this leg (one direction)

# Drift parameters
drift_speed_kts = 1.94  # knots
drift_speed_ms = drift_speed_kts * 1852 / 3600  # m/s = 0.998 m/s
blackout_rate = 1.0  # per year
blackout_per_hour = blackout_rate / (365.25 * 24)

# Repair time distribution (lognormal)
repair_data = {
    'use_lognormal': 1,
    'std': 1.0,    # shape parameter (sigma of log)
    'loc': 0.0,    # location
    'scale': 1.0,  # scale (median repair time in hours)
}

# Wind rose: uniform
rose_prob = 1.0 / 8.0  # 0.125 per direction

# Drift direction: 315 deg compass (NW)
drift_direction = 315

# Lateral distribution: single Gaussian with sigma based on leg width
# In OMRAT, sigma comes from the traffic distribution fitting
# For this example, use a typical value
lateral_sigma = 500.0  # meters (typical for a wide lane)

print("SHIP & DRIFT PARAMETERS:")
print(f"  Ship type: Oil products tanker 225-250m")
print(f"  Draught: {ship_draught} m (> polygon depth 12m -> grounding hazard)")
print(f"  Speed: {ship_speed_kts} kts")
print(f"  Frequency: {ship_freq} ships/year")
print(f"  Drift speed: {drift_speed_kts} kts = {drift_speed_ms:.3f} m/s")
print(f"  Blackout rate: {blackout_rate}/year = {blackout_per_hour:.6e}/hour")
print(f"  Rose probability (NW=315): {rose_prob}")
print(f"  Lateral sigma: {lateral_sigma} m")
print()

# =============================================================================
# 3. CONSTRUCT OMRAT OBJECTS
# =============================================================================

leg_state = LegState(
    leg_id="LEG_3",
    line=leg_line,
    mean_offset_m=0.0,  # No offset (start_from='leg_center')
    lateral_sigma_m=lateral_sigma,
)

ship_state = ShipState(
    draught_m=ship_draught,
    anchor_d=7.0,  # Not used in this simple example (no anchoring zones)
)

reach_distance = 50000.0  # meters (max drift reach)
cfg = DriftConfig(
    reach_distance_m=reach_distance,
    corridor_sigma_multiplier=3.0,
    use_leg_offset_for_distance=False,
)

# =============================================================================
# 4. STEP-BY-STEP CALCULATION
# =============================================================================

print("CALCULATION STEPS:")
print("-" * 80)

# Step 1: Exposure (base rate)
hours_present = (leg_line.length / (ship_speed_kts * 1852)) * ship_freq
base = hours_present * blackout_per_hour
print(f"  Step 1 - Exposure:")
print(f"    hours_present = L / (V * 1852) * freq")
print(f"                  = {leg_line.length:.1f} / ({ship_speed_kts} * 1852) * {ship_freq}")
print(f"                  = {hours_present:.4f} hours/year")
print(f"    base = hours_present * blackout_per_hour")
print(f"         = {hours_present:.4f} * {blackout_per_hour:.6e}")
print(f"         = {base:.6e}")
print()

# Step 2: Probability hole (what fraction of lateral distribution hits polygon)
# Using analytical probability
leg_coords = np.array(leg_line.coords)
leg_start_np = leg_coords[0]
leg_end_np = leg_coords[-1]
leg_vec = leg_end_np - leg_start_np
leg_len = leg_line.length
leg_dir = leg_vec / leg_len
perp_dir = np.array([-leg_dir[1], leg_dir[0]])

# Drift direction vector (compass to math)
math_deg = compass_to_math_deg(drift_direction)
angle_rad = np.radians(math_deg)
drift_vec = np.array([np.cos(angle_rad), np.sin(angle_rad)])

# Lateral distribution
lateral_dist = stats.norm(0, lateral_sigma)
lateral_range = 5.0 * lateral_sigma  # +/- 5 sigma

polygon_rings = _extract_polygon_rings(depth_polygon)

hole_pct = compute_probability_analytical(
    leg_start=leg_start_np,
    leg_vec=leg_vec,
    perp_dir=perp_dir,
    drift_vec=drift_vec,
    distance=reach_distance,
    lateral_range=lateral_range,
    polygon_rings=polygon_rings,
    dists=[lateral_dist],
    weights=np.array([1.0]),
    n_slices=200,
)

print(f"  Step 2 - Probability hole (analytical):")
print(f"    drift_direction = {drift_direction} deg compass = {math_deg:.1f} deg math")
print(f"    drift_vec = ({drift_vec[0]:.4f}, {drift_vec[1]:.4f})")
print(f"    lateral_range = 5 * sigma = {lateral_range:.0f} m")
print(f"    n_slices = 200")
print(f"    hole_pct = {hole_pct:.6e}")
print(f"    (fraction of distribution that intersects polygon in this direction)")
print()

# Step 3: Edge-level distribution of hole
# Get polygon edges and their individual distances
edges = list(zip(poly_utm_coords[:-1], poly_utm_coords[1:]))
print(f"  Step 3 - Edge distances and hole distribution:")
print(f"    Polygon has {len(edges)} edges")

# Build corridor for edge filtering
corridor = build_directional_corridor(leg_state, drift_direction, cfg)

# For each edge, compute its directional distance and check if it faces the drift
edge_data = []
math_deg = compass_to_math_deg(drift_direction)
drift_ux = np.cos(np.radians(math_deg))
drift_uy = np.sin(np.radians(math_deg))

for i, (p1, p2) in enumerate(edges):
    d_avg = edge_average_distance_m(
        leg_state, drift_direction, (p1, p2), use_leg_offset=False
    )
    # Check if edge intersects corridor
    edge_line = LineString([p1, p2])
    in_corridor = corridor.intersects(edge_line)

    # For a CCW polygon (Shapely default), outward normal of edge p1->p2 is (dy, -dx).
    ex, ey = p2[0] - p1[0], p2[1] - p1[1]
    # Shapely exterior is CCW by default; outward normal = (ey, -ex)
    nx, ny = ey, -ex
    # Edge faces drift source if normal points AGAINST drift direction
    faces_drift = (nx * drift_ux + ny * drift_uy) < 0

    edge_data.append({
        'idx': i,
        'p1': p1,
        'p2': p2,
        'dist': d_avg,
        'in_corridor': in_corridor,
        'faces_drift': faces_drift,
        'length': edge_line.length,
    })
    status = ""
    if not in_corridor:
        status = " [outside corridor]"
    elif not faces_drift:
        status = " [shadowed - faces away from drift]"
    if d_avg is not None:
        print(f"    Edge {i} ({p1[0]:.0f},{p1[1]:.0f})->({p2[0]:.0f},{p2[1]:.0f}): "
              f"dist={d_avg:.1f}m, in_corridor={in_corridor}, faces_drift={faces_drift}, "
              f"len={edge_line.length:.1f}m{status}")
    else:
        print(f"    Edge {i}: no valid directional distance (not facing drift)")

# Only edges that are in corridor AND face into the drift can be hit
valid_edges = [e for e in edge_data
               if e['in_corridor'] and e['faces_drift'] and e['dist'] is not None]
shadowed_edges = [e for e in edge_data
                  if e['in_corridor'] and not e['faces_drift'] and e['dist'] is not None]
total_overlap = sum(e['length'] for e in valid_edges)
print()
print(f"    Total overlap of valid edges: {total_overlap:.1f} m")
for e in valid_edges:
    e['edge_hole'] = hole_pct * (e['length'] / total_overlap)
    print(f"    Edge {e['idx']}: edge_hole = {hole_pct:.6e} * "
          f"{e['length']:.1f}/{total_overlap:.1f} = {e['edge_hole']:.6e}")
print()

# Step 4: Probability of not being repaired (per edge)
print(f"  Step 4 - P(not repaired) per edge:")
for e in valid_edges:
    drift_time_s = e['dist'] / drift_speed_ms
    drift_time_h = drift_time_s / 3600
    p_nr = get_not_repaired(repair_data, drift_speed_ms, e['dist'])
    e['p_nr'] = p_nr
    print(f"    Edge {e['idx']}: dist={e['dist']:.1f}m, "
          f"drift_time={drift_time_h:.2f}h, P(not repaired)={p_nr:.6e}")
print()

# Step 5: Final grounding probability per edge
print(f"  Step 5 - Grounding probability per edge:")
print(f"    Formula: P_ground = base * rose * hole_edge * P(not_repaired)")
print()
total_grounding = 0.0
for e in valid_edges:
    p_ground = base * rose_prob * e['edge_hole'] * e['p_nr']
    e['p_ground'] = p_ground
    total_grounding += p_ground
    print(f"    Edge {e['idx']}: {base:.4e} * {rose_prob} * "
          f"{e['edge_hole']:.4e} * {e['p_nr']:.4e} = {p_ground:.4e}")

print()
print(f"  TOTAL grounding probability (NW direction, this ship category):")
print(f"    P_total = {total_grounding:.6e} events/year")
print()

# =============================================================================
# 5. COMPARISON: start_from = 'leg_center' vs 'distribution_center'
# =============================================================================

print("=" * 80)
print("DISTANCE MODE COMPARISON")
print("=" * 80)
print()

# Mode (a): Default -- from leg LINE (mean_offset_m = 0)
print(f"  Mode (a): start_from = 'leg_center' (DEFAULT)")
print(f"    Reference: the leg LINE itself (mean_offset_m = 0)")
print(f"    Distance measured from each edge vertex back to the leg line")
print(f"    along the drift direction (reverse ray).")
for e in valid_edges:
    p1_dist = directional_distance_to_point_from_offset_leg(
        leg_state, drift_direction, Point(e['p1']), use_leg_offset=False)
    p2_dist = directional_distance_to_point_from_offset_leg(
        leg_state, drift_direction, Point(e['p2']), use_leg_offset=False)
    print(f"    Edge {e['idx']}: v0={p1_dist:.1f}m, v1={p2_dist:.1f}m, "
          f"avg={e['dist']:.1f}m")

print()

# Mode (b): From distribution center (mean offset line)
# When traffic is offset from the leg, the mean position differs per direction.
# We use two offsets: +500m (direction A) and -500m (direction B).
example_offset = 500.0
leg_state_offset = LegState(
    leg_id='LEG_3', line=leg_line,
    mean_offset_m=example_offset, lateral_sigma_m=lateral_sigma)
print(f"  Mode (b): start_from = 'distribution_center'")
print(f"    Reference: the mean-offset line (offset = +{example_offset}m from leg)")
print(f"    Distance measured from each edge vertex back to the offset line")
print(f"    along the drift direction (reverse ray).")
for e in valid_edges:
    p1_dist = directional_distance_to_point_from_offset_leg(
        leg_state_offset, drift_direction, Point(e['p1']), use_leg_offset=True)
    p2_dist = directional_distance_to_point_from_offset_leg(
        leg_state_offset, drift_direction, Point(e['p2']), use_leg_offset=True)
    d_vals = [d for d in (p1_dist, p2_dist) if d is not None]
    avg = sum(d_vals) / len(d_vals) if d_vals else float('nan')
    p1_str = f"{p1_dist:.1f}" if p1_dist is not None else "N/A"
    p2_str = f"{p2_dist:.1f}" if p2_dist is not None else "N/A"
    print(f"    Edge {e['idx']}: v0={p1_str}m, v1={p2_str}m, avg={avg:.1f}m")

print()
print("  Both modes measure from each edge vertex back to the reference line")
print("  along the drift direction.  'distribution_center' accounts for traffic")
print("  not being centered on the charted leg.")
print()

# =============================================================================
# 6. GENERATE FIGURE 1: Main overview with traffic lines
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# --- Corridor with hole AND shadow behind polygon ---
max_target_dist = max(e['dist'] for e in valid_edges)
cfg_display = DriftConfig(reach_distance_m=1.2 * max_target_dist, corridor_sigma_multiplier=3.0)
corridor_display = build_directional_corridor(leg_state, drift_direction, cfg_display)
from shapely.geometry import MultiPolygon
math_rad = np.radians(compass_to_math_deg(drift_direction))

# Build shadow zone: quad-based sweep in drift direction (NW)
# Note: we use drift_ux/drift_uy directly (correct NW vector) instead of
# create_obstacle_shadow which uses a different compass convention.
from shapely.affinity import translate as _translate
from shapely.ops import unary_union as _unary_union
_corr_diag = np.sqrt((corridor_display.bounds[2]-corridor_display.bounds[0])**2 +
                     (corridor_display.bounds[3]-corridor_display.bounds[1])**2)
_extrude = _corr_diag * 2
_far_poly = _translate(depth_polygon, xoff=drift_ux*_extrude, yoff=drift_uy*_extrude)
_orig_coords = list(depth_polygon.exterior.coords)[:-1]
_far_coords = list(_far_poly.exterior.coords)[:-1]
_quads = []
for _qi in range(len(_orig_coords)):
    _qj = (_qi + 1) % len(_orig_coords)
    _q = Polygon([_orig_coords[_qi], _orig_coords[_qj],
                  _far_coords[_qj], _far_coords[_qi]])
    if _q.is_valid and _q.area > 0:
        _quads.append(_q)
shadow_poly = _unary_union([depth_polygon, _far_poly] + _quads)
# Corridor minus the full shadow (shadow includes the obstacle itself)
corridor_with_hole = corridor_display.difference(shadow_poly)

# Draw the visible corridor
def _draw_geom(ax_obj, geom, alpha, fc, ec, lw, label=None):
    if geom.is_empty:
        return
    if geom.geom_type == 'Polygon':
        ax_obj.fill(*geom.exterior.xy, alpha=alpha, fc=fc, ec=ec, lw=lw, label=label)
        for interior in geom.interiors:
            ax_obj.fill(*interior.xy, fc='white', ec='darkred', lw=1)
    elif geom.geom_type == 'MultiPolygon':
        first = True
        for g in geom.geoms:
            ax_obj.fill(*g.exterior.xy, alpha=alpha, fc=fc, ec=ec, lw=lw,
                        label=label if first else None)
            for interior in g.interiors:
                ax_obj.fill(*interior.xy, fc='white', ec='darkred', lw=1)
            first = False

_draw_geom(ax, corridor_with_hole, 0.08, 'blue', 'blue', 0.5, 'Drift corridor')

# Draw shadow zone to show blocked area behind polygon
shadow_visible = corridor_display.intersection(shadow_poly).difference(depth_polygon)
_draw_geom(ax, shadow_visible, 0.08, 'gray', 'none', 0)
if not shadow_visible.is_empty:
    ax.plot([], [], color='gray', ls='-', lw=6, alpha=0.15, label='Shadow (blocked by polygon)')

# --- Traffic lines (multiple semi-transparent lines on both sides of leg) ---
n_traffic_lines = 15
max_offset = 2.5 * lateral_sigma
offsets = np.linspace(-max_offset, max_offset, n_traffic_lines)
# Weight by normal distribution for visual density
weights = stats.norm(0, lateral_sigma).pdf(offsets)
weights = weights / weights.max()
for offset, w in zip(offsets, weights):
    shifted = leg_line.parallel_offset(offset, 'left')
    if shifted.geom_type == 'LineString' and not shifted.is_empty:
        ax.plot(*shifted.xy, color='gray', lw=1, alpha=float(w) * 0.4, zorder=3)
# Label traffic
ax.plot([], [], color='gray', lw=2, alpha=0.4, label='Ship traffic (lateral dist.)')

# Plot central leg
ax.plot(*leg_line.xy, 'k-', linewidth=2.5, label='Traffic leg (center)', zorder=5)
ax.plot(*leg_line.xy, 'wo', markersize=5, zorder=6)

# Plot polygon
poly_x = [c[0] for c in poly_utm_coords]
poly_y = [c[1] for c in poly_utm_coords]
ax.fill(poly_x, poly_y, alpha=0.4, fc='brown', ec='darkred', lw=2, label='Depth polygon (12m)')

# Draw drift direction arrow from leg center
center = leg_line.centroid
arrow_len = 5000
math_rad = np.radians(compass_to_math_deg(drift_direction))
dx, dy = arrow_len * np.cos(math_rad), arrow_len * np.sin(math_rad)
ax.annotate('', xy=(center.x + dx, center.y + dy), xytext=(center.x, center.y),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.text(center.x + dx*0.5 + 200, center.y + dy*0.5 + 200,
        f'Drift NW\n(315 deg)', color='blue', fontsize=9, ha='left')

# Draw distance lines from actual vertices for valid edges
dist_colors = ['red', 'green', 'purple', 'orange', 'teal',
               'darkred', 'darkgreen', 'navy']
drift_perp = np.array([-np.sin(math_rad), np.cos(math_rad)])

# Precompute per-vertex distances for drawing
edge_draw_data = []
for i, e in enumerate(valid_edges):
    d0 = directional_distance_to_point_from_offset_leg(
        leg_state, drift_direction, Point(e['p1']), use_leg_offset=False)
    d1 = directional_distance_to_point_from_offset_leg(
        leg_state, drift_direction, Point(e['p2']), use_leg_offset=False)
    edge_draw_data.append((d0, d1))
    c = dist_colors[i % len(dist_colors)]
    # Draw from vertex 0 back to leg
    if d0 is not None:
        bx0 = e['p1'][0] - d0 * np.cos(math_rad)
        by0 = e['p1'][1] - d0 * np.sin(math_rad)
        ax.plot([bx0, e['p1'][0]], [by0, e['p1'][1]],
                '--', color=c, lw=1.2, zorder=4, alpha=0.6)
    # Draw from vertex 1 back to leg
    if d1 is not None:
        bx1 = e['p2'][0] - d1 * np.cos(math_rad)
        by1 = e['p2'][1] - d1 * np.sin(math_rad)
        ax.plot([bx1, e['p2'][0]], [by1, e['p2'][1]],
                '--', color=c, lw=1.2, zorder=4, alpha=0.6)

# Place text labels stacked vertically in a column to the left, no overlap
n_edges = len(valid_edges)
# Find a good column position: left of polygon, in the corridor
col_x = min(c[0] for c in poly_utm_coords) - 2500
col_y_top = min(c[1] for c in poly_utm_coords) - 1500
label_spacing = 800
for i, e in enumerate(valid_edges):
    c = dist_colors[i % len(dist_colors)]
    d0, d1 = edge_draw_data[i]
    d0_s = f'{d0:.0f}' if d0 is not None else 'N/A'
    d1_s = f'{d1:.0f}' if d1 is not None else 'N/A'
    ty = col_y_top - i * label_spacing
    ax.text(col_x, ty, f'E{e["idx"]}: v0={d0_s}, v1={d1_s}, avg={e["dist"]:.0f}m',
            color=c, fontsize=7, ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=c, alpha=0.8),
            zorder=10)
    # Thin leader line from label to edge midpoint
    emx = (e['p1'][0] + e['p2'][0]) / 2
    emy = (e['p1'][1] + e['p2'][1]) / 2
    ax.annotate('', xy=(emx, emy), xytext=(col_x + 3000, ty),
                arrowprops=dict(arrowstyle='-', color=c, lw=0.5, alpha=0.4),
                zorder=4)

# Highlight valid (front-facing) edges; show shadowed edges in gray
for e in valid_edges:
    ax.plot([e['p1'][0], e['p2'][0]], [e['p1'][1], e['p2'][1]],
            'r-', linewidth=3, zorder=7)
    ax.plot(e['p1'][0], e['p1'][1], 'ro', markersize=3, zorder=8)
    ax.plot(e['p2'][0], e['p2'][1], 'ro', markersize=3, zorder=8)
for e in shadowed_edges:
    ax.plot([e['p1'][0], e['p2'][0]], [e['p1'][1], e['p2'][1]],
            '-', color='gray', linewidth=2, zorder=6, alpha=0.5)

# Annotations box
textstr = (
    f"Ship: Tanker 225-250m, draught={ship_draught}m\n"
    f"Freq: {ship_freq} ships/yr, speed={ship_speed_kts}kts\n"
    f"Drift: {drift_speed_kts}kts, dir=315 (NW)\n"
    f"Blackout: {blackout_rate}/yr\n"
    f"Lateral sigma: {lateral_sigma}m\n"
    f"Hole: {hole_pct:.4e}\n"
    f"Total P(grounding): {total_grounding:.4e}/yr"
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props, family='monospace')

# Formula box
formula = (
    r"$P_{ground} = \frac{L}{V \cdot 1852} \cdot f \cdot \lambda_{bo} "
    r"\cdot r_p \cdot h_{edge} \cdot P_{NR}(d)$"
    "\n\n"
    r"$P_{NR} = 1 - F_{repair}\left(\frac{d}{V_{drift}}\right)$"
)
ax.text(0.02, 0.02, formula, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('Easting (m)', fontsize=10)
ax.set_ylabel('Northing (m)', fontsize=10)
ax.set_title('Level 1: Drifting Grounding - Distance from Leg Line', fontsize=12)
ax.legend(loc='upper right', fontsize=9)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# --- Zoomed inset: polygon detail with shadow + front-facing edges ---
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
ax_zoom = inset_axes(ax, width="40%", height="40%", loc='lower right', borderpad=1.5)

pad = 600
px = [c[0] for c in poly_utm_coords]
py = [c[1] for c in poly_utm_coords]
zoom_box = box(min(px)-pad, min(py)-pad, max(px)+pad, max(py)+pad)

# Corridor (without shadow) in inset
corridor_clip = corridor_with_hole.intersection(zoom_box)
_draw_geom(ax_zoom, corridor_clip, 0.12, 'blue', 'blue', 0.5)

# Shadow in inset
shadow_clip = shadow_visible.intersection(zoom_box) if not shadow_visible.is_empty else Polygon()
_draw_geom(ax_zoom, shadow_clip, 0.10, 'gray', 'none', 0)

# Polygon in inset
ax_zoom.fill(poly_x, poly_y, alpha=0.4, fc='brown', ec='darkred', lw=2)

# Front-facing (hittable) edges colored, shadowed edges in gray
for i, e in enumerate(valid_edges):
    c = dist_colors[i % len(dist_colors)]
    ax_zoom.plot([e['p1'][0], e['p2'][0]], [e['p1'][1], e['p2'][1]],
                 '-', color=c, linewidth=2.5, zorder=7)
    ax_zoom.plot(e['p1'][0], e['p1'][1], 'o', color=c, markersize=5, zorder=8,
                 markeredgecolor='black', markeredgewidth=0.5)
    ax_zoom.plot(e['p2'][0], e['p2'][1], 's', color=c, markersize=5, zorder=8,
                 markeredgecolor='black', markeredgewidth=0.5)
for e in shadowed_edges:
    ax_zoom.plot([e['p1'][0], e['p2'][0]], [e['p1'][1], e['p2'][1]],
                 '-', color='gray', linewidth=1.5, zorder=6, alpha=0.5)

# Measurement rays in inset for front-facing edges
for i, e in enumerate(valid_edges):
    c = dist_colors[i % len(dist_colors)]
    d0, d1 = edge_draw_data[i]
    if d0 is not None:
        bx = e['p1'][0] - d0 * np.cos(math_rad)
        by = e['p1'][1] - d0 * np.sin(math_rad)
        ax_zoom.plot([bx, e['p1'][0]], [by, e['p1'][1]],
                     '--', color=c, lw=1, alpha=0.5)
    if d1 is not None:
        bx = e['p2'][0] - d1 * np.cos(math_rad)
        by = e['p2'][1] - d1 * np.sin(math_rad)
        ax_zoom.plot([bx, e['p2'][0]], [by, e['p2'][1]],
                     '--', color=c, lw=1, alpha=0.5)

# Drift arrow in inset
center_z = depth_polygon.centroid
ax_zoom.annotate('', xy=(center_z.x - 300*np.cos(math_rad), center_z.y - 300*np.sin(math_rad)),
                 xytext=(center_z.x + 300*np.cos(math_rad), center_z.y + 300*np.sin(math_rad)),
                 arrowprops=dict(arrowstyle='<-', color='blue', lw=1.5))

ax_zoom.set_xlim(min(px)-pad, max(px)+pad)
ax_zoom.set_ylim(min(py)-pad, max(py)+pad)
ax_zoom.set_aspect('equal')
ax_zoom.grid(True, alpha=0.3)
ax_zoom.set_title('Front-facing edges + shadow', fontsize=8, fontweight='bold')
ax_zoom.tick_params(labelsize=6)

# Connect inset to main axes (NW and SE corners)
mark_inset(ax, ax_zoom, loc1=1, loc2=3, fc='none', ec='0.5', lw=1, ls='--')

output_dir = os.path.dirname(__file__)
fig_path = os.path.join(output_dir, 'level_1_single_polygon.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_path}")
plt.close()

# =============================================================================
# 7. GENERATE FIGURE 2: Two mean-offset lines (one per traffic direction)
#    Zoom on LEG side to show where reverse rays land + bimodal traffic
# =============================================================================

from drifting.engine import _offset_line_perpendicular

# Two mean offsets: +500m and -500m from the leg, one per traffic direction
mean_offset_A = 500.0   # direction A ships travel
mean_offset_B = -500.0  # direction B ships travel (opposite side)
offset_line_A = _offset_line_perpendicular(leg_line, mean_offset_A)
offset_line_B = _offset_line_perpendicular(leg_line, mean_offset_B)

# Build leg states for each offset
leg_state_A = LegState(leg_id='LEG_3', line=leg_line,
                       mean_offset_m=mean_offset_A, lateral_sigma_m=lateral_sigma)
leg_state_B = LegState(leg_id='LEG_3', line=leg_line,
                       mean_offset_m=mean_offset_B, lateral_sigma_m=lateral_sigma)

fig2, ax2 = plt.subplots(1, 1, figsize=(16, 11))

# Corridor with hole and shadow (reuse from Fig 1)
_draw_geom(ax2, corridor_with_hole, 0.08, 'blue', 'blue', 0.5, 'Drift corridor')
_draw_geom(ax2, shadow_visible, 0.06, 'gray', 'none', 0)

# --- Bimodal traffic lines: two groups centered on offset_A and offset_B ---
n_per_group = 25
sigma_each = lateral_sigma * 0.6  # narrower per-direction spread for clearer peaks
offsets_A = np.linspace(mean_offset_A - 2.5*sigma_each, mean_offset_A + 2.5*sigma_each, n_per_group)
offsets_B = np.linspace(mean_offset_B - 2.5*sigma_each, mean_offset_B + 2.5*sigma_each, n_per_group)
weights_A = stats.norm(mean_offset_A, sigma_each).pdf(offsets_A)
weights_B = stats.norm(mean_offset_B, sigma_each).pdf(offsets_B)
w_max = max(weights_A.max(), weights_B.max())
weights_A = weights_A / w_max
weights_B = weights_B / w_max

for off_val, w in zip(offsets_A, weights_A):
    shifted = leg_line.parallel_offset(off_val, 'left')
    if shifted.geom_type == 'LineString' and not shifted.is_empty:
        ax2.plot(*shifted.xy, color='gray', lw=1, alpha=float(w) * 0.45, zorder=3)
for off_val, w in zip(offsets_B, weights_B):
    shifted = leg_line.parallel_offset(off_val, 'left')
    if shifted.geom_type == 'LineString' and not shifted.is_empty:
        ax2.plot(*shifted.xy, color='gray', lw=1, alpha=float(w) * 0.45, zorder=3)
ax2.plot([], [], color='gray', lw=2, alpha=0.45, label='Ship traffic (bimodal)')

# Central leg
ax2.plot(*leg_line.xy, 'k-', linewidth=2.5, label='Traffic leg (center)', zorder=5)
ax2.plot(*leg_line.xy, 'wo', markersize=5, zorder=6)

# Mean-offset lines
ax2.plot(*offset_line_A.xy, color='darkorange', linewidth=2.5, linestyle='--',
         label=f'Mean-offset A (+{mean_offset_A:.0f}m)', zorder=5)
ax2.plot(*offset_line_B.xy, color='deepskyblue', linewidth=2.5, linestyle='--',
         label=f'Mean-offset B ({mean_offset_B:.0f}m)', zorder=5)

# Polygon
ax2.fill(poly_x, poly_y, alpha=0.4, fc='brown', ec='darkred', lw=2, label='Depth polygon (12m)')

# Drift arrow
ax2.annotate('', xy=(center.x + dx, center.y + dy), xytext=(center.x, center.y),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax2.text(center.x + dx*0.5 + 200, center.y + dy*0.5 + 200,
         f'Drift NW\n(315 deg)', color='blue', fontsize=9, ha='left')

# Compute per-vertex distances to all three references for front-facing edges
# Measurement lines run from the vertex to offset B (the farthest reference line)
edge_1b_data = []
southernmost_idx = None
southernmost_y = float('inf')

for i, e in enumerate(valid_edges):
    c = dist_colors[i % len(dist_colors)]
    for vi, v in enumerate([e['p1'], e['p2']]):
        pt = Point(v)
        d_leg = directional_distance_to_point_from_offset_leg(
            leg_state, drift_direction, pt, use_leg_offset=False)
        d_offA = directional_distance_to_point_from_offset_leg(
            leg_state_A, drift_direction, pt, use_leg_offset=True)
        d_offB = directional_distance_to_point_from_offset_leg(
            leg_state_B, drift_direction, pt, use_leg_offset=True)

        edge_1b_data.append({
            'edge_idx': e['idx'], 'vi': vi, 'v': v, 'color': c,
            'd_leg': d_leg, 'd_offA': d_offA, 'd_offB': d_offB,
        })

        if v[1] < southernmost_y:
            southernmost_y = v[1]
            southernmost_idx = len(edge_1b_data) - 1

        # Draw measurement line from vertex to offset B (stop there, do not extend past)
        if d_offB is not None:
            bx_B = v[0] - d_offB * np.cos(math_rad)
            by_B = v[1] - d_offB * np.sin(math_rad)
            ax2.plot([bx_B, v[0]], [by_B, v[1]],
                     '--', color=c, lw=1.0, zorder=4, alpha=0.4)

# Mark vertex dots for front-facing edges
for e in valid_edges:
    ax2.plot([e['p1'][0], e['p2'][0]], [e['p1'][1], e['p2'][1]],
             'r-', linewidth=3, zorder=7)
    ax2.plot(e['p1'][0], e['p1'][1], 'ro', markersize=3, zorder=8)
    ax2.plot(e['p2'][0], e['p2'][1], 'ro', markersize=3, zorder=8)
# Shadowed edges in gray
for e in shadowed_edges:
    ax2.plot([e['p1'][0], e['p2'][0]], [e['p1'][1], e['p2'][1]],
             '-', color='gray', linewidth=2, zorder=6, alpha=0.5)

# For the SOUTHERNMOST vertex, highlight its measurement line thicker
if southernmost_idx is not None:
    sd = edge_1b_data[southernmost_idx]
    v = sd['v']
    d_offB = sd['d_offB']
    sc = sd['color']

    if d_offB is not None:
        bx_B = v[0] - d_offB * np.cos(math_rad)
        by_B = v[1] - d_offB * np.sin(math_rad)
        ax2.plot([bx_B, v[0]], [by_B, v[1]],
                 '-', color=sc, lw=2.5, zorder=6, alpha=0.8)

        # Mark hit points on leg and both offsets
        if sd['d_leg'] is not None:
            bx_leg = v[0] - sd['d_leg'] * np.cos(math_rad)
            by_leg = v[1] - sd['d_leg'] * np.sin(math_rad)
            ax2.plot(bx_leg, by_leg, 'ko', markersize=7, zorder=9)
        if sd['d_offA'] is not None:
            bx_A = v[0] - sd['d_offA'] * np.cos(math_rad)
            by_A = v[1] - sd['d_offA'] * np.sin(math_rad)
            ax2.plot(bx_A, by_A, 's', color='darkorange', markersize=7,
                     zorder=9, markeredgecolor='black', markeredgewidth=0.5)
        ax2.plot(bx_B, by_B, 's', color='deepskyblue', markersize=7,
                 zorder=9, markeredgecolor='black', markeredgewidth=0.5)

# Info box
explain = (
    "TWO MEAN OFFSETS:\n"
    f"Offset A: +{mean_offset_A:.0f}m (orange)\n"
    f"Offset B: {mean_offset_B:.0f}m (blue)\n\n"
    "Ships in direction A are centered\n"
    "on offset A, direction B on offset B.\n\n"
    "Distance measured from each edge\n"
    "VERTEX back to the reference line\n"
    "along drift direction (reverse ray).\n\n"
    "Southernmost vertex shows all\n"
    "three distances explicitly."
)
ax2.text(0.02, 0.98, explain, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         family='monospace')

ax2.set_xlabel('Easting (m)', fontsize=10)
ax2.set_ylabel('Northing (m)', fontsize=10)
ax2.set_title('Level 1b: Two Mean Offsets per Traffic Direction', fontsize=12)
ax2.legend(loc='upper right', fontsize=9)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# --- Zoomed inset on the LEG SIDE ---
from mpl_toolkits.axes_grid1.inset_locator import inset_axes as inset_axes2, mark_inset as mark_inset2
ax_z2 = inset_axes2(ax2, width="40%", height="40%", loc='lower right', borderpad=1.5)

# Determine zoom bounds from where rays hit the leg area
# Use the southernmost vertex measurement line as the anchor
if southernmost_idx is not None and d_leg is not None:
    sd = edge_1b_data[southernmost_idx]
    v = sd['v']
    bx_leg = v[0] - sd['d_leg'] * np.cos(math_rad)
    by_leg = v[1] - sd['d_leg'] * np.sin(math_rad)
    # Zoom centered on the leg hit area, wide enough to show both offsets
    z_cx, z_cy = bx_leg, by_leg
    z_half = 2000  # 2km half-width
    z_xmin, z_xmax = z_cx - z_half, z_cx + z_half
    z_ymin, z_ymax = z_cy - z_half, z_cy + z_half
    zoom_box2 = box(z_xmin, z_ymin, z_xmax, z_ymax)

    # Draw bimodal traffic lines in inset
    for off_val, w in zip(offsets_A, weights_A):
        shifted = leg_line.parallel_offset(off_val, 'left')
        if shifted.geom_type == 'LineString' and not shifted.is_empty:
            clip = shifted.intersection(zoom_box2)
            if clip.geom_type == 'LineString' and not clip.is_empty:
                ax_z2.plot(*clip.xy, color='gray', lw=1.5, alpha=float(w)*0.5, zorder=3)
    for off_val, w in zip(offsets_B, weights_B):
        shifted = leg_line.parallel_offset(off_val, 'left')
        if shifted.geom_type == 'LineString' and not shifted.is_empty:
            clip = shifted.intersection(zoom_box2)
            if clip.geom_type == 'LineString' and not clip.is_empty:
                ax_z2.plot(*clip.xy, color='gray', lw=1.5, alpha=float(w)*0.5, zorder=3)

    # Leg and offset lines in inset
    for line_geom, clr, ls, lw in [
        (leg_line, 'black', '-', 3),
        (offset_line_A, 'darkorange', '--', 2.5),
        (offset_line_B, 'deepskyblue', '--', 2.5),
    ]:
        clip = line_geom.intersection(zoom_box2)
        if clip.geom_type == 'LineString' and not clip.is_empty:
            ax_z2.plot(*clip.xy, color=clr, linewidth=lw, linestyle=ls, zorder=5)

    # Draw ALL measurement lines in inset (vertex to offset B only)
    for ed in edge_1b_data:
        v = ed['v']
        db = ed['d_offB']
        if db is not None:
            bx = v[0] - db * np.cos(math_rad)
            by = v[1] - db * np.sin(math_rad)
            ax_z2.plot([bx, v[0]], [by, v[1]],
                       '--', color=ed['color'], lw=1, alpha=0.5, zorder=4)

    # Mark hit points for ALL vertices on all three lines
    for ed in edge_1b_data:
        v = ed['v']
        if ed['d_leg'] is not None:
            hx = v[0] - ed['d_leg'] * np.cos(math_rad)
            hy = v[1] - ed['d_leg'] * np.sin(math_rad)
            ax_z2.plot(hx, hy, 'o', color=ed['color'], markersize=4, zorder=9,
                       markeredgecolor='black', markeredgewidth=0.3)
        if ed['d_offA'] is not None:
            hx = v[0] - ed['d_offA'] * np.cos(math_rad)
            hy = v[1] - ed['d_offA'] * np.sin(math_rad)
            ax_z2.plot(hx, hy, 's', color='darkorange', markersize=4, zorder=9,
                       markeredgecolor='black', markeredgewidth=0.3)
        if ed['d_offB'] is not None:
            hx = v[0] - ed['d_offB'] * np.cos(math_rad)
            hy = v[1] - ed['d_offB'] * np.sin(math_rad)
            ax_z2.plot(hx, hy, 's', color='deepskyblue', markersize=4, zorder=9,
                       markeredgecolor='black', markeredgewidth=0.3)

    # Highlight the southernmost measurement line in the inset (to offset B)
    if southernmost_idx is not None:
        sd = edge_1b_data[southernmost_idx]
        v = sd['v']
        if sd['d_offB'] is not None:
            bx = v[0] - sd['d_offB'] * np.cos(math_rad)
            by = v[1] - sd['d_offB'] * np.sin(math_rad)
            ax_z2.plot([bx, v[0]], [by, v[1]],
                       '-', color=sd['color'], lw=2.5, alpha=0.8, zorder=6)

    # Distance labels ONLY for the highlighted (southernmost/red) line, placed south
    if southernmost_idx is not None:
        sd_label = edge_1b_data[southernmost_idx]
        v = sd_label['v']
        for ref_key, ref_color, ref_name in [
            ('d_offA', 'darkorange', 'A'),
            ('d_leg', 'black', 'Leg'),
            ('d_offB', 'deepskyblue', 'B'),
        ]:
            d = sd_label[ref_key]
            if d is not None:
                hx = v[0] - d * np.cos(math_rad)
                hy = v[1] - d * np.sin(math_rad)
                if z_xmin <= hx <= z_xmax and z_ymin <= hy <= z_ymax:
                    ax_z2.text(hx, hy - 300, f'{ref_name}: {d:.0f}m',
                               fontsize=8, color=ref_color, ha='center', va='top',
                               fontweight='bold', zorder=11,
                               bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                         ec=ref_color, alpha=0.85, lw=0.5))

    ax_z2.set_xlim(z_xmin, z_xmax)
    ax_z2.set_ylim(z_ymin, z_ymax)
    ax_z2.set_aspect('equal')
    ax_z2.grid(True, alpha=0.3)
    ax_z2.set_title('Zoom: ray hits on leg / offset A / offset B', fontsize=7, fontweight='bold')
    ax_z2.tick_params(labelsize=5)
    ax_z2.plot([], [], 'ko', markersize=4, label='On leg')
    ax_z2.plot([], [], 's', color='darkorange', markersize=4, label='On offset A')
    ax_z2.plot([], [], 's', color='deepskyblue', markersize=4, label='On offset B')
    ax_z2.legend(fontsize=5, loc='upper left')
    mark_inset2(ax2, ax_z2, loc1=2, loc2=3, fc='none', ec='0.5', lw=1, ls='--')

fig2_path = os.path.join(output_dir, 'level_1b_distribution_center.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig2_path}")
plt.close()

print()
print("SUMMARY")
print("=" * 80)
print(f"  Grounding probability for vertex 7432B79C edge from LEG_3:")
print(f"  One ship category (Tanker 225-250m), one direction (NW):")
print(f"  P = {total_grounding:.6e} events/year")
print()
print("  Components breakdown:")
print(f"    base (exposure)   = {base:.6e}")
print(f"    rose_prob (NW)    = {rose_prob}")
print(f"    hole_pct (total)  = {hole_pct:.6e}")
print(f"    The hole is distributed across valid edges proportional to their length")
print(f"    P(not repaired) depends on distance from leg to each edge")
