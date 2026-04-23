"""
Level 2: Drifting Grounding - Two Legs, Multiple Ship Categories
================================================================

Building on Level 1, this example adds:
    - A second traffic leg (Leg 6) that also has ships drifting toward the polygon
    - Multiple ship categories with different draughts and frequencies

Only ships with draught > polygon_depth are grounding hazards.
The calculation is the same formula per (leg, ship, direction) then summed.

From Level 2 onward, we always use start_from='leg_center' (distance from leg LINE).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import LineString, Polygon, Point
from scipy import stats
import pyproj

from drifting.engine import (
    LegState, ShipState, DriftConfig,
    compass_to_math_deg,
    directional_distance_to_point_from_offset_leg,
    corridor_width_m,
    build_directional_corridor,
    edge_average_distance_m,
)
from compute.basic_equations import get_not_repaired
from geometries.analytical_probability import (
    compute_probability_analytical,
    _extract_polygon_rings,
)


# =============================================================================
# 1. GEOMETRY SETUP
# =============================================================================

# Coordinate transformation: WGS84 -> UTM 33N
proj_wgs84 = pyproj.CRS("EPSG:4326")
proj_utm = pyproj.CRS("EPSG:32633")
transformer = pyproj.Transformer.from_crs(proj_wgs84, proj_utm, always_xy=True)

def to_utm(lon, lat):
    return transformer.transform(lon, lat)

# Leg 3: WP_2_End_Point (55.16728, 14.24187) -> WP_3_End_Point (55.39937, 14.59271)
# Bearing ~41 deg NE, length ~34 km
leg3_start = to_utm(14.24187, 55.16728)
leg3_end = to_utm(14.59271, 55.39937)
leg3_line = LineString([leg3_start, leg3_end])

# Leg 6: WP_5_End_Point (55.10675, 14.19053) -> WP_2_End_Point (55.16728, 14.24187)
# Bearing ~41 deg NE, length ~7.5 km
leg6_start = to_utm(14.19053, 55.10675)
leg6_end = to_utm(14.24187, 55.16728)
leg6_line = LineString([leg6_start, leg6_end])

# Depth polygon BD5A4C46 (depth=12m) -- more natural shape with 8 vertices
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
poly_utm_coords = [to_utm(lon, lat) for lon, lat in poly_vertices_lonlat]
depth_polygon = Polygon(poly_utm_coords)

print("=" * 80)
print("LEVEL 2: Two Legs, Multiple Ship Categories")
print("=" * 80)
print()
print("GEOMETRY:")
print(f"  Leg 3: length={leg3_line.length:.0f}m, bearing ~41 deg NE")
print(f"  Leg 6: length={leg6_line.length:.0f}m, bearing ~41 deg NE")
print(f"  Polygon BD5A4C46: depth=12m")
print()

# =============================================================================
# 2. SHIP CATEGORIES
# =============================================================================

# Multiple ship categories with different draughts
# Only ships with draught > 12m can ground on this polygon
ship_categories = [
    {'name': 'Oil tanker 225-250m',      'draught': 14.27, 'speed_kts': 12.5, 'freq_leg3': 610, 'freq_leg6': 50},
    {'name': 'General cargo 225-250m',   'draught': 11.82, 'speed_kts': 13.0, 'freq_leg3': 450, 'freq_leg6': 40},
    {'name': 'Bulk carrier 250-275m',    'draught': 16.53, 'speed_kts': 13.5, 'freq_leg3': 180, 'freq_leg6': 15},
    {'name': 'Container 275-300m',       'draught': 13.50, 'speed_kts': 18.0, 'freq_leg3': 95,  'freq_leg6': 8},
    {'name': 'Passenger 100-125m',       'draught': 5.80,  'speed_kts': 16.0, 'freq_leg3': 320, 'freq_leg6': 280},
]

polygon_depth = 12.0  # meters

print("SHIP CATEGORIES:")
print(f"  {'Name':<30} {'Draught':>8} {'Grounds?':>8} {'Freq L3':>8} {'Freq L6':>8}")
print(f"  {'-'*30} {'-------':>8} {'-------':>8} {'-------':>8} {'-------':>8}")
for s in ship_categories:
    grounds = 'YES' if s['draught'] > polygon_depth else 'no'
    print(f"  {s['name']:<30} {s['draught']:>7.2f}m {grounds:>8} {s['freq_leg3']:>8} {s['freq_leg6']:>8}")
print()
print(f"  Only ships with draught > {polygon_depth}m can ground on this polygon.")
print(f"  Categories that ground: {sum(1 for s in ship_categories if s['draught'] > polygon_depth)}/5")
print()

# =============================================================================
# 3. DRIFT PARAMETERS
# =============================================================================

drift_speed_kts = 1.94
drift_speed_ms = drift_speed_kts * 1852 / 3600
blackout_rate = 1.0  # per year
blackout_per_hour = blackout_rate / (365.25 * 24)
rose_prob = 1.0 / 8.0

# Different drift directions per leg:
#   LEG_3: NW (315) -- polygon is NW of Leg 3
#   LEG_6: N  (0)   -- polygon is roughly N of Leg 6
drift_directions = {'LEG_3': 315, 'LEG_6': 0}

repair_data = {
    'use_lognormal': 1,
    'std': 1.0,
    'loc': 0.0,
    'scale': 1.0,
}

lateral_sigma = 500.0
reach_distance = 50000.0

cfg = DriftConfig(
    reach_distance_m=reach_distance,
    corridor_sigma_multiplier=3.0,
    use_leg_offset_for_distance=False,
)

# =============================================================================
# 4. COMPUTE PER-LEG PROBABILITY HOLES
# =============================================================================

legs = [
    {'name': 'LEG_3', 'line': leg3_line, 'start': leg3_start, 'end': leg3_end},
    {'name': 'LEG_6', 'line': leg6_line, 'start': leg6_start, 'end': leg6_end},
]

polygon_rings = _extract_polygon_rings(depth_polygon)

print("PROBABILITY HOLES (fraction of lateral distribution hitting polygon):")
print("-" * 80)

for leg_info in legs:
    line = leg_info['line']
    leg_coords = np.array(line.coords)
    leg_start_np = leg_coords[0]
    leg_end_np = leg_coords[-1]
    leg_vec = leg_end_np - leg_start_np
    leg_len = line.length
    leg_dir = leg_vec / leg_len
    perp_dir = np.array([-leg_dir[1], leg_dir[0]])

    dd = drift_directions[leg_info['name']]
    leg_info['drift_direction'] = dd
    math_deg = compass_to_math_deg(dd)
    angle_rad = np.radians(math_deg)
    drift_vec = np.array([np.cos(angle_rad), np.sin(angle_rad)])

    lateral_dist = stats.norm(0, lateral_sigma)
    lateral_range = 5.0 * lateral_sigma

    hole = compute_probability_analytical(
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
    leg_info['hole_pct'] = hole
    dir_label = 'NW' if dd == 315 else 'N'
    print(f"  {leg_info['name']} (drift {dir_label}/{dd}): hole_pct = {hole:.6e} (length={leg_len:.0f}m)")

print()

# =============================================================================
# 5. COMPUTE EDGE DISTANCES PER LEG
# =============================================================================

edges = list(zip(poly_utm_coords[:-1], poly_utm_coords[1:]))

print("EDGE DISTANCES (from leg LINE in respective drift directions):")
print("-" * 80)

for leg_info in legs:
    dd = leg_info['drift_direction']
    leg_state = LegState(
        leg_id=leg_info['name'],
        line=leg_info['line'],
        mean_offset_m=0.0,
        lateral_sigma_m=lateral_sigma,
    )
    leg_info['leg_state'] = leg_state

    # Get edge distances using the leg's drift direction
    corridor = build_directional_corridor(leg_state, dd, cfg)
    leg_info['corridor'] = corridor

    # Compute drift vector for front-facing filter
    math_deg = compass_to_math_deg(dd)
    drift_ux = np.cos(np.radians(math_deg))
    drift_uy = np.sin(np.radians(math_deg))

    edge_data = []
    for i, (p1, p2) in enumerate(edges):
        d_avg = edge_average_distance_m(
            leg_state, dd, (p1, p2), use_leg_offset=False
        )
        edge_line = LineString([p1, p2])
        in_corridor = corridor.intersects(edge_line)
        # Front-facing: outward normal opposes drift direction (CCW polygon)
        ex, ey = p2[0] - p1[0], p2[1] - p1[1]
        nx, ny = ey, -ex
        faces_drift = (nx * drift_ux + ny * drift_uy) < 0
        edge_data.append({
            'idx': i, 'p1': p1, 'p2': p2, 'dist': d_avg,
            'in_corridor': in_corridor, 'faces_drift': faces_drift,
            'length': edge_line.length,
        })

    valid_edges = [e for e in edge_data
                   if e['in_corridor'] and e['faces_drift'] and e['dist'] is not None]
    shadowed_edges = [e for e in edge_data
                      if e['in_corridor'] and not e['faces_drift'] and e['dist'] is not None]
    leg_info['edge_data'] = edge_data
    leg_info['valid_edges'] = valid_edges
    leg_info['shadowed_edges'] = shadowed_edges
    leg_info['drift_ux'] = drift_ux
    leg_info['drift_uy'] = drift_uy

    # Store simple edge_dists for backward compat
    edge_dists = [e['dist'] for e in edge_data]
    leg_info['edge_dists'] = edge_dists
    dir_label = 'NW' if dd == 315 else 'N'
    print(f"  {leg_info['name']} (drift {dir_label}/{dd}):")
    for e in edge_data:
        status = ''
        if not e['in_corridor']:
            status = ' [outside corridor]'
        elif not e['faces_drift']:
            status = ' [shadowed]'
        d_str = f"{e['dist']:.0f}m" if e['dist'] is not None else "N/A"
        print(f"    Edge {e['idx']}: {d_str}, faces_drift={e['faces_drift']}{status}")
    print(f"    Front-facing: {len(valid_edges)} edges, shadowed: {len(shadowed_edges)} edges")
    print()

# =============================================================================
# 6. PER-CATEGORY, PER-LEG GROUNDING CALCULATION
# =============================================================================

print("GROUNDING PROBABILITY PER (LEG, SHIP CATEGORY):")
print("=" * 80)
print()
print("Formula: P = (L/(V*1852)) * freq * lambda_bo * rose * hole_edge * P_NR(d)")
print()

results = []
total_grounding = 0.0

for leg_info in legs:
    line = leg_info['line']
    hole_pct = leg_info['hole_pct']
    valid_edges = leg_info['valid_edges']

    # Distribute hole across front-facing edges (proportional to edge length)
    valid_edge_data = []
    total_len = sum(e['length'] for e in valid_edges)
    for e in valid_edges:
        edge_hole = hole_pct * (e['length'] / total_len) if total_len > 0 else 0
        valid_edge_data.append({**e, 'edge_hole': edge_hole})

    leg_info['valid_edge_data'] = valid_edge_data

    print(f"  {leg_info['name']} (length={line.length:.0f}m, hole={hole_pct:.4e}):")
    print(f"  {'Ship category':<30} {'Draught':>7} {'Freq':>5} {'Base':>10} "
          f"{'P_ground':>12} {'Note':>10}")
    print(f"  {'-'*30} {'-'*7} {'-'*5} {'-'*10} {'-'*12} {'-'*10}")

    for ship in ship_categories:
        freq = ship['freq_leg3'] if 'LEG_3' in leg_info['name'] else ship['freq_leg6']

        if ship['draught'] <= polygon_depth:
            print(f"  {ship['name']:<30} {ship['draught']:>6.2f}m {freq:>5} "
                  f"{'---':>10} {'---':>12} {'skip':>10}")
            continue

        hours_present = (line.length / (ship['speed_kts'] * 1852)) * freq
        base = hours_present * blackout_per_hour

        # Sum over all valid edges
        ship_total = 0.0
        for e in valid_edge_data:
            p_nr = get_not_repaired(repair_data, drift_speed_ms, e['dist'])
            contrib = base * rose_prob * e['edge_hole'] * p_nr
            ship_total += contrib

        results.append({
            'leg': leg_info['name'],
            'ship': ship['name'],
            'draught': ship['draught'],
            'freq': freq,
            'base': base,
            'p_ground': ship_total,
        })
        total_grounding += ship_total

        print(f"  {ship['name']:<30} {ship['draught']:>6.2f}m {freq:>5} "
              f"{base:>10.4e} {ship_total:>12.4e}")

    print()

print("=" * 80)
print(f"TOTAL GROUNDING (all legs, all ships, direction NW):")
print(f"  P_total = {total_grounding:.6e} events/year")
print()

# Show contribution breakdown
print("CONTRIBUTION BREAKDOWN:")
for r in sorted(results, key=lambda x: x['p_ground'], reverse=True):
    pct = 100 * r['p_ground'] / total_grounding if total_grounding > 0 else 0
    print(f"  {r['leg']} / {r['ship']:<28}: {r['p_ground']:.4e} ({pct:.1f}%)")

# =============================================================================
# 7. GENERATE FIGURE
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Helper to draw polygon/multipolygon
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

# Plot corridors with shadow zones (shortened for readability)
from shapely.affinity import translate as _translate
from shapely.ops import unary_union as _unary_union

shadow_data = {}
for leg_info in legs:
    dd = leg_info['drift_direction']
    dux = leg_info['drift_ux']
    duy = leg_info['drift_uy']
    # Per-leg display corridor: extend to 1.2x the farthest valid edge distance
    leg_max_dist = max((e['dist'] for e in leg_info['valid_edges']), default=10000)
    cfg_disp = DriftConfig(reach_distance_m=1.2 * leg_max_dist, corridor_sigma_multiplier=3.0)
    corridor_display = build_directional_corridor(leg_info['leg_state'], dd, cfg_disp)

    # Build quad-sweep shadow in correct drift direction
    _diag = np.sqrt((corridor_display.bounds[2]-corridor_display.bounds[0])**2 +
                    (corridor_display.bounds[3]-corridor_display.bounds[1])**2)
    _ext = _diag * 2
    _far = _translate(depth_polygon, xoff=dux*_ext, yoff=duy*_ext)
    _oc = list(depth_polygon.exterior.coords)[:-1]
    _fc = list(_far.exterior.coords)[:-1]
    _quads = []
    for _qi in range(len(_oc)):
        _qj = (_qi + 1) % len(_oc)
        _q = Polygon([_oc[_qi], _oc[_qj], _fc[_qj], _fc[_qi]])
        if _q.is_valid and _q.area > 0:
            _quads.append(_q)
    shadow_poly = _unary_union([depth_polygon, _far] + _quads)

    corridor_with_hole = corridor_display.difference(shadow_poly)
    shadow_visible = corridor_display.intersection(shadow_poly).difference(depth_polygon)

    shadow_data[leg_info['name']] = {
        'corridor_display': corridor_display,
        'corridor_with_hole': corridor_with_hole,
        'shadow_visible': shadow_visible,
    }

    # Draw corridor for both legs, blue for LEG_3, darkgreen for LEG_6
    if leg_info['name'] == 'LEG_3':
        _draw_geom(ax, corridor_with_hole, 0.06, 'blue', 'blue', 0.5, 'Drift corridor')
        _draw_geom(ax, shadow_visible, 0.06, 'gray', 'none', 0, 'Shadow zone')
    else:
        _draw_geom(ax, corridor_with_hole, 0.06, 'darkgreen', 'darkgreen', 0.5, 'Drift corridor (LEG_6)')
        _draw_geom(ax, shadow_visible, 0.10, 'darkgreen', 'none', 0, 'Shadow zone (LEG_6)')

# Overlay hatch where both shadows overlap in main image
if not shadow_data['LEG_3']['shadow_visible'].is_empty and not shadow_data['LEG_6']['shadow_visible'].is_empty:
    overlap = shadow_data['LEG_3']['shadow_visible'].intersection(shadow_data['LEG_6']['shadow_visible'])
    if not overlap.is_empty:
        from matplotlib.patches import PathPatch
        import matplotlib as mpl
        if overlap.geom_type == 'Polygon':
            patch = PathPatch(mpl.path.Path(list(overlap.exterior.coords)),
                              facecolor='none', edgecolor='black', lw=0.8,
                              hatch='///', zorder=10)
            ax.add_patch(patch)
        elif overlap.geom_type == 'MultiPolygon':
            for g in overlap.geoms:
                patch = PathPatch(mpl.path.Path(list(g.exterior.coords)),
                                  facecolor='none', edgecolor='black', lw=0.8,
                                  hatch='///', zorder=10)
                ax.add_patch(patch)

# Plot legs
colors_legs = ['black', 'darkgreen']
for i, leg_info in enumerate(legs):
    line = leg_info['line']
    ax.plot(*line.xy, color=colors_legs[i], linewidth=3, label=f"{leg_info['name']} ({line.length:.0f}m)")
    # Mark endpoints
    coords = list(line.coords)
    ax.plot(coords[0][0], coords[0][1], 'o', color=colors_legs[i], markersize=8)
    ax.plot(coords[-1][0], coords[-1][1], 's', color=colors_legs[i], markersize=8)

# Plot polygon
poly_x = [c[0] for c in poly_utm_coords]
poly_y = [c[1] for c in poly_utm_coords]
ax.fill(poly_x, poly_y, alpha=0.4, fc='brown', ec='darkred', lw=2,
        label=f'Depth polygon (12m)')

# Draw drift direction arrows (one per leg)
drift_colors = {'LEG_3': 'blue', 'LEG_6': 'darkgreen'}
for leg_info in legs:
    dd = leg_info['drift_direction']
    center = leg_info['line'].centroid
    math_rad = np.radians(compass_to_math_deg(dd))
    arrow_len = 5000
    dx, dy = arrow_len * np.cos(math_rad), arrow_len * np.sin(math_rad)
    c = drift_colors[leg_info['name']]
    ax.annotate('', xy=(center.x + dx, center.y + dy), xytext=(center.x, center.y),
                arrowprops=dict(arrowstyle='->', color=c, lw=2.5))
    dir_label = 'NW' if dd == 315 else 'N'
    ax.text(center.x + dx * 0.6, center.y + dy * 0.6 + 300,
            f'{leg_info["name"]}\nDrift {dir_label} ({dd})', color=c, fontsize=9, ha='center')

# Distance annotations for Leg 3 (main contributor) -- one front-facing edge
math_rad_leg3 = np.radians(compass_to_math_deg(drift_directions['LEG_3']))
leg3_valid = legs[0]['valid_edge_data']
if leg3_valid:
    e = leg3_valid[0]
    mid = ((e['p1'][0]+e['p2'][0])/2, (e['p1'][1]+e['p2'][1])/2)
    back_x = mid[0] - e['dist'] * np.cos(math_rad_leg3)
    back_y = mid[1] - e['dist'] * np.sin(math_rad_leg3)
    ax.plot([back_x, mid[0]], [back_y, mid[1]], 'r--', lw=1.2, alpha=0.6)
    label_x = (back_x + mid[0]) / 2
    label_y = (back_y + mid[1]) / 2
    ax.text(label_x, label_y, f'L3: {e["dist"]:.0f}m', fontsize=8, color='red',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7, ec='none'))

# Distance annotation for Leg 6 -- one front-facing edge
math_rad_leg6 = np.radians(compass_to_math_deg(drift_directions['LEG_6']))
leg6_valid = legs[1]['valid_edge_data']
if leg6_valid:
    e = leg6_valid[0]
    mid = ((e['p1'][0]+e['p2'][0])/2, (e['p1'][1]+e['p2'][1])/2)
    back_x = mid[0] - e['dist'] * np.cos(math_rad_leg6)
    back_y = mid[1] - e['dist'] * np.sin(math_rad_leg6)
    ax.plot([back_x, mid[0]], [back_y, mid[1]], '--', color='darkgreen', lw=1.2, alpha=0.6)
    label_x = (back_x + mid[0]) / 2
    label_y = (back_y + mid[1]) / 2
    ax.text(label_x, label_y, f'L6: {e["dist"]:.0f}m', fontsize=8, color='darkgreen',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7, ec='none'))

# Highlight front-facing edges, show shadowed in gray
for leg_info in legs:
    for e in leg_info['valid_edges']:
        ax.plot([e['p1'][0], e['p2'][0]], [e['p1'][1], e['p2'][1]],
                'r-', linewidth=3, zorder=7)
    for e in leg_info['shadowed_edges']:
        ax.plot([e['p1'][0], e['p2'][0]], [e['p1'][1], e['p2'][1]],
                '-', color='gray', linewidth=2, zorder=6, alpha=0.5)

# Summary text box
textstr = (
    f"Drift speed: {drift_speed_kts}kts\n"
    f"LEG_3: drift NW (315)\n"
    f"LEG_6: drift N (0)\n"
    f"Blackout: {blackout_rate}/yr\n"
    f"Rose: {rose_prob} (uniform)\n"
    f"\nShip categories grounding: 3/5\n"
    f"(draught > {polygon_depth}m)\n"
    f"\nTotal P(ground) = {total_grounding:.4e}/yr\n"
    f"\nTop contributor:\n"
    f"  {results[0]['ship']}\n"
    f"  from {results[0]['leg']}: {results[0]['p_ground']:.4e}"
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props, family='monospace')

ax.set_xlabel('Easting (m)', fontsize=10)
ax.set_ylabel('Northing (m)', fontsize=10)
ax.set_title('Level 2: Two Legs, Multiple Ship Categories (NW + N drift)', fontsize=12)
ax.legend(loc='upper right', fontsize=9)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Zoomed inset on the grounding polygon

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from shapely.geometry import box as shapely_box
# Enlarge and move the inset further down
ax_zoom = inset_axes(ax, width="38%", height="38%", loc='lower right',
                     borderpad=3)
pad = 800
zoom_box = shapely_box(min(poly_x)-pad, min(poly_y)-pad, max(poly_x)+pad, max(poly_y)+pad)

# --- LEG_3 (main contributor) ---
sd3 = shadow_data['LEG_3']
shadow_clip3 = sd3['shadow_visible'].intersection(zoom_box) if not sd3['shadow_visible'].is_empty else Polygon()
_draw_geom(ax_zoom, shadow_clip3, 0.10, 'gray', 'none', 0)
corr_clip3 = sd3['corridor_with_hole'].intersection(zoom_box)
_draw_geom(ax_zoom, corr_clip3, 0.10, 'blue', 'blue', 0.5)

# --- LEG_6 (now also shown in inset) ---
sd6 = shadow_data['LEG_6']
shadow_clip6 = sd6['shadow_visible'].intersection(zoom_box) if not sd6['shadow_visible'].is_empty else Polygon()
if not shadow_clip6.is_empty:
    # Draw LEG_6 shadow in the same color as main image (darkgreen, semi-transparent)
    _draw_geom(ax_zoom, shadow_clip6, 0.18, 'darkgreen', 'darkgreen', 2.0)

# If both LEG_3 and LEG_6 shadows overlap, overlay a hatch or darker color
from shapely.ops import unary_union
if not shadow_clip3.is_empty and not shadow_clip6.is_empty:
    overlap = shadow_clip3.intersection(shadow_clip6)
    if not overlap.is_empty:
        # Overlay with black hatching to indicate blocked from both directions
        from matplotlib.patches import PathPatch
        import matplotlib as mpl
        if overlap.geom_type == 'Polygon':
            patch = PathPatch(mpl.path.Path(list(overlap.exterior.coords)),
                              facecolor='none', edgecolor='black', lw=0.8,
                              hatch='///', zorder=10)
            ax_zoom.add_patch(patch)
        elif overlap.geom_type == 'MultiPolygon':
            for g in overlap.geoms:
                patch = PathPatch(mpl.path.Path(list(g.exterior.coords)),
                                  facecolor='none', edgecolor='black', lw=0.8,
                                  hatch='///', zorder=10)
                ax_zoom.add_patch(patch)
corr_clip6 = sd6['corridor_with_hole'].intersection(zoom_box)
_draw_geom(ax_zoom, corr_clip6, 0.10, 'darkgreen', 'darkgreen', 0.4)

ax_zoom.fill(poly_x, poly_y, alpha=0.4, fc='brown', ec='darkred', lw=2)
# Front-facing edges in red (LEG_3) and darkgreen (LEG_6), shadowed in gray
for e in legs[0]['valid_edges']:
    ax_zoom.plot([e['p1'][0], e['p2'][0]], [e['p1'][1], e['p2'][1]],
                 'r-', linewidth=2.5, zorder=7)
for e in legs[0]['shadowed_edges']:
    ax_zoom.plot([e['p1'][0], e['p2'][0]], [e['p1'][1], e['p2'][1]],
                 '-', color='gray', linewidth=1.5, zorder=6, alpha=0.5)
for e in legs[1]['valid_edges']:
    ax_zoom.plot([e['p1'][0], e['p2'][0]], [e['p1'][1], e['p2'][1]],
                 '-', color='darkgreen', linewidth=2.0, zorder=7)
for e in legs[1]['shadowed_edges']:
    ax_zoom.plot([e['p1'][0], e['p2'][0]], [e['p1'][1], e['p2'][1]],
                 '-', color='gray', linewidth=1.2, zorder=6, alpha=0.5)
ax_zoom.set_xlim(min(poly_x)-pad, max(poly_x)+pad)
ax_zoom.set_ylim(min(poly_y)-pad, max(poly_y)+pad)
ax_zoom.set_aspect('equal')
ax_zoom.grid(True, alpha=0.3)
ax_zoom.set_title('Polygon + shadow', fontsize=9, fontweight='bold')
ax_zoom.tick_params(labelsize=7)
mark_inset(ax, ax_zoom, loc1=4, loc2=2, fc='none', ec='0.5', lw=1, ls='--')

output_dir = os.path.dirname(__file__)
fig_path = os.path.join(output_dir, 'level_2_two_legs_multi_ships.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {fig_path}")
plt.close()
