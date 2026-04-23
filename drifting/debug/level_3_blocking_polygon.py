"""
Level 3: Drifting Grounding with a Blocking Polygon (Shadow Coverage)
=====================================================================

Building on Level 2, this example demonstrates how an upstream obstacle
SHADOWS a grounding polygon in OMRAT's drift model.  The shadow is purely
GEOMETRIC -- it is not a multiplicative 'remaining' factor carried from
one obstacle to the next.  A blocker simply prevents drifting ships from
reaching the portion of the target that lies within the blocker's shadow.

Algorithm
---------
1. Compute the blocker's shadow polygon:  sweep the blocker along the
   drift direction (quad-sweep algorithm, same as used by OMRAT's drift
   corridor clipping).

2. Split the target into an unshadowed part and a shadowed part:
        target_unshadowed = target - blocker_shadow
        target_shadowed   = target - target_unshadowed

3. The effective probability hole for the target is the analytical hole
   of its unshadowed subregion only:
        h_target_eff = hole(target_unshadowed)

   Equivalently:
        shadow_coverage = 1 - h_target_eff / h_target
        h_target_eff    = h_target * (1 - shadow_coverage)

4. The grounding probability for the target becomes

        P(target) = base * r_p * h_target_eff * P_NR(d_target)

   with NO multiplicative 'remaining' factor from the blocker.  The
   blocker only changes the target's effective hole.

Three shadow-coverage scenarios are computed on the SAME target polygon
(BD5A4C46, 12 m, 8-vertex shape from levels 1-2):

    Scenario A -- 0% coverage
        Blocker offset laterally so its shadow misses the target entirely.

    Scenario B -- ~30% coverage
        Blocker sized and placed so ~30% of the target falls in its shadow.

    Scenario C -- 100% coverage
        Blocker wider than the target on the cross-drift axis and placed
        directly upstream; the target lies entirely in its shadow.

All values printed below come directly from OMRAT's actual functions:
compute_probability_analytical(), edge_average_distance_m(),
get_not_repaired().
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, Point, box, MultiPolygon
from shapely.affinity import translate as _translate
from shapely.ops import unary_union as _unary_union
from scipy import stats
import pyproj

from drifting.engine import (
    LegState, DriftConfig,
    compass_to_math_deg,
    build_directional_corridor,
    edge_average_distance_m,
    directional_distance_to_point_from_offset_leg,
)
from compute.basic_equations import get_not_repaired
from geometries.analytical_probability import (
    compute_probability_analytical,
    _extract_polygon_rings,
)


# =============================================================================
# 1. GEOMETRY SETUP (same as Level 1/2)
# =============================================================================

proj_wgs84 = pyproj.CRS("EPSG:4326")
proj_utm = pyproj.CRS("EPSG:32633")
transformer = pyproj.Transformer.from_crs(proj_wgs84, proj_utm, always_xy=True)

def to_utm(lon, lat):
    return transformer.transform(lon, lat)

# Leg 3
leg3_start = to_utm(14.24187, 55.16728)
leg3_end = to_utm(14.59271, 55.39937)
leg3_line = LineString([leg3_start, leg3_end])

# Target polygon (same 8-vertex shape as L1/L2)
target_vertices_lonlat = [
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
target_utm = [to_utm(lon, lat) for lon, lat in target_vertices_lonlat]
target_polygon = Polygon(target_utm)

# =============================================================================
# 2. PARAMETERS
# =============================================================================

ship_draught = 14.27
ship_speed_kts = 12.5
ship_freq = 610
drift_speed_kts = 1.94
drift_speed_ms = drift_speed_kts * 1852 / 3600     # 0.998 m/s
blackout_rate = 1.0
blackout_per_hour = blackout_rate / (365.25 * 24)
rose_prob = 1.0 / 8.0
lateral_sigma = 500.0
reach_distance = 50000.0
drift_direction = 315                               # NW

repair_data = {'use_lognormal': 1, 'std': 1.0, 'loc': 0.0, 'scale': 1.0}

cfg = DriftConfig(reach_distance_m=reach_distance, corridor_sigma_multiplier=3.0)
leg_state = LegState(leg_id="LEG_3", line=leg3_line,
                      mean_offset_m=0.0, lateral_sigma_m=lateral_sigma)

math_deg = compass_to_math_deg(drift_direction)
angle_rad = np.radians(math_deg)
drift_vec_unit = np.array([np.cos(angle_rad), np.sin(angle_rad)])
cross_drift_vec = np.array([-drift_vec_unit[1], drift_vec_unit[0]])
drift_ux, drift_uy = drift_vec_unit[0], drift_vec_unit[1]

print("=" * 80)
print("LEVEL 3: Drifting Grounding with a Blocking Polygon (Shadow Coverage)")
print("=" * 80)
print()
print("PARAMETERS:")
print(f"  Leg length L            = {leg3_line.length:.1f} m")
print(f"  Ship speed V            = {ship_speed_kts} kts")
print(f"  Frequency f             = {ship_freq} ships/year")
print(f"  Drift speed V_drift     = {drift_speed_kts} kts ({drift_speed_ms:.3f} m/s)")
print(f"  Blackout rate lambda_bo = {blackout_rate}/year ({blackout_per_hour:.4e}/hour)")
print(f"  Rose probability r_p    = {rose_prob:.4f} (uniform)")
print(f"  Lateral sigma           = {lateral_sigma} m")
print(f"  Drift direction         = {drift_direction} deg compass ({math_deg:.0f} deg math)")
print(f"  Drift vector            = ({drift_ux:.4f}, {drift_uy:.4f})")
print(f"  Reach distance          = {reach_distance:.0f} m")
print()

# =============================================================================
# 3. ANALYTICAL HOLE HELPER AND QUAD-SWEEP SHADOW HELPER
# =============================================================================

leg_coords = np.array(leg3_line.coords)
leg_start_np = leg_coords[0]
leg_vec = leg_coords[-1] - leg_coords[0]
leg_len = leg3_line.length
leg_dir = leg_vec / leg_len
perp_dir = np.array([-leg_dir[1], leg_dir[0]])

lateral_dist = stats.norm(0, lateral_sigma)
lateral_range = 5.0 * lateral_sigma

def compute_hole(geom):
    """Analytical probability hole for any Polygon or MultiPolygon geom."""
    if geom is None or geom.is_empty:
        return 0.0
    rings = _extract_polygon_rings(geom)
    if not rings:
        return 0.0
    return compute_probability_analytical(
        leg_start=leg_start_np, leg_vec=leg_vec, perp_dir=perp_dir,
        drift_vec=drift_vec_unit, distance=reach_distance,
        lateral_range=lateral_range, polygon_rings=rings,
        dists=[lateral_dist], weights=np.array([1.0]), n_slices=400,
    )

def build_quad_shadow(poly, extrude_length):
    """Quad-sweep shadow: sweep each polygon edge along the drift direction."""
    far = _translate(poly, xoff=drift_ux * extrude_length,
                     yoff=drift_uy * extrude_length)
    orig = list(poly.exterior.coords)[:-1]
    far_c = list(far.exterior.coords)[:-1]
    quads = []
    for qi in range(len(orig)):
        qj = (qi + 1) % len(orig)
        q = Polygon([orig[qi], orig[qj], far_c[qj], far_c[qi]])
        if q.is_valid and q.area > 0:
            quads.append(q)
    return _unary_union([poly, far] + quads)

# =============================================================================
# 4. TARGET HOLE (reference value) AND DISTANCE
# =============================================================================

target_hole = compute_hole(target_polygon)

target_vertices = list(target_polygon.exterior.coords)[:-1]
target_dists = [directional_distance_to_point_from_offset_leg(
                    leg_state, drift_direction, Point(v))
                for v in target_vertices]
target_dists = [d for d in target_dists if d is not None]
d_target = float(np.mean(target_dists))
p_nr_target = get_not_repaired(repair_data, drift_speed_ms, d_target)

print("TARGET POLYGON (no blocker):")
print(f"  Reference hole  h_target = {target_hole:.6e}")
print(f"  Mean distance   d_target = {d_target:.1f} m")
print(f"  P_NR(d_target)           = {p_nr_target:.6e}")
print()

# =============================================================================
# 5. CONSTRUCT THE THREE BLOCKERS AT 0%, 30%, 100% SHADOW COVERAGE
# =============================================================================

def cross_drift_interval(poly_coords):
    projs = [np.dot(np.array(pt), cross_drift_vec) for pt in poly_coords]
    return min(projs), max(projs)

target_cd_lo, target_cd_hi = cross_drift_interval(target_vertices)
target_cd_center = 0.5 * (target_cd_lo + target_cd_hi)
target_cd_width = target_cd_hi - target_cd_lo
target_centroid = np.array([target_polygon.centroid.x, target_polygon.centroid.y])

# Common along-drift placement for the blocker: 4 km upstream of the target
blocker_center_along = target_centroid - 4000.0 * drift_vec_unit
blocker_half_along = 400.0   # 800 m deep in drift direction

# Compute shadow polygons (quad-sweep).  Extrude at least target_distance + 5 km.
extrude_len = d_target + 5000.0

def make_blocker(center_cross_offset_m, half_cross_m):
    """Rectangular blocker aligned to the drift axis."""
    center_pt = blocker_center_along + center_cross_offset_m * cross_drift_vec
    cx = half_cross_m * cross_drift_vec
    ay = blocker_half_along * drift_vec_unit
    coords = [
        tuple(center_pt - cx - ay),
        tuple(center_pt + cx - ay),
        tuple(center_pt + cx + ay),
        tuple(center_pt - cx + ay),
        tuple(center_pt - cx - ay),
    ]
    return coords, Polygon(coords)

# Scenario A -- 0% coverage: blocker shifted WELL clear of target on cross-drift
blocker_A_offset = +1500.0   # meters along the cross-drift axis from target center
blocker_A_half_cross = 300.0
blocker_A_utm, blocker_A_poly = make_blocker(blocker_A_offset, blocker_A_half_cross)

# Scenario B -- ~30% hole coverage.  Find a blocker cross-drift half-width
# that yields ~30% hole coverage by a small search.  The blocker sits on one
# side of the target's cross-drift interval so the shadow covers a band of
# one edge rather than straddling the centre.
def _coverage_for_half(half_cross_m):
    cx_offset = (target_cd_hi - half_cross_m) - target_cd_center
    _, poly = make_blocker(cx_offset, half_cross_m)
    sh = build_quad_shadow(poly, extrude_len)
    unshadow = target_polygon.difference(sh)
    h = compute_hole(unshadow)
    return 1.0 - h / target_hole if target_hole > 0 else 0.0

# Binary search for the half-width that gives ~30% coverage
lo, hi = 50.0, target_cd_width
best_half = 0.5 * 0.5 * target_cd_width
for _ in range(30):
    mid = 0.5 * (lo + hi)
    cov = _coverage_for_half(mid)
    if cov < 0.30:
        lo = mid
    else:
        hi = mid
    best_half = mid
blocker_B_half_cross = best_half
blocker_B_center_cross = (target_cd_hi - blocker_B_half_cross) - target_cd_center
blocker_B_utm, blocker_B_poly = make_blocker(blocker_B_center_cross, blocker_B_half_cross)

# Scenario C -- 100% coverage: blocker wider than target
blocker_C_half_cross = 0.5 * target_cd_width + 200.0
blocker_C_utm, blocker_C_poly = make_blocker(0.0, blocker_C_half_cross)

shadow_A = build_quad_shadow(blocker_A_poly, extrude_len)
shadow_B = build_quad_shadow(blocker_B_poly, extrude_len)
shadow_C = build_quad_shadow(blocker_C_poly, extrude_len)

# =============================================================================
# 6. EFFECTIVE TARGET HOLE PER SCENARIO
# =============================================================================

def scenario_stats(name, blocker_poly, shadow_poly):
    target_unshadowed = target_polygon.difference(shadow_poly)
    target_shadowed = target_polygon.intersection(shadow_poly)
    area_total = target_polygon.area
    area_shadowed = target_shadowed.area if not target_shadowed.is_empty else 0.0
    coverage_area = area_shadowed / area_total if area_total > 0 else 0.0

    h_unshadowed = compute_hole(target_unshadowed)
    coverage_hole = 1.0 - (h_unshadowed / target_hole) if target_hole > 0 else 0.0
    P_target = base * rose_prob * h_unshadowed * p_nr_target

    # Blocker's own grounding (if it is a grounding hazard) -- informational only.
    h_blocker = compute_hole(blocker_poly)
    blocker_verts = list(blocker_poly.exterior.coords)[:-1]
    blocker_dists = [directional_distance_to_point_from_offset_leg(
                          leg_state, drift_direction, Point(v))
                      for v in blocker_verts]
    blocker_dists = [d for d in blocker_dists if d is not None]
    d_blocker = float(np.mean(blocker_dists)) if blocker_dists else 0.0
    p_nr_blocker = get_not_repaired(repair_data, drift_speed_ms, d_blocker)
    P_blocker = base * rose_prob * h_blocker * p_nr_blocker

    return {
        'name': name,
        'blocker_poly': blocker_poly,
        'shadow_poly': shadow_poly,
        'target_unshadowed': target_unshadowed,
        'target_shadowed': target_shadowed,
        'coverage_area': coverage_area,
        'coverage_hole': coverage_hole,
        'h_target_unshadowed': h_unshadowed,
        'P_target': P_target,
        'h_blocker': h_blocker,
        'd_blocker': d_blocker,
        'p_nr_blocker': p_nr_blocker,
        'P_blocker': P_blocker,
    }

# Base exposure factor (same for every scenario)
hours_present = (leg_len / (ship_speed_kts * 1852)) * ship_freq
base = hours_present * blackout_per_hour

print("EXPOSURE BASE:")
print(f"  hours_present = L / (V * 1852) * f = {leg_len:.1f} / ({ship_speed_kts} * 1852) * {ship_freq}")
print(f"                = {hours_present:.4f} hours/year")
print(f"  base          = hours_present * blackout_per_hour = {hours_present:.4f} * {blackout_per_hour:.4e}")
print(f"                = {base:.6e}")
print()

# Baseline (no blocker)
P_target_baseline = base * rose_prob * target_hole * p_nr_target
print("BASELINE (no blocker):")
print(f"  P(target) = base * r_p * h_target * P_NR(d_target)")
print(f"            = {base:.4e} * {rose_prob} * {target_hole:.4e} * {p_nr_target:.4e}")
print(f"            = {P_target_baseline:.6e}")
print()

scenarios = [
    scenario_stats("A: 0% coverage  (offset blocker)",  blocker_A_poly, shadow_A),
    scenario_stats("B: ~30% coverage (partial shadow)", blocker_B_poly, shadow_B),
    scenario_stats("C: 100% coverage (full shadow)",    blocker_C_poly, shadow_C),
]

print("SHADOW COVERAGE AND EFFECTIVE TARGET HOLE:")
print("-" * 80)
print(f"  {'Scenario':<36} {'cov(area)':>10} {'cov(hole)':>10} {'h_eff':>12} {'P(target)':>12}")
for s in scenarios:
    print(f"  {s['name']:<36} {s['coverage_area']*100:>9.1f}% {s['coverage_hole']*100:>9.1f}% "
          f"{s['h_target_unshadowed']:>12.4e} {s['P_target']:>12.4e}")
print()
print("  Notes:")
print("    cov(area) = geometric area fraction of the target inside the blocker shadow.")
print("    cov(hole) = 1 - h(unshadowed) / h(target)  -- fraction of the probability")
print("                hole removed by the shadow.  This is the meaningful coverage")
print("                figure because the lateral PDF weights different parts of the")
print("                target differently, so area and hole coverage usually differ.")
print()

print("PER-SCENARIO DETAIL:")
print("-" * 80)
for s in scenarios:
    print(f"  {s['name']}")
    print(f"    blocker hole       h_blocker   = {s['h_blocker']:.6e}")
    print(f"    blocker distance   d_blocker   = {s['d_blocker']:.1f} m")
    print(f"    blocker P_NR                    = {s['p_nr_blocker']:.6e}")
    print(f"    blocker contribution P_blocker = {s['P_blocker']:.6e}")
    print(f"    target effective hole h_eff     = {s['h_target_unshadowed']:.6e}")
    print(f"    target coverage (hole fraction) = {s['coverage_hole']*100:.2f}%")
    print(f"    target contribution  P(target) = base * r_p * h_eff * P_NR(d_target)")
    print(f"                                    = {base:.4e} * {rose_prob} * "
          f"{s['h_target_unshadowed']:.4e} * {p_nr_target:.4e}")
    print(f"                                    = {s['P_target']:.6e}")
    print()

# =============================================================================
# 7. FIGURE: 3 scenarios side-by-side
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(21, 9))

def draw_geom(ax_obj, geom, alpha, fc, ec, lw, label=None):
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

for ax, s in zip(axes, scenarios):
    # Corridor for display
    max_dist = d_target
    cfg_disp = DriftConfig(reach_distance_m=1.2 * max_dist, corridor_sigma_multiplier=3.0)
    corridor_display = build_directional_corridor(leg_state, drift_direction, cfg_disp)

    # Corridor minus target (show target as hole in corridor)
    t_shadow_display = build_quad_shadow(target_polygon, extrude_len)
    corridor_with_t_hole = corridor_display.difference(t_shadow_display)
    draw_geom(ax, corridor_with_t_hole, 0.06, 'blue', 'blue', 0.5, 'Drift corridor')

    # Blocker's own shadow (visible as an orange band behind the blocker).
    block_shadow_visible = corridor_display.intersection(s['shadow_poly']).difference(s['blocker_poly'])
    draw_geom(ax, block_shadow_visible, 0.15, 'orange', 'none', 0)
    if not block_shadow_visible.is_empty:
        ax.plot([], [], color='orange', lw=6, alpha=0.3, label='Blocker shadow')

    # Traffic lines
    n_lines = 13
    max_offset = 2.5 * lateral_sigma
    offsets = np.linspace(-max_offset, max_offset, n_lines)
    weights = stats.norm(0, lateral_sigma).pdf(offsets)
    weights = weights / weights.max()
    for off, w in zip(offsets, weights):
        shifted = leg3_line.parallel_offset(off, 'left')
        if shifted.geom_type == 'LineString' and not shifted.is_empty:
            ax.plot(*shifted.xy, color='gray', lw=1, alpha=float(w) * 0.35, zorder=3)
    ax.plot([], [], color='gray', lw=2, alpha=0.35, label='Ship traffic')

    # Leg
    ax.plot(*leg3_line.xy, 'k-', linewidth=2.5, label='Leg 3', zorder=5)

    # Target polygon: draw the unshadowed part in brown, the shadowed part in gray
    draw_geom(ax, s['target_unshadowed'], 0.50, 'brown', 'darkred', 2, 'Target (unshadowed)')
    if not s['target_shadowed'].is_empty:
        draw_geom(ax, s['target_shadowed'], 0.55, 'dimgray', 'black', 1.5, 'Target (shadowed)')

    # Blocker polygon
    bx = [c[0] for c in s['blocker_poly'].exterior.coords]
    by = [c[1] for c in s['blocker_poly'].exterior.coords]
    ax.fill(bx, by, alpha=0.7, fc='darkorange', ec='k', lw=1.5, label='Blocker')

    # Drift arrow
    center = leg3_line.centroid
    dx = 5000 * drift_ux
    dy = 5000 * drift_uy
    ax.annotate('', xy=(center.x + dx, center.y + dy),
                xytext=(center.x, center.y),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(center.x + dx * 0.5 + 200, center.y + dy * 0.5 + 200,
            f'Drift NW\n(315 deg)', color='blue', fontsize=9)

    # Summary box
    txt = (
        f"Shadow coverage (hole) = {s['coverage_hole']*100:.1f}%\n"
        f"h_target (ref)   = {target_hole:.4e}\n"
        f"h_target (eff)   = {s['h_target_unshadowed']:.4e}\n"
        f"d_target         = {d_target:.0f} m\n"
        f"P_NR(d_target)   = {p_nr_target:.4e}\n"
        f"P(target)        = {s['P_target']:.4e}\n"
        f"(baseline P     = {P_target_baseline:.4e})"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', fc='wheat', alpha=0.85),
            family='monospace')

    # Formula box
    formula = (
        r"$h_{\mathrm{eff}} = \mathrm{hole}(\mathrm{target} - "
        r"\mathrm{shadow}(\mathrm{blocker}))$" "\n"
        r"$P_{\mathrm{target}} = \mathrm{base}\cdot r_p\cdot "
        r"h_{\mathrm{eff}}\cdot P_{NR}(d_{\mathrm{target}})$"
    )
    ax.text(0.02, 0.02, formula, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

    ax.set_title(s['name'], fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')

plt.suptitle('Level 3: Shadow Coverage determines the Effective Target Hole',
             fontsize=13, y=0.995)
plt.tight_layout()
fig_path = os.path.join(os.path.dirname(__file__), 'level_3_blocking_polygon.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_path}")
plt.close()

# =============================================================================
# 8. SUMMARY
# =============================================================================

print("SUMMARY")
print("=" * 80)
print(f"  Baseline (no blocker)            : P(target) = {P_target_baseline:.6e}")
for s in scenarios:
    print(f"  {s['name']:<36}: P(target) = {s['P_target']:.6e} "
          f"(coverage {s['coverage_hole']*100:.1f}%)")
print()
print("  Key point: the blocker's effect on the TARGET is determined by the")
print("  fraction of the target's probability hole that falls inside the blocker's")
print("  shadow.  No multiplicative 'remaining' factor is needed -- the blocker")
print("  only removes the rays it physically intercepts.  The blocker may itself")
print("  be a grounding or allision hazard and is accounted for as its own")
print("  obstacle with its own hole and distance.")
