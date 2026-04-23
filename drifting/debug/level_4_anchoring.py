"""
Level 4: Drifting Grounding with Anchoring
==========================================

Building on Level 3, this example adds an anchoring zone.  An anchoring
polygon is an area where ships can attempt to drop anchor if the water
depth is shallower than ``anchor_d * draught``.

Unlike a blocking polygon (Level 3) which purely geometrically prevents
ships from reaching the target, anchoring is a PROBABILISTIC event: a
ship that drifts through an anchoring zone anchors successfully with
probability ``a_p`` (typical 0.7) and continues drifting with probability
``(1 - a_p)``.

The correct formulation uses the SAME geometric shadow-coverage idea as
Level 3, combined with the probabilistic effect of anchoring:

    shadow_a          = quad-sweep shadow of anchor polygon along drift
    target_in_shadow  = target polygon INSIDE the anchor shadow
    h_a_on_target     = hole(target intersected with shadow_a)

    effective target hole
        h_target_eff = h_target - a_p * h_a_on_target

    grounding probability
        P(target) = base * r_p * h_target_eff * P_NR(d_target)

    anchoring probability (ships saved on the anchoring polygon itself)
        P(anchor) = base * r_p * a_p * h_anchor

This is consistent with Level 3 in the limit a_p = 1 (every anchoring
attempt succeeds, so the anchor polygon behaves like a pure blocker).
There is NO multiplicative ``remaining`` factor.

Setup
-----
    - Leg 3 (same as L1-L3)
    - Target grounding polygon BD5A4C46 (12 m, 8-vertex shape)
    - Anchor polygon: 50 m depth, sized so that 100% of the target rays
      pass through it (so the anchoring effect is clearly visible)
    - anchor_p = 0.70, anchor_d = 7.0, draught = 14.27 m
        => anchoring threshold = 7 * 14.27 = 99.9 m
        => 50 m < 99.9 m -> anchoring applies
        => 12 m < 14.27 m -> target grounds
    - Drift NW (315 deg)

Three scenarios are shown so the anchor shadow coverage of the target is
easy to read off:

    A) No anchor polygon              -> baseline grounding
    B) Anchor shadowing 30% of target
    C) Anchor shadowing 100% of target

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
    directional_distance_to_point_from_offset_leg,
)
from compute.basic_equations import get_not_repaired
from geometries.analytical_probability import (
    compute_probability_analytical,
    _extract_polygon_rings,
)


# =============================================================================
# 1. GEOMETRY
# =============================================================================

proj_wgs84 = pyproj.CRS("EPSG:4326")
proj_utm = pyproj.CRS("EPSG:32633")
transformer = pyproj.Transformer.from_crs(proj_wgs84, proj_utm, always_xy=True)

def to_utm(lon, lat):
    return transformer.transform(lon, lat)

leg3_start = to_utm(14.24187, 55.16728)
leg3_end = to_utm(14.59271, 55.39937)
leg3_line = LineString([leg3_start, leg3_end])

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
drift_speed_ms = drift_speed_kts * 1852 / 3600
blackout_rate = 1.0
blackout_per_hour = blackout_rate / (365.25 * 24)
rose_prob = 1.0 / 8.0
lateral_sigma = 500.0
reach_distance = 50000.0
drift_direction = 315

anchor_p = 0.70
anchor_d = 7.0
anchor_depth = 50.0
polygon_depth = 12.0

repair_data = {'use_lognormal': 1, 'std': 1.0, 'loc': 0.0, 'scale': 1.0}

cfg = DriftConfig(reach_distance_m=reach_distance, corridor_sigma_multiplier=3.0)
leg_state = LegState(leg_id="LEG_3", line=leg3_line,
                      mean_offset_m=0.0, lateral_sigma_m=lateral_sigma)

math_deg = compass_to_math_deg(drift_direction)
angle_rad = np.radians(math_deg)
drift_vec_unit = np.array([np.cos(angle_rad), np.sin(angle_rad)])
cross_drift_vec = np.array([-drift_vec_unit[1], drift_vec_unit[0]])
drift_ux, drift_uy = drift_vec_unit[0], drift_vec_unit[1]

anchoring_threshold = anchor_d * ship_draught

print("=" * 80)
print("LEVEL 4: Drifting Grounding with Anchoring (shadow-coverage model)")
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
print(f"  Ship draught            = {ship_draught} m")
print(f"  Target depth            = {polygon_depth} m  (< draught -> grounds)")
print(f"  Anchor depth            = {anchor_depth} m  (< {anchoring_threshold:.1f} m -> anchors)")
print(f"  anchor_p                = {anchor_p}")
print(f"  anchor_d                = {anchor_d}")
print(f"  Drift direction         = {drift_direction} deg compass")
print()

# =============================================================================
# 3. ANALYTICAL HOLE HELPER + QUAD-SWEEP SHADOW HELPER
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
# 4. TARGET STATS
# =============================================================================

target_hole = compute_hole(target_polygon)
target_vertices = list(target_polygon.exterior.coords)[:-1]
target_dists = [directional_distance_to_point_from_offset_leg(
                    leg_state, drift_direction, Point(v))
                for v in target_vertices]
target_dists = [d for d in target_dists if d is not None]
d_target = float(np.mean(target_dists))
p_nr_target = get_not_repaired(repair_data, drift_speed_ms, d_target)

hours_present = (leg_len / (ship_speed_kts * 1852)) * ship_freq
base = hours_present * blackout_per_hour

P_target_baseline = base * rose_prob * target_hole * p_nr_target

print("TARGET POLYGON (no anchor):")
print(f"  h_target = {target_hole:.6e}")
print(f"  d_target = {d_target:.1f} m")
print(f"  P_NR(d_target) = {p_nr_target:.6e}")
print(f"  base = L / (V*1852) * f * blackout_per_hour = {base:.6e}")
print(f"  Baseline P(target) = base * r_p * h_target * P_NR = {P_target_baseline:.6e}")
print()

# =============================================================================
# 5. BUILD ANCHOR POLYGONS FOR 3 COVERAGE SCENARIOS
# =============================================================================

def cross_drift_interval(pts):
    projs = [np.dot(np.array(pt), cross_drift_vec) for pt in pts]
    return min(projs), max(projs)

target_cd_lo, target_cd_hi = cross_drift_interval(target_vertices)
target_cd_center = 0.5 * (target_cd_lo + target_cd_hi)
target_cd_width = target_cd_hi - target_cd_lo
target_centroid = np.array([target_polygon.centroid.x, target_polygon.centroid.y])

# Anchor placed 6 km upstream of target
anchor_center_along = target_centroid - 6000.0 * drift_vec_unit
anchor_half_along = 500.0
extrude_len = d_target + 5000.0

def make_anchor(center_cross_offset_m, half_cross_m):
    center = anchor_center_along + center_cross_offset_m * cross_drift_vec
    cx = half_cross_m * cross_drift_vec
    ay = anchor_half_along * drift_vec_unit
    coords = [
        tuple(center - cx - ay),
        tuple(center + cx - ay),
        tuple(center + cx + ay),
        tuple(center - cx + ay),
        tuple(center - cx - ay),
    ]
    return coords, Polygon(coords)

# Scenario A: no anchor at all (None)
# Scenario B: ~30% coverage -- binary search on half_cross
def coverage_for_half(half_m):
    offset = (target_cd_hi - half_m) - target_cd_center
    _, poly = make_anchor(offset, half_m)
    sh = build_quad_shadow(poly, extrude_len)
    unshadow = target_polygon.difference(sh)
    h = compute_hole(unshadow)
    return 1.0 - h / target_hole if target_hole > 0 else 0.0

lo, hi = 50.0, target_cd_width
best = 0.5 * target_cd_width
for _ in range(30):
    mid = 0.5 * (lo + hi)
    c = coverage_for_half(mid)
    if c < 0.30:
        lo = mid
    else:
        hi = mid
    best = mid
anchor_B_half_cross = best
anchor_B_offset = (target_cd_hi - anchor_B_half_cross) - target_cd_center
_, anchor_B_poly = make_anchor(anchor_B_offset, anchor_B_half_cross)

# Scenario C: 100% coverage -- wider than target
anchor_C_half_cross = 0.5 * target_cd_width + 200.0
_, anchor_C_poly = make_anchor(0.0, anchor_C_half_cross)

# =============================================================================
# 6. COMPUTE SCENARIO RESULTS
# =============================================================================

def scenario_stats(name, anchor_poly):
    if anchor_poly is None:
        return {
            'name': name,
            'anchor_poly': None,
            'shadow': None,
            'target_unshadow': target_polygon,
            'target_in_shadow': Polygon(),
            'h_anchor': 0.0,
            'h_a_on_target': 0.0,
            'coverage_hole': 0.0,
            'h_target_eff': target_hole,
            'P_anchor': 0.0,
            'P_target': P_target_baseline,
            'd_anchor': 0.0,
        }
    shadow = build_quad_shadow(anchor_poly, extrude_len)
    target_unshadow = target_polygon.difference(shadow)
    target_in_shadow = target_polygon.intersection(shadow)
    h_unshadow = compute_hole(target_unshadow)
    h_a_on_target = target_hole - h_unshadow   # derive from holes, not re-integrate
    coverage_hole = (h_a_on_target / target_hole) if target_hole > 0 else 0.0
    h_target_eff = target_hole - anchor_p * h_a_on_target
    # P(target) = base * r_p * h_target_eff * P_NR(d_target)
    P_target = base * rose_prob * h_target_eff * p_nr_target

    # anchor's own hole + mean distance for P(anchor)
    h_anchor = compute_hole(anchor_poly)
    anchor_verts = list(anchor_poly.exterior.coords)[:-1]
    anchor_dists = [directional_distance_to_point_from_offset_leg(
                        leg_state, drift_direction, Point(v))
                    for v in anchor_verts]
    anchor_dists = [d for d in anchor_dists if d is not None]
    d_anchor = float(np.mean(anchor_dists)) if anchor_dists else 0.0
    # P(anchor) = base * r_p * a_p * h_anchor
    P_anchor = base * rose_prob * anchor_p * h_anchor
    return {
        'name': name,
        'anchor_poly': anchor_poly,
        'shadow': shadow,
        'target_unshadow': target_unshadow,
        'target_in_shadow': target_in_shadow,
        'h_anchor': h_anchor,
        'h_a_on_target': h_a_on_target,
        'coverage_hole': coverage_hole,
        'h_target_eff': h_target_eff,
        'P_anchor': P_anchor,
        'P_target': P_target,
        'd_anchor': d_anchor,
    }

scenarios = [
    scenario_stats("A: no anchor            ", None),
    scenario_stats("B: anchor shadow ~30%   ", anchor_B_poly),
    scenario_stats("C: anchor shadow 100%   ", anchor_C_poly),
]

print("RESULTS:")
print("-" * 80)
print(f"  {'Scenario':<30} {'cov(hole)':>10} {'h_eff':>12} "
      f"{'h_anchor':>12} {'P(anchor)':>12} {'P(target)':>12}")
for s in scenarios:
    print(f"  {s['name']:<30} {s['coverage_hole']*100:>9.1f}% "
          f"{s['h_target_eff']:>12.4e} {s['h_anchor']:>12.4e} "
          f"{s['P_anchor']:>12.4e} {s['P_target']:>12.4e}")
print()

print("FORMULAS:")
print("-" * 80)
print("  h_target_eff = h(target) - a_p * h(target intersected with anchor_shadow)")
print("  P(target)    = base * r_p * h_target_eff * P_NR(d_target)")
print("  P(anchor)    = base * r_p * a_p * h(anchor)")
print()

print("PER-SCENARIO DETAIL:")
print("-" * 80)
for s in scenarios:
    print(f"  {s['name']}")
    print(f"    h_anchor                     = {s['h_anchor']:.6e}")
    if s['anchor_poly'] is not None:
        print(f"    h(target & anchor_shadow)    = {s['h_a_on_target']:.6e}")
        print(f"    coverage (hole fraction)     = {s['coverage_hole']*100:.2f}%")
        print(f"    h_target_eff = h_target - a_p * h(target & shadow)")
        print(f"                 = {target_hole:.4e} - {anchor_p} * {s['h_a_on_target']:.4e}")
        print(f"                 = {s['h_target_eff']:.6e}")
    else:
        print(f"    (no anchor; h_target_eff = h_target = {target_hole:.4e})")
    print(f"    P(anchor)  = base * r_p * a_p * h_anchor")
    print(f"               = {base:.4e} * {rose_prob} * {anchor_p} * {s['h_anchor']:.4e}")
    print(f"               = {s['P_anchor']:.6e}")
    print(f"    P(target)  = base * r_p * h_target_eff * P_NR(d_target)")
    print(f"               = {base:.4e} * {rose_prob} * {s['h_target_eff']:.4e} * {p_nr_target:.4e}")
    print(f"               = {s['P_target']:.6e}")
    print()

# =============================================================================
# 7. FIGURE
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(21, 9))

def draw_geom(ax_obj, geom, alpha, fc, ec, lw, label=None):
    if geom is None or geom.is_empty:
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
    cfg_disp = DriftConfig(reach_distance_m=1.2 * d_target, corridor_sigma_multiplier=3.0)
    corridor_display = build_directional_corridor(leg_state, drift_direction, cfg_disp)

    t_shadow_display = build_quad_shadow(target_polygon, extrude_len)
    corridor_with_t_hole = corridor_display.difference(t_shadow_display)
    draw_geom(ax, corridor_with_t_hole, 0.06, 'blue', 'blue', 0.5, 'Drift corridor')

    # Anchor shadow (visible band behind anchor polygon)
    if s['shadow'] is not None:
        anchor_shadow_visible = (corridor_display.intersection(s['shadow'])
                                 .difference(s['anchor_poly']))
        draw_geom(ax, anchor_shadow_visible, 0.15, 'cyan', 'none', 0)
        if not anchor_shadow_visible.is_empty:
            ax.plot([], [], color='cyan', lw=6, alpha=0.3, label='Anchor shadow (a_p)')

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

    # Target: unshadowed (full colour) vs shadowed (tinted)
    draw_geom(ax, s['target_unshadow'], 0.50, 'brown', 'darkred', 2, 'Target (unshadowed)')
    if s['target_in_shadow'] is not None and not s['target_in_shadow'].is_empty:
        draw_geom(ax, s['target_in_shadow'], 0.55, 'plum', 'purple', 2,
                  'Target (in anchor shadow)')

    # Anchor polygon
    if s['anchor_poly'] is not None:
        ax_in_corridor = corridor_display.intersection(s['anchor_poly'])
        if not ax_in_corridor.is_empty:
            if ax_in_corridor.geom_type == 'Polygon':
                ax.fill(*ax_in_corridor.exterior.xy, alpha=0.50, fc='cyan', ec='teal',
                        lw=2, hatch='///', label=f'Anchor (50 m, a_p={anchor_p})')
            else:
                first = True
                for g in ax_in_corridor.geoms:
                    ax.fill(*g.exterior.xy, alpha=0.50, fc='cyan', ec='teal', lw=2, hatch='///',
                            label=(f'Anchor (50 m, a_p={anchor_p})' if first else None))
                    first = False

    # Drift arrow
    center = leg3_line.centroid
    dx = 5000 * drift_ux
    dy = 5000 * drift_uy
    ax.annotate('', xy=(center.x + dx, center.y + dy),
                xytext=(center.x, center.y),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(center.x + dx * 0.5 + 200, center.y + dy * 0.5 + 200,
            f'Drift NW\n(315 deg)', color='blue', fontsize=9)

    txt = (
        f"coverage(hole) = {s['coverage_hole']*100:.1f}%\n"
        f"h_anchor       = {s['h_anchor']:.4e}\n"
        f"h_target       = {target_hole:.4e}\n"
        f"h_target_eff   = {s['h_target_eff']:.4e}\n"
        f"P(anchor)      = {s['P_anchor']:.4e}\n"
        f"P(target)      = {s['P_target']:.4e}\n"
        f"(baseline P    = {P_target_baseline:.4e})"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', fc='wheat', alpha=0.85),
            family='monospace')

    formula = (
        r"$h_{\mathrm{eff}} = h_{\mathrm{target}} - a_p\cdot "
        r"h(\mathrm{target}\cap \mathrm{shadow}_a)$" "\n"
        r"$P(\mathrm{target}) = \mathrm{base}\cdot r_p\cdot "
        r"h_{\mathrm{eff}}\cdot P_{NR}(d_{\mathrm{target}})$" "\n"
        r"$P(\mathrm{anchor}) = \mathrm{base}\cdot r_p\cdot "
        r"a_p\cdot h_{\mathrm{anchor}}$"
    )
    ax.text(0.02, 0.02, formula, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

    ax.set_title(s['name'], fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')

plt.suptitle('Level 4: Anchoring reduces target hole by a_p * shadow coverage',
             fontsize=13, y=0.995)
plt.tight_layout()
fig_path = os.path.join(os.path.dirname(__file__), 'level_4_anchoring.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_path}")
plt.close()

# =============================================================================
# 8. SUMMARY
# =============================================================================

print("SUMMARY")
print("=" * 80)
print(f"  Baseline P(target)                    = {P_target_baseline:.6e}")
for s in scenarios[1:]:
    red = 100.0 * (1.0 - s['P_target'] / P_target_baseline) if P_target_baseline > 0 else 0.0
    print(f"  {s['name']} P(target)     = {s['P_target']:.6e} "
          f"(-{red:.1f}% from baseline, P(anchor)={s['P_anchor']:.4e})")
print()
print("  Key point: a_p applies only to the fraction of the target's probability")
print("  hole that falls inside the anchor's shadow.  Anchoring outside that")
print("  shadow has no effect on the target.  In the limit a_p -> 1 the anchor")
print("  behaves like a pure blocker (Level 3 full-coverage case).")
