"""
Level 5: Complete Drifting Cascade -- Anchoring + Allision + Grounding
======================================================================

The complete OMRAT drifting calculation with three obstacle types in the
drift path:

    - Anchoring polygon (50 m depth)      -- probabilistic, a_p success
    - Allision structure (wind turbine)    -- geometric blocker (terminates)
    - Grounding polygon (12 m depth)       -- terminates on the rays it
                                              intercepts

Each obstacle contributes to the risk total based on the CORRECT combined
shadow-coverage formulation (no multiplicative ``remaining`` cascade).

Mathematical formulation
------------------------
Let

    h(X)      = analytical probability hole of region X
    shadow(P) = quad-sweep shadow polygon of P along the drift direction
    a_p       = anchoring success probability

Define the blocker shadow (from the structure):

    X_target_reachable = target - shadow(structure)

Define the anchor coverage over the reachable target:

    X_target_anchored  = X_target_reachable intersected with shadow(anchor)

Effective holes:

    h_target_eff  = h(X_target_reachable) - a_p * h(X_target_anchored)
    h_struct_eff  = h(struct) - a_p * h(struct intersected with shadow(anchor))
    h_anchor_eff  = h(anchor)          (anchoring is unconditional -- all ships
                                        that drift through the anchor polygon
                                        are counted)

Probability contributions:

    P(anchor)    = base * r_p * a_p * h_anchor_eff
    P(allision)  = base * r_p * h_struct_eff * P_NR(d_struct)
    P(ground)    = base * r_p * h_target_eff * P_NR(d_target)

The total accident rate is P(allision) + P(ground); P(anchor) is shown
separately since those ships are saved.

Setup
-----
    - Leg 3 (same as L1-L4)
    - Target grounding polygon BD5A4C46 (12 m, 8-vertex shape)
    - Anchoring polygon (50 m) placed upstream, wider than target
    - Allision structure (100 m x 100 m) placed between leg and target
      directly in front of the target on the cross-drift axis
    - anchor_p = 0.70, anchor_d = 7.0, draught = 14.27 m
    - Drift NW (315 deg)

The structure directly blocks part of the target; the anchor reduces the
remaining target rays by the a_p * coverage factor.
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
# 1. GEOMETRY (same leg + target as levels 1-4)
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
struct_height = 20.0

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
print("LEVEL 5: Complete Cascade -- Anchoring + Allision + Grounding")
print("=" * 80)
print()
print("PARAMETERS:")
print(f"  Leg length L            = {leg3_line.length:.1f} m")
print(f"  Ship speed V            = {ship_speed_kts} kts")
print(f"  Frequency f             = {ship_freq} ships/year")
print(f"  Drift speed V_drift     = {drift_speed_kts} kts ({drift_speed_ms:.3f} m/s)")
print(f"  Blackout rate lambda_bo = {blackout_rate}/year ({blackout_per_hour:.4e}/hour)")
print(f"  Rose probability r_p    = {rose_prob:.4f}")
print(f"  Lateral sigma           = {lateral_sigma} m")
print(f"  Ship draught            = {ship_draught} m")
print(f"  anchor_p, anchor_d      = {anchor_p}, {anchor_d}")
print(f"  Drift direction         = {drift_direction} deg compass ({math_deg:.0f} deg math)")
print()

# =============================================================================
# 3. HOLE AND SHADOW HELPERS
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

def directional_distance_mean(poly):
    verts = list(poly.exterior.coords)[:-1]
    ds = [directional_distance_to_point_from_offset_leg(
              leg_state, drift_direction, Point(v))
          for v in verts]
    ds = [d for d in ds if d is not None]
    return float(np.mean(ds)) if ds else 0.0

# =============================================================================
# 4. TARGET STATS
# =============================================================================

target_hole = compute_hole(target_polygon)
d_target = directional_distance_mean(target_polygon)
p_nr_target = get_not_repaired(repair_data, drift_speed_ms, d_target)

hours_present = (leg_len / (ship_speed_kts * 1852)) * ship_freq
base = hours_present * blackout_per_hour
extrude_len = d_target + 5000.0

P_target_baseline = base * rose_prob * target_hole * p_nr_target

print("TARGET POLYGON (no other obstacles):")
print(f"  h_target       = {target_hole:.6e}")
print(f"  d_target       = {d_target:.1f} m")
print(f"  P_NR(d_target) = {p_nr_target:.6e}")
print(f"  base           = L/(V*1852) * f * blackout_per_hour = {base:.6e}")
print(f"  Baseline P(target) = {P_target_baseline:.6e}")
print()

# =============================================================================
# 5. PLACE ANCHOR AND STRUCTURE ON THE DRIFT AXIS
# =============================================================================

target_centroid = np.array([target_polygon.centroid.x, target_polygon.centroid.y])

# Target cross-drift bounds (for sizing anchor to fully cover target)
target_verts = list(target_polygon.exterior.coords)[:-1]
target_cd = [np.dot(np.array(p), cross_drift_vec) for p in target_verts]
target_cd_lo, target_cd_hi = min(target_cd), max(target_cd)
target_cd_width = target_cd_hi - target_cd_lo

# Anchor (50 m depth): 8 km upstream, wider than target to give 100% coverage
anchor_center = target_centroid - 8000.0 * drift_vec_unit
anchor_half_cross = 0.5 * target_cd_width + 300.0
anchor_half_along = 500.0
def _rect(center, half_cross, half_along):
    cx = half_cross * cross_drift_vec
    ay = half_along * drift_vec_unit
    return Polygon([tuple(center - cx - ay),
                    tuple(center + cx - ay),
                    tuple(center + cx + ay),
                    tuple(center - cx + ay),
                    tuple(center - cx - ay)])
anchor_polygon = _rect(anchor_center, anchor_half_cross, anchor_half_along)

# Structure (allision hazard): 4 km upstream of target, placed directly in
# front of the target on the cross-drift axis.  We size it so the shadow
# visibly intercepts a meaningful fraction of the target -- 300 m half-width
# cross-drift x 100 m half-length along-drift (e.g. a large platform or a
# short cluster of turbine foundations).
struct_center = target_centroid - 4000.0 * drift_vec_unit
struct_half_cross = 300.0
struct_half_along = 100.0
struct_polygon = _rect(struct_center, struct_half_cross, struct_half_along)

# =============================================================================
# 6. HOLES AND DISTANCES FOR EACH OBSTACLE
# =============================================================================

h_anchor = compute_hole(anchor_polygon)
h_struct = compute_hole(struct_polygon)
d_anchor = directional_distance_mean(anchor_polygon)
d_struct = directional_distance_mean(struct_polygon)
p_nr_struct = get_not_repaired(repair_data, drift_speed_ms, d_struct)

print("OBSTACLES:")
print("-" * 80)
print(f"  Anchor   (50 m, anchoring) : h = {h_anchor:.6e}, d = {d_anchor:.1f} m")
print(f"  Struct   (allision)         : h = {h_struct:.6e}, d = {d_struct:.1f} m")
print(f"  Target   (12 m, grounding)  : h = {target_hole:.6e}, d = {d_target:.1f} m")
print()

# =============================================================================
# 7. SHADOW POLYGONS AND COMBINED COVERAGE
# =============================================================================

shadow_struct = build_quad_shadow(struct_polygon, extrude_len)
shadow_anchor = build_quad_shadow(anchor_polygon, extrude_len)

# Target reachable after structure (drop structure-shadowed part entirely)
target_reachable = target_polygon.difference(shadow_struct)
h_target_reach = compute_hole(target_reachable)

# Of the reachable target, how much falls inside the anchor shadow?
target_reach_in_anchor = target_reachable.intersection(shadow_anchor)
h_target_reach_in_anchor = compute_hole(target_reach_in_anchor)

# Effective holes
h_target_eff = h_target_reach - anchor_p * h_target_reach_in_anchor

# Struct effective hole: reduce by a_p * fraction of struct rays passing
# through the anchor shadow
struct_in_anchor = struct_polygon.intersection(shadow_anchor)
h_struct_in_anchor = compute_hole(struct_in_anchor)
h_struct_eff = h_struct - anchor_p * h_struct_in_anchor

print("SHADOW COVERAGES:")
print("-" * 80)
cov_struct_on_target = (target_hole - h_target_reach) / target_hole if target_hole > 0 else 0.0
cov_anchor_on_reach = (h_target_reach_in_anchor / h_target_reach) if h_target_reach > 0 else 0.0
cov_anchor_on_struct = (h_struct_in_anchor / h_struct) if h_struct > 0 else 0.0
print(f"  Structure shadow covers {cov_struct_on_target*100:.1f}% of target hole (removed entirely)")
print(f"  Anchor shadow covers {cov_anchor_on_reach*100:.1f}% of REACHABLE target hole "
      f"(reduced by factor a_p = {anchor_p})")
print(f"  Anchor shadow covers {cov_anchor_on_struct*100:.1f}% of structure hole "
      f"(reduced by factor a_p)")
print()

# =============================================================================
# 8. PROBABILITIES
# =============================================================================

P_anchor = base * rose_prob * anchor_p * h_anchor
P_struct = base * rose_prob * h_struct_eff * p_nr_struct
P_ground = base * rose_prob * h_target_eff * p_nr_target

print("PROBABILITY CALCULATIONS:")
print("-" * 80)
print(f"  P(anchor)  = base * r_p * a_p * h_anchor")
print(f"             = {base:.4e} * {rose_prob} * {anchor_p} * {h_anchor:.4e}")
print(f"             = {P_anchor:.6e}")
print()
print(f"  h_struct_eff = h_struct - a_p * h(struct ^ shadow_anchor)")
print(f"              = {h_struct:.4e} - {anchor_p} * {h_struct_in_anchor:.4e}")
print(f"              = {h_struct_eff:.6e}")
print(f"  P(allision) = base * r_p * h_struct_eff * P_NR(d_struct)")
print(f"              = {base:.4e} * {rose_prob} * {h_struct_eff:.4e} * {p_nr_struct:.4e}")
print(f"              = {P_struct:.6e}")
print()
print(f"  h_target_eff = h(target - shadow_struct) - a_p * h((target - shadow_struct) ^ shadow_anchor)")
print(f"               = {h_target_reach:.4e} - {anchor_p} * {h_target_reach_in_anchor:.4e}")
print(f"               = {h_target_eff:.6e}")
print(f"  P(ground)   = base * r_p * h_target_eff * P_NR(d_target)")
print(f"              = {base:.4e} * {rose_prob} * {h_target_eff:.4e} * {p_nr_target:.4e}")
print(f"              = {P_ground:.6e}")
print()

total_accident = P_struct + P_ground
print(f"  Accident rate = P(allision) + P(ground) = {total_accident:.6e}/year")
print(f"  P(anchor)                               = {P_anchor:.6e}/year")
print()

# =============================================================================
# 9. COMPARISON SCENARIOS
# =============================================================================

def run(include_anchor, include_struct):
    """Compute P(anchor), P(allision), P(ground) for a given set of obstacles."""
    sh_s = shadow_struct if include_struct else Polygon()
    sh_a = shadow_anchor if include_anchor else Polygon()
    a_eff = anchor_p if include_anchor else 0.0

    tr = target_polygon.difference(sh_s)
    h_tr = compute_hole(tr)
    h_tr_anch = compute_hole(tr.intersection(sh_a))
    h_tgt_eff = h_tr - a_eff * h_tr_anch
    P_tgt = base * rose_prob * h_tgt_eff * p_nr_target

    if include_struct:
        h_struct_in_a = compute_hole(struct_polygon.intersection(sh_a))
        h_str_eff = h_struct - a_eff * h_struct_in_a
        P_str = base * rose_prob * h_str_eff * p_nr_struct
    else:
        P_str = 0.0

    P_anc = base * rose_prob * a_eff * h_anchor if include_anchor else 0.0
    return P_anc, P_str, P_tgt

print("COMPARISON SCENARIOS:")
print("-" * 80)
print(f"  {'scenario':<32} {'P(anchor)':>12} {'P(allision)':>12} {'P(ground)':>12} {'Accident':>12}")
for name, a, s in [
    ("target only",              False, False),
    ("target + anchor",          True,  False),
    ("target + structure",       False, True),
    ("target + anchor + struct", True,  True),
]:
    pa, ps, pg = run(a, s)
    print(f"  {name:<32} {pa:>12.4e} {ps:>12.4e} {pg:>12.4e} {ps+pg:>12.4e}")
print()

# =============================================================================
# 10. FIGURE
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(14, 11))

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

cfg_disp = DriftConfig(reach_distance_m=1.2 * d_target, corridor_sigma_multiplier=3.0)
corridor_display = build_directional_corridor(leg_state, drift_direction, cfg_disp)

# Corridor minus target (show target polygon as a hole) and minus struct
corridor_with_holes = corridor_display.difference(
    _unary_union([build_quad_shadow(target_polygon, extrude_len), shadow_struct]))
draw_geom(ax, corridor_with_holes, 0.06, 'blue', 'blue', 0.5, 'Drift corridor')

# Structure shadow (visible band)
struct_shadow_visible = corridor_display.intersection(shadow_struct).difference(struct_polygon)
draw_geom(ax, struct_shadow_visible, 0.15, 'red', 'none', 0)
if not struct_shadow_visible.is_empty:
    ax.plot([], [], color='red', lw=6, alpha=0.3, label='Structure shadow')

# Anchor shadow (visible band)
anchor_shadow_visible = corridor_display.intersection(shadow_anchor).difference(anchor_polygon)
draw_geom(ax, anchor_shadow_visible, 0.15, 'cyan', 'none', 0)
if not anchor_shadow_visible.is_empty:
    ax.plot([], [], color='cyan', lw=6, alpha=0.3, label='Anchor shadow')

# Traffic
n_lines = 15
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

# Anchor polygon
anchor_in_corr = corridor_display.intersection(anchor_polygon)
if not anchor_in_corr.is_empty:
    if anchor_in_corr.geom_type == 'Polygon':
        ax.fill(*anchor_in_corr.exterior.xy, alpha=0.45, fc='cyan', ec='teal',
                lw=2, hatch='///', label=f'Anchor (50 m, a_p={anchor_p})')
    else:
        first = True
        for g in anchor_in_corr.geoms:
            ax.fill(*g.exterior.xy, alpha=0.45, fc='cyan', ec='teal', lw=2, hatch='///',
                    label=(f'Anchor (50 m, a_p={anchor_p})' if first else None))
            first = False

# Structure
sx = [c[0] for c in struct_polygon.exterior.coords]
sy = [c[1] for c in struct_polygon.exterior.coords]
ax.fill(sx, sy, alpha=0.85, fc='red', ec='darkred', lw=2, label='Structure (allision)', zorder=7)

# Target: split unshadowed / struct-shadowed / anchor-shadowed-but-reachable
target_reach_not_anchor = target_reachable.difference(shadow_anchor)
target_reach_in_anchor_geom = target_reachable.intersection(shadow_anchor)
target_struct_shadowed = target_polygon.intersection(shadow_struct)
draw_geom(ax, target_reach_not_anchor, 0.55, 'brown', 'darkred', 2,
          'Target: fully reachable')
draw_geom(ax, target_reach_in_anchor_geom, 0.55, 'mediumorchid', 'purple', 2,
          'Target: anchor-reduced')
draw_geom(ax, target_struct_shadowed, 0.55, 'dimgray', 'black', 2,
          'Target: struct-shadowed (0)')

# Drift arrow
center = leg3_line.centroid
dx = 5000 * drift_ux
dy = 5000 * drift_uy
ax.annotate('', xy=(center.x + dx, center.y + dy),
            xytext=(center.x, center.y),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
ax.text(center.x + dx * 0.5 + 200, center.y + dy * 0.5 + 200,
        f'Drift NW\n(315 deg)', color='blue', fontsize=9)

# Text box with the full breakdown
cascade_text = (
    "CASCADE (shadow-coverage model):\n"
    f"  Anchor (50 m, ~{d_anchor:.0f} m):\n"
    f"    h_anchor = {h_anchor:.4e}\n"
    f"    P(anchor) = base*r_p*a_p*h_anchor\n"
    f"             = {P_anchor:.4e}\n"
    f"  Structure (~{d_struct:.0f} m):\n"
    f"    h_struct = {h_struct:.4e}\n"
    f"    anchor cov(struct) = {cov_anchor_on_struct*100:.1f}%\n"
    f"    h_struct_eff = {h_struct_eff:.4e}\n"
    f"    P(allision) = {P_struct:.4e}\n"
    f"  Target (~{d_target:.0f} m):\n"
    f"    struct cov(target) = {cov_struct_on_target*100:.1f}%\n"
    f"    h(reach) = {h_target_reach:.4e}\n"
    f"    anchor cov(reach)  = {cov_anchor_on_reach*100:.1f}%\n"
    f"    h_target_eff = {h_target_eff:.4e}\n"
    f"    P(ground) = {P_ground:.4e}\n"
    f"---\n"
    f"  Accident rate = {total_accident:.4e}/yr"
)
ax.text(0.02, 0.98, cascade_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

formula = (
    r"$h_{\mathrm{tgt,eff}} = h(\mathrm{tgt} - S_s) - a_p\, h((\mathrm{tgt} - S_s)\cap S_a)$"
    "\n"
    r"$h_{\mathrm{str,eff}} = h(\mathrm{str}) - a_p\, h(\mathrm{str}\cap S_a)$"
    "\n"
    r"$P_i = \mathrm{base}\cdot r_p\cdot h_{i,\mathrm{eff}}\cdot P_{NR}(d_i)$, "
    r"$P_{\mathrm{anchor}} = \mathrm{base}\cdot r_p\cdot a_p\cdot h_{\mathrm{anchor}}$"
)
ax.text(0.02, 0.02, formula, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', fc='wheat', alpha=0.9))

ax.set_title('Level 5: Complete Drifting Cascade\n'
             'Anchor (probabilistic) + Structure (blocker) + Grounding',
             fontsize=13)
ax.legend(loc='lower right', fontsize=8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(__file__), 'level_5_complete_cascade.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_path}")
plt.close()

# =============================================================================
# 11. SUMMARY
# =============================================================================

print("SUMMARY")
print("=" * 80)
print(f"  Baseline P(target) (target only)         = {P_target_baseline:.6e}")
print(f"  P(anchor)                                = {P_anchor:.6e}")
print(f"  P(allision) (struct, a_p-reduced)        = {P_struct:.6e}")
print(f"  P(grounding) (shadow-and-anchor-reduced) = {P_ground:.6e}")
print(f"  Accident rate (allision + grounding)     = {total_accident:.6e}/year")
print()
print("  The structure's shadow REMOVES target rays it physically intercepts.")
print("  The anchor's shadow REDUCES (by factor a_p) the target rays that also")
print("  pass through the anchor zone.  The structure itself is reduced by the")
print("  anchor for struct-rays that pass through the anchor zone first.")
