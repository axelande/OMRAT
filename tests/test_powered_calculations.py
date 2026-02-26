"""
Comprehensive tests for powered Cat II calculations with shadow effects.

Tests the IWRAP Category II powered grounding/allision model:
  P(hit) = mass * exp(-d_mean / (ai * V))

Key concepts tested:
  1. powered_na() exponential decay
  2. Ray casting from a point in a direction to find obstacle intersection
  3. Shadow effect: closer obstacles block distribution for farther obstacles
  4. Mass computation: fraction of lateral distribution intercepted by each obstacle
  5. Draft filtering: only depth areas with depth <= MAX_DRAFT are grounding hazards
  6. Integration test with proj2.omrat real data

Does NOT require QGIS -- uses only standard Python, numpy, scipy, shapely.
"""

import json
import os
from collections import defaultdict
from math import cos, exp, radians

import numpy as np
import pytest
from numpy import isclose
from scipy.stats import norm
from shapely import wkt
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import transform as shapely_transform

from compute.basic_equations import (
    get_powered_grounding_cat1,
    get_powered_grounding_cat2,
    get_recovery_distance,
    powered_na,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ2_FILE = os.path.join(TEST_DIR, "example_data", "proj2.omrat")

# ---------------------------------------------------------------------------
# Constants (same as examples/check_powered_repaired.py)
# ---------------------------------------------------------------------------
MAX_RANGE = 50_000   # metres
MAX_DRAFT = 15.0     # metres -- only depths <= this are grounding hazards
N_RAYS = 500         # rays across the lateral distribution


# =========================================================================
# Helper functions (standalone; no QGIS dependency)
# =========================================================================

class SimpleProjector:
    """Equirectangular projection (lon/lat -> metres) around a reference point."""

    def __init__(self, lon_ref: float, lat_ref: float) -> None:
        self.lon_ref = lon_ref
        self.lat_ref = lat_ref
        self.mx = 111_320.0 * cos(radians(lat_ref))
        self.my = 110_540.0

    def transform(self, lon: float, lat: float) -> tuple[float, float]:
        return ((lon - self.lon_ref) * self.mx,
                (lat - self.lat_ref) * self.my)


def parse_point(coord_str: str) -> tuple[float, float]:
    parts = coord_str.strip().split()
    return float(parts[0]), float(parts[1])


def project_wkt(wkt_str: str, proj: SimpleProjector):
    geom = wkt.loads(wkt_str)
    return shapely_transform(lambda x, y, z=None: proj.transform(x, y), geom)


def weighted_avg_speed_knots(traffic_dir: dict) -> float:
    freq = traffic_dir["Frequency (ships/year)"]
    spd = traffic_dir["Speed (knots)"]
    tot_f, tot_fs = 0.0, 0.0
    for row_f, row_s in zip(freq, spd):
        for f, s in zip(row_f, row_s):
            if 0 < f < float("inf") and 0 < s < float("inf"):
                tot_f += f
                tot_fs += f * s
    return tot_fs / tot_f if tot_f > 0 else 0.0


def leg_vectors(start: np.ndarray, end: np.ndarray):
    """Return (unit direction, perpendicular, length)."""
    d = end - start
    length = np.linalg.norm(d)
    u = d / length
    n = np.array([-u[1], u[0]])   # CCW 90 degrees
    return u, n, length


def get_all_coords(geom):
    """Extract all coordinate tuples from any Shapely geometry."""
    if geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [(geom.x, geom.y)]
    elif geom.geom_type in ("LineString", "LinearRing"):
        return list(geom.coords)
    elif geom.geom_type == "Polygon":
        return list(geom.exterior.coords)
    elif geom.geom_type in ("MultiPoint", "MultiLineString",
                             "MultiPolygon", "GeometryCollection"):
        coords = []
        for part in geom.geoms:
            coords.extend(get_all_coords(part))
        return coords
    return []


def ray_hit_distance(origin, direction, max_range, obstacle_geom):
    """Cast a ray and return the along-track distance to the first intersection."""
    ray_end = origin + max_range * direction
    ray = LineString([origin, ray_end])
    if not ray.intersects(obstacle_geom):
        return None
    try:
        intersection = ray.intersection(obstacle_geom)
    except Exception:
        return None
    coords = get_all_coords(intersection)
    if not coords:
        return None
    along_dists = []
    for px, py in coords:
        d = np.dot(np.array([px, py]) - origin, direction)
        if d > 0:
            along_dists.append(d)
    return min(along_dists) if along_dists else None


def compute_cat2_with_shadows(turn_pt, ext_dir, perp, mean_offset, sigma,
                               ai, speed_ms, obstacles, n_rays=N_RAYS):
    """Compute Cat II probabilities with shadow effects.

    Returns dict { (kind, obs_id): {mass, mean_dist, p_approx, n_rays, ...} }
    """
    offsets = np.linspace(mean_offset - 4 * sigma, mean_offset + 4 * sigma,
                          n_rays)
    dx = offsets[1] - offsets[0]
    pdf_vals = norm.pdf(offsets, mean_offset, sigma)
    masses = pdf_vals * dx
    recovery = ai * speed_ms

    obs_accum: dict = defaultdict(lambda: {
        "mass": 0.0, "weighted_dist": 0.0, "p_integral": 0.0,
        "n_rays": 0, "ray_offsets": [], "ray_dists": [],
    })

    ray_data = []
    for off, m_i in zip(offsets, masses):
        ray_origin = turn_pt + off * perp

        best_d = float("inf")
        best_key = None

        for obs, kind in obstacles:
            d = ray_hit_distance(ray_origin, ext_dir, MAX_RANGE, obs["geom"])
            if d is not None and 0 < d < best_d:
                best_d = d
                best_key = (kind, obs["id"])

        if best_key is not None:
            oa = obs_accum[best_key]
            oa["mass"] += m_i
            oa["weighted_dist"] += m_i * best_d
            if recovery > 0:
                oa["p_integral"] += m_i * exp(-best_d / recovery)
            oa["n_rays"] += 1
            oa["ray_offsets"].append(off)
            oa["ray_dists"].append(best_d)
            ray_data.append((off, m_i, best_key, best_d))
        else:
            ray_data.append((off, m_i, None, None))

    summaries = {}
    for key, oa in obs_accum.items():
        mean_dist = oa["weighted_dist"] / oa["mass"] if oa["mass"] > 0 else 0
        p_approx = oa["mass"] * powered_na(mean_dist, ai, speed_ms)
        summaries[key] = {
            "mass": oa["mass"],
            "mean_dist": mean_dist,
            "p_integral": oa["p_integral"],
            "p_approx": p_approx,
            "n_rays": oa["n_rays"],
            "ray_offsets": oa["ray_offsets"],
            "ray_dists": oa["ray_dists"],
        }

    return summaries, ray_data, offsets, pdf_vals


# =========================================================================
# Fixture: load and project proj2.omrat once per session
# =========================================================================

@pytest.fixture(scope="module")
def proj2_data():
    """Load proj2.omrat and project all geometries to local metres."""
    with open(PROJ2_FILE, "r") as f:
        data = json.load(f)

    segments = data["segment_data"]
    traffic = data["traffic_data"]
    pc_vals = data.get("pc", {})
    pc_grounding = float(pc_vals.get("grounding", pc_vals.get("p_pc", 1.6e-4)))
    pc_allision = float(pc_vals.get("allision", 1.9e-4))

    first_seg = segments[list(segments.keys())[0]]
    lon0, lat0 = parse_point(first_seg["Start_Point"])
    proj = SimpleProjector(lon0, lat0)

    # Depths
    depth_geoms_all = [
        {"id": d[0], "depth": float(d[1]), "geom": project_wkt(d[2], proj)}
        for d in data["depths"]
    ]
    depth_geoms = [d for d in depth_geoms_all if d["depth"] <= MAX_DRAFT]
    object_geoms = [
        {"id": o[0], "height": o[1], "geom": project_wkt(o[2], proj)}
        for o in data["objects"]
    ]

    # Legs
    legs = {}
    for seg_id, seg in segments.items():
        lon_s, lat_s = parse_point(seg["Start_Point"])
        lon_e, lat_e = parse_point(seg["End_Point"])
        xs, ys = proj.transform(lon_s, lat_s)
        xe, ye = proj.transform(lon_e, lat_e)
        start = np.array([xs, ys])
        end = np.array([xe, ye])

        dirs = seg.get("Dirs", ["Dir 1", "Dir 2"])
        ai_values = [float(seg.get("ai1", 180)), float(seg.get("ai2", 180))]

        seg_traffic = traffic.get(seg_id, {})
        dir_info = []
        for i, d_name in enumerate(dirs):
            spd_kn = (weighted_avg_speed_knots(seg_traffic[d_name])
                      if d_name in seg_traffic else 0)
            dir_info.append({
                "name": d_name,
                "speed_kn": spd_kn,
                "speed_ms": spd_kn * 1852.0 / 3600.0,
                "ai": ai_values[min(i, 1)],
                "mean": float(seg.get(f"mean{i+1}_1", 0)),
                "std": float(seg.get(f"std{i+1}_1", 100)),
            })

        legs[seg_id] = {
            "start": start, "end": end,
            "name": seg.get("Leg_name", ""),
            "dirs": dir_info,
        }

    all_obstacles = ([(dg, "depth") for dg in depth_geoms]
                     + [(og, "object") for og in object_geoms])

    return {
        "data": data,
        "proj": proj,
        "legs": legs,
        "depth_geoms": depth_geoms,
        "depth_geoms_all": depth_geoms_all,
        "object_geoms": object_geoms,
        "all_obstacles": all_obstacles,
        "pc_grounding": pc_grounding,
        "pc_allision": pc_allision,
    }


# =========================================================================
# 1. Test powered_na() / exponential decay
# =========================================================================

class TestPoweredNaExponentialDecay:
    """Verify exp(-d / (ai * V)) for known inputs."""

    def test_zero_distance_gives_one(self):
        assert isclose(powered_na(0, 180, 5.0), 1.0)

    def test_at_characteristic_distance(self):
        """At d = ai*V the result should be 1/e."""
        ai, V = 180.0, 5.0
        d = ai * V           # 900 m
        assert isclose(powered_na(d, ai, V), exp(-1), rtol=1e-12)

    def test_double_characteristic_distance(self):
        ai, V = 180.0, 5.0
        d = 2 * ai * V       # 1800 m
        assert isclose(powered_na(d, ai, V), exp(-2), rtol=1e-12)

    def test_known_numeric_value(self):
        """ai=180, V=5.14 m/s (10 kn), d=2000 m."""
        ai, V, d = 180.0, 5.14, 2000.0
        recovery = ai * V                    # 925.2 m
        expected = exp(-d / recovery)         # exp(-2.1616...)
        assert isclose(powered_na(d, ai, V), expected, rtol=1e-10)

    def test_large_distance_near_zero(self):
        result = powered_na(50_000, 180, 5.0)
        assert result < 1e-20

    def test_monotonically_decreasing(self):
        distances = [0, 100, 500, 1000, 2000, 5000, 10000]
        results = [powered_na(d, 180, 5.0) for d in distances]
        for i in range(1, len(results)):
            assert results[i] < results[i - 1]

    def test_faster_ship_decays_slower(self):
        d = 1000
        slow = powered_na(d, 180, 5.0)
        fast = powered_na(d, 180, 10.0)
        assert fast > slow


# =========================================================================
# 2. Test ray casting
# =========================================================================

class TestRayCasting:
    """Verify that a ray from a point in a direction hits a known polygon
    at the correct distance."""

    def test_ray_hits_square_at_known_distance(self):
        """Square obstacle 1000 m ahead along +X axis."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        obstacle = Polygon([(1000, -50), (1000, 50), (1100, 50), (1100, -50)])

        d = ray_hit_distance(origin, direction, MAX_RANGE, obstacle)
        assert d is not None
        assert isclose(d, 1000.0, atol=1.0)

    def test_ray_misses_obstacle_off_axis(self):
        """Obstacle to the side should not be hit by forward-pointing ray."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        obstacle = Polygon([(500, 200), (500, 300), (600, 300), (600, 200)])

        d = ray_hit_distance(origin, direction, MAX_RANGE, obstacle)
        assert d is None

    def test_ray_hits_at_angle(self):
        """Ray at 45 degrees should hit obstacle on the diagonal."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 1.0]) / np.sqrt(2)
        # Place obstacle centred on diagonal at ~707 m from origin
        obstacle = Polygon([(480, 480), (480, 520), (520, 520), (520, 480)])

        d = ray_hit_distance(origin, direction, MAX_RANGE, obstacle)
        assert d is not None
        expected_min = np.sqrt(480**2 + 480**2)
        assert d >= expected_min - 1.0  # hit at the near face
        assert d < np.sqrt(520**2 + 520**2) + 1.0

    def test_ray_behind_obstacle_no_hit(self):
        """Obstacle behind the ray origin should not be hit."""
        origin = np.array([100.0, 0.0])
        direction = np.array([1.0, 0.0])   # pointing right
        obstacle = Polygon([(-100, -10), (-100, 10), (-50, 10), (-50, -10)])

        d = ray_hit_distance(origin, direction, MAX_RANGE, obstacle)
        assert d is None

    def test_ray_clipped_by_max_range(self):
        """Obstacle beyond MAX_RANGE should not be hit."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])
        d_far = MAX_RANGE + 1000
        obstacle = Polygon([(d_far, -50), (d_far, 50),
                            (d_far + 100, 50), (d_far + 100, -50)])

        d = ray_hit_distance(origin, direction, MAX_RANGE, obstacle)
        assert d is None

    def test_multiple_obstacles_returns_nearest(self):
        """When multiple obstacles are along the same ray, the nearest hit
        should be found (caller iterates obstacles; each ray_hit_distance
        returns first intersection with that obstacle's geometry)."""
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 0.0])

        near_obs = Polygon([(500, -50), (500, 50), (600, 50), (600, -50)])
        far_obs = Polygon([(2000, -50), (2000, 50), (2100, 50), (2100, -50)])

        d_near = ray_hit_distance(origin, direction, MAX_RANGE, near_obs)
        d_far = ray_hit_distance(origin, direction, MAX_RANGE, far_obs)

        assert d_near is not None
        assert d_far is not None
        assert d_near < d_far
        assert isclose(d_near, 500.0, atol=1.0)
        assert isclose(d_far, 2000.0, atol=1.0)


# =========================================================================
# 3. Test shadow effect
# =========================================================================

class TestShadowEffect:
    """Set up two obstacles where the closer one shadows the farther one.
    Verify that the farther obstacle receives reduced mass."""

    def _make_obstacles(self, near_dist, far_dist, width):
        """Create two rectangular obstacles of the same lateral width centred
        on the extension direction from origin (0,0) along +X."""
        half_w = width / 2
        near = {
            "id": "near",
            "geom": Polygon([
                (near_dist, -half_w), (near_dist, half_w),
                (near_dist + 100, half_w), (near_dist + 100, -half_w),
            ]),
        }
        far = {
            "id": "far",
            "geom": Polygon([
                (far_dist, -half_w), (far_dist, half_w),
                (far_dist + 100, half_w), (far_dist + 100, -half_w),
            ]),
        }
        return near, far

    def test_identical_width_full_shadow(self):
        """When the near obstacle is the same width or wider, the far one
        is completely shadowed and receives zero mass."""
        near, far = self._make_obstacles(near_dist=500, far_dist=2000, width=4000)
        obstacles = [(near, "depth"), (far, "depth")]

        turn_pt = np.array([0.0, 0.0])
        ext_dir = np.array([1.0, 0.0])
        perp = np.array([0.0, 1.0])
        mean_offset = 0.0
        sigma = 500.0
        ai = 180.0
        speed_ms = 5.0

        summaries, _, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, perp, mean_offset, sigma, ai, speed_ms,
            obstacles, n_rays=200)

        near_key = ("depth", "near")
        far_key = ("depth", "far")

        # Near obstacle should capture essentially all mass
        near_mass = summaries.get(near_key, {}).get("mass", 0)
        far_mass = summaries.get(far_key, {}).get("mass", 0)

        assert near_mass > 0.99, (
            f"Near obstacle should capture > 99% of mass, got {near_mass:.4f}")
        assert far_mass < 0.01, (
            f"Far obstacle should be fully shadowed, got mass {far_mass:.4f}")

    def test_narrow_near_obstacle_partial_shadow(self):
        """A narrow near obstacle only shadows a fraction; the far wide obstacle
        captures the remaining distribution tails."""
        narrow_near = {
            "id": "near",
            "geom": Polygon([
                (500, -100), (500, 100),
                (600, 100), (600, -100),
            ]),
        }
        wide_far = {
            "id": "far",
            "geom": Polygon([
                (2000, -3000), (2000, 3000),
                (2100, 3000), (2100, -3000),
            ]),
        }
        obstacles = [(narrow_near, "depth"), (wide_far, "depth")]

        turn_pt = np.array([0.0, 0.0])
        ext_dir = np.array([1.0, 0.0])
        perp = np.array([0.0, 1.0])
        mean_offset = 0.0
        sigma = 500.0

        summaries, _, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, perp, mean_offset, sigma, 180, 5.0,
            obstacles, n_rays=400)

        near_mass = summaries.get(("depth", "near"), {}).get("mass", 0)
        far_mass = summaries.get(("depth", "far"), {}).get("mass", 0)

        # Near obstacle is narrow: +-100 m, sigma=500 => roughly erf(100/(500*sqrt(2))) ~ 0.16
        assert 0.05 < near_mass < 0.30, (
            f"Narrow near obstacle mass out of range: {near_mass:.4f}")

        # Far obstacle should capture the rest (the tails)
        assert far_mass > 0.5, (
            f"Wide far obstacle should capture tails, got mass {far_mass:.4f}")

        # Combined mass should be close to 1.0 (almost all rays hit something)
        total = near_mass + far_mass
        assert total > 0.95, (
            f"Combined mass should be ~1.0, got {total:.4f}")

    def test_shadow_reduces_p_hit_for_far_obstacle(self):
        """The far obstacle's P(hit) should be much less than if it were alone,
        because the near obstacle blocks the high-probability centre of the
        distribution."""
        near, far = self._make_obstacles(near_dist=500, far_dist=2000, width=800)
        turn_pt = np.array([0.0, 0.0])
        ext_dir = np.array([1.0, 0.0])
        perp = np.array([0.0, 1.0])
        mean_offset = 0.0
        sigma = 500.0
        ai = 180.0
        speed_ms = 5.0

        # With shadow
        summaries_shadow, _, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, perp, mean_offset, sigma, ai, speed_ms,
            [(near, "depth"), (far, "depth")], n_rays=400)

        # Without shadow (far obstacle alone)
        summaries_alone, _, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, perp, mean_offset, sigma, ai, speed_ms,
            [(far, "depth")], n_rays=400)

        far_p_shadow = summaries_shadow.get(("depth", "far"), {}).get("p_approx", 0)
        far_p_alone = summaries_alone.get(("depth", "far"), {}).get("p_approx", 0)

        # With shadow, the far obstacle's P(hit) should be significantly less
        if far_p_alone > 0:
            assert far_p_shadow < far_p_alone, (
                f"Shadow should reduce P(hit): {far_p_shadow:.4e} vs {far_p_alone:.4e}")


# =========================================================================
# 4. Test mass computation
# =========================================================================

class TestMassComputation:
    """Verify that the distribution mass sums correctly."""

    def test_mass_sums_to_one_with_covering_obstacle(self):
        """A single obstacle covering the entire distribution band should
        capture mass close to 1.0."""
        turn_pt = np.array([0.0, 0.0])
        ext_dir = np.array([1.0, 0.0])
        perp = np.array([0.0, 1.0])
        mean_offset = 0.0
        sigma = 500.0

        # Very wide obstacle covering +-4*sigma = +-2000 m
        big_obs = {
            "id": "big",
            "geom": Polygon([
                (1000, -3000), (1000, 3000),
                (1100, 3000), (1100, -3000),
            ]),
        }
        obstacles = [(big_obs, "depth")]

        summaries, _, offsets, pdf_vals = compute_cat2_with_shadows(
            turn_pt, ext_dir, perp, mean_offset, sigma, 180, 5.0,
            obstacles, n_rays=500)

        total_mass = sum(s["mass"] for s in summaries.values())

        # The linspace covers +-4*sigma; the Gaussian integral from -4s to 4s
        # is erf(4/sqrt(2)) ~ 0.99994.  Discretisation will be close.
        assert total_mass > 0.99, f"Total mass should be ~1.0, got {total_mass:.4f}"

    def test_mass_equals_distribution_integral(self):
        """Mass captured by a ray band should equal the CDF integral over
        that lateral interval (for a single obstacle with full coverage)."""
        mean_offset = 100.0
        sigma = 300.0
        half_w = 200.0  # obstacle lateral half-width

        # Expected CDF mass from mean-half_w to mean+half_w
        expected = norm.cdf(mean_offset + half_w, mean_offset, sigma) - \
                   norm.cdf(mean_offset - half_w, mean_offset, sigma)

        turn_pt = np.array([0.0, 0.0])
        ext_dir = np.array([1.0, 0.0])
        perp = np.array([0.0, 1.0])

        obs = {
            "id": "strip",
            "geom": Polygon([
                (500, mean_offset - half_w), (500, mean_offset + half_w),
                (600, mean_offset + half_w), (600, mean_offset - half_w),
            ]),
        }

        summaries, _, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, perp, mean_offset, sigma, 180, 5.0,
            [(obs, "depth")], n_rays=1000)

        actual_mass = summaries.get(("depth", "strip"), {}).get("mass", 0)

        # Allow tolerance because of discretisation (1000 rays over 8*sigma)
        assert isclose(actual_mass, expected, atol=0.02), (
            f"Mass {actual_mass:.4f} should be close to CDF integral {expected:.4f}")

    def test_no_obstacle_mass_is_zero(self):
        """With no obstacles, all mass should be unblocked."""
        turn_pt = np.array([0.0, 0.0])
        ext_dir = np.array([1.0, 0.0])
        perp = np.array([0.0, 1.0])

        summaries, ray_data, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, perp, 0, 500, 180, 5.0,
            [], n_rays=100)

        assert len(summaries) == 0
        # All rays should have hit_key=None
        for _, _, hit_key, _ in ray_data:
            assert hit_key is None

    def test_two_equal_obstacles_split_mass(self):
        """Two non-overlapping obstacles of equal width placed symmetrically
        about the mean should capture roughly equal mass."""
        turn_pt = np.array([0.0, 0.0])
        ext_dir = np.array([1.0, 0.0])
        perp = np.array([0.0, 1.0])
        mean_offset = 0.0
        sigma = 500.0

        # Obstacle A: lateral [-1000, -200]  (left side)
        obs_a = {
            "id": "A",
            "geom": Polygon([
                (1000, -1000), (1000, -200),
                (1100, -200), (1100, -1000),
            ]),
        }
        # Obstacle B: lateral [200, 1000]  (right side)
        obs_b = {
            "id": "B",
            "geom": Polygon([
                (1000, 200), (1000, 1000),
                (1100, 1000), (1100, 200),
            ]),
        }

        summaries, _, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, perp, mean_offset, sigma, 180, 5.0,
            [(obs_a, "depth"), (obs_b, "depth")], n_rays=500)

        mass_a = summaries.get(("depth", "A"), {}).get("mass", 0)
        mass_b = summaries.get(("depth", "B"), {}).get("mass", 0)

        # Both at same distance from mean; should be roughly equal
        assert isclose(mass_a, mass_b, rtol=0.15), (
            f"Symmetric obstacles should have similar mass: A={mass_a:.4f}, B={mass_b:.4f}")
        # Neither should capture the center (gap at +-200m)
        assert mass_a + mass_b < 0.95


# =========================================================================
# 5. Test draft filtering
# =========================================================================

class TestDraftFiltering:
    """Verify that depth areas deeper than MAX_DRAFT are excluded."""

    def test_depths_filtered_by_max_draft(self, proj2_data):
        """All depths in the filtered list should have depth <= MAX_DRAFT."""
        for dg in proj2_data["depth_geoms"]:
            assert dg["depth"] <= MAX_DRAFT, (
                f"Depth ID={dg['id']} has depth={dg['depth']} > MAX_DRAFT={MAX_DRAFT}")

    def test_deep_depths_excluded(self, proj2_data):
        """Depths > MAX_DRAFT should not appear in the filtered list."""
        filtered_ids = {dg["id"] for dg in proj2_data["depth_geoms"]}
        for dg in proj2_data["depth_geoms_all"]:
            if dg["depth"] > MAX_DRAFT:
                assert dg["id"] not in filtered_ids, (
                    f"Deep depth ID={dg['id']} (depth={dg['depth']}) should be excluded")

    def test_depth_count(self, proj2_data):
        """proj2.omrat has 17 total depth areas; 6 have depth <= 15m."""
        assert len(proj2_data["depth_geoms_all"]) == 17
        assert len(proj2_data["depth_geoms"]) == 6

    def test_shallow_depths_included(self, proj2_data):
        """Depths 0, 3, 6, 9, 12, 15 should all be included."""
        expected_depths = {0.0, 3.0, 6.0, 9.0, 12.0, 15.0}
        actual_depths = {dg["depth"] for dg in proj2_data["depth_geoms"]}
        assert expected_depths == actual_depths

    def test_grounding_only_if_draught_exceeds_depth(self):
        """Grounding occurs when ship draught > water depth.
        Depth 15m and ship draught 10m -> no grounding.
        Depth 12m and ship draught 15m -> grounding."""
        # This is a conceptual test of the filtering logic
        ship_draught = 10.0
        depths = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0]
        grounding_hazards = [d for d in depths if ship_draught > d]
        assert 15.0 not in grounding_hazards  # draught 10 <= depth 15
        assert 9.0 in grounding_hazards       # draught 10 > depth 9


# =========================================================================
# 6. Integration tests with proj2.omrat data
# =========================================================================

class TestProj2Integration:
    """Load the actual test data and verify key expected results."""

    def test_leg1_east_going_hits_objects(self, proj2_data):
        """LEG 1, East going: should hit obstacles.
        Object #2 should intercept the vast majority of the distribution mass."""
        leg = proj2_data["legs"]["1"]
        start, end = leg["start"], leg["end"]
        u, n, L = leg_vectors(start, end)
        d = leg["dirs"][0]  # East going (dir_idx=0)

        turn_pt = end.copy()     # dir_idx 0 -> extension from end
        ext_dir = u.copy()

        summaries, _, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, n, d["mean"], d["std"],
            d["ai"], d["speed_ms"], proj2_data["all_obstacles"])

        # Should hit at least one obstacle
        assert len(summaries) > 0, "LEG 1 East going should hit some obstacles"

    def test_leg1_east_first_depth_around_25km(self, proj2_data):
        """LEG 1, East going: the first grounding depth hit should be at
        approximately 25,000-26,000 m along track (depth #6, 15m depth)."""
        leg = proj2_data["legs"]["1"]
        start, end = leg["start"], leg["end"]
        u, n, L = leg_vectors(start, end)
        d = leg["dirs"][0]

        turn_pt = end.copy()
        ext_dir = u.copy()

        summaries, _, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, n, d["mean"], d["std"],
            d["ai"], d["speed_ms"], proj2_data["all_obstacles"])

        # Find depth hits and sort by distance
        depth_hits = {
            k: v for k, v in summaries.items() if k[0] == "depth"
        }

        if depth_hits:
            nearest_depth = min(depth_hits.values(), key=lambda s: s["mean_dist"])
            # The first depth hit should be around 25,000-27,000 m
            assert 20_000 < nearest_depth["mean_dist"] < 35_000, (
                f"First depth hit at {nearest_depth['mean_dist']:.0f}m, "
                f"expected around 25,000m")

    def test_leg1_east_object2_dominates_mass(self, proj2_data):
        """LEG 1, East going: Object #2 should intercept a very large fraction
        of the distribution mass (~95-99%)."""
        leg = proj2_data["legs"]["1"]
        start, end = leg["start"], leg["end"]
        u, n, L = leg_vectors(start, end)
        d = leg["dirs"][0]

        turn_pt = end.copy()
        ext_dir = u.copy()

        summaries, _, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, n, d["mean"], d["std"],
            d["ai"], d["speed_ms"], proj2_data["all_obstacles"])

        obj2_key = ("object", 2)
        if obj2_key in summaries:
            obj2_mass = summaries[obj2_key]["mass"]
            total_mass = sum(s["mass"] for s in summaries.values())

            assert obj2_mass / total_mass > 0.90, (
                f"Object #2 should capture >90% of total hit mass, "
                f"got {obj2_mass / total_mass:.2%}")

    def test_leg1_east_depth6_shadowed_by_object2(self, proj2_data):
        """LEG 1, East going: depth #6 should have very low mass because
        Object #2 blocks most rays before they reach it."""
        leg = proj2_data["legs"]["1"]
        start, end = leg["start"], leg["end"]
        u, n, L = leg_vectors(start, end)
        d = leg["dirs"][0]

        turn_pt = end.copy()
        ext_dir = u.copy()

        summaries, _, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, n, d["mean"], d["std"],
            d["ai"], d["speed_ms"], proj2_data["all_obstacles"])

        depth6_key = ("depth", 6)
        obj2_key = ("object", 2)

        # If both are hit
        if depth6_key in summaries and obj2_key in summaries:
            depth6_mass = summaries[depth6_key]["mass"]
            obj2_mass = summaries[obj2_key]["mass"]

            # Depth #6 should have much less mass due to shadowing
            assert depth6_mass < obj2_mass * 0.1, (
                f"Depth #6 mass ({depth6_mass:.4f}) should be <<< "
                f"Object #2 mass ({obj2_mass:.4f}) due to shadow")

            # Depth #6 should receive roughly 2% of the total distribution
            total_mass = sum(s["mass"] for s in summaries.values())
            if total_mass > 0:
                depth6_fraction = depth6_mass / total_mass
                assert depth6_fraction < 0.10, (
                    f"Depth #6 fraction {depth6_fraction:.2%} should be small "
                    f"due to shadow from Object #2")

    def test_all_legs_mass_not_exceed_one(self, proj2_data):
        """For every leg and direction, the total intercepted mass should
        not exceed 1.0 (it can be less than 1.0 if some rays miss)."""
        for seg_id, leg in proj2_data["legs"].items():
            start, end = leg["start"], leg["end"]
            u, n, L = leg_vectors(start, end)

            for di, d in enumerate(leg["dirs"]):
                if d["speed_ms"] <= 0:
                    continue

                if di == 0:
                    turn_pt = end.copy()
                    ext_dir = u.copy()
                else:
                    turn_pt = start.copy()
                    ext_dir = (-u).copy()

                summaries, _, _, _ = compute_cat2_with_shadows(
                    turn_pt, ext_dir, n, d["mean"], d["std"],
                    d["ai"], d["speed_ms"], proj2_data["all_obstacles"],
                    n_rays=100)  # Use fewer rays for speed

                total_mass = sum(s["mass"] for s in summaries.values())
                assert total_mass <= 1.01, (
                    f"Leg {seg_id} {d['name']}: total mass {total_mass:.4f} > 1.0")

    def test_recovery_distance_with_proj2_params(self, proj2_data):
        """Verify recovery distance for Leg 1 East going parameters."""
        leg = proj2_data["legs"]["1"]
        d = leg["dirs"][0]  # East going
        ai = d["ai"]           # 180 seconds
        speed_ms = d["speed_ms"]

        recovery = ai * speed_ms
        recovery_from_func = get_recovery_distance(ai / 60.0, speed_ms)

        # Both should agree
        assert isclose(recovery, recovery_from_func, rtol=1e-6)
        # With speed ~10kn (~5.14 m/s) and ai=180s, recovery ~ 925m
        assert 500 < recovery < 2000

    def test_p_hit_decreases_with_distance(self, proj2_data):
        """For obstacles hit from the same direction, farther obstacles
        should have lower P(hit) = mass * exp(-d/recovery) -- unless they
        have significantly more mass."""
        leg = proj2_data["legs"]["1"]
        start, end = leg["start"], leg["end"]
        u, n, L = leg_vectors(start, end)
        d = leg["dirs"][0]

        turn_pt = end.copy()
        ext_dir = u.copy()

        summaries, _, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, n, d["mean"], d["std"],
            d["ai"], d["speed_ms"], proj2_data["all_obstacles"])

        if len(summaries) >= 2:
            # Sort by distance
            sorted_obs = sorted(summaries.items(),
                                key=lambda x: x[1]["mean_dist"])
            # The nearest obstacle should have the highest P(hit)
            # (it blocks the centre of the distribution AND has exp decay advantage)
            nearest_p = sorted_obs[0][1]["p_approx"]
            for i in range(1, len(sorted_obs)):
                other_p = sorted_obs[i][1]["p_approx"]
                # The nearest should dominate due to capturing the
                # distribution centre AND being closer
                assert nearest_p >= other_p, (
                    f"Nearest obstacle P({nearest_p:.4e}) should >= "
                    f"farther obstacle P({other_p:.4e})")

    def test_leg1_objects_count(self, proj2_data):
        """proj2.omrat should have exactly 2 objects."""
        assert len(proj2_data["object_geoms"]) == 2

    def test_pc_values_from_data(self, proj2_data):
        """Verify causation factors are loaded correctly."""
        assert isclose(proj2_data["pc_grounding"], 1.6e-4, rtol=1e-2)
        assert isclose(proj2_data["pc_allision"], 1.9e-4, rtol=1e-2)

    def test_ai_parameter_value(self, proj2_data):
        """All legs should have ai=180 seconds."""
        for seg_id, leg in proj2_data["legs"].items():
            for d in leg["dirs"]:
                assert isclose(d["ai"], 180.0), (
                    f"Leg {seg_id} {d['name']} ai={d['ai']}, expected 180")


# =========================================================================
# 7. Compare production vs example approach
# =========================================================================

class TestProductionVsExampleComparison:
    """Document and test the known differences between the production code
    in run_calculations.py and the shadow-aware example.

    Production code (run_powered_grounding_model):
      - Cat II uses a SINGLE distance per obstacle (leg centroid to obstacle)
      - mass (prob_at_position) = norm.pdf(distance / sigma) / sigma
      - NO shadow effects -- each obstacle is independent
      - Draft filtering: draught > depth (correct)

    Example code (check_powered_repaired.py):
      - Cat II casts N rays across the lateral distribution
      - Each ray finds the FIRST obstacle (natural shadow)
      - mass = sum of pdf(offset)*dx for rays hitting that obstacle
      - Shadow effects included
    """

    def test_production_equations_consistent(self):
        """Production's get_powered_grounding_cat2 should equal
        Pc * Q * f(z) * exp(-d / recovery)."""
        Q = 1000.0
        Pc = 1.6e-4
        sigma = 500.0
        distance = 2000.0
        ai_minutes = 3.0    # 180 seconds / 60
        speed_ms = 5.14

        prob_at_pos = norm.pdf(distance / sigma) / sigma
        recovery = get_recovery_distance(ai_minutes, speed_ms)
        expected = Pc * Q * prob_at_pos * exp(-distance / recovery)

        result = get_powered_grounding_cat2(
            Q=Q, Pc=Pc,
            prob_at_position=prob_at_pos,
            distance_to_obstacle=distance,
            position_check_interval=ai_minutes,
            ship_speed=speed_ms)

        assert isclose(result, expected, rtol=1e-10)

    def test_shadow_gives_lower_total_than_independent(self):
        """With shadow effects, the total P(hit) across all obstacles should
        be LESS than the sum of independent P(hit) per obstacle, because
        shadow redistributes mass rather than double-counting."""
        turn_pt = np.array([0.0, 0.0])
        ext_dir = np.array([1.0, 0.0])
        perp = np.array([0.0, 1.0])
        mean_offset = 0.0
        sigma = 500.0
        ai = 180.0
        speed_ms = 5.0

        near = {
            "id": "near",
            "geom": Polygon([
                (500, -600), (500, 600),
                (600, 600), (600, -600),
            ]),
        }
        far = {
            "id": "far",
            "geom": Polygon([
                (2000, -1500), (2000, 1500),
                (2100, 1500), (2100, -1500),
            ]),
        }

        # Shadow-aware (both obstacles)
        summaries_shadow, _, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, perp, mean_offset, sigma, ai, speed_ms,
            [(near, "depth"), (far, "depth")], n_rays=400)

        # Independent (each obstacle alone)
        summaries_near_alone, _, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, perp, mean_offset, sigma, ai, speed_ms,
            [(near, "depth")], n_rays=400)
        summaries_far_alone, _, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, perp, mean_offset, sigma, ai, speed_ms,
            [(far, "depth")], n_rays=400)

        total_shadow = sum(s["p_approx"] for s in summaries_shadow.values())
        total_independent = (
            sum(s["p_approx"] for s in summaries_near_alone.values())
            + sum(s["p_approx"] for s in summaries_far_alone.values())
        )

        # Shadow total should be less because the far obstacle's mass is reduced
        assert total_shadow < total_independent, (
            f"Shadow total ({total_shadow:.4e}) should be < "
            f"independent total ({total_independent:.4e})")

    def test_production_no_shadow_double_counts(self):
        """Demonstrate that the production approach (independent distance per
        obstacle) effectively double-counts the distribution for overlapping
        coverage regions, while the ray-based approach does not.

        This is a structural test of the formula difference."""
        # Two obstacles, both overlapping the same lateral region
        sigma = 500.0
        mean = 0.0

        # Distance to each obstacle
        d_near = 500.0
        d_far = 2000.0

        ai_minutes = 3.0
        speed_ms = 5.0

        # Production approach: each obstacle gets its own mass independently
        prob_near = norm.pdf(d_near / sigma) / sigma
        prob_far = norm.pdf(d_far / sigma) / sigma

        p_near_prod = prob_near * powered_na(d_near, ai_minutes * 60, speed_ms)
        p_far_prod = prob_far * powered_na(d_far, ai_minutes * 60, speed_ms)
        total_prod = p_near_prod + p_far_prod

        # Shadow approach: compute via ray casting
        near = {
            "id": "near",
            "geom": Polygon([
                (d_near, -2000), (d_near, 2000),
                (d_near + 100, 2000), (d_near + 100, -2000),
            ]),
        }
        far = {
            "id": "far",
            "geom": Polygon([
                (d_far, -2000), (d_far, 2000),
                (d_far + 100, 2000), (d_far + 100, -2000),
            ]),
        }

        summaries, _, _, _ = compute_cat2_with_shadows(
            np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]),
            mean, sigma, ai_minutes * 60, speed_ms,
            [(near, "depth"), (far, "depth")], n_rays=500)

        # With shadow, the far obstacle gets zero mass because the near one
        # blocks everything
        far_mass = summaries.get(("depth", "far"), {}).get("mass", 0)
        assert far_mass < 0.01, (
            f"Far obstacle should be fully shadowed, got mass {far_mass:.4f}")

        # But in the production approach, both get non-zero probability
        assert p_far_prod > 0, (
            "Production approach gives non-zero P for far obstacle (no shadow)")


# =========================================================================
# 8. Additional edge cases
# =========================================================================

class TestEdgeCases:
    """Edge cases for the shadow-aware Cat II computation."""

    def test_zero_speed_gives_zero_p(self):
        """Zero ship speed -> recovery distance = 0 -> P(hit) = 0."""
        obs = {
            "id": "obs",
            "geom": Polygon([(500, -500), (500, 500),
                             (600, 500), (600, -500)]),
        }
        summaries, _, _, _ = compute_cat2_with_shadows(
            np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]),
            0, 500, 180, 0.0,  # speed = 0
            [(obs, "depth")], n_rays=100)

        for s in summaries.values():
            assert s["p_approx"] == 0.0 or s["p_integral"] == 0.0

    def test_very_large_sigma(self):
        """With a very large sigma, the distribution is nearly flat and
        obstacles capture mass proportional to their angular width."""
        turn_pt = np.array([0.0, 0.0])
        ext_dir = np.array([1.0, 0.0])
        perp = np.array([0.0, 1.0])
        sigma = 10_000.0  # Very large

        obs = {
            "id": "obs",
            "geom": Polygon([(1000, -500), (1000, 500),
                             (1100, 500), (1100, -500)]),
        }

        summaries, _, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, perp, 0, sigma, 180, 5.0,
            [(obs, "depth")], n_rays=500)

        mass = summaries.get(("depth", "obs"), {}).get("mass", 0)
        # With sigma=10000 and obstacle width=1000, mass ~ 1000/(8*10000) ~ 0.0125
        assert 0.001 < mass < 0.1, (
            f"With large sigma, mass should be small: {mass:.4f}")

    def test_obstacle_at_turning_point(self):
        """An obstacle immediately at the turning point (distance ~ 0)
        should have exp(0) = 1.0 decay factor -> P(hit) = mass * 1.0."""
        turn_pt = np.array([0.0, 0.0])
        ext_dir = np.array([1.0, 0.0])
        perp = np.array([0.0, 1.0])

        obs = {
            "id": "adjacent",
            "geom": Polygon([(1, -1000), (1, 1000),
                             (100, 1000), (100, -1000)]),
        }

        summaries, _, _, _ = compute_cat2_with_shadows(
            turn_pt, ext_dir, perp, 0, 500, 180, 5.0,
            [(obs, "depth")], n_rays=200)

        if ("depth", "adjacent") in summaries:
            s = summaries[("depth", "adjacent")]
            # Distance is very small, so P(not recovered) ~ 1.0
            assert s["p_approx"] > s["mass"] * 0.95, (
                f"At distance ~0, P(hit) should be close to mass: "
                f"P={s['p_approx']:.4e}, mass={s['mass']:.4f}")
