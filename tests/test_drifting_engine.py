"""Unit tests for ``drifting.engine``.

The engine module is used by ``compute.drifting_model`` (for
``directional_distance_to_point_from_offset_leg``,  ``LegState``, and
``compass_to_math_deg``) and by ``geometries.get_drifting_overlap``
(same two names).  The other public functions -- ``build_directional_corridor``,
``evaluate_leg_direction``, ``edge_average_distance_m`` etc. -- are
exercised by the diagnostic scripts in ``drifting/debug/`` but had
little direct test coverage.

This file covers them directly.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from shapely.geometry import (
    GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon,
    Point, Polygon, box,
)
from shapely.geometry.polygon import LinearRing

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drifting.engine import (
    DIRECTIONS_COMPASS_DEG,
    DepthTarget,
    StructureTarget,
    ShipState,
    LegState,
    DriftConfig,
    TargetHit,
    available_targets,
    interesting_targets,
    compass_to_math_deg,
    _offset_line_perpendicular,
    build_directional_corridor,
    corridor_width_m,
    directional_distance_m,
    directional_distance_to_point_from_offset_leg,
    edge_average_distance_m,
    coverage_percent,
    edge_hit_percent,
    evaluate_leg_direction,
)


# ---------------------------------------------------------------------------
# Fixtures: simple east-going leg and a selection of targets
# ---------------------------------------------------------------------------

@pytest.fixture
def east_leg() -> LegState:
    """10 km east-going leg at y=0, zero offset, 1 m lateral sigma."""
    return LegState(
        leg_id='L1',
        line=LineString([(0, 0), (10_000, 0)]),
        mean_offset_m=0.0,
        lateral_sigma_m=1.0,
    )


@pytest.fixture
def default_cfg() -> DriftConfig:
    return DriftConfig(reach_distance_m=5_000.0)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestDirectionsConstant:
    def test_eight_cardinals_and_ordinals(self):
        assert DIRECTIONS_COMPASS_DEG == (0, 45, 90, 135, 180, 225, 270, 315)


# ---------------------------------------------------------------------------
# Dataclasses (frozen / immutable)
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_depth_target_is_frozen(self):
        d = DepthTarget('d1', 10.0, Point(0, 0))
        with pytest.raises(Exception):
            d.target_id = 'other'  # type: ignore[misc]

    def test_structure_target_basic(self):
        s = StructureTarget('s1', 25.0, Point(1, 2))
        assert s.target_id == 's1'
        assert s.top_height_m == 25.0

    def test_ship_state_defaults(self):
        s = ShipState(draught_m=10.0, anchor_d=2.0)
        assert s.ship_height_m is None
        assert s.respect_structure_height is False

    def test_leg_state_basic(self, east_leg):
        assert east_leg.leg_id == 'L1'
        assert east_leg.line.length == 10_000.0

    def test_drift_config_defaults(self):
        c = DriftConfig(reach_distance_m=100.0)
        assert c.corridor_sigma_multiplier == 3.0
        assert c.use_leg_offset_for_distance is False

    def test_target_hit_fields(self):
        t = TargetHit(
            leg_id='L1', direction_deg=90, role='grounding',
            target_id='d1', distance_m=150.0, coverage_percent=25.0,
        )
        assert t.role == 'grounding'
        assert t.distance_m == 150.0


# ---------------------------------------------------------------------------
# available_targets / interesting_targets
# ---------------------------------------------------------------------------

class TestAvailableTargets:
    def test_counts_iterables(self):
        depths = [DepthTarget('d1', 5.0, Point(0, 0))] * 3
        structs = [StructureTarget('s1', 20.0, Point(0, 0))] * 2
        assert available_targets(depths, structs) == {
            'depth_count': 3, 'structure_count': 2,
        }

    def test_empty(self):
        assert available_targets([], []) == {
            'depth_count': 0, 'structure_count': 0,
        }


class TestInterestingTargets:
    def test_shallow_depths_kept_for_grounding(self):
        """Grounding: depth < draught."""
        depths = [
            DepthTarget('shallow', 3.0, Point(0, 0)),
            DepthTarget('medium', 8.0, Point(0, 0)),
            DepthTarget('deep', 20.0, Point(0, 0)),
        ]
        ship = ShipState(draught_m=10.0, anchor_d=2.0)  # anchor_limit = 20
        result = interesting_targets(ship, depths, [])
        ids = [d.target_id for d in result['grounding_depths']]
        # shallow (3<10) and medium (8<10) qualify.
        assert sorted(ids) == ['medium', 'shallow']

    def test_anchoring_uses_draught_times_anchor_d(self):
        depths = [
            DepthTarget('a', 5.0, Point(0, 0)),
            DepthTarget('b', 15.0, Point(0, 0)),
            DepthTarget('c', 30.0, Point(0, 0)),
        ]
        # draught=10, anchor_d=2.0 -> anchor_limit = 20
        ship = ShipState(draught_m=10.0, anchor_d=2.0)
        ids = [d.target_id for d in interesting_targets(ship, depths, [])['anchoring_depths']]
        assert sorted(ids) == ['a', 'b']  # depth 30 rejected

    def test_structures_filtered_by_ship_height_when_enabled(self):
        structs = [
            StructureTarget('s1', 10.0, Point(0, 0)),
            StructureTarget('s2', 20.0, Point(0, 0)),
            StructureTarget('s3', 30.0, Point(0, 0)),
        ]
        ship = ShipState(
            draught_m=10.0, anchor_d=2.0,
            ship_height_m=15.0, respect_structure_height=True,
        )
        result = interesting_targets(ship, [], structs)
        # Only structures <= 15 m: s1.
        assert [s.target_id for s in result['structures']] == ['s1']

    def test_structures_unfiltered_when_respect_height_false(self):
        structs = [
            StructureTarget('s1', 10.0, Point(0, 0)),
            StructureTarget('s2', 30.0, Point(0, 0)),
        ]
        ship = ShipState(draught_m=10.0, anchor_d=2.0, respect_structure_height=False)
        result = interesting_targets(ship, [], structs)
        assert len(result['structures']) == 2


# ---------------------------------------------------------------------------
# compass_to_math_deg
# ---------------------------------------------------------------------------

class TestCompassToMathDeg:
    @pytest.mark.parametrize("compass, math_deg", [
        (0, 90.0),     # N -> math 90
        (90, 0.0),     # E -> math 0
        (180, 270.0),  # S -> math 270
        (270, 180.0),  # W -> math 180
        (45, 45.0),    # NE -> math 45
        (315, 135.0),  # NW -> math 135
    ])
    def test_cardinals(self, compass, math_deg):
        assert compass_to_math_deg(compass) == pytest.approx(math_deg)

    def test_wraparound_above_360(self):
        # 450 compass degrees is 90 (E) -> math 0.
        assert compass_to_math_deg(450) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _offset_line_perpendicular
# ---------------------------------------------------------------------------

class TestOffsetLinePerpendicular:
    def test_zero_offset_returns_same_line(self):
        line = LineString([(0, 0), (100, 0)])
        result = _offset_line_perpendicular(line, 0.0)
        assert result is line

    def test_positive_offset_shifts_left(self):
        """Shapely's 'left' side means +y for an east-going line."""
        line = LineString([(0, 0), (100, 0)])
        result = _offset_line_perpendicular(line, 10.0)
        # All y-coordinates shifted +10.
        ys = [c[1] for c in result.coords]
        assert all(y == pytest.approx(10.0, abs=1e-6) for y in ys)

    def test_negative_offset_shifts_right(self):
        line = LineString([(0, 0), (100, 0)])
        result = _offset_line_perpendicular(line, -10.0)
        ys = [c[1] for c in result.coords]
        assert all(y == pytest.approx(-10.0, abs=1e-6) for y in ys)

    def test_tiny_offset_treated_as_zero(self):
        """Offsets below 1e-12 short-circuit to the original line."""
        line = LineString([(0, 0), (100, 0)])
        assert _offset_line_perpendicular(line, 1e-15) is line

    def test_degenerate_line_falls_back_to_original(self):
        """A zero-length line can't be offset -- function returns input."""
        line = LineString([(5, 5), (5, 5)])
        result = _offset_line_perpendicular(line, 10.0)
        # Shapely returns an empty shifted line; function falls back.
        assert result is line

    def test_bad_offset_hits_exception_fallback(self, monkeypatch):
        """If parallel_offset raises, the function falls back to input."""
        line = LineString([(0, 0), (10, 0)])

        def boom(*args, **kwargs):
            raise RuntimeError("synthetic")

        monkeypatch.setattr(LineString, 'parallel_offset', boom)
        result = _offset_line_perpendicular(line, 5.0)
        assert result is line

    def test_multilinestring_result_picks_longest(self, monkeypatch):
        """When parallel_offset returns a MultiLineString, the longest
        segment is kept as the shifted line."""
        line = LineString([(0, 0), (100, 0)])
        long_piece = LineString([(0, 5), (80, 5)])
        short_piece = LineString([(85, 5), (100, 5)])
        fake_mls = MultiLineString([long_piece, short_piece])
        monkeypatch.setattr(
            LineString, 'parallel_offset', lambda self, *a, **k: fake_mls
        )
        result = _offset_line_perpendicular(line, 5.0)
        assert isinstance(result, LineString)
        # The longest piece has length 80.
        assert result.length == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# build_directional_corridor
# ---------------------------------------------------------------------------

class TestBuildDirectionalCorridor:
    def test_east_drift_extends_east(self, east_leg):
        cfg = DriftConfig(reach_distance_m=5_000.0, corridor_sigma_multiplier=3.0)
        # compass 90 = east
        poly = build_directional_corridor(east_leg, 90, cfg)
        assert isinstance(poly, Polygon)
        # Polygon extends east of the leg (max x > 10 km).
        minx, miny, maxx, maxy = poly.bounds
        assert maxx > 10_000 + 1_000

    def test_south_drift_extends_south(self, east_leg):
        """For a north-up coordinate system, south drift shifts down."""
        cfg = DriftConfig(reach_distance_m=3_000.0)
        # compass 180 = south -> math 270 -> dy = sin(270°)*3000 = -3000
        poly = build_directional_corridor(east_leg, 180, cfg)
        minx, miny, maxx, maxy = poly.bounds
        assert miny < -1_000  # extends south of leg (y=0)

    def test_width_scales_with_sigma_multiplier(self, east_leg):
        # High sigma multiplier -> wider corridor.
        wide = build_directional_corridor(
            east_leg, 90, DriftConfig(reach_distance_m=1_000.0, corridor_sigma_multiplier=100.0),
        )
        narrow = build_directional_corridor(
            east_leg, 90, DriftConfig(reach_distance_m=1_000.0, corridor_sigma_multiplier=1.0),
        )
        # Wide corridor has larger area.
        assert wide.area > narrow.area


# ---------------------------------------------------------------------------
# corridor_width_m
# ---------------------------------------------------------------------------

class TestCorridorWidth:
    def test_default_no_direction_uses_base(self, east_leg):
        cfg = DriftConfig(reach_distance_m=1_000.0, corridor_sigma_multiplier=3.0)
        # sigma=1 -> spread = 3*1 = 3 -> base_width = 6
        assert corridor_width_m(east_leg, cfg) == pytest.approx(6.0)

    def test_drift_parallel_uses_base_width(self, east_leg):
        """Drift east along an east-going leg -> width ~= 2*spread."""
        cfg = DriftConfig(reach_distance_m=1_000.0, corridor_sigma_multiplier=3.0)
        # compass 90 = east, parallel to the leg
        # cross-drift axis is perpendicular (north) -> leg's cross-extent is 0
        w = corridor_width_m(east_leg, cfg, 90)
        assert w == pytest.approx(6.0)  # base width (leg has no cross-extent)

    def test_drift_perpendicular_uses_leg_length(self, east_leg):
        """Drift north across east-going leg -> width ~= leg length."""
        cfg = DriftConfig(reach_distance_m=1_000.0, corridor_sigma_multiplier=1.0)
        # compass 0 = north -> cross-drift axis = east -> extent = leg length (10 km)
        w = corridor_width_m(east_leg, cfg, 0)
        assert w == pytest.approx(10_000.0)

    def test_degenerate_leg_returns_base(self):
        leg = LegState(
            leg_id='deg', line=LineString([(0, 0), (0, 0)]),
            mean_offset_m=0.0, lateral_sigma_m=1.0,
        )
        cfg = DriftConfig(reach_distance_m=100.0)
        assert corridor_width_m(leg, cfg, 90) >= 1.0

    def test_empty_linestring_leg_returns_base(self):
        """Empty LineString -> len(coords) < 2 -> base width."""
        leg = LegState(
            leg_id='e', line=LineString(),
            mean_offset_m=0.0, lateral_sigma_m=1.0,
        )
        cfg = DriftConfig(reach_distance_m=100.0, corridor_sigma_multiplier=3.0)
        # sigma=1 -> spread = 3 -> base_width = 6
        assert corridor_width_m(leg, cfg, 0) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# directional_distance_to_point_from_offset_leg
# ---------------------------------------------------------------------------

class TestDirectionalDistanceToPointFromOffsetLeg:
    def test_point_directly_north_compass_0(self, east_leg):
        """Point at (5000, 500) with compass direction N (0°)."""
        # Compass 0 -> math 90 -> drift vector (0, 1), reverse ray goes (0, -1).
        # A point north of the leg maps to itself going back down to the leg.
        d = directional_distance_to_point_from_offset_leg(
            east_leg, direction_deg=0,
            point=Point(5_000, 500),
        )
        assert d == pytest.approx(500.0, abs=1.0)

    def test_point_upstream_returns_none(self, east_leg):
        """Point behind the leg in drift direction -> no intersection."""
        # Drift is east (compass 90). A point west of the leg (x=-100) is
        # behind the leg; the reverse ray goes further east, never hits.
        d = directional_distance_to_point_from_offset_leg(
            east_leg, direction_deg=90,
            point=Point(-100, 0),
        )
        # The reverse-ray nearest-points fallback returns None when the
        # dot product is negative.
        assert d is None

    def test_point_on_leg_returns_zero_or_none(self, east_leg):
        """A point on the leg should be distance ~0 or None."""
        d = directional_distance_to_point_from_offset_leg(
            east_leg, direction_deg=0,
            point=Point(5_000, 0),
        )
        if d is not None:
            assert d == pytest.approx(0.0, abs=1.0)

    def test_empty_point_returns_none(self, east_leg):
        assert directional_distance_to_point_from_offset_leg(
            east_leg, 0, Point()
        ) is None

    def test_none_point_returns_none(self, east_leg):
        assert directional_distance_to_point_from_offset_leg(
            east_leg, 0, None  # type: ignore[arg-type]
        ) is None

    def test_use_leg_offset_shifts_reference(self):
        """With non-zero mean offset and use_leg_offset=True, distance is
        measured from the shifted leg."""
        leg = LegState(
            leg_id='L', line=LineString([(0, 0), (100, 0)]),
            mean_offset_m=10.0, lateral_sigma_m=1.0,
        )
        # Point at y=20. Without offset: distance = 20. With offset=10 left: distance = 10.
        d_no_offset = directional_distance_to_point_from_offset_leg(
            leg, 0, Point(50, 20), use_leg_offset=False,
        )
        d_with_offset = directional_distance_to_point_from_offset_leg(
            leg, 0, Point(50, 20), use_leg_offset=True,
        )
        assert d_no_offset == pytest.approx(20.0, abs=0.5)
        assert d_with_offset == pytest.approx(10.0, abs=0.5)

    def test_multilinestring_intersection_when_ray_is_collinear(self):
        """If the ray is colinear with multiple leg segments, the
        intersection is a MultiLineString.  Exercises the ``"MultiLineString"``
        branch."""
        # Two vertical segments at x=50, separated by a detour.
        leg = LegState(
            leg_id='M',
            line=LineString([(50, 90), (50, 70), (40, 70), (40, 60), (50, 60), (50, 40)]),
            mean_offset_m=0.0, lateral_sigma_m=1.0,
        )
        # Point above the leg on the collinear x=50 axis; ray going south
        # hits both vertical segments along their length.
        d = directional_distance_to_point_from_offset_leg(
            leg, direction_deg=0, point=Point(50, 100),
        )
        # Minimum distance to any crossing = 100 - 90 = 10.
        assert d == pytest.approx(10.0, abs=1.0)

    def test_geometrycollection_intersection(self, monkeypatch):
        """Stub the intersection result to be a GeometryCollection to
        exercise that branch of the type-dispatch."""
        # We can't easily produce a GeometryCollection intersection from
        # shapely's intersection() on two LineStrings, so monkey-patch the
        # intersect result.
        from drifting.engine import (
            directional_distance_to_point_from_offset_leg as fn,
        )
        leg = LegState(
            leg_id='G', line=LineString([(0, 0), (100, 0)]),
            mean_offset_m=0.0, lateral_sigma_m=1.0,
        )
        # Build a GeometryCollection of a Point, a LineString, and a Polygon.
        # The Polygon's ``coords`` access raises -> exercises the except branch.
        gc = GeometryCollection([
            Point(50, 0),
            LineString([(45, 0), (55, 0)]),
            box(20, -5, 30, 5),  # polygon: raises on .coords
        ])
        # LineString.intersection is a method; monkey-patch via a subclass trick
        # is fragile -- instead, monkey-patch the LineString.intersection bound
        # method on the specific start_line instance we'll create.
        import drifting.engine as eng
        original_intersection = LineString.intersection

        def fake_intersection(self, other):
            # Return our GC only on the very first call from the fn.
            return gc

        monkeypatch.setattr(LineString, 'intersection', fake_intersection)
        d = fn(leg, 0, Point(50, 100))
        assert d is not None
        # At least one sub-geom with coords contributed a point.
        assert d >= 0.0

    def test_multi_intersection_via_self_crossing_leg(self):
        """A self-crossing leg produces a MultiPoint intersection with the
        reverse ray.  Exercises the ``gt == "MultiPoint"`` branch."""
        # Self-crossing leg: two ~horizontal segments at different y.
        leg = LegState(
            leg_id='Z',
            line=LineString([(0, -50), (100, -50), (50, -100), (50, 0)]),
            mean_offset_m=0.0, lateral_sigma_m=1.0,
        )
        # Point above the leg; reverse ray from (75, 50) going south crosses
        # the horizontal arm at y=-50 and the diagonal at y=-75.
        # Minimum distance = 100 (to y=-50) from y=50.
        d = directional_distance_to_point_from_offset_leg(
            leg, direction_deg=0, point=Point(75, 50),
        )
        assert d is not None
        assert d == pytest.approx(100.0, abs=1.0)  # nearest crossing at y=-50


# ---------------------------------------------------------------------------
# directional_distance_m (whole-geometry wrapper)
# ---------------------------------------------------------------------------

class TestDirectionalDistanceM:
    def test_point_geom(self, east_leg):
        d = directional_distance_m(east_leg, 0, Point(5_000, 300))
        assert d == pytest.approx(300.0, abs=1.0)

    def test_polygon_returns_minimum_over_vertices(self, east_leg):
        # Box centered at (5000, 500), 200 m wide / 200 m tall.
        poly = box(4_900, 400, 5_100, 600)
        d = directional_distance_m(east_leg, 0, poly)
        # Minimum y-vertex is 400 m above leg.
        assert d == pytest.approx(400.0, abs=1.0)

    def test_linestring_geom(self, east_leg):
        ls = LineString([(100, 50), (200, 150)])
        d = directional_distance_m(east_leg, 0, ls)
        assert d == pytest.approx(50.0, abs=1.0)

    def test_empty_geom_returns_none(self, east_leg):
        assert directional_distance_m(east_leg, 0, Polygon()) is None

    def test_none_geom_returns_none(self, east_leg):
        assert directional_distance_m(east_leg, 0, None) is None  # type: ignore[arg-type]

    def test_upstream_polygon_returns_none(self, east_leg):
        """Polygon entirely south of east-going leg when drifting north --
        all vertices are upstream of the reverse ray."""
        poly = box(100, -100, 200, -50)
        assert directional_distance_m(east_leg, 0, poly) is None

    def test_multipoint_geom(self, east_leg):
        """MultiPoint iterates its sub-points."""
        mp = MultiPoint([(100, 100), (200, 200), (300, 50)])
        d = directional_distance_m(east_leg, 0, mp)
        # Minimum y = 50 -> distance 50.
        assert d == pytest.approx(50.0, abs=1.0)

    def test_multilinestring_geom(self, east_leg):
        """MultiLineString extends coords across sub-lines."""
        mls = MultiLineString([[(0, 100), (10, 100)], [(0, 50), (10, 50)]])
        d = directional_distance_m(east_leg, 0, mls)
        assert d == pytest.approx(50.0, abs=1.0)

    def test_multipolygon_geom(self, east_leg):
        """MultiPolygon iterates sub-polygon exterior coords."""
        mp = MultiPolygon([
            box(100, 200, 200, 300),
            box(100, 500, 200, 600),
        ])
        d = directional_distance_m(east_leg, 0, mp)
        # Minimum y vertex across both boxes = 200.
        assert d == pytest.approx(200.0, abs=1.0)

    def test_geometry_collection_geom(self, east_leg):
        """GeometryCollection: only sub-geoms with .coords contribute.
        (Polygon has no .coords so it's effectively skipped here.)"""
        gc = GeometryCollection([
            Point(100, 80),
            LineString([(200, 120), (210, 140)]),
        ])
        d = directional_distance_m(east_leg, 0, gc)
        # Minimum y: Point at 80, LineString endpoints 120/140 -> 80.
        assert d == pytest.approx(80.0, abs=1.0)

    def test_linear_ring_geom(self, east_leg):
        """LinearRing is handled via the LineString/LinearRing branch."""
        ring = LinearRing([(100, 200), (200, 200), (200, 300), (100, 300)])
        d = directional_distance_m(east_leg, 0, ring)
        assert d == pytest.approx(200.0, abs=1.0)

    def test_empty_point_returns_none(self, east_leg):
        """An empty Point short-circuits at ``target_geom.is_empty``."""
        assert directional_distance_m(east_leg, 0, Point()) is None

    def test_geom_with_no_extractable_coords_returns_none(self, east_leg):
        """A GeometryCollection of sub-polygons (no .coords attr) yields
        an empty ``coords`` list -> returns None."""
        gc = GeometryCollection([box(100, 100, 200, 200)])
        # Polygon has no .coords attribute at the top level -> skipped.
        assert directional_distance_m(east_leg, 0, gc) is None


# ---------------------------------------------------------------------------
# edge_average_distance_m
# ---------------------------------------------------------------------------

class TestEdgeAverageDistanceM:
    def test_two_endpoint_average(self, east_leg):
        edge = ((0, 100), (10, 200))
        # Compass 0 = north drift: endpoints at y=100 and y=200 -> avg = 150.
        d = edge_average_distance_m(east_leg, 0, edge)
        assert d == pytest.approx(150.0, abs=1.0)

    def test_single_valid_endpoint(self, east_leg):
        """One endpoint upstream (returns None), average = the other."""
        edge = ((0, 100), (10, -50))  # second endpoint below leg
        d = edge_average_distance_m(east_leg, 0, edge)
        assert d == pytest.approx(100.0, abs=1.0)

    def test_both_upstream_returns_none(self, east_leg):
        edge = ((0, -50), (10, -100))
        assert edge_average_distance_m(east_leg, 0, edge) is None


# ---------------------------------------------------------------------------
# coverage_percent
# ---------------------------------------------------------------------------

class TestCoveragePercent:
    def test_full_overlap_is_100(self):
        corr = box(0, 0, 100, 100)
        target = box(10, 10, 20, 20)
        assert coverage_percent(corr, target) == pytest.approx(100.0)

    def test_no_overlap_is_zero(self):
        corr = box(0, 0, 10, 10)
        target = box(100, 100, 110, 110)
        assert coverage_percent(corr, target) == 0.0

    def test_partial_overlap(self):
        corr = box(0, 0, 100, 100)
        target = box(90, 90, 110, 110)
        # target area = 400, intersection = 100 -> 25%
        assert coverage_percent(corr, target) == pytest.approx(25.0)

    def test_empty_target_returns_zero(self):
        assert coverage_percent(box(0, 0, 10, 10), Polygon()) == 0.0

    def test_zero_area_target_returns_zero(self):
        # A LineString has area=0.
        assert coverage_percent(box(0, 0, 10, 10), LineString([(0, 0), (5, 5)])) == 0.0


# ---------------------------------------------------------------------------
# edge_hit_percent
# ---------------------------------------------------------------------------

class TestEdgeHitPercent:
    def test_empty_target_returns_zero(self):
        assert edge_hit_percent(box(0, 0, 10, 10), Polygon(), width_m=100.0) == 0.0

    def test_zero_width_returns_zero(self):
        assert edge_hit_percent(box(0, 0, 10, 10), box(1, 1, 2, 2), width_m=0.0) == 0.0

    def test_overlap_length_over_width(self):
        """Box boundary inside corridor: overlap length = full perimeter."""
        # Obstacle square [40,60]x[40,60] fully inside corridor [0,100]x[0,100].
        # Its boundary length = 4*20 = 80.
        # With width_m=80 -> frac = 1.0 -> 100%.
        pct = edge_hit_percent(box(0, 0, 100, 100), box(40, 60, 40, 60), width_m=80.0)
        # Degenerate box (zero size) yields empty boundary -> expect 0 here.
        # Use a real 20x20 obstacle instead.
        pct = edge_hit_percent(box(0, 0, 100, 100), box(40, 40, 60, 60), width_m=80.0)
        assert pct == pytest.approx(100.0, rel=1e-6)

    def test_fraction_clamped_to_100(self):
        """Even for width_m smaller than overlap, result stays <= 100."""
        # 20x20 box boundary is 80 m. width_m=10 would naively give 800%.
        pct = edge_hit_percent(box(0, 0, 100, 100), box(40, 40, 60, 60), width_m=10.0)
        assert pct == pytest.approx(100.0)

    def test_no_overlap_returns_zero(self):
        # Target far outside corridor.
        pct = edge_hit_percent(box(0, 0, 10, 10), box(100, 100, 200, 200), width_m=50.0)
        assert pct == 0.0

    def test_exception_falls_back_to_coverage_percent(self, monkeypatch):
        """If boundary ops throw, the function falls back to
        ``coverage_percent`` using the area overlap."""
        target = box(40, 40, 60, 60)

        # Force target.boundary to raise.
        original_boundary = Polygon.boundary
        def broken_boundary(self):
            raise RuntimeError("synthetic")
        monkeypatch.setattr(Polygon, 'boundary', property(broken_boundary))

        # Corridor fully contains target -> coverage_percent = 100.
        pct = edge_hit_percent(box(0, 0, 100, 100), target, width_m=50.0)
        assert pct == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# evaluate_leg_direction  (full pipeline)
# ---------------------------------------------------------------------------

class TestEvaluateLegDirection:
    def test_all_three_roles_produced(self, east_leg, default_cfg):
        """A corridor that hits a grounding depth, an anchoring depth, and
        a structure returns one TargetHit per role (and direction)."""
        # Depths: shallow (grounding), medium (anchoring only).
        grounding_depth = DepthTarget(
            'g1', depth_m=3.0,
            geometry=box(4_900, -200, 5_100, -100),  # south of leg
        )
        anchoring_depth = DepthTarget(
            'a1', depth_m=15.0,  # >= draught 10, so only anchoring
            geometry=box(4_800, -300, 5_000, -200),
        )
        structure = StructureTarget(
            's1', top_height_m=25.0,
            geometry=box(5_200, -100, 5_400, -50),
        )
        ship = ShipState(draught_m=10.0, anchor_d=2.0)

        hits = evaluate_leg_direction(
            east_leg, ship,
            direction_deg=180,  # south
            depths=[grounding_depth, anchoring_depth],
            structures=[structure],
            cfg=default_cfg,
        )

        roles = {h.role for h in hits}
        assert 'grounding' in roles
        assert 'anchoring' in roles
        assert 'structure' in roles

    def test_no_hits_when_corridor_misses_targets(self, east_leg, default_cfg):
        """Targets far away from the corridor produce no hits."""
        depth = DepthTarget(
            'd1', depth_m=3.0,
            geometry=box(100_000, 100_000, 101_000, 101_000),  # far away
        )
        ship = ShipState(draught_m=10.0, anchor_d=2.0)
        hits = evaluate_leg_direction(
            east_leg, ship, 90, [depth], [], default_cfg,
        )
        assert hits == []

    def test_structure_outside_corridor_is_skipped(self, east_leg, default_cfg):
        """A structure outside the corridor is skipped via
        ``if not corridor.intersects(s.geometry): continue``."""
        # Structure far north (opposite side from south-drift corridor).
        struct = StructureTarget(
            'far', top_height_m=30.0,
            geometry=box(4_900, 100_000, 5_100, 101_000),
        )
        ship = ShipState(draught_m=10.0, anchor_d=2.0)
        hits = evaluate_leg_direction(
            east_leg, ship, 180,  # south drift
            [], [struct], default_cfg,
        )
        assert hits == []

    def test_hits_carry_leg_id_and_direction(self, east_leg, default_cfg):
        depth = DepthTarget(
            'd1', depth_m=3.0,
            geometry=box(4_900, -200, 5_100, -100),
        )
        ship = ShipState(draught_m=10.0, anchor_d=2.0)
        hits = evaluate_leg_direction(
            east_leg, ship, 180, [depth], [], default_cfg,
        )
        assert hits
        for h in hits:
            assert h.leg_id == east_leg.leg_id
            assert h.direction_deg == 180

    def test_structure_height_filtering_applied(self, east_leg, default_cfg):
        """A short ship respecting structure height misses tall structures."""
        tall = StructureTarget(
            's_tall', top_height_m=30.0,
            geometry=box(4_900, -200, 5_100, -100),
        )
        short = StructureTarget(
            's_short', top_height_m=5.0,
            geometry=box(4_900, -200, 5_100, -100),
        )
        ship = ShipState(
            draught_m=10.0, anchor_d=2.0,
            ship_height_m=10.0, respect_structure_height=True,
        )
        hits = evaluate_leg_direction(
            east_leg, ship, 180, [], [tall, short], default_cfg,
        )
        # Only the short structure (<=10m) qualifies.
        assert all(h.target_id == 's_short' for h in hits if h.role == 'structure')

    def test_none_dist_skipped_for_all_roles(self, east_leg, default_cfg, monkeypatch):
        """If ``directional_distance_m`` returns None for a target that
        intersects the corridor, the target is skipped (``if dist is None:
        continue`` branches in evaluate_leg_direction)."""
        import drifting.engine as eng

        # Force directional_distance_m to always return None.
        monkeypatch.setattr(eng, 'directional_distance_m', lambda *a, **k: None)

        # Provide one target of each role, each overlapping the south corridor.
        grounding = DepthTarget('g', 3.0, box(4_900, -200, 5_100, -100))
        anchoring = DepthTarget('a', 15.0, box(4_800, -300, 5_000, -200))
        struct = StructureTarget('s', 25.0, box(5_200, -100, 5_400, -50))
        ship = ShipState(draught_m=10.0, anchor_d=2.0)

        hits = eng.evaluate_leg_direction(
            east_leg, ship, 180,
            [grounding, anchoring], [struct], default_cfg,
        )
        # All three roles tripped the ``if dist is None: continue`` branch.
        assert hits == []
