"""Depth simplification (pure parts): drift reach quantiles, the detail
limit suggestion, the drift corridor zone and the keep / clip / drop rule.
Runs standalone (``--noconftest -p no:qgis``)."""
from __future__ import annotations

import math

import pytest
from scipy import stats
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, Point, box

from compute.drift_reach import REACH_QUANTILE, drift_reach_m, repair_time_quantile_h
from geometries.depth_simplify import (
    DEFAULT_REPAIR_QUANTILE,
    FALLBACK_SPREAD_M,
    NM_M,
    REPAIR_QUANTILES,
    area_km2,
    class_values,
    corridor_length_m,
    corridor_zone,
    depth_value,
    describe,
    drift_lines,
    lateral_spread_m,
    max_traffic_draught,
    simplify_depths,
    suggest_detail_limit,
)

LOGNORMAL = {'use_lognormal': True, 'std': 0.97, 'loc': 0.2, 'scale': 0.5, 'func': ''}


def _box_wkt(x0, y0, x1, y1):
    return box(x0, y0, x1, y1).wkt


# ---------------------------------------------------------------------------
# compute.drift_reach
# ---------------------------------------------------------------------------

class TestDriftReach:
    def test_lognormal_quantile_matches_scipy(self):
        expected = stats.lognorm(0.97, loc=0.2, scale=0.5).ppf(0.9999)
        assert repair_time_quantile_h(LOGNORMAL, 0.9999) == pytest.approx(expected)

    def test_kattegatt_reaches(self):
        """1 kn and the Kattegatt repair time: 5.0 nm at 99 %, 18.6 nm at 99.99 %."""
        drift = {'speed': 1.0, 'repair': LOGNORMAL}
        assert drift_reach_m(drift) / NM_M == pytest.approx(4.97, abs=0.01)
        assert drift_reach_m(drift, 0.9999) / NM_M == pytest.approx(18.62, abs=0.02)

    def test_default_quantile_is_the_corridor_one(self):
        drift = {'speed': 2.0, 'repair': LOGNORMAL}
        assert drift_reach_m(drift) == drift_reach_m(drift, REACH_QUANTILE)
        assert REACH_QUANTILE == 0.99

    def test_reach_scales_with_speed(self):
        one = drift_reach_m({'speed': 1.0, 'repair': LOGNORMAL}, 0.999)
        three = drift_reach_m({'speed': 3.0, 'repair': LOGNORMAL}, 0.999)
        assert three == pytest.approx(3 * one)

    def test_weibull(self):
        rep = {'dist_type': 'weibull', 'wb_shape': 1.5, 'wb_loc': 0.0, 'wb_scale': 2.0}
        assert repair_time_quantile_h(rep, 0.99) == pytest.approx(
            stats.weibull_min(c=1.5, scale=2.0).ppf(0.99))

    def test_user_cdf_by_bisection(self):
        rep = {'func': '1 - exp(-x)'}
        assert repair_time_quantile_h(rep, 0.99) == pytest.approx(-math.log(0.01), rel=1e-6)

    @pytest.mark.parametrize('drift', [{}, {'speed': 0.0, 'repair': LOGNORMAL}, {'speed': 1.0, 'repair': {}}])
    def test_no_reach_without_speed_or_repair_time(self, drift):
        assert drift_reach_m(drift) is None


# ---------------------------------------------------------------------------
# Detail limit
# ---------------------------------------------------------------------------

class TestDetailLimit:
    def test_class_values_sorted_and_distinct(self):
        rows = [['1', '6.0', 'x'], ['2', '0', 'x'], ['3', '6', 'x'], ['4', 'n/a', 'x']]
        assert class_values(rows) == [0.0, 6.0]

    def test_range_text_reads_as_its_lower_bound(self):
        assert depth_value('48-51') == 48.0
        assert depth_value('-3') == -3.0
        assert math.isnan(depth_value('deep'))

    def test_max_traffic_draught_ignores_cells_without_ships(self):
        td = {'1': {'East going': {'Frequency (ships/year)': [[10, 0], [0, 5]],
                                   'Draught (meters)': [[6.0, 20.0], [math.inf, 9.5]]}},
              '2': {'West going': {'Frequency (ships/year)': [[1]], 'Draught (meters)': [[math.nan]]}}}
        assert max_traffic_draught(td) == 9.5
        assert max_traffic_draught({}) is None

    def test_max_traffic_draught_includes_squat(self):
        """Squat at the cell speed (Barras, cargo Cb 0.72): 9.5 m at 10 kn grounds like 10.22 m."""
        td = {'1': {'East going': {'Frequency (ships/year)': [[0], [5]],
                                   'Draught (meters)': [[0.0], [9.5]],
                                   'Speed (knots)': [[0.0], [10.0]]}}}
        names = ['Fishing', 'Cargo, all ships of this type']
        assert max_traffic_draught(td, True, names) == pytest.approx(10.22)
        assert max_traffic_draught(td, False, names) == pytest.approx(9.5)

    def test_suggest_first_class_deeper_than_draught(self):
        classes = [0, 3, 6, 9, 12, 15, 18, 21, 30]
        assert suggest_detail_limit(classes, 16.08) == 18
        assert suggest_detail_limit(classes, 18.0) == 21
        assert suggest_detail_limit(classes, 40.0) == 30
        assert suggest_detail_limit(classes, None) == 30
        assert suggest_detail_limit([], 10.0) is None


# ---------------------------------------------------------------------------
# Drift corridor zone
# ---------------------------------------------------------------------------

LEG = {'Start_Point': '11.000000 56.000000', 'End_Point': '11.100000 56.000000', 'line_length': 6200.0,
       'weight1_1': 100, 'mean1_1': 0.0, 'std1_1': 500.0, 'weight2_1': 100, 'mean2_1': 0.0, 'std2_1': 500.0}
STRUCTURE = box(11.045, 56.030, 11.055, 56.036)        # 3.3 km north of the leg


def _traffic(dir0, dir1):
    return {'East going': {'Frequency (ships/year)': [[dir0]]}, 'West going': {'Frequency (ships/year)': [[dir1]]}}


class TestLateralSpread:
    def test_one_normal_component(self):
        assert lateral_spread_m(LEG, 0) == pytest.approx(2500.0)

    def test_weighted_mixture_with_uniform_part(self):
        seg = {'weight1_1': 1, 'std1_1': 300.0, 'weight1_2': 1, 'std1_2': 400.0,
               'u_p1': 2, 'u_min1': -600.0, 'u_max1': 600.0}
        var = (300.0 ** 2 + 400.0 ** 2 + 2 * 1200.0 ** 2 / 12.0) / 4.0
        assert lateral_spread_m(seg, 0) == pytest.approx(5.0 * math.sqrt(var))

    def test_second_direction_reads_its_own_fields(self):
        seg = dict(LEG, std2_1=200.0)
        assert lateral_spread_m(seg, 1) == pytest.approx(1000.0)

    def test_no_distribution_falls_back_like_the_model(self):
        assert lateral_spread_m({}, 0) == FALLBACK_SPREAD_M
        assert lateral_spread_m({'weight1_1': 0, 'std1_1': 500.0}, 0) == FALLBACK_SPREAD_M


class TestCorridorLength:
    DRIFT = {'speed': 1.0, 'repair': LOGNORMAL}

    def test_model_reach_is_the_99_percent_drift(self):
        assert corridor_length_m(self.DRIFT, {'1': LEG}) == pytest.approx(drift_reach_m(self.DRIFT))

    def test_model_reach_is_capped_at_ten_legs(self):
        short = dict(LEG, line_length=500.0)
        assert corridor_length_m(self.DRIFT, {'1': short}) == pytest.approx(5000.0)

    def test_longer_quantile_adds_margin(self):
        short = dict(LEG, line_length=500.0)
        assert corridor_length_m(self.DRIFT, {'1': short}, 0.9999) == pytest.approx(
            drift_reach_m(self.DRIFT, 0.9999))

    def test_without_repair_time_the_cap_is_used(self):
        assert corridor_length_m({'speed': 1.0}, {'1': LEG}) == pytest.approx(62_000.0)
        assert corridor_length_m({}, {}) is None


class TestDriftLines:
    def test_directions_with_ships_only(self):
        lines = drift_lines({'1': LEG, '2': dict(LEG)}, {'1': _traffic(10, 0), '2': _traffic(0, 0)})
        assert [(seg, k) for seg, k, _, _ in lines] == [('1', 0)]

    def test_second_direction_runs_end_to_start(self):
        (_, k, line, spread), = drift_lines({'1': LEG}, {'1': _traffic(0, 5)})
        assert k == 1 and list(line.coords) == [(11.1, 56.0), (11.0, 56.0)] and spread == pytest.approx(2500.0)


class TestCorridorZone:
    LINES = [('1', 0, LineString([(11.0, 56.0), (11.1, 56.0)]), 2500.0)]

    def test_corridors_drifting_onto_the_structure(self):
        zone, hits = corridor_zone(self.LINES, [STRUCTURE.wkt], 9000.0)
        compass = {h['compass'] for h in hits}
        assert 0 in compass and 180 not in compass and 90 not in compass
        assert zone.contains(STRUCTURE.centroid)
        assert zone.contains(Point(11.05, 56.015))           # between the leg and the structure

    def test_corridor_cut_just_past_the_structure(self):
        zone, _ = corridor_zone(self.LINES, [STRUCTURE.wkt], 9000.0)
        assert zone.bounds[3] < 56.08                          # 12 km north without the cut

    def test_rose_without_northward_drift(self):
        rose = {str(c): (0.0 if c == 0 else 0.125) for c in range(0, 360, 45)}
        _, hits = corridor_zone(self.LINES, [STRUCTURE.wkt], 9000.0, rose)
        assert 0 not in {h['compass'] for h in hits}

    def test_structure_out_of_reach(self):
        """The corridor spans the lateral spread too: 1 km of it plus 1 km of drift stays short of 3.3 km."""
        narrow = [('1', 0, LineString([(11.0, 56.0), (11.1, 56.0)]), 1000.0)]
        assert corridor_zone(narrow, [STRUCTURE.wkt], 1000.0) == (None, [])
        assert corridor_zone(narrow, [STRUCTURE.wkt], 2500.0)[0] is not None

    def test_nothing_without_structures_or_legs(self):
        assert corridor_zone(self.LINES, [], 9000.0) == (None, [])
        assert corridor_zone([], [STRUCTURE.wkt], 9000.0) == (None, [])
        assert corridor_zone(self.LINES, ['not wkt'], 9000.0) == (None, [])

    def test_area_km2(self):
        assert area_km2(box(11.0, 56.0, 11.1, 56.1)) == pytest.approx(6.22 * 11.12, rel=0.01)
        assert area_km2(None) == 0.0


# ---------------------------------------------------------------------------
# simplify_depths
# ---------------------------------------------------------------------------

class TestSimplifyDepths:
    ROWS = [
        ['1', '0', _box_wkt(0, 0, 1, 1)],
        ['2', '6', _box_wkt(1, 0, 2, 1)],
        ['3', '9', _box_wkt(2, 0, 3, 1)],     # deeper, half inside the zone
        ['4', '12', _box_wkt(5, 0, 6, 1)],    # deeper, outside the zone
    ]
    ZONE = box(2.5, -1, 4, 2)

    def test_keep_clip_drop(self):
        rows, summary = simplify_depths(self.ROWS, 6.0, self.ZONE)
        assert [r[:2] for r in rows] == [['1', '0'], ['2', '6'], ['3', '9']]
        assert rows[0][2] == self.ROWS[0][2]
        assert shapely_wkt.loads(rows[2][2]).equals(box(2.5, 0, 3, 1))
        assert (summary['kept'], summary['clipped'], summary['dropped']) == (2, 1, 1)
        assert summary['classes_before'] == [0, 6, 9, 12]
        assert summary['classes_after'] == [0, 6, 9]
        assert summary['vertices_before'] == 20 and summary['vertices_after'] == 15

    def test_without_zone_everything_deeper_goes(self):
        rows, summary = simplify_depths(self.ROWS, 6.0, None)
        assert [r[0] for r in rows] == ['1', '2']
        assert summary['dropped'] == 2

    def test_limit_at_deepest_class_changes_nothing(self):
        rows, summary = simplify_depths(self.ROWS, 12.0, None)
        assert rows == [[str(v) for v in r] for r in self.ROWS]
        assert summary['clipped'] == summary['dropped'] == 0

    def test_unparsable_depth_is_kept(self):
        rows, _ = simplify_depths([['1', 'n/a', _box_wkt(0, 0, 1, 1)]], 0.0, None)
        assert len(rows) == 1

    def test_touching_zone_gives_no_sliver(self):
        """A deep polygon that only touches the zone along an edge is dropped, not kept as a line."""
        rows, summary = simplify_depths([['1', '30', _box_wkt(0, 0, 1, 1)]], 6.0, box(1, 0, 2, 1))
        assert rows == [] and summary['dropped'] == 1

    def test_describe(self):
        _, summary = simplify_depths(self.ROWS, 6.0, self.ZONE)
        text = describe(summary)
        assert 'Depth classes 4 -> 3' in text and 'vertices 20 -> 15' in text


def test_quantile_constants():
    """The model reach (99 %) is the default; the longer ones are margin."""
    assert REPAIR_QUANTILES[0] == DEFAULT_REPAIR_QUANTILE == REACH_QUANTILE
    assert NM_M == 1852.0


class TestDepthValue:
    """One rule for a depth class: the number, or an interval's shallower end."""

    @pytest.mark.parametrize('text, expected', [
        ('10', 10.0), (7.5, 7.5), ('-5', -5.0), ('0-10', 0.0), ('48-51', 48.0),
        ('10 - 20', 10.0), ('-10--5', -10.0), ('20-10', 10.0),
    ])
    def test_values(self, text, expected):
        from geometries.depth_simplify import depth_value
        assert depth_value(text) == expected

    @pytest.mark.parametrize('text', ['', 'deep', 'a-b-7.5', None])
    def test_unreadable_is_nan(self, text):
        import math
        from geometries.depth_simplify import depth_value
        assert math.isnan(depth_value(text))
