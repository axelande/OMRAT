"""Drift settings that reach the model since 2026-11-10, and the direction-2 lateral side.

* **Start drifting from** (``compute/drift_start.py``): ``distribution_center``
  measures the drift distance from the leg offset by the mean of the
  direction's lateral distribution.
* **Squat** (``compute/squat.py``, two Causation factors tick boxes): the
  grounding draught includes the squat at the ship's speed, in drifting and in
  powered grounding; anchoring keeps the static draught.
* The drift model draws direction 2 as the reversed line; its lateral
  distribution is mirrored onto that line's own axis (it sailed on the wrong
  side of the leg before).

The ``TestEndToEnd`` class runs the drifting model and needs the QGIS
conftest (``qgis_app``); everything else runs standalone.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from scipy.stats import norm, uniform
from shapely.geometry import LineString, box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.data_preparation import clean_traffic, get_distribution  # noqa: E402
from compute.drift_start import mixture_mean, uses_distribution_center  # noqa: E402
from compute.squat import cb_for_type_name, cell_squat_m, squat_on  # noqa: E402
from geometries.analytical_probability import compute_probability_holes_analytical  # noqa: E402
from geometries.drift_overlap_geometry import (  # noqa: E402
    compute_min_distance_by_object, create_polygon_from_line,
)

NORTH, SOUTH = 2, 6   # drift-direction index in the hole tables (math angle / 45)


def _seg(**over) -> dict:
    seg = {
        'Start_Point': '0 0', 'End_Point': '10000 0', 'Dirs': ['East going', 'West going'],
        'mean1_1': 0.0, 'std1_1': 300.0, 'weight1_1': 1.0,
        'mean2_1': 0.0, 'std2_1': 300.0, 'weight2_1': 1.0,
        'u_p1': 0.0, 'u_min1': 0.0, 'u_max1': 0.0,
        'u_p2': 0.0, 'u_min2': 0.0, 'u_max2': 0.0,
    }
    seg.update(over)
    return seg


def _cells(draught=11.5, speed=10.0) -> dict:
    grid = [[0.0, 0.0] for _ in range(3)]
    freq, spd, dr, h = (copy.deepcopy(grid) for _ in range(4))
    freq[2][1], spd[2][1], dr[2][1], h[2][1] = 100.0, speed, draught, 20.0
    return {'Frequency (ships/year)': freq, 'Speed (knots)': spd, 'Draught (meters)': dr,
            'Ship heights (meters)': h}


def _data(drift: dict, pc: dict | None = None, **seg_over) -> dict:
    return {
        'pc': pc if pc is not None else {},
        'segment_data': {'1': _seg(**seg_over)},
        'traffic_data': {'1': {'East going': _cells(), 'West going': _cells()}},
        'ship_categories': {'types': ['Fishing', 'Passenger, all ships of this type',
                                      'Cargo, all ships of this type']},
        'drift': drift,
    }


# ---------------------------------------------------------------------------
# Direction 2 on the right side of the leg
# ---------------------------------------------------------------------------

class TestDirectionTwoSide:
    def test_mirror_negates_means_and_flips_the_uniform_part(self):
        seg = _seg(mean2_1=250.0, u_p2=0.1, u_min2=-100.0, u_max2=400.0)
        plain, _ = get_distribution(seg, 1)
        mirrored, w = get_distribution(seg, 1, mirror=True)
        assert plain[0].mean() == pytest.approx(250.0)
        assert mirrored[0].mean() == pytest.approx(-250.0)
        assert mirrored[0].std() == pytest.approx(300.0)
        assert mirrored[-1].support() == pytest.approx((-400.0, 100.0))
        assert w[-1] == pytest.approx(0.1)

    def test_clean_traffic_mirrors_direction_two_only(self):
        out = clean_traffic(_data({}, mean1_1=500.0, mean2_1=500.0))
        (_, d1, _, _, _), (line2, d2, _, _, _) = out
        assert d1[0].mean() == pytest.approx(500.0)
        assert d2[0].mean() == pytest.approx(-500.0)
        assert list(line2.coords)[0] == (10000.0, 0.0)   # drawn End -> Start

    def test_both_directions_hit_the_same_side(self):
        """Ships 800 m north of an eastward leg reach a box 1-2 km north -- in both directions."""
        obj = gpd.GeoDataFrame(geometry=[box(4000, 1000, 6000, 2000)])
        out = clean_traffic(_data({}, mean1_1=800.0, mean2_1=800.0))
        lines = [o[0] for o in out]
        holes = compute_probability_holes_analytical(
            lines, [o[1] for o in out], [o[2] for o in out], [obj], distance=900)
        assert holes[0][NORTH][0] > 0.15
        assert holes[1][NORTH][0] == pytest.approx(holes[0][NORTH][0], rel=1e-6)
        assert holes[1][SOUTH][0] == pytest.approx(holes[0][SOUTH][0], rel=1e-6, abs=1e-12)

    def test_coverage_polygon_moves_perpendicular_to_the_leg(self):
        """A north-drawn leg with mean +50: the polygon moves 50 m west (left), not north."""
        poly = create_polygon_from_line(LineString([(0, 0), (0, 100)]), [norm(loc=50, scale=5)], [1.0])
        assert poly.centroid.x == pytest.approx(-50.0, abs=1.0)
        assert poly.centroid.y == pytest.approx(50.0, abs=1.0)


# ---------------------------------------------------------------------------
# Start drifting from
# ---------------------------------------------------------------------------

class TestStartFrom:
    def test_setting(self):
        assert uses_distribution_center({'start_from': 'distribution_center'})
        assert not uses_distribution_center({'start_from': 'leg_center'})
        assert not uses_distribution_center({})
        assert not uses_distribution_center(None)

    def test_mixture_mean_skips_zero_weights(self):
        dists = [norm(loc=100, scale=10), norm(loc=0, scale=1), uniform(loc=200, scale=200)]
        assert mixture_mean(dists, [3.0, 0.0, 1.0]) == pytest.approx((3 * 100 + 300) / 4)
        assert mixture_mean([norm(0, 1)], [0.0]) == 0.0

    def test_distance_from_the_offset_line(self):
        line = LineString([(0, 0), (10000, 0)])
        obj = gpd.GeoDataFrame(geometry=[box(4000, 2000, 6000, 3000)])
        dists, wgts = [norm(loc=500, scale=300)], [1.0]
        plain = compute_min_distance_by_object([line], [dists], [wgts], [obj], distance=5000)
        shifted = compute_min_distance_by_object(
            [line], [dists], [wgts], [obj], distance=5000,
            distance_lines=[LineString([(0, 500), (10000, 500)])])
        assert plain[0][NORTH][0] == pytest.approx(2000.0)
        assert shifted[0][NORTH][0] == pytest.approx(1500.0)

    def test_object_behind_the_offset_line_keeps_its_leg_distance(self):
        line = LineString([(0, 0), (10000, 0)])
        obj = gpd.GeoDataFrame(geometry=[box(4000, 200, 6000, 400)])
        dists, wgts = [norm(loc=800, scale=300)], [1.0]
        shifted = compute_min_distance_by_object(
            [line], [dists], [wgts], [obj], distance=5000,
            distance_lines=[LineString([(0, 800), (10000, 800)])])
        assert shifted[0][NORTH][0] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Squat
# ---------------------------------------------------------------------------

class TestSquat:
    def test_tick_boxes_default_on(self):
        assert squat_on({}, 'squat_drifting') and squat_on(None, 'squat_powered')
        assert not squat_on({'squat_drifting': False}, 'squat_drifting')
        assert not squat_on({'squat_powered': 'false'}, 'squat_powered')
        assert squat_on({'squat_powered': 'true'}, 'squat_powered')

    def test_block_coefficient_by_name(self):
        assert cb_for_type_name('Tanker, all ships of this type') == 0.82
        assert cb_for_type_name('Chemical / gas tanker') == 0.67
        assert cb_for_type_name('Cargo, all ships of this type') == 0.72
        assert cb_for_type_name('Passenger cruise') == 0.70
        assert cb_for_type_name('High speed craft (HSC), all ships of this type') == 0.65
        assert cb_for_type_name('Spare') == 0.75
        assert cb_for_type_name(None, 19) == 0.82   # OMRAT's own taxonomy by index

    def test_barras_at_the_ship_speed(self):
        assert cell_squat_m(10.0, 0.72) == pytest.approx(0.72)
        assert cell_squat_m(2.0, 0.72) == pytest.approx(0.0288)
        assert cell_squat_m(float('inf'), 0.72) == 0.0
        assert cell_squat_m(0.0, 0.72) == 0.0

    def test_clean_traffic_adds_squat_to_the_grounding_draught_only(self):
        cell = clean_traffic(_data({}))[0][3][0]                 # tick box default on
        assert cell['draught'] == pytest.approx(11.5 + 0.72)    # cargo Cb 0.72 at 10 kn
        assert cell['anchor_draught'] == pytest.approx(11.5)
        cell = clean_traffic(_data({}, {'squat_drifting': False}))[0][3][0]
        assert cell['draught'] == pytest.approx(11.5)

    def test_anchoring_uses_the_static_draught(self):
        """7 x 11.5 m = 80.5 m: an 82 m area is not anchorable, although 7 x (11.5 + squat) is deeper."""
        from compute.drifting_model import DriftingModelMixin

        host = DriftingModelMixin.__new__(DriftingModelMixin)
        depths = [{'depth': 12.0}, {'depth': 82.0}]
        obs = host._collect_cell_obstacles(
            0, 0, 12.22, 7.0, [], depths, [], [[[100.0, 200.0]] * 8], [], [[[0.5, 0.5]] * 8], None,
            anchor_draught=11.5)
        kinds = {(o[0], o[1]) for o in obs}
        assert ('grounding', 0) in kinds          # 12 m < 12.22 m with squat
        assert ('anchoring', 0) in kinds
        assert ('anchoring', 1) not in kinds


# ---------------------------------------------------------------------------
# Whole drifting model (QGIS conftest)
# ---------------------------------------------------------------------------

def _model_project(drift_over: dict, traffic_dirs=('East going', 'West going'), **seg_over) -> dict:
    from tests.test_cascade_minimal import _build_minimal_project
    from compute.basic_equations import default_blackout_by_ship_type

    data = _build_minimal_project()
    data['drift'].update(drift_over)
    data['drift']['blackout_by_ship_type'] = default_blackout_by_ship_type()
    data['segment_data']['1'].update(seg_over)
    for name in list(data['traffic_data']['1']):
        if name not in traffic_dirs:
            blk = data['traffic_data']['1'][name]
            blk['Frequency (ships/year)'] = [[0.0 for _ in row] for row in blk['Frequency (ships/year)']]
    return data


def _run(data: dict) -> tuple[float, float]:
    from tests.test_cascade_minimal import _mock_parent
    from compute.run_calculations import Calculation

    calc = Calculation(_mock_parent())
    calc.set_progress_callback(lambda c, t, m: True)
    return calc.run_drifting_model(copy.deepcopy(data))


class TestEndToEnd:
    def test_direction_two_matches_direction_one_on_the_same_side(self, qgis_app):
        a1, g1 = _run(_model_project({}, ('East going',), mean1_1=800.0))
        a2, g2 = _run(_model_project({}, ('West going',), mean2_1=800.0))
        assert a1 > 0
        assert a2 == pytest.approx(a1, rel=1e-6)
        assert g2 == pytest.approx(g1, rel=1e-6)

    def test_distribution_center_starts_closer_to_the_structure(self, qgis_app):
        """Both lanes 800 m north, the structure ~0.9-1.3 km north: shorter drift, more allision."""
        seg = {'mean1_1': 800.0, 'mean2_1': 800.0}
        a_leg, _ = _run(_model_project({'start_from': 'leg_center'}, **seg))
        a_dist, _ = _run(_model_project({'start_from': 'distribution_center'}, **seg))
        assert a_dist > a_leg * 1.05

    def test_drift_squat_grounds_on_a_deeper_class(self, qgis_app):
        """Draught 11.5 m, depth 12 m: no grounding without squat; 10 kn cargo squat 0.72 m reaches it."""
        def _project(on):
            data = _model_project({})
            data['pc']['squat_drifting'] = on
            for blk in data['traffic_data']['1'].values():
                blk['Draught (meters)'][18][1] = 11.5
            return data
        _, g_off = _run(_project(False))
        _, g_on = _run(_project(True))
        assert g_off == 0.0
        assert g_on > 0.0

    def test_powered_grounding_draughts_include_squat(self, qgis_app):
        from compute.powered_model import PoweredModelMixin

        host = PoweredModelMixin.__new__(PoweredModelMixin)
        td = {'1': {'East going': _cells(), 'West going': _cells()}}
        names = ['Fishing', 'Passenger, all ships of this type', 'Cargo, all ships of this type']
        assert host._collect_draught_set(td) == {11.5}
        assert sorted(host._collect_draught_set(td, names)) == pytest.approx([11.5 + 0.72])
