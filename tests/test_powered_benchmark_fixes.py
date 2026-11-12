"""Powered-model fixes from the Kattegat benchmark (KNOWN_ISSUES 21, 25, 26, 27).

* P1: the projection is centred on the model, so results do not depend on
  which leg is first, and distances are right within 0.05 %.
* P2: Cat II is summed ray by ray, so it does not depend on how the water
  is cut into polygons.
* D: the rays cover the whole lateral mixture with its weights.
* I4: two polygons never share an id in the caster.

Runs with the QGIS conftest (the caster module imports matplotlib's Qt
backend; the allision run builds a result layer).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.get_powered_overlap import (  # noqa: E402
    ModelProjector, SimpleProjector, _build_leg_entry, _compute_cat1_in_lane, _compute_cat2_with_shadows,
    _unique_ids, cat2_probability, model_projector,
)

ORIGIN = np.array([0.0, 0.0])
ALONG = np.array([1.0, 0.0])
PERP = np.array([0.0, 1.0])


# ---------------------------------------------------------------------------
# P1  projection
# ---------------------------------------------------------------------------

def _geodesic_m(lon1, lat1, lon2, lat2) -> float:
    from pyproj import Geod
    return Geod(ellps='WGS84').inv(lon1, lat1, lon2, lat2)[2]


def _proj_dist(proj, lon1, lat1, lon2, lat2) -> float:
    x1, y1 = proj.transform(lon1, lat1)
    x2, y2 = proj.transform(lon2, lat2)
    return float(np.hypot(x2 - x1, y2 - y1))


class TestProjection:
    # Kattegat: model centre ~ 11.6 E / 57.1 N, the farm at ~ 11.9 E / 56.6 N,
    # the first leg in the original route table at Skagen (10.78 E / 57.83 N).
    CENTRE = (11.6, 57.1)
    FARM = (11.9, 56.6)

    @pytest.mark.parametrize('dlon, dlat', [(0.0166, 0.0), (0.0, 0.009), (0.012, 0.006)])
    def test_model_projector_distances_within_0_05_percent(self, dlon, dlat):
        proj = ModelProjector(*self.CENTRE)
        lon, lat = self.FARM
        true = _geodesic_m(lon, lat, lon + dlon, lat + dlat)
        assert _proj_dist(proj, lon, lat, lon + dlon, lat + dlat) == pytest.approx(true, rel=5e-4)

    def test_the_old_first_leg_projection_was_3_percent_short(self):
        lon, lat = self.FARM
        true = _geodesic_m(lon, lat, lon + 0.0166, lat)
        old = _proj_dist(SimpleProjector(10.78, 57.83), lon, lat, lon + 0.0166, lat)
        assert old / true == pytest.approx(0.967, abs=0.004)

    def test_model_projector_is_centred_on_the_legs(self):
        segs = {'a': {'Start_Point': '10.0 56.0', 'End_Point': '11.0 56.5'},
                'b': {'Start_Point': '12.0 58.0', 'End_Point': '11.5 57.0'},
                'c': {}}
        proj = model_projector(segs)
        assert (proj.lon_ref, proj.lat_ref) == (11.0, 57.0)
        with pytest.raises(ValueError):
            model_projector({})

    def test_arrays_and_scalars(self):
        proj = ModelProjector(11.0, 57.0)
        xs, ys = proj.transform(np.array([11.0, 11.1]), np.array([57.0, 57.0]))
        x, y = proj.transform(11.1, 57.0)
        assert xs[0] == pytest.approx(0.0) and ys[0] == pytest.approx(0.0)
        assert (xs[1], ys[1]) == (pytest.approx(x), pytest.approx(y))

    def test_leg_order_does_not_change_the_result(self):
        """Before the fix the first leg's start was the projection origin."""
        sys.path.insert(0, str(ROOT / 'tests'))
        import test_powered_model as tpm
        data = tpm._minimal_powered_data(include_depths=False)
        far = copy.deepcopy(data['segment_data']['1'])
        far.update({'Start_Point': '13.0 56.4', 'End_Point': '13.2 56.4', 'Leg_name': 'far', 'Segment_Id': '2'})
        first = dict(data, segment_data={'2': far, '1': data['segment_data']['1']})
        second = dict(data, segment_data={'1': data['segment_data']['1'], '2': far})
        totals = []
        for d in (first, second):
            host = tpm._make_mixin_host('LEPPoweredAllision')
            totals.append(host.run_powered_allision_model(d))
        assert totals[0] > 0
        assert totals[0] == pytest.approx(totals[1], rel=1e-12)


# ---------------------------------------------------------------------------
# P2  Cat II per ray
# ---------------------------------------------------------------------------

SLANTED = Polygon([(50, -60), (300, -60), (300, 60), (250, 60)])    # west face: ray distance grows with offset


def _cat2_total(obstacles, recovery=150.0):
    summaries, *_ = _compute_cat2_with_shadows(ORIGIN, ALONG, PERP, 0.0, 30.0, 30.0, recovery / 30.0, obstacles)
    return sum(cat2_probability(s, recovery) for s in summaries.values()), summaries


class TestCat2PerRay:
    def test_sum_of_rays_not_mean_distance(self):
        s = {'mass': 0.5, 'mean_dist': 1.6, 'ray_masses': [0.25, 0.25], 'ray_dists': [0.2, 3.0]}
        assert cat2_probability(s, 1.0) == pytest.approx(0.25 * (np.exp(-0.2) + np.exp(-3.0)))
        assert cat2_probability(s, 1.0) > 0.5 * np.exp(-1.6)           # Jensen: the old form was lower
        assert cat2_probability({'mass': 0.5, 'mean_dist': 1.6}, 1.0) == pytest.approx(0.5 * np.exp(-1.6))
        assert cat2_probability(s, 0.0) == 0.0

    def test_splitting_an_obstacle_does_not_change_cat2(self):
        whole, _ = _cat2_total([({'id': 'w', 'geom': SLANTED}, 'depth')])
        south = SLANTED.intersection(box(-1e4, -1e4, 1e4, 0.0))
        north = SLANTED.intersection(box(-1e4, 0.0, 1e4, 1e4))
        split, summaries = _cat2_total([({'id': 's', 'geom': south}, 'depth'), ({'id': 'n', 'geom': north}, 'depth')])
        assert len(summaries) == 2
        assert split == pytest.approx(whole, rel=1e-9)

    def test_edge_lists_follow_the_rays(self):
        _, summaries = _cat2_total([({'id': 'w', 'geom': SLANTED}, 'depth')])
        s = next(iter(summaries.values()))
        assert len(s['ray_masses']) == len(s['ray_dists']) == s['n_rays']
        edges = s['edges']
        assert sum(len(e['ray_masses']) for e in edges.values()) == s['n_rays']
        assert sum(cat2_probability(e, 150.0) for e in edges.values()) == pytest.approx(
            cat2_probability(s, 150.0), rel=1e-12)


# ---------------------------------------------------------------------------
# D  whole lateral mixture
# ---------------------------------------------------------------------------

LEG_1_6_DIR1 = [('n', -300.0, 350.0, 0.8), ('n', -2100.0, 350.0, 0.2)]
SHOAL = box(2000.0, -2600.0, 3000.0, -1900.0)     # under the second lane, inside the leg


class TestWholeMixture:
    def test_shoal_under_the_second_lane_gets_its_ships(self):
        obstacles = [({'id': 'shoal', 'geom': SHOAL}, 'depth')]
        with_mix, *_ = _compute_cat1_in_lane(ORIGIN, ALONG, PERP, -300.0, 350.0, 10_000.0, obstacles,
                                             components=LEG_1_6_DIR1)
        first_only, *_ = _compute_cat1_in_lane(ORIGIN, ALONG, PERP, -300.0, 350.0, 10_000.0, obstacles)
        assert with_mix[('depth', 'shoal')]['mass'] == pytest.approx(0.1279, abs=2e-3)
        assert sum(s['mass'] for s in first_only.values()) < 1e-5

    def test_leg_entry_carries_the_mixture(self):
        seg = {'Start_Point': '11.0 56.0', 'End_Point': '11.1 56.0', 'Dirs': ['East going', 'West going'],
               'mean1_1': -300.0, 'std1_1': 350.0, 'weight1_1': 80.0,
               'mean1_2': -2100.0, 'std1_2': 350.0, 'weight1_2': 20.0,
               'mean2_1': 300.0, 'std2_1': 200.0, 'weight2_1': 0.0,
               'u_min1': 0.0, 'u_max1': 0.0, 'u_p1': 0}
        entry = _build_leg_entry('1', seg, {}, ModelProjector(11.05, 56.0))
        comps = entry['dirs'][0]['components']
        assert [(k, a, b) for k, a, b, _w in comps] == [('n', -300.0, 350.0), ('n', -2100.0, 350.0)]
        assert [w for *_x, w in comps] == pytest.approx([0.8, 0.2])
        # No weights at all: one normal from mean/std_1, as before.
        assert entry['dirs'][1]['components'] == [('n', 300.0, 200.0, 1.0)]


# ---------------------------------------------------------------------------
# I4  unique ids in the caster
# ---------------------------------------------------------------------------

class TestUniqueIds:
    def test_repeated_ids_get_a_suffix(self):
        obs = _unique_ids([{'id': '3'}, {'id': '3'}, {'id': '4'}, {'id': '3'}])
        assert [o['id'] for o in obs] == ['3', '3#2', '4', '3#3']

    def test_two_polygons_with_one_id_stay_two_obstacles(self):
        from geometries.get_powered_overlap import _build_legs_and_obstacles
        near = 'POLYGON((11.12 55.995, 11.13 55.995, 11.13 56.005, 11.12 56.005, 11.12 55.995))'
        far = 'POLYGON((11.20 55.995, 11.21 55.995, 11.21 56.005, 11.20 56.005, 11.20 55.995))'
        data = {'segment_data': {'1': {'Start_Point': '11.0 56.0', 'End_Point': '11.1 56.0'}},
                'depths': [['3', '5', near], ['3', '5', far]]}
        _legs, obstacles, *_ = _build_legs_and_obstacles(data, ModelProjector(11.1, 56.0), 'grounding', 10.0)
        assert sorted(o['id'] for o, _k in obstacles) == ['3', '3#2']
