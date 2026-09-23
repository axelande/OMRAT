"""Per-edge split of powered grounding / allision probabilities.

Before this change the powered result layer spread each obstacle's total
over its boundary edges with a cosine heuristic: every similarly oriented
edge got the same value, the per-leg columns were identical on all edges,
and the edges of one obstacle summed to many times the obstacle total
(about 170x for obstacle 3_51 of the Kattegat comparison run).  The ray
caster now records the edge each ray hits and the model emits
``by_obstacle_segment_legdir``; these tests pin that chain end to end.

Needs the QGIS conftest (the model's finalize step builds QgsVectorLayers).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from geometries.boundary_edges import boundary_edges  # noqa: E402
from geometries.get_powered_overlap import (  # noqa: E402
    _build_hit_and_edge_matrices, _compute_cat1_in_lane, _compute_cat2_with_shadows,
)
from compute.powered_model import _edge_shares, _iter_hit_probs  # noqa: E402
import test_powered_model as _tpm  # noqa: E402

ORIGIN = np.array([0.0, 0.0])
ALONG = np.array([1.0, 0.0])
PERP = np.array([0.0, 1.0])


def _west_face_index(geom) -> int:
    """Index of the edge facing the rays (both x equal to the min x)."""
    edges = boundary_edges(geom)
    xmin = min(min(e[0], e[2]) for e in edges)
    hits = [i for i, (x1, _, x2, _) in enumerate(edges) if x1 == xmin and x2 == xmin]
    assert len(hits) == 1
    return hits[0]


# ---------------------------------------------------------------------------
# Ray caster: which edge did each ray hit?
# ---------------------------------------------------------------------------

class TestEdgeMatrix:
    def test_box_ahead_all_rays_hit_the_front_face(self):
        geom = box(50, -50, 100, 50)
        offsets = np.linspace(-80, 80, 17)
        hit, edge = _build_hit_and_edge_matrices(
            offsets, [({'id': 'b', 'geom': geom}, 'depth')], ORIGIN, ALONG, PERP)
        # The caster's crossing test is half-open (y_min <= y < y_max).
        inside = (offsets >= -50) & (offsets < 50)
        assert np.all(edge[inside, 0] == _west_face_index(geom))
        assert np.all(edge[~inside, 0] == -1)
        assert np.all(np.isinf(hit[~inside, 0]))
        assert np.allclose(hit[inside, 0], 50.0)

    def test_edge_masses_sum_to_obstacle_mass_cat2(self):
        geom = box(50, -50, 100, 50)
        summaries, _, _, _ = _compute_cat2_with_shadows(
            ORIGIN, ALONG, PERP, 0.0, 30.0, 180.0, 5.0,
            [({'id': 'b', 'geom': geom}, 'depth')])
        s = summaries[('depth', 'b')]
        assert s['edges']
        assert sum(e['mass'] for e in s['edges'].values()) == pytest.approx(s['mass'])
        assert sum(e['n_rays'] for e in s['edges'].values()) == s['n_rays']
        assert set(s['edges']) == {_west_face_index(geom)}

    def test_edge_masses_sum_to_obstacle_mass_cat1(self):
        geom = box(50, -50, 100, 50)
        summaries, _, _, _ = _compute_cat1_in_lane(
            ORIGIN, ALONG, PERP, 0.0, 30.0, 500.0,
            [({'id': 'b', 'geom': geom}, 'depth')])
        s = summaries[('depth', 'b')]
        assert sum(e['mass'] for e in s['edges'].values()) == pytest.approx(s['mass'])
        assert set(s['edges']) == {_west_face_index(geom)}

    def test_stepped_obstacle_splits_over_its_front_faces(self):
        """Three front faces at different distances get three different masses.

        This is the shape of the demo used to explain the fix: a staircase
        contour whose back face and along-ray side faces can never be hit.
        """
        stair = Polygon([(1500, -600), (3500, -600), (3500, 900), (2500, 900),
                         (2500, 300), (2000, 300), (2000, -200), (1500, -200)])
        edges = boundary_edges(stair)
        summaries, _, _, _ = _compute_cat2_with_shadows(
            ORIGIN, ALONG, PERP, 0.0, 300.0, 180.0, 6.0,
            [({'id': 'stair', 'geom': stair}, 'depth')])
        s = summaries[('depth', 'stair')]
        front = {i for i, (x1, _, x2, _) in enumerate(edges) if x1 == x2 and x1 < 3500}
        assert set(s['edges']) == front
        masses = [s['edges'][i]['mass'] for i in sorted(front)]
        assert len(set(round(m, 12) for m in masses)) == 3      # all distinct
        assert sum(masses) == pytest.approx(s['mass'])
        # The closer face has the shorter mean distance.
        by_x = sorted(front, key=lambda i: edges[i][0])
        dists = [s['edges'][i]['mean_dist'] for i in by_x]
        assert dists == sorted(dists)


# ---------------------------------------------------------------------------
# Model: shares and the report block
# ---------------------------------------------------------------------------

class TestEdgeShares:
    def test_shares_sum_to_one_and_weight_by_distance(self):
        edges = {0: {'mass': 0.5, 'mean_dist': 1000.0}, 1: {'mass': 0.5, 'mean_dist': 3000.0}}
        cat2 = _edge_shares(edges, recovery=1000.0)
        cat1 = _edge_shares(edges, recovery=None)
        assert sum(cat2.values()) == pytest.approx(1.0)
        assert cat1 == {0: pytest.approx(0.5), 1: pytest.approx(0.5)}
        assert cat2[0] > cat2[1]          # the near edge keeps more under Cat II decay

    def test_empty_or_zero_mass_gives_empty(self):
        assert _edge_shares(None, 100.0) == {}
        assert _edge_shares({0: {'mass': 0.0, 'mean_dist': 10.0}}, None) == {}

    def test_iter_hit_probs_yields_shares(self):
        geom = box(50, -50, 100, 50)
        summaries, ray_data, offsets, pdf = _compute_cat2_with_shadows(
            ORIGIN, ALONG, PERP, 0.0, 30.0, 180.0, 5.0,
            [({'id': 'b', 'geom': geom}, 'depth')])
        comp = {'summaries': summaries, 'cat1': {'summaries': {}}}
        out = list(_iter_hit_probs(comp, recovery=900.0))
        assert len(out) == 1
        category, key, p_hit, shares = out[0]
        assert category == 'cat2' and key == ('depth', 'b') and p_hit > 0
        assert sum(shares.values()) == pytest.approx(1.0)


class TestReportEdgeBlock:
    def test_grounding_edges_sum_to_obstacle_total(self):
        host = _tpm._make_mixin_host('LEPPoweredGrounding')
        host.run_powered_grounding_model(_tpm._minimal_powered_data(include_objects=False))
        rep = host.powered_grounding_report
        seg_root = rep['by_obstacle_segment_legdir']['d1']
        edge_sum = sum(v for seg in seg_root.values() for v in seg.values())
        assert edge_sum == pytest.approx(rep['by_obstacle']['d1'], rel=1e-12)
        # Only the face towards the leg end is ever reached.
        from shapely import wkt as _wkt
        geom = _wkt.loads(_tpm._minimal_powered_data()['depths'][0][2])
        assert set(seg_root) == {f"seg_{_west_face_index(geom)}"}
        # Per leg-direction entries mirror by_obstacle_leg.
        for dir_key, total in rep['by_obstacle_leg']['d1'].items():
            per_dir = sum(seg.get(dir_key, 0.0) for seg in seg_root.values())
            assert per_dir == pytest.approx(total, rel=1e-12)

    def test_allision_report_carries_edge_block(self):
        host = _tpm._make_mixin_host('LEPPoweredAllision')
        host.run_powered_allision_model(_tpm._minimal_powered_data(include_depths=False))
        rep = host.powered_allision_report
        seg_root = rep['by_obstacle_segment_legdir']['s1']
        edge_sum = sum(v for seg in seg_root.values() for v in seg.values())
        assert edge_sum == pytest.approx(rep['by_obstacle']['s1'], rel=1e-12)

    def test_empty_report_has_the_key(self):
        host = _tpm._make_mixin_host('LEPPoweredGrounding')
        host.run_powered_grounding_model(_tpm._minimal_powered_data(include_traffic=False))
        assert host.powered_grounding_report['by_obstacle_segment_legdir'] == {}


# ---------------------------------------------------------------------------
# Result layer: one value per edge, summing to the object probability
# ---------------------------------------------------------------------------

def _layer_rows(layer):
    names = [f.name() for f in layer.fields()]
    return names, [dict(zip(names, f.attributes())) for f in layer.getFeatures()]


class TestPoweredLayerPerEdge:
    def test_edges_sum_to_object_probability_and_unhit_edges_are_zero(self, qgis_iface):
        host = _tpm._make_mixin_host('LEPPoweredGrounding')
        data = _tpm._minimal_powered_data(include_objects=False)
        host._last_depths_original = [
            {'id': 'd1', 'depth': 12.0, 'wkt': data['depths'][0][2], 'wkt_wgs84': data['depths'][0][2]},
        ]
        host.run_powered_grounding_model(data)
        layer = host.powered_grounding_layer
        assert layer is not None
        names, rows = _layer_rows(layer)
        assert len(rows) == 4
        obj_p = {r['object_probability'] for r in rows}
        assert len(obj_p) == 1                                   # object total is per obstacle
        edge_vals = [r['total_edge_probability'] for r in rows]
        assert sum(edge_vals) == pytest.approx(obj_p.pop(), rel=1e-9)
        assert sum(1 for v in edge_vals if v == 0.0) == 3        # back and side faces untouched
        # The per-leg column follows the edge, not the obstacle.
        assert 'leg_1_0' in names
        leg_vals = [r['leg_1_0'] for r in rows]
        assert sum(1 for v in leg_vals if v == 0.0) == 3
        assert max(leg_vals) == pytest.approx(max(edge_vals))

    def test_layer_falls_back_to_heuristic_without_edge_block(self, qgis_iface):
        from geometries.result_layers import create_powered_grounding_layer
        wkt = 'POLYGON((14.21 55.195, 14.24 55.195, 14.24 55.205, 14.21 55.205, 14.21 55.195))'
        report = {
            'by_obstacle': {'d1': 1e-4},
            'by_obstacle_leg': {'d1': {'1:0': 1e-4}},
        }
        seg_data = {'1': {'Start_Point': '14.0 55.2', 'End_Point': '14.2 55.2'}}
        layer = create_powered_grounding_layer(
            report, [{'id': 'd1', 'depth': 12.0, 'wkt_wgs84': wkt}],
            add_to_project=False, segment_data=seg_data)
        assert layer is not None
        _, rows = _layer_rows(layer)
        assert len(rows) == 4
        assert all(r['leg_1_0'] == pytest.approx(1e-4) for r in rows)   # old behaviour, kept
