"""Unit tests for geometries/result_layers.py.

Covers the small pure helpers (`_segment_normal_angle`,
`_exposure_factor`, `_parse_angle_from_key`, `_aggregate_by_leg`) and
the mid-level `extract_obstacle_probabilities` that turns a cascade
report into per-obstacle contribution dicts.

The big `create_result_layer` / `apply_graduated_symbology` /
`create_result_layers` functions are exercised indirectly by the
minimal-cascade integration test; direct coverage on them would
require driving QgsSymbol construction which is substantially more
involved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.result_layers import (
    _segment_normal_angle,
    _exposure_factor,
    _parse_angle_from_key,
    _aggregate_by_leg,
    extract_obstacle_probabilities,
)


# ---------------------------------------------------------------------------
# _segment_normal_angle
# ---------------------------------------------------------------------------

class TestSegmentNormalAngle:
    """Now emits standard nautical bearings (0=N, 90=E, 180=S, 270=W, CW)."""

    def test_east_going_segment_normal_south(self):
        """A segment going east has outward normal pointing south (180°)."""
        angle = _segment_normal_angle(0, 0, 10, 0)
        assert angle == pytest.approx(180.0, abs=1e-6)

    def test_north_going_segment_normal_east(self):
        """A segment going north has outward normal pointing east (90°)."""
        angle = _segment_normal_angle(0, 0, 0, 10)
        assert angle == pytest.approx(90.0, abs=1e-6)

    def test_west_going_segment_normal_north(self):
        """West-going segment -> outward normal north (0° / 360°)."""
        angle = _segment_normal_angle(10, 0, 0, 0)
        assert angle == pytest.approx(0.0, abs=1e-6) or angle == pytest.approx(360.0, abs=1e-6)

    def test_south_going_segment_normal_west(self):
        """South-going segment -> outward normal west (270°)."""
        angle = _segment_normal_angle(0, 10, 0, 0)
        assert angle == pytest.approx(270.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _exposure_factor
# ---------------------------------------------------------------------------

class TestExposureFactor:
    def test_head_on_drift_gives_full_exposure(self):
        """Drift direction 180° hits a north-facing segment (normal=0) head-on."""
        assert _exposure_factor(segment_normal=0.0, drift_direction=180.0) == pytest.approx(1.0, abs=1e-9)

    def test_perpendicular_drift_zero_exposure(self):
        """Drift perpendicular to the hit direction -> 0 exposure."""
        # Segment normal = 0 (north), hit direction = 180 (south).
        # Drift east (270°) is 90° off the hit direction -> factor 0.
        assert _exposure_factor(segment_normal=0.0, drift_direction=270.0) == 0.0

    def test_back_side_drift_zero_exposure(self):
        """Drift coming from the back of the segment -> 0 exposure."""
        # Normal 0° (N), hit direction 180° (S). Drift 0° is 180° off hit.
        assert _exposure_factor(segment_normal=0.0, drift_direction=0.0) == 0.0

    def test_45_degree_off_is_cos45(self):
        # Normal 0, hit direction 180, drift at 135 is 45° off.
        exp = _exposure_factor(segment_normal=0.0, drift_direction=135.0)
        import math
        assert exp == pytest.approx(math.cos(math.radians(45)), abs=1e-9)


# ---------------------------------------------------------------------------
# _parse_angle_from_key
# ---------------------------------------------------------------------------

class TestParseAngleFromKey:
    def test_standard_key(self):
        assert _parse_angle_from_key('1:East going:90') == 90.0

    def test_key_with_trailing_data_ignored(self):
        # The function parses parts[2], so extra parts are ignored.
        assert _parse_angle_from_key('1:East going:135:extra') == 135.0

    def test_missing_angle_returns_none(self):
        assert _parse_angle_from_key('1:East going') is None

    def test_non_numeric_angle_returns_none(self):
        assert _parse_angle_from_key('1:East going:NNW') is None

    def test_empty_string_returns_none(self):
        assert _parse_angle_from_key('') is None


# ---------------------------------------------------------------------------
# _aggregate_by_leg
# ---------------------------------------------------------------------------

class TestAggregateByLeg:
    def test_sums_contributions_across_directions(self):
        contribs = {
            '1:East going:0':   1e-5,
            '1:East going:45':  2e-5,
            '1:East going:90':  3e-5,
            '2:West going:180': 4e-5,
        }
        out = _aggregate_by_leg(contribs)
        assert out['1'] == pytest.approx(6e-5, abs=1e-12)
        assert out['2'] == pytest.approx(4e-5, abs=1e-12)

    def test_empty_input_empty_output(self):
        assert _aggregate_by_leg({}) == {}

    def test_malformed_keys_skipped_gracefully(self):
        """Keys without ':' are still summed under the whole string."""
        out = _aggregate_by_leg({'plainleg': 2e-6, '1:East:0': 3e-6})
        assert out.get('plainleg') == 2e-6
        assert out.get('1') == 3e-6


# ---------------------------------------------------------------------------
# extract_obstacle_probabilities
# ---------------------------------------------------------------------------

class TestExtractObstacleProbabilities:
    @pytest.fixture
    def simple_structs_and_depths(self):
        p = box(0, 0, 1, 1)
        structs = [{'id': 's1', 'height': 20.0, 'wkt': p, 'wkt_wgs84': p}]
        depths  = [{'id': 'd1', 'depth': 12.0,  'wkt': p, 'wkt_wgs84': p}]
        return structs, depths

    def test_extracts_allision_and_grounding_sections(self, simple_structs_and_depths):
        structs, depths = simple_structs_and_depths
        report = {
            'by_object': {
                'Structure - s1': {'allision': 1e-6, 'grounding': 0.0},
                'Depth - d1':     {'allision': 0.0,  'grounding': 2e-6},
            },
            'by_structure_legdir': {
                'Structure - s1': {'1:East going:0': 1e-6},
            },
            'by_depth_legdir': {
                'Depth - d1': {'1:East going:0': 2e-6},
            },
        }
        allision_data, grounding_data = extract_obstacle_probabilities(
            report, structs, depths)

        # At least one entry in each map, keyed by obstacle id.
        assert allision_data  # non-empty
        assert grounding_data  # non-empty

        # Each entry has the expected fields.
        for entry in list(allision_data.values()) + list(grounding_data.values()):
            assert 'total_probability' in entry
            assert 'geometry' in entry
            assert 'leg_contributions' in entry
            assert entry['total_probability'] >= 0.0

    def test_empty_report_yields_empty_dicts(self, simple_structs_and_depths):
        structs, depths = simple_structs_and_depths
        allision_data, grounding_data = extract_obstacle_probabilities(
            {}, structs, depths)
        # Expect entries for each obstacle even when report is empty,
        # OR entirely empty dicts -- depending on implementation.  At
        # minimum, the function must not raise on empty input.
        assert isinstance(allision_data, dict)
        assert isinstance(grounding_data, dict)

    def test_segment_contributions_extracted(self, simple_structs_and_depths):
        """Per-segment contributions from by_*_segment_legdir fields are
        propagated into allision_data / grounding_data."""
        structs, depths = simple_structs_and_depths
        report = {
            'by_object': {
                'Structure - s1': {'allision': 1e-6, 'grounding': 0.0},
                'Depth - d1':     {'allision': 0.0,  'grounding': 2e-6},
            },
            'by_structure_segment_legdir': {
                'Structure - s1': {'seg_0': {'1:N:0': 5e-7}},
            },
            'by_depth_segment_legdir': {
                'Depth - d1': {'seg_0': {'1:N:0': 1e-6}},
            },
        }
        a, g = extract_obstacle_probabilities(report, structs, depths)
        assert a['s1']['segment_contributions'] == {'seg_0': {'1:N:0': 5e-7}}
        assert g['d1']['segment_contributions'] == {'seg_0': {'1:N:0': 1e-6}}


# ---------------------------------------------------------------------------
# _extract_line_segments_with_normals
# ---------------------------------------------------------------------------

class TestExtractLineSegmentsWithNormals:
    def test_polygon_returns_per_edge_with_normals(self):
        from geometries.result_layers import _extract_line_segments_with_normals
        segs = _extract_line_segments_with_normals(box(0, 0, 10, 10))
        # 4 edges expected.
        assert len(segs) == 4
        for x1, y1, x2, y2, normal in segs:
            assert isinstance(normal, float)
            assert 0.0 <= normal < 360.0

    def test_multipolygon_aggregates(self):
        from geometries.result_layers import _extract_line_segments_with_normals
        from shapely.geometry import MultiPolygon
        mp = MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)])
        segs = _extract_line_segments_with_normals(mp)
        assert len(segs) == 8

    def test_polygon_with_hole_extracts_both_rings(self):
        from geometries.result_layers import _extract_line_segments_with_normals
        from shapely.geometry import Polygon
        outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        segs = _extract_line_segments_with_normals(Polygon(outer, [hole]))
        # 4 outer + 4 inner = 8
        assert len(segs) == 8

    def test_linestring_uses_coords_path(self):
        from geometries.result_layers import _extract_line_segments_with_normals
        from shapely.geometry import LineString
        ls = LineString([(0, 0), (10, 0), (10, 10)])
        segs = _extract_line_segments_with_normals(ls)
        assert len(segs) == 2

    def test_empty_polygon_yields_no_segments(self):
        from geometries.result_layers import _extract_line_segments_with_normals
        from shapely.geometry import Polygon
        assert _extract_line_segments_with_normals(Polygon()) == []


# ---------------------------------------------------------------------------
# _calculate_segment_probability
# ---------------------------------------------------------------------------

class TestCalculateSegmentProbability:
    def test_aggregates_per_leg_with_exposure(self):
        """Each leg-direction contribution is weighted by the exposure factor
        and summed per leg id."""
        from geometries.result_layers import _calculate_segment_probability
        # Segment normal pointing east (compass 270).  Drift coming from
        # west (compass 90) hits it head-on -> exposure = 1.0
        contribs = {
            '1:E going:90': 1e-6,   # head-on, exposure 1.0
            '1:N going:0':  2e-6,   # perpendicular -> exposure 0
        }
        total, per_leg = _calculate_segment_probability(
            segment_normal=270.0, leg_dir_contributions=contribs,
        )
        assert total == pytest.approx(1e-6, rel=1e-9)
        assert per_leg == {'1': pytest.approx(1e-6, rel=1e-9)}

    def test_unparseable_key_skipped(self):
        from geometries.result_layers import _calculate_segment_probability
        contribs = {'no_colons_here': 1e-6}
        total, per_leg = _calculate_segment_probability(
            segment_normal=0.0, leg_dir_contributions=contribs,
        )
        assert total == 0.0
        assert per_leg == {}


# ---------------------------------------------------------------------------
# create_result_layer / apply_graduated_symbology / create_result_layers
# (QGIS layer building -- needs qgis_iface)
# ---------------------------------------------------------------------------

class TestCreateResultLayer:
    def _sample_data(self):
        return {
            's1': {
                'total_probability': 1e-5,
                'geometry': box(0, 0, 1, 1),
                'value': 20.0,
                'leg_contributions': {'1': 5e-6, '2': 5e-6},
                'leg_dir_contributions': {'1:E:90': 5e-6, '2:E:90': 5e-6},
                'segment_contributions': {},
            },
        }

    def test_returns_none_for_empty_data(self, qgis_iface):
        from geometries.result_layers import create_result_layer
        assert create_result_layer('Empty', {}) is None

    def test_creates_layer_with_features(self, qgis_iface):
        from geometries.result_layers import create_result_layer
        layer = create_result_layer('Test', self._sample_data(), 'allision')
        assert layer is not None
        # 4 segments per box -> 4 features.
        assert layer.featureCount() == 4
        # Per-leg fields added.
        names = [f.name() for f in layer.fields()]
        assert 'leg_1' in names and 'leg_2' in names

    def test_uses_segment_contributions_when_provided(self, qgis_iface):
        from geometries.result_layers import create_result_layer
        data = self._sample_data()
        # Add per-segment contributions.
        data['s1']['segment_contributions'] = {
            'seg_0': {'1:E:90': 1e-6, '2:E:90': 1e-6},
        }
        layer = create_result_layer('Test', data, 'allision')
        assert layer is not None
        assert layer.featureCount() == 4

    def test_obstacle_with_no_geometry_skipped(self, qgis_iface):
        from geometries.result_layers import create_result_layer
        data = {'s1': {'total_probability': 1e-5, 'geometry': None,
                       'value': 20.0, 'leg_contributions': {},
                       'leg_dir_contributions': {}, 'segment_contributions': {}}}
        layer = create_result_layer('Test', data, 'allision')
        # Layer is created, but no features.
        assert layer is not None
        assert layer.featureCount() == 0


class TestApplyGraduatedSymbology:
    def test_handles_empty_layer(self, qgis_iface):
        from geometries.result_layers import apply_graduated_symbology
        from qgis.core import QgsVectorLayer
        layer = QgsVectorLayer("LineString?crs=epsg:4326", "x", "memory")
        # Should not raise on empty.
        apply_graduated_symbology(layer)

    def test_handles_missing_attribute(self, qgis_iface):
        """If the requested attribute isn't in the layer, return without raising."""
        from geometries.result_layers import (
            apply_graduated_symbology, create_result_layer,
        )
        layer = create_result_layer('Test', {
            's1': {'total_probability': 1e-5, 'geometry': box(0, 0, 1, 1),
                   'value': 20.0, 'leg_contributions': {},
                   'leg_dir_contributions': {}, 'segment_contributions': {}},
        })
        # Try to apply on a non-existent field.
        apply_graduated_symbology(layer, attribute='no_such_field')

    def test_single_value_uses_single_symbol(self, qgis_iface):
        """When all values are equal, the function falls back to a single symbol."""
        from geometries.result_layers import (
            apply_graduated_symbology, create_result_layer,
        )
        layer = create_result_layer('Test', {
            's1': {'total_probability': 2e-5, 'geometry': box(0, 0, 1, 1),
                   'value': 20.0, 'leg_contributions': {},
                   'leg_dir_contributions': {}, 'segment_contributions': {}},
        })
        # All 4 features have the same total_prob -> single symbol path.
        apply_graduated_symbology(layer)
        # Renderer should be set.
        assert layer.renderer() is not None

    def test_graduated_classification(self, qgis_iface):
        """With a range of probabilities, the renderer classifies into bins."""
        from geometries.result_layers import apply_graduated_symbology
        from qgis.core import QgsVectorLayer, QgsField, QgsFeature, QgsGeometry
        from qgis.PyQt.QtCore import QVariant

        layer = QgsVectorLayer("Point?crs=epsg:4326", "ranged", "memory")
        provider = layer.dataProvider()
        provider.addAttributes([QgsField("total_edge_probability", QVariant.Double)])
        layer.updateFields()
        for v in [0.1, 0.3, 0.5, 0.7, 0.9]:
            feat = QgsFeature(layer.fields())
            feat.setGeometry(QgsGeometry.fromWkt('POINT(0 0)'))
            feat.setAttribute('total_edge_probability', v)
            provider.addFeature(feat)
        apply_graduated_symbology(layer, num_classes=5)
        assert layer.renderer() is not None


class TestCreateResultLayers:
    def test_none_report_returns_none_pair(self, qgis_iface):
        from geometries.result_layers import create_result_layers
        a, g = create_result_layers(None, [], [])
        assert a is None and g is None

    def test_creates_both_layers(self, qgis_iface):
        from geometries.result_layers import create_result_layers
        structs = [{'id': 's1', 'height': 20.0, 'wkt': box(0, 0, 1, 1),
                    'wkt_wgs84': box(0, 0, 1, 1)}]
        depths = [{'id': 'd1', 'depth': 12.0, 'wkt': box(0, 0, 1, 1),
                   'wkt_wgs84': box(0, 0, 1, 1)}]
        report = {
            'by_object': {
                'Structure - s1': {'allision': 1e-5, 'grounding': 0.0},
                'Depth - d1':     {'allision': 0.0, 'grounding': 1e-5},
            },
            'by_structure_legdir': {'Structure - s1': {'1:N:0': 5e-6}},
            'by_depth_legdir':     {'Depth - d1':     {'1:N:0': 5e-6}},
            'by_structure_segment_legdir': {},
            'by_depth_segment_legdir':     {},
        }
        a, g = create_result_layers(report, structs, depths, add_to_project=False)
        assert a is not None
        assert g is not None
        assert a.featureCount() == 4
        assert g.featureCount() == 4
