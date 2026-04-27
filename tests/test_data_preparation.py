"""Unit tests for compute/data_preparation.py.

Covers the pure functions: get_distribution, clean_traffic,
safe_load_wkt, load_areas, split_structures_and_depths,
transform_to_utm, prepare_traffic_lists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, MultiPolygon, Polygon, box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.data_preparation import (
    get_distribution,
    clean_traffic,
    safe_load_wkt,
    load_areas,
    split_structures_and_depths,
    transform_to_utm,
    prepare_traffic_lists,
    _is_qgis_available,
)


# ---------------------------------------------------------------------------
# get_distribution
# ---------------------------------------------------------------------------

class TestGetDistribution:
    def test_single_gaussian_component(self):
        seg = {
            'mean1_1': 0.0, 'std1_1': 100.0, 'weight1_1': 10.0,
            'mean1_2': 0.0, 'std1_2': 0.0, 'weight1_2': 0.0,
            'mean1_3': 0.0, 'std1_3': 0.0, 'weight1_3': 0.0,
            'u_min1': 0.0, 'u_max1': 0.0, 'u_p1': 0.0,
        }
        dists, weights = get_distribution(seg, direction=0)
        # Returns 4 entries: 3 Gaussians + 1 uniform.
        assert len(dists) == 4 and len(weights) == 4
        assert weights[0] == 10.0 and weights[1] == 0.0 and weights[2] == 0.0
        assert weights[3] == 0.0

    def test_missing_fields_filled_with_zero_weight(self):
        """When mean/std/weight keys are missing, the function still
        returns placeholder distributions with zero weights."""
        seg = {'u_p1': 0.0, 'u_min1': 0.0, 'u_max1': 0.0}
        dists, weights = get_distribution(seg, direction=0)
        assert len(dists) == 4
        assert all(w == 0 for w in weights)

    def test_uniform_component_added_when_u_p_positive(self):
        seg = {
            'mean1_1': 0.0, 'std1_1': 100.0, 'weight1_1': 1.0,
            'u_min1': -5.0, 'u_max1': 5.0, 'u_p1': 2.0,
        }
        dists, weights = get_distribution(seg, direction=0)
        # uniform is the 4th entry
        assert weights[3] == 2.0
        # its sample range checks pass
        assert dists[3].support() == (-5.0, 5.0)

    def test_direction_index_offsets_keys(self):
        """direction=1 maps to the dir2 keys (weight2_1 etc.)."""
        seg = {
            'mean2_1': 3.0, 'std2_1': 50.0, 'weight2_1': 7.0,
            'u_p2': 0.0, 'u_min2': 0, 'u_max2': 0,
        }
        dists, weights = get_distribution(seg, direction=1)
        assert weights[0] == 7.0


# ---------------------------------------------------------------------------
# safe_load_wkt
# ---------------------------------------------------------------------------

class TestSafeLoadWkt:
    def test_loads_valid_wkt(self):
        g = safe_load_wkt('POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))')
        assert isinstance(g, Polygon)
        assert g.area == 1.0

    def test_empty_string_returns_none(self):
        assert safe_load_wkt('') is None
        assert safe_load_wkt('   ') is None

    def test_none_returns_none(self):
        assert safe_load_wkt(None) is None

    def test_invalid_wkt_returns_none(self):
        assert safe_load_wkt('THIS IS NOT WKT') is None


# ---------------------------------------------------------------------------
# load_areas
# ---------------------------------------------------------------------------

class TestLoadAreas:
    def test_loads_mixed_objects_and_depths(self):
        data = {
            'objects': [['struct_1', '12.0', 'POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))']],
            'depths': [['depth_1', '6.0', 'POLYGON ((2 2, 3 2, 3 3, 2 3, 2 2))']],
        }
        out = load_areas(data)
        types = sorted(o['type'] for o in out)
        assert types == ['Depth', 'Structure']
        struct = next(o for o in out if o['type'] == 'Structure')
        assert struct['id'] == 'struct_1' and struct['height'] == '12.0'

    def test_skips_malformed_rows(self):
        data = {'objects': [['only_two']], 'depths': []}
        assert load_areas(data) == []

    def test_skips_invalid_wkt(self):
        data = {'objects': [['1', '10.0', 'not-wkt']], 'depths': []}
        assert load_areas(data) == []


# ---------------------------------------------------------------------------
# split_structures_and_depths
# ---------------------------------------------------------------------------

class TestSplitStructuresAndDepths:
    def test_splits_multipolygons(self):
        mp_wkt = MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)]).wkt
        data = {
            'objects': [['m1', '8.0', mp_wkt]],
            'depths': [],
        }
        structs, depths = split_structures_and_depths(data)
        # Two polygons in MultiPolygon -> two struct entries.
        assert len(structs) == 2 and len(depths) == 0
        assert {s['id'] for s in structs} == {'m1_0', 'm1_1'}

    def test_single_polygon_kept_intact(self):
        wkt = 'POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))'
        data = {'objects': [], 'depths': [['d1', '9.0', wkt]]}
        _, depths = split_structures_and_depths(data)
        assert len(depths) == 1
        assert depths[0]['id'] == 'd1' and depths[0]['depth'] == 9.0

    def test_height_cast_failure_skips_row(self):
        data = {'objects': [['s1', 'not-a-number', 'POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))']],
                'depths': []}
        structs, _ = split_structures_and_depths(data)
        assert structs == []


# ---------------------------------------------------------------------------
# transform_to_utm
# ---------------------------------------------------------------------------

class TestTransformToUtm:
    def test_northern_hemisphere_zone_33n_for_sweden(self):
        """A line near lon 14, lat 55 should hit UTM zone 33N = EPSG:32633."""
        line = LineString([(14.0, 55.0), (15.0, 55.0)])
        poly = box(14.1, 55.1, 14.2, 55.2)
        _, _, epsg = transform_to_utm([line], [poly])
        assert epsg == 32633

    def test_southern_hemisphere_uses_327xx(self):
        line = LineString([(0.0, -30.0), (1.0, -30.0)])
        _, _, epsg = transform_to_utm([line], [])
        assert 32700 < epsg < 32800

    def test_transform_preserves_geometry_types(self):
        line = LineString([(14.0, 55.0), (14.1, 55.1)])
        poly = box(14.2, 55.2, 14.25, 55.25)
        t_lines, t_polys, _ = transform_to_utm([line], [poly])
        assert len(t_lines) == 1 and len(t_polys) == 1
        assert isinstance(t_lines[0], LineString)
        assert isinstance(t_polys[0], Polygon)
        # After transform, coordinates are in meters (much larger than
        # the original lon/lat values).
        x, y = list(t_lines[0].coords)[0]
        assert abs(x) > 100_000


# ---------------------------------------------------------------------------
# clean_traffic + prepare_traffic_lists
# ---------------------------------------------------------------------------

def _sample_traffic_data():
    return {
        'segment_data': {
            '1': {
                'Start_Point': '14.0 55.0', 'End_Point': '14.1 55.0',
                'Dirs': ['East going', 'West going'],
                'Width': 1000,
                'mean1_1': 0.0, 'std1_1': 100.0, 'weight1_1': 1.0,
                'mean1_2': 0.0, 'std1_2': 0.0, 'weight1_2': 0.0,
                'mean1_3': 0.0, 'std1_3': 0.0, 'weight1_3': 0.0,
                'mean2_1': 0.0, 'std2_1': 100.0, 'weight2_1': 1.0,
                'mean2_2': 0.0, 'std2_2': 0.0, 'weight2_2': 0.0,
                'mean2_3': 0.0, 'std2_3': 0.0, 'weight2_3': 0.0,
                'u_min1': 0.0, 'u_max1': 0.0, 'u_p1': 0.0,
                'u_min2': 0.0, 'u_max2': 0.0, 'u_p2': 0.0,
            },
        },
        'traffic_data': {
            '1': {
                'East going': {
                    'Frequency (ships/year)': [[0, 5], [0, 0]],
                    'Speed (knots)': [[0.0, 12.0], [0.0, 0.0]],
                    'Draught (meters)': [[0.0, 6.5], [0.0, 0.0]],
                    'Ship heights (meters)': [[0.0, 20.0], [0.0, 0.0]],
                    'Ship Beam (meters)': [[0.0, 15.0], [0.0, 0.0]],
                },
                'West going': {
                    'Frequency (ships/year)': [[0, 0], [0, 0]],
                    'Speed (knots)': [[0.0, 0.0], [0.0, 0.0]],
                    'Draught (meters)': [[0.0, 0.0], [0.0, 0.0]],
                    'Ship heights (meters)': [[0.0, 0.0], [0.0, 0.0]],
                    'Ship Beam (meters)': [[0.0, 0.0], [0.0, 0.0]],
                },
            },
        },
    }


class TestCleanTraffic:
    def test_returns_entry_per_segment_direction(self):
        out = clean_traffic(_sample_traffic_data())
        assert len(out) == 2  # two directions

    def test_positive_frequencies_produce_leg_traffic_entries(self):
        out = clean_traffic(_sample_traffic_data())
        # First entry (East going) has one non-zero cell: (0, 1) with freq=5.
        _, _, _, east_traffic, east_name = out[0]
        assert len(east_traffic) == 1
        info = east_traffic[0]
        assert info['freq'] == 5
        assert info['ship_type'] == 0
        assert info['ship_size'] == 1
        assert info['speed'] == 12.0
        assert info['direction'] == 'East going'
        assert 'Leg 1' in east_name

    def test_west_going_has_no_traffic(self):
        out = clean_traffic(_sample_traffic_data())
        _, _, _, west_traffic, _ = out[1]
        assert west_traffic == []

    def test_string_frequency_handled(self):
        data = _sample_traffic_data()
        data['traffic_data']['1']['East going']['Frequency (ships/year)'][0][1] = '5'
        out = clean_traffic(data)
        _, _, _, east_traffic, _ = out[0]
        assert east_traffic and east_traffic[0]['freq'] == 5

    def test_empty_string_frequency_skipped(self):
        data = _sample_traffic_data()
        data['traffic_data']['1']['East going']['Frequency (ships/year)'][0][1] = ''
        out = clean_traffic(data)
        _, _, _, east_traffic, _ = out[0]
        assert east_traffic == []


class TestPrepareTrafficLists:
    def test_returns_four_parallel_lists(self):
        data = _sample_traffic_data()
        lines, dists, weights, names = prepare_traffic_lists(data)
        assert len(lines) == len(dists) == len(weights) == len(names) == 2
        assert all(isinstance(l, LineString) for l in lines)
        assert all(isinstance(d, list) for d in dists)
        assert all(isinstance(w, list) for w in weights)
        assert all(isinstance(n, str) for n in names)


# ---------------------------------------------------------------------------
# _is_qgis_available  and the standalone (pyproj) transform branch
# ---------------------------------------------------------------------------

class TestIsQgisAvailable:
    def test_real_qgis_returns_bool(self):
        """With the real OSGeo4W QGIS lib loaded, QCS.isValid() returns bool."""
        # Can't force-mock the class; just verify it returns a bool.
        result = _is_qgis_available()
        assert isinstance(result, bool)

    def test_exception_returns_false(self, monkeypatch):
        """If constructing the CRS raises, the function returns False."""
        import compute.data_preparation as dp

        class Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("synthetic")

        monkeypatch.setattr(dp, 'QgsCoordinateReferenceSystem', Boom)
        assert _is_qgis_available() is False


class TestTransformToUtmStandaloneFallback:
    def test_standalone_path_uses_pyproj(self, monkeypatch):
        """When _is_qgis_available returns False, the pyproj fallback is used."""
        import compute.data_preparation as dp
        monkeypatch.setattr(dp, '_is_qgis_available', lambda: False)

        line = LineString([(14.0, 55.0), (14.1, 55.0)])
        t_lines, t_polys, epsg = transform_to_utm([line], [])
        assert epsg == 32633  # Sweden -> UTM 33N
        assert isinstance(t_lines[0], LineString)
        # Coords should now be in UTM meters (~400k m, ~6.1M m for 55N).
        x, y = list(t_lines[0].coords)[0]
        assert abs(x) > 100_000
        assert abs(y) > 1_000_000


# ---------------------------------------------------------------------------
# split_structures_and_depths -- remaining error paths
# ---------------------------------------------------------------------------

class TestSplitStructuresAndDepthsEdges:
    def test_malformed_structure_row_skipped(self):
        """Row with fewer than 3 elements -> unpack raises -> continue."""
        data = {'objects': [['only_two']], 'depths': []}
        structs, _ = split_structures_and_depths(data)
        assert structs == []

    def test_invalid_structure_wkt_skipped(self):
        data = {'objects': [['s1', '10.0', 'not-wkt']], 'depths': []}
        structs, _ = split_structures_and_depths(data)
        assert structs == []

    def test_malformed_depth_row_skipped(self):
        data = {'objects': [], 'depths': [['only_two']]}
        _, depths = split_structures_and_depths(data)
        assert depths == []

    def test_invalid_depth_wkt_skipped(self):
        data = {'objects': [], 'depths': [['d1', '5.0', 'not-wkt']]}
        _, depths = split_structures_and_depths(data)
        assert depths == []

    def test_depth_value_cast_failure_skipped(self):
        data = {'objects': [],
                'depths': [['d1', 'not-a-number', 'POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))']]}
        _, depths = split_structures_and_depths(data)
        assert depths == []

    def test_depth_multipolygon_split(self):
        mp_wkt = MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)]).wkt
        data = {'objects': [], 'depths': [['d1', '5.0', mp_wkt]]}
        _, depths = split_structures_and_depths(data)
        assert len(depths) == 2
        assert {d['id'] for d in depths} == {'d1_0', 'd1_1'}


# ---------------------------------------------------------------------------
# load_areas -- remaining error paths
# ---------------------------------------------------------------------------

class TestLoadAreasEdges:
    def test_invalid_depth_wkt_skipped(self):
        data = {'objects': [], 'depths': [['d1', '5.0', 'not-wkt']]}
        assert load_areas(data) == []

    def test_malformed_depth_row_skipped(self):
        data = {'objects': [], 'depths': [['only_two']]}
        assert load_areas(data) == []
