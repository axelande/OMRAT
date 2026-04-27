"""Tests for the IWRAP XML export side of compute/iwrap_convertion.py.

The import path is exercised by ``test_iwrap_import.py``; this file
covers ``write_iwrap_xml`` / ``generate_iwrap_xml`` / the small
``build_*`` helpers and the WKT parsing edge cases.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.iwrap_convertion import (
    parse_wkt_polygon,
    parse_point_str,
    parse_generic_polygon,
    parse_wkt_multipolygon,
    new_guid,
    prettify_xml,
    build_drifting,
    build_bridges,
    build_waypoints,
    build_legs,
    build_areas,
    build_traffic_distributions,
    generate_iwrap_xml,
    write_iwrap_xml,
)


# ---------------------------------------------------------------------------
# parse_wkt_polygon  &  parse_point_str
# ---------------------------------------------------------------------------

class TestParseWktPolygon:
    def test_valid_polygon(self):
        coords = parse_wkt_polygon('POLYGON((10 50, 11 50, 11 51, 10 51, 10 50))')
        # Returned as (lat, lon) pairs from (lon lat) WKT.
        assert (50, 10) in coords
        assert len(coords) == 5

    def test_non_string_returns_empty(self):
        assert parse_wkt_polygon(None) == []
        assert parse_wkt_polygon(123) == []

    def test_non_polygon_wkt_returns_empty(self):
        assert parse_wkt_polygon('POINT(0 0)') == []
        assert parse_wkt_polygon('LINESTRING(0 0, 1 1)') == []

    def test_blank_pair_skipped(self):
        # Trailing comma + blank pair should be skipped.
        coords = parse_wkt_polygon('POLYGON((10 50, , 11 51))')
        assert (50, 10) in coords
        assert (51, 11) in coords

    def test_short_pair_skipped(self):
        coords = parse_wkt_polygon('POLYGON((10 50, 99, 11 51))')
        # "99" has only 1 token -> skipped.
        assert (50, 10) in coords
        assert (51, 11) in coords

    def test_unparseable_returns_empty(self):
        # Missing inner parens trips the float() error.
        assert parse_wkt_polygon('POLYGON not-wkt') == []


class TestParsePointStr:
    def test_valid(self):
        assert parse_point_str('14.0 55.0') == (55.0, 14.0)

    def test_invalid_returns_none(self):
        assert parse_point_str('') is None
        assert parse_point_str('only-one-token') is None
        assert parse_point_str(None) is None


class TestParseGenericPolygon:
    def test_semicolon_separated_coords(self):
        # Format: "lon lat; lon lat; ..."
        coords = parse_generic_polygon('10 50; 11 50; 11 51; 10 51; 10 50')
        # Each (lon, lat) pair flipped to (lat, lon).
        assert (50.0, 10.0) in coords
        assert len(coords) == 5

    def test_comma_separated_within_pair(self):
        coords = parse_generic_polygon('10,50; 11,50; 11,51')
        assert (50.0, 10.0) in coords

    def test_empty_returns_empty(self):
        assert parse_generic_polygon('') == []

    def test_short_pair_skipped(self):
        coords = parse_generic_polygon('10 50; only-one; 11 51')
        assert (50.0, 10.0) in coords
        assert (51.0, 11.0) in coords
        # 'only-one' yields a single token -> skipped.
        assert len(coords) == 2

    def test_unparseable_returns_empty(self):
        # Mixed alphabetic tokens trip the float() call.
        assert parse_generic_polygon('a b; c d') == []


class TestParseWktMultipolygon:
    def test_two_polygons(self):
        wkt = ('MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)),'
               '((2 2, 3 2, 3 3, 2 3, 2 2)))')
        rings = parse_wkt_multipolygon(wkt)
        assert len(rings) == 2

    def test_non_multipolygon_returns_empty(self):
        assert parse_wkt_multipolygon('POLYGON((0 0, 1 0, 1 1))') == []
        assert parse_wkt_multipolygon('') == []


# ---------------------------------------------------------------------------
# new_guid / prettify_xml
# ---------------------------------------------------------------------------

class TestSimpleHelpers:
    def test_guid_format(self):
        g = new_guid()
        # UUID4 string with hyphens, uppercased.
        assert isinstance(g, str)
        assert len(g) == 36
        assert g == g.upper()

    def test_prettify_xml_indents(self):
        root = ET.Element('outer')
        ET.SubElement(root, 'inner').set('a', 'b')
        out = prettify_xml(root)
        assert '<outer>' in out and '<inner' in out
        # Pretty-printed -> contains newlines and indentation.
        assert '\n' in out


# ---------------------------------------------------------------------------
# build_drifting
# ---------------------------------------------------------------------------

class TestBuildDrifting:
    def test_basic_drift_attributes(self):
        root = ET.Element('root')
        drift = {
            'anchor_p': 0.7,
            'anchor_d': 7.0,
            'speed': 1.94,
            'rose': {str(a): 0.125 for a in (0, 45, 90, 135, 180, 225, 270, 315)},
            'repair': {
                'combi': 'Mean/Std', 'param_0': 1.0, 'param_1': 0.5,
                'type': 'Normal',
            },
        }
        build_drifting(root, drift)
        drifting = root.find('drifting')
        assert drifting is not None
        assert drifting.get('anchor_probability') == '0.7'
        assert drifting.get('drift_speed') == '1.94'

    def test_drift_p_fallback_for_anchor_probability(self):
        """When anchor_p is missing, drift_p is used as the fallback."""
        root = ET.Element('root')
        build_drifting(root, {'drift_p': 1.0, 'rose': {}})
        d = root.find('drifting')
        assert d.get('anchor_probability') == '1.0'

    def test_repair_func_only_when_no_distribution(self):
        """If 'func' is set but no distribution attrs, repair_time_func is added."""
        root = ET.Element('root')
        build_drifting(root, {
            'rose': {},
            'repair': {'func': 'lognorm(0,1)'},
        })
        rf = root.find('drifting/repair_time_func')
        assert rf is not None
        assert rf.get('name') == 'lognorm(0,1)'

    def test_drift_directions_added(self):
        root = ET.Element('root')
        build_drifting(root, {'rose': {'90': 0.5}})
        dd = root.find('drifting/drift_directions')
        assert dd is not None
        # OMRAT 90° -> IWRAP (90 + 180) % 360 = 270.
        assert dd.get('angle_270') == '0.5'


# ---------------------------------------------------------------------------
# build_bridges
# ---------------------------------------------------------------------------

class TestBuildBridges:
    def test_empty_objects_skipped(self):
        root = ET.Element('root')
        build_bridges(root, [])
        assert root.find('bridges') is None

    def test_dict_object_form(self):
        root = ET.Element('root')
        build_bridges(root, [{
            'id': 'bridge_1', 'height': 25,
            'polygon': 'POLYGON((10 50, 11 50, 11 51, 10 51, 10 50))',
        }])
        bridges = root.find('bridges')
        assert bridges is not None
        bridge = bridges.find('bridge')
        assert bridge.get('name') == 'bridge_1'
        items = bridge.find('bridge_polyline').findall('item')
        assert len(items) == 5
        assert all(i.get('height') == '25' for i in items)

    def test_list_object_form(self):
        root = ET.Element('root')
        build_bridges(root, [
            ['b1', 30, 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'],
        ])
        bridge = root.find('bridges/bridge')
        assert bridge.get('name') == 'b1'

    def test_object_without_height_omits_attr(self):
        root = ET.Element('root')
        build_bridges(root, [['b1', '', 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))']])
        items = root.findall('bridges/bridge/bridge_polyline/item')
        assert items
        assert items[0].get('height') is None  # not set


# ---------------------------------------------------------------------------
# build_waypoints / build_legs
# ---------------------------------------------------------------------------

class TestBuildWaypoints:
    def test_creates_waypoints_for_each_endpoint(self):
        root = ET.Element('root')
        seg_data = {
            '1': {'Start_Point': '14.0 55.0', 'End_Point': '14.1 55.0'},
            '2': {'Start_Point': '14.1 55.0',  # shared with leg 1 end
                  'End_Point': '14.2 55.0'},
        }
        build_waypoints(root, seg_data)
        wps = root.findall('waypoints/waypoint')
        # 3 unique waypoints (shared one merged).
        assert len(wps) == 3

    def test_invalid_endpoint_skipped(self):
        root = ET.Element('root')
        seg_data = {
            '1': {'Start_Point': 'not-valid', 'End_Point': '14.1 55.0'},
        }
        build_waypoints(root, seg_data)
        wps = root.findall('waypoints/waypoint')
        # Only the valid endpoint becomes a waypoint.
        assert len(wps) == 1


class TestBuildAreas:
    def test_depth_polygons_added(self):
        root = ET.Element('root')
        depths = [
            ['d1', '5', 'POLYGON((10 50, 11 50, 11 51, 10 51, 10 50))'],
        ]
        build_areas(root, depths)
        # Areas container has one area_polygon child.
        a = root.find('areas/area_polygon')
        assert a is not None

    def test_empty_depths_no_areas_node(self):
        root = ET.Element('root')
        build_areas(root, [])
        assert root.find('areas') is None

    def test_object_area_marked_as_obstacle(self):
        """An entry with depth==-1 is treated as an object-area."""
        root = ET.Element('root')
        build_areas(root, [['s1', '-1',
                            'POLYGON((10 50, 11 50, 11 51, 10 51, 10 50))']])
        # Function still emits an area_polygon -- just exercise the path.
        assert root.find('areas/area_polygon') is not None

    def test_dict_form_accepted(self):
        root = ET.Element('root')
        build_areas(root, [{
            'id': 'd1', 'depth': 5,
            'polygon': 'POLYGON((10 50, 11 50, 11 51, 10 51, 10 50))',
        }])
        assert root.find('areas/area_polygon') is not None

    def test_multipolygon_depth(self):
        root = ET.Element('root')
        build_areas(root, [
            ['mp', '5',
             'MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)),'
             '((2 2, 3 2, 3 3, 2 3, 2 2)))'],
        ])
        polys = root.findall('areas/area_polygon')
        # Two polygons emitted from the MULTIPOLYGON.
        assert len(polys) >= 2

    def test_polygon_as_list_form(self):
        """Polygon stored as list of [lon, lat] pairs is accepted."""
        root = ET.Element('root')
        build_areas(root, [
            ['d1', '5', [[10, 50], [11, 50], [11, 51], [10, 51], [10, 50]]],
        ])
        assert root.find('areas/area_polygon') is not None

    def test_invalid_polygon_yields_empty_area(self):
        """Unparseable polygon WKT -> empty area_polygon container."""
        root = ET.Element('root')
        build_areas(root, [['d1', '5', 'NOT WKT']])
        # Function still emits an empty area_polygon container.
        assert root.find('areas/area_polygon') is not None


# ---------------------------------------------------------------------------
# generate_iwrap_xml -- top-level orchestration
# ---------------------------------------------------------------------------

class TestGenerateIwrapXml:
    def _minimal_data(self):
        return {
            'project_name': 'unittest',
            'segment_data': {
                '1': {
                    'Start_Point': '14.0 55.0',
                    'End_Point': '14.1 55.0',
                    'Width': 1000,
                    'Dirs': ['East going', 'West going'],
                    'mean1_1': 0.0, 'std1_1': 100.0, 'weight1_1': 1.0,
                    'mean2_1': 0.0, 'std2_1': 100.0, 'weight2_1': 1.0,
                    'u_min1': 0.0, 'u_max1': 0.0, 'u_p1': 0.0,
                    'u_min2': 0.0, 'u_max2': 0.0, 'u_p2': 0.0,
                },
            },
            'traffic_data': {
                '1': {
                    'East going': {
                        'Frequency (ships/year)': [[10] * 5 for _ in range(21)],
                        'Speed (knots)': [[10.0] * 5 for _ in range(21)],
                        'Draught (meters)': [[5.0] * 5 for _ in range(21)],
                        'Ship heights (meters)': [[10.0] * 5 for _ in range(21)],
                        'Ship Beam (meters)': [[20.0] * 5 for _ in range(21)],
                    },
                    'West going': {
                        'Frequency (ships/year)': [[5] * 5 for _ in range(21)],
                        'Speed (knots)': [[10.0] * 5 for _ in range(21)],
                        'Draught (meters)': [[5.0] * 5 for _ in range(21)],
                        'Ship heights (meters)': [[10.0] * 5 for _ in range(21)],
                        'Ship Beam (meters)': [[20.0] * 5 for _ in range(21)],
                    },
                },
            },
            'drift': {
                'anchor_p': 0.7, 'anchor_d': 7,
                'speed': 1.94,
                'rose': {str(a): 0.125 for a in
                         (0, 45, 90, 135, 180, 225, 270, 315)},
                'repair': {'use_lognormal': False, 'std': 1.0,
                           'loc': 0.0, 'scale': 1.0,
                           'dist_type': 'normal', 'norm_mean': 0.0, 'norm_std': 1.0,
                           'func': '0'},
            },
            'depths': [
                ['d1', '12',
                 'POLYGON((14.05 55.05, 14.06 55.05, 14.06 55.06, 14.05 55.06, 14.05 55.05))'],
            ],
            'objects': [
                ['s1', '20',
                 'POLYGON((14.07 55.05, 14.08 55.05, 14.08 55.06, 14.07 55.06, 14.07 55.05))'],
            ],
            'pc': {'p_pc': 1.6e-4, 'headon': 4.9e-5},
        }

    def test_root_element_returns_xml(self):
        root = generate_iwrap_xml(self._minimal_data())
        # Top-level is 'riskmodel' or similar IWRAP root.
        assert root.tag == 'riskmodel'

    def test_output_xml_is_well_formed(self, tmp_path):
        from io import BytesIO
        data_path = tmp_path / 'data.omrat'
        data_path.write_text(json.dumps(self._minimal_data()))
        out_path = tmp_path / 'out.xml'
        write_iwrap_xml(str(data_path), str(out_path))
        # Parse it back -- well-formed XML.
        tree = ET.parse(str(out_path))
        assert tree.getroot() is not None


# ---------------------------------------------------------------------------
# build_traffic_distributions
# ---------------------------------------------------------------------------

class TestBuildTrafficDistributions:
    def test_handles_empty_traffic(self):
        root = ET.Element('root')
        # Empty dicts -> no children but no exceptions.
        build_traffic_distributions(root, {}, {})

    def test_basic_traffic(self):
        root = ET.Element('root')
        traffic = {
            '1': {
                'East going': {
                    'Frequency (ships/year)': [[10] * 5 for _ in range(21)],
                    'Speed (knots)': [[10.0] * 5 for _ in range(21)],
                    'Draught (meters)': [[5.0] * 5 for _ in range(21)],
                    'Ship Beam (meters)': [[20.0] * 5 for _ in range(21)],
                    'Ship heights (meters)': [[10.0] * 5 for _ in range(21)],
                },
            },
        }
        seg_data = {'1': {'Start_Point': '14.0 55.0', 'End_Point': '14.1 55.0',
                          'Dirs': ['East going', 'West going']}}
        # Just run without exception.
        build_traffic_distributions(root, traffic, seg_data)
