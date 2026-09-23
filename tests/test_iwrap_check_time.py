# -*- coding: utf-8 -*-
"""IWRAP export/import must carry the per-direction position check interval.

Before v0.16.2 the exporter wrote ``grounding_check_time="0"`` on every
``manoeuvring_aspects_leg`` (IWRAP then used its global default, 180 s)
and put ``ai1`` on the leg's ``max_bearing_angle``; the importer read it
back from there.  OMRAT's ``ai <= 0`` ("Cat II off" for that direction)
cannot be expressed through the check time, so the exporter zeroes the
leg extension past the waypoint the flow would overshoot instead, and
the importer maps a zero extension back to ``ai = 0`` (kattegatt_test4,
2026-09-21).
"""
import xml.etree.ElementTree as ET

from compute.iwrap_convertion import (
    LEG_EXTENSION_M,
    _CAT2_OFF_KEY,
    _parse_global_settings_el,
    _parse_legs_el,
    _parse_mal_el,
    _parse_waypoints_el,
    build_legs,
    build_manoeuvring_aspects_legs,
    build_waypoints,
    parse_iwrap_xml,
)


def _seg(ai1, ai2, name='LEG_1_1'):
    return {
        'Leg_name': name, 'Width': 200,
        'Start_Point': '10.0 57.0', 'End_Point': '10.1 57.0',
        'ai1': ai1, 'ai2': ai2,
        'mean1_1': 0.0, 'std1_1': 100.0, 'weight1_1': 100.0,
        'mean2_1': 0.0, 'std2_1': 100.0, 'weight2_1': 100.0,
    }


def _mals_by_guid(root):
    return {m.get('guid'): m for m in root.iter('manoeuvring_aspects_leg')}


class TestExport:
    def test_check_time_written_per_direction(self):
        root = ET.Element('riskmodel')
        guids = build_manoeuvring_aspects_legs(root, {'1': _seg(30.0, 10.0)}, {})
        by_guid = _mals_by_guid(root)
        assert by_guid[guids['1']['ftl']].get('grounding_check_time') == '30'
        assert by_guid[guids['1']['ltf']].get('grounding_check_time') == '10'

    def test_extensions_follow_ai(self):
        root = ET.Element('riskmodel')
        build_legs(root, {'1': _seg(30.0, 10.0)}, {})
        leg = root.find('legs/leg')
        assert leg.get('max_extension_first') == LEG_EXTENSION_M
        assert leg.get('max_extension_last') == LEG_EXTENSION_M
        # seconds must never end up on the bearing-angle attribute
        assert leg.get('max_bearing_angle') == '0'

    def test_ai_zero_zeroes_extension_past_overshoot_waypoint(self):
        """Direction 1 (first->last) overshoots the *last* waypoint."""
        root = ET.Element('riskmodel')
        build_legs(root, {'1': _seg(0.0, 120.0), '2': _seg(30.0, 0.0, 'LEG_1_2')}, {})
        legs = root.findall('legs/leg')
        assert legs[0].get('max_extension_last') == '0'
        assert legs[0].get('max_extension_first') == LEG_EXTENSION_M
        assert legs[1].get('max_extension_first') == '0'
        assert legs[1].get('max_extension_last') == LEG_EXTENSION_M

    def test_ai_zero_leaves_check_time_zero(self):
        root = ET.Element('riskmodel')
        guids = build_manoeuvring_aspects_legs(root, {'1': _seg(0.0, 120.0)}, {})
        by_guid = _mals_by_guid(root)
        assert by_guid[guids['1']['ftl']].get('grounding_check_time') == '0'
        assert by_guid[guids['1']['ltf']].get('grounding_check_time') == '120'


_XML_TEMPLATE = """
<riskmodel>
  <waypoints>
    <waypoint guid="A" latitude="57.0" longitude="10.0"/>
    <waypoint guid="B" latitude="57.0" longitude="10.1"/>
  </waypoints>
  <manoeuvring_aspects_legs>
    <manoeuvring_aspects_leg guid="F" name="LEG_1_1" grounding_check_time="{check_ftl}">
      <mixed_dist scale="1"><mixed_dist_item param_0="0" param_1="100" type="Normal" weight="1"/></mixed_dist>
    </manoeuvring_aspects_leg>
    <manoeuvring_aspects_leg guid="L" name="LEG_1_1" grounding_check_time="{check_ltf}">
      <mixed_dist scale="1"><mixed_dist_item param_0="0" param_1="100" type="Normal" weight="1"/></mixed_dist>
    </manoeuvring_aspects_leg>
  </manoeuvring_aspects_legs>
  <legs>
    <leg guid="G" name="LEG_1_1" max_width="200" max_bearing_angle="45"
         max_extension_first="{ext_first}" max_extension_last="{ext_last}"
         first_waypoint_guid="A" last_waypoint_guid="B"
         man_aspects_first_to_last_guid="F" man_aspects_last_to_first_guid="L"/>
  </legs>
  <global_settings>
    <misc meantime_between_checks="240"/>
  </global_settings>
</riskmodel>"""


def _xml(check_ftl, check_ltf, ext_first, ext_last):
    return ET.fromstring(_XML_TEMPLATE.format(
        check_ftl=check_ftl, check_ltf=check_ltf, ext_first=ext_first, ext_last=ext_last))


def _import_segment(root):
    wp = _parse_waypoints_el(root.find('waypoints'))
    mal = _parse_mal_el(root.find('manoeuvring_aspects_legs'))
    result = {'pc': {}, 'segment_data': _parse_legs_el(root.find('legs'), wp, mal, False)}
    _parse_global_settings_el(root.find('global_settings'), result, False)
    return result['segment_data']['1']


class TestImport:
    def test_per_direction_check_time_wins(self):
        seg = _import_segment(_xml(30, 10, 50000, 50000))
        assert seg['ai1'] == 30.0
        assert seg['ai2'] == 10.0

    def test_zero_check_time_falls_back_to_global(self):
        seg = _import_segment(_xml(0, 0, 50000, 50000))
        assert seg['ai1'] == 240.0
        assert seg['ai2'] == 240.0

    def test_zero_extension_keeps_cat2_off(self):
        seg = _import_segment(_xml(0, 120, 50000, 0))
        assert seg['ai1'] == 0.0
        assert seg['ai2'] == 120.0
        seg = _import_segment(_xml(30, 0, 0, 50000))
        assert seg['ai1'] == 30.0
        assert seg['ai2'] == 0.0

    def test_bearing_angle_is_not_ai(self):
        seg = _import_segment(_xml(0, 0, 50000, 50000))
        assert seg['ai1'] != 45.0

    def test_marker_stripped_by_parse_iwrap_xml(self, tmp_path):
        path = tmp_path / 'm.xml'
        path.write_text(ET.tostring(_xml(0, 120, 50000, 0), encoding='unicode'), encoding='utf-8')
        segs = parse_iwrap_xml(str(path))['segment_data']
        assert _CAT2_OFF_KEY not in segs['1']
        assert segs['1']['ai1'] == 0.0


def test_round_trip_preserves_ai():
    src = {'1': _seg(30.0, 10.0), '2': _seg(0.0, 120.0, 'LEG_1_2'), '3': _seg(60.0, 0.0, 'LEG_1_3')}
    root = ET.Element('riskmodel')
    wp_lookup = build_waypoints(root, src)
    mal_guids = build_manoeuvring_aspects_legs(root, src, {})
    build_legs(root, src, wp_lookup)
    for leg_el, sid in zip(root.findall('legs/leg'), src):
        leg_el.set('man_aspects_first_to_last_guid', mal_guids[sid]['ftl'])
        leg_el.set('man_aspects_last_to_first_guid', mal_guids[sid]['ltf'])
    gs = ET.SubElement(root, 'global_settings')
    ET.SubElement(gs, 'misc').set('meantime_between_checks', '240')

    wp = _parse_waypoints_el(root.find('waypoints'))
    mal = _parse_mal_el(root.find('manoeuvring_aspects_legs'))
    result = {'pc': {}, 'segment_data': _parse_legs_el(root.find('legs'), wp, mal, False)}
    _parse_global_settings_el(root.find('global_settings'), result, False)
    got = {k: (v['ai1'], v['ai2']) for k, v in result['segment_data'].items()}
    assert got == {'1': (30.0, 10.0), '2': (0.0, 120.0), '3': (60.0, 0.0)}
