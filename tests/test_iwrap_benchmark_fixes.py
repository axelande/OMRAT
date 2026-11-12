"""IWRAP export / import fixes from the Kattegat benchmark (KNOWN_ISSUES 16-22).

* E5 lateral sign: IWRAP's positive = starboard of the sailing direction.
* E1 waypoint routing: junction matrices <-> ``leg_leg_distribution`` rows.
* I1 length classes: OMRAT's 350-400 m goes to IWRAP's 325-350, merged.
* E2 passenger / fast-ferry reduction factor: the project's own, default 1.
* E3 missing draught / speed / beam: filled as a run fills them, never 0.
* E4 drifting: both blackout classes and ``min_anchor_dist_from_ground``.
* I4 depth part ids: every imported area keeps its own id.
"""
from __future__ import annotations

import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute import passenger_reduction  # noqa: E402
from compute.iwrap_convertion import (  # noqa: E402
    IWRAP_LENGTH_CLASSES,
    _parse_areas_el,
    blackout_classes,
    export_junction_rows,
    generate_iwrap_xml,
    iwrap_length_class,
    length_class_merges,
    parse_iwrap_xml,
    powered_width,
)
from geometries.junctions import junction_id_for_point  # noqa: E402

TYPES = [
    'Fishing', 'Towing', 'Dredging or underwater ops', 'Diving ops', 'Military ops',
    'Sailing', 'Pleasure Craft', 'High speed craft (HSC)', 'Pilot Vessel',
    'Search and Rescue vessel', 'Tug', 'Port Tender', 'Anti-pollution equipment',
    'Law Enforcement', 'Spare', 'Medical Transport',
    'Noncombatant ship according to RR Resolution No. 18',
    'Passenger, all ships of this type', 'Cargo, all ships of this type',
    'Tanker, all ships of this type', 'Other Type, all ships of this type',
]
INTERVALS = [{'min': float(lo), 'max': float(lo + 25), 'label': f'{lo}.0 - {lo + 25}.0'} for lo in range(0, 350, 25)]
INTERVALS.append({'min': 350.0, 'max': 400.0, 'label': '350.0 - 400.0'})
CARGO = TYPES.index('Cargo, all ships of this type')


def _seg(name: str, sp: str, ep: str, **extra) -> dict:
    seg = {
        'Leg_name': name, 'Start_Point': sp, 'End_Point': ep, 'Width': 2000,
        'Dirs': ['East going', 'West going'],
        'mean1_1': -300.0, 'std1_1': 200.0, 'weight1_1': 100.0,
        'mean2_1': 300.0, 'std2_1': 200.0, 'weight2_1': 100.0,
        'u_min1': 0.0, 'u_max1': 0.0, 'u_p1': 0, 'u_min2': 0.0, 'u_max2': 0.0, 'u_p2': 0,
        'ai1': 180.0, 'ai2': 180.0,
    }
    seg.update(extra)
    return seg


def _block(freq: float = 0.0) -> dict:
    n, m = len(TYPES), len(INTERVALS)

    def mat(v):
        return [[v] * m for _ in range(n)]

    block = {'Frequency (ships/year)': mat(0.0), 'Speed (knots)': mat(12.0), 'Draught (meters)': mat(6.0),
             'Ship heights (meters)': mat(20.0), 'Ship Beam (meters)': mat(15.0), 'Scaling (%)': mat(100.0)}
    block['Frequency (ships/year)'][CARGO][5] = freq
    return block


def _data(segments: dict, **extra) -> dict:
    data = {
        'project_name': 'benchmark-fixes',
        'segment_data': segments,
        'traffic_data': {sid: {'East going': _block(100.0), 'West going': _block(50.0)} for sid in segments},
        'ship_categories': {'types': list(TYPES), 'length_intervals': copy.deepcopy(INTERVALS)},
        'drift': {'anchor_p': 0.7, 'anchor_d': 7, 'speed': 1.0, 'drift_p': 1,
                  'rose': {str(a): 0.125 for a in range(0, 360, 45)},
                  'repair': {'use_lognormal': True, 'std': 0.95, 'loc': 0.2, 'scale': 0.85}},
        'depths': [], 'objects': [], 'pc': {},
    }
    data.update(extra)
    return data


def _round_trip(tmp_path, data: dict) -> dict:
    out = tmp_path / 'model.xml'
    ET.ElementTree(generate_iwrap_xml(data)).write(out, encoding='utf-8', xml_declaration=True)
    return parse_iwrap_xml(str(out))


def _mal_items(root, leg_name: str, attr: str) -> list:
    leg = next(el for el in root.iter('leg') if el.get('name') == leg_name)
    mal = next(el for el in root.iter('manoeuvring_aspects_leg') if el.get('guid') == leg.get(attr))
    return list(mal.iter('mixed_dist_item'))


# ---------------------------------------------------------------------------
# E5  lateral sign
# ---------------------------------------------------------------------------

class TestLateralSign:
    def test_both_lanes_are_written_on_starboard(self):
        root = generate_iwrap_xml(_data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')}))
        ftl = _mal_items(root, 'LEG_1', 'man_aspects_first_to_last_guid')
        ltf = _mal_items(root, 'LEG_1', 'man_aspects_last_to_first_guid')
        # OMRAT: dir 1 at -300 (starboard of the drawn way), dir 2 at +300 (its starboard).
        # IWRAP: positive = starboard of the sailing direction, so +300 for both.
        assert float(ftl[0].get('param_0')) == pytest.approx(300.0)
        assert float(ltf[0].get('param_0')) == pytest.approx(300.0)

    def test_uniform_bounds_follow_the_sign(self):
        seg = _seg('LEG_1', '10.0 56.0', '10.1 56.0', u_min1=-1000.0, u_max1=-200.0, u_p1=10,
                   u_min2=200.0, u_max2=900.0, u_p2=10)
        root = generate_iwrap_xml(_data({'1': seg}))
        ftl_u = [i for i in _mal_items(root, 'LEG_1', 'man_aspects_first_to_last_guid') if i.get('type') == 'Uniform']
        ltf_u = [i for i in _mal_items(root, 'LEG_1', 'man_aspects_last_to_first_guid') if i.get('type') == 'Uniform']
        assert (float(ftl_u[0].get('param_0')), float(ftl_u[0].get('param_1'))) == (200.0, 1000.0)
        assert (float(ltf_u[0].get('param_0')), float(ltf_u[0].get('param_1'))) == (200.0, 900.0)

    def test_round_trip_keeps_each_lane_on_its_side(self, tmp_path):
        seg = _seg('LEG_1', '10.0 56.0', '10.1 56.0', u_min1=-1000.0, u_max1=-200.0, u_p1=10)
        back = _round_trip(tmp_path, _data({'1': seg}))['segment_data']['1']
        assert back['mean1_1'] == pytest.approx(-300.0)
        assert back['mean2_1'] == pytest.approx(300.0)
        assert (back['u_min1'], back['u_max1']) == (pytest.approx(-1000.0), pytest.approx(-200.0))

    def test_native_iwrap_lanes_come_in_on_starboard(self, tmp_path):
        """A native IWRAP file has +100 in both directions (traffic keeps right)."""
        root = generate_iwrap_xml(_data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')}))
        for item in root.iter('mixed_dist_item'):
            item.set('param_0', '100')
        out = tmp_path / 'native.xml'
        ET.ElementTree(root).write(out, encoding='utf-8', xml_declaration=True)
        seg = parse_iwrap_xml(str(out))['segment_data']['1']
        assert seg['mean1_1'] == pytest.approx(-100.0)   # starboard of the drawn direction
        assert seg['mean2_1'] == pytest.approx(100.0)    # port of the drawn = starboard of dir 2


# ---------------------------------------------------------------------------
# E1  waypoint routing
# ---------------------------------------------------------------------------

def _three_legs() -> dict:
    return {
        '1': _seg('LEG_A', '10.0 56.0', '10.1 56.0'),
        '2': _seg('LEG_B', '10.1 56.0', '10.2 56.0'),
        '3': _seg('LEG_C', '10.1 56.0', '10.2 56.1'),
    }


def _user_junction() -> dict:
    jid = junction_id_for_point((10.1, 56.0))
    return {jid: {
        'point': [10.1, 56.0], 'legs': {'1': 'end', '2': 'start', '3': 'start'}, 'source': 'user',
        'transitions': {'1': {'2': 0.8, '3': 0.2}, '2': {'1': 0.9, '3': 0.1}, '3': {'1': 0.7, '2': 0.3}},
    }}


class TestWaypointRouting:
    def test_junction_matrix_becomes_routing_rows(self):
        root = generate_iwrap_xml(_data(_three_legs(), junctions=_user_junction()))
        guid = {el.get('guid'): el.get('name') for el in root.iter('leg')}
        rows = {(guid[r.get('from_leg_uid')], guid[r.get('to_leg_uid')]): float(r.get('fraction'))
                for r in root.iter('leg_leg_distribution')}
        assert rows == {
            ('LEG_A', 'LEG_B'): pytest.approx(0.8), ('LEG_A', 'LEG_C'): pytest.approx(0.2),
            ('LEG_B', 'LEG_A'): pytest.approx(0.9), ('LEG_B', 'LEG_C'): pytest.approx(0.1),
            ('LEG_C', 'LEG_A'): pytest.approx(0.7), ('LEG_C', 'LEG_B'): pytest.approx(0.3),
        }

    def test_rows_are_manual_shares(self):
        """AUTOMATIC makes IWRAP split equally and ignore the fraction (seen in IWRAP, 2026-11-09)."""
        root = generate_iwrap_xml(_data(_three_legs(), junctions=_user_junction()))
        rows = list(root.iter('leg_leg_distribution'))
        assert rows and {r.get('method') for r in rows} == {'MANUAL'}

    def test_every_other_leg_gets_a_row_and_shares_sum_to_one(self):
        junctions = _user_junction()
        junctions[next(iter(junctions))]['transitions']['1'] = {'2': 2.0 / 3.0, '3': 1.0 / 3.0}
        junctions[next(iter(junctions))]['transitions']['2'] = {'1': 1.0}          # no share to leg 3
        rows = export_junction_rows(_three_legs(), junctions)
        by_in: dict = {}
        for _key, in_leg, out_leg, frac in rows:
            by_in.setdefault(in_leg, {})[out_leg] = frac
        assert by_in['2'] == {'1': 1.0, '3': 0.0}                                  # explicit zero
        assert sum(by_in['1'].values()) == 1.0                                      # exact after rounding
        assert by_in['1'] == {'2': pytest.approx(0.666667, abs=1e-6), '3': pytest.approx(0.333333, abs=1e-6)}

    def test_import_takes_iwrap_split_of_automatic_rows(self, tmp_path):
        """An IWRAP-saved model: one MANUAL row, the AUTOMATIC ones hold IWRAP's equal remainder."""
        root = generate_iwrap_xml(_data(_three_legs(), junctions=_user_junction()))
        guid = {el.get('name'): el.get('guid') for el in root.iter('leg')}
        for r in root.iter('leg_leg_distribution'):
            if r.get('from_leg_uid') == guid['LEG_A']:
                manual = r.get('to_leg_uid') == guid['LEG_B']
                r.set('method', 'MANUAL' if manual else 'AUTOMATIC')
                r.set('fraction', '0.2345' if manual else '0.7655')
        out = tmp_path / 'iwrap_saved.xml'
        ET.ElementTree(root).write(out, encoding='utf-8', xml_declaration=True)
        j = next(iter(parse_iwrap_xml(str(out))['junctions'].values()))
        assert j['transitions']['1'] == {'2': pytest.approx(0.2345), '3': pytest.approx(0.7655)}

    def test_no_leg_routes_to_itself(self):
        root = generate_iwrap_xml(_data(_three_legs()))
        rows = list(root.iter('leg_leg_distribution'))
        assert rows, 'geometric defaults must still be written'
        assert all(r.get('from_leg_uid') != r.get('to_leg_uid') for r in rows)

    def test_rows_sit_on_the_shared_waypoint(self):
        root = generate_iwrap_xml(_data(_three_legs(), junctions=_user_junction()))
        holders = [wp for wp in root.iter('waypoint') if list(wp.iter('leg_leg_distribution'))]
        assert len(holders) == 1
        assert (float(holders[0].get('latitude')), float(holders[0].get('longitude'))) == (56.0, 10.1)

    def test_two_leg_chain_continues_fully(self):
        legs = {'1': _seg('LEG_A', '10.0 56.0', '10.1 56.0'), '2': _seg('LEG_B', '10.1 56.0', '10.2 56.05')}
        rows = export_junction_rows(legs, None)
        assert sorted((r[1], r[2], r[3]) for r in rows) == [('1', '2', 1.0), ('2', '1', 1.0)]

    def test_round_trip_restores_the_matrix_as_user(self, tmp_path):
        back = _round_trip(tmp_path, _data(_three_legs(), junctions=_user_junction()))
        junctions = back['junctions']
        assert len(junctions) == 1
        j = next(iter(junctions.values()))
        assert j['source'] == 'user'
        assert j['transitions']['1'] == {'2': pytest.approx(0.8), '3': pytest.approx(0.2)}
        assert j['transitions']['3'] == {'1': pytest.approx(0.7), '2': pytest.approx(0.3)}

    def test_old_self_pairs_import_as_no_matrix(self, tmp_path):
        root = generate_iwrap_xml(_data(_three_legs()))
        for row in root.iter('leg_leg_distribution'):
            row.set('to_leg_uid', row.get('from_leg_uid'))
        out = tmp_path / 'old.xml'
        ET.ElementTree(root).write(out, encoding='utf-8', xml_declaration=True)
        assert 'junctions' not in parse_iwrap_xml(str(out))


# ---------------------------------------------------------------------------
# I1  length classes
# ---------------------------------------------------------------------------

class TestLengthClasses:
    def test_iwrap_class_names(self):
        assert IWRAP_LENGTH_CLASSES[0] == '0-25' and IWRAP_LENGTH_CLASSES[-1] == '325-350'
        assert iwrap_length_class({'min': 0.0, 'max': 25.0}) == '0-25'
        assert iwrap_length_class({'min': 350.0, 'max': 400.0}) == '325-350'
        assert iwrap_length_class({'min': 400.0, 'max': 1000.0}) == '325-350'
        assert iwrap_length_class({'min': 0.0, 'max': 50.0}) == '25-50'   # midpoint rule

    def test_merges_are_listed(self):
        assert length_class_merges({'length_intervals': INTERVALS}) == [('350-400', '325-350')]
        assert length_class_merges(None) == []

    def test_largest_ships_are_merged_into_the_last_class(self):
        data = _data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')})
        block = data['traffic_data']['1']['East going']
        block['Frequency (ships/year)'][CARGO][5] = 0.0
        block['Frequency (ships/year)'][CARGO][13] = 30.0   # 325-350
        block['Frequency (ships/year)'][CARGO][14] = 10.0   # 350-400
        block['Speed (knots)'][CARGO][13] = 14.0
        block['Speed (knots)'][CARGO][14] = 18.0
        block['Ship Beam (meters)'][CARGO][14] = 54.0
        root = generate_iwrap_xml(data)
        td = next(root.iter('traffic_distribution'))   # leg 1, first-to-last
        cargo = next(st for st in td.iter('shiptype') if st.get('name') == 'General cargo ship')
        cats = {c.get('name'): c for c in cargo.iter('category')}
        assert set(cats) == {'325-350'}
        assert float(cats['325-350'].get('freq')) == pytest.approx(40.0)
        assert float(cats['325-350'].get('speed')) == pytest.approx(15.0)   # (30*14 + 10*18) / 40
        assert float(cats['325-350'].get('width')) == pytest.approx(24.75)  # (30*15 + 10*54) / 40


# ---------------------------------------------------------------------------
# E2  passenger / fast-ferry reduction factor
# ---------------------------------------------------------------------------

class TestPassengerReduction:
    def _misc(self, data):
        return generate_iwrap_xml(data).find('global_settings/misc')

    def test_default_is_iwrap_20(self):
        misc = self._misc(_data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')}))
        assert misc.get('passengership_reductionfactor') == '20'
        assert misc.get('fastferry_reduction_factor') == '20'

    def test_no_reduction_is_exported_as_1(self):
        pc = {passenger_reduction.PASSENGER_KEY: 1.0, passenger_reduction.FAST_FERRY_KEY: 1.0}
        misc = self._misc(_data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')}, pc=pc))
        assert misc.get('passengership_reductionfactor') == '1'
        assert misc.get('fastferry_reduction_factor') == '1'

    def test_project_factor_is_exported(self):
        pc = {passenger_reduction.PASSENGER_KEY: 20.0, passenger_reduction.FAST_FERRY_KEY: 10.0}
        misc = self._misc(_data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')}, pc=pc))
        assert misc.get('passengership_reductionfactor') == '20'
        assert misc.get('fastferry_reduction_factor') == '10'

    def test_import_reads_the_factors(self, tmp_path):
        pc = {passenger_reduction.PASSENGER_KEY: 20.0, passenger_reduction.FAST_FERRY_KEY: 20.0}
        back = _round_trip(tmp_path, _data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')}, pc=pc))
        assert back['pc'][passenger_reduction.PASSENGER_KEY] == 20.0
        assert back['pc'][passenger_reduction.FAST_FERRY_KEY] == 20.0


# ---------------------------------------------------------------------------
# E3  missing draught / beam
# ---------------------------------------------------------------------------

class TestFilledTraffic:
    def test_missing_draught_and_beam_are_filled_not_zero(self):
        data = _data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')})
        block = data['traffic_data']['1']['East going']
        block['Draught (meters)'][CARGO][5] = float('inf')
        block['Ship Beam (meters)'][CARGO][5] = 0.0
        before = json.dumps(data, default=str)
        root = generate_iwrap_xml(data)
        td = next(root.iter('traffic_distribution'))
        cargo = next(st for st in td.iter('shiptype') if st.get('name') == 'General cargo ship')
        cat = next(c for c in cargo.iter('category') if c.get('name') == '125-150')
        assert float(cat.get('draught')) == pytest.approx(6.0)        # same type and length, other direction
        assert float(cat.get('width')) == pytest.approx(137.5 / 6.5, abs=0.01)  # class midpoint / 6.5
        assert json.dumps(data, default=str) == before                # the project is not changed


# ---------------------------------------------------------------------------
# E4  drifting settings
# ---------------------------------------------------------------------------

class TestDriftSettings:
    def _drift_with_types(self):
        from compute.basic_equations import default_blackout_by_ship_type
        rates = default_blackout_by_ship_type(TYPES)
        return {k: (0.2 if v < 1.0 else 0.8) for k, v in rates.items()}

    def test_blackout_classes_from_per_type_rates(self):
        assert blackout_classes({'blackout_by_ship_type': self._drift_with_types()}, TYPES) == (0.2, 0.8)
        assert blackout_classes({}, TYPES) == (None, None)

    def test_both_blackout_classes_and_anchor_distance_are_written(self):
        data = _data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')})
        data['drift']['blackout_by_ship_type'] = self._drift_with_types()
        drifting = generate_iwrap_xml(data).find('drifting')
        assert drifting.get('blackout_roro_passenger') == '0.2'
        assert drifting.get('blackout_other') == '0.8'
        assert drifting.get('min_anchor_dist_from_ground') == '3'

    def test_import_restores_per_type_blackout(self, tmp_path):
        data = _data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')})
        data['drift']['blackout_by_ship_type'] = self._drift_with_types()
        back = _round_trip(tmp_path, data)
        rates = back['drift']['blackout_by_ship_type']
        assert rates[TYPES.index('Passenger, all ships of this type')] == pytest.approx(0.2)
        assert rates[CARGO] == pytest.approx(0.8)
        assert back['drift']['min_anchor_dist_from_ground'] == 3.0
        assert '_iwrap_blackout_classes' not in back['drift']


# ---------------------------------------------------------------------------
# I4  depth part ids
# ---------------------------------------------------------------------------

def _area(name: str, depth: str, area_type: str, x0: float) -> str:
    pts = [(x0, 56.0), (x0 + 0.01, 56.0), (x0 + 0.01, 56.01), (x0, 56.01), (x0, 56.0)]
    items = ''.join(f'<item lat="{lat}" lon="{lon}"/>' for lon, lat in pts)
    return f'<area_polygon name="{name}" depth="{depth}" type="{area_type}"><polygon>{items}</polygon></area_polygon>'


class TestAreaIds:
    def test_every_part_keeps_its_own_id(self):
        xml = '<areas>' + ''.join([
            _area('depth_3_1', '3', '0', 10.0), _area('depth_3_2', '3', '0', 10.1),
            _area('depth_4', '6', '0', 10.2), _area('depth_4', '6', '0', 10.3),
            _area('object_s1', '-20', '1', 10.4), _area('depth_s1', '9', '0', 10.5),
        ]) + '</areas>'
        depths, objects = _parse_areas_el(ET.fromstring(xml))
        assert [d[0] for d in depths] == ['3_1', '3_2', '4', '4_2', 's1']
        assert [o[0] for o in objects] == ['s1']

    def test_round_trip_of_a_multipolygon_gives_one_id_per_part(self, tmp_path):
        data = _data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')}, depths=[[
            '3', '3.0',
            'MULTIPOLYGON(((10.0 56.0, 10.01 56.0, 10.01 56.01, 10.0 56.01, 10.0 56.0)),'
            '((10.1 56.0, 10.11 56.0, 10.11 56.01, 10.1 56.01, 10.1 56.0)))'
        ]])
        ids = [d[0] for d in _round_trip(tmp_path, data)['depths']]
        assert ids == ['3_1', '3_2']


# ---------------------------------------------------------------------------
# Powered width: IWRAP's max_width_powered covers OMRAT's ray span
# ---------------------------------------------------------------------------

def _leg_el(root, name: str):
    return next(el for el in root.iter('leg') if el.get('name') == name)


class TestPoweredWidth:
    def test_span_wider_than_the_leg_widens_only_the_powered_width(self):
        # dir 1: -300 +/- 6 x 200 -> [-1500, 900]; dir 2: [-900, 1500] -> 2 x 1500
        root = generate_iwrap_xml(_data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')}))
        leg = _leg_el(root, 'LEG_1')
        assert leg.get('max_width') == '2000'
        assert leg.get('max_width_powered') == '3000'

    def test_narrow_distribution_keeps_the_leg_width(self):
        seg = _seg('LEG_1', '10.0 56.0', '10.1 56.0', std1_1=50.0, std2_1=50.0)
        assert powered_width(seg) == 2000

    def test_uniform_part_and_rounding(self):
        seg = _seg('LEG_1', '10.0 56.0', '10.1 56.0', u_min1=-4010.0, u_max1=-200.0, u_p1=10)
        assert powered_width(seg) == 8100

    def test_every_component_counts(self):
        seg = _seg('LEG_1', '10.0 56.0', '10.1 56.0', mean1_2=2500.0, std1_2=500.0, weight1_2=5.0)
        assert powered_width(seg) == 2 * (2500 + 6 * 500)

    def test_matches_the_ray_span(self):
        from compute.ray_sampling import mixture_span
        from geometries.distribution_curves import mixture
        seg = _seg('LEG_1', '10.0 56.0', '10.1 56.0', mean2_1=450.0, std2_1=333.0)
        half = max(abs(b) for d in (0, 1) for b in mixture_span(mixture(seg, d)))
        assert powered_width(seg) - 2 * half == pytest.approx(0.0, abs=100.0)
        assert powered_width(seg) >= 2 * half

    def test_round_trip_keeps_the_leg_width(self, tmp_path):
        back = _round_trip(tmp_path, _data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')}))
        assert back['segment_data']['1']['Width'] == 2000


# ---------------------------------------------------------------------------
# Traffic scaling: IWRAP gets the frequencies the calculation uses
# ---------------------------------------------------------------------------

def _total_freq(traffic: dict) -> float:
    return sum(float(q) for dirs in traffic.values() for block in dirs.values()
               for row in block['Frequency (ships/year)'] for q in row)


class TestScalingExport:
    def _scaled(self, percent: float) -> dict:
        data = _data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')})
        data['traffic_data']['1']['East going']['Scaling (%)'][CARGO][5] = percent
        data['traffic_scaling'] = {'global_percent': percent, 'follow_global': [True] * len(TYPES)}
        return data

    def test_scaled_frequencies_reach_the_xml(self, tmp_path):
        back = _round_trip(tmp_path, self._scaled(150.0))
        assert _total_freq(back['traffic_data']) == pytest.approx(150.0 + 50.0)

    def test_prepared_copy_is_baked_once_and_project_untouched(self):
        from compute.traffic_redirect import prepare_export_data
        data = self._scaled(150.0)
        original = copy.deepcopy(data)
        once, summary = prepare_export_data(data)
        twice, again = prepare_export_data(once)
        assert data == original
        assert summary['scaled_cells'] == 1 and again['scaled_cells'] == 0
        assert _total_freq(twice['traffic_data']) == pytest.approx(200.0)
        block = twice['traffic_data']['1']['East going']
        assert all(v == 100.0 for row in block['Scaling (%)'] for v in row)
        assert twice['traffic_scaling']['global_percent'] == 100.0
        # generate_iwrap_xml bakes again for direct callers: still a no-op
        root = generate_iwrap_xml(twice)
        assert root.find('legs') is not None

    def test_scaling_comes_before_the_redirects(self):
        """As in a run: a suppressed leg's ships are scaled on their own leg first."""
        from compute.traffic_redirect import prepare_export_data
        segs = {
            '1': _seg('LEG_1', '10.0 56.0', '10.1 56.0', suppressed=True,
                      traffic_redirect=[{'from_dir': 0, 'leg': '2', 'dir': 0, 'share': 100.0}]),
            '2': _seg('LEG_2', '10.0 56.1', '10.1 56.1'),
        }
        data = _data(segs)
        data['traffic_data']['1']['East going']['Scaling (%)'][CARGO][5] = 200.0
        prepared, _ = prepare_export_data(data)
        assert '1' not in prepared['traffic_data']
        cell = prepared['traffic_data']['2']['East going']['Frequency (ships/year)'][CARGO][5]
        assert cell == pytest.approx(100.0 + 2 * 100.0)


# ---------------------------------------------------------------------------
# Structure heights: written as IWRAP's negative depth, read back as height
# ---------------------------------------------------------------------------

STRUCT = 'POLYGON((10.05 56.0, 10.06 56.0, 10.06 56.01, 10.05 56.01, 10.05 56.0))'


def _structure_areas(root) -> list:
    return [a for a in root.iter('area_polygon') if a.get('type') == '1']


class TestStructureHeights:
    def test_height_is_written_as_negative_depth(self):
        data = _data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')}, objects=[['s1', '25.0', STRUCT]])
        (area,) = _structure_areas(generate_iwrap_xml(data))
        assert area.get('depth') == '-25'
        assert area.get('structure_type') == 'Other'
        assert area.get('name') == 'object_s1'

    @pytest.mark.parametrize('height', ['0', '', 'n/a', '-3'])
    def test_no_usable_height_is_minus_one(self, height):
        data = _data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')}, objects=[['s1', height, STRUCT]])
        (area,) = _structure_areas(generate_iwrap_xml(data))
        assert area.get('depth') == '-1'

    def test_round_trip_keeps_the_height(self, tmp_path):
        other = STRUCT.replace('10.05', '10.07').replace('10.06', '10.08')
        data = _data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')},
                     objects=[['s1', '25.0', STRUCT], ['s2', '12.5', other]])
        back = _round_trip(tmp_path, data)
        assert sorted((o[0], float(o[1])) for o in back['objects']) == [('s1', 25.0), ('s2', 12.5)]
        assert back['depths'] == []

    def test_negative_depth_area_is_not_mistaken_for_a_structure(self):
        data = _data({'1': _seg('LEG_1', '10.0 56.0', '10.1 56.0')}, depths=[['d1', '-10', STRUCT]])
        root = generate_iwrap_xml(data)
        assert _structure_areas(root) == []
