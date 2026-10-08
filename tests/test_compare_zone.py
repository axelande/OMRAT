"""Pure tests for the Compare tab's common-zone comparison (``omrat_utils.compare_zone``)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.compare_zone import build_zone_table, zone_from_structures, zone_values  # noqa: E402

PARK = 'POLYGON ((16.80 56.00, 16.81 56.00, 16.81 56.01, 16.80 56.01, 16.80 56.00))'


def _traffic(q):
    return {'East going': {'Frequency (ships/year)': [[q]]}, 'West going': {'Frequency (ships/year)': [[0.0]]}}


def _snap(objects=True, suppress_far=False):
    seg = {
        '1': {'Leg_name': 'IN', 'Start_Point': '16.700000 56.000000', 'End_Point': '16.900000 56.000000'},
        '2': {'Leg_name': 'HALF', 'Start_Point': '16.900000 56.000000', 'End_Point': '18.900000 56.000000'},
        '3': {'Leg_name': 'FAR', 'Start_Point': '19.000000 57.000000', 'End_Point': '19.500000 57.000000',
              'suppressed': suppress_far},
    }
    return {
        'segment_data': seg,
        'traffic_data': {'1': _traffic(1000.0), '2': _traffic(2000.0), '3': _traffic(5000.0)},
        'objects': [['1', '0', PARK]] if objects else [],
    }


OUTPUTS = {
    'collision': {
        'by_leg': {'1': {'head_on': 1e-3, 'overtaking': 0.0}, '2': {'head_on': 4e-3, 'overtaking': 2e-3},
                   '3': {'head_on': 9.0, 'overtaking': 9.0}},
        'by_waypoint': {'16.900000 56.000000': {'crossing': 3e-3, 'merging': 1e-3, 'bend': 0.0},
                        '18.900000 56.000000': {'crossing': 9.0, 'merging': 0.0, 'bend': 0.0}},
    },
    'drifting': {'totals': {'allision': 0.01, 'grounding': 0.5}},
    'powered': {'allision': {'totals': {'allision': 2e-4}}},
}


def _length_km(a, b):
    """Length in the zone's own projection (centred on the structures)."""
    (x1, y1), (x2, y2) = PROJ(*a), PROJ(*b)
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / 1000.0


ZONE, PROJ, N_STRUCT = zone_from_structures(_snap(), None, 20.0)
VALS = zone_values(_snap(), OUTPUTS, ZONE, PROJ)


class TestZoneValues:

    def test_zone_built_from_structures(self):
        assert N_STRUCT == 1
        assert ZONE.area / 1e6 == pytest.approx(3.1416 * 20 ** 2, rel=0.08)

    def test_leg_length_and_traffic_work(self):
        len_in = _length_km((16.7, 56.0), (16.9, 56.0))
        inside_half = VALS['leg_km'] - len_in
        assert 0 < inside_half < _length_km((16.9, 56.0), (18.9, 56.0)) / 2
        expected_work = (1000.0 * len_in + 2000.0 * inside_half) / 1e6
        assert VALS['ship_km_m'] == pytest.approx(expected_work)

    def test_leg_collisions_scaled_by_share_inside(self):
        len_in = _length_km((16.7, 56.0), (16.9, 56.0))
        share_half = (VALS['leg_km'] - len_in) / _length_km((16.9, 56.0), (18.9, 56.0))
        assert VALS['head_on'] == pytest.approx(1e-3 + 4e-3 * share_half, rel=1e-6)
        assert VALS['overtaking'] == pytest.approx(2e-3 * share_half, rel=1e-6)

    def test_waypoints_inside_only(self):
        assert VALS['crossing'] == pytest.approx(3e-3)
        assert VALS['merging'] == pytest.approx(1e-3)

    def test_rate_and_totals(self):
        assert VALS['rate'] == pytest.approx(VALS['collisions'] / VALS['ship_km_m'])
        assert VALS['total'] == pytest.approx(VALS['collisions'] + 0.01 + 2e-4)

    def test_suppressed_leg_left_out_of_traffic_work(self):
        snap = _snap()
        snap['segment_data']['1']['suppressed'] = True
        vals = zone_values(snap, OUTPUTS, ZONE, PROJ)
        assert vals['ship_km_m'] < VALS['ship_km_m']

    def test_missing_collision_sidecar_gives_none(self):
        vals = zone_values(_snap(), {}, ZONE, PROJ)
        assert vals['head_on'] is None and vals['collisions'] is None and vals['total'] is None
        assert vals['leg_km'] == pytest.approx(VALS['leg_km'])


class TestZoneTable:
    def test_zone_from_run_b_only_still_compares_both(self):
        """The run without the park gets the same zone as the run with it."""
        rows, info = build_zone_table(_snap(objects=False), _snap(), OUTPUTS, OUTPUTS, 20.0)
        assert info.startswith('Zone: within 20 km of 1 structure(s)')
        by = {r[0]: r for r in rows}
        assert by['Leg length in zone (km)'][1] == by['Leg length in zone (km)'][2]
        assert by['Crossing collision in zone'][3] == '+0.000e+00'

    def test_structure_in_both_runs_counts_once(self):
        _rows, info = build_zone_table(_snap(), _snap(), OUTPUTS, OUTPUTS, 20.0)
        assert info.startswith('Zone: within 20 km of 1 structure(s)')

    def test_no_structures(self):
        rows, info = build_zone_table(_snap(objects=False), _snap(objects=False), OUTPUTS, OUTPUTS)
        assert rows == []
        assert 'structures' in info

    def test_radius_changes_the_result(self):
        small, _ = build_zone_table(_snap(), _snap(), OUTPUTS, OUTPUTS, 5.0)
        large, _ = build_zone_table(_snap(), _snap(), OUTPUTS, OUTPUTS, 50.0)
        km = {r[0]: r for r in small}['Leg length in zone (km)'][1]
        km_large = {r[0]: r for r in large}['Leg length in zone (km)'][1]
        assert float(km.replace(',', '')) < float(km_large.replace(',', ''))
