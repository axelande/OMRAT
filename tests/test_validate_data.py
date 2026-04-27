"""Unit tests for the pydantic schema in omrat_utils/validate_data.py.

These verify that the schema classes accept plausible .omrat payloads
and reject malformed ones, so schema drift between the GUI save path
and the on-disk format surfaces at test time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.validate_data import (
    PC, Rose, Repair, Drift, TrafficDirectionData, TrafficLeg, TrafficData,
    Segment, SegmentData, PolygonEntry, Depths, Objects,
    LengthInterval, ShipCategoriesModel, RootModelSchema,
)


def _sample_drift():
    return {
        'drift_p': 1,
        'anchor_p': 0.7,
        'anchor_d': 7,
        'speed': 1.94,
        'rose': {str(a): 0.125 for a in (0, 45, 90, 135, 180, 225, 270, 315)},
        'repair': {
            'func': 'norm.cdf(x)', 'std': 1.0, 'loc': 0.0, 'scale': 1.0,
            'use_lognormal': False,
        },
    }


def _sample_segment():
    return {
        'Start_Point': '14 55', 'End_Point': '15 55',
        'Dirs': ['East going', 'West going'], 'Width': 1000,
        'line_length': 100.0, 'Route_Id': 1, 'Leg_name': 'leg 1', 'Segment_Id': '1',
        'mean1_1': 0.0, 'std1_1': 100.0, 'weight1_1': 1.0,
        'mean2_1': 0.0, 'std2_1': 100.0, 'weight2_1': 1.0,
        'mean1_2': 0.0, 'std1_2': 0.0, 'weight1_2': 0.0,
        'mean1_3': 0.0, 'std1_3': 0.0, 'weight1_3': 0.0,
        'mean2_2': 0.0, 'std2_2': 0.0, 'weight2_2': 0.0,
        'mean2_3': 0.0, 'std2_3': 0.0, 'weight2_3': 0.0,
        'u_min1': 0.0, 'u_max1': 0.0, 'u_p1': 0, 'ai1': 180.0,
        'u_min2': 0.0, 'u_max2': 0.0, 'u_p2': 0, 'ai2': 180.0,
    }


def _sample_direction_data():
    return {
        'Frequency (ships/year)': [[0.0]],
        'Speed (knots)': [[0.0]],
        'Draught (meters)': [[0.0]],
        'Ship heights (meters)': [[0.0]],
        'Ship Beam (meters)': [[0.0]],
    }


# ---------------------------------------------------------------------------
# PC: causation factors
# ---------------------------------------------------------------------------

class TestPC:
    def test_accepts_minimal(self):
        pc = PC(p_pc=0.1, d_pc=0.2)
        assert pc.p_pc == 0.1
        assert pc.d_pc == 0.2
        # defaults populated
        assert pc.headon == 4.9e-5
        assert pc.grounding == 1.6e-4

    def test_overrides_defaults(self):
        pc = PC(p_pc=0.1, d_pc=0.2, headon=1e-3)
        assert pc.headon == 1e-3

    def test_missing_required_raises(self):
        with pytest.raises(ValidationError):
            PC(d_pc=0.2)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Rose / Repair / Drift
# ---------------------------------------------------------------------------

class TestRoseRepairDrift:
    def test_rose_accepts_any_string_keyed_floats(self):
        r = Rose.model_validate({'0': 0.125, 'foo': 0.1})
        assert r.root == {'0': 0.125, 'foo': 0.1}

    def test_repair_requires_all_fields(self):
        Repair(func='x', std=1.0, loc=0.0, scale=1.0, use_lognormal=False)
        with pytest.raises(ValidationError):
            Repair(std=1.0, loc=0.0, scale=1.0, use_lognormal=False)  # type: ignore[call-arg]

    def test_drift_accepts_sample(self):
        Drift.model_validate(_sample_drift())

    def test_drift_rejects_wrong_types(self):
        bad = _sample_drift()
        bad['anchor_p'] = 'not-a-number'
        with pytest.raises(ValidationError):
            Drift.model_validate(bad)


# ---------------------------------------------------------------------------
# Traffic nested models
# ---------------------------------------------------------------------------

class TestTraffic:
    def test_direction_data_aliases(self):
        d = TrafficDirectionData.model_validate(_sample_direction_data())
        assert d.Frequency_ships_per_year == [[0.0]]

    def test_leg_requires_both_directions(self):
        with pytest.raises(ValidationError):
            TrafficLeg.model_validate({'East going': _sample_direction_data()})

    def test_full_traffic_data_roundtrip(self):
        leg = {'East going': _sample_direction_data(), 'West going': _sample_direction_data()}
        td = TrafficData.model_validate({'1': leg, '2': leg})
        assert set(td.root.keys()) == {'1', '2'}


# ---------------------------------------------------------------------------
# Segment / SegmentData
# ---------------------------------------------------------------------------

class TestSegment:
    def test_accepts_sample(self):
        Segment.model_validate(_sample_segment())

    def test_segment_data_dict_form(self):
        seg = _sample_segment()
        sd = SegmentData.model_validate({'1': seg, '2': seg})
        assert set(sd.root.keys()) == {'1', '2'}


# ---------------------------------------------------------------------------
# Depths / Objects / ShipCategories
# ---------------------------------------------------------------------------

class TestPolygonsAndShipCategories:
    def test_polygon_entry_list_of_strings(self):
        PolygonEntry.model_validate(['1', '6.0', 'POLYGON ((0 0, 1 1, 1 0, 0 0))'])

    def test_depths_wraps_list(self):
        Depths.model_validate([['1', '0.0', 'POLYGON ((0 0, 1 1, 1 0, 0 0))']])

    def test_objects_wraps_list(self):
        Objects.model_validate([['1', '10', 'POLYGON ((0 0, 1 1, 1 0, 0 0))']])

    def test_length_interval_accepts_float_or_string(self):
        LengthInterval(min=0.0, max=25.0, label='0-25')
        LengthInterval(min='', max='', label='open')

    def test_ship_categories_optional_selection_mode(self):
        sc = ShipCategoriesModel(
            types=['Cargo', 'Tanker'],
            length_intervals=[LengthInterval(min=0.0, max=25.0, label='0-25')],
        )
        assert sc.selection_mode is None


# ---------------------------------------------------------------------------
# RootModelSchema
# ---------------------------------------------------------------------------

class TestRootModelSchema:
    def _minimal_payload(self):
        return {
            'pc': {'p_pc': 0.1, 'd_pc': 0.2},
            'drift': _sample_drift(),
            'traffic_data': {
                '1': {'East going': _sample_direction_data(),
                      'West going': _sample_direction_data()},
            },
            'segment_data': {'1': _sample_segment()},
            'depths': [],
            'objects': [],
        }

    def test_valid_root_accepted(self):
        payload = self._minimal_payload()
        RootModelSchema.model_validate(payload)

    def test_ship_categories_optional(self):
        payload = self._minimal_payload()
        payload['ship_categories'] = {
            'types': ['Cargo'],
            'length_intervals': [{'min': 0.0, 'max': 25.0, 'label': '0-25'}],
        }
        RootModelSchema.model_validate(payload)

    def test_missing_required_section_raises(self):
        payload = self._minimal_payload()
        del payload['drift']
        with pytest.raises(ValidationError):
            RootModelSchema.model_validate(payload)
