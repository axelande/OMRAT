"""Standalone tests for ``compute/leg_labels.py`` (no QGIS).

Run with ``-p no:qgis --noconftest``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.leg_labels import (  # noqa: E402
    direction_name, leg_display, leg_field_map, leg_field_name, leg_label,
    leg_name, pair_label, parse_leg_key,
)

SEG = {
    '9': {'Leg_name': 'LEG_1_2_b', 'Dirs': ['East going', 'West going']},
    '8': {'Leg_name': 'LEG_3_7_b', 'Dirs': ['North going', 'South going']},
    '15': {'Leg_name': 'LEG_1_1_c', 'Dirs': ['East going', 'West going']},
    '40': {'Leg_name': 'LEG_1_2_b', 'Dirs': ['East going', 'West going']},   # duplicate name
    '41': {'Leg_name': '', 'Dirs': ['East going', 'West going']},            # blank name
}


def test_parse_leg_key_forms():
    assert parse_leg_key('9') == ('9', None, None)
    assert parse_leg_key('9:0') == ('9', '0', None)
    assert parse_leg_key('9:North going:270') == ('9', 'North going', '270')
    assert parse_leg_key(9) == ('9', None, None)


def test_names_and_directions():
    assert leg_name(SEG, '9') == 'LEG_1_2_b'
    assert leg_name(SEG, 9) == 'LEG_1_2_b'
    assert leg_name(SEG, '99') is None
    assert leg_name(SEG, '41') is None
    assert leg_name(None, '9') is None
    assert direction_name(SEG, '9', 0) == 'East going'
    assert direction_name(SEG, '9', '1') == 'West going'
    assert direction_name(SEG, '99', '1') == 'dir 1'
    assert direction_name(SEG, '9', 'North going') == 'North going'
    assert direction_name(SEG, '9', None) is None


def test_labels():
    assert leg_label(SEG, '9') == 'LEG_1_2_b'
    assert leg_label(SEG, '9:0') == 'LEG_1_2_b East going'
    assert leg_label(SEG, '8:1') == 'LEG_3_7_b South going'
    assert leg_label(SEG, '9:North going:270') == 'LEG_1_2_b North going 270 deg'
    # Unknown leg: the key itself, unchanged.
    assert leg_label(SEG, '99:0') == '99:0'
    assert leg_label({}, '9:0') == '9:0'


def test_display_appends_the_key_only_when_a_name_exists():
    assert leg_display(SEG, '9:0') == 'LEG_1_2_b East going (9:0)'
    assert leg_display(SEG, '99:0') == '99:0'


def test_field_names_are_safe_and_fall_back_to_the_old_form():
    assert leg_field_name(SEG, '9:0') == 'LEG_1_2_b_East_going'
    assert leg_field_name(SEG, '9') == 'LEG_1_2_b'
    assert leg_field_name(SEG, '99:0') == 'leg_99_0'
    assert leg_field_name(None, '7') == 'leg_7'
    assert leg_field_name({'3': {'Leg_name': 'Leg 3 (A/B)', 'Dirs': ['N', 'S']}}, '3:0') == 'Leg_3_A_B_N'


def test_field_map_is_unique_even_for_duplicate_leg_names():
    m = leg_field_map(SEG, ['9:0', '40:0', '8:1', '99:0'])
    assert m['9:0'] == 'LEG_1_2_b_East_going'
    assert m['40:0'] == 'LEG_1_2_b_East_going_id40_0'
    assert m['8:1'] == 'LEG_3_7_b_South_going'
    assert m['99:0'] == 'leg_99_0'
    assert len(set(m.values())) == 4
    # Same key twice is not a collision.
    assert leg_field_map(SEG, ['9:0', '9:0'])['9:0'] == 'LEG_1_2_b_East_going'


def test_pair_label_from_tuple_and_from_sidecar_text():
    assert pair_label(SEG, ('9', '8')) == 'LEG_1_2_b (9) x LEG_3_7_b (8)'
    assert pair_label(SEG, "('9', '8')") == 'LEG_1_2_b (9) x LEG_3_7_b (8)'
    assert pair_label(SEG, ('9', '99')) == 'LEG_1_2_b (9) x 99'
    assert pair_label(SEG, 'something else') == 'something else'
