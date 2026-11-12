"""Smallest route angle at a waypoint (compute/encounter_angle.py)."""
import math

import pytest

from compute import encounter_angle as ea
from compute.iwrap_defaults import default_pc_values


def test_default_is_ten_degrees():
    assert ea.DEFAULT_DEG == 10.0
    assert ea.min_angle_deg(None) == 10.0
    assert ea.min_angle_deg({}) == 10.0
    assert ea.min_angle_rad({}) == pytest.approx(math.radians(10.0))


def test_default_pc_values_carry_the_angle():
    assert default_pc_values()[ea.KEY] == pytest.approx(10.0)


@pytest.mark.parametrize('value, expected', [
    (5.0, 5.0), ('7.5', 7.5), (0.0, ea.MIN_DEG), (-3.0, ea.MIN_DEG), (90.0, ea.MAX_DEG),
    (None, 10.0), ('abc', 10.0), (float('nan'), 10.0),
])
def test_value_is_read_and_clamped(value, expected):
    assert ea.min_angle_deg({ea.KEY: value}) == pytest.approx(expected)


def test_limit_is_symmetric_about_parallel_and_opposite():
    lim = math.radians(10.0)
    assert not ea.angle_counts(math.radians(9.9), lim)
    assert ea.angle_counts(math.radians(10.0), lim)
    assert ea.angle_counts(math.radians(90.0), lim)
    assert ea.angle_counts(math.radians(170.0), lim)
    assert not ea.angle_counts(math.radians(170.1), lim)


def test_describe():
    assert ea.describe({}) == '10 to 170 degrees between the arrival courses'


def test_schema_default_matches():
    from omrat_utils.validate_data import PC
    assert PC(p_pc=1.6e-4, d_pc=1.0).min_encounter_angle_deg == ea.DEFAULT_DEG


def test_compare_explain_attributes_the_angle_to_causation():
    from omrat_utils.compare_explain import _PC_KEYS
    assert ea.KEY in _PC_KEYS
