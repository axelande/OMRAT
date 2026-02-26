"""Tests for the get_type() AIS TOC-to-UI-category mapping."""
import pytest

from omrat_utils.handle_ais import get_type


# fmt: off
@pytest.mark.parametrize("toc, expected_index, label", [
    # 0: Fishing
    (30,  0,  "Fishing"),
    # 1: Towing
    (31,  1,  "Towing"),
    (32,  1,  "Towing"),
    # 2: Dredging or underwater ops
    (33,  2,  "Dredging or underwater ops"),
    # 3: Diving ops
    (34,  3,  "Diving ops"),
    # 4: Military ops
    (35,  4,  "Military ops"),
    # 5: Sailing
    (36,  5,  "Sailing"),
    # 6: Pleasure Craft
    (37,  6,  "Pleasure Craft"),
    # 7: High speed craft (HSC) — full range 40-49
    (40,  7,  "HSC lower bound"),
    (44,  7,  "HSC mid"),
    (49,  7,  "HSC upper bound"),
    # 8: Pilot Vessel
    (50,  8,  "Pilot Vessel"),
    # 9: Search and Rescue vessel
    (51,  9,  "SAR vessel"),
    # 10: Tug
    (52, 10,  "Tug"),
    # 11: Port Tender
    (53, 11,  "Port Tender"),
    # 12: Anti-pollution equipment
    (54, 12,  "Anti-pollution equipment"),
    # 13: Law Enforcement
    (55, 13,  "Law Enforcement"),
    # 14: Spare
    (56, 14,  "Spare"),
    (57, 14,  "Spare"),
    # 15: Medical Transport
    (58, 15,  "Medical Transport"),
    # 16: Noncombatant ship
    (59, 16,  "Noncombatant ship"),
    # 17: Passenger — full range 60-69
    (60, 17,  "Passenger lower bound"),
    (65, 17,  "Passenger mid"),
    (69, 17,  "Passenger upper bound"),
    # 18: Cargo — full range 70-79
    (70, 18,  "Cargo lower bound"),
    (75, 18,  "Cargo mid"),
    (79, 18,  "Cargo upper bound"),
    # 19: Tanker — full range 80-89
    (80, 19,  "Tanker lower bound"),
    (85, 19,  "Tanker mid"),
    (89, 19,  "Tanker upper bound"),
    # 20: Other Type — everything else
    (0,  20,  "Other (0)"),
    (20, 20,  "Other (WIG 20)"),
    (29, 20,  "Other (WIG 29)"),
    (38, 20,  "Other (reserved 38)"),
    (39, 20,  "Other (reserved 39)"),
    (90, 20,  "Other (90)"),
    (99, 20,  "Other (99)"),
])
# fmt: on
def test_get_type(toc, expected_index, label):
    assert get_type(toc) == expected_index, f"Failed for TOC {toc} ({label})"


def test_get_type_accepts_float():
    """TOC values from the database may arrive as floats."""
    assert get_type(30.0) == 0
    assert get_type(80.0) == 19
    assert get_type(70.5) == 18
