"""Standalone tests for the audit / Choices & Deltas renderer.

Run with::

    /c/OSGeo4W/apps/Python312/python.exe -m pytest -p no:qgis \
        --noconftest tests/test_audit_report.py -v
"""
from __future__ import annotations

import pytest

from omrat_utils.audit_report import (
    build_choices_and_deltas_markdown,
    parse_point_wkt,
    haversine_m,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_parse_point_wkt_handles_z():
    assert parse_point_wkt("Point (12.5 55.3)") == pytest.approx((12.5, 55.3))
    assert parse_point_wkt("POINT Z (12.5 55.3 0)") == pytest.approx((12.5, 55.3))
    assert parse_point_wkt("garbage") is None
    assert parse_point_wkt("") is None


def test_haversine_m_known_distance():
    # The two southern-junction positions from ore2 vs ore3.
    p_ore2 = (12.718924, 55.510670)
    p_ore3 = (12.718554, 55.503552)
    d = haversine_m(p_ore2, p_ore3)
    # Should be roughly 790-820 m (latitude shift ~0.0071 degrees).
    assert 750 < d < 850


# ---------------------------------------------------------------------------
# Minimal synthetic project — exercises every branch deterministically.
# ---------------------------------------------------------------------------
def _baseline_data() -> dict:
    """A 'default everything' project that should produce zero deltas."""
    from compute.iwrap_defaults import IWRAP_PC_DEFAULTS, IWRAP_DRIFT_DEFAULTS, IWRAP_ROSE_DEFAULT, IWRAP_REPAIR_DEFAULT
    return {
        'pc': dict(IWRAP_PC_DEFAULTS),
        'drift': {
            **{k: v for k, v in IWRAP_DRIFT_DEFAULTS.items()
               if k not in ('rose', 'repair')},
            'rose': dict(IWRAP_ROSE_DEFAULT),
            'repair': dict(IWRAP_REPAIR_DEFAULT),
        },
        'segment_data': {
            '1': {
                'Start_Point': 'Point (12.0 55.0)',
                'End_Point': 'Point (12.1 55.1)',
            },
        },
        'segments_imported': {
            '1': {
                'Start_Point': 'Point (12.0 55.0)',
                'End_Point': 'Point (12.1 55.1)',
            },
        },
        'traffic_data': {'1': {'East going': {}, 'West going': {}}},
        'depths': [['1', '20', 'POLYGON ((...))']],
        'objects': [],
        'junctions': {},
        'consequence': {
            'oil_onboard': [[100.0, 100.0]],
            'spill_probability': [[97, 1, 1, 1]],
            'spill_fraction': [[0, 10, 30, 100]],
            'catastrophe_levels': [
                {'name': 'Minor', 'quantity': 50.0},
                {'name': 'Major', 'quantity': 500.0},
                {'name': 'Catastrophic', 'quantity': 5000.0},
            ],
        },
    }


def test_baseline_renders_no_pc_deltas():
    md = build_choices_and_deltas_markdown(_baseline_data())
    assert "All causation factors match IWRAP defaults." in md
    # No warning glyphs in pc rows.
    pc_block = md.split("### Drift parameters")[0]
    assert "⚠" not in pc_block


def test_modified_pc_shows_delta():
    d = _baseline_data()
    d['pc']['overtaking'] = 6.0e-5  # mirror ore3
    d['pc']['crossing'] = 1.0e-4
    md = build_choices_and_deltas_markdown(d)
    assert "pc.overtaking" in md
    assert "−45%" in md or "-45%" in md
    assert "pc.crossing" in md
    assert "−23%" in md or "-23%" in md


def test_drift_speed_and_rose_changes_flagged():
    d = _baseline_data()
    d['drift']['speed'] = 1.4
    d['drift']['rose'] = {
        '0': 0.1022, '45': 0.1022, '90': 0.157, '135': 0.1022,
        '180': 0.1022, '225': 0.1022, '270': 0.2298, '315': 0.1022,
    }
    md = build_choices_and_deltas_markdown(d)
    assert "Drift speed (kn)" in md
    assert "+40%" in md
    assert "non-uniform" in md
    assert "270" in md  # dominant direction surfaced


def test_waypoint_move_flagged_in_geometry_section():
    d = _baseline_data()
    # Move End_Point ~800 m south (~0.0072 deg lat).
    d['segment_data']['1']['End_Point'] = 'Point (12.1 55.0928)'
    md = build_choices_and_deltas_markdown(d)
    assert "End Point" in md
    # Distance should be ~800 m -- accept any 700–900 m.
    assert any(f"{m} m" in md for m in range(700, 900))


def test_baseline_geometry_section_when_no_import_snapshot():
    d = _baseline_data()
    d['segments_imported'] = {}  # legacy project
    md = build_choices_and_deltas_markdown(d)
    assert "No imported-geometry baseline on file." in md


def test_missing_depths_warns_about_zero_grounding():
    d = _baseline_data()
    d['depths'] = []
    md = build_choices_and_deltas_markdown(d)
    assert "drifting/powered grounding will be 0" in md


def test_ships_without_speed_or_draught_say_how_they_were_filled():
    """KNOWN_ISSUES 14: the report says which ships had no speed / draught
    and what the run used instead (the snapshot is taken before the fill)."""
    inf = float('inf')
    d = _baseline_data()
    md = build_choices_and_deltas_markdown(d)
    assert "| Ships without speed | none ✓ |" in md
    assert "| Ships without draught | none ✓ |" in md
    d['traffic_data'] = {
        '1': {'East going': {'Frequency (ships/year)': [[4, 6]], 'Speed (knots)': [[inf, 12.0]],
                             'Draught (meters)': [[5.0, 7.0]]}},
        '2': {'East going': {'Frequency (ships/year)': [[3, 0]], 'Speed (knots)': [[11.0, inf]],
                             'Draught (meters)': [[6.0, inf]]}},
    }
    md = build_choices_and_deltas_markdown(d)
    assert ("| Ships without speed | ⚠ 4 ships/yr in 1 traffic cell(s); filled with the traffic-weighted "
            "mean speed of the other legs (same type and length: 4) |") in md
    assert "| Ships without draught | none ✓ |" in md
    assert d['traffic_data']['1']['East going']['Speed (knots)'][0][0] == inf   # report never fills
