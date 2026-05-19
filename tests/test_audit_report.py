"""Standalone tests for the audit / Choices & Deltas renderer.

Run with::

    /c/OSGeo4W/apps/Python312/python.exe -m pytest -p no:qgis \
        --noconftest tests/test_audit_report.py -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from omrat_utils.audit_report import (
    build_choices_and_deltas_markdown,
    parse_point_wkt,
    haversine_m,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


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


# ---------------------------------------------------------------------------
# Real .omrat files: ore1 / ore2 / ore3.
# ---------------------------------------------------------------------------
ORE_FILES = {
    'ore1': REPO_ROOT / 'ore1_20260518_061939.omrat',
    'ore2': REPO_ROOT / 'ore2_20260518_062542.omrat',
    'ore3': REPO_ROOT / 'ore3_20260518_064027.omrat',
}


def _load(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


@pytest.mark.skipif(
    not ORE_FILES['ore1'].exists(),
    reason="ore* fixture files not present in working tree",
)
def test_ore1_audit_calls_out_missing_depths():
    md = build_choices_and_deltas_markdown(_load(ORE_FILES['ore1']))
    # ore1 had 0 depths -> grounding silently zero; audit should call it out.
    assert "drifting/powered grounding will be 0" in md
    # ore1 used default pc -- no deltas.
    pc_block = md.split("### Drift parameters")[0]
    assert "⚠" not in pc_block


@pytest.mark.skipif(
    not ORE_FILES['ore2'].exists(),
    reason="ore* fixture files not present in working tree",
)
def test_ore2_audit_clean_pc_but_complete_inputs():
    md = build_choices_and_deltas_markdown(_load(ORE_FILES['ore2']))
    # ore2 has depths now -> no "will be 0" warning.
    assert "drifting/powered grounding will be 0" not in md
    # Still default pc.
    pc_block = md.split("### Drift parameters")[0]
    assert "⚠" not in pc_block


@pytest.mark.skipif(
    not ORE_FILES['ore3'].exists(),
    reason="ore* fixture files not present in working tree",
)
def test_ore3_audit_flags_overtaking_and_crossing_pc_changes():
    md = build_choices_and_deltas_markdown(_load(ORE_FILES['ore3']))
    # ore3 dropped overtaking 1.1e-4 -> 6e-5 (-45%) and crossing 1.3e-4 -> 1e-4 (-23%).
    assert "pc.overtaking" in md
    assert "pc.crossing" in md
    # The bend / grounding / allision rows still match defaults.
    # We at least expect two delta warnings (overtaking + crossing) in pc.
    pc_block = md.split("### Drift parameters")[0]
    assert pc_block.count("⚠") >= 2


@pytest.mark.skipif(
    not ORE_FILES['ore3'].exists(),
    reason="ore* fixture files not present in working tree",
)
def test_ore3_audit_reports_no_imported_baseline_when_legacy():
    # ore3 was saved before segments_imported was introduced — the audit
    # should report a missing baseline rather than crash.
    md = build_choices_and_deltas_markdown(_load(ORE_FILES['ore3']))
    assert "No imported-geometry baseline on file." in md


@pytest.mark.skipif(
    not all(p.exists() for p in ORE_FILES.values()),
    reason="ore* fixture files not present in working tree",
)
def test_dump_ore_audit_sections(capsys):
    """Render each project so a human can eyeball the output.

    Not a strict assertion — kept so the rendered Markdown gets printed
    by ``pytest -s`` for visual sanity-checking.
    """
    for name, path in ORE_FILES.items():
        md = build_choices_and_deltas_markdown(_load(path))
        print(f"\n=== {name} =====================================================\n")
        print(md)
    assert capsys.readouterr().out.count("Choices & Deltas") == 3
