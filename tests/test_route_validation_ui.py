"""Standalone test for the route-validation-pass driver.

The QGIS-specific zoom + dialog-exec paths are bypassed by passing a
``show_dialog`` callable to :func:`run_validation_pass`, so the test can
exercise the orchestration without spinning up Qt or the canvas.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _seg(start: str, end: str, length_m: float = 100_000) -> dict:
    return {
        'Start_Point': start,
        'End_Point': end,
        'line_length': length_m,
        'Width': 5000,
        'Route_Id': 1,
        'Leg_name': 'LEG',
    }


def _make_omrat_stub(segment_data: dict, traffic_data: dict | None = None):
    return SimpleNamespace(
        segment_data=segment_data,
        traffic_data=traffic_data or {},
        main_widget=None,
        iface=None,
        junctions=None,
    )


def test_run_validation_pass_applies_user_choice_for_merge():
    from geometries.route_validation import parse_wkt_point
    from omrat_utils.route_validation_ui import (
        ValidationOutcome,
        run_validation_pass,
        _MergeChoice,
    )

    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0"),
        '2': _seg("15.0001 55.0", "16.0 55.0"),
    }
    omrat = _make_omrat_stub(sd)

    # Stub dialog: always pick point_a for merges; always split crossings.
    def fake(kind, payload):
        if kind == 'merge':
            return _MergeChoice(target=payload.point_a)
        return True

    outcome = run_validation_pass(omrat, show_dialog=fake)
    assert outcome.merges_applied == 1
    # Verify the merge actually happened.
    assert parse_wkt_point(sd['2']['Start_Point']) == (14.0, 55.0) or \
           parse_wkt_point(sd['2']['Start_Point']) == (15.0, 55.0)


def test_run_validation_pass_skips_when_user_cancels():
    from omrat_utils.route_validation_ui import (
        run_validation_pass, _MergeChoice,
    )
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0"),
        '2': _seg("15.0001 55.0", "16.0 55.0"),
    }
    omrat = _make_omrat_stub(sd)

    def fake(kind, payload):
        if kind == 'merge':
            return _MergeChoice(target=None)  # skip
        return False  # also skip splits

    outcome = run_validation_pass(omrat, show_dialog=fake)
    assert outcome.merges_applied == 0
    assert outcome.splits_applied == 0
    assert outcome.skipped >= 1


def test_run_validation_pass_splits_x_crossing():
    from omrat_utils.route_validation_ui import run_validation_pass
    sd = {
        '1': _seg("14.0 55.0", "16.0 56.0", 200_000),
        '2': _seg("14.0 56.0", "16.0 55.0", 200_000),
    }
    omrat = _make_omrat_stub(sd)
    outcome = run_validation_pass(
        omrat,
        show_dialog=lambda kind, payload: True if kind == 'split' else None,
    )
    assert outcome.splits_applied == 1
    assert len(sd) == 4  # original 2 + 2 new sub-legs


def test_validation_pass_handles_clean_project():
    from omrat_utils.route_validation_ui import run_validation_pass
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0"),
        '2': _seg("15.0 55.0", "16.0 55.0"),
    }
    omrat = _make_omrat_stub(sd)
    calls = []

    def fake(kind, payload):
        calls.append(kind)
        return None

    outcome = run_validation_pass(omrat, show_dialog=fake)
    assert outcome.merges_applied == 0
    assert outcome.splits_applied == 0
    assert calls == []  # no prompts shown for a clean project


def test_validation_refreshes_junction_handler_after_edits():
    """The handler.rebuild_from_segments path is exercised when present."""
    from omrat_utils.handle_junctions import Junctions
    from omrat_utils.route_validation_ui import (
        run_validation_pass, _MergeChoice,
    )
    sd = {
        '1': _seg("14.0 55.0", "15.0 55.0"),
        '2': _seg("15.0001 55.0", "16.0 55.0"),
    }
    omrat = _make_omrat_stub(sd)
    omrat.junctions = Junctions(omrat)
    omrat.junctions.rebuild_from_segments()
    assert len(omrat.junctions) == 0  # before merge: no junction
    run_validation_pass(
        omrat,
        show_dialog=lambda kind, payload: (
            _MergeChoice(target=payload.point_a) if kind == 'merge' else False
        ),
    )
    assert len(omrat.junctions) == 1  # after merge: real junction
