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


# ---------------------------------------------------------------------------
# Crossings created by an edit / before Run model (2026-10-16)
# ---------------------------------------------------------------------------

def _crossed():
    """Leg 1 west-east, leg 2 south-north across it."""
    sd = {
        '1': dict(_seg("14.0 55.0", "15.0 55.0"), Leg_name='LEG_1_1'),
        '2': dict(_seg("14.5 54.5", "14.5 55.5"), Leg_name='LEG_2_1'),
    }
    return sd, _make_omrat_stub(sd)


@pytest.mark.parametrize('answer, n_legs', [(True, 4), (False, 2), (None, 2)])
def test_offer_split_new_crossings(monkeypatch, answer, n_legs):
    from omrat_utils import route_validation_ui as rvu
    sd, omrat = _crossed()
    asked = []
    monkeypatch.setattr(rvu, '_ask_split_now', lambda _om, text: asked.append(text) or answer)
    split = rvu.offer_split_new_crossings(omrat, before=set())
    assert len(asked) == 1 and 'LEG_1_1 x LEG_2_1' in asked[0]
    assert split == (1 if answer else 0)
    assert len(sd) == n_legs


def test_offer_split_ignores_crossings_that_were_there_before(monkeypatch):
    from geometries.route_validation import crossing_pairs
    from omrat_utils import route_validation_ui as rvu
    sd, omrat = _crossed()
    monkeypatch.setattr(rvu, '_ask_split_now', lambda *_a: pytest.fail("must not ask"))
    assert rvu.offer_split_new_crossings(omrat, before=crossing_pairs(sd)) == 0


def test_headless_prompts_never_block():
    from omrat_utils import route_validation_ui as rvu
    _sd, omrat = _crossed()
    omrat.testing = True
    assert rvu._ask_split_now(omrat, 'x') is None
    assert rvu._ask_run_choice(omrat, 'x') == 'run'


@pytest.mark.parametrize('choice, n_legs', [('split', 4), ('run', 2), ('cancel', 2)])
def test_check_crossings_before_run(monkeypatch, choice, n_legs):
    from omrat_utils import route_validation_ui as rvu
    sd, omrat = _crossed()
    asked = []
    monkeypatch.setattr(rvu, '_ask_run_choice', lambda _om, text: asked.append(text) or choice)
    assert rvu.check_crossings_before_run(omrat, ('collision',)) == choice
    assert len(asked) == 1 and 'counts as zero' in asked[0]
    assert len(sd) == n_legs


def test_check_crossings_only_for_collision_runs_and_live_legs(monkeypatch):
    from omrat_utils import route_validation_ui as rvu
    sd, omrat = _crossed()
    monkeypatch.setattr(rvu, '_ask_run_choice', lambda *_a: pytest.fail("must not ask"))
    assert rvu.check_crossings_before_run(omrat, ('drifting', 'powered_grounding')) == 'run'
    sd['2']['suppressed'] = True
    assert rvu.check_crossings_before_run(omrat, ('collision',)) == 'run'
