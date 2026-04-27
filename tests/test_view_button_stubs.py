"""Regression tests for the 5 stubbed "View" buttons in the main widget.

Demonstrates two new pytest-qgis helpers:
  * ``qgis_message_log`` fixture -- captures QgsMessageLog.logMessage
  * ``make_memory_layer`` -- not used here but imported to keep the
    demo import surface honest.

Each placeholder slot in omrat.py should log a Qgis.Warning; once real
visualisation pipelines are wired up, update or replace these tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from qgis.core import Qgis

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def view_stub_target(omrat, qgis_message_log):
    """Return the omrat plugin and the message-log capture for a test."""
    return omrat, qgis_message_log


class TestViewButtonStubs:
    @pytest.mark.parametrize("method_name, expected_tag", [
        ("show_drift_grounding",      "Drift grounding"),
        ("show_overtaking_collision", "Overtaking collision"),
        ("show_head_on_collision",    "Head-on collision"),
        ("show_crossing_collision",   "Crossing collision"),
        ("show_merging_collision",    "Merging collision"),
    ])
    def test_stub_logs_warning(self, view_stub_target, method_name, expected_tag):
        """Each stubbed slot logs a Qgis.Warning naming the button."""
        omrat, log = view_stub_target

        getattr(omrat, method_name)()

        entry = log.find(expected_tag, level=Qgis.Warning)
        assert entry is not None, (
            f"{method_name!r} should log a Qgis.Warning mentioning "
            f"{expected_tag!r}; captured entries: {log.entries}"
        )
        assert "not implemented" in entry.message.lower()
        assert entry.tag == "OMRAT"


class TestViewButtonsWiredUp:
    """The 5 view-button widgets must have their clicked signal wired
    to the stub slots -- otherwise the 'not implemented' warning is
    never reached."""

    @pytest.mark.parametrize("widget_name, slot_name", [
        ("pbViewDriftingGrounding",   "show_drift_grounding"),
        ("pbViewOvertakingCollision", "show_overtaking_collision"),
        ("pbViewHeadOnCollision",     "show_head_on_collision"),
        ("pbViewCrossingCollision",   "show_crossing_collision"),
        ("pbViewMergingCollision",    "show_merging_collision"),
    ])
    def test_button_click_triggers_stub(
        self, view_stub_target, widget_name, slot_name
    ):
        omrat, log = view_stub_target
        button = getattr(omrat.main_widget, widget_name)
        button.click()

        # The stub logs the "not implemented" message as soon as the
        # button is clicked, so the captured log must include a
        # Warning entry from OMRAT.
        assert any(
            entry.level == Qgis.Warning and entry.tag == "OMRAT"
            for entry in log.entries
        ), (
            f"Clicking {widget_name} did not log a Qgis.Warning from "
            f"OMRAT -- the clicked signal is probably not wired to "
            f"{slot_name!r}.  Captured: {log.entries}"
        )
