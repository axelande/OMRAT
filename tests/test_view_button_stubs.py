"""Wire-up tests for the 5 "View" buttons on the main widget.

Each ``show_X`` slot now delegates to
:meth:`AccidentResultsMixin._dispatch_view`, which:

  1. asks the user to pick a single run from the Previous-runs table
     (popping up a ``QMessageBox`` when nothing/too many rows are
     selected), and
  2. on success, hands the loaded run data to the matching visualiser
     on ``self.calc``.

The tests below verify that:

  * each slot dispatches to ``_dispatch_view`` with its own slot name
    (so renaming a slot can never silently leave a button orphaned),
  * each button widget's ``clicked`` signal actually fires the slot.

The earlier version of these tests asserted a ``Qgis.Warning`` log
entry written by a placeholder stub.  Those stubs have been replaced
with the real dispatch path, so the log assertion no longer applies.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_VIEW_SLOTS = [
    ("show_drift_grounding",      "Drifting grounding"),
    ("show_overtaking_collision", "Overtaking collision"),
    ("show_head_on_collision",    "Head-on collision"),
    ("show_crossing_collision",   "Crossing collision"),
    ("show_merging_collision",    "Merging collision"),
]


class TestViewSlotDispatch:
    @pytest.mark.parametrize("slot_name, _label", _VIEW_SLOTS)
    def test_slot_routes_through_dispatch_view(self, omrat, slot_name, _label):
        """Calling ``omrat.show_X()`` must call ``_dispatch_view('show_X')``.

        Patching ``_dispatch_view`` short-circuits the rest of the chain
        (which otherwise pops up a ``QMessageBox`` to ask the user to
        pick a run from an empty Previous-runs table).
        """
        with patch.object(omrat, '_dispatch_view') as dispatch:
            getattr(omrat, slot_name)()
        dispatch.assert_called_once_with(slot_name)


class TestViewButtonsWiredUp:
    """Each row's "View" button (column 2 of TWAccidentResults) must fire
    its slot.  The legacy ``pbView*`` widgets on ``main_widget`` are
    hollow stubs kept only so old code that calls ``LEPDriftAllision``
    etc. still resolves; the live, wired-up buttons live inside the
    table cells."""

    @pytest.mark.parametrize("slot_name, label", _VIEW_SLOTS)
    def test_button_click_triggers_slot(self, omrat, slot_name, label):
        from omrat_utils.accident_results_mixin import AccidentResultsMixin

        # Find the row whose label matches this parametrisation.
        row = next(
            i for i, spec in enumerate(AccidentResultsMixin._ACCIDENT_ROWS)
            if spec[0] == label
        )
        tw = omrat.main_widget.TWAccidentResults
        button = tw.cellWidget(row, 2)
        assert button is not None, (
            f"Row {row} ({label!r}) has no cellWidget — the View button "
            f"was never created."
        )
        with patch.object(omrat, '_dispatch_view') as dispatch:
            button.click()
        assert dispatch.call_count >= 1, (
            f"Clicking the View button on row {row} ({label!r}) did not "
            f"reach _dispatch_view — the clicked signal is probably not "
            f"wired to {slot_name!r}."
        )
        # And the dispatch was given THIS slot's name (not a sibling's,
        # which would silently route to the wrong visualiser).
        assert dispatch.call_args.args[0] == slot_name
