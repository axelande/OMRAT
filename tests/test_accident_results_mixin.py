"""Unit tests for ``omrat_utils.accident_results_mixin`` pure-data parts.

The mixin's behaviour falls into three groups:

* ``_VIEW_DISPATCH`` and ``_ACCIDENT_ROWS`` -- compile-time dispatch
  tables that the runtime relies on; verify each accident row maps to
  a slot that lives on the mixin.
* ``_dispatch_view`` -- pure dispatch logic, exercisable with a stub
  ``self`` object that fakes ``calc`` and the selected-run helpers.
* The other methods either touch Qt widgets (``_setup_*``,
  ``_populate_*``) or fire popups, both of which need a Qt event loop
  via the QGIS conftest fixture and therefore can't run here.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Static dispatch tables
# ---------------------------------------------------------------------------
class TestDispatchTables:
    def _import(self):
        from omrat_utils.accident_results_mixin import AccidentResultsMixin
        return AccidentResultsMixin

    def test_every_row_has_a_view_dispatch_entry(self):
        Mix = self._import()
        for _label, _le_name, _pb_name, slot_name in Mix._ACCIDENT_ROWS:
            assert slot_name in Mix._VIEW_DISPATCH, (
                f"Slot {slot_name!r} from _ACCIDENT_ROWS is not wired "
                "in _VIEW_DISPATCH"
            )

    def test_every_view_dispatch_slot_has_a_method(self):
        Mix = self._import()
        for slot_name in Mix._VIEW_DISPATCH:
            assert callable(getattr(Mix, slot_name, None)), (
                f"_VIEW_DISPATCH advertises slot {slot_name!r} but the "
                "mixin has no callable with that name."
            )

    def test_dispatch_methods_match_calc_method_names(self):
        Mix = self._import()
        # The calc-method strings we hand off to ``getattr(self.calc, ...)``
        # must be plausibly real names -- guard against typos by checking
        # every entry is a non-empty identifier.
        for slot_name, (method_name, label, _key) in Mix._VIEW_DISPATCH.items():
            assert method_name and method_name.replace('_', '').isalnum(), (
                f"Slot {slot_name!r} has a malformed calc-method name "
                f"{method_name!r}"
            )
            assert label, f"Slot {slot_name!r} has empty popup label"


# ---------------------------------------------------------------------------
# Dispatcher wiring
# ---------------------------------------------------------------------------
class _FakeHost:
    """Concrete subclass of ``AccidentResultsMixin`` for direct testing.

    A ``MagicMock`` stand-in for ``self`` doesn't work because the
    ``self._VIEW_DISPATCH.get(...)`` lookup returns another mock; we
    need real attribute resolution against the class-level dict.
    """

    def __init__(self):
        self._calc = MagicMock()
        self._calc.collision_report = {'live': True}
        self._selected = MagicMock()
        self._inputs = (
            {'segment_data': {}},
            {'overtaking': {}},
            {'by_leg_direction': {}},
        )
        self.main_widget = MagicMock()
        self.tr = lambda s: s

    @property
    def calc(self):
        return self._calc

    @calc.setter
    def calc(self, value):
        self._calc = value

    def _require_single_selected_run(self):
        return self._selected

    def _load_run_inputs_and_collision_report(self, run):
        return self._inputs

    def _selected_run_ids(self):
        return ['fake']


def _make_host():
    """A concrete object with the mixin grafted on so ``_dispatch_view``
    sees the class-level ``_VIEW_DISPATCH``."""
    from omrat_utils.accident_results_mixin import AccidentResultsMixin

    class _Host(_FakeHost, AccidentResultsMixin):
        pass

    return _Host()


class TestDispatchView:
    """Exercise ``_dispatch_view`` without a Qt event loop."""

    def test_unknown_slot_is_a_noop(self):
        host = _make_host()
        host._dispatch_view('nonexistent_slot')
        host.calc.run_drift_visualization.assert_not_called()

    def test_drift_route_uses_run_drift_visualization(self):
        host = _make_host()
        host._dispatch_view('show_drift_allision')
        host.calc.run_drift_visualization.assert_called_once()
        host.calc.run_collision_breakdown_dialog.assert_not_called()

    def test_collision_route_uses_breakdown_dialog(self):
        host = _make_host()
        host._dispatch_view('show_overtaking_collision')
        host.calc.run_collision_breakdown_dialog.assert_called_once_with(
            'overtaking',
        )

    def test_collision_route_restores_live_report(self):
        host = _make_host()
        host._inputs = (
            {'segment_data': {}}, {'crossing': 1.0}, None,
        )
        host.calc.collision_report = {'live': True}
        host._dispatch_view('show_crossing_collision')
        # After the call the live report is restored.
        assert host.calc.collision_report == {'live': True}

    def test_no_run_selected_short_circuits(self):
        host = _make_host()
        host._selected = None
        host._dispatch_view('show_drift_grounding')
        host.calc.run_drift_grounding_visualization.assert_not_called()

    def test_missing_calc_short_circuits(self):
        host = _make_host()
        host.calc = None
        # Should silently no-op without raising.
        host._dispatch_view('show_drift_grounding')
