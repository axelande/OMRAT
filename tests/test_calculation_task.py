"""Unit tests for ``compute.calculation_task.CalculationTask``.

``CalculationTask`` is a :class:`QgsTask` wrapper that orchestrates the
four calculation phases (drifting, ship collisions, powered grounding,
powered allision) while propagating cancellation and progress through
the QGIS task manager.  These tests run the task logic directly
(without involving the real task manager) using mocked ``calc`` objects.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def task(qgis_app):
    from compute.calculation_task import CalculationTask
    calc = MagicMock()
    calc.run_drifting_model = MagicMock(return_value=(1e-5, 2e-5))
    calc.run_ship_collision_model = MagicMock(return_value={})
    calc.run_powered_grounding_model = MagicMock(return_value=3e-5)
    calc.run_powered_allision_model = MagicMock(return_value=4e-5)
    calc.set_progress_callback = MagicMock()
    t = CalculationTask("test-desc", calc, {'any': 'payload'})
    return t, calc


# ---------------------------------------------------------------------------
# __init__ / simple accessors
# ---------------------------------------------------------------------------

class TestInit:
    def test_stores_calc_and_data(self, task):
        t, calc = task
        assert t.calc is calc
        assert t.data == {'any': 'payload'}
        assert t.exception is None
        assert t.error_msg is None

    def test_set_progress_callback_stores_callable(self, task):
        t, _ = task
        cb = MagicMock()
        t.set_progress_callback(cb)
        assert t._progress_callback is cb

    def test_update_description_prefixes_omrat(self, task):
        t, _ = task
        t._update_description("phase 1")
        # QgsTask.description() should now include the OMRAT prefix.
        assert 'OMRAT' in t.description()
        assert 'phase 1' in t.description()


# ---------------------------------------------------------------------------
# run() -- happy path
# ---------------------------------------------------------------------------

class TestRunSuccess:
    def test_all_four_phases_invoked(self, task):
        t, calc = task
        assert t.run() is True
        calc.run_drifting_model.assert_called_once_with({'any': 'payload'})
        calc.run_ship_collision_model.assert_called_once()
        calc.run_powered_grounding_model.assert_called_once()
        calc.run_powered_allision_model.assert_called_once()

    def test_progress_wrapper_injected_when_calc_supports_it(self, task):
        t, calc = task
        t.run()
        calc.set_progress_callback.assert_called_once()
        # The injected wrapper is a callable.
        wrapper = calc.set_progress_callback.call_args.args[0]
        assert callable(wrapper)

    def test_progress_wrapper_emits_signal_and_returns_true(self, task):
        t, calc = task
        received = []

        def on_progress(completed, total, msg):
            received.append((completed, total, msg))

        t.progress_updated.connect(on_progress)
        t.run()
        # Grab the wrapper and call it.
        wrapper = calc.set_progress_callback.call_args.args[0]
        assert wrapper(5, 10, "halfway") is True
        assert any(msg == "halfway" for _, _, msg in received)

    def test_progress_wrapper_returns_false_when_cancelled(self, task, monkeypatch):
        """If the task is cancelled, the injected wrapper tells the calc to stop."""
        t, calc = task
        t.run()
        wrapper = calc.set_progress_callback.call_args.args[0]
        # Simulate cancellation by forcing isCanceled to return True.
        monkeypatch.setattr(t, 'isCanceled', lambda: True)
        assert wrapper(1, 100, "cancel!") is False

    def test_progress_wrapper_handles_zero_total(self, task):
        """If `total` is 0 the wrapper must not divide by zero."""
        t, calc = task
        t.run()
        wrapper = calc.set_progress_callback.call_args.args[0]
        # Should not raise.
        assert wrapper(0, 0, "idle") is True


# ---------------------------------------------------------------------------
# run() -- cancellation between phases
# ---------------------------------------------------------------------------

class TestRunCancellation:
    def test_cancel_after_drifting_returns_false(self, qgis_app):
        """Cancel mid-flight: the task returns False and short-circuits."""
        from compute.calculation_task import CalculationTask
        cancel_state = {'cancelled': False}

        calc = MagicMock()
        def fake_drifting(data):
            cancel_state['cancelled'] = True
            return (0.0, 0.0)
        calc.run_drifting_model.side_effect = fake_drifting

        t = CalculationTask("test", calc, {})
        # After the first phase we flip isCanceled() to return True.
        original_is_canceled = t.isCanceled
        def is_canceled_now():
            return cancel_state['cancelled']
        t.isCanceled = is_canceled_now

        result = t.run()
        assert result is False
        calc.run_drifting_model.assert_called_once()
        # Phase 2 (ship collisions) must NOT be invoked after cancel.
        calc.run_ship_collision_model.assert_not_called()

    def test_cancel_after_ship_collisions(self, qgis_app):
        """Cancel after ship-collision phase short-circuits before powered."""
        from compute.calculation_task import CalculationTask
        phases_done = {'count': 0}

        calc = MagicMock()
        def track(_):
            phases_done['count'] += 1
            return None
        calc.run_drifting_model.side_effect = track
        calc.run_ship_collision_model.side_effect = track
        calc.run_powered_grounding_model.side_effect = track

        t = CalculationTask("test", calc, {})
        # Cancel only after phase 2 has completed.
        def is_canceled_now():
            return phases_done['count'] >= 2
        t.isCanceled = is_canceled_now

        assert t.run() is False
        # Drifting + ship collisions ran; powered phases did not.
        calc.run_drifting_model.assert_called_once()
        calc.run_ship_collision_model.assert_called_once()
        calc.run_powered_grounding_model.assert_not_called()


# ---------------------------------------------------------------------------
# run() -- exception handling
# ---------------------------------------------------------------------------

class TestRunExceptions:
    def test_drifting_raises_captures_and_returns_false(self, qgis_app):
        from compute.calculation_task import CalculationTask
        calc = MagicMock()
        calc.run_drifting_model.side_effect = RuntimeError("drift failed")

        t = CalculationTask("test", calc, {})
        assert t.run() is False
        assert isinstance(t.exception, RuntimeError)
        assert "drift failed" in (t.error_msg or '')

    def test_calc_without_progress_callback_still_runs(self, qgis_app):
        """A calc object missing set_progress_callback is fine."""
        from compute.calculation_task import CalculationTask

        class Bare:
            """Does not inherit MagicMock, so no auto-set_progress_callback."""
            def run_drifting_model(self, data): return (0, 0)
            def run_ship_collision_model(self, data): return {}
            def run_powered_grounding_model(self, data): return 0
            def run_powered_allision_model(self, data): return 0

        t = CalculationTask("test", Bare(), {})
        assert t.run() is True


# ---------------------------------------------------------------------------
# finished()
# ---------------------------------------------------------------------------

class TestFinished:
    def test_success_emits_calculation_finished(self, task):
        t, calc = task
        seen = []
        t.calculation_finished.connect(lambda obj: seen.append(obj))
        t.finished(True)
        assert seen == [calc]

    def test_failure_emits_calculation_failed_with_error(self, task):
        t, _ = task
        t.error_msg = "boom"
        # Make sure isCanceled reads False -> the "failed" branch fires.
        t.isCanceled = lambda: False
        seen = []
        t.calculation_failed.connect(lambda msg: seen.append(msg))
        t.finished(False)
        assert seen == ["boom"]

    def test_failure_without_error_msg_uses_default(self, task):
        t, _ = task
        t.error_msg = None
        t.isCanceled = lambda: False
        seen = []
        t.calculation_failed.connect(lambda msg: seen.append(msg))
        t.finished(False)
        assert seen == ["Unknown error"]

    def test_cancelled_does_not_emit_failed(self, task):
        """If the task was cancelled, calculation_failed must NOT fire."""
        t, _ = task
        t.isCanceled = lambda: True
        fail_calls = []
        t.calculation_failed.connect(lambda msg: fail_calls.append(msg))
        t.finished(False)
        assert fail_calls == []


# ---------------------------------------------------------------------------
# cancel()
# ---------------------------------------------------------------------------

class TestCancel:
    def test_cancel_flips_is_canceled(self, task):
        t, _ = task
        assert t.isCanceled() is False
        t.cancel()
        # QgsTask.cancel() sets the internal flag.
        assert t.isCanceled() is True
