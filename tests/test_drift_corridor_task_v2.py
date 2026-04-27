"""Unit tests for ``geometries.drift_corridor_task_v2.DriftCorridorTask``.

Mirrors the pattern used by ``test_calculation_task.py`` -- the task is
driven directly (without involving the QGIS task manager) against a
mocked generator.
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
    from geometries.drift_corridor_task_v2 import DriftCorridorTask
    gen = MagicMock()
    gen.generate_corridors = MagicMock(return_value=[
        {'id': 'c1'}, {'id': 'c2'},
    ])
    gen.set_progress_callback = MagicMock()
    t = DriftCorridorTask(
        'corridor test', gen,
        depth_threshold=-5.0,
        height_threshold=20.0,
        target_prob=1e-3,
    )
    return t, gen


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_stores_parameters(self, task):
        t, gen = task
        assert t.generator is gen
        assert t.depth_threshold == -5.0
        assert t.height_threshold == 20.0
        assert t.target_prob == 1e-3
        assert t.exception is None
        assert t.error_msg is None
        assert t.corridors == []


# ---------------------------------------------------------------------------
# run() -- happy path
# ---------------------------------------------------------------------------

class TestRunSuccess:
    def test_generator_called_with_stored_thresholds(self, task):
        t, gen = task
        assert t.run() is True
        gen.generate_corridors.assert_called_once_with(-5.0, 20.0, 1e-3)

    def test_corridors_stored_on_task(self, task):
        t, _ = task
        t.run()
        assert t.corridors == [{'id': 'c1'}, {'id': 'c2'}]

    def test_progress_callback_injected(self, task):
        t, gen = task
        t.run()
        gen.set_progress_callback.assert_called_once()
        wrapper = gen.set_progress_callback.call_args.args[0]
        assert callable(wrapper)

    def test_progress_wrapper_emits_signal(self, task):
        t, gen = task
        received = []
        t.progress_updated.connect(lambda c, tot, m: received.append((c, tot, m)))
        t.run()
        wrapper = gen.set_progress_callback.call_args.args[0]
        assert wrapper(5, 10, "half") is True
        assert ('half' in msg for _, _, msg in received)

    def test_progress_wrapper_returns_false_when_cancelled(self, task, monkeypatch):
        t, gen = task
        t.run()
        wrapper = gen.set_progress_callback.call_args.args[0]
        monkeypatch.setattr(t, 'isCanceled', lambda: True)
        assert wrapper(1, 2, "cancel") is False

    def test_zero_total_in_progress_is_safe(self, task):
        t, gen = task
        t.run()
        wrapper = gen.set_progress_callback.call_args.args[0]
        # Should not divide by zero.
        assert wrapper(0, 0, "idle") is True


# ---------------------------------------------------------------------------
# run() -- cancellation and exceptions
# ---------------------------------------------------------------------------

class TestRunFailureModes:
    def test_cancel_after_generate_returns_false(self, qgis_app):
        from geometries.drift_corridor_task_v2 import DriftCorridorTask
        state = {'done': False}
        gen = MagicMock()
        def fake_gen(*a, **k):
            state['done'] = True
            return [{'id': 'x'}]
        gen.generate_corridors.side_effect = fake_gen

        t = DriftCorridorTask('x', gen, -5.0, 20.0)
        t.isCanceled = lambda: state['done']
        assert t.run() is False

    def test_exception_captured(self, qgis_app):
        from geometries.drift_corridor_task_v2 import DriftCorridorTask
        gen = MagicMock()
        gen.generate_corridors.side_effect = RuntimeError('gen broke')

        t = DriftCorridorTask('x', gen, -5.0, 20.0)
        assert t.run() is False
        assert isinstance(t.exception, RuntimeError)
        assert 'gen broke' in (t.error_msg or '')


# ---------------------------------------------------------------------------
# finished() / cancel()
# ---------------------------------------------------------------------------

class TestFinishedAndCancel:
    def test_success_emits_corridors(self, task):
        t, _ = task
        t.corridors = [{'id': 'a'}]
        seen = []
        t.corridors_generated.connect(lambda lst: seen.append(lst))
        t.finished(True)
        assert seen == [[{'id': 'a'}]]

    def test_failure_emits_generation_failed(self, task):
        t, _ = task
        t.error_msg = 'boom'
        t.isCanceled = lambda: False
        seen = []
        t.generation_failed.connect(lambda m: seen.append(m))
        t.finished(False)
        assert seen == ['boom']

    def test_failure_without_error_msg_uses_default(self, task):
        t, _ = task
        t.error_msg = None
        t.isCanceled = lambda: False
        seen = []
        t.generation_failed.connect(lambda m: seen.append(m))
        t.finished(False)
        assert seen == ['Unknown error']

    def test_cancelled_does_not_emit_failed(self, task):
        t, _ = task
        t.isCanceled = lambda: True
        seen = []
        t.generation_failed.connect(lambda m: seen.append(m))
        t.finished(False)
        assert seen == []

    def test_cancel_flips_is_canceled(self, task):
        t, _ = task
        assert t.isCanceled() is False
        t.cancel()
        assert t.isCanceled() is True
