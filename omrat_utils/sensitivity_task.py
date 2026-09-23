"""Background ``QgsTask`` that drives :func:`compute.sensitivity.run_sensitivity`.

Every model run of the analysis gets its own :class:`Calculation` whose
parent is a stub, so the perturbed results never reach the Run Analysis
tab's line-edits (the models write their totals to ``self.p.main_widget``
as they finish).  Only the phases a parameter can influence are run --
see ``ParameterSpec.phases``.

Signals (all delivered on the main thread):

* ``progress_updated(done, total, message)`` -- one partial run started
  or finished
* ``phase_progress(percent, message)`` -- movement *inside* a partial run,
  forwarded from the models' own progress reporting (throttled)
* ``parameter_done(ParameterResult)`` -- one parameter has both its
  ``-delta`` and ``+delta`` values, so the dialog can grow its ranking
  while the long runs continue
* ``analysis_finished(SensitivityResult)`` -- also emitted after a
  cancel, with ``result.cancelled`` set and the parameters done so far
* ``analysis_failed(message)``
"""
from __future__ import annotations

import traceback
from typing import Any, Callable, Iterable

from qgis.core import Qgis, QgsMessageLog, QgsTask
from qgis.PyQt.QtCore import pyqtSignal

from compute.sensitivity import (
    PHASE_LABELS, BaselineReports, ParameterSpec, SensitivityResult,
    reports_from_calc, run_sensitivity,
)


class _Null:
    """Swallows every attribute access / call; falsy, empty and ``''``.

    Stands in for the plugin's ``main_widget`` so ``setText(...)`` on the
    result line-edits, ``LEReportPath.text()`` and friends are no-ops.
    """

    def __getattr__(self, name: str) -> '_Null':
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> '_Null':
        return self

    def __bool__(self) -> bool:
        return False

    def __len__(self) -> int:
        return 0

    def __iter__(self):
        return iter(())

    def __str__(self) -> str:
        return ''

    def __float__(self) -> float:
        return 0.0


class StubParent:
    """Minimal stand-in for ``OMRAT`` accepted by ``Calculation``."""

    def __init__(self) -> None:
        self.main_widget = _Null()
        self.iface = _Null()
        self.testing = True


_PHASE_METHODS: tuple[tuple[str, str], ...] = (
    ('drifting', 'run_drifting_model'),
    ('collision', 'run_ship_collision_model'),
    ('powered_grounding', 'run_powered_grounding_model'),
    ('powered_allision', 'run_powered_allision_model'),
)


def run_phases_headless(
    data: dict[str, Any], phases: Iterable[str],
    is_cancelled=None,
    on_progress: Callable[[int, int, int, int, str], None] | None = None,
) -> BaselineReports:
    """Run ``phases`` on a fresh stub-parented Calculation; return its reports.

    ``data`` must already be a private copy: traffic scaling is applied in
    place exactly like ``CalculationTask.run`` does, and so are the
    suppressed-leg traffic redirects.  ``on_progress`` gets
    ``(phase_index, n_phases, done, total, message)`` straight from the
    models' own progress reporting (``done/total`` is 0..100 within the
    phase), so callers can show movement inside a long run.
    """
    from compute.data_preparation import apply_traffic_scaling
    from compute.run_calculations import Calculation
    from compute.traffic_redirect import apply_traffic_redirects

    cancelled = is_cancelled or (lambda: False)
    wanted = tuple(phases)
    ordered = [(name, method) for name, method in _PHASE_METHODS if name in wanted]
    n_phases = len(ordered)
    state = {'idx': 0}

    def _progress(done: int, total: int, message: str) -> bool:
        if on_progress is not None:
            try:
                on_progress(state['idx'], n_phases, int(done), int(total), str(message))
            except Exception:  # nosec B110 - progress display must never abort a run
                pass
        return not cancelled()

    calc = Calculation(StubParent())
    calc.set_progress_callback(_progress)
    apply_traffic_scaling(data)
    apply_traffic_redirects(data)
    for idx, (name, method) in enumerate(ordered):
        if cancelled():
            break
        state['idx'] = idx
        _progress(0, 100, f'{PHASE_LABELS[name]} - starting')
        getattr(calc, method)(data)
    return reports_from_calc(calc)


class SensitivityTask(QgsTask):
    """Run the OAT sensitivity analysis in the QGIS task manager."""

    progress_updated = pyqtSignal(int, int, str)
    phase_progress = pyqtSignal(float, str)
    parameter_done = pyqtSignal(object)
    analysis_finished = pyqtSignal(object)
    analysis_failed = pyqtSignal(str)

    def __init__(
        self,
        data: dict[str, Any],
        specs: list[ParameterSpec],
        delta: float,
        baseline: BaselineReports | None = None,
        description: str = 'OMRAT: sensitivity analysis',
    ) -> None:
        super().__init__(description, QgsTask.Flag.CanCancel)
        self.data = data
        self.specs = list(specs)
        self.delta = float(delta)
        self.baseline = baseline
        self.result: SensitivityResult | None = None
        self.error_msg: str | None = None
        # Position within the run plan, kept so in-phase progress can be
        # mapped onto the overall bar.
        self._done = 0
        self._total = max(1, len(specs))
        self._item_label = ''
        self._last_pct = -1.0
        self._last_msg = ''

    # -- worker thread -------------------------------------------------
    def _on_progress(self, done: int, total: int, message: str) -> None:
        """Called by the runner when a partial run starts / finishes."""
        self._done, self._total = int(done), max(1, int(total))
        self._item_label = str(message)
        self.setProgress(min(100.0, 100.0 * self._done / self._total))
        self.setDescription(f'OMRAT sensitivity: {message}')
        self.progress_updated.emit(self._done, self._total, str(message))

    def _on_phase_progress(self, phase_idx: int, n_phases: int, done: int, total: int, message: str) -> None:
        """Called by the models inside a partial run; maps onto the overall bar."""
        within_phase = (done / total) if total > 0 else 0.0
        within_item = (phase_idx + min(1.0, max(0.0, within_phase))) / max(1, n_phases)
        pct = min(100.0, 100.0 * (self._done + within_item) / self._total)
        if pct - self._last_pct < 0.1 and message == self._last_msg:
            return  # throttle: the cascade loop reports very often
        self._last_pct, self._last_msg = pct, message
        self.setProgress(pct)
        text = f'{self._item_label} -- {message}' if self._item_label else message
        self.setDescription(f'OMRAT sensitivity: {text}')
        self.phase_progress.emit(pct, text)

    def _run_phases(self, data: dict[str, Any], phases: tuple[str, ...]) -> BaselineReports:
        return run_phases_headless(
            data, phases, is_cancelled=self.isCanceled, on_progress=self._on_phase_progress)

    def run(self) -> bool:
        QgsMessageLog.logMessage(
            f'Sensitivity analysis started: {len(self.specs)} parameters, '
            f'+/- {self.delta * 100:.0f} %', 'OMRAT', Qgis.MessageLevel.Info)
        try:
            self.result = run_sensitivity(
                self.data, self.specs, self._run_phases, delta=self.delta,
                baseline=self.baseline, is_cancelled=self.isCanceled,
                on_progress=self._on_progress,
                on_result=self.parameter_done.emit,
            )
            if self.result.cancelled:
                QgsMessageLog.logMessage(
                    'Sensitivity analysis cancelled; keeping the parameters finished so far',
                    'OMRAT', Qgis.MessageLevel.Warning)
            else:
                self.setProgress(100)
            return True
        except Exception as exc:  # noqa: BLE001 - reported through the signal
            self.error_msg = f'{exc}\n{traceback.format_exc()}'
            QgsMessageLog.logMessage(
                f'Sensitivity analysis failed: {self.error_msg}', 'OMRAT', Qgis.MessageLevel.Critical)
            return False

    # -- main thread ---------------------------------------------------
    def finished(self, result: bool) -> None:
        if self.isCanceled():
            # Cancelled before or during the runs: hand back whatever was
            # gathered (possibly nothing) flagged as cancelled.
            if self.result is None:
                self.result = SensitivityResult(
                    delta=self.delta, baseline=self.baseline or BaselineReports())
            self.result.cancelled = True
            self.analysis_finished.emit(self.result)
        elif result and self.result is not None:
            self.analysis_finished.emit(self.result)
        else:
            self.analysis_failed.emit(self.error_msg or 'Unknown error')
