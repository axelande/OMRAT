"""QGIS-fixture tests for the sensitivity-analysis dialog and headless runner.

Needs the conftest (``QgsApplication``): run *without* ``--noconftest``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.sensitivity import (  # noqa: E402
    ACCIDENT_KEYS, ALL_PHASES, BaselineReports, build_parameter_specs,
)
from omrat_utils.storage import Storage  # noqa: E402

EXAMPLE = ROOT / 'tests' / 'example_data' / 'proj.omrat'


@pytest.fixture
def loaded(omrat):
    Storage(omrat).load_from_path(str(EXAMPLE))
    omrat.main_widget.LEModelName.setText('sens_test')
    return omrat


def _run_task_inline(task) -> None:
    """Stand-in for the QGIS task manager: run + finish on this thread."""
    ok = task.run()
    task.finished(ok)


def test_button_is_wired_and_dialog_opens(loaded):
    assert hasattr(loaded.main_widget, 'pbSensitivity')
    loaded.open_sensitivity_dialog()
    dlg = loaded._sensitivity_dialog
    assert dlg is not None and dlg.isVisible()
    try:
        specs = dlg.selected_specs()
        assert specs, 'everything ticked by default'
        assert {s.group for s in specs} >= {'Causation factors', 'Traffic', 'Drift settings', 'Legs'}
        # No finished run in this session -> only "recompute" is offered.
        assert not dlg.rbReuse.isEnabled() and dlg.rbRecompute.isChecked()
        dlg._select_instant()
        assert all(s.analytic for s in dlg.selected_specs())
        assert '0 need model re-runs' in dlg.lblPlan.text()
        # Second click raises the same dialog instead of opening another.
        loaded.open_sensitivity_dialog()
        assert loaded._sensitivity_dialog is dlg
    finally:
        dlg.close()


def test_headless_phases_leave_the_tab_untouched(loaded):
    from omrat_utils.gather_data import GatherData
    from omrat_utils.sensitivity_task import run_phases_headless

    data = GatherData(loaded).get_all_for_save()
    before = loaded.main_widget.LEPDriftAllision.text()
    seen: list[tuple] = []
    reports = run_phases_headless(
        json.loads(json.dumps(data)), ('collision', 'powered_allision'),
        on_progress=lambda *args: seen.append(args))
    assert set(reports.totals) == set(ACCIDENT_KEYS)
    # In-phase progress is forwarded with (phase_idx, n_phases, done, total, msg).
    assert seen and all(len(t) == 5 and t[1] == 2 for t in seen)
    assert {t[0] for t in seen} == {0, 1}
    assert any('starting' in t[4] for t in seen)
    assert all(math.isfinite(v) for v in reports.totals.values())
    # Only the requested phases produced numbers.
    assert reports.totals['drift_allision'] == 0.0
    assert reports.totals['drift_grounding'] == 0.0
    # The plugin's own result line-edits were not written to.
    assert loaded.main_widget.LEPDriftAllision.text() == before


def test_dialog_runs_analysis_and_writes_report(loaded, tmp_path):
    from omrat_utils.sensitivity_dialog import SensitivityDialog
    from omrat_utils.gather_data import GatherData

    data = GatherData(loaded).get_all_for_save()
    data['project_name'] = 'sens_test'
    specs = build_parameter_specs(data)
    baseline = BaselineReports(
        totals={k: 1e-3 for k in ACCIDENT_KEYS},
        powered_cat={'powered_grounding': {'cat1': 4e-4, 'cat2': 6e-4}},
        by_cell={'head_on': {'0_0': 5e-4, '1_0': 5e-4}},
    )
    dlg = SensitivityDialog(
        loaded.main_widget, data, specs, live_baseline=baseline,
        out_dir=tmp_path, run_name='sens_test', task_starter=_run_task_inline,
    )
    try:
        assert dlg.rbReuse.isEnabled() and dlg.rbReuse.isChecked()
        dlg._select_instant()
        dlg.sbDelta.setValue(10)
        dlg.start()
        # Inline starter -> finished synchronously.
        assert dlg._task is None
        result = dlg._result
        assert result is not None and not result.cancelled
        assert {r.spec.key for r in result.results} == {s.key for s in specs if s.analytic}
        assert dlg.table.rowCount() == len(result.results)
        # Ranked: first row has the largest swing; header shows the delta.
        assert dlg.table.horizontalHeaderItem(4).text() == 'At -10 %'
        first = dlg.table.item(0, 1).text()
        assert first.startswith('Traffic volume')
        # Re-ranking by a single accident type works without re-running.
        idx = dlg.cbOutput.findData('head_on')
        dlg.cbOutput.setCurrentIndex(idx)
        labels = [dlg.table.item(r, 1).text() for r in range(dlg.table.rowCount())]
        assert labels[0] in ('Traffic volume (all ships)',)
        assert dlg.table.item(0, 8).text() == '+2.00'  # elasticity of Q on head-on
        # Report written to the output folder.
        written = sorted(tmp_path.glob('sens_test_sensitivity_*'))
        assert [p.suffix for p in written] == ['.json', '.md']
        md = written[1].read_text(encoding='utf-8')
        assert '# OMRAT sensitivity analysis' in md
        assert 'Causation factor: head-on' in md
        payload = json.loads(written[0].read_text(encoding='utf-8'))
        assert payload['format'] == 'omrat-sensitivity/1'
        assert payload['delta'] == pytest.approx(0.1)
        assert dlg.pbSave.isEnabled() and dlg.pbPlot.isEnabled()
    finally:
        dlg.close()


def test_task_cancel_before_start_reports_cancelled(loaded):
    from omrat_utils.sensitivity_task import SensitivityTask

    task = SensitivityTask({'segment_data': {}}, [], 0.2, baseline=None)
    got: list = []
    task.analysis_finished.connect(got.append)
    task.cancel()
    task.finished(False)
    assert len(got) == 1 and got[0].cancelled and got[0].results == []
    assert ALL_PHASES == ('drifting', 'collision', 'powered_grounding', 'powered_allision')
