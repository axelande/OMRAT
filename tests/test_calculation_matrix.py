"""Calculation matrix (run only some model parts) + dock reopen wiring.

Needs the QGIS conftest (``omrat`` fixture).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reopen(omrat, monkeypatch, times: int = 2) -> None:
    """Close the dock and open it again, like the toolbar icon does."""
    monkeypatch.setattr(omrat, '_ask_save_on_close', lambda: 'discard')
    for _ in range(times):
        assert omrat.main_widget.close() is True
        assert omrat.pluginIsActive is False
        omrat.run()
        assert omrat.pluginIsActive is True


def _counting_message_box(calls: list):
    return SimpleNamespace(
        information=lambda *a, **k: calls.append(a),
        critical=lambda *a, **k: calls.append(a),
    )


class TestReopenWiresOnce:
    def test_update_all_distributions_fires_once_after_reopen(self, omrat, monkeypatch):
        import omrat_utils.route_validation_ui as rvu
        monkeypatch.setattr(rvu, 'run_validation_pass',
                            lambda _o: SimpleNamespace(merges_applied=0, splits_applied=0))
        _reopen(omrat, monkeypatch)
        calls = []
        monkeypatch.setattr(omrat.ais, 'update_legs', lambda *a, **k: calls.append(a))
        omrat.main_widget.pbUpdateAIS.setEnabled(True)
        omrat.main_widget.pbUpdateAIS.click()
        assert len(calls) == 1

    def test_run_model_fires_once_after_reopen(self, omrat, monkeypatch):
        import omrat as omrat_mod
        _reopen(omrat, monkeypatch)
        shown = []
        monkeypatch.setattr(omrat_mod, 'QMessageBox', _counting_message_box(shown))
        # "Already running" is the cheapest path through run_calculation.
        omrat._current_task = object()
        omrat.main_widget.pbRunModel.setEnabled(True)
        omrat.main_widget.pbRunModel.click()
        omrat._current_task = None
        assert len(shown) == 1

    def test_no_duplicate_widgets_after_reopen(self, omrat, monkeypatch):
        from qgis.PyQt.QtWidgets import QMenuBar, QPushButton
        _reopen(omrat, monkeypatch, times=3)
        add_buttons = [b for b in omrat.main_widget.findChildren(QPushButton)
                       if b.text() == 'Add selected run results to map']
        assert len(add_buttons) == 1
        assert len(omrat.main_widget.findChildren(QMenuBar)) == 1


class TestDialog:
    def test_preselects_the_given_parts(self, omrat):
        from omrat_utils.calculation_matrix_dialog import CalculationMatrixDialog
        dlg = CalculationMatrixDialog(('drifting',))
        assert dlg.selected_phases() == ('drifting',)

    def test_ok_needs_at_least_one_part(self, omrat):
        from qgis.PyQt.QtWidgets import QDialogButtonBox
        from compute.run_phases import PHASES
        from omrat_utils.calculation_matrix_dialog import CalculationMatrixDialog
        dlg = CalculationMatrixDialog(None)
        ok = dlg.buttons.button(QDialogButtonBox.StandardButton.Ok)
        assert dlg.selected_phases() == PHASES
        dlg._set_all(False)
        assert dlg.selected_phases() == ()
        assert ok.isEnabled() is False
        dlg._set_all(True)
        assert ok.isEnabled() is True

    def test_accepting_stores_the_selection(self, omrat, monkeypatch):
        import omrat_utils.calculation_matrix_dialog as cmd
        from qgis.PyQt.QtWidgets import QDialog

        class FakeDialog:
            def __init__(self, phases, parent=None):
                pass

            def exec(self):
                return QDialog.DialogCode.Accepted

            def selected_phases(self):
                return ('powered_grounding',)

        monkeypatch.setattr(cmd, 'CalculationMatrixDialog', FakeDialog)
        omrat.open_calculation_matrix()
        assert omrat.run_phases == ('powered_grounding',)


class TestRunModelButton:
    def test_partial_selection_shows_on_run_model(self, omrat, tmp_path):
        omrat.main_widget.LEModelName.setText('m')
        omrat.main_widget.LEReportPath.setText(str(tmp_path))
        omrat.set_run_phases(['powered_allision'])
        btn = omrat.main_widget.pbRunModel
        assert btn.text() == 'Run model (1 of 4 parts)'
        assert btn.isEnabled()
        assert 'Powered allision' in btn.toolTip()
        omrat.set_run_phases(None)
        assert btn.text() == 'Run model'
        assert btn.toolTip() == ''


class TestRunCalculation:
    def _prepare(self, omrat, monkeypatch, tmp_path):
        import omrat as omrat_mod
        omrat.main_widget.LEModelName.setText('m')
        omrat.main_widget.LEReportPath.setText(str(tmp_path))
        captured = {}

        def fake_task(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(omrat_mod, 'CalculationTask', fake_task)
        monkeypatch.setattr(omrat_mod, 'QgsApplication', MagicMock())
        return captured

    def test_selection_reaches_the_task(self, omrat, monkeypatch, tmp_path):
        captured = self._prepare(omrat, monkeypatch, tmp_path)
        omrat.set_run_phases(['collision', 'drifting'])
        omrat.run_calculation()
        omrat._current_task = None
        assert captured['phases'] == ('drifting', 'collision')

    def test_empty_selection_does_not_start(self, omrat, monkeypatch, tmp_path):
        import omrat as omrat_mod
        captured = self._prepare(omrat, monkeypatch, tmp_path)
        shown = []
        monkeypatch.setattr(omrat_mod, 'QMessageBox', _counting_message_box(shown))
        omrat.run_phases = ()
        omrat.run_calculation()
        assert captured == {}
        assert len(shown) == 1
        assert omrat._current_task is None

    def test_finish_blanks_the_skipped_rows(self, omrat, monkeypatch):
        mw = omrat.main_widget
        mw.LEPDriftAllision.setText('1.000e-03')
        mw.LEPHeadOnCollision.setText('3.000e-03')
        mw.LEPPoweredAllision.setText('2.000e-03')
        monkeypatch.setattr(omrat, '_auto_save_run', lambda calc: None)
        omrat._on_calculation_finished(
            SimpleNamespace(run_phases=('powered_allision',), consequence_result=None))
        assert mw.LEPDriftAllision.text() == ''
        assert mw.LEPHeadOnCollision.text() == ''
        assert mw.LEPPoweredAllision.text() == '2.000e-03'
        # Blank = not calculated; a calculated zero shows '0'.
        assert mw.TWAccidentResults.item(0, 1).text() == ''
        assert mw.TWAccidentResults.item(2, 1).text() == omrat._format_result(2e-3)


class TestCalculationState:
    def test_begin_run_forgets_the_previous_run(self, omrat):
        from compute.run_calculations import Calculation
        calc = Calculation(omrat)
        calc.drifting_report = {'totals': {'allision': 1.0}}
        calc.collision_report = {'totals': {'head_on': 1.0}}
        calc.consequence_result = {'levels': []}
        calc._last_structures = [{'id': 'old'}]
        data = {'depths': [['7', '12', 'POLYGON((0 0, 1 0, 1 1, 0 0))']]}
        calc.begin_run(data, ['powered_allision'])
        assert calc.drifting_report is None
        assert calc.collision_report is None
        assert calc.consequence_result is None
        assert calc._last_structures == []
        assert calc.run_phases == ('powered_allision',)
        # Powered grounding's layer needs these even without drifting.
        assert [d['id'] for d in calc._last_depths_original] == ['7']

    def test_drifting_without_traffic_reports_zero_not_none(self, omrat):
        from compute.run_calculations import Calculation
        calc = Calculation(omrat)
        calc.begin_run({}, None)
        calc.run_drifting_model({})
        assert calc.drifting_report == {'totals': {'allision': 0.0, 'grounding': 0.0, 'anchoring': 0.0}}

    def test_partial_run_is_no_sensitivity_baseline(self, omrat):
        omrat.calc.begin_run({}, ['powered_allision'])
        omrat.calc.powered_allision_report = {'totals': {'allision': 1.0}}
        assert omrat._live_sensitivity_baseline() is None


class TestOutputs:
    def test_report_names_the_skipped_parts(self, omrat, tmp_path):
        calc = SimpleNamespace(
            drifting_report=None, collision_report=None, powered_grounding_report=None,
            powered_allision_report={'totals': {'allision': 1e-4}, 'by_obstacle': {'s1': 1e-4}},
            run_phases=('powered_allision',),
            _last_structures=[],
            _last_powered_allision_structs=[{'id': 's1', 'height': 25.0}],
            generate_drifting_report_markdown=lambda data: 'STALE DRIFTING SECTION',
        )
        path = omrat._write_results_markdown(
            calc_object=calc, out_dir=tmp_path, base_filename='m_20261002_120000.gpkg',
            run_name='m', timestamp='2026-10-02 12:00:00', snapshot={})
        text = path.read_text(encoding='utf-8')
        assert ('Partial run (Calculation matrix): Drifting, Ship-ship collisions, '
                'Powered grounding not calculated') in text
        assert 'STALE DRIFTING SECTION' not in text
        assert '| s1 | 25.0 |' in text

    def test_view_explains_a_skipped_part(self, omrat, monkeypatch):
        import omrat_utils.accident_results_mixin as arm
        shown = []
        monkeypatch.setattr(arm, 'QMessageBox', _counting_message_box(shown))
        run = SimpleNamespace(name='r1', totals_dict=lambda: {
            'drift_allision': None, 'powered_allision': 0.0})
        assert omrat._not_calculated_in_run('show_drift_allision', run, 'Drifting allision') is True
        assert len(shown) == 1
        # Zero is a calculated value.
        assert omrat._not_calculated_in_run('show_powered_allision', run, 'Powered allision') is False
        assert len(shown) == 1
