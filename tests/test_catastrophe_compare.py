"""QGIS-fixture tests for the catastrophe table on the Run Analysis tab.

* ``_auto_save_run`` writes a ``<stem>.consequence.json`` sidecar next to
  the per-run GeoPackage;
* ``TWCatastropheResults`` shows saved runs only (mirroring the accident
  table, 2026-11-09): the ticked baseline once, values only, then each
  other selected run with its exceedance and its change, read from that
  sidecar; no baseline -> values only;
* runs without a sidecar show a dash; an empty selection leaves only the
  two fixed columns.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DASH = '—'


class _FakeSettings:
    store: dict = {}

    def value(self, key, default=None, type=None):  # noqa: A002
        return self.store.get(key, default)

    def setValue(self, key, value):
        self.store[key] = value

    def remove(self, key):
        self.store.pop(key, None)


@pytest.fixture
def history(tmp_path, monkeypatch):
    """Route every ``RunHistory()`` in the mixins to a temp sqlite file."""
    import omrat_utils.run_history as rh
    import omrat_utils.run_history_mixin as rhm
    real = rh.RunHistory
    db = tmp_path / 'history.sqlite'
    monkeypatch.setattr(rh, 'RunHistory', lambda db_path=None: real(db))
    _FakeSettings.store = {}
    monkeypatch.setattr(rhm, 'QSettings', _FakeSettings)
    return real(db)


def _result(scale: float) -> dict:
    return {
        'levels': [
            {'name': 'Minor', 'quantity': 50.0, 'exceedance': 1e-3 * scale},
            {'name': 'Major', 'quantity': 500.0, 'exceedance': 2e-4 * scale},
            {'name': 'Catastrophic', 'quantity': 5000.0, 'exceedance': 3e-5 * scale},
        ],
        'total_spill_frequency': 1e-3 * scale,
        'by_accident': {},
    }


def _save_run_with_sidecar(history, out_dir: Path, name: str, result: dict | None) -> int:
    filename = f'{name}_20260101_000000.gpkg'
    run_id = history.save_run(name, totals={}, output_dir=str(out_dir), output_filename=filename)
    if result is not None:
        with (out_dir / f'{name}_20260101_000000.consequence.json').open('w', encoding='utf-8') as f:
            json.dump(result, f)
    return run_id


def _row_of(tw, run_id) -> int:
    from qgis.PyQt.QtCore import Qt
    for row in range(tw.rowCount()):
        if int(tw.item(row, 0).data(Qt.ItemDataRole.UserRole)) == run_id:
            return row
    raise AssertionError(f'run {run_id} not in table')


def _select(omrat, *run_ids):
    tw_runs = omrat.main_widget.TWPreviousRuns
    tw_runs.clearSelection()
    from qgis.PyQt.QtCore import QItemSelectionModel
    sel = tw_runs.selectionModel()
    for rid in run_ids:
        row = _row_of(tw_runs, rid)
        sel.select(
            tw_runs.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )
    omrat._on_previous_runs_selection_changed()


class TestSidecarWriter:
    def test_writes_consequence_json(self, omrat, tmp_path):
        calc = SimpleNamespace(consequence_result=_result(1.0))
        path = omrat._write_consequence_report_sidecar(
            calc_object=calc, out_dir=tmp_path, base_filename='run_20260101_000000.gpkg',
        )
        assert path == tmp_path / 'run_20260101_000000.consequence.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        assert payload['levels'][1]['name'] == 'Major'
        assert payload['levels'][1]['exceedance'] == pytest.approx(2e-4)

    def test_skips_when_no_result(self, omrat, tmp_path):
        calc = SimpleNamespace(consequence_result=None)
        assert omrat._write_consequence_report_sidecar(
            calc_object=calc, out_dir=tmp_path, base_filename='x.gpkg',
        ) is None
        assert not list(tmp_path.iterdir())

    def test_loader_roundtrip_and_missing(self, tmp_path):
        from omrat_utils.run_history import RunMeta
        from omrat_utils.run_history_mixin import load_consequence_sidecar
        run = RunMeta(run_id=1, name='r', timestamp='t', output_dir=str(tmp_path),
                      output_filename='r_20260101_000000.gpkg')
        assert load_consequence_sidecar(run) is None
        (tmp_path / 'r_20260101_000000.consequence.json').write_text(
            json.dumps(_result(1.0)), encoding='utf-8',
        )
        assert load_consequence_sidecar(run)['levels'][0]['name'] == 'Minor'
        assert load_consequence_sidecar(RunMeta(run_id=2, name='x', timestamp='t')) is None


def _tick_baseline(omrat, run_id) -> None:
    """Tick Baseline on the run's row, as a user would."""
    from qgis.PyQt.QtCore import Qt
    tw = omrat.main_widget.TWPreviousRuns
    tw.item(_row_of(tw, run_id), 1).setCheckState(Qt.CheckState.Checked)


def _headers(tw) -> list[str]:
    return [tw.horizontalHeaderItem(c).text() for c in range(tw.columnCount())]


FIXED = ['Catastrophe level', 'Threshold (m^3)']


class TestCatastropheCompare:
    def test_selected_run_fills_rows_and_values(self, omrat, history, tmp_path):
        omrat._baseline_runs = None
        rid = _save_run_with_sidecar(history, tmp_path, 'test14', _result(2.0))
        omrat.refresh_previous_runs_table()
        tw = omrat.main_widget.TWCatastropheResults
        _select(omrat)
        assert tw.rowCount() == 0
        assert _headers(tw) == FIXED

        _select(omrat, rid)
        assert _headers(tw) == FIXED + ['test14']     # no baseline: values only
        assert [tw.item(r, 0).text() for r in range(tw.rowCount())] == ['Minor', 'Major', 'Catastrophic']
        assert tw.item(0, 1).text() == '50.00'
        assert tw.item(0, 2).text() == '2.000e-03'
        assert tw.item(1, 2).text() == '4.000e-04'

        _select(omrat)
        assert tw.rowCount() == 0
        assert _headers(tw) == FIXED

    def test_baseline_once_then_value_and_change(self, omrat, history, tmp_path):
        omrat._baseline_runs = None
        id_base = _save_run_with_sidecar(history, tmp_path, 'base-run', _result(1.0))
        id_other = _save_run_with_sidecar(history, tmp_path, 'other', _result(3.0))
        omrat.refresh_previous_runs_table()
        _tick_baseline(omrat, id_base)
        tw = omrat.main_widget.TWCatastropheResults

        _select(omrat, id_other, id_base)
        assert _headers(tw) == FIXED + ['base-run (baseline)', 'other', 'Change (%)']
        assert tw.item(0, 2).text() == '1.000e-03'
        assert tw.item(0, 3).text() == '3.000e-03'
        for r in range(3):
            assert tw.item(r, 4).text() == '+200.0%'

    def test_baseline_shown_without_a_selection(self, omrat, history, tmp_path):
        omrat._baseline_runs = None
        id_base = _save_run_with_sidecar(history, tmp_path, 'base-run', _result(1.0))
        omrat.refresh_previous_runs_table()
        _tick_baseline(omrat, id_base)
        _select(omrat)
        tw = omrat.main_widget.TWCatastropheResults
        assert _headers(tw) == FIXED + ['base-run (baseline)']
        assert tw.rowCount() == 3

    def test_run_without_sidecar_shows_dashes(self, omrat, history, tmp_path):
        omrat._baseline_runs = None
        id_base = _save_run_with_sidecar(history, tmp_path, 'base', _result(1.0))
        rid_old = _save_run_with_sidecar(history, tmp_path, 'old', None)
        omrat.refresh_previous_runs_table()
        _tick_baseline(omrat, id_base)
        tw = omrat.main_widget.TWCatastropheResults

        _select(omrat, rid_old)
        assert _headers(tw) == FIXED + ['base (baseline)', 'old', 'Change (%)']
        assert 'Re-run' in tw.horizontalHeaderItem(3).toolTip()
        for r in range(3):
            assert tw.item(r, 3).text() == DASH
            assert tw.item(r, 4).text() == DASH

    def test_two_runs_and_an_extra_level_row(self, omrat, history, tmp_path):
        omrat._baseline_runs = None
        res_b = _result(1.0)
        res_b['levels'].append({'name': 'Extreme', 'quantity': 50000.0, 'exceedance': 1e-6})
        rid_a = _save_run_with_sidecar(history, tmp_path, 'a', _result(1.0))
        rid_b = _save_run_with_sidecar(history, tmp_path, 'b', res_b)
        omrat.refresh_previous_runs_table()
        tw = omrat.main_widget.TWCatastropheResults

        _select(omrat, rid_a, rid_b)
        assert _headers(tw) == FIXED + ['b', 'a']     # table order: newest first
        assert tw.rowCount() == 4
        assert tw.item(3, 0).text() == 'Extreme'
        assert tw.item(3, 1).text() == '50000.00'
        assert tw.item(3, 2).text() == '1.000e-06'
        assert tw.item(3, 3).text() == DASH           # a has no such level

    def test_follows_the_display_mode(self, omrat, history, tmp_path):
        omrat._baseline_runs = None
        id_base = _save_run_with_sidecar(history, tmp_path, 'base', _result(1.0))
        id_other = _save_run_with_sidecar(history, tmp_path, 'other', _result(2.0))
        omrat.refresh_previous_runs_table()
        _tick_baseline(omrat, id_base)
        _select(omrat, id_other)
        w = omrat.main_widget
        tw = w.TWCatastropheResults
        assert tw.item(0, 4).text() == '+100.0%'

        w.cbResultDisplayMode.setCurrentIndex(1)
        assert tw.item(0, 2).text() == '1,000'        # 1 / 1e-3 years
        assert tw.item(0, 3).text() == '500'
        assert tw.item(0, 4).text() == '-50.0%'       # change of the years
        assert 'years between' in w.lblCatastropheResults.text()

        w.cbResultDisplayMode.setCurrentIndex(0)
        assert tw.item(0, 2).text() == '1.000e-03'
        assert tw.item(0, 4).text() == '+100.0%'
        assert 'events/year' in w.lblCatastropheResults.text()

    def test_copy_table_with_headers(self, omrat, history, tmp_path, monkeypatch):
        omrat._baseline_runs = None
        id_base = _save_run_with_sidecar(history, tmp_path, 'base', _result(1.0))
        id_other = _save_run_with_sidecar(history, tmp_path, 'other', _result(2.0))
        monkeypatch.setattr(omrat, '_decimal_point', lambda: '.')   # drawn under '.'
        omrat.refresh_previous_runs_table()
        _tick_baseline(omrat, id_base)
        _select(omrat, id_other)
        monkeypatch.setattr(omrat, '_decimal_point', lambda: ',')   # copied under ','
        tw = omrat.main_widget.TWCatastropheResults
        lines = omrat._result_table_tsv(tw, whole=True).split('\n')
        assert lines[0].split('\t') == FIXED + ['base (baseline)', 'other', 'Change (%)']
        assert lines[1].split('\t') == ['Minor', '50', '0,001', '0,002', '100']
        assert lines[2].split('\t') == ['Major', '500', '0,0002', '0,0004', '100']
