"""QGIS-fixture tests for the catastrophe table on the Run Analysis tab.

* ``_auto_save_run`` writes a ``<stem>.consequence.json`` sidecar next to
  the per-run GeoPackage;
* selecting rows in ``TWPreviousRuns`` adds one exceedance + one delta
  column per run to ``TWCatastropheResults`` (mirroring the accident
  table), read from that sidecar;
* runs without a sidecar show a dash; clearing the selection restores the
  live three-column layout.
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


class TestCatastropheCompare:
    def test_selected_run_fills_columns_from_sidecar(self, omrat, history, tmp_path):
        omrat._main_run_id = None
        rid = _save_run_with_sidecar(history, tmp_path, 'test14', _result(2.0))
        omrat.refresh_previous_runs_table()
        # No live result (nothing run this session) -> rows come from the run.
        omrat._populate_catastrophe_results_table(None)
        tw = omrat.main_widget.TWCatastropheResults
        assert tw.rowCount() == 0

        _select(omrat, rid)

        assert tw.columnCount() == 5
        assert tw.horizontalHeaderItem(3).text() == 'test14'
        assert tw.horizontalHeaderItem(4).text() == 'Δ vs test14 %'
        assert [tw.item(r, 0).text() for r in range(tw.rowCount())] == ['Minor', 'Major', 'Catastrophic']
        assert tw.item(0, 1).text() == '50.00'
        assert tw.item(0, 2).text() == DASH          # no live value
        assert tw.item(0, 3).text() == '2.000e-03'
        assert tw.item(1, 3).text() == '4.000e-04'
        assert tw.item(0, 4).text() == '+0.0%'       # baseline is the run itself
        omrat._reset_catastrophe_table_to_base()

    def test_live_result_is_baseline_and_restored_on_clear(self, omrat, history, tmp_path):
        omrat._main_run_id = None
        rid = _save_run_with_sidecar(history, tmp_path, 'other', _result(2.0))
        omrat.refresh_previous_runs_table()
        omrat._populate_catastrophe_results_table(_result(1.0))
        tw = omrat.main_widget.TWCatastropheResults
        assert tw.columnCount() == 3 and tw.rowCount() == 3
        assert tw.item(0, 2).text() == '1.000e-03'

        _select(omrat, rid)
        assert tw.columnCount() == 5
        assert tw.horizontalHeaderItem(4).text() == 'Δ vs current %'
        for r in range(3):
            assert tw.item(r, 4).text() == '+100.0%'

        omrat.main_widget.TWPreviousRuns.clearSelection()
        omrat._on_previous_runs_selection_changed()
        assert tw.columnCount() == 3
        assert tw.rowCount() == 3
        assert tw.item(0, 2).text() == '1.000e-03'
        assert tw.horizontalHeaderItem(2).text() == 'Exceedance (events/year)'

    def test_main_run_is_baseline(self, omrat, history, tmp_path):
        omrat._main_run_id = None
        id_main = _save_run_with_sidecar(history, tmp_path, 'main-run', _result(1.0))
        id_other = _save_run_with_sidecar(history, tmp_path, 'other', _result(3.0))
        omrat._set_main_run_id(id_main)
        omrat.refresh_previous_runs_table()
        omrat._populate_catastrophe_results_table(None)
        tw = omrat.main_widget.TWCatastropheResults

        _select(omrat, id_other)
        assert tw.horizontalHeaderItem(4).text() == 'Δ vs main (main-run) %'
        for r in range(3):
            assert tw.item(r, 4).text() == '+200.0%'
        omrat._reset_catastrophe_table_to_base()

    def test_run_without_sidecar_shows_dashes(self, omrat, history, tmp_path):
        omrat._main_run_id = None
        rid_old = _save_run_with_sidecar(history, tmp_path, 'old', None)
        omrat.refresh_previous_runs_table()
        omrat._populate_catastrophe_results_table(_result(1.0))
        tw = omrat.main_widget.TWCatastropheResults

        _select(omrat, rid_old)
        assert tw.columnCount() == 5
        assert tw.horizontalHeaderItem(3).text() == 'old'
        assert 'Re-run' in tw.horizontalHeaderItem(3).toolTip()
        for r in range(3):
            assert tw.item(r, 3).text() == DASH
            assert tw.item(r, 4).text() == DASH
        omrat._reset_catastrophe_table_to_base()

    def test_two_runs_two_column_pairs_and_extra_level_row(self, omrat, history, tmp_path):
        omrat._main_run_id = None
        res_b = _result(1.0)
        res_b['levels'].append({'name': 'Extreme', 'quantity': 50000.0, 'exceedance': 1e-6})
        rid_a = _save_run_with_sidecar(history, tmp_path, 'a', _result(1.0))
        rid_b = _save_run_with_sidecar(history, tmp_path, 'b', res_b)
        omrat.refresh_previous_runs_table()
        omrat._populate_catastrophe_results_table(_result(1.0))
        tw = omrat.main_widget.TWCatastropheResults

        _select(omrat, rid_a, rid_b)
        assert tw.columnCount() == 7
        assert tw.rowCount() == 4
        assert tw.item(3, 0).text() == 'Extreme'
        assert tw.item(3, 1).text() == '50000.00'
        assert tw.item(3, 2).text() == DASH        # live has no such level
        col_b = 3 if tw.horizontalHeaderItem(3).text() == 'b' else 5
        col_a = 8 - col_b
        assert tw.item(3, col_b).text() == '1.000e-06'
        assert tw.item(3, col_a).text() == DASH
        assert tw.item(3, col_b + 1).text() == DASH  # no baseline for that level

        omrat.main_widget.TWPreviousRuns.clearSelection()
        omrat._on_previous_runs_selection_changed()
        assert tw.rowCount() == 3 and tw.columnCount() == 3
