"""QGIS-fixture tests for the Run Analysis / Compare UI changes:

* ``TWPreviousRuns`` has a checkable **Baseline** column (one per
  project list); ``TWAccidentResults`` shows the baseline once (values
  only) and every other selected run with its value and its change
  against it -- no column for the live model (2026-11-09);
* the result tables copy to Excel as numbers (locale decimal point);
* ``TWAccidentResults`` carries three bold summary rows;
* loading legs from file draws the tangent-line layer;
* ``snapshot_layers.add_snapshot_to_project`` builds one ordered group.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXAMPLE = ROOT / 'tests' / 'example_data' / 'proj.omrat'


class _FakeSettings:
    """Stand-in for QSettings so tests never touch the user profile."""

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


def _totals(scale: float) -> dict:
    return {
        'drift_allision': 1e-4 * scale, 'drift_grounding': 2e-4 * scale,
        'powered_allision': 3e-4 * scale, 'powered_grounding': 4e-4 * scale,
        'overtaking': 1e-5 * scale, 'head_on': 2e-5 * scale,
        'crossing': 3e-5 * scale, 'merging': 4e-5 * scale, 'bend': 5e-5 * scale,
    }


def _row_of(tw, run_id) -> int:
    from qgis.PyQt.QtCore import Qt
    for row in range(tw.rowCount()):
        if int(tw.item(row, 0).data(Qt.ItemDataRole.UserRole)) == run_id:
            return row
    raise AssertionError(f'run {run_id} not in table')


# ---------------------------------------------------------------------------
# Baseline column
# ---------------------------------------------------------------------------

def _select(omrat, *run_ids):
    from qgis.PyQt.QtCore import QItemSelectionModel
    tw_runs = omrat.main_widget.TWPreviousRuns
    tw_runs.clearSelection()
    sel = tw_runs.selectionModel()
    for rid in run_ids:
        sel.select(
            tw_runs.model().index(_row_of(tw_runs, rid), 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )
    omrat._on_previous_runs_selection_changed()


def _tick_baseline(omrat, run_id, ticked: bool = True) -> None:
    from qgis.PyQt.QtCore import Qt
    tw = omrat.main_widget.TWPreviousRuns
    tw.item(_row_of(tw, run_id), 1).setCheckState(
        Qt.CheckState.Checked if ticked else Qt.CheckState.Unchecked)


def _headers(tw) -> list[str]:
    return [tw.horizontalHeaderItem(c).text() for c in range(tw.columnCount())]


class TestBaselineColumn:
    def test_headers_have_baseline_right_of_name(self, omrat, history):
        omrat._baseline_runs = None
        history.save_run('base', totals=_totals(1.0))
        omrat.refresh_previous_runs_table()
        tw = omrat.main_widget.TWPreviousRuns
        assert _headers(tw) == ['Name', 'Baseline', 'Date', 'Duration', 'Version', 'Project']

    def test_baseline_cells_are_checkable_and_unchecked(self, omrat, history):
        from qgis.PyQt.QtCore import Qt
        omrat._baseline_runs = None
        history.save_run('base', totals=_totals(1.0))
        omrat.refresh_previous_runs_table()
        item = omrat.main_widget.TWPreviousRuns.item(0, 1)
        assert item.flags() & Qt.ItemFlag.ItemIsUserCheckable
        assert item.checkState() == Qt.CheckState.Unchecked

    def test_ticking_sets_baseline_and_unticks_others(self, omrat, history):
        from qgis.PyQt.QtCore import Qt
        omrat._baseline_runs = None
        id_a = history.save_run('a', totals=_totals(1.0))
        id_b = history.save_run('b', totals=_totals(2.0))
        omrat.refresh_previous_runs_table()
        tw = omrat.main_widget.TWPreviousRuns
        _tick_baseline(omrat, id_a)
        assert omrat._get_baseline_run_id() == id_a
        _tick_baseline(omrat, id_b)
        assert omrat._get_baseline_run_id() == id_b
        assert tw.item(_row_of(tw, id_a), 1).checkState() == Qt.CheckState.Unchecked
        stored = json.loads(_FakeSettings.store['omrat/baseline_runs'])
        assert list(stored.values()) == [id_b]

    def test_unticking_clears_baseline(self, omrat, history):
        omrat._baseline_runs = None
        id_a = history.save_run('a', totals=_totals(1.0))
        omrat.refresh_previous_runs_table()
        _tick_baseline(omrat, id_a)
        _tick_baseline(omrat, id_a, ticked=False)
        assert omrat._get_baseline_run_id() is None
        assert 'omrat/baseline_runs' not in _FakeSettings.store

    def test_refresh_keeps_tick_and_forgets_deleted_baseline(self, omrat, history):
        from qgis.PyQt.QtCore import Qt
        omrat._baseline_runs = None
        id_a = history.save_run('a', totals=_totals(1.0))
        omrat.refresh_previous_runs_table()
        _tick_baseline(omrat, id_a)
        omrat._baseline_runs = None            # re-read from the settings
        omrat.refresh_previous_runs_table()
        tw = omrat.main_widget.TWPreviousRuns
        assert tw.item(_row_of(tw, id_a), 1).checkState() == Qt.CheckState.Checked
        history.delete_run(id_a)
        omrat.refresh_previous_runs_table()
        assert omrat._get_baseline_run_id() is None
        assert omrat._baseline_map() == {}

    def test_old_main_run_setting_is_not_read(self, omrat, history):
        omrat._baseline_runs = None
        id_a = history.save_run('a', totals=_totals(1.0))
        _FakeSettings.store['omrat/main_run_id'] = id_a
        omrat.refresh_previous_runs_table()
        assert omrat._get_baseline_run_id() is None


# ---------------------------------------------------------------------------
# Run columns of the accident table
# ---------------------------------------------------------------------------

class TestRunColumns:
    def test_baseline_once_then_value_and_change(self, omrat, history):
        omrat._baseline_runs = None
        id_base = history.save_run('base-run', totals=_totals(1.0))
        id_other = history.save_run('other', totals=_totals(2.0))
        omrat.refresh_previous_runs_table()
        _tick_baseline(omrat, id_base)
        _select(omrat, id_other)

        tw = omrat.main_widget.TWAccidentResults
        assert _headers(tw) == ['Accident type', 'base-run (baseline)', 'other', 'Change (%)', 'View']
        assert tw.item(1, 1).text() == '2.000e-04'
        assert tw.item(1, 2).text() == '4.000e-04'
        for row in range(9):
            assert tw.item(row, 3).text() == '+100.0%'
        # Summary rows: bold, in every run column.
        assert tw.item(9, 0).text() == 'All grounding'
        assert tw.item(9, 1).text() == '6.000e-04'
        assert tw.item(9, 2).text() == '1.200e-03'
        assert tw.item(9, 3).text() == '+100.0%'
        assert tw.item(9, 1).font().bold() and tw.item(9, 3).font().bold()
        assert 'base-run' in omrat.main_widget.lblCompareHint.text()

    def test_selected_baseline_is_shown_once(self, omrat, history):
        omrat._baseline_runs = None
        id_base = history.save_run('base-run', totals=_totals(1.0))
        id_other = history.save_run('other', totals=_totals(2.0))
        omrat.refresh_previous_runs_table()
        _tick_baseline(omrat, id_base)
        _select(omrat, id_base, id_other)
        tw = omrat.main_widget.TWAccidentResults
        assert _headers(tw) == ['Accident type', 'base-run (baseline)', 'other', 'Change (%)', 'View']

    def test_baseline_shown_without_a_selection(self, omrat, history):
        omrat._baseline_runs = None
        id_base = history.save_run('base-run', totals=_totals(1.0))
        omrat.refresh_previous_runs_table()
        _tick_baseline(omrat, id_base)
        _select(omrat)
        tw = omrat.main_widget.TWAccidentResults
        assert _headers(tw) == ['Accident type', 'base-run (baseline)', 'View']

    def test_no_baseline_values_only_with_a_hint(self, omrat, history):
        omrat._baseline_runs = None
        id_a = history.save_run('a', totals=_totals(1.0))
        id_b = history.save_run('b', totals=_totals(2.0))
        omrat.refresh_previous_runs_table()
        _select(omrat, id_a, id_b)
        tw = omrat.main_widget.TWAccidentResults
        assert _headers(tw) == ['Accident type', 'b', 'a', 'View']   # newest first
        assert 'Tick Baseline' in omrat.main_widget.lblCompareHint.text()

    def test_no_live_column(self, omrat, history):
        """The hidden LEP* fields (written by compute) are not shown: a
        finished run reaches the table through the run history."""
        omrat._baseline_runs = None
        _set_all_leps(omrat.main_widget)
        _select(omrat)
        tw = omrat.main_widget.TWAccidentResults
        assert _headers(tw) == ['Accident type', 'View']
        assert 'Select runs' in omrat.main_widget.lblCompareHint.text()

    def test_unticking_removes_the_change_columns(self, omrat, history):
        omrat._baseline_runs = None
        id_base = history.save_run('base-run', totals=_totals(1.0))
        id_other = history.save_run('other', totals=_totals(2.0))
        omrat.refresh_previous_runs_table()
        _tick_baseline(omrat, id_base)
        _select(omrat, id_other)
        _tick_baseline(omrat, id_base, ticked=False)
        tw = omrat.main_widget.TWAccidentResults
        assert _headers(tw) == ['Accident type', 'other', 'View']

    def test_view_buttons_stay_in_the_last_column(self, omrat, history):
        omrat._baseline_runs = None
        id_a = history.save_run('a', totals=_totals(1.0))
        omrat.refresh_previous_runs_table()
        _select(omrat, id_a)
        tw = omrat.main_widget.TWAccidentResults
        last = tw.columnCount() - 1
        assert all(tw.cellWidget(r, last) is not None for r in range(9))
        assert tw.cellWidget(9, last) is None      # summary rows have no View


# ---------------------------------------------------------------------------
# Copy to Excel
# ---------------------------------------------------------------------------

class TestCopyToExcel:
    def _fill(self, omrat, history, monkeypatch):
        omrat._baseline_runs = None
        id_base = history.save_run('base', totals=_totals(1.0))
        id_other = history.save_run('other', totals=_totals(2.0))
        # Drawn under a '.' locale, copied under ',': the separator is the
        # one at copy time, whatever the machine's locale (CI runs in C, a
        # Swedish desktop hid that it was baked in when drawing).
        monkeypatch.setattr(omrat, '_decimal_point', lambda: '.')
        omrat.refresh_previous_runs_table()
        _tick_baseline(omrat, id_base)
        _select(omrat, id_other)
        monkeypatch.setattr(omrat, '_decimal_point', lambda: ',')
        return omrat.main_widget.TWAccidentResults

    def test_whole_table_with_headers_as_numbers(self, omrat, history, monkeypatch):
        tw = self._fill(omrat, history, monkeypatch)
        lines = omrat._result_table_tsv(tw, whole=True).split('\n')
        assert lines[0].split('\t') == ['Accident type', 'base (baseline)', 'other', 'Change (%)']
        assert lines[1].split('\t') == ['Drifting allision', '0,0001', '0,0002', '100']
        assert lines[9].split('\t') == ['Bend collision', '5e-05', '0,0001', '100']
        assert lines[10].split('\t') == ['All grounding', '0,0006', '0,0012', '100']
        assert len(lines) == 13

    def test_selection_without_headers(self, omrat, history, monkeypatch):
        from qgis.PyQt.QtWidgets import QTableWidgetSelectionRange
        tw = self._fill(omrat, history, monkeypatch)
        tw.clearSelection()
        tw.setRangeSelected(QTableWidgetSelectionRange(0, 2, 1, 4), True)   # incl. View
        assert omrat._result_table_tsv(tw) == '0,0002\t100\n0,0004\t100'

    def test_years_mode_copies_years(self, omrat, history, monkeypatch):
        tw = self._fill(omrat, history, monkeypatch)
        omrat.main_widget.cbResultDisplayMode.setCurrentIndex(1)
        lines = omrat._result_table_tsv(tw, whole=True).split('\n')
        # drift allision 1e-4 -> 10000 years; doubled -> 5000 years, -50 %
        assert lines[1].split('\t') == ['Drifting allision', '10000', '5000', '-50']

    def test_ctrl_c_reaches_the_table_copy(self, omrat, history, monkeypatch):
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtTest import QTest
        tw = self._fill(omrat, history, monkeypatch)
        calls = []
        monkeypatch.setattr(omrat, '_copy_result_table', lambda t, whole=False: calls.append((t, whole)))
        QTest.keyClick(tw, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
        assert calls == [(tw, False)]


# ---------------------------------------------------------------------------
# Summary rows
# ---------------------------------------------------------------------------

class TestSummaryRows:
    def test_three_bold_rows_below_accidents(self, omrat):
        tw = omrat.main_widget.TWAccidentResults
        assert tw.rowCount() == 12
        assert [tw.item(r, 0).text() for r in (9, 10, 11)] == [
            'All grounding', 'All allision', 'All collisions',
        ]
        assert tw.item(9, 0).font().bold()
        assert tw.cellWidget(9, tw.columnCount() - 1) is None  # no View button


# ---------------------------------------------------------------------------
# Display mode: frequency <-> years between incidents
# ---------------------------------------------------------------------------

def _set_all_leps(w) -> None:
    w.LEPDriftingGrounding.setText('1.000e-03')
    w.LEPPoweredGrounding.setText('2.000e-03')
    w.LEPDriftAllision.setText('5.000e-04')
    w.LEPPoweredAllision.setText('0')
    for name in ('LEPOvertakingCollision', 'LEPHeadOnCollision',
                 'LEPCrossingCollision', 'LEPMergingCollision', 'LEPBendCollision'):
        getattr(w, name).setText('1.000e-05')


class TestResultDisplayMode:
    def test_combo_defaults_to_frequency(self, omrat, history):
        cb = omrat.main_widget.cbResultDisplayMode
        assert cb.count() == 2
        assert cb.itemData(0) == 'frequency'
        assert cb.itemData(1) == 'return_period'
        assert cb.currentIndex() == 0

    def test_switching_rerenders_values_and_change(self, omrat, history):
        from omrat_utils.accident_summary import RESULT_DISPLAY_SETTING
        omrat._baseline_runs = None
        totals = _totals(1.0)
        totals['powered_allision'] = 0.0
        id_base = history.save_run('base', totals=totals)
        id_other = history.save_run('other', totals=_totals(2.0))
        omrat.refresh_previous_runs_table()
        _tick_baseline(omrat, id_base)
        _select(omrat, id_other)
        w = omrat.main_widget
        tw = w.TWAccidentResults
        assert tw.item(1, 1).text() == '2.000e-04'

        w.cbResultDisplayMode.setCurrentIndex(1)
        assert tw.item(1, 1).text() == '5,000'      # drifting grounding 1 / 2e-4
        assert tw.item(1, 2).text() == '2,500'
        assert tw.item(1, 3).text() == '-50.0%'     # change of the years
        assert tw.item(2, 1).text() == '∞'     # zero frequency
        assert tw.item(2, 3).text() == '—'     # change from infinity: undefined
        assert tw.item(9, 1).text() == '1,667'      # All grounding 1 / 6e-4
        assert tw.item(9, 1).font().bold()
        assert _FakeSettings.store[RESULT_DISPLAY_SETTING] == 'return_period'

        w.cbResultDisplayMode.setCurrentIndex(0)
        assert tw.item(1, 1).text() == '2.000e-04'
        assert tw.item(1, 3).text() == '+100.0%'
        assert tw.item(2, 1).text() == '0'


# ---------------------------------------------------------------------------
# Tangent lines on file load
# ---------------------------------------------------------------------------

class TestTangentOnLoad:
    def test_load_lines_draws_tangent_layer(self, omrat):
        from qgis.core import QgsProject
        omrat.qgis_geoms.clear()
        segs = {
            '1': {'Segment_Id': '1', 'Route_Id': 1, 'Leg_name': 'L1',
                  'Start_Point': '12.00 56.00', 'End_Point': '12.20 56.10', 'Width': 4000},
            '2': {'Segment_Id': '2', 'Route_Id': 1, 'Leg_name': 'L2',
                  'Start_Point': '12.20 56.10', 'End_Point': '12.40 56.30', 'Width': 6000},
        }
        omrat.load_lines({'segment_data': segs})
        layer = omrat.qgis_geoms.tangent_layer
        assert layer is not None
        assert QgsProject.instance().mapLayer(layer.id()) is layer
        assert layer.featureCount() == 2
        types = sorted(f['type'] for f in layer.getFeatures())
        assert types == ['Tangent Line 1', 'Tangent Line 2']

    def test_stale_tangent_layer_is_recreated(self, omrat):
        from qgis.core import QgsProject, QgsPointXY
        omrat.qgis_geoms.clear()
        omrat.qgis_geoms.create_offset_lines(
            QgsPointXY(12.0, 56.0), QgsPointXY(12.2, 56.1), 2000, 1,
        )
        first = omrat.qgis_geoms.tangent_layer
        QgsProject.instance().removeMapLayer(first.id())
        omrat.qgis_geoms.create_offset_lines(
            QgsPointXY(12.0, 56.0), QgsPointXY(12.2, 56.1), 2000, 1,
        )
        second = omrat.qgis_geoms.tangent_layer
        assert second is not first
        assert QgsProject.instance().mapLayer(second.id()) is second
        assert second.featureCount() == 1


# ---------------------------------------------------------------------------
# Snapshot -> grouped layers
# ---------------------------------------------------------------------------

class TestSnapshotLayers:
    def test_group_order_and_contents(self, omrat):
        from qgis.core import QgsProject
        from omrat_utils.snapshot_layers import add_snapshot_to_project
        with EXAMPLE.open('r', encoding='utf-8') as f:
            snap = json.load(f)
        built = add_snapshot_to_project(snap, 'Compare A: proj', color='red')
        group = built['group']
        try:
            root = QgsProject.instance().layerTreeRoot()
            assert root.findGroup('Compare A: proj') is group
            names = [child.name() for child in group.children()]
            # Top to bottom: results (none here), tangents, legs, structures, depths.
            assert names == ['Tangent Lines', 'Legs', 'Structures', 'Depth Areas']
            by_name = {lyr.name(): lyr for lyr in built['model']}
            n_legs = len(snap['segment_data'])
            assert by_name['Legs'].featureCount() == n_legs
            assert by_name['Tangent Lines'].featureCount() == n_legs
            assert by_name['Depth Areas'].featureCount() == len(snap['depths'])
            assert by_name['Structures'].featureCount() == len(snap['objects'])
            assert built['results'] == []
            for lyr in built['model']:
                assert QgsProject.instance().mapLayer(lyr.id()) is lyr
        finally:
            root.removeChildNode(group)

    def test_perpendicular_helper(self):
        from omrat_utils.snapshot_layers import perpendicular_through_midpoint
        ends = perpendicular_through_midpoint((0.0, 0.0), (1000.0, 0.0), 250.0)
        assert ends is not None
        (ax, ay), (bx, by) = ends
        assert abs(ax - 500.0) < 1e-9 and abs(bx - 500.0) < 1e-9
        assert sorted((ay, by)) == [-250.0, 250.0]
        assert perpendicular_through_midpoint((1.0, 1.0), (1.0, 1.0), 10.0) is None


class TestSnapshotTangentPosition:
    def test_build_tangent_layer_honours_tangent_pos(self, omrat):
        from omrat_utils.snapshot_layers import build_tangent_layer
        seg = {
            '1': {'Start_Point': '14.0 55.0', 'End_Point': '14.2 55.0', 'Width': 2000,
                  'Route_Id': 1, 'Leg_name': 'L1', 'Segment_Id': '1', 'Tangent_Pos': 0.2},
            '2': {'Start_Point': '14.0 56.0', 'End_Point': '14.2 56.0', 'Width': 2000,
                  'Route_Id': 1, 'Leg_name': 'L2', 'Segment_Id': '2'},
        }
        layer = build_tangent_layer(seg, 'Tangent Lines', color='#000000')
        assert layer is not None
        mids = {}
        for f in layer.getFeatures():
            pts = f.geometry().asPolyline()
            mids[f['type']] = ((pts[0].x() + pts[-1].x()) / 2, (pts[0].y() + pts[-1].y()) / 2)
        assert mids['Tangent Line 1'][0] == pytest.approx(14.04, abs=2e-3)
        assert mids['Tangent Line 1'][1] == pytest.approx(55.0, abs=2e-4)
        # No Tangent_Pos -> midpoint, as before.
        assert mids['Tangent Line 2'][0] == pytest.approx(14.1, abs=2e-3)
