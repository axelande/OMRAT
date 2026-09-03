"""QGIS-fixture tests for the Run Analysis / Compare UI changes:

* ``TWPreviousRuns`` has a checkable **Main** column and the accident
  table's delta columns are computed against the main run;
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
# Main column
# ---------------------------------------------------------------------------

class TestMainColumn:
    def test_headers_have_main_right_of_name(self, omrat, history):
        omrat._main_run_id = None
        history.save_run('base', totals=_totals(1.0))
        omrat.refresh_previous_runs_table()
        tw = omrat.main_widget.TWPreviousRuns
        headers = [tw.horizontalHeaderItem(c).text() for c in range(tw.columnCount())]
        assert headers == ['Name', 'Main', 'Date', 'Duration']

    def test_main_cells_are_checkable_and_unchecked(self, omrat, history):
        from qgis.PyQt.QtCore import Qt
        omrat._main_run_id = None
        history.save_run('base', totals=_totals(1.0))
        omrat.refresh_previous_runs_table()
        item = omrat.main_widget.TWPreviousRuns.item(0, 1)
        assert item.flags() & Qt.ItemFlag.ItemIsUserCheckable
        assert item.checkState() == Qt.CheckState.Unchecked

    def test_ticking_sets_main_and_unticks_others(self, omrat, history):
        from qgis.PyQt.QtCore import Qt
        omrat._main_run_id = None
        id_a = history.save_run('a', totals=_totals(1.0))
        id_b = history.save_run('b', totals=_totals(2.0))
        omrat.refresh_previous_runs_table()
        tw = omrat.main_widget.TWPreviousRuns
        tw.item(_row_of(tw, id_a), 1).setCheckState(Qt.CheckState.Checked)
        assert omrat._get_main_run_id() == id_a
        tw.item(_row_of(tw, id_b), 1).setCheckState(Qt.CheckState.Checked)
        assert omrat._get_main_run_id() == id_b
        assert tw.item(_row_of(tw, id_a), 1).checkState() == Qt.CheckState.Unchecked
        assert _FakeSettings.store['omrat/main_run_id'] == id_b

    def test_unticking_clears_main(self, omrat, history):
        from qgis.PyQt.QtCore import Qt
        omrat._main_run_id = None
        id_a = history.save_run('a', totals=_totals(1.0))
        omrat.refresh_previous_runs_table()
        tw = omrat.main_widget.TWPreviousRuns
        item = tw.item(_row_of(tw, id_a), 1)
        item.setCheckState(Qt.CheckState.Checked)
        item.setCheckState(Qt.CheckState.Unchecked)
        assert omrat._get_main_run_id() is None

    def test_refresh_keeps_tick_and_forgets_deleted_main(self, omrat, history):
        from qgis.PyQt.QtCore import Qt
        omrat._main_run_id = None
        id_a = history.save_run('a', totals=_totals(1.0))
        omrat._set_main_run_id(id_a)
        omrat.refresh_previous_runs_table()
        tw = omrat.main_widget.TWPreviousRuns
        assert tw.item(_row_of(tw, id_a), 1).checkState() == Qt.CheckState.Checked
        history.delete_run(id_a)
        omrat.refresh_previous_runs_table()
        assert omrat._get_main_run_id() is None


class TestComparisonAgainstMain:
    def test_delta_uses_main_run(self, omrat, history):
        omrat._main_run_id = None
        id_main = history.save_run('main-run', totals=_totals(1.0))
        id_other = history.save_run('other', totals=_totals(2.0))
        omrat._set_main_run_id(id_main)
        omrat.refresh_previous_runs_table()
        tw_runs = omrat.main_widget.TWPreviousRuns
        tw_runs.clearSelection()
        tw_runs.selectRow(_row_of(tw_runs, id_other))
        omrat._on_previous_runs_selection_changed()

        tw = omrat.main_widget.TWAccidentResults
        # base 3 + probability + delta; columns: type, current, other, delta, View
        assert tw.columnCount() == 5
        assert tw.horizontalHeaderItem(2).text() == 'other'
        assert tw.horizontalHeaderItem(3).text() == 'Δ vs main (main-run) %'
        # every accident row doubled -> +100 %
        for row in range(9):
            assert tw.item(row, 3).text() == '+100.0%'
        # summary rows too
        assert tw.item(9, 0).text() == 'All grounding'
        assert tw.item(9, 2).text() == '1.200e-03'  # (2e-4 + 4e-4) * 2
        assert tw.item(9, 3).text() == '+100.0%'
        assert tw.item(9, 3).font().bold()
        omrat._reset_accident_table_to_base()

    def test_without_main_falls_back_to_current(self, omrat, history):
        omrat._main_run_id = None
        id_a = history.save_run('a', totals=_totals(1.0))
        omrat.refresh_previous_runs_table()
        omrat.main_widget.LEPDriftAllision.setText('1.000e-04')
        tw_runs = omrat.main_widget.TWPreviousRuns
        tw_runs.clearSelection()
        tw_runs.selectRow(_row_of(tw_runs, id_a))
        omrat._on_previous_runs_selection_changed()
        tw = omrat.main_widget.TWAccidentResults
        assert tw.horizontalHeaderItem(3).text() == 'Δ vs current %'
        assert tw.item(0, 3).text() == '+0.0%'
        omrat._reset_accident_table_to_base()


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
        assert tw.cellWidget(9, 2) is None  # no View button

    def test_summary_follows_lep_edits(self, omrat):
        w = omrat.main_widget
        w.LEPDriftingGrounding.setText('1.000e-03')
        w.LEPPoweredGrounding.setText('2.000e-03')
        w.LEPDriftAllision.setText('5.000e-04')
        w.LEPPoweredAllision.setText('')
        for name in ('LEPOvertakingCollision', 'LEPHeadOnCollision',
                     'LEPCrossingCollision', 'LEPMergingCollision', 'LEPBendCollision'):
            getattr(w, name).setText('1.000e-05')
        tw = w.TWAccidentResults
        assert tw.item(9, 1).text() == '3.000e-03'
        assert tw.item(10, 1).text() == '5.000e-04'
        assert tw.item(11, 1).text() == '5.000e-05'


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
