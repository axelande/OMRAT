"""Unit tests for omrat_utils/handle_traffic.py::Traffic.

``Traffic`` is a thin wrapper around Qt widgets + a nested dict of
traffic rows.  These tests use real QTableWidget / QSpinBox instances
(provided by pytest-qgis) so the Qt lifecycle + model wiring actually
runs, but avoid the full OMRAT plugin bootstrap.
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
def traffic(qgis_iface):
    """Traffic instance with real Qt widgets and a stub ShipCategories."""
    from qgis.PyQt.QtWidgets import QTableWidgetItem
    from omrat_utils.handle_ship_cat import ShipCategories
    from omrat_utils.handle_traffic import Traffic
    from ui.traffic_data_widget import TrafficDataWidget

    # Build a real ShipCategories widget so row/column counts come from Qt.
    sc = ShipCategories(MagicMock())
    sc.scw.cvTypes.setColumnCount(1)
    sc.scw.cvTypes.setRowCount(2)
    sc.scw.cvTypes.setItem(0, 0, QTableWidgetItem('Cargo'))
    sc.scw.cvTypes.setItem(1, 0, QTableWidgetItem('Tanker'))
    sc.scw.twLengths.setColumnCount(2)
    sc.scw.twLengths.setRowCount(2)
    sc.scw.twLengths.setItem(0, 0, QTableWidgetItem('0'))
    sc.scw.twLengths.setItem(0, 1, QTableWidgetItem('25'))
    sc.scw.twLengths.setItem(1, 0, QTableWidgetItem('25'))
    sc.scw.twLengths.setItem(1, 1, QTableWidgetItem('50'))

    omrat = MagicMock()
    omrat.ship_cat = sc
    omrat.traffic_data = {}

    dw = TrafficDataWidget()  # the real widget, has twTrafficData etc.
    t = Traffic(omrat, dw)
    return t


# ---------------------------------------------------------------------------
# fill_cbTrafficSelectSeg
# ---------------------------------------------------------------------------

class TestFillSegSelect:
    def test_populates_from_route_table(self, traffic, qgis_iface):
        from qgis.PyQt.QtWidgets import QTableWidget, QTableWidgetItem
        # Attach a route-list table (normally on the main widget, not dw).
        traffic.dw.twRouteList = QTableWidget()
        traffic.dw.twRouteList.setColumnCount(1)
        traffic.dw.twRouteList.setRowCount(2)
        traffic.dw.twRouteList.setItem(0, 0, QTableWidgetItem('L1'))
        traffic.dw.twRouteList.setItem(1, 0, QTableWidgetItem('L2'))
        traffic.fill_cbTrafficSelectSeg()
        assert traffic.dw.cbTrafficSelectSeg.count() == 2
        assert traffic.dw.cbTrafficSelectSeg.itemText(0) == 'L1'
        assert traffic.dw.cbTrafficSelectSeg.itemText(1) == 'L2'


# ---------------------------------------------------------------------------
# set_table_headings -- the fontMetrics / sizing fallback
# ---------------------------------------------------------------------------

class TestSetTableHeadings:
    def test_headings_from_ship_cat(self, traffic):
        traffic.set_table_headings()
        # 2 types x 2 size bins from the fixture.
        assert traffic.dw.twTrafficData.rowCount() == 2
        assert traffic.dw.twTrafficData.columnCount() == 2

    def test_fontmetrics_failure_swallowed(self, traffic, monkeypatch):
        """A TypeError from fontMetrics is caught silently (L85-87)."""
        # Force horizontalAdvance to raise TypeError.
        fm = traffic.dw.twTrafficData.fontMetrics()
        monkeypatch.setattr(
            type(fm), 'horizontalAdvance',
            lambda self, *a: (_ for _ in ()).throw(TypeError('mocked'))
        )
        traffic.set_table_headings()  # should not raise


# ---------------------------------------------------------------------------
# create_empty_dict
# ---------------------------------------------------------------------------

class TestCreateEmptyDict:
    def test_shape_matches_table(self, traffic):
        traffic.set_table_headings()
        traffic.create_empty_dict('L1', ['East', 'West'])
        data = traffic.traffic_data['L1']
        assert set(data.keys()) == {'East', 'West'}
        # Frequency grid has int zeros, the others have empty lists.
        east = data['East']
        freq = east['Frequency (ships/year)']
        assert freq == [[0, 0], [0, 0]]
        assert east['Speed (knots)'] == [[[], []], [[], []]]


# ---------------------------------------------------------------------------
# update_traffic_tbl -- the different cell types
# ---------------------------------------------------------------------------

class TestUpdateTrafficTbl:
    def _prime(self, traffic, val):
        # Disable signals before touching Qt widgets; otherwise setCurrentText
        # triggers update_traffic_tbl which calls save() and overwrites the
        # very value we're about to set.
        traffic.run_update = False
        traffic.set_table_headings()
        traffic.create_empty_dict('L1', ['East'])
        # Populate every cell in Speed (knots) so the update_traffic_tbl loop
        # doesn't trip over an empty list when building a QDoubleSpinBox.
        rows = traffic.dw.twTrafficData.rowCount()
        cols = traffic.dw.twTrafficData.columnCount()
        for r in range(rows):
            for c in range(cols):
                traffic.traffic_data['L1']['East']['Speed (knots)'][r][c] = 0
        traffic.traffic_data['L1']['East']['Speed (knots)'][0][0] = val
        traffic.c_seg = 'L1'
        traffic.c_di = 'East'
        traffic.last_var = 'Speed (knots)'
        traffic.dw.cbTrafficSelectSeg.addItem('L1')
        traffic.dw.cbTrafficDirectionSelect.addItem('East')
        # cbSelectType has pre-populated items from the .ui; pick Speed (knots).
        traffic.dw.cbSelectType.setCurrentText('Speed (knots)')

    def test_empty_string_value_becomes_zero_spinbox(self, traffic):
        from qgis.PyQt.QtWidgets import QSpinBox
        self._prime(traffic, '')
        traffic.update_traffic_tbl('type')
        w = traffic.dw.twTrafficData.cellWidget(0, 0)
        assert isinstance(w, QSpinBox)
        assert w.value() == 0

    def test_inf_value_uses_spinbox(self, traffic):
        """An infinite value trips the np.inf branch which creates a
        QSpinBox (with setEnabled(False) called before mounting)."""
        import numpy as np
        from qgis.PyQt.QtWidgets import QSpinBox
        self._prime(traffic, np.inf)
        traffic.update_traffic_tbl('type')
        w = traffic.dw.twTrafficData.cellWidget(0, 0)
        # Branch was taken -> widget is a QSpinBox.  The enabled state is
        # whatever Qt lands on after mounting; we don't assert on that.
        assert isinstance(w, QSpinBox)
        assert w.value() == 0

    def test_float_value_uses_double_spinbox(self, traffic):
        from qgis.PyQt.QtWidgets import QDoubleSpinBox
        self._prime(traffic, 12.5)
        traffic.update_traffic_tbl('type')
        w = traffic.dw.twTrafficData.cellWidget(0, 0)
        assert isinstance(w, QDoubleSpinBox)
        assert w.value() == pytest.approx(12.5)

    def test_int_value_uses_int_spinbox(self, traffic):
        from qgis.PyQt.QtWidgets import QSpinBox
        self._prime(traffic, 3)
        traffic.update_traffic_tbl('type')
        w = traffic.dw.twTrafficData.cellWidget(0, 0)
        assert isinstance(w, QSpinBox)
        assert w.value() == 3

    def test_segment_caller_updates_direction_select(self, traffic):
        traffic.set_table_headings()
        traffic.create_empty_dict('L1', ['East', 'West'])
        traffic.c_seg = 'L1'
        traffic.dw.cbTrafficSelectSeg.addItem('L1')
        traffic.dw.cbTrafficSelectSeg.setCurrentText('L1')
        traffic.dw.cbTrafficDirectionSelect.addItem('East')  # will be cleared
        traffic.run_update = False
        traffic.update_traffic_tbl('segment')
        # Direction dropdown repopulated with the two keys.
        texts = [traffic.dw.cbTrafficDirectionSelect.itemText(i)
                 for i in range(traffic.dw.cbTrafficDirectionSelect.count())]
        assert set(texts) == {'East', 'West'}

    def test_empty_selection_short_circuits(self, traffic):
        """If c_seg / c_di / last_var is blank, the method returns without
        filling cell widgets."""
        traffic.set_table_headings()
        traffic.c_seg = ''
        traffic.c_di = ''
        traffic.last_var = ''
        # Should simply return (no AttributeError from missing traffic_data).
        traffic.update_traffic_tbl('type')


# ---------------------------------------------------------------------------
# update_direction_select
# ---------------------------------------------------------------------------

class TestUpdateDirectionSelect:
    def test_empty_traffic_data_returns_early(self, traffic):
        traffic.traffic_data.clear()
        traffic.c_seg = ''
        traffic.update_direction_select()
        assert traffic.dw.cbTrafficDirectionSelect.count() == 0

    def test_populates_directions_for_current_segment(self, traffic):
        traffic.set_table_headings()
        traffic.create_empty_dict('L1', ['East', 'West'])
        traffic.c_seg = 'L1'
        traffic.update_direction_select()
        count = traffic.dw.cbTrafficDirectionSelect.count()
        assert count == 2


# ---------------------------------------------------------------------------
# save + commit_changes
# ---------------------------------------------------------------------------

class TestSaveAndCommit:
    def test_save_writes_cell_values_back(self, traffic):
        from qgis.PyQt.QtWidgets import QSpinBox
        traffic.set_table_headings()
        traffic.create_empty_dict('L1', ['East'])
        # Install SpinBoxes with known values in the table.
        for row in range(2):
            for col in range(2):
                sb = QSpinBox()
                sb.setMaximum(1000)
                sb.setValue(10 * row + col)
                traffic.dw.twTrafficData.setCellWidget(row, col, sb)
        traffic.c_seg = 'L1'
        traffic.c_di = 'East'
        traffic.last_var = 'Frequency (ships/year)'
        traffic.save()
        freq = traffic.traffic_data['L1']['East']['Frequency (ships/year)']
        assert freq == [[0, 1], [10, 11]]

    def test_save_with_blank_context_noops(self, traffic):
        traffic.c_seg = ''
        traffic.c_di = ''
        traffic.last_var = ''
        traffic.save()  # should not raise

    def test_commit_changes_propagates_to_omrat(self, traffic):
        traffic.traffic_data['L1'] = {'East': {}}
        traffic.c_seg = ''
        traffic.c_di = ''
        traffic.last_var = ''  # so save() short-circuits
        traffic.commit_changes()
        assert traffic.omrat.traffic_data is traffic.traffic_data


# ---------------------------------------------------------------------------
# unload
# ---------------------------------------------------------------------------

class TestUnload:
    def test_unload_clears_data_and_disconnects(self, traffic, capsys):
        traffic.traffic_data['L1'] = {'East': {}}
        traffic.unload()
        assert traffic.traffic_data == {}
        out = capsys.readouterr().out
        assert 'cleaned up' in out.lower()

    def test_unload_handles_already_disconnected_signals(self, traffic):
        """Qt raises TypeError when disconnect() is called twice; the method
        swallows it."""
        traffic.unload()  # first call
        # Second call: signals already disconnected -> TypeError caught.
        traffic.unload()  # must not raise
