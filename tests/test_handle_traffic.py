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


# ---------------------------------------------------------------------------
# Scaling (%) -- the "easy option" frequency multiplier
# ---------------------------------------------------------------------------

class TestScalingDefaultsAndShape:
    def test_create_empty_dict_seeds_scaling_at_100(self, traffic):
        """``Scaling (%)`` is a sibling of Frequency etc. and defaults to
        100 in every cell, so a project compute identically until the
        user touches it."""
        from omrat_utils.handle_traffic import SCALING_VAR
        traffic.set_table_headings()
        traffic.create_empty_dict('L1', ['East'])
        scaling = traffic.traffic_data['L1']['East'][SCALING_VAR]
        assert scaling == [[100.0, 100.0], [100.0, 100.0]]

    def test_create_empty_dict_keeps_frequency_at_zero(self, traffic):
        """Sanity-check the existing Frequency default didn't regress when
        Scaling was added to the variable list."""
        traffic.set_table_headings()
        traffic.create_empty_dict('L1', ['East'])
        freq = traffic.traffic_data['L1']['East']['Frequency (ships/year)']
        assert freq == [[0, 0], [0, 0]]

    def test_ensure_scaling_present_seeds_missing(self, traffic):
        """Legacy projects have no Scaling matrix; ensure_scaling_present
        adds a default-100 matrix shaped from Frequency."""
        from omrat_utils.handle_traffic import SCALING_VAR
        traffic.set_table_headings()
        traffic.traffic_data['L1'] = {
            'East': {
                'Frequency (ships/year)': [[5.0, 0.0], [0.0, 0.0]],
            },
        }
        traffic.ensure_scaling_present()
        assert traffic.traffic_data['L1']['East'][SCALING_VAR] == [
            [100.0, 100.0], [100.0, 100.0],
        ]

    def test_ensure_scaling_present_leaves_existing_alone(self, traffic):
        """An already-edited Scaling matrix must not be overwritten."""
        from omrat_utils.handle_traffic import SCALING_VAR
        traffic.set_table_headings()
        traffic.traffic_data['L1'] = {
            'East': {
                'Frequency (ships/year)': [[5.0, 0.0], [0.0, 0.0]],
                SCALING_VAR: [[130.0, 100.0], [100.0, 100.0]],
            },
        }
        traffic.ensure_scaling_present()
        assert traffic.traffic_data['L1']['East'][SCALING_VAR] == [
            [130.0, 100.0], [100.0, 100.0],
        ]

    def test_ensure_scaling_present_pads_undersized_rows(self, traffic):
        """If columns were added since save, the existing rows are padded
        with 100 -- new bins default to no scaling."""
        from omrat_utils.handle_traffic import SCALING_VAR
        traffic.set_table_headings()
        # Freq has 2 cols; existing Scaling has 1 -> right-pad to 2.
        traffic.traffic_data['L1'] = {
            'East': {
                'Frequency (ships/year)': [[5.0, 0.0], [0.0, 0.0]],
                SCALING_VAR: [[130.0], [100.0]],
            },
        }
        traffic.ensure_scaling_present()
        scaling = traffic.traffic_data['L1']['East'][SCALING_VAR]
        assert scaling[0] == [130.0, 100.0]
        assert scaling[1] == [100.0, 100.0]


class TestEnsureFollowGlobal:
    def test_seeds_default_true_per_row(self, traffic):
        traffic.set_table_headings()
        traffic.ensure_follow_global()
        flags = traffic._project_scaling()['follow_global']
        assert flags == [True, True]

    def test_extends_when_more_types(self, traffic):
        traffic.set_table_headings()
        traffic.omrat.traffic_scaling = {
            'global_percent': 100.0,
            'follow_global': [False],
        }
        traffic.ensure_follow_global(n_types=3)
        flags = traffic._project_scaling()['follow_global']
        # Existing False preserved, new rows default True.
        assert flags == [False, True, True]

    def test_truncates_when_fewer_types(self, traffic):
        traffic.set_table_headings()
        traffic.omrat.traffic_scaling = {
            'global_percent': 100.0,
            'follow_global': [True, False, True, False],
        }
        traffic.ensure_follow_global(n_types=2)
        flags = traffic._project_scaling()['follow_global']
        assert flags == [True, False]


class TestApplyGlobalScaling:
    def test_broadcasts_to_all_ticked_rows(self, traffic):
        """A global value floods every (seg, dir, type, len) cell of
        types whose ``follow_global`` bit is True."""
        from omrat_utils.handle_traffic import SCALING_VAR
        traffic.set_table_headings()
        traffic.create_empty_dict('L1', ['East', 'West'])
        traffic.ensure_follow_global()
        # Default is [True, True]; broadcast 130 should hit both rows.
        traffic.apply_global_scaling(value=130.0, refresh_table=False)
        for direction in ('East', 'West'):
            mat = traffic.traffic_data['L1'][direction][SCALING_VAR]
            assert mat == [[130.0, 130.0], [130.0, 130.0]]
        # And the project-level state remembers the new global.
        assert traffic._project_scaling()['global_percent'] == pytest.approx(130.0)

    def test_skips_unticked_rows(self, traffic):
        """A row whose checkbox is unticked keeps its existing values."""
        from omrat_utils.handle_traffic import SCALING_VAR
        traffic.set_table_headings()
        traffic.create_empty_dict('L1', ['East'])
        # Untick row 0 (Cargo) -- it must remain at 100 even after
        # broadcasting 130.  Row 1 (Tanker) follows the global.
        traffic.omrat.traffic_scaling = {
            'global_percent': 100.0,
            'follow_global': [False, True],
        }
        traffic.apply_global_scaling(value=130.0, refresh_table=False)
        mat = traffic.traffic_data['L1']['East'][SCALING_VAR]
        assert mat[0] == [100.0, 100.0]   # unticked row left alone
        assert mat[1] == [130.0, 130.0]   # ticked row follows global

    def test_default_value_pulled_from_project_scaling(self, traffic):
        """``value=None`` falls back to the stored global_percent."""
        from omrat_utils.handle_traffic import SCALING_VAR
        traffic.set_table_headings()
        traffic.create_empty_dict('L1', ['East'])
        # Pretend main_widget is missing the spinbox so the fallback
        # path is taken (project_scaling.global_percent).
        traffic.omrat.main_widget = None
        traffic.omrat.traffic_scaling = {
            'global_percent': 75.0,
            'follow_global': [True, True],
        }
        traffic.apply_global_scaling(value=None, refresh_table=False)
        mat = traffic.traffic_data['L1']['East'][SCALING_VAR]
        assert mat == [[75.0, 75.0], [75.0, 75.0]]


class TestFollowGlobalToggle:
    def test_newly_ticked_row_rebroadcasts_global(self, traffic):
        """Toggling a row from unticked -> ticked pushes the current
        global value back into that row's cells."""
        from omrat_utils.handle_traffic import SCALING_VAR
        traffic.set_table_headings()
        traffic.create_empty_dict('L1', ['East'])
        traffic.omrat.traffic_scaling = {
            'global_percent': 150.0,
            'follow_global': [False, False],
        }
        # Both rows currently at 100; row 0 has been edited but never
        # broadcast.  Tick row 1 -> row 1 should jump to 150.
        traffic._on_follow_global_toggled(1, True)
        mat = traffic.traffic_data['L1']['East'][SCALING_VAR]
        assert mat[0] == [100.0, 100.0]  # still unticked
        assert mat[1] == [150.0, 150.0]
        assert traffic._project_scaling()['follow_global'] == [False, True]

    def test_untick_only_flips_the_flag(self, traffic):
        """Unticking a row leaves the cell values where they are; the
        user keeps whatever the global last broadcast (no surprise
        zeros)."""
        from omrat_utils.handle_traffic import SCALING_VAR
        traffic.set_table_headings()
        traffic.create_empty_dict('L1', ['East'])
        traffic.omrat.traffic_scaling = {
            'global_percent': 130.0,
            'follow_global': [True, True],
        }
        traffic.apply_global_scaling(value=130.0, refresh_table=False)
        traffic._on_follow_global_toggled(0, False)
        mat = traffic.traffic_data['L1']['East'][SCALING_VAR]
        assert mat[0] == [130.0, 130.0]  # cell values stay
        assert traffic._project_scaling()['follow_global'] == [False, True]


class TestScalingCellEditedUnticks:
    def test_manual_edit_unticks_the_row(self, traffic):
        """Editing a cell in the Scaling view is the "manual override"
        signal -- the row's follow_global flag flips off so future
        global broadcasts skip it."""
        traffic.set_table_headings()
        traffic.create_empty_dict('L1', ['East'])
        traffic.omrat.traffic_scaling = {
            'global_percent': 100.0,
            'follow_global': [True, True],
        }
        # Simulate the spinbox.valueChanged callback for row 0.
        traffic._on_scaling_cell_edited(0, 130.0)
        assert traffic._project_scaling()['follow_global'] == [False, True]

    def test_edit_on_already_unticked_row_is_noop(self, traffic):
        """If the row is already manual, an edit must not flip anything
        else and must not raise."""
        traffic.set_table_headings()
        traffic.omrat.traffic_scaling = {
            'global_percent': 100.0,
            'follow_global': [False, True],
        }
        traffic._on_scaling_cell_edited(0, 130.0)  # already False
        assert traffic._project_scaling()['follow_global'] == [False, True]


class TestSaveRoundTripsScalingValues:
    """Editing the Scaling table and calling ``save`` must persist the
    typed values back into ``traffic_data`` -- the same path Frequency
    uses, so a 130 % entry survives a save / re-open."""

    def test_save_stores_scaling_doublespinbox_values(self, traffic):
        from qgis.PyQt.QtWidgets import QDoubleSpinBox
        from omrat_utils.handle_traffic import SCALING_VAR
        traffic.set_table_headings()
        traffic.create_empty_dict('L1', ['East'])
        # Install QDoubleSpinBoxes in every cell with known scaling values.
        values = [[130.5, 100.0], [50.0, 200.0]]
        for r in range(2):
            for c in range(2):
                sb = QDoubleSpinBox()
                sb.setRange(0.0, 100000.0)
                sb.setDecimals(1)
                sb.setValue(values[r][c])
                traffic.dw.twTrafficData.setCellWidget(r, c, sb)
        traffic.c_seg = 'L1'
        traffic.c_di = 'East'
        traffic.last_var = SCALING_VAR
        traffic.save()
        stored = traffic.traffic_data['L1']['East'][SCALING_VAR]
        assert stored[0][0] == pytest.approx(130.5)
        assert stored[1][1] == pytest.approx(200.0)
