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


FREQ = 'Frequency (ships/year)'


@pytest.fixture
def locale_sv():
    """Swedish number format (decimal comma), as Excel uses on the user's machine."""
    from qgis.PyQt.QtCore import QLocale
    prev = QLocale()
    QLocale.setDefault(QLocale('sv_SE'))
    yield
    QLocale.setDefault(prev)


@pytest.fixture
def locale_en():
    from qgis.PyQt.QtCore import QLocale
    prev = QLocale()
    QLocale.setDefault(QLocale('en_US'))
    yield
    QLocale.setDefault(prev)


def _show(traffic, var=FREQ):
    """Render leg L1 / East / ``var`` in the table (2 types x 2 lengths)."""
    traffic.run_update = False
    if 'L1' not in traffic.traffic_data:
        traffic.set_table_headings()
        traffic.create_empty_dict('L1', ['East'])
        traffic.dw.cbTrafficSelectSeg.addItem('LEG_1_L1', 'L1')
        traffic.dw.cbTrafficDirectionSelect.addItem('East')
    traffic.c_seg, traffic.c_di = 'L1', 'East'
    traffic.dw.cbSelectType.setCurrentText(var)
    traffic.update_traffic_tbl('type')
    traffic.run_update = True
    return traffic.traffic_data['L1']['East']


def _select(traffic, cells):
    from qgis.PyQt.QtCore import QItemSelectionModel
    tw = traffic.dw.twTrafficData
    tw.clearSelection()
    for r, c in cells:
        tw.selectionModel().select(tw.model().index(r, c), QItemSelectionModel.SelectionFlag.Select)


# ---------------------------------------------------------------------------
# fill_cbTrafficSelectSeg
# ---------------------------------------------------------------------------

class TestFillSegSelect:
    def test_populates_from_route_table(self, traffic, qgis_iface):
        from qgis.PyQt.QtWidgets import QTableWidget, QTableWidgetItem
        # Route list has at least 3 columns: Segment_Id(0), Route_Id(1), Leg_name(2).
        traffic.dw.twRouteList = QTableWidget()
        traffic.dw.twRouteList.setColumnCount(3)
        traffic.dw.twRouteList.setRowCount(2)
        traffic.dw.twRouteList.setItem(0, 0, QTableWidgetItem('1'))
        traffic.dw.twRouteList.setItem(0, 2, QTableWidgetItem('LEG_1_1'))
        traffic.dw.twRouteList.setItem(1, 0, QTableWidgetItem('2'))
        traffic.dw.twRouteList.setItem(1, 2, QTableWidgetItem('LEG_1_2'))
        traffic.fill_cbTrafficSelectSeg()
        assert traffic.dw.cbTrafficSelectSeg.count() == 2
        # Display text is the leg name; stored data is the segment ID.
        assert traffic.dw.cbTrafficSelectSeg.itemText(0) == 'LEG_1_1'
        assert traffic.dw.cbTrafficSelectSeg.itemText(1) == 'LEG_1_2'
        assert traffic.dw.cbTrafficSelectSeg.itemData(0) == '1'
        assert traffic.dw.cbTrafficSelectSeg.itemData(1) == '2'


# ---------------------------------------------------------------------------
# set_table_headings -- the fontMetrics / sizing fallback
# ---------------------------------------------------------------------------

class TestSetTableHeadings:
    def test_headings_from_ship_cat(self, traffic):
        traffic.set_table_headings()
        # 2 types x 2 size bins from the fixture + 1 Total row/col each.
        assert traffic.dw.twTrafficData.rowCount() == 3
        assert traffic.dw.twTrafficData.columnCount() == 3

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
        rows = traffic._data_rows
        cols = traffic._data_cols
        for r in range(rows):
            for c in range(cols):
                traffic.traffic_data['L1']['East']['Speed (knots)'][r][c] = 0
        traffic.traffic_data['L1']['East']['Speed (knots)'][0][0] = val
        traffic.c_seg = 'L1'
        traffic.c_di = 'East'
        traffic.last_var = 'Speed (knots)'
        traffic.dw.cbTrafficSelectSeg.addItem('LEG_1_L1', 'L1')
        traffic.dw.cbTrafficDirectionSelect.addItem('East')
        # cbSelectType has pre-populated items from the .ui; pick Speed (knots).
        traffic.dw.cbSelectType.setCurrentText('Speed (knots)')

    def _cell(self, traffic, r=0, c=0):
        return traffic.dw.twTrafficData.item(r, c)

    def test_cells_are_plain_items_not_spinboxes(self, traffic):
        self._prime(traffic, 3)
        traffic.update_traffic_tbl('type')
        assert traffic.dw.twTrafficData.cellWidget(0, 0) is None
        assert self._cell(traffic).text() == '3'

    def test_empty_string_value_is_blank_and_kept(self, traffic):
        self._prime(traffic, '')
        traffic.update_traffic_tbl('type')
        assert self._cell(traffic).text() == ''
        traffic.save()
        assert traffic.traffic_data['L1']['East']['Speed (knots)'][0][0] == ''

    def test_inf_value_is_blank_editable_and_survives_save(self, traffic):
        """Before 2026-10-16 an inf cell was a disabled spinbox showing 0,
        and save() wrote that 0 back."""
        import numpy as np
        from qgis.PyQt.QtCore import Qt
        self._prime(traffic, np.inf)
        traffic.update_traffic_tbl('type')
        item = self._cell(traffic)
        assert item.text() == ''
        assert item.toolTip()
        assert bool(item.flags() & Qt.ItemFlag.ItemIsEditable)
        traffic.save()
        assert traffic.traffic_data['L1']['East']['Speed (knots)'][0][0] == np.inf

    def test_float_value_uses_locale_decimal_point(self, traffic, locale_sv):
        self._prime(traffic, 12.5)
        traffic.update_traffic_tbl('type')
        assert self._cell(traffic).text() == '12,5'

    def test_viewing_never_rounds_the_stored_value(self, traffic):
        self._prime(traffic, 12.3456)
        traffic.update_traffic_tbl('type')
        traffic.save()
        assert traffic.traffic_data['L1']['East']['Speed (knots)'][0][0] == 12.3456

    def test_large_fractional_frequency_not_clamped(self, traffic):
        """QDoubleSpinBox's default maximum (99.99) clamped 1234.5 ships/year."""
        self._prime(traffic, 0)
        traffic.traffic_data['L1']['East']['Frequency (ships/year)'][0][0] = 1234.5
        traffic.last_var = 'Frequency (ships/year)'
        traffic.dw.cbSelectType.setCurrentText('Frequency (ships/year)')
        traffic.update_traffic_tbl('type')
        traffic.save()
        assert traffic.traffic_data['L1']['East']['Frequency (ships/year)'][0][0] == 1234.5

    def test_segment_caller_updates_direction_select(self, traffic):
        traffic.set_table_headings()
        traffic.create_empty_dict('L1', ['East', 'West'])
        traffic.c_seg = 'L1'
        # addItem(display, userData): display is the leg name, data is the seg key.
        traffic.dw.cbTrafficSelectSeg.addItem('LEG_1_L1', 'L1')
        traffic.dw.cbTrafficSelectSeg.setCurrentIndex(0)
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
    def test_typed_values_are_written_through(self, traffic, locale_en):
        data = _show(traffic)
        tw = traffic.dw.twTrafficData
        for row in range(2):
            for col in range(2):
                tw.item(row, col).setText(str(10 * row + col))   # fires itemChanged
        assert data[FREQ] == [[0, 1], [10, 11]]

    def test_save_picks_up_text_changed_without_signal(self, traffic, locale_en):
        data = _show(traffic)
        tw = traffic.dw.twTrafficData
        tw.blockSignals(True)
        tw.item(1, 0).setText('42')
        tw.blockSignals(False)
        traffic.save()
        assert data[FREQ][1][0] == 42

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

    def test_typed_scaling_values_stored(self, traffic, locale_sv):
        from omrat_utils.handle_traffic import SCALING_VAR
        data = _show(traffic, SCALING_VAR)
        tw = traffic.dw.twTrafficData
        tw.item(0, 0).setText('130,5')
        tw.item(1, 1).setText('200')
        traffic.save()
        assert data[SCALING_VAR][0][0] == pytest.approx(130.5)
        assert data[SCALING_VAR][1][1] == pytest.approx(200.0)
        assert data[SCALING_VAR][0][1] == pytest.approx(100.0)

    def test_typed_scaling_value_unticks_its_row(self, traffic, locale_en):
        from omrat_utils.handle_traffic import SCALING_VAR
        _show(traffic, SCALING_VAR)
        traffic._project_scaling()['follow_global'] = [True, True]
        traffic.dw.twTrafficData.item(1, 0).setText('150')
        assert traffic._project_scaling()['follow_global'] == [True, False]


class TestExcelCopyPaste:
    """Plain-text cells (2026-10-16): select, copy to Excel, paste back."""

    def test_invalid_text_reverts_and_keeps_value(self, traffic, locale_en):
        data = _show(traffic)
        tw = traffic.dw.twTrafficData
        tw.item(0, 0).setText('12')
        tw.item(0, 0).setText('abc')
        assert tw.item(0, 0).text() == '12' and data[FREQ][0][0] == 12
        tw.item(0, 0).setText('-5')
        assert data[FREQ][0][0] == 12

    def test_typed_text_is_normalised(self, traffic, locale_sv):
        data = _show(traffic, 'Speed (knots)')
        tw = traffic.dw.twTrafficData
        tw.item(0, 1).setText('12.50')          # dot typed in a Swedish locale
        assert data['Speed (knots)'][0][1] == 12.5
        assert tw.item(0, 1).text() == '12,5'

    def test_totals_follow_edits(self, traffic, locale_en):
        _show(traffic)
        tw = traffic.dw.twTrafficData
        tw.item(0, 0).setText('10')
        tw.item(1, 0).setText('5')
        tw.item(1, 1).setText('3')
        assert tw.item(0, 2).text() == '10'
        assert tw.item(2, 0).text() == '15'
        assert tw.item(2, 2).text() == '18'

    def test_totals_are_read_only_and_not_selectable(self, traffic):
        from qgis.PyQt.QtCore import Qt
        _show(traffic)
        flags = traffic.dw.twTrafficData.item(2, 2).flags()
        assert not flags & Qt.ItemFlag.ItemIsEditable
        assert not flags & Qt.ItemFlag.ItemIsSelectable

    def test_copy_selection_as_tsv(self, traffic, locale_sv):
        data = _show(traffic, 'Speed (knots)')
        data['Speed (knots)'][0] = [12.5, 8]
        data['Speed (knots)'][1] = [float('inf'), 14.25]
        _show(traffic, 'Speed (knots)')
        traffic.dw.twTrafficData.selectAll()
        text = traffic.copy_selection()
        assert text == '12,5\t8\n\t14,25'          # totals left out, inf blank

    def test_copy_puts_text_on_the_clipboard(self, traffic, monkeypatch):
        from omrat_utils import handle_traffic
        clip = MagicMock()
        monkeypatch.setattr(handle_traffic.QApplication, 'clipboard', staticmethod(lambda: clip))
        _show(traffic)
        traffic.dw.twTrafficData.selectAll()
        text = traffic.copy_selection()
        clip.setText.assert_called_once_with(text)

    def test_paste_reads_the_clipboard(self, traffic, locale_en, monkeypatch):
        from omrat_utils import handle_traffic
        clip = MagicMock()
        clip.text.return_value = '5\t6'
        monkeypatch.setattr(handle_traffic.QApplication, 'clipboard', staticmethod(lambda: clip))
        data = _show(traffic)
        _select(traffic, [(1, 0)])
        traffic.paste_clipboard()
        assert data[FREQ][1] == [5, 6]

    def test_copy_with_headers(self, traffic, locale_en):
        data = _show(traffic)
        data[FREQ][0] = [3, 4]
        _show(traffic)
        lines = traffic.copy_selection(with_headers=True).split('\n')
        assert lines[0].split('\t')[1:] == ['0 - 25', '25 - 50']
        assert lines[0].startswith('LEG_1_L1 - East - Frequency')
        assert lines[1].split('\t') == ['Cargo', '3', '4']
        assert len(lines) == 3

    def test_paste_block_at_selected_cell(self, traffic, locale_sv):
        data = _show(traffic)
        _select(traffic, [(0, 1)])
        plan = traffic.paste_clipboard('7\n9\n11\r\n')    # 3 rows, table has 2
        assert data[FREQ] == [[0, 7], [0, 9]]
        assert plan.clipped == 1

    def test_paste_swedish_excel_numbers(self, traffic, locale_sv):
        data = _show(traffic, 'Speed (knots)')
        _select(traffic, [(0, 0)])
        traffic.paste_clipboard('12,5\t1 234,5\r\n8\t\r\n')
        assert data['Speed (knots)'][0] == [12.5, 1234.5]
        assert data['Speed (knots)'][1][0] == 8.0
        assert data['Speed (knots)'][1][1] == float('inf')   # blank = no observation

    def test_paste_back_what_was_copied_with_headers(self, traffic, locale_en):
        data = _show(traffic)
        data[FREQ][0] = [3, 4]
        data[FREQ][1] = [5, 6]
        _show(traffic)
        text = traffic.copy_selection(with_headers=True)
        edited = text.replace('\t3\t', '\t30\t')            # "edited in Excel"
        _select(traffic, [(1, 1)])                         # anchor ignored for a labelled block
        traffic.paste_clipboard(edited)
        assert data[FREQ] == [[30, 4], [5, 6]]

    def test_one_value_fills_the_selection(self, traffic, locale_en):
        data = _show(traffic)
        _select(traffic, [(0, 0), (0, 1), (1, 1)])
        traffic.paste_clipboard('9')
        assert data[FREQ] == [[9, 9], [0, 9]]

    def test_paste_skips_invalid_cells(self, traffic, locale_en):
        data = _show(traffic)
        _select(traffic, [(0, 0)])
        plan = traffic.paste_clipboard('1\tx\n-3\t4')
        assert data[FREQ] == [[1, 0], [0, 4]]
        assert len(plan.cells) == 4

    def test_paste_updates_totals(self, traffic, locale_en):
        _show(traffic)
        _select(traffic, [(0, 0)])
        traffic.paste_clipboard('1\t2\n3\t4')
        assert traffic.dw.twTrafficData.item(2, 2).text() == '10'

    def test_paste_scaling_unticks_rows(self, traffic, locale_en):
        from omrat_utils.handle_traffic import SCALING_VAR
        _show(traffic, SCALING_VAR)
        traffic._project_scaling()['follow_global'] = [True, True]
        _select(traffic, [(0, 0)])
        traffic.paste_clipboard('120\t130')
        assert traffic._project_scaling()['follow_global'] == [False, True]

    def test_delete_clears_per_variable(self, traffic, locale_en):
        import numpy as np
        data = _show(traffic)
        data[FREQ][0] = [3, 4]
        data['Speed (knots)'][0] = [12.0, 13.0]
        _show(traffic)
        _select(traffic, [(0, 0)])
        traffic.clear_selection()
        assert data[FREQ][0] == [0, 4]
        _show(traffic, 'Speed (knots)')
        _select(traffic, [(0, 1)])
        traffic.clear_selection()
        assert data['Speed (knots)'][0] == [12.0, np.inf]

    def test_ctrl_c_ctrl_v_and_delete_keys(self, traffic, locale_en, monkeypatch):
        # Routing only: the system clipboard on Windows can be held by
        # another process, so the copy / paste bodies are tested directly.
        from qgis.PyQt.QtCore import QEvent, Qt
        from qgis.PyQt.QtGui import QKeyEvent
        from qgis.PyQt.QtWidgets import QApplication
        data = _show(traffic)
        data[FREQ][1] = [3, 4]
        _show(traffic)
        calls = []
        monkeypatch.setattr(traffic, 'copy_selection', lambda *a, **k: calls.append('copy'))
        monkeypatch.setattr(traffic, 'paste_clipboard', lambda *a, **k: calls.append('paste'))
        tw = traffic.dw.twTrafficData
        ctrl = Qt.KeyboardModifier.ControlModifier
        assert QApplication.sendEvent(tw, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_C, ctrl))
        assert QApplication.sendEvent(tw, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_V, ctrl))
        assert calls == ['copy', 'paste']
        _select(traffic, [(1, 1)])
        no_mod = Qt.KeyboardModifier.NoModifier
        QApplication.sendEvent(tw, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, no_mod))
        assert data[FREQ][1] == [3, 0]

    def test_context_menu(self, traffic):
        _show(traffic)
        labels = [a.text() for a in traffic._build_table_menu().actions()]
        assert labels == ['Copy', 'Copy table with headers', 'Paste', 'Clear', 'Select all']
