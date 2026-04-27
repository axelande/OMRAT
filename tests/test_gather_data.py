"""Unit tests for omrat_utils/gather_data.py.

Focuses on the pure-Python helpers and the ship-categories save/load
roundtrip.  The `populate()` path with a real QGIS fixture is exercised
by `test_load_data.py`; this file covers the rest.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.gather_data import (
    GatherData, dict_ndarray_to_list, dict_list_to_ndarray,
)


@pytest.fixture
def gd_minimal():
    """GatherData instance with a stub parent sufficient for the pure
    helpers."""
    plugin = MagicMock()
    gd = GatherData(plugin)
    return gd


# ---------------------------------------------------------------------------
# normalize_depths_for_ui / normalize_objects_for_ui
# ---------------------------------------------------------------------------

class TestNormalizeDepthsAndObjects:
    def test_depths_passthrough_for_list_rows(self, gd_minimal):
        rows = gd_minimal.normalize_depths_for_ui(
            [['d1', '6.0', 'POLY(...)']])
        assert rows == [['d1', '6.0', 'POLY(...)']]

    def test_depths_non_list_returns_empty(self, gd_minimal):
        assert gd_minimal.normalize_depths_for_ui(None) == []  # type: ignore[arg-type]
        assert gd_minimal.normalize_depths_for_ui('not a list') == []  # type: ignore[arg-type]

    def test_depths_short_string_falls_back(self, gd_minimal):
        """Strings shorter than 3 characters raise IndexError during
        unpack and trigger the fallback `[str(item), '', '']`."""
        rows = gd_minimal.normalize_depths_for_ui(['ab'])
        assert rows == [['ab', '', '']]

    def test_depths_long_string_treated_as_sequence(self, gd_minimal):
        """Long strings are indexable, so the first 3 characters are
        used (known behaviour of the current indexing)."""
        rows = gd_minimal.normalize_depths_for_ui(['just-a-str'])
        assert rows == [['j', 'u', 's']]

    def test_objects_passthrough(self, gd_minimal):
        rows = gd_minimal.normalize_objects_for_ui(
            [['s1', '12.0', 'POLY(...)']])
        assert rows == [['s1', '12.0', 'POLY(...)']]

    def test_objects_non_list_returns_empty(self, gd_minimal):
        assert gd_minimal.normalize_objects_for_ui(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# obtain_table_data
# ---------------------------------------------------------------------------

class TestObtainTableData:
    def test_reads_all_cells_as_text(self, gd_minimal):
        tbl = MagicMock()
        tbl.rowCount.return_value = 2
        tbl.columnCount.return_value = 3
        # 6 cells: cell.text() returns str(r*10 + c)
        tbl.item.side_effect = lambda r, c: SimpleNamespace(
            text=lambda r=r, c=c: f'{r}-{c}'
        )
        data = gd_minimal.obtain_table_data(tbl)
        assert data == [['0-0', '0-1', '0-2'], ['1-0', '1-1', '1-2']]

    def test_none_items_skipped(self, gd_minimal):
        tbl = MagicMock()
        tbl.rowCount.return_value = 1
        tbl.columnCount.return_value = 3
        tbl.item.side_effect = [None, SimpleNamespace(text=lambda: 'x'), None]
        data = gd_minimal.obtain_table_data(tbl)
        # The None values are simply skipped; that row ends up with
        # fewer than 3 entries.
        assert data == [['x']]


# ---------------------------------------------------------------------------
# get_ship_categories_for_save (real widgets via qgis_iface fixture)
# ---------------------------------------------------------------------------

class TestGetShipCategoriesForSave:
    def test_reads_types_and_intervals_from_widgets(self, qgis_iface):
        """Exercise the real ShipCategoriesWidget path: populate
        cvTypes / twLengths, then confirm the save dict round-trips."""
        from unittest.mock import patch
        with patch('omrat_utils.handle_ais.DB'):
            from omrat_utils.handle_ship_cat import ShipCategories
            from qgis.PyQt.QtWidgets import QTableWidgetItem

            sc = ShipCategories(MagicMock())
            # Populate the ship-types table.
            sc.scw.cvTypes.setColumnCount(1)
            sc.scw.cvTypes.setRowCount(2)
            sc.scw.cvTypes.setItem(0, 0, QTableWidgetItem('Cargo'))
            sc.scw.cvTypes.setItem(1, 0, QTableWidgetItem('Tanker'))

            # Populate the length-intervals table.
            sc.scw.twLengths.setColumnCount(2)
            sc.scw.twLengths.setRowCount(2)
            sc.scw.twLengths.setItem(0, 0, QTableWidgetItem('0'))
            sc.scw.twLengths.setItem(0, 1, QTableWidgetItem('25'))
            sc.scw.twLengths.setItem(1, 0, QTableWidgetItem('25'))
            sc.scw.twLengths.setItem(1, 1, QTableWidgetItem('50'))

            parent = SimpleNamespace(ship_cat=sc)
            gd = GatherData(parent)

            data = gd.get_ship_categories_for_save()
            assert data['types'] == ['Cargo', 'Tanker']
            assert len(data['length_intervals']) == 2
            assert data['length_intervals'][0] == {
                'min': 0.0, 'max': 25.0, 'label': '0 - 25'
            }
            assert data['length_intervals'][1] == {
                'min': 25.0, 'max': 50.0, 'label': '25 - 50'
            }

    def test_missing_widgets_returns_default(self, gd_minimal):
        # Parent doesn't have a ship_cat widget tree; get_ship_categories
        # should return the empty-list defaults without raising.
        gd_minimal.p = SimpleNamespace()
        data = gd_minimal.get_ship_categories_for_save()
        assert data == {'types': [], 'length_intervals': []}

    def test_empty_min_max_row_skipped(self, qgis_iface):
        """Rows where both min and max are empty strings are skipped."""
        from unittest.mock import patch
        with patch('omrat_utils.handle_ais.DB'):
            from omrat_utils.handle_ship_cat import ShipCategories
            from qgis.PyQt.QtWidgets import QTableWidgetItem

            sc = ShipCategories(MagicMock())
            sc.scw.twLengths.setColumnCount(2)
            sc.scw.twLengths.setRowCount(2)
            # First row is empty, second is valid.
            sc.scw.twLengths.setItem(0, 0, QTableWidgetItem(''))
            sc.scw.twLengths.setItem(0, 1, QTableWidgetItem(''))
            sc.scw.twLengths.setItem(1, 0, QTableWidgetItem('10'))
            sc.scw.twLengths.setItem(1, 1, QTableWidgetItem('20'))
            sc.scw.cvTypes.setColumnCount(1)
            sc.scw.cvTypes.setRowCount(0)

            parent = SimpleNamespace(ship_cat=sc)
            gd = GatherData(parent)
            data = gd.get_ship_categories_for_save()
            assert len(data['length_intervals']) == 1
            assert data['length_intervals'][0]['min'] == 10.0


# ---------------------------------------------------------------------------
# copy_depths_and_objects (reads from QTableWidgets)
# ---------------------------------------------------------------------------

class TestCopyDepthsAndObjects:
    def test_copies_rows_with_three_cells(self, gd_minimal):
        # Configure the parent's main_widget.twDepthList / twObjectList
        # to simulate two depths and one object.
        plugin = gd_minimal.p
        plugin.main_widget.twDepthList.rowCount.return_value = 2
        plugin.main_widget.twDepthList.columnCount.return_value = 3

        def depth_item(r, c):
            values = {
                (0, 0): 'd0', (0, 1): '0', (0, 2): 'POLYGON((0 0,1 0,1 1,0 1,0 0))',
                (1, 0): 'd1', (1, 1): '6', (1, 2): 'POLYGON((2 2,3 2,3 3,2 3,2 2))',
            }
            return SimpleNamespace(text=lambda r=r, c=c: values[(r, c)])

        plugin.main_widget.twDepthList.item.side_effect = depth_item

        plugin.main_widget.twObjectList.rowCount.return_value = 1
        plugin.main_widget.twObjectList.columnCount.return_value = 3

        def object_item(r, c):
            values = {(0, 0): 'o0', (0, 1): '20', (0, 2): 'POLYGON((5 5,6 5,6 6,5 6,5 5))'}
            return SimpleNamespace(text=lambda r=r, c=c: values[(r, c)])

        plugin.main_widget.twObjectList.item.side_effect = object_item

        gd_minimal.data = {'depths': [], 'objects': []}
        gd_minimal.copy_depths_and_objects()
        assert gd_minimal.data['depths'] == [
            ['d0', '0', 'POLYGON((0 0,1 0,1 1,0 1,0 0))'],
            ['d1', '6', 'POLYGON((2 2,3 2,3 3,2 3,2 2))'],
        ]
        assert gd_minimal.data['objects'] == [
            ['o0', '20', 'POLYGON((5 5,6 5,6 6,5 6,5 5))'],
        ]

    def test_incomplete_rows_skipped(self, gd_minimal):
        """Rows with fewer than 3 cells are dropped by the guard inside
        copy_depths_and_objects."""
        plugin = gd_minimal.p
        plugin.main_widget.twDepthList.rowCount.return_value = 1
        plugin.main_widget.twDepthList.columnCount.return_value = 2
        plugin.main_widget.twDepthList.item.side_effect = [
            SimpleNamespace(text=lambda: 'only'),
            SimpleNamespace(text=lambda: 'two'),
        ]
        plugin.main_widget.twObjectList.rowCount.return_value = 0
        plugin.main_widget.twObjectList.columnCount.return_value = 0

        gd_minimal.data = {'depths': [], 'objects': []}
        gd_minimal.copy_depths_and_objects()
        assert gd_minimal.data['depths'] == []

    def test_incomplete_object_rows_skipped(self, gd_minimal):
        """Object rows with fewer than 3 cells trip the len(row) < 3 guard."""
        plugin = gd_minimal.p
        plugin.main_widget.twDepthList.rowCount.return_value = 0
        plugin.main_widget.twDepthList.columnCount.return_value = 0
        plugin.main_widget.twObjectList.rowCount.return_value = 1
        plugin.main_widget.twObjectList.columnCount.return_value = 2
        plugin.main_widget.twObjectList.item.side_effect = [
            SimpleNamespace(text=lambda: 'only'),
            SimpleNamespace(text=lambda: 'two'),
        ]
        gd_minimal.data = {'depths': [], 'objects': []}
        gd_minimal.copy_depths_and_objects()
        assert gd_minimal.data['objects'] == []


# ---------------------------------------------------------------------------
# dict_ndarray_to_list / dict_list_to_ndarray
# ---------------------------------------------------------------------------

class TestDictConverters:
    def test_ndarray_to_list_converts_nested(self):
        import numpy as np
        data = {'a': {'x': np.array([1, 2, 3]), 'y': 'literal'}}
        out = dict_ndarray_to_list(data)
        assert out['a']['x'] == [1, 2, 3]
        assert out['a']['y'] == 'literal'

    def test_ndarray_to_list_passthrough_non_arrays(self):
        data = {'a': {'x': [1, 2, 3]}}  # already list
        out = dict_ndarray_to_list(data)
        assert out['a']['x'] == [1, 2, 3]

    def test_list_to_ndarray_converts_lists(self):
        import numpy as np
        data = {'a': {'x': [1, 2, 3], 'y': 42}}
        out = dict_list_to_ndarray(data)
        assert isinstance(out['a']['x'], np.ndarray)
        assert out['a']['y'] == 42


# ---------------------------------------------------------------------------
# get_segment_tbl
# ---------------------------------------------------------------------------

class TestGetSegmentTbl:
    def test_reads_6_cols_per_segment(self, gd_minimal):
        from unittest.mock import MagicMock
        plugin = gd_minimal.p
        plugin.main_widget.twRouteList = MagicMock()

        def item(r, c):
            values = {
                (0, 0): 'seg_A', (0, 1): '1', (0, 2): 'leg_a',
                (0, 3): '14.0 55.0', (0, 4): '14.2 55.0', (0, 5): '1000',
            }
            return SimpleNamespace(text=lambda r=r, c=c: values.get((r, c), ''))

        plugin.main_widget.twRouteList.item.side_effect = item

        gd_minimal.data = {'segment_data': {'X': {}}}  # one segment
        gd_minimal.get_segment_tbl()

        seg = gd_minimal.data['segment_data']['X']
        assert seg['Segment_Id'] == 'seg_A'
        assert seg['Leg_name'] == 'leg_a'
        assert seg['Start_Point'] == '14.0 55.0'
        assert seg['Width'] == '1000'

    def test_returns_early_on_missing_item(self, gd_minimal):
        """If the QTableWidget.item() returns None mid-loop, the function
        short-circuits without raising."""
        from unittest.mock import MagicMock
        plugin = gd_minimal.p
        plugin.main_widget.twRouteList = MagicMock()
        plugin.main_widget.twRouteList.item.return_value = None

        gd_minimal.data = {'segment_data': {'X': {'Segment_Id': 'keep'}}}
        gd_minimal.get_segment_tbl()
        # Value left unchanged because function exited early.
        assert gd_minimal.data['segment_data']['X']['Segment_Id'] == 'keep'


# ---------------------------------------------------------------------------
# get_ship_categories_for_save -- remaining branches
# ---------------------------------------------------------------------------

class TestGetShipCategoriesRemainingPaths:
    def test_float_conversion_failure_keeps_string(self, qgis_iface):
        """If min / max cells contain non-numeric text, vmin/vmax stay as strings."""
        from unittest.mock import patch
        with patch('omrat_utils.handle_ais.DB'):
            from omrat_utils.handle_ship_cat import ShipCategories
            from qgis.PyQt.QtWidgets import QTableWidgetItem

            sc = ShipCategories(MagicMock())
            sc.scw.twLengths.setColumnCount(2)
            sc.scw.twLengths.setRowCount(1)
            sc.scw.twLengths.setItem(0, 0, QTableWidgetItem('abc'))
            sc.scw.twLengths.setItem(0, 1, QTableWidgetItem('xyz'))
            sc.scw.cvTypes.setColumnCount(1)
            sc.scw.cvTypes.setRowCount(0)

            parent = SimpleNamespace(ship_cat=sc)
            gd = GatherData(parent)
            out = gd.get_ship_categories_for_save()
            # Non-numeric min/max are kept as strings.
            assert out['length_intervals'] == [
                {'min': 'abc', 'max': 'xyz', 'label': 'abc - xyz'}
            ]

    def test_selection_mode_simple_ais(self, qgis_iface):
        from unittest.mock import patch
        with patch('omrat_utils.handle_ais.DB'):
            from omrat_utils.handle_ship_cat import ShipCategories
            sc = ShipCategories(MagicMock())
            # Only test the code that reads radio buttons; stub them.
            sc.scw.radioButton = MagicMock()
            sc.scw.radioButton.isChecked.return_value = True
            sc.scw.radioButton_2 = MagicMock()
            sc.scw.radioButton_2.isChecked.return_value = False
            # Empty the tables so the result focuses on selection_mode.
            sc.scw.twLengths.setColumnCount(2)
            sc.scw.twLengths.setRowCount(0)
            sc.scw.cvTypes.setColumnCount(1)
            sc.scw.cvTypes.setRowCount(0)

            parent = SimpleNamespace(ship_cat=sc)
            gd = GatherData(parent)
            out = gd.get_ship_categories_for_save()
            assert out.get('selection_mode') == 'simple_ais'

    def test_selection_mode_manual(self, qgis_iface):
        from unittest.mock import patch
        with patch('omrat_utils.handle_ais.DB'):
            from omrat_utils.handle_ship_cat import ShipCategories
            sc = ShipCategories(MagicMock())
            sc.scw.radioButton = MagicMock()
            sc.scw.radioButton.isChecked.return_value = False
            sc.scw.radioButton_2 = MagicMock()
            sc.scw.radioButton_2.isChecked.return_value = True
            sc.scw.twLengths.setColumnCount(2)
            sc.scw.twLengths.setRowCount(0)
            sc.scw.cvTypes.setColumnCount(1)
            sc.scw.cvTypes.setRowCount(0)

            parent = SimpleNamespace(ship_cat=sc)
            gd = GatherData(parent)
            out = gd.get_ship_categories_for_save()
            assert out.get('selection_mode') == 'manual'

    def test_selection_mode_radio_exception_kept_none(self, qgis_iface):
        from unittest.mock import patch
        with patch('omrat_utils.handle_ais.DB'):
            from omrat_utils.handle_ship_cat import ShipCategories
            sc = ShipCategories(MagicMock())
            # isChecked raises -> silently swallowed, no selection_mode key.
            sc.scw.radioButton = MagicMock()
            sc.scw.radioButton.isChecked.side_effect = RuntimeError('boom')
            sc.scw.radioButton_2 = MagicMock()
            sc.scw.twLengths.setColumnCount(2)
            sc.scw.twLengths.setRowCount(0)
            sc.scw.cvTypes.setColumnCount(1)
            sc.scw.cvTypes.setRowCount(0)

            parent = SimpleNamespace(ship_cat=sc)
            gd = GatherData(parent)
            out = gd.get_ship_categories_for_save()
            assert 'selection_mode' not in out


# ---------------------------------------------------------------------------
# normalize_objects_for_ui -- exception fallback
# ---------------------------------------------------------------------------

class TestNormalizeObjectsFallback:
    def test_short_string_falls_back(self, gd_minimal):
        """An unindexable or too-short entry produces [str(item), '', '']."""
        rows = gd_minimal.normalize_objects_for_ui(['ab'])  # too short for [0][1][2]
        assert rows == [['ab', '', '']]


# ---------------------------------------------------------------------------
# get_all_for_save -- full stitching
# ---------------------------------------------------------------------------

class TestGetAllForSave:
    def test_assembles_top_level_keys(self, gd_minimal):
        """Stub the parent's sub-objects so get_all_for_save can stitch
        the top-level dict together."""
        import numpy as np
        from unittest.mock import MagicMock
        plugin = gd_minimal.p
        plugin.causation_f.data = {'p_pc': 1.6e-4}
        plugin.drift_values = {'speed': 1.94}
        plugin.distributions.change_dist_segment = MagicMock()
        plugin.distributions.last_id = '1'
        plugin.traffic_data = {}
        plugin.segment_data = {
            '1': {'dist1': np.array([1, 2, 3]), 'dist2': np.array([4, 5, 6])},
        }
        # twRouteList.item returns None so get_segment_tbl exits early.
        plugin.main_widget.twRouteList.item.return_value = None
        # Empty depths/objects tables.
        plugin.main_widget.twDepthList.rowCount.return_value = 0
        plugin.main_widget.twDepthList.columnCount.return_value = 0
        plugin.main_widget.twObjectList.rowCount.return_value = 0
        plugin.main_widget.twObjectList.columnCount.return_value = 0
        # ship_cat stubs so get_ship_categories_for_save finishes.
        plugin.ship_cat.scw.cvTypes.rowCount.return_value = 0
        plugin.ship_cat.scw.twLengths.rowCount.return_value = 0

        out = gd_minimal.get_all_for_save()
        assert out['pc'] == {'p_pc': 1.6e-4}
        assert out['drift'] == {'speed': 1.94}
        assert out['depths'] == []
        assert out['objects'] == []
        # dist1 was converted to list.
        assert out['segment_data']['1']['dist1'] == [1, 2, 3]


# ---------------------------------------------------------------------------
# populate_ship_categories
# ---------------------------------------------------------------------------

class TestPopulateShipCategories:
    def test_populates_types_and_intervals(self, qgis_iface):
        from unittest.mock import patch
        with patch('omrat_utils.handle_ais.DB'):
            from omrat_utils.handle_ship_cat import ShipCategories

            sc = ShipCategories(MagicMock())
            parent = SimpleNamespace(ship_cat=sc, traffic=MagicMock())
            parent.traffic.set_table_headings = MagicMock()
            gd = GatherData(parent)

            gd.populate_ship_categories({
                'types': ['Cargo', 'Tanker'],
                'length_intervals': [
                    {'min': 0, 'max': 25, 'label': '0-25'},
                    {'min': 25, 'max': 50, 'label': '25-50'},
                ],
            })

            assert sc.scw.cvTypes.rowCount() == 2
            assert sc.scw.cvTypes.item(0, 0).text() == 'Cargo'
            assert sc.scw.twLengths.rowCount() == 2
            assert sc.scw.twLengths.item(1, 1).text() == '50'
            # traffic.set_table_headings was called to refresh the traffic table.
            parent.traffic.set_table_headings.assert_called_once()

    def test_empty_categories_still_calls_set_table_headings(self, qgis_iface):
        from unittest.mock import patch
        with patch('omrat_utils.handle_ais.DB'):
            from omrat_utils.handle_ship_cat import ShipCategories

            sc = ShipCategories(MagicMock())
            parent = SimpleNamespace(ship_cat=sc, traffic=MagicMock())
            parent.traffic.set_table_headings = MagicMock()
            gd = GatherData(parent)
            gd.populate_ship_categories({})
            parent.traffic.set_table_headings.assert_called_once()
