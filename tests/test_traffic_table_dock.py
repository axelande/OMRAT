"""The Traffic Data table in the *real* plugin dock (QGIS conftest).

``tests/test_handle_traffic.py`` drives ``Traffic`` on the standalone
``TrafficDataWidget``, which skips the dock styling -- and that styling
made every dock table single-row select, so ranges could not be
selected (2026-10-16).  QGIS also binds Ctrl+C / Ctrl+V / Del / Ctrl+A
to *feature* actions in its main window; the table must claim them.
"""
from __future__ import annotations

import pytest

FREQ = 'Frequency (ships/year)'


@pytest.fixture
def dock(omrat):
    """Leg 77 with traffic, shown on the Traffic tab of the real dock."""
    from qgis.PyQt.QtWidgets import QTableWidgetItem
    omrat.testing = True
    omrat.segment_data['77'] = {
        'Start_Point': '14.000000 55.000000', 'End_Point': '14.200000 55.000000',
        'Width': 5000, 'Route_Id': 1, 'Leg_name': 'LEG_1_77', 'Segment_Id': '77',
        'Dirs': ['East going', 'West going'], 'line_length': 12000.0, 'Tangent_Pos': 0.5,
    }
    omrat.traffic.create_empty_dict('77', ['East going', 'West going'])
    omrat.reset_route_table()
    tbl = omrat.main_widget.twRouteList
    tbl.setRowCount(1)
    for col, text in enumerate(['77', '1', 'LEG_1_77', '14.000000 55.000000', '14.200000 55.000000',
                                '5000', '50']):
        tbl.setItem(0, col, QTableWidgetItem(text))
    omrat.run_traffic_module()
    omrat.main_widget.cbSelectType.setCurrentText(FREQ)
    omrat.traffic.update_traffic_tbl('segment')
    msgs: list[str] = []
    omrat.notifier.display_message = lambda m, **_k: msgs.append(m)
    yield omrat.traffic, msgs
    omrat.segment_data.pop('77', None)
    omrat.traffic_data.pop('77', None)


def test_traffic_table_selects_ranges_like_a_spreadsheet(omrat, dock):
    from qgis.PyQt.QtWidgets import QAbstractItemView
    tw = omrat.main_widget.twTrafficData
    assert tw.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
    assert tw.selectionBehavior() == QAbstractItemView.SelectionBehavior.SelectItems
    assert tw.isCornerButtonEnabled()
    # The other dock tables keep the single-row style.
    assert omrat.main_widget.twRouteList.selectionMode() == QAbstractItemView.SelectionMode.SingleSelection


def test_several_rows_can_be_selected_and_copied(omrat, dock):
    from qgis.PyQt.QtCore import QItemSelectionModel
    traffic, _msgs = dock
    m = traffic._current_matrix()
    m[0][0], m[1][0], m[2][0] = 1, 2, 3
    traffic.update_traffic_tbl('type')
    tw = omrat.main_widget.twTrafficData
    for r in (0, 1, 2):
        tw.selectionModel().select(tw.model().index(r, 0), QItemSelectionModel.SelectionFlag.Select)
    assert traffic.copy_selection() == '1\n2\n3'


@pytest.mark.parametrize('key, mod', [
    ('Key_C', 'ControlModifier'), ('Key_V', 'ControlModifier'),
    ('Key_A', 'ControlModifier'), ('Key_Delete', 'NoModifier'),
])
def test_table_claims_keys_qgis_binds_to_feature_actions(omrat, dock, key, mod):
    from qgis.PyQt.QtCore import QEvent, Qt
    from qgis.PyQt.QtGui import QKeyEvent
    from qgis.PyQt.QtWidgets import QApplication
    ev = QKeyEvent(QEvent.Type.ShortcutOverride, getattr(Qt.Key, key), getattr(Qt.KeyboardModifier, mod))
    ev.ignore()
    QApplication.sendEvent(omrat.main_widget.twTrafficData, ev)
    assert ev.isAccepted()


def test_menu_paste_reports_what_changed(omrat, dock, monkeypatch):
    from omrat_utils import handle_traffic
    traffic, msgs = dock
    text = traffic.copy_selection(with_headers=True)
    clip = type('Clip', (), {'text': lambda self: text, 'setText': lambda self, t: None})()
    monkeypatch.setattr(handle_traffic.QApplication, 'clipboard', staticmethod(lambda: clip))
    acts = {a.text(): a for a in traffic._build_table_menu().actions()}
    acts['Paste'].trigger()                       # the unchanged copy: nothing changes...
    assert msgs[-1] == f'Pasted 315 cell(s) into {FREQ}, 0 changed'   # ...and it says so
    text = text.replace('\t0\t', '\t9\t', 1)
    acts['Paste'].trigger()
    assert msgs[-1].endswith('1 changed')
    assert traffic._current_matrix()[0][0] == 9


def test_block_pasted_into_one_cell_editor_lands_as_a_block(omrat, dock, monkeypatch):
    """The cell editor's own right-click Paste puts the whole TSV in one
    cell; it is pasted as a block from that cell instead."""
    from omrat_utils import handle_traffic
    traffic, _msgs = dock
    block = '4\t5\n6\t7'
    clip = type('Clip', (), {'text': lambda self: block, 'setText': lambda self, t: None})()
    monkeypatch.setattr(handle_traffic.QApplication, 'clipboard', staticmethod(lambda: clip))
    omrat.main_widget.twTrafficData.item(2, 3).setText(block.replace('\n', ' '))  # what the editor holds
    m = traffic._current_matrix()
    assert [m[2][3], m[2][4], m[3][3], m[3][4]] == [4, 5, 6, 7]
    assert omrat.main_widget.twTrafficData.item(2, 3).text() == '4'


def test_paste_with_nothing_to_paste_says_so(omrat, dock):
    traffic, msgs = dock
    assert traffic.paste_clipboard('') is None
    assert msgs[-1] == 'Paste: the clipboard holds no table cells.'
    traffic.c_seg = ''
    assert traffic.paste_clipboard('1') is None
    assert msgs[-1].startswith('Paste: pick a leg')
