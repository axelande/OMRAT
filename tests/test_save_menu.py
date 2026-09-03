"""File -> Save / Save as... on the live plugin (QGIS fixture)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _menu_texts(omrat):
    from qgis.PyQt.QtWidgets import QMenuBar
    texts = []
    for bar in omrat.main_widget.findChildren(QMenuBar):
        for top in bar.actions():
            menu = top.menu()
            if menu is not None and top.text() == 'File':
                texts.extend(a.text() for a in menu.actions() if not a.isSeparator())
    return texts


class TestSaveMenu:
    def test_menu_has_save_and_save_as(self, omrat):
        texts = _menu_texts(omrat)
        assert 'Save' in texts
        assert 'Save as...' in texts
        assert texts.index('Save') < texts.index('Save as...') < texts.index('Load')

    def test_save_without_path_falls_back_to_dialog(self, omrat, monkeypatch, tmp_path):
        import omrat_utils.storage as storage_mod
        target = str(tmp_path / 'first.omrat')
        omrat.project_path = None
        monkeypatch.setattr(storage_mod.Storage, 'new_file_path', lambda self, *a, **k: (target, ''))
        omrat.save_work()
        assert (tmp_path / 'first.omrat').exists()
        assert omrat.project_path == target

    def test_save_reuses_known_path_without_dialog(self, omrat, monkeypatch, tmp_path):
        import omrat_utils.storage as storage_mod
        target = str(tmp_path / 'known.omrat')
        omrat.project_path = target

        def boom(self, *a, **k):
            raise AssertionError('Save must not open a dialog when the path is known')

        monkeypatch.setattr(storage_mod.Storage, 'new_file_path', boom)
        omrat.save_work()
        assert (tmp_path / 'known.omrat').exists()
        assert omrat.main_widget.windowTitle().endswith('known.omrat')

    def test_save_as_always_asks(self, omrat, monkeypatch, tmp_path):
        import omrat_utils.storage as storage_mod
        omrat.project_path = str(tmp_path / 'a.omrat')
        asked = []

        def dialog(self, *a, **k):
            asked.append(a)
            return (str(tmp_path / 'b.omrat'), '')

        monkeypatch.setattr(storage_mod.Storage, 'new_file_path', dialog)
        omrat.save_work_as()
        assert asked and asked[0][3] == 'a.omrat'   # pre-filled with the current name
        assert (tmp_path / 'b.omrat').exists()
        assert omrat.project_path == str(tmp_path / 'b.omrat')

    def test_clear_model_forgets_path(self, omrat, tmp_path):
        omrat.project_path = str(tmp_path / 'x.omrat')
        omrat.refresh_project_title()
        assert omrat.main_widget.windowTitle().endswith('x.omrat')
        omrat.clear_model()
        assert omrat.project_path is None
        assert not omrat.main_widget.windowTitle().endswith('x.omrat')


class TestCloseWithUnsavedChanges:
    def _make_dirty(self, omrat):
        omrat.segment_data['901'] = {
            'Start_Point': '14.000000 55.000000', 'End_Point': '14.200000 55.000000',
            'Width': 5000, 'Route_Id': 1, 'Leg_name': 'LEG_1_901', 'Segment_Id': '901',
            'Dirs': ['East going', 'West going'], 'line_length': 12000.0, 'Tangent_Pos': 0.5,
        }

    def test_fresh_plugin_is_clean(self, omrat):
        assert omrat.has_unsaved_changes() is False
        assert omrat.confirm_close() is True

    def test_edit_makes_model_dirty_and_save_marks_clean(self, omrat, monkeypatch, tmp_path):
        import omrat_utils.storage as storage_mod
        self._make_dirty(omrat)
        assert omrat.has_unsaved_changes() is True
        monkeypatch.setattr(storage_mod.Storage, 'new_file_path',
                            lambda self, *a, **k: (str(tmp_path / 'c.omrat'), ''))
        omrat.save_work()
        assert omrat.has_unsaved_changes() is False

    def test_cancel_keeps_dock_open(self, omrat, monkeypatch):
        self._make_dirty(omrat)
        monkeypatch.setattr(omrat, '_ask_save_on_close', lambda: 'cancel')
        assert omrat.confirm_close() is False
        # The dock's closeEvent honours the veto.
        assert omrat.main_widget.close() is False
        assert omrat.pluginIsActive is True

    def test_discard_closes(self, omrat, monkeypatch):
        self._make_dirty(omrat)
        monkeypatch.setattr(omrat, '_ask_save_on_close', lambda: 'discard')
        assert omrat.confirm_close() is True

    def test_save_choice_writes_known_file_and_closes(self, omrat, monkeypatch, tmp_path):
        import omrat_utils.storage as storage_mod
        self._make_dirty(omrat)
        omrat.project_path = str(tmp_path / 'known.omrat')
        monkeypatch.setattr(storage_mod.Storage, 'new_file_path',
                            lambda self, *a, **k: (_ for _ in ()).throw(AssertionError('no dialog expected')))
        monkeypatch.setattr(omrat, '_ask_save_on_close', lambda: 'save')
        assert omrat.confirm_close() is True
        assert (tmp_path / 'known.omrat').exists()

    def test_save_as_cancelled_keeps_dock_open(self, omrat, monkeypatch):
        import omrat_utils.storage as storage_mod
        self._make_dirty(omrat)
        monkeypatch.setattr(storage_mod.Storage, 'new_file_path', lambda self, *a, **k: ('', ''))
        monkeypatch.setattr(omrat, '_ask_save_on_close', lambda: 'save_as')
        assert omrat.confirm_close() is False
        assert omrat.has_unsaved_changes() is True

    def test_save_as_completed_closes(self, omrat, monkeypatch, tmp_path):
        import omrat_utils.storage as storage_mod
        self._make_dirty(omrat)
        monkeypatch.setattr(storage_mod.Storage, 'new_file_path',
                            lambda self, *a, **k: (str(tmp_path / 'new.omrat'), ''))
        monkeypatch.setattr(omrat, '_ask_save_on_close', lambda: 'save_as')
        assert omrat.confirm_close() is True
        assert omrat.project_path == str(tmp_path / 'new.omrat')

    def test_clear_model_is_clean_baseline(self, omrat):
        self._make_dirty(omrat)
        omrat.clear_model()
        assert omrat.has_unsaved_changes() is False

    def test_prompt_not_shown_when_clean(self, omrat, monkeypatch):
        called = []
        monkeypatch.setattr(omrat, '_ask_save_on_close', lambda: called.append(1) or 'cancel')
        assert omrat.confirm_close() is True
        assert called == []


class TestStripRunTimestamp:
    def test_strip(self):
        from omrat_utils.storage import Storage
        assert Storage.strip_run_timestamp('test14_20260827_232733.omrat') == 'test14.omrat'
        assert Storage.strip_run_timestamp('baltic1.omrat') == 'baltic1.omrat'
        assert Storage.strip_run_timestamp('_20260827_232733.omrat') == '_20260827_232733.omrat'
        assert Storage.strip_run_timestamp('x_2026_1.omrat') == 'x_2026_1.omrat'


class TestSaveOntoReadOnlySnapshot:
    """A project loaded from a run snapshot (written read-only) is simply
    overwritten by Save -- no prompt, no PermissionError."""

    @staticmethod
    def _read_only(path):
        import stat
        path.chmod(path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)

    @staticmethod
    def _restore(path):
        import stat
        if path.exists():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)

    def test_save_overwrites_read_only_file_without_dialog(self, omrat, monkeypatch, tmp_path):
        import os
        import omrat_utils.storage as storage_mod
        snapshot = tmp_path / 'test14_20260827_232733.omrat'
        snapshot.write_text('{}')
        self._read_only(snapshot)
        omrat.project_path = str(snapshot)
        monkeypatch.setattr(storage_mod.Storage, 'new_file_path',
                            lambda self, *a, **k: (_ for _ in ()).throw(AssertionError('no dialog expected')))
        try:
            omrat.save_work()
            assert os.access(str(snapshot), os.W_OK)
            assert snapshot.read_text() != '{}'
            assert omrat.project_path == str(snapshot)
            assert omrat.has_unsaved_changes() is False
            omrat.save_work()   # second save: still no dialog
        finally:
            self._restore(snapshot)

    def test_save_as_from_read_only_prefills_name_without_timestamp(self, omrat, monkeypatch, tmp_path):
        import omrat_utils.storage as storage_mod
        snapshot = tmp_path / 'test14_20260827_232733.omrat'
        snapshot.write_text('{}')
        self._read_only(snapshot)
        omrat.project_path = str(snapshot)
        asked = []

        def dialog(self, *a, **k):
            asked.append(a)
            return (str(tmp_path / 'renamed.omrat'), '')

        monkeypatch.setattr(storage_mod.Storage, 'new_file_path', dialog)
        try:
            omrat.save_work_as()
        finally:
            self._restore(snapshot)
        assert asked and asked[0][3] == 'test14.omrat'
        assert (tmp_path / 'renamed.omrat').exists()
        assert snapshot.read_text() == '{}'


class TestTrafficEditsCountAsUnsaved:
    def _leg_with_traffic(self, omrat):
        from qgis.PyQt.QtWidgets import QTableWidgetItem
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
        omrat.traffic.update_traffic_tbl('segment')

    def test_get_all_for_save_flushes_traffic_table(self, omrat):
        from omrat_utils.gather_data import GatherData
        calls = []
        omrat.traffic.save = lambda: calls.append(1)
        GatherData(omrat).get_all_for_save()
        assert calls == [1]

    def test_spinbox_edit_makes_model_dirty(self, omrat):
        self._leg_with_traffic(omrat)
        omrat.main_widget.cbSelectType.setCurrentText('Frequency (ships/year)')
        omrat.traffic.update_traffic_tbl('type')
        omrat.mark_project_saved()
        assert omrat.has_unsaved_changes() is False
        cell = omrat.main_widget.twTrafficData.cellWidget(0, 0)
        assert cell is not None
        cell.setValue(cell.value() + 7)
        assert omrat.has_unsaved_changes() is True
        # and the value is what gets saved
        from omrat_utils.gather_data import GatherData
        data = GatherData(omrat).get_all_for_save()
        di = omrat.traffic.c_di
        assert data['traffic_data']['77'][di]['Frequency (ships/year)'][0][0] == cell.value()


class TestCloseRemovesLayers:
    def test_closing_dock_clears_model_and_layers(self, omrat, monkeypatch):
        from qgis.core import QgsProject
        seg = {
            '77': {
                'Start_Point': '14.000000 55.000000', 'End_Point': '14.200000 55.000000',
                'Width': 5000, 'Route_Id': 1, 'Leg_name': 'LEG_1_77', 'Segment_Id': '77',
                'Dirs': ['East going', 'West going'], 'line_length': 12000.0, 'Tangent_Pos': 0.5,
            },
        }
        omrat.segment_data = dict(seg)
        omrat.load_lines({'segment_data': seg})
        assert QgsProject.instance().mapLayersByName('LEG_1_77')
        assert QgsProject.instance().mapLayersByName('Tangent Line')
        monkeypatch.setattr(omrat, '_ask_save_on_close', lambda: 'discard')

        assert omrat.main_widget.close() is True

        assert omrat.segment_data == {}
        assert omrat.traffic_data == {}
        assert QgsProject.instance().mapLayersByName('LEG_1_77') == []
        assert QgsProject.instance().mapLayersByName('Tangent Line') == []
        assert omrat.pluginIsActive is False

    def test_cancel_keeps_layers(self, omrat, monkeypatch):
        from qgis.core import QgsProject
        seg = {
            '78': {
                'Start_Point': '14.000000 56.000000', 'End_Point': '14.200000 56.000000',
                'Width': 5000, 'Route_Id': 1, 'Leg_name': 'LEG_1_78', 'Segment_Id': '78',
                'Dirs': ['East going', 'West going'], 'line_length': 12000.0, 'Tangent_Pos': 0.5,
            },
        }
        omrat.segment_data = dict(seg)
        omrat.load_lines({'segment_data': seg})
        monkeypatch.setattr(omrat, '_ask_save_on_close', lambda: 'cancel')
        assert omrat.main_widget.close() is False
        assert QgsProject.instance().mapLayersByName('LEG_1_78')
        assert '78' in omrat.segment_data
