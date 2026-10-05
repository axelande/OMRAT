"""Previous runs grouped per project + OMRAT version column (QGIS conftest).

Every ``RunHistory()`` is routed to a temp sqlite file and QSettings to a
dict, so the user's own history is never touched.
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

EXAMPLE = ROOT / 'tests' / 'example_data' / 'proj.omrat'


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
    import omrat_utils.run_history as rh
    import omrat_utils.run_history_mixin as rhm
    real = rh.RunHistory
    db = tmp_path / 'history.sqlite'
    monkeypatch.setattr(rh, 'RunHistory', lambda db_path=None: real(db))
    _FakeSettings.store = {}
    monkeypatch.setattr(rhm, 'QSettings', _FakeSettings)
    return real(db)


@pytest.fixture
def plugin(omrat, history):
    """Plugin with no current project: empty name, no output folder."""
    omrat._main_run_id = None
    omrat.main_widget.LEProjectName.setText('')
    omrat.main_widget.LEReportPath.setText('')
    omrat._runs_project_filter = None
    return omrat


def _shown_names(omrat) -> list[str]:
    tw = omrat.main_widget.TWPreviousRuns
    return [tw.item(r, 0).text() for r in range(tw.rowCount())]


def _combo_texts(omrat) -> list[str]:
    cb = omrat.main_widget.cbRunProject
    return [cb.itemText(i) for i in range(cb.count())]


def _three_runs(history, tmp_path):
    history.save_run('a1', output_dir=str(tmp_path / 'x'), project='Alpha', omrat_version='0.16.11')
    history.save_run('a2', output_dir=str(tmp_path / 'x'), project='Alpha', omrat_version='0.16.11')
    history.save_run('k1', output_dir=str(tmp_path / 'Kattegat'))


class TestColumns:
    def test_version_and_project_columns(self, plugin, history, tmp_path):
        history.save_run('new', output_dir=str(tmp_path / 'Kattegat'), omrat_version='0.16.11')
        history.save_run('old', output_dir=str(tmp_path / 'Kattegat'))
        plugin.refresh_previous_runs_table()
        tw = plugin.main_widget.TWPreviousRuns
        headers = [tw.horizontalHeaderItem(c).text() for c in range(tw.columnCount())]
        assert headers == ['Name', 'Main', 'Date', 'Duration', 'Version', 'Project']
        versions = {tw.item(r, 0).text(): tw.item(r, 4).text() for r in range(tw.rowCount())}
        assert versions == {'new': '0.16.11', 'old': '—'}


class TestProjectFilter:
    def test_no_current_project_shows_all(self, plugin, history, tmp_path):
        _three_runs(history, tmp_path)
        plugin.refresh_previous_runs_table()
        assert _combo_texts(plugin) == ['All projects (3)', 'Alpha (2)', 'Kattegat (1)']
        assert sorted(_shown_names(plugin)) == ['a1', 'a2', 'k1']
        tw = plugin.main_widget.TWPreviousRuns
        assert not tw.isColumnHidden(5)

    def test_follows_the_project_name(self, plugin, history, tmp_path):
        _three_runs(history, tmp_path)
        plugin.main_widget.LEProjectName.setText('Alpha')
        plugin.show_current_project_runs()
        assert sorted(_shown_names(plugin)) == ['a1', 'a2']
        assert plugin.main_widget.cbRunProject.currentText() == 'Alpha (2)'
        assert plugin.main_widget.TWPreviousRuns.isColumnHidden(5)

    def test_untagged_runs_follow_the_output_folder(self, plugin, history, tmp_path):
        _three_runs(history, tmp_path)
        (tmp_path / 'Kattegat').mkdir()
        plugin.main_widget.LEReportPath.setText(str(tmp_path / 'Kattegat'))
        plugin.show_current_project_runs()
        assert _shown_names(plugin) == ['k1']

    def test_picking_a_project_sticks(self, plugin, history, tmp_path):
        _three_runs(history, tmp_path)
        plugin.main_widget.LEProjectName.setText('Alpha')
        plugin.show_current_project_runs()
        cb = plugin.main_widget.cbRunProject
        cb.setCurrentIndex(cb.findData('Kattegat'))
        assert _shown_names(plugin) == ['k1']
        plugin.refresh_previous_runs_table()
        assert _shown_names(plugin) == ['k1']
        cb.setCurrentIndex(0)   # All projects
        assert len(_shown_names(plugin)) == 3

    def test_main_run_of_another_project_is_kept(self, plugin, history, tmp_path):
        """Filtering must not mistake a hidden main run for a deleted one."""
        _three_runs(history, tmp_path)
        k1 = next(r.run_id for r in history.list_runs() if r.name == 'k1')
        plugin._set_main_run_id(k1)
        plugin.main_widget.LEProjectName.setText('Alpha')
        plugin.show_current_project_runs()
        assert plugin._get_main_run_id() == k1


class TestSetProject:
    def test_retag_selected_runs(self, plugin, history, tmp_path, monkeypatch):
        _three_runs(history, tmp_path)
        ids = [r.run_id for r in history.list_runs() if r.name in ('a2', 'k1')]
        asked = {}

        def fake_ask(n, labels, current, parent=None):
            asked.update(n=n, labels=labels)
            return 'Baltic', True

        monkeypatch.setattr(plugin, '_ask_project_name', fake_ask)
        plugin._set_project_for_runs(ids)
        assert asked == {'n': 2, 'labels': ['Alpha', 'Kattegat']}
        labels = {r.name: r.project_label() for r in history.list_runs()}
        assert labels == {'a1': 'Alpha', 'a2': 'Baltic', 'k1': 'Baltic'}

    def test_empty_name_goes_back_to_the_folder(self, plugin, history, tmp_path, monkeypatch):
        _three_runs(history, tmp_path)
        a1 = next(r.run_id for r in history.list_runs() if r.name == 'a1')
        monkeypatch.setattr(plugin, '_ask_project_name', lambda *a, **k: ('', True))
        plugin._set_project_for_runs([a1])
        assert history.get_run(a1).project_label() == 'x'

    def test_cancel_changes_nothing(self, plugin, history, tmp_path, monkeypatch):
        _three_runs(history, tmp_path)
        a1 = next(r.run_id for r in history.list_runs() if r.name == 'a1')
        monkeypatch.setattr(plugin, '_ask_project_name', lambda *a, **k: ('Other', False))
        plugin._set_project_for_runs([a1])
        assert history.get_run(a1).project == 'Alpha'


class TestSaveRun:
    def test_run_records_project_and_version_and_is_shown(self, plugin, history, tmp_path):
        from omrat_utils.gather_data import plugin_version
        out = tmp_path / 'Kattegat'
        out.mkdir()
        history.save_run('other', output_dir=str(tmp_path / 'y'), project='Other')
        plugin.main_widget.LEModelName.setText('m')
        plugin.main_widget.LEReportPath.setText(str(out))
        # The name in the input snapshot (taken at kick-off) wins.
        plugin.main_widget.LEProjectName.setText('Changed during the run')
        plugin._run_input_snapshot = {'project_name': 'ESCOW'}
        plugin._auto_save_run(SimpleNamespace())
        run = next(r for r in history.list_runs() if r.name == 'm')
        assert run.project == 'ESCOW'
        assert run.omrat_version == plugin_version() != ''
        plugin.refresh_previous_runs_table()
        assert _shown_names(plugin) == ['m']

    def test_untagged_run_is_shown_under_its_folder(self, plugin, history, tmp_path):
        out = tmp_path / 'Kattegat'
        out.mkdir()
        plugin.main_widget.LEModelName.setText('m')
        plugin.main_widget.LEReportPath.setText(str(out))
        plugin._run_input_snapshot = {}
        plugin._auto_save_run(SimpleNamespace())
        assert history.list_runs()[0].project is None
        plugin.refresh_previous_runs_table()
        assert plugin.main_widget.cbRunProject.currentText() == 'Kattegat (1)'


class TestProjectFile:
    def test_project_name_survives_save_and_load(self, plugin, tmp_path):
        from omrat_utils.storage import Storage
        Storage(plugin).load_from_path(str(EXAMPLE))
        assert plugin.main_widget.LEProjectName.text() == ''
        plugin.main_widget.LEProjectName.setText('  Kattegat study ')
        target = tmp_path / 'named.omrat'
        assert Storage(plugin).store_all(str(target)) == str(target)
        assert json.loads(target.read_text())['project_name'] == 'Kattegat study'
        plugin.clear_model()
        assert plugin.main_widget.LEProjectName.text() == ''
        Storage(plugin).load_from_path(str(target))
        assert plugin.main_widget.LEProjectName.text() == 'Kattegat study'

    def test_empty_name_is_not_written(self, plugin, tmp_path):
        from omrat_utils.storage import Storage
        Storage(plugin).load_from_path(str(EXAMPLE))
        target = tmp_path / 'plain.omrat'
        Storage(plugin).store_all(str(target))
        assert 'project_name' not in json.loads(target.read_text())

    def test_file_without_a_name_keeps_the_field(self, plugin):
        """Merging a file that names no project must not wipe the name."""
        from omrat_utils.storage import Storage
        plugin.main_widget.LEProjectName.setText('Keep me')
        Storage(plugin).load_from_path(str(EXAMPLE))
        assert plugin.main_widget.LEProjectName.text() == 'Keep me'


# ---------------------------------------------------------------------------
# File -> Manage previous runs...
# ---------------------------------------------------------------------------

def _runs_with_files(history, tmp_path):
    """a (file on disk, 0.16.9), b (file missing, 0.16.10), c (other project)."""
    folder = tmp_path / 'Kattegat'
    folder.mkdir()
    (folder / 'a.gpkg').write_bytes(b'gpkg')
    history.save_run('a', output_dir=str(folder), output_filename='a.gpkg', omrat_version='0.16.9')
    history.save_run('b', output_dir=str(folder), output_filename='b.gpkg', omrat_version='0.16.10')
    history.save_run('c', output_dir=str(tmp_path / 'other'), output_filename='c.gpkg', project='Baltic')


def _dialog_names(dlg) -> list[str]:
    return [dlg.table.item(r, 0).text() for r in range(dlg.table.rowCount())]


def _select(dlg, names) -> None:
    from qgis.PyQt.QtCore import QItemSelectionModel
    dlg.table.clearSelection()
    flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    for row in range(dlg.table.rowCount()):
        if dlg.table.item(row, 0).text() in names:
            dlg.table.selectionModel().select(dlg.table.model().index(row, 0), flags)


class TestManageRunsDialog:
    def test_file_menu_entry_opens_the_dialog(self, plugin, history, monkeypatch):
        """It used to only switch tab -- invisible from Run Analysis."""
        from qgis.PyQt.QtWidgets import QMenuBar
        import omrat_utils.previous_runs_dialog as prd
        opened = []
        monkeypatch.setattr(prd.PreviousRunsDialog, 'exec', lambda self: opened.append(self) or 0)
        bar = plugin.main_widget.findChildren(QMenuBar)[0]
        file_menu = next(a.menu() for a in bar.actions() if a.text() == 'File')
        action = next(a for a in file_menu.actions() if a.text() == 'Manage previous runs...')
        action.trigger()
        assert len(opened) == 1

    def test_lists_every_project(self, plugin, history, tmp_path):
        from omrat_utils.previous_runs_dialog import COL_FILE, COL_PROJECT, PreviousRunsDialog
        _runs_with_files(history, tmp_path)
        dlg = PreviousRunsDialog(plugin)
        assert sorted(_dialog_names(dlg)) == ['a', 'b', 'c']
        projects = {dlg.table.item(r, 0).text(): dlg.table.item(r, COL_PROJECT).text()
                    for r in range(dlg.table.rowCount())}
        assert projects == {'a': 'Kattegat', 'b': 'Kattegat', 'c': 'Baltic'}
        files = {dlg.table.item(r, 0).text(): dlg.table.item(r, COL_FILE).text()
                 for r in range(dlg.table.rowCount())}
        assert files == {'a': 'ok', 'b': 'missing', 'c': 'missing'}
        assert '3 runs in 2 projects, 2 with a missing result file' in dlg.info.text()
        assert str(history.db_path) in dlg.info.text()
        dlg.cb_project.setCurrentIndex(dlg.cb_project.findData('Kattegat'))
        assert sorted(_dialog_names(dlg)) == ['a', 'b']

    def test_select_missing(self, plugin, history, tmp_path):
        from omrat_utils.previous_runs_dialog import PreviousRunsDialog
        _runs_with_files(history, tmp_path)
        dlg = PreviousRunsDialog(plugin)
        dlg.select_missing()
        names = {r.name for r in dlg._selected_runs()}
        assert names == {'b', 'c'}
        assert dlg.pb_delete.isEnabled() and not dlg.pb_open_folder.isEnabled()

    def test_versions_sort_numerically(self, plugin, history, tmp_path):
        from qgis.PyQt.QtCore import Qt
        from omrat_utils.previous_runs_dialog import COL_VERSION, PreviousRunsDialog
        _runs_with_files(history, tmp_path)
        dlg = PreviousRunsDialog(plugin)
        dlg.table.sortItems(COL_VERSION, Qt.SortOrder.AscendingOrder)
        versions = [dlg.table.item(r, COL_VERSION).text() for r in range(dlg.table.rowCount())]
        assert versions.index('0.16.9') < versions.index('0.16.10')

    def test_set_project_from_the_dialog(self, plugin, history, tmp_path, monkeypatch):
        from omrat_utils.previous_runs_dialog import PreviousRunsDialog
        _runs_with_files(history, tmp_path)
        dlg = PreviousRunsDialog(plugin)
        parents = []

        def fake_ask(n, labels, current, parent=None):
            parents.append(parent)
            return 'Baltic', True

        monkeypatch.setattr(plugin, '_ask_project_name', fake_ask)
        _select(dlg, {'a', 'b'})
        dlg.set_project()
        assert parents == [dlg]
        assert {r.project_label() for r in history.list_runs()} == {'Baltic'}
        assert dlg.cb_project.itemText(1) == 'Baltic (3)'

    def test_delete_from_the_dialog(self, plugin, history, tmp_path, monkeypatch):
        import omrat_utils.run_history_mixin as rhm
        from qgis.PyQt.QtWidgets import QMessageBox
        from omrat_utils.previous_runs_dialog import PreviousRunsDialog
        _runs_with_files(history, tmp_path)
        dlg = PreviousRunsDialog(plugin)
        asked = []

        class _Box:
            StandardButton = QMessageBox.StandardButton

            @staticmethod
            def question(parent, *a, **k):
                asked.append(parent)
                return QMessageBox.StandardButton.Yes

        monkeypatch.setattr(rhm, 'QMessageBox', _Box)
        _select(dlg, {'a'})
        dlg.delete(remove_files=True)
        assert asked == [dlg]
        assert sorted(r.name for r in history.list_runs()) == ['b', 'c']
        assert not (tmp_path / 'Kattegat' / 'a.gpkg').exists()
        assert sorted(_dialog_names(dlg)) == ['b', 'c']

    def test_open_folder(self, plugin, history, tmp_path, monkeypatch):
        import omrat_utils.previous_runs_dialog as prd
        _runs_with_files(history, tmp_path)
        dlg = prd.PreviousRunsDialog(plugin)
        opened = []
        monkeypatch.setattr(prd.QDesktopServices, 'openUrl', lambda url: opened.append(url.toLocalFile()))
        _select(dlg, {'a'})
        assert dlg.pb_open_folder.isEnabled()
        dlg.open_folder()
        assert [Path(p) for p in opened] == [tmp_path / 'Kattegat']
