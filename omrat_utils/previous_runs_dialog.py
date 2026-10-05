"""File -> **Manage previous runs...**: the whole run history in one dialog.

The dock's Previous runs table shows one project at a time and is for
picking runs to compare.  This dialog is for housekeeping across
projects: re-tag runs (**Set project...**), delete them (optionally with
their GeoPackage), open a run's output folder and find runs whose result
file is gone.  Every change goes through the same ``RunHistoryMixin``
helpers as the dock's right-click menu.
"""
from __future__ import annotations

import re

from qgis.PyQt.QtCore import QItemSelectionModel, Qt, QUrl
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
    QHeaderView, QLabel, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)

from omrat_utils.run_history_mixin import ALL_PROJECTS, _format_duration

COLUMNS = ('Name', 'Project', 'Date', 'Duration', 'Version', 'Output folder', 'Result file')
COL_NAME, COL_PROJECT, COL_DATE, COL_DURATION, COL_VERSION, COL_FOLDER, COL_FILE = range(len(COLUMNS))
MISSING = 'missing'
_FOLDER_MAX_WIDTH = 320
# A role of our own for sort keys (int: PyQt6 enums do not add).
_USER_ROLE = Qt.ItemDataRole.UserRole
_SORT_ROLE = int(getattr(_USER_ROLE, 'value', _USER_ROLE)) + 1


def version_sort_key(version: str | None) -> str:
    """'0.16.10' after '0.16.9': numeric parts zero-padded."""
    parts = re.split(r'([0-9]+)', version or '')
    return ''.join(p.zfill(6) if p.isdigit() else p for p in parts)


class _SortItem(QTableWidgetItem):
    """Sorts on ``_SORT_ROLE`` when set (durations, versions), else text."""

    def __lt__(self, other):
        mine, theirs = self.data(_SORT_ROLE), other.data(_SORT_ROLE)
        if mine is not None and theirs is not None:
            return mine < theirs
        return super().__lt__(other)


def _item(text: str, sort_key=None) -> QTableWidgetItem:
    item = _SortItem(text)
    if sort_key is not None:
        item.setData(_SORT_ROLE, sort_key)
    return item


class PreviousRunsDialog(QDialog):
    """Every run in the history, with project / version / file status."""

    def __init__(self, plugin, parent=None):
        super().__init__(parent)
        self.plugin = plugin
        self.setWindowTitle(self.tr('Manage previous runs'))
        self.resize(980, 540)
        self._runs = []

        self.info = QLabel(self)
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.cb_project = QComboBox(self)
        self.cb_project.setMinimumWidth(240)

        # Sorting is safe here: no cell widgets, and rows map to runs through
        # the run_id stored on the Name item, not through the row index.
        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        # Newest first until the user clicks a header.
        self.table.horizontalHeader().setSortIndicator(COL_DATE, Qt.SortOrder.DescendingOrder)

        self.pb_set_project = QPushButton(self.tr('Set project...'), self)
        self.pb_open_folder = QPushButton(self.tr('Open output folder'), self)
        self.pb_select_missing = QPushButton(self.tr('Select runs with missing files'), self)
        self.pb_delete = QPushButton(self.tr('Delete from history'), self)
        self.pb_delete_files = QPushButton(self.tr('Delete + remove .gpkg file'), self)
        self.pb_set_project.setToolTip(self.tr(
            'Move the selected runs to another project.  Clear the name to go '
            'back to the output folder name.'))
        self.pb_select_missing.setToolTip(self.tr(
            'Select the runs whose result GeoPackage is no longer on disk'))
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        close.rejected.connect(self.reject)

        top = QHBoxLayout()
        top.addWidget(self.info, 1)
        top.addWidget(QLabel(self.tr('Project:'), self))
        top.addWidget(self.cb_project)
        actions = QHBoxLayout()
        for btn in (self.pb_set_project, self.pb_open_folder, self.pb_select_missing):
            actions.addWidget(btn)
        actions.addStretch(1)
        actions.addWidget(self.pb_delete)
        actions.addWidget(self.pb_delete_files)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.table)
        layout.addLayout(actions)
        layout.addWidget(close)

        self.cb_project.currentIndexChanged.connect(lambda *_: self._fill_table())
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.pb_set_project.clicked.connect(self.set_project)
        self.pb_open_folder.clicked.connect(self.open_folder)
        self.pb_select_missing.clicked.connect(self.select_missing)
        self.pb_delete.clicked.connect(lambda: self.delete(remove_files=False))
        self.pb_delete_files.clicked.connect(lambda: self.delete(remove_files=True))
        self.reload()

    # ------------------------------------------------------------------
    def reload(self) -> None:
        """Re-read the history and redraw (keeps the project filter)."""
        from omrat_utils.run_history import RunHistory, project_counts
        history = RunHistory()
        self._runs = history.list_runs()
        try:
            history.backfill_versions(self._runs)
        except Exception:  # nosec B110 - a missing version must not hide the list
            pass
        counts = project_counts(self._runs)
        wanted = self.cb_project.currentData()
        was_blocked = self.cb_project.blockSignals(True)
        try:
            self.cb_project.clear()
            self.cb_project.addItem(self.tr('All projects ({n})').format(n=len(self._runs)), ALL_PROJECTS)
            for label, n in counts.items():
                self.cb_project.addItem(f'{label} ({n})', label)
            self.cb_project.setCurrentIndex(max(0, self.cb_project.findData(wanted)))
        finally:
            self.cb_project.blockSignals(was_blocked)
        n_missing = sum(1 for r in self._runs if self._is_missing(r))
        self.info.setText(self.tr(
            '{runs} runs in {projects} projects, {missing} with a missing result file.\n'
            'History file: {path}').format(
                runs=len(self._runs), projects=len(counts), missing=n_missing, path=history.db_path))
        self._fill_table()

    @staticmethod
    def _is_missing(run) -> bool:
        gpkg = run.gpkg_path()
        return gpkg is None or not gpkg.is_file()

    def _fill_table(self) -> None:
        project = self.cb_project.currentData()
        runs = [r for r in self._runs
                if project in (None, ALL_PROJECTS) or r.project_label() == project]
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            name = _item(run.name)
            name.setData(Qt.ItemDataRole.UserRole, run.run_id)
            self.table.setItem(row, COL_NAME, name)
            project_item = _item(run.project_label())
            if not (run.project or '').strip():
                project_item.setToolTip(self.tr('Not tagged: the name of the output folder'))
            self.table.setItem(row, COL_PROJECT, project_item)
            self.table.setItem(row, COL_DATE, _item(run.timestamp or ''))
            self.table.setItem(row, COL_DURATION, _item(
                _format_duration(run.duration_seconds), float(run.duration_seconds or 0.0)))
            self.table.setItem(row, COL_VERSION, _item(
                run.omrat_version or '—', version_sort_key(run.omrat_version)))
            folder = _item(run.output_dir or '')
            folder.setToolTip(run.output_dir or '')
            self.table.setItem(row, COL_FOLDER, folder)
            missing = self._is_missing(run)
            status = _item(self.tr(MISSING) if missing else self.tr('ok'))
            if missing:
                gpkg = run.gpkg_path()
                status.setToolTip(str(gpkg) if gpkg is not None else self.tr('No result file recorded'))
            self.table.setItem(row, COL_FILE, status)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        # Long paths would squeeze everything else; the tooltip has it all.
        self.table.setColumnWidth(COL_FOLDER, min(self.table.columnWidth(COL_FOLDER), _FOLDER_MAX_WIDTH))
        self._update_buttons()

    # ------------------------------------------------------------------
    def selected_run_ids(self) -> list[int]:
        out: list[int] = []
        for index in self.table.selectionModel().selectedRows():
            item = self.table.item(index.row(), COL_NAME)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) is not None:
                out.append(int(item.data(Qt.ItemDataRole.UserRole)))
        return out

    def _selected_runs(self):
        ids = set(self.selected_run_ids())
        return [r for r in self._runs if r.run_id in ids]

    def _update_buttons(self) -> None:
        selected = self._selected_runs()
        for btn in (self.pb_set_project, self.pb_delete, self.pb_delete_files):
            btn.setEnabled(bool(selected))
        self.pb_open_folder.setEnabled(len(selected) == 1 and bool(selected[0].output_dir))

    def select_missing(self) -> None:
        sm = self.table.selectionModel()
        self.table.clearSelection()
        flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
        for row in range(self.table.rowCount()):
            item = self.table.item(row, COL_FILE)
            if item is not None and item.text() == self.tr(MISSING):
                sm.select(self.table.model().index(row, 0), flags)

    def set_project(self) -> None:
        ids = self.selected_run_ids()
        if ids:
            self.plugin._set_project_for_runs(ids, parent=self)
            self.reload()

    def delete(self, remove_files: bool) -> None:
        ids = self.selected_run_ids()
        if ids:
            self.plugin._delete_runs(ids, delete_gpkg=remove_files, parent=self)
            self.reload()

    def open_folder(self) -> None:
        from pathlib import Path
        selected = self._selected_runs()
        if len(selected) != 1 or not selected[0].output_dir:
            return
        folder = Path(selected[0].output_dir)
        if not folder.is_dir():
            QMessageBox.information(
                self, self.tr('Folder missing'),
                self.tr('The output folder no longer exists:\n{path}').format(path=folder))
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
