"""Compare tab -> **More info (collisions)...**: why did the collisions change?

A non-modal dialog that runs :func:`omrat_utils.compare_explain.explain_collisions`
in a cancellable :class:`CompareExplainTask` and shows, per accident type,

1. *Why* -- the change split over the input groups that differ
   (diverging bar per group),
2. *Largest items* -- legs / leg pairs with their per-group split
   (optionally summed per split route), with **Show on map**,
3. *Findings* -- the plain-language findings for that type.

The explanation is cached on the plugin per snapshot pair (path + mtime),
so reopening the dialog for the same two runs is instant.  Everything
numerical lives in :mod:`omrat_utils.compare_explain`.
"""
from __future__ import annotations

import html
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

from qgis.core import (
    Qgis, QgsApplication, QgsCoordinateReferenceSystem, QgsGeometry,
    QgsMessageLog, QgsPointXY, QgsTask,
)
from qgis.PyQt.QtCore import QRectF, Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar, QPushButton,
    QSplitter, QStyledItemDelegate, QTableWidget, QTableWidgetItem, QTextBrowser,
    QVBoxLayout, QWidget,
)

from omrat_utils.compare_explain import (
    GROUP_LABELS, TOTAL, TYPE_LABELS, TYPES, CollisionExplanation,
    explain_collisions, findings_for, items_for, leg_endpoints, to_markdown,
)

_BAR_ROLE = Qt.ItemDataRole.UserRole + 1
_SORT_ROLE = Qt.ItemDataRole.UserRole + 2
_LEGS_ROLE = Qt.ItemDataRole.UserRole + 3
_INC = QColor(200, 70, 60)
_DEC = QColor(60, 110, 200)


def _fmt(value: float | None) -> str:
    if value is None:
        return '-'
    if value == 0:
        return '0'
    return f'{value:+.3e}'


class CompareExplainTask(QgsTask):
    """Runs the collision model on every needed mix of the two snapshots."""

    progress_updated = pyqtSignal(int, int, str)
    explanation_ready = pyqtSignal(object)
    explanation_failed = pyqtSignal(str)

    def __init__(self, snap_a: Mapping[str, Any], snap_b: Mapping[str, Any],
                 outputs_a: Mapping[str, Any] | None = None,
                 outputs_b: Mapping[str, Any] | None = None) -> None:
        super().__init__('OMRAT: why did the collisions change?', QgsTask.Flag.CanCancel)
        self.snap_a, self.snap_b = snap_a, snap_b
        self.outputs_a, self.outputs_b = outputs_a, outputs_b
        self.result: CollisionExplanation | None = None
        self.error_msg: str | None = None

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.setProgress(100.0 * done / max(1, total))
        self.progress_updated.emit(int(done), int(total), str(message))

    def run(self) -> bool:
        try:
            self.result = explain_collisions(
                self.snap_a, self.snap_b, self.outputs_a, self.outputs_b,
                on_progress=self._on_progress, is_cancelled=self.isCanceled)
            return self.result is not None
        except Exception as exc:  # noqa: BLE001 - reported through the signal
            self.error_msg = f'{exc}\n{traceback.format_exc()}'
            return False

    def finished(self, result: bool) -> None:
        if result and self.result is not None:
            self.explanation_ready.emit(self.result)
        elif self.isCanceled():
            self.explanation_failed.emit('Cancelled.')
        else:
            QgsMessageLog.logMessage(f'Collision explanation failed: {self.error_msg}', 'OMRAT',
                                     Qgis.MessageLevel.Critical)
            self.explanation_failed.emit(self.error_msg or 'Unknown error')


class _BarDelegate(QStyledItemDelegate):
    """Diverging bar: red to the right for an increase, blue to the left for a decrease."""

    def paint(self, painter, option, index) -> None:  # noqa: D401 - Qt override
        super().paint(painter, option, index)
        frac = index.data(_BAR_ROLE)
        if frac is None:
            return
        rect = option.rect.adjusted(4, 4, -4, -4)
        mid = rect.left() + rect.width() / 2.0
        width = rect.width() / 2.0 * max(-1.0, min(1.0, float(frac)))
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_INC if width >= 0 else _DEC)
        left = mid if width >= 0 else mid + width
        painter.drawRect(QRectF(left, rect.top(), abs(width), rect.height()))
        painter.setPen(option.palette.mid().color())
        painter.drawLine(int(mid), rect.top(), int(mid), rect.bottom())
        painter.restore()


class _SortItem(QTableWidgetItem):
    """Shows text, sorts on ``_SORT_ROLE``."""

    def __lt__(self, other: Any) -> bool:
        a, b = self.data(_SORT_ROLE), other.data(_SORT_ROLE)
        if a is not None and b is not None:
            return float(a) < float(b)
        return super().__lt__(other)


def _num_item(value: float | None, text: str | None = None) -> QTableWidgetItem:
    item = _SortItem(_fmt(value) if text is None else text)
    item.setData(_SORT_ROLE, None if value is None else abs(value))
    item.setTextAlignment(int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
    return item


class CompareExplainDialog(QDialog):
    """Non-modal *More info* dialog for one pair of snapshots."""

    def __init__(
        self,
        parent: QWidget | None,
        snap_a: Mapping[str, Any],
        snap_b: Mapping[str, Any],
        name_a: str,
        name_b: str,
        outputs_a: Mapping[str, Any] | None = None,
        outputs_b: Mapping[str, Any] | None = None,
        acc_type: str = TOTAL,
        explanation: CollisionExplanation | None = None,
        canvas: Any = None,
        on_ready: Callable[[CollisionExplanation], None] | None = None,
        task_starter: Callable[[CompareExplainTask], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle('More info - why did the collisions change?')
        self.setWindowFlag(Qt.WindowType.WindowMinMaxButtonsHint, True)
        self._snap_a, self._snap_b = snap_a, snap_b
        self._outputs_a, self._outputs_b = outputs_a, outputs_b
        self._name_a, self._name_b = name_a, name_b
        self._canvas = canvas
        self._bands: list[Any] = []
        self._on_ready = on_ready
        self._task_starter = task_starter or (lambda t: QgsApplication.taskManager().addTask(t))
        self._task: CompareExplainTask | None = None
        self.explanation: CollisionExplanation | None = None
        self._build_ui()
        idx = self.cbType.findData(acc_type)
        self.cbType.setCurrentIndex(idx if idx >= 0 else 0)
        if explanation is not None:
            self.set_explanation(explanation)
        else:
            self.start()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        runs = QLabel(f'Run A: <b>{html.escape(self._name_a)}</b> &nbsp;&nbsp;|&nbsp;&nbsp; '
                      f'Run B: <b>{html.escape(self._name_b)}</b>')
        root.addWidget(runs)

        top = QHBoxLayout()
        top.addWidget(QLabel('Accident type:'))
        self.cbType = QComboBox()
        for key in TYPES + (TOTAL,):
            self.cbType.addItem(TYPE_LABELS[key], key)
        self.cbType.currentIndexChanged.connect(self._refresh)
        top.addWidget(self.cbType)
        self.lblSummary = QLabel('')
        self.lblSummary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        top.addWidget(self.lblSummary, 1)
        root.addLayout(top)

        prog = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        prog.addWidget(self.progress, 1)
        self.lblStatus = QLabel('')
        prog.addWidget(self.lblStatus, 2)
        self.pbCancel = QPushButton('Cancel')
        self.pbCancel.clicked.connect(self.cancel)
        prog.addWidget(self.pbCancel)
        self.progressRow = QWidget()
        self.progressRow.setLayout(prog)
        root.addWidget(self.progressRow)

        split = QSplitter(Qt.Orientation.Vertical)

        why = QGroupBox('1. Why - share of the change per input group')
        why_l = QVBoxLayout(why)
        self.twGroups = QTableWidget(0, 4)
        self.twGroups.setHorizontalHeaderLabels(['Input group', 'Contribution', 'Share of change', ''])
        self.twGroups.setItemDelegateForColumn(3, _BarDelegate(self.twGroups))
        self.twGroups.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.twGroups.verticalHeader().setVisible(False)
        hdr = self.twGroups.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        why_l.addWidget(self.twGroups)
        self.lblIdentical = QLabel('')
        self.lblIdentical.setWordWrap(True)
        why_l.addWidget(self.lblIdentical)
        split.addWidget(why)

        items = QGroupBox('2. Largest items (legs for head-on / overtaking / bend, leg pairs for crossing / merging)')
        items_l = QVBoxLayout(items)
        row = QHBoxLayout()
        self.cbMergeSplit = QCheckBox('Merge split legs (LEG_1_2_a + LEG_1_2_b)')
        self.cbMergeSplit.setToolTip('Sum the sub-legs of a split route.  Moving a split point moves length from '
                                     'one sub-leg to the other, which looks like two large changes that cancel.')
        self.cbMergeSplit.toggled.connect(self._refresh_items)
        row.addWidget(self.cbMergeSplit)
        row.addStretch(1)
        self.pbShowMap = QPushButton('Show on map')
        self.pbShowMap.setToolTip('Highlight the legs of the selected rows (from run B, or run A when the leg '
                                  'exists only there) and zoom to them.')
        self.pbShowMap.clicked.connect(self.show_selected_on_map)
        self.pbShowMap.setEnabled(self._canvas is not None)
        row.addWidget(self.pbShowMap)
        items_l.addLayout(row)
        self.twItems = QTableWidget(0, 0)
        self.twItems.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.twItems.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.twItems.verticalHeader().setVisible(False)
        self.twItems.doubleClicked.connect(lambda *_: self.show_selected_on_map())
        items_l.addWidget(self.twItems)
        split.addWidget(items)

        found = QGroupBox('3. Findings')
        found_l = QVBoxLayout(found)
        self.tbFindings = QTextBrowser()
        self.tbFindings.setOpenExternalLinks(False)
        found_l.addWidget(self.tbFindings)
        split.addWidget(found)
        split.setSizes([200, 300, 300])
        root.addWidget(split, 1)

        buttons = QHBoxLayout()
        self.lblRuns = QLabel('')
        buttons.addWidget(self.lblRuns, 1)
        self.pbExport = QPushButton('Export to Markdown...')
        self.pbExport.clicked.connect(self.export_markdown)
        buttons.addWidget(self.pbExport)
        close = QPushButton('Close')
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        root.addLayout(buttons)
        self.resize(1100, 850)
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.progressRow.setVisible(running)
        self.pbCancel.setEnabled(running)
        has = self.explanation is not None
        self.pbExport.setEnabled(has and not running)
        self.cbMergeSplit.setEnabled(has)

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._task is not None:
            return
        self.explanation = None
        self.progress.setValue(0)
        self.lblStatus.setText('Starting...')
        self.lblSummary.setText('Calculating...')
        task = CompareExplainTask(self._snap_a, self._snap_b, self._outputs_a, self._outputs_b)
        task.progress_updated.connect(self._on_progress)
        task.explanation_ready.connect(self._on_ready_signal)
        task.explanation_failed.connect(self._on_failed)
        self._task = task
        self._set_running(True)
        self._task_starter(task)

    def cancel(self) -> None:
        if self._task is not None:
            self.lblStatus.setText('Cancelling after the current model run...')
            self._task.cancel()

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.progress.setValue(int(round(100.0 * done / max(1, total))))
        self.lblStatus.setText(message)

    def _on_ready_signal(self, expl: CollisionExplanation) -> None:
        self._task = None
        self.set_explanation(expl)
        if self._on_ready is not None:
            self._on_ready(expl)

    def _on_failed(self, message: str) -> None:
        self._task = None
        self._set_running(False)
        first = message.splitlines()[0] if message else 'Unknown error'
        self.lblSummary.setText(f'Not calculated: {first}')
        if first != 'Cancelled.':
            QMessageBox.critical(self, 'More info', f'The explanation failed:\n{first}')

    # ------------------------------------------------------------------
    # Showing a result
    # ------------------------------------------------------------------
    def set_explanation(self, expl: CollisionExplanation) -> None:
        self.explanation = expl
        self._set_running(False)
        how = 'exact Shapley values' if expl.exact else 'Shapley values averaged over fixed orders'
        self.lblRuns.setText(f'{expl.runs} collision-model runs in {expl.seconds:.0f} s ({how}).')
        self._refresh()

    def current_type(self) -> str:
        return str(self.cbType.currentData() or TOTAL)

    def _refresh(self, *_args) -> None:
        expl = self.explanation
        if expl is None:
            return
        t = self.current_type()
        a, b, d = expl.totals_a[t], expl.totals_b[t], expl.delta(t)
        pct = f' ({100.0 * d / a:+.1f} %)' if a else ''
        self.lblSummary.setText(f'Run A {a:.3e}  ->  Run B {b:.3e}   change {_fmt(d)}{pct} per year')
        self._fill_groups(expl, t)
        self._refresh_items()
        self._fill_findings(expl, t)

    def _fill_groups(self, expl: CollisionExplanation, t: str) -> None:
        rows = expl.contribution_rows(t)
        tw = self.twGroups
        tw.setRowCount(len(rows))
        delta = expl.delta(t)
        scale = max([abs(v) for _, v in rows] + [abs(delta)]) or 1.0
        for r, (g, v) in enumerate(rows):
            name = QTableWidgetItem(GROUP_LABELS[g])
            name.setToolTip('\n'.join(expl.differences.get(g) or []) or GROUP_LABELS[g])
            tw.setItem(r, 0, name)
            tw.setItem(r, 1, _num_item(v))
            share = '0 %' if v == 0 else (f'{100.0 * v / delta:.1f} %' if delta else '-')
            tw.setItem(r, 2, _num_item(v, share))
            bar = QTableWidgetItem('')
            bar.setData(_BAR_ROLE, v / scale)
            tw.setItem(r, 3, bar)
        tw.resizeColumnsToContents()
        tw.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        ident = [GROUP_LABELS[g] for g in expl.identical_groups()]
        self.lblIdentical.setText(('Identical in A and B (no contribution): ' + ', '.join(ident) + '.')
                                  if ident else '')

    def _refresh_items(self, *_args) -> None:
        expl = self.explanation
        if expl is None:
            return
        t = self.current_type()
        rows = items_for(expl, t, merge_split=self.cbMergeSplit.isChecked())
        cols = expl.columns()
        headers = ['Item'] + (['Type'] if t == TOTAL else []) + ['Run A', 'Run B', 'Change'] + \
            [GROUP_LABELS[g].split(' (')[0] for g in cols]
        tw = self.twItems
        tw.setSortingEnabled(False)
        tw.clear()
        tw.setColumnCount(len(headers))
        tw.setHorizontalHeaderLabels(headers)
        tw.setRowCount(len(rows))
        for r, item in enumerate(rows):
            c = 0
            label = QTableWidgetItem(item.label)
            tip = [item.label]
            if item.angle_a is not None or item.angle_b is not None:
                tip.append('Angle: A {} / B {} deg'.format(
                    '-' if item.angle_a is None else f'{item.angle_a:.1f}',
                    '-' if item.angle_b is None else f'{item.angle_b:.1f}'))
            label.setToolTip('\n'.join(tip))
            label.setData(_LEGS_ROLE, list(item.legs))
            tw.setItem(r, c, label)
            c += 1
            if t == TOTAL:
                tw.setItem(r, c, QTableWidgetItem(TYPE_LABELS[item.acc_type].split()[0]))
                c += 1
            tw.setItem(r, c, _num_item(item.a, f'{item.a:.3e}'))
            tw.setItem(r, c + 1, _num_item(item.b, f'{item.b:.3e}'))
            change = _num_item(item.delta)
            change.setForeground(_INC if item.delta > 0 else _DEC)
            tw.setItem(r, c + 2, change)
            for k, g in enumerate(cols):
                tw.setItem(r, c + 3 + k, _num_item(item.by_group.get(g, 0.0)))
        tw.setSortingEnabled(True)
        tw.resizeColumnsToContents()

    def _fill_findings(self, expl: CollisionExplanation, t: str) -> None:
        parts = []
        for f in findings_for(expl, t):
            colour = '#b35900' if f.level == 'warning' else '#202020'
            mark = '&#9888; ' if f.level == 'warning' else ''
            parts.append(f'<p style="margin-bottom:2px"><b style="color:{colour}">{mark}'
                         f'{html.escape(f.title)}</b></p>')
            if f.details:
                parts.append('<ul style="margin-top:0">' + ''.join(
                    f'<li>{html.escape(d)}</li>' for d in f.details) + '</ul>')
        self.tbFindings.setHtml(''.join(parts) or '<p>No findings.</p>')

    # ------------------------------------------------------------------
    # Map
    # ------------------------------------------------------------------
    def selected_legs(self) -> list[tuple[str, str]]:
        legs: list[tuple[str, str]] = []
        for idx in self.twItems.selectionModel().selectedRows(0) if self.twItems.selectionModel() else []:
            for leg in self.twItems.item(idx.row(), 0).data(_LEGS_ROLE) or []:
                if tuple(leg) not in legs:
                    legs.append(tuple(leg))
        return legs

    def clear_map(self) -> None:
        for band in self._bands:
            try:
                band.reset(Qgis.GeometryType.Line)
                if self._canvas is not None:
                    self._canvas.scene().removeItem(band)
            except Exception:  # nosec B110 B112
                pass
        self._bands = []

    def show_selected_on_map(self) -> int:
        """Highlight the selected rows' legs; returns the number of legs drawn."""
        self.clear_map()
        if self._canvas is None:
            return 0
        lines = leg_endpoints(self._snap_a, self._snap_b, self.selected_legs())
        if not lines:
            return 0
        from qgis.gui import QgsRubberBand
        crs = QgsCoordinateReferenceSystem('EPSG:4326')
        extent = None
        for p, q in lines:
            geom = QgsGeometry.fromPolylineXY([QgsPointXY(*p), QgsPointXY(*q)])
            band = QgsRubberBand(self._canvas, Qgis.GeometryType.Line)
            band.setColor(QColor(255, 0, 200, 200))
            band.setWidth(6)
            band.setToGeometry(geom, crs)
            self._bands.append(band)
            box = band.asGeometry().boundingBox()
            if extent is None:
                extent = box
            else:
                extent.combineExtentWith(box)
        if extent is not None:
            extent.scale(1.3)
            self._canvas.setExtent(extent)
            self._canvas.refresh()
        return len(lines)

    # ------------------------------------------------------------------
    # Export / close
    # ------------------------------------------------------------------
    def export_markdown(self, target: str | None = None) -> Path | None:
        if self.explanation is None:
            return None
        if not target:
            start = Path(self._name_b).stem + '_collision_explanation.md'
            target, _f = QFileDialog.getSaveFileName(self, 'Export explanation', start, 'Markdown (*.md)')
        if not target:
            return None
        path = Path(target)
        path.write_text(to_markdown(self.explanation, self._name_a, self._name_b), encoding='utf-8')
        return path

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self.clear_map()
        super().closeEvent(event)


__all__ = ['CompareExplainDialog', 'CompareExplainTask']
