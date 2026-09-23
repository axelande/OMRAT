"""Sensitivity-analysis dialog (Run Analysis tab -> **Sensitivity analysis...**).

The dialog lets the user

* pick the parameters to test (grouped tree, all ticked by default),
* choose the one-at-a-time perturbation (``+/- N %``),
* decide whether the last finished model run serves as the baseline or
  a fresh baseline is computed first,
* start the analysis as a cancellable ``QgsTask`` and watch the ranking
  fill in parameter by parameter,
* switch the output the ranking is based on (all accidents, all
  grounding, ..., a single accident type) without re-running anything,
* save the ranking as Markdown + JSON (written automatically to the
  Run Analysis output folder when one is set) and show a tornado plot.

Everything numerical lives in :mod:`compute.sensitivity`; this module is
Qt only.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from qgis.core import Qgis, QgsApplication, QgsMessageLog
from qgis.PyQt.QtCore import QSettings, Qt
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar,
    QPushButton, QRadioButton, QScrollArea, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from compute.sensitivity import (
    OUTPUT_KEYS, OUTPUT_LABELS, PHASE_LABELS, PHASES, BaselineReports,
    ParameterResult, ParameterSpec, SensitivityResult, baseline_value,
    build_run_plan, dump_json, phase_counts, rank_results, result_to_markdown,
)
from omrat_utils.run_history import slug
from omrat_utils.sensitivity_task import SensitivityTask

DELTA_SETTING = 'omrat/sensitivity_delta_pct'
OUTPUT_SETTING = 'omrat/sensitivity_output'

_RESULT_HEADERS = (
    '#', 'Parameter', 'Group', 'Current value', 'At -d', 'At +d',
    'Swing', 'Swing %', 'Elasticity', 'Method',
)


def _fmt_prob(value: float | None) -> str:
    if value is None:
        return '-'
    if value == 0:
        return '0'
    return f'{value:.3e}'


def _fmt_value(value: float | None, unit: str) -> str:
    if value is None:
        return '-'
    text = f'{value:.4g}'
    return f'{text} {unit}' if unit else text


class SensitivityDialog(QDialog):
    """Non-modal dialog owning one :class:`SensitivityTask` at a time."""

    def __init__(
        self,
        parent: QWidget | None,
        data: dict[str, Any],
        specs: list[ParameterSpec],
        live_baseline: BaselineReports | None = None,
        out_dir: Path | None = None,
        run_name: str = '',
        task_starter: Callable[[SensitivityTask], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle('OMRAT sensitivity analysis')
        self.setWindowFlag(Qt.WindowType.WindowMinMaxButtonsHint, True)
        self._data = data
        self._specs = list(specs)
        self._live_baseline = live_baseline
        self._out_dir = out_dir
        self._run_name = run_name or 'model'
        self._task_starter = task_starter or self._default_task_starter
        self._task: SensitivityTask | None = None
        self._result: SensitivityResult | None = None
        self._live_result: SensitivityResult | None = None
        self._saved_paths: list[Path] = []
        self._build_ui()
        self._populate_tree()
        self._update_plan_label()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        settings = QSettings()
        root = QVBoxLayout(self)

        intro = QLabel(
            'Each selected parameter is changed one at a time by -d and +d around its current '
            'value and the accident totals are recomputed.  Causation factors and traffic volume '
            'are evaluated instantly from the baseline results; the other parameters re-run only '
            'the model phases they can influence, which can take a long time on large projects.')
        intro.setWordWrap(True)
        root.addWidget(intro)

        # --- settings row ------------------------------------------------
        opts = QGroupBox('Settings')
        grid = QGridLayout(opts)
        grid.addWidget(QLabel('Perturbation d [%]:'), 0, 0)
        self.sbDelta = QSpinBox()
        self.sbDelta.setRange(1, 90)
        self.sbDelta.setValue(int(settings.value(DELTA_SETTING, 20, type=int) or 20))
        self.sbDelta.setToolTip('Every parameter is tested at (100 - d) % and (100 + d) % of its current value.')
        grid.addWidget(self.sbDelta, 0, 1)

        grid.addWidget(QLabel('Rank by:'), 0, 2)
        self.cbOutput = QComboBox()
        for key in OUTPUT_KEYS:
            self.cbOutput.addItem(OUTPUT_LABELS[key], key)
        stored = str(settings.value(OUTPUT_SETTING, 'total') or 'total')
        idx = self.cbOutput.findData(stored)
        self.cbOutput.setCurrentIndex(idx if idx >= 0 else 0)
        self.cbOutput.currentIndexChanged.connect(self._refresh_results_table)
        grid.addWidget(self.cbOutput, 0, 3)

        grid.addWidget(QLabel('Baseline:'), 1, 0)
        self.rbReuse = QRadioButton('Reuse the results of the last model run')
        self.rbReuse.setToolTip(
            'Fastest.  Assumes the inputs have not changed since that run finished; '
            'if they have, choose "Recompute".')
        self.rbRecompute = QRadioButton('Recompute the baseline first (all phases)')
        has_live = self._live_baseline is not None and any(self._live_baseline.totals.values())
        self.rbReuse.setEnabled(has_live)
        if has_live:
            self.rbReuse.setChecked(True)
        else:
            self.rbRecompute.setChecked(True)
            self.rbReuse.setToolTip('No finished model run in this session -- run the model first, or recompute.')
        grid.addWidget(self.rbReuse, 1, 1, 1, 3)
        grid.addWidget(self.rbRecompute, 2, 1, 1, 3)
        self.rbReuse.toggled.connect(self._update_plan_label)
        self.sbDelta.valueChanged.connect(self._update_plan_label)
        root.addWidget(opts)

        splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(splitter, 1)

        # --- parameter tree -------------------------------------------
        tree_box = QGroupBox('Parameters to test')
        tree_layout = QVBoxLayout(tree_box)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(['Parameter', 'Current value', 'Model phases re-run'])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tree.itemChanged.connect(self._on_tree_item_changed)
        tree_layout.addWidget(self.tree)
        btn_row = QHBoxLayout()
        self.pbAll = QPushButton('Select all')
        self.pbInstant = QPushButton('Instant only')
        self.pbInstant.setToolTip('Only the parameters that need no model re-run.')
        self.pbNone = QPushButton('Select none')
        self.pbAll.clicked.connect(lambda: self._set_all(True))
        self.pbInstant.clicked.connect(self._select_instant)
        self.pbNone.clicked.connect(lambda: self._set_all(False))
        btn_row.addWidget(self.pbAll)
        btn_row.addWidget(self.pbInstant)
        btn_row.addWidget(self.pbNone)
        btn_row.addStretch(1)
        self.lblPlan = QLabel('')
        self.lblPlan.setWordWrap(True)
        btn_row.addWidget(self.lblPlan, 2)
        tree_layout.addLayout(btn_row)
        splitter.addWidget(tree_box)

        # --- results --------------------------------------------------
        res_box = QGroupBox('Ranking')
        res_layout = QVBoxLayout(res_box)
        self.table = QTableWidget(0, len(_RESULT_HEADERS))
        self.table.setHorizontalHeaderLabels(list(_RESULT_HEADERS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        res_layout.addWidget(self.table)
        self.lblBaseline = QLabel('')
        self.lblBaseline.setWordWrap(True)
        res_layout.addWidget(self.lblBaseline)
        splitter.addWidget(res_box)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        # --- progress + buttons --------------------------------------
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)
        self.lblStatus = QLabel('')
        self.lblStatus.setWordWrap(True)
        root.addWidget(self.lblStatus)

        buttons = QHBoxLayout()
        self.pbRun = QPushButton('Run analysis')
        self.pbRun.clicked.connect(self.start)
        self.pbCancel = QPushButton('Cancel run')
        self.pbCancel.setEnabled(False)
        self.pbCancel.clicked.connect(self.cancel)
        self.pbPlot = QPushButton('Tornado plot...')
        self.pbPlot.setEnabled(False)
        self.pbPlot.clicked.connect(self.show_tornado)
        self.pbSave = QPushButton('Save report...')
        self.pbSave.setEnabled(False)
        self.pbSave.clicked.connect(self.save_report_as)
        self.pbClose = QPushButton('Close')
        self.pbClose.clicked.connect(self.close)
        for b in (self.pbRun, self.pbCancel, self.pbPlot, self.pbSave):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.pbClose)
        root.addLayout(buttons)
        self.resize(980, 760)

    def _populate_tree(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        for spec in self._specs:
            grp = groups.get(spec.group)
            if grp is None:
                grp = QTreeWidgetItem([spec.group, '', ''])
                grp.setFlags(grp.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
                grp.setCheckState(0, Qt.CheckState.Checked)
                self.tree.addTopLevelItem(grp)
                groups[spec.group] = grp
            phases = 'none (instant)' if spec.analytic else ', '.join(PHASE_LABELS[p] for p in spec.phases)
            item = QTreeWidgetItem([
                spec.label, _fmt_value(baseline_value(self._data, spec), spec.unit), phases,
            ])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked)
            item.setData(0, Qt.ItemDataRole.UserRole, spec.key)
            grp.addChild(item)
        self.tree.expandAll()
        for col in range(3):
            self.tree.resizeColumnToContents(col)
        self.tree.blockSignals(False)

    # ------------------------------------------------------------------
    # Parameter selection
    # ------------------------------------------------------------------
    def _iter_param_items(self):
        for g in range(self.tree.topLevelItemCount()):
            grp = self.tree.topLevelItem(g)
            for c in range(grp.childCount()):
                yield grp.child(c)

    def selected_specs(self) -> list[ParameterSpec]:
        by_key = {s.key: s for s in self._specs}
        out: list[ParameterSpec] = []
        for item in self._iter_param_items():
            if item.checkState(0) == Qt.CheckState.Checked:
                spec = by_key.get(str(item.data(0, Qt.ItemDataRole.UserRole)))
                if spec is not None:
                    out.append(spec)
        return out

    def set_selected_keys(self, keys: set[str]) -> None:
        self.tree.blockSignals(True)
        for item in self._iter_param_items():
            key = str(item.data(0, Qt.ItemDataRole.UserRole))
            item.setCheckState(0, Qt.CheckState.Checked if key in keys else Qt.CheckState.Unchecked)
        self.tree.blockSignals(False)
        self._update_plan_label()

    def _set_all(self, checked: bool) -> None:
        self.set_selected_keys({s.key for s in self._specs} if checked else set())

    def _select_instant(self) -> None:
        self.set_selected_keys({s.key for s in self._specs if s.analytic})

    def _on_tree_item_changed(self, *_args) -> None:
        self._update_plan_label()

    def delta(self) -> float:
        return self.sbDelta.value() / 100.0

    def _update_plan_label(self, *_args) -> None:
        specs = self.selected_specs()
        plan = build_run_plan(specs, self.delta())
        n_computed = sum(1 for s in specs if not s.analytic)
        counts = phase_counts(plan)
        parts = [f'{PHASE_LABELS[p]} x{n}' for p, n in counts.items() if n]
        extra = 0 if (self.rbReuse.isChecked() and self.rbReuse.isEnabled()) else 1
        text = (f'{len(specs)} parameters selected, {len(specs) - n_computed} instant, '
                f'{n_computed} need model re-runs: {len(plan)} partial runs'
                + (f' ({", ".join(parts)})' if parts else '')
                + (' plus one full baseline run.' if extra else '.'))
        self.lblPlan.setText(text)
        self.pbRun.setEnabled(bool(specs) and self._task is None)

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    @staticmethod
    def _default_task_starter(task: SensitivityTask) -> None:
        QgsApplication.taskManager().addTask(task)

    def start(self) -> None:
        if self._task is not None:
            return
        specs = self.selected_specs()
        if not specs:
            QMessageBox.information(self, 'Sensitivity analysis', 'Select at least one parameter.')
            return
        settings = QSettings()
        settings.setValue(DELTA_SETTING, self.sbDelta.value())
        settings.setValue(OUTPUT_SETTING, self.cbOutput.currentData())
        baseline = self._live_baseline if (self.rbReuse.isEnabled() and self.rbReuse.isChecked()) else None
        self._result = None
        self._live_result = SensitivityResult(
            delta=self.delta(), baseline=baseline or BaselineReports(),
            started=time.strftime('%Y-%m-%d %H:%M:%S'),
            project_name=str(self._data.get('project_name') or self._run_name),
        )
        self._saved_paths = []
        self.table.setRowCount(0)
        self.lblBaseline.setText('')
        self.progress.setValue(0)
        self.lblStatus.setText('Starting...')
        task = SensitivityTask(self._data, specs, self.delta(), baseline=baseline)
        task.progress_updated.connect(self._on_progress)
        task.phase_progress.connect(self._on_phase_progress)
        task.parameter_done.connect(self._on_parameter_done)
        task.analysis_finished.connect(self._on_finished)
        task.analysis_failed.connect(self._on_failed)
        self._task = task
        self._set_running(True)
        self._task_starter(task)

    def cancel(self) -> None:
        if self._task is not None:
            self.lblStatus.setText('Cancelling after the current run...')
            self._task.cancel()

    def _set_running(self, running: bool) -> None:
        self.pbRun.setEnabled(not running and bool(self.selected_specs()))
        self.pbCancel.setEnabled(running)
        for w in (self.tree, self.sbDelta, self.rbReuse, self.rbRecompute, self.pbAll, self.pbInstant, self.pbNone):
            w.setEnabled(not running)
        if not running and self._live_baseline is None:
            self.rbReuse.setEnabled(False)
        has_result = self._result is not None and bool(self._result.results)
        self.pbSave.setEnabled(has_result and not running)
        self.pbPlot.setEnabled(has_result and not running)

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self._runs_done, self._runs_total = done, total
        if total > 0:
            self.progress.setValue(int(round(100 * done / total)))
        self.lblStatus.setText(f'Run {done}/{total}: {message}')

    def _on_phase_progress(self, pct: float, message: str) -> None:
        """Movement inside a partial run (the bar would otherwise sit still for hours)."""
        self.progress.setValue(int(pct))
        done = getattr(self, '_runs_done', 0)
        total = getattr(self, '_runs_total', 0)
        self.lblStatus.setText(f'Run {done + 1}/{total} ({pct:.1f} % overall): {message}')

    def _on_parameter_done(self, pr: ParameterResult) -> None:
        if self._live_result is None:
            return
        self._live_result.results.append(pr)
        if self._task is not None and self._task.result is not None:
            # Baseline may have been computed by the task; mirror it so
            # the live ranking has the right denominator.
            self._live_result.baseline = self._task.result.baseline
        self._result = self._live_result
        self._refresh_results_table()

    def _on_finished(self, result: SensitivityResult) -> None:
        self._result = result
        self._task = None
        self._refresh_results_table()
        if result.cancelled:
            self.lblStatus.setText(
                f'Cancelled: {len(result.results)} parameters finished.')
        else:
            self.progress.setValue(100)
            self.lblStatus.setText(f'Done: {len(result.results)} parameters evaluated.')
        self._set_running(False)
        if result.results:
            self._auto_save()

    def _on_failed(self, message: str) -> None:
        self._task = None
        self._set_running(False)
        self.lblStatus.setText('Failed.')
        QgsMessageLog.logMessage(f'Sensitivity analysis failed: {message}', 'OMRAT', Qgis.MessageLevel.Critical)
        first_line = message.splitlines()[0] if message else 'Unknown error'
        QMessageBox.critical(self, 'Sensitivity analysis failed', first_line)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def current_output_key(self) -> str:
        return str(self.cbOutput.currentData() or 'total')

    def _refresh_results_table(self, *_args) -> None:
        result = self._result
        self.table.setRowCount(0)
        if result is None or not result.results:
            return
        key = self.current_output_key()
        QSettings().setValue(OUTPUT_SETTING, key)
        rows = rank_results(result, key)
        d_pct = f'{result.delta * 100:.0f}'
        self.table.setHorizontalHeaderLabels([
            '#', 'Parameter', 'Group', 'Current value', f'At -{d_pct} %', f'At +{d_pct} %',
            'Swing', 'Swing %', 'Elasticity', 'Method',
        ])
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            method = 'analytical' if not r.computed else 're-run: ' + ', '.join(PHASE_LABELS[p] for p in r.phases)
            values = [
                str(i + 1), r.label, r.group, _fmt_value(r.base_value, r.unit),
                _fmt_prob(r.output_minus), _fmt_prob(r.output_plus),
                ('-' if r.swing == 0 else f'{r.swing:+.3e}'),
                ('-' if r.swing_pct is None else f'{r.swing_pct:+.1f} %'),
                ('-' if r.elasticity is None else f'{r.elasticity:+.2f}'),
                method,
            ]
            for c, text in enumerate(values):
                item = QTableWidgetItem(text)
                if c not in (1, 2, 9):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, c, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        base = rows[0].base_output if rows else 0.0
        self.lblBaseline.setText(
            f'Baseline {OUTPUT_LABELS[key]}: {_fmt_prob(base)} events/year.  '
            'Swing = value at +d minus value at -d.  Elasticity = relative change of the output per '
            'relative change of the input (1 = proportional, 2 = quadratic, 0 = no effect).')

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def write_reports(self, directory: Path, stem: str) -> list[Path]:
        """Write ``<stem>.md`` and ``<stem>.json``; return the paths."""
        if self._result is None:
            return []
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        md_path = directory / f'{stem}.md'
        json_path = directory / f'{stem}.json'
        md_path.write_text(result_to_markdown(self._result, self.current_output_key()), encoding='utf-8')
        dump_json(self._result, json_path)
        return [md_path, json_path]

    def _auto_save(self) -> None:
        if self._out_dir is None:
            self.lblStatus.setText(self.lblStatus.text() + '  Use "Save report..." to keep the ranking.')
            return
        try:
            stem = f'{slug(self._run_name)}_sensitivity_{time.strftime("%Y%m%d_%H%M%S")}'
            self._saved_paths = self.write_reports(Path(self._out_dir), stem)
            self.lblStatus.setText(self.lblStatus.text() + f'  Report written to {self._saved_paths[0]}')
        except Exception as exc:  # noqa: BLE001
            QgsMessageLog.logMessage(f'Could not write sensitivity report: {exc}', 'OMRAT', Qgis.MessageLevel.Warning)

    def save_report_as(self) -> None:
        if self._result is None:
            return
        start = str(self._out_dir or Path.home())
        default = Path(start) / f'{slug(self._run_name)}_sensitivity.md'
        path, _filter = QFileDialog.getSaveFileName(
            self, 'Save sensitivity report', str(default), 'Markdown report (*.md)')
        if not path:
            return
        p = Path(path)
        try:
            paths = self.write_reports(p.parent, p.stem)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, 'Save failed', str(exc))
            return
        self.lblStatus.setText('Saved ' + ' and '.join(str(x) for x in paths))

    # ------------------------------------------------------------------
    # Tornado plot
    # ------------------------------------------------------------------
    def tornado_png(self, path: Path, max_rows: int = 15) -> Path | None:
        """Render the current ranking as a tornado plot; ``None`` without matplotlib."""
        if self._result is None or not self._result.results:
            return None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception:  # noqa: BLE001 - optional dependency
            return None
        key = self.current_output_key()
        rows = [r for r in rank_results(self._result, key) if r.base_output > 0][:max_rows]
        if not rows:
            return None
        rows = rows[::-1]  # largest swing on top
        base = rows[0].base_output
        labels = [r.label for r in rows]
        lo = [(r.output_minus - base) / base * 100.0 for r in rows]
        hi = [(r.output_plus - base) / base * 100.0 for r in rows]
        fig, ax = plt.subplots(figsize=(10, 0.45 * len(rows) + 1.8))
        ax.barh(labels, lo, color='#c0504d', label=f'-{self._result.delta * 100:.0f} %')
        ax.barh(labels, hi, color='#4f81bd', label=f'+{self._result.delta * 100:.0f} %')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel(f'Change in {OUTPUT_LABELS[key]} relative to baseline [%]')
        ax.set_title('OMRAT sensitivity: one-at-a-time perturbation')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    def show_tornado(self) -> None:
        tmp = Path(tempfile.gettempdir()) / f'omrat_sensitivity_{int(time.time())}.png'
        out = self.tornado_png(tmp)
        if out is None:
            QMessageBox.information(
                self, 'Tornado plot',
                'No plot available (matplotlib missing or no results).  The ranking table above '
                'holds the same information.')
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f'Tornado plot -- {OUTPUT_LABELS[self.current_output_key()]}')
        layout = QVBoxLayout(dlg)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        label = QLabel()
        label.setPixmap(QPixmap(str(out)))
        scroll.setWidget(label)
        layout.addWidget(scroll)
        row = QHBoxLayout()
        save = QPushButton('Save PNG...')

        def _save() -> None:
            start = str(self._out_dir or Path.home())
            target, _f = QFileDialog.getSaveFileName(
                dlg, 'Save tornado plot', str(Path(start) / f'{slug(self._run_name)}_tornado.png'),
                'PNG image (*.png)')
            if target:
                self.tornado_png(Path(target))

        save.clicked.connect(_save)
        row.addWidget(save)
        row.addStretch(1)
        close = QPushButton('Close')
        close.clicked.connect(dlg.close)
        row.addWidget(close)
        layout.addLayout(row)
        dlg.resize(1000, 700)
        dlg.exec()

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._task is not None:
            answer = QMessageBox.question(
                self, 'Sensitivity analysis running',
                'The analysis is still running.  Cancel it and close?')
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.cancel()
        super().closeEvent(event)


__all__ = ['SensitivityDialog', 'DELTA_SETTING', 'OUTPUT_SETTING', 'PHASES']
