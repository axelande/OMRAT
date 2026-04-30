"""Run-history workflow, factored out of ``omrat.OMRAT``.

Owns everything tied to "what happens after **Run Model** finishes":

* writing the per-run GeoPackage, ``.omrat`` snapshot, combined Markdown
  report, and JSON sidecars (collision / drifting reports);
* the **Run Analysis** tab's previous-runs table (load / refresh /
  context menu / row selection -> compare columns);
* the output-folder picker and the Run-Model-button gate.

Pure data work (sqlite metadata, slug generation, GeoPackage writing,
report markdown) lives in:

* :mod:`omrat_utils.run_history` -- master sqlite DB + ``RunMeta``,
* :mod:`omrat_utils.run_persistence` -- per-run GeoPackage I/O,
* :mod:`omrat_utils.full_report` -- combined Markdown report.

This mixin only owns the dock-widget plumbing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox

if TYPE_CHECKING:
    pass


def _fmt(v: float | None) -> str:
    """Format a probability total for the previous-runs / accident tables."""
    if v is None:
        return ''
    if v == 0:
        return '0'
    return f'{v:.3e}'


def _format_duration(seconds: float | None) -> str:
    """Human-readable elapsed-time string for the Duration column."""
    if seconds is None:
        return '—'
    if seconds < 60:
        return f'{seconds:.1f}s'
    if seconds < 3600:
        return f'{seconds / 60:.1f}m'
    return f'{seconds / 3600:.1f}h'


def _qt_enum(klass, name: str, *scoped_paths: str):
    """Look up ``name`` (or any of the scoped paths) on ``klass``.

    Crosses the PyQt5 / PyQt6 enum-naming gap: PyQt5 exposes
    ``Qt.UserRole`` flat, PyQt6 requires ``Qt.ItemDataRole.UserRole``.
    """
    if hasattr(klass, name):
        return getattr(klass, name)
    for path in scoped_paths:
        obj = klass
        try:
            for part in path.split('.'):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError(
        f"{klass.__name__} has no enum named {name!r} (tried {scoped_paths})"
    )


class RunHistoryMixin:
    """Auto-save flow + previous-runs table + output-folder gating."""

    # The accident-row order is shared with ``AccidentResultsMixin`` --
    # we declare it here too because the run-history's
    # "fill comparison columns" path indexes into the accident table.
    _ACCIDENT_TOTAL_KEYS: tuple[str, ...] = (
        'drift_allision', 'drift_grounding',
        'powered_allision', 'powered_grounding',
        'overtaking', 'head_on', 'crossing', 'bend',
    )

    # ------------------------------------------------------------------
    # Auto-save after a successful run
    # ------------------------------------------------------------------
    def _auto_save_run(self, calc_object: Any) -> None:
        """Write per-run GeoPackage + record metadata in the master DB."""
        from omrat_utils.run_history import (
            RunHistory, make_run_filename, totals_from_calc,
        )
        from omrat_utils.run_persistence import write_run_results

        out_dir = self._get_output_dir()
        if out_dir is None:
            raise RuntimeError(
                "No output folder configured on the Run Analysis tab. "
                "Set one and re-run."
            )

        name = self._auto_save_run_name()
        ts_struct = time.localtime()
        ts_text = time.strftime('%Y-%m-%d %H:%M:%S', ts_struct)
        filename = make_run_filename(name, ts_struct)
        gpkg_path = Path(out_dir) / filename

        QgsMessageLog.logMessage(
            f"Writing per-run results to {gpkg_path}", 'OMRAT', Qgis.Info,
        )
        try:
            written_layers = write_run_results(
                calc_object, gpkg_path,
                structures=getattr(calc_object, '_last_structures', None),
                depths=getattr(calc_object, '_last_depths', None),
                depths_original=getattr(
                    calc_object, '_last_depths_original', None,
                ),
                segment_data=self.segment_data,
            )
        except Exception as exc:
            import traceback
            QgsMessageLog.logMessage(
                f"GeoPackage write failed: {exc}\n{traceback.format_exc()}",
                'OMRAT', Qgis.Critical,
            )
            written_layers = []

        duration = self._compute_run_duration()

        history = RunHistory()
        run_id = history.save_run(
            name=name,
            timestamp=ts_text,
            duration_seconds=duration,
            totals=totals_from_calc(calc_object),
            output_dir=str(out_dir),
            output_filename=filename,
        )

        snapshot = getattr(self, '_run_input_snapshot', None)
        omrat_path = self._maybe_write_input_snapshot(
            snapshot, Path(out_dir), filename,
        )
        md_path = self._safe_write_results_markdown(
            calc_object, Path(out_dir), filename, name, ts_text, snapshot,
        )
        self._safe_write_collision_sidecar(
            calc_object, Path(out_dir), filename,
        )
        self._safe_write_drifting_sidecar(
            calc_object, Path(out_dir), filename,
        )

        self._log_run_save_summary(
            name=name, run_id=run_id, gpkg_path=gpkg_path,
            written_layers=written_layers, duration=duration,
            omrat_path=omrat_path, md_path=md_path,
        )

    def _auto_save_run_name(self) -> str:
        try:
            name = self.main_widget.LEModelName.text().strip()
        except Exception:
            name = ''
        if not name:
            name = time.strftime('run_%Y%m%d_%H%M%S')
        return name

    def _compute_run_duration(self) -> float | None:
        try:
            start = getattr(self, '_run_started_at', None)
            if start is not None:
                return float(time.monotonic() - start)
        except Exception:
            pass
        return None

    def _maybe_write_input_snapshot(
        self,
        snapshot: dict | None,
        out_dir: Path,
        filename: str,
    ) -> Path | None:
        if snapshot is None:
            return None
        omrat_path = out_dir / (Path(filename).stem + '.omrat')
        try:
            self._write_input_snapshot(omrat_path, snapshot)
            return omrat_path
        except Exception as exc:
            QgsMessageLog.logMessage(
                f"Failed to write input snapshot '{omrat_path}': {exc}",
                'OMRAT', Qgis.Warning,
            )
            return None

    def _safe_write_results_markdown(
        self, calc_object, out_dir, filename, run_name, ts_text, snapshot,
    ) -> Path | None:
        try:
            return self._write_results_markdown(
                calc_object=calc_object,
                out_dir=out_dir,
                base_filename=filename,
                run_name=run_name,
                timestamp=ts_text,
                snapshot=snapshot,
            )
        except Exception as exc:
            import traceback
            QgsMessageLog.logMessage(
                f"Markdown report write failed: {exc}\n{traceback.format_exc()}",
                'OMRAT', Qgis.Warning,
            )
            return None

    def _safe_write_collision_sidecar(
        self, calc_object, out_dir, filename,
    ) -> None:
        try:
            self._write_collision_report_sidecar(
                calc_object=calc_object,
                out_dir=out_dir,
                base_filename=filename,
            )
        except Exception as exc:
            QgsMessageLog.logMessage(
                f"Collision-report sidecar write failed: {exc}",
                'OMRAT', Qgis.Warning,
            )

    def _safe_write_drifting_sidecar(
        self, calc_object, out_dir, filename,
    ) -> None:
        try:
            self._write_drifting_report_sidecar(
                calc_object=calc_object,
                out_dir=out_dir,
                base_filename=filename,
            )
        except Exception as exc:
            QgsMessageLog.logMessage(
                f"Drifting-report sidecar write failed: {exc}",
                'OMRAT', Qgis.Warning,
            )

    @staticmethod
    def _log_run_save_summary(
        *, name, run_id, gpkg_path, written_layers, duration,
        omrat_path, md_path,
    ) -> None:
        dur_text = (
            f"duration={duration:.1f}s" if duration is not None
            else "duration=n/a"
        )
        snap_text = (
            f", snapshot={omrat_path.name}" if omrat_path is not None else ''
        )
        md_text = f", report={md_path.name}" if md_path is not None else ''
        QgsMessageLog.logMessage(
            f"Run '{name}' saved (run_id={run_id}, "
            f"layers={len(written_layers)}, "
            f"{dur_text}{snap_text}{md_text}).  GeoPackage: {gpkg_path}",
            'OMRAT', Qgis.Success,
        )

    # ------------------------------------------------------------------
    # File writers (markdown + json sidecars + .omrat snapshot)
    # ------------------------------------------------------------------
    def _write_results_markdown(
        self,
        *,
        calc_object: Any,
        out_dir: Path,
        base_filename: str,
        run_name: str,
        timestamp: str,
        snapshot: dict | None,
    ) -> Path | None:
        """Write the combined ``<name>_results_<timestamp>.md`` report.

        Returns the path of the written file, or ``None`` when there
        wasn't enough data to produce a useful report.
        """
        from omrat_utils.full_report import build_full_report_markdown

        drifting_md: str | None = None
        try:
            if hasattr(calc_object, 'generate_drifting_report_markdown'):
                drifting_md = calc_object.generate_drifting_report_markdown(
                    snapshot,
                )
        except Exception:
            drifting_md = None

        powered_grounding_report = getattr(
            calc_object, 'powered_grounding_report', None,
        )
        powered_allision_report = getattr(
            calc_object, 'powered_allision_report', None,
        )
        collision_report = getattr(calc_object, 'collision_report', None)

        if (
            not drifting_md
            and not powered_grounding_report
            and not powered_allision_report
            and not collision_report
        ):
            return None

        content = build_full_report_markdown(
            run_name=run_name,
            timestamp=timestamp,
            data=snapshot or {},
            drifting_md=drifting_md,
            powered_grounding_report=powered_grounding_report,
            powered_allision_report=powered_allision_report,
            collision_report=collision_report,
            structures_meta=getattr(calc_object, '_last_structures', None),
            depths_meta=(
                getattr(calc_object, '_last_depths_original', None)
                or getattr(calc_object, '_last_depths', None)
            ),
        )
        md_path = out_dir / self._results_md_filename(base_filename)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with md_path.open('w', encoding='utf-8') as f:
            f.write(content)
        return md_path

    @staticmethod
    def _results_md_filename(base_filename: str) -> str:
        """``<slug>_<timestamp>.gpkg`` -> ``<slug>_results_<timestamp>.md``."""
        stem = Path(base_filename).stem
        slug, _, ts_part = stem.rpartition('_')
        if not slug:
            slug = stem
            ts_part = ''
        if ts_part:
            return f"{slug}_results_{ts_part}.md"
        return f"{slug}_results.md"

    @staticmethod
    def _write_collision_report_sidecar(
        *, calc_object: Any, out_dir: Path, base_filename: str,
    ) -> Path | None:
        report = getattr(calc_object, 'collision_report', None)
        if not report:
            return None
        path = out_dir / (Path(base_filename).stem + '.collision.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        return path

    @staticmethod
    def _write_drifting_report_sidecar(
        *, calc_object: Any, out_dir: Path, base_filename: str,
    ) -> Path | None:
        report = getattr(calc_object, 'drifting_report', None)
        if not report:
            return None
        path = out_dir / (Path(base_filename).stem + '.drifting.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        return path

    def _write_input_snapshot(self, path: Path, data: dict) -> None:
        """Write ``data`` as a read-only ``.omrat`` snapshot."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        try:
            import stat as _stat
            mode = path.stat().st_mode
            path.chmod(
                mode & ~_stat.S_IWUSR & ~_stat.S_IWGRP & ~_stat.S_IWOTH,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Output folder + Run Model gating
    # ------------------------------------------------------------------
    def _get_output_dir(self) -> Path | None:
        """Resolved + validated output folder, or ``None`` if not set."""
        try:
            text = (self.main_widget.LEReportPath.text() or '').strip()
        except Exception:
            text = ''
        if not text:
            return None
        path = Path(text)
        if not path.is_dir():
            return None
        return path

    def choose_output_folder(self) -> None:
        """Pop a folder picker into ``LEReportPath``."""
        initial = ''
        try:
            initial = (self.main_widget.LEReportPath.text() or '').strip()
        except Exception:
            initial = ''
        if not initial:
            initial = str(Path.home())
        chosen = QFileDialog.getExistingDirectory(
            self.main_widget,
            'Select output folder for OMRAT result GeoPackages',
            initial,
        )
        if chosen:
            self.main_widget.LEReportPath.setText(chosen)
            try:
                QSettings().setValue('omrat/output_dir', chosen)
            except Exception:
                pass
            self._update_run_model_enabled()

    def _restore_output_dir(self) -> None:
        """At plugin start, restore the last-used output folder."""
        try:
            value = QSettings().value('omrat/output_dir', '', type=str)
            if value:
                self.main_widget.LEReportPath.setText(value)
        except Exception:
            pass

    def _update_run_model_enabled(self) -> None:
        """Run Model button is gated on a writable output folder + name."""
        try:
            btn = self.main_widget.pbRunModel
        except Exception:
            return
        try:
            run_name = (self.main_widget.LEModelName.text() or '').strip()
        except Exception:
            run_name = ''
        has_dir = self._get_output_dir() is not None
        enabled = has_dir and bool(run_name)
        try:
            btn.setEnabled(enabled)
            if enabled:
                btn.setToolTip('')
            elif not has_dir and not run_name:
                btn.setToolTip(
                    'Set a model name and pick an output folder first',
                )
            elif not has_dir:
                btn.setToolTip('Pick an output folder first')
            else:
                btn.setToolTip('Set a model name first')
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Previous-runs table on the Results tab
    # ------------------------------------------------------------------
    def _setup_previous_runs_table(self) -> None:
        """Configure ``TWPreviousRuns`` columns + add the load button."""
        from qgis.PyQt import QtCore, QtWidgets
        Qt = QtCore.Qt
        AIV = QtWidgets.QAbstractItemView
        tw = self.main_widget.TWPreviousRuns

        headers = ['Name', 'Date', 'Duration']
        tw.setColumnCount(len(headers))
        tw.setHorizontalHeaderLabels(headers)
        tw.horizontalHeader().setStretchLastSection(True)
        tw.setSelectionBehavior(_qt_enum(
            AIV, 'SelectRows', 'SelectionBehavior.SelectRows',
        ))
        tw.setSelectionMode(_qt_enum(
            AIV, 'ExtendedSelection', 'SelectionMode.ExtendedSelection',
        ))
        tw.setEditTriggers(_qt_enum(
            AIV, 'NoEditTriggers', 'EditTrigger.NoEditTriggers',
        ))
        tw.setContextMenuPolicy(_qt_enum(
            Qt, 'CustomContextMenu', 'ContextMenuPolicy.CustomContextMenu',
        ))
        tw.customContextMenuRequested.connect(
            self._previous_runs_context_menu,
        )

        sel_model = tw.selectionModel()
        if sel_model is not None:
            sel_model.selectionChanged.connect(
                lambda *_: self._on_previous_runs_selection_changed(),
            )
        self._add_load_run_to_map_button(tw)

    def _add_load_run_to_map_button(self, tw) -> None:
        from qgis.PyQt import QtWidgets
        try:
            parent_widget = tw.parentWidget()
            layout = (
                parent_widget.layout()
                if parent_widget is not None else None
            )
            btn = QtWidgets.QPushButton('Add selected run results to map')
            btn.setEnabled(False)
            btn.clicked.connect(self._add_selected_run_to_map)
            self._pb_add_run_to_map = btn
            if layout is not None:
                idx = layout.indexOf(tw)
                if idx >= 0:
                    layout.insertWidget(idx + 1, btn)
                else:
                    layout.addWidget(btn)
        except Exception as exc:
            QgsMessageLog.logMessage(
                f'Could not add "Add to map" button: {exc}',
                'OMRAT', Qgis.Warning,
            )

    def _select_latest_previous_run(self) -> None:
        """Select row 0 (most recent) so View buttons default to it."""
        try:
            tw = self.main_widget.TWPreviousRuns
        except Exception:
            return
        if tw.rowCount() == 0:
            return
        try:
            tw.clearSelection()
            tw.selectRow(0)
            tw.setCurrentCell(0, 0)
        except Exception:
            pass

    def refresh_previous_runs_table(self) -> None:
        """Reload ``TWPreviousRuns`` from the master history DB."""
        from omrat_utils.run_history import RunHistory
        from qgis.PyQt import QtCore, QtWidgets
        user_role = _qt_enum(
            QtCore.Qt, 'UserRole', 'ItemDataRole.UserRole',
        )
        tw = self.main_widget.TWPreviousRuns
        try:
            runs = RunHistory().list_runs()
        except Exception as exc:
            QgsMessageLog.logMessage(
                f'Could not read run history: {exc}', 'OMRAT', Qgis.Warning,
            )
            tw.setRowCount(0)
            return
        tw.setRowCount(len(runs))
        for i, run in enumerate(runs):
            cells = [
                run.name,
                run.timestamp,
                _format_duration(run.duration_seconds),
            ]
            for j, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if j == 0:
                    item.setData(user_role, run.run_id)
                tw.setItem(i, j, item)
        tw.resizeColumnsToContents()

    def _selected_run_ids(self) -> list[int]:
        from qgis.PyQt.QtCore import Qt
        user_role = _qt_enum(Qt, 'UserRole', 'ItemDataRole.UserRole')
        tw = self.main_widget.TWPreviousRuns
        out: list[int] = []
        seen: set[int] = set()
        sel_model = tw.selectionModel()
        if sel_model is None:
            return out
        for idx in sel_model.selectedRows():
            item = tw.item(idx.row(), 0)
            if item is None:
                continue
            run_id = item.data(user_role)
            if run_id is not None and int(run_id) not in seen:
                out.append(int(run_id))
                seen.add(int(run_id))
        return out

    def _on_previous_runs_selection_changed(self) -> None:
        """Add / remove comparison columns on TWAccidentResults to
        match the rows selected in TWPreviousRuns."""
        from omrat_utils.run_history import RunHistory
        run_ids = self._selected_run_ids()
        try:
            self._pb_add_run_to_map.setEnabled(len(run_ids) == 1)
        except Exception:
            pass
        if not run_ids:
            self._reset_accident_table_to_base()
            return
        try:
            runs = RunHistory().compare_runs(run_ids)
        except Exception as exc:
            QgsMessageLog.logMessage(
                f'Could not load previous-run details: {exc}',
                'OMRAT', Qgis.Warning,
            )
            self._reset_accident_table_to_base()
            return
        if not runs:
            self._reset_accident_table_to_base()
            return
        self._fill_result_fields_from_runs(runs)

    def _fill_result_fields_from_runs(self, runs) -> None:
        """Add one comparison column per selected row in TWPreviousRuns."""
        from qgis.PyQt.QtWidgets import QTableWidgetItem
        tw = getattr(self.main_widget, 'TWAccidentResults', None)
        if tw is None:
            return

        self._reset_accident_table_to_base()
        if not runs:
            return

        current_baseline = self._capture_current_run_baseline(tw)
        baseline_is_current = any(b is not None for b in current_baseline)
        if not baseline_is_current and runs:
            try:
                first_totals = runs[0].totals_dict()
            except Exception:
                first_totals = {}
            current_baseline = [
                first_totals.get(k) for k in self._ACCIDENT_TOTAL_KEYS
            ]
        delta_header = (
            'Δ vs current %' if baseline_is_current
            else f'Δ vs {getattr(runs[0], "name", "run 1")} %'
        )

        view_col = tw.columnCount() - 1
        for run_idx, run in enumerate(runs):
            try:
                totals = run.totals_dict()
            except Exception:
                totals = {}
            try:
                run_label = run.name
            except Exception:
                run_label = f'Run {run_idx + 1}'

            tw.insertColumn(view_col)
            tw.insertColumn(view_col + 1)
            view_col += 2
            prob_col = view_col - 2
            delta_col = view_col - 1
            tw.setHorizontalHeaderItem(
                prob_col, QTableWidgetItem(str(run_label)),
            )
            tw.setHorizontalHeaderItem(
                delta_col, QTableWidgetItem(delta_header),
            )

            for row, key in enumerate(self._ACCIDENT_TOTAL_KEYS):
                v = totals.get(key)
                tw.setItem(row, prob_col, QTableWidgetItem(_fmt(v)))
                tw.setItem(
                    row, delta_col,
                    QTableWidgetItem(
                        self._format_delta_pct(v, current_baseline[row]),
                    ),
                )

    def _capture_current_run_baseline(self, tw) -> list[float | None]:
        """Read the live LEP*-derived probabilities out of column 1."""
        baseline: list[float | None] = []
        for row in range(len(self._ACCIDENT_TOTAL_KEYS)):
            txt = ''
            try:
                item = tw.item(row, 1)
                if item is not None:
                    txt = item.text()
            except Exception:
                txt = ''
            try:
                baseline.append(float(txt))
            except (TypeError, ValueError):
                baseline.append(None)
        return baseline

    @staticmethod
    def _format_delta_pct(v: Any, base: Any) -> str:
        try:
            base_f = float(base) if base is not None else None
        except (TypeError, ValueError):
            base_f = None
        if v is None or base_f is None or base_f == 0:
            return '—'
        rel = (float(v) - base_f) / base_f * 100.0
        return f"{rel:+.1f}%"

    # ------------------------------------------------------------------
    # Context menu + load-to-map + delete
    # ------------------------------------------------------------------
    def _previous_runs_context_menu(self, pos) -> None:
        from qgis.PyQt.QtWidgets import QMenu
        tw = self.main_widget.TWPreviousRuns
        run_ids = self._selected_run_ids()
        if not run_ids:
            return
        menu = QMenu(tw)
        if len(run_ids) == 1:
            menu.addAction(
                'Add results to map',
                lambda: self._add_selected_run_to_map(),
            )
        menu.addSeparator()
        menu.addAction(
            'Delete from history',
            lambda: self._delete_runs(run_ids),
        )
        menu.addAction(
            'Delete from history + remove .gpkg file',
            lambda: self._delete_runs(run_ids, delete_gpkg=True),
        )
        menu.exec_(tw.viewport().mapToGlobal(pos))

    def _add_selected_run_to_map(self) -> None:
        """Load the selected run's per-run GeoPackage onto the canvas."""
        from omrat_utils.run_history import RunHistory
        from omrat_utils.run_persistence import load_run_results_to_map
        run_ids = self._selected_run_ids()
        if len(run_ids) != 1:
            return
        run = RunHistory().get_run(run_ids[0])
        if run is None:
            return
        gpkg_path = run.gpkg_path()
        if gpkg_path is None or not gpkg_path.is_file():
            QMessageBox.warning(
                self.main_widget, 'GeoPackage missing',
                f"The result file for run '{run.name}' could not be "
                f"found.\nExpected at: {gpkg_path}",
            )
            return
        try:
            new_layers = load_run_results_to_map(gpkg_path, run.name)
        except Exception as exc:
            import traceback
            QgsMessageLog.logMessage(
                f'Failed to load run on map: {exc}\n'
                f'{traceback.format_exc()}',
                'OMRAT', Qgis.Warning,
            )
            return
        for layer in new_layers:
            if layer is not None:
                self._history_layers.append(layer)

    def _delete_runs(
        self, run_ids: list[int], *, delete_gpkg: bool = False,
    ) -> None:
        from omrat_utils.run_history import RunHistory
        yes = _qt_enum(QMessageBox, 'Yes', 'StandardButton.Yes')
        msg = (
            f'Delete {len(run_ids)} run(s) from history'
            + (' AND remove the .gpkg files from disk?' if delete_gpkg else '?')
        )
        confirm = QMessageBox.question(
            self.main_widget, 'Delete runs', msg,
        )
        if confirm != yes:
            return
        history = RunHistory()
        for run_id in run_ids:
            try:
                history.delete_run(run_id, delete_gpkg=delete_gpkg)
            except Exception as exc:
                QgsMessageLog.logMessage(
                    f'Failed to delete run {run_id}: {exc}',
                    'OMRAT', Qgis.Warning,
                )
        self.refresh_previous_runs_table()

    def open_previous_runs_dialog(self) -> None:
        """File menu shortcut -- focuses the Results tab + the runs table."""
        try:
            self.main_widget.tabWidget.setCurrentWidget(
                self.main_widget.tab_9,
            )
        except Exception:
            pass
        self.refresh_previous_runs_table()
