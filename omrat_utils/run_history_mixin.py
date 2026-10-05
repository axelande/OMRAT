"""Run-history workflow, factored out of ``omrat.OMRAT``.

Owns everything tied to "what happens after **Run Model** finishes":

* writing the per-run GeoPackage, ``.omrat`` snapshot, combined Markdown
  report, and JSON sidecars (collision / drifting / consequence
  reports, powered per-leg shares);
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

from omrat_utils.accident_summary import (
    ACCIDENT_TOTAL_KEYS, summary_values,
)

if TYPE_CHECKING:
    pass

# QSettings key holding the run_id of the "main" run that the
# comparison columns on the Run Analysis tab are computed against.
MAIN_RUN_SETTING = 'omrat/main_run_id'

# ``cbRunProject`` item data for "All projects".
ALL_PROJECTS = '__all_projects__'

_NO_VERSION_TIP = (
    'Not recorded: the run is older than the version stamp, '
    'or its .omrat snapshot is missing.'
)


def consequence_sidecar_path(run) -> Path | None:
    """``<gpkg stem>.consequence.json`` for ``run`` (a ``RunMeta``), or
    ``None`` when the run has no GeoPackage path recorded."""
    try:
        gpkg_path = run.gpkg_path() if hasattr(run, 'gpkg_path') else None
    except Exception:  # nosec B110 B112
        gpkg_path = None
    if gpkg_path is None:
        return None
    stem = Path(gpkg_path).with_suffix('')
    return Path(str(stem) + '.consequence.json')


def load_consequence_sidecar(run) -> dict | None:
    """Read the catastrophe-exceedance result saved next to ``run``.

    Returns the ``compute_catastrophe_exceedance`` dict, or ``None`` when
    the sidecar is missing (runs recorded before it was written) or
    unreadable.
    """
    path = consequence_sidecar_path(run)
    if path is None or not path.is_file():
        return None
    try:
        with path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception as exc:
        QgsMessageLog.logMessage(
            f"Could not read {path}: {exc}", 'OMRAT', Qgis.MessageLevel.Warning,
        )
        return None
    return payload if isinstance(payload, dict) else None


def powered_sidecar_path(run) -> Path | None:
    """``<gpkg stem>.powered.json`` for ``run``, or ``None`` without a GeoPackage path."""
    from compute.powered_breakdown import SIDECAR_SUFFIX
    try:
        gpkg_path = run.gpkg_path() if hasattr(run, 'gpkg_path') else None
        if gpkg_path is None:
            return None
        gpkg = Path(gpkg_path)
        return gpkg.with_name(gpkg.stem + SIDECAR_SUFFIX)
    except Exception:  # nosec B110 B112
        return None


def load_powered_sidecar(run) -> dict | None:
    """``{"allision": {...}, "grounding": {...}}`` saved next to ``run``.

    The blocks are ``compute.powered_breakdown.summarize_report`` output
    (totals + per leg-direction frequencies).  ``None`` for runs recorded
    before the sidecar existed, or when it cannot be read.
    """
    path = powered_sidecar_path(run)
    if path is None or not path.is_file():
        return None
    try:
        with path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception as exc:
        QgsMessageLog.logMessage(
            f"Could not read {path}: {exc}", 'OMRAT', Qgis.MessageLevel.Warning,
        )
        return None
    return payload if isinstance(payload, dict) else None


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
    _ACCIDENT_TOTAL_KEYS: tuple[str, ...] = ACCIDENT_TOTAL_KEYS

    # Column indices of ``TWPreviousRuns``.
    _RUNS_COL_NAME = 0
    _RUNS_COL_MAIN = 1
    _RUNS_COL_DATE = 2
    _RUNS_COL_DURATION = 3
    _RUNS_COL_VERSION = 4
    _RUNS_COL_PROJECT = 5

    # ------------------------------------------------------------------
    # Auto-save after a successful run
    # ------------------------------------------------------------------
    def _auto_save_run(self, calc_object: Any) -> None:
        """Write per-run GeoPackage + record metadata in the master DB."""
        from omrat_utils.gather_data import plugin_version
        from omrat_utils.run_history import (
            NO_PROJECT_LABEL, RunHistory, folder_project, make_run_filename,
            totals_from_calc,
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
            f"Writing per-run results to {gpkg_path}", 'OMRAT', Qgis.MessageLevel.Info,
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
                'OMRAT', Qgis.MessageLevel.Critical,
            )
            written_layers = []

        duration = self._compute_run_duration()

        snapshot = getattr(self, '_run_input_snapshot', None)
        project = self._run_project_tag(snapshot)
        history = RunHistory()
        run_id = history.save_run(
            name=name,
            timestamp=ts_text,
            duration_seconds=duration,
            totals=totals_from_calc(calc_object),
            output_dir=str(out_dir),
            output_filename=filename,
            project=project,
            omrat_version=plugin_version(),
        )
        # Show this run's project: the refresh after a run selects row 0.
        self._runs_project_filter = project or folder_project(str(out_dir)) or NO_PROJECT_LABEL

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
        self._safe_write_consequence_sidecar(
            calc_object, Path(out_dir), filename,
        )
        self._safe_write_powered_sidecar(
            calc_object, Path(out_dir), filename,
        )

        self._log_run_save_summary(
            name=name, run_id=run_id, gpkg_path=gpkg_path,
            written_layers=written_layers, duration=duration,
            omrat_path=omrat_path, md_path=md_path,
        )

    def _run_project_tag(self, snapshot: dict | None = None) -> str:
        """Project name of the run: the one in the input snapshot taken at
        kick-off, else the field.  '' = untagged (output folder name)."""
        if isinstance(snapshot, dict) and snapshot.get('project_name'):
            return str(snapshot['project_name']).strip()
        return self._project_name_text()

    def _project_name_text(self) -> str:
        try:
            return (self.main_widget.LEProjectName.text() or '').strip()
        except Exception:  # nosec B110 B112
            return ''

    def _current_project_label(self) -> str:
        """Project a run started now would be grouped under."""
        from omrat_utils.run_history import folder_project
        name = self._project_name_text()
        if name:
            return name
        try:
            folder = (self.main_widget.LEReportPath.text() or '').strip()
        except Exception:  # nosec B110 B112
            folder = ''
        return folder_project(folder)

    def _auto_save_run_name(self) -> str:
        try:
            name = self.main_widget.LEModelName.text().strip()
        except Exception:  # nosec B110 B112
            name = ''
        if not name:
            name = time.strftime('run_%Y%m%d_%H%M%S')
        return name

    def _compute_run_duration(self) -> float | None:
        try:
            start = getattr(self, '_run_started_at', None)
            if start is not None:
                return float(time.monotonic() - start)
        except Exception:  # nosec B110 B112
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
                'OMRAT', Qgis.MessageLevel.Warning,
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
                'OMRAT', Qgis.MessageLevel.Warning,
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
                'OMRAT', Qgis.MessageLevel.Warning,
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
                'OMRAT', Qgis.MessageLevel.Warning,
            )

    def _safe_write_consequence_sidecar(
        self, calc_object, out_dir, filename,
    ) -> None:
        try:
            self._write_consequence_report_sidecar(
                calc_object=calc_object,
                out_dir=out_dir,
                base_filename=filename,
            )
        except Exception as exc:
            QgsMessageLog.logMessage(
                f"Consequence-report sidecar write failed: {exc}",
                'OMRAT', Qgis.MessageLevel.Warning,
            )

    def _safe_write_powered_sidecar(
        self, calc_object, out_dir, filename,
    ) -> None:
        try:
            self._write_powered_report_sidecar(
                calc_object=calc_object,
                out_dir=out_dir,
                base_filename=filename,
            )
        except Exception as exc:
            QgsMessageLog.logMessage(
                f"Powered-report sidecar write failed: {exc}",
                'OMRAT', Qgis.MessageLevel.Warning,
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
            'OMRAT', Qgis.MessageLevel.Success,
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
        from compute.run_phases import PHASE_LABELS, skipped_phases
        from omrat_utils.full_report import build_full_report_markdown

        drifting_md: str | None = None
        try:
            # No drifting report = drifting was not calculated; the
            # generator would otherwise render a section of zeros.
            if (getattr(calc_object, 'drifting_report', None) is not None
                    and hasattr(calc_object, 'generate_drifting_report_markdown')):
                drifting_md = calc_object.generate_drifting_report_markdown(
                    snapshot,
                )
        except Exception:  # nosec B110 B112
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

        run_phases = getattr(calc_object, 'run_phases', None)
        skipped = (
            [PHASE_LABELS[p] for p in skipped_phases(run_phases)]
            if isinstance(run_phases, (tuple, list)) else []
        )
        content = build_full_report_markdown(
            run_name=run_name,
            timestamp=timestamp,
            data=snapshot or {},
            drifting_md=drifting_md,
            powered_grounding_report=powered_grounding_report,
            powered_allision_report=powered_allision_report,
            collision_report=collision_report,
            # Drifting sets ``_last_structures``; a run without drifting
            # still has the powered allision model's own list (heights).
            structures_meta=(
                getattr(calc_object, '_last_structures', None)
                or getattr(calc_object, '_last_powered_allision_structs', None)
            ),
            depths_meta=(
                getattr(calc_object, '_last_depths_original', None)
                or getattr(calc_object, '_last_depths', None)
            ),
            skipped_parts=skipped,
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

    @staticmethod
    def _write_consequence_report_sidecar(
        *, calc_object: Any, out_dir: Path, base_filename: str,
    ) -> Path | None:
        """Persist ``calc.consequence_result`` as ``<stem>.consequence.json``
        so the catastrophe table can be re-filled for a previous run."""
        result = getattr(calc_object, 'consequence_result', None)
        if not isinstance(result, dict) or not result.get('levels'):
            return None
        path = out_dir / (Path(base_filename).stem + '.consequence.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        return path

    @staticmethod
    def _write_powered_report_sidecar(
        *, calc_object: Any, out_dir: Path, base_filename: str,
    ) -> Path | None:
        """Persist the per leg-direction powered frequencies as ``<stem>.powered.json``
        so the powered View can show each leg's share for a previous run."""
        from compute.powered_breakdown import SIDECAR_SUFFIX, sidecar_payload
        payload = sidecar_payload(
            getattr(calc_object, 'powered_allision_report', None),
            getattr(calc_object, 'powered_grounding_report', None),
        )
        if payload is None:
            return None
        path = out_dir / (Path(base_filename).stem + SIDECAR_SUFFIX)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, default=str)
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
        except Exception:  # nosec B110 B112
            pass

    # ------------------------------------------------------------------
    # Output folder + Run Model gating
    # ------------------------------------------------------------------
    def _get_output_dir(self) -> Path | None:
        """Resolved + validated output folder, or ``None`` if not set."""
        try:
            text = (self.main_widget.LEReportPath.text() or '').strip()
        except Exception:  # nosec B110 B112
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
        except Exception:  # nosec B110 B112
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
            except Exception:  # nosec B110 B112
                pass
            self._update_run_model_enabled()

    def _restore_output_dir(self) -> None:
        """At plugin start, restore the last-used output folder."""
        try:
            value = QSettings().value('omrat/output_dir', '', type=str)
            if value:
                self.main_widget.LEReportPath.setText(value)
        except Exception:  # nosec B110 B112
            pass

    def _update_run_model_enabled(self) -> None:
        """Run Model button is gated on a writable output folder + name."""
        try:
            btn = self.main_widget.pbRunModel
        except Exception:  # nosec B110 B112
            return
        try:
            run_name = (self.main_widget.LEModelName.text() or '').strip()
        except Exception:  # nosec B110 B112
            run_name = ''
        has_dir = self._get_output_dir() is not None
        enabled = has_dir and bool(run_name)
        try:
            btn.setEnabled(enabled)
            if enabled:
                # Names the parts a partial Calculation matrix leaves out.
                tooltip = getattr(self, '_run_model_tooltip', None)
                btn.setToolTip(tooltip() if callable(tooltip) else '')
            elif not has_dir and not run_name:
                btn.setToolTip(
                    'Set a model name and pick an output folder first',
                )
            elif not has_dir:
                btn.setToolTip('Pick an output folder first')
            else:
                btn.setToolTip('Set a model name first')
        except Exception:  # nosec B110 B112
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

        headers = ['Name', 'Main', 'Date', 'Duration', 'Version', 'Project']
        tw.setColumnCount(len(headers))
        tw.setHorizontalHeaderLabels(headers)
        tw.horizontalHeader().setStretchLastSection(True)
        try:
            tw.horizontalHeaderItem(self._RUNS_COL_MAIN).setToolTip(
                'Tick one run to make it the "main" run.  The delta '
                'columns of the accident table are then computed against '
                'it instead of against the current model.'
            )
        except Exception:  # nosec B110 B112
            pass
        try:
            tw.horizontalHeaderItem(self._RUNS_COL_VERSION).setToolTip(
                'OMRAT version that produced the run')
            tw.horizontalHeaderItem(self._RUNS_COL_PROJECT).setToolTip(
                'Project name of the run, else the name of its output folder.  '
                'Right-click -> Set project... to change it.')
        except Exception:  # nosec B110 B112
            pass
        tw.itemChanged.connect(self._on_previous_runs_item_changed)
        cb = getattr(self.main_widget, 'cbRunProject', None)
        if cb is not None:
            cb.currentIndexChanged.connect(self._on_run_project_filter_changed)
        le = getattr(self.main_widget, 'LEProjectName', None)
        if le is not None:
            le.editingFinished.connect(self._on_project_name_edited)
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
                'OMRAT', Qgis.MessageLevel.Warning,
            )

    def _select_latest_previous_run(self) -> None:
        """Select row 0 (most recent) so View buttons default to it."""
        try:
            tw = self.main_widget.TWPreviousRuns
        except Exception:  # nosec B110 B112
            return
        if tw.rowCount() == 0:
            return
        try:
            tw.clearSelection()
            tw.selectRow(0)
            tw.setCurrentCell(0, 0)
        except Exception:  # nosec B110 B112
            pass

    def refresh_previous_runs_table(self) -> None:
        """Reload ``TWPreviousRuns`` from the master history DB, showing
        the runs of the project picked in ``cbRunProject``."""
        from omrat_utils.run_history import RunHistory
        from qgis.PyQt import QtCore, QtWidgets
        user_role = _qt_enum(
            QtCore.Qt, 'UserRole', 'ItemDataRole.UserRole',
        )
        tw = self.main_widget.TWPreviousRuns
        try:
            history = RunHistory()
            all_runs = history.list_runs()
        except Exception as exc:
            QgsMessageLog.logMessage(
                f'Could not read run history: {exc}', 'OMRAT', Qgis.MessageLevel.Warning,
            )
            tw.setRowCount(0)
            return
        try:
            # Runs from before the version column: read their snapshot once.
            history.backfill_versions(all_runs)
        except Exception as exc:  # nosec B110 B112 - the table must still show
            QgsMessageLog.logMessage(
                f'Could not fill in run versions: {exc}', 'OMRAT', Qgis.MessageLevel.Warning,
            )
        main_id = self._get_main_run_id()
        if main_id is not None and main_id not in {r.run_id for r in all_runs}:
            # The main run was deleted from history: forget it.
            self._set_main_run_id(None)
            main_id = None
        project = self._fill_run_project_combo(all_runs)
        runs = [r for r in all_runs if project is None or r.project_label() == project]
        checked = _qt_enum(QtCore.Qt, 'Checked', 'CheckState.Checked')
        unchecked = _qt_enum(QtCore.Qt, 'Unchecked', 'CheckState.Unchecked')
        flag_enabled = _qt_enum(QtCore.Qt, 'ItemIsEnabled', 'ItemFlag.ItemIsEnabled')
        flag_checkable = _qt_enum(
            QtCore.Qt, 'ItemIsUserCheckable', 'ItemFlag.ItemIsUserCheckable',
        )
        flag_selectable = _qt_enum(
            QtCore.Qt, 'ItemIsSelectable', 'ItemFlag.ItemIsSelectable',
        )
        was_blocked = tw.blockSignals(True)
        try:
            tw.setRowCount(len(runs))
            for i, run in enumerate(runs):
                name_item = QtWidgets.QTableWidgetItem(run.name)
                name_item.setData(user_role, run.run_id)
                tw.setItem(i, self._RUNS_COL_NAME, name_item)

                main_item = QtWidgets.QTableWidgetItem('')
                main_item.setFlags(flag_enabled | flag_checkable | flag_selectable)
                main_item.setCheckState(
                    checked if run.run_id == main_id else unchecked,
                )
                main_item.setToolTip('Compare against this run')
                tw.setItem(i, self._RUNS_COL_MAIN, main_item)

                tw.setItem(
                    i, self._RUNS_COL_DATE,
                    QtWidgets.QTableWidgetItem(run.timestamp),
                )
                tw.setItem(
                    i, self._RUNS_COL_DURATION,
                    QtWidgets.QTableWidgetItem(
                        _format_duration(run.duration_seconds),
                    ),
                )
                version_item = QtWidgets.QTableWidgetItem(run.omrat_version or '\u2014')
                if not run.omrat_version:
                    version_item.setToolTip(_NO_VERSION_TIP)
                tw.setItem(i, self._RUNS_COL_VERSION, version_item)
                project_item = QtWidgets.QTableWidgetItem(run.project_label())
                project_item.setToolTip(
                    'Project name' if (run.project or '').strip()
                    else 'Not tagged: the name of the output folder')
                tw.setItem(i, self._RUNS_COL_PROJECT, project_item)
        finally:
            tw.blockSignals(was_blocked)
        # One project shown: its name in every row would be noise.
        tw.setColumnHidden(self._RUNS_COL_PROJECT, project is not None)
        tw.resizeColumnsToContents()

    def _fill_run_project_combo(self, all_runs) -> str | None:
        """Refill ``cbRunProject`` and return the project to show (``None``
        = all).  No explicit choice (``_runs_project_filter is None``)
        follows the current project, or all projects when it has no runs."""
        from omrat_utils.run_history import project_counts
        counts = project_counts(all_runs)
        wanted = getattr(self, '_runs_project_filter', None)
        if wanted is not None and wanted != ALL_PROJECTS and wanted not in counts:
            # Its runs were deleted or moved to another project.
            self._runs_project_filter = wanted = None
        if wanted is None:
            current = self._current_project_label()
            wanted = current if current in counts else ALL_PROJECTS
        cb = getattr(self.main_widget, 'cbRunProject', None)
        if cb is not None:
            was_blocked = cb.blockSignals(True)
            try:
                cb.clear()
                cb.addItem(f'All projects ({len(all_runs)})', ALL_PROJECTS)
                for label, n in counts.items():
                    cb.addItem(f'{label} ({n})', label)
                cb.setCurrentIndex(max(0, cb.findData(wanted)))
            finally:
                cb.blockSignals(was_blocked)
        return None if wanted == ALL_PROJECTS else wanted

    def _on_run_project_filter_changed(self, index: int) -> None:
        cb = getattr(self.main_widget, 'cbRunProject', None)
        if cb is None or index < 0:
            return
        self._runs_project_filter = cb.itemData(index)
        self.refresh_previous_runs_table()

    def _on_project_name_edited(self) -> None:
        """A new project name moves the default filter along with it."""
        if getattr(self, '_runs_project_filter', None) is None:
            self.refresh_previous_runs_table()

    def show_current_project_runs(self) -> None:
        """Drop an explicit filter choice and show the current project
        (after a project is loaded or the model is cleared)."""
        self._runs_project_filter = None
        self.refresh_previous_runs_table()

    # ------------------------------------------------------------------
    # "Main" run (comparison baseline)
    # ------------------------------------------------------------------
    def _get_main_run_id(self) -> int | None:
        """run_id of the user-designated main run, or ``None``."""
        cached = getattr(self, '_main_run_id', None)
        if cached is not None:
            return int(cached)
        try:
            value = QSettings().value(MAIN_RUN_SETTING, None)
        except Exception:  # nosec B110 B112
            value = None
        if value in (None, ''):
            return None
        try:
            run_id = int(value)
        except (TypeError, ValueError):
            return None
        self._main_run_id = run_id
        return run_id

    def _set_main_run_id(self, run_id: int | None) -> None:
        self._main_run_id = run_id
        try:
            settings = QSettings()
            if run_id is None:
                settings.remove(MAIN_RUN_SETTING)
            else:
                settings.setValue(MAIN_RUN_SETTING, int(run_id))
        except Exception:  # nosec B110 B112
            pass

    def _main_run_meta(self):
        """``RunMeta`` of the main run, or ``None`` (unset / deleted)."""
        run_id = self._get_main_run_id()
        if run_id is None:
            return None
        from omrat_utils.run_history import RunHistory
        try:
            run = RunHistory().get_run(run_id)
        except Exception:  # nosec B110 B112
            return None
        if run is None:
            self._set_main_run_id(None)
        return run

    def _on_previous_runs_item_changed(self, item) -> None:
        """Checkbox in the Main column toggled -> exactly one main run."""
        from qgis.PyQt import QtCore
        if item is None or item.column() != self._RUNS_COL_MAIN:
            return
        tw = self.main_widget.TWPreviousRuns
        user_role = _qt_enum(QtCore.Qt, 'UserRole', 'ItemDataRole.UserRole')
        checked = _qt_enum(QtCore.Qt, 'Checked', 'CheckState.Checked')
        unchecked = _qt_enum(QtCore.Qt, 'Unchecked', 'CheckState.Unchecked')
        name_item = tw.item(item.row(), self._RUNS_COL_NAME)
        run_id = name_item.data(user_role) if name_item is not None else None
        if run_id is None:
            return
        run_id = int(run_id)
        if item.checkState() == checked:
            self._set_main_run_id(run_id)
            was_blocked = tw.blockSignals(True)
            try:
                for row in range(tw.rowCount()):
                    other = tw.item(row, self._RUNS_COL_MAIN)
                    if other is not None and row != item.row():
                        other.setCheckState(unchecked)
            finally:
                tw.blockSignals(was_blocked)
        elif self._get_main_run_id() == run_id:
            self._set_main_run_id(None)
        # Re-fill the comparison columns against the new baseline.
        self._on_previous_runs_selection_changed()

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
        """Add / remove comparison columns on TWAccidentResults and
        TWCatastropheResults to match the rows selected in
        TWPreviousRuns."""
        from omrat_utils.run_history import RunHistory
        run_ids = self._selected_run_ids()
        try:
            self._pb_add_run_to_map.setEnabled(len(run_ids) == 1)
        except Exception:  # nosec B110 B112
            pass
        if not run_ids:
            self._reset_accident_table_to_base()
            self._reset_catastrophe_table_to_base()
            return
        try:
            runs = RunHistory().compare_runs(run_ids)
        except Exception as exc:
            QgsMessageLog.logMessage(
                f'Could not load previous-run details: {exc}',
                'OMRAT', Qgis.MessageLevel.Warning,
            )
            self._reset_accident_table_to_base()
            self._reset_catastrophe_table_to_base()
            return
        if not runs:
            self._reset_accident_table_to_base()
            self._reset_catastrophe_table_to_base()
            return
        self._fill_result_fields_from_runs(runs)
        self._fill_catastrophe_table_from_runs(runs)

    def _fill_result_fields_from_runs(self, runs) -> None:
        """Add one comparison column per selected row in TWPreviousRuns."""
        from qgis.PyQt.QtWidgets import QTableWidgetItem
        tw = getattr(self.main_widget, 'TWAccidentResults', None)
        if tw is None:
            return

        self._reset_accident_table_to_base()
        if not runs:
            return

        current_baseline, delta_header = self._comparison_baseline(tw, runs)
        baseline_summary = summary_values(
            dict(zip(self._ACCIDENT_TOTAL_KEYS, current_baseline)),
        )
        n_accidents = len(self._ACCIDENT_TOTAL_KEYS)

        view_col = tw.columnCount() - 1
        for run_idx, run in enumerate(runs):
            try:
                totals = run.totals_dict()
            except Exception:  # nosec B110 B112
                totals = {}
            try:
                run_label = run.name
            except Exception:  # nosec B110 B112
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
                tw.setItem(row, prob_col, QTableWidgetItem(self._format_result(v)))
                tw.setItem(
                    row, delta_col,
                    QTableWidgetItem(
                        self._format_delta_pct(v, current_baseline[row]),
                    ),
                )
            for offset, v in enumerate(summary_values(totals)):
                row = n_accidents + offset
                if row >= tw.rowCount():
                    break
                self._set_bold_item(tw, row, prob_col, self._format_result(v))
                self._set_bold_item(
                    tw, row, delta_col,
                    self._format_delta_pct(v, baseline_summary[offset]),
                )

    def _comparison_baseline(self, tw, runs) -> tuple[list[float | None], str]:
        """Pick the baseline the delta columns are computed against.

        Priority: the user-designated *main* run (Main checkbox in
        ``TWPreviousRuns``), then the current model's live values, then
        the first selected run.
        """
        main_run = self._main_run_meta()
        if main_run is not None:
            try:
                main_totals = main_run.totals_dict()
            except Exception:  # nosec B110 B112
                main_totals = {}
            baseline = [main_totals.get(k) for k in self._ACCIDENT_TOTAL_KEYS]
            return baseline, f'Δ vs main ({main_run.name}) %'

        baseline = self._capture_current_run_baseline(tw)
        if any(b is not None for b in baseline):
            return baseline, 'Δ vs current %'
        try:
            first_totals = runs[0].totals_dict()
        except Exception:  # nosec B110 B112
            first_totals = {}
        baseline = [first_totals.get(k) for k in self._ACCIDENT_TOTAL_KEYS]
        return baseline, f'Δ vs {getattr(runs[0], "name", "run 1")} %'

    @staticmethod
    def _set_bold_item(tw, row: int, col: int, text: str) -> None:
        from qgis.PyQt.QtWidgets import QTableWidgetItem
        item = QTableWidgetItem(text)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        tw.setItem(row, col, item)

    def _capture_current_run_baseline(self, tw) -> list[float | None]:
        """Live annual frequencies of the nine accident rows.

        Taken from the hidden ``LEP*`` widgets (via
        ``_current_accident_totals``) rather than the visible cells,
        which may be showing years between incidents.
        """
        try:
            totals = self._current_accident_totals(tw)
        except Exception:  # nosec B110 B112
            totals = {}
        return [totals.get(k) for k in self._ACCIDENT_TOTAL_KEYS]

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
        menu.addAction(
            'Set project...',
            lambda: self._set_project_for_runs(run_ids),
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
        menu.exec(tw.viewport().mapToGlobal(pos))

    def _ask_project_name(
        self, n_runs: int, labels: list[str], current: str, parent=None,
    ) -> tuple[str, bool]:
        from qgis.PyQt.QtWidgets import QInputDialog
        return QInputDialog.getItem(
            parent or self.main_widget, 'Set project',
            f'Project for {n_runs} run(s).\n'
            'Pick one, type a new name, or clear the text to use the output folder name.',
            labels, labels.index(current) if current in labels else 0, True,
        )

    def _set_project_for_runs(self, run_ids: list[int], parent=None) -> None:
        """Re-tag runs saved under the wrong project (select all rows of a
        project first to rename it).  ``parent``: the window asking."""
        from omrat_utils.run_history import RunHistory, project_counts
        history = RunHistory()
        all_runs = history.list_runs()
        ids = set(run_ids)
        selected = [r for r in all_runs if r.run_id in ids]
        if not selected:
            return
        text, ok = self._ask_project_name(
            len(selected), list(project_counts(all_runs)), selected[0].project_label(),
            parent=parent)
        if not ok:
            return
        history.set_project([r.run_id for r in selected], text)
        self.refresh_previous_runs_table()

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
                'OMRAT', Qgis.MessageLevel.Warning,
            )
            return
        for layer in new_layers:
            if layer is not None:
                self._history_layers.append(layer)

    def _delete_runs(
        self, run_ids: list[int], *, delete_gpkg: bool = False, parent=None,
    ) -> None:
        from omrat_utils.run_history import RunHistory
        yes = _qt_enum(QMessageBox, 'Yes', 'StandardButton.Yes')
        msg = (
            f'Delete {len(run_ids)} run(s) from history'
            + (' AND remove the .gpkg files from disk?' if delete_gpkg else '?')
        )
        confirm = QMessageBox.question(
            parent or self.main_widget, 'Delete runs', msg,
        )
        if confirm != yes:
            return
        history = RunHistory()
        for run_id in run_ids:
            if self._get_main_run_id() == run_id:
                self._set_main_run_id(None)
            try:
                history.delete_run(run_id, delete_gpkg=delete_gpkg)
            except Exception as exc:
                QgsMessageLog.logMessage(
                    f'Failed to delete run {run_id}: {exc}',
                    'OMRAT', Qgis.MessageLevel.Warning,
                )
        self.refresh_previous_runs_table()

    def open_previous_runs_dialog(self) -> None:
        """File -> Manage previous runs...: every run of every project in
        one dialog.  It used to only switch to the Run Analysis tab, so
        from that tab the menu entry seemed to do nothing."""
        from omrat_utils.previous_runs_dialog import PreviousRunsDialog
        PreviousRunsDialog(self, self.main_widget).exec()
        self.refresh_previous_runs_table()
