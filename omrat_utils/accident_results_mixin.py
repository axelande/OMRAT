"""Accident-results table + per-row View dispatcher, factored out of
``omrat.OMRAT``.

Owns the ``TWAccidentResults`` table on the Run Analysis tab plus the
View-button slots that open the interactive visualiser for the
currently-selected run in ``TWPreviousRuns``:

* :meth:`AccidentResultsMixin._setup_accident_results_table` builds
  the table, creates the legacy ``LEP*`` line-edits as hidden widgets
  (so existing compute / test code that calls ``setText`` still works),
  and wires the per-row ``View`` button.
* :meth:`AccidentResultsMixin._dispatch_view` is the single entry-point
  every ``show_*`` slot delegates to.  It reads the selected run from
  ``TWPreviousRuns``, loads its ``.omrat`` snapshot + JSON sidecars,
  and feeds them to the matching calc-method on ``self.calc``.

Pure-data work for the various interactive popups lives in:

* :mod:`compute.visualization` -- drift / powered visualisers,
* :mod:`compute.ship_collision_model` -- collision breakdown dialogs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtWidgets import QMessageBox

from omrat_utils.run_history_mixin import _qt_enum

if TYPE_CHECKING:
    pass


class AccidentResultsMixin:
    """``TWAccidentResults`` setup + per-row View-button dispatcher."""

    # (Accident type label, LEP* widget name on main_widget,
    #  View pushbutton name on main_widget, slot on self).
    # Order is the row order of the table.
    _ACCIDENT_ROWS: tuple[tuple[str, str, str, str], ...] = (
        ('Drifting allision', 'LEPDriftAllision',
         'pbViewDriftingAllision', 'show_drift_allision'),
        ('Drifting grounding', 'LEPDriftingGrounding',
         'pbViewDriftingGrounding', 'show_drift_grounding'),
        ('Powered allision', 'LEPPoweredAllision',
         'pbViewPoweredAllision', 'show_powered_allision'),
        ('Powered grounding', 'LEPPoweredGrounding',
         'pbViewPoweredGrounding', 'show_powered_grounding'),
        ('Overtaking collision', 'LEPOvertakingCollision',
         'pbViewOvertakingCollision', 'show_overtaking_collision'),
        ('Head-on collision', 'LEPHeadOnCollision',
         'pbViewHeadOnCollision', 'show_head_on_collision'),
        ('Crossing collision', 'LEPCrossingCollision',
         'pbViewCrossingCollision', 'show_crossing_collision'),
        ('Merging collision', 'LEPMergingCollision',
         'pbViewMergingCollision', 'show_merging_collision'),
    )

    # View slot -> (calc-method-name, label, optional breakdown key).
    _VIEW_DISPATCH: dict[str, tuple[str, str, str | None]] = {
        'show_drift_allision': (
            'run_drift_visualization', 'Drifting allision', None,
        ),
        'show_drift_grounding': (
            'run_drift_grounding_visualization', 'Drifting grounding', None,
        ),
        'show_powered_allision': (
            'run_powered_allision_visualization', 'Powered allision', None,
        ),
        'show_powered_grounding': (
            'run_powered_grounding_visualization',
            'Powered grounding', None,
        ),
        'show_overtaking_collision': (
            'run_collision_breakdown_dialog',
            'Overtaking collision', 'overtaking',
        ),
        'show_head_on_collision': (
            'run_collision_breakdown_dialog',
            'Head-on collision', 'head_on',
        ),
        'show_crossing_collision': (
            'run_collision_breakdown_dialog',
            'Crossing collision', 'crossing',
        ),
        'show_merging_collision': (
            'run_collision_breakdown_dialog',
            'Merging collision', 'merging',
        ),
    }

    # ------------------------------------------------------------------
    # Table setup
    # ------------------------------------------------------------------
    def _setup_accident_results_table(self) -> None:
        """Configure ``TWAccidentResults`` and create the legacy LEP*
        line-edits as hidden Python attributes on ``main_widget``.

        Many compute / test paths still call ``LEPDriftAllision.setText``
        etc., so we keep the same widget names available.  Edits are
        forwarded to the new table cell via ``_on_lep_text_changed``.
        """
        from qgis.PyQt import QtCore, QtWidgets

        Qt = QtCore.Qt
        tw = getattr(self.main_widget, 'TWAccidentResults', None)
        if tw is None:
            return

        self._ensure_legacy_lep_widgets()
        self._configure_accident_table(tw)
        self._populate_accident_rows(tw)
        self._wire_clipboard_copy_shortcut(tw)
        self._setup_catastrophe_results_table()

    def _setup_catastrophe_results_table(self) -> None:
        """Configure ``TWCatastropheResults`` -- the small annual-frequency
        table that sits below ``TWAccidentResults`` on the Run Analysis
        tab.  Headers are set here; rows get (re)populated each run from
        ``_populate_catastrophe_results_table``.
        """
        from qgis.PyQt import QtWidgets
        from qgis.PyQt.QtWidgets import QHeaderView

        AIV = QtWidgets.QAbstractItemView
        tw = getattr(self.main_widget, 'TWCatastropheResults', None)
        if tw is None:
            return
        tw.setColumnCount(3)
        tw.setHorizontalHeaderLabels([
            'Catastrophe level', 'Threshold (m^3)', 'Exceedance (events/year)',
        ])
        tw.setRowCount(0)
        tw.verticalHeader().setVisible(False)
        tw.setEditTriggers(_qt_enum(
            AIV, 'NoEditTriggers', 'EditTrigger.NoEditTriggers',
        ))
        tw.setSelectionBehavior(_qt_enum(
            AIV, 'SelectRows', 'SelectionBehavior.SelectRows',
        ))
        try:
            mode_stretch = _qt_enum(
                QHeaderView, 'Stretch', 'ResizeMode.Stretch',
            )
            mode_resize = _qt_enum(
                QHeaderView, 'ResizeToContents', 'ResizeMode.ResizeToContents',
            )
            tw.horizontalHeader().setSectionResizeMode(0, mode_stretch)
            tw.horizontalHeader().setSectionResizeMode(1, mode_resize)
            tw.horizontalHeader().setSectionResizeMode(2, mode_resize)
        except Exception:
            pass
        self._wire_clipboard_copy_shortcut(tw)

    def _reset_accident_table_to_base(self) -> None:
        """Strip per-run comparison columns from ``TWAccidentResults``.

        ``_fill_result_fields_from_runs`` (in :class:`RunHistoryMixin`)
        inserts two extra columns per selected run before the View column.
        Selecting nothing or selecting different runs has to put the table
        back to its 3-column base layout (Accident type / Probability /
        View) before a fresh fill -- this method does that.
        """
        from qgis.PyQt import QtWidgets

        tw = getattr(self.main_widget, 'TWAccidentResults', None)
        if tw is None:
            return
        # Base layout has exactly three columns; anything past that came
        # from a previous-runs comparison fill.  Drop them right-to-left
        # so column indices stay stable.
        while tw.columnCount() > 3:
            tw.removeColumn(tw.columnCount() - 2)
        # Reset headers + the View column header to the canonical labels
        # so the comparison-fill labels don't bleed across selections.
        tw.setHorizontalHeaderLabels(['Accident type', 'Probability', 'View'])

    def _populate_catastrophe_results_table(self, consequence_result) -> None:
        """Populate ``TWCatastropheResults`` from a ``consequence_result``
        dict produced by :func:`compute.consequence.compute_catastrophe_exceedance`.

        Rows are written in the same (ascending volume) order returned by
        the calculation.  ``None`` or missing levels clear the table.
        """
        from qgis.PyQt import QtWidgets

        tw = getattr(self.main_widget, 'TWCatastropheResults', None)
        if tw is None:
            return
        levels: list = []
        if isinstance(consequence_result, dict):
            levels = list(consequence_result.get('levels', []) or [])
        tw.setRowCount(len(levels))
        for r, lvl in enumerate(levels):
            try:
                name = str(lvl.get('name', ''))
                qty = float(lvl.get('quantity', 0.0))
                exceed = float(lvl.get('exceedance', 0.0))
            except Exception:
                continue
            tw.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            tw.setItem(r, 1, QtWidgets.QTableWidgetItem(f'{qty:.2f}'))
            tw.setItem(r, 2, QtWidgets.QTableWidgetItem(f'{exceed:.3e}'))

    def _ensure_legacy_lep_widgets(self) -> None:
        """Create the ``LEP*`` / ``pbView*`` widgets as hidden children
        of ``main_widget`` if they don't already exist.

        They used to live in the .ui file; we keep them programmatically
        so external callers (compute models, tests) that do
        ``LEPDriftAllision.setText`` keep working.
        """
        from qgis.PyQt import QtWidgets
        for _label, le_name, pb_name, _slot_name in self._ACCIDENT_ROWS:
            le = getattr(self.main_widget, le_name, None)
            if le is None:
                le = QtWidgets.QLineEdit(self.main_widget)
                le.setVisible(False)
                setattr(self.main_widget, le_name, le)
            pb = getattr(self.main_widget, pb_name, None)
            if pb is None:
                pb = QtWidgets.QPushButton(self.main_widget)
                pb.setVisible(False)
                setattr(self.main_widget, pb_name, pb)

    @staticmethod
    def _configure_accident_table(tw) -> None:
        from qgis.PyQt import QtWidgets
        from qgis.PyQt.QtWidgets import QHeaderView
        AIV = QtWidgets.QAbstractItemView

        headers = ['Accident type', 'Probability', 'View']
        tw.setColumnCount(len(headers))
        tw.setHorizontalHeaderLabels(headers)
        tw.setRowCount(len(AccidentResultsMixin._ACCIDENT_ROWS))
        tw.verticalHeader().setVisible(False)
        tw.setEditTriggers(_qt_enum(
            AIV, 'NoEditTriggers', 'EditTrigger.NoEditTriggers',
        ))
        tw.setSelectionBehavior(_qt_enum(
            AIV, 'SelectRows', 'SelectionBehavior.SelectRows',
        ))
        tw.setSelectionMode(_qt_enum(
            AIV, 'ExtendedSelection', 'SelectionMode.ExtendedSelection',
        ))
        tw.horizontalHeader().setStretchLastSection(False)
        try:
            mode_stretch = _qt_enum(
                QHeaderView, 'Stretch', 'ResizeMode.Stretch',
            )
            mode_resize = _qt_enum(
                QHeaderView,
                'ResizeToContents', 'ResizeMode.ResizeToContents',
            )
            tw.horizontalHeader().setSectionResizeMode(0, mode_stretch)
            tw.horizontalHeader().setSectionResizeMode(1, mode_resize)
            tw.horizontalHeader().setSectionResizeMode(2, mode_resize)
        except Exception:
            pass

    def _populate_accident_rows(self, tw) -> None:
        from qgis.PyQt import QtWidgets
        for row, (label, le_name, _pb_name, slot_name) in enumerate(
            self._ACCIDENT_ROWS,
        ):
            tw.setItem(row, 0, QtWidgets.QTableWidgetItem(label))
            le = getattr(self.main_widget, le_name, None)
            text = le.text() if le is not None else ''
            tw.setItem(row, 1, QtWidgets.QTableWidgetItem(text))

            btn = QtWidgets.QPushButton(self.tr('View'))
            slot = getattr(self, slot_name, None)
            if callable(slot):
                btn.clicked.connect(slot)
            else:
                btn.setEnabled(False)
            tw.setCellWidget(row, 2, btn)

            if le is not None:
                try:
                    le.textChanged.connect(
                        lambda txt, r=row: self._on_lep_text_changed(r, txt),
                    )
                except Exception:
                    pass

    def _wire_clipboard_copy_shortcut(self, tw) -> None:
        try:
            from qgis.PyQt.QtGui import QKeySequence, QShortcut
            for ks in (
                QKeySequence.StandardKey.Copy,
                QKeySequence('Ctrl+Insert'),
            ):
                sc = QShortcut(ks, tw)
                sc.activated.connect(
                    lambda t=tw: self._copy_table_selection_to_clipboard(t),
                )
        except Exception:
            pass

    @staticmethod
    def _copy_table_selection_to_clipboard(tw) -> None:
        """Copy the currently-selected cells of ``tw`` to the clipboard
        as tab-separated values.

        Whole-row selection yields one TSV line per selected row;
        individual cell selection yields the bounding rectangle with
        empty cells where there's no selection.  Cell widgets (the View
        buttons in column 2) are emitted as empty strings.
        """
        try:
            from qgis.PyQt.QtWidgets import QApplication
        except Exception:
            return
        ranges = tw.selectedRanges()
        if not ranges:
            return
        rows = sorted({
            r for rng in ranges
            for r in range(rng.topRow(), rng.bottomRow() + 1)
        })
        cols = sorted({
            c for rng in ranges
            for c in range(rng.leftColumn(), rng.rightColumn() + 1)
        })
        if not rows or not cols:
            return
        lines: list[str] = []
        for r in rows:
            cells: list[str] = []
            for c in cols:
                if tw.cellWidget(r, c) is not None:
                    cells.append('')
                    continue
                item = tw.item(r, c)
                cells.append(item.text() if item is not None else '')
            lines.append('\t'.join(cells))
        QApplication.clipboard().setText('\n'.join(lines))

    def _on_lep_text_changed(self, row: int, text: str) -> None:
        """Forward edits on a hidden LEP* into the new accident table."""
        try:
            from qgis.PyQt.QtWidgets import QTableWidgetItem
            tw = self.main_widget.TWAccidentResults
            tw.setItem(row, 1, QTableWidgetItem(text))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # View dispatch
    # ------------------------------------------------------------------
    def _require_single_selected_run(self):
        """Return the single selected ``RunMeta`` or surface a popup.

        Returns ``None`` (after showing a popup) if 0 or >1 rows are
        selected in ``TWPreviousRuns``.
        """
        try:
            run_ids = self._selected_run_ids()
        except Exception:
            run_ids = []
        if len(run_ids) == 0:
            QMessageBox.information(
                self.main_widget,
                self.tr('Select a run'),
                self.tr(
                    "Pick a run in the Previous-runs table at the top of "
                    "the Run Analysis tab before clicking View."
                ),
            )
            return None
        if len(run_ids) > 1:
            QMessageBox.information(
                self.main_widget,
                self.tr('Select only one run'),
                self.tr(
                    "View shows the breakdown of a single run.  Select "
                    "exactly one row in the Previous-runs table."
                ),
            )
            return None
        try:
            from omrat_utils.run_history import RunHistory
            runs = RunHistory().compare_runs(run_ids)
        except Exception as exc:
            self.show_error_popup(str(exc), '_require_single_selected_run')
            return None
        return runs[0] if runs else None

    def _load_run_inputs_and_collision_report(
        self, run,
    ) -> tuple[dict | None, dict | None, dict | None]:
        """Load ``data`` (.omrat), collision_report, drifting_report
        from disk for ``run``.  Any may be ``None`` when the file
        isn't present (older runs that pre-date the sidecar code).
        """
        data = None
        collision_report = None
        drifting_report = None
        try:
            gpkg_path = (
                run.gpkg_path() if hasattr(run, 'gpkg_path') else None
            )
        except Exception:
            gpkg_path = None
        if gpkg_path is None:
            return data, collision_report, drifting_report
        stem = Path(gpkg_path).with_suffix('')
        for path, label, target in (
            (stem.with_suffix('.omrat'), 'data', 'data'),
            (Path(str(stem) + '.collision.json'), 'collision', 'cr'),
            (Path(str(stem) + '.drifting.json'), 'drifting', 'dr'),
        ):
            if not path.is_file():
                continue
            try:
                with path.open('r', encoding='utf-8') as f:
                    payload = json.load(f)
            except Exception as exc:
                QgsMessageLog.logMessage(
                    f"Could not read {path}: {exc}",
                    'OMRAT', Qgis.Warning,
                )
                continue
            if target == 'data':
                data = payload
            elif target == 'cr':
                collision_report = payload
            elif target == 'dr':
                drifting_report = payload
        return data, collision_report, drifting_report

    def _dispatch_view(self, slot_name: str) -> None:
        spec = self._VIEW_DISPATCH.get(slot_name)
        if spec is None:
            return
        method_name, _label, breakdown_key = spec

        run = self._require_single_selected_run()
        if run is None:
            return
        data, collision_report, drifting_report = (
            self._load_run_inputs_and_collision_report(run)
        )
        if data is None:
            QMessageBox.information(
                self.main_widget,
                self.tr('Snapshot missing'),
                self.tr(
                    "No .omrat snapshot was found next to this run's "
                    "GeoPackage, so the interactive visualiser cannot "
                    "rebuild its inputs.  Re-run the model to produce "
                    "a snapshot, or pick a newer run."
                ),
            )
            return
        if self.calc is None:
            return
        method = getattr(self.calc, method_name, None)
        if not callable(method):
            return
        if method_name == 'run_collision_breakdown_dialog':
            self._invoke_collision_breakdown(
                method, breakdown_key, collision_report,
            )
            return
        self._invoke_drift_or_powered_visualiser(
            method, data, drifting_report,
        )

    def _invoke_collision_breakdown(
        self, method, breakdown_key, collision_report,
    ) -> None:
        previous = getattr(self.calc, 'collision_report', None)
        try:
            self.calc.collision_report = collision_report or {}
            method(breakdown_key)
        finally:
            self.calc.collision_report = previous

    def _invoke_drift_or_powered_visualiser(
        self, method, data, drifting_report,
    ) -> None:
        prev_dr = getattr(self.calc, 'drifting_report', None)
        try:
            if drifting_report is not None:
                self.calc.drifting_report = drifting_report
            method(data)
        finally:
            self.calc.drifting_report = prev_dr

    # ------------------------------------------------------------------
    # Public View slots (one per accident type)
    # ------------------------------------------------------------------
    def show_drift_allision(self):
        self._dispatch_view('show_drift_allision')

    def show_drift_grounding(self):
        self._dispatch_view('show_drift_grounding')

    def show_powered_allision(self):
        self._dispatch_view('show_powered_allision')

    def show_powered_grounding(self):
        self._dispatch_view('show_powered_grounding')

    def show_overtaking_collision(self):
        self._dispatch_view('show_overtaking_collision')

    def show_head_on_collision(self):
        self._dispatch_view('show_head_on_collision')

    def show_crossing_collision(self):
        self._dispatch_view('show_crossing_collision')

    def show_merging_collision(self):
        self._dispatch_view('show_merging_collision')
