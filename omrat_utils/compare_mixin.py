"""Compare-Models tab slots, factored out of ``omrat.OMRAT``.

The Compare tab lets the user load two ``.omrat`` snapshots side-by-side
and renders three diff tables (accidents, settings, leg distances).
The pure-data work lives in :mod:`omrat_utils.compare`; this mixin only
owns the **dock-widget side**:

* file-picker buttons that drive the ``LECompareA`` / ``LECompareB``
  line-edits,
* the Compare button that fills the three QTables, and
* the optional "Add both runs as map layers (red + blue)" button which
  loads each run's GeoPackage onto the canvas with a uniform tint.

The mixin is composed onto :class:`omrat.OMRAT`.  Public method names
(``_setup_compare_tab``, ``_run_compare`` and friends) are unchanged so
existing wiring keeps working.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox

if TYPE_CHECKING:
    pass


class CompareMixin:
    """All Compare-tab slot wiring + result-rendering."""

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _setup_compare_tab(self) -> None:
        """Wire the Compare tab's pickers + Compare button."""
        widget = self.main_widget
        if not hasattr(widget, 'pbCompareABrowse'):
            return  # Compare tab not present in this build of the .ui

        widget.pbCompareABrowse.clicked.connect(
            lambda: self._pick_compare_snapshot(widget.LECompareA),
        )
        widget.pbCompareBBrowse.clicked.connect(
            lambda: self._pick_compare_snapshot(widget.LECompareB),
        )
        widget.pbRunCompare.clicked.connect(self._run_compare)
        if hasattr(widget, 'pbAddCompareLayers'):
            widget.pbAddCompareLayers.clicked.connect(
                self._add_compare_layers,
            )

        widget.TWCompareAccidents.setColumnCount(5)
        widget.TWCompareAccidents.setHorizontalHeaderLabels(
            ['Accident type', 'Run A', 'Run B', 'Δ abs', 'Δ %'],
        )
        widget.TWCompareSettings.setColumnCount(3)
        widget.TWCompareSettings.setHorizontalHeaderLabels(
            ['Setting', 'Run A', 'Run B'],
        )
        widget.TWCompareLegs.setColumnCount(5)
        widget.TWCompareLegs.setHorizontalHeaderLabels(
            ['Leg', 'Run A length (m)', 'Run B length (m)', 'Δ m', 'Δ %'],
        )

    # ------------------------------------------------------------------
    # File picker
    # ------------------------------------------------------------------
    def _pick_compare_snapshot(self, line_edit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.main_widget,
            self.tr('Select .omrat snapshot'),
            line_edit.text() or '',
            'OMRAT snapshots (*.omrat *.OMRAT);;All files (*.*)',
        )
        if path:
            line_edit.setText(path)

    # ------------------------------------------------------------------
    # Compare button: render the three tables
    # ------------------------------------------------------------------
    def _run_compare(self) -> None:
        from omrat_utils.compare import (
            build_accident_table,
            build_leg_distance_table,
            build_settings_table,
            load_snapshot,
        )
        widget = self.main_widget
        path_a = (widget.LECompareA.text() or '').strip()
        path_b = (widget.LECompareB.text() or '').strip()
        if not path_a or not path_b:
            QMessageBox.information(
                widget,
                self.tr('Compare'),
                self.tr('Pick two .omrat snapshots before clicking Compare.'),
            )
            return
        try:
            snap_a = load_snapshot(path_a)
            snap_b = load_snapshot(path_b)
        except Exception as exc:
            self.show_error_popup(str(exc), '_run_compare')
            return

        accident_rows = build_accident_table(Path(path_a), Path(path_b))
        settings_rows = build_settings_table(snap_a, snap_b)
        leg_rows = build_leg_distance_table(snap_a, snap_b)

        self._fill_compare_table(widget.TWCompareAccidents, accident_rows)
        self._fill_compare_table(widget.TWCompareSettings, settings_rows)
        self._fill_compare_table(widget.TWCompareLegs, leg_rows)

        widget.lblCompareSummary.setText(
            self.tr(
                "Run A: {a}\nRun B: {b}\n"
                "Accident rows: {na} | Settings differences: {ns} | Legs: {nl}"
            ).format(
                a=Path(path_a).name, b=Path(path_b).name,
                na=len(accident_rows), ns=len(settings_rows), nl=len(leg_rows),
            )
        )

    @staticmethod
    def _fill_compare_table(tw, rows: list[list[str]]) -> None:
        from qgis.PyQt.QtWidgets import QTableWidgetItem
        tw.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                tw.setItem(r, c, QTableWidgetItem(val))

    # ------------------------------------------------------------------
    # Add both runs as map layers
    # ------------------------------------------------------------------
    def _add_compare_layers(self) -> None:
        """Load both run GeoPackages onto the canvas, A=red, B=blue."""
        from omrat_utils.run_persistence import load_run_results_to_map
        widget = self.main_widget
        report: list[str] = []
        loaded_total = 0
        for label, le, color in (
            ('A', widget.LECompareA, 'red'),
            ('B', widget.LECompareB, 'blue'),
        ):
            text = (le.text() or '').strip()
            if not text:
                report.append(f"Run {label}: empty path -- skipped.")
                continue
            omrat_path = Path(text)
            gpkg = omrat_path.with_suffix('.gpkg')
            if not gpkg.is_file():
                report.append(
                    f"Run {label}: GeoPackage not found at {gpkg}"
                )
                continue
            try:
                layers = load_run_results_to_map(gpkg, f'compare_{label}')
            except Exception as exc:
                self.show_error_popup(str(exc), '_add_compare_layers')
                return
            n = len(layers)
            loaded_total += n
            report.append(f"Run {label}: loaded {n} layer(s) from {gpkg.name}")
            for lyr in layers:
                self._tint_layer(lyr, color)

        msg_body = "\n".join(report) if report else "Nothing was attempted."
        if loaded_total == 0:
            QMessageBox.information(
                self.main_widget,
                self.tr('Compare: no layers loaded'),
                self.tr(
                    "{body}\n\n"
                    "Pick the .omrat snapshot files for run A and run B "
                    "(use the ... buttons), then try again.  The "
                    "GeoPackage with the same stem must exist next to "
                    "each .omrat snapshot."
                ).format(body=msg_body),
            )
        else:
            QgsMessageLog.logMessage(
                f"Compare layers added ({loaded_total} total):\n{msg_body}",
                'OMRAT', Qgis.Success,
            )

    @staticmethod
    def _tint_layer(layer, color_name: str) -> None:
        """Replace the layer's renderer with a single-symbol tint
        (red / blue) so run A and run B are visually distinct.

        Best-effort: keeps the graduated symbology if anything in the
        chain raises.
        """
        try:
            from qgis.core import (
                Qgis as _Qgis,
                QgsFillSymbol,
                QgsLineSymbol,
                QgsMarkerSymbol,
                QgsSingleSymbolRenderer,
            )
        except Exception:
            return
        try:
            geom_type = layer.geometryType()
            if geom_type == _Qgis.GeometryType.Line:
                sym = QgsLineSymbol.createSimple(
                    {'color': color_name, 'width': '0.6'},
                )
            elif geom_type == _Qgis.GeometryType.Polygon:
                sym = QgsFillSymbol.createSimple(
                    {
                        'color': color_name, 'style': 'no',
                        'outline_color': color_name, 'outline_width': '0.6',
                    },
                )
            else:
                sym = QgsMarkerSymbol.createSimple(
                    {'color': color_name, 'size': '2.5'},
                )
            layer.setRenderer(QgsSingleSymbolRenderer(sym))
            layer.triggerRepaint()
        except Exception:
            pass
