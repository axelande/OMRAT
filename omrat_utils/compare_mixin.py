"""Compare-Models tab slots, factored out of ``omrat.OMRAT``.

The Compare tab lets the user load two ``.omrat`` snapshots side-by-side
and renders its diff tables on five sub-tabs (``twCompareResults``):

* **Totals** -- accidents, summary rows and oil spill
  (:func:`omrat_utils.compare.build_accident_table`),
* **Settings & inputs** -- parameters (``build_settings_table``) and
  inputs (:mod:`omrat_utils.compare_inputs`),
* **Breakdown** -- per leg / waypoint / object changes
  (:mod:`omrat_utils.compare_breakdown`) with a category filter,
* **Zone** -- the common area around the structures
  (:mod:`omrat_utils.compare_zone`), re-computed when the radius changes,
* **Leg lengths** -- ``build_leg_distance_table``.

**More info (collisions)...** under the Totals table opens
:class:`omrat_utils.compare_explain_dialog.CompareExplainDialog`, which
explains *why* each ship-ship collision type changed
(:mod:`omrat_utils.compare_explain`).  The explanation is cached per
snapshot pair (path + mtime) in ``_compare_explain_cache``.

The pure-data work lives in those modules; this mixin only owns the
**dock-widget side**: the file pickers, the Compare button, the two
"add to QGIS" buttons (one layer group per model, A red / B blue, and
the B - A difference layers).

The mixin is composed onto :class:`omrat.OMRAT`.  Public method names
(``_setup_compare_tab``, ``_run_compare`` and friends) are unchanged so
existing wiring keeps working.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem

_ALL_CATEGORIES = 'All categories'


class _NumericItem(QTableWidgetItem):
    """Table item that shows formatted text but sorts on the number."""

    def __init__(self, value: float | None, text: str) -> None:
        super().__init__(text)
        self._value = value

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, _NumericItem):
            a = self._value if self._value is not None else float('-inf')
            b = other._value if other._value is not None else float('-inf')
            return a < b
        return super().__lt__(other)


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

        if hasattr(widget, 'pbCompareDiff'):
            widget.pbCompareDiff.clicked.connect(self._add_compare_diff_layers)
        if hasattr(widget, 'pbCompareMoreInfo'):
            widget.pbCompareMoreInfo.clicked.connect(self._open_compare_explain)
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
        if hasattr(widget, 'TWCompareInputs'):
            widget.TWCompareInputs.setColumnCount(4)
            widget.TWCompareInputs.setHorizontalHeaderLabels(['Group', 'Input', 'Run A', 'Run B'])
        if hasattr(widget, 'TWCompareBreakdown'):
            from omrat_utils.compare_breakdown import CATEGORIES
            widget.TWCompareBreakdown.setColumnCount(6)
            widget.TWCompareBreakdown.setHorizontalHeaderLabels(
                ['Category', 'Item', 'Run A', 'Run B', 'Δ (B - A)', 'Share of |Δ| (%)'],
            )
            cb = widget.cbCompareBreakdownFilter
            cb.clear()
            cb.addItems([_ALL_CATEGORIES, *CATEGORIES])
            cb.currentIndexChanged.connect(self._filter_compare_breakdown)
        if hasattr(widget, 'TWCompareZone'):
            widget.TWCompareZone.setColumnCount(5)
            widget.TWCompareZone.setHorizontalHeaderLabels(
                ['Zone figure', 'Run A', 'Run B', 'Δ abs', 'Δ %'],
            )
            widget.sbCompareZoneKm.valueChanged.connect(self._refresh_compare_zone)
        self._compare_cache: dict[str, Any] | None = None
        self._compare_explain_cache: dict[tuple, Any] = {}
        self._compare_explain_dlg = None

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
            load_run_outputs,
            load_snapshot,
        )
        from omrat_utils.compare_breakdown import build_breakdown_rows, top_movers
        from omrat_utils.compare_inputs import build_input_table, describe
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
        out_a = load_run_outputs(path_a)
        out_b = load_run_outputs(path_b)

        accident_rows = build_accident_table(Path(path_a), Path(path_b), out_a, out_b)
        settings_rows = build_settings_table(snap_a, snap_b)
        leg_rows = build_leg_distance_table(snap_a, snap_b)
        self._fill_compare_table(widget.TWCompareAccidents, accident_rows)
        self._fill_compare_table(widget.TWCompareSettings, settings_rows)
        self._fill_compare_table(widget.TWCompareLegs, leg_rows)

        input_rows: list[list[str]] = []
        breakdown_rows: list[list[Any]] = []
        if hasattr(widget, 'TWCompareInputs'):
            input_rows = build_input_table(snap_a, snap_b)
            self._fill_compare_table(widget.TWCompareInputs, input_rows)
        if hasattr(widget, 'TWCompareBreakdown'):
            breakdown_rows = build_breakdown_rows(snap_a, snap_b, out_a, out_b)
            self._fill_breakdown_table(widget.TWCompareBreakdown, breakdown_rows)
            self._filter_compare_breakdown()
        self._compare_cache = {
            'snap_a': snap_a, 'snap_b': snap_b, 'out_a': out_a, 'out_b': out_b,
        }
        self._refresh_compare_zone()

        def _files(out: dict) -> str:
            return ', '.join(sorted(out)) or self.tr('none')

        widget.lblCompareSummary.setText(self.tr('Run A: {a}   |   Run B: {b}').format(
            a=Path(path_a).name, b=Path(path_b).name))
        lines = [
            self.tr('Result files: A {fa}; B {fb}.').format(fa=_files(out_a), fb=_files(out_b)),
            self.tr('Accident rows: {na} | Settings differences: {ns} | Input differences: {ni} '
                    '({gi}) | Breakdown items: {nb} | Legs: {nl}').format(
                na=len(accident_rows), ns=len(settings_rows), ni=len(input_rows),
                gi=describe(input_rows), nb=len(breakdown_rows), nl=len(leg_rows)),
        ]
        top = top_movers(breakdown_rows, 1)
        if top:
            lines.append(self.tr('Largest single change: {item} ({delta:+.3e} per year)').format(
                item=top[0][1], delta=top[0][4]))
        if hasattr(widget, 'lblCompareDetails'):
            widget.lblCompareDetails.setText('\n'.join(lines))

    @staticmethod
    def _fill_compare_table(tw, rows: list[list[str]]) -> None:
        tw.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                tw.setItem(r, c, QTableWidgetItem(val))
        tw.resizeColumnsToContents()
        tw.horizontalHeader().setStretchLastSection(True)

    @staticmethod
    def _fill_breakdown_table(tw, rows: list[list[Any]]) -> None:
        """``[category, item, a, b, delta, share]`` rows; numbers sort numerically."""
        tw.setSortingEnabled(False)
        tw.setRowCount(len(rows))
        for r, (category, item, a, b, delta, share) in enumerate(rows):
            tw.setItem(r, 0, QTableWidgetItem(category))
            tw.setItem(r, 1, QTableWidgetItem(item))
            tw.setItem(r, 2, _NumericItem(a, '—' if a is None else f'{a:.3e}'))
            tw.setItem(r, 3, _NumericItem(b, '—' if b is None else f'{b:.3e}'))
            tw.setItem(r, 4, _NumericItem(delta, f'{delta:+.3e}'))
            tw.setItem(r, 5, _NumericItem(share, f'{share:.1f}'))
        tw.setSortingEnabled(True)
        tw.resizeColumnsToContents()
        tw.horizontalHeader().setStretchLastSection(True)

    def _filter_compare_breakdown(self, *_args) -> None:
        """Show only the rows of the category picked in ``cbCompareBreakdownFilter``."""
        widget = self.main_widget
        if not hasattr(widget, 'TWCompareBreakdown'):
            return
        tw = widget.TWCompareBreakdown
        wanted = widget.cbCompareBreakdownFilter.currentText()
        for r in range(tw.rowCount()):
            item = tw.item(r, 0)
            category = item.text() if item is not None else ''
            tw.setRowHidden(r, wanted not in ('', _ALL_CATEGORIES) and category != wanted)

    def _refresh_compare_zone(self, *_args) -> None:
        """(Re)build the Zone table from the last Compare, with the current radius."""
        from omrat_utils.compare_zone import build_zone_table
        widget = self.main_widget
        cache = getattr(self, '_compare_cache', None)
        if not cache or not hasattr(widget, 'TWCompareZone'):
            return
        try:
            rows, info = build_zone_table(
                cache['snap_a'], cache['snap_b'], cache['out_a'], cache['out_b'],
                radius_km=float(widget.sbCompareZoneKm.value()),
            )
        except Exception as exc:
            rows, info = [], self.tr('Zone comparison failed: {e}').format(e=exc)
            QgsMessageLog.logMessage(f'Compare zone failed: {exc}', 'OMRAT', Qgis.MessageLevel.Warning)
        self._fill_compare_table(widget.TWCompareZone, rows)
        widget.lblCompareZoneInfo.setText(info)

    # ------------------------------------------------------------------
    # More info: why did the collisions change?
    # ------------------------------------------------------------------
    @staticmethod
    def _compare_explain_key(path_a: str, path_b: str) -> tuple:
        """Cache key: both snapshots' resolved path and modification time."""
        key = []
        for path in (path_a, path_b):
            p = Path(path)
            try:
                key.append((str(p.resolve()), p.stat().st_mtime))
            except OSError:
                key.append((str(p), None))
        return tuple(key)

    def _remember_compare_explanation(self, key: tuple, explanation: Any) -> None:
        cache = getattr(self, '_compare_explain_cache', None)
        if cache is None:
            cache = self._compare_explain_cache = {}
        cache[key] = explanation
        while len(cache) > 4:
            cache.pop(next(iter(cache)))

    def _open_compare_explain(self) -> None:
        """Open (or raise) *More info*: why did each collision type change?

        The accident row selected in the Totals table picks the type shown
        first; any non-collision row shows all collisions.
        """
        from omrat_utils.compare import load_run_outputs, load_snapshot
        from omrat_utils.compare_explain import TOTAL, type_for_label
        from omrat_utils.compare_explain_dialog import CompareExplainDialog
        widget = self.main_widget
        path_a = (widget.LECompareA.text() or '').strip()
        path_b = (widget.LECompareB.text() or '').strip()
        if not path_a or not path_b:
            QMessageBox.information(
                widget, self.tr('More info'),
                self.tr('Pick two .omrat snapshots before clicking More info.'))
            return
        acc_type = TOTAL
        tw = widget.TWCompareAccidents
        item = tw.item(tw.currentRow(), 0) if tw.currentRow() >= 0 else None
        if item is not None:
            acc_type = type_for_label(item.text())
        key = self._compare_explain_key(path_a, path_b)
        dlg = getattr(self, '_compare_explain_dlg', None)
        if dlg is not None:
            try:
                if dlg.isVisible() and getattr(dlg, 'cache_key', None) == key:
                    idx = dlg.cbType.findData(acc_type)
                    if idx >= 0:
                        dlg.cbType.setCurrentIndex(idx)
                    dlg.raise_()
                    dlg.activateWindow()
                    return
                dlg.close()
            except RuntimeError:
                pass  # underlying C++ object already deleted
            self._compare_explain_dlg = None
        try:
            snap_a = load_snapshot(path_a)
            snap_b = load_snapshot(path_b)
        except Exception as exc:
            self.show_error_popup(str(exc), '_open_compare_explain')
            return
        iface = getattr(self, 'iface', None)
        canvas = iface.mapCanvas() if iface is not None else None
        dlg = CompareExplainDialog(
            widget, snap_a, snap_b, Path(path_a).name, Path(path_b).name,
            load_run_outputs(path_a), load_run_outputs(path_b),
            acc_type=acc_type,
            explanation=(getattr(self, '_compare_explain_cache', None) or {}).get(key),
            canvas=canvas,
            on_ready=lambda expl, key=key: self._remember_compare_explanation(key, expl),
        )
        dlg.cache_key = key
        self._compare_explain_dlg = dlg
        dlg.show()

    # ------------------------------------------------------------------
    # Add both runs as map layers
    # ------------------------------------------------------------------
    def _add_compare_layers(self) -> None:
        """Add both models to QGIS, each in its own layer group.

        Per model the group holds (bottom to top) depth areas,
        structures, legs, tangent lines and -- when ``<stem>.gpkg`` sits
        next to the ``.omrat`` file -- the result layers.  Run A is
        tinted red, run B blue.
        """
        from omrat_utils.compare import load_snapshot
        from omrat_utils.snapshot_layers import add_snapshot_to_project
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
            if not omrat_path.is_file():
                report.append(f"Run {label}: file not found: {omrat_path}")
                continue
            try:
                snapshot = load_snapshot(omrat_path)
            except Exception as exc:
                report.append(f"Run {label}: could not read {omrat_path.name}: {exc}")
                continue
            gpkg = omrat_path.with_suffix('.gpkg')
            try:
                built = add_snapshot_to_project(
                    snapshot,
                    f"Compare {label}: {omrat_path.stem}",
                    gpkg_path=gpkg if gpkg.is_file() else None,
                    color=color,
                    run_label=f'compare_{label}',
                )
            except Exception as exc:
                self.show_error_popup(str(exc), '_add_compare_layers')
                return
            for lyr in built['results']:
                self._tint_layer(lyr, color)
            self._remember_compare_group(built)
            n_model = len(built['model'])
            n_res = len(built['results'])
            loaded_total += n_model + n_res
            if gpkg.is_file():
                report.append(
                    f"Run {label}: {n_model} model layer(s) + {n_res} result "
                    f"layer(s) from {gpkg.name}"
                )
            else:
                report.append(
                    f"Run {label}: {n_model} model layer(s); no GeoPackage "
                    f"({gpkg.name}) so result layers were skipped"
                )

        msg_body = "\n".join(report) if report else "Nothing was attempted."
        if loaded_total == 0:
            QMessageBox.information(
                self.main_widget,
                self.tr('Compare: no layers loaded'),
                self.tr(
                    "{body}\n\n"
                    "Pick the .omrat snapshot files for run A and run B "
                    "(use the ... buttons), then try again."
                ).format(body=msg_body),
            )
        else:
            QgsMessageLog.logMessage(
                f"Compare layers added ({loaded_total} total):\n{msg_body}",
                'OMRAT', Qgis.MessageLevel.Success,
            )

    # ------------------------------------------------------------------
    # Difference layers (B - A)
    # ------------------------------------------------------------------
    def _compare_gpkg_paths(self) -> tuple[Path | None, Path | None, list[str]]:
        """``<stem>.gpkg`` next to each picked snapshot, plus problems found."""
        widget = self.main_widget
        problems: list[str] = []
        out: list[Path | None] = []
        for label, le in (('A', widget.LECompareA), ('B', widget.LECompareB)):
            text = (le.text() or '').strip()
            if not text:
                problems.append(f"Run {label}: empty path.")
                out.append(None)
                continue
            gpkg = Path(text).with_suffix('.gpkg')
            if not gpkg.is_file():
                problems.append(f"Run {label}: no GeoPackage next to the snapshot ({gpkg.name}).")
                out.append(None)
                continue
            out.append(gpkg)
        return out[0], out[1], problems

    def _add_compare_diff_layers(self) -> None:
        """Add one *difference* layer (B - A) per result layer to QGIS.

        Features are joined on their keys (obstacle edge, leg or
        waypoint); ``delta`` is the absolute change in events/year and the
        layers are styled on it (blue = decrease, red = increase).  See
        ``omrat_utils.result_diff`` for why ``pct_change`` is not used for
        the style.
        """
        from omrat_utils.result_diff import add_diff_group, build_diff_layers, summary_lines

        gpkg_a, gpkg_b, problems = self._compare_gpkg_paths()
        if gpkg_a is None or gpkg_b is None:
            QMessageBox.information(
                self.main_widget, self.tr('Compare: difference layers'),
                self.tr(
                    "{body}\n\nBoth runs need their result GeoPackage next to the "
                    ".omrat snapshot (they are written together by Run model)."
                ).format(body="\n".join(problems)),
            )
            return
        label = f'{gpkg_b.stem} - {gpkg_a.stem}'
        try:
            layers, summaries = build_diff_layers(gpkg_a, gpkg_b, label='B - A')
        except Exception as exc:
            self.show_error_popup(str(exc), '_add_compare_diff_layers')
            return
        if not layers:
            QMessageBox.information(
                self.main_widget, self.tr('Compare: difference layers'),
                self.tr('Neither GeoPackage holds a result layer to compare.'),
            )
            return
        built = add_diff_group(layers, f'Compare diff: {label}')
        self._remember_compare_group(built)
        body = "\n".join(summary_lines(summaries))
        QgsMessageLog.logMessage(
            f"Difference layers added ({len(layers)}):\n{body}", 'OMRAT', Qgis.MessageLevel.Success)
        try:
            self.main_widget.lblCompareSummary.setText(
                f'Difference layers (B - A) added: {len(layers)}.  ' + '  '.join(summary_lines(summaries)))
        except Exception:  # nosec B110 B112
            pass

    def _remember_compare_group(self, built: dict) -> None:
        """Track the group + layers so ``clear_model`` removes them."""
        try:
            groups = getattr(self, '_compare_groups', None)
            if groups is None:
                groups = []
                self._compare_groups = groups
            groups.append(built['group'])
            history = getattr(self, '_history_layers', None)
            if history is not None:
                history.extend(built['model'])
                history.extend(built['results'])
        except Exception:  # nosec B110 B112
            pass

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
        except Exception:  # nosec B110 B112
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
        except Exception:  # nosec B110 B112
            pass
