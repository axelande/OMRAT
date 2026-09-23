"""**Settings > Ship type mapping...** dialog.

Owns the UI side of the custom IMO / MMSI -> ship type mapping: enable
flag, schema / table, CSV import and export, and a read-only preview of
the table.  The database connection is the one the AIS handler holds
(``omrat.ais.db``), so the AIS connection has to be configured first.
All SQL and file handling lives in :mod:`omrat_utils.ship_type_map`;
this module only moves values between the widgets and those helpers,
which keeps it testable with a mocked widget.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem

from omrat_utils.ship_type_map import (
    DEFAULT_TABLE, SHIP_TYPE_NAMES, MappingRow, ShipTypeMapConfig, count_mapping,
    fetch_mapping_rows, import_mapping, read_mapping_csv, write_mapping_csv,
)

if TYPE_CHECKING:
    from omrat import OMRAT

PREVIEW_LIMIT = 1000
_MAX_ERRORS_SHOWN = 20


def _default_widget_factory():
    from ui.ship_type_map_widget import ShipTypeMapWidget
    return ShipTypeMapWidget(None)


class ShipTypeMapping:
    """Controller for the ship type mapping dialog."""

    def __init__(self, omrat: "OMRAT", widget_factory: Callable[[], Any] | None = None):
        self.omrat = omrat
        self.dlg = (widget_factory or _default_widget_factory)()
        self.dlg.pbImport.clicked.connect(self.import_csv)
        self.dlg.pbExport.clicked.connect(self.export_csv)
        self.dlg.pbRefresh.clicked.connect(self.refresh)
        self.dlg.accepted.connect(self.apply)

    # ------------------------------------------------------------ helpers

    @staticmethod
    def tr(text: str) -> str:
        return QCoreApplication.translate("ShipTypeMapping", text)

    @property
    def ais(self):
        return self.omrat.ais

    def _db(self):
        db = getattr(self.ais, "db", None)
        if db is None:
            QMessageBox.warning(
                self.dlg, self.tr("Ship type mapping"),
                self.tr("No AIS database connection. Configure Settings > AIS connection settings first."),
            )
        return db

    def _config_from_ui(self) -> ShipTypeMapConfig:
        return ShipTypeMapConfig(
            enabled=bool(self.dlg.cbEnabled.isChecked()),
            schema=self.dlg.leSchema.text().strip(),
            table=self.dlg.leTable.text().strip() or DEFAULT_TABLE,
        )

    def _table_config(self) -> ShipTypeMapConfig | None:
        """Config for reading / writing the table, regardless of the enable box."""
        cfg = self._config_from_ui()
        cfg.enabled = True
        if not cfg.is_valid():
            QMessageBox.warning(
                self.dlg, self.tr("Ship type mapping"),
                self.tr("Enter the schema (and table) the mapping table lives in. "
                        "Both must be plain SQL names (letters, digits, underscore)."),
            )
            return None
        return cfg

    # ------------------------------------------------------------ lifecycle

    def run(self) -> None:
        cfg = self.ais.type_map
        self.dlg.cbEnabled.setChecked(bool(cfg.enabled))
        self.dlg.leSchema.setText(cfg.schema)
        self.dlg.leTable.setText(cfg.table or DEFAULT_TABLE)
        self.refresh(quiet=True)
        self.dlg.exec()

    def apply(self) -> ShipTypeMapConfig:
        """Save button: store the config on the AIS handler and in QSettings."""
        cfg = self._config_from_ui()
        self.ais.set_type_map(cfg)
        return cfg

    # ------------------------------------------------------------ preview

    def refresh(self, *_args, quiet: bool = False) -> None:
        """Fill the status line and preview table from the database."""
        cfg = self._config_from_ui()
        cfg.enabled = True
        db = getattr(self.ais, "db", None)
        self.dlg.twPreview.setRowCount(0)
        if db is None:
            self.dlg.lblStatus.setText(self.tr("No AIS database connection (see Settings > AIS connection settings)."))
            return
        if not cfg.is_valid():
            if not quiet or cfg.schema:
                self.dlg.lblStatus.setText(self.tr("Schema and table must be plain SQL names."))
            else:
                self.dlg.lblStatus.setText(self.tr("Enter a schema and import a CSV to create the table."))
            return
        counts = count_mapping(db, cfg)
        if counts is None:
            self.dlg.lblStatus.setText(
                self.tr("Table {0} does not exist yet -- Import CSV... creates it.").format(cfg.qualified_name())
            )
            return
        total, n_imo, n_mmsi = counts
        rows = fetch_mapping_rows(db, cfg, limit=PREVIEW_LIMIT) or []
        shown = self.tr(" (first {0} shown)").format(PREVIEW_LIMIT) if total > len(rows) else ""
        self.dlg.lblStatus.setText(
            self.tr("{0}: {1} rows, {2} with IMO, {3} with MMSI{4}").format(
                cfg.qualified_name(), total, n_imo, n_mmsi, shown)
        )
        self._fill_preview(rows)

    def _fill_preview(self, rows: list[MappingRow]) -> None:
        tw = self.dlg.twPreview
        tw.setRowCount(len(rows))
        for r, row in enumerate(rows):
            name = SHIP_TYPE_NAMES[row.ship_type] if 0 <= row.ship_type < len(SHIP_TYPE_NAMES) else "?"
            cells = (
                "" if row.mmsi is None else str(row.mmsi),
                "" if row.imo is None else str(row.imo),
                f"{row.ship_type}: {name}",
                row.note or "",
            )
            for c, text in enumerate(cells):
                tw.setItem(r, c, QTableWidgetItem(text))
        tw.resizeColumnsToContents()

    # ------------------------------------------------------------ import / export

    def import_csv(self, *_args) -> int | None:
        """Load a CSV into the table (created when missing, rows replaced)."""
        cfg = self._table_config()
        if cfg is None:
            return None
        db = self._db()
        if db is None:
            return None
        path, _ = QFileDialog.getOpenFileName(
            self.dlg, self.tr("Ship type mapping CSV"), "",
            self.tr("CSV files (*.csv *.txt);;All files (*)"),
        )
        if not path:
            return None
        try:
            rows, errors = read_mapping_csv(path)
        except Exception as exc:
            QMessageBox.critical(self.dlg, self.tr("Ship type mapping"), str(exc))
            return None
        if not rows:
            QMessageBox.critical(
                self.dlg, self.tr("Ship type mapping"),
                self.tr("Nothing could be imported:\n") + "\n".join(errors[:_MAX_ERRORS_SHOWN]),
            )
            return None
        if errors:
            shown = "\n".join(errors[:_MAX_ERRORS_SHOWN])
            if len(errors) > _MAX_ERRORS_SHOWN:
                shown += self.tr("\n... and {0} more").format(len(errors) - _MAX_ERRORS_SHOWN)
            answer = QMessageBox.question(
                self.dlg, self.tr("Ship type mapping"),
                self.tr("{0} rows could not be read and will be skipped:\n\n{1}\n\n"
                        "Replace the table with the remaining {2} rows?").format(len(errors), shown, len(rows)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return None
        try:
            written = import_mapping(db, cfg, rows, replace=True)
        except Exception as exc:
            QMessageBox.critical(
                self.dlg, self.tr("Ship type mapping"),
                self.tr("Could not write {0}:\n{1}").format(cfg.qualified_name(), exc),
            )
            return None
        QgsMessageLog.logMessage(
            f"Ship type mapping: wrote {written} rows to {cfg.qualified_name()} from {path}",
            "OMRAT", Qgis.MessageLevel.Info,
        )
        self.dlg.cbEnabled.setChecked(True)
        self.refresh()
        return written

    def export_csv(self, *_args) -> int | None:
        """Write the whole table to a CSV the user can edit and re-import."""
        cfg = self._table_config()
        if cfg is None:
            return None
        db = self._db()
        if db is None:
            return None
        rows = fetch_mapping_rows(db, cfg, limit=None)
        if rows is None:
            QMessageBox.warning(
                self.dlg, self.tr("Ship type mapping"),
                self.tr("Table {0} could not be read.").format(cfg.qualified_name()),
            )
            return None
        path, _ = QFileDialog.getSaveFileName(
            self.dlg, self.tr("Export ship type mapping"), f"{cfg.table}.csv",
            self.tr("CSV files (*.csv);;All files (*)"),
        )
        if not path:
            return None
        try:
            write_mapping_csv(path, rows)
        except Exception as exc:
            QMessageBox.critical(self.dlg, self.tr("Ship type mapping"), str(exc))
            return None
        self.dlg.lblStatus.setText(self.tr("Wrote {0} rows to {1}").format(len(rows), path))
        return len(rows)
