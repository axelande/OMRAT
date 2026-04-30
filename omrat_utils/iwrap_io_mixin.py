"""IWRAP XML import / export slots, factored out of ``omrat.OMRAT``.

OMRAT can interoperate with the IWRAP reference implementation via
its XML format; the conversion logic itself lives in
:mod:`compute.iwrap_convertion` (``write_iwrap_xml`` / ``parse_iwrap_xml``).
This mixin only owns the **dock-widget side** of the workflow:

* prompts the user for a target file path,
* gathers / populates the OMRAT data dict around the conversion call,
* surfaces success / failure as a Qt popup,
* triggers the schema validation + canvas refresh after import.

The mixin is composed onto :class:`omrat.OMRAT` so the public ``export_to_iwrap``
and ``import_from_iwrap`` slots stay reachable from the existing menu
wiring without changing call sites.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import TYPE_CHECKING

from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox

from compute.iwrap_convertion import parse_iwrap_xml, write_iwrap_xml
from omrat_utils.gather_data import GatherData

if TYPE_CHECKING:
    from omrat import OMRAT


class IwrapIOMixin:
    """File-dialog-driven IWRAP export / import slots."""

    # The mixin is glued onto ``OMRAT`` so static analysers know which
    # attributes to expect on ``self`` -- using a forward-reference
    # ``OMRAT`` alias keeps mypy happy without a runtime import cycle.
    if TYPE_CHECKING:
        # Re-declared for the type checker; ``OMRAT`` provides the
        # actual values at runtime.
        from compute.run_calculations import Calculation
        from qgis._gui import QgisInterface
        from qgis.core import QgsVectorLayer

        iface: QgisInterface
        main_widget: object
        traffic: object
        traffic_data: dict
        calc: Calculation | None

        def show_error_popup(self, message: str, function_name: str) -> None: ...
        def clear_model(self) -> None: ...
        def _confirm_before_load(self, action_label: str) -> str: ...
        def tr(self, message: str) -> str: ...

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_to_iwrap(self) -> None:
        """Export current project data to IWRAP XML format."""
        try:
            filename = self._ask_iwrap_export_path()
            if not filename:
                return
            self._do_iwrap_export(filename)
            QMessageBox.information(
                self.main_widget,
                self.tr('Export Successful'),
                self.tr(f'Project successfully exported to:\n{filename}'),
            )
        except Exception as exc:
            self.show_error_popup(str(exc), 'export_to_iwrap')

    def _ask_iwrap_export_path(self) -> str:
        filename, _ = QFileDialog.getSaveFileName(
            self.main_widget,
            self.tr('Export to IWRAP XML'),
            '',
            'IWRAP XML Files (*.xml);;All Files (*.*)',
        )
        if not filename:
            return ''
        if not filename.lower().endswith('.xml'):
            filename += '.xml'
        return filename

    def _do_iwrap_export(self, filename: str) -> None:
        gd = GatherData(self)
        data = gd.get_all_for_save()
        # The IWRAP writer expects a JSON file on disk -- stage to a
        # temp file, run the converter, clean up.
        temp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.omrat', delete=False, encoding='utf-8',
        )
        temp_path = temp.name
        try:
            json.dump(data, temp, indent=2)
            temp.close()
            write_iwrap_xml(temp_path, filename)
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------
    def import_from_iwrap(self) -> None:
        """Import project data from IWRAP XML format."""
        try:
            filename = self._ask_iwrap_import_path()
            if not filename:
                return
            choice = self._confirm_before_load('Import')
            if choice == 'cancel':
                return
            if choice == 'clear':
                self.clear_model()

            data = parse_iwrap_xml(filename, debug=True)
            data = self._validate_iwrap_payload(data)
            self._apply_iwrap_payload(data)

            QMessageBox.information(
                self.main_widget,
                self.tr('Import Successful'),
                self.tr(
                    f'Project successfully imported from:\n{filename}\n\n'
                    f'Project: {data.get("project_name", "Unknown")}\n'
                    f'Segments: {len(data.get("segment_data", {}))}'
                ),
            )
        except Exception as exc:
            self.show_error_popup(str(exc), 'import_from_iwrap')

    def _ask_iwrap_import_path(self) -> str:
        filename, _ = QFileDialog.getOpenFileName(
            self.main_widget,
            self.tr('Import from IWRAP XML'),
            '',
            'IWRAP XML Files (*.xml);;All Files (*.*)',
        )
        if not filename:
            return ''
        if not os.path.exists(filename):
            QMessageBox.warning(
                self.main_widget,
                self.tr('File Not Found'),
                self.tr(f'The file does not exist:\n{filename}'),
            )
            return ''
        return filename

    def _validate_iwrap_payload(self, data: dict) -> dict:
        """Run schema validation + best-effort normalisation on the
        imported payload.

        Validation failure is non-fatal: we surface a warning popup
        and let the user proceed with whatever the converter produced.
        """
        from omrat_utils.storage import Storage
        storage = Storage(self)
        data = storage._normalize_legacy_to_schema(data)
        try:
            from omrat_utils.validate_data import RootModelSchema
            RootModelSchema.model_validate(data)
        except Exception as validation_error:
            QMessageBox.warning(
                self.main_widget,
                self.tr('Import Validation Error'),
                self.tr(
                    f'The imported data failed validation:\n'
                    f'{validation_error}\n\n'
                    f'The data will still be loaded, but there may be '
                    f'missing or incorrect values.'
                ),
            )
        return data

    def _apply_iwrap_payload(self, data: dict) -> None:
        """Push the imported dict into the live OMRAT widgets and
        traffic table."""
        gd = GatherData(self)
        gd.populate(data)
        # ``populate`` rebinds ``self.traffic_data`` to a fresh dict;
        # mirror it onto the Traffic helper so the table refresh
        # below sees the new data.
        self.traffic.traffic_data = self.traffic_data
        self.traffic.update_direction_select()
        self.traffic.update_traffic_tbl('segment')
