"""Run Analysis tab -> **Sensitivity analysis...** slot, factored out of ``omrat.OMRAT``.

Gathers the live project exactly like **Run model** does
(``GatherData.get_all_for_save``), builds the parameter registry, takes
the last finished run as an optional baseline and opens
:class:`~omrat_utils.sensitivity_dialog.SensitivityDialog`.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtWidgets import QMessageBox

if TYPE_CHECKING:  # pragma: no cover
    from compute.sensitivity import BaselineReports


class SensitivityMixin:
    """Button wiring + dialog lifecycle for the sensitivity analysis."""

    def _setup_sensitivity_button(self) -> None:
        btn = getattr(self.main_widget, 'pbSensitivity', None)
        if btn is None:
            return  # button not present in this build of the .ui
        btn.clicked.connect(self.open_sensitivity_dialog)

    def _live_sensitivity_baseline(self) -> 'BaselineReports | None':
        """Reports of the last finished run in this session, or ``None``."""
        from compute.sensitivity import reports_from_calc
        calc = getattr(self, 'calc', None)
        if calc is None:
            return None
        if not any(getattr(calc, attr, None) for attr in (
                'drifting_report', 'collision_report',
                'powered_grounding_report', 'powered_allision_report')):
            return None
        return reports_from_calc(calc)

    def _sensitivity_input_data(self) -> dict[str, Any]:
        from omrat_utils.gather_data import GatherData
        data = copy.deepcopy(GatherData(self).get_all_for_save())
        try:
            name = (self.main_widget.LEModelName.text() or '').strip()
        except Exception:  # nosec B110 B112
            name = ''
        if name:
            data['project_name'] = name
        return data

    def open_sensitivity_dialog(self) -> None:
        """Open (or raise) the non-modal sensitivity dialog."""
        from compute.sensitivity import build_parameter_specs
        from omrat_utils.sensitivity_dialog import SensitivityDialog

        existing = getattr(self, '_sensitivity_dialog', None)
        if existing is not None:
            try:
                if existing.isVisible():
                    existing.raise_()
                    existing.activateWindow()
                    return
            except RuntimeError:
                pass  # underlying C++ object already deleted
            self._sensitivity_dialog = None

        data = self._sensitivity_input_data()
        if not (data.get('segment_data') or {}):
            QMessageBox.information(
                self.main_widget, self.tr('Sensitivity analysis'),
                self.tr('Draw or load at least one leg with traffic before running a sensitivity analysis.'),
            )
            return
        specs = build_parameter_specs(data)
        baseline = self._live_sensitivity_baseline()
        try:
            run_name = (self.main_widget.LEModelName.text() or '').strip()
        except Exception:  # nosec B110 B112
            run_name = ''
        out_dir = None
        get_dir = getattr(self, '_get_output_dir', None)
        if callable(get_dir):
            out_dir = get_dir()
        dlg = SensitivityDialog(
            self.main_widget, data, specs, live_baseline=baseline,
            out_dir=out_dir, run_name=run_name or 'model',
        )
        self._sensitivity_dialog = dlg
        dlg.show()
        QgsMessageLog.logMessage(
            f'Sensitivity dialog opened with {len(specs)} parameters'
            + (' (live baseline available)' if baseline is not None else ''),
            'OMRAT', Qgis.MessageLevel.Info)


__all__ = ['SensitivityMixin']
