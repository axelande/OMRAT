# -*- coding: utf-8 -*-
"""
Result widget for displaying OMRAT calculation results.

This widget displays results for grounding, allision, and ship-ship collision
calculations in a dialog format.
"""

import os
from typing import Any

from qgis.PyQt import QtGui, QtWidgets, uic
from qgis.PyQt.QtCore import pyqtSignal

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'result.ui'))


class ResultWidget(QtWidgets.QDialog, FORM_CLASS):
    """Dialog widget for displaying OMRAT calculation results."""

    def __init__(self, parent: Any = None):
        """Constructor.

        Args:
            parent: Parent widget, defaults to None.
        """
        super(ResultWidget, self).__init__(parent)
        self.setupUi(self)

        # Result type selector
        self.cbResType: QtWidgets.QComboBox

        # Result tables and trees
        self.twRes: QtWidgets.QTableWidget
        self.treewSegment: QtWidgets.QTreeWidget
        self.treewObject: QtWidgets.QTreeWidget

        # Ship-ship collision result fields
        self.LEPHeadOnCollision: QtWidgets.QLineEdit
        self.LEPOvertakingCollision: QtWidgets.QLineEdit
        self.LEPCrossingCollision: QtWidgets.QLineEdit
        self.LEPMergingCollision: QtWidgets.QLineEdit
        self.LEPTotalCollision: QtWidgets.QLineEdit

        # Dialog buttons
        self.buttonBox: QtWidgets.QDialogButtonBox

    def update_collision_results(self, collision_data: dict[str, float]) -> None:
        """Update the collision result fields with calculation results.

        Args:
            collision_data: Dictionary containing collision frequencies with keys:
                - 'head_on': Head-on collision frequency
                - 'overtaking': Overtaking collision frequency
                - 'crossing': Crossing collision frequency
                - 'bend': Bend/merging collision frequency
                - 'total': Total collision frequency
        """
        # Format collision frequencies for display (scientific notation)
        def format_freq(value: float) -> str:
            """Format frequency value for display."""
            if value == 0.0:
                return "0.0"
            elif value < 0.001:
                return f"{value:.2e}"
            else:
                return f"{value:.6f}"

        head_on = collision_data.get('head_on', 0.0)
        overtaking = collision_data.get('overtaking', 0.0)
        crossing = collision_data.get('crossing', 0.0)
        bend = collision_data.get('bend', 0.0)  # 'bend' maps to merging in UI
        total = collision_data.get('total', 0.0)

        self.LEPHeadOnCollision.setText(format_freq(head_on))
        self.LEPOvertakingCollision.setText(format_freq(overtaking))
        self.LEPCrossingCollision.setText(format_freq(crossing))
        self.LEPMergingCollision.setText(format_freq(bend))
        self.LEPTotalCollision.setText(format_freq(total))

    def clear_collision_results(self) -> None:
        """Clear all collision result fields."""
        self.LEPHeadOnCollision.clear()
        self.LEPOvertakingCollision.clear()
        self.LEPCrossingCollision.clear()
        self.LEPMergingCollision.clear()
        self.LEPTotalCollision.clear()
