# -*- coding: utf-8 -*-
"""Dialog for editing the per-accident spill fraction matrix (% of full tank)."""

import os

from qgis.PyQt import QtWidgets, uic

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'spill_fraction.ui'))


class SpillFractionWidget(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.twSpillFraction: QtWidgets.QTableWidget
        self.buttonBox: QtWidgets.QDialogButtonBox
