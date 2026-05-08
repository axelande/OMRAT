# -*- coding: utf-8 -*-
"""Dialog for editing the conditional spill-level probability matrix."""

import os

from qgis.PyQt import QtWidgets, uic

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'spill_probability.ui'))


class SpillProbabilityWidget(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.twSpillProbability: QtWidgets.QTableWidget
        self.lblRowSumStatus: QtWidgets.QLabel
        self.buttonBox: QtWidgets.QDialogButtonBox
