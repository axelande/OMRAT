# -*- coding: utf-8 -*-
"""Dialog for editing the maximum oil onboard matrix (m^3)."""

import os

from qgis.PyQt import QtWidgets, uic

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'oil_onboard.ui'))


class OilOnboardWidget(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.twOilOnboard: QtWidgets.QTableWidget
        self.buttonBox: QtWidgets.QDialogButtonBox
