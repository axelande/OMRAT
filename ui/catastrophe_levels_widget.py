# -*- coding: utf-8 -*-
"""Dialog for editing the user-defined catastrophe levels."""

import os

from qgis.PyQt import QtWidgets, uic

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'catastrophe_levels.ui'))


class CatastropheLevelsWidget(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.twCatastropheLevels: QtWidgets.QTableWidget
        self.pbAddLevel: QtWidgets.QPushButton
        self.pbRemoveLevel: QtWidgets.QPushButton
        self.buttonBox: QtWidgets.QDialogButtonBox
