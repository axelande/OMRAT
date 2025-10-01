# -*- coding: utf-8 -*-
import os
from typing import Any, Self

from qgis.PyQt import QtGui, QtWidgets, uic
from qgis.PyQt.QtCore import pyqtSignal

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'causation_factors.ui'))


class CausationFactorsWidget(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent: Any=None):
        """Constructor."""
        super(CausationFactorsWidget, self).__init__(parent)
        self.setupUi(self)
        self.lePoweredPc: QtWidgets.QLineEdit
        self.leDriftingPc: QtWidgets.QLineEdit
        self.buttonBox: QtWidgets.QDialogButtonBox
