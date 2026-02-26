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
        # Grounding/Allision causation factors
        self.lePoweredPc: QtWidgets.QLineEdit
        self.leDriftingPc: QtWidgets.QLineEdit

        # Ship-Ship collision causation factors
        self.leHeadOnCf: QtWidgets.QLineEdit
        self.leOvertakingCf: QtWidgets.QLineEdit
        self.leCrossingCf: QtWidgets.QLineEdit
        self.leBendCf: QtWidgets.QLineEdit
        self.leGroundingCf: QtWidgets.QLineEdit
        self.leAllisionCf: QtWidgets.QLineEdit

        # Dialog buttons
        self.buttonBox: QtWidgets.QDialogButtonBox
