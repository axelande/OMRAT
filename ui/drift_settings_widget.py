# -*- coding: utf-8 -*-
import os

from qgis.PyQt import QtGui, QtWidgets, uic
from qgis.PyQt.QtCore import pyqtSignal

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'drift_settings.ui'))


class DriftSettingsWidget(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(DriftSettingsWidget, self).__init__(parent)
        self.setupUi(self)
        self.leAnchorMaxDepth: QtWidgets.QLineEdit
        self.leAnchorProb: QtWidgets.QLineEdit
        self.leDriftProb: QtWidgets.QLineEdit
        self.leDriftE: QtWidgets.QLineEdit
        self.leDriftNE: QtWidgets.QLineEdit
        self.leDriftNW: QtWidgets.QLineEdit
        self.leDriftS: QtWidgets.QLineEdit
        self.leDriftSE: QtWidgets.QLineEdit
        self.leDriftSW: QtWidgets.QLineEdit
        self.leDriftW: QtWidgets.QLineEdit
        self.leDriftN: QtWidgets.QLineEdit
        self.leDriftSpeed: QtWidgets.QLineEdit
        self.leRepairLoc: QtWidgets.QLineEdit
        self.leRepairScale: QtWidgets.QLineEdit
        self.leRepairStd: QtWidgets.QLineEdit
        self.leRepairFunc: QtWidgets.QTextEdit
        self.rbLogNormal: QtWidgets.QRadioButton
        self.rbUserDefined: QtWidgets.QRadioButton
        self.pbTestRepair: QtWidgets.QPushButton
