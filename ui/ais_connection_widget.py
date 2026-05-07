# -*- coding: utf-8 -*-
import os

from qgis.PyQt import QtGui, QtWidgets, uic
from qgis.PyQt.QtCore import pyqtSignal

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'ais_connection.ui'))


class AISConnectionWidget(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None) -> QtWidgets.QDialog:
        """Constructor."""
        super(AISConnectionWidget, self).__init__(parent)
        self.setupUi(self)
        self.leDBHost: QtWidgets.QLineEdit
        self.SBPort: QtWidgets.QSpinBox
        self.leDBName: QtWidgets.QLineEdit
        self.leUserName: QtWidgets.QLineEdit
        self.lePassword: QtWidgets.QLineEdit
        self.leProvider: QtWidgets.QLineEdit
        self.SBYear: QtWidgets.QSpinBox
        self.leMaxDev: QtWidgets.QLineEdit
        # External vessel-data lookup (optional).  ``gbExtVessel`` is a
        # checkable QGroupBox; its ``isChecked()`` doubles as the master
        # enable flag for the per-column line edits below.
        self.gbExtVessel: QtWidgets.QGroupBox
        self.leExtSchema: QtWidgets.QLineEdit
        self.leExtTable: QtWidgets.QLineEdit
        self.leExtMmsiCol: QtWidgets.QLineEdit
        self.leExtLoaCol: QtWidgets.QLineEdit
        self.leExtBeamCol: QtWidgets.QLineEdit
        self.leExtShipTypeCol: QtWidgets.QLineEdit
        self.leExtAirDraughtCol: QtWidgets.QLineEdit
        # Multiply per-ping frequency counts by (one year / observed
        # coverage) so a 48 h ingestion is reported as annualised traffic.
        self.cbRecalcFullYear: QtWidgets.QCheckBox