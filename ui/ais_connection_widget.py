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