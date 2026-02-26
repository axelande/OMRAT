# -*- coding: utf-8 -*-
import os

from qgis.PyQt import QtGui, QtWidgets, uic
from qgis.PyQt.QtCore import pyqtSignal

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'show_geometric_results.ui'))


class ShowGeomRes(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(ShowGeomRes, self).__init__(parent)
        self.setupUi(self)
        
        self.result_layout:QtWidgets.QVBoxLayout
