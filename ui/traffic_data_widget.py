# -*- coding: utf-8 -*-
"""

"""

import os

from qgis.PyQt import QtGui, QtWidgets, uic
from qgis.PyQt.QtCore import pyqtSignal

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'traffic_data.ui'))


class TrafficDataWidget(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(TrafficDataWidget, self).__init__(parent)
        self.setupUi(self)
        self.twTrafficData: QtWidgets.QTableWidget
        self.cbTrafficSelectSeg: QtWidgets.QComboBox
        self.cbSelectType: QtWidgets.QComboBox
        self.cbTrafficDirectionSelect: QtWidgets.QComboBox
