# -*- coding: utf-8 -*-
import os

from qgis.PyQt import QtWidgets, uic

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'ship_type_map.ui'))


class ShipTypeMapWidget(QtWidgets.QDialog, FORM_CLASS):
    """Settings > Ship type mapping... dialog (IMO / MMSI -> OMRAT category)."""

    def __init__(self, parent=None):
        super(ShipTypeMapWidget, self).__init__(parent)
        self.setupUi(self)
        self.lblHelp: QtWidgets.QLabel
        self.cbEnabled: QtWidgets.QCheckBox
        self.leSchema: QtWidgets.QLineEdit
        self.leTable: QtWidgets.QLineEdit
        self.pbImport: QtWidgets.QPushButton
        self.pbExport: QtWidgets.QPushButton
        self.pbRefresh: QtWidgets.QPushButton
        self.lblStatus: QtWidgets.QLabel
        self.twPreview: QtWidgets.QTableWidget
        self.buttonBox: QtWidgets.QDialogButtonBox
