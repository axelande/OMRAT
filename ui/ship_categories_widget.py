# -*- coding: utf-8 -*-
import os

from qgis.PyQt import QtWidgets, uic

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'ship_categories.ui'))


class ShipCategoriesWidget(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(ShipCategoriesWidget, self).__init__(parent)
        self.setupUi(self)

        self.cvTypes: QtWidgets.QTableWidget
        self.twLengths: QtWidgets.QTableWidget
