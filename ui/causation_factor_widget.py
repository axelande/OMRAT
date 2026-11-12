# -*- coding: utf-8 -*-
import os
from typing import Any

from qgis.PyQt import QtWidgets, uic

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'causation_factors.ui'))


class CausationFactorsWidget(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent: Any = None):
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
        self.leMergingCf: QtWidgets.QLineEdit
        self.leBendCf: QtWidgets.QLineEdit
        # Smallest route angle at a waypoint (compute/encounter_angle.py), default 10 deg
        self.sbMinEncounterAngle: QtWidgets.QDoubleSpinBox
        self.leGroundingCf: QtWidgets.QLineEdit        # Cat II (missed turn)
        self.leAllisionCf: QtWidgets.QLineEdit         # Cat II (missed turn)
        self.leGroundingCat1Cf: QtWidgets.QLineEdit    # Cat I (obstacle in lane)
        self.leAllisionCat1Cf: QtWidgets.QLineEdit     # Cat I (obstacle in lane)

        # Squat in the grounding models (compute/squat.py), default on
        self.cbSquatDrifting: QtWidgets.QCheckBox
        self.cbSquatPowered: QtWidgets.QCheckBox

        # Passenger ship / fast ferry reduction factor (IWRAP's), default 20, 1 = none
        self.sbPassengerReduction: QtWidgets.QDoubleSpinBox
        self.sbFastFerryReduction: QtWidgets.QDoubleSpinBox

        # Dialog buttons
        self.buttonBox: QtWidgets.QDialogButtonBox
