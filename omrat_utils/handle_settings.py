from typing import Any
from qgis.PyQt.QtWidgets import QDialogButtonBox

from omrat_utils.repair_time import Repair
from ui.drift_settings_widget import DriftSettingsWidget

class DriftSettings:
    def __init__(self, parent):
        self.parent = parent
        self.dsw = DriftSettingsWidget(None)
        self.repair = Repair(self)
        rose = {'0': 12.5, '45': 12.5, '90': 12.5, '135': 12.5, '180': 12.5, '225': 12.5, '270': 12.5, '315': 12.5}
        repair = {'func': "",
                  'std': .95,
                  'loc': .2,
                  'scale': .85,
                  'use_lognormal': True}
        # This is set here as default values, however it is overwritten while loading user data
        self.drift_values:dict[str, Any] = {'drift_p': 1, 'anchor_p': .95,'anchor_d': 7, 'speed':1, 'rose': rose, 'repair': repair}
        
    def commit_changes(self):
        n = float(self.dsw.leDriftN.text())
        ne = float(self.dsw.leDriftNE.text())
        e = float(self.dsw.leDriftE.text())
        se = float(self.dsw.leDriftSE.text())
        s = float(self.dsw.leDriftS.text())
        sw = float(self.dsw.leDriftSW.text())
        w = float(self.dsw.leDriftW.text())
        nw = float(self.dsw.leDriftNW.text())
        rose = {'0': n, '45': ne, '90': e, '135': se, '180': s, '225': sw, '270': w, '315': nw}
        speed = float(self.dsw.leDriftSpeed.text())
        drift_p = float(self.dsw.leDriftProb.text())
        anchor_p = float(self.dsw.leAnchorProb.text())
        anchor_d = float(self.dsw.leAnchorMaxDepth.text())
        repair = {'func': self.dsw.leRepairFunc.toPlainText(),
                  'std': float(self.dsw.leRepairStd.text()),
                  'loc': float(self.dsw.leRepairLoc.text()),
                  'scale': float(self.dsw.leRepairScale.text()),
                  'use_lognormal': self.dsw.rbLogNormal.isChecked()}
        self.drift_values = {'drift_p': drift_p, 'anchor_p':anchor_p,'anchor_d':anchor_d, 'speed':speed, 'rose': rose, 
                             'repair': repair}
        self.parent.drift_values = self.drift_values
        
    def discard_changes(self):
        pass
    
    def unload(self):
        self.dsw.pbTestRepair.clicked.disconnect()
        while self.dsw.canRepairViewLay.count():
            item = self.dsw.canRepairViewLay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()  # Properly delete the widget

    def populate_drift(self):
        """Populates the drift fields with the "drift_values" dict """
        self.dsw.leDriftN.setText(f"{self.drift_values['rose']['0']}")
        self.dsw.leDriftNE.setText(f"{self.drift_values['rose']['45']}")
        self.dsw.leDriftE.setText(f"{self.drift_values['rose']['90']}")
        self.dsw.leDriftSE.setText(f"{self.drift_values['rose']['135']}")
        self.dsw.leDriftS.setText(f"{self.drift_values['rose']['180']}")
        self.dsw.leDriftSW.setText(f"{self.drift_values['rose']['225']}")
        self.dsw.leDriftW.setText(f"{self.drift_values['rose']['270']}")
        self.dsw.leDriftNW.setText(f"{self.drift_values['rose']['315']}")
        self.dsw.leDriftSpeed.setText(f"{self.drift_values['speed']}")
        self.dsw.leAnchorProb.setText(f"{self.drift_values['anchor_p']}")
        self.dsw.leAnchorMaxDepth.setText(f"{self.drift_values['anchor_d']}")
        self.dsw.leDriftProb.setText(f"{self.drift_values['drift_p']}")
        self.dsw.leRepairFunc.setText(f"{self.drift_values['repair']['func']}")
        self.dsw.leRepairStd.setText(f"{self.drift_values['repair']['std']}")
        self.dsw.leRepairLoc.setText(f"{self.drift_values['repair']['loc']}")
        self.dsw.leRepairScale.setText(f"{self.drift_values['repair']['scale']}")
        self.dsw.rbLogNormal.setChecked(int(self.drift_values['repair']['use_lognormal']))

    def run(self):
        self.populate_drift()
        self.dsw.show()
        # Get the button box
        self.buttonBox = self.dsw.findChild(QDialogButtonBox, 'buttonBox')
        self.dsw.pbTestRepair.clicked.connect(self.repair.test_evaluate)
        self.dsw.rbLogNormal.toggled.connect(self.repair.test_evaluate)
        self.dsw.rbUserDefined.toggled.connect(self.repair.test_evaluate)
        self.dsw.leRepairStd.textChanged.connect(self.repair.test_evaluate)
        self.dsw.leRepairLoc.textChanged.connect(self.repair.test_evaluate)
        self.dsw.leRepairScale.textChanged.connect(self.repair.test_evaluate)
        
        # Connect the accepted signal to your custom slot
        self.buttonBox.accepted.connect(self.commit_changes)
        
        # Optionally, connect the rejected signal to a different slot
        self.buttonBox.rejected.connect(self.discard_changes)
        self.dsw.exec_()