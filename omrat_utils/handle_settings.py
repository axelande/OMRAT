from typing import Any
from qgis.PyQt.QtWidgets import QDialogButtonBox, QLineEdit

from omrat_utils.repair_time import Repair
from ui.drift_settings_widget import DriftSettingsWidget

class DriftSettings:
    def __init__(self, parent):
        self.parent = parent
        self.dsw = DriftSettingsWidget(None)
        self.repair = Repair(self)
        rose = {'0': .125, '45': .125, '90': .125, '135': .125, '180': .125, '225': .125, '270': .125, '315': .125}
        repair: dict[str,str|float|bool] = {'func': "",
                  'std': .95,
                  'loc': .2,
                  'scale': .85,
                  'use_lognormal': True}
        # This is set here as default values, however it is overwritten while loading user data
        self.drift_values:dict[str, Any] = {'drift_p': 1, 'anchor_p': .95,'anchor_d': 7, 'speed': 1 * 3600 / 1852, 
                                            'rose': rose, 'repair': repair}
        
    def adjust_directions(self, changed_widget: QLineEdit) -> None:
        widgets: list[Any] = [self.dsw.leDriftN, self.dsw.leDriftNE, self.dsw.leDriftNW, self.dsw.leDriftE, 
                              self.dsw.leDriftSE, self.dsw.leDriftS, self.dsw.leDriftSW, self.dsw.leDriftW]
        
        total_weight = 100
        changed_value = float(changed_widget.text())

        # Calculate the remaining weight
        remaining_weight = total_weight - changed_value

        # Distribute the remaining weight proportionally among the other widgets
        other_widgets = [w for w in widgets if w != changed_widget]
        other_values = [
            float(w.text()) if hasattr(w, 'text') and w.text() != '' else w.value() if hasattr(w, 'value') else 0
            for w in other_widgets
        ]
        other_total = sum(other_values)

        if other_total == 0:
            # If all other values are zero, distribute equally
            for w in other_widgets:
                w.setText(str(remaining_weight / len(other_widgets)))
        else:
            # Adjust the other values proportionally
            for w, value in zip(other_widgets, other_values):
                adjusted_value = (value / other_total) * remaining_weight
                w.setText(str(round(adjusted_value, 2)))
        # Ensure the total sum is exactly 100
        self.ensure_total_sum(widgets)

    def ensure_total_sum(self, widgets:list[QLineEdit]):
        """Ensure the total sum of weights equals 100."""
        total = sum(
            float(w.text()) if w.text() != '' else 0
            for w in widgets
        )
        difference = 100 - total

        # Adjust the last widget to make the total exactly 100
        last_widget = widgets[-1]
        last_value = float(last_widget.text()) if last_widget.text() != '' else 0
        last_widget.setText(str(last_value + difference))
        
    def commit_changes(self):
        n = float(self.dsw.leDriftN.text()) / 100
        ne = float(self.dsw.leDriftNE.text()) / 100
        e = float(self.dsw.leDriftE.text()) / 100
        se = float(self.dsw.leDriftSE.text()) / 100
        s = float(self.dsw.leDriftS.text()) / 100
        sw = float(self.dsw.leDriftSW.text()) / 100
        w = float(self.dsw.leDriftW.text()) / 100
        nw = float(self.dsw.leDriftNW.text()) / 100
        rose = {'0': n, '45': ne, '90': e, '135': se, '180': s, '225': sw, '270': w, '315': nw}
        speed = float(self.dsw.leDriftSpeed.text()) * 1852 / 3600
        drift_p = float(self.dsw.leDriftProb.text())
        anchor_p = float(self.dsw.leAnchorProb.text())
        anchor_d = float(self.dsw.leAnchorMaxDepth.text())
        repair: dict[str,str|float|bool] = {'func': self.dsw.leRepairFunc.toPlainText(),
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
        widgets: list[Any] = [self.dsw.leDriftN, self.dsw.leDriftNE, self.dsw.leDriftNW, self.dsw.leDriftE, 
                              self.dsw.leDriftSE, self.dsw.leDriftS, self.dsw.leDriftSW, self.dsw.leDriftW]
        for widget in widgets:
            if hasattr(widget, 'editingFinished'):
                widget.editingFinished.disconnect()
            elif hasattr(widget, 'leaveEvent'):
                widget.leaveEvent.disconnect()
        while self.dsw.canRepairViewLay.count():
            item = self.dsw.canRepairViewLay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()  # Properly delete the widget

    def populate_drift(self):
        """Populates the drift fields with the "drift_values" dict """
        self.dsw.leDriftN.setText(f"{self.drift_values['rose']['0'] * 100}")
        self.dsw.leDriftNE.setText(f"{self.drift_values['rose']['45'] * 100}")
        self.dsw.leDriftE.setText(f"{self.drift_values['rose']['90'] * 100}")
        self.dsw.leDriftSE.setText(f"{self.drift_values['rose']['135'] * 100}")
        self.dsw.leDriftS.setText(f"{self.drift_values['rose']['180'] * 100}")
        self.dsw.leDriftSW.setText(f"{self.drift_values['rose']['225'] * 100}")
        self.dsw.leDriftW.setText(f"{self.drift_values['rose']['270'] * 100}")
        self.dsw.leDriftNW.setText(f"{self.drift_values['rose']['315'] * 100}")
        self.dsw.leDriftSpeed.setText(f"{round(self.drift_values['speed'] * 3600 /1852, 3)}")
        self.dsw.leAnchorProb.setText(f"{self.drift_values['anchor_p']}")
        self.dsw.leAnchorMaxDepth.setText(f"{self.drift_values['anchor_d']}")
        self.dsw.leDriftProb.setText(f"{self.drift_values['drift_p']}")
        self.dsw.leRepairFunc.setText(f"{self.drift_values['repair']['func']}")
        self.dsw.leRepairStd.setText(f"{self.drift_values['repair']['std']}")
        self.dsw.leRepairLoc.setText(f"{self.drift_values['repair']['loc']}")
        self.dsw.leRepairScale.setText(f"{self.drift_values['repair']['scale']}")
        self.dsw.rbLogNormal.setChecked(self.drift_values['repair']['use_lognormal'])

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
        widgets: list[Any] = [self.dsw.leDriftN, self.dsw.leDriftNE, self.dsw.leDriftNW, self.dsw.leDriftE, 
                        self.dsw.leDriftSE, self.dsw.leDriftS, self.dsw.leDriftSW, self.dsw.leDriftW]
        for widget in widgets:
            if hasattr(widget, 'editingFinished'):
                widget.editingFinished.connect(lambda w=widget: self.adjust_directions(w))
            elif hasattr(widget, 'leaveEvent'):
                widget.leaveEvent.connect(lambda w=widget: self.adjust_directions(w))
                
        # Connect the accepted signal to your custom slot
        self.buttonBox.accepted.connect(self.commit_changes)
        
        # Optionally, connect the rejected signal to a different slot
        self.buttonBox.rejected.connect(self.discard_changes)
        self.dsw.exec_()