from qgis.PyQt.QtWidgets import QDialogButtonBox

from ui.causation_factor_widget import CausationFactorsWidget

class CausationFactors:
    def __init__(self, parent):
        self.p = parent
        self.cfw = CausationFactorsWidget()
        self.data = {'p_pc': 1.6E-4, 'd_pc':1}
        
    def commit_changes(self):
        p_pc = float(self.cfw.lePoweredPc.text())
        d_pc = float(self.cfw.leDriftingPc.text())
        self.data = {'p_pc': p_pc, 'd_pc': d_pc}
    
    def set_values(self):
        self.cfw.lePoweredPc.setText(f"{self.data['p_pc']}")
        self.cfw.leDriftingPc.setText(f"{self.data['d_pc']}")
    
    def run(self):
        self.dsw.show()
        self.set_values()
        self.buttonBox = self.dsw.findChild(QDialogButtonBox, 'buttonBox')
        self.buttonBox.accepted.connect(self.commit_changes)
        self.dsw.exec_()