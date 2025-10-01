from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omrat import OMRAT


from qgis.PyQt.QtWidgets import QDialogButtonBox

from ui.causation_factor_widget import CausationFactorsWidget

class CausationFactors:
    def __init__(self, parent: "OMRAT") -> None:
        self.p = parent
        self.cfw = CausationFactorsWidget()
        self.data: dict[str, float] = {'p_pc': 1.6E-4, 'd_pc':1}
        
    def commit_changes(self):
        p_pc = float(self.cfw.lePoweredPc.text())
        d_pc = float(self.cfw.leDriftingPc.text())
        self.data = {'p_pc': p_pc, 'd_pc': d_pc}
    
    def set_values(self):
        self.cfw.lePoweredPc.setText(f"{self.data['p_pc']}")
        self.cfw.leDriftingPc.setText(f"{self.data['d_pc']}")
    
    def run(self):
        self.cfw.show()
        self.set_values()
        self.buttonBox: QDialogButtonBox | None = self.cfw.findChild(QDialogButtonBox, 'buttonBox')
        if isinstance(self.buttonBox, QDialogButtonBox):
            self.buttonBox.accepted.connect(self.commit_changes)
        self.cfw.exec_()
