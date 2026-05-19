from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omrat import OMRAT


from qgis.PyQt.QtWidgets import QDialogButtonBox

from compute.iwrap_defaults import default_pc_values
from ui.causation_factor_widget import CausationFactorsWidget

class CausationFactors:
    def __init__(self, parent: "OMRAT") -> None:
        self.p = parent
        self.cfw = CausationFactorsWidget()
        self.data: dict[str, float] = default_pc_values()
        
    def commit_changes(self):
        # Powered and drifting causation factors
        self.data['p_pc'] = float(self.cfw.lePoweredPc.text())
        self.data['d_pc'] = float(self.cfw.leDriftingPc.text())

        # Collision causation factors
        self.data['headon'] = float(self.cfw.leHeadOnCf.text())
        self.data['overtaking'] = float(self.cfw.leOvertakingCf.text())
        self.data['crossing'] = float(self.cfw.leCrossingCf.text())
        self.data['bend'] = float(self.cfw.leBendCf.text())

        # Grounding and allision causation factors
        self.data['grounding'] = float(self.cfw.leGroundingCf.text())
        self.data['allision'] = float(self.cfw.leAllisionCf.text())
    
    def set_values(self):
        # Powered and drifting causation factors
        self.cfw.lePoweredPc.setText(f"{self.data['p_pc']}")
        self.cfw.leDriftingPc.setText(f"{self.data['d_pc']}")

        # Collision causation factors
        self.cfw.leHeadOnCf.setText(f"{self.data['headon']}")
        self.cfw.leOvertakingCf.setText(f"{self.data['overtaking']}")
        self.cfw.leCrossingCf.setText(f"{self.data['crossing']}")
        self.cfw.leBendCf.setText(f"{self.data['bend']}")

        # Grounding and allision causation factors
        self.cfw.leGroundingCf.setText(f"{self.data['grounding']}")
        self.cfw.leAllisionCf.setText(f"{self.data['allision']}")
    
    def run(self):
        self.cfw.show()
        self.set_values()
        self.buttonBox: QDialogButtonBox | None = self.cfw.findChild(QDialogButtonBox, 'buttonBox')
        if isinstance(self.buttonBox, QDialogButtonBox):
            self.buttonBox.accepted.connect(self.commit_changes)
        self.cfw.exec()
