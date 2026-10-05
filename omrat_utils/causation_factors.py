from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omrat import OMRAT


from qgis.PyQt.QtWidgets import QDialogButtonBox

from compute.cat1_start_zone import (
    DEFAULT_ENABLED, DEFAULT_PCT, KEY_ENABLED, KEY_PCT, start_zone_enabled, start_zone_pct,
)
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
        self.data['merging'] = float(self.cfw.leMergingCf.text())
        self.data['bend'] = float(self.cfw.leBendCf.text())

        # Grounding and allision causation factors (Cat II = missed turn,
        # Cat I = obstacle already in the lane)
        self.data['grounding'] = float(self.cfw.leGroundingCf.text())
        self.data['allision'] = float(self.cfw.leAllisionCf.text())
        self.data['grounding_cat1'] = float(self.cfw.leGroundingCat1Cf.text())
        self.data['allision_cat1'] = float(self.cfw.leAllisionCat1Cf.text())

        # Category I start zone (compute/cat1_start_zone.py)
        cb = getattr(self.cfw, 'cbCat1SkipStart', None)
        sb = getattr(self.cfw, 'sbCat1SkipStartPct', None)
        if cb is not None and sb is not None:
            self.data[KEY_ENABLED] = bool(cb.isChecked())
            self.data[KEY_PCT] = float(sb.value())

    def set_values(self):
        # Powered and drifting causation factors
        self.cfw.lePoweredPc.setText(f"{self.data['p_pc']}")
        self.cfw.leDriftingPc.setText(f"{self.data['d_pc']}")

        # Collision causation factors
        self.cfw.leHeadOnCf.setText(f"{self.data['headon']}")
        self.cfw.leOvertakingCf.setText(f"{self.data['overtaking']}")
        self.cfw.leCrossingCf.setText(f"{self.data['crossing']}")
        self.cfw.leMergingCf.setText(f"{self.data['merging']}")
        self.cfw.leBendCf.setText(f"{self.data['bend']}")

        # Grounding and allision causation factors
        self.cfw.leGroundingCf.setText(f"{self.data['grounding']}")
        self.cfw.leAllisionCf.setText(f"{self.data['allision']}")
        self.cfw.leGroundingCat1Cf.setText(f"{self.data['grounding_cat1']}")
        self.cfw.leAllisionCat1Cf.setText(f"{self.data['allision_cat1']}")

        # Category I start zone; projects saved before the setting existed
        # get the defaults (on, 10 %) the same way compute does.
        cb = getattr(self.cfw, 'cbCat1SkipStart', None)
        sb = getattr(self.cfw, 'sbCat1SkipStartPct', None)
        if cb is not None and sb is not None:
            cb.setChecked(start_zone_enabled(self.data) if KEY_ENABLED in self.data else DEFAULT_ENABLED)
            sb.setValue(start_zone_pct(self.data) if KEY_PCT in self.data else DEFAULT_PCT)
            sb.setEnabled(cb.isChecked())
            try:
                cb.toggled.disconnect(sb.setEnabled)
            except (TypeError, RuntimeError):
                pass
            cb.toggled.connect(sb.setEnabled)

    def run(self):
        self.cfw.show()
        self.set_values()
        self.buttonBox: QDialogButtonBox | None = self.cfw.findChild(QDialogButtonBox, 'buttonBox')
        if isinstance(self.buttonBox, QDialogButtonBox):
            self.buttonBox.accepted.connect(self.commit_changes)
        self.cfw.exec()
