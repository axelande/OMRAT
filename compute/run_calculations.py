"""
Facade module for OMRAT calculations.

Assembles the Calculation class from mixins and re-exports public symbols
for backward compatibility. All calculation logic lives in the individual
mixin modules.
"""
import sys
from typing import Any, TYPE_CHECKING, Callable

import numpy as np
from qgis.PyQt.QtWidgets import QWidget

sys.path.append('.')

# --- Backward-compatible re-exports ---
from compute.drift_corridor_geometry import (  # noqa: F401
    _compass_idx_to_math_idx,
    _extract_obstacle_segments,
    _create_drift_corridor,
    _segment_intersects_corridor,
)
from compute.data_preparation import (  # noqa: F401
    get_distribution,
    clean_traffic,
    safe_load_wkt,
    load_areas,
    split_structures_and_depths,
    transform_to_utm,
    prepare_traffic_lists,
)

# --- Mixin imports ---
from compute.drifting_model import DriftingModelMixin
from compute.ship_collision_model import ShipCollisionModelMixin
from compute.powered_model import PoweredModelMixin
from compute.drifting_report import DriftingReportMixin
from compute.visualization import VisualizationMixin

if TYPE_CHECKING:
    from omrat import OMRAT


class Calculation(
    DriftingModelMixin,
    ShipCollisionModelMixin,
    PoweredModelMixin,
    DriftingReportMixin,
    VisualizationMixin,
):
    """Main calculation facade -- composes all model mixins."""

    def __init__(self, parent: "OMRAT") -> None:
        self.p = parent
        self.canvas: QWidget | None = None
        self.drifting_report: dict[str, Any] | None = None
        self._progress_callback: Callable[[int, int, str], bool] | None = None
        # Store metadata for result layer generation
        self._last_structures: list[dict[str, Any]] = []
        self._last_depths: list[dict[str, Any]] = []
        self.allision_result_layer = None
        self.grounding_result_layer = None
        # Ship-ship collision attributes
        self.ship_collision_prob: float = 0.0
        self.collision_report: dict[str, Any] | None = None
        # Drifting model attributes
        self.drifting_allision_prob: float = 0.0
        self.drifting_grounding_prob: float = 0.0

    def set_progress_callback(self, callback: Callable[[int, int, str], bool]) -> None:
        """
        Set a callback function for progress updates.

        Args:
            callback: Function that takes (completed, total, message) and returns bool.
                     Should return False to cancel the operation, True to continue.
        """
        self._progress_callback = callback

    def _report_progress(self, phase: str, phase_progress: float, message: str) -> bool:
        """
        Report progress across multiple calculation phases.

        Phases and their weight in overall progress:
        - 'spatial': 0-60% (probability hole calculations - expensive)
        - 'cascade': 60-90% (traffic cascade - moderate)
        - 'layers': 90-100% (result layer creation - fast)

        Args:
            phase: One of 'spatial', 'cascade', 'layers'
            phase_progress: Progress within the phase (0.0 to 1.0)
            message: Status message to display

        Returns:
            True to continue, False to cancel
        """
        if not self._progress_callback:
            return True

        # Phase weights (must sum to 1.0)
        phase_weights = {
            'spatial': (0.0, 0.60),   # 0% to 60%
            'cascade': (0.60, 0.90),  # 60% to 90%
            'layers': (0.90, 1.0),    # 90% to 100%
        }

        start, end = phase_weights.get(phase, (0.0, 1.0))
        overall_progress = start + (end - start) * min(1.0, max(0.0, phase_progress))

        # Report as percentage (0-100)
        return self._progress_callback(
            int(overall_progress * 100),
            100,
            message
        )

    def get_no_ship_h(self, data: dict[str, Any]) -> list[float]:
        no_ships: list[float] = []
        td: dict[str, dict[str, np.ndarray]] = data['traffic_data']
        for leg, leg_dirs in td.items():
            leg_length = data['segment_data'][leg]['line_length']
            for v in leg_dirs.values():
                freq = np.array(v['Frequency (ships/year)'])
                speed = np.array(v['Speed (knots)'])
                h = leg_length / (speed * 1852 / 3600)
                no_ships.append(float(np.sum(freq * h)))
        return no_ships
