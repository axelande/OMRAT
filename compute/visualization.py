"""Visualization mixin for the calculation runner.

Extracts the drift, powered-allision, and powered-grounding visualisation
methods so they can be composed into ``CalculationRunner`` without bloating
the main module.
"""

from typing import Any

import geopandas as gpd
import shapely.wkt as sw

try:
    from shapely import make_valid as shp_make_valid
except ImportError:
    shp_make_valid = None

from compute.data_preparation import (
    prepare_traffic_lists,
    transform_to_utm,
    split_structures_and_depths,
)
from geometries.get_drifting_overlap import DriftingOverlapVisualizer
from geometries.get_powered_overlap import (
    PoweredOverlapVisualizer,
    SimpleProjector as _PoweredProjector,
    _build_legs_and_obstacles,
    _parse_point,
)
from ui.show_geom_res import ShowGeomRes


class VisualizationMixin:
    """Mixin providing interactive visualisation dialogs.

    Expects the host class to expose ``self.p.main_widget`` (the parent
    QWidget used for dialogs) and ``self.canvas`` (the QGIS map canvas).
    """

    def run_drift_visualization(self, data: dict[str, Any]) -> None:
        if not data.get('traffic_data'):
            return

        lines, distributions, weights, line_names = prepare_traffic_lists(data)

        # Prepare objects
        objects = [sw.loads(wkt) for _, _, wkt in data['objects']]
        transformed_lines, transformed_objects, utm_crs = transform_to_utm(lines, objects)
        # Fix invalid geometries for visualization and overlap computation
        fixed_objects = []
        for obj in transformed_objects:
            try:
                if shp_make_valid is not None:
                    fixed = shp_make_valid(obj)
                else:
                    fixed = obj.buffer(0)
            except Exception:
                fixed = obj
            if fixed is not None:
                fixed_objects.append(fixed)
        transformed_objects = fixed_objects
        longest_length = max(line.length for line in transformed_lines)
        transformed_objects_gdf = [gpd.GeoDataFrame(geometry=[obj]) for obj in transformed_objects]

        # Create and show dialog
        dialog = ShowGeomRes(self.p.main_widget)
        DriftingOverlapVisualizer.show_in_dialog(
            dialog,
            transformed_lines,
            line_names,
            transformed_objects_gdf,
            distributions,
            weights,
            data = data,
            distance=longest_length * 3.0
        )
        dialog.exec_()

    def run_powered_allision_visualization(self, data: dict[str, Any]) -> None:
        """Show an interactive Cat II powered allision visualisation dialog.

        Uses the same ``ShowGeomRes`` dialog as the drifting visualisation but
        populates it with shadow-aware Cat II ray-casting plots showing how
        ships that miss a turn may hit objects (structures).
        """
        if not data.get('traffic_data') or not data.get('segment_data'):
            return
        try:
            max_draft = float(data.get('max_draft', 15.0))
        except (TypeError, ValueError):
            max_draft = 15.0

        dialog = ShowGeomRes(self.p.main_widget)
        PoweredOverlapVisualizer.show_in_dialog(
            dialog, data, mode="allision", max_draft=max_draft,
        )
        dialog.exec_()

    def run_powered_grounding_visualization(self, data: dict[str, Any]) -> None:
        """Show an interactive Cat II powered grounding visualisation dialog.

        Uses the same ``ShowGeomRes`` dialog as the drifting visualisation but
        populates it with shadow-aware Cat II ray-casting plots showing how
        ships that miss a turn may run aground on shallow depth areas.
        """
        if not data.get('traffic_data') or not data.get('segment_data'):
            return
        try:
            max_draft = float(data.get('max_draft', 15.0))
        except (TypeError, ValueError):
            max_draft = 15.0

        dialog = ShowGeomRes(self.p.main_widget)
        PoweredOverlapVisualizer.show_in_dialog(
            dialog, data, mode="grounding", max_draft=max_draft,
        )
        dialog.exec_()
