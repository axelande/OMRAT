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
            data=data,
            distance=longest_length * 3.0,
            drifting_report=getattr(self, 'drifting_report', None),
            accident_kind='allision',
        )
        dialog.exec()  # ``exec_`` was dropped in PyQt6 (QGIS 4).

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
        dialog.exec()

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
        dialog.exec()

    def run_drift_grounding_visualization(self, data: dict[str, Any]) -> None:
        """Show drifting-grounding overlap (drift counterpart to powered).

        Reuses the drifting visualiser but substitutes shallow depth
        polygons (depths whose value <= ``max_draft``) for the structure
        objects.  Highlights the overlap between drift corridors and
        grounding hazards so the user can see which depth areas drive
        the drifting-grounding probability.
        """
        if not data.get('traffic_data'):
            return
        depths = data.get('depths') or []
        if not depths:
            return
        try:
            max_draft = float(data.get('max_draft', 15.0))
        except (TypeError, ValueError):
            max_draft = 15.0

        lines, distributions, weights, line_names = prepare_traffic_lists(data)

        # Filter depths to grounding hazards (depth <= ship draft).
        hazards = []
        for entry in depths:
            try:
                _did, val, wkt = entry
                if float(val) <= max_draft:
                    hazards.append(sw.loads(wkt))
            except Exception:
                continue
        if not hazards:
            return
        transformed_lines, transformed_objects, _utm_crs = transform_to_utm(
            lines, hazards,
        )
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
        if not transformed_objects:
            return
        longest_length = max(line.length for line in transformed_lines)
        objects_gdf = [gpd.GeoDataFrame(geometry=[obj]) for obj in transformed_objects]
        dialog = ShowGeomRes(self.p.main_widget)
        DriftingOverlapVisualizer.show_in_dialog(
            dialog,
            transformed_lines,
            line_names,
            objects_gdf,
            distributions,
            weights,
            data=data,
            distance=longest_length * 3.0,
            drifting_report=getattr(self, 'drifting_report', None),
            accident_kind='grounding',
        )
        dialog.exec()

    def run_collision_breakdown_dialog(self, encounter_type: str) -> None:
        """Show a breakdown of a ship-ship collision encounter type.

        * head-on / overtaking  -> per leg (single-leg phenomena).
        * crossing / merging    -> per leg-pair "leg_a -> leg_b" with
          the shared waypoint.
        * bend                  -> per leg-pair "leg_a -> next_leg"
          attributed to leg_a's end waypoint.

        The dialog is read-only and adds no map layer; it answers
        "which legs / leg-pairs drive this number?".
        """
        from qgis.PyQt.QtWidgets import (
            QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QHeaderView,
        )
        report = getattr(self, 'collision_report', None) or {}

        wanted = encounter_type
        rows: list[tuple[str, ...]] = []
        headers: list[str] = []

        if wanted in ('crossing', 'merging'):
            by_leg_pair: dict[str, dict[str, Any]] = (
                report.get('by_leg_pair', {}) or {}
            )
            headers = ['Leg pair', 'Waypoint (lon lat)', 'Angle°', 'Probability']
            for label, rec in by_leg_pair.items():
                v = float(rec.get(wanted, 0.0) or 0.0)
                if v <= 0.0:
                    continue
                rows.append((
                    label,
                    str(rec.get('waypoint', '')),
                    f"{float(rec.get('angle_deg', 0.0) or 0.0):.1f}",
                    f"{v:.3e}",
                ))
            # Sort descending by probability (last column).
            rows.sort(key=lambda r: -float(r[-1]))
        elif wanted == 'bend':
            bend_by_pair: dict[str, dict[str, Any]] = (
                report.get('bend_by_pair', {}) or {}
            )
            headers = ['Leg pair', 'Waypoint (lon lat)', 'Probability']
            for label, rec in bend_by_pair.items():
                v = float(rec.get('bend', 0.0) or 0.0)
                if v <= 0.0:
                    continue
                rows.append((
                    label,
                    str(rec.get('waypoint', '')),
                    f"{v:.3e}",
                ))
            rows.sort(key=lambda r: -float(r[-1]))
        else:
            # head_on / overtaking are single-leg phenomena.
            by_leg: dict[str, dict[str, float]] = report.get('by_leg', {}) or {}
            headers = ['Leg', 'Probability']
            for leg_id, leg_vals in by_leg.items():
                if not isinstance(leg_vals, dict):
                    continue
                v = float(leg_vals.get(wanted, 0.0) or 0.0)
                if v <= 0.0:
                    continue
                rows.append((str(leg_id), f"{v:.3e}"))
            rows.sort(key=lambda r: -float(r[-1]))

        if not rows:
            return

        dialog = QDialog(self.p.main_widget)
        title = wanted.replace('_', '-').title()
        if wanted in ('crossing', 'merging', 'bend'):
            dialog.setWindowTitle(f"{title} collision per leg-pair / waypoint")
        else:
            dialog.setWindowTitle(f"{title} collision per leg")

        layout = QVBoxLayout(dialog)
        tw = QTableWidget(len(rows), len(headers), dialog)
        tw.setHorizontalHeaderLabels(headers)
        tw.verticalHeader().setVisible(False)
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                tw.setItem(r, c, QTableWidgetItem(val))
        try:
            tw.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.Stretch,
            )
        except Exception:
            try:
                tw.horizontalHeader().setSectionResizeMode(
                    0, QHeaderView.Stretch,
                )
            except Exception:
                pass
        layout.addWidget(tw)
        dialog.resize(560, 540)
        dialog.exec()
