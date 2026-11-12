"""Visualization mixin for the calculation runner.

Extracts the drift, powered-allision, and powered-grounding visualisation
methods so they can be composed into ``CalculationRunner`` without bloating
the main module.
"""

import copy
from typing import Any

import geopandas as gpd
import shapely.wkt as sw

try:
    from shapely import make_valid as shp_make_valid
except ImportError:
    shp_make_valid = None

from compute.data_preparation import prepare_traffic_lists, transform_to_utm
from geometries.get_drifting_overlap import DriftingOverlapVisualizer
from geometries.get_powered_overlap import PoweredOverlapVisualizer
from ui.show_geom_res import ShowGeomRes


# Default size for the powered / drifting matplotlib visualisation
# dialogs.  The .ui file ships with 721x534 which crams a 14x12-inch
# figure into a postage stamp; resizing at runtime overrides that
# without having to edit the .ui.
_VIZ_DIALOG_DEFAULT_SIZE = (1400, 900)
_VIZ_DIALOG_MIN_SIZE = (900, 600)


def _with_speed_restrictions(data: dict[str, Any]) -> dict[str, Any]:
    """``data`` with its speed restriction zones applied (on a copy).

    The powered View casts its own rays from the run's snapshot; Cat II
    must use the speed at the turning point like the model does
    (``compute/speed_restriction.py``).  Without zones -- or if they cannot
    be applied -- the data is used as it is.
    """
    from compute.speed_restriction import ZONES_KEY, apply_speed_restrictions
    if not data.get(ZONES_KEY):
        return data
    try:
        work = copy.deepcopy(data)
        apply_speed_restrictions(work)
        return work
    except Exception:  # nosec B110 - the View falls back to the stored speeds
        return data


def _prepare_viz_dialog(dialog) -> None:
    """Apply a larger default size + size grip to a visualisation dialog.

    The visualisation `.ui` is shared with the collision breakdown popup
    which deliberately stays small; we expand here so the matplotlib
    canvas has room to breathe and the user can drag the corner to grow
    further.
    """
    try:
        dialog.resize(*_VIZ_DIALOG_DEFAULT_SIZE)
        dialog.setMinimumSize(*_VIZ_DIALOG_MIN_SIZE)
        dialog.setSizeGripEnabled(True)
    except Exception:  # nosec B110 B112
        pass


def add_draught_bin_picker(dialog, bins: list[float], show) -> None:
    """Row above the powered grounding figure: which depth bin it shows.

    ``run_powered_grounding_model`` casts the rays once per depth bin (a ship
    grounds on every depth class not deeper than its draught, + squat;
    ``grounding_draught_bins``).  The figure shows one bin at a time, the
    deepest by default; picking another calls ``show(max_draft)``, which rebuilds
    the figure.  The table's frequencies are the model's, over every bin.
    """
    from qgis.PyQt.QtWidgets import QComboBox, QHBoxLayout, QLabel

    row = QHBoxLayout()
    label = QLabel()
    row.addWidget(label)
    combo = None
    if not bins:
        label.setText("No ship in the traffic draws enough water to ground on any depth area.")
    else:
        label.setText("Obstacles: depth areas down to")
        combo = QComboBox()
        for b in bins:
            combo.addItem(f"{b:g} m", b)
        combo.setCurrentIndex(len(bins) - 1)
        combo.setToolTip(
            "The model casts the rays once per depth bin: a ship grounds on every depth area "
            "not deeper than its draught (+ squat when ticked).  The figure shows one bin; "
            "the frequencies in the table are the model's, summed over every bin."
        )
        combo.currentIndexChanged.connect(lambda i: show(float(combo.itemData(i))))
        row.addWidget(combo)
    row.addStretch(1)
    layout = dialog.layout()
    if layout is not None:
        layout.insertLayout(0, row)
    dialog.draught_bin_combo = combo


class VisualizationMixin:
    """Mixin providing interactive visualisation dialogs.

    Expects the host class to expose ``self.p.main_widget`` (the parent
    QWidget used for dialogs) and ``self.canvas`` (the QGIS map canvas).

    ``self.powered_view_reports`` (set by the Run Analysis View for the
    selected run, from ``<stem>.powered.json``) feeds the Frequency and
    % of total columns of the powered dialogs.
    """

    def _powered_view_report(self, mode: str) -> dict[str, Any] | None:
        reports = getattr(self, 'powered_view_reports', None)
        if not isinstance(reports, dict):
            return None
        block = reports.get(mode)
        return block if isinstance(block, dict) else None

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
            except Exception:  # nosec B110 B112
                fixed = obj
            if fixed is not None:
                fixed_objects.append(fixed)
        transformed_objects = fixed_objects
        longest_length = max(line.length for line in transformed_lines)
        transformed_objects_gdf = [gpd.GeoDataFrame(geometry=[obj]) for obj in transformed_objects]

        # Create and show dialog
        dialog = ShowGeomRes(self.p.main_widget)
        _prepare_viz_dialog(dialog)
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
        dialog.exec()

    def run_powered_allision_visualization(self, data: dict[str, Any]) -> None:
        """Show an interactive powered allision visualisation dialog.

        Uses the same ``ShowGeomRes`` dialog as the drifting visualisation but
        populates it with shadow-aware ray-casting plots: Cat I rays along the
        leg for structures inside the lane and Cat II rays from the turning
        point for ships that miss a turn.
        """
        if not data.get('traffic_data') or not data.get('segment_data'):
            return
        try:
            max_draft = float(data.get('max_draft', 15.0))
        except (TypeError, ValueError):
            max_draft = 15.0

        dialog = ShowGeomRes(self.p.main_widget)
        _prepare_viz_dialog(dialog)
        PoweredOverlapVisualizer.show_in_dialog(
            dialog, _with_speed_restrictions(data), mode="allision", max_draft=max_draft,
            report=self._powered_view_report('allision'),
        )
        dialog.exec()

    def run_powered_grounding_visualization(self, data: dict[str, Any]) -> None:
        """Show an interactive powered grounding visualisation dialog.

        Uses the same ``ShowGeomRes`` dialog as the drifting visualisation but
        populates it with shadow-aware ray-casting plots: Cat I rays along the
        leg for shallow areas inside the lane and Cat II rays from the turning
        point for ships that miss a turn.
        """
        if not data.get('traffic_data') or not data.get('segment_data'):
            return
        from compute.powered_model import grounding_draught_bins

        view_data = _with_speed_restrictions(data)
        bins = grounding_draught_bins(view_data)
        report = self._powered_view_report('grounding')
        dialog = ShowGeomRes(self.p.main_widget)
        _prepare_viz_dialog(dialog)

        def _show(max_draft: float) -> None:
            PoweredOverlapVisualizer.show_in_dialog(
                dialog, view_data, mode="grounding", max_draft=max_draft, report=report,
            )

        add_draught_bin_picker(dialog, bins, _show)
        _show(bins[-1] if bins else -1.0)
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
        # Every depth class some ship of the traffic grounds on (draught,
        # + squat when the drifting Squat box is ticked), not a fixed 15 m.
        from compute.powered_model import grounding_draught_bins
        from compute.squat import KEY_DRIFTING
        bins = grounding_draught_bins(data, squat_key=KEY_DRIFTING)
        max_draft = bins[-1] if bins else -1.0

        lines, distributions, weights, line_names = prepare_traffic_lists(data)

        # Filter depths to grounding hazards (depth <= ship draft).
        hazards = []
        for entry in depths:
            try:
                _did, val, wkt = entry
                if float(val) <= max_draft:
                    hazards.append(sw.loads(wkt))
            except Exception:  # nosec B110 B112
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
            except Exception:  # nosec B110 B112
                fixed = obj
            if fixed is not None:
                fixed_objects.append(fixed)
        transformed_objects = fixed_objects
        if not transformed_objects:
            return
        longest_length = max(line.length for line in transformed_lines)
        objects_gdf = [gpd.GeoDataFrame(geometry=[obj]) for obj in transformed_objects]
        dialog = ShowGeomRes(self.p.main_widget)
        _prepare_viz_dialog(dialog)
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
        from compute.collision_breakdown import build_breakdown_rows
        report = getattr(self, 'collision_report', None) or {}

        wanted = encounter_type
        headers, rows = build_breakdown_rows(
            report, wanted, segment_data=getattr(self.p, 'segment_data', None))
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
        except Exception:  # nosec B110 B112
            try:
                tw.horizontalHeader().setSectionResizeMode(
                    0, QHeaderView.ResizeMode.Stretch,
                )
            except Exception:  # nosec B110 B112
                pass
        layout.addWidget(tw)
        dialog.resize(560, 540)
        dialog.exec()
