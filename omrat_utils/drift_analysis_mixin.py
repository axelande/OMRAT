"""Drift-corridor analysis runner, factored out of ``omrat.OMRAT``.

This mixin owns everything that fires when the user clicks
**Run Drift Analysis** on the dock's Drift Analysis tab:

* kicks off a ``DriftCorridorTask`` in QGIS's background-task manager,
* updates the status label as progress events come in,
* on success, adds one categorised polygon layer per leg with the 8
  direction polygons coloured.

The pure corridor-generation maths lives in
:class:`compute.drift_corridor.DriftCorridorGenerator` and the QGIS
task wrapper in :class:`compute.drift_corridor.DriftCorridorTask`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCategorizedSymbolRenderer,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsFillSymbol,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
    QgsRendererCategory,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QMetaType

if TYPE_CHECKING:
    pass


# Direction -> hex colour for the categorised renderer.  Kept at module
# level so tests / external callers can introspect the palette.
_DIRECTION_COLOURS: dict[str, str] = {
    'N':  '#e41a1c',
    'NE': '#377eb8',
    'E':  '#4daf4a',
    'SE': '#984ea3',
    'S':  '#ff7f00',
    'SW': '#ffff33',
    'W':  '#a65628',
    'NW': '#f781bf',
}


class DriftAnalysisMixin:
    """Drift-analysis tab slots and corridor-layer rendering."""

    # ------------------------------------------------------------------
    # Public slot
    # ------------------------------------------------------------------
    def run_drift_analysis(self) -> None:
        """Run drift corridor analysis as a background task."""
        from compute.drift_corridor import (
            DriftCorridorGenerator,
            DriftCorridorTask,
        )
        try:
            self._clear_drift_corridor_layers()

            depth_threshold = float(self.main_widget.leDepthThreshold.text() or 10)
            height_threshold = float(self.main_widget.leHeightThreshold.text() or 10)

            self.main_widget.label_drift_status.setText("Collecting data...")
            self.main_widget.pbRunDriftAnalysis.setEnabled(False)

            # Qt widgets can only be touched from the main thread, so we
            # pre-collect the input data here and feed only plain dicts
            # into the background task.
            generator = DriftCorridorGenerator(self)
            generator.precollect_data(depth_threshold, height_threshold)

            self.main_widget.label_drift_status.setText(
                "Starting corridor generation...",
            )

            task = DriftCorridorTask(
                "Generating Drift Corridors",
                generator,
                depth_threshold,
                height_threshold,
            )
            task.progress_updated.connect(self._on_drift_progress)
            task.corridors_generated.connect(self._on_drift_complete)
            task.generation_failed.connect(self._on_drift_failed)
            self._drift_task = task

            task_manager = QgsApplication.taskManager()
            if task_manager is not None:
                task_manager.addTask(task)
            else:
                # Fallback for environments without a task manager
                # (unit tests, headless invocations).
                self.main_widget.label_drift_status.setText(
                    "Running (no task manager)...",
                )
                corridors = generator.generate_corridors(
                    depth_threshold, height_threshold,
                )
                self._on_drift_complete(corridors)

        except Exception as exc:
            self.main_widget.label_drift_status.setText(f"Error: {exc}")
            self.main_widget.pbRunDriftAnalysis.setEnabled(True)
            QgsMessageLog.logMessage(
                f"Drift analysis failed: {exc}", "OMRAT", Qgis.Critical,
            )

    # ------------------------------------------------------------------
    # Task signal handlers
    # ------------------------------------------------------------------
    def _on_drift_progress(self, completed: int, total: int, message: str) -> None:
        """Handle progress updates from drift corridor task."""
        if total > 0:
            pct = int((completed / total) * 100)
            self.main_widget.label_drift_status.setText(f"{message} ({pct}%)")
        else:
            self.main_widget.label_drift_status.setText(message)

    def _on_drift_complete(self, corridors: list) -> None:
        """Handle successful completion of drift corridor generation."""
        self.main_widget.pbRunDriftAnalysis.setEnabled(True)

        if not corridors:
            self.main_widget.label_drift_status.setText(
                "No corridors generated. Add routes first.",
            )
            return

        self._create_corridor_layers(corridors)
        num_legs = len(set(c['leg_index'] for c in corridors))
        self.main_widget.label_drift_status.setText(
            f"Generated {len(corridors)} corridors for {num_legs} legs"
        )

    def _on_drift_failed(self, error_msg: str) -> None:
        """Handle drift corridor generation failure."""
        self.main_widget.pbRunDriftAnalysis.setEnabled(True)
        self.main_widget.label_drift_status.setText(f"Error: {error_msg}")
        QgsMessageLog.logMessage(
            f"Drift analysis failed: {error_msg}", "OMRAT", Qgis.Critical,
        )

    # ------------------------------------------------------------------
    # Layer rendering
    # ------------------------------------------------------------------
    def _clear_drift_corridor_layers(self) -> None:
        """Remove previously-created drift-corridor layers from the
        project so a fresh run doesn't pile up duplicates."""
        for layer in getattr(self, 'drift_corridor_layers', []):
            try:
                project = QgsProject.instance()
                if project is not None and layer is not None:
                    project.removeMapLayer(layer.id())
            except Exception:
                pass
        self.drift_corridor_layers.clear()

    def _create_corridor_layers(self, corridors: list) -> None:
        """Build one polygon memory layer per leg from the corridor list."""
        legs: dict[int, list[dict]] = {}
        for corridor in corridors:
            legs.setdefault(corridor['leg_index'], []).append(corridor)

        for leg_idx, leg_corridors in legs.items():
            vl = self._build_corridor_layer_for_leg(leg_idx, leg_corridors)
            if vl is None:
                continue
            self._style_corridor_layer_categorized(vl)
            self.drift_corridor_layers.append(vl)
            project = QgsProject.instance()
            if project is not None:
                project.addMapLayer(vl)

    def _build_corridor_layer_for_leg(
        self,
        leg_idx: int,
        leg_corridors: list[dict],
    ) -> QgsVectorLayer | None:
        layer_name = f"Drift Corridors - Leg {leg_idx + 1}"
        vl = QgsVectorLayer("Polygon?crs=EPSG:4326", layer_name, "memory")

        fields = QgsFields()
        fields.append(QgsField("direction", QMetaType.Type.QString))
        fields.append(QgsField("angle", QMetaType.Type.Int))
        fields.append(QgsField("leg_index", QMetaType.Type.Int))

        provider = vl.dataProvider()
        if provider is None:
            return None
        provider.addAttributes(fields.toList())
        vl.updateFields()

        for corridor in leg_corridors:
            poly = corridor['polygon']
            if poly.is_empty:
                continue
            attrs = [
                corridor['direction'],
                corridor['angle'],
                corridor['leg_index'],
            ]
            if poly.geom_type == 'MultiPolygon':
                geoms = list(poly.geoms)
            else:
                geoms = [poly]
            for geom in geoms:
                feat = QgsFeature(fields)
                feat.setGeometry(QgsGeometry.fromWkt(geom.wkt))
                feat.setAttributes(attrs)
                provider.addFeature(feat)
        vl.updateExtents()
        return vl

    def _style_corridor_layer_categorized(
        self, layer: QgsVectorLayer,
    ) -> None:
        """Apply categorized styling to a corridor layer based on the
        ``direction`` field."""
        categories = []
        for direction, color in _DIRECTION_COLOURS.items():
            symbol = QgsFillSymbol.createSimple({
                'color': color,
                'outline_color': color,
                'outline_width': '0.5',
            })
            if symbol:
                symbol.setOpacity(0.3)
            categories.append(
                QgsRendererCategory(direction, symbol, direction),
            )
        renderer = QgsCategorizedSymbolRenderer('direction', categories)
        layer.setRenderer(renderer)
        layer.triggerRepaint()
