"""Interactive drifting-overlap visualiser (top-level facade).

The Drift-grounding / Drift-allision **View** dialog opens an
interactive 3-panel matplotlib figure inside a QGIS dock dialog:

* ``ax1`` (top-left) -- route legs, click to pick one,
* ``ax2`` (top-right) -- the 8 swept drift-direction polygons of the
  picked leg,
* ``ax3`` (bottom)  -- weighted PDF + failure-remains curve + the
  intersection extent of the polygon clicked in ``ax2``.

On the right, a sortable ``QTableWidget`` lists every
``(leg, direction)`` pair with its contribution to drift allision /
grounding probability so the user can pick a row directly.

This module hosts :class:`DriftingOverlapVisualizer` which orchestrates
those three axes plus the sidebar.  Pure-data helpers
(``create_polygon_from_line``, ``compute_min_distance_by_object``, ...)
live in :mod:`geometries.drift_overlap_geometry`; the bottom-axis plot
in :mod:`geometries.drift_overlap_plot`; the sidebar table builder in
:mod:`geometries.drift_overlap_sidebar`.

Backwards compatibility: the old top-level functions are re-exported
from here so callers (compute / tests) don't need to update imports.
"""
from __future__ import annotations

from typing import Any

import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from geometries.drift_overlap_geometry import (
    compare_polygons_with_objs,
    compute_coverages_and_distances,
    compute_min_distance_by_object,
    create_polygon_from_line,
    directional_distances_to_points,
    directional_min_distance_reverse_ray,
    estimate_weighted_overlap,
    extend_polygon_in_directions,
)
from geometries.drift_overlap_plot import visualize
from geometries.drift_overlap_sidebar import (
    DIRECTION_LABELS,
    build_overlap_sidebar,
)
from ui.show_geom_res import ShowGeomRes

__all__ = [
    'DriftingOverlapVisualizer',
    'compare_polygons_with_objs',
    'compute_coverages_and_distances',
    'compute_min_distance_by_object',
    'create_polygon_from_line',
    'directional_distances_to_points',
    'directional_min_distance_reverse_ray',
    'estimate_weighted_overlap',
    'extend_polygon_in_directions',
    'visualize',
]


class DriftingOverlapVisualizer:
    """Drives the 3-panel matplotlib figure for the View dialog."""

    # Re-export so external callers that referenced
    # ``DriftingOverlapVisualizer._DIRECTION_LABELS`` still work.
    _DIRECTION_LABELS = DIRECTION_LABELS

    def __init__(
        self,
        fig: Figure,
        ax1: Axes,
        ax2: Axes,
        ax3: Axes,
        lines: list[LineString],
        line_names: list[str],
        objs_gdf_list: list[gpd.GeoDataFrame],
        distributions: list[list[Any]],
        weights: list[list[float]],
        data: dict[str, Any],
        distance: float = 50_000,
    ) -> None:
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.lines = lines
        self.line_names = line_names
        self.objs_gdf_list = objs_gdf_list
        self.distributions = distributions
        self.weights = weights
        self.data = data
        self.distance = distance

        self.current_line: LineString | None = None
        self.current_base_polygon: Polygon | None = None
        self.current_extended_polygons: list[BaseGeometry] | None = None
        self.current_centre_lines: list[LineString] | None = None
        self.current_coverages: list[float] | None = None
        self.current_distances: list[np.ndarray] | None = None
        self.current_weight: list[float] | None = None
        self.current_distribution: list[Any] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_visualization(self) -> None:
        self.setup_plot()
        self.connect_events()
        self.simulate_initial_selection()

    def setup_plot(self) -> None:
        assert isinstance(self.ax1, Axes)
        for line, name in zip(self.lines, self.line_names):
            x, y = line.xy
            # picker=15 gives a click tolerance that's reliable on
            # high-DPI displays (default 5 px is too tight).
            self.ax1.plot(x, y, label=name, picker=15, linewidth=2)
        self.ax1.set_title("Route legs (select via the table on the right)")
        self.ax2.set_title("Drift directions")
        self.ax3.set_title("Probability Density Function (PDF)")

    def connect_events(self) -> None:
        self.fig.canvas.mpl_connect('pick_event', self.on_line_click)
        self.fig.canvas.mpl_connect(
            'button_press_event', self.on_polygon_click,
        )
        # Fallback so clicks that miss the thin leg line still pick the
        # nearest leg.  Without it the user has to land on the exact
        # pixels of a 2-pt matplotlib line.
        self.fig.canvas.mpl_connect(
            'button_press_event', self._on_ax1_click_fallback,
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_ax1_click_fallback(self, event: Any) -> None:
        if event.inaxes != self.ax1:
            return
        if event.xdata is None or event.ydata is None:
            return
        nearest_idx = self._find_nearest_leg(event.xdata, event.ydata)
        if nearest_idx is None:
            return

        line_artists = self.ax1.get_lines()
        if nearest_idx < len(line_artists):
            class _MockPickEvent:
                def __init__(self, artist):
                    self.artist = artist
            self.on_line_click(_MockPickEvent(line_artists[nearest_idx]))

    def _find_nearest_leg(self, click_x: float, click_y: float) -> int | None:
        nearest_idx: int | None = None
        nearest_dist = float('inf')
        click = np.array([click_x, click_y])
        for i, line in enumerate(self.lines):
            try:
                xs, ys = line.xy
            except Exception:
                continue
            for x, y in zip(xs, ys):
                d = float(np.linalg.norm(np.array([x, y]) - click))
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_idx = i
        if nearest_idx is None:
            return None
        # Threshold: 5 % of the larger axis range.
        xlim = self.ax1.get_xlim()
        ylim = self.ax1.get_ylim()
        threshold = 0.05 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        if nearest_dist > threshold:
            return None
        return nearest_idx

    def on_line_click(self, event: Any) -> None:
        idx = self._get_selected_line_index(event)
        if idx is None:
            return
        self._select_leg(idx)

    def on_polygon_click(self, event: Any) -> None:
        if event.inaxes != self.ax2:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.current_extended_polygons is None:
            return
        for i, polygon in enumerate(self.current_extended_polygons):
            if polygon.contains(Point(event.xdata, event.ydata)):
                self._render_polygon_panel(i)
                self.fig.canvas.draw()
                return

    # ------------------------------------------------------------------
    # Leg / polygon selection
    # ------------------------------------------------------------------
    def _select_leg(self, idx: int, direction_idx: int = 0) -> None:
        """Programmatic equivalent of clicking on a leg in ``ax1``."""
        if idx < 0 or idx >= len(self.lines):
            return
        self._set_current_line_state(idx)
        if (self.current_line is None
                or self.current_distribution is None
                or self.current_weight is None):
            return

        self.ax2.clear()
        self.ax3.clear()
        self.current_base_polygon = create_polygon_from_line(
            self.current_line,
            self.current_distribution,
            self.current_weight,
        )
        self.current_extended_polygons, self.current_centre_lines = (
            extend_polygon_in_directions(
                self.current_base_polygon, self.distance,
            )
        )
        results = compare_polygons_with_objs(
            self.current_extended_polygons, self.objs_gdf_list,
        )
        covered = self._update_coverages_and_distances(results)
        self._plot_polygons(covered)
        self._plot_objects()
        self._auto_select_polygon(direction_idx)
        self._sync_sidebar(idx, direction_idx)

    def _auto_select_polygon(self, direction_idx: int) -> None:
        """Auto-click polygon ``direction_idx`` so the PDF is drawn.

        Without this the bottom panel goes blank after every leg
        switch and the user has to click ``ax2`` separately to bring
        it back.
        """
        if not self.current_extended_polygons:
            return
        if not (0 <= direction_idx < len(self.current_extended_polygons)):
            return
        if self.current_coverages is None:
            return
        if len(self.current_coverages) < len(self.current_extended_polygons):
            return
        poly = self.current_extended_polygons[direction_idx]
        try:
            centroid = poly.centroid
        except Exception:
            return

        class _MockButtonEvent:
            def __init__(self, x, y, inaxes):
                self.xdata = x
                self.ydata = y
                self.inaxes = inaxes

        self.on_polygon_click(_MockButtonEvent(
            centroid.x, centroid.y, self.ax2,
        ))

    def _sync_sidebar(self, leg_idx: int, direction_idx: int) -> None:
        """Reflect the current ``(leg, direction)`` in the sidebar."""
        sidebar = getattr(self, 'sidebar', None)
        if sidebar is None:
            return
        from qgis.PyQt.QtCore import Qt as _Qt
        target_row = -1
        for r in range(sidebar.rowCount()):
            item = sidebar.item(r, 0)
            if item is None:
                continue
            payload = item.data(_Qt.ItemDataRole.UserRole)
            if (
                isinstance(payload, tuple)
                and len(payload) == 2
                and int(payload[0]) == leg_idx
                and int(payload[1]) == direction_idx
            ):
                target_row = r
                break
        if target_row < 0:
            return
        blocked = sidebar.blockSignals(True)
        try:
            sidebar.selectRow(target_row)
        finally:
            sidebar.blockSignals(blocked)

    # ------------------------------------------------------------------
    # Coverage / distance computation
    # ------------------------------------------------------------------
    def _obstacle_distances_per_polygon(
        self, results: dict[str, list[list[bool]]],
    ) -> list[np.ndarray]:
        """Distances from contributing obstacle vertices to the leg line.

        ``compute_coverages_and_distances`` returns distances measured
        from intersection-polygon vertices to a single point on the
        centre-line.  When the drift corridor is a buffer that crosses
        the leg, those vertices include points right on the leg
        (distance ~0) even when the obstacle itself sits 10-20 km
        away.  The visualisation wants the *obstacle's* offset from
        the leg, so we sample obstacle exterior coords and project
        each onto the leg with shapely's perpendicular distance.
        """
        from shapely.geometry import MultiPolygon as _MultiPolygon
        if (
            self.current_extended_polygons is None
            or self.current_line is None
        ):
            return []
        out: list[np.ndarray] = []
        leg_line = self.current_line
        for i in range(len(self.current_extended_polygons)):
            dists: list[float] = []
            for gi, gdf in enumerate(self.objs_gdf_list or []):
                for j, obj in enumerate(gdf.geometry):
                    try:
                        if not results[f"Polygon_{i}"][gi][j]:
                            continue
                    except Exception:
                        continue
                    geoms_to_sample = (
                        list(obj.geoms)
                        if isinstance(obj, _MultiPolygon)
                        else [obj]
                    )
                    for g in geoms_to_sample:
                        try:
                            coords = (
                                list(g.exterior.coords)
                                if hasattr(g, 'exterior')
                                else list(g.coords)
                            )
                        except Exception:
                            continue
                        for cx, cy in coords:
                            try:
                                d = float(
                                    Point(cx, cy).distance(leg_line)
                                )
                            except Exception:
                                continue
                            if np.isfinite(d):
                                dists.append(d)
            out.append(np.array(dists) if dists else np.array([]))
        return out

    def _update_coverages_and_distances(
        self, results: dict[str, list[list[bool]]],
    ) -> list[bool]:
        """Reshape flat per-(polygon, gdf, obj) arrays into per-polygon.

        ``compute_coverages_and_distances`` returns FLAT lists indexed
        by ``(polygon, gdf, obj)``.  Both ``on_polygon_click`` and the
        sidebar consume per-polygon aggregates, so we reshape: sum the
        per-object coverages and concatenate the per-object distance
        arrays.  Then override per-polygon distances with
        obstacle-vs-leg distances for the visualisation -- the PDF
        x-axis becomes "perpendicular distance from leg" instead of
        "Euclidean from polygon vertex to centerline midpoint".
        """
        flat_coverages, flat_distances, covered = (
            compute_coverages_and_distances(
                self.current_extended_polygons,
                self.current_centre_lines,
                self.current_distribution,
                self.current_weight,
                self.objs_gdf_list,
                results,
            )
        )
        n_polys = len(self.current_extended_polygons or [])
        n_total_objs = sum(len(gdf) for gdf in (self.objs_gdf_list or []))
        self.current_coverages = []
        self.current_distances = []
        for i in range(n_polys):
            cov, dists = self._aggregate_polygon_slice(
                i, n_total_objs, flat_coverages, flat_distances,
            )
            self.current_coverages.append(cov)
            self.current_distances.append(dists)
        self._override_with_obstacle_distances(results)
        return covered

    @staticmethod
    def _aggregate_polygon_slice(
        i: int,
        n_total_objs: int,
        flat_coverages: list[float],
        flat_distances: list[Any],
    ) -> tuple[float, np.ndarray]:
        """Sum coverage + concat distances for one polygon's slice."""
        if n_total_objs == 0:
            return 0.0, np.array([])
        start = i * n_total_objs
        end = start + n_total_objs
        poly_cov = float(
            sum(float(c or 0.0) for c in flat_coverages[start:end])
        )
        # ``compute_coverages_and_distances`` appends a 0-d scalar
        # ``np.ndarray([])`` for non-intersecting objects and a 1-d
        # array for intersecting ones.  Coerce to 1-d before
        # filtering so ``np.concatenate`` doesn't reject the mix.
        non_empty: list[np.ndarray] = []
        for d in flat_distances[start:end]:
            arr = np.atleast_1d(np.asarray(d))
            if arr.size > 0:
                non_empty.append(arr)
        poly_dists = (
            np.concatenate(non_empty) if non_empty else np.array([])
        )
        return poly_cov, poly_dists

    def _override_with_obstacle_distances(
        self, results: dict[str, list[list[bool]]],
    ) -> None:
        try:
            obs_dists = self._obstacle_distances_per_polygon(results)
            if (
                obs_dists
                and self.current_distances is not None
                and len(obs_dists) == len(self.current_distances)
            ):
                self.current_distances = obs_dists
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug(
                f"Could not compute obstacle distances: {exc}"
            )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _plot_polygons(self, covered: list[bool]) -> None:
        gpd.GeoSeries(self.current_base_polygon).plot(
            ax=self.ax2, color='blue', alpha=0.5,
        )
        assert self.current_extended_polygons is not None
        for i, polygon in enumerate(self.current_extended_polygons):
            assert isinstance(polygon, Polygon)
            alpha = 0.3 if covered[i] else 0.1
            mpl_polygon = MplPolygon(
                np.array(polygon.exterior.coords),
                closed=True, alpha=alpha,
            )
            self.ax2.add_patch(mpl_polygon)

    def _plot_objects(self) -> None:
        for objs_gdf in self.objs_gdf_list:
            objs_gdf.plot(
                ax=self.ax2, color='red', alpha=0.7, label='Objects',
            )
        self.ax2.set_title("Drift directions")
        self.fig.canvas.draw()

    def _render_polygon_panel(self, i: int) -> None:
        """Render the bottom PDF panel for polygon ``i``."""
        cov = (
            self.current_coverages[i]
            if (
                self.current_coverages is not None
                and i < len(self.current_coverages)
            )
            else None
        )
        if (
            self.current_distances is not None
            and self.current_distribution is not None
            and cov is not None
            and i < len(self.current_distances)
            and getattr(self.current_distances[i], 'size', 0) > 0
        ):
            visualize(
                self.ax3,
                distances=self.current_distances[i],
                distributions=self.current_distribution,
                weights=self.current_weight,
                weighted_overlap=cov,
                data=self.data,
            )
            return
        self.ax3.clear()
        self.ax3.text(
            0.5, 0.5,
            "No intersection between this drift direction\n"
            "and any object/depth -- nothing to plot.",
            transform=self.ax3.transAxes,
            ha='center', va='center',
            fontsize=10, color='gray',
        )

    def _get_selected_line_index(self, event: Any) -> int | None:
        for i, name in enumerate(self.line_names):
            if event.artist.get_label() == name:
                return i
        return None

    def _set_current_line_state(self, idx: int) -> None:
        self.current_line = self.lines[idx]
        self.current_weight = self.weights[idx]
        self.current_distribution = self.distributions[idx]

    def simulate_initial_selection(self) -> None:
        """Auto-pick the last leg + first polygon to seed the panels."""

        class _MockPickEvent:
            def __init__(self, artist: Any):
                self.artist = artist

        class _MockButtonEvent:
            def __init__(self, x: float, y: float, inaxes: Axes | None):
                self.xdata = x
                self.ydata = y
                self.inaxes = inaxes

        if self.ax1.lines:
            line_artist = self.ax1.lines[-1]
            self.on_line_click(_MockPickEvent(line_artist))
        if (
            self.current_extended_polygons is not None
            and self.current_coverages is not None
            and len(self.current_coverages)
            >= len(self.current_extended_polygons)
        ):
            centroid = self.current_extended_polygons[0].centroid
            self.on_polygon_click(_MockButtonEvent(
                centroid.x, centroid.y, self.ax2,
            ))

    # ------------------------------------------------------------------
    # Dialog factory
    # ------------------------------------------------------------------
    @classmethod
    def show_in_dialog(
        cls,
        dialog: ShowGeomRes,
        lines: list[LineString],
        line_names: list[str],
        objs_gdf_list: list[gpd.GeoDataFrame],
        distributions: list[list[Any]],
        weights: list[list[float]],
        data: dict[Any],
        distance: float = 50_000,
        drifting_report: dict[str, Any] | None = None,
        accident_kind: str = 'allision',
    ) -> "DriftingOverlapVisualizer":
        """Build the dialog content: figure, sidebar, signal wiring."""
        cls._clear_dialog_layout(dialog)
        fig, ax1, ax2, ax3 = cls._build_figure_axes()
        canvas = FigureCanvas(fig)

        sidebar = build_overlap_sidebar(
            line_names, drifting_report, accident_kind,
        )
        cls._install_canvas_and_sidebar(dialog, canvas, sidebar)

        dov = cls(
            fig, ax1, ax2, ax3, lines, line_names, objs_gdf_list,
            distributions, weights, data, distance=distance,
        )
        dov.canvas = canvas
        dov.sidebar = sidebar

        dov._wire_sidebar(sidebar)
        # Set up axes + click handlers but skip ``simulate_initial_selection``
        # -- the sidebar selection below drives the initial leg pick so
        # there's no double-draw race.
        dov.setup_plot()
        dov.connect_events()
        dov._initial_select_via_sidebar(sidebar, len(lines))

        try:
            canvas.draw()
        except Exception:
            pass
        return dov

    @staticmethod
    def _clear_dialog_layout(dialog: ShowGeomRes) -> None:
        layout = dialog.result_layout
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            layout.removeWidget(widget)
            widget.deleteLater()

    @staticmethod
    def _build_figure_axes() -> tuple[Figure, Axes, Axes, Axes]:
        """Build the 3-panel matplotlib figure used by the dialog.

        ``ax2`` (drift directions) gets the wider half of the top row
        because the legend is no longer on ``ax1``; ``ax1`` only needs
        room for the leg lines since the sidebar table drives leg picks.
        """
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(
            2, 2, height_ratios=[1, 1.2], width_ratios=[1, 2],
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        for ax in (ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(
                left=False, bottom=False,
                labelleft=False, labelbottom=False,
            )
        return fig, ax1, ax2, ax3

    @staticmethod
    def _install_canvas_and_sidebar(
        dialog: ShowGeomRes, canvas: FigureCanvas, sidebar,
    ) -> None:
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(canvas)
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        dialog.result_layout.addWidget(splitter)

    def _wire_sidebar(self, sidebar) -> None:
        from qgis.PyQt.QtCore import Qt as _Qt
        n_dirs = len(self._DIRECTION_LABELS)

        def _on_sidebar_selection_changed():
            rows = {idx.row() for idx in sidebar.selectedIndexes()}
            if not rows:
                return
            row = next(iter(rows))
            # Sorting scrambles rows, so we read the (leg_idx, dir_idx)
            # tuple stored on the Leg cell's UserRole.
            leg_item = sidebar.item(row, 0)
            payload = (
                leg_item.data(_Qt.ItemDataRole.UserRole)
                if leg_item is not None else None
            )
            if isinstance(payload, tuple) and len(payload) == 2:
                leg_idx, dir_idx = int(payload[0]), int(payload[1])
            else:
                leg_idx, dir_idx = divmod(row, n_dirs)
            self._select_leg(leg_idx, direction_idx=dir_idx)

        sidebar.itemSelectionChanged.connect(_on_sidebar_selection_changed)

    def _initial_select_via_sidebar(
        self, sidebar, n_lines: int,
    ) -> None:
        """Trigger the initial selection through the sidebar.

        Falls back to ``simulate_initial_selection`` if the sidebar
        signal didn't fire (e.g. zero-row table).
        """
        try:
            last_leg = max(0, n_lines - 1)
            sidebar.selectRow(last_leg * len(self._DIRECTION_LABELS))
        except Exception:
            pass
        if self.current_extended_polygons is None:
            try:
                self.simulate_initial_selection()
            except Exception:
                pass
