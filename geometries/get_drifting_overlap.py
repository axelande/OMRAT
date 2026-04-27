from typing import Any, Callable, Sequence
import geopandas as gpd
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt
from matplotlib.backend_bases import PickEvent
import numpy as np
from qgis.PyQt.QtWidgets import QLabel, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.gridspec as gridspec
from shapely.affinity import translate
from shapely.geometry import LineString, Polygon, Point, MultiPolygon
from shapely.geometry.base import BaseGeometry
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from compute.basic_equations import get_not_repaired
from ui.show_geom_res import ShowGeomRes


def create_polygon_from_line(
    line: LineString,
    distributions: list[Any],
    weights: list[float]
) -> Polygon:
    """
    Create a polygon from a LineString based on multiple distributions and their weights.
    """
    # Ensure weights sum to 1
    weights = np.array(weights) / np.sum(weights)

    # Calculate the weighted mean (mu) and standard deviation (std) for the combined distribution
    weighted_mu = sum(weight * dist.mean() for dist, weight in zip(distributions, weights))
    weighted_std = np.sqrt(sum(weight * (dist.std() ** 2) for dist, weight in zip(distributions, weights)))

    # Translate the line by the weighted mean
    moved_line = translate(line, xoff=0, yoff=weighted_mu)

    # Calculate the range for 99.9999% coverage (±4.89 standard deviations)
    coverage_range = 4.89 * weighted_std

    # Create a buffer around the moved line
    polygon = moved_line.buffer(coverage_range)
    return polygon


def extend_polygon_in_directions(
    polygon: Polygon,
    distance: float
) -> tuple[list[BaseGeometry], list[LineString]]:
    """
    Extend a polygon in 8 directions (0°, 45°, ..., 315°) by a given distance,
    creating 8 separate polygons that span from the original polygon to the extended areas.
    """
    extended_polygons: list[BaseGeometry] = []  # List to store the 8 separate polygons
    centre_lines: list[LineString] = []

    # Degenerate/empty geometry can occur for pathological distributions.
    # Return empty placeholders rather than crashing downstream processing.
    if polygon is None or polygon.is_empty:
        return [Polygon()] * 8, [LineString()] * 8

    for angle in range(0, 360, 45):
        # Translate the polygon in the given direction
        dx = distance * np.cos(np.radians(angle))
        dy = distance * np.sin(np.radians(angle))
        translated_polygon = translate(polygon, xoff=dx, yoff=dy)

        # Create a polygon that spans from the original polygon to the translated polygon
        connecting_polygon = polygon.union(translated_polygon).convex_hull
        extended_polygons.append(connecting_polygon)
        try:
            c0 = polygon.representative_point()
            c1 = translated_polygon.representative_point()
            centre_lines.append(LineString([(c0.x, c0.y), (c1.x, c1.y)]))
        except Exception:
            centre_lines.append(LineString())

    return extended_polygons, centre_lines


def compare_polygons_with_objs(
    extended_polygons: list[BaseGeometry],
    objs_gdf_list: list[gpd.GeoDataFrame]
) -> dict[str, list[list[bool]]]:
    """
    Compare the extended polygons with the objects in a list of GeoDataFrames.

    Parameters:
    - extended_polygons: List of polygons to compare.
    - objs_gdf_list: List of GeoDataFrames containing objects.

    Returns:
    - results: Dictionary with overlap results for each polygon and GeoDataFrame.
    """
    results: dict[str, Any] = {}
    for i, polygon in enumerate(extended_polygons):
        results[f"Polygon_{i}"] = []
        for objs_gdf in objs_gdf_list:
            intersects = objs_gdf.intersects(polygon)
            results[f"Polygon_{i}"].append(intersects.tolist())
    return results


def estimate_weighted_overlap(
    intersection: BaseGeometry,
    line: LineString,
    distributions: list[Any],
    weights: list[float]
) -> tuple[float, np.ndarray]:
    """
    Estimate the weighted overlap of the intersection area based on multiple distributions.

    Parameters:
    - intersection: The intersection Polygon between the coverage polygon and the object.
    - line: The original or extended LineString.
    - distributions: A list of distribution functions (e.g., norm, uniform).
    - weights: A list of weights corresponding to each distribution.

    Returns:
    - Weighted percentage of the intersection area.
    """
    # Ensure weights sum to 1
    weights = np.array(weights) / np.sum(weights)

    # Find the closest point on the line to the intersection polygon
    closest_point = line.interpolate(line.project(intersection.centroid))
    
    if isinstance(intersection, Polygon):

        # Sample points within the intersection polygon
        sample_points = np.array(intersection.exterior.coords)
    
    elif isinstance(intersection, MultiPolygon):
        sample_points = np.vstack([np.array(poly.exterior.coords) for poly in intersection.geoms])
    else:
        raise ValueError("Unkown geom type type")
    sample_points = np.atleast_2d(sample_points)
    # Calculate the distance of each point from the closest point on the line
    distances = np.sqrt((sample_points[:, 0] - closest_point.x)**2 + (sample_points[:, 1] - closest_point.y)**2)

    # Calculate the combined probability density
    combined_probabilities = np.zeros_like(distances)
    for dist, weight in zip(distributions, weights):
        combined_probabilities += weight * dist.pdf(distances)

    # Calculate the weighted overlap as the sum of combined probabilities
    weighted_overlap = combined_probabilities.sum() * 100  # Convert to percentage
    return weighted_overlap, distances


def compute_coverages_and_distances(
    extended_polygons: list[BaseGeometry],
    centre_lines: list[LineString],
    distributions: list[Any],
    weights: list[float],
    objs_gdf_list: list[gpd.GeoDataFrame],
    results: dict[str, list[list[bool]]],
) -> tuple[list[float], list[Any], list[bool]]:
    """Compute weighted coverage + distance arrays per (polygon, object).

    Pure-data extract of ``DriftingOverlapVisualizer._update_coverages_and_distances``
    so the loop logic can be tested in isolation.

    Returns
    -------
    (coverages, distances, covered)
        - coverages: flat list, one entry per (polygon, gdf, obj) triple.
        - distances: flat list of np.ndarray distance arrays, same length.
        - covered: per-polygon flag (any object intersected).
    """
    coverages: list[float] = []
    distances: list[Any] = []
    covered: list[bool] = []
    for i, polygon in enumerate(extended_polygons):
        covered.append(False)
        for objs_gdf_idx, objs_gdf in enumerate(objs_gdf_list):
            for j, obj in enumerate(objs_gdf.geometry):
                if results[f"Polygon_{i}"][objs_gdf_idx][j]:
                    intersection = polygon.intersection(obj)
                    coverage, dists = estimate_weighted_overlap(
                        intersection, centre_lines[i], distributions, weights
                    )
                    coverages.append(coverage)
                    distances.append(dists)
                    covered[i] = True
                else:
                    coverages.append(0)
                    distances.append(np.ndarray([]))
    return coverages, distances, covered


def directional_distances_to_points(
    points: np.ndarray,
    leg: LineString,
    compass_angle_deg: float,
    use_leg_offset: bool = False,
) -> np.ndarray:
    """Return per-point along-drift distances from ``leg``, vectorised.

    For each point ``p`` in ``points``, casts a reverse ray against
    ``compass_angle_deg`` and finds the along-drift distance to the first
    intersection with ``leg``.  Points whose reverse ray misses every
    segment (ray parallel to leg, point outside the lateral band, etc.)
    fall back to :func:`directional_distance_to_point_from_offset_leg` for
    robust endpoint / collinear handling.

    Parameters
    ----------
    points
        ``(K, 2)`` array of world-frame ``(x, y)`` coordinates.
    leg
        Leg :class:`~shapely.geometry.LineString`.
    compass_angle_deg
        Drift direction (0 = N, 90 = E, ...).
    use_leg_offset
        Forwarded to the scalar fallback when a point misses every segment.
        The vectorised path only supports the default (centerline) case,
        which matches OMRAT's runtime ``mean_offset_m=0.0``.

    Returns
    -------
    np.ndarray
        ``(K,)`` array of distances, ``+inf`` for points that miss both the
        vectorised path and the scalar fallback.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty(0)
    if points.ndim == 1:
        points = points.reshape(1, 2)

    n = points.shape[0]
    if leg is None or leg.is_empty:
        return np.full(n, np.inf)

    # Local import avoids pulling drifting.engine at module import time.
    from drifting.engine import compass_to_math_deg
    import math

    leg_coords = np.asarray(leg.coords, dtype=float)
    if leg_coords.shape[0] < 2:
        return np.full(n, np.inf)

    math_deg = compass_to_math_deg(float(compass_angle_deg))
    rad = math.radians(math_deg)
    ux = math.cos(rad)
    uy = math.sin(rad)
    u = np.array([ux, uy])
    u_perp = np.array([-uy, ux])

    origin = leg_coords[0]
    verts_rel = points - origin
    leg_rel = leg_coords - origin
    verts_along = verts_rel @ u
    verts_perp = verts_rel @ u_perp
    leg_along = leg_rel @ u
    leg_perp = leg_rel @ u_perp

    seg_a0 = leg_along[:-1]
    seg_a1 = leg_along[1:]
    seg_p0 = leg_perp[:-1]
    seg_p1 = leg_perp[1:]

    ray_a = verts_along[:, None]
    ray_p = verts_perp[:, None]
    p_min = np.minimum(seg_p0, seg_p1)[None, :]
    p_max = np.maximum(seg_p0, seg_p1)[None, :]
    dp = (seg_p1 - seg_p0)[None, :]

    crosses = (ray_p >= p_min) & (ray_p <= p_max) & (dp != 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (ray_p - seg_p0[None, :]) / dp
        along_int = seg_a0[None, :] + t * (seg_a1 - seg_a0)[None, :]
    dist = ray_a - along_int
    valid = crosses & (dist >= 0)
    dist = np.where(valid, dist, np.inf)
    per_point_min = np.min(dist, axis=1)

    finite = np.isfinite(per_point_min)
    if finite.all():
        return per_point_min

    # Fallback: vectorised "nearest-point on leg, projected onto drift".
    # Mirrors the scalar fallback in
    # :func:`drifting.engine.directional_distance_to_point_from_offset_leg`:
    # if a point's reverse ray missed every leg segment (drift parallel to
    # a segment, or the point lies off the lateral band of every segment)
    # we take the closest point on the leg polyline and project the vector
    # from it to the original point onto the drift direction.  Negative
    # projections remain misses.
    miss_idx = np.where(~finite)[0]
    miss_pts = points[miss_idx]

    # Nearest point on each leg segment, for every missing input point.
    seg_p0_xy = leg_coords[:-1]
    seg_p1_xy = leg_coords[1:]
    seg_v = seg_p1_xy - seg_p0_xy           # (M, 2)
    seg_len_sq = (seg_v * seg_v).sum(axis=1)  # (M,)
    # Avoid division-by-zero for zero-length segments.
    safe_len_sq = np.where(seg_len_sq > 0, seg_len_sq, 1.0)

    # Broadcast diff: (K, M, 2)
    diff = miss_pts[:, None, :] - seg_p0_xy[None, :, :]
    t = np.einsum('kmi,mi->km', diff, seg_v) / safe_len_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    # Zero-length segments: closest point is the single endpoint.
    t = np.where(seg_len_sq[None, :] > 0, t, 0.0)
    nearest = seg_p0_xy[None, :, :] + t[:, :, None] * seg_v[None, :, :]
    delta = miss_pts[:, None, :] - nearest  # (K, M, 2)
    dist2 = (delta * delta).sum(axis=2)     # (K, M)
    best_seg = np.argmin(dist2, axis=1)     # (K,)
    k_range = np.arange(miss_pts.shape[0])
    near = nearest[k_range, best_seg]       # (K, 2)

    # Project (point - nearest) onto the drift direction.  Positive
    # projection means the point is downstream of the leg and
    # ``dot`` is the along-drift distance.  Negative means the
    # reverse ray points away from the leg so it's still a miss.
    vec = miss_pts - near
    dot = vec[:, 0] * ux + vec[:, 1] * uy
    positive = dot >= 0
    per_point_min[miss_idx[positive]] = dot[positive]
    return per_point_min


def directional_min_distance_reverse_ray(
    intersection: BaseGeometry,
    leg: LineString,
    compass_angle_deg: float,
) -> float | None:
    """Minimum along-drift distance from ``leg`` to any vertex of ``intersection``.

    For every vertex of ``intersection`` we cast a reverse ray back along the
    anti-drift direction and intersect with ``leg``; the vertex-to-leg
    distance along the drift direction is the minimum of these.

    This is the correct "directional travel distance" for failure-time
    (:math:`P_{NR}`) calculations: an obstacle that sits near the lateral
    corridor but not in the drift direction returns :code:`None` (no ray
    hits the leg), while an obstacle genuinely downstream returns its
    along-drift distance to the nearest vertex.

    Implementation
    --------------
    Rather than calling shapely ``LineString.intersection`` once per vertex
    (each allocation + intersection is ~70 us), we transform the leg and
    all vertices into a local frame where the drift direction is +x.  In
    that frame every reverse ray becomes a horizontal line ``y = vertex_y``
    and ray/segment hit reduces to an edge-crossing test that vectorises
    over ``(vertices x leg-segments)``.  Vertices whose rays don't hit any
    leg segment fall back to the scalar engine function for robust
    handling of collinear / endpoint cases.

    Args:
        intersection: Polygon or MultiPolygon (typically the clip of an
            obstacle against the leg's extended drift corridor).
        leg: The leg LineString (UTM).
        compass_angle_deg: Drift direction in compass degrees (0 = N,
            90 = E, 180 = S, 270 = W).

    Returns:
        Minimum along-drift distance in metres, or ``None`` if no vertex
        of ``intersection`` is reachable by drifting from ``leg`` in that
        direction.
    """
    if leg is None or leg.is_empty:
        return None

    coords: list[tuple[float, float]] = []
    if isinstance(intersection, Polygon):
        coords = list(intersection.exterior.coords)
        for hole in intersection.interiors:
            coords.extend(hole.coords)
    elif isinstance(intersection, MultiPolygon):
        for poly in intersection.geoms:
            coords.extend(poly.exterior.coords)
            for hole in poly.interiors:
                coords.extend(hole.coords)
    else:
        return None

    if not coords:
        return None

    verts = np.asarray(coords, dtype=float)
    dists = directional_distances_to_points(
        verts, leg, compass_angle_deg, use_leg_offset=False,
    )
    finite = np.isfinite(dists)
    if not finite.any():
        return None
    return float(dists[finite].min())


def visualize(
    ax3: Axes,
    distances: np.ndarray,
    distributions: list[Any],
    weights: list[float],
    weighted_overlap: float,
    data:dict[Any]
) -> None:
    ax3.clear()
    if weighted_overlap is None:
        return

    # Generate distances for the PDF curve
    x: np.ndarray = np.linspace(0, max(distances) * 1.5, 500)
    assert isinstance(x, np.ndarray)
    combined_pdf = np.zeros_like(x)
    weights = np.array(weights) / np.sum(weights)

    for dist, weight in zip(distributions, weights):
        combined_pdf += weight * dist.pdf(x)

    # Plot the combined PDF curve
    ax3.plot(x, combined_pdf, label='Prob. leg overlap', color='blue')
    not_repaireds= []
    for x_ in x:
        not_repaireds.append(get_not_repaired(data['drift']['repair'], data['drift']['speed'],x_))
    ax3.plot(x, not_repaireds, label='Prob. failure remains', color='red')

    # Highlight the extent of the intersection
    intersection_min = distances.min()
    intersection_max = distances.max()
    ax3.axvspan(intersection_min, intersection_max, color='green', alpha=0.3, label='Intersection Extent')

    # Add labels and legend
    # ax3.set_title(f"Weighted Overlap: {weighted_overlap:.3e}")
    ax3.set_xlabel("Distance from Closest Point")
    ax3.set_ylabel("Probability Density")
    ax3.legend()
    plt.suptitle(f"Interactive Visualization: Click on an Extended Polygon")
    plt.tight_layout()
    ax3.figure.canvas.draw()

class DriftingOverlapVisualizer:
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

        # State variables
        self.current_line: LineString | None = None
        self.current_base_polygon: Polygon | None = None
        self.current_extended_polygons: list[BaseGeometry] | None = None
        self.current_centre_lines: list[LineString] | None = None
        self.current_coverages: list[float] | None = None
        self.current_distances: list[np.ndarray] | None = None
        self.current_weight: list[float] | None = None
        self.current_distribution: list[Any] | None = None
        
        
    def run_visualization(self):
        self.setup_plot()
        self.connect_events()
        self.simulate_initial_selection()

    def setup_plot(self) -> None:
        assert(isinstance(self.ax1, Axes))
        for line, name in zip(self.lines, self.line_names):
            x, y = line.xy
            self.ax1.plot(x, y, label=name, picker=True)
        self.ax1.set_title("Select a Line")
        legend1 = self.ax1.legend()
        self.ax2.set_title("Base and Extended Polygons")
        self.ax3.set_title("Probability Density Function (PDF)")

        # --- Make legend entries interactive ---
        for legline, _ in zip(legend1.get_lines(), self.ax1.get_lines()):
            legline.set_picker(True)  # Enable picking on the legend line

        def on_legend_pick(event:PickEvent):
            legline = event.artist
            # Find the corresponding original line
            for lline, oline in zip(legend1.get_lines(), self.ax1.get_lines()):
                if legline == lline:
                    visible = not oline.get_visible()
                    oline.set_visible(visible)
                    # Optionally, fade legend entry if line is hidden
                    legline.set_alpha(1.0 if visible else 0.2)
                    self.fig.canvas.draw()
                    break

        self.fig.canvas.mpl_connect('pick_event', on_legend_pick)

    def connect_events(self) -> None:
        self.fig.canvas.mpl_connect('pick_event', self.on_line_click)
        self.fig.canvas.mpl_connect('button_press_event', self.on_polygon_click)

    def on_line_click(self, event: Any) -> None:
        """Handle line selection and update the visualization."""
        idx: int | None = self._get_selected_line_index(event)
        if idx is None:
            return
        self._set_current_line_state(idx)
        if self.current_line is None or self.current_distribution is None or self.current_weight is None:
            return
        
        self.ax2.clear()
        self.ax3.clear()
        self.current_base_polygon = create_polygon_from_line(self.current_line, self.current_distribution, self.current_weight)
        self.current_extended_polygons, self.current_centre_lines = extend_polygon_in_directions(self.current_base_polygon, self.distance)
        results = compare_polygons_with_objs(self.current_extended_polygons, self.objs_gdf_list)
        covered = self._update_coverages_and_distances(results)
        self._plot_polygons(covered)
        self._plot_objects()

    def _update_coverages_and_distances(
        self, results: dict[str, list[list[bool]]]
    ) -> list[bool]:
        coverages, distances, covered = compute_coverages_and_distances(
            self.current_extended_polygons,
            self.current_centre_lines,
            self.current_distribution,
            self.current_weight,
            self.objs_gdf_list,
            results,
        )
        self.current_coverages = coverages
        self.current_distances = distances
        return covered

    def _plot_polygons(self, covered: list[bool]) -> None:
        gpd.GeoSeries(self.current_base_polygon).plot(ax=self.ax2, color='blue', alpha=0.5)
        assert(self.current_extended_polygons != None)
        for i, polygon in enumerate(self.current_extended_polygons):
            assert(isinstance(polygon, Polygon))
            alpha = 0.3 if covered[i] else 0.1
            mpl_polygon = MplPolygon(np.array(polygon.exterior.coords), closed=True, alpha=alpha)
            self.ax2.add_patch(mpl_polygon)

    def _plot_objects(self) -> None:
        for objs_gdf in self.objs_gdf_list:
            objs_gdf.plot(ax=self.ax2, color='red', alpha=0.7, label='Objects')
        self.ax2.set_title("Drift directions")
        self.fig.canvas.draw()

    def _get_selected_line_index(self, event: Any) -> int | None:
        """Return the index of the selected line based on the event."""
        for i, name in enumerate(self.line_names):
            if event.artist.get_label() == name:
                return i
        return None

    def _set_current_line_state(self, idx: int) -> None:
        """Set the current line, weights, and distributions based on the selected index."""
        self.current_line = self.lines[idx]
        self.current_weight = self.weights[idx]
        self.current_distribution = self.distributions[idx]
        
    def on_polygon_click(self, event: Any) -> None:
        # Only respond to clicks in ax2
        if event.inaxes != self.ax2:
            print("wrong axes")
            return
        if event.xdata is None or event.ydata is None:
            print("No x/y")
            return
        if self.current_extended_polygons is None:
            print("extended poly is None")
            return
        clicked = False
        for i, polygon in enumerate(self.current_extended_polygons):
            if polygon.contains(Point(event.xdata, event.ydata)):
                print(f"Polygon {i} was clicked!, {self.current_coverages[i]=}")
                clicked = True
                if self.current_distances is not None and self.current_distribution is not None and self.current_coverages is not None:
                    if len(self.current_distances[i].shape) > 0:
                        visualize(
                            self.ax3,
                            distances=self.current_distances[i],
                            distributions=self.current_distribution,
                            weights=self.current_weight,
                            weighted_overlap=self.current_coverages[i],
                            data=self.data
                        )
                else:
                    self.ax3.clear()
                self.fig.canvas.draw()
                break
        if not clicked:
            print("No polygon was clicked.")

    def simulate_initial_selection(self) -> None:
        class MockPickEvent:
            def __init__(self, artist: Any):
                self.artist = artist

        class MockButtonEvent:
            def __init__(self, x: float, y: float, inaxes: Axes | None):
                self.xdata = x
                self.ydata = y
                self.inaxes = inaxes

        if self.ax1.lines:
            line_artist = self.ax1.lines[-1]
            self.on_line_click(MockPickEvent(line_artist))
        if self.current_extended_polygons is not None:
            centroid = self.current_extended_polygons[0].centroid
            self.on_polygon_click(MockButtonEvent(centroid.x, centroid.y, self.ax2))

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
        distance: float = 50_000
    ) -> "DriftingOverlapVisualizer":
        # Remove any existing canvas
        layout = dialog.result_layout
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            layout.removeWidget(widget)
            widget.deleteLater()

        # Create figure and axes
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        ax1.set_aspect('equal')
        for ax in (ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Add canvas to dialog
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        # Use a label in the dialog for result text
        # Create and run the visualizer
        dov = cls(
            fig, ax1, ax2, ax3, lines, line_names, objs_gdf_list,
            distributions, weights, data, distance=distance
        )
        dov.canvas = canvas  # Optionally store for later
        dov.run_visualization()
        return dov

def compute_min_distance_by_object(
    lines: list[LineString],
    distributions: list[list[Any]],
    weights: list[list[float]],
    objs_gdf_list: list[gpd.GeoDataFrame],
    distance: float
) -> list[list[list[float | None]]]:
    """
    For each leg and for each of the 8 drift directions, compute the minimum distance
    from the leg's centre line to each object's intersection footprint, or None if no intersection.

    Returns a 3-level list indexed as [leg_index][direction_index][object_index].
    The value is the minimal distance (float) if intersecting, else None.
    """
    per_leg_dir_obj: list[list[list[float | None]]] = []
    for idx, (line, dist, wgt) in enumerate(zip(lines, distributions, weights)):
        # Prepare containers for this leg: 8 directions x N objects
        n_objs = sum(len(gdf) for gdf in objs_gdf_list)
        # Map flattened object index to (gdf_idx, row_idx)
        obj_index_map: list[tuple[int, int]] = []
        for gi, gdf in enumerate(objs_gdf_list):
            for ri in range(len(gdf)):
                obj_index_map.append((gi, ri))

        per_dir: list[list[float | None]] = []

        base_polygon = create_polygon_from_line(line, dist, wgt)
        extended_polygons, centre_lines = extend_polygon_in_directions(base_polygon, distance)

        # For each of the 8 directions
        for d_idx, polygon in enumerate(extended_polygons):
            # d_idx maps to the math-angle convention (0=East, 90=North, ...)
            # because extend_polygon_in_directions translates by angle=d_idx*45
            # using cos/sin directly.  Convert to the compass angle that
            # directional_distance_to_point_from_offset_leg expects.
            math_angle = (d_idx * 45) % 360
            compass_angle = (90 - math_angle) % 360

            # Initialize as None (no intersection)
            min_dists: list[float | None] = [None] * n_objs
            # Iterate per GeoDataFrame and each geometry
            flat_idx = 0
            for gi, objs_gdf in enumerate(objs_gdf_list):
                for ri, obj in enumerate(objs_gdf.geometry):
                    if polygon.intersects(obj):
                        intersection = polygon.intersection(obj)
                        try:
                            # Reverse-ray distance from leg to any vertex of
                            # the intersection, along the drift direction.
                            # This returns None for vertices that are not
                            # reachable by drifting (i.e. "behind" the leg),
                            # which is the correct behaviour for the
                            # directional distance used in P_NR.
                            md = directional_min_distance_reverse_ray(
                                intersection, line, compass_angle,
                            )
                            if md is not None:
                                prev = min_dists[flat_idx]
                                if prev is None or md < prev:
                                    min_dists[flat_idx] = md
                        except Exception:
                            # Keep None on failure
                            pass
                    flat_idx += 1
            per_dir.append(min_dists)
        per_leg_dir_obj.append(per_dir)
    return per_leg_dir_obj
