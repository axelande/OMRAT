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

    for angle in range(0, 360, 45):
        # Translate the polygon in the given direction
        dx = distance * np.cos(np.radians(angle))
        dy = distance * np.sin(np.radians(angle))
        translated_polygon = translate(polygon, xoff=dx, yoff=dy)

        # Create a polygon that spans from the original polygon to the translated polygon
        connecting_polygon = polygon.union(translated_polygon).convex_hull
        extended_polygons.append(connecting_polygon)
        centre_lines.append(LineString((polygon.centroid, translated_polygon.centroid)))

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
    x:np.ndarray = np.linspace(0, max(distances) * 1.5, 500)
    assert(x, np.ndarray)
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
        self.current_coverages = []
        self.current_distances = []
        covered: list[bool] = []

        for i, polygon in enumerate(self.current_extended_polygons):
            covered.append(False)
            for objs_gdf_idx, objs_gdf in enumerate(self.objs_gdf_list):
                for j, obj in enumerate(objs_gdf.geometry):
                    if results[f"Polygon_{i}"][objs_gdf_idx][j]:
                        intersection = polygon.intersection(obj)
                        coverage, distances = estimate_weighted_overlap(
                            intersection, self.current_centre_lines[i], self.current_distribution, self.current_weight
                        )
                        self.current_coverages.append(coverage)
                        self.current_distances.append(distances)
                        covered[i] = True
                    else:
                        self.current_coverages.append(0)
                        self.current_distances.append(np.ndarray([]))
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

def compute_total_overlap(
    lines: list[LineString],
    distributions: list[list[Any]],
    weights: list[list[float]],
    objs_gdf_list: list[gpd.GeoDataFrame],
    distance: float
) -> list[list[list[float]]]:
    """
    Compute the total weighted overlap between all legs (lines) and all objects.
    Returns a list of list of list which corresponds legs[the 8 drift directions[distance avg side]]
    """
    total_overlap:list[list[list[float]]] = []
    for idx, (line, dist, wgt) in enumerate(zip(lines, distributions, weights)):
        total_overlap.append([])
        base_polygon = create_polygon_from_line(line, dist, wgt)
        extended_polygons, centre_lines = extend_polygon_in_directions(base_polygon, distance)
        results = compare_polygons_with_objs(extended_polygons, objs_gdf_list)
        for i, polygon in enumerate(extended_polygons):
            total_overlap[idx].append([])
            for objs_gdf_idx, objs_gdf in enumerate(objs_gdf_list):
                for j, obj in enumerate(objs_gdf.geometry):
                    if results[f"Polygon_{i}"][objs_gdf_idx][j]:
                        intersection = polygon.intersection(obj)
                        _, distances = estimate_weighted_overlap(intersection, centre_lines[i], dist, wgt)
                        for k in range(len(distances)-1):
                            total_overlap[idx][i].append((distances[k] + distances[k+1])/2)
    return total_overlap

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
            # Initialize as None (no intersection)
            min_dists: list[float | None] = [None] * n_objs
            # Iterate per GeoDataFrame and each geometry
            flat_idx = 0
            for gi, objs_gdf in enumerate(objs_gdf_list):
                for ri, obj in enumerate(objs_gdf.geometry):
                    if polygon.intersects(obj):
                        intersection = polygon.intersection(obj)
                        try:
                            _, distances = estimate_weighted_overlap(
                                intersection, centre_lines[d_idx], dist, wgt
                            )
                            if distances.size > 0:
                                md = float(np.min(distances))
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

def compute_leg_overlap_fraction(
    lines: list[LineString],
    distributions: list[list[Any]],
    weights: list[list[float]],
    objs_gdf_list: list[gpd.GeoDataFrame]
) -> list[list[float]]:
    """
    Estimate, per leg and per object, the fraction of the leg length that
    overlaps laterally with each object based on the lateral spread (coverage_range).

    Returns a 2-level list indexed as [leg_index][object_index] -> fraction in [0,1].
    """
    fractions: list[list[float]] = []
    # Flatten objects index map to consistent ordering
    obj_index_map: list[tuple[int, int]] = []
    for gi, gdf in enumerate(objs_gdf_list):
        for ri in range(len(gdf)):
            obj_index_map.append((gi, ri))

    for line, dists, wgts in zip(lines, distributions, weights):
        # Compute weighted std used in coverage
        w = np.array(wgts)
        if w.sum() == 0:
            # Avoid division by zero
            w = np.ones_like(w)
        w = w / w.sum()
        # Combined std as in create_polygon_from_line
        weighted_std = float(np.sqrt(sum(weight * (dist.std() ** 2) for dist, weight in zip(dists, w))))
        coverage_range = 4.89 * weighted_std
        # Symmetric strip around the original leg
        strip = line.buffer(coverage_range)
        leg_len = max(line.length, 1e-9)

        per_obj_frac: list[float] = []
        for gi, ri in obj_index_map:
            obj = objs_gdf_list[gi].geometry.iloc[ri]
            try:
                inter = strip.intersection(obj)
                if inter.is_empty:
                    per_obj_frac.append(0.0)
                else:
                    overlap_geom = line.intersection(inter)
                    frac = float(overlap_geom.length / leg_len) if overlap_geom.length > 0 else 0.0
                    per_obj_frac.append(max(0.0, min(1.0, frac)))
            except Exception:
                per_obj_frac.append(0.0)
        fractions.append(per_obj_frac)
    return fractions

def compute_dir_overlap_fraction_by_object(
    lines: list[LineString],
    distributions: list[list[Any]],
    weights: list[list[float]],
    objs_gdf_list: list[gpd.GeoDataFrame],
    distance: float
) -> list[list[list[float]]]:
    """
    For each leg and each of the 8 drift directions, estimate the fraction of the leg
    length that overlaps with each object by expanding the direction polygon slightly
    and measuring the portion of the original leg inside the intersection.

    Returns a 3-level list indexed as [leg][direction][object] with fractions in [0,1].
    """
    per_leg_dir_obj: list[list[list[float]]] = []
    # Flatten object indexing consistent with other helpers
    obj_index_map: list[tuple[int, int]] = []
    for gi, gdf in enumerate(objs_gdf_list):
        for ri in range(len(gdf)):
            obj_index_map.append((gi, ri))

    for line, dists, wgts in zip(lines, distributions, weights):
        # Weighted std for coverage expansion
        w = np.array(wgts)
        if w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()
        weighted_std = float(np.sqrt(sum(weight * (dist.std() ** 2) for dist, weight in zip(dists, w))))
        coverage_range = 4.89 * weighted_std

        base_polygon = create_polygon_from_line(line, dists, w.tolist())
        extended_polygons, centre_lines = extend_polygon_in_directions(base_polygon, distance)

        per_dir: list[list[float]] = []
        for d_idx, poly in enumerate(extended_polygons):
            # Slightly expand polygon to form a corridor strip
            expanded = poly.buffer(coverage_range)
            # Use the directional centre line for overlap measurement
            dir_line = centre_lines[d_idx]
            dir_len = max(dir_line.length, 1e-9)
            dir_fracs: list[float] = []
            for gi, ri in obj_index_map:
                obj = objs_gdf_list[gi].geometry.iloc[ri]
                try:
                    inter = expanded.intersection(obj)
                    if inter.is_empty:
                        dir_fracs.append(0.0)
                    else:
                        overlap_geom = dir_line.intersection(inter)
                        frac = float(overlap_geom.length / dir_len) if overlap_geom.length > 0 else 0.0
                        dir_fracs.append(max(0.0, min(1.0, frac)))
                except Exception:
                    dir_fracs.append(0.0)
            per_dir.append(dir_fracs)
        per_leg_dir_obj.append(per_dir)
    return per_leg_dir_obj

def compute_dir_leg_overlap_fraction_by_object(
    lines: list[LineString],
    distributions: list[list[Any]],
    weights: list[list[float]],
    objs_gdf_list: list[gpd.GeoDataFrame],
    distance: float
) -> list[list[list[float]]]:
    """
    For each leg and each of the 8 drift directions, estimate the fraction of the
    ORIGINAL leg length that lies inside the intersection of the directional corridor
    (expanded polygon) and each object. This respects the "fraction of the leg overlapping"
    semantics but allows directional reach beyond the immediate strip around the leg.

    Returns [leg][direction][object] fractions in [0,1].
    """
    per_leg_dir_obj: list[list[list[float]]] = []
    obj_index_map: list[tuple[int, int]] = []
    for gi, gdf in enumerate(objs_gdf_list):
        for ri in range(len(gdf)):
            obj_index_map.append((gi, ri))

    for line, dists, wgts in zip(lines, distributions, weights):
        w = np.array(wgts)
        if w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()
        weighted_std = float(np.sqrt(sum(weight * (dist.std() ** 2) for dist, weight in zip(dists, w))))
        coverage_range = 4.89 * weighted_std

        base_polygon = create_polygon_from_line(line, dists, w.tolist())
        extended_polygons, _ = extend_polygon_in_directions(base_polygon, distance)

        leg_len = max(line.length, 1e-9)
        per_dir: list[list[float]] = []
        for poly in extended_polygons:
            expanded = poly.buffer(coverage_range)
            dir_fracs: list[float] = []
            for gi, ri in obj_index_map:
                obj = objs_gdf_list[gi].geometry.iloc[ri]
                try:
                    inter = expanded.intersection(obj)
                    if inter.is_empty:
                        dir_fracs.append(0.0)
                    else:
                        overlap_geom = line.intersection(inter)
                        frac = float(overlap_geom.length / leg_len) if overlap_geom.length > 0 else 0.0
                        dir_fracs.append(max(0.0, min(1.0, frac)))
                except Exception:
                    dir_fracs.append(0.0)
            per_dir.append(dir_fracs)
        per_leg_dir_obj.append(per_dir)
    return per_leg_dir_obj

