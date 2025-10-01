from typing import Any

import geopandas as gpd
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QLabel
from shapely.affinity import translate
from shapely.geometry import LineString, Polygon, Point
from shapely.geometry.base import BaseGeometry


def create_polygon_from_line(line: LineString, distributions: list[Any], weights: list[float]) -> Polygon:
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


def extend_polygon_in_directions(polygon: Polygon, distance: float) -> tuple[list[BaseGeometry], list[LineString]]:
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


def compare_polygons_with_objs(extended_polygons: list, objs_gdf_list: list):
    """
    Compare the extended polygons with the objects in a list of GeoDataFrames.

    Parameters:
    - extended_polygons: List of polygons to compare.
    - objs_gdf_list: List of GeoDataFrames containing objects.

    Returns:
    - results: Dictionary with overlap results for each polygon and GeoDataFrame.
    """
    results = {}
    for i, polygon in enumerate(extended_polygons):
        results[f"Polygon_{i}"] = []
        for objs_gdf in objs_gdf_list:
            intersects = objs_gdf.intersects(polygon)
            results[f"Polygon_{i}"].append(intersects.tolist())
    return results


def estimate_weighted_overlap(intersection: Polygon, line: LineString, distributions: list, weights: list) -> float:
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

    # Sample points within the intersection polygon
    sample_points = np.array(intersection.exterior.coords)

    # Calculate the distance of each point from the closest point on the line
    distances = np.sqrt((sample_points[:, 0] - closest_point.x)**2 + (sample_points[:, 1] - closest_point.y)**2)

    # Calculate the combined probability density
    combined_probabilities = np.zeros_like(distances)
    for dist, weight in zip(distributions, weights):
        combined_probabilities += weight * dist.pdf(distances)

    # Calculate the weighted overlap as the sum of combined probabilities
    weighted_overlap = combined_probabilities.sum() * 100  # Convert to percentage
    return weighted_overlap, distances


def visualize(ax2, line, intersection, distances, distributions, weights, weighted_overlap: float):
    ax2.clear()
    if weighted_overlap is None:
        return

    # Generate distances for the PDF curve
    x = np.linspace(0, max(distances) * 1.5, 500)
    combined_pdf = np.zeros_like(x)
    weights = np.array(weights) / np.sum(weights)

    for dist, weight in zip(distributions, weights):
        combined_pdf += weight * dist.pdf(x)

    # Plot the combined PDF curve
    ax2.plot(x, combined_pdf, label='Combined PDF', color='blue')

    # Highlight the extent of the intersection
    intersection_min = distances.min()
    intersection_max = distances.max()
    ax2.axvspan(intersection_min, intersection_max, color='green', alpha=0.3, label='Intersection Extent')

    # Add labels and legend
    ax2.set_title(f"Weighted Overlap: {weighted_overlap:.3e}")
    ax2.set_xlabel("Distance from Closest Point")
    ax2.set_ylabel("Probability Density")
    ax2.legend()
    plt.suptitle(f"Interactive Visualization: Click on an Extended Polygon")
    plt.tight_layout()
    ax2.figure.canvas.draw()

def visualize_interactive(fig, ax1, ax2, ax3, lines, line_names, objs_gdf_list, distributions, weights, result_text:QLabel, distance:float=50_000):
    """
    Interactive visualization with three subplots:
    - ax1: Example lines to select from.
    - ax2: Base polygon and extended polygons for the selected line.
    - ax3: Probability density function (PDF) for the selected extended polygon.
    """

    # Plot the example lines in ax1
    for i, (line, name) in enumerate(zip(lines, line_names)):
        x, y = line.xy
        ax1.plot(x, y, label=name, picker=True)  # Enable picking for the lines
    ax1.set_title("Select a Line")
    ax1.legend()

    # Placeholder for ax2 and ax3
    ax2.set_title("Base and Extended Polygons")
    ax3.set_title("Probability Density Function (PDF)")

    # Variables to store the current state
    current_line = None
    current_base_polygon = None
    current_extended_polygons = None
    current_centre_lines = None
    current_coverages = None
    current_distances = None
    current_weight = None
    current_distribution = None

    def on_line_click(event):
        """Handle clicks on lines in ax1."""
        nonlocal current_line, current_base_polygon, current_extended_polygons, current_centre_lines, current_coverages, current_distances, current_distribution, current_weight

        # Find the clicked line
        for i, (line, name) in enumerate(zip(lines, line_names)):
            if event.artist.get_label() == name:
                current_line = line
                current_weight = weights[i]
                current_distribution = distributions[i]
                break

        # Clear ax2 and ax3
        ax2.clear()
        ax3.clear()

        # Create the base polygon and extended polygons for the selected line
        current_base_polygon = create_polygon_from_line(current_line, current_distribution, current_weight)
        current_extended_polygons, current_centre_lines = extend_polygon_in_directions(current_base_polygon, distance)

        # Compare the extended polygons with the objects
        results = compare_polygons_with_objs(current_extended_polygons, objs_gdf_list)

        # Calculate coverages and distances
        current_coverages = []
        current_distances = []
        covered = [False, False, False, False, False, False, False, False]
        total_weighted_overlap = 0 
        for i, polygon in enumerate(current_extended_polygons):
            for objs_gdf_idx, objs_gdf in enumerate(objs_gdf_list):
                for j, obj in enumerate(objs_gdf.geometry):
                    if results[f"Polygon_{i}"][objs_gdf_idx][j]:  # Check if overlap is True
                        intersection = polygon.intersection(obj)
                        coverage, distances = estimate_weighted_overlap(
                            intersection, current_centre_lines[i], current_distribution, current_weight
                        )
                        current_coverages.append(coverage)
                        current_distances.append(distances)
                        if coverage is not None:
                            total_weighted_overlap += coverage  # <-- Accumulate
                        covered[i] = True
                    else:
                        current_coverages.append(None)
                        current_distances.append(None)

        # Plot the base polygon and extended polygons in ax2
        gpd.GeoSeries(current_base_polygon).plot(ax=ax2, color='blue', alpha=0.5)
        print(covered)
        for i, polygon in enumerate(current_extended_polygons):
            if covered[i]:
                mpl_polygon = MplPolygon(np.array(polygon.exterior.coords), closed=True, alpha=0.3)
            else:
                mpl_polygon = MplPolygon(np.array(polygon.exterior.coords), closed=True, alpha=0.1)
            ax2.add_patch(mpl_polygon)
        for objs_gdf in objs_gdf_list:
            objs_gdf.plot(ax=ax2, color='red', alpha=0.7, label='Objects')
        ax2.set_title("Drift directions")
        fig.canvas.draw()
        result_text.setText(f"Sum of weighted overlaps for this line: {total_weighted_overlap:.3e}")

    def on_polygon_click(event):
        """Handle clicks on extended polygons in ax2."""
        if current_extended_polygons is None:
            return

        # Check if the click is inside any extended polygon
        for i, polygon in enumerate(current_extended_polygons):
            if polygon.contains(Point(event.xdata, event.ydata)):
                # Generate the PDF plot for the clicked polygon in ax3
                if current_distances is not None:
                    visualize(ax3, current_centre_lines[i], objs_gdf_list[0], distances=current_distances[i], distributions=current_distribution, weights=current_weight, weighted_overlap=current_coverages[i])
                else:
                    ax3.clear()
                break
    
    # Connect the click events to the handlers
    fig.canvas.mpl_connect('pick_event', on_line_click)  # For selecting lines in ax1
    fig.canvas.mpl_connect('button_press_event', on_polygon_click)  # For selecting polygons in ax2

    # Simulate a pick event for the first line
    class MockPickEvent:
        def __init__(self, artist):
            self.artist = artist

    # Simulate a button press event for the first polygon
    class MockButtonEvent:
        def __init__(self, x, y):
            self.xdata = x
            self.ydata = y

    # After plotting the lines in ax1, get the first line artist
    line_artist = ax1.lines[-1]  # or find by label if needed

    # Trigger on_line_click for the first line
    on_line_click(MockPickEvent(line_artist))

    # After on_line_click, you have current_extended_polygons available
    # Get a point inside the first extended polygon (e.g., its centroid)
    if current_extended_polygons:
        centroid = current_extended_polygons[0].centroid
        on_polygon_click(MockButtonEvent(centroid.x, centroid.y))
