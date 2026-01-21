# -*- coding: utf-8 -*-
"""
Main drift corridor generator class.

Orchestrates corridor generation for all legs and directions,
handling data collection, coordinate transformations, and progress reporting.
"""

from typing import TYPE_CHECKING

from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.validation import make_valid
from pyproj import CRS

from .constants import DIRECTIONS
from .coordinates import get_utm_crs, transform_geometry
from .distribution import get_projection_distance, get_distribution_width
from .corridor import create_projected_corridor
from .clipping import clip_corridor_at_obstacles

if TYPE_CHECKING:
    from omrat import OMRAT


class DriftCorridorGenerator:
    """
    Generates drift corridors for QGIS layers.

    Handles the complete workflow:
    1. Pre-collect data from Qt widgets (main thread)
    2. Generate corridors for all legs in all 8 directions (can run in background)
    3. Report progress and handle cancellation
    """

    def __init__(self, plugin: 'OMRAT'):
        self.plugin = plugin
        self._progress_callback = None
        self._cancelled = False
        self._precollected_data: dict | None = None

    def clear_cache(self) -> None:
        """Clear any cached data."""
        self._precollected_data = None
        self._cancelled = False
        self._progress_callback = None

    def set_progress_callback(self, callback) -> None:
        """
        Set a callback function for progress updates.

        The callback signature is: callback(completed: int, total: int, message: str) -> bool
        Return False from callback to cancel generation.
        """
        self._progress_callback = callback

    def _report_progress(self, completed: int, total: int, message: str) -> bool:
        """Report progress and check for cancellation."""
        if self._progress_callback:
            result = self._progress_callback(completed, total, message)
            if result is False:
                self._cancelled = True
                return False
        return True

    def precollect_data(self, depth_threshold: float, height_threshold: float) -> None:
        """
        Pre-collect all data from Qt widgets (must be called from main thread).

        This reads all necessary data from UI widgets and stores it for use
        in generate_corridors() which can run in a background thread.

        Args:
            depth_threshold: Depths <= this value are considered obstacles
            height_threshold: Heights <= this value are considered obstacles
        """
        from qgis.core import QgsMessageLog, Qgis

        depth_obstacles = self._get_depth_obstacles(depth_threshold)
        structure_obstacles = self._get_structure_obstacles(height_threshold)

        self._precollected_data = {
            'legs': self._get_legs_from_routes(),
            'depth_obstacles': depth_obstacles,
            'structure_obstacles': structure_obstacles,
            'lateral_std': self._get_distribution_std(),
            'repair_params': self._get_repair_params(),
            'drift_speed': self._get_drift_speed_ms(),
        }

        # Log detailed info about obstacles
        total_depth_area = sum(p.area for p, v in depth_obstacles) if depth_obstacles else 0
        total_struct_area = sum(p.area for p, v in structure_obstacles) if structure_obstacles else 0

        QgsMessageLog.logMessage(
            f"Pre-collected data: {len(self._precollected_data['legs'])} legs, "
            f"{len(depth_obstacles)} depth obstacles (total area: {total_depth_area:.2f}), "
            f"{len(structure_obstacles)} structure obstacles (total area: {total_struct_area:.2f})",
            "OMRAT", Qgis.Info
        )

        if depth_obstacles:
            depths = [v for p, v in depth_obstacles]
            QgsMessageLog.logMessage(
                f"Depth obstacle values: min={min(depths)}, max={max(depths)}, threshold={depth_threshold}",
                "OMRAT", Qgis.Info
            )

    def _get_legs_from_routes(self) -> list[LineString]:
        """Extract LineString geometries from route layers."""
        legs = []
        for layer in self.plugin.qgis_geoms.vector_layers:
            for feature in layer.getFeatures():
                geom = feature.geometry()
                if geom and not geom.isNull():
                    wkt = geom.asWkt()
                    try:
                        from shapely import wkt as shapely_wkt
                        shapely_geom = shapely_wkt.loads(wkt)
                        if isinstance(shapely_geom, LineString):
                            legs.append(shapely_geom)
                    except Exception:
                        pass
        return legs

    def _get_depth_obstacles(self, depth_threshold: float) -> list[tuple[Polygon, float]]:
        """
        Get depth polygons that are shallower than threshold.

        For depth intervals like "0-10", we use the UPPER bound (10m) as the max depth.
        Areas with max depth <= threshold are considered obstacles (grounding risk).

        For single values with bin detection, we use (value + bin_width) as max depth.
        """
        from qgis.core import QgsMessageLog, Qgis

        obstacles = []
        table = self.plugin.main_widget.twDepthList

        # First pass: detect bin width from depth values
        bin_width = self._detect_depth_bin_width(table)

        # Second pass: collect obstacles
        for row in range(table.rowCount()):
            try:
                depth_item = table.item(row, 1)
                if depth_item is None:
                    continue
                depth_text = depth_item.text().strip()

                # Parse depth value
                depth = self._parse_depth_value(depth_text, bin_width)
                if depth is None or depth > depth_threshold:
                    continue

                wkt_item = table.item(row, 2)
                if wkt_item is None:
                    continue
                wkt = wkt_item.text()
                if not wkt or not wkt.strip():
                    continue

                from shapely import wkt as shapely_wkt
                shapely_geom = shapely_wkt.loads(wkt)

                # Handle both Polygon and MultiPolygon
                if isinstance(shapely_geom, Polygon):
                    obstacles.append((shapely_geom, depth))
                elif isinstance(shapely_geom, MultiPolygon):
                    for poly in shapely_geom.geoms:
                        if not poly.is_empty:
                            obstacles.append((poly, depth))

            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Error parsing depth row {row}: {e}", "OMRAT", Qgis.Warning
                )

        return obstacles

    def _detect_depth_bin_width(self, table) -> float:
        """Detect bin width from depth values in the table."""
        depth_values = []
        for row in range(table.rowCount()):
            depth_item = table.item(row, 1)
            if depth_item is None:
                continue
            depth_text = depth_item.text().strip()
            try:
                if '-' in depth_text and not depth_text.startswith('-'):
                    continue  # Skip intervals for bin detection
                elif depth_text.startswith('-'):
                    if '--' in depth_text:
                        continue
                    val = abs(float(depth_text))
                else:
                    val = float(depth_text)
                depth_values.append(val)
            except ValueError:
                continue

        # Calculate most common difference
        if len(depth_values) >= 2:
            sorted_depths = sorted(set(depth_values))
            if len(sorted_depths) >= 2:
                diffs = [sorted_depths[i+1] - sorted_depths[i]
                         for i in range(len(sorted_depths)-1)]
                if diffs:
                    from collections import Counter
                    diff_counts = Counter(round(d, 1) for d in diffs)
                    most_common_diff = diff_counts.most_common(1)[0][0]
                    if most_common_diff > 0:
                        return most_common_diff
        return 0.0

    def _parse_depth_value(self, depth_text: str, bin_width: float) -> float | None:
        """Parse a depth value from text, returning the max depth for the cell."""
        try:
            if '-' in depth_text and not depth_text.startswith('-'):
                # Interval like "0-10" -> use upper bound
                parts = depth_text.split('-')
                return float(parts[-1])
            elif depth_text.startswith('-'):
                if '--' in depth_text:
                    # Negative interval like "-10--5" -> use upper bound (less negative)
                    parts = depth_text.split('--')
                    return abs(float(parts[0]))
                else:
                    # Single negative value -> add bin width
                    return abs(float(depth_text)) + bin_width
            else:
                # Single positive value -> add bin width
                return float(depth_text) + bin_width
        except ValueError:
            return None

    def _get_structure_obstacles(self, height_threshold: float) -> list[tuple[Polygon, float]]:
        """Get structure polygons that are lower than threshold."""
        from qgis.core import QgsMessageLog, Qgis

        obstacles = []
        table = self.plugin.main_widget.twObjectList

        for row in range(table.rowCount()):
            try:
                height_item = table.item(row, 1)
                if height_item is None:
                    continue
                height = float(height_item.text())

                if height > height_threshold:
                    continue

                wkt_item = table.item(row, 2)
                if wkt_item is None:
                    continue
                wkt = wkt_item.text()

                from shapely import wkt as shapely_wkt
                shapely_geom = shapely_wkt.loads(wkt)

                if isinstance(shapely_geom, Polygon):
                    obstacles.append((shapely_geom, height))
                elif isinstance(shapely_geom, MultiPolygon):
                    for poly in shapely_geom.geoms:
                        if not poly.is_empty:
                            obstacles.append((poly, height))

            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Error parsing structure row {row}: {e}", "OMRAT", Qgis.Warning
                )

        return obstacles

    def _get_distribution_std(self) -> float:
        """Get the lateral distribution standard deviation."""
        try:
            std1 = float(self.plugin.main_widget.leNormStd1_1.text() or 0)
            if std1 > 0:
                return std1
        except (ValueError, AttributeError):
            pass
        return 100.0

    def _get_repair_params(self) -> dict:
        """Get repair time distribution parameters."""
        drift_values = self.plugin.drift_values
        return {
            'use_lognormal': drift_values.get('use_lognormal', 1),
            'std': drift_values.get('std', 0.95),
            'loc': drift_values.get('loc', 0.2),
            'scale': drift_values.get('scale', 0.85),
        }

    def _get_drift_speed_ms(self) -> float:
        """Get drift speed in m/s."""
        drift_values = self.plugin.drift_values
        speed_kts = float(drift_values.get('speed', 1.94))
        return speed_kts * 1852.0 / 3600.0

    def generate_corridors(self, depth_threshold: float, height_threshold: float,
                           target_prob: float = 1e-3) -> list[dict]:
        """
        Generate drift corridors for all legs in all 8 directions.

        Args:
            depth_threshold: Depths <= this value create shadows
            height_threshold: Heights <= this value create shadows
            target_prob: Target probability for projection distance

        Returns:
            List of corridor dicts with: direction, angle, leg_index, polygon (WGS84)
        """
        from qgis.core import QgsMessageLog, Qgis

        self._cancelled = False

        # Use pre-collected data if available
        if self._precollected_data is not None:
            data = self._precollected_data
            legs = data['legs']
            depth_obstacles = data['depth_obstacles']
            structure_obstacles = data['structure_obstacles']
            lateral_std = data['lateral_std']
            repair_params = data['repair_params']
            drift_speed = data['drift_speed']
        else:
            legs = self._get_legs_from_routes()
            depth_obstacles = self._get_depth_obstacles(depth_threshold)
            structure_obstacles = self._get_structure_obstacles(height_threshold)
            lateral_std = self._get_distribution_std()
            repair_params = self._get_repair_params()
            drift_speed = self._get_drift_speed_ms()

        if not legs:
            QgsMessageLog.logMessage("No legs found from routes", "OMRAT", Qgis.Warning)
            return []

        total_work = len(legs) * len(DIRECTIONS)
        completed_work = 0

        if not self._report_progress(0, total_work, "Initializing..."):
            return []

        # Calculate parameters
        half_width = get_distribution_width(lateral_std, 0.99) / 2
        projection_dist = get_projection_distance(repair_params, drift_speed, target_prob)
        projection_dist = min(projection_dist, 50000)

        QgsMessageLog.logMessage(
            f"Corridor params: half_width={half_width:.1f}m, projection_dist={projection_dist:.1f}m",
            "OMRAT", Qgis.Info
        )

        wgs84 = CRS("EPSG:4326")
        all_obstacles = depth_obstacles + structure_obstacles
        corridors = []

        for leg_idx, leg in enumerate(legs):
            if self._cancelled:
                return corridors

            # Generate corridors for this leg
            leg_corridors, completed_work = self._generate_leg_corridors(
                leg, leg_idx, len(legs),
                all_obstacles, half_width, projection_dist,
                wgs84, completed_work, total_work
            )
            corridors.extend(leg_corridors)

        self._report_progress(total_work, total_work,
                              f"Complete: {len(corridors)} corridors generated")
        self._precollected_data = None

        return corridors

    def _generate_leg_corridors(self, leg: LineString, leg_idx: int, total_legs: int,
                                all_obstacles: list, half_width: float, projection_dist: float,
                                wgs84: CRS, completed_work: int, total_work: int) -> tuple[list[dict], int]:
        """Generate corridors for a single leg in all 8 directions."""
        corridors = []

        centroid = leg.centroid
        if not (-180 <= centroid.x <= 180 and -90 <= centroid.y <= 90):
            return corridors, completed_work + len(DIRECTIONS)

        try:
            utm_crs = get_utm_crs(centroid.x, centroid.y)
            leg_utm = transform_geometry(leg, wgs84, utm_crs)
        except Exception:
            return corridors, completed_work + len(DIRECTIONS)

        # Transform obstacles to UTM
        if not self._report_progress(completed_work, total_work,
                                     f"Transforming obstacles for leg {leg_idx + 1}/{total_legs}..."):
            return corridors, completed_work

        obstacles_utm = self._transform_obstacles_to_utm(all_obstacles, wgs84, utm_crs)

        for dir_name, angle in DIRECTIONS.items():
            if self._cancelled:
                return corridors, completed_work

            if not self._report_progress(completed_work, total_work,
                                         f"Leg {leg_idx + 1}/{total_legs} - {dir_name}"):
                return corridors, completed_work

            # Create and clip corridor
            corridor_wgs84 = self._create_single_corridor(
                leg_utm, half_width, angle, projection_dist,
                obstacles_utm, utm_crs, wgs84,
                f"Leg {leg_idx} {dir_name}: "
            )

            if corridor_wgs84 is not None:
                corridors.append({
                    'direction': dir_name,
                    'angle': angle,
                    'leg_index': leg_idx,
                    'polygon': corridor_wgs84,
                })

            completed_work += 1

        return corridors, completed_work

    def _transform_obstacles_to_utm(self, obstacles: list, wgs84: CRS, utm_crs: CRS) -> list:
        """Transform all obstacles to UTM coordinates."""
        obstacles_utm = []
        for poly, value in obstacles:
            try:
                poly_utm = transform_geometry(poly, wgs84, utm_crs)
                poly_utm = make_valid(poly_utm)
                if not poly_utm.is_empty:
                    obstacles_utm.append((poly_utm, value))
            except Exception:
                pass
        return obstacles_utm

    def _create_single_corridor(self, leg_utm: LineString, half_width: float,
                                angle: float, projection_dist: float,
                                obstacles_utm: list, utm_crs: CRS, wgs84: CRS,
                                log_prefix: str) -> Polygon | None:
        """Create a single corridor for one direction."""
        # Create corridor
        corridor_utm = create_projected_corridor(leg_utm, half_width, angle, projection_dist)

        if corridor_utm.is_empty:
            return None

        # Clip at obstacles
        if obstacles_utm:
            corridor_utm = clip_corridor_at_obstacles(
                corridor_utm, obstacles_utm, angle, log_prefix
            )

        if corridor_utm.is_empty:
            return None

        # Transform back to WGS84 and validate
        try:
            corridor_wgs84 = transform_geometry(corridor_utm, utm_crs, wgs84)
            corridor_wgs84 = make_valid(corridor_wgs84)

            if corridor_wgs84.is_empty:
                return None

            # Validate bounds
            bounds = corridor_wgs84.bounds
            if not (-180 <= bounds[0] <= 180 and -180 <= bounds[2] <= 180 and
                    -90 <= bounds[1] <= 90 and -90 <= bounds[3] <= 90):
                return None

            # Check for unreasonably large corridors (likely transformation errors)
            if (bounds[2] - bounds[0]) > 2 or (bounds[3] - bounds[1]) > 2:
                return None

            return corridor_wgs84
        except Exception:
            return None
