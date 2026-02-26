"""
Create QGIS layers showing grounding and allision results with probability attributes.

Each layer contains lines representing obstacles that were hit, with attributes:
- total_probability: Sum of all leg contributions
- leg_X_prob: Contribution from each leg (aggregated across all directions)

Layers are styled with graduated symbology:
- Red = highest probability
- Yellow = medium probability
- Green = lowest probability

Per-segment probability calculation:
Each segment's probability is calculated based on its orientation relative to drift directions.
A segment can only be "hit" by drift from directions that face the segment's exposed side.
For example, a north-facing edge (normal pointing north) can only be hit by southward drift.
"""
from typing import Any
import numpy as np
from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsField,
    QgsProject,
    QgsGraduatedSymbolRenderer,
    QgsRendererRange,
    QgsLineSymbol,
)
from qgis.PyQt.QtGui import QColor
import logging

logger = logging.getLogger(__name__)

# Drift directions used in the model (compass convention: 0=North, 90=West, etc.)
DRIFT_DIRECTIONS = {
    'North': 0,
    'NorthWest': 45,
    'West': 90,
    'SouthWest': 135,
    'South': 180,
    'SouthEast': 225,
    'East': 270,
    'NorthEast': 315,
}


def _segment_normal_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the outward normal angle of a segment.

    For a polygon with counter-clockwise vertices (which is the standard for
    WGS84 GeoJSON/WKT), the outward normal points to the RIGHT of the segment
    direction when walking along the boundary.

    Args:
        x1, y1: Start point of segment
        x2, y2: End point of segment

    Returns:
        Angle in degrees using OMRAT convention:
        0°=North, 90°=West, 180°=South, 270°=East (CCW from North)
    """
    # Segment direction vector
    dx = x2 - x1
    dy = y2 - y1

    # Right perpendicular (outward normal for CCW polygon)
    # For CCW polygon, outward = right = rotate segment direction -90°
    # Right perpendicular of (dx, dy) is (dy, -dx)
    nx = dy
    ny = -dx

    # Convert to OMRAT angle convention (0=North, 90=West, 180=South, 270=East)
    # This is CCW from North, so we use: -atan2(nx, ny)
    # atan2(nx, ny) gives CW from North (standard compass bearing)
    # Negating gives CCW from North (OMRAT convention)
    omrat_angle = (-np.degrees(np.arctan2(nx, ny))) % 360

    return omrat_angle


def _exposure_factor(segment_normal: float, drift_direction: float) -> float:
    """
    Calculate how exposed a segment is to drift from a given direction.

    A segment is fully exposed (factor=1.0) when drift comes directly toward
    the segment's face (i.e., drift direction is OPPOSITE to the segment normal).
    Exposure decreases with cos(angle) and is 0 when the drift is perpendicular
    or coming from behind the segment.

    Example:
    - A segment facing north (normal=0°) is hit by ships drifting SOUTH (toward 180°)
    - Ships drifting south are moving toward the north-facing edge

    Args:
        segment_normal: Outward normal angle of segment (OMRAT degrees: 0=N, 90=W, 180=S, 270=E)
        drift_direction: Direction ships drift TOWARD (OMRAT degrees)

    Returns:
        Exposure factor between 0.0 and 1.0
    """
    # The drift direction is where ships are drifting TO.
    # A segment facing north (normal=0) is hit by ships drifting SOUTH (toward 180).
    # The drift direction that hits a segment is OPPOSITE to the segment's normal.
    #
    # So we compare drift_direction with (segment_normal + 180) % 360

    # Direction that would hit this segment head-on
    hit_direction = (segment_normal + 180) % 360

    # Angular difference between actual drift and the hit direction
    angle_diff = abs(drift_direction - hit_direction)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    # Exposure is cos(angle_diff), clamped to [0, 1]
    # Full exposure when angle_diff = 0 (drift comes directly at segment face)
    # Zero exposure when angle_diff >= 90 (drift perpendicular or from behind)
    if angle_diff >= 90:
        return 0.0

    return np.cos(np.radians(angle_diff))


def _parse_angle_from_key(leg_dir_key: str) -> float | None:
    """
    Parse drift angle from leg-direction key.

    Key format: "seg_id:direction_label:angle" (e.g., "1:NNW:45")
    The angle is the third component (d_idx * 45 from run_calculations.py).

    Returns:
        Drift angle in degrees (0, 45, 90, etc.) or None if parsing fails
    """
    parts = leg_dir_key.split(':')
    if len(parts) >= 3:
        try:
            return float(parts[2])
        except (ValueError, TypeError):
            pass
    return None


def _aggregate_by_leg(leg_contributions: dict[str, float]) -> dict[str, float]:
    """
    Aggregate leg-direction contributions to per-leg totals.

    The leg keys are in format "seg_id:direction:angle" (e.g., "1:North:0").
    We want to sum all directions for each leg to get per-leg totals.

    Args:
        leg_contributions: Dict of {leg_dir_key: probability}

    Returns:
        Dict of {leg_id: total_probability}
    """
    by_leg: dict[str, float] = {}
    for leg_dir_key, contrib in leg_contributions.items():
        # Parse leg key: "seg_id:direction:angle"
        parts = leg_dir_key.split(':')
        if parts:
            leg_id = parts[0]  # Just the segment ID
            by_leg[leg_id] = by_leg.get(leg_id, 0.0) + contrib
    return by_leg


def extract_obstacle_probabilities(
    report: dict[str, Any],
    structures: list[dict[str, Any]],
    depths: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Extract probability contributions per obstacle from the drifting report.

    Args:
        report: The drifting_report from Calculation
        structures: List of structure metadata dicts with 'id', 'height', 'wkt', 'wkt_wgs84'
        depths: List of depth metadata dicts with 'id', 'depth', 'wkt', 'wkt_wgs84'

    Returns:
        (allision_data, grounding_data) where each is a dict:
        {
            'obstacle_id': {
                'total_probability': float,
                'geometry': BaseGeometry (in WGS84),
                'value': float (height or depth),
                'leg_contributions': {'leg_id': float, ...},  # Aggregated by leg
                'leg_dir_contributions': {'leg_dir_key': float, ...},  # Raw leg-direction data
                'segment_contributions': {'seg_idx': {'leg_dir_key': float, ...}, ...}  # Per-segment
            }
        }
    """
    by_object = report.get('by_object', {})

    # Build reverse lookup: obstacle_id -> metadata
    struct_lookup = {s['id']: s for s in structures}
    depth_lookup = {d['id']: d for d in depths}

    allision_data: dict[str, dict[str, Any]] = {}
    grounding_data: dict[str, dict[str, Any]] = {}

    # Extract totals per object
    for obj_key, obj_vals in by_object.items():
        if not isinstance(obj_vals, dict):
            continue

        allision_prob = obj_vals.get('allision', 0.0)
        grounding_prob = obj_vals.get('grounding', 0.0)

        # Parse object key: "Structure - id" or "Depth - id"
        if obj_key.startswith('Structure - '):
            obj_id = obj_key.replace('Structure - ', '')
            if obj_id in struct_lookup and allision_prob > 0:
                meta = struct_lookup[obj_id]
                # Use WGS84 geometry if available, otherwise fall back to UTM
                geom = meta.get('wkt_wgs84') or meta.get('wkt')
                allision_data[obj_id] = {
                    'total_probability': allision_prob,
                    'geometry': geom,
                    'value': meta['height'],
                    'leg_contributions': {},
                    'leg_dir_contributions': {},  # Raw leg-direction data for per-segment calc
                    'segment_contributions': {},  # Per-segment per leg-direction
                }
        elif obj_key.startswith('Depth - '):
            obj_id = obj_key.replace('Depth - ', '')
            if obj_id in depth_lookup and grounding_prob > 0:
                meta = depth_lookup[obj_id]
                # Use WGS84 geometry if available, otherwise fall back to UTM
                geom = meta.get('wkt_wgs84') or meta.get('wkt')
                grounding_data[obj_id] = {
                    'total_probability': grounding_prob,
                    'geometry': geom,
                    'value': meta['depth'],
                    'leg_contributions': {},
                    'leg_dir_contributions': {},  # Raw leg-direction data for per-segment calc
                    'segment_contributions': {},  # Per-segment per leg-direction
                }

    # Extract per-leg contributions from by_structure_legdir (for allision)
    by_struct_legdir = report.get('by_structure_legdir', {})
    for struct_key, leg_contribs in by_struct_legdir.items():
        if not isinstance(leg_contribs, dict):
            continue
        obj_id = struct_key.replace('Structure - ', '')
        if obj_id in allision_data:
            # Store raw leg-direction contributions for per-segment calculation
            allision_data[obj_id]['leg_dir_contributions'] = dict(leg_contribs)
            # Aggregate contributions by leg ID (sum across all directions)
            aggregated = _aggregate_by_leg(leg_contribs)
            allision_data[obj_id]['leg_contributions'] = aggregated

    # Extract per-leg contributions for grounding from by_depth_legdir
    by_depth_legdir = report.get('by_depth_legdir', {})
    for depth_key, leg_contribs in by_depth_legdir.items():
        if not isinstance(leg_contribs, dict):
            continue
        obj_id = depth_key.replace('Depth - ', '')
        if obj_id in grounding_data:
            # Store raw leg-direction contributions for per-segment calculation
            grounding_data[obj_id]['leg_dir_contributions'] = dict(leg_contribs)
            # Aggregate contributions by leg ID (sum across all directions)
            aggregated = _aggregate_by_leg(leg_contribs)
            grounding_data[obj_id]['leg_contributions'] = aggregated

    # Extract per-segment contributions from by_structure_segment_legdir
    by_struct_seg_legdir = report.get('by_structure_segment_legdir', {})
    for struct_key, seg_data in by_struct_seg_legdir.items():
        if not isinstance(seg_data, dict):
            continue
        obj_id = struct_key.replace('Structure - ', '')
        if obj_id in allision_data:
            allision_data[obj_id]['segment_contributions'] = dict(seg_data)

    # Extract per-segment contributions from by_depth_segment_legdir
    by_depth_seg_legdir = report.get('by_depth_segment_legdir', {})
    for depth_key, seg_data in by_depth_seg_legdir.items():
        if not isinstance(seg_data, dict):
            continue
        obj_id = depth_key.replace('Depth - ', '')
        if obj_id in grounding_data:
            grounding_data[obj_id]['segment_contributions'] = dict(seg_data)

    return allision_data, grounding_data


def _extract_line_segments_with_normals(geom) -> list[tuple[float, float, float, float, float]]:
    """
    Extract individual line segments from a polygon boundary with their normal angles.

    For a rectangular polygon, this returns 4 separate LineStrings with normals.
    Each segment connects two consecutive vertices.

    IMPORTANT: This function normalizes polygon orientation to CCW (counter-clockwise)
    before extracting segments. This ensures consistent outward normal calculation.

    For CCW polygons:
    - Exterior ring goes counter-clockwise
    - Interior (hole) rings go clockwise
    - Outward normal = rotate segment vector 90° clockwise (right-hand rule)

    Args:
        geom: A shapely geometry (Polygon, MultiPolygon, LineString, etc.)

    Returns:
        List of (x1, y1, x2, y2, normal_angle) tuples representing line segments
        where normal_angle is the outward-facing direction in compass degrees
    """
    from shapely.geometry import Polygon, MultiPolygon, LineString, LinearRing
    from shapely.geometry import polygon as shapely_polygon

    segments: list[tuple[float, float, float, float, float]] = []

    def extract_from_ring(ring_coords):
        """Extract segments from a ring's coordinates."""
        coords = list(ring_coords)
        for i in range(len(coords) - 1):
            x1, y1 = coords[i][:2]
            x2, y2 = coords[i + 1][:2]
            # Skip zero-length segments
            if (x1, y1) != (x2, y2):
                normal = _segment_normal_angle(x1, y1, x2, y2)
                segments.append((x1, y1, x2, y2, normal))

    if isinstance(geom, Polygon):
        # Normalize polygon to CCW exterior, CW holes using shapely's orient()
        # This ensures consistent outward normal calculation
        oriented_geom = shapely_polygon.orient(geom, sign=1.0)  # 1.0 = CCW exterior
        # Extract exterior ring segments
        extract_from_ring(oriented_geom.exterior.coords)
        # Extract interior rings (holes) if any
        for interior in oriented_geom.interiors:
            extract_from_ring(interior.coords)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            segments.extend(_extract_line_segments_with_normals(poly))
    elif isinstance(geom, (LineString, LinearRing)):
        extract_from_ring(geom.coords)
    elif hasattr(geom, 'boundary'):
        # For other geometry types, try to get boundary
        boundary = geom.boundary
        if hasattr(boundary, 'coords'):
            extract_from_ring(boundary.coords)
        elif hasattr(boundary, 'geoms'):
            # MultiLineString boundary
            for line in boundary.geoms:
                extract_from_ring(line.coords)

    return segments


def _calculate_segment_probability(
    segment_normal: float,
    leg_dir_contributions: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    Calculate the probability for a specific segment based on its orientation.

    Each segment's probability is the sum of contributions from each leg-direction,
    weighted by how exposed the segment is to drift from that direction.

    Args:
        segment_normal: Outward normal angle of segment (OMRAT degrees: 0=N, 90=W, 180=S, 270=E)
        leg_dir_contributions: Dict of {leg_dir_key: probability}
            where leg_dir_key is "seg_id:direction_label:angle" (e.g., "1:NNW:45")

    Returns:
        (total_probability, per_leg_contributions)
        where per_leg_contributions is {leg_id: probability} aggregated by leg
    """
    segment_total = 0.0
    per_leg: dict[str, float] = {}

    for leg_dir_key, contrib in leg_dir_contributions.items():
        # Parse drift angle from key (e.g., "1:NNW:45" -> 45.0)
        drift_angle = _parse_angle_from_key(leg_dir_key)
        if drift_angle is None:
            continue

        # Calculate exposure factor for this segment to this drift direction
        exposure = _exposure_factor(segment_normal, drift_angle)

        # Segment contribution = original contribution × exposure factor
        segment_contrib = contrib * exposure
        segment_total += segment_contrib

        # Aggregate by leg
        parts = leg_dir_key.split(':')
        if parts:
            leg_id = parts[0]
            per_leg[leg_id] = per_leg.get(leg_id, 0.0) + segment_contrib

    return segment_total, per_leg


def create_result_layer(
    name: str,
    obstacle_data: dict[str, dict[str, Any]],
    layer_type: str = 'allision',
) -> QgsVectorLayer | None:
    """
    Create a QGIS vector layer with obstacle geometries and probability attributes.

    Each obstacle's boundary is split into individual line segments. All segments
    of an obstacle share the same probability values since the underlying calculation
    computes probability per obstacle (not per segment).

    Geometries are expected to already be in WGS84.

    Args:
        name: Layer name
        obstacle_data: Dict from extract_obstacle_probabilities (geometries in WGS84)
        layer_type: 'allision' or 'grounding'

    Returns:
        QgsVectorLayer with features and attributes in WGS84
    """
    if not obstacle_data:
        logger.info(f"No {layer_type} data to create layer")
        return None

    # Collect all unique leg keys for attribute columns
    all_leg_keys: set[str] = set()
    for obs_id, obs_data in obstacle_data.items():
        all_leg_keys.update(obs_data.get('leg_contributions', {}).keys())

    # Create layer with Line geometry in WGS84
    layer = QgsVectorLayer("LineString?crs=epsg:4326", name, "memory")
    provider = layer.dataProvider()

    if provider is None:
        logger.error(f"Failed to get data provider for layer {name}")
        return None

    # Define attributes
    fields = [
        QgsField("obstacle_id", QVariant.String),
        QgsField("segment_idx", QVariant.Int),  # Index of segment within obstacle
        QgsField("total_prob", QVariant.Double),  # Per-segment probability
        QgsField("obs_total", QVariant.Double),   # Total obstacle probability (for reference)
        QgsField("normal_deg", QVariant.Double),  # Segment normal direction (compass degrees)
        QgsField("value", QVariant.Double),  # height or depth
    ]

    # Add per-leg attributes with leg ID as field name
    leg_key_to_field: dict[str, str] = {}
    for leg_key in sorted(all_leg_keys):
        # Use leg ID directly as field name (e.g., "leg_1", "leg_2")
        safe_name = f"leg_{leg_key}"
        leg_key_to_field[leg_key] = safe_name
        fields.append(QgsField(safe_name, QVariant.Double))

    provider.addAttributes(fields)
    layer.updateFields()

    # Add features - one per line segment with per-segment probability
    features = []
    for obs_id, obs_data in obstacle_data.items():
        geom = obs_data.get('geometry')
        if geom is None:
            continue

        # Extract individual line segments with their normal angles
        try:
            segments = _extract_line_segments_with_normals(geom)
        except Exception as e:
            logger.warning(f"Failed to extract segments for {obs_id}: {e}")
            continue

        if not segments:
            logger.warning(f"No segments extracted for {obs_id}")
            continue

        # Get leg-direction contributions for per-segment calculation
        leg_dir_contribs = obs_data.get('leg_dir_contributions', {})
        obs_total_prob = obs_data.get('total_probability', 0.0)

        # Get per-segment contributions from the report (calculated based on corridor intersection)
        segment_contributions = obs_data.get('segment_contributions', {})

        # Check if we have per-segment data (new behavior) or need to fall back
        has_segment_data = bool(segment_contributions)

        # Get aggregated leg contributions for fallback
        obs_leg_contribs = obs_data.get('leg_contributions', {})

        # Create one feature per segment
        # Per-segment contributions are tracked based on actual drift corridor intersection
        # Falls back to obstacle total if per-segment data is not available
        for seg_idx, (x1, y1, x2, y2, normal_angle) in enumerate(segments):
            if has_segment_data:
                # Use per-segment contribution data from corridor intersection
                seg_key = f"seg_{seg_idx}"
                seg_legdir_contribs = segment_contributions.get(seg_key, {})

                # Calculate segment total probability from its leg-direction contributions
                seg_prob = sum(seg_legdir_contribs.values()) if seg_legdir_contribs else 0.0

                # Calculate per-leg contributions for this segment
                seg_leg_contribs: dict[str, float] = {}
                for leg_dir_key, contrib in seg_legdir_contribs.items():
                    parts = leg_dir_key.split(':')
                    if parts:
                        leg_id = parts[0]
                        seg_leg_contribs[leg_id] = seg_leg_contribs.get(leg_id, 0.0) + contrib
            else:
                # Fallback: all segments share the obstacle's total probability
                # This maintains backward compatibility with older reports
                seg_prob = obs_total_prob
                seg_leg_contribs = obs_leg_contribs

            # Create WKT for line segment (already in WGS84)
            wkt = f"LINESTRING({x1} {y1}, {x2} {y2})"
            qgs_geom = QgsGeometry.fromWkt(wkt)

            feat = QgsFeature(layer.fields())
            feat.setGeometry(qgs_geom)

            # Set attributes
            feat.setAttribute("obstacle_id", obs_id)
            feat.setAttribute("segment_idx", seg_idx)
            feat.setAttribute("total_prob", seg_prob)  # Per-segment probability
            feat.setAttribute("obs_total", obs_total_prob)  # Total obstacle probability
            feat.setAttribute("normal_deg", normal_angle)  # Segment orientation
            feat.setAttribute("value", obs_data.get('value', 0.0))

            # Set per-leg probabilities (segment-specific)
            for leg_key, field_name in leg_key_to_field.items():
                feat.setAttribute(field_name, seg_leg_contribs.get(leg_key, 0.0))

            features.append(feat)

    provider.addFeatures(features)

    # Log statistics
    probs = [f["total_prob"] for f in features if f["total_prob"] is not None]
    if probs:
        logger.info(
            f"Created {layer_type} layer with {len(features)} segments from "
            f"{len(obstacle_data)} obstacles. Probability range: {min(probs):.2e} - {max(probs):.2e}"
        )
    else:
        logger.info(f"Created {layer_type} layer with {len(features)} segments from {len(obstacle_data)} obstacles")

    return layer


def apply_graduated_symbology(
    layer: QgsVectorLayer,
    attribute: str = "total_prob",
    num_classes: int = 5,
) -> None:
    """
    Apply graduated symbology to a layer: green (low) -> yellow (medium) -> red (high).

    Args:
        layer: The vector layer to style
        attribute: The attribute field to classify by
        num_classes: Number of classification bins
    """
    if layer is None or layer.featureCount() == 0:
        return

    # Get min/max values
    idx = layer.fields().indexOf(attribute)
    if idx < 0:
        logger.warning(f"Attribute {attribute} not found in layer")
        return

    values = [f[attribute] for f in layer.getFeatures() if f[attribute] is not None and f[attribute] > 0]
    if not values:
        logger.warning("No valid values for symbology")
        return

    min_val = min(values)
    max_val = max(values)

    if min_val >= max_val:
        # All same value - use single symbol
        symbol = QgsLineSymbol.createSimple({
            'color': 'yellow',
            'width': '0.8',
        })
        from qgis.core import QgsSingleSymbolRenderer
        renderer = QgsSingleSymbolRenderer(symbol)
        layer.setRenderer(renderer)
        layer.triggerRepaint()
        return

    # Create ranges manually: green -> yellow -> red
    ranges = []
    step = (max_val - min_val) / num_classes

    # Color gradient: green (low probability) to red (high probability)
    colors = [
        QColor(0, 255, 0),      # Green - lowest
        QColor(128, 255, 0),    # Yellow-green
        QColor(255, 255, 0),    # Yellow - medium
        QColor(255, 128, 0),    # Orange
        QColor(255, 0, 0),      # Red - highest
    ]

    for i in range(num_classes):
        lower = min_val + i * step
        upper = min_val + (i + 1) * step

        # Get color for this range
        color = colors[min(i, len(colors) - 1)]

        # Create symbol
        symbol = QgsLineSymbol.createSimple({
            'color': color.name(),
            'width': str(0.5 + 0.3 * i),  # Thicker lines for higher probability
        })

        # Create range
        label = f"{lower:.2e} - {upper:.2e}"
        rng = QgsRendererRange(lower, upper, symbol, label)
        ranges.append(rng)

    # Create graduated renderer
    renderer = QgsGraduatedSymbolRenderer(attribute, ranges)
    renderer.setMode(QgsGraduatedSymbolRenderer.Custom)

    layer.setRenderer(renderer)
    layer.triggerRepaint()


def create_result_layers(
    report: dict[str, Any],
    structures: list[dict[str, Any]],
    depths: list[dict[str, Any]],
    add_to_project: bool = True,
) -> tuple[QgsVectorLayer | None, QgsVectorLayer | None]:
    """
    Create grounding and allision result layers from calculation results.

    The geometries are expected to be in WGS84 (from 'wkt_wgs84' field in metadata).

    Args:
        report: The drifting_report from Calculation
        structures: List of structure metadata dicts with 'wkt_wgs84'
        depths: List of depth metadata dicts with 'wkt_wgs84'
        add_to_project: If True, add layers to the current QGIS project

    Returns:
        (allision_layer, grounding_layer)
    """
    if report is None:
        logger.warning("No report data provided")
        return None, None

    # Extract probability data (uses WGS84 geometries)
    allision_data, grounding_data = extract_obstacle_probabilities(
        report, structures, depths
    )

    # Create layers (geometries are already in WGS84)
    allision_layer = create_result_layer(
        "Allision Results",
        allision_data,
        "allision",
    )

    grounding_layer = create_result_layer(
        "Grounding Results",
        grounding_data,
        "grounding",
    )

    # Apply symbology
    if allision_layer is not None:
        apply_graduated_symbology(allision_layer)
        if add_to_project:
            QgsProject.instance().addMapLayer(allision_layer)
            logger.info("Added Allision Results layer to project")

    if grounding_layer is not None:
        apply_graduated_symbology(grounding_layer)
        if add_to_project:
            QgsProject.instance().addMapLayer(grounding_layer)
            logger.info("Added Grounding Results layer to project")

    return allision_layer, grounding_layer
