# Drift Corridor Geometry Specification

## Overview

When a ship loses propulsion, it drifts in the wind direction. The **drift corridor** represents the area where a drifting ship could end up. Obstacles (shallow depths, structures) block the corridor - ships ground/stop when they hit these obstacles.

## Coordinate System

- **Drift angle**: Degrees where 0=North, 45=NorthWest, 90=West, 135=SouthWest, 180=South, 225=SouthEast, 270=East, 315=NorthEast
- **UTM coordinates**: All calculations done in meters (UTM projection), the input are gathered in wgs84 (EPSG:4326)
- For N drift (0°): Ship drifts from south to north (increasing Y)

## Step 1: Create Base Corridor

The corridor is a polygon extending from the ship's route leg in the drift direction.

```
Input:
- leg: LineString of the route segment
- distributions: How the ships are spread laterally
- projection_dist: How far the corridor extends in drift direction (meters)
- drift_angle_deg: Wind/drift direction

Output:
- corridor: Polygon representing the full drift area before obstacles
```

The corridor is essentially a rectangle:
- Width: `99.9% of all distributions`
- Length: From leg start to `projection_dist` in the drift direction

## Step 2: Identify Obstacles

Obstacles are areas where ships cannot drift through:
- **Depth obstacles**: Areas shallower than ship draft (ships ground)
- **Structure obstacles**: Fixed structures (platforms, wind turbines)

```
Input:
- depth_layer: Polygons with depth values,
there are several layers with different depths, however these could be unioned together based on the depth, either the depth are more shallow than the draught_threshold value or it is not
- structure_layer: Polygons of structures including their height
- draught_threshold: Ship's draught (meters)
- height_threshold: Ship's height (meters)

Output:
- obstacles: List of (polygon, value) tuples that intersect the corridor
```
## Step 3: Clip Corridor at Obstacles (THE KEY ALGORITHM)

### Concept
When an obstacle **partially covers** the corridor width:
- Ships at lateral positions that hit the obstacle will **ground/stop there**
- Ships at lateral positions that miss the obstacle can **drift past** it

### Blocking Lines
For each obstacle, create **vertical blocking lines** (for N/S drift) or **horizontal blocking lines** (for E/W drift) or **Tilted blocking lines** (for NE, NW, SW, SE drift) at the obstacle's lateral boundaries.

### Visual Example (N Drift)

```
                    CORRIDOR TOP (corridor_maxy)
    ┌─────────────────────────────────────────────┐
    │    │               │                        │
    │    │    BLOCKED    │         OPEN           │
    │    │    (ships hit │         (ships drift   │
    │    │     obstacle) │          past)         │
    │    │               │                        │
    │    ┌─────────────┐ │                        │
    │    │ OBSTACLE    │ │                        │
    │    │ (depth or   / │                        │
    │    │ structure) |  │                        │
    │    │             \ │                        │
    │    │              \│                        │
    │    └───────────────┘                        │
    │    │               │                        │
    │    ↑               ↑                        │
    │      blocking lines                         │
    │                                             │
    │              OPEN CORRIDOR                  │
    │              (ships can drift here)         │
    │                                             │
    ├─────────────────────────────────────────────┤
    │                  LEG                        │
    └─────────────────────────────────────────────┘
```
Observe that the obstacle "contains" a small hole on the west side that also should be blocked etc.

## Key Principles

1. **Straight blocking lines**: Use the actual shape. This creates clean vertical/horizontal blocking lines.

2. **Block behind, not the obstacle itself**: The blocking area extends from the obstacle's front edge to the corridor boundary. Ships CAN reach the obstacle (they ground there).

3. **Lateral boundaries matter**: Only the left/right (or top/bottom for E/W drift) edges of obstacles create blocking lines. The obstacle's actual shape doesn't matter for blocking - only its extent perpendicular to drift direction.

4. **Multiple obstacles**: Each obstacle creates its own blocking rectangle. The union of all blocking rectangles is subtracted from the corridor.

5. **Result is corridor minus blocked areas**: The final corridor polygon is `corridor.difference(all_blocking_areas)`.

## WKT Output Files

For debugging, write these files:
- `obstacle_intersection_{direction}.wkt`: The obstacle polygons within the corridor
- `blocking_line_{direction}.wkt`: The vertical/horizontal blocking lines
- `shadow_{direction}.wkt`: The blocking rectangles (areas to subtract)

## Connections
leDepthThreshold -> LineEdit for the draught_threshold
leHeightThreshold -> LineEdit for the height_threshold
pbRunDriftAnalysis -> Pushbutton to trigger the insertion of these polygons in QGIS

## Other
- It is important that there is a unload function in the new class so everything that is added could be removed when the plugin is reloaded.
- The task of running this should be handelled within a QgsTask so there is a progress update the the possibility to cancel the generation of cooridors
- If the value of leDepthThreshold is updated and pbRunDriftAnalysis is pressed the cooridor should be updated (removed and added again with the updated geometry)