# Drift Corridor Geometry Specification v2

## Overview

This document refines and clarifies the original `drift_geom.md` specification based on implementation experience. The key challenge is handling **irregular, jagged depth contours** from real-world bathymetric data (e.g., GEBCO grids).

When a ship loses propulsion, it drifts in the wind direction. The **drift corridor** represents the area where a drifting ship could end up. Obstacles (shallow depths, structures) block the corridor - ships ground/stop when they hit these obstacles.

## Comparison with Original Specification

| Aspect | Original (drift_geom.md) | Revised (this document) |
|--------|--------------------------|-------------------------|
| Blocking approach | "Straight blocking lines" at lateral boundaries | **Contour-following** - use actual obstacle shape |
| Obstacle handling | Implied simple polygons | Must handle **scattered MultiPolygons** from gridded data |
| Shadow shape | Rectangular boxes | **Irregular shapes** following obstacle contours |
| Holes in obstacles | Mentioned but unclear | Explicitly handled - holes create open passages |

## Coordinate System (Unchanged)

- **Drift angle**: Degrees where 0=North, 45=NorthWest, 90=West, 135=SouthWest, 180=South, 225=SouthEast, 270=East, 315=NorthEast
- **UTM coordinates**: All calculations done in meters (UTM projection), input gathered in WGS84 (EPSG:4326)
- For N drift (0°): Ship drifts from south to north (increasing Y)

### Mathematical Conversion

To convert from nautical/compass convention to mathematical angles for vector calculations:
```
math_angle = 90 + compass_angle  (then use standard cos/sin)
```

This gives:
- N (0°) → math 90° → vector (0, 1) → ships drift northward
- E (270°) → math 360° → vector (1, 0) → ships drift eastward
- S (180°) → math 270° → vector (0, -1) → ships drift southward
- W (90°) → math 180° → vector (-1, 0) → ships drift westward

## The Core Problem: What Failed

### Issue 1: Bounding Box Shadows Create Wrong Shapes

The original spec suggested using rectangular blocking zones based on obstacle bounding boxes:

```
WHAT WE IMPLEMENTED (WRONG):

    ┌─────────────────────────────────────────────┐
    │████████████████│                            │
    │████████████████│         OPEN               │
    │████████████████│                            │
    │    ┌──────────┐│                            │
    │    │ OBSTACLE ││                            │
    │    │  (actual │ │                            │
    │    │   shape) │ │                            │
    │    └──────────┘│                            │
    │████████████████│  ← Rectangular notch       │
    │████████████████│    cuts into corridor      │
    │                                             │
    │              OPEN CORRIDOR                  │
    └─────────────────────────────────────────────┘

The bounding box approach creates a sharp rectangular edge
that doesn't match the actual irregular obstacle shape.
```

### Issue 2: Real Depth Data is Irregular

GEBCO and similar bathymetric data produces **MultiPolygons with many scattered, irregular parts**:

```
ACTUAL DEPTH OBSTACLES (example):

    ┌─────────────────────────────────────────────┐
    │                                             │
    │      ▓▓▓                ▓▓▓▓               │
    │     ▓▓▓▓▓              ▓▓▓▓▓▓              │
    │    ▓▓▓▓▓▓▓     ▓▓▓    ▓▓▓▓▓▓▓▓             │
    │   ▓▓▓▓▓▓▓▓▓   ▓▓▓▓▓   ▓▓▓▓▓▓▓▓▓            │
    │  ▓▓▓▓▓▓▓▓▓▓▓ ▓▓▓▓▓▓▓ ▓▓▓▓▓▓▓▓▓▓▓           │
    │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓          │
    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓         │
    │                                             │
    └─────────────────────────────────────────────┘

These are jagged, irregular contours - NOT clean rectangles.
Using bounding boxes on these creates artificial rectangular cuts.
```

## Correct Behavior: Contour-Following Shadows

### Principle: The Corridor Should Touch But Not Overlap Obstacles

The drift corridor should:
1. **Extend UP TO** the obstacle boundary (ships can reach and ground on obstacles)
2. **NOT overlap** the obstacle area (ships cannot be inside shallow water)
3. **Follow the actual contour** of the obstacle, not a rectangular approximation

### Visual Example: Correct vs Incorrect

```
INCORRECT (bounding box):           CORRECT (contour-following):

┌──────────────────────┐           ┌──────────────────────┐
│████████│             │           │                      │
│████████│             │           │                      │
│   ▓▓▓▓▓│             │           │   ▓▓▓▓▓              │
│  ▓▓▓▓▓▓│             │           │  ▓▓▓▓▓▓▓             │
│ ▓▓▓▓▓▓▓│             │           │ ▓▓▓▓▓▓▓▓▓            │
│        │             │           │                      │
│  OPEN  │             │           │     OPEN             │
└──────────────────────┘           └──────────────────────┘
     ↑                                   ↑
Rectangular cut                    Corridor follows the
doesn't match                      actual depth contour
obstacle shape

█ = Blocked area (shadow)
▓ = Obstacle (depth/structure)
```

### Algorithm: Sweep-Based Shadow Creation

For each obstacle polygon intersecting the corridor:

1. **Get the obstacle's intersection** with the corridor
2. **For each part** of the intersection (handling MultiPolygon):
   - Create a shadow that extends from the obstacle's "front edge" to the corridor boundary
   - The shadow should follow the obstacle's actual shape, not its bounding box

#### Approach: Buffered Sweep

Instead of simple bounding boxes, use a **directional buffer/sweep**:

```python
def create_contour_following_shadow(obstacle, drift_angle, corridor):
    """
    Create shadow that follows the obstacle's actual contour.

    The shadow extends from the obstacle in the drift direction
    to the corridor boundary, following the obstacle's shape.
    """
    # 1. Get the projection distance needed
    drift_distance = calculate_distance_to_corridor_edge(obstacle, drift_angle, corridor)

    # 2. Create directional offset of obstacle shape
    # This "sweeps" the obstacle shape in the drift direction
    dx, dy = drift_vector(drift_angle, drift_distance)
    swept_obstacle = translate(obstacle, dx, dy)

    # 3. The shadow is the convex hull of original + swept positions
    # OR use a more precise approach with parallel edges
    shadow = unary_union([obstacle, swept_obstacle]).convex_hull

    # 4. Clip to corridor bounds
    return shadow.intersection(corridor)
```

#### Alternative: Vertex-Based Blocking Lines

For more precise control, trace blocking lines from each vertex:

```python
def create_vertex_based_shadow(obstacle, drift_angle, corridor_bounds):
    """
    Create blocking lines from each vertex of the obstacle.
    """
    blocking_lines = []

    for vertex in obstacle.exterior.coords:
        # Create line from vertex to corridor boundary in drift direction
        end_point = extend_to_boundary(vertex, drift_angle, corridor_bounds)
        blocking_lines.append(LineString([vertex, end_point]))

    # Connect blocking lines to form shadow polygon
    # This preserves the obstacle's lateral shape
```

## Handling Scattered MultiPolygons

When depth data produces scattered polygon parts with gaps between them:

```
SCATTERED OBSTACLES:

    ┌─────────────────────────────────────────────┐
    │                                             │
    │     ▓▓▓        GAP        ▓▓▓▓             │
    │    ▓▓▓▓▓                 ▓▓▓▓▓▓            │
    │   ▓▓▓▓▓▓▓               ▓▓▓▓▓▓▓▓           │
    │                                             │
    │              CORRIDOR                       │
    └─────────────────────────────────────────────┘

Ships in the GAP can drift PAST the obstacles!
```

### Correct Handling:

1. **Process each polygon part separately** - don't union or convex hull them
2. **Create individual shadows** for each part
3. **The gaps remain open** - ships at those lateral positions can drift through

```python
def process_obstacles(obstacles, drift_angle, corridor):
    all_shadows = []

    for obstacle in obstacles:
        if isinstance(obstacle, MultiPolygon):
            # Process each part separately - gaps remain open
            for part in obstacle.geoms:
                shadow = create_contour_following_shadow(part, drift_angle, corridor)
                all_shadows.append(shadow)
        else:
            shadow = create_contour_following_shadow(obstacle, drift_angle, corridor)
            all_shadows.append(shadow)

    return unary_union(all_shadows)
```

## Key Differences from Original Spec

### 1. Shape Preservation
- **Original**: "Straight blocking lines" implying rectangular shadows
- **Revised**: Shadows should follow the actual obstacle contour shape

### 2. Obstacle-Corridor Relationship
- **Original**: Unclear whether corridor should overlap obstacles
- **Revised**: Corridor should touch but NOT overlap obstacles. Ships can reach the obstacle (and ground there), but the corridor polygon itself should not include the obstacle area.

### 3. MultiPolygon Handling
- **Original**: Not explicitly addressed
- **Revised**: Each polygon part creates its own shadow; gaps between parts remain open

### 4. Blocking Direction
- **Original**: "Block behind, not the obstacle itself"
- **Revised**: The blocking zone (shadow) includes:
  - The area from the obstacle's front edge (facing against drift) to corridor boundary
  - The corridor should be the original corridor MINUS these shadows
  - Result: corridor extends UP TO obstacles but not through them

## Implementation Checklist

- [ ] Use nautical/compass convention (0°=North)
- [ ] Convert to UTM for accurate distance calculations
- [ ] For each obstacle:
  - [ ] Get intersection with corridor
  - [ ] Handle MultiPolygon by processing each part separately
  - [ ] Create shadow that follows obstacle contour (not bounding box)
  - [ ] Shadow extends from obstacle to corridor boundary in drift direction
- [ ] Final corridor = original corridor - union(all shadows)
- [ ] Transform result back to WGS84

## Debug Output Files

For troubleshooting, write these WKT files:
- `corridor_base_{direction}.wkt` - The unclipped corridor
- `obstacles_{direction}.wkt` - Obstacle intersections with corridor
- `shadows_{direction}.wkt` - The blocking zones
- `corridor_final_{direction}.wkt` - The clipped corridor result

## UI Connections (Unchanged)

- `leDepthThreshold` → LineEdit for the draught_threshold
- `leHeightThreshold` → LineEdit for the height_threshold
- `pbRunDriftAnalysis` → PushButton to trigger corridor generation
- Use QgsTask for background processing with progress updates
- Implement unload() to clean up layers when plugin reloads
