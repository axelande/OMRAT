# OMRAT Computation Procedures

This document details the computation flow when `run_drifting_model()` is triggered in OMRAT, including all functions called, data assumptions, coordinate systems, and mathematical formulas.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Entry Point: run_drifting_model()](#2-entry-point-run_drifting_model)
3. [Phase 1: Data Preparation](#3-phase-1-data-preparation)
4. [Phase 2: Spatial Precomputation](#4-phase-2-spatial-precomputation)
5. [Phase 3: Traffic Cascade Calculation](#5-phase-3-traffic-cascade-calculation)
6. [Phase 4: Report Generation](#6-phase-4-report-generation)
7. [Key Formulas](#7-key-formulas)
8. [Coordinate Systems](#8-coordinate-systems)
9. [Data Assumptions](#9-data-assumptions)
10. [Calculation Parameters](#10-calculation-parameters)

---

## 1. Overview

The OMRAT drifting model calculates the probability of **allision** (collision with fixed structures) and **grounding** (running aground on shallow depths) for drifting vessels. The calculation follows this high-level flow:

```
run_drifting_model()
    ├── _build_transformed()           # Phase 1: Data preparation
    │   ├── prepare_traffic_lists()
    │   ├── split_structures_and_depths()
    │   └── transform_to_utm()
    │
    ├── _precompute_spatial()          # Phase 2: Spatial calculations
    │   ├── compute_min_distance_by_object()
    │   ├── compute_probability_holes_smart_hybrid()
    │   │   └── compute_probability_holes_pdf_corrected()
    │   └── compute_overlap_fractions()
    │
    ├── _iterate_traffic_and_sum()     # Phase 3: Cascade calculation
    │   ├── Build obstacle list per leg/direction
    │   ├── Sort by distance (cascade order)
    │   ├── Process anchoring → allision → grounding
    │   └── Track per-segment contributions
    │
    └── create_result_layers()         # Phase 4: Output
        └── Generate QGIS layers
```

---

## 2. Entry Point: run_drifting_model()

**Location:** `compute/run_calculations.py`, method `Calculation.run_drifting_model()`

**Input:** `data: dict[str, Any]` - The project data dictionary containing:
- `traffic_data`: Ship traffic per segment/direction
- `segment_data`: Route segment definitions and lateral distributions
- `objects`: Structure definitions `[id, height, wkt_polygon]`
- `depths`: Depth definitions `[id, depth_value, wkt_polygon]`
- `drift`: Drift parameters (speed, repair times, wind rose, anchoring)
- `pc`: Probability correction factors

**Output:** `tuple[float, float]` - (total_allision_probability, total_grounding_probability)

**Early Exit Conditions:**
- Missing `traffic_data` or `segment_data` → returns (0.0, 0.0)
- No structures AND no depths → returns (0.0, 0.0)

---

## 3. Phase 1: Data Preparation

### 3.1 prepare_traffic_lists()

Extracts traffic data into parallel lists:

```python
def prepare_traffic_lists(data) -> tuple[
    list[LineString],      # lines - route geometries (WGS84)
    list[list[Any]],       # distributions - lateral distributions per leg
    list[list[float]],     # weights - distribution weights per leg
    list[str]              # line_names - "Leg {segment}-{direction}"
]
```

### 3.2 clean_traffic()

Parses the traffic matrix into per-ship-category records.

**Each route segment produces TWO legs** (one per travel direction):

```
Route Segment "1" (Start ↔ End)
    ├── Leg "1-NNW" (k=0): Start_Point → End_Point
    │   ├── Lateral distributions for direction 1
    │   └── Traffic matrix for direction 1
    │
    └── Leg "1-SSE" (k=1): End_Point → Start_Point
        ├── Lateral distributions for direction 2
        └── Traffic matrix for direction 2
```

The geometry is **reversed** for the second direction (`k=1`), so the LineString always
points in the travel direction. Each direction has its own lateral distribution
parameters and its own ship traffic matrix.

```python
def clean_traffic(data) -> list[tuple[
    LineString,            # Route geometry (direction-specific)
    list[Any],             # Lateral distributions (3 normal + 1 uniform)
    list[float],           # Distribution weights
    list[dict],            # Ship categories: {freq, speed, draught, height, ship_type, ship_size}
    str                    # Leg name, e.g. "Leg 1-NNW"
]]
```

**Lateral Distribution Structure:**
- Up to 3 normal distributions: `norm(mean{d}_{i}, std{d}_{i})` with weights `weight{d}_{i}`
- 1 uniform distribution: `uniform(u_min{d}, u_max{d})` with weight `u_p{d}`
- Where `d` = direction index + 1 (1 or 2), corresponding to `k=0` or `k=1`

By the time downstream functions receive the `lines` list, each element is already
a single-direction leg. A project with 4 route segments produces 8 legs (4 segments × 2 directions).

### 3.3 split_structures_and_depths()

Separates obstacles into structures (allision targets) and depths (grounding targets):

```python
def split_structures_and_depths(data) -> tuple[
    list[dict],  # structures: {'id', 'height', 'wkt'}
    list[dict]   # depths: {'id', 'depth', 'wkt'}
]
```

**Important:** MultiPolygons are split into individual Polygons, each retaining the original ID with suffix `_0`, `_1`, etc.

### 3.4 transform_to_utm()

Transforms all geometries from WGS84 (EPSG:4326) to appropriate UTM zone:

```python
def transform_to_utm(lines, objects) -> tuple[
    list[LineString],      # transformed_lines (UTM)
    list[Polygon],         # transformed_objects (UTM)
    int                    # utm_epsg code
]
```

**UTM Zone Selection:**
- Northern hemisphere: EPSG 326XX
- Southern hemisphere: EPSG 327XX
- Zone determined by: `int((centroid_longitude + 180) / 6) + 1`

### 3.5 Geometry Validation

Invalid geometries are fixed using:
1. `shapely.make_valid()` (preferred)
2. `geometry.buffer(0)` (fallback)

Both WKT (UTM) and WKT_WGS84 versions are stored for consistent segment indexing.

---

## 4. Phase 2: Spatial Precomputation

**Location:** `Calculation._precompute_spatial()`

### 4.1 compute_min_distance_by_object()

Calculates minimum distance from each leg to each obstacle for each drift direction.

**Output:** `[leg_idx][direction_idx][object_idx] = distance_meters`

### 4.2 compute_probability_holes_smart_hybrid()

**Location:** `geometries/smart_hybrid_probability_holes.py`

Calculates the "probability hole" — the fraction of ships on a leg that would drift into an obstacle.

Uses the accurate `dblquad` integration method (`geometries/calculate_probability_holes.py`)
for both structures and depths.

### 4.3 compute_probability_holes() — Semi-analytical Integration

**Location:** `geometries/calculate_probability_holes.py`

For each leg/direction/obstacle combination, performs a 2D numerical integration using
`scipy.integrate.dblquad`:

1. **Integration domain:**
   - `s ∈ [0, 1]` — parameter along the leg (0 = start, 1 = end)
   - `y ∈ [-5σ, +5σ]` — lateral offset from leg centreline

2. **Integrand:**
   For each point `(s, y)`, compute the starting position along the leg with
   the lateral offset, then shoot a ray of length `reach_distance` in the drift
   direction. If the ray intersects the obstacle geometry, return the lateral
   PDF value `Σ(wᵢ × PDFᵢ(y))`; otherwise return 0.

3. **Normalise:**
   ```python
   hole_pct = dblquad_result / leg_length
   ```

This gives a pure geometric probability — no empirical decay factors or correction
multipliers. Distance-dependent attenuation is handled separately in the cascade
via `get_not_repaired()` (§7.4).

**Drift Directions (math convention):**
- Index 0 = 0° = East
- Index 1 = 45° = Northeast
- Index 2 = 90° = North
- Index 3 = 135° = Northwest
- Index 4 = 180° = West
- Index 5 = 225° = Southwest
- Index 6 = 270° = South
- Index 7 = 315° = Southeast

---

## 5. Phase 3: Traffic Cascade Calculation

**Location:** `Calculation._iterate_traffic_and_sum()`

### 5.1 Loop Structure

The outer loop iterates over **legs**, not route segments. Since `clean_traffic()` already
expanded each route segment into two legs (one per travel direction, see §3.2), the two
travel directions are implicitly covered by the leg loop — there is no separate direction loop.

```
For each leg (= segment × direction, e.g. 4 segments → 8 legs):
    For each ship category (freq > 0, speed > 0):
        For each drift direction (0-7, 8 wind rose directions):
            Build obstacle list
            Sort by distance
            Process cascade
```

For example, with 4 route segments and 10 ship categories per leg:
- 8 legs × 10 ship categories × 8 drift directions = 640 cascade iterations

### 5.2 Base Calculation

```python
hours_present = (line_length_m / (ship_speed_kts × 1852)) × ship_frequency
blackout_per_hour = drift_p / (365 × 24)
base = hours_present × blackout_per_hour
```

Where:
- `line_length_m`: Route segment length in meters
- `ship_speed_kts`: Ship speed in knots
- `ship_frequency`: Ships per year
- `drift_p`: Annual blackout probability (typically 1.0)

### 5.3 Wind Rose Probability

```python
rp = rose_vals[angle] / sum(rose_vals)
```

Where `angle = d_idx × 45` (compass convention: 0=North, 45=NE, 90=East, etc.)

### 5.4 Obstacle Filtering

**Structures (Allision):**
- Only included if `structure_height < ship_height`

**Depths (Grounding):**
- Only included if `depth < ship_draught`

**Depths (Anchoring):**
- Only included if `depth < anchor_d × ship_draught`
- Where `anchor_d` is the anchor depth multiplier (user-configurable, default 7.0)

### 5.5 Cascade Processing

Obstacles are sorted by distance (closest first). For each obstacle:

```python
remaining_prob = 1.0  # Initially, all ships still drifting

for obstacle in sorted_by_distance:
    if obstacle.type == 'anchoring':
        anchor_contrib = base × rp × remaining_prob × anchor_p × hole_pct
        remaining_prob *= (1.0 - anchor_p × hole_pct)

    elif obstacle.type == 'allision':
        p_nr = get_not_repaired(repair_params, drift_speed, distance)
        contrib = base × rp × remaining_prob × hole_pct × p_nr
        remaining_prob *= (1.0 - hole_pct)

    elif obstacle.type == 'grounding':
        p_nr = get_not_repaired(repair_params, drift_speed, distance)
        contrib = base × rp × remaining_prob × hole_pct × p_nr
        remaining_prob *= (1.0 - hole_pct)
```

### 5.6 get_not_repaired()

**Location:** `compute/basic_equations.py`

Calculates probability that a ship has NOT repaired before reaching the obstacle:

```python
drift_time_hours = distance / drift_speed / 3600

if use_lognormal:
    distribution = lognorm(std, loc, scale)
    prob_not_repaired = 1 - distribution.cdf(drift_time_hours)
else:
    prob_not_repaired = 1 - eval(custom_function)
```

### 5.7 Per-Segment Attribution

For structures and depths, contributions are distributed to individual polygon segments:

1. **Extract segments** from polygon boundary (normalized to CCW orientation)
2. **Check intersection** with drift corridor
3. **Filter by drift direction:**
   - Segment must face the drift direction (outward normal opposes drift)
   - Segment must be ahead of leg in drift direction
4. **Distribute contribution** equally among intersecting segments

**Segment Normal Calculation:**
```python
# For CCW polygon, outward normal is right perpendicular
seg_vec = (x2 - x1, y2 - y1)
outward_normal = (seg_vec.y, -seg_vec.x)  # Rotate 90° clockwise
```

---

## 6. Phase 4: Report Generation

### 6.1 Report Structure

```python
report = {
    'totals': {'allision': float, 'grounding': float, 'anchoring': float},
    'by_leg_direction': {
        'leg_id:direction:angle': {
            'contrib_allision': float,
            'contrib_grounding': float,
            'base_hours': float,
            'ship_categories': {...}
        }
    },
    'by_object': {'Structure - id': {...}, 'Depth - id': {...}},
    'by_structure_legdir': {'Structure - id': {'leg:dir:angle': float}},
    'by_depth_legdir': {'Depth - id': {'leg:dir:angle': float}},
    'by_anchoring_legdir': {'Anchoring - id': {'leg:dir:angle': float}},
    'by_structure_segment_legdir': {'Structure - id': {'seg_0': {...}, 'seg_1': {...}}},
    'by_depth_segment_legdir': {'Depth - id': {'seg_0': {...}, 'seg_1': {...}}},
    'by_anchoring_segment_legdir': {'Anchoring - id': {'seg_0': {...}}}
}
```

### 6.2 Result Layers

**Location:** `geometries/result_layers.py`

Creates QGIS vector layers visualizing:
- **Allision Results:** Line segments colored by probability
- **Grounding Results:** Line segments colored by probability

Each segment includes attributes:
- `obstacle_id`: Original obstacle ID
- `segment_idx`: Segment index within obstacle
- `total_prob`: Total probability for this segment
- `normal_deg`: Outward normal angle (OMRAT convention)
- `leg_1`, `leg_2`, etc.: Per-leg contributions

---

## 7. Key Formulas

### 7.1 Allision/Grounding Contribution

```
contribution = base × rp × remaining_prob × hole_pct × p_nr
```

Where:
- `base = hours_present × blackout_per_hour`
- `rp = wind_rose_probability[direction]`
- `remaining_prob = probability ships still drifting (after earlier obstacles)`
- `hole_pct = geometric probability of hitting obstacle`
- `p_nr = probability ship NOT repaired before reaching obstacle`

### 7.2 Anchoring Shadow

```
remaining_prob *= (1.0 - anchor_p × hole_pct)
```

Where:
- `anchor_p = probability of successful anchoring`
- Ships that successfully anchor don't contribute to subsequent obstacles

### 7.3 Repair Time Distribution (Distance Attenuation)

**Location:** `compute/basic_equations.py` → `get_not_repaired()`

The probability that a drifting ship has NOT been repaired by the time it reaches an
obstacle at a given distance. This is the only distance-dependent factor in the
contribution formula — the probability hole (`hole_pct`) is purely geometric.

```python
drift_time = distance / drift_speed          # seconds
drift_time_hours = drift_time / 3600         # convert to hours
P(not_repaired) = 1 - CDF_lognorm(drift_time_hours; std, loc, scale)
```

Default: Lognormal distribution (configurable via UI or custom function).

### 7.4 Reach Distance

```
reach_distance = drift_speed × 3600 × t99
```

Where `t99 = lognorm.ppf(0.99)` (99th percentile repair time in hours)

---

## 8. Coordinate Systems

### 8.1 Angle Conventions

**Compass Convention (Wind Rose, UI):**
- 0° = North
- 45° = Northeast
- 90° = East
- 135° = Southeast
- 180° = South
- 225° = Southwest
- 270° = West
- 315° = Northwest
- Direction is clockwise from North

**Math Convention (Internal Calculations):**
- 0° = East
- 90° = North
- 180° = West
- 270° = South
- Direction is counter-clockwise from East

**Conversion:**
```python
math_angle = (90 - compass_angle) % 360
compass_angle = (90 - math_angle) % 360
```

**OMRAT Normal Convention (Result Layers):**
- 0° = North
- 90° = West
- 180° = South
- 270° = East
- Direction is counter-clockwise from North

### 8.2 Coordinate Reference Systems

- **Input data:** WGS84 (EPSG:4326)
- **Calculations:** UTM (EPSG:326XX or 327XX)
- **Output layers:** WGS84 (EPSG:4326)

### 8.3 Polygon Orientation

All polygons are normalized to **counter-clockwise (CCW)** exterior orientation using `shapely.geometry.polygon.orient(geom, sign=1.0)`.

This ensures consistent outward normal calculation:
- For CCW polygon, outward normal = rotate segment vector 90° clockwise

---

## 9. Data Assumptions

### 9.1 Traffic Data

- Ship frequencies are in ships/year
- Ship speeds are in knots
- Ship draughts are in meters
- Ship heights are in meters
- Traffic matrix has dimensions [ship_type][ship_size]

### 9.2 Obstacles

- Structure heights are in meters (above sea level)
- Depth values are in meters (positive = below sea level)
- All geometries are valid polygons in WKT format (WGS84)
- MultiPolygons are automatically split into individual polygons

### 9.3 Drift Parameters

- `drift_p`: Annual blackout probability (default 1.0 = 100%)
- `speed`: Drift speed in knots
- `anchor_p`: Successful anchoring probability
- `anchor_d`: Anchor depth multiplier (depth < anchor_d × draught)
- `rose`: Wind rose probabilities by compass direction

### 9.4 Repair Parameters

```python
repair = {
    'use_lognormal': 1,  # or 0 for custom function
    'std': float,        # lognormal std parameter
    'loc': float,        # lognormal loc parameter
    'scale': float,      # lognormal scale parameter
    'func': str          # custom function if use_lognormal=0
}
```

---

## 10. Calculation Parameters

### 10.1 Lateral Distribution

- Default sigma range: 5σ (captures 99.99% of distribution)
- Distributions are typically Gaussian mixtures fit to AIS data

### 10.3 Progress Phases

The calculation reports progress in three phases:
- **Spatial (0-60%):** Probability hole calculations
- **Cascade (60-90%):** Traffic iteration
- **Layers (90-100%):** Result layer generation

---

## Appendix: Function Reference

### Core Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `run_drifting_model()` | run_calculations.py | Main entry point |
| `_build_transformed()` | run_calculations.py | Data preparation |
| `_precompute_spatial()` | run_calculations.py | Spatial calculations |
| `_iterate_traffic_and_sum()` | run_calculations.py | Cascade calculation |
| `compute_probability_holes_pdf_corrected()` | pdf_corrected_fast_probability_holes.py | Probability holes |
| `get_not_repaired()` | basic_equations.py | Repair probability |
| `create_result_layers()` | result_layers.py | QGIS layer output |

### Helper Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `clean_traffic()` | run_calculations.py | Parse traffic data |
| `transform_to_utm()` | run_calculations.py | Coordinate transform |
| `_extract_obstacle_segments()` | run_calculations.py | Polygon segmentation |
| `_segment_intersects_corridor()` | run_calculations.py | Segment-corridor check |
| `_compass_idx_to_math_idx()` | run_calculations.py | Angle convention conversion |
