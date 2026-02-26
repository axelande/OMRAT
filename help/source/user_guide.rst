.. _user_guide:

==========
User Guide
==========

This chapter provides step-by-step instructions for using OMRAT to
perform a maritime risk assessment.

.. contents:: In this chapter
   :local:
   :depth: 2


Quick Start
===========

A typical OMRAT workflow consists of five steps:

1. **Define routes** -- Digitise shipping lanes on the map
2. **Add traffic data** -- Enter ship frequencies and dimensions
3. **Define obstacles** -- Add depth and structure polygons
4. **Configure parameters** -- Set drift settings and causation factors
5. **Run calculations** -- Execute the risk assessment

Each step is described in detail below.


Step 1: Defining Routes
========================

Opening the Plugin
-------------------

Click the OMRAT icon in the QGIS toolbar, or go to **Plugins** > **Open
Maritime Risk Analysis Tool** > **Omrat**. The OMRAT dock widget opens
on the right side of the QGIS window.

Digitising a Route
--------------------

1. Go to the **Route** tab in the OMRAT widget.
2. Click **Add Route** to start digitising.
3. Click on the map to set the first waypoint. A point marker appears.
4. Click again to create a leg segment between the first and second
   points. A blue line appears on the map.
5. Continue clicking to add more segments to the route. Each click
   creates a new segment from the previous point.
6. Click **Stop Route** when the route is complete.

Each segment is automatically assigned:

- A **Segment ID** and **Route ID**
- **Direction labels** (e.g., "North going" / "South going") based on
  the segment orientation
- A default **width** of 5000 m (editable in the route table)

Route Width
-----------

The route width is shown as a perpendicular line at the midpoint of each
segment. To change the width:

1. Edit the **Width** column in the route table
2. The visual offset line updates automatically


Step 2: Traffic Data
=====================

Selecting a Segment
--------------------

1. Go to the **Traffic** tab.
2. Select a segment from the **Segment** dropdown.
3. Select a direction from the **Direction** dropdown.
4. Select a data variable (Frequency, Speed, Draught, Height, Beam).

Entering Data
--------------

The traffic table has rows for each ship type and columns for each ship
size category. Enter:

- **Frequency** (ships/year): Integer count of vessels per year
- **Speed** (knots): Average speed for the ship type/size
- **Draught** (metres): Average draught
- **Height** (metres): Average height above waterline
- **Beam** (metres): Average beam (width)

Importing from AIS
-------------------

If you have access to an AIS database:

1. Go to **Settings** > **AIS connection settings**
2. Enter the database connection parameters
3. Click **Update AIS** next to a segment in the route table
4. OMRAT queries the database for vessel passages and populates the
   traffic table automatically

Ship Categories
----------------

To customise ship type and size classifications:

1. Go to **Settings** > **Ship Categories**
2. Edit the type names and size bin boundaries
3. The traffic table dimensions will update accordingly


Step 3: Defining Obstacles
===========================

Depth Polygons
--------------

Depths represent bathymetry (water depth) areas. Ships can ground in
shallow areas.

**Manual entry:**

1. Go to the **Depths** tab
2. Click **Add Simple Depth**
3. Enter a depth value and draw a polygon on the map

**Loading from shapefile:**

1. Click **Load Depth**
2. Select a shapefile containing depth polygons

**GEBCO download:**

1. Enter your OpenTopography API key
2. Set the bounding box extension (%)
3. Click **Get GEBCO Depths**
4. The plugin downloads and vectorises bathymetry data automatically

Structure Polygons
-------------------

Structures are physical objects (bridges, wind turbines, platforms) that
ships can collide with.

1. Go to the **Objects** tab
2. Click **Add Simple Object** or **Load Object**
3. Enter a height value for the structure


Step 4: Configuration
======================

Drift Settings
---------------

Go to **Settings** > **Drift settings** to configure:

- **Drift probability**: Blackout frequency
- **Anchor probability**: Likelihood of successful anchoring
- **Max anchor depth**: Maximum depth for anchoring (metres)
- **Drift speed**: Speed of drifting ship (m/s)
- **Wind rose**: Probability of wind from each of 8 directions
  (must sum to 100%)
- **Repair time**: Lognormal distribution parameters or custom function

Causation Factors
------------------

Go to **Settings** > **Causation Factors** to configure:

- Powered grounding causation factor (default: :math:`1.6 \times 10^{-4}`)
- Drifting causation factor (default: 1.0)
- Ship-ship collision factors (head-on, overtaking, crossing, bend)

Lateral Distributions
-----------------------

Click on a segment in the route table to view and edit its lateral
traffic distribution on the **Distributions** tab:

- Up to 3 normal distributions with mean, std, and weight
- 1 uniform distribution with min, max, and probability
- Weights are normalised to sum to 100%
- A plot shows the combined distribution


Step 5: Running Calculations
=============================

Full Risk Assessment
---------------------

1. Go to the **Results** tab
2. Optionally enter a model name
3. Click **Run Model**
4. The calculation runs in the background (check the QGIS Task Manager
   for progress)
5. Results appear in the results fields:

   - **Drift Allision**: Probability of drifting into structures
   - **Powered Allision**: Probability of powered collision with structures
   - **Drift Grounding**: Probability of drifting onto shallow water
   - **Powered Grounding**: Probability of powered grounding
   - **Overtaking Collision**: Overtaking collision frequency
   - **Head-On Collision**: Head-on collision frequency
   - **Crossing Collision**: Crossing collision frequency
   - **Merging Collision**: Merging collision frequency

Drift Corridor Analysis
-------------------------

For a visual analysis of drift corridors:

1. Go to the **Drift Analysis** tab
2. Set depth and height thresholds
3. Click **Run Drift Analysis**
4. Coloured polygon layers appear on the map, one per leg, showing the
   8-directional drift corridors

Viewing Detailed Results
-------------------------

Click the **View** buttons next to each result to see detailed
breakdowns by segment and obstacle.


Saving and Loading Projects
============================

Saving
------

Go to **File** > **Save** and choose a location for the ``.omrat`` file.
All routes, traffic data, obstacles, settings, and results are saved.

Loading
-------

Go to **File** > **Load** and select an ``.omrat`` file. You can choose
to:

- **Clear & Load**: Replace the current model completely
- **Merge**: Add the loaded data to the existing model

IWRAP Import/Export
--------------------

- **Export**: **File** > **Export to IWRAP XML** -- saves the current
  model in IWRAP-compatible XML format
- **Import**: **File** > **Import from IWRAP XML** -- loads an IWRAP XML
  file into OMRAT


Interpreting Results
====================

Result values are **annual accident frequencies** -- the expected number
of accidents per year for each type.

- Values in the range :math:`10^{-3}` to :math:`10^{-2}` indicate a
  relatively frequent accident type
- Values in the range :math:`10^{-5}` to :math:`10^{-4}` are typical for
  well-managed waterways
- Values below :math:`10^{-6}` are very rare events

Results can be compared against:

- IALA risk acceptance criteria
- Historical accident data for the area
- Results from IWRAP Mk2 for cross-validation


Tips and Best Practices
========================

- Always check that **lateral distributions** are correctly configured
  for each segment -- these have a large impact on results
- Verify that **depth polygons** cover the relevant shallow areas --
  missing depth data leads to underestimated grounding risk
- Use the **drift corridor visualisation** to sanity-check that
  corridors reach the correct obstacles
- Start with **default causation factors** and adjust only if local
  data supports different values
- For large study areas, consider running calculations overnight as
  the Monte Carlo integration can be time-consuming
