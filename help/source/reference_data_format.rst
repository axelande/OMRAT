.. _reference-data-format:

============================
``.omrat`` data format
============================

This chapter is the **field-level reference** for the ``.omrat`` JSON
project file and the ``data`` dict passed into every risk model.  Use
it when you're writing a project file by hand, migrating from another
tool, reviewing a test fixture, or tracing a number back to its input.

The file format is **stable for reading** across versions -- older
projects load into newer OMRAT releases via
:class:`~omrat_utils.storage.Storage._normalize_legacy_to_schema`.
The schema validated at load time is defined by
:class:`~omrat_utils.schema.RootModelSchema` (Pydantic).

.. contents:: In this chapter
   :local:
   :depth: 2


Top-level layout
================

An ``.omrat`` file is a single JSON object.  The top-level keys are:

.. code-block:: json

   {
     "pc":              { ... causation factors ... },
     "drift":           { ... drift settings + wind rose + repair ... },
     "traffic_data":    { ... per-leg, per-direction traffic matrices ... },
     "segment_data":    { ... per-leg geometry + distributions + ai ... },
     "depths":          [ [id, depth_m, wkt_polygon], ... ],
     "objects":         [ [id, height_m, wkt_polygon], ... ],
     "ship_categories": { ... type names + LOA bins ... }
   }

Some optional keys are also emitted by the plugin when present:
``results``, ``drifting_report``, ``collision_report``.  These are
calculation **outputs**, not inputs; they're round-tripped so a saved
project reopens with its last result visible.


``pc`` -- causation factors
===========================

.. code-block:: json

   "pc": {
     "p_pc": 0.00016,
     "d_pc": 1e-4,
     "headon": 4.9e-5,
     "overtaking": 1.1e-4,
     "crossing": 1.3e-4,
     "bend": 1.3e-4,
     "grounding": 1.6e-4,
     "allision": 1.9e-4,
     "allision_drifting_rf": 1.0,
     "grounding_drifting_rf": 1.0
   }

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Type
     - Used by
   * - ``headon``, ``overtaking``, ``crossing``, ``bend``
     - float
     - ``run_ship_collision_model``.  Multiplied by the per-pair
       geometric candidate count.
   * - ``grounding``, ``allision``
     - float
     - ``run_powered_grounding_model`` / ``run_powered_allision_model``
       as the :math:`P_c` in :math:`N_{II} = P_c Q m \exp(...)`.
   * - ``allision_drifting_rf``, ``grounding_drifting_rf``
     - float
     - Risk-reduction factor applied to drifting totals after the
       cascade (defaults 1.0 -- set <1 to model e.g. pilot-on-board
       adjustment for a whole area).
   * - ``p_pc``, ``d_pc``
     - float
     - Legacy fields.  Not read by the current risk path.


``drift`` -- drifting model parameters
========================================

.. code-block:: json

   "drift": {
     "drift_p": 1.0,
     "blackout_by_ship_type": {"18": 1.0, "9": 0.1, ...},
     "anchor_p": 0.7,
     "anchor_d": 7.0,
     "speed": 1.94,
     "rose": {"0": 0.125, "45": 0.125, "90": 0.125, "135": 0.125,
              "180": 0.125, "225": 0.125, "270": 0.125, "315": 0.125},
     "repair": {
       "use_lognormal": true,
       "dist_type": "lognormal",
       "std": 1.0,
       "loc": 0.0,
       "scale": 1.0,
       "func": "__import__('scipy.stats', fromlist=['norm'])...cdf(x)"
     }
   }

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Meaning
   * - ``drift_p``
     - Blackout rate in **events per ship-year**.  Default 1.0.
       Multiplied by ``blackout_by_ship_type`` for per-type overrides.
   * - ``blackout_by_ship_type``
     - Dict from ship-type index (string key of integer 0-20) to a
       multiplier.  IWRAP uses 0.1 for RoRo/Passenger, 1.0 otherwise.
   * - ``anchor_p``
     - Conditional probability that a drifting ship in anchorable
       water successfully anchors (saves itself).  Default 0.7.
   * - ``anchor_d``
     - Anchor-depth factor.  A ship with draught :math:`T` can anchor
       in water of depth :math:`< a_d \cdot T`.
   * - ``speed``
     - Drift speed in **knots**.  Converted to m/s internally.
   * - ``rose``
     - Wind rose -- eight compass-angle keys ("0" ... "315") mapping
       to probability weights.  Sum should be 1.0 (normalisation is
       applied if it isn't).
   * - ``repair.dist_type``
     - ``"lognormal"``, ``"weibull"``, or ``"normal"``.  Controls how
       the repair-time CDF is built.
   * - ``repair.std`` / ``loc`` / ``scale``
     - Parameters of the selected distribution.  ``use_lognormal=True``
       selects the lognormal path directly; otherwise ``func`` is
       evaluated (see :func:`compute.basic_equations.get_not_repaired`).
   * - ``repair.func``
     - A Python expression evaluated with ``x`` bound to drift-time
       in hours.  The two shapes OMRAT pattern-matches to a fast
       analytical path are ``.norm(loc=..., scale=...).cdf(x)`` and
       ``.weibull_min(c=..., loc=..., scale=...).cdf(x)``.


``traffic_data`` -- traffic matrix per leg/direction
====================================================

.. code-block:: json

   "traffic_data": {
     "1": {
       "East going": {
         "Frequency (ships/year)": [[0, 12, ...], [...], ...],
         "Speed (knots)":           [[0, 10, ...], [...], ...],
         "Draught (meters)":        [[0, 5.0, ...], [...], ...],
         "Ship heights (meters)":   [[0, 18.0, ...], [...], ...],
         "Ship Beam (meters)":      [[0, 22.0, ...], [...], ...]
       },
       "West going": { ... }
     },
     "2": { ... }
   }

Keys are **segment IDs** (strings).  Nested keys are **direction
labels** matching the segment's ``Dirs`` array in ``segment_data``.
Each variable is a **2-D matrix** with shape
``(n_ship_types, n_loa_bins)`` -- for the default ship categories
that's ``(21, 15)``.

Cells can be ``""``, ``0``, or a numeric value.  Empty cells are
treated as zero and skipped in the inner loops.


``segment_data`` -- per-leg geometry and parameters
====================================================

.. code-block:: json

   "segment_data": {
     "1": {
       "Start_Point": "14.0 55.2",
       "End_Point":   "14.2 55.2",
       "Dirs":        ["East going", "West going"],
       "Width":       1000,
       "line_length": 12850.7,
       "Route_Id":    1,
       "Leg_name":    "leg 1",
       "Segment_Id":  "1",
       "bearing":     90.0,
       "ai1":         180.0,
       "ai2":         180.0,
       "mean1_1": 0.0, "std1_1": 200.0, "weight1_1": 1.0,
       "mean1_2": 0.0, "std1_2": 0.0,   "weight1_2": 0.0,
       "mean1_3": 0.0, "std1_3": 0.0,   "weight1_3": 0.0,
       "mean2_1": 0.0, "std2_1": 200.0, "weight2_1": 1.0,
       "mean2_2": 0.0, "std2_2": 0.0,   "weight2_2": 0.0,
       "mean2_3": 0.0, "std2_3": 0.0,   "weight2_3": 0.0,
       "u_min1": 0.0, "u_max1": 0.0, "u_p1": 0.0,
       "u_min2": 0.0, "u_max2": 0.0, "u_p2": 0.0,
       "dist1": [], "dist2": [],
       "bend_angle": 0.0
     }
   }

Geometry
--------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Key
     - Meaning
   * - ``Start_Point`` / ``End_Point``
     - ``"lon lat"`` as a single space-separated string.  EPSG:4326.
   * - ``Width``
     - Display width (metres).  Not used by the risk integration.
   * - ``line_length``
     - Leg length (metres).  Recomputed whenever geometry changes.
   * - ``Route_Id`` / ``Leg_name`` / ``Segment_Id``
     - Human labels used by the UI and reports.
   * - ``bearing``
     - Compass bearing of Start -> End, degrees.  0 = N, 90 = E,
       clockwise.
   * - ``ai1``, ``ai2``
     - IWRAP position-check interval (seconds) for directions 1 and
       2.  Plugged into :math:`\exp(-d/(a_i V))` for powered models.
   * - ``bend_angle``
     - Bend angle at the downstream waypoint, in degrees.  > 5 deg
       enables the bend-collision formula for this leg.

Lateral distribution
--------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Key
     - Meaning
   * - ``mean{d}_{i}``
     - Mean of the :math:`i`-th Gaussian component for direction
       :math:`d` (d = 1 or 2; i = 1, 2, 3).  Metres.
   * - ``std{d}_{i}``
     - Standard deviation of the same.
   * - ``weight{d}_{i}``
     - Component weight (weights are normalised to sum to 1).
   * - ``u_min{d}`` / ``u_max{d}`` / ``u_p{d}``
     - Uniform-component bounds and weight.
   * - ``dist1`` / ``dist2``
     - Legacy raw-PDF arrays (unused in the current risk path).


``depths`` -- bathymetry polygons
==================================

A flat array of ``[id, depth_m, wkt_polygon]`` triples:

.. code-block:: json

   "depths": [
     ["d1",  "5.0",  "POLYGON((14.08 55.22, ...))"],
     ["d2",  "10.0", "POLYGON((...))"],
     ...
   ]

* ``id`` is a short label used in results (``d1``, ``d2``, ...).
* ``depth`` is stored as a **string** of a number -- OMRAT casts to
  float internally.  Chart-datum metres below MSL.
* ``wkt`` is a valid shapely / OGC WKT polygon in EPSG:4326.
  MultiPolygons are accepted; OMRAT splits them into components on
  load.

Used by: drifting grounding (as a hazard), drifting anchoring (if
shallow enough), powered grounding (depth-bin filtering).


``objects`` -- structure polygons
==================================

Same shape as ``depths`` but with ``height`` instead of depth:

.. code-block:: json

   "objects": [
     ["s1", "20.0", "POLYGON((14.09 55.208, ...))"],
     ["s2", "50.0", "POLYGON((...))"]
   ]

* ``height`` -- height above waterline (metres).  A ship with
  ``ship_height < object_height`` passes under the structure without
  colliding (powered allision only).

Used by: drifting allision (any drift contact), powered allision (with
height filter).


``ship_categories`` -- type + LOA bin definitions
====================================================

.. code-block:: json

   "ship_categories": {
     "types": [
       "Other ship", "Search & rescue", "Sailing vessel", ...
     ],
     "length_intervals": [
       {"min": 0.0,    "max": 50.0,   "label": "0-50"},
       {"min": 50.0,   "max": 100.0,  "label": "50-100"},
       ...
     ],
     "selection_mode": "ais"
   }

* ``types`` -- display names for the 21 ship-type rows in the Traffic
  matrix.
* ``length_intervals`` -- the LOA bin columns.  Each bin has
  ``min`` / ``max`` (metres) and a ``label``.  The **midpoint** of a
  bin is used to estimate ship length when the traffic cell stores
  only frequency.
* ``selection_mode`` -- ``"ais"`` or ``"manual"``.  Controls whether
  the UI wires **Update AIS** buttons.


Output keys (round-tripped, not inputs)
=========================================

When a project is saved **after** a successful run, the risk results
are written into the same file for convenience.  They are NOT inputs
-- loading a file ignores them; **Run Model** overwrites them.

.. code-block:: json

   "results": {
     "drifting_allision_prob": 1.148e-01,
     "drifting_grounding_prob": 8.329e-03,
     "ship_collision_prob": 5.2e-06,
     "powered_grounding_prob": 3.1e-05,
     "powered_allision_prob": 1.7e-06
   },
   "drifting_report": {
     "totals":   {"allision": ..., "grounding": ..., "anchoring": ...},
     "by_leg_direction": { ... },
     "by_object":         { ... },
     "by_structure_legdir": { ... },
     "by_depth_legdir":     { ... },
     "by_anchoring_legdir": { ... },
     "by_structure_segment_legdir": { ... },
     "by_depth_segment_legdir":     { ... },
     "by_anchoring_segment_legdir": { ... },
     "debug_obstacles": { ... if debug_trace is enabled ... }
   },
   "collision_report": {
     "totals": {"head_on": ..., "overtaking": ..., "crossing": ..., "bend": ..., "total": ...},
     "by_leg": { ... },
     "causation_factors": {...}
   }


Optional / developer-only flags
=================================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Meaning
   * - ``data['use_analytical']``
     - ``True`` (default) -> analytical cross-section CDF integration.
       ``False`` -> Monte Carlo sampler.  See :ref:`code-flow-drifting`.
   * - ``data['drift']['debug_trace']``
     - ``True`` -> per-obstacle debug block added to the drifting
       report (see "Debug trace" in :ref:`code-flow-drifting`).
   * - ``data['drift']['use_leg_offset_for_distance']``
     - ``True`` -> measure directional distances from the leg's
       mean-offset line instead of the centerline.  Default
       ``False`` (OMRAT's runtime uses ``mean_offset_m = 0`` anyway).


Run history: master DB + per-run GeoPackages
=============================================

OMRAT splits run history into two locations to keep the master
database small even after many runs.

Master history database: ``omrat_history.sqlite``
--------------------------------------------------

One file in the user's app-data folder (see :ref:`user_guide`),
auto-created on first run.  Holds **only metadata** -- one row per
run, no spatial features -- so a plain SQLite file suffices and the
extension correctly advertises that.

Existing installs that have an older ``omrat_history.gpkg`` (or
``omrat_runs.gpkg`` from earlier releases) are auto-migrated on the
next plugin start: the file is renamed in place; nothing is lost
because the format is identical.

Single table: ``omrat_runs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Meaning
   * - ``run_id``
     - Auto-increment primary key.
   * - ``name``
     - Model name from the Run Analysis tab, or
       ``run_YYYYMMDD_HHMMSS`` if left blank.
   * - ``timestamp``
     - Save time as ``YYYY-MM-DD HH:MM:SS``.
   * - ``duration_seconds``
     - Wall time of the run, in seconds (``time.monotonic`` based).
   * - ``drift_allision``, ``drift_grounding``, ``drift_anchoring``
     - Drifting cascade totals (annual frequency).
   * - ``powered_grounding``, ``powered_allision``
     - Cat II powered totals.
   * - ``head_on``, ``overtaking``, ``crossing``, ``bend``,
       ``ship_collision_total``
     - Ship-collision totals.
   * - ``output_dir``
     - Folder containing the per-run GeoPackage (the user-selected
       output folder at the time of the run).
   * - ``output_filename``
     - Filename of the per-run GeoPackage inside ``output_dir``.
       Built as ``<slug(name)>_<YYYYMMDD_HHMMSS>.gpkg`` so each
       run has a unique filename.
   * - ``notes``
     - Free-text user note (currently unused; reserved).

Per-run GeoPackages
-------------------

One file per run, written under the user-selected output folder.
Filename: ``<slug(name)>_<YYYYMMDD_HHMMSS>.gpkg``.  Each contains up
to six layers (only those with non-zero contributions are written):

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Layer
     - Geometry
     - Attributes
   * - ``drifting_allision``
     - Polygon
     - ``obstacle_id``, segment-level + per-leg breakdown columns
       inherited from the live result-layer factory.
   * - ``drifting_grounding``
     - Polygon
     - same shape as Allision.
   * - ``powered_grounding``
     - Polygon
     - ``obstacle_id``, ``value`` (depth), ``total_prob``,
       per-leg ``leg_<id>`` columns.
   * - ``powered_allision``
     - Polygon
     - ``obstacle_id``, ``value`` (height), ``total_prob``,
       per-leg ``leg_<id>`` columns.
   * - ``collision_lines``
     - LineString
     - ``leg_id``, ``head_on``, ``overtaking``, ``combined``.
   * - ``collision_points``
     - Point
     - ``waypoint``, ``crossing``, ``bend``, ``combined``.

Open a per-run file directly in QGIS (it's a standard OGC
GeoPackage), or use **Add selected run results to map** on the
Previous-runs table to load all six layers in one click with
graduated styling applied.


Minimal valid project (synthetic)
==================================

The smallest project that passes Pydantic validation and runs the
drifting cascade is in ``tests/test_cascade_minimal.py``
(``_build_minimal_project``): one east-going 10 km leg, one 12 m
depth polygon NW of the leg, one 20 m structure N of the leg, a
single cargo-type cell with 100 ships/year at 10 knots / 13 m
draught.  Useful as a starting skeleton when building a fixture.
