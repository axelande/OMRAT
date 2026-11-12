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
:meth:`~omrat_utils.storage.Storage._normalize_legacy_to_schema`.
The schema validated at save and load time is
:class:`~omrat_utils.validate_data.RootModelSchema` (Pydantic, in
``omrat_utils/validate_data.py``); keys it does not name are ignored.
Before the check, both the writer (``GatherData.get_all_for_save``) and
the reader run ``omrat_utils/project_sanitize.sanitize_project``: empty
observation cells become their mean (``Infinity`` when empty), legs never
shown in the Distributions tab get its defaults, and traffic blocks whose
leg no longer exists are dropped.

.. contents:: In this chapter
   :local:
   :depth: 2


Top-level layout
================

An ``.omrat`` file is a single JSON object.  The top-level keys are:

.. code-block:: text

   {
     "omrat_version":     "0.18.1",
     "project_name":      "My Analysis",
     "pc":                { ... causation factors ... },
     "drift":             { ... drift settings + wind rose + repair ... },
     "traffic_data":      { ... per-leg, per-direction traffic matrices ... },
     "segment_data":      { ... per-leg geometry + distributions + ai ... },
     "waypoints":         { ... shared leg end nodes ... },
     "depths":            [ [id, depth_m, wkt_polygon], ... ],
     "objects":           [ [id, height_m, wkt_polygon], ... ],
     "ship_categories":   { ... type names + LOA bins ... },
     "junctions":         { ... transition matrix per junction ... },
     "consequence":       { ... oil-spill consequence inputs ... },
     "traffic_scaling":   { ... global multiplier + per-row follow flags ... },
     "layer_styles":      { ... QGIS styles of the OMRAT layers ... },
     "segments_imported": { ... first endpoints of each leg ... },
     "speed_restriction_zones": [ ... ]
   }

Only ``pc``, ``drift``, ``traffic_data``, ``segment_data``, ``depths``
and ``objects`` are required.  Every other block is optional; when it
is missing the load uses the default (waypoints derived from the leg
ends, geometric junction matrices, default consequence inputs, 100 %
scaling, no stored styles).

``project_name`` (optional string) is the **Project name** from the Run
Analysis tab; Previous runs groups the project's runs under it, and the
IWRAP export uses it as the model name.  It is left out when the field
is empty, and older files without it load unchanged.  ``omrat_version``
records the OMRAT release that wrote the file.
``speed_restriction_zones`` (optional, see below) holds the speed
restriction zones and is left out when there are none.

Results are **not** stored in the project file.  Each run writes them
next to its GeoPackage (see `Files next to each run`_); the run's input
snapshot ``<stem>.omrat`` has the same format as a project file.


``pc`` -- causation factors
===========================

.. code-block:: json

   "pc": {
     "p_pc": 0.00016,
     "d_pc": 1,
     "headon": 4.9e-5,
     "overtaking": 1.1e-4,
     "crossing": 1.3e-4,
     "merging": 1.3e-4,
     "bend": 1.3e-4,
     "grounding": 1.6e-4,
     "allision": 1.9e-4,
     "grounding_cat1": 1.6e-4,
     "allision_cat1": 1.9e-4,
     "cat1_skip_start": true,
     "cat1_skip_start_pct": 10.0,
     "min_encounter_angle_deg": 10.0,
     "passenger_reduction": 20.0,
     "fast_ferry_reduction": 20.0,
     "squat_drifting": true,
     "squat_powered": true,
     "allision_drifting_rf": 1.0,
     "grounding_drifting_rf": 1.0
   }

``p_pc`` and ``d_pc`` are required by the schema.  Any other key the
file lacks takes the default shown (``default_pc_values`` in
``compute/iwrap_defaults.py``, or 1.0 for the two drifting ``*_rf``
factors); the Causation factors dialog edits all of them except the
``*_rf`` factors.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Type
     - Used by
   * - ``headon``, ``overtaking``, ``crossing``, ``merging``,
       ``bend``
     - float
     - ``run_ship_collision_model``.  Multiplied by the per-pair
       geometric candidate count.  ``merging`` was added in v0.14.0;
       a project file written by an earlier version gets the default
       (1.3e-4, the crossing figure).
   * - ``min_encounter_angle_deg``
     - float
     - Smallest angle (degrees, 1-45, default 10) between two routes
       meeting at a waypoint for a crossing / merging / bend encounter to
       count (``compute/encounter_angle.py``,
       :ref:`angle-limit-at-waypoints`).  Head-on and overtaking are not
       affected.  Not carried by the IWRAP export.
   * - ``passenger_reduction``, ``fast_ferry_reduction``
     - float
     - IWRAP's reduction factor :math:`F` for passenger ships and fast
       ferries (default 20; 1 = no reduction;
       ``compute/passenger_reduction.py``).  Ship-ship: the causation
       factor of an encounter is multiplied by :math:`1/\sqrt{F}` per
       such ship.  Powered grounding / allision: their frequency is
       divided by :math:`F`.  Drifting is not affected.
   * - ``grounding``, ``allision``
     - float
     - ``run_powered_grounding_model`` / ``run_powered_allision_model``
       as the :math:`P_c` in :math:`N_{II} = P_c Q m \exp(...)`.
       These are **Category II** (missed-turn) factors, so IWRAP
       export writes them to ``p_grounding_no_turn_causation`` /
       ``p_allision_no_turn_causation`` -- see
       :ref:`iwrap-causation-mapping`.
   * - ``grounding_cat1``, ``allision_cat1``
     - float
     - The same two models, as the :math:`P_c` in the **Category I**
       (obstacle already in the lane) term :math:`N_I = P_c Q m`.
       Added in v0.15.0; a project file without the keys uses the
       IWRAP default (identical to the Cat II figure).  IWRAP export
       writes them to ``p_grounding_causation`` /
       ``p_allision_causation``.
   * - ``cat1_skip_start``, ``cat1_skip_start_pct``
     - bool, float
     - Category I start zone (default on, 10 %;
       ``compute/cat1_start_zone.py``, :ref:`cat1-start-zone`): in-lane
       hits within the first ``cat1_skip_start_pct`` % of a leg, seen
       from the waypoint the ships come from, are ignored.  Category II
       is untouched.  Not carried by the IWRAP export.
   * - ``squat_drifting``, ``squat_powered``
     - bool
     - Add squat (Barras open water, :math:`C_b V^2 / 100`, at the
       cell's speed; ``compute/squat.py``) to the draught in drifting
       grounding and powered grounding.  Default on.  Anchoring keeps
       the static draught; powered allision (height clearance) is
       unchanged.
   * - ``allision_drifting_rf``, ``grounding_drifting_rf``
     - float
     - Risk-reduction factor applied to every drifting allision /
       grounding contribution (defaults 1.0 -- set <1 to model e.g.
       pilot-on-board adjustment for a whole area).  The run's totals,
       breakdowns and result layers all include it; the drifting report
       also lists the totals before it (``totals_before_rf``).
   * - ``p_pc``, ``d_pc``
     - float
     - Legacy fields.  ``p_pc`` is still a fallback for
       ``grounding`` and is kept in step with it on IWRAP import;
       ``d_pc`` is not read by the current risk path.


.. _iwrap-causation-mapping:

IWRAP causation-factor mapping
==============================

IWRAP splits powered grounding and allision into two categories,
each with its own causation factor:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - IWRAP attribute
     - Meaning
   * - ``p_grounding_causation``,
       ``p_allision_causation``
     - **Category I** -- the obstacle already lies in the lane; no
       course change was required.
   * - ``p_grounding_no_turn_causation``,
       ``p_allision_no_turn_causation``
     - **Category II** -- a turn *was* required at a waypoint and
       the ship failed to make it.

OMRAT models both categories (see :ref:`powered`).  ``grounding`` and
``allision`` are the Category-II factors, ``grounding_cat1`` and
``allision_cat1`` the Category-I ones:

* **Export** writes each to its own attribute.  A project file that
  predates the ``*_cat1`` keys exports its Category-II value on the
  Category-I attribute rather than a constant, because IWRAP computes
  the Category-I geometry regardless of what we write -- scoring it
  with a factor nobody chose is how the pre-v0.14.0 exporter
  produced unaccountable numbers.
* **Import** puts ``p_*_causation`` on ``*_cat1`` and
  ``p_*_no_turn_causation`` on ``grounding`` / ``allision``.  The
  Category-II keys also accept the Category-I attribute as a fallback
  when a file sets only that one, with ``_no_turn`` winning when both
  are present.

The two directions are driven by ``_CF_EXPORT_MAP`` and
``_CF_IMPORT_MAP`` in ``compute/iwrap_convertion.py``; a test
asserts they name the same attribute set, so a round-trip through
IWRAP XML cannot silently lose a factor.

.. warning:: Changed in v0.14.0

   Before v0.14.0 the exporter ignored ``pc`` entirely and wrote a
   fixed block of constants, and the importer put IWRAP's allision
   causation into an unread key.  Both directions now round-trip.
   With IWRAP's own defaults the two categories carry identical
   values, so this only changes results for a project that tunes
   them -- but a benchmark comparison is exactly that case.


``drift`` -- drifting model parameters
========================================

.. code-block:: text

   "drift": {
     "drift_p": 1,
     "blackout_by_ship_type": {"18": 1.0, "9": 0.1, ...},
     "anchor_p": 0.7,
     "anchor_d": 7,
     "speed": 1.0,
     "start_from": "leg_center",
     "rose": {"0": 0.125, "45": 0.125, "90": 0.125, "135": 0.125,
              "180": 0.125, "225": 0.125, "270": 0.125, "315": 0.125},
     "repair": {
       "use_lognormal": true,
       "std": 0.95,
       "loc": 0.2,
       "scale": 0.85,
       "func": ""
     }
   }

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Meaning
   * - ``drift_p``
     - Blackout rate in **events per ship-year**.  Default 1.  Used for
       every ship type that has no entry in ``blackout_by_ship_type``.
   * - ``blackout_by_ship_type``
     - Dict from ship-type index (string key of the traffic-matrix row)
       to that type's own blackout rate in events per ship-year; it
       replaces ``drift_p`` for the type.  The default is 0.1 for
       passenger, ferry, ro-ro, cruise and high-speed-craft types (matched
       by name) and 1.0 for the rest, as IWRAP's two blackout classes.
   * - ``anchor_p``
     - Conditional probability that a drifting ship in anchorable
       water successfully anchors (saves itself).  Default 0.7.
   * - ``anchor_d``
     - Anchor-depth factor.  A ship with draught :math:`T` can anchor
       in water of depth :math:`< a_d \cdot T`.
   * - ``speed``
     - Drift speed in **knots** (default 1.0).  Converted to m/s
       internally.
   * - ``start_from``
     - **Start drifting from** (``compute/drift_start.py``):
       ``"leg_center"`` (default) measures the drift distance to an
       obstacle from the leg line; ``"distribution_center"`` from a line
       offset by the mean of the direction's lateral distribution.
   * - ``rose``
     - Wind rose -- eight compass-angle keys ("0" = N ... "315" = NW,
       the drift direction) mapping to probability weights.  The
       Drift settings dialog scales them to sum to 1.0 on OK.
   * - ``repair.use_lognormal`` / ``std`` / ``loc`` / ``scale``
     - ``use_lognormal=true`` uses a lognormal repair time in hours
       with shape ``std``, ``loc`` and ``scale``; otherwise ``func`` is
       evaluated (see :func:`compute.basic_equations.get_not_repaired`).
   * - ``repair.func``
     - A Python expression evaluated with ``x`` bound to drift-time
       in hours.  The two shapes OMRAT pattern-matches to a fast
       analytical path are ``.norm(loc=..., scale=...).cdf(x)`` and
       ``.weibull_min(c=..., loc=..., scale=...).cdf(x)``.
   * - ``repair.dist_type``
     - Optional, written by the IWRAP import (``"normal"`` /
       ``"weibull"``, with ``norm_mean`` / ``norm_std`` or ``wb_*``
       parameters) to say which shape ``func`` holds; the drift reach
       (``compute/drift_reach.py``) reads it.

``squat_mode`` (a former Drift settings combo) and
``use_leg_offset_for_distance`` are no longer read: squat lives in
``pc`` (``squat_drifting`` / ``squat_powered``) and the start line in
``start_from``.  Old files that still carry them load unchanged.


``traffic_data`` -- traffic matrix per leg/direction
====================================================

.. code-block:: text

   "traffic_data": {
     "1": {
       "East going": {
         "Frequency (ships/year)": [[0, 12, ...], [...], ...],
         "Speed (knots)":           [[0, 10, ...], [...], ...],
         "Draught (meters)":        [[0, 5.0, ...], [...], ...],
         "Ship heights (meters)":   [[0, 18.0, ...], [...], ...],
         "Ship Beam (meters)":      [[0, 22.0, ...], [...], ...],
         "Scaling (%)":             [[100, 100, ...], [...], ...]
       },
       "West going": { ... }
     },
     "2": { ... }
   }

Keys are **segment IDs** (strings) and must exist in ``segment_data``
(orphan blocks are dropped on save, load and run).  Each leg has
exactly two nested blocks keyed by the **direction labels** of the
segment's ``Dirs`` array.  Compute and the IWRAP export read the blocks
**by position**: the first block is ``Dirs[0]``, the ships sailing the
drawn direction (``Start_Point -> End_Point``).
Each variable is a **2-D matrix** with shape
``(n_ship_types, n_loa_bins)`` -- for the default ship categories
that's ``(21, 15)``.  Rows are ship types, columns length classes.

Cells are numbers.  A ``Frequency`` of ``0`` means no ships.  In the
other matrices ``Infinity`` (``inf``) means "ships but no observation"
(what an AIS update writes for such a cell).  Every run fills missing
speeds and draughts first (``compute/traffic_gaps.py``, see
:ref:`traffic-gaps`); a ship height of ``0`` or ``inf`` is unknown and
gets no height clearance.

``Scaling (%)`` is a per-cell **frequency multiplier in percent**
(default ``100`` -- no scaling).  Applied once per run, on the run's
private copy, by
:func:`compute.traffic_pipeline.prepare_traffic_for_run`
(:func:`compute.traffic_scaling.apply_traffic_scaling`, re-exported by
``compute.data_preparation``):
``Q_effective = Q * Scaling / 100``.  Every model reads
``Frequency (ships/year)`` afterwards, so the multiplier is inherited
by ship-ship, powered, drifting, and consequence in one shot.  AIS
refresh + IWRAP import never overwrite this column, so a saved
``+30 %`` survives every traffic update.


``traffic_scaling`` -- project-level scaling state
====================================================

.. code-block:: text

   "traffic_scaling": {
     "global_percent": 130.0,
     "follow_global": [true, false, true, true, ...]
   }

Companion to the per-cell ``Scaling (%)`` matrices in ``traffic_data``.

* ``global_percent`` -- the value of the **Global scaling [%]** spinbox
  on the Traffic Data tab.  Re-broadcast into the per-cell matrix whenever
  it changes, but only for ship-type rows whose ``follow_global`` flag
  is ``true``.  Compute never reads this number directly -- it reads
  the per-cell matrix that the broadcast has already updated.
* ``follow_global`` -- one boolean per ship-type row.  ``true`` means
  "this type's Scaling cells track the global spinbox".  Toggling a
  row ``true -> false`` keeps the current cell values; toggling
  ``false -> true`` re-broadcasts the current global.  **Editing a
  Scaling cell manually auto-unticks the row** -- the typed value is
  the explicit "I want this row at X %" signal.

Optional -- legacy projects pre-date this feature.  The legacy
normaliser in :class:`omrat_utils.storage.Storage` seeds the block
with ``global_percent = 100.0`` and ``follow_global = []`` (the
empty list grows back to ``n_ship_types`` on first save).


``speed_restriction_zones`` -- speed limits in areas
====================================================

.. code-block:: json

   "speed_restriction_zones": [
     {"id": 1, "name": "Anholt OWF", "enabled": true,
      "limit_kn": 10.0, "compliance_pct": 100.0,
      "wkt": "Polygon ((11.75 56.55, 11.90 56.55, 11.90 56.65, 11.75 56.65, 11.75 56.55))",
      "rules": [
        {"ship_type": 17, "length_min": null, "length_max": null,
         "restricted": false, "limit_kn": null, "compliance_pct": null},
        {"ship_type": null, "length_min": 100.0, "length_max": null,
         "restricted": true, "limit_kn": 6.0, "compliance_pct": null}
      ]}
   ]

Optional; left out when the project has no zones.  Written by
**Settings > Speed restriction zones...** (:ref:`speed-restriction-zones`),
read by :mod:`compute.speed_restriction`.

* ``wkt`` -- polygon or multipolygon in EPSG:4326.  Enabled zones may not
  overlap (shared edges are fine).
* ``limit_kn`` / ``compliance_pct`` -- the zone default.
* ``rules`` -- tried in order, the first match wins.  ``ship_type`` is the
  row index of the traffic matrix (``null`` = all types);
  ``length_min`` / ``length_max`` are length class boundaries (a class
  ``(lo, hi]`` matches when ``lo >= length_min`` and ``hi <= length_max``);
  ``restricted: false`` exempts the ships, otherwise ``limit_kn`` /
  ``compliance_pct`` override the zone (``null`` = the zone value).

Compute applies the zones on its private copy after the gap fill, the
scaling and the suppressed-leg moves (the last step of
:func:`compute.traffic_pipeline.prepare_traffic_for_run`); overlapping
or invalid enabled zones stop **Run model**.  ``traffic_data`` in the
file always holds the
measured speeds.  On that copy a leg in a zone gets the time-equivalent
``Speed (knots)`` and two extra matrices, ``Speed at start point (knots)``
and ``Speed at end point (knots)``, which are never saved.


``segment_data`` -- per-leg geometry and parameters
====================================================

.. code-block:: json

   "segment_data": {
     "1": {
       "Start_Point": "14.000000 55.200000",
       "End_Point":   "14.200000 55.200000",
       "start_wp":    "1",
       "end_wp":      "2",
       "Dirs":        ["East going", "West going"],
       "Width":       5000,
       "Tangent_Pos": 0.5,
       "line_length": 12850.7,
       "Route_Id":    1,
       "Leg_name":    "LEG_1_1",
       "Segment_Id":  "1",
       "ai1":         180.0,
       "ai2":         180.0,
       "mean1_1": -200.0, "std1_1": 300.0, "weight1_1": 100,
       "mean1_2": 0.0, "std1_2": 0.0,   "weight1_2": 0,
       "mean1_3": 0.0, "std1_3": 0.0,   "weight1_3": 0,
       "mean2_1": 200.0, "std2_1": 300.0, "weight2_1": 100,
       "mean2_2": 0.0, "std2_2": 0.0,   "weight2_2": 0,
       "mean2_3": 0.0, "std2_3": 0.0,   "weight2_3": 0,
       "u_min1": 0.0, "u_max1": 0.0, "u_p1": 0,
       "u_min2": 0.0, "u_max2": 0.0, "u_p2": 0,
       "dist1": [], "dist2": [],
       "traffic_locked": false
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
     - ``"lon lat"`` as a single space-separated string with six
       decimals.  EPSG:4326.  Junctions match on these exact strings.
   * - ``start_wp`` / ``end_wp``
     - Ids of the shared nodes in the top-level ``waypoints`` block.
       ``Start_Point`` / ``End_Point`` are the derived copy of those
       nodes.  Optional; derived from the coordinates when missing.
   * - ``Dirs``
     - The two direction labels (``"East going"``, ``"West going"``,
       ...).  ``Dirs[0]`` is always the drawn direction; the names come
       from 90-degree bearing sectors and only label the traffic blocks.
   * - ``Width``
     - Length (metres) of the tangent line, which is also the AIS
       passage line; written to IWRAP as ``max_width``.  Not used by
       the risk integration.
   * - ``Tangent_Pos``
     - Where the tangent / AIS passage line crosses the leg, as a
       fraction ``0..1`` from ``Start_Point`` (default ``0.5``).  Only
       the AIS sample reads it (:ref:`moving-the-tangent-line`).
   * - ``line_length``
     - Leg length (metres).  Recomputed whenever geometry changes.
   * - ``Route_Id`` / ``Leg_name`` / ``Segment_Id``
     - ``Segment_Id`` is the leg's global id (the ``segment_data``
       key; never reused).  ``Leg_name`` is ``LEG_<route>_<n>`` with
       the per-route leg number; outputs name legs by it.
   * - ``ai1``, ``ai2``
     - IWRAP position-check interval (seconds) for directions 1 and
       2.  Plugged into :math:`\exp(-d/(a_i V))` for the powered
       Category II term.  ``ai <= 0`` switches Category II off for
       that leg and direction.
   * - ``traffic_locked``
     - **AIS lock** (route table): ``true`` keeps AIS updates from
       overwriting the leg's traffic (:ref:`copy-traffic`).  Missing =
       ``false``.
   * - ``traffic_source``
     - Segment id of the leg the traffic was copied from.  It makes the
       junction matrix default to 100 % continuation between the two
       legs; it is removed when the lock is released or an AIS update
       writes the leg.

``bend_angle`` (written by the IWRAP import) is no longer read: bend
collisions are found from the routes at each waypoint.

Scenario: suppressed leg
------------------------

.. code-block:: json

   "suppressed": true,
   "traffic_redirect": [
     {"from_dir": 0, "leg": "5",  "dir": 0, "share": 80.0},
     {"from_dir": 0, "leg": "15", "dir": 0, "share": 20.0},
     {"from_dir": 1, "leg": "5",  "dir": 1, "share": 100.0}
   ]

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Key
     - Meaning
   * - ``suppressed``
     - ``true`` leaves the leg out of the calculation and the IWRAP
       export (:ref:`suppress-leg`).  Missing = ``false``.
   * - ``suppressed_with``
     - Segment id of the *lead* leg this leg is suppressed together with
       (:ref:`suppress-route`).  The leg is left out without moving any
       traffic, and its own ``traffic_redirect`` is ignored.  Missing on a
       leg that is not in a group.
   * - ``traffic_redirect``
     - Where the suppressed leg's ships go.  ``from_dir`` and ``dir``
       are direction indices (``0`` = the drawn direction,
       ``Dirs[0]``; ``1`` = the reverse) on the suppressed leg and on
       the target ``leg`` (a segment id).  ``share`` is the percentage
       of the ``from_dir`` ships that sail the target.  The list is
       kept when the leg is restored and is only used while
       ``suppressed`` is ``true``.

The redirect is applied by
:func:`compute.traffic_redirect.apply_traffic_redirects` right after
the traffic scaling, inside
:func:`compute.traffic_pipeline.prepare_traffic_for_run`.  It works on the
calculation's private copy of the data: frequencies are added to the
target cells, speed / draught / height / beam become frequency-weighted
means, and the suppressed legs are then removed from ``segment_data``,
``traffic_data`` and ``junctions``.  The saved file always keeps the
original traffic.

Lateral distribution
--------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Key
     - Meaning
   * - ``mean{d}_{i}``
     - Mean of the :math:`i`-th Gaussian component for direction
       :math:`d` (d = 1 or 2; i = 1, 2, 3).  Metres on the leg's
       lateral axis, the same axis for both directions: **negative =
       starboard of the drawn direction**.
   * - ``std{d}_{i}``
     - Standard deviation of the same.  A component with ``std`` or
       weight ``0`` is left out.
   * - ``weight{d}_{i}``
     - Component weight.  The Distributions tab stores percent; the
       weights and ``u_p{d}`` are normalised by their sum.
   * - ``u_min{d}`` / ``u_max{d}`` / ``u_p{d}``
     - Uniform-component bounds (same axis) and weight.
   * - ``dist1`` / ``dist2``
     - The lateral offsets of the ships the last AIS update saw on the
       passage line, per direction (the histogram on the Distributions
       tab).  Not read by the risk models.


``waypoints`` -- shared leg end nodes
======================================

.. code-block:: json

   "waypoints": {
     "1": [14.0, 55.2],
     "2": [14.2, 55.2]
   }

``{wp_id: [lon, lat]}``.  Legs hang on these nodes through ``start_wp``
/ ``end_wp``, so moving a node moves every leg on it.  The block is
derived: on load and save it is rebuilt from the leg ends, keeping the
ids whose coordinates are unchanged, so a file without it (or with a
stale one) loads unchanged.


``junctions`` -- transition matrices
=====================================

.. code-block:: json

   "junctions": {
     "j_14.200000_55.200000": {
       "point": [14.2, 55.2],
       "legs": {"1": "end", "2": "start", "3": "start"},
       "transitions": {"1": {"2": 0.7, "3": 0.3}, "2": {"1": 1.0}, "3": {"1": 1.0}},
       "source": "ais"
     }
   }

One entry per point where two or more leg ends meet (:ref:`junctions`).
``legs`` says which end of each leg touches the point;
``transitions[in_leg][out_leg]`` is the share of the ships arriving on
``in_leg`` that continue on ``out_leg`` (rows sum to 1).  ``source`` is
``"geometry"`` (deflection-angle default), ``"ais"`` (counted from AIS
transits) or ``"user"`` (edited in **Settings > Junction transition
matrix**; kept through every refresh).  The ship-ship model uses the
matrices for crossing, merging and bend encounters.  Missing or empty =
geometric defaults.


``depths`` -- bathymetry polygons
==================================

A flat array of ``[id, depth_m, wkt_polygon]`` triples:

.. code-block:: text

   "depths": [
     ["d1",  "5.0",  "POLYGON((14.08 55.22, ...))"],
     ["d2",  "10.0", "POLYGON((...))"],
     ...
   ]

* ``id`` is a string label used in results (the Depths table's id,
  e.g. ``"1"``).
* ``depth`` is stored as a **string** of a number -- OMRAT casts to
  float internally.  Chart-datum metres below MSL.
* ``wkt`` is a valid shapely / OGC WKT polygon in EPSG:4326.
  MultiPolygons are accepted; a run splits them into components named
  ``<id>_<i>`` in the results.

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

.. code-block:: text

   "ship_categories": {
     "types": [
       "Other ship", "Search & rescue", "Sailing vessel", ...
     ],
     "length_intervals": [
       {"min": 0.0,    "max": 50.0,   "label": "0-50"},
       {"min": 50.0,   "max": 100.0,  "label": "50-100"},
       ...
     ],
     "selection_mode": "simple_ais"
   }

* ``types`` -- display names for the ship-type rows of the traffic
  matrices (21 by default).
* ``length_intervals`` -- the LOA bin columns.  Each bin has
  ``min`` / ``max`` (metres) and a ``label``.  The **midpoint** of a
  bin is the ship length the ship-ship model uses (the traffic
  matrices store no length); a missing beam is estimated from it.
* ``selection_mode`` -- ``"simple_ais"`` or ``"manual"``: the option
  chosen in the Ship categories dialog (the IWRAP import writes
  ``"ais"``).  Saved for reference; it is not read back.


``consequence`` -- oil-spill inputs
====================================

.. code-block:: text

   "consequence": {
     "oil_onboard":        [[...], ...],
     "spill_probability":  [[97, 1, 1, 1], ...],
     "spill_fraction":     [[0, 10, 30, 100], ...],
     "catastrophe_levels": [{"name": "Minor", "quantity": 50.0}, ...]
   }

The four inputs of the **Consequence** menu (:ref:`consequence`):
oil on board in m\ :sup:`3` per ship type x length class, and per
accident type (9 rows, in the order of the accident table) x spill level
the spill probability and the spilled fraction, both in percent; the
catastrophe levels need at least two rows.  A file without the block
gets the defaults.


``layer_styles`` and ``segments_imported``
==========================================

* ``layer_styles`` -- ``{legs|tangent|depths|structures|speed_zones:
  <QML text>}``, the QGIS symbology, labeling and rendering of the OMRAT
  layers, re-applied on load.  A restyle in the Layers panel therefore
  counts as an unsaved change.
* ``segments_imported`` -- ``{segment_id: {Start_Point, End_Point}}``,
  the endpoints each leg had when it was first drawn or imported.
  Written once; the audit section of the run report compares them with
  the current legs to show waypoints that moved.


Optional / developer-only flags
=================================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Meaning
   * - ``data['drift']['debug_trace']``
     - ``True`` -> per-obstacle debug block added to the drifting
       report (see "Debug trace" in :ref:`code-flow-drifting`).

Two flags of older versions are no longer read: ``data['use_analytical']``
(the probability holes always use the analytical cross-section
integration) and ``data['drift']['use_leg_offset_for_distance']``
(replaced by ``drift['start_from']``).


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

Table ``omrat_runs``
~~~~~~~~~~~~~~~~~~~~

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
     - Powered totals (Category I + Category II).
   * - ``head_on``, ``overtaking``, ``crossing``, ``merging``,
       ``bend``, ``ship_collision_total``
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

A total of a model part the run did not calculate (**Calculation
matrix**) is ``NULL`` -- shown empty, never as zero.  There is no
column migration: adding an accident type changes this table, and a
history file written by an older OMRAT then has to be deleted.

Table ``omrat_run_info``
~~~~~~~~~~~~~~~~~~~~~~~~

Project tag and OMRAT version per run.  It is a separate table so that
adding it needs no change to ``omrat_runs``: it is created on the next
plugin start in any existing history file, runs recorded before it
simply have no row, and an older OMRAT still reads ``omrat_runs``
unchanged.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Meaning
   * - ``run_id``
     - The ``omrat_runs`` row this belongs to.
   * - ``project``
     - The project name (**Project name** on the Run Analysis tab, or
       **Set project...**).  ``NULL`` = not tagged: the run is grouped
       under the last folder of ``output_dir``.
   * - ``omrat_version``
     - OMRAT version that produced the run.  Filled in once from the
       run's ``.omrat`` snapshot for older runs; ``''`` when the
       snapshot carries no version.

Per-run GeoPackages
-------------------

One file per run, written under the user-selected output folder.
Filename: ``<slug(name)>_<YYYYMMDD_HHMMSS>.gpkg``.  Each contains up
to six layers (only those the run produced are written).  The four
obstacle layers have one LineString per boundary edge of each hit
depth or structure polygon, and one column per leg (drifting:
``LEG_1_2_b``; powered: ``LEG_1_2_b_East_going``, the route-table name):

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Layer
     - Geometry
     - Attributes
   * - ``drifting_allision``
     - LineString (per edge)
     - ``obstacle_id``, ``segment_idx``, ``total_edge_probability``,
       ``object_probability``, ``normal_deg``, ``value``, the edge
       back-tracing columns (``edge_dist_m``, ``reach_width_m``,
       ``edge_p_nr``, ...) and the per-leg columns.
   * - ``drifting_grounding``
     - LineString (per edge)
     - same shape as ``drifting_allision``.
   * - ``powered_grounding``
     - LineString (per edge)
     - ``obstacle_id``, ``segment_idx``, ``total_edge_probability``
       (the rays that hit this edge), ``object_probability``,
       ``value`` (depth) and the per-leg columns.
   * - ``powered_allision``
     - LineString (per edge)
     - same columns, ``value`` = structure height.
   * - ``collision_lines``
     - LineString
     - ``leg_id``, ``leg_name`` (route-table name), ``head_on``,
       ``overtaking``, ``combined``.
   * - ``collision_points``
     - Point
     - ``waypoint``, ``crossing``, ``merging``, ``bend``,
       ``combined``.

Open a per-run file directly in QGIS (it's a standard OGC
GeoPackage), or use **Add selected run results to map** (or the
right-click **Add results to map**) on the Previous-runs table to load
its layers in one click with graduated styling applied.

Files next to each run
----------------------

Each run also writes these files beside its GeoPackage, named after
its stem (``<stem>`` = ``<slug(name)>_<YYYYMMDD_HHMMSS>``).  Each is
written only when there is something to put in it, and each is
optional when read back: a run recorded before a file existed simply
lacks it, and the dialog that needs it falls back as described.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - File
     - Content and reader
   * - ``<stem>.omrat``
     - Read-only snapshot of the project inputs.  The **View** dialogs
       and the Compare tab rebuild their inputs from it.
   * - ``<slug(name)>_<YYYYMMDD>_results_<HHMMSS>.md``
     - Combined Markdown report of the run.
   * - ``<stem>.collision.json``
     - The ship-ship collision report: ``totals`` (``head_on``,
       ``overtaking``, ``crossing``, ``merging``, ``bend``, ``total``),
       ``by_leg``, ``by_waypoint``, ``by_leg_pair``, ``bend_by_pair``,
       ``by_cell``, ``causation_factors``,
       ``min_encounter_angle_deg`` (collision **View** tables, Compare
       tab).
   * - ``<stem>.drifting.json``
     - The drifting report: ``totals`` (``allision``, ``grounding``,
       ``anchoring``), ``by_leg_direction``, ``by_object``, the
       ``by_structure_* / by_depth_* / by_anchoring_*`` leg-direction
       and per-edge maps, ``by_cell_allision`` /
       ``by_cell_grounding``, ``edge_calc_meta`` (per obstacle edge and
       leg-direction: hole share, reachable width, edge distance, ...,
       for back-tracing; the result layers' edge columns come from it)
       and, with ``debug_trace``, ``debug_obstacles``.  Read by the drifting **View** table
       (contributions and % of total per leg, direction and drift
       direction) and the Compare tab.
   * - ``<stem>.consequence.json``
     - The catastrophe-exceedance result (catastrophe table for
       previous runs).
   * - ``<stem>.powered.json``
     - Per leg-direction powered frequencies for the powered **View**
       table (Frequency and % of total).  One block per mode, left out
       when that model produced no report.  Keys of
       ``by_leg_direction`` are ``"<Segment_Id>:<dir_idx>"`` (``0`` =
       the drawn direction).  Without this file, allision is
       recomputed from the snapshot and grounding shows ``--``.

Shape of ``<stem>.powered.json`` (the ``grounding`` block, when
present, has the same keys with ``"grounding"`` as the total):

.. code-block:: json

   {
     "allision": {
       "totals": {"allision": 1.07e-03, "cat1": 1.07e-03, "cat2": 3.36e-06},
       "by_leg_direction": {"17:0": 6.10e-04, "17:1": 4.58e-04, "9:0": 1.18e-06},
       "by_obstacle": {"1_0": 1.07e-03},
       "causation_factor": 1.9e-04,
       "causation_factor_cat1": 1.9e-04
     }
   }


Minimal valid project (synthetic)
==================================

The smallest project that passes Pydantic validation and runs the
drifting cascade is in ``tests/test_cascade_minimal.py``
(``_build_minimal_project``): one east-going 10 km leg, one 12 m
depth polygon NW of the leg, one 20 m structure N of the leg, a
single cargo-type cell with 100 ships/year at 10 knots / 13 m
draught.  Useful as a starting skeleton when building a fixture.
