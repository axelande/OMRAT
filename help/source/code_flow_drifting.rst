.. _code-flow-drifting:

==========================================
Code Flow: Drifting Model
==========================================

This chapter walks the drifting model one function at a time, in the
order that the calls fire when a user presses **Run Model**.  Pair it
with the theory chapter :ref:`drifting`, which derives the formulas
that each function below implements.

.. contents:: In this chapter
   :local:
   :depth: 2


Entry point
===========

:meth:`DriftingModelMixin.run_drifting_model` is phase 1 of
:class:`~compute.calculation_task.CalculationTask` (see
:ref:`code-flow`).  It takes a single ``data`` dict and writes

* ``self.drifting_allision_prob``
* ``self.drifting_grounding_prob``
* ``self.drifting_report``
* ``self.allision_result_layer``, ``self.grounding_result_layer``
* ``LEPDriftAllision.setText(...)`` / ``LEPDriftingGrounding.setText(...)``

.. container:: source-code-ref pipeline

   **Entry:** ``compute/drifting_model.py:1608`` -- `run_drifting_model() <https://github.com/axelande/OMRAT/blob/main/compute/drifting_model.py#L1608>`__


Top-level call tree
===================

::

   run_drifting_model(data)
     |
     +-- _build_transformed(data)                                         # UTM projection + geom prep
     |     +-- prepare_traffic_lists(data)                                # compute/data_preparation.py
     |     +-- split_structures_and_depths(data)
     |     +-- transform_to_utm(lines, structure+depth wkts)
     |     +-- shapely.make_valid  (per obstacle)
     |
     +-- _compute_reach_distance(data, longest_length)                    # drift-speed * T99 of repair dist
     |
     +-- merge depths by unique level (optional optimisation)             # shapely.unary_union per level
     |
     +-- _precompute_spatial(...)                                         # = "spatial" progress phase
     |     +-- compute_min_distance_by_object(...)                        # geometries/get_drifting_overlap.py
     |     +-- compute_probability_holes_analytical(...)                  # geometries/analytical_probability.py
     |           +-- ThreadPoolExecutor (per leg x direction)
     |                 +-- _compute_single_direction_analytical(...)
     |                       +-- compute_probability_analytical(...)
     |                             +-- _vectorized_edge_y_intervals(...)
     |                             +-- _merge_intervals_across_slices(...)
     |                             +-- dist.cdf(array)                    # batched scipy CDF
     |
     +-- _precompute_shadow_layer(...)                                    # = "shadow" progress phase (0..50%)
     |     +-- ThreadPoolExecutor (per leg x direction)
     |           +-- _shadow_task(leg_idx, d_idx)
     |                 +-- _create_drift_corridor(...)
     |                 +-- _build_blocker_shadow(..., shadow_memo)
     |                 |     +-- extract_polygons(geom)
     |                 |     +-- create_obstacle_shadow(p, compass, bounds)  # cached per (id(geom),dir)
     |                 +-- _edge_geom_for(poly)
     |                       +-- _extract_obstacle_segments(poly)
     |                       +-- _edge_weighted_holes(...)                # batched drift filter + shapely
     |                       |     +-- segment_corridor_overlap_length(...)
     |                       +-- directional_distances_to_points(...)    # vectorised drift distance
     |                       +-- get_not_repaired(...)                   # analytical ndtr / Weibull
     |
     +-- _precompute_bucket_memo(...)                                     # = "shadow" progress phase (50..100%)
     |     +-- ThreadPoolExecutor (per (leg, dir, ship-bucket))
     |           +-- _compute_bucket(...)
     |                 +-- geom.difference(blocker_union)
     |                 +-- _analytical_hole_for_geom(reach, ...)
     |                       +-- extract_polygons + _extract_polygon_rings
     |                       +-- compute_probability_analytical(...)
     |                 +-- shapely.unary_union
     |
     +-- _iterate_traffic_and_sum(...)                                    # = "cascade" progress phase (60..90%)
     |     +-- clean_traffic(data)
     |     +-- for each leg, ship cell, direction:
     |           _process_cell_direction(...)
     |             +-- lookup bucket_memo entry
     |             +-- per-entry: accumulate allision/grounding/anchoring
     |             +-- _update_report(...)
     |             +-- _update_anchoring_report(...)
     |             +-- _add_direct_segment_contrib(...)
     |
     +-- create_result_layers(report, ...)                                # = "layers" phase
     +-- _auto_generate_drifting_report(data)                             # writes Markdown


The sections below open each box above, in the order the code executes.


:meth:`_build_transformed`: lat/lon -> UTM
==========================================

:meth:`DriftingModelMixin._build_transformed` is a one-time geometry
prep step that runs entirely on the calculation thread.  It returns
eight parallel lists used by every downstream helper.

.. container:: source-code-ref pipeline

   **Source:** ``compute/drifting_model.py:859`` -- `_build_transformed() <https://github.com/axelande/OMRAT/blob/main/compute/drifting_model.py#L859>`__

Calls performed:

1. :func:`compute.data_preparation.prepare_traffic_lists(data)` - returns
   ``(lines, distributions, weights, line_names)``.

   * ``lines`` is a list of :class:`shapely.geometry.LineString` in
     EPSG:4326 (one per leg).
   * ``distributions`` is ``list[list[scipy.stats.rv_frozen]]`` -- one
     list per leg with *up to three* superposed distributions parsed
     from ``mean1_1``/``std1_1``/``weight1_1`` ... etc.
   * ``weights`` is the matching list of floats summing to 1.
   * ``line_names`` are the human labels shown in reports.

2. :func:`compute.data_preparation.split_structures_and_depths(data)` -
   converts the ``depths`` / ``objects`` lists of
   ``[id, value, wkt]`` into two lists of dicts with parsed shapely
   geometries.  MultiPolygons are split into their component polygons
   so every entry has a single :class:`shapely.geometry.Polygon`.

3. :func:`compute.data_preparation.transform_to_utm(lines, obstacle_geoms)` -
   picks the UTM zone best covering the combined bbox, transforms
   every leg and obstacle once via QGIS
   :class:`~qgis.core.QgsCoordinateTransform` when running inside
   QGIS, or :mod:`pyproj` otherwise.  All subsequent geometry math is
   in metres.

4. :func:`shapely.make_valid` on every obstacle.  If the result is a
   MultiPolygon it is again split into components.  Each final polygon
   is stored with its WGS84 version (``wkt_wgs84``) so that result
   layers generated later use the same vertex order as the UTM
   version.

Output:
``(lines, distributions, weights, line_names, structures,
depths, structs_gdfs, depths_gdfs, transformed_lines)``.


:meth:`_compute_reach_distance`
================================

Computes the "99th percentile drift distance": how far a ship can drift
before the repair distribution says it almost certainly got its engine
back.

.. container:: source-code-ref pipeline

   **Source:** ``compute/drifting_model.py:92`` -- `_compute_reach_distance() <https://github.com/axelande/OMRAT/blob/main/compute/drifting_model.py#L92>`__

Logic:

* If ``drift.repair.dist_type == 'weibull'``, uses
  ``scipy.stats.weibull_min(...)``.ppf(0.99).
* Else if ``drift.repair.use_lognormal``, uses
  ``scipy.stats.lognorm(...).ppf(0.99)``.
* Multiplies by the drift speed (m/s) to get metres.
* Caps at ``10 x longest_leg_length`` so that a poorly-parameterised
  repair distribution can't make the reach distance degenerate.


Depth merging (optional)
========================

Before the spatial precompute, :meth:`run_drifting_model` examines the
set of depth polygons.  If there are more polygons than unique depth
values, the method merges all polygons with depth <= boundary into one
cumulative polygon per boundary via :func:`shapely.unary_union`.  The
result is:

* ``merged_depths_gdfs`` / ``merged_depths_meta`` - a much smaller set
  of obstacles used by the expensive spatial phase.
* ``threshold_to_idx`` - maps any draught or anchor threshold to the
  index of the correct merged polygon.  The cascade uses this to look
  up grounding / anchoring obstacles per ship cell in O(1).

This is the single biggest source of speedup for projects with dense
bathymetry (tens of depth contours map to just a handful of unique
levels).


Phase 1: :meth:`_precompute_spatial` (``spatial``, 0--40 %)
============================================================

Ship-independent pre-computation of everything that depends on leg
geometry only.

.. container:: source-code-ref pipeline

   **Source:** ``compute/drifting_model.py:986`` -- `_precompute_spatial() <https://github.com/axelande/OMRAT/blob/main/compute/drifting_model.py#L986>`__

Returns four lists, each indexed ``[leg_idx][math_dir_idx][obstacle_idx]``:

* ``struct_min_dists`` - nearest along-drift distance from leg to
  each structure (or ``None`` if the obstacle is out of reach in that
  direction).  Computed by
  :func:`geometries.get_drifting_overlap.compute_min_distance_by_object`.
* ``depth_min_dists`` - same for depth (grounding / anchoring)
  polygons.
* ``struct_probability_holes`` - :math:`h_X` = probability of drifting
  far enough in the obstacle's direction to hit it, from a random
  start on the leg.  Computed by
  :func:`geometries.analytical_probability.compute_probability_holes_analytical`
  (default) or
  :func:`geometries.calculate_probability_holes.compute_probability_holes`
  (Monte Carlo, opt-in via ``data['use_analytical'] = False``).
* ``depth_probability_holes`` - same for depth polygons.

Inside the analytical path
--------------------------

:func:`compute_probability_holes_analytical` parallelises per
``(leg, direction)`` across a :class:`~concurrent.futures.ThreadPoolExecutor`
with ``cpu_count() - 1`` workers.  Each worker calls
:func:`_compute_single_direction_analytical`, which for each object
calls :func:`compute_probability_analytical`:

1. Slice the leg into 100 cross-sections (``s_values`` = midpoints of
   100 equal intervals along the leg).
2. For every edge of the obstacle and every slice, solve for the
   ``y``-range of ``perp_offset`` values whose drift ray crosses that
   edge.  This is done in one batched numpy call
   (:func:`_vectorized_edge_y_intervals`) over all slices x all edges.
3. Per slice, merge the valid intervals into disjoint ones
   (:func:`_merge_intervals_across_slices`, batched over all slices).
4. Flatten every merged interval into two 1-D arrays ``los`` and
   ``his`` across every slice.
5. Evaluate the weighted distribution CDF **once** per distribution on
   both arrays (scipy's ``dist.cdf`` is vectorised over array input),
   subtract, and sum.  Divide by ``n_slices`` to get the probability
   hole.

Performance note: this used to call ``dist.cdf`` and
``_merge_intervals_vectorized`` per slice, which dominated
``compute_probability_analytical``.  Batching both turned the work
into a handful of scipy calls per obstacle (see
:ref:`code-flow`'s performance timeline).


Phase 2: :meth:`_precompute_shadow_layer` (``shadow``, 0--50 %)
================================================================

For every ``(leg, direction)`` pair, build:

* the drift corridor polygon,
* the quad-sweep shadow for every reachable obstacle,
* per-edge geometry (edge length fractions of the corridor overlap,
  along-drift distance of each edge, :math:`P_{NR}` at that distance).

.. container:: source-code-ref pipeline

   **Source:** ``compute/drifting_model.py:334`` -- `_precompute_shadow_layer() <https://github.com/axelande/OMRAT/blob/main/compute/drifting_model.py#L334>`__

Pre-computation outside the thread pool:

1. Build ``leg_precomputed[leg_idx]`` with ``dists_dir``, ``w_dir``
   (normalised weights), ``lateral_spread`` = 5 x weighted std of the
   leg's lateral distribution, and a :class:`drifting.engine.LegState`
   for the scalar fallback path.
2. Compute a **global shadow bounds**: the bbox of all legs + all
   obstacle geometries, padded by ``max(1 km, reach_distance)``.  This
   is passed to every call to :func:`create_obstacle_shadow`, which
   derives its extrude distance from the corridor bounds.  Using a
   single global bound lets a per-polygon shadow cache hit across
   legs, because the extrude distance (and thus the shadow shape) is
   now the same for every call.

Per-task work (``_shadow_task(leg_idx, d_idx)``):

1. Build the drift corridor via
   :func:`compute.drift_corridor_geometry._create_drift_corridor`: two
   rectangles (leg rect + drifted rect) unioned, falling back to
   convex hull if union produces a MultiPolygon.
2. For every structure and every depth (filtered via
   ``struct_min_dists`` / ``depth_min_dists`` to those within
   ``reach_distance``):

   a. Build the quad-sweep shadow -> :meth:`_build_blocker_shadow`.
   b. Build per-edge geometry -> inner ``_edge_geom_for(poly)``.

3. Return ``(key, entry)`` where ``entry`` has
   ``corridor``, ``bounds``, ``dists_list``, ``weights_arr``,
   ``lateral_spread``, ``shadow`` (dict of ``(type, idx) -> Polygon``),
   and ``edge_geom`` (dict of ``(type, idx) -> list[edge-dict]``).

Futures are collected with :func:`concurrent.futures.as_completed`,
progress is reported every ~5 % of completed tasks, and a cancellation
check aborts the remaining work if the user clicks stop.

.. container:: source-code-ref pipeline

   **Shadow cache helper:** ``compute/drifting_model.py:161`` -- `_build_blocker_shadow() <https://github.com/axelande/OMRAT/blob/main/compute/drifting_model.py#L161>`__

:meth:`_build_blocker_shadow`
-----------------------------

Memoises the **full** obstacle shadow (union over all Polygon
components) by ``(id(geom), compass_angle)``.  Because every leg passes
the same Python object for a given obstacle, the cache hits
``(legs - 1) x 8`` times per obstacle.

Implementation:

1. If ``id(geom), compass_angle`` already has a cached shadow, return
   it.
2. Call :func:`geometries.drift.shadow.extract_polygons(geom)` to split
   MultiPolygons.
3. For each component :class:`Polygon` call
   :func:`geometries.drift.shadow.create_obstacle_shadow`, which:

   a. Computes an extrude distance = ``2 * corridor_diagonal``.
   b. Translates the polygon that far in the drift direction.
   c. Builds one quad per original edge connecting original ->
      translated vertices.  Quads with zero shoelace area are skipped
      (the previous implementation called ``quad.is_valid`` /
      ``quad.area`` per quad, which was >1 M shapely dispatch calls
      per proj.omrat run).
   d. Unions original + translated + quads into the final shadow
      polygon.

4. If more than one component, :func:`shapely.ops.unary_union` the
   per-component shadows.
5. Cache and return.


Inner :func:`_edge_geom_for`
----------------------------

Works per obstacle inside ``_shadow_task``.  The result
(``[{'seg_idx', 'len_frac', 'edge_dist', 'edge_p_nr'}, ...]``) is used
by the cascade to split the obstacle's hole probability across
individual polygon edges.

1. ``segments = _extract_obstacle_segments(poly)`` - polygon edges as
   a list of ``((x0,y0), (x1,y1))`` tuples.
2. ``raw = self._edge_weighted_holes(poly, drift_corridor, math_angle,
   line, 1.0, None)`` - for each segment that survives the
   drift-direction pre-filter, returns
   ``(seg_idx, overlap_length)``.  The pre-filter runs numpy-batched
   so most rejected segments never touch shapely; the survivors go
   through :func:`segment_corridor_overlap_length` for the actual
   corridor intersection.
3. Collect the two endpoints of every selected edge into a flat
   ``(2N, 2)`` array and call
   :func:`geometries.get_drifting_overlap.directional_distances_to_points`
   **once**.  That function returns per-point along-drift distances
   via a single vectorised edge-crossing pass, falling back to a
   vectorised nearest-point projection for points that miss every
   leg segment.
4. For each edge, average its two endpoint distances ->
   ``edge_dist``.
5. :math:`P_{NR}` = :func:`compute.basic_equations.get_not_repaired`.
   That function compiles the repair ``func`` string or matches the
   common ``norm.cdf`` / ``weibull_min.cdf`` patterns and caches a
   pure-:func:`scipy.special.ndtr` closure, so repeated calls are
   O(1) in Python -- no scipy frozen-distribution dispatch.


Phase 3: :meth:`_precompute_bucket_memo` (``shadow``, 50--100 %)
==================================================================

Eagerly evaluates the **shadow-coverage cascade** for every distinct
"ship bucket" per ``(leg, direction)``.  A bucket is the set of
``(obstacle_type, obstacle_idx)`` tuples a ship sees given its
draught (grounding index) and anchor threshold (anchoring index).
Many ships share the same bucket, so doing the carving once per bucket
turns the later traffic cascade into pure arithmetic.

.. container:: source-code-ref pipeline

   **Source:** ``compute/drifting_model.py:617`` -- `_precompute_bucket_memo() <https://github.com/axelande/OMRAT/blob/main/compute/drifting_model.py#L617>`__

Per bucket:

1. Sort obstacles by along-drift distance.
2. Walk the sorted list maintaining a running ``blocker_union`` (for
   grounding/allision) and ``anchor_union`` (for anchoring).  The
   shadows come from the memo built in Phase 2.
3. For each obstacle:

   * Carve ``reach = geom.difference(blocker_union)`` - the part of
     the obstacle still reachable past closer blockers.
   * ``h_reach`` = :meth:`_analytical_hole_for_geom(reach, ...)` -
     the probability hole of the carved region, computed via the same
     :func:`compute_probability_analytical` used in Phase 1.
   * ``h_in_anchor`` = probability hole of the carved region that
     ALSO intersects the anchoring region - so anchoring ships are
     subtracted out correctly.

4. Result: ``memo[(leg, dir, bucket_key)] = [{'obs_type', 'obs_idx',
   'dist', 'hole_pct', 'h_reach', 'h_in_anchor'}, ...]``.

Parallelised the same way as the shadow layer.


Phase 4: :meth:`_iterate_traffic_and_sum` (``cascade``, 60--90 %)
==================================================================

Walks the ``traffic_data`` dict and accumulates contributions using the
bucket memo.  The outer structure is:

::

   for leg_idx, line in enumerate(transformed_lines):
       for cell in ship_cells[leg_idx]:
           for d_idx in range(8):
               a_delta, g_delta, an_delta = _process_cell_direction(...)

.. container:: source-code-ref pipeline

   **Source:** ``compute/drifting_model.py:1067`` -- `_iterate_traffic_and_sum() <https://github.com/axelande/OMRAT/blob/main/compute/drifting_model.py#L1067>`__

Each ship cell contributes:

.. math::

   \mathrm{base}_{i,k} =
     \frac{L_i}{v_k \cdot 1852/3600} \cdot f_{i,k}
     \cdot \frac{\lambda_{bo}(\mathrm{ship\_type})}{365.25 \cdot 24}

and the per-direction contribution is:

.. math::

   \Delta = \mathrm{base}_{i,k} \cdot r_{p,d} \cdot h_{\mathrm{eff}}
            \cdot P_{NR}(d_{\mathrm{edge}})

:meth:`_process_cell_direction` does the inner work:

.. container:: source-code-ref pipeline

   **Source:** ``compute/drifting_model.py:1296`` -- `_process_cell_direction() <https://github.com/axelande/OMRAT/blob/main/compute/drifting_model.py#L1296>`__

1. Build the obstacle list for this cell (respecting draught +
   anchor threshold), compute ``bucket_key`` from the tuple of sorted
   ``(type, idx)`` pairs.
2. Look up ``entries = bucket_memo[(leg_idx, d_idx, bucket_key)]``.
   On a memo miss (rare), fall back to the pre-computed ``hole_pct``
   without any shadow carving.
3. For each entry:

   * Compute ``h_eff = max(0, h_reach - anchor_p * h_in_anchor)``.
   * If the obstacle has precomputed per-edge geometry, sum
     ``contrib = base * r_p * (h_eff * len_frac) * edge_p_nr`` across
     edges.  Otherwise fall back to the obstacle-level contrib.
   * Call :meth:`_update_report` / :meth:`_update_anchoring_report`
     to record the per-leg-direction breakdown.
   * Call :meth:`_add_direct_segment_contrib` to record the
     per-edge contribution so the result layer can colour the exact
     polygon edges that dominated the risk.

4. Accumulate ``(allision_delta, grounding_delta, anchoring_delta)``.

Progress is reported approximately every 1 % of cascade completion.


Completion (``layers``, 90--100 %)
===================================

After the cascade finishes :meth:`run_drifting_model` does three
things on the calculation thread:

1. Applies the user's risk-reduction factors
   (``pc.allision_drifting_rf``, ``pc.grounding_drifting_rf``) to the
   totals and stores the final numbers on ``self``.
2. Calls :func:`geometries.result_layers.create_result_layers` to
   build two :class:`~qgis.core.QgsVectorLayer` objects that colour
   each obstacle polygon by its contribution to the total risk.  Layer
   creation is graduated by Jenks natural breaks (or single-class
   fallback).
3. Calls :meth:`_auto_generate_drifting_report` which delegates to
   :meth:`DriftingReportBuilderMixin.write_drifting_report_markdown`
   (mixin from ``compute/drifting_report_builder.py``).  The report
   writes a Markdown file with:

   * Totals per accident type,
   * Per leg-direction tables,
   * Per obstacle tables (with drill-down to per-segment
     contributions),
   * Debug obstacle blocks when ``drift.debug_trace = True``.

Finally the two result line-edits are updated:

.. code-block:: python

   self.p.main_widget.LEPDriftAllision.setText(f"{self.drifting_allision_prob:.3e}")
   self.p.main_widget.LEPDriftingGrounding.setText(f"{self.drifting_grounding_prob:.3e}")


Function reference
==================

A flat list of every function reached from
:meth:`run_drifting_model`, grouped by source file.

``compute/drifting_model.py``
-----------------------------

* ``run_drifting_model`` - orchestrator (this chapter).
* ``_build_transformed`` - UTM projection + make_valid.
* ``_compute_reach_distance`` - T99 repair distance in metres.
* ``_precompute_spatial`` - min-distances + probability holes.
* ``_precompute_shadow_layer`` - corridor/shadow/edge geom per
  ``(leg, dir)``.
* ``_precompute_bucket_memo`` - shadow cascade per ship bucket.
* ``_iterate_traffic_and_sum`` - traffic cascade.
* ``_process_cell_direction`` - per cell x direction contribution.
* ``_build_blocker_shadow`` - cached per-obstacle quad-sweep union.
* ``_edge_weighted_holes`` - distribute hole across polygon edges by
  corridor overlap length.
* ``_analytical_hole_for_geom`` - probability hole for a carved
  geometry; wraps :func:`compute_probability_analytical`.
* ``_update_report`` / ``_update_anchoring_report`` /
  ``_add_direct_segment_contrib`` - report bookkeeping (in
  :class:`DriftingReportBuilderMixin`).
* ``_auto_generate_drifting_report`` - writes Markdown report.

``compute/data_preparation.py``
-------------------------------

* ``prepare_traffic_lists`` - builds ``(lines, distributions, weights,
  line_names)`` for every leg.
* ``split_structures_and_depths`` - parses WKT, splits MultiPolygons.
* ``transform_to_utm`` - picks UTM zone; uses QGIS or pyproj.
* ``clean_traffic`` - flattens ``traffic_data`` into per-leg cell
  lists.
* ``get_distribution`` - parses a segment's lateral distributions.

``compute/basic_equations.py``
------------------------------

* ``get_not_repaired`` - analytical :math:`P_{NR}`; dispatches to
  cached :func:`scipy.special.ndtr` (normal / lognormal) or direct
  Weibull formula, else compiles ``func`` once and re-uses.
* ``powered_na`` - shared helper used by the powered model.
* ``get_drifting_prob`` / ``get_drift_time`` - legacy scalar helpers.
* ``SHIP_TYPE_NAMES`` / ``default_blackout_by_ship_type`` - per-type
  blackout rate table.

``compute/drift_corridor_geometry.py``
--------------------------------------

* ``_create_drift_corridor`` - leg-rect + drift-rect unioned.
* ``_extract_obstacle_segments`` - polygon edges as
  ``[(p0, p1), ...]``.
* ``_segment_intersects_corridor`` - kept for tests / callers outside
  the hot path.
* ``segment_corridor_overlap_length`` - drift-direction pre-filter
  then corridor intersection in one shapely pass.
* ``_compass_idx_to_math_idx`` - 0..7 compass -> math direction.

``geometries/analytical_probability.py``
----------------------------------------

* ``compute_probability_holes_analytical`` - top-level batch across
  all legs/directions/objects.  ThreadPoolExecutor.
* ``_compute_single_direction_analytical`` - per ``(leg, dir)``
  worker.
* ``compute_probability_analytical`` - per obstacle; batched CDF.
* ``_vectorized_edge_y_intervals`` - numpy batch of
  (slice x edge) y-intervals.
* ``_merge_intervals_vectorized`` - single-slice merge (tests /
  callers outside the hot path).
* ``_merge_intervals_across_slices`` - flat merged intervals across
  every slice (hot-path helper).
* ``_extract_polygon_rings`` - polygon rings as numpy arrays.

``geometries/get_drifting_overlap.py``
--------------------------------------

* ``compute_min_distance_by_object`` - per
  ``(leg, dir, obstacle)`` minimum along-drift distance.
* ``directional_distances_to_points`` - vectorised per-point along-
  drift distance (used by ``_edge_geom_for``).
* ``directional_min_distance_reverse_ray`` - wraps the helper above
  to return the min across a polygon's vertices.

``geometries/drift/shadow.py``
------------------------------

* ``create_obstacle_shadow`` - quad-sweep shadow polygon.
* ``_create_edge_quads`` - N quad polygons per obstacle edge
  (shoelace-filtered, no shapely validity).
* ``extract_polygons`` - flatten Polygon / MultiPolygon /
  GeometryCollection to a list of Polygons.

``geometries/result_layers.py``
-------------------------------

* ``create_result_layers`` - builds allision + grounding
  :class:`~qgis.core.QgsVectorLayer` from the drifting report.

``geometries/calculate_probability_holes.py``
---------------------------------------------

Monte-Carlo alternative to the analytical path, used when
``data['use_analytical'] = False``.  Same pipeline shape with
``ThreadPoolExecutor`` parallelism, but every ``p_hole`` is
estimated by sampling rather than integrated analytically.


Debug trace
===========

Setting ``data['drift']['debug_trace'] = True`` enables a per-obstacle
debug path in :meth:`_iterate_traffic_and_sum` that records
``exposure_factor``, ``base``, ``rp``, ``freq``, ``dist``, ``h_eff``,
and the accumulated contribution into ``report['debug_obstacles']``.
The debug block is embedded in the auto-generated Markdown report,
giving a line-by-line breakdown of every obstacle / leg / direction.
