.. _api:

=============
API Reference
=============

This chapter provides a reference for OMRAT's key modules, classes, and
functions.

.. contents:: In this chapter
   :local:
   :depth: 2


compute.basic_equations
========================

Core mathematical formulas for risk calculations.

**Source file:** ``compute/basic_equations.py`` (`View on GitHub <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py>`__)

Drifting Functions
------------------

.. function:: get_Fcoll(na, pc)

   Calculate accident frequency from candidates and causation factor.

   :param na: Number of accident candidates
   :param pc: Causation probability
   :return: Accident frequency (:math:`F = N_A \times P_C`)

   **Line:** 67 | `Source <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py#L67>`__

.. function:: get_drifting_prob(Fb, line_length, ship_speed)

   Calculate the probability of a blackout during transit.

   :param Fb: Blackout frequency (events/year)
   :param line_length: Leg length
   :param ship_speed: Ship speed, per hour in the unit of
      ``line_length`` (``line_length / ship_speed`` is the transit time
      :math:`h` in hours)
   :return: Blackout probability

   Uses: :math:`P = 1 - \exp(-F_b / (24 \times 365.25) \times h)`

   **Line:** 78 | `Source <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py#L78>`__

.. function:: get_drift_time(distance, drift_speed)

   Estimate the time to drift a given distance.

   :param distance: Drift distance (metres)
   :param drift_speed: Drift speed (m/s)
   :return: Drift time (seconds)

   **Line:** 86 | `Source <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py#L86>`__

.. function:: get_not_repaired(data, drift_speed, dist)

   Get the probability that the engine is not repaired before drifting
   distance ``dist``.

   :param data: Repair time parameters dict
   :param drift_speed: Drift speed (m/s)
   :param dist: Distance to obstacle (metres)
   :return: Probability of not being repaired

   **Line:** 205 | `Source <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py#L205>`__

.. function:: repairtime_function(data, x)

   Evaluate the repair time CDF at time ``x``.

   :param data: Repair parameters dict
   :param x: Time (hours)
   :return: Cumulative repair probability

   **Line:** 93 | `Source <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py#L93>`__

Ship-Ship Collision Functions
------------------------------

.. function:: lateral_overlap_probability(comps1, comps2, half_width)

   Hansen's geometric probability :math:`P_G = P(|X - Y| \le B)` for two
   lateral mixtures on the same axis, summed component by component
   (:math:`\sum_i \sum_j w_i w_j P_G(i, j)`; normal and uniform parts in
   closed form).  ``comps`` are ``geometries.distribution_curves.mixture``
   components, ``half_width`` the mean half-breadth of the two ships.
   This is what ``run_ship_collision_model`` uses for head-on and
   overtaking.

   **Line:** 481 | `Source <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py#L481>`__

.. function:: head_on_candidates_from_overlap(Q1, Q2, V1, V2, P_G, L_w)

   Head-on candidates per year for a given :math:`P_G`:
   :math:`(Q_1/V_1)(Q_2/V_2)(V_1 + V_2) P_G L_w`.  Used by the model.

   **Line:** 515 | `Source <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py#L515>`__

.. function:: overtaking_candidates_from_overlap(Q_fast, Q_slow, V_fast, V_slow, P_G, L_w)

   Overtaking candidates per year for a given :math:`P_G`:
   :math:`(Q_f/V_f)(Q_s/V_s)(V_f - V_s) P_G L_w`; 0 unless
   :math:`V_f > V_s`.  Used by the model.

   **Line:** 524 | `Source <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py#L524>`__

.. function:: get_head_on_collision_candidates(Q1, Q2, V1, V2, mu1, mu2, sigma1, sigma2, B1, B2, L_w)

   Calculate geometric head-on collision candidates (Hansen Eq. 4.2--4.4)
   for one normal distribution per direction.  Not called by the model
   (which uses the mixture functions above); kept for reference and tests.

   :param Q1, Q2: Traffic volumes (ships/year)
   :param V1, V2: Ship speeds (m/s)
   :param mu1, mu2: Mean lateral positions (m)
   :param sigma1, sigma2: Standard deviations (m)
   :param B1, B2: Ship beams (m)
   :param L_w: Leg length (m)
   :return: Geometric collision candidates per year

   **Line:** 338 | `Source <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py#L338>`__

.. function:: get_overtaking_collision_candidates(Q_fast, Q_slow, V_fast, V_slow, mu_fast, mu_slow, sigma_fast, sigma_slow, B_fast, B_slow, L_w)

   Calculate geometric overtaking collision candidates (Hansen Eq. 4.5)
   for one normal distribution per direction.  Returns 0 if
   ``V_fast <= V_slow``.  Not called by the model; kept for reference and
   tests.

   :return: Geometric collision candidates per year

   **Line:** 535 | `Source <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py#L535>`__

.. function:: get_crossing_collision_candidates(Q1, Q2, V1, V2, L1, L2, B1, B2, theta)

   Calculate geometric crossing collision candidates (Pedersen 1995 /
   Hansen Eq. 4.6).  Used for crossing, merging and bend encounters at a
   waypoint.  Returns 0 if the courses are parallel (``sin(theta) ~ 0``).

   :param L1, L2: Ship lengths (m) -- the length-class midpoints
   :param theta: Angle between the two courses (radians)
   :return: Geometric collision candidates per year

   **Line:** 626 | `Source <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py#L626>`__

Powered Grounding Functions
----------------------------

The powered models do not call these closed-form helpers: they cast rays
over the lateral mixture (``geometries/get_powered_overlap.py``) and
sum Category II ray by ray with
:func:`geometries.get_powered_overlap.cat2_probability`.  The helpers
below are the textbook formulas, kept for reference and tests.

.. function:: powered_na(distance, mean_time, ship_speed)

   :math:`\exp(-d / (a_i V))` with ``mean_time`` = :math:`a_i` (seconds)
   and ``ship_speed`` = :math:`V` (m/s); 0 when :math:`a_i V \le 0`
   (``ai <= 0`` switches Category II off).

   **Line:** 102 | `Source <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py#L102>`__

.. function:: get_recovery_distance(position_check_minutes, ship_speed)

   Calculate recovery distance :math:`a = \lambda \times V`.

   :param position_check_minutes: Check interval (minutes)
   :param ship_speed: Ship speed (m/s)
   :return: Recovery distance (metres)

   **Line:** 754 | `Source <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py#L754>`__

.. function:: get_powered_grounding_cat1(Q, Pc, prob_in_obstacle)

   Category I powered grounding -- ships sailing directly into obstacle.

   :param Q: Traffic volume (ships/year)
   :param Pc: Causation factor
   :param prob_in_obstacle: Fraction of distribution overlapping obstacle
   :return: Expected groundings per year

   **Line:** 778 | `Source <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py#L778>`__

.. function:: get_powered_grounding_cat2(Q, Pc, prob_at_position, distance_to_obstacle, position_check_interval, ship_speed)

   Category II powered grounding -- ships failing to turn at bend.

   :param distance_to_obstacle: Distance from bend (metres)
   :param position_check_interval: Check interval (minutes)
   :return: Expected groundings per year

   **Line:** 816 | `Source <https://github.com/axelande/OMRAT/blob/main/compute/basic_equations.py#L816>`__


compute.run_calculations
========================

**Source file:** ``compute/run_calculations.py`` (`View on GitHub <https://github.com/axelande/OMRAT/blob/main/compute/run_calculations.py>`__)

.. class:: Calculation(omrat)

   Main calculation orchestrator.

   **Line:** 45 | `Source <https://github.com/axelande/OMRAT/blob/main/compute/run_calculations.py#L45>`__

   Composed from mixins: ``DriftingModelMixin``, ``ShipCollisionModelMixin``,
   ``PoweredModelMixin``, ``DriftingReportMixin``, ``VisualizationMixin``
   (the bases of the class).
   ``DriftingReportMixin`` only renders the drifting section; the run's
   combined Markdown report is assembled by
   ``omrat_utils/full_report.py`` (``build_full_report_markdown``) when
   the run is saved.

   .. method:: begin_run(data, phases=None)

      Called by ``CalculationTask.run`` before the first phase: resets
      every phase's report / layer attributes to their not-calculated
      values (``compute.run_phases.reset_results``), clears
      ``consequence_result``, stores the selected ``run_phases`` and
      re-derives the depth metadata the powered grounding layer needs.

   .. method:: run_drifting_model(data)

      Execute the complete drifting risk calculation pipeline:
      drift corridors, analytical probability-hole integration
      (cross-section CDF), shadow cascade, repair time application,
      wind rose weighting.

   .. method:: run_ship_collision_model(data)

      Calculate ship-ship collision frequencies for all encounter types
      (head-on, overtaking, crossing, merging, bend).

   .. method:: run_powered_grounding_model(data)

      Calculate powered grounding (Category I and II).

      Implementation details:

      - Groups draughts (plus squat when ticked) into effective depth
        bins based on depth contours.
      - Reuses cached shadow/ray-casting computations per bin.
      - Emits progress updates per obstacle-bin step for long runs.

   .. method:: run_powered_allision_model(data)

      Calculate powered allision (Category I and II) with structures.

   .. method:: run_drift_visualization(data)

      Generate drift corridor visualisation layers without full
      probability calculation.


compute.calculation_task
========================

**Source file:** ``compute/calculation_task.py`` (`View on GitHub <https://github.com/axelande/OMRAT/blob/main/compute/calculation_task.py>`__)

.. class:: CalculationTask(description, calc_object, data, phases=None)

   ``QgsTask`` that runs one **Run model** in a background thread.
   ``run()`` prepares the traffic
   (:func:`compute.traffic_pipeline.prepare_traffic_for_run`, each step
   logged), calls ``Calculation.begin_run``, runs the selected phases in
   ``compute.run_phases.PHASES`` order and finally the oil-spill
   consequence step.  Signals: ``progress_updated(int, int, str)``,
   ``calculation_finished(object)``, ``calculation_failed(str)``.


compute.run_phases
==================

**Source file:** ``compute/run_phases.py`` (`View on GitHub <https://github.com/axelande/OMRAT/blob/main/compute/run_phases.py>`__)

The model parts of a run (:ref:`Calculation matrix <calculation-matrix>`).  Pure Python.

* ``PHASES`` -- ``('drifting', 'collision', 'powered_grounding',
  'powered_allision')``, the run order; ``PHASE_LABELS``,
  ``PHASE_TOTALS`` (history keys per phase), ``PHASE_METHODS``
  (``Calculation`` method per phase), ``PHASE_RESULT_ATTRS``
  (attributes and their not-calculated values).
* ``normalize_phases(phases)``, ``skipped_phases``, ``is_partial``,
  ``skipped_totals``, ``describe_selection``, ``reset_results(calc,
  phases=None)``.


compute.traffic_redirect
========================

Suppressed legs and their traffic redirects (:ref:`suppress-leg`).
Pure Python, no QGIS dependency.

**Source file:** ``compute/traffic_redirect.py`` (`View on GitHub <https://github.com/axelande/OMRAT/blob/main/compute/traffic_redirect.py>`__)

.. function:: apply_traffic_redirects(data)

   Move every suppressed leg's traffic onto its targets and remove the
   suppressed legs from ``data`` (in place, on a private copy).
   Returns ``{'moves': [...], 'removed': [...], 'warnings': [...]}``.

.. function:: prepare_export_data(data)

   ``(copy, summary)``: a deep copy with the redirects applied, used by
   the IWRAP export.  ``data`` is not changed.

.. function:: merge_direction(target, source, share_pct)

   Add ``share_pct`` % of one direction's ships to another: frequencies
   add, per-cell attributes become frequency-weighted means.  Returns
   the ships per year moved.

.. function:: auto_target_dir(segment_data, src, from_dir, dst)

   The direction of ``dst`` closest in bearing to direction
   ``from_dir`` of ``src``.  Pre-fills **To direction** in the dialog.

.. function:: drop_legs_from_junctions(junctions, removed, segment_data)

   Serialised junctions without the removed legs (rows renormalised,
   geometric fallback for rows that only fed a removed leg).

.. function:: after_split(segment_data, parent, sub_ids)

   Keep redirects consistent after ``split_leg_at_points``.


compute.speed_restriction
=========================

Speed restriction zones (:ref:`speed-restriction-zones`).  Pure Python
(shapely), no QGIS dependency.

**Source file:** ``compute/speed_restriction.py`` (`View on GitHub <https://github.com/axelande/OMRAT/blob/main/compute/speed_restriction.py>`__)

.. function:: apply_speed_restrictions(data)

   Restrict the speeds of ``data['traffic_data']`` in place (compute copy
   only): the time-equivalent ``Speed (knots)`` along each leg in a zone
   plus ``Speed at start point (knots)`` / ``Speed at end point (knots)``.
   Returns ``{'zones': [...], 'legs': [...]}``.  Raises
   ``SpeedRestrictionError`` for overlapping or invalid enabled zones.

.. function:: speed_matrix(dir_data, at=None)

   The speed matrix a model reads: ``at='start'`` / ``'end'`` = the speed
   at that waypoint (crossing, merging, bend, powered Cat II), ``None`` =
   along the leg.  Falls back to ``Speed (knots)``.

.. function:: cell_limit(zone, ship_type, length_idx, bounds)

   ``(limit_kn, compliance)`` for one traffic cell, ``None`` when exempt;
   the first matching rule wins.

.. function:: blocking_problems(data)

   The messages that stop Run model: overlapping or invalid enabled zones.


compute.traffic_pipeline
========================

**Source file:** ``compute/traffic_pipeline.py`` (`View on GitHub <https://github.com/axelande/OMRAT/blob/main/compute/traffic_pipeline.py>`__)

.. function:: prepare_traffic_for_run(data)

   The traffic steps every run takes on its private copy, in order: gap
   fill, traffic scaling, suppressed-leg redirects, speed restriction
   zones.  Used by ``CalculationTask.run``,
   ``sensitivity_task.run_phases_on_stub`` and
   ``compare_explain.run_collision_model``.  Returns
   ``{'gaps': [...], 'redirect': {...}, 'speed': {...}}``; raises
   ``SpeedRestrictionError`` for overlapping or invalid enabled zones.


Other compute helpers
=====================

Small pure-Python modules (no QGIS) that own one rule each:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - What it owns
   * - ``compute/traffic_gaps.py``
     - ``fill_traffic_gaps(data)``: cells with ships but no speed /
       draught get the same type and length on other legs, then the
       same type, then the same length, then 10 kn / 5 m
       (:ref:`traffic-gaps`); ``preview_traffic_gaps`` counts them
       without changing the data.
   * - ``compute/traffic_scaling.py``
     - ``apply_traffic_scaling(data)``: ``Frequency * Scaling (%) / 100``
       in place (re-exported by ``compute.data_preparation``).
   * - ``compute/encounter_angle.py``
     - ``pc['min_encounter_angle_deg']`` (``KEY``, ``DEFAULT_DEG`` = 10,
       clamp 1-45): ``min_angle_deg(pc)``, ``min_angle_rad(pc)``,
       ``angle_counts(theta_rad, limit_rad)``.
   * - ``compute/passenger_reduction.py``
     - Passenger ship / fast ferry reduction factor (default 20):
       ``factor(pc, key)``, ``collision_scales``, ``powered_scales``.
   * - ``compute/squat.py``
     - ``pc['squat_drifting']`` / ``pc['squat_powered']`` (default on):
       ``squat_on(pc, key)``, ``squat_for_row(names, row, speed_kts)``
       (Barras open water, :math:`C_b V^2/100`).
   * - ``compute/drift_start.py``
     - ``drift['start_from']``: ``uses_distribution_center(drift)``,
       ``mixture_mean(distributions, weights)``.
   * - ``compute/cat1_start_zone.py``
     - Category I start zone (``cat1_skip_start`` /
       ``cat1_skip_start_pct``, default on, 10 %):
       ``start_zone_fraction(pc)``.
   * - ``compute/ray_sampling.py``
     - Powered ray sampling over the lateral mixture (``RAY_SPAN_SIGMA``
       = 6, ``N_RAYS`` = 1000, ``MAX_RAYS`` = 3000): ``mixture_rays``,
       ``mixture_span``.
   * - ``compute/drift_reach.py``
     - Repair-time quantiles and the drift reach
       (``REACH_QUANTILE`` = 0.99): ``repair_time_quantile_h``,
       ``drift_reach_m``.
   * - ``compute/leg_labels.py``
     - Leg names in outputs: ``leg_label``, ``leg_display``,
       ``leg_field_map``, ``pair_label``.
   * - ``compute/iwrap_defaults.py``
     - IWRAP default values: ``default_pc_values()``,
       ``default_drift_values()``.


geometries.drift
=================

**Source file:** ``geometries/drift/generator.py`` (`View on GitHub <https://github.com/axelande/OMRAT/blob/main/geometries/drift/generator.py>`__)

Drift Corridor Generation
--------------------------

.. class:: DriftCorridorGenerator(omrat)

   Main orchestrator for drift corridor generation.

   **Line:** 107 | `Source <https://github.com/axelande/OMRAT/blob/main/geometries/drift/generator.py#L107>`__

   .. method:: precollect_data(depth_threshold, height_threshold)

      Pre-collect data from UI in the main thread.

   .. method:: generate_corridors(depth_threshold, height_threshold, target_prob=1e-3)

      Generate corridors for all legs and 8 directions.

      :return: List of corridor dicts with keys: ``direction``,
               ``angle``, ``leg_index``, ``polygon``,
               ``anchor_polygon``, ``deep_polygon``

Corridor Construction
---------------------

.. function:: create_base_surface(leg, half_width)

   Create the base distribution rectangle for a leg.

.. function:: create_projected_corridor(leg, half_width, drift_angle_deg, projection_dist)

   Create the full unclipped corridor by projecting the base surface.

.. function:: create_obstacle_shadow(obstacle, drift_angle_deg, corridor_bounds)

   Create the blocking shadow zone behind an obstacle.

.. function:: clip_corridor_at_obstacles(corridor, obstacles, drift_angle_deg)

   Clip corridor by removing all shadow zones.

Distribution Utilities
-----------------------

.. function:: get_projection_distance(repair_params, drift_speed_ms, target_prob=1e-3, max_distance=50000)

   Calculate maximum drift distance from repair time distribution.

.. function:: get_distribution_width(std, coverage=0.99)

   Calculate lateral strip width for given distribution coverage.

Coordinate Utilities
---------------------

.. function:: get_utm_crs(lon, lat)

   Return the appropriate UTM CRS for a geographic location.

.. function:: transform_geometry(geom, from_crs, to_crs)

   Transform a Shapely geometry between coordinate reference systems.

.. function:: compass_to_vector(angle_deg, distance)

   Convert compass angle and distance to (dx, dy) vector in UTM.


geometries.analytical_probability
==================================

.. function:: compute_probability_holes_analytical(lines, distributions, weights, objs_gdf_list, distance, progress_callback=None, n_slices=100)

   The probability holes the drifting model uses: the probability that
   vessels drifting from each leg hit each obstacle, by analytical
   cross-section CDF integration over ``n_slices`` slices per leg.
   Deterministic; same arguments and return shape as
   ``compute_probability_holes`` below.


geometries.calculate_probability_holes
=======================================

.. function:: compute_probability_holes(lines, distributions, weights, objs_gdf_list, distance, progress_callback=None)

   Compute probability that vessels will hit each obstacle using
   Monte Carlo integration with parallel execution.  Not selectable in
   a run: kept as an independent cross-check for tests and examples.

   :param lines: List of leg LineStrings (UTM)
   :param distributions: List of distribution parameter lists
   :param weights: List of weight lists
   :param objs_gdf_list: List of obstacle polygon lists (one per direction)
   :param distance: Maximum drift distance (metres)
   :param progress_callback: Optional ``(completed, total, msg) -> bool``
   :return: 3D array ``[leg][direction][obstacle]`` of probabilities

   **Line:** 684 | `Source <https://github.com/axelande/OMRAT/blob/main/geometries/calculate_probability_holes.py#L684>`__


geometries.get_powered_overlap
==============================

The ray caster both powered models and the powered **View** share
(:ref:`shadow-ray-casting`).

.. function:: cat2_probability(summary, recovery)

   Category II hit probability of one obstacle (or edge) summary,
   summed ray by ray: :math:`\sum_r m_r \exp(-d_r / (a_i V))` with
   ``recovery`` = :math:`a_i V`; 0 when ``recovery <= 0``.

   **Line:** 406 | `Source <https://github.com/axelande/OMRAT/blob/main/geometries/get_powered_overlap.py#L406>`__

.. function:: model_projector(segment_data)

   The ``ModelProjector`` the powered models and the View project with:
   transverse Mercator about the centre of the legs' bounding box
   (defined in ``geometries/model_projection.py``, re-exported here).

   **Line:** 85 | `Source <https://github.com/axelande/OMRAT/blob/main/geometries/model_projection.py#L85>`__


geometries.result_layers
=========================

.. function:: create_result_layers(report, structures, depths, add_to_project=True, segment_data=None)

   Create the drifting QGIS result layers: one LineString per obstacle
   boundary edge with per-edge probability attributes.

   :param report: Drifting report dict from calculation
   :param structures: List of structure dicts
   :param depths: List of depth dicts
   :param segment_data: Names the per-leg columns after the route table
      (``LEG_1_2_b``); without it they fall back to the id form
   :return: ``(allision_layer, grounding_layer)``

.. function:: create_powered_grounding_layer(powered_grounding_report, depths, add_to_project=True, segment_data=None)
              create_powered_allision_layer(powered_allision_report, structures, add_to_project=True, segment_data=None)

   Powered result layers: one LineString per boundary edge of each hit
   depth / structure polygon; ``total_edge_probability`` comes from the
   edges the rays hit (``by_obstacle_segment_legdir``), so the edges of
   an obstacle sum to its probability.

.. function:: create_collision_layers(collision_report, segment_data, add_to_project=True)

   Ship-ship result layers, returned as ``(line_layer, point_layer)``:
   one line per leg (head-on, overtaking) and one point per waypoint
   (crossing, merging, bend).


omrat_utils
============

Key data management classes:

.. class:: Traffic(omrat, widget)

   Manages the traffic data table UI and underlying data structures.

.. class:: OObject(omrat)

   Manages depth and structure obstacle layers and data.

.. class:: DriftSettings(omrat)

   Manages drift parameters (wind rose, speed, repair time).

.. class:: CausationFactors(omrat)

   Manages causation factor values.

.. class:: Distributions(omrat)

   Manages lateral traffic distribution parameters and visualisation.

.. class:: AIS(omrat)

   Manages AIS database connections and traffic data import.

.. class:: GatherData(omrat)

   Collects all data for save/load and populates from saved data.

.. class:: Storage(omrat)

   Handles project file I/O (``.omrat`` JSON format).

.. class:: ShipCategories(omrat)

   Manages ship type and size bin definitions.

.. class:: Junctions(omrat=None)

   ``omrat_utils/handle_junctions.py``: the live junction registry and
   its transition matrices (geometry / AIS / user sources).

.. class:: Consequence(omrat)

   ``omrat_utils/handle_consequence.py``: the oil-spill consequence
   inputs and their dialogs.

.. class:: SpeedRestrictionZones(omrat)

   ``omrat_utils/speed_restriction_zones.py``: the speed restriction
   zones (``omrat.speed_zones``), their map layer and save / load.

Pure helpers (no Qt / QGIS) used by the dialogs and reports:

* ``omrat_utils/project_sanitize.py`` -- ``sanitize_project(data)``,
  the fixes applied on save, load and run (see
  :ref:`reference-data-format`).
* ``omrat_utils/copy_traffic.py`` -- Copy traffic and the AIS lock
  (``copy_leg_traffic``, ``release_copy``, ``misaligned_targets``).
* ``omrat_utils/traffic_links.py`` -- copy / redirect / suppressed-with
  links between legs (``build_links``, ``link_groups``, ``leg_label``).
* ``omrat_utils/traffic_summary.py`` --
  ``build_traffic_summary(data, *, project_path='', generated=None)``,
  the content and checks of the Traffic report (:ref:`traffic-report`);
  ``omrat_utils/traffic_report_html.py`` --
  ``render_traffic_report_html(summary, *, decimal='.', csv_stem='traffic')``,
  one self-contained HTML page.
* ``omrat_utils/table_clipboard.py`` -- number parsing / formatting and
  TSV copy-paste for the Traffic Data table.
* ``omrat_utils/result_compare.py``, ``compare.py``,
  ``compare_inputs.py``, ``compare_breakdown.py``, ``compare_zone.py``,
  ``compare_explain.py`` -- the Run Analysis result tables and the
  Compare tab.

Dialogs with a headless core (used by the tests):

* ``omrat_utils/waypoint_editor_dialog.py`` -- **Settings > Waypoint
  editor...**; ``apply_detach(omrat, wp_id, leg_id, side, gap_m)``,
  ``apply_attach(omrat, leg_id, side, wp_id)`` (:ref:`waypoint-editor`).
* ``omrat_utils/speed_restriction_dialog.py`` -- **Settings > Speed
  restriction zones...** (:ref:`speed-restriction-zones`).
* ``omrat_utils/depth_simplify_dialog.py`` -- **Depths > Simplify...**
  (:ref:`simplify-depths`).
* ``omrat_utils/copy_traffic_dialog.py`` (``apply_copy``),
  ``suppress_leg_dialog.py`` (``apply_suppression``) -- Routes tab.
* ``omrat_utils/traffic_report_dialog.py`` --
  ``write_traffic_report(omrat, path, data=None)``,
  ``open_traffic_report(omrat, path=None)``.
* ``omrat_utils/calculation_matrix_dialog.py`` -- **Calculation
  matrix...** (:ref:`Calculation matrix <calculation-matrix>`); ``sensitivity_dialog.py`` /
  ``sensitivity_task.py`` -- **Sensitivity analysis...**
  (:ref:`sensitivity-analysis`).
