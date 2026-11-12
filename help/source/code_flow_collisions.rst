.. _code-flow-collisions:

===================================================
Code Flow: Ship-Ship Collision Model
===================================================

This chapter walks the ship-ship collision model one function at a
time, in the order the calls fire when a user presses **Run model**.
Pair it with :ref:`collisions`, which derives the formulas of the five
encounter types (head-on, overtaking, crossing, merging, bend).

.. contents:: In this chapter
   :local:
   :depth: 2


Entry point
===========

:meth:`ShipCollisionModelMixin.run_ship_collision_model` is phase 2 of
:class:`~compute.calculation_task.CalculationTask` (see
:ref:`code-flow`).  It takes the same ``data`` dict as the other
phases and writes

* ``self.ship_collision_prob`` -- total frequency (accidents/year)
* ``self.collision_report`` -- per-type totals plus per-leg, per-waypoint,
  per-leg-pair and per-cell breakdowns
* ``self.collision_line_layer`` / ``self.collision_point_layer`` -- the
  result layers
* ``LEPHeadOnCollision`` / ``LEPOvertakingCollision`` /
  ``LEPCrossingCollision`` / ``LEPMergingCollision`` /
  ``LEPBendCollision`` line-edit text

.. container:: source-code-ref pipeline

   **Entry:** ``compute/ship_collision_model.py:1070`` -- `run_ship_collision_model() <https://github.com/axelande/OMRAT/blob/main/compute/ship_collision_model.py#L1070>`__


Top-level call tree
===================

::

   run_ship_collision_model(data)
     |
     +-- collision_type_scale(data, pc)              # passenger / fast ferry 1/sqrt(F) per type row
     +-- min_angle_rad(pc)                           # compute/encounter_angle.py
     +-- _extract_pc_and_intervals(data, pc)
     |
     +-- _run_per_leg_loop(...)  -> _process_single_leg(leg) for each leg:
     |     _calc_head_on_collisions(...)
     |       +-- lateral_mixture(seg_info, 0 / 1)                     # geometries/distribution_curves.py
     |       +-- _dir_arrays / _extract_speed_ms / _extract_beam
     |       +-- get_loa_midpoint / estimate_beam
     |       +-- _overlap -> lateral_overlap_probability(...)         # compute/basic_equations.py
     |       +-- head_on_candidates_from_overlap(...)
     |
     |     _calc_overtaking_collisions(...)
     |       +-- lateral_mixture(seg_info, dir)
     |       +-- _collect_ship_cells(...)
     |       +-- _overlap -> lateral_overlap_probability(...)
     |       +-- overtaking_candidates_from_overlap(...)
     |
     +-- _run_crossing_and_merge_waypoints(...)
     |     _calc_crossing_collisions(...)            # route pairs per waypoint: crossing / merging / bend
     |       +-- _route_registry -> refresh_junction_registry(...)    # geometries/junctions.py
     |       +-- model_projector_or_none(segment_data)                # geometries/model_projection.py
     |       +-- for each waypoint: _junction_route_encounters(...)
     |             +-- _junction_routes (_junction_outward_bearing, speed at the waypoint)
     |             +-- _route_pair_kind (_paths_interleave)
     |             +-- angle_counts(theta, limit)
     |             +-- _crossing_ship_cell_loop -> get_crossing_collision_candidates(...)
     |
     +-- _split_bend_over_legs / _fill_result_totals
     +-- _assemble_collision_report(...)
     +-- _finalize_collision_layers(...)             # geometries/result_layers.py
     +-- _update_collision_ui(result)                # LEP* line-edits


Unlike the drifting model, the ship-collision phase has **no geometric
precompute step**.  All heavy numerics live inside pure-math functions
in :mod:`compute.basic_equations`:
:func:`~compute.basic_equations.lateral_overlap_probability` (the lateral
overlap :math:`P_G` of two mixtures),
:func:`~compute.basic_equations.head_on_candidates_from_overlap`,
:func:`~compute.basic_equations.overtaking_candidates_from_overlap` and
:func:`~compute.basic_equations.get_crossing_collision_candidates`
(crossing, merging and bend).
Everything in the mixin is bookkeeping around those.


Data pulled from ``data``
=========================

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - Use
   * - ``traffic_data[leg][dir]['Frequency (ships/year)']``
     - Frequency matrix, rows = ship types, columns = length classes.
       Same structure as the drifting model uses.  Values are read
       **after** :func:`compute.traffic_pipeline.prepare_traffic_for_run`
       has filled gaps, multiplied each cell by its ``Scaling (%) / 100``
       (the "easy option" global / per-row scaling on the Traffic Data
       tab), moved the traffic of suppressed legs and applied the speed
       restriction zones -- all inherited transparently.
   * - ``traffic_data[leg][dir]['Speed (knots)']``
     - Per-cell speed along the leg (head-on, overtaking).  A cell can
       hold a scalar or a list; the code averages lists.  A missing speed
       (``inf`` or <= 0) is filled before the run
       (:mod:`compute.traffic_gaps`); ``_extract_speed_ms`` still falls
       back to 10 knots.
   * - ``traffic_data[leg][dir]['Speed at start point (knots)']`` /
       ``['Speed at end point (knots)']``
     - Speed at the waypoint, used by crossing, merging and bend
       (:func:`compute.speed_restriction.speed_matrix`).  Only present on
       the run's copy of a leg touched by a speed restriction zone;
       otherwise ``Speed (knots)`` is used.
   * - ``traffic_data[leg][dir]['Ship Beam (meters)']``
     - Per-cell beam.  Falls back to ``estimate_beam(L)`` when missing
       (not finite or <= 0), with ``L`` the length-class midpoint.
   * - ``segment_data[leg]`` (``mean1_1``, ``std1_1``, ``weight1_1``,
       ``u_min1``, ``u_max1``, ``u_p1`` ... per direction)
     - Lateral distribution used by the head-on and overtaking formulas.
       :func:`geometries.distribution_curves.mixture` turns it into the
       weighted components of each direction (weights normalised by
       their sum, uniform part included); the whole mixture is used.
   * - ``segment_data[leg]['line_length']``
     - Leg length :math:`L_w` (m) for head-on and overtaking.
   * - ``segment_data[leg]['Start_Point'] / 'End_Point'`` and
       ``data['junctions']``
     - Waypoints where legs meet, their transition matrices (stored, or
       the geometric default) and the arrival courses of the routes.
   * - ``pc.headon`` / ``pc.overtaking`` / ``pc.crossing`` /
       ``pc.merging`` / ``pc.bend``
     - Causation factors.  The per-type geometric candidate count is
       multiplied by these to get the accident frequency.
   * - ``pc.min_encounter_angle_deg``
     - Smallest angle between two routes at a waypoint (default 10
       degrees, :mod:`compute.encounter_angle`).
   * - ``pc.passenger_reduction`` / ``pc.fast_ferry_reduction`` and
       ``ship_categories.types``
     - Reduction factor :math:`F` (default 20); the causation factor of
       a cell pair is multiplied by :math:`1/\sqrt{F}` per passenger ship
       or fast ferry (:mod:`compute.passenger_reduction`).
   * - ``ship_categories.length_intervals``
     - Defines the length classes (``[{'min','max'}]``).
       ``get_loa_midpoint`` returns the centre of a class: the ship
       length in the crossing diameter and the input of the beam
       estimate.


:meth:`run_ship_collision_model`: orchestrator
==============================================

.. container:: source-code-ref pipeline

   **Source:** ``compute/ship_collision_model.py:1070`` -- `run_ship_collision_model() <https://github.com/axelande/OMRAT/blob/main/compute/ship_collision_model.py#L1070>`__

Step by step:

1. Set the per-run state from ``data['pc']``: ``_collision_type_scale``
   (:func:`collision_type_scale`, ``None`` when both reduction factors
   are 1) and ``_route_angle_limit_rad``
   (:func:`compute.encounter_angle.min_angle_rad`).
2. Short-circuit if ``traffic_data`` or ``segment_data`` is empty; set
   totals to zero and return.
3. Read the five causation factors from ``data['pc']``
   (:meth:`_extract_pc_and_intervals`; defaults are the IWRAP values:
   head-on 4.9e-5, overtaking 1.1e-4, crossing / merging / bend 1.3e-4).
4. For every leg (:meth:`_run_per_leg_loop` ->
   :meth:`_process_single_leg`), call two per-leg helpers:

   * :meth:`_calc_head_on_collisions` - same-leg, opposite-direction.
   * :meth:`_calc_overtaking_collisions` - same-leg, same-direction
     with different speeds.

   Each helper returns a scalar frequency for that leg.  Progress is
   reported at 80 % of the ``spatial`` phase because the waypoint
   encounters come next.

5. Call :meth:`_calc_crossing_collisions` once (through
   :meth:`_run_crossing_and_merge_waypoints`) - it visits every waypoint
   where legs meet and returns the crossing, merging and bend totals.
   Progress is reported inside the helper (``cascade`` phase).
6. :meth:`_split_bend_over_legs` gives each of the two legs of a bend
   half of it in ``by_leg``; :meth:`_fill_result_totals` sums the five
   types into ``result``.
7. Build the result report (:meth:`_assemble_collision_report`):

   .. code-block:: python

      self.collision_report = {
          'totals': result,          # head_on, overtaking, crossing, merging, bend, total
          'by_leg': by_leg,          # per leg: head_on / overtaking / half of each bend
          'by_waypoint': {...},      # "lon lat": crossing / merging / bend
          'by_leg_pair': {...},      # "in1->in2": crossing / merging / bend, angle, waypoint
          'bend_by_pair': {...},
          'by_cell': {...},          # per type: "{ship_type}_{length_class}" -> frequency
          'causation_factors': {...},
          'min_encounter_angle_deg': ...,   # the angle limit used, default 10
      }

8. Build the result layers (:meth:`_finalize_collision_layers`) and
   write the five result line-edits (:meth:`_update_collision_ui`),
   swallowing ``Exception`` because Qt widgets can disappear mid-run in
   pytest contexts.


Shared helpers
==============

These static helpers are used by the per-leg and per-waypoint routines.

:func:`_overlap`
----------------

.. container:: source-code-ref pipeline

   **Source:** ``compute/ship_collision_model.py:69`` -- `_overlap() <https://github.com/axelande/OMRAT/blob/main/compute/ship_collision_model.py#L69>`__

Returns the geometric probability :math:`P_G` of two lateral mixtures
for the mean half-breadth :math:`(B_1 + B_2)/2` of a ship pair, cached per
half-breadth for one leg and direction pair.  The mixtures come from
:func:`geometries.distribution_curves.mixture` (``lateral_mixture``):
up to three normal components and the uniform part of one direction,
weights normalised to sum to 1, on the leg's lateral axis (negative =
starboard of the drawn direction, the same axis for both directions).
:func:`compute.basic_equations.lateral_overlap_probability` sums
:math:`w_i w_j P_G(i, j)` over every component pair, normal x uniform
and uniform x uniform in closed form.

Until 2026-11-09 the model collapsed each direction into one normal
(mixture mean, total variance).

:func:`get_loa_midpoint(loa_idx, length_intervals)`
---------------------------------------------------

Returns the centre of length class ``loa_idx`` -- the *column* of the
traffic matrices (the loops pass ``type_j`` / ``type_l``; the historical
loop names have rows and columns the other way round).  Used as the ship
length in the crossing diameter and for beam estimation when the traffic
cell doesn't provide a beam.

:func:`estimate_beam(loa)`
--------------------------

``loa / 6.5`` - a typical length-to-beam ratio.  Only used as a
fallback.

:func:`_extract_speed_ms` / :func:`_extract_beam`
-------------------------------------------------

Read one cell's speed (knots -> m/s) and beam.  A speed that is not
finite or not above 0 becomes 10 knots; a beam that is not finite or not
above 0 becomes the ``estimate_beam`` fallback.


:meth:`_calc_head_on_collisions`
================================

.. container:: source-code-ref pipeline

   **Source:** ``compute/ship_collision_model.py:299`` -- `_calc_head_on_collisions() <https://github.com/axelande/OMRAT/blob/main/compute/ship_collision_model.py#L299>`__

Structure:

1. Fetch direction 1 and direction 2 cells
   (``leg_dirs[dir1]`` / ``leg_dirs[dir2]``).  If only one direction
   exists, head-on is zero.
2. Extract per-cell ``freq``, ``speed``, ``beam`` arrays
   (:meth:`_dir_arrays`).  For a cell that holds a scalar speed, that
   scalar is used; for a list/array, the mean is used.
3. Get the lateral mixture of each direction,
   ``comps1 = lateral_mixture(seg_info, 0)`` and
   ``comps2 = lateral_mixture(seg_info, 1)``.  No mixture -> head-on is
   zero.
4. Double loop (:meth:`_head_on_outer_loop` /
   :meth:`_head_on_inner_loop`) over every ``(loa_i, type_j)`` cell in
   direction 1 and every ``(loa_k, type_l)`` cell in direction 2 (row =
   ship type, column = length class).  For each pair:

   .. code-block:: python

      p_g = self._overlap(overlap_cache, comps1, comps2, b1, b2)
      n_g = head_on_candidates_from_overlap(q1, q2, v1_ms, v2_ms, p_g, leg_length_m)
      contrib = n_g * pc_headon * self._pair_scale(scale, loa_i, loa_k)

   ``_pair_scale`` is the passenger ship / fast ferry reduction of the two
   ship-type rows (1 when it is off).  Half of ``contrib`` goes to each
   cell's ``by_cell`` entry.

The nested loop runs per leg; for a busy leg with 21 ship types x 5
LOA bins per direction there are ~10 k pairs, each a handful of
arithmetic ops.  No shapely.

The core formula is in
:func:`compute.basic_equations.head_on_candidates_from_overlap`, with
:math:`P_G` from
:func:`compute.basic_equations.lateral_overlap_probability`.  See
:ref:`collisions` for the math.


:meth:`_calc_overtaking_collisions`
====================================

.. container:: source-code-ref pipeline

   **Source:** ``compute/ship_collision_model.py:423`` -- `_calc_overtaking_collisions() <https://github.com/axelande/OMRAT/blob/main/compute/ship_collision_model.py#L423>`__

Iterates each direction independently (overtaking is same-direction
by definition).  For each direction:

1. Flatten every non-zero cell into a ``ship_cells`` list of
   ``(loa, type, freq, speed_ms, beam)`` (:meth:`_collect_ship_cells`).
2. Double loop over pairs ``(fast, slow)`` where ``v_fast > v_slow``.
   Both ships come from the same direction's mixture, so
   ``p_g = self._overlap(cache, comps, comps, b_fast, b_slow)`` and the
   pair goes to :func:`overtaking_candidates_from_overlap`.
3. Multiply by ``pc.overtaking`` and the pair's passenger ship / fast
   ferry reduction, and accumulate into ``leg_overtaking``.


:meth:`_calc_bend_collisions` (removed)
========================================

Removed in November 2026.  The bend is now a route pair in
:meth:`_calc_crossing_collisions`: the two reverse routes of one turn
(``A -> B`` and ``B -> A``), crossing formula per cell pair at the angle
between their arrival courses, ``pc.bend``.  See :ref:`collisions`.


:meth:`_calc_crossing_collisions`
==================================

.. container:: source-code-ref pipeline

   **Source:** ``compute/ship_collision_model.py:501`` -- `_calc_crossing_collisions() <https://github.com/axelande/OMRAT/blob/main/compute/ship_collision_model.py#L501>`__

This is the only function that does cross-leg geometry.

1. :meth:`_route_registry` gives every waypoint where legs meet, with
   its junction matrix (stored, or the geometric default).
2. :meth:`_junction_routes` builds the routes at one waypoint: in-leg ->
   out-leg, carrying the in-leg's arriving direction times
   ``T[in][out]``, with the arrival course and the speed at the waypoint.
3. :meth:`_route_pair_kind` classifies each pair of routes: same out-leg
   = merging, the two reverse routes of a turn = bend, four legs whose
   paths interleave = crossing, anything else = no waypoint encounter.
4. The angle is the one between the two arrival courses, from the leg
   bearings at the waypoint measured in ``model_projector(segment_data)``
   (:mod:`geometries.model_projection`, the projection of the powered
   models; raw lon/lat degrees until 2026-11-12).  A pair within
   ``pc.min_encounter_angle_deg`` (default 10 degrees,
   :mod:`compute.encounter_angle`) of parallel or opposite is skipped:
   the crossing formula divides by the sine of the angle.
5. For every ``(cell_i, cell_j)`` pair of the two routes
   (:meth:`_crossing_ship_cell_loop`), with ``l1`` / ``l2`` the
   length-class midpoints of the cells' columns, call
   :func:`get_crossing_collision_candidates`:

   .. code-block:: python

      n_g = get_crossing_collision_candidates(
          Q1=q1, Q2=q2, V1=v1_ms, V2=v2_ms,
          L1=l1, L2=l2, B1=b1, B2=b2,
          theta=crossing_angle_rad,
      )

   and multiply by the causation factor of the kind (``pc.crossing``,
   ``pc.merging`` or ``pc.bend``) and the pair's passenger ship / fast
   ferry reduction before accumulating
   (:meth:`_accumulate_crossing_contrib`: ``by_waypoint``,
   ``by_leg_pair`` keyed by the two in-legs, and half to each cell's
   ``by_cell`` entry).

6. Report progress (``cascade`` phase) after every waypoint.


The pure-math helpers
=====================

All collision math lives in :mod:`compute.basic_equations`.  Each
helper is a short numpy expression with no external dependencies.

.. container:: source-code-ref pipeline

   **Source:** ``compute/basic_equations.py``

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Formula (see :ref:`collisions` for derivation)
   * - ``lateral_overlap_probability``
     - :math:`P_G = \sum_i \sum_j w_i w_j P(|X_i - Y_j| \le B)` over the
       components of two lateral mixtures, :math:`B = (B_1 + B_2)/2`.
   * - ``head_on_candidates_from_overlap``
     - :math:`N_G = \frac{Q_1}{V_1}\frac{Q_2}{V_2} (V_1 + V_2) P_G L_w / \text{s/yr}`.
   * - ``overtaking_candidates_from_overlap``
     - Same structure but :math:`V_{ij} = V_\mathrm{fast} -
       V_\mathrm{slow}` and only pairs with fast > slow contribute.
   * - ``get_crossing_collision_candidates``
     - :math:`N_G = \frac{Q_1 Q_2}{V_1 V_2} \frac{V_{ij} D_{ij}}{|\sin\theta|} / \text{s/yr}`,
       :math:`V_{ij}` from the law of cosines and :math:`D_{ij}` the
       Pedersen (1995) collision diameter.
   * - ``get_head_on_collision_candidates`` /
       ``get_overtaking_collision_candidates``
     - The same equations with one normal per direction (Hansen eq.
       4.2-4.5); not called by the model any more.

``sec/year`` = :math:`365.25 \times 24 \times 3600`.


Output
======

After :meth:`_calc_crossing_collisions` returns, the orchestrator
assembles:

.. code-block:: python

   result = {
       'head_on': total_head_on,
       'overtaking': total_overtaking,
       'crossing': total_crossing,
       'merging': total_merging,
       'bend': total_bend,
       'total': sum of all five,
   }
   self.ship_collision_prob = result['total']
   self.collision_report = {
       'totals': result,
       'by_leg': {leg: {'head_on': ..., 'overtaking': ..., 'bend': ...}, ...},
       'by_waypoint': {...}, 'by_leg_pair': {...}, 'bend_by_pair': {...},
       'by_cell': {'head_on': {...}, 'overtaking': {...}, 'crossing': {...},
                   'merging': {...}, 'bend': {...}},
       'causation_factors': {...},
       'min_encounter_angle_deg': ...,
   }

The five line-edits (``LEPHeadOnCollision``, ``LEPOvertakingCollision``,
``LEPCrossingCollision``, ``LEPMergingCollision``, ``LEPBendCollision``)
are set with ``f"{result[...]:.3e}"``.  ``collision_report`` is saved
next to the run as ``<stem>.collision.json`` (the source of the
collision **View** breakdown and the Compare tab), feeds the
ship-collision section of the run's Markdown report
(``omrat_utils/full_report.py``) and, through ``by_cell``, the oil-spill
consequence step (:ref:`consequence`).


Function reference
==================

``compute/ship_collision_model.py`` (all methods of
:class:`ShipCollisionModelMixin`)

* ``run_ship_collision_model`` - orchestrator.
* ``_calc_head_on_collisions`` - per leg.
* ``_calc_overtaking_collisions`` - per leg, per direction.
* ``_calc_crossing_collisions`` - route pairs at every waypoint: crossing, merging, bend.
* ``_route_registry`` / ``_junction_routes`` / ``_route_pair_kind`` /
  ``_junction_route_encounters`` / ``_crossing_ship_cell_loop`` - the
  waypoint encounters.
* ``_overlap`` - cached lateral overlap :math:`P_G` of two mixtures.
* ``_parse_point`` / ``_points_match`` / ``_calc_bearing`` /
  ``_junction_outward_bearing`` / ``_end_at`` - geometry helpers
  (bearings in the model's projection).
* ``get_loa_midpoint`` / ``estimate_beam`` / ``_extract_speed_ms`` /
  ``_extract_beam`` - length, beam and speed per cell with fallbacks.
* ``_split_bend_over_legs`` / ``_assemble_collision_report`` - report.
* ``collision_type_scale`` (module function) - passenger ship / fast
  ferry reduction per ship-type row.

``compute/basic_equations.py``

* ``lateral_overlap_probability``
* ``head_on_candidates_from_overlap``
* ``overtaking_candidates_from_overlap``
* ``get_crossing_collision_candidates``

``geometries/distribution_curves.py``

* ``mixture`` - the weighted components (up to three normals and the
  uniform part) of one direction, from the segment's
  ``mean/std/weight`` and ``u_min/u_max/u_p`` fields.

``compute/encounter_angle.py``, ``compute/passenger_reduction.py``,
``geometries/junctions.py`` (``refresh_junction_registry``) and
``geometries/model_projection.py`` (``model_projector_or_none``,
``projected_bearing``).
