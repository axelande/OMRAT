.. _code-flow-collisions:

===================================================
Code Flow: Ship-Ship Collision Model
===================================================

This chapter walks the ship-ship collision model one function at a
time, in the order the calls fire when a user presses **Run Model**.
Pair it with :ref:`collisions`, which derives the four collision
formulas (head-on, overtaking, crossing, bend).

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
* ``self.collision_report`` -- per-leg + per-type breakdown
* ``LEPHeadOnCollision`` / ``LEPOvertakingCollision`` /
  ``LEPCrossingCollision`` / ``LEPMergingCollision`` line-edit text

.. container:: source-code-ref pipeline

   **Entry:** ``compute/ship_collision_model.py:526`` -- `run_ship_collision_model() <https://github.com/axelande/OMRAT/blob/main/compute/ship_collision_model.py#L526>`__


Top-level call tree
===================

::

   run_ship_collision_model(data)
     |
     +-- for each leg:
     |     _calc_head_on_collisions(...)
     |       +-- _get_weighted_mu_sigma(seg_info, dir)
     |       |     +-- get_distribution(seg_info, dir)                  # compute/data_preparation.py
     |       +-- get_loa_midpoint / estimate_beam
     |       +-- get_head_on_collision_candidates(...)                  # compute/basic_equations.py
     |
     |     _calc_overtaking_collisions(...)
     |       +-- _get_weighted_mu_sigma(...)
     |       +-- get_overtaking_collision_candidates(...)
     |
     |     _calc_bend_collisions(...)
     |       +-- get_bend_collision_candidates(...)
     |
     +-- _calc_crossing_collisions(...)                                 # across leg pairs
     |     +-- _parse_point / _points_match / _calc_bearing
     |     +-- get_crossing_collision_candidates(...)
     |
     +-- aggregate into result dict
     +-- update LEP* line-edits


Unlike the drifting model, the ship-collision phase has **no geometric
precompute step**.  All heavy numerics live inside four pure-math
functions in :mod:`compute.basic_equations`:
:func:`~compute.basic_equations.get_head_on_collision_candidates`,
:func:`~compute.basic_equations.get_overtaking_collision_candidates`,
:func:`~compute.basic_equations.get_crossing_collision_candidates`,
:func:`~compute.basic_equations.get_bend_collision_candidates`.
Everything in the mixin is bookkeeping around those.


Data pulled from ``data``
=========================

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - Use
   * - ``traffic_data[leg][dir]['Frequency (ships/year)']``
     - Per-LOA, per-type frequency matrix.  Same structure as the
       drifting model uses.
   * - ``traffic_data[leg][dir]['Speed (knots)']``
     - Per-cell speed.  A cell can hold a scalar or a list; the code
       averages lists.
   * - ``traffic_data[leg][dir]['Ship Beam (meters)']``
     - Per-cell beam.  Falls back to ``estimate_beam(LOA)`` when
       missing.
   * - ``segment_data[leg]`` (``mean1_1``, ``std1_1``, ``weight1_1``
       ... per direction)
     - Lateral distribution parameters used by the head-on and
       overtaking formulas.  ``_get_weighted_mu_sigma`` reduces them
       to a single :math:`(\mu, \sigma)` per direction.
   * - ``segment_data[leg]['bend_angle']``
     - Used only when it exceeds 5 degrees; drives the bend-collision
       formula.
   * - ``segment_data[leg]['Start_Point'] / 'End_Point' / 'bearing'``
     - Used by crossing-collision geometry to find waypoints and
       crossing angles between pairs of legs.
   * - ``pc.headon`` / ``pc.overtaking`` / ``pc.crossing`` /
       ``pc.bend``
     - Causation factors.  The per-type geometric candidate count is
       multiplied by these to get the accident frequency.
   * - ``ship_categories.length_intervals``
     - Defines the LOA bins (``[{'min','max'}]``).  ``get_loa_midpoint``
       returns the centre of each bin for speed / beam estimation.


:meth:`run_ship_collision_model`: orchestrator
==============================================

.. container:: source-code-ref pipeline

   **Source:** ``compute/ship_collision_model.py:526`` -- `run_ship_collision_model() <https://github.com/axelande/OMRAT/blob/main/compute/ship_collision_model.py#L526>`__

Step by step:

1. Short-circuit if ``traffic_data`` or ``segment_data`` is empty; set
   totals to zero and return.
2. Read causation factors from ``data['pc']`` (default to IALA
   recommended values).
3. For every leg (outer loop is in this function), call three
   per-leg helpers:

   * :meth:`_calc_head_on_collisions` - same-leg, opposite-direction.
   * :meth:`_calc_overtaking_collisions` - same-leg, same-direction
     with different speeds.
   * :meth:`_calc_bend_collisions` - same-leg geometric bend.

   Each helper returns a scalar frequency for that leg.  Progress is
   reported at 80 % of the ``spatial`` phase because crossing
   collisions come next and still need the remaining 20 %.

4. Call :meth:`_calc_crossing_collisions` once for the full
   ``leg_keys`` list - it iterates every leg pair and looks for
   shared waypoints.  Progress is reported inside the helper
   (``cascade`` phase).
5. Build the result report:

   .. code-block:: python

      self.collision_report = {
          'totals': result,   # head_on, overtaking, crossing, bend, total
          'by_leg': by_leg,   # per leg dict of head_on / overtaking / bend
          'causation_factors': {...},
      }

6. Write the four result line-edits in one block, swallowing
   ``Exception`` because Qt widgets can disappear mid-run in pytest
   contexts.


Shared helpers
==============

Three static helpers are used by every per-leg routine.

:func:`_get_weighted_mu_sigma`
-------------------------------

.. container:: source-code-ref pipeline

   **Source:** ``compute/ship_collision_model.py:58`` -- `_get_weighted_mu_sigma() <https://github.com/axelande/OMRAT/blob/main/compute/ship_collision_model.py#L58>`__

Reads the per-direction lateral distributions (up to three superposed
Gaussians) via :func:`compute.data_preparation.get_distribution`,
normalises the weights to sum to 1, and reduces them to a single
``(mu, sigma)`` pair using the mixture-of-Gaussians variance identity:

.. math::

   \mathrm{Var}[X] = \sum_i w_i (\sigma_i^2 + \mu_i^2) - (\sum_i w_i \mu_i)^2

This is what the collision formulas expect as input.  If the resulting
:math:`\sigma` is below 1 m the helper raises ``ValueError`` to catch
misconfigured distributions early (the formula divides by sigma).

:func:`get_loa_midpoint(loa_idx, length_intervals)`
---------------------------------------------------

Returns the centre of LOA bin ``loa_idx``.  Used for beam estimation
when the traffic cell doesn't provide beam directly.

:func:`estimate_beam(loa)`
--------------------------

``loa / 6.5`` - a typical length-to-beam ratio.  Only used as a
fallback.


:meth:`_calc_head_on_collisions`
================================

.. container:: source-code-ref pipeline

   **Source:** ``compute/ship_collision_model.py:138`` -- `_calc_head_on_collisions() <https://github.com/axelande/OMRAT/blob/main/compute/ship_collision_model.py#L138>`__

Structure:

1. Fetch direction 1 and direction 2 cells
   (``leg_dirs[dir1]`` / ``leg_dirs[dir2]``).  If only one direction
   exists, head-on is zero.
2. Extract per-cell ``freq``, ``speed``, ``beam`` arrays.  For a
   cell that holds a scalar speed, that scalar is used; for a
   list/array, the mean is used.
3. Get ``(mu1, sigma1)`` and ``(mu2, sigma2)`` via
   :meth:`_get_weighted_mu_sigma`.
4. Double loop over every ``(loa_i, type_j)`` cell in direction 1 and
   every ``(loa_k, type_l)`` cell in direction 2.  For each pair:

   .. code-block:: python

      n_g = get_head_on_collision_candidates(
          Q1=q1, Q2=q2, V1=v1_ms, V2=v2_ms,
          mu1=mu1_lat, mu2=mu2_lat,
          sigma1=sigma1_lat, sigma2=sigma2_lat,
          B1=b1, B2=b2, L_w=leg_length_m,
      )
      leg_head_on += n_g * pc_headon

The nested loop runs per leg; for a busy leg with 21 ship types x 5
LOA bins per direction there are ~10 k pairs, each a handful of
arithmetic ops.  No shapely.

The core formula is in
:func:`compute.basic_equations.get_head_on_collision_candidates`.  See
:ref:`collisions` for the math.


:meth:`_calc_overtaking_collisions`
====================================

.. container:: source-code-ref pipeline

   **Source:** ``compute/ship_collision_model.py:239`` -- `_calc_overtaking_collisions() <https://github.com/axelande/OMRAT/blob/main/compute/ship_collision_model.py#L239>`__

Iterates each direction independently (overtaking is same-direction
by definition).  For each direction:

1. Flatten every non-zero cell into a ``ship_cells`` list of
   ``(loa, type, freq, speed_ms, beam)``.
2. Double loop over pairs ``(fast, slow)`` where ``v_fast > v_slow``.
   The slower ship's cell and the faster ship's cell are passed to
   :func:`get_overtaking_collision_candidates` along with
   ``(mu_ot, sigma_ot)`` = the direction's weighted lateral
   distribution.
3. Multiply by ``pc.overtaking`` and accumulate into
   ``leg_overtaking``.


:meth:`_calc_bend_collisions`
==============================

.. container:: source-code-ref pipeline

   **Source:** ``compute/ship_collision_model.py:313`` -- `_calc_bend_collisions() <https://github.com/axelande/OMRAT/blob/main/compute/ship_collision_model.py#L313>`__

Bend collisions model a ship that fails to turn at the leg's
downstream waypoint and continues straight.

1. Loop the leg's direction cells to accumulate an **average**
   frequency (``avg_freq``) and average length / beam (``avg_length``,
   ``avg_beam``).  The averaging is a simple running mean weighted
   by non-zero cell count.
2. Read ``segment_data[leg]['bend_angle']``.  If <= 5 degrees, return 0
   (no meaningful bend).
3. Otherwise, call :func:`get_bend_collision_candidates`:

   .. code-block:: python

      n_g = get_bend_collision_candidates(
          Q=avg_freq, P_no_turn=0.01,
          L=avg_length, B=avg_beam,
          theta=bend_angle_rad,
      )

   ``P_no_turn`` is hard-coded at 0.01 (IALA default).
4. Multiply by ``pc.bend`` and return.


:meth:`_calc_crossing_collisions`
==================================

.. container:: source-code-ref pipeline

   **Source:** ``compute/ship_collision_model.py:367`` -- `_calc_crossing_collisions() <https://github.com/axelande/OMRAT/blob/main/compute/ship_collision_model.py#L367>`__

This is the only function that does cross-leg geometry.

1. Outer double loop over every ``(leg1, leg2)`` pair with
   ``j > i`` to avoid double-counting.
2. Parse each leg's ``Start_Point`` and ``End_Point`` via
   :meth:`_parse_point`.
3. Short-circuit unless the two legs share a waypoint
   (:meth:`_points_match` comparing start/end pairs).  This is the
   cheap filter that keeps the O(L^2) loop tractable.
4. Compute each leg's bearing from the stored ``bearing`` field or
   from :meth:`_calc_bearing(start, end)`.
5. Crossing angle = ``abs(bearing1 - bearing2)`` reduced to
   ``[0, 90]``.  Angles below ~0.1 rad are treated as parallel and
   skipped.
6. For every ``(dir1_cell, dir2_cell)`` pair on the two legs, call
   :func:`get_crossing_collision_candidates`:

   .. code-block:: python

      n_g = get_crossing_collision_candidates(
          Q1=q1, Q2=q2, V1=v1_ms, V2=v2_ms,
          L1=l1, L2=l2, B1=b1, B2=b2,
          theta=crossing_angle_rad,
      )

   and multiply by ``pc.crossing`` before accumulating.

7. Report progress (``cascade`` phase) after every pair.


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
   * - ``get_head_on_collision_candidates``
     - :math:`N_G = \frac{Q_1}{V_1}\frac{Q_2}{V_2} V_{ij} P_G L_w / \text{s/yr}`,
       with :math:`P_G` the Gaussian lateral-overlap probability.
   * - ``get_overtaking_collision_candidates``
     - Same structure but :math:`V_{ij} = |V_\mathrm{fast} -
       V_\mathrm{slow}|` and only pairs with fast > slow contribute.
   * - ``get_crossing_collision_candidates``
     - :math:`N_G = \frac{Q_1 Q_2}{V_1 V_2} \frac{D}{\sin\theta}
       (L_1 + L_2 \sin\theta + B_1 + B_2 \cos\theta) / \text{s/yr}`.
   * - ``get_bend_collision_candidates``
     - :math:`N_G = Q \cdot P_\mathrm{no\_turn} \cdot (L + B
       \tan(\theta))`, simplified from Hansen for a single-leg bend.

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
       'bend': total_bend,
       'total': sum of all four,
   }
   self.ship_collision_prob = result['total']
   self.collision_report = {
       'totals': result,
       'by_leg': {leg: {'head_on': ..., 'overtaking': ..., 'bend': ...}, ...},
       'causation_factors': {...},
   }

The four line-edits (``LEPHeadOnCollision``, ``LEPOvertakingCollision``,
``LEPCrossingCollision``, ``LEPMergingCollision``) are set with
``f"{result[...]:.3e}"``.  ``collision_report`` is the input to the
ship-collision Markdown report written later by
:class:`DriftingReportMixin` (despite the name, the mixin writes all
result reports).


Function reference
==================

``compute/ship_collision_model.py`` (all methods of
:class:`ShipCollisionModelMixin`)

* ``run_ship_collision_model`` - orchestrator.
* ``_calc_head_on_collisions`` - per leg.
* ``_calc_overtaking_collisions`` - per leg, per direction.
* ``_calc_bend_collisions`` - per leg, angle > 5 deg.
* ``_calc_crossing_collisions`` - all leg pairs with shared waypoints.
* ``_get_weighted_mu_sigma`` - lateral distribution reduction.
* ``_parse_point`` / ``_points_match`` / ``_calc_bearing`` - geometry
  helpers.
* ``get_loa_midpoint`` / ``estimate_beam`` - LOA/beam fallback
  estimates.

``compute/basic_equations.py``

* ``get_head_on_collision_candidates``
* ``get_overtaking_collision_candidates``
* ``get_crossing_collision_candidates``
* ``get_bend_collision_candidates``

``compute/data_preparation.py``

* ``get_distribution`` - extract up to three superposed distributions
  from one segment's ``mean/std/weight`` fields.
