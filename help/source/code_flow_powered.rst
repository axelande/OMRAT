.. _code-flow-powered:

=======================================================
Code Flow: Powered Grounding & Allision (Cat I + II)
=======================================================

This chapter walks OMRAT's IWRAP **Category I** and **Category II**
powered-grounding and powered-allision calculations one function at a
time, in the order the calls fire when a user presses **Run model**.
Pair it with :ref:`powered`, which derives the Cat I formula
:math:`N_I = P_c Q \cdot \mathrm{mass}` and the Cat II formula
:math:`N_{II} = P_c Q \sum_r m_r \exp(-d_r/(a_i V))` (summed over the
rays that hit the obstacle) and explains the shadow model.

.. contents:: In this chapter
   :local:
   :depth: 2


Entry points
============

Powered grounding is phase 3 and powered allision is phase 4 of
:class:`~compute.calculation_task.CalculationTask` (see
:ref:`code-flow`).  They share the same ray-cast engine and the same
obstacle projection step, but the obstacle sources differ:

* **Grounding** uses the depth polygons and filters by ship draught:
  only depths not deeper than the ship's draught (plus squat, when
  ticked) are hazards.  One
  computation is run per **depth bin** -- a bin is the set of depths a
  draught-class "sees", so all draughts in the same bin share one
  shadow-aware result.
* **Allision** uses the object polygons and filters by ship height: if
  a ship of known height is lower than a structure it passes under
  (bridge case).

Both phases call :func:`geometries.get_powered_overlap._run_all_computations`
to produce the shadow-aware ray-cast summaries, then iterate traffic to
multiply by per-cell frequency.

.. container:: source-code-ref pipeline

   **Grounding entry:** ``compute/powered_model.py:523`` -- `run_powered_grounding_model() <https://github.com/axelande/OMRAT/blob/main/compute/powered_model.py#L523>`__

   **Allision entry:** ``compute/powered_model.py:569`` -- `run_powered_allision_model() <https://github.com/axelande/OMRAT/blob/main/compute/powered_model.py#L569>`__


Top-level call tree
===================

::

   run_powered_grounding_model(data)
     |
     +-- _pc_pair(pc, "grounding")                                       # (Cat II, Cat I) factors
     +-- powered_type_scale(data)                                        # 1/F per passenger / fast ferry row
     +-- model_projector(segment_data)                                   # ModelProjector, centred on the model
     +-- _collect_draught_set(traffic, squat_names)                      # draught (+ squat) per cell
     +-- _build_unique_depths(depths)
     +-- _precompute_bin_results(draughts, depths, data, proj)
     |     for each depth bin:
     |     _build_legs_and_obstacles(data, proj, mode="grounding", max_draft=bin)
     |       +-- _load_depth_obstacles -> _unique_ids                    # MultiPolygon -> <id>_<i>
     |       +-- _project_wkt_geom(geom, proj)                           # shapely.ops.transform
     |       +-- _build_leg_entry(...)                                   # lateral mixture + avg speed
     |             +-- _weighted_avg_speed_knots(traffic, turning_point(i))
     |     _run_all_computations(legs, all_obstacles, cat1_start_zone_frac)
     |       +-- per (leg, dir) with traffic:
     |             _leg_vectors(start, end)
     |             _compute_cat2_with_shadows(turn_pt, ext_dir, perp,    # vectorised ray cast
     |                                         mean, std, ai, V, obstacles, components)
     |               +-- _ray_grid -> mixture_rays(components)           # 1000-3000 rays
     |               +-- _build_hit_and_edge_matrices(...)               # (n_rays x n_edges) crossings
     |               |     +-- _extract_edges_local(obs_geom, turn_pt, u, u_perp)
     |               +-- _accumulate_obs_hits(...)                       # first hit per ray, per edge
     |               +-- _build_summaries(...)
     |             _compute_cat1_in_lane(origin, ext_dir, perp, mean, std, L,
     |                                   obstacles, start_zone_frac, components)
     +-- _sum_grounding_contribs(...)
     |     for each traffic cell x direction:
     |       lookup comps by depth bin; _iter_hit_probs(comp, ai * V)
     |       Cat II: cat2_probability = sum_r m_r * exp(-d_r/(ai*V));  Cat I: mass
     |       _ContribAccumulator.add(...)                                # per obstacle / edge / cell / category
     +-- _finalize_grounding(...)                                        # report + result layer

::

   run_powered_allision_model(data)
     |
     +-- model_projector + _build_legs_and_obstacles(mode="allision")    # max_draft=0 flags objects only
     +-- _run_all_computations(legs, all_obstacles, cat1_start_zone_frac) # same as grounding
     +-- _sum_allision_contribs(...)                                     # for each comp x traffic cell:
     |     skip when known ship_height < obstacle_height (bridge clearance)
     |     sum * pc_allision / pc_allision_cat1
     +-- _finalize_allision(...)


Shared inputs
=============

Both phases consume ``data``:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - Use
   * - ``segment_data[leg]['Start_Point'] / 'End_Point'``
     - Leg endpoints in lon/lat.  The centre of their bounding box (all
       legs) is the projection centre.
   * - ``segment_data[leg]['ai1'] / 'ai2'``
     - Position check interval (seconds, default 180) per direction.
       Plugged into the Cat II exponential:
       :math:`\exp(-d / (a_i V))`; ``ai <= 0`` gives no Cat II.
   * - ``segment_data[leg]['weight{d}_{k}'] / 'mean{d}_{k}' / 'std{d}_{k}'``,
       ``'u_min{d}' / 'u_max{d}' / 'u_p{d}'``
     - Per-direction lateral distribution: the whole mixture (normals
       1-3 with their weights plus the uniform part, weights normalised,
       ``geometries.distribution_curves.mixture``) is sampled by the rays
       (:ref:`ray-sampling`).  ``mean{d}_1`` / ``std{d}_1`` alone are the
       fallback for a leg without weights and draw the band in the View.
   * - ``traffic_data[leg][dir]['Frequency (ships/year)'] / 'Speed (knots)'``
     - Per-cell frequency and speed.  Read **after**
       :func:`compute.traffic_pipeline.prepare_traffic_for_run` has
       filled missing speeds and draughts, applied the per-cell
       ``Scaling (%)`` multiplier (see the :ref:`reference-data-format`
       chapter), moved the traffic of suppressed legs and applied the
       speed restriction zones.  Cat II reads the speed at the turning
       point (``speed_matrix(dir_data, turning_point(dir_idx))``).
   * - ``traffic_data[leg][dir]['Draught (meters)']`` (grounding only)
     - Selects the depth bin (draught + squat when
       ``pc['squat_powered']`` is on).
   * - ``traffic_data[leg][dir]['Ship heights (meters)']`` (allision
       only)
     - Filters out ships low enough to pass under the structure; 0 or
       ``inf`` is unknown and gets no clearance.
   * - ``objects`` / ``depths``
     - Obstacle geometries (WKT) + their ``height`` / ``depth``
       attribute.
   * - ``ship_categories['types']``
     - Ship type names: the passenger ship / fast ferry rows and the
       squat block coefficient.
   * - ``pc.grounding`` / ``pc.allision``,
       ``pc.grounding_cat1`` / ``pc.allision_cat1``
     - Cat II and Cat I causation factors.
       :math:`N = P_c \cdot Q \cdot \ldots`.
   * - ``pc.cat1_skip_start`` / ``pc.cat1_skip_start_pct``
     - Cat I start zone (default on, 10 %; :ref:`cat1-start-zone`).
   * - ``pc.squat_powered``
     - Squat added to the draught (grounding only, default on).
   * - ``pc.passenger_reduction`` / ``pc.fast_ferry_reduction``
     - Reduction factor :math:`F` (default 20): the frequency of those
       ship types is divided by :math:`F`.


Projection: :class:`ModelProjector`
===================================

The powered-model geometry is all done in one local metric frame
centred on the **model**: :func:`model_projector(segment_data)` builds a
:class:`ModelProjector` about the centre of the bounding box of every
leg's end points.  It is a spherical transverse Mercator about that
central meridian, scaled with the WGS84 radii of curvature at the
centre latitude (the prime-vertical radius :math:`N_0` across, the
meridian radius :math:`M_0` along), so distances across a sea area the
size of the Kattegat are right within 0.05 % and angles within 0.1
degree.  Pure numpy, no pyproj.

.. container:: source-code-ref pipeline

   **Source:** ``geometries/model_projection.py:38`` -- `ModelProjector <https://github.com/axelande/OMRAT/blob/main/geometries/model_projection.py#L38>`__

.. code-block:: python

   class ModelProjector:
       def transform(self, lon, lat):
           lam = np.radians(lon) - self._lam0
           phi = np.radians(lat)
           x = self.n0 * np.arctanh(np.cos(phi) * np.sin(lam))
           y = self.m0 * (np.arctan2(np.tan(phi), np.cos(lam)) - self._phi0)
           return x, y

Used by :func:`_build_legs_and_obstacles` to project legs and every
obstacle once.  Both powered models and the **View** use the same
projector.  Until 2026-11-09 the powered models used the equirectangular
:class:`SimpleProjector` around the *first* leg's start point (one
:math:`\cos\varphi_0` for the whole model), so east-west distances were
a few percent off away from that leg and the result depended on the
order of the route table.  :class:`SimpleProjector` remains for other,
non-QGIS uses.


:meth:`run_powered_grounding_model`
====================================

.. container:: source-code-ref pipeline

   **Source:** ``compute/powered_model.py:523`` -- `run_powered_grounding_model() <https://github.com/axelande/OMRAT/blob/main/compute/powered_model.py#L523>`__

Flow:

1. Short-circuit if traffic / segments / depths are empty.
2. Read the Cat II / Cat I causation factors (``_pc_pair``) and the
   passenger ship / fast ferry scale (``powered_type_scale``: ``1/F``
   per ship-type row, ``None`` when both factors are 1).
3. Build the :class:`ModelProjector` with ``model_projector(segment_data)``.
4. ``_collect_draught_set``: every unique positive, finite ``draught``
   across traffic, plus the cell's squat at its leg speed when
   ``pc['squat_powered']`` is on (``compute/squat.py``).  Defaults to
   ``{5.0}`` if no traffic has draught set.
5. **Depth binning** (``_precompute_bin_results``).  For each unique
   draught, find the largest ``depth_value`` not above it
   (``_depth_bin_key``).  Two ships whose draughts fall in the same bin
   "see" the same obstacle set, so we avoid recomputing the shadow
   geometry.  Bins are sorted so the progress bar advances
   monotonically; ``None`` is included if some draughts are below every
   depth polygon (no obstacles).  Per bin:

   * ``_build_legs_and_obstacles(data, proj, mode='grounding', max_draft=bin)``
     - returns ``(legs, all_obstacles, depth_geoms, ...)``.  Only
     depths :math:`\le` ``max_draft`` become obstacles.
   * ``bin_results[bin] = _run_all_computations(legs, all_obstacles,
     cat1_start_zone_frac=start_zone_fraction(pc))``
     - per ``(leg, direction)`` shadow-aware ray cast.
   * Progress reported via the progress callback
     (``"Powered grounding: obstacle bin i/n"``).

6. Iterate the traffic matrix (``_sum_grounding_contribs``):

   .. code-block:: python

      for leg_key, leg_dirs in traffic_data.items():
          for dir_idx, (dir_key, dir_data) in enumerate(leg_dirs.items()):
              ai = ai_per_dir[min(dir_idx, 1)]
              spd_arr = speed_matrix(dir_data, turning_point(dir_idx))
              for loa_i, freq_row in enumerate(dir_data['Frequency (ships/year)']):   # row = ship type
                  for type_j, q in enumerate(freq_row):                                # column = length class
                      if q <= 0: continue
                      q *= scale[loa_i]                     # 1/F for passenger ships / fast ferries
                      draught = ... + squat                 # per cell, 5 m if missing
                      speed_ms = ...                        # per cell, 10 kn if missing
                      for comp in bin_results[_depth_bin_key(draught)]:
                          if (comp['seg_id'], comp['dir_idx']) != (leg_key, dir_idx): continue
                          for category, obs_key, p_hit, shares in _iter_hit_probs(comp, ai * speed_ms):
                              acc.add(category, pc[category] * q * p_hit, ...)

7. ``_finalize_grounding``: store ``powered_grounding_report``
   (``acc.report``, plus ``cat1_start_zone``, ``ray_sampling`` and
   ``squat``), build the result layer, write
   ``LEPPoweredGrounding.setText(f"{total:.3e}")`` and return
   ``total``.


:meth:`run_powered_allision_model`
===================================

.. container:: source-code-ref pipeline

   **Source:** ``compute/powered_model.py:569`` -- `run_powered_allision_model() <https://github.com/axelande/OMRAT/blob/main/compute/powered_model.py#L569>`__

Structurally identical to grounding but:

* **No depth binning and no squat.**  The obstacle set is fixed
  (``objects``), so :func:`_run_all_computations` is called exactly
  once.
* **Height filter.**  For each obstacle summary, the inner loop
  (``_sum_allision_contribs``) skips ship cells whose known
  ``ship_height < obj_height``: those pass under the structure (bridge
  case).  A height of 0 or ``inf`` is unknown (``known_height``) and
  gets no clearance.
* Uses ``pc.allision`` (Cat II) and ``pc.allision_cat1`` (Cat I) as the
  causation factors.

Everything else (projector build, leg + obstacle build, ray cast,
cell iteration) is the same pattern as grounding.


:func:`_build_legs_and_obstacles`
=================================

.. container:: source-code-ref pipeline

   **Source:** ``geometries/get_powered_overlap.py:692`` -- `_build_legs_and_obstacles() <https://github.com/axelande/OMRAT/blob/main/geometries/get_powered_overlap.py#L692>`__

Purpose: given ``data`` + projector + mode (+ ``max_draft``), produce
all the geometry the ray cast needs in one local-frame dict.

Returns
``(legs, all_obstacles, depth_geoms, depth_geoms_deep, object_geoms)``:

* ``legs[seg_id]`` is a dict with:

  ``{'start', 'end', 'name', 'start_wkt', 'end_wkt',
  'dirs': [{'name', 'speed_kn', 'speed_ms', 'ai', 'mean', 'std',
  'components'}, ...]}``.  ``start`` / ``end`` are numpy ``(x, y)`` in
  metres; ``components`` is the direction's lateral mixture
  (``distribution_curves.mixture``; one normal from ``mean`` / ``std``
  when the leg has no weights).
* ``all_obstacles`` is a list of ``(obstacle_dict, kind)`` where
  ``kind`` is ``'depth'`` or ``'object'``.  Obstacle dicts carry
  ``'id'``, ``'geom'`` (shapely Polygon in local frame), and either
  ``'depth'`` or ``'height'``.  A MultiPolygon is split into one
  obstacle per part (``'<id>_<i>'``) and ``_unique_ids`` gives a
  repeated id ``#2``, ``#3``, ...: the caster adds hits up per id, so
  two polygons sharing one would be a single obstacle with colliding
  edge numbers.
* ``depth_geoms`` / ``depth_geoms_deep`` split the depths at
  ``max_draft``: only shallow ones become obstacles; the "deep" list
  is used by the visualiser to colour the safe-water background.
* ``object_geoms`` is the projected object list; used by the
  visualiser to draw structures separately from ray-hit summaries.

Per-direction speed is computed by
:func:`_weighted_avg_speed_knots(traffic_dir, turning_point(i))` which
sums ``freq * speed`` over every cell with a positive, finite frequency
and speed -- the speed at the direction's turning point -- and divides
by the total frequency.  It only decides whether the direction has
traffic (``speed_ms > 0``) and feeds the View; the models use each
cell's own speed.


:func:`_run_all_computations`
=============================

.. container:: source-code-ref pipeline

   **Source:** ``geometries/get_powered_overlap.py:718`` -- `_run_all_computations() <https://github.com/axelande/OMRAT/blob/main/geometries/get_powered_overlap.py#L718>`__

Loops every ``(seg_id, dir_idx)`` with non-zero speed and returns a
list of ``computation`` dicts:

.. code-block:: python

   {
       'seg_id', 'leg', 'dir_idx', 'dir_info',
       'turn_pt', 'ext_dir', 'perp',
       'summaries': {(kind, obs_id): {'mass', 'mean_dist', 'p_integral',
                                       'p_approx', 'n_rays',
                                       'ray_offsets', 'ray_dists',
                                       'ray_masses', 'obs', 'kind',
                                       'edges': {edge_idx: {'mass', 'mean_dist',
                                                            'n_rays', 'ray_masses',
                                                            'ray_dists'}}}},
       'ray_data', 'offsets', 'pdf_vals',
       'start', 'end',
       'cat1': {'origin', 'along_dir', 'length', 'start_zone_m',
                'summaries': {(kind, obs_id): {...same keys...}},
                'ray_data'},
   }

One entry per ``(leg, dir)`` that has obstacle hits in *either*
category.  ``summaries`` / ``ray_data`` are the Cat II result (rays
from the turning point onwards); the ``cat1`` block holds the Cat I
result (rays along the leg itself).

Setup per ``(leg, dir)``:

* ``u, n, L = _leg_vectors(start, end)`` -- unit leg direction, unit
  perpendicular, leg length.
* ``origin = start if dir_idx == 0 else end`` -- the waypoint the
  ships come from; Cat I rays start here.
* ``turn_pt = end if dir_idx == 0 else start`` -- ships fail to turn
  at the downstream waypoint, so the Cat II ray origin is the turning
  point.
* ``ext_dir = u`` for direction 0, ``-u`` for direction 1 (shared by
  both categories).

Then ``_compute_cat2_with_shadows(turn_pt, ext_dir, n, d['mean'],
d['std'], d['ai'], d['speed_ms'], all_obstacles,
components=d['components'])`` and ``_compute_cat1_in_lane(origin,
ext_dir, n, d['mean'], d['std'], L, all_obstacles,
start_zone_frac=cat1_start_zone_frac, components=d['components'])`` do
the work.  ``cat1_start_zone_frac`` is
``compute.cat1_start_zone.start_zone_fraction(pc)`` (0 when the setting
is off); Cat II never uses it.


:func:`_compute_cat1_in_lane`
=============================

.. container:: source-code-ref pipeline

   **Source:** ``geometries/get_powered_overlap.py`` -- `_compute_cat1_in_lane() <https://github.com/axelande/OMRAT/blob/main/geometries/get_powered_overlap.py>`__

The Category-I counterpart of the function below.  It samples the same
rays (``_ray_grid``) and reuses :func:`_build_hit_and_edge_matrices`
and :func:`_accumulate_obs_hits` with three differences:

* The rays start at the *origin* waypoint and ``max_range`` is the
  **leg length**, so only obstacles between the two waypoints are
  hit.  Anything past the turning point is left to Cat II, which
  keeps the two categories disjoint.
* ``min_range = start_zone_frac * L``: an edge crossing inside the
  start zone does not count, ray by ray (see :ref:`cat1-start-zone`).
* :func:`_build_cat1_summaries` sets ``p_approx = p_integral = mass``:
  there is no ``exp(-d / (a_i V))`` factor because the ship is already
  on a collision course (Hansen eq. 4.15).  ``mean_dist`` is still
  recorded (distance from the origin waypoint) for the visualiser.

In :mod:`compute.powered_model`, :func:`_iter_hit_probs` yields
``('cat2', key, cat2_probability(s, recovery), edge_shares)`` -- the sum
``sum_r m_r exp(-d_r / recovery)`` over the obstacle's rays, skipped
when ``recovery = ai * V`` is not positive -- for the Cat II summaries
and ``('cat1', key, mass, edge_shares)`` for the Cat I ones.  The
caller multiplies by ``Q`` and by the category's causation factor
(``pc['grounding']`` / ``pc['allision']`` for Cat II,
``pc['grounding_cat1']`` / ``pc['allision_cat1']`` for Cat I) and the
``_ContribAccumulator`` sums the result per obstacle, leg-direction,
cell and category (``by_category``).  ``edge_shares``
(``_edge_shares``: each hit edge weighted like the obstacle, normalised)
splits it over the obstacle's boundary edges into
``by_obstacle_segment_legdir[obs]["seg_<k>"][dir_key]``, which the
result layer reads; the edges of an obstacle sum to its
``by_obstacle`` value.


:func:`_compute_cat2_with_shadows`
==================================

.. container:: source-code-ref pipeline

   **Source:** ``geometries/get_powered_overlap.py:466`` -- `_compute_cat2_with_shadows() <https://github.com/axelande/OMRAT/blob/main/geometries/get_powered_overlap.py#L466>`__

This is the **core** of the powered model.  It casts parallel rays
across the direction's whole lateral mixture (every component's
``mean +/- 6 sigma`` and the uniform bounds, ``compute/ray_sampling.py``,
:ref:`ray-sampling`) and for each ray keeps the **first obstacle** it
hits (that's what produces the shadowing: closer obstacles block
farther ones automatically).

The implementation is fully vectorised in numpy -- no per-ray Python
loop over shapely.

Steps:

1. Build the rays (``_ray_grid`` -> ``mixture_rays(components)``):
   evenly spaced offsets over ``mixture_span``, ``N_RAYS = 1000`` for
   one component and more for a wider mixture (the narrowest component
   keeps a 0.012 sigma spacing, at most ``MAX_RAYS = 3000``).
   ``masses[i] = pdf(offsets[i]) * dx`` (mixture density) is the share
   of ships on that ray.  Without components it falls back to
   ``ray_offsets(mean, sigma)`` of one normal.
2. :func:`_build_hit_and_edge_matrices` skips an obstacle whose bounding
   box lies wholly behind the origin or beyond ``max_range`` (50 km for
   Cat II), then calls
   :func:`_extract_edges_local(geom, turn_pt, ext_dir, perp)`.  This
   transforms the polygon's boundary into the local ``(along, lateral)``
   frame where every ray is the horizontal line ``y = offset_i`` and
   returns an ``(M, 2, 2)`` array of edges.
3. For each obstacle compute a ``(n_rays, n_edges)`` matrix of
   along-track crossings in **one** numpy expression:

   .. code-block:: python

      # Edge (y0,x0) -> (y1,x1); ray at y = ray_ys
      crosses = (ray_ys >= y_min) & (ray_ys < y_max) & (dy != 0)
      t = (ray_ys - y0) / dy
      along = x0 + t * (x1 - x0)
      valid = crosses & (along > min_range) & (along < max_range)
      along = where(valid, along, inf)
      best_edge = argmin(along, axis=1)            # nearest edge per ray
      hit_matrix[:, obs_idx] = along[arange(n_rays), best_edge]
      edge_matrix[:, obs_idx] = best_edge          # -1 for a miss

4. Per ray pick the first-hit obstacle:
   ``best_obs_idx = argmin(hit_matrix, axis=1)``,
   ``best_dists = hit_matrix[arange(n_rays), best_obs_idx]``.  Rays
   that miss every obstacle keep ``inf``.
5. Accumulate per-obstacle stats (``mass``, ``weighted_dist``,
   ``p_integral``, ``ray_offsets``, ``ray_dists``, ``ray_masses``, and
   the same per hit edge under ``edges``) by walking the rays once
   (``_accumulate_obs_hits``).  Everything is O(n_rays) Python work
   because the heavy numerical work already happened.
6. Each obstacle summary (``_build_summaries``):

   * ``mean_dist = weighted_dist / mass``
   * ``p_integral = sum_i m_i * exp(-dist_i / (ai * V))`` at the
     direction's mean speed; ``p_approx`` is the same value.  They rank
     the legs in the View when a run has no powered report.
   * The model does not use them: it calls
     :func:`cat2_probability(summary, ai * V)` with each traffic cell's
     own speed, which sums ``ray_masses`` x ``exp(-ray_dists / (ai V))``
     ray by ray.  Until 2026-11-09 it used
     ``mass * exp(-mean_dist / (ai V))``, which understates the sum
     because ``exp`` is convex.

This replaces an earlier ray-by-ray shapely loop that dominated
end-to-end runtime; the vectorised form is ~74 x faster on the same
500-ray x many-obstacle problem.


:func:`_extract_edges_local`
============================

.. container:: source-code-ref pipeline

   **Source:** ``geometries/get_powered_overlap.py:229`` -- `_extract_edges_local() <https://github.com/axelande/OMRAT/blob/main/geometries/get_powered_overlap.py#L229>`__

Helper used by :func:`_build_hit_and_edge_matrices` (both categories).
The edges come from :func:`geometries.boundary_edges.boundary_edges`
(rings oriented counter-clockwise, exterior then holes, zero-length
edges dropped) -- the same numbering the powered result layers use, so
edge ``k`` of the caster is ``seg_<k>`` of the layer.  It subtracts
``turn_pt`` and projects onto ``along_dir`` and ``perp_dir`` (both unit
vectors in world frame) using two matrix multiplications, producing an
``(M, 2, 2)`` edge array in the local frame.

Returns ``None`` when there are no edges (a Point / MultiPoint -- a
zero-width ray can never hit a point -- or an empty geometry).


Output: the traffic accumulation loop
=====================================

After :func:`_run_all_computations` returns, the mixin method walks
``traffic_data`` to apply per-cell frequency and causation factors.
Each ``computation['summaries']`` entry holds the rays that hit **one
obstacle** in **one (leg, direction)**; the traffic loop multiplies the
cell's frequency ``q`` (``/ F`` for passenger ships and fast ferries)
by :math:`P_c \sum_r m_r \exp(-d_r/(a_i V))` for Cat II or
:math:`P_c \cdot \mathrm{mass}` for Cat I and adds it to the running
total.

Only three filters can exclude a contribution:

* Grounding: the depth polygon's depth must be :math:`\le` ship's
  draught (+ squat).  This is implicitly enforced by the depth-bin
  selection (``max_draft`` passed to :func:`_build_legs_and_obstacles`).
* Allision: ``ship_height >= obj_height`` when the ship height is known.
  Explicit check in the traffic loop; lets low ships pass under tall
  structures (e.g., a tug under a high-clearance bridge).
* Cat II only: ``ai * V > 0`` (``ai <= 0`` switches Cat II off for that
  leg direction).

The total is written directly to
``self.p.main_widget.LEPPoweredGrounding`` /
``LEPPoweredAllision`` and returned as a float; the report
(``powered_grounding_report`` / ``powered_allision_report``: ``totals``
with the ``cat1`` / ``cat2`` split, ``by_obstacle``, ``by_obstacle_leg``,
``by_obstacle_segment_legdir``, ``by_cell``, ``by_category``,
``cat1_start_zone``, ``ray_sampling``) and the result layer are kept on
the calculation.


Interactive visualiser
======================

The **View** button of the powered rows re-uses the same
``computations`` list to build an interactive matplotlib view.  The
same :func:`_run_all_computations` call -- on the same
``model_projector``, with the project's Cat I start zone and the run
snapshot's speed restriction zones applied
(``compute/visualization.py:_with_speed_restrictions``) -- produces the
ray summaries; the visualiser walks them to draw the overview (selected
leg, heading arrow and the outlines of the two local windows; legend
and how-to beside it) on top, and Cat I (the leg in its own frame) |
Cat II geometry over the Cat II probability
(:func:`cat2_probability`, ``sum_r m_r exp(-d_r / (ai V))`` = mass x
mean survival) below.  The local panels are true to scale:
``local_view`` sizes each window around the largest hits (the next ones
and the ray start / turning point join while the window stays compact)
and to the panel's shape after a layout pass, and the lateral
distribution is an inset at their left edge.  The (leg, direction) is
picked in the table only.

The table next to the figure comes from
``compute/powered_breakdown.py:build_sidebar_rows``: one row per
(leg, direction) with the route-table leg name, the annual frequency
(sum of ``by_obstacle_leg[obs]["<seg>:<dir>"]``) and its **% of
total**.  The frequencies come from ``<stem>.powered.json``, written
next to the run's GeoPackage by
``RunHistoryMixin._write_powered_report_sidecar`` and handed to the
dialog as ``calc.powered_view_reports``.  For a run saved before that
sidecar existed, allision is rebuilt exactly from the View's own ray
casts (``allision_report_from_computations``, the same accumulation as
``run_powered_allision_model``); grounding is ray cast once per draught
bin in the model, so it is not rebuilt and its columns show ``--``.

.. container:: source-code-ref pipeline

   **Visualiser:** ``geometries/get_powered_overlap.py:946`` -- `PoweredOverlapVisualizer <https://github.com/axelande/OMRAT/blob/main/geometries/get_powered_overlap.py#L946>`__


Function reference
==================

``compute/powered_model.py`` (:class:`PoweredModelMixin`)

* ``run_powered_grounding_model`` -- entry point, draught binning.
* ``run_powered_allision_model`` -- entry point, height filter.
* ``_collect_draught_set`` / ``_precompute_bin_results`` -- draughts
  (+ squat) and one ray cast per depth bin.
* ``_sum_grounding_contribs`` / ``_sum_allision_contribs`` -- the
  traffic loops; ``_finalize_grounding`` / ``_finalize_allision`` --
  report, result layer, line edit.
* ``_iter_hit_probs`` / ``_edge_shares`` -- per-obstacle hit
  probability per category and its split over the boundary edges.
* ``_ContribAccumulator`` -- per obstacle / leg-direction / edge / cell
  / category totals.
* ``powered_type_scale`` -- ``1/F`` per passenger ship / fast ferry row
  (``compute/passenger_reduction.py``).
* ``_pc_pair`` -- Cat II and Cat I causation factors (IWRAP defaults
  when missing).
* ``known_height`` -- a ship height the clearance check can use.

``geometries/model_projection.py`` (pure numpy; re-exported by
``get_powered_overlap``)

* ``ModelProjector`` / ``model_projector`` -- transverse Mercator centred
  on the model, used by both powered models, the View and the collision
  model's waypoint bearings.
* ``SimpleProjector`` -- equirectangular lon/lat -> metres.
* ``projected_bearing`` -- compass bearing between two projected points.

``geometries/get_powered_overlap.py``

* ``_parse_point`` -- parse "x y" -> ``(float, float)``.
* ``_project_wkt_geom`` -- projects a shapely geometry via
  :func:`shapely.ops.transform`.
* ``_weighted_avg_speed_knots`` -- per-cell freq-weighted speed.
* ``_leg_vectors`` -- leg unit vector, perpendicular, length.
* ``_build_legs_and_obstacles`` -- construct local-frame legs +
  obstacles.
* ``_load_depth_obstacles`` / ``_load_object_obstacles`` /
  ``_unique_ids`` -- projected obstacles, one id per polygon.
* ``_build_leg_entry`` -- one leg with its per-direction mixture,
  ``ai`` and mean speed.
* ``_run_all_computations`` -- per ``(leg, dir)`` ray-cast.
* ``_ray_grid`` -- ray offsets / masses (``compute/ray_sampling.py``).
* ``_compute_cat2_with_shadows`` -- vectorised shadow-aware ray
  cast (Cat II).
* ``_compute_cat1_in_lane`` -- the same along the leg (Cat I), with the
  start zone.
* ``_build_hit_and_edge_matrices`` / ``_build_hit_matrix`` -- distance
  and hit edge per (ray, obstacle).
* ``_accumulate_obs_hits`` -- first hit per ray, summed per obstacle
  and edge.
* ``cat2_probability`` -- ``sum_r m_r exp(-d_r / recovery)`` of one
  obstacle or edge summary.
* ``_extract_edges_local`` -- polygon boundaries in the leg's local
  frame.
* ``_get_all_coords`` / ``_ray_hit_distance`` -- kept for tests (not
  on the hot path anymore).
* ``_powered_na`` -- :math:`\exp(-d/(a_i V))` scalar helper.
* ``local_view`` -- true-scale window of a Cat I / Cat II panel,
  zoomed to the hits.
* ``PoweredOverlapVisualizer`` -- interactive matplotlib panel.

``compute/powered_breakdown.py`` (QGIS-free)

* ``build_sidebar_rows`` -- (leg, direction) rows: name, frequency,
  % of total, per-ship hit probability.
* ``leg_direction_frequencies`` / ``report_total`` -- per leg-direction
  frequencies and the total from a model report or a sidecar block.
* ``summarize_report`` / ``sidecar_payload`` -- the ``.powered.json``
  content.
* ``allision_report_from_computations`` -- rebuilds the allision report
  from the View's ray casts for runs without a sidecar.

``compute/basic_equations.py``

* ``powered_na`` -- same formula as ``_powered_na``, exposed as a
  module-level function for tests.

``compute/ray_sampling.py``, ``compute/cat1_start_zone.py``,
``compute/squat.py``, ``compute/passenger_reduction.py`` (QGIS-free)

* ``RAY_SPAN_SIGMA`` / ``N_RAYS`` / ``MAX_RAYS`` / ``mixture_rays`` --
  the ray sampling.
* ``start_zone_fraction`` / ``start_zone_settings`` -- the Cat I start
  zone.
* ``squat_on`` / ``squat_for_row`` -- squat added to the grounding
  draught.
* ``powered_scales`` -- ``1/F`` per ship-type row.
