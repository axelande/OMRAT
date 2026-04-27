.. _code-flow-powered:

===================================================
Code Flow: Powered Grounding & Allision (Cat II)
===================================================

This chapter walks OMRAT's IWRAP **Category II** powered-grounding and
powered-allision calculations one function at a time, in the order the
calls fire when a user presses **Run Model**.  Pair it with
:ref:`powered`, which derives the Cat II formula
:math:`N_{II} = P_c Q \cdot \mathrm{mass} \cdot
\exp(-d_\mathrm{mean}/(a_i V))` and explains the shadow model.

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
  only depths shallower than the ship's draught are hazards.  One
  computation is run per **depth bin** -- a bin is the set of depths a
  draught-class "sees", so all draughts in the same bin share one
  shadow-aware result.
* **Allision** uses the object polygons and filters by ship height: if
  a ship is shorter than a structure it passes under (bridge case).

Both phases call :func:`geometries.get_powered_overlap._run_all_computations`
to produce the shadow-aware ray-cast summaries, then iterate traffic to
multiply by per-cell frequency.

.. container:: source-code-ref pipeline

   **Grounding entry:** ``compute/powered_model.py:28`` -- `run_powered_grounding_model() <https://github.com/axelande/OMRAT/blob/main/compute/powered_model.py#L28>`__

   **Allision entry:** ``compute/powered_model.py:204`` -- `run_powered_allision_model() <https://github.com/axelande/OMRAT/blob/main/compute/powered_model.py#L204>`__


Top-level call tree
===================

::

   run_powered_grounding_model(data)
     |
     +-- _parse_point(first_seg["Start_Point"])
     +-- SimpleProjector(lon0, lat0)                                     # equirect projection origin
     +-- for each ship-draught, pick a depth bin
     +-- for each bin:
     |     _build_legs_and_obstacles(data, proj, mode="grounding", max_draft=bin)
     |       +-- sw.loads(wkt_str) per depth polygon
     |       +-- _project_wkt_geom(geom, proj)                           # shapely.ops.transform
     |       +-- _weighted_avg_speed_knots(seg_traffic[d_name])          # per direction avg speed
     |     _run_all_computations(legs, all_obstacles)
     |       +-- per (leg, dir):
     |             _leg_vectors(start, end)
     |             _compute_cat2_with_shadows(turn_pt, ext_dir, perp,    # vectorised ray cast
     |                                         mean, std, ai, V, obstacles)
     |               +-- _extract_edges_local(obs_geom, origin, u, u_perp)
     |               +-- numpy broadcast: (n_rays x n_edges) crossings
     |               +-- per-obstacle argmin + summary dict
     +-- for each traffic cell x direction:
           lookup comps by draught-bin; sum per-obstacle mass * exp(-d_mean/(ai*V))

::

   run_powered_allision_model(data)
     |
     +-- SimpleProjector + _build_legs_and_obstacles(mode="allision")    # max_draft=0 flags objects only
     +-- _run_all_computations(legs, all_obstacles)                      # same as grounding
     +-- for each comp x traffic cell:
           filter by ship_height < obstacle_height (bridge clearance)
           sum * pc_allision


Shared inputs
=============

Both phases consume ``data``:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - Use
   * - ``segment_data[leg]['Start_Point'] / 'End_Point'``
     - Leg endpoints in lon/lat.  The first leg's ``Start_Point``
       defines the projector origin.
   * - ``segment_data[leg]['ai1'] / 'ai2'``
     - Position check interval (seconds) per direction.  Plugged into
       the Cat II exponential:
       :math:`\exp(-d / (a_i V))`.
   * - ``segment_data[leg]['mean1_1'] / 'std1_1' / 'mean2_1' / 'std2_1'``
     - Per-direction lateral distribution.  A single Gaussian is
       assumed (the first superposed component); multi-component
       distributions would need the analytical probability path (see
       :ref:`code-flow-drifting`).
   * - ``traffic_data[leg][dir]['Frequency (ships/year)'] / 'Speed (knots)'``
     - Per-cell frequency and speed; speed is averaged if the cell
       holds a list.
   * - ``traffic_data[leg][dir]['Draught (meters)']`` (grounding only)
     - Selects the depth bin.
   * - ``traffic_data[leg][dir]['Ship heights (meters)']`` (allision
       only)
     - Filters out ships short enough to pass under the structure.
   * - ``objects`` / ``depths``
     - Obstacle geometries (WKT) + their ``height`` / ``depth``
       attribute.
   * - ``pc.grounding`` / ``pc.allision``
     - Causation factors.  :math:`N_{II} = P_c \cdot Q \cdot \ldots`.


Projection: :class:`SimpleProjector`
====================================

The powered-model geometry is all done in a local **equirectangular**
frame centred on the first leg's start point.  This is cheaper than
UTM (no zone selection, no pyproj dependency) and fine for
per-leg-scale distances since the rays are only cast a few tens of
kilometres.

.. container:: source-code-ref pipeline

   **Source:** ``geometries/get_powered_overlap.py:51`` -- `SimpleProjector <https://github.com/axelande/OMRAT/blob/main/geometries/get_powered_overlap.py#L51>`__

.. code-block:: python

   class SimpleProjector:
       def __init__(self, lon_ref, lat_ref):
           self.mx = 111_320.0 * cos(radians(lat_ref))
           self.my = 110_540.0

       def transform(self, lon, lat):
           return ((lon - self.lon_ref) * self.mx,
                   (lat - self.lat_ref) * self.my)

Used by :func:`_build_legs_and_obstacles` to project legs and every
obstacle once.


:meth:`run_powered_grounding_model`
====================================

.. container:: source-code-ref pipeline

   **Source:** ``compute/powered_model.py:28`` -- `run_powered_grounding_model() <https://github.com/axelande/OMRAT/blob/main/compute/powered_model.py#L28>`__

Flow:

1. Short-circuit if traffic / segments / depths are empty.
2. Build a :class:`SimpleProjector` from the first leg's
   ``Start_Point``.
3. Collect every unique non-zero ``draught`` across traffic.  Defaults
   to ``{5.0}`` if no traffic has draught set.
4. **Depth binning.**  For each unique draught, find the largest
   ``depth_value`` below it (``_depth_bin_key``).  Two ships whose
   draughts fall in the same bin "see" the same obstacle set, so we
   avoid recomputing the shadow geometry.  Bins are sorted so the
   progress bar advances monotonically; ``None`` is included if some
   draughts are below every depth polygon (no obstacles).
5. Per bin:

   * ``_build_legs_and_obstacles(data, proj, mode='grounding', max_draft=bin)``
     - returns ``(legs, all_obstacles, depth_geoms, ...)``.  Only
     depths :math:`\le` ``max_draft`` become obstacles.
   * ``bin_results[bin] = _run_all_computations(legs, all_obstacles)``
     - per ``(leg, direction)`` shadow-aware ray cast.
   * Progress reported via the top-level progress callback with
     phase ``"grounding"``.

6. Iterate the traffic matrix:

   .. code-block:: python

      for leg_key, leg_dirs in traffic_data.items():
          for dir_idx, (dir_key, dir_data) in enumerate(leg_dirs.items()):
              ai_seconds = ai_per_dir[min(dir_idx, 1)]
              for loa_i, freq_row in enumerate(dir_data['Frequency (ships/year)']):
                  for type_j, q in enumerate(freq_row):
                      if q <= 0: continue
                      draught, speed = ...  # per-cell
                      comps = bin_results[_depth_bin_key(draught)]
                      for comp in comps:
                          if comp['seg_id'] != leg_key: continue
                          if comp['dir_idx'] != dir_idx: continue
                          for key, s in comp['summaries'].items():
                              mass, d_mean = s['mass'], s['mean_dist']
                              recovery = ai_seconds * speed_ms
                              total += pc_grounding * q * mass * exp(-d_mean / recovery)

7. Write ``LEPPoweredGrounding.setText(f"{total:.3e}")`` and return
   ``total``.


:meth:`run_powered_allision_model`
===================================

.. container:: source-code-ref pipeline

   **Source:** ``compute/powered_model.py:204`` -- `run_powered_allision_model() <https://github.com/axelande/OMRAT/blob/main/compute/powered_model.py#L204>`__

Structurally identical to grounding but:

* **No depth binning.**  The obstacle set is fixed (``objects``), so
  :func:`_run_all_computations` is called exactly once.
* **Height filter.**  For each obstacle summary, the inner loop
  skips ship cells whose ``ship_height < obj_height``: those pass
  under the structure (bridge case).
* Uses ``pc.allision`` as the causation factor.

Everything else (projector build, leg + obstacle build, ray cast,
cell iteration) is the same pattern as grounding.


:func:`_build_legs_and_obstacles`
=================================

.. container:: source-code-ref pipeline

   **Source:** ``geometries/get_powered_overlap.py:322`` -- `_build_legs_and_obstacles() <https://github.com/axelande/OMRAT/blob/main/geometries/get_powered_overlap.py#L322>`__

Purpose: given ``data`` + projector + mode (+ ``max_draft``), produce
all the geometry the ray cast needs in one local-frame dict.

Returns
``(legs, all_obstacles, depth_geoms, depth_geoms_deep, object_geoms)``:

* ``legs[seg_id]`` is a dict with:

  ``{'start', 'end', 'name', 'start_wkt', 'end_wkt',
  'dirs': [{'name', 'speed_kn', 'speed_ms', 'ai', 'mean', 'std'},
  ...]}``.  ``start`` / ``end`` are numpy ``(x, y)`` in metres.
* ``all_obstacles`` is a list of ``(obstacle_dict, kind)`` where
  ``kind`` is ``'depth'`` or ``'object'``.  Obstacle dicts carry
  ``'id'``, ``'geom'`` (shapely Polygon in local frame), and either
  ``'depth'`` or ``'height'``.
* ``depth_geoms`` / ``depth_geoms_deep`` split the depths at
  ``max_draft``: only shallow ones become obstacles; the "deep" list
  is used by the visualiser to colour the safe-water background.
* ``object_geoms`` is the projected object list; used by the
  visualiser to draw structures separately from ray-hit summaries.

Per-direction speed is computed by
:func:`_weighted_avg_speed_knots(traffic_dir)` which sums
``freq * speed`` over every cell with non-zero frequency and divides
by the total frequency.


:func:`_run_all_computations`
=============================

.. container:: source-code-ref pipeline

   **Source:** ``geometries/get_powered_overlap.py:424`` -- `_run_all_computations() <https://github.com/axelande/OMRAT/blob/main/geometries/get_powered_overlap.py#L424>`__

Loops every ``(seg_id, dir_idx)`` with non-zero speed and returns a
list of ``computation`` dicts:

.. code-block:: python

   {
       'seg_id', 'leg', 'dir_idx', 'dir_info',
       'turn_pt', 'ext_dir', 'perp',
       'summaries': {(kind, obs_id): {'mass', 'mean_dist', 'p_integral',
                                       'p_approx', 'n_rays',
                                       'ray_offsets', 'ray_dists',
                                       'obs', 'kind'}},
       'ray_data', 'offsets', 'pdf_vals',
       'start', 'end',
   }

One entry per ``(leg, dir)`` that has obstacle hits.

Setup per ``(leg, dir)``:

* ``u, n, L = _leg_vectors(start, end)`` -- unit leg direction, unit
  perpendicular, leg length.
* ``turn_pt = end if dir_idx == 0 else start`` -- ships fail to turn
  at the downstream waypoint, so the ray origin is the turning point.
* ``ext_dir = u`` for direction 0, ``-u`` for direction 1.

Then ``_compute_cat2_with_shadows(turn_pt, ext_dir, n, d['mean'],
d['std'], d['ai'], d['speed_ms'], all_obstacles)`` does the work.


:func:`_compute_cat2_with_shadows`
==================================

.. container:: source-code-ref pipeline

   **Source:** ``geometries/get_powered_overlap.py:205`` -- `_compute_cat2_with_shadows() <https://github.com/axelande/OMRAT/blob/main/geometries/get_powered_overlap.py#L205>`__

This is the **core** of the powered model.  It casts
``N_RAYS = 500`` parallel rays across the lateral distribution and for
each ray keeps the **first obstacle** it hits (that's what produces
the shadowing: closer obstacles block farther ones automatically).

The implementation is fully vectorised in numpy -- no per-ray Python
loop over shapely.

Steps:

1. Build the ray offsets: ``offsets = linspace(mean - 4 sigma, mean +
   4 sigma, 500)``.  ``masses[i] = pdf(offsets[i]) * dx`` is the mass
   of the lateral distribution attributable to that ray.
2. For each obstacle call
   :func:`_extract_edges_local(geom, turn_pt, ext_dir, perp)`.  This
   transforms the polygon's boundary into the local ``(along, lateral)``
   frame where every ray is the horizontal line ``y = offset_i`` and
   returns an ``(M, 2, 2)`` array of edges.
3. For each obstacle compute a ``(n_rays, n_edges)`` matrix of
   along-track crossings in **one** numpy expression:

   .. code-block:: python

      # Edge (y0,x0) -> (y1,x1); ray at y = ray_ys
      crosses = (ray_ys >= y_min) & (ray_ys <= y_max) & (dy != 0)
      t = (ray_ys - y0) / dy
      along = x0 + t * (x1 - x0)
      valid = crosses & (along > 0) & (along < MAX_RANGE)
      along = where(valid, along, inf)
      hit_matrix[:, obs_idx] = along.min(axis=1)   # nearest edge per ray

4. Per ray pick the first-hit obstacle:
   ``best_obs_idx = argmin(hit_matrix, axis=1)``,
   ``best_dists = hit_matrix[arange(n_rays), best_obs_idx]``.  Rays
   that miss every obstacle keep ``inf``.
5. Accumulate per-obstacle stats (``mass``, ``weighted_dist``,
   ``p_integral``, ``ray_offsets``, ``ray_dists``) by walking the 500
   rays once.  Everything is O(N_RAYS) Python work because the heavy
   numerical work already happened.
6. Each obstacle summary:

   * ``mean_dist = weighted_dist / mass``
   * ``p_approx = mass * _powered_na(mean_dist, ai, speed_ms)`` --
     the closed-form Cat II probability at the mean distance,
     equivalent for small dispersions.
   * ``p_integral = sum_i m_i * exp(-dist_i / (ai * V))`` -- the
     properly-integrated Cat II probability (used for tighter bounds
     / visualisation).

This replaces an earlier ray-by-ray shapely loop that dominated
end-to-end runtime; the vectorised form is ~74 x faster on the same
500-ray x many-obstacle problem.


:func:`_extract_edges_local`
============================

.. container:: source-code-ref pipeline

   **Source:** ``geometries/get_powered_overlap.py:121`` -- `_extract_edges_local() <https://github.com/axelande/OMRAT/blob/main/geometries/get_powered_overlap.py#L121>`__

Helper used by :func:`_compute_cat2_with_shadows`.  Walks the
geometry (Polygon, MultiPolygon, LineString, MultiLineString,
GeometryCollection) collecting the exterior + interior rings of every
component.  For every ring it subtracts ``turn_pt`` and projects onto
``along_dir`` and ``perp_dir`` (both unit vectors in world frame)
using two matrix multiplications, producing an ``(M, 2, 2)`` edge
array in the local frame.

Returns ``None`` for Point / MultiPoint geometries -- a zero-width ray
can never hit a point.  Returns ``None`` for empty geometries.


Output: the traffic accumulation loop
=====================================

After :func:`_run_all_computations` returns, the mixin method walks
``traffic_data`` to apply per-cell frequency and causation factors.
Each ``computation['summaries']`` entry gives the mass fraction and
mean distance for **one obstacle** in **one (leg, direction)**; the
traffic loop multiplies by the cell's frequency ``q`` and adds
``P_c \cdot q \cdot mass \cdot \exp(-d/(a_i V))`` to the running total.

Only two filters can exclude a contribution:

* Grounding: the depth polygon's depth must be :math:`\le` ship's
  draught.  This is implicitly enforced by the depth-bin selection
  (``max_draft`` passed to :func:`_build_legs_and_obstacles`).
* Allision: ``ship_height >= obj_height``.  Explicit check in the
  traffic loop; lets short ships pass under tall structures (e.g., a
  tug under a high-clearance bridge).

The result is written directly to
``self.p.main_widget.LEPPoweredGrounding`` /
``LEPPoweredAllision`` and returned as a float.


Interactive visualiser
======================

For "Show analysis" (previously called "Run analysis") the plugin
re-uses the same ``computations`` list to build an interactive
matplotlib view.  The same
:func:`_run_all_computations` call produces the ray summaries; the
visualiser walks them to draw the overview map, detail panel, and
waterfall breakdown.

.. container:: source-code-ref pipeline

   **Visualiser:** ``geometries/get_powered_overlap.py:473`` -- `PoweredOverlapVisualizer <https://github.com/axelande/OMRAT/blob/main/geometries/get_powered_overlap.py#L473>`__


Function reference
==================

``compute/powered_model.py`` (:class:`PoweredModelMixin`)

* ``run_powered_grounding_model`` -- entry point, draught binning.
* ``run_powered_allision_model`` -- entry point, height filter.

``geometries/get_powered_overlap.py``

* ``SimpleProjector`` -- equirectangular lon/lat -> metres.
* ``_parse_point`` -- parse "x y" -> ``(float, float)``.
* ``_project_wkt_geom`` -- projects a shapely geometry via
  :func:`shapely.ops.transform`.
* ``_weighted_avg_speed_knots`` -- per-cell freq-weighted speed.
* ``_leg_vectors`` -- leg unit vector, perpendicular, length.
* ``_build_legs_and_obstacles`` -- construct local-frame legs +
  obstacles.
* ``_run_all_computations`` -- per ``(leg, dir)`` ray-cast.
* ``_compute_cat2_with_shadows`` -- vectorised shadow-aware ray
  cast.
* ``_extract_edges_local`` -- polygon boundaries in the leg's local
  frame.
* ``_get_all_coords`` / ``_ray_hit_distance`` -- kept for the
  interactive visualiser and tests (not on the hot path anymore).
* ``_powered_na`` -- :math:`\exp(-d/(a_i V))` scalar helper.
* ``find_closest_computation_index`` -- maps a click to a
  computation (used by the visualiser).
* ``PoweredOverlapVisualizer`` -- interactive matplotlib panel.

``compute/basic_equations.py``

* ``powered_na`` -- same formula as ``_powered_na``, exposed as a
  module-level function for tests and the drifting cascade.
