.. _workflows:

==========
Workflows
==========

Short end-to-end recipes for common tasks.  Each one assumes you
already have OMRAT open and a project loaded.  If any step refers to
a tab or field you don't recognise, see :ref:`user_guide`.

.. contents:: Recipes
   :local:
   :depth: 1


Build a project from scratch (no AIS)
======================================

#. **Routes tab** -- click **Add**, click the waypoints on the map,
   then **Stop**.  Start each new route at an existing waypoint (the
   click snaps to it) where routes meet.
#. **Traffic Data tab** -- for every segment, select the direction
   and enter the Frequency / Speed / Draught / Ship heights matrices
   manually.  A single busy ship-type row is enough for a first pass.
   Leave **Scaling (%)** at 100 unless you are running a what-if.
#. **Depths tab** -- click **Load** to import a polygon layer, or use
   **Fetch GEBCO depth** with your OpenTopography key (the key goes
   in the **API Key** field on that same tab).
#. **Objects tab** -- **Load** a polygon layer with a ``height``
   attribute, or **Add manually**, draw the polygon and click **Save**.
#. **Settings -> Drift settings** -- check the wind rose (default is
   uniform) and the repair distribution (default is lognormal with
   IWRAP defaults).  Adjust if you have local data.
#. Back on the **Routes tab**, scroll down to the lateral
   distribution panel and check every segment's distribution (it is
   also drawn as a curve on the leg's tangent line on the map).  Adjust
   **Mean1** / **std 1** / **weight 1** (and the other components) per
   direction until it matches the expected track spread.
#. **Run Analysis tab** -- set **Name of the Run** and
   **File path**, then **Run model**.


Build a project from AIS
=========================

#. **Settings -> AIS connection settings** -- fill in the database
   parameters.  (**Settings -> Database setup wizard...** will create
   and populate the database if you don't have one; see
   :ref:`database-setup`.)
#. Digitise the route as above (**Routes** tab).
#. Click **Update all distributions** on the Routes tab.  This fires
   PostgreSQL queries against your AIS schema and populates the
   traffic matrix for *every* segment and both directions at once.
   It first runs the route-validation pass, which may prompt you about
   near-coincident waypoints and crossing legs, and afterwards
   re-derives the junction transition matrices (:ref:`junctions`); the
   matrix editor opens by itself when a junction joins three or more
   legs.
#. Open the **Traffic report...** (Traffic Data tab, or the button in
   the message after the update) and check its warnings
   (:ref:`traffic-report`).
#. Continue from step 3 of the "from scratch" recipe.

Tip: the AIS queries can be slow on large schemas.  The time per leg
is written to the QGIS log tab.  Any **Scaling (%)** values you set by
hand are preserved across the refresh -- the matrices (frequency, speed,
draught, heights, beam) and each direction's lateral distribution are
replaced, the latter by one normal distribution fitted to the AIS
crossings.  Legs with **AIS lock** ticked (for example legs that got
their traffic with **Copy traffic...**, :ref:`copy-traffic`) are
skipped.


Import an existing IWRAP XML
============================

If you have an existing IWRAP Mk2 project:

#. **File -> Import from IWRAP XML**.
#. Select the ``.xml`` file.

OMRAT parses the IWRAP schema and fills in segments, traffic,
obstacles, causation factors and drift settings.  Each direction's
lateral distribution is read from IWRAP's mixed (normal + uniform)
distribution, and IWRAP's leg-to-leg routing fractions become user
junction matrices.  IWRAP's 14 ship types are merged into OMRAT rows.

Cross-validate by running the risk and comparing against IWRAP.  The
two tools make different modelling choices in a few places (crossing
and bend rules, the Category I start zone, the reduction factor,
blackout classes, squat, ...); :ref:`iwrap-differences` lists them and
what to set before comparing.


Export for use in IWRAP
========================

#. **File -> Export to IWRAP XML**.
#. Pick a location and filename.

The emitted XML is IWRAP Mk2 compatible.  OMRAT settings with no
IWRAP counterpart -- speed restriction zones, the Category I start
zone, squat, the smallest encounter angle -- are not written; see
:ref:`iwrap-differences`.  The traffic scaling is written into the
frequencies, as the calculation uses them.  OMRAT's 21 ship
types are grouped into IWRAP's 14 and length classes above 350 m are
merged into IWRAP's last class; the export message lists such merges.

Suppressed legs (see :ref:`suppress-leg`) do not exist in IWRAP.  The
export leaves them out and adds their traffic to the target legs, the
same way the OMRAT calculation does.  A warning lists the suppressed
legs and every move (ships per year) before the file is written;
**Cancel** stops the export.  The OMRAT project itself is not changed.


Assess a wind farm that blocks an existing lane
===============================================

The AIS data shows today's traffic, but once the farm is built the ships
on the lane through it take a detour.

#. Build the project from AIS as above, with the lane through the farm
   area and the legs of the detour(s) around it.
#. **Routes tab -> Copy traffic...** -- where a route is split into
   sub-legs at crossings, copy the traffic of its cleanest leg onto the
   other sub-legs, at least on the detour that will receive the ships
   (:ref:`copy-traffic`).  The Suppress leg dialog recommends this the
   first time it opens.
#. **Run Analysis tab** -- run the model and tick **Baseline** on it in
   **Previous runs**.
#. Add the wind farm as a structure on the **Objects** tab.
#. **Routes tab -> Suppress leg...** -- pick the lane through the farm
   and give each direction its targets, following the two rules and the
   worked examples in :ref:`suppress-leg`: 100 % on every leg of a single
   detour, or a split such as 80 % / 20 % between two alternative
   routes.  If the lane is split into several legs, give only the leg
   with the best AIS sample the targets and tick the others under
   **Suppress together with this leg** (:ref:`suppress-route`).  The
   lane is drawn dashed.
#. Press **Traffic links** to check on the map that the arrows go where
   you expect (:ref:`traffic-links`).
#. Run the model again: the **Change (%)** column next to the baseline
   gives the change per accident type.  For the details, compare the
   two runs on the **Compare** tab (**Breakdown**, **Add difference
   layers to QGIS (B - A)**).  The difference shows where the risk
   moves to: the detour legs, their junctions and the farm's edges.

**Restore leg** in the same dialog brings the lane back without losing
the detour settings.


Inspect the dominant obstacle on a risk number
===============================================

If a result is surprisingly high, you usually want to know which
single polygon is contributing the most.

#. Run the model.
#. Make sure the run you care about is selected in **Previous runs**
   (the newest is selected automatically).
#. Click **View** on the row of interest in the **Accident
   probabilities** table (e.g. *Drifting grounding*).
#. A dialog opens.  For the drifting and powered rows, its table
   lists every leg and direction (route-table names) with its
   contribution and **% of total**, largest first; pick a row and the
   figure shows that leg with the obstacles it hits, named in the
   panel legends.  For the collision rows it is a per-leg (or
   per-leg-pair) table with the same **% of total** column.

Alternatively, click **Add selected run results to map** and inspect
the **Grounding Results** layer -- its features are the boundary
edges of each depth contour, coloured by contribution (red =
highest).  Click any edge to open its attribute row, which carries a
column per contributing leg, named as in the route table (e.g.
``LEG_1_2_b``).


Debug why a number looks wrong
===============================

Enable the debug trace:

#. Open the saved ``.omrat`` JSON in a text editor and add
   ``"debug_trace": true`` to the ``drift`` block (the UI has no
   switch for it).  Load the file again; pressing OK in **Settings ->
   Drift settings** afterwards rebuilds the block without the flag.
#. Run the model again.
#. Open ``<name>_<timestamp>.drifting.json`` next to the run's
   GeoPackage.  Its ``debug_obstacles`` block lists every
   ``(leg, direction, obstacle)`` triple with its contribution and the
   contribution-weighted distance, probability hole, :math:`P_{NR}`,
   exposure factor, rose probability, base exposure, and frequency.

For the most forensic view, use the worked-example scripts in
``drifting/debug/level_1`` ... ``level_5`` which recompute a single
scenario end-to-end and print every intermediate variable.  These are
the same examples referenced from :ref:`drifting`.


Speed up a slow calculation
===========================

OMRAT already uses most of the reasonable optimisations (shadow
caching, batched CDF calls, vectorised ray-casting).  If a run takes
longer than you expect:

#. Watch the OMRAT task in the QGIS task manager: the drifting model
   names its current phase (shadows, bucket memo, structure / depth
   probabilities, traffic cascade, ...), so you can see where the time
   goes.
#. The **shadow precompute** phase is dominated by obstacle polygon
   complexity.  If you have a coastline polygon with tens of
   thousands of vertices, consider simplifying it in QGIS
   (``Vector -> Geometry Tools -> Simplify``) before loading.
   Many fine depth classes far from the structures cost time too:
   **Simplify...** on the Depths tab keeps the shallow classes
   everywhere and the deep ones only near the structures
   (:ref:`simplify-depths`).
#. The **analytical probability holes** phase scales with
   ``n_slices`` (default 100) x number of obstacles x number of legs
   x 8 directions.  You can reduce ``n_slices`` in
   :func:`~geometries.analytical_probability.compute_probability_holes_analytical`
   for a quick-and-dirty estimate, at the cost of some accuracy.
#. If only some accident types changed, recalculate just those with
   **Calculation matrix...** next to Run model
   (:ref:`Calculation matrix <calculation-matrix>`).
#. For very complex projects, run the **analysis** track instead
   (Drift Analysis tab).  It skips the risk integration entirely and
   just draws the corridors -- orders of magnitude cheaper.


Reproduce an old result
========================

OMRAT is deterministic: the probability holes are always computed
analytically.  Given the same ``.omrat`` file and the same code
version, **Run model** produces the same numbers on any machine.  Every
run keeps its input snapshot ``<name>_<timestamp>.omrat``, and
**Previous runs** shows the OMRAT version each run was made with.


Run a calculation without the QGIS UI (headless)
=================================================

The risk calculation doesn't intrinsically depend on QGIS.  You can
load an ``.omrat`` file, build the calculation object, and run it
from a Python script.  See
the standalone scripts under ``examples/`` -- e.g.
``examples/run_collision_breakdown.py`` and
``examples/check_powered_repaired.py`` -- for minimal working
patterns.

The UI-free path is useful for batch runs (sweeping parameters over a
scenario tree) or for integrating OMRAT into a larger pipeline.


Compare two result sets
=========================

Use the **Compare** tab:

#. Run the first scenario.  Every run writes an input snapshot
   ``<name>_<timestamp>.omrat`` next to its GeoPackage.
#. Change whatever you want (e.g. reduce traffic on one leg) and run
   again.
#. On the **Compare** tab, browse to the two snapshots as
   **Run A (.omrat)** and **Run B (.omrat)**, then click **Compare**.
   The sub-tabs show the accident totals side by side (**Totals**),
   the settings and inputs that differ (**Settings & inputs**), the
   change per leg, waypoint and object (**Breakdown**), the change in
   a zone around the structures (**Zone**) and the per-leg route
   distances (**Leg lengths**).  **More info (collisions)...**
   splits the change of each ship-ship collision type over the inputs
   that caused it.
#. **Add both models to QGIS (grouped: A = red, B = blue)** puts both
   geometries on the canvas; **Add difference layers to QGIS (B - A)**
   maps where the accident probability changed.

For a quick numeric comparison without leaving the Run Analysis tab,
tick **Baseline** on one run in **Previous runs** and select the
others: the **Accident probabilities** table shows the baseline first
and a value and **Change (%)** column for every selected run.

If you want a raw diff instead, the snapshots are plain JSON -- open
both in ``git diff --no-index`` or VS Code's diff view.
