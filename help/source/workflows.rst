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
   then **Stop**.
#. **Traffic Data tab** -- for every segment, select the direction
   and enter the Frequency / Speed / Draught / Ship heights matrices
   manually.  A single busy ship-type row is enough for a first pass.
   Leave **Scaling (%)** at 100 unless you are running a what-if.
#. **Depths tab** -- click **Load** to import a polygon layer, or use
   **Fetch GEBCO depth** with your OpenTopography key (the key goes
   in the **API Key** field on that same tab).
#. **Objects tab** -- **Load** a polygon layer with a ``height``
   attribute, or **Add manually** and draw polygons.
#. **Settings -> Drift settings** -- check the wind rose (default is
   uniform) and the repair distribution (default is lognormal with
   IWRAP defaults).  Adjust if you have local data.
#. Back on the **Routes tab**, scroll down to the lateral
   distribution panel and check the PDF plot for every segment.
   Adjust ``mean1_1`` / ``std1_1`` / ``weight1_1`` until the plot
   matches the expected track spread.
#. **Run Analysis tab** -- set **Name of the model** and
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
   It also re-derives the junction transition matrices
   (:ref:`junctions`) and runs the route-validation pass, which may
   prompt you about near-coincident waypoints and crossing legs.
#. Continue from step 3 of the "from scratch" recipe.

Tip: the AIS queries can be slow on large schemas.  The time per leg
is written to the QGIS log tab.  Any **Scaling (%)** values you set by
hand are preserved across the refresh -- only Frequency, Speed,
Draught and Height are overwritten.


Import an existing IWRAP XML
============================

If you have an existing IWRAP Mk2 project:

#. **File -> Import from IWRAP XML**.
#. Select the ``.xml`` file.

OMRAT parses the IWRAP schema and fills in segments, traffic,
obstacles, and causation factors.  Lateral distributions are
converted from IWRAP's (mean, std) per segment.

Cross-validate by running the risk and comparing against IWRAP.
Small differences (few percent) are expected -- OMRAT uses analytical
CDF integration where IWRAP uses Monte Carlo.


Export for use in IWRAP
========================

#. **File -> Export to IWRAP XML**.
#. Pick a location and filename.

The emitted XML is IWRAP Mk2 compatible.  Anything OMRAT computes
that has no IWRAP counterpart (e.g. the anchoring branch) is
dropped from the export.


Inspect the dominant obstacle on a risk number
===============================================

If a result is surprisingly high, you usually want to know which
single polygon is contributing the most.

#. Run the model.
#. Make sure the run you care about is selected in **Previous runs**
   (the newest is selected automatically).
#. Click **View** on the row of interest in the **Accident
   probabilities** table (e.g. *Drifting grounding*).
#. A dialog opens with per-obstacle contributions sorted by
   probability.

Alternatively, click **Add selected run results to map** and inspect
the **Grounding Results** layer -- its features are the boundary
edges of each depth contour, coloured by contribution (red =
highest).  Click any edge to open its attribute row, which carries a
``leg_<id>`` column per contributing leg.


Debug why a number looks wrong
===============================

Enable the debug trace:

#. Open the saved ``.omrat`` JSON in a text editor, or set it from
   the UI if your release exposes the flag: ``drift.debug_trace =
   true``.
#. Run the model again.
#. Open the auto-generated Markdown report -- its path is shown in
   the QGIS log tab (look for ``Drifting report written to:``).  The
   report now has a **Debug Obstacles** section listing every
   ``(leg, direction, obstacle)`` triple with its contribution,
   distance, probability hole, :math:`P_{NR}`, exposure factor, rose
   probability, base exposure, and frequency.

For the most forensic view, use the worked-example scripts in
``drifting/debug/level_1`` ... ``level_5`` which recompute a single
scenario end-to-end and print every intermediate variable.  These are
the same examples referenced from :ref:`drifting`.


Speed up a slow calculation
===========================

OMRAT already uses most of the reasonable optimisations (shadow
caching, batched CDF calls, vectorised ray-casting).  If a run takes
longer than you expect:

#. Look in the QGIS log tab.  A log line tells you the largest phase
   (shadow precompute, bucket memo, cascade, ...).
#. The **shadow precompute** phase is dominated by obstacle polygon
   complexity.  If you have a coastline polygon with tens of
   thousands of vertices, consider simplifying it in QGIS
   (``Vector -> Geometry Tools -> Simplify``) before loading.
#. The **analytical probability holes** phase scales with
   ``n_slices`` (default 100) x number of obstacles x number of legs
   x 8 directions.  You can reduce ``n_slices`` in
   :func:`~geometries.analytical_probability.compute_probability_holes_analytical`
   for a quick-and-dirty estimate, at the cost of some accuracy.
#. For very complex projects, run the **analysis** track instead
   (Drift Analysis tab).  It skips the risk integration entirely and
   just draws the corridors -- orders of magnitude cheaper.


Reproduce an old result
========================

OMRAT is deterministic **when using the analytical probability path**
(the default).  Given the same ``.omrat`` file and the same code
version, **Run model** produces bit-identical numbers on any machine.

The Monte Carlo path (``use_analytical=False``) is not deterministic
unless you set a random seed before invoking it.


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
   You get accident probabilities side by side, a list of which
   settings actually differ, and per-leg route distances.
   **Add both runs as map layers (red + blue)** puts both geometries
   on the canvas.

For a quick numeric comparison without leaving the Run Analysis tab,
select several rows in **Previous runs**: the result fields switch to
side-by-side form with a relative difference against the first
selected run.

If you want a raw diff instead, the snapshots are plain JSON -- open
both in ``git diff --no-index`` or VS Code's diff view.
