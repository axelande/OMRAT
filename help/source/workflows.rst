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

#. **Route tab** -- **Add Route**, click waypoints on the map,
   **Stop Route**.
#. **Traffic tab** -- for every segment, select direction and enter
   the Frequency / Speed / Draught / Height / Beam matrix manually.
   A single busy ship type row is enough for a first pass.
#. **Depths tab** -- load a shapefile or use **Get GEBCO Depths**
   with your OpenTopography key.
#. **Objects tab** -- load a shapefile with a ``height`` attribute or
   draw polygons manually.
#. **Settings -> Drift settings** -- check the wind rose (default is
   uniform) and the repair distribution (default is lognormal with
   IWRAP defaults).  Adjust if you have local data.
#. **Distributions tab** -- for every segment, check the PDF plot.
   Adjust ``mean1_1`` / ``std1_1`` / ``weight1_1`` until the plot
   matches the expected track spread.
#. **Results tab -> Run Model.**


Build a project from AIS
=========================

#. **Settings -> AIS connection settings** -- fill in the database
   parameters.
#. Digitise the route as above (**Route** tab).
#. Select each segment in turn and click **Update AIS**.  This fires
   a PostgreSQL query against your AIS schema; OMRAT populates the
   traffic matrix for both directions automatically.
#. Continue from step 3 of the "from scratch" recipe.

Tip: the AIS queries can be slow on large schemas.  The time per leg
is written to the QGIS log tab.


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
#. Click **View** next to the result field of interest (e.g. Drift
   Grounding).
#. A dialog opens with per-obstacle contributions sorted by
   probability.

Alternatively, inspect the **Drifting grounding results** layer on
the map -- it is coloured by contribution (red = highest).  Click
any polygon to open its attribute row with per-leg and per-direction
totals.


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
version, **Run Model** produces bit-identical numbers on any machine.

The Monte Carlo path (``use_analytical=False``) is not deterministic
unless you set a random seed before invoking it.


Run a calculation without the QGIS UI (headless)
=================================================

The risk calculation doesn't intrinsically depend on QGIS.  You can
load an ``.omrat`` file, build the calculation object, and run it
from a Python script.  See
``tests/diagnostics/profile_drifting_e2e.py`` for a minimal example
(it's also what was used to produce the end-to-end benchmark numbers
in :ref:`code-flow`).

The UI-free path is useful for batch runs (sweeping parameters over a
scenario tree) or for integrating OMRAT into a larger pipeline.


Compare two result sets
=========================

There is no built-in diff tool, but the pattern is:

#. Run the first scenario, **File -> Save** as ``scenario_a.omrat``.
#. Change whatever you want (e.g. reduce traffic on one leg).
#. Run again, **File -> Save** as ``scenario_b.omrat``.
#. Open both JSON files in a diff tool (``git diff --no-index``,
   VS Code's built-in diff, etc.).  Every result field is in the
   ``results`` section, and the full per-obstacle drifting report is
   in ``drifting_report``.
