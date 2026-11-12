.. _code-flow:

==========================================
Code Flow: From "Run model" to Results
==========================================

This chapter is the **implementation-side companion** to the theory
chapters (:ref:`drifting`, :ref:`collisions`, :ref:`powered`).  Those
chapters explain *what* OMRAT calculates -- the formulas, the physical
meaning of each term, and the inputs the analyst supplies.  This chapter
explains *how* those formulas are executed inside the code: which
function the GUI invokes, what that function calls next, and how results
reach the run history and the result tables on the Run Analysis tab.

The goal is that a developer can read this chapter alongside the source
and confidently trace any line in a calculation result back to the code
that produced it.

.. contents:: In this chapter
   :local:
   :depth: 2


Reading map
===========

This chapter gives the **common frame** (GUI button, background task,
phase order, progress reporting).  Each accident type's detailed call
tree lives in its own chapter:

* :ref:`code-flow-drifting` -- ``run_drifting_model`` internals.
* :ref:`code-flow-collisions` -- ``run_ship_collision_model`` internals.
* :ref:`code-flow-powered` -- powered grounding + allision internals.

Pair each of these with its theory counterpart:

* :ref:`drifting` <-> :ref:`code-flow-drifting`
* :ref:`collisions` <-> :ref:`code-flow-collisions`
* :ref:`powered` <-> :ref:`code-flow-powered`


Entry point: the "Run model" button
====================================

The user triggers a run by clicking **Run model** on the main plugin
dialog.  The wiring between that button and the background calculation
is set up when the dialog is constructed:

.. container:: source-code-ref pipeline

   **Button wiring:** ``omrat.py:1338`` -- `self.main_widget.pbRunModel.clicked.connect(self.run_calculation) <https://github.com/axelande/OMRAT/blob/main/omrat.py#L1338>`__

When the button is clicked, :meth:`omrat.OMRAT.run_calculation` runs on
the Qt main thread:

1. It refuses to start a second run while one is running
   (``_current_task``).
2. :class:`~omrat_utils.gather_data.GatherData` pulls every field from
   the UI -- segments, traffic, depths, objects, drift params, causation
   factors, rose, repair distribution, junctions, speed restriction
   zones -- into one plain Python ``dict`` named ``data``
   (``get_all_for_save``).  This dict is the sole input the rest of the
   pipeline consumes.
3. Checks, each of which can stop the run with a message: a model name
   and an output folder are set (Run Analysis tab); at least one part is
   ticked in the **Calculation matrix** (``self.run_phases``,
   :ref:`Calculation matrix <calculation-matrix>`); the enabled speed restriction zones do
   not overlap and are valid (``_speed_zones_ok_for_run`` ->
   ``compute.speed_restriction.blocking_problems``).
4. Crossing check (``route_validation_ui.check_crossings_before_run``,
   only when ship-ship collisions are part of the run): legs that cross
   without a shared waypoint add no crossing collisions, so the user is
   offered **Split and run** / **Run anyway** / **Cancel**.  After a
   split ``data`` is gathered again.
5. A :class:`~compute.calculation_task.CalculationTask` is constructed
   with the existing :class:`~compute.run_calculations.Calculation`
   object, the ``data`` dict and the selected ``phases``.  A deep copy
   of ``data`` is kept as ``_run_input_snapshot`` (written as the run's
   ``.omrat`` snapshot) and the start time for the run's duration.
6. Three Qt signals are wired:

   - ``progress_updated`` -> :meth:`_on_calculation_progress` (logs lines
     to the QGIS message log),
   - ``calculation_finished`` -> :meth:`_on_calculation_finished` (saves
     the run to the history and shows it in the result tables),
   - ``calculation_failed`` -> :meth:`_on_calculation_failed` (surfaces
     the exception to the user).

7. The task is handed to QGIS's task manager
   (``QgsApplication.taskManager().addTask(task)``), which starts its
   ``run()`` method in a background thread so the UI stays responsive.

.. container:: source-code-ref pipeline

   **Orchestrator:** ``omrat.py:952`` -- `run_calculation() <https://github.com/axelande/OMRAT/blob/main/omrat.py#L952>`__


Background orchestration: :class:`~compute.calculation_task.CalculationTask`
============================================================================

``CalculationTask`` subclasses :class:`qgis.core.QgsTask` so QGIS can
schedule it and show progress in the task-manager tray.  Everything
inside its ``run()`` method executes on the background thread.

.. container:: source-code-ref pipeline

   **Class:** ``compute/calculation_task.py:14`` -- `CalculationTask <https://github.com/axelande/OMRAT/blob/main/compute/calculation_task.py#L14>`__

Progress plumbing
-----------------

After preparing the traffic and calling ``begin_run`` (see below) and
before invoking the risk models, ``run()`` installs a **progress
wrapper** on the ``Calculation`` object
(``CalculationTask._make_progress_wrapper``):

.. code-block:: python

   def progress_wrapper(completed, total, message) -> bool:
       if self.isCanceled():
           return False                   # signal the calc to stop
       if total > 0:
           self.setProgress(int(completed / total * 100))
       self._update_description(message)  # shown in QGIS task tray
       self.progress_updated.emit(completed, total, message)
       return True                        # continue

   self.calc.set_progress_callback(self._make_progress_wrapper())

Every risk model calls ``self._report_progress(phase, phase_progress,
message)`` at key milestones (see
``compute/run_calculations.py:_report_progress``).  That helper converts
``(phase, 0.0..1.0)`` into an overall 0--100 percentage using fixed
phase weights:

.. list-table:: Progress phase weights inside a single risk model
   :header-rows: 1

   * - Phase
     - Band
     - What it covers
   * - ``spatial``
     - 0 -- 40 %
     - Probability hole + min-distance pre-computation
   * - ``shadow``
     - 40 -- 60 %
     - Shadow polygons + per-obstacle edge geometry
   * - ``cascade``
     - 60 -- 90 %
     - Traffic cascade (per-ship lookups, cheap)
   * - ``layers``
     - 90 -- 100 %
     - Result-layer creation

.. container:: source-code-ref pipeline

   **Progress conversion:** ``compute/run_calculations.py:108`` -- `_report_progress() <https://github.com/axelande/OMRAT/blob/main/compute/run_calculations.py#L108>`__


The four phases, in order
-------------------------

Before the phases start, ``run()`` prepares its private copy of
``data`` with :func:`compute.traffic_pipeline.prepare_traffic_for_run`
(``CalculationTask._prepare_traffic``), in this order:

1. :func:`compute.traffic_gaps.fill_traffic_gaps` -- cells with ships
   but no speed / draught get a value (:ref:`traffic-gaps`);
2. :func:`compute.traffic_scaling.apply_traffic_scaling` -- the
   ``Scaling (%)`` matrices;
3. :func:`compute.traffic_redirect.apply_traffic_redirects` -- moves the
   traffic of suppressed legs onto their target legs and removes them
   from the segments, the traffic and the junction matrices (see
   :ref:`suppress-leg`);
4. :func:`compute.speed_restriction.apply_speed_restrictions` -- speeds
   in the speed restriction zones (:ref:`speed-restriction-zones`); the
   summary is kept as ``calc.speed_restriction_summary`` for the run
   report.

Every fill, move and zone effect is written to the QGIS log.

``CalculationTask.run()`` then executes the phases **sequentially** on
the same ``Calculation`` instance.  Each phase writes its results into
attributes of ``self.calc`` (it also sets hidden ``LEP*`` line-edits,
which are no longer shown).

.. list-table:: Phases invoked by :meth:`CalculationTask.run`
   :header-rows: 1
   :widths: 5 30 20 45

   * - #
     - Method on ``Calculation``
     - Source
     - Outputs written
   * - 1
     - :meth:`run_drifting_model`
     - ``compute/drifting_model.py``
     - ``drifting_allision_prob``, ``drifting_grounding_prob``,
       ``drifting_report``, ``allision_result_layer``,
       ``grounding_result_layer``, ``LEPDriftAllision.setText``,
       ``LEPDriftingGrounding.setText``
   * - 2
     - :meth:`run_ship_collision_model`
     - ``compute/ship_collision_model.py``
     - ``ship_collision_prob``, ``collision_report``,
       ``collision_line_layer``, ``collision_point_layer``,
       ``LEPHeadOnCollision``, ``LEPOvertakingCollision``,
       ``LEPCrossingCollision``, ``LEPMergingCollision``,
       ``LEPBendCollision``
   * - 3
     - :meth:`run_powered_grounding_model`
     - ``compute/powered_model.py``
     - ``powered_grounding_report``, ``powered_grounding_layer``,
       ``LEPPoweredGrounding``, return: total frequency
   * - 4
     - :meth:`run_powered_allision_model`
     - ``compute/powered_model.py``
     - ``powered_allision_report``, ``powered_allision_layer``,
       ``LEPPoweredAllision``, return: total frequency

``run()`` only runs the phases in ``CalculationTask.phases`` (the
**Calculation matrix** selection, ``compute/run_phases.py``; default all
four, always in the order above).  Before the first phase it calls
``Calculation.begin_run``, which resets every phase's report / layer
attributes to their not-calculated values (``PHASE_RESULT_ATTRS``) and
re-derives ``_last_depths_original`` from ``data``.  A skipped phase
therefore leaves a ``None`` report, which ``totals_from_calc`` records
as an empty history total -- never a zero, and never the previous
run's value.

After the last phase ``run()`` runs the oil-spill consequence step
(:func:`compute.consequence.compute_catastrophe_exceedance`) on
whichever reports the run produced and stores it as
``calc.consequence_result``; a failure there is logged and leaves
``None``.

Between every phase, ``run()`` checks ``self.isCanceled()`` so the user
can stop a long calculation.  If a phase raises an exception, ``run()``
stores the message on ``self.error_msg`` and returns ``False``; the
``finished()`` callback then emits ``calculation_failed`` on the main
thread.


Where the inputs come from
==========================

Inside every phase, one dict (``data``) acts as the single source of
truth.  Its keys map onto specific UI widgets / files:

.. list-table:: Top-level keys of the calculation ``data`` dict
   :header-rows: 1
   :widths: 22 78

   * - Key
     - What it holds
   * - ``segment_data``
     - Per-leg geometry, direction labels (``Dirs[0]`` = the drawn
       direction), lateral distributions (``mean``/``std``/``weight``
       per direction plus the uniform part), ``ai1``/``ai2`` (IWRAP
       position check interval, seconds), line length, suppression /
       redirect fields.  Keys are leg IDs.  Populated by the Routes tab.
   * - ``traffic_data``
     - Per-leg, per-direction matrices of frequency, speed, draught,
       beam, height and ``Scaling (%)``.  Rows = ship types; columns =
       LOA bins.  Populated by the traffic tab or AIS import.
   * - ``depths``
     - List of ``[id, depth_m, wkt_polygon]`` triples for every depth
       contour.  Populated by the depths tab.
   * - ``objects``
     - List of ``[id, height_m, wkt_polygon]`` triples for every
       structure (bridge pier, wind turbine foundation, ...).  Populated
       by the objects tab.
   * - ``drift``
     - Blackout rate ``drift_p``, per-type blackout overrides
       ``blackout_by_ship_type``, drift speed ``speed`` (knots), anchor
       probability ``anchor_p``, anchor-depth factor ``anchor_d``,
       repair distribution under ``repair`` (``use_lognormal``,
       ``std``/``loc``/``scale`` or ``func`` string), wind rose under
       ``rose``, drift start line ``start_from``.
   * - ``pc``
     - Causation factors per accident type (``headon``, ``overtaking``,
       ``crossing``, ``merging``, ``bend``, ``grounding`` /
       ``allision`` and their ``*_cat1`` counterparts,
       ``allision_drifting_rf``, ``grounding_drifting_rf``) plus the
       model settings edited in the same dialog
       (``min_encounter_angle_deg``, ``cat1_skip_start`` /
       ``cat1_skip_start_pct``, ``passenger_reduction`` /
       ``fast_ferry_reduction``, ``squat_drifting`` /
       ``squat_powered``).
   * - ``ship_categories``
     - LOA bin definitions and ship-type names, used by the
       ship-collision model for ship length (bin midpoint) and missing
       beams, and to label the traffic matrix rows.
   * - ``junctions``
     - Transition matrix per waypoint where legs meet; read by the
       ship-ship model for crossing / merging / bend (geometric defaults
       when missing).
   * - ``speed_restriction_zones``
     - Speed restriction zones, applied in the traffic preparation.
   * - ``consequence``
     - Oil on board, spill probabilities / fractions and catastrophe
       levels for the consequence step.

The :class:`~omrat_utils.gather_data.GatherData` helper reads each
widget and builds this dict in one shot:
``gd.get_all_for_save()``.  Every key is described in
:ref:`reference-data-format`.  The same dict format is what
:class:`~omrat_utils.storage.Storage` serialises to ``.omrat`` files, so
a loaded project can be handed to ``run_calculation`` without
conversion.

.. container:: source-code-ref pipeline

   **UI -> dict:** ``omrat_utils/gather_data.py`` -- `GatherData <https://github.com/axelande/OMRAT/blob/main/omrat_utils/gather_data.py>`__


The ``Calculation`` facade
==========================

:class:`~compute.run_calculations.Calculation` is an empty class that
pulls in five mixins at import time.  Each mixin holds one clearly
scoped piece of the pipeline:

.. code-block:: python

   class Calculation(
       DriftingModelMixin,
       ShipCollisionModelMixin,
       PoweredModelMixin,
       DriftingReportMixin,
       VisualizationMixin,
   ):
       """Main calculation facade -- composes all model mixins."""

.. container:: source-code-ref pipeline

   **Facade:** ``compute/run_calculations.py:45`` -- `Calculation <https://github.com/axelande/OMRAT/blob/main/compute/run_calculations.py#L45>`__

The mixins communicate only through attributes on ``self``, which means
each risk-model mixin can be tested in isolation with a mock parent.
See ``tests/test_cascade_minimal.py`` for the smallest possible
end-to-end drive.

.. list-table:: Mixin source files
   :header-rows: 1

   * - Mixin
     - Source
     - Covered in
   * - ``DriftingModelMixin``
     - ``compute/drifting_model.py``
     - :ref:`code-flow-drifting`
   * - ``ShipCollisionModelMixin``
     - ``compute/ship_collision_model.py``
     - :ref:`code-flow-collisions`
   * - ``PoweredModelMixin``
     - ``compute/powered_model.py``
     - :ref:`code-flow-powered`
   * - ``DriftingReportMixin``
     - ``compute/drifting_report.py``
     - Drifting section of the run's Markdown report; the whole report
       is assembled by ``omrat_utils/full_report.py`` when the run is
       saved
   * - ``VisualizationMixin``
     - ``compute/visualization.py``
     - The **View** dialogs (drifting / powered overlap, collision
       breakdown) opened from the result tables


Completion and UI fan-out
==========================

When the selected phases finish (or one fails), QGIS runs the task's
``finished(result)`` method on the main thread.  The main-thread handler
emits whichever of these signals applies:

- ``calculation_finished(self.calc)`` -> connected in ``omrat.py`` to
  :meth:`_on_calculation_finished`, which:

  1. Blanks the result rows of the parts the run skipped
     (``_clear_accident_results(skipped_totals(...))``).
  2. Calls :meth:`_auto_save_run` to persist the finished run (see
     "Run-history persistence" below).
  3. Calls :meth:`refresh_previous_runs_table` so the
     ``TWPreviousRuns`` table on the Run Analysis tab picks up the new
     entry, and :meth:`_select_latest_previous_run` to select it.  The
     selection fills ``TWAccidentResults`` and
     ``TWCatastropheResults``, which show saved runs only (the new run
     next to the baseline, :ref:`previous-runs-table`).

- ``calculation_failed(error_msg)`` -> connected to
  :meth:`_on_calculation_failed`, which logs the error and shows it in
  a message box.

A cancelled run emits neither signal.  The results therefore appear on
the Run Analysis tab only once the run has been saved.

Run-history persistence
-----------------------

:meth:`_auto_save_run` (``RunHistoryMixin``) writes:

1. in the output folder of the Run Analysis tab, the per-run
   GeoPackage ``<slug(name)>_<YYYYMMDD_HHMMSS>.gpkg``
   (``omrat_utils/run_persistence.write_run_results``);
2. one metadata row in the master history database
   ``omrat_history.sqlite`` in the user's app-data folder
   (:class:`omrat_utils.run_history.RunHistory`, ``save_run``, tagged
   with the project name and the OMRAT version);
3. next to the GeoPackage, the read-only input snapshot
   ``<stem>.omrat``, the combined Markdown report (``omrat_utils/full_report.py``) and the
   ``.collision.json`` / ``.drifting.json`` / ``.consequence.json`` /
   ``.powered.json`` sidecars.

See :ref:`reference-data-format` for the schema.  ``RunHistory`` is
plain ``sqlite3``, so the persistence layer is fully unit-testable
without a QGIS instance (``tests/test_run_history.py``); loading a run's
layers onto the map lives in ``omrat_utils/run_persistence.py``.

The table on the Run Analysis tab (``TWPreviousRuns``) is populated by
:meth:`refresh_previous_runs_table` and supports a right-click context
menu with **Add results to map** (one run), **Set project...**,
**Delete from history** and **Delete from history + remove .gpkg
file**.  **File -> Manage previous runs...** opens a dialog with every
run of every project.

.. container:: source-code-ref pipeline

   **Persistence:** ``omrat_utils/run_history.py`` -- `RunHistory <https://github.com/axelande/OMRAT/blob/main/omrat_utils/run_history.py>`__

.. container:: source-code-ref pipeline

   **Completion handler:** ``omrat.py:1080`` -- `_on_calculation_finished() <https://github.com/axelande/OMRAT/blob/main/omrat.py#L1080>`__
