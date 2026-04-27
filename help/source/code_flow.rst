.. _code-flow:

==========================================
Code Flow: From "Run Model" to Results
==========================================

This chapter is the **implementation-side companion** to the theory
chapters (:ref:`drifting`, :ref:`collisions`, :ref:`powered`).  Those
chapters explain *what* OMRAT calculates -- the formulas, the physical
meaning of each term, and the inputs the analyst supplies.  This chapter
explains *how* those formulas are executed inside the code: which
function the GUI invokes, what that function calls next, and how results
flow back to the result line-edits on the main dialog.

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


Entry point: the "Run Model" button
====================================

The user triggers a run by clicking **Run Model** on the main plugin
dialog.  The wiring between that button and the background calculation
is set up when the dialog is constructed:

.. container:: source-code-ref pipeline

   **Button wiring:** ``omrat.py:1152`` -- `self.main_widget.pbRunModel.clicked.connect(self.run_calculation) <https://github.com/axelande/OMRAT/blob/main/omrat.py#L1152>`__

When the button is clicked, :meth:`omrat.OMRAT.run_calculation` runs on
the Qt main thread:

1. :class:`~omrat_utils.gather_data.GatherData` pulls every field from
   the UI -- segments, traffic, depths, objects, drift params, causation
   factors, rose, repair distribution -- into one plain Python ``dict``
   named ``data``.  This dict is the sole input the rest of the pipeline
   consumes.
2. A :class:`~compute.calculation_task.CalculationTask` is constructed
   with the existing :class:`~compute.run_calculations.Calculation`
   object and the ``data`` dict.
3. Three Qt signals are wired:

   - ``progress_updated`` -> :meth:`_on_calculation_progress` (logs lines
     to the QGIS message log),
   - ``calculation_finished`` -> :meth:`_on_calculation_finished` (fans
     the results out to result-visualisation helpers),
   - ``calculation_failed`` -> :meth:`_on_calculation_failed` (surfaces
     the exception to the user).

4. The task is handed to QGIS's task manager
   (``QgsApplication.taskManager().addTask(task)``), which starts its
   ``run()`` method in a background thread so the UI stays responsive.

.. container:: source-code-ref pipeline

   **Orchestrator:** ``omrat.py:599`` -- `run_calculation() <https://github.com/axelande/OMRAT/blob/main/omrat.py#L599>`__


Background orchestration: :class:`~compute.calculation_task.CalculationTask`
============================================================================

``CalculationTask`` subclasses :class:`qgis.core.QgsTask` so QGIS can
schedule it and show progress in the task-manager tray.  Everything
inside its ``run()`` method executes on the background thread.

.. container:: source-code-ref pipeline

   **Class:** ``compute/calculation_task.py:12`` -- `CalculationTask <https://github.com/axelande/OMRAT/blob/main/compute/calculation_task.py#L12>`__

Progress plumbing
-----------------

Before invoking the four risk models, ``run()`` installs a
**progress wrapper** on the ``Calculation`` object:

.. code-block:: python

   def progress_wrapper(completed, total, message) -> bool:
       if self.isCanceled():
           return False                   # signal the calc to stop
       if total > 0:
           self.setProgress(int(completed / total * 100))
       self._update_description(message)  # shown in QGIS task tray
       self.progress_updated.emit(completed, total, message)
       return True                        # continue

   self.calc.set_progress_callback(progress_wrapper)

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

   **Progress conversion:** ``compute/run_calculations.py:80`` -- `_report_progress() <https://github.com/axelande/OMRAT/blob/main/compute/run_calculations.py#L80>`__


The four phases, in order
-------------------------

``CalculationTask.run()`` executes four phases **sequentially** on the
same ``Calculation`` instance.  Each phase writes its results into
attributes of ``self.calc`` and also pushes formatted numbers straight
into the result line-edits on the main dialog.

.. list-table:: Phases invoked by :meth:`CalculationTask.run`
   :header-rows: 1
   :widths: 5 30 20 45

   * - #
     - Method on ``Calculation``
     - Source
     - Outputs written
   * - 1
     - :meth:`run_drifting_model`
     - ``compute/drifting_model.py:1608``
     - ``drifting_allision_prob``, ``drifting_grounding_prob``,
       ``drifting_report``, ``allision_result_layer``,
       ``grounding_result_layer``, ``LEPDriftAllision.setText``,
       ``LEPDriftingGrounding.setText``
   * - 2
     - :meth:`run_ship_collision_model`
     - ``compute/ship_collision_model.py:526``
     - ``ship_collision_prob``, ``collision_report``,
       ``LEPHeadOnCollision``, ``LEPOvertakingCollision``,
       ``LEPCrossingCollision``, ``LEPMergingCollision``
   * - 3
     - :meth:`run_powered_grounding_model`
     - ``compute/powered_model.py:28``
     - ``LEPPoweredGrounding``, return: total frequency
   * - 4
     - :meth:`run_powered_allision_model`
     - ``compute/powered_model.py:204``
     - ``LEPPoweredAllision``, return: total frequency

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
     - Per-leg geometry, direction labels, lateral distributions
       (``mean``/``std``/``weight`` per direction), ``ai1``/``ai2`` (IWRAP
       position check interval, seconds), bend angle, line length.  Keys
       are leg IDs.  Populated by the legs/segments tab.
   * - ``traffic_data``
     - Per-leg, per-direction matrices of frequency, speed, draught,
       beam, height.  Rows = ship types; columns = LOA bins.  Populated
       by the traffic tab or AIS import.
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
       ``rose``.
   * - ``pc``
     - Causation factors per accident type (``grounding``, ``allision``,
       ``headon``, ``overtaking``, ``crossing``, ``bend``,
       ``allision_drifting_rf``, ``grounding_drifting_rf``).
   * - ``ship_categories``
     - LOA bin definitions and ship-type names, used by the
       ship-collision model to estimate ship beam from LOA and to label
       the traffic matrix rows.

The :class:`~omrat_utils.gather_data.GatherData` helper reads each
widget and builds this dict in one shot:
``gd.get_all_for_save()``.  The same dict format is what
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

   **Facade:** ``compute/run_calculations.py:44`` -- `Calculation <https://github.com/axelande/OMRAT/blob/main/compute/run_calculations.py#L44>`__

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
     - Report markdown generation (written by the drifting model)
   * - ``VisualizationMixin``
     - ``compute/visualization.py``
     - Map / plot helpers invoked after a run finishes


Completion and UI fan-out
==========================

When all four phases finish (or one fails), QGIS runs the task's
``finished(result)`` method on the main thread.  The main-thread handler
emits whichever of these signals applies:

- ``calculation_finished(self.calc)`` -> connected in ``omrat.py`` to
  :meth:`_on_calculation_finished`, which:

  1. Calls :meth:`_auto_save_run` to persist the finished run to the
     history GeoPackage (see "Run-history persistence" below).
  2. Calls :meth:`refresh_previous_runs_table` so the
     ``TWPreviousRuns`` table on the Results tab picks up the new
     entry.
  3. (via :class:`VisualizationMixin`) redraws the result layers,
     writes the drifting report Markdown file if a path is
     configured, and refreshes the overview panel.

- ``calculation_failed(error_msg)`` -> connected to
  :meth:`_on_calculation_failed`, which surfaces the error in the
  message log.

Because the line-edits on the main dialog were already set from inside
the background phases, the user typically sees the numerical results
before the ``finished`` signal fires.

Run-history persistence
-----------------------

:meth:`_auto_save_run` delegates to
:class:`omrat_utils.run_history.RunHistory`, which writes a single
GeoPackage at the user's app-data location (see
:ref:`reference-data-format` for the schema).  The ``RunHistory``
class is intentionally QGIS-soft: every method except
``load_run_layers`` is plain ``sqlite3``, so the persistence layer is
fully unit-testable without a QGIS instance
(``tests/test_run_history.py`` covers save / list / get / compare /
delete with 28 tests).

The table on the Results tab (``TWPreviousRuns``) is populated by
:meth:`refresh_previous_runs_table` and supports a right-click context
menu with **Load on map**, **Compare selected**, and **Delete**.
**File -> Manage previous runs...** is a shortcut to the same view.

.. container:: source-code-ref pipeline

   **Persistence:** ``omrat_utils/run_history.py`` -- `RunHistory <https://github.com/axelande/OMRAT/blob/main/omrat_utils/run_history.py>`__

.. container:: source-code-ref pipeline

   **Completion handler:** ``omrat.py:635`` -- `_on_calculation_finished() <https://github.com/axelande/OMRAT/blob/main/omrat.py#L635>`__
