.. _concepts:

=========
Concepts
=========

This chapter is a glossary of the terms used throughout the rest of
the documentation and the plugin UI.  Each entry has a short
definition and pointers to where the concept matters (which tab,
which tab field, which function).

Read this once, then use it as a lookup when you hit an unfamiliar
word.

.. contents:: Glossary
   :local:
   :depth: 1


Route, segment, leg
====================

A **route** is a polyline on the map representing a shipping lane.
OMRAT splits a route into **segments** (also called **legs**) at
vertices you digitise.  Each segment is a single straight line between
two waypoints.

Every segment has its own:

* Geometry (two endpoints in lon/lat).
* Traffic matrix (per direction).
* Lateral distribution (per direction).
* Width (display only).

**Where:** Route tab.  **Why it matters:** every accident type is
computed per segment and summed.


Direction
==========

Each segment has **two directions of travel** -- e.g. "North going"
and "South going" on a meridional leg, or "East going" / "West going"
on a parallel leg.  OMRAT auto-labels these based on the segment's
compass bearing when you digitise.

Ship-ship head-on collisions arise between a segment's two
directions.  Overtaking happens within one direction.  Powered
calculations consider each direction independently (they define
different "bend turning points").

**Where:** Route tab (``Dirs``), Traffic tab (per-direction matrix).


Traffic cell
=============

One entry in the traffic matrix: the number of ships of a given
**type** (row) in a given **LOA bin** (column) per year.  Each cell
also stores a representative speed, draught, height, and beam.

A cell with zero frequency contributes zero to every accident type and
is skipped in the inner loops.  There are typically 21 types x 15 LOA
bins = 315 cells per direction, of which 10-30 are non-zero for a
real project.

**Where:** Traffic tab.


Wind rose
==========

An 8-value probability distribution over the compass directions
(N, NE, E, SE, S, SW, W, NW) telling OMRAT how often the wind (and
thus the drift direction) blows from each direction.  Values must sum
to 1.

**Where:** Settings -> Drift settings.  **Why it matters:** the
drifting-model sum weights the 8-direction contributions by these
probabilities.


Drift corridor
===============

The swept polygon a drifting ship from a given segment in a given
direction could reach before being picked up by repair, anchoring, or
an obstacle.  Conceptually it is the union of:

* The segment's lateral band (the ship could start anywhere within
  the lateral distribution).
* That band translated by ``reach_distance`` in the drift direction,
  where ``reach_distance`` = drift speed * 99th-percentile repair
  time.

**Where:** Drift Analysis tab for visualisation.

.. figure:: _static/screenshots/ui_drift_corridor.png
   :width: 80%
   :alt: Drift corridors shown on the map for a single leg

   Drift corridors for a single leg, one per compass direction.


Obstacle shadow
================

For each obstacle in a given drift direction, the **shadow** is the
extruded polygon behind the obstacle in that direction.  A drifting
ship that enters the shadow has already grounded / collided on the
obstacle that casts the shadow, so the shadow blocks farther obstacles
from contributing to that ship's risk.

Shadows are why OMRAT doesn't double-count: if a ship grounds on a
near reef, the same ship can't also ground on a deep reef beyond.

**Where:** ``compute/drifting_model.py:_build_blocker_shadow``,
``geometries/drift/shadow.py:create_obstacle_shadow``.


Probability hole (:math:`h_X`)
================================

Given a leg, a drift direction, and an obstacle :math:`X`, the
**probability hole** is the probability that a drifting ship starting
uniformly along the leg (laterally distributed by the segment's PDF)
eventually reaches :math:`X`.  It is a pure geometric integral --
it does not yet include repair time.

OMRAT computes this with an **analytical cross-section CDF
integration** (default, fast) or with a Monte Carlo sampler (opt-in
via ``data['use_analytical'] = False``).

**Where:** ``geometries/analytical_probability.py:compute_probability_analytical``.


Repair-time distribution and :math:`P_{NR}`
=============================================

The probability that the crew has *not* repaired the engine after
drifting time :math:`t` is

.. math::

   P_{NR}(t) = 1 - F_\mathrm{repair}(t)

where :math:`F_\mathrm{repair}` is the CDF of a Lognormal, Weibull, or
Normal distribution (user-configurable).  OMRAT pattern-matches the
common shapes and uses :func:`scipy.special.ndtr` directly for a ~650x
speedup over the scipy frozen-distribution path.

**Where:** Settings -> Drift settings -> Repair.  Implemented in
``compute/basic_equations.py:get_not_repaired``.


Anchoring
==========

A drifting ship in water shallow enough (``depth < anchor_d *
draught``) **may** successfully anchor.  This is modelled as a
conditional probability ``anchor_p``.  Anchoring events are summed
separately from grounding/allision -- they represent *saved* ships.

**Where:** Settings -> Drift settings (``anchor_p``, ``anchor_d``).
Output: the drifting report's ``totals['anchoring']``.


Causation factor
=================

The probability that a geometric accident candidate becomes an actual
accident.  IWRAP derives these from historical data; OMRAT ships the
standard values:

.. list-table::
   :header-rows: 1

   * - Accident type
     - Default :math:`P_C`
   * - Head-on
     - :math:`4.9 \times 10^{-5}`
   * - Overtaking
     - :math:`1.1 \times 10^{-4}`
   * - Crossing
     - :math:`1.3 \times 10^{-4}`
   * - Bend
     - :math:`1.3 \times 10^{-4}`
   * - Powered grounding
     - :math:`1.6 \times 10^{-4}`
   * - Allision
     - :math:`1.9 \times 10^{-4}`
   * - Drifting
     - :math:`1.0` (no avoidance -- the ship is powerless)

**Where:** Settings -> Causation Factors.


Category I vs Category II (powered)
=====================================

IWRAP splits powered risk into two categories:

* **Category I** -- the ship under power navigates into an obstacle on
  the normal route (e.g. bad pilotage on a straight leg).  Not
  computed by OMRAT's current release; the hazard is assumed absorbed
  into the drifting path via the distribution tail.
* **Category II** -- the ship fails to turn at a bend / waypoint and
  continues straight into whatever is ahead.  This is what
  ``run_powered_grounding_model`` and ``run_powered_allision_model``
  compute.

**Why the distinction matters:** Category II uses an exponential
decay :math:`\exp(-d/(a_i V))` that models the chance the crew regains
control before hitting the obstacle.  Category I uses a different
exposure formula and currently falls outside OMRAT.


Lateral distribution
=====================

Per segment, per direction, a PDF describing where ships sit across
the leg.  In OMRAT this is a mixture of up to three Gaussian
components and one uniform component.  Practical projects mostly use
a single Gaussian (mean 0, sigma derived from AIS track spread).

**Where:** Distributions tab.  **Why:** drifting corridor width,
Cat II ray spread, ship-ship collision overlap.


Standard nautical compass convention
======================================

OMRAT uses **nautical bearings**: 0 deg = North, 90 deg = East, 180
deg = South, 270 deg = West, measured **clockwise** from north.  The
wind rose, segment bearings, and drift directions all use this
convention.

Internal math uses the standard **math** convention (0 deg = East,
counter-clockwise).  The conversion is
:math:`\theta_\mathrm{math} = (90^\circ - \theta_\mathrm{compass}) \bmod 360^\circ`,
implemented once in
``drifting/engine.py:compass_to_math_deg``.


The ``data`` dict
==================

Everything that flows from the UI into the calculation is packed into
a single Python dict named ``data``.  :ref:`reference-data-format`
documents every key.  The same dict is serialised to ``.omrat`` JSON
files.


Run Model vs Run Analysis
==========================

Two buttons, different things:

* **Run Model** (Results tab) -- the risk calculation.  Returns
  numbers.  This is what
  :class:`~compute.calculation_task.CalculationTask` orchestrates
  (:ref:`code-flow`).
* **Run Drift Analysis** (Drift Analysis tab) -- visual corridor
  generation only.  Does not return risk numbers.

Don't mix them up -- running the analysis doesn't produce the numbers
you see in the result line-edits.
