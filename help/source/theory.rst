.. _theory:

=======================
Theory (what is calculated)
=======================

This is the **theory track** -- a short umbrella that tells you where
to find each accident type's derivation and the few global
conventions shared by all of them.  For the call-tree companion (the
"how" track), see :ref:`code-flow`.

.. contents:: In this chapter
   :local:
   :depth: 1


The IWRAP framework in one equation
====================================

Every accident frequency OMRAT reports has the form

.. math::

   F_\mathrm{accident} = N_A \cdot P_C

* :math:`N_A` is the **geometric candidate count** -- how often an
  accident *could* happen from geometry + traffic alone.
* :math:`P_C` is the **causation factor** -- the conditional
  probability that a candidate becomes an actual accident (the crew
  fails to avoid, the machinery fails, etc.).

The accident-type chapters derive :math:`N_A` from first principles
and give sources for :math:`P_C`:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Chapter
     - Covers
   * - :ref:`drifting`
     - Drifting grounding / allision / anchoring.  The largest
       chapter, with five worked examples in
       ``drifting/debug/level_1`` ... ``level_5``.
   * - :ref:`collisions`
     - Head-on, overtaking, crossing, bend collisions (Hansen eq.
       4.2-4.4, Pedersen).
   * - :ref:`powered`
     - IWRAP Category II powered grounding + allision
       (:math:`N_{II} = P_c Q m \exp(-d/(a_i V))`).


Default causation factors
==========================

OMRAT ships the IALA default table.  These are the values most
published studies use unless there's local data.

.. list-table:: Default causation factors (IALA / Fujii / Pedersen)
   :header-rows: 1
   :widths: 35 22 22 21

   * - Accident type
     - IALA default
     - Fujii (1974)
     - Notes
   * - Head-on collision
     - :math:`4.9 \times 10^{-5}`
     - :math:`4.9 \times 10^{-5}`
     - TSS present helps; higher in narrow lanes.
   * - Overtaking collision
     - :math:`1.1 \times 10^{-4}`
     - :math:`1.1 \times 10^{-4}`
     -
   * - Crossing collision
     - :math:`1.3 \times 10^{-4}`
     - :math:`1.2 \times 10^{-4}`
     - Pedersen value.
   * - Bend collision
     - :math:`1.3 \times 10^{-4}`
     - --
     - Pedersen value.
   * - Powered grounding
     - :math:`1.6 \times 10^{-4}`
     - :math:`1.6 \times 10^{-4}`
     -
   * - Allision (structure)
     - :math:`1.9 \times 10^{-4}`
     - :math:`1.9 \times 10^{-4}`
     -
   * - Drifting
     - :math:`1.0`
     - --
     - No avoidance -- the ship is powerless.

Local adjustment factors are typically applied on top:

* **Ferry / passenger vessels** -- divide by 20 (two navigators,
  well-known route).
* **Pilot on board** -- divide by 3 (COWIconsult).
* **Poor visibility (3-10 %)** -- multiply by 2.
* **Poor visibility (10-30 %)** -- multiply by 8.

Adjust these in **Settings -> Causation Factors** if your project
needs them.


Lateral traffic distribution
==============================

Ship positions across a lane are modelled as a **mixture** of up to
three Gaussian components and one uniform component:

.. math::

   f(z) = \sum_{i=1}^{3} w_i \; \phi\!\left(\frac{z - \mu_i}{\sigma_i}\right)
        + w_u \; U(z; a, b)

where

* :math:`z` is the lateral distance from the leg centreline (m),
* :math:`w_i`, :math:`\mu_i`, :math:`\sigma_i` are the weight, mean,
  and standard deviation of Gaussian :math:`i`,
* :math:`w_u` is the weight of the uniform component on
  :math:`[a, b]`,
* weights are normalised so :math:`\sum w_i + w_u = 1`.

You can fit the mixture from AIS data or set it by hand.  Values are
stored per segment per direction (``mean1_1``, ``std1_1``,
``weight1_1``, ``u_min1``, ``u_max1``, ``u_p1``, etc.).

Implemented in
:func:`compute.data_preparation.get_distribution`.


Coordinate systems
===================

* **WGS84 (EPSG:4326)** -- all stored geometry, user inputs, and the
  ``.omrat`` JSON file.  Lon/lat.
* **UTM** -- used internally for metric calculations.  OMRAT picks
  the zone from the study-area centroid and projects once per run.

The drifting model lives in UTM end-to-end.  The powered model uses
a cheaper per-project **equirectangular** projection centred on the
first leg's start point (``SimpleProjector``), which is fine at the
per-leg length scale where rays travel tens of kilometres.


Compass convention
===================

OMRAT uses **standard nautical bearings** everywhere: 0 = N, 90 = E,
180 = S, 270 = W, measured **clockwise from north**.  The wind rose,
stored segment bearings, and drift directions all follow this
convention.

Internal math uses the standard **math** convention (0 = E,
counter-clockwise).  The conversion is:

.. math::

   \theta_\mathrm{math} = (90^\circ - \theta_\mathrm{compass}) \bmod 360^\circ

Canonical implementation:
``drifting/engine.py:compass_to_math_deg``.  Callers that need an
``(x, y)`` step-vector directly use ``geometries/drift/coordinates.py:compass_to_vector``.

.. list-table:: Direction -> compass bearing
   :header-rows: 1
   :widths: 15 15 70

   * - Direction
     - Bearing
     - Vector (+X=East, +Y=North)
   * - N
     - 0
     - (0, +d)
   * - NE
     - 45
     - (+d/sqrt(2), +d/sqrt(2))
   * - E
     - 90
     - (+d, 0)
   * - SE
     - 135
     - (+d/sqrt(2), -d/sqrt(2))
   * - S
     - 180
     - (0, -d)
   * - SW
     - 225
     - (-d/sqrt(2), -d/sqrt(2))
   * - W
     - 270
     - (-d, 0)
   * - NW
     - 315
     - (-d/sqrt(2), +d/sqrt(2))


References
===========

* Friis-Hansen, P. (2008). *IWRAP MK II - Basic Modelling Principles
  for Prediction of Collision and Grounding Frequencies.* Technical
  University of Denmark.
* Pedersen, P.T. (1995). *Collision and Grounding Mechanics.* WEMT'95.
* Fujii, Y. et al. (1974). *Some factors affecting the frequency of
  accidents in marine traffic.* Journal of Navigation, 27.
* Talavera, A. et al. (2013). *Application of Dempster-Shafer theory
  for the quantification and propagation of the uncertainty caused by
  the use of AIS data.* Reliability Engineering and System Safety,
  111, 95-105.
* Engberg, P.C. (2017). *IWRAP Mk2 v5.3.0 Manual.* GateHouse A/S.
