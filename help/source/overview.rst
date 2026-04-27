.. _overview:

===============
Project Overview
===============

What OMRAT does
================

OMRAT takes **three kinds of input** and produces **one kind of output**.

Inputs:

#. A **shipping route** -- one or more polyline segments on a map.
#. **Traffic** per segment -- how many ships of each type pass per
   year, their speed, draught, beam, and height above waterline.
#. **Obstacles** -- depth polygons (bathymetry) and structure polygons
   (bridges, wind turbines, piers).

Output: **expected annual frequency** for each accident type.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Accident type
     - When it occurs
   * - **Drifting grounding**
     - A ship loses propulsion, drifts with wind/current, grounds on a
       shallow depth polygon before the crew can restart the engine.
   * - **Drifting allision**
     - Same but the drifting ship hits a structure.
   * - **Drifting anchoring**
     - The drifting ship successfully anchors before it grounds.
   * - **Powered grounding**
     - A ship under power fails to turn at a bend, continues straight,
       grounds on shallower water ahead.
   * - **Powered allision**
     - Same but hits a structure.
   * - **Head-on collision**
     - Two ships on the same leg travelling in opposite directions.
   * - **Overtaking collision**
     - Same leg, same direction, different speeds.
   * - **Crossing collision**
     - Two legs share a waypoint at a non-trivial angle.
   * - **Bend collision**
     - Same leg, one ship fails to turn at a bend.


Who OMRAT is for
=================

* **Port / fairway designers** doing quantitative risk assessments.
* **Environmental authorities** estimating baseline risk for a sea
  area before permitting new infrastructure.
* **Researchers** comparing IWRAP-style methodology outputs against
  historical accident data.
* **IWRAP users** who want an open-source alternative and can already
  import / export XML.

OMRAT is not a routing or navigation tool.  It does not simulate
individual ship movements.  It is a **statistical** tool: for a given
traffic pattern it returns *how often* each accident type is expected.


The methodology in one paragraph
================================

OMRAT implements the IWRAP framework (Friis-Hansen 2008, Pedersen
1995): every accident frequency is decomposed into a **geometric
candidate count** (how often *could* an accident happen based only on
geometry and traffic) multiplied by a **causation factor** (how often
does an accident *actually* happen given a candidate encounter).

.. math::

   F_\mathrm{accident} = N_A \cdot P_C

:math:`N_A` is derived from the route, traffic, and obstacles.
:math:`P_C` comes from published tables (defaults: Fujii 1974, IALA
IWRAP manual).  See :ref:`theory` for the full reference table and
:ref:`drifting` / :ref:`collisions` / :ref:`powered` for each accident
type's derivation.


Background and funding
======================

OMRAT has been developed with funding from:

* **Naturvardsverket** -- Swedish Environmental Protection Agency.
* **RISE** -- Research Institutes of Sweden.

It is licensed under GPL v2+.  The source is at
https://github.com/axelande/OMRAT.

The mathematical foundations come from:

* Pedersen, P.T. (1995). *Collision and Grounding Mechanics.* WEMT'95.
* Friis-Hansen, P. (2008). *IWRAP MK II - Basic Modelling Principles
  for Prediction of Collision and Grounding Frequencies.* Technical
  University of Denmark.
* Fujii, Y. et al. (1974). *Some factors affecting the frequency of
  accidents in marine traffic.* Journal of Navigation, 27.


What's next
============

* Never installed OMRAT? -> :ref:`installation`.
* Installed and curious what a first run looks like? -> :ref:`quickstart`.
* Want to know what a "leg" is? -> :ref:`concepts`.
* Ready to build your own project? -> :ref:`user_guide`.
