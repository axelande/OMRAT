.. _iwrap-differences:

=====================================
Differences between OMRAT and IWRAP
=====================================

OMRAT implements the IWRAP methodology.  An OMRAT project can be exported
to IWRAP Mk II (**File > Export to IWRAP XML**) and imported back
(**File > Import from IWRAP XML**).  The two tools still make different
modelling choices in a few places, so the same model does not give the
same numbers in both.  This chapter lists the differences known today:
what IWRAP does, what OMRAT does, what it means for the results, and how
to line the two up when you compare them.

What IWRAP does was found by running exported models in IWRAP Mk II and
reading its job folder (2026).  Where a rule was inferred from IWRAP's
results rather than from its documentation, the text says so.


Before you compare results
==========================

#. **Turn off OMRAT's Category I start zone** (**Settings > Causation
   Factors**, *Ignore Category I hits within the first ... %*).  IWRAP has
   no such rule and the export cannot express it.  See
   :ref:`cat1-start-zone`.
#. **Set the passenger ship / fast ferry reduction factor in IWRAP's
   program settings.**  IWRAP's job engine takes that factor from its
   program settings, not from the model file, so the value the export
   writes is not used.  Both tools default to 20.
#. **Check the causation factors.**  IWRAP uses the factors in the model
   file, so an exported model runs with OMRAT's values.  If you edit the
   model in IWRAP, compare again: OMRAT's defaults are, for example, 4.9e-5
   for head-on and 1.9e-4 for powered allision, and an IWRAP installation
   can carry other values.
#. **Leave out speed restriction zones**, or compare without them: the
   export ignores them (:ref:`speed-restriction-zones`).
#. **Untick both Squat boxes** (**Settings > Causation factors**) and set
   **Start drifting from** to *Leg center* (**Settings > Drift settings**).
   The export writes no squat and IWRAP measures the drift from the leg.
#. **Give fast ferries IWRAP's blackout rate.**  OMRAT's default puts high
   speed craft with the passenger ships (0.1 per year); IWRAP puts fast
   ferries in its *other* class.  Set the type to the *other* rate in the
   Drift settings' blackout table when you compare drifting results.
#. **Compare per accident type.**  Crossing, bend and powered grounding
   follow different rules in the two tools (below), so a total over all
   types hides differences that cancel.


Where to look in an IWRAP job folder
------------------------------------

Each IWRAP calculation writes a job folder (``grisk_jobs/{date_time}``):

* ``model/model.xml`` -- the model the job ran, including the causation
  factors it used;
* ``main_settings.csv`` -- the program settings the job ran with,
  including the passenger ship and fast ferry reduction factors;
* ``results/main_result.csv`` -- the totals per accident type;
* ``results/result_*.csv`` -- the results per leg, waypoint, area edge
  and ship type.


Inputs and the export
=====================

.. list-table::
   :widths: 14 28 28 30
   :header-rows: 1

   * - Topic
     - IWRAP
     - OMRAT
     - Effect, and how to line them up
   * - Ship types
     - 14 fixed types.
     - The project's own list (the 21 AIS types by default).
     - The export merges OMRAT's types into IWRAP's 14.  Where several
       types meet in one cell the ships add up, and speed, draught, beam
       and height become ship-weighted means with unknown values left out.
       The import maps the 14 types onto OMRAT rows the same way.  Results
       per ship type are comparable on IWRAP's 14 types only.
   * - Length classes
     - 14 classes of 25 m, ``0-25`` to ``325-350``.
     - The project's own classes.
     - The export maps each OMRAT class onto IWRAP's.  Ships longer than
       350 m go into the last class; the export message lists the merges.
   * - Lateral distribution
     - One distribution per direction; positive = starboard of the sailing
       direction.
     - One axis per leg for both directions; negative = starboard of the
       drawn direction.
     - The export and the import convert the sign and write the weights as
       fractions.  Nothing to do.
   * - Leg width
     - ``max_width`` per leg, and ``max_width_powered``: the powered models
       count only ships within half of it from the leg line.
     - ``Width`` is the length of the AIS passage line (the tangent line)
       and the width of the drawn lane.  It limits no model.
     - The export writes ``Width`` as ``max_width``.  ``max_width_powered``
       is twice the largest lateral offset OMRAT's powered model casts rays
       to (every component's mean +/- 6 sigma and the uniform bounds, both
       directions), rounded up to 100 m and never less than ``Width``.  Exports made before
       November 2026 wrote ``Width`` there too, so IWRAP left out the ships
       outside the leg width (see `Powered grounding and allision`_).
   * - Routing at waypoints
     - ``leg_leg_distribution`` rows per waypoint.  ``AUTOMATIC`` rows split
       the remaining traffic equally and ignore the fraction; ``MANUAL``
       rows use it.
     - The junction transition matrix (from geometry, AIS or the user; see
       :ref:`junctions`).
     - The export writes every matrix as ``MANUAL`` rows that sum to 1.  The
       import reads the fractions back as user matrices.
   * - Traffic scaling
     - A traffic adjustment factor on the model and per ship type.
     - The ``Scaling (%)`` matrix and the global percentage on the
       Traffic tab, applied to the frequencies at the start of every run.
     - The export writes the frequencies with ``Scaling (%)`` applied, as
       the OMRAT calculation uses them, and both IWRAP factors as 1; the
       export message says how many cells were scaled.  The scaling itself
       does not travel: an import reads the scaled frequencies back at
       100 %.
   * - Passenger ship / fast ferry reduction
     - One reduction factor for passenger ships and one for fast ferries
       (default 20), taken from IWRAP's program settings
       (``main_settings.csv`` of the job), not from the model file.
     - ``pc`` keys ``passenger_reduction`` / ``fast_ferry_reduction``
       (default 20; **Settings > Causation factors**).  Collisions: the
       causation factor is divided by the square root of the factor for
       every passenger ship or fast ferry in the encounter; powered
       grounding and allision: their frequency is divided by the factor;
       drifting is not reduced.  Which types count follows the ship type
       name.
     - The export writes both factors to the model's ``global_settings``,
       but the IWRAP job uses its program settings: set the same values
       there.
   * - Missing traffic values
     - Uses the values in the model.
     - Fills a missing speed or draught before every run
       (:ref:`traffic-gaps`).
     - The export writes the filled values, and a missing beam as length /
       6.5, so both tools get the same numbers.
   * - Position check interval ``ai`` = 0
     - No way to switch Category II off: ``grounding_check_time = 0`` means
       the global default (180 s).
     - Category II is off for that leg and direction.
     - The export sets the leg extension past the waypoint that flow would
       overshoot to 0; the import reads it back as ``ai = 0``.
   * - Suppressed legs
     - --
     - A scenario: the traffic of a suppressed leg is moved to other legs
       (:ref:`suppress-leg`).
     - The export writes the scenario (traffic moved, suppressed legs left
       out) after a warning that lists the moves.
   * - Speed restriction zones
     - No equivalent.
     - Lower the speed on the legs inside a zone
       (:ref:`speed-restriction-zones`).
     - Not exported.  A model with zones gives different results in IWRAP.


Ship-ship collisions
====================

.. list-table::
   :widths: 14 28 28 30
   :header-rows: 1

   * - Topic
     - IWRAP
     - OMRAT
     - Effect, and how to line them up
   * - Head-on and overtaking
     - The whole lateral distribution of both directions.
     - The same: every normal component and the uniform part
       (:ref:`collisions`).
     - Close agreement with the same inputs.
   * - Crossing and merging
     - Every pair of routes through a waypoint (in-leg -> out-leg) with
       different in-legs.  Same out-leg: merging.  The two reverse routes
       of one turn (A -> B and B -> A): bend.  Every other pair: crossing,
       including two routes that share one leg in opposite directions.
       Inferred from IWRAP's results per waypoint: a pair that shares a leg
       counts when the starboard lanes of the two routes cross.  Going
       clockwise from the shared leg, the out-leg of the route that arrives
       on the shared leg comes before the in-leg of the route that leaves
       on it.
     - Same out-leg: merging.  Four different legs whose paths interleave
       around the waypoint: crossing.  Two routes that share a leg meet on
       that leg, which the head-on model already counts, so they are not a
       waypoint encounter.
     - IWRAP's crossing is higher wherever routes share legs, for example a
       Y-junction with traffic in both directions.  It cannot be changed in
       either tool.
   * - Bend
     - The two reverse routes through a turn (A -> B and B -> A), crossing
       formula at 180 degrees minus the turn, with every ship of both
       routes, once per waypoint.  Inferred from IWRAP's results per
       waypoint.  No turns under 10 degrees (the crossing formula's 10-170
       degree range).
     - The same rule (see :ref:`collisions`), with the same angle limit:
       turns under 10 degrees give no bend.
     - The theory document (Friis-Hansen 2008, section 4.4) gives no
       formula, only that at a two-leg bend a ship becomes a collision
       candidate if it does not change course, with an omission probability
       :math:`P_0 = 0.01`.  Neither tool applies :math:`P_0`.  Before
       November 2026 OMRAT's bend was the 1 % non-turners against the
       turners of their own route and is not comparable.
   * - Angle limit
     - 10 to 170 degrees (theory document, eq. 4.6).  Waypoints that turn
       less than 10 degrees get no bend in IWRAP's results.
     - The same by default: route pairs within 10 degrees of parallel or
       opposite are skipped (the crossing formula divides by the sine of
       the angle).  The limit is a setting in **Causation factors**
       (``pc.min_encounter_angle_deg``); the IWRAP export does not carry
       it, so keep it at 10 for a comparison.  Before November 2026 it was
       0.1 rad (5.7 degrees).  The angle is measured in OMRAT's map
       projection; before November 2026 it came from raw longitude /
       latitude differences, which distorts it at high latitude.
     - Pairs close to opposite (gentle turns) are sensitive to the limit.
   * - Ship length and breadth
     - Built-in dimensions per ship type and length class, from Lloyd's
       register data (theory document, appendix).  Inferred from IWRAP's
       results per cell: a type and size with no ships in that table gets
       little or no size, and the smallest classes get the table's mean
       length rather than the middle of the class.
     - The middle of the length class and the beam given in the traffic
       (length / 6.5 when it is missing).
     - In waypoint encounters IWRAP is lower for ships beyond its table
       (for example general cargo ships over 250 m) and higher for the
       smallest craft; a few percent overall.  No setting lines them up.


Powered grounding and allision
==============================

.. list-table::
   :widths: 14 28 28 30
   :header-rows: 1

   * - Topic
     - IWRAP
     - OMRAT
     - Effect, and how to line them up
   * - Lateral extent
     - Ships within ``max_width_powered / 2`` of the leg line.  Inferred
       from IWRAP's results: hits outside it are missing, and the share of
       the distribution beyond it is lost, not spread over the rest.
     - The whole lateral mixture, each normal component's mean +/- 6 sigma
       and the uniform part (:ref:`ray-sampling`).
     - The export sets ``max_width_powered`` to cover OMRAT's rays (see
       *Leg width* above).  Beyond the AIS passage line the tails come from
       the fitted distribution, not from observed ships.  If a lane's sigma
       is close to half the leg width, the traffic probably reaches beyond
       the passage line, and a wider leg would sample it.
   * - Category I start zone
     - None.
     - Category I ignores hits in the first 10 % of the leg from the
       origin waypoint (default on; :ref:`cat1-start-zone`).
     - Not exported.  Switch it off in OMRAT to compare.  It can remove
       most of a structure's Category I allision when the structure is
       close to a waypoint.
   * - Category II probability
     - Sum over rays of ship share x ``exp(-d / (ai V))``.
     - The same, summed ray by ray (see :ref:`powered`).
     - Nothing to do.
   * - Depth against draught
     - Optional safety margin (``grounding_safety_margin``), optional squat
       (``calc_squat`` on the model and on each leg), optional fixed
       draught.
     - A depth area is a hazard when it is not deeper than the cell's
       draught plus squat.  Squat is on by default for drifting and for powered
       grounding (two boxes in **Settings > Causation factors**): Barras,
       open water, at the ship's speed on the leg.  No safety margin.
     - The export writes margin 0, ``calc_squat = false`` on the model and
       no fixed draught.  IWRAP adds ``calc_squat = true`` to every leg when
       it saves a model; whether the model-level setting still wins has not
       been tested, and IWRAP's own squat formula is not documented.
       Untick OMRAT's Squat boxes to compare.
   * - Structure height
     - Structure areas, and bridges with a height.
     - A ship whose height is known and lower than the structure's height
       passes it (both categories).  A ship height of 0 or ``inf`` (AIS
       gives none) is unknown and gets no clearance.
     - The export writes every structure as a structure area with depth
       = minus its height, as IWRAP itself stores the spans of a bridge
       (``-1`` when the height is 0 or missing), and the import reads the
       height back from it.  Whether IWRAP applies that clearance to a
       structure that is not a bridge has not been tested; give the
       structures height 0 in OMRAT if the results differ there.
   * - Causation factors
     - Category I (``p_*_causation``) and Category II
       (``p_*_no_turn_causation``) per accident type, read from the model
       file.
     - ``grounding_cat1`` / ``allision_cat1`` and ``grounding`` /
       ``allision``.
     - Mapped one to one by the export and the import (see
       :ref:`iwrap-causation-mapping`).


Drifting
========

.. list-table::
   :widths: 14 28 28 30
   :header-rows: 1

   * - Topic
     - IWRAP
     - OMRAT
     - Effect, and how to line them up
   * - Blackout rate
     - Two classes, ``blackout_roro_passenger`` and ``blackout_other``.
       Fast ferries are in the *other* class: IWRAP's drift results per
       ship type show it.
     - One rate per ship type.  The default is 0.1 per year for types whose
       name contains passenger, ferry, ro-pax, ro-ro, cruise, high speed or
       HSC, and 1.0 for the rest: fast ferries go with the passenger ships
       (several propulsion units).
     - The export gives each IWRAP class the rate most of its OMRAT types
       carry; the import gives every type its class rate, fast ferries the
       passenger rate.  A per-type rate that differs from the rest of its
       class cannot be exported, and IWRAP applies *other* to fast ferries.
       To compare, give OMRAT's fast ferries the *other* rate.
   * - Anchoring
     - ``anchor_probability``, ``max_anchor_depth`` and
       ``min_anchor_dist_from_ground`` (default 3).
     - Anchor probability and maximum anchoring depth.  There is no minimum
       distance rule.  A leg that lies inside anchorable water can anchor
       from distance 0 (see :ref:`drifting`).
     - The export writes ``min_anchor_dist_from_ground`` explicitly, so
       IWRAP does not fall back on its own default.  Results can still
       differ for small ships that sail close to anchorable water.
   * - Where the drift starts
     - The leg line (assumed; not compared yet).
     - **Settings > Drift settings > Start drifting from**: the leg line
       (default) or, per direction, a line through the mean of the lateral
       distribution (:ref:`drifting-level1b`).
     - Use *Leg center* to compare.
