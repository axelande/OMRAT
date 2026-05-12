.. _consequence:

============================================
Oil-spill consequence -- catastrophe rates
============================================

After the per-accident-type frequencies are calculated (drifting,
powered, collision), OMRAT can roll them up into **annual catastrophe
exceedance frequencies**: how often, on average per year, an oil spill
larger than a user-defined size is expected to happen.

The module is reached via the **Consequence** menu in the plugin
toolbar.  Four dialogs configure the inputs; the actual calculation
runs as the last phase of **Run Model** and writes its result into the
"Catastrophe results" table on the Run Analysis tab.


At a glance
===========

For each ship cell ``(ship_type, length_interval)`` the model multiplies:

* the **annual accident frequency** (events/year, from the per-cell
  breakdown the four model phases emit),
* a **conditional spill probability** (rows 0-3 = "no spill", "small
  spill", "medium spill", "catastrophic spill", in percent),
* a **spill fraction** of the ship's oil-onboard quantity (also in
  percent), and
* the cell's **oil onboard** (m\ :sup:`3`).

It then counts each (accident, level) contribution toward every
catastrophe threshold whose quantity it exceeds.  The output is a
list of ``{level_name, quantity, exceedance}`` rows ordered by
quantity ascending.


The four input dialogs
======================

Open each from **Consequence > ...** in the menu bar.

Maximum oil onboard
-------------------

A ship-type x length-interval matrix in m\ :sup:`3`.  Defaults are:

* Tankers: ``80 x average_length`` (m\ :sup:`3`) per cell, where
  average length is the midpoint of the cell's length interval and the
  open-ended top bin uses ``min + 50``.
* Everything else: 100 m\ :sup:`3` per cell.

Override any cell directly; the dialog uses spinboxes ranged 0 to
10\ :sup:`7`, two decimals, 10 m\ :sup:`3` step.

Spill probability per accident
------------------------------

8 rows (one per accident category) x 4 columns (no spill / small /
medium / catastrophic) in percent.  **Each row must sum to 100% +/-
0.05%** -- the dialog refuses to save until it does.

Defaults:

* Drifting accidents: ``[98, 2, 0, 0]`` -- almost all benign.
* All other accidents: ``[97, 1, 1, 1]``.

Spill fraction per accident
---------------------------

Same 8 x 4 shape; each cell is the fraction of the ship's oil onboard
that escapes for the given (accident, spill-level) combination, in
percent.  No row constraint -- you can choose any spill-fraction
distribution.  Default is ``[0, 10, 30, 100]`` for every accident.

Catastrophe levels
------------------

A list of ``{name, quantity_m3}`` rows; **at least two are required**.
The Run Analysis "Catastrophe results" table will have one row per
catastrophe level, sorted by quantity ascending.  Defaults are
Minor / Major / Catastrophic at 50 / 500 / 5000 m\ :sup:`3`.


Worked example
==============

Suppose a single tanker cell sees 0.001 powered groundings per year,
the tanker's oil onboard is 100 000 m\ :sup:`3`, and the spill
probability for powered grounding is the default ``[97, 1, 1, 1]``
with fractions ``[0, 10, 30, 100]``.

Per year the cell produces:

* 0.001 x 0.97 = 9.7e-4 "no spill" events (volume 0).
* 0.001 x 0.01 = 1e-5 "small spill" events of 10 000 m\ :sup:`3`.
* 0.001 x 0.01 = 1e-5 "medium spill" events of 30 000 m\ :sup:`3`.
* 0.001 x 0.01 = 1e-5 "catastrophic" events of 100 000 m\ :sup:`3`.

With the default thresholds:

* >50 m\ :sup:`3` ("Minor") fires 3e-5 times/yr (all three spill levels).
* >500 m\ :sup:`3` ("Major") fires 3e-5 times/yr (the smallest spill is
  already 10 000 m\ :sup:`3`).
* >5000 m\ :sup:`3` ("Catastrophic") fires 3e-5 times/yr.

The table renders one row per level with the per-year exceedance.


Headless validation
===================

For batch / CI use, ``compute.consequence.validate_consequence`` runs
the same constraints the dialogs enforce (row sums, minimum two
catastrophe levels, non-negative values) and returns a structured
report instead of popping a UI:

.. code-block:: python

    from compute.consequence import validate_consequence
    rep = validate_consequence(consequence_block)
    if not rep.ok:
        print("\n".join(rep.errors))


Behind the scenes
=================

* Code: ``compute/consequence.py:compute_catastrophe_exceedance``.
* Defaults / reshaping: ``omrat_utils/consequence_defaults.py``.
* UI handler (live state, dialog wiring, row-sum validation):
  ``omrat_utils/handle_consequence.py``.
* Result table: ``omrat_utils/accident_results_mixin.py
  ._populate_catastrophe_results_table`` writes into the
  ``TWCatastropheResults`` widget.

Per-cell accident frequencies feeding the consequence calc come from
each model phase's ``by_cell`` dict (or ``by_cell_allision`` /
``by_cell_grounding`` for drifting).  Pair-wise collisions split each
pair contribution 50/50 between the participating cells; bend
distributes proportionally to per-cell traffic on the leg.


Limitations and TODOs
=====================

* Tankers split between **laden and ballast** voyages would carry
  different spill volumes -- not yet modelled.  Today every voyage
  uses the configured ``oil_onboard``.
* **Merging collisions** share the IWRAP causation factor with
  ``crossing``.  The model already separates them in ``by_cell`` based
  on a 30 deg crossing-angle threshold, but to be a fully separate
  accident type ``merging`` needs its own ``pc`` entry.
