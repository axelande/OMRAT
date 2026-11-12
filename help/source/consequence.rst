.. _consequence:

============================================
Oil-spill consequence -- catastrophe rates
============================================

After the per-accident-type frequencies are calculated (drifting,
powered, collision), OMRAT can roll them up into **annual catastrophe
exceedance frequencies**: how often, on average per year, an oil spill
larger than a user-defined size is expected to happen.

The module is reached via the **Consequence** menu in the OMRAT
window's menu bar.  Four dialogs configure the inputs; the actual
calculation runs as the last step of **Run model** (on the reports of
whichever model phases that run calculated) and writes its result into
the "Catastrophe results" table on the Run Analysis tab.


At a glance
===========

For each ship cell ``(ship_type, length_interval)`` and each of the nine
accident types the model multiplies:

* the **annual accident frequency** (events/year, from the per-cell
  breakdown the four model phases emit),
* a **conditional spill probability** per spill level (columns 0-3 =
  "No spill", "Minor spill", "Major spill", "Total loss", in percent),
* a **spill fraction** of the ship's oil-onboard quantity (also in
  percent), and
* the cell's **oil onboard** (m\ :sup:`3`).

It then counts each (accident, spill level) contribution toward every
catastrophe threshold whose quantity it exceeds.  The output is a
list of ``{name, quantity, exceedance}`` rows ordered by quantity
ascending, plus the total spill frequency, the mean spilled volume per
year and a per-accident-type ``{frequency, spill_m3}`` breakdown.


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

9 rows (one per accident type: drifting allision, drifting grounding,
powered allision, powered grounding, overtaking, head-on, crossing,
merging, bend) x 4 columns (No spill / Minor spill / Major spill /
Total loss) in percent.  **Each row must sum to 100% +/- 0.05%** -- the
dialog refuses to save until it does.

Defaults:

* Drifting accidents: ``[98, 2, 0, 0]`` -- almost all benign.
* All other accidents: ``[97, 1, 1, 1]``.

Spill fraction per accident
---------------------------

Same 9 x 4 shape; each cell is the fraction of the ship's oil onboard
that escapes for the given (accident, spill-level) combination, in
percent (0 to 100).  No row constraint -- you can choose any
spill-fraction distribution.  Default is ``[0, 10, 30, 100]`` for every
accident.

Catastrophe levels
------------------

A list of ``{name, quantity}`` rows (quantity in m\ :sup:`3`); **at
least two are required**.
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

* 0.001 x 0.97 = 9.7e-4 "No spill" events (volume 0).
* 0.001 x 0.01 = 1e-5 "Minor spill" events of 10 000 m\ :sup:`3`.
* 0.001 x 0.01 = 1e-5 "Major spill" events of 30 000 m\ :sup:`3`.
* 0.001 x 0.01 = 1e-5 "Total loss" events of 100 000 m\ :sup:`3`.

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
* Result table: ``TWCatastropheResults`` on the Run Analysis tab shows
  the saved runs (baseline + selection), read from each run's
  ``<stem>.consequence.json`` sidecar
  (``omrat_utils/accident_results_mixin.py._show_run_comparison``).

Per-cell accident frequencies feeding the consequence calc come from
each model phase's ``by_cell`` dict (or ``by_cell_allision`` /
``by_cell_grounding`` for drifting).  Pair-wise collisions -- head-on,
overtaking, crossing, merging and bend alike -- split each pair
contribution 50/50 between the two participating cells.  Cell keys are
``"{ship_type}_{length_class}"`` and index the ``oil_onboard`` matrix.


Limitations and TODOs
=====================

* A ship in ballast floats higher and carries less oil than a laden
  one -- not yet modelled.  Today every voyage uses the configured
  ``oil_onboard``; letting it follow the draught of the traffic is
  planned.
* **Merging collisions** (two routes leaving a waypoint on the same leg)
  are their own accident type with their own row, ``pc['merging']``
  entry and spill-matrix rows.  The merging causation factor defaults to
  the crossing value, because IWRAP publishes no separate figure.
