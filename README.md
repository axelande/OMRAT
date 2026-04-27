# OMRAT — Open Maritime Risk Analysis Tool

[![License: GPL v2+](https://img.shields.io/badge/License-GPL%20v2%2B-blue.svg)](LICENSE)
[![QGIS](https://img.shields.io/badge/QGIS-3.30%2B-green.svg)](https://qgis.org)
[![Python](https://img.shields.io/badge/Python-3.9%2B-yellow.svg)](https://www.python.org)

OMRAT is a QGIS plugin that calculates the **expected annual frequency**
of maritime accidents — drifting groundings, drifting allisions,
powered groundings, powered allisions, and ship-ship collisions
(head-on, overtaking, crossing, bend) — for any user-defined shipping
route. It implements the IWRAP methodology (Friis-Hansen 2008,
Pedersen 1995) as an open-source alternative to the IALA reference
tool, with import / export compatibility for IWRAP Mk2 XML.

Funded by **Naturvårdsverket** (Swedish Environmental Protection
Agency) and **RISE** (Research Institutes of Sweden).

## Documentation

The full user and developer guide is published at
**<https://axelande.github.io/OMRAT/>** and ships with the plugin under
[`help/source/`](help/source/).

| If you want to … | Read |
|---|---|
| install OMRAT and run the example project in 10 minutes | [Quickstart](help/source/quickstart.rst) |
| learn what every tab and dialog does | [User guide](help/source/user_guide.rst) |
| look up a term (leg, corridor, shadow, *P_NR* …) | [Concepts](help/source/concepts.rst) |
| understand **what** is calculated (the math) | [Theory overview](help/source/theory.rst), [Drifting](help/source/drifting.rst), [Collisions](help/source/collisions.rst), [Powered](help/source/powered.rst) |
| understand **how** it's calculated (the call tree) | [Code flow overview](help/source/code_flow.rst), [Drifting](help/source/code_flow_drifting.rst), [Collisions](help/source/code_flow_collisions.rst), [Powered](help/source/code_flow_powered.rst) |
| open or write a `.omrat` project file by hand | [Data format reference](help/source/reference_data_format.rst) |

The documentation deliberately follows a **two-track** model: every
accident type has both a *theory* chapter (formulas, assumptions) and
a *code-flow* chapter (which functions are called, in what order, by
what). They cross-link so you can switch tracks as you read.

## Installation

The recommended path is via the QGIS Plugin Manager
(**Plugins → Manage and Install Plugins → search "OMRAT"**).
First-run installation also offers to install the
[`qpip`](https://github.com/opengisch/qpip) helper plugin which then
auto-installs every Python package OMRAT needs from
[`requirements.txt`](requirements.txt).

For a development install, see
[`help/source/installation.rst`](help/source/installation.rst). The
plugin's `pb_tool.cfg` deploys to:

```
%APPDATA%\QGIS\QGIS<n>\profiles\default\python\plugins\Omrat
```

so `pb_tool deploy` from the source root is the typical iteration
loop.

## What you get

After clicking **Run Model** with a route, traffic table, depths, and
structures defined, OMRAT writes:

- the per-accident-type total probabilities into the result fields on
  the Run Analysis tab,
- a `<model_name>_<YYYYMMDD_HHMMSS>.gpkg` with **six** result layers
  (drifting allision/grounding polygons, powered allision/grounding
  polygons with per-leg attributes, ship-collision lines per leg,
  ship-collision points per shared waypoint),
- a row in the master history database
  (`%APPDATA%\OMRAT\omrat_history.sqlite`) referencing the per-run
  GeoPackage so you can compare runs and reload past results onto the
  map without re-running the calculation.

A markdown drift report and per-leg / per-obstacle drill-down dialogs
are also produced. See the [User guide](help/source/user_guide.rst)
for screenshots and the field-by-field walk-through.

## Project status

Active development. The current release (see [`metadata.txt`](metadata.txt))
covers:

- ✅ Drifting model (allision / grounding / anchoring) with shadow
  cascade.
- ✅ Powered Cat II grounding + allision with vectorised ray casting.
- ✅ Ship-ship collisions (head-on, overtaking, crossing, bend).
- ✅ IWRAP Mk2 XML import / export.
- ✅ AIS database import (PostgreSQL/PostGIS).
- ✅ Per-run GeoPackage history with side-by-side comparison.

Open work (high level):

- Make the calculation IWRAP-bit-identical for cross-validation.
- Add consequence modelling — collision energies, oil-spill volumes,
  structure-damage thresholds.
- Powered Cat I (failure-on-route) once the supporting traffic
  segmentation lands.

## Contributing

Pull requests welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for
the contributor licence note (GPL v2+), dev setup, test commands, and
the PR checklist.

Run the standalone test suite (no QGIS needed) with:

```bash
C:/OSGeo4W/apps/Python312/python.exe -m pytest -p no:qgis --noconftest tests/
```

## Citing

If you use OMRAT in a publication, please cite via the
[`CITATION.cff`](CITATION.cff) file (GitHub renders a *"Cite this
repository"* button from it). Cite the underlying methodology
references too:

- Pedersen, P.T. (1995). *Collision and Grounding Mechanics.* WEMT'95.
- Friis-Hansen, P. (2008). *IWRAP MK II — Basic Modelling Principles
  for Prediction of Collision and Grounding Frequencies.* Technical
  University of Denmark.

## Licence

GNU General Public License **v2 or any later version (GPL v2+)** — see
[`LICENSE`](LICENSE). You can use, modify and redistribute the code,
including commercially, but distributed modifications must remain
under GPL.
