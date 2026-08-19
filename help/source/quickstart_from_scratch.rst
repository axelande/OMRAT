.. _quickstart_from_scratch:

==============================
Quickstart - from scratch
==============================

The :ref:`quickstart` chapter walks you through the **example project**.
This chapter shows the same workflow but **starting with a blank
QGIS project**: how to draw your first leg, set its width, bring in
depth and object data, and produce a first risk number.

.. note::

   A real-world project usually pulls its traffic counts from a
   PostGIS AIS database via the **Update AIS** / **Update all
   distributions** buttons -- not from manual entry.  Setting that
   database up (schema, raw-NMEA / CSV ingestion, vessel-lookup
   table, AIS connection settings) is its own multi-step task and is
   covered end-to-end in :ref:`database-setup`.  If you'll need real
   AIS traffic for your project, **read that chapter first** so the
   database is ready before you reach the Traffic tab step below;
   otherwise you'll either fill the matrix in by hand or come back
   later.

.. contents:: In this chapter
   :local:
   :depth: 1


1. Open the plugin (no project loaded)
=========================================

Open QGIS, click the OMRAT icon, and dock the panel on the right.
With nothing loaded the Routes tab looks like this:

.. figure:: _static/screenshots/quickstart/qs_01_empty_routes_tab.png
   :width: 80%
   :alt: Empty Routes tab, before the first leg is drawn.

   Routes tab on a fresh project.  ``twRouteList`` is empty and the
   distribution panel below shows zeroed defaults.

.. figure:: _static/screenshots/quickstart/qs_02_canvas_blank.png
   :width: 80%
   :alt: Empty QGIS canvas with no layers.

   The QGIS canvas before any layer is added.  Most users begin by
   adding a basemap (XYZ Tiles → OpenStreetMap is fine).


2. Place legs on the map
==========================

A *leg* is a single straight segment of a route.  OMRAT calculates risk
**per leg**, so a curved or branching route is approximated with
several short legs joined end to end.

To draw a leg:

#. Click **Place leg** on the Routes tab toolbar.  The cursor turns
   into a crosshair.
#. Click the start point on the canvas.
#. Click the end point.  The leg is added to ``twRouteList`` and a
   blue line appears on the canvas.
#. Repeat for each leg.  Click **Stop route** to leave place-leg mode.

After one leg the Routes tab shows the new row:

.. figure:: _static/screenshots/quickstart/qs_03_route_after_first_leg.png
   :width: 80%
   :alt: Routes tab after a single leg has been placed.

   The first leg appears in ``twRouteList`` (Segment 1, Route 1).
   Adjust **Width** to the width of the corridor in metres
   (5000 = a 5 km wide corridor).

.. figure:: _static/screenshots/quickstart/qs_04_canvas_with_legs.png
   :width: 80%
   :alt: QGIS canvas showing the placed leg as a blue line.

   The leg as drawn on the canvas.  Two grey "offset" lines mark the
   width of the corridor.

.. tip::
   Endpoints that exactly coincide between legs (within metres, not
   visually) are treated as **shared vertices**: dragging one with the
   QGIS Vertex Tool moves every connected leg's endpoint together,
   so a curved route stays connected when you re-route.


3. Bring in depth data
========================

Powered grounding and drifting grounding both rely on a depth contour
layer.  OMRAT consumes depth polygons (one per discrete depth value).

Where to get depth data:

* **EMODnet Bathymetry** (https://emodnet.ec.europa.eu/en/bathymetry) -
  free 1/16 arc-min DTM for European seas.  Download a tile, contour
  it in QGIS (*Raster → Extraction → Contour Polygons*), then load
  the resulting polygons into the Depths tab.
* **GEBCO via OpenTopography** - the Depths tab has a built-in downloader
  that fetches GEBCO depth data directly inside OMRAT.  See
  :ref:`gebco-api-key` below for how to get the free API key required.
* **Local hydrographic offices** - national bathymetry products are
  usually higher resolution but require a license.
* **ENCs** (Electronic Navigational Charts) - if you have access to
  S-57/S-101 chart data, the *Depth Areas* (DEPARE) layer is exactly
  what OMRAT wants.

Once you have polygons in QGIS, switch to the **Depths** tab and use
**Add depth from layer** to import them.  The tab is empty before any
depths are added:

.. figure:: _static/screenshots/quickstart/qs_05_depths_tab_empty.png
   :width: 80%
   :alt: Empty Depths tab.

   Depths tab on a fresh project.  Each row links one polygon to its
   depth value (in metres, positive downwards).


.. _gebco-api-key:

3a. Fetching GEBCO depths directly (OpenTopography)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OMRAT can download GEBCO bathymetry for your project area and convert
it to depth polygons in one click, without leaving QGIS.  You need a
**free** OpenTopography API key first.

**Step 1 — Create an OpenTopography account**

   Go to https://portal.opentopography.org and click **Create new account** (top
   right).  Fill in a username, e-mail address, and password, then
   confirm your e-mail.

**Step 2 — Generate an API key**

   Log in, then click your username in the top-right corner and choose
   **MyOpenTopo → My Account**.  Scroll to the **API Access** section
   and click **Request API Key**.  The key is a short alphanumeric
   string (e.g. ``abc123def456``).  Copy it.

**Step 3 — Paste the key into OMRAT**

   In the Depths tab, paste the key into the **API Key** field (the
   text box to the right of the label "API Key:").  OMRAT saves it
   automatically between sessions so you only need to do this once.

**Step 4 — Enter your bounding box**

   Fill in the four coordinate boxes:

   * **LL Lat / LL Lon** — latitude and longitude of the south-west
     (lower-left) corner of the area you want to download.
   * **UR Lat / UR Lon** — latitude and longitude of the north-east
     (upper-right) corner.

   A good starting point is to add a small buffer around your routes.
   You can read off coordinates from the QGIS status bar (shown when
   you hover over the canvas).

**Step 5 — Set depth intervals and fetch**

   Enter the maximum depth of interest in **Max depth (m)** and a
   contour step in **Depth interval (m)**.  Click **Update depth**
   to preview the contour levels, then click
   **Fetch GEBCO depth**.  OMRAT downloads the GeoTIFF, vectorizes it,
   and adds one row per depth interval to the table automatically.

.. note::

   GEBCO resolution is 15 arc-seconds (~450 m at mid-latitudes).  It
   is appropriate for open-water or offshore risk assessments; for
   nearshore or port approaches you should use higher-resolution
   hydrographic charts or ENCs.


4. Define obstacles (objects)
==============================

Bridges, wind-park footprints, and other surface structures go on the
**Objects** tab.  The workflow is identical to Depths: import a polygon
layer (or digitise polygons by hand) and set the structure's height
in metres above sea level.

.. figure:: _static/screenshots/quickstart/qs_06_objects_tab_empty.png
   :width: 80%
   :alt: Empty Objects tab.

   Objects tab on a fresh project.  Powered allision and drifting
   allision use the polygons listed here.

Common sources for object polygons:

* **OpenStreetMap** - bridges, piers and offshore wind farms tagged
  ``man_made=*`` are usually present.
* **EMODnet Human Activities** - offshore platforms, wind-farm
  layouts.
* **National marine spatial planning portals**.


5. Settings: drift, ship categories, causation, AIS
=====================================================

Open the relevant **Settings** dialog from the dock's gear menu.

.. figure:: _static/screenshots/quickstart/qs_07_drift_settings_wind_rose.png
   :width: 80%
   :alt: Drift Settings dialog showing the wind-rose.

   Drift Settings.  The wind-rose drives drifting risk -- start from a
   uniform 1/8 distribution if you don't yet have site-specific data,
   then refine using a local meteorological reanalysis (ERA5, MERRA-2).

The other Settings dialogs are documented in :ref:`user_guide`:

* **Ship Categories** - the type/size matrix that maps AIS rows to
  the cells of the traffic table.
* **Causation Factors** - the per-accident-type Pc values.
* **AIS Connection** - host/database/user for the optional AIS
  Postgres database.  If you don't have an AIS database, fill the
  traffic table in by hand on the Traffic tab.


6. Run the model
==================

Switch to the **Run Analysis** tab, fill in:

* **Name of the model** - a short slug for this scenario.
* **File path** - a folder.  Each Run writes three files into this
  folder:

  - ``<name>_<timestamp>.gpkg`` - the result layers.
  - ``<name>_<timestamp>.omrat`` - a snapshot of the inputs (read-only).
  - ``<name>_results_<timestamp>.md`` - a Markdown report covering
    every accident type.

Click **Run Model**.  The Run button is greyed out until both fields
are set, and a popup spells out which is missing if you click anyway.

When the run finishes the **Accident probabilities** table populates
with one row per accident type and a **View** button per row that
opens the matching driver visualisation.


7. Inspect the results
========================

The **Previous runs** table at the top of the Run Analysis tab keeps
every run.  Select a row and click **Add selected run results to map**
to load that run's GeoPackage as styled QGIS layers.  Two runs can be
compared side-by-side from the **Compare** tab.

.. seealso::
   * :ref:`quickstart` -- same flow, but with the example project.
   * :ref:`user_guide` -- detailed reference for every tab.
   * :ref:`concepts` -- glossary of the OMRAT-specific vocabulary.
