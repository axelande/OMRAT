.. _installation:

============
Installation
============

Prerequisites
=============

OMRAT requires:

- **QGIS 3.0+** (recommended: QGIS 3.28 LTS or later)
- **Python 3.9+** (bundled with QGIS)
- The following Python packages (most are bundled with QGIS):

  - NumPy
  - SciPy
  - Shapely
  - GeoPandas
  - Matplotlib
  - Pydantic
  - pyproj

Optional dependencies:

- **psycopg2** -- for AIS database connectivity
- **requests** -- for GEBCO bathymetry download


Installing the Plugin
=====================

From QGIS Plugin Manager
-------------------------

1. Open QGIS
2. Go to **Plugins** > **Manage and Install Plugins...**
3. Search for "OMRAT" or "Maritime Risk"
4. Click **Install Plugin**

From Source (Development)
--------------------------

1. Clone the repository::

      git clone https://github.com/axelande/OMRAT.git

2. Copy or symlink the OMRAT folder into your QGIS plugin directory:

   - **Linux**: ``~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/``
   - **Windows**: ``%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\``
   - **macOS**: ``~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/``

3. Restart QGIS and enable the plugin from the Plugin Manager.


Installing Additional Dependencies
===================================

If any Python packages are missing, install them using the QGIS Python
environment. On Windows with OSGeo4W::

    C:\OSGeo4W\bin\python3.exe -m pip install pydantic

On Linux::

    python3 -m pip install pydantic --user


Verifying the Installation
==========================

After installation, you should see the OMRAT icon in the QGIS toolbar.
Click it to open the plugin dock widget. The widget should display tabs
for: Route, Traffic, Depths, Objects, Distributions, Results, and Drift
Analysis.
