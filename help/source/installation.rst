.. _installation:

============
Installation
============

Prerequisites
=============

OMRAT requires:

- **QGIS 3.30 or newer, QGIS 4 included** (``qgisMinimumVersion=3.30``
  and ``qgisMaximumVersion=4.9`` in ``metadata.txt``).  The code is
  checked for Qt 6 / QGIS 4 compatibility before every release.
- **Python 3.9+** (bundled with QGIS).
- A handful of scientific Python packages listed in ``requirements.txt``
  (NumPy, Pandas, SciPy, Shapely, GeoPandas, Matplotlib, Pydantic,
  Requests, defusedxml; pyproj comes with GeoPandas).  If you install
  OMRAT via the QGIS Plugin Manager the **qpip** plugin dependency
  installs these automatically — see below.
- **psycopg2** -- the PostgreSQL driver.  OMRAT imports it at start-up,
  so it must be present even if you never use an AIS database.  It
  ships with the standard QGIS installers; on Linux install your
  distribution's ``python3-psycopg2`` package if it is missing.

Optional:

- **aissegments** and **aisdb** -- needed for the in-plugin AIS
  ingestion pipeline (Database setup wizard).  Listed in
  ``requirements.txt`` and installed automatically by qpip; see
  :ref:`database-setup` for the full ingestion workflow.  ``aisdb`` is
  only listed for Python older than 3.13 (it has no wheels for newer
  versions); without it the NMEA and aisdb-CSV inputs cannot be read,
  the generic CSV and Parquet inputs still can.
- **pyarrow** -- only for ingesting ``.parquet`` AIS files; its version
  must match the Arrow library bundled with QGIS (see
  :ref:`database-setup`).


Installing the Plugin (recommended)
====================================

Via QGIS Plugin Manager
------------------------

This is the normal install path.  QGIS handles both the plugin and its
Python-package dependencies.

1. Open QGIS.
2. **Plugins** -> **Manage and Install Plugins...**.
3. Search for "OMRAT" or "Maritime Risk" and click **Install Plugin**.

When OMRAT installs, QGIS sees ``plugin_dependencies=qpip`` in
``metadata.txt`` and offers to install **qpip** (the QGIS Python
Package Installer, https://github.com/opengisch/qpip) if it isn't
already present.  Accept the prompt.

qpip then reads OMRAT's ``requirements.txt`` and installs any missing
Python packages into the QGIS Python environment automatically.  On
first run you may see a short progress dialog while pip fetches the
packages.

Enable OMRAT in the plugin manager (the checkbox) and you'll see the
OMRAT icon appear in the QGIS toolbar.

.. note::

   If qpip's first-run install is blocked by a corporate firewall or
   proxy, you can install the dependencies manually -- see
   :ref:`installation-manual-deps`.

From Source (developers)
------------------------

1. Clone the repository::

      git clone https://github.com/axelande/OMRAT.git

2. Copy or symlink the OMRAT folder into your QGIS plugin directory:

   - **Linux**: ``~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/``
   - **Windows**: ``%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\``
   - **macOS**: ``~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/``

   QGIS 4 keeps its profiles in a ``QGIS4`` folder instead of ``QGIS3``.

3. Restart QGIS and enable the plugin from the Plugin Manager.  qpip
   will still be offered on first launch and handle the dependency
   install as described above.

4. For running the test suite outside QGIS, see
   :ref:`installation-manual-deps` -- the dev-only packages
   (``pytest``, ``pytest-cov``, ``pytest-qgis``, ``flake8``,
   ``bandit``, ...) are in ``requirements_dev.txt`` and are **not**
   needed to run the plugin itself.


.. _installation-manual-deps:

Manual dependency install (fallback)
=====================================

Only needed if qpip isn't available or is blocked (e.g. offline
install, restrictive network).

On Windows with OSGeo4W::

    C:\OSGeo4W\bin\python3.exe -m pip install -r requirements.txt

On Linux/macOS with the system QGIS Python::

    python3 -m pip install --user -r requirements.txt

``requirements.txt`` holds only the runtime packages.  To run the test
suite, install ``requirements_dev.txt`` the same way.


Verifying the Installation
==========================

After installation, you should see the OMRAT icon on its own **OMRAT**
toolbar in QGIS.  Click it to open the OMRAT window (a separate window
next to the QGIS main window).  It should have the menus File,
Settings, Consequence and Help, and the tabs Routes, Traffic Data,
Depths, Objects, Run Analysis, Drift Analysis and Compare.

If the plugin fails to load:

- Check **Plugins** -> **Installed** for a red error badge on OMRAT;
  hover it to see the import error.
- The most common cause is a missing Python dependency that qpip did
  not install.  Try the :ref:`installation-manual-deps` fallback.
- QGIS message log (**View** -> **Panels** -> **Log Messages Panel**)
  has an **OMRAT** tab with plugin-side errors.
