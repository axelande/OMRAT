.. _installation:

============
Installation
============

Prerequisites
=============

OMRAT requires:

- **QGIS 3.30+** (``qgisMinimumVersion=3.30`` in ``metadata.txt``).
  The plugin is Qt 6 compatible (``supportsQt6=True``).
- **Python 3.9+** (bundled with QGIS).
- A handful of scientific Python packages listed in ``requirements.txt``
  (NumPy, SciPy, Shapely, GeoPandas, Matplotlib, Pydantic, pyproj,
  Requests, Pandas).  If you install OMRAT via the QGIS Plugin Manager
  the **qpip** plugin dependency installs these automatically — see
  below.

Optional:

- **psycopg2** -- for PostgreSQL/AIS database connectivity (only needed
  if you import traffic from a remote AIS database).


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
   - **Windows**: ``%APPDATA%\\QGIS\\QGIS3\\profiles\\default\\python\\plugins\\``
   - **macOS**: ``~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/``

3. Restart QGIS and enable the plugin from the Plugin Manager.  qpip
   will still be offered on first launch and handle the dependency
   install as described above.

4. For running the test suite outside QGIS, see
   :ref:`installation-manual-deps` -- the dev-only packages
   (``pytest``, ``pytest-cov``, ``pytest-qgis``) are in
   ``requirements.txt`` but are **not** needed to run the plugin
   itself.


.. _installation-manual-deps:

Manual dependency install (fallback)
=====================================

Only needed if qpip isn't available or is blocked (e.g. offline
install, restrictive network).

On Windows with OSGeo4W::

    C:\\OSGeo4W\\bin\\python3.exe -m pip install -r requirements.txt

On Linux/macOS with the system QGIS Python::

    python3 -m pip install --user -r requirements.txt

You can also install just the runtime packages (skip the ``pytest*``
entries) if you don't plan to run the test suite.


Verifying the Installation
==========================

After installation, you should see the OMRAT icon in the QGIS toolbar.
Click it to open the plugin dock widget.  The widget should display
tabs for: Route, Traffic, Depths, Objects, Distributions, Results, and
Drift Analysis.

If the plugin fails to load:

- Check **Plugins** -> **Installed** for a red error badge on OMRAT;
  hover it to see the import error.
- The most common cause is a missing Python dependency that qpip did
  not install.  Try the :ref:`installation-manual-deps` fallback.
- QGIS message log (**View** -> **Panels** -> **Log Messages Panel**)
  has an **OMRAT** tab with plugin-side errors.
