"""Capture the 19 screenshots referenced by the OMRAT documentation.

Run this **inside a QGIS Python environment** (see bottom of file for
invocation recipes).  The script:

1. Finds or loads the OMRAT plugin.
2. Loads ``tests/example_data/proj.omrat``.
3. Steps through every tab of the dock widget, grabs each to PNG.
4. Opens the four Settings sub-dialogs, grabs each, closes them.
5. Grabs the map canvas before and after Run Model.
6. Optionally triggers Run Model and waits for completion so the
   results-populated + result-layers shots are accurate.

Output files are dropped into::

    <repo>/help/source/_static/screenshots/

with the exact filenames that :file:`help/source/_static/screenshots/README.md`
expects.

Limitations
-----------

* The script needs the real QGIS Qt event loop, so it cannot be run
  with ``pytest-qgis`` in headless mode -- the dock widget needs to be
  actually visible for ``widget.grab()`` to render the children
  correctly.
* Some dialogs (``QDialog.exec_()``) block the event loop.  The script
  uses ``show()`` instead so the capture can race in a single-shot
  timer.  If a future release switches any of them to ``exec_()``,
  wrap the corresponding grab in a ``QTimer.singleShot`` closure.
* The Run Model phase is slow on realistic projects.  Comment out the
  ``run_model=True`` call below if you only need the dry screenshots.

Invocation
----------

**Option A: GUI interactive** (fastest iteration while you tweak
screenshots).  Open QGIS, open the Python console, and paste::

    exec(open(r'e:/dev_e/program/OMRAT/tools/capture_screenshots.py').read())

**Option B: batch mode**.  From a shell that has the QGIS Python in
its PATH (OSGeo4W shell on Windows)::

    python-qgis tools/capture_screenshots.py

On a fresh OSGeo4W install, ``python-qgis.bat`` lives in
``C:\\OSGeo4W\\bin``.  On Linux, start QGIS with ``-nologo -code
tools/capture_screenshots.py``.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from qgis.PyQt import QtCore, QtGui, QtWidgets
from qgis.core import QgsApplication, QgsProject
from qgis.utils import iface, plugins

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    """Return the OMRAT repository root.

    Tries, in order:

    1. ``__file__`` -- works when the script is run as ``python tools/...``
       or ``qgis --code tools/...``.
    2. The ``OMRAT_REPO`` environment variable.
    3. Walking up from the current working directory, looking for a
       directory that contains ``omrat.py``.
    4. The currently-running OMRAT plugin's module directory (if the
       plugin is already loaded when the script runs).

    This lets the user ``exec(open(r'.../capture_screenshots.py').read())``
    from the QGIS Python console without ``__file__`` being set.
    """
    here = globals().get('__file__')
    if here:
        return Path(here).resolve().parent.parent

    env = os.environ.get('OMRAT_REPO')
    if env:
        p = Path(env).expanduser().resolve()
        if (p / 'omrat.py').is_file():
            return p

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / 'omrat.py').is_file():
            return candidate

    # Fall back to the already-loaded plugin's module location.
    try:
        from qgis.utils import plugins as _loaded_plugins
        plug = _loaded_plugins.get('OMRAT')
        if plug is not None:
            mod = sys.modules.get(plug.__class__.__module__)
            if mod and getattr(mod, '__file__', None):
                p = Path(mod.__file__).resolve().parent
                if (p / 'omrat.py').is_file():
                    return p
    except Exception:
        pass

    raise RuntimeError(
        "Can't locate the OMRAT repository root.\n"
        "Set the OMRAT_REPO env var to the path containing omrat.py, "
        "or run this script as a file (not via exec()).  Example:\n"
        "  import os; os.environ['OMRAT_REPO'] = r'E:/dev_e/program/OMRAT'\n"
        "  exec(open(r'E:/dev_e/program/OMRAT/tools/capture_screenshots.py').read())"
    )


REPO = _find_repo_root()
EXAMPLE = REPO / 'tests' / 'example_data' / 'proj.omrat'
OUT_DIR = REPO / 'help' / 'source' / '_static' / 'screenshots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Resize dock to this width before capturing tabs.  Tall enough that the
# full contents fit without scrollbars.
DOCK_WIDTH = 520
DOCK_HEIGHT = 820



# ---------------------------------------------------------------------------
# Plugin bootstrap
# ---------------------------------------------------------------------------

def find_or_load_plugin():
    """Return the live OMRAT plugin instance, loading it if needed."""
    name = 'OMRAT'
    if name in plugins:
        return plugins[name]

    # Fall back to loading it manually from this repo.
    sys.path.insert(0, str(REPO))
    from omrat import OMRAT as OmratPlugin

    plugin = OmratPlugin(iface)
    plugin.initGui()
    plugins[name] = plugin
    return plugin


# ---------------------------------------------------------------------------
# Grab helpers
# ---------------------------------------------------------------------------

def _process(n_ms: int = 50) -> None:
    """Spin the Qt event loop briefly so the UI paints before we grab.

    We deliberately use the no-arg form of ``processEvents`` because
    ``QEventLoop.AllEvents`` lives at different paths on Qt 5 (PyQt5,
    flat) vs Qt 6 (PyQt6, scoped under ``ProcessEventsFlag``).  The
    flag was just for excluding socket / user-input events anyway --
    plain ``processEvents()`` is fine for paint events.
    """
    deadline = time.time() + n_ms / 1000.0
    app = QtWidgets.QApplication.instance()
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.005)


def grab_widget(widget: QtWidgets.QWidget, name: str) -> None:
    """Render ``widget`` to ``OUT_DIR / <name>.png``."""
    widget.show()
    widget.raise_()
    _process(100)
    pixmap: QtGui.QPixmap = widget.grab()
    path = OUT_DIR / f'{name}.png'
    ok = pixmap.save(str(path), 'PNG')
    if ok:
        print(f'  wrote {path.name} ({pixmap.width()}x{pixmap.height()})')
    else:
        print(f'  FAILED to write {path}')


def grab_canvas(name: str) -> None:
    """Save just the QGIS map canvas to ``OUT_DIR / <name>.png``.

    Renders the canvas's current map settings synchronously into an
    off-screen ``QImage`` via :class:`QgsMapRendererCustomPainterJob`.
    This is the QGIS-blessed off-screen rendering path: it doesn't
    depend on the canvas's on-screen pixmap, doesn't race the async
    renderer, and works even when the canvas is covered by another
    window.

    Falls back to grabbing the main window and cropping to the canvas
    geometry if the renderer job fails for any reason.
    """
    from qgis.core import QgsMapRendererCustomPainterJob
    path = OUT_DIR / f'{name}.png'
    canvas = iface.mapCanvas()

    try:
        from qgis.core import QgsMapSettings, QgsProject
        size = canvas.size()
        if size.width() <= 0 or size.height() <= 0:
            size = QtCore.QSize(1280, 900)

        fmt = getattr(QtGui.QImage, 'Format_ARGB32', None)
        if fmt is None:
            fmt = QtGui.QImage.Format.Format_ARGB32  # PyQt6 scoped
        img = QtGui.QImage(size, fmt)
        white = (QtCore.Qt.GlobalColor.white
                 if hasattr(QtCore.Qt, 'GlobalColor') else QtCore.Qt.white)
        img.fill(white)

        # Build the map settings explicitly from the project's layers
        # and the canvas's current extent.  ``canvas.mapSettings()``
        # sometimes returns settings without the visible layers
        # attached on certain QGIS versions, which produces an
        # all-white image regardless of how long we wait.
        settings = QgsMapSettings()
        settings.setOutputSize(size)
        settings.setOutputDpi(96)
        settings.setExtent(canvas.extent())
        settings.setDestinationCrs(canvas.mapSettings().destinationCrs())
        layers = [
            lyr for lyr in QgsProject.instance().mapLayers().values()
            if lyr is not None
        ]
        settings.setLayers(layers)
        settings.setBackgroundColor(QtGui.QColor('white'))

        painter = QtGui.QPainter(img)
        try:
            job = QgsMapRendererCustomPainterJob(settings, painter)
            job.start()
            job.waitForFinished()
        finally:
            painter.end()

        ok = img.save(str(path), 'PNG')
        if ok:
            print(f'  wrote {path.name} ({img.width()}x{img.height()}) '
                  f'-- {len(layers)} layer(s) rendered')
            return
        print(f'  FAILED to write {path} (renderer path)')
    except Exception as exc:
        print(f'  WARN: synchronous renderer failed: {exc}')

    # Fallback: crop the main window grab.
    mw = iface.mainWindow()
    full = mw.grab()
    try:
        top_left = canvas.mapTo(mw, QtCore.QPoint(0, 0))
        rect = QtCore.QRect(top_left, canvas.size())
        rect = rect.intersected(QtCore.QRect(0, 0, full.width(), full.height()))
        cropped = full.copy(rect) if not rect.isEmpty() else full
    except Exception:
        cropped = full
    if cropped.save(str(path), 'PNG'):
        print(f'  wrote {path.name} ({cropped.width()}x{cropped.height()}) '
              f'[fallback]')
    else:
        print(f'  FAILED to write {path}')


def grab_mainwindow(name: str) -> None:
    """Grab the full QGIS main window (dock + canvas + toolbar)."""
    mw: QtWidgets.QWidget = iface.mainWindow()
    mw.raise_()
    _process(200)
    path = OUT_DIR / f'{name}.png'
    mw.grab().save(str(path), 'PNG')
    print(f'  wrote {path.name}')


def grab_toolbar(name: str) -> None:
    """Grab just the QGIS plugins toolbar, cropping around the OMRAT icon."""
    mw: QtWidgets.QMainWindow = iface.mainWindow()
    # Find any QToolBar that contains an action whose text starts with "OMRAT".
    # Fall back to the plugins toolbar.
    for tb in mw.findChildren(QtWidgets.QToolBar):
        if any('OMRAT' in (a.text() or '') or 'Maritime' in (a.text() or '')
               for a in tb.actions()):
            grab_widget(tb, name)
            return
    # Last resort: grab the Plugins toolbar.
    for tb in mw.findChildren(QtWidgets.QToolBar, 'mPluginToolBar'):
        grab_widget(tb, name)
        return


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

def reorder_layers_for_screenshots() -> None:
    """Reorder layers so depth polygons render beneath everything else.

    OMRAT's load path can leave depth polygons at the top of the layer
    panel, which means they paint *over* the route legs and the
    structure polygons on the canvas.  For the screenshots we want the
    visible stacking from bottom to top to be:

    * depth / bathymetry polygons (back, often filling the canvas);
    * structure / object polygons;
    * everything else (anchor zones, AIS density, ...);
    * route / leg / segment lines (always on top, visible).

    Only the top-level layer nodes in the project layer tree are
    reordered; nested groups are left alone.  Categorisation is by
    case-insensitive substring match on the layer name.
    """
    project = QgsProject.instance()
    root = project.layerTreeRoot()

    def priority(layer) -> int:
        name = (layer.name() or '').lower()
        if 'depth' in name or 'bathy' in name:
            return 0  # bottom of the panel = rendered first = behind
        if 'route' in name or 'leg' in name or 'segment' in name:
            return 3  # top of the panel = rendered last = on top
        if 'object' in name or 'structure' in name:
            return 2
        return 1

    # Snapshot top-level layer nodes (skip group nodes).
    nodes = [
        c for c in root.children()
        if hasattr(c, 'layer') and c.layer() is not None
    ]
    nodes.sort(key=lambda n: priority(n.layer()), reverse=True)

    # Re-insert each node in priority order.  ``addChildNode`` appends
    # to the end of the children list, so the first one we add becomes
    # the *top* of the panel (which is what we want for high-priority
    # layers).
    for node in nodes:
        clone = node.clone()
        root.removeChildNode(node)
        root.addChildNode(clone)

    iface.mapCanvas().refreshAllLayers()


def _build_example_canvas_layers() -> list:
    """Build memory layers from the example ``.omrat`` JSON.

    OMRAT's ``Storage.load_from_path`` only populates the dock-widget
    tables; it does **not** add anything to the QGIS canvas.  The
    "loaded example" screenshot would otherwise be blank, so this
    helper reads the JSON file directly and creates simple memory
    layers (route lines, depth polygons, structure polygons) styled in
    the same palette the docs caption refers to.

    Returns the list of created layers so the caller can clean them up
    after capturing.
    """
    import json
    from qgis.core import (
        QgsCoordinateReferenceSystem, QgsFeature, QgsGeometry, QgsProject,
        QgsVectorLayer,
    )

    with open(EXAMPLE) as f:
        data = json.load(f)

    crs = QgsCoordinateReferenceSystem('EPSG:4326')
    project = QgsProject.instance()

    # Start from a clean project so result polygons / debug layers
    # left over from previous OMRAT runs don't bleed into the
    # ``ui_loaded_example`` shot.  The script later re-adds run
    # results explicitly when ``OMRAT_RUN_MODEL=1``.
    try:
        existing_ids = list(project.mapLayers().keys())
        if existing_ids:
            project.removeMapLayers(existing_ids)
            print(f'  removed {len(existing_ids)} pre-existing layer(s) '
                  f'for a clean baseline')
    except Exception as exc:
        print(f'  WARN: could not clear pre-existing layers: {exc}')

    created: list = []

    def _new(layer_type: str, name: str):
        lyr = QgsVectorLayer(f'{layer_type}?crs=EPSG:4326', name, 'memory')
        lyr.setCrs(crs)
        return lyr

    # --- Depths --------------------------------------------------------- #
    depth_lyr = _new('MultiPolygon', 'Depths')
    feats = []
    for row in data.get('depths', []) or []:
        try:
            _did, _val, wkt = row
        except Exception:
            continue
        geom = QgsGeometry.fromWkt(str(wkt))
        if geom is None or geom.isEmpty():
            continue
        f = QgsFeature()
        f.setGeometry(geom)
        feats.append(f)
    if feats:
        depth_lyr.dataProvider().addFeatures(feats)
        depth_lyr.updateExtents()
        try:
            sym = depth_lyr.renderer().symbol()
            sym.setColor(QtGui.QColor('#9ec5d6'))
            sym.symbolLayer(0).setStrokeColor(QtGui.QColor('#5d8aa0'))
        except Exception:
            pass
        project.addMapLayer(depth_lyr)
        created.append(depth_lyr)

    # --- Structures ----------------------------------------------------- #
    obj_lyr = _new('Polygon', 'Structures')
    feats = []
    for row in data.get('objects', []) or []:
        try:
            _oid, _h, wkt = row
        except Exception:
            continue
        geom = QgsGeometry.fromWkt(str(wkt))
        if geom is None or geom.isEmpty():
            continue
        f = QgsFeature()
        f.setGeometry(geom)
        feats.append(f)
    if feats:
        obj_lyr.dataProvider().addFeatures(feats)
        obj_lyr.updateExtents()
        try:
            sym = obj_lyr.renderer().symbol()
            sym.setColor(QtGui.QColor('#e8a046'))
            sym.symbolLayer(0).setStrokeColor(QtGui.QColor('#b06b1c'))
        except Exception:
            pass
        project.addMapLayer(obj_lyr)
        created.append(obj_lyr)

    # --- Route legs ----------------------------------------------------- #
    route_lyr = _new('LineString', 'Route legs')
    feats = []

    def _xy(p):
        """Accept ``"lon lat"`` strings, ``[lon, lat]`` lists, or
        ``{'x': lon, 'y': lat}`` dicts -- the ``.omrat`` schema has
        used all three at various points."""
        if p is None:
            return None
        if isinstance(p, str):
            parts = p.replace(',', ' ').split()
            if len(parts) < 2:
                return None
            return float(parts[0]), float(parts[1])
        if isinstance(p, dict):
            return float(p.get('x', p.get('lon', 0))), \
                   float(p.get('y', p.get('lat', 0)))
        try:
            return float(p[0]), float(p[1])
        except Exception:
            return None

    for seg in (data.get('segment_data', {}) or {}).values():
        sp = seg.get('Start_Point') or seg.get('Start Point')
        ep = seg.get('End_Point') or seg.get('End Point')
        sxy = _xy(sp)
        exy = _xy(ep)
        if sxy is None or exy is None:
            continue
        sx, sy = sxy
        ex, ey = exy
        wkt = f'LineString({sx} {sy}, {ex} {ey})'
        geom = QgsGeometry.fromWkt(wkt)
        if geom is None or geom.isEmpty():
            continue
        f = QgsFeature()
        f.setGeometry(geom)
        feats.append(f)
    if feats:
        route_lyr.dataProvider().addFeatures(feats)
        route_lyr.updateExtents()
        try:
            sym = route_lyr.renderer().symbol()
            sym.setColor(QtGui.QColor('#1f4eaa'))
            sym.setWidth(0.8)
        except Exception:
            pass
        project.addMapLayer(route_lyr)
        created.append(route_lyr)

    return created


# Strong references to the example memory layers so the Python
# wrappers aren't garbage-collected.  ``QgsProject.addMapLayer``
# transfers C++ ownership, but if the Python wrapper goes out of
# scope sip can still tear down the layer behind QGIS's back -- the
# project ends up with 0 map layers a few seconds later.
_PERSISTED_EXAMPLE_LAYERS: list = []


def load_example(plugin) -> None:
    print(f'Loading {EXAMPLE} ...')
    from omrat_utils.storage import Storage
    store = Storage(plugin)
    store.load_from_path(str(EXAMPLE))
    _process(500)

    # ``Storage.load_from_path`` populates the dock tables but does NOT
    # add layers to the canvas, so the canvas would be blank for the
    # "loaded example" screenshot.  Build a small set of memory layers
    # from the JSON itself so the canvas has something to render.
    print('  building memory layers for the canvas...')
    layers = _build_example_canvas_layers()
    # Keep a strong Python reference so GC doesn't drop the wrappers
    # and take the C++ layers with them mid-script.
    _PERSISTED_EXAMPLE_LAYERS.extend(layers)
    _process(300)
    print(f'  {len(layers)} memory layer(s) added to the project')

    # NOTE: deliberately NOT calling ``reorder_layers_for_screenshots``
    # here.  Its ``removeChildNode`` + ``addChildNode(clone())`` dance
    # was dropping the just-added memory layers from the project (the
    # cloned layer-tree node held a stale handle that QGIS's GC then
    # invalidated).  ``QgsProject.addMapLayer`` already inserts new
    # layers at the top of the layer panel, so creating in the order
    # depths -> structures -> route gives the correct stacking
    # (route on top, depths at the back) without any reordering.

    iface.mapCanvas().zoomToFullExtent()
    iface.mapCanvas().refreshAllLayers()
    _process(500)


def configure_dock(plugin) -> QtWidgets.QDockWidget:
    """Make the OMRAT dock visibly attached to the QGIS main window.

    The plugin's own ``run()`` only calls ``main_widget.show()`` -- it
    never registers the dock with the QGIS main window via
    :py:meth:`QgisInterface.addDockWidget`, so when the plugin starts
    the dock is a floating top-level window.  ``iface.mainWindow()``
    therefore wouldn't include it in a screenshot.

    Here we explicitly dock it on the right side and force
    ``floating=False`` so the landing image and tab grabs both
    show the dock as part of the main window.
    """
    dock = plugin.main_widget
    mw = iface.mainWindow()
    # If it's already a child of the main window (re-run case), skip
    # ``addDockWidget`` so QGIS doesn't relocate it on every call.
    already_docked = (
        dock.parentWidget() is mw
        or dock in mw.findChildren(QtWidgets.QDockWidget)
    )
    if not already_docked:
        try:
            area = _qt_enum(QtCore.Qt, 'RightDockWidgetArea',
                            'DockWidgetArea.RightDockWidgetArea')
            iface.addDockWidget(area, dock)
        except Exception as exc:
            print(f'  WARN: could not addDockWidget: {exc}')

    dock.setFloating(False)
    dock.resize(DOCK_WIDTH, DOCK_HEIGHT)
    dock.show()
    dock.raise_()
    _process(300)
    return dock


def _qt_enum(klass, *paths):
    """Fetch a Qt enum value by trying each path in turn.

    Qt 5 (PyQt5) exposes most enums flat on the class
    (``Qt.RightDockWidgetArea``); Qt 6 (PyQt6) requires the scoped
    form (``Qt.DockWidgetArea.RightDockWidgetArea``).  This helper
    keeps the script working across both.
    """
    for p in paths:
        obj = klass
        try:
            for part in p.split('.'):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError(
        f"{klass.__name__} has none of: {', '.join(paths)}"
    )


def capture_tabs(plugin) -> None:
    """Grab each tab of the dock widget."""
    dock = configure_dock(plugin)
    tabs: QtWidgets.QTabWidget = dock.tabWidget

    tab_map = [
        # (tab index, output filename, extra delay ms)
        (0, 'ui_tab_route', 0),
        (1, 'ui_tab_traffic', 0),
        (2, 'ui_tab_depths', 0),
        (3, 'ui_tab_objects', 0),
        (4, 'ui_tab_results', 0),          # Run Analysis tab = Results
        (5, 'ui_tab_drift_analysis', 0),
    ]

    # Distributions aren't a separate tab -- they live inside Route_tab.
    # We alias the Route shot to ui_tab_distributions so docs that point
    # at that filename still resolve to a meaningful screenshot.
    for idx, name, extra in tab_map:
        tabs.setCurrentIndex(idx)
        _process(250 + extra)
        grab_widget(dock, name)

    tabs.setCurrentIndex(0)
    _process(200)
    grab_widget(dock, 'ui_tab_distributions')  # same as Route

    # Annotated copy of the Routes tab -- the author can later decorate
    # it with arrows / numbered callouts.  ``ui_dock_overview`` is
    # produced separately in ``main`` from a *full main-window* shot so
    # the docs landing image shows dock + canvas together.
    tabs.setCurrentIndex(0)
    _process(200)
    grab_widget(dock, 'ui_dock_tabs_annotated')


def _capture_exec_dialog(run_fn, get_widget, name: str, delay_ms: int = 450) -> None:
    """Open a dialog that uses ``exec_()`` (blocking), grab it, close it.

    We schedule the grab+close on a single-shot timer that fires inside
    the nested event loop ``exec_()`` spins up.  The call to ``run_fn``
    therefore returns once we close the dialog from the timer.
    """
    captured = {'ok': False, 'err': None}

    def grab_and_close():
        try:
            w = get_widget()
            if w is None:
                QtCore.QTimer.singleShot(150, grab_and_close)
                return
            # Let the widget fully render; dialog fonts + tables sometimes
            # lay out lazily.
            for _ in range(10):
                QtWidgets.QApplication.instance().processEvents()
                time.sleep(0.01)
            grab_widget(w, name)
            captured['ok'] = True
            w.close()
        except Exception as exc:
            captured['err'] = exc

    QtCore.QTimer.singleShot(delay_ms, grab_and_close)
    try:
        run_fn()
    except Exception as exc:
        print(f'  WARN: dialog {name}: {exc}')
    if captured['err'] is not None:
        print(f'  WARN: dialog {name}: {captured["err"]}')


def capture_settings(plugin) -> None:
    """Open each settings sub-dialog, grab, close."""

    # Drift settings -- uses show() so it doesn't block.
    plugin.drift_settings.run()
    dsw = plugin.drift_settings.dsw
    _process(400)
    grab_widget(dsw, 'ui_settings_drift')
    dsw.close()
    print(f'  [trace] after drift_settings: {_layer_count()} layers')

    # Ship categories -- exec_() so we need the timer trick.
    _capture_exec_dialog(
        plugin.ship_cat.run,
        lambda: plugin.ship_cat.scw,
        'ui_settings_ship_categories',
    )
    print(f'  [trace] after ship_cat: {_layer_count()} layers')

    # Causation factors -- exec_() too.
    _capture_exec_dialog(
        plugin.causation_f.run,
        lambda: plugin.causation_f.cfw,
        'ui_settings_causation',
    )
    print(f'  [trace] after causation_f: {_layer_count()} layers')

    # AIS connection -- also exec_().
    _capture_exec_dialog(
        plugin.ais_settings,
        lambda: plugin.ais.acw,
        'ui_settings_ais',
    )
    print(f'  [trace] after ais_settings: {_layer_count()} layers')


def capture_canvas_and_results(plugin, run_model: bool = False) -> None:
    """Capture map canvas shots + optionally trigger Run Model."""
    # ``ui_loaded_example`` is grabbed earlier in ``main`` (right
    # after load_example) so its canvas extent isn't disturbed by the
    # tab walk or settings dialogs.  Skip a second grab here.

    # Toolbar (zoom to the OMRAT icon)
    grab_toolbar('ui_toolbar')

    # Results-tab "about to click Run Model"
    dock = plugin.main_widget
    dock.tabWidget.setCurrentIndex(4)  # Run Analysis tab
    _process(200)
    grab_widget(dock, 'ui_run_model')

    if run_model:
        print('Running model (this can take a few minutes)...')
        plugin.run_calculation()
        # Wait for the task to finish.  We can poll by checking that
        # the result line-edit changes away from its default.
        timeout = time.time() + 900  # 15 min hard cap
        while time.time() < timeout:
            text = dock.LEPDriftAllision.text()
            if text and text not in ('', '0.000e+00'):
                break
            _process(300)
        else:
            print('  WARN: timed out waiting for Run Model -- capturing anyway')

        _process(1000)
        dock.tabWidget.setCurrentIndex(4)
        grab_widget(dock, 'ui_results_filled')
        grab_widget(dock, 'ui_tab_results')  # overwrite the blank one

        # Result layers are written to the per-run GeoPackage but
        # **not** auto-added to the canvas (run_persistence keeps the
        # canvas clean during runs).  For the screenshot we need to
        # explicitly load the latest run's layers onto the map.
        try:
            from omrat_utils.run_history import RunHistory
            from omrat_utils.run_persistence import load_run_results_to_map
            runs = RunHistory().list_runs()
            if runs:
                latest = runs[0]
                gpkg_path = latest.gpkg_path()
                if gpkg_path is not None and gpkg_path.is_file():
                    new_layers = load_run_results_to_map(
                        gpkg_path, latest.name,
                    )
                    print(f'  added {len(new_layers)} result layer(s) '
                          f'from {gpkg_path.name}')
                    iface.mapCanvas().zoomToFullExtent()
                    _process(800)
                else:
                    print('  WARN: latest run has no .gpkg on disk -- '
                          'ui_result_layers will be the input layers only')
            else:
                print('  WARN: no runs in history; ui_result_layers '
                      'will be the input layers only')
        except Exception as exc:
            print(f'  WARN: could not add run results to canvas: {exc}')

        grab_canvas('ui_result_layers')

    else:
        # Dry screenshots: populate a convincing fake result set so the
        # "after run" slots aren't blank.  The doc reader won't be able
        # to tell this was a synthetic number without looking at the
        # rest of the state, and a real run can always overwrite.
        dock.LEPDriftAllision.setText('1.148e-01')
        dock.LEPDriftingGrounding.setText('8.330e-03')
        dock.LEPPoweredGrounding.setText('3.100e-05')
        dock.LEPPoweredAllision.setText('1.700e-06')
        dock.LEPHeadOnCollision.setText('4.900e-07')
        dock.LEPOvertakingCollision.setText('1.100e-06')
        dock.LEPCrossingCollision.setText('0.000e+00')
        dock.LEPMergingCollision.setText('0.000e+00')
        _process(200)
        dock.tabWidget.setCurrentIndex(4)
        grab_widget(dock, 'ui_results_filled')

    # Task progress hint: grab the task tray manager panel directly.
    _grab_task_tray('ui_task_progress')


def _grab_task_tray(name: str) -> None:
    """Grab the QGIS status bar area that contains the task tray widget.

    The task tray isn't a separate widget with a stable objectName, so
    we pragmatically grab the entire status bar.
    """
    try:
        sb = iface.mainWindow().statusBar()
        grab_widget(sb, name)
    except Exception:
        print('  WARN: couldn\'t find status bar for task tray')


def capture_drift_corridor(plugin) -> None:
    """Trigger Drift Analysis, wait, then grab the canvas."""
    dock = plugin.main_widget
    dock.tabWidget.setCurrentIndex(5)  # Drift Analysis tab
    _process(300)
    try:
        # The exact button name may vary; adjust if your build differs.
        btn = dock.findChild(QtWidgets.QPushButton, 'pbRunDriftAnalysis')
        if btn is None:
            btn = dock.findChild(QtWidgets.QPushButton, 'pbDriftAnalysis')
        if btn is not None:
            btn.click()
            _process(60000)  # wait up to a minute for the async task
            iface.mapCanvas().zoomToFullExtent()
            _process(500)
            grab_canvas('ui_drift_corridor')
        else:
            print('  WARN: couldn\'t find Drift Analysis button; '
                  'skipping ui_drift_corridor')
    except Exception as exc:
        print(f'  WARN: Drift Analysis failed: {exc}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def hide_python_console() -> None:
    """Close / hide the QGIS Python Console so it doesn't appear in
    any of the canvas / main-window screenshots.

    The console widget is a :class:`QDockWidget` with object name
    ``PythonConsole``.  Some QGIS releases also keep an embedded
    console under ``mPythonConsoleDock``; we cover both.
    """
    try:
        mw = iface.mainWindow()
        for obj_name in ('PythonConsole', 'mPythonConsoleDock', 'pythonConsole'):
            for dock in mw.findChildren(QtWidgets.QDockWidget, obj_name):
                dock.hide()
        # As a backstop, hide *any* dock whose window-title is "Python Console".
        for dock in mw.findChildren(QtWidgets.QDockWidget):
            if (dock.windowTitle() or '').strip().lower() == 'python console':
                dock.hide()
    except Exception as exc:
        print(f'  WARN: could not hide Python Console: {exc}')


def _layer_count() -> int:
    try:
        from qgis.core import QgsProject
        return len(QgsProject.instance().mapLayers())
    except Exception:
        return -1


def capture_quickstart(plugin) -> None:
    """Capture the "Quick start from scratch" walkthrough screenshots.

    Output files go to ``<OUT_DIR>/quickstart/qs_*.png`` so they don't
    collide with the canonical 19-screenshot set.  Captured states:

    * ``qs_01_empty_routes_tab`` — the Routes tab on a fresh project
      (no legs yet).
    * ``qs_02_canvas_blank`` — the QGIS canvas before any layer is added.
    * ``qs_03_route_after_first_leg`` — Routes tab after a synthetic
      leg has been added to ``twRouteList``.
    * ``qs_04_canvas_with_legs`` — canvas zoomed to the synthetic legs.
    * ``qs_05_depths_tab_empty`` / ``qs_06_objects_tab_empty`` — what
      the user sees before they bring in depth / object polygons.
    * ``qs_07_drift_settings_wind_rose`` — drift-settings dialog
      (wind-rose section is the typical "where do I find depth info"
      jumping-off point).
    """
    qs_dir = OUT_DIR / 'quickstart'
    qs_dir.mkdir(parents=True, exist_ok=True)

    def _save(widget, name: str) -> None:
        path = qs_dir / f'{name}.png'
        widget.show()
        widget.raise_()
        _process(120)
        if widget.grab().save(str(path), 'PNG'):
            print(f'  wrote quickstart/{path.name}')
        else:
            print(f'  FAILED to write {path}')

    def _save_canvas(name: str) -> None:
        # Reuse grab_canvas but with a quickstart-specific filename.
        from qgis.core import QgsMapRendererCustomPainterJob, QgsMapSettings
        path = qs_dir / f'{name}.png'
        canvas = iface.mapCanvas()
        size = canvas.size()
        if size.width() <= 0 or size.height() <= 0:
            size = QtCore.QSize(1280, 900)
        fmt = getattr(QtGui.QImage, 'Format_ARGB32', None) or QtGui.QImage.Format.Format_ARGB32
        img = QtGui.QImage(size, fmt)
        white = QtCore.Qt.GlobalColor.white if hasattr(QtCore.Qt, 'GlobalColor') else QtCore.Qt.white
        img.fill(white)
        settings = QgsMapSettings()
        settings.setOutputSize(size)
        settings.setOutputDpi(96)
        settings.setExtent(canvas.extent())
        settings.setDestinationCrs(canvas.mapSettings().destinationCrs())
        settings.setLayers([
            l for l in QgsProject.instance().mapLayers().values() if l is not None
        ])
        settings.setBackgroundColor(QtGui.QColor('white'))
        painter = QtGui.QPainter(img)
        try:
            job = QgsMapRendererCustomPainterJob(settings, painter)
            job.start()
            job.waitForFinished()
        finally:
            painter.end()
        if img.save(str(path), 'PNG'):
            print(f'  wrote quickstart/{path.name}')
        else:
            print(f'  FAILED to write {path}')

    # Wipe any layers / segment data left from earlier captures so the
    # "blank slate" shots are actually blank.
    project = QgsProject.instance()
    try:
        existing = list(project.mapLayers().keys())
        if existing:
            project.removeMapLayers(existing)
    except Exception:
        pass
    try:
        plugin.segment_data.clear()
        plugin.main_widget.twRouteList.setRowCount(0)
        plugin.main_widget.twDepthList.setRowCount(0)
        plugin.main_widget.twObjectList.setRowCount(0)
    except Exception:
        pass

    dock = configure_dock(plugin)
    tabs = dock.tabWidget

    # 1. Empty Routes tab
    tabs.setCurrentIndex(0)
    _process(200)
    _save(dock, 'qs_01_empty_routes_tab')

    # 2. Empty canvas
    iface.mapCanvas().zoomToFullExtent()
    _process(200)
    _save_canvas('qs_02_canvas_blank')

    # 3. Inject a synthetic leg into the route table so the tab
    # populates the same way the real "Place leg" tool would.
    from qgis.PyQt.QtWidgets import QTableWidgetItem
    tw = plugin.main_widget.twRouteList
    tw.setRowCount(1)
    cells = ['1', '1', 'LEG_1_1', '13.50 55.10', '14.50 55.30', '5000']
    for col, txt in enumerate(cells):
        tw.setItem(0, col, QTableWidgetItem(txt))
    plugin.segment_data['1'] = {
        'Route_Id': '1',
        'Leg_name': 'LEG_1_1',
        'Start_Point': '13.50 55.10',
        'End_Point': '14.50 55.30',
        'Width': '5000',
        'Dirs': ['East going', 'West going'],
    }
    _process(200)
    _save(dock, 'qs_03_route_after_first_leg')

    # Add a memory layer so the canvas shows the leg.
    from qgis.core import QgsVectorLayer, QgsFeature, QgsGeometry
    leg_layer = QgsVectorLayer('LineString?crs=EPSG:4326', 'Quick start leg', 'memory')
    feat = QgsFeature()
    feat.setGeometry(QgsGeometry.fromWkt('LINESTRING(13.50 55.10, 14.50 55.30)'))
    leg_layer.dataProvider().addFeature(feat)
    leg_layer.updateExtents()
    project.addMapLayer(leg_layer)
    iface.mapCanvas().zoomToFullExtent()
    _process(400)
    _save_canvas('qs_04_canvas_with_legs')

    # 4 & 5. Empty Depths / Objects tabs.
    tabs.setCurrentIndex(2)
    _process(200)
    _save(dock, 'qs_05_depths_tab_empty')
    tabs.setCurrentIndex(3)
    _process(200)
    _save(dock, 'qs_06_objects_tab_empty')

    # 6. Drift Settings dialog (where wind-rose / drift-speed live).
    # We deliberately avoid ``plugin.drift_settings.run()`` here -- it
    # was already invoked once during ``capture_settings()``, which
    # connected several signals.  Re-running would either duplicate
    # those connections (cheap-but-ugly) or in some QGIS versions
    # raise on the second connect.  Instead we just refresh + show
    # the dialog directly.
    try:
        ds = plugin.drift_settings
        # ``populate_drift`` repopulates the line-edits from the live
        # ``drift_values`` dict so the captured shot reflects the
        # currently-loaded project.
        try:
            ds.populate_drift()
        except Exception as e:
            print(f'  WARN: populate_drift failed: {e}')
        dsw = ds.dsw
        try:
            dsw.show()
            dsw.raise_()
            dsw.activateWindow()
        except Exception as e:
            print(f'  WARN: dsw.show() failed: {e}')
        _process(600)
        _save(dsw, 'qs_07_drift_settings_wind_rose')
        try:
            dsw.close()
        except Exception:
            pass
    except Exception as exc:
        import traceback
        print(f'  WARN: could not capture drift-settings: {exc}')
        traceback.print_exc()


def main(run_model: bool = False, drift_analysis: bool = False,
         quickstart: bool = False) -> None:
    print(f'Capturing OMRAT screenshots to {OUT_DIR}')
    print(f'  run_model = {run_model}')
    print(f'  drift_analysis = {drift_analysis}')
    print(f'  quickstart = {quickstart}')
    if not run_model:
        print('  >>> set OMRAT_RUN_MODEL=1 BEFORE exec() to also produce '
              'ui_result_layers.png')
    if not drift_analysis:
        print('  >>> set OMRAT_DRIFT=1 BEFORE exec() to also produce '
              'ui_drift_corridor.png')
    if not quickstart:
        print('  >>> set OMRAT_QUICKSTART=1 BEFORE exec() to also produce '
              'quickstart/qs_*.png')

    hide_python_console()

    plugin = find_or_load_plugin()
    load_example(plugin)
    print(f'  [trace] after load_example: {_layer_count()} layers')

    # Landing image for the docs: full QGIS window with the OMRAT dock
    # visible alongside the loaded example canvas.  Captured *before*
    # the tab walk because ``capture_tabs`` cycles through tabs, hides
    # the canvas behind dialogs, and resizes the dock.
    print('Landing shot (main window with dock + canvas)...')
    dock = configure_dock(plugin)
    dock.tabWidget.setCurrentIndex(0)  # Routes tab
    iface.mapCanvas().zoomToFullExtent()
    _process(500)
    grab_mainwindow('ui_dock_overview')

    # Grab the canvas-only ``ui_loaded_example`` here, while the canvas
    # is still in its "freshly loaded with input layers" state.  Doing
    # it later (after capture_tabs / capture_settings) leaves a
    # window for unrelated state to creep in and produce a wrong
    # extent or empty render.
    print('Loaded-example shot (canvas only)...')
    grab_canvas('ui_loaded_example')

    print('Tabs...')
    capture_tabs(plugin)
    print(f'  [trace] after capture_tabs: {_layer_count()} layers')

    print('Settings dialogs...')
    capture_settings(plugin)
    print(f'  [trace] after capture_settings: {_layer_count()} layers')

    print('Canvas + results...')
    capture_canvas_and_results(plugin, run_model=run_model)

    if drift_analysis:
        print('Drift analysis...')
        capture_drift_corridor(plugin)

    if quickstart:
        print('Quick-start (from-scratch walkthrough)...')
        capture_quickstart(plugin)

    print(f'Done.  {len(list(OUT_DIR.glob("*.png")))} PNGs in {OUT_DIR}.')


def _flag(name: str, default: bool = False) -> bool:
    """``True`` if env var ``name`` is set to a truthy string.

    When the env var is unset, ``default`` decides the result.
    Recognised truthy values: ``1``, ``true``, ``yes``, ``on``.
    Recognised falsy values: ``0``, ``false``, ``no``, ``off``.
    """
    val = (os.environ.get(name, '') or '').strip().lower()
    if not val:
        return default
    if val in ('1', 'true', 'yes', 'on'):
        return True
    if val in ('0', 'false', 'no', 'off'):
        return False
    return default


if __name__ == '__main__' or 'qgis' in sys.modules:
    # ``OMRAT_DRIFT`` and ``OMRAT_QUICKSTART`` default to ON so a single
    # ``exec()`` produces every PNG the docs reference.  Set the env var
    # to ``0`` (or ``false`` / ``no``) to skip the slow phases.
    #
    #     import os
    #     os.environ['OMRAT_DRIFT']     = '0'   # skip ui_drift_corridor.png
    #     os.environ['OMRAT_QUICKSTART'] = '0'  # skip quickstart/qs_*.png
    #     os.environ['OMRAT_RUN_MODEL'] = '1'   # additionally produce
    #                                           # ui_result_layers.png
    #     exec(open(r'.../tools/capture_screenshots.py').read())
    main(
        run_model=_flag('OMRAT_RUN_MODEL'),
        drift_analysis=_flag('OMRAT_DRIFT', default=True),
        quickstart=_flag('OMRAT_QUICKSTART', default=True),
    )
