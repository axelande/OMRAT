"""Capture the 19 screenshots referenced by the OMRAT documentation.

Run this **inside a QGIS Python environment** (see bottom of file for
invocation recipes).  The script:

1. Finds the running OMRAT plugin (or loads it) and opens its window.
2. Loads ``tests/example_data/proj.omrat`` the way File > Load does
   (from a temporary copy), so OMRAT draws its own layers.
3. Captures the screen with QGIS and the OMRAT window side by side.
4. Steps through every tab of the OMRAT window, grabs each to PNG.
5. Opens the four Settings sub-dialogs, grabs each, closes them.
6. Grabs the map canvas before and after Run Model.
7. Optionally triggers Run Model and waits for completion so the
   results-populated + result-layers shots are accurate.

The OMRAT window is shot as the user sees it: a window of its own,
never docked into the QGIS main window (the plugin never docks it), at
the size that shows every tab without scroll bars.

The script replaces the model open in OMRAT and every layer on the map.
It stops when OMRAT has unsaved changes.

Output files are dropped into::

    <repo>/help/source/_static/screenshots/

with the exact filenames that :file:`help/source/_static/screenshots/README.md`
expects.

Limitations
-----------

* The script needs the real QGIS Qt event loop, so it cannot be run
  with ``pytest-qgis`` in headless mode -- the OMRAT window needs to be
  actually visible for ``widget.grab()`` to render the children
  correctly, and the landing image is a capture of the screen.
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
import shutil
import sys
import tempfile
import time
from pathlib import Path

from qgis.PyQt import QtCore, QtGui, QtWidgets
from qgis.core import QgsProject
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

# Landing image: QGIS and the OMRAT window side by side, each with a
# drawn title bar, on a plain background (pixels at 100 % scaling).
LANDING_GAP_PX = 12
LANDING_TITLE_PX = 30
LANDING_BACKGROUND = '#d5dbe3'
LANDING_FRAME = '#8a9099'
LANDING_TITLE_BG = '#f3f4f6'
# The widest the OMRAT window gets in the landing image, as a share of
# the screen width.
MAX_SIDE_SHARE = 0.5
# The Traffic tab shot is widened for more of the matrix, up to this
# width (the whole matrix needs ~2100 px: too small on a docs page).
MAX_TRAFFIC_WIDTH = 1200


# ---------------------------------------------------------------------------
# Plugin bootstrap
# ---------------------------------------------------------------------------


def _running_plugin():
    """The OMRAT instance QGIS loaded, whatever key it is registered under.

    QGIS keys a plugin by its folder name: ``Omrat`` for an install from
    the plugin repository, the checkout's name for a development link.
    Looking up only ``'OMRAT'`` missed the installed plugin and built a
    second instance next to it -- a second toolbar icon, and a window
    whose menu bar and buttons were never wired.
    """
    for plugin in plugins.values():
        if type(plugin).__name__ == 'OMRAT' and hasattr(plugin, 'main_widget'):
            return plugin
    return None


def find_or_load_plugin():
    """Return the OMRAT plugin with its window open, loading it if needed."""
    plugin = _running_plugin()
    if plugin is None:
        # Not enabled in this QGIS: load it from this repo.
        sys.path.insert(0, str(REPO))
        from omrat import OMRAT as OmratPlugin

        plugin = OmratPlugin(iface)
        plugin.initGui()
        plugins['OMRAT'] = plugin
    module = sys.modules.get(type(plugin).__module__)
    source = Path(getattr(module, '__file__', '') or '.').resolve().parent
    print(f'  OMRAT code: {source}')
    if source != REPO.resolve():
        print(f'  NOTE: that is not {REPO} -- deploy the plugin first if the '
              f'two differ, the screenshots show the code QGIS runs')
    # The toolbar icon's handler: opens the window, and on the first open
    # wires its buttons and builds the menu bar.
    plugin.run()
    _process(300)
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
    """Grab the full QGIS main window (canvas + panels + toolbars)."""
    mw: QtWidgets.QWidget = iface.mainWindow()
    mw.raise_()
    _process(200)
    path = OUT_DIR / f'{name}.png'
    mw.grab().save(str(path), 'PNG')
    print(f'  wrote {path.name}')


def _window_image(widget: QtWidgets.QWidget) -> QtGui.QImage:
    """``widget.grab()`` as an image in device pixels."""
    _process(100)
    image = widget.grab().toImage()
    image.setDevicePixelRatio(1.0)
    return image


def _with_title_bar(image: QtGui.QImage, title: str, scale: float) -> QtGui.QImage:
    """``image`` in a 1 px frame under a plain title bar with ``title``."""
    bar = round(LANDING_TITLE_PX * scale)
    out = QtGui.QImage(image.width() + 2, image.height() + bar + 2,
                       QtGui.QImage.Format.Format_ARGB32)
    out.fill(QtGui.QColor(LANDING_FRAME))
    painter = QtGui.QPainter(out)
    try:
        painter.fillRect(1, 1, image.width(), bar, QtGui.QColor(LANDING_TITLE_BG))
        font = painter.font()
        font.setPixelSize(round(13 * scale))
        painter.setFont(font)
        painter.setPen(QtGui.QColor('#202124'))
        pad = round(10 * scale)
        painter.drawText(
            QtCore.QRect(1 + pad, 1, image.width() - 2 * pad, bar),
            int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
            title,
        )
        painter.drawImage(1, 1 + bar, image)
    finally:
        painter.end()
    return out


def grab_desktop(plugin, name: str) -> None:
    """QGIS with the OMRAT window beside it: the docs landing image.

    Composed from the two windows' own ``grab()`` -- QGIS on the left, the
    OMRAT window on the right, each under a drawn title bar -- never a
    capture of the screen: that picks up whatever else is on it (another
    application in front, the wrong monitor), and the image goes into the
    public docs.  QGIS is resized so the pair is one screen wide and gets
    its own size and state back afterwards.
    """
    mw: QtWidgets.QWidget = iface.mainWindow()
    dock = plugin.main_widget
    screen = mw.screen() or QtWidgets.QApplication.primaryScreen()
    avail = screen.availableGeometry()
    saved = mw.saveGeometry()
    was_maximized = mw.isMaximized()
    try:
        show_omrat_window(plugin)  # every tab name, no scroll bars
        dock_w = min(dock.width(), int(avail.width() * MAX_SIDE_SHARE))
        height = dock.height()
        _resize_window(dock, QtCore.QSize(dock_w, height))
        mw.showNormal()
        _process(400)
        mw.resize(max(400, avail.width() - dock.width() - 3 * LANDING_GAP_PX - 4), height)
        _process(600)
        iface.mapCanvas().zoomToFullExtent()
        iface.mapCanvas().refreshAllLayers()
        _process(1500)

        left_raw, right_raw = _window_image(mw), _window_image(dock)
        scale = left_raw.width() / max(1, mw.width())  # device pixels per pixel
        left = _with_title_bar(left_raw, mw.windowTitle() or 'QGIS', scale)
        right = _with_title_bar(right_raw, dock.windowTitle() or 'OMRAT', scale)
        gap = round(LANDING_GAP_PX * scale)
        image = QtGui.QImage(left.width() + right.width() + 3 * gap,
                             max(left.height(), right.height()) + 2 * gap,
                             QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor(LANDING_BACKGROUND))
        painter = QtGui.QPainter(image)
        try:
            painter.drawImage(gap, gap, left)
            painter.drawImage(2 * gap + left.width(), gap, right)
        finally:
            painter.end()
        path = OUT_DIR / f'{name}.png'
        if image.save(str(path), 'PNG'):
            print(f'  wrote {path.name} ({image.width()}x{image.height()})')
        else:
            print(f'  FAILED to write {path}')
    finally:
        mw.restoreGeometry(saved)
        if was_maximized:
            mw.showMaximized()
        _process(400)
        show_omrat_window(plugin)


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

def _clean_slate(plugin) -> None:
    """An empty OMRAT model and an empty map.

    OMRAT's own ``clear_model`` first (its layers, tables and data), then
    whatever else is on the map.  Removing OMRAT's leg layers behind its
    back takes the legs out of the model too -- that is how the route
    table of the earlier screenshots ended up empty.
    """
    plugin.clear_model()
    project = QgsProject.instance()
    left = list(project.mapLayers().keys())
    if left:
        project.removeMapLayers(left)
        print(f'  removed {len(left)} other layer(s) for a clean map')
    _process(200)


def load_example(plugin) -> None:
    """Load the example project the way File > Load does.

    OMRAT then draws its own layers (legs, tangent lines with the lateral
    distributions, depth areas, structures) in its own styles.  A
    temporary copy is loaded: the loaded file becomes the project path,
    and a later File > Save in this QGIS session must not overwrite the
    tracked example.
    """
    _clean_slate(plugin)
    copy = Path(tempfile.mkdtemp(prefix='omrat_screenshots_')) / EXAMPLE.name
    shutil.copy2(EXAMPLE, copy)
    print(f'Loading {EXAMPLE.name} (copy in {copy.parent}) ...')
    # The Storage class of the plugin's own code (installed or this repo).
    module = sys.modules.get(type(plugin).__module__)
    storage_cls = getattr(module, 'Storage', None)
    if storage_cls is None:
        from omrat_utils.storage import Storage as storage_cls
    storage_cls(plugin).load_from_path(str(copy))
    show_runs = getattr(plugin, 'show_current_project_runs', None)
    if callable(show_runs):
        show_runs()
    _process(500)
    print(f'  {plugin.main_widget.twRouteList.rowCount()} leg(s) in the route '
          f'table, {_layer_count()} layer(s) on the map')

    iface.mapCanvas().zoomToFullExtent()
    iface.mapCanvas().refreshAllLayers()
    _process(500)


def _preferred_size(dock) -> QtCore.QSize:
    """Every tab without scroll bars, menu bar included."""
    preferred = getattr(dock, 'preferred_size', None)
    return preferred() if callable(preferred) else dock.sizeHint()


def _resize_window(dock, size: QtCore.QSize) -> None:
    """Resize the OMRAT window; it may run past the bottom of the screen.

    ``grab()`` renders the part off the screen too, so a window a few
    pixels taller than the screen (1007 px of contents under a title bar,
    1032 px on a 1080 screen with the taskbar) still gets its shots
    without scroll bars.  The title bar stays on the screen;
    ``fit_omrat_window`` puts the whole window back on it at the end.
    """
    dock.resize(size)
    _process(250)
    screen, handle = dock.screen(), dock.windowHandle()
    if screen is not None and handle is not None:
        avail = screen.availableGeometry()
        m = handle.frameMargins()
        geo = dock.geometry()  # the client area
        right_most = avail.left() + avail.width() - m.right() - geo.width()
        left = max(avail.left() + m.left(), min(geo.left(), right_most))
        if geo.bottom() + m.bottom() > avail.bottom():
            top = avail.top() + m.top()
        else:
            top = max(geo.top(), avail.top() + m.top())
        if (left, top) != (geo.left(), geo.top()):
            # setGeometry() places the client area; move() uses the frame.
            dock.setGeometry(left, top, geo.width(), geo.height())
            _process(150)
    if dock.width() < size.width() or dock.height() < size.height():
        print(f'  NOTE: the OMRAT window is {dock.width()}x{dock.height()}, '
              f'not {size.width()}x{size.height()}: the shots may get scroll bars')


def show_omrat_window(plugin) -> QtWidgets.QDockWidget:
    """Show OMRAT the way the user sees it: a window of its own.

    The plugin never docks its window (``OMRATMainWidget`` is a parentless
    top-level window).  Earlier versions of this script docked it into the
    QGIS main window at 520 x 820: every tab shot was cut off on the
    right, had scroll bars and lacked the File / Settings menu bar, and
    the window stayed docked after the script.  A window docked by such a
    run is taken out again here.

    Size: ``preferred_size`` (every tab without scroll bars), widened
    until the tab bar shows every tab name instead of scroll arrows.
    """
    dock = plugin.main_widget
    if dock.parentWidget() is not None:
        try:
            iface.removeDockWidget(dock)
        except Exception as exc:
            print(f'  WARN: could not undock the OMRAT window: {exc}')
        dock.setParent(None)
    ensure_frame = getattr(dock, 'ensure_window_frame', None)
    if callable(ensure_frame):
        ensure_frame()  # title bar and border, as the toolbar icon does
    dock.showNormal()
    dock.raise_()
    dock.activateWindow()
    _process(200)
    size = _preferred_size(dock)
    _resize_window(dock, size)
    tab_bar = dock.tabWidget.tabBar()
    missing = tab_bar.sizeHint().width() - tab_bar.width()
    if missing > 0:
        _resize_window(dock, QtCore.QSize(size.width() + missing + 4, size.height()))
    return dock


def fit_omrat_window(plugin) -> None:
    """Leave the OMRAT window wholly on the screen, as the plugin keeps it."""
    fit = getattr(plugin.main_widget, '_fit_on_screen', None)
    if callable(fit):
        fit()
        _process(150)


def _widen_for_table(dock, table: QtWidgets.QTableWidget, max_width: int) -> None:
    """Widen the window for more of ``table``'s columns, up to ``max_width``."""
    extra = table.horizontalHeader().length() - table.viewport().width()
    if extra > 0:
        width = min(dock.width() + extra + 4, max(max_width, dock.width()))
        _resize_window(dock, QtCore.QSize(width, dock.height()))


def capture_tabs(plugin) -> None:
    """Grab each tab of the OMRAT window."""
    dock = show_omrat_window(plugin)
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
    normal_size = dock.size()
    for idx, name, extra in tab_map:
        tabs.setCurrentIndex(idx)
        _process(250 + extra)
        traffic = getattr(dock, 'twTrafficData', None)
        if name == 'ui_tab_traffic' and traffic is not None:
            # More of the matrix next to the Traffic scaling group.
            _widen_for_table(dock, traffic, MAX_TRAFFIC_WIDTH)
        grab_widget(dock, name)
        if dock.size() != normal_size:
            _resize_window(dock, normal_size)

    tabs.setCurrentIndex(0)
    _process(200)
    grab_widget(dock, 'ui_tab_distributions')  # same as Route

    # Annotated copy of the Routes tab -- the author can later decorate
    # it with arrows / numbered callouts.  ``ui_dock_overview`` is
    # produced separately in ``main`` from a capture of the screen so
    # the docs landing image shows QGIS and the OMRAT window together.
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
        # "after run" slots aren't blank.  Writes go straight to
        # ``TWAccidentResults``: the table shows saved runs only, and the
        # hidden ``LEP*`` line-edits are never displayed.
        #
        # Synthetic numbers per row, in the same order as
        # ``AccidentResultsMixin._ACCIDENT_ROWS``.  A real run can
        # always overwrite.
        fake_probs = [
            '1.148e-01',  # Drifting allision
            '8.330e-03',  # Drifting grounding
            '1.700e-06',  # Powered allision
            '3.100e-05',  # Powered grounding
            '1.100e-06',  # Overtaking
            '4.900e-07',  # Head-on
            '0.000e+00',  # Crossing
            '0.000e+00',  # Merging
        ]
        try:
            # Make sure the table is configured + the legacy LE shims
            # exist; idempotent.  Plugin attaches the mixin to itself,
            # so the bound method lives on the plugin, not the dock.
            setup = getattr(plugin, '_setup_accident_results_table', None)
            if callable(setup):
                setup()
        except Exception as exc:
            print(f'  WARN: accident-results setup failed: {exc}')

        tw = getattr(dock, 'TWAccidentResults', None)
        render = getattr(plugin, '_render_accident_comparison', None)
        if tw is not None and callable(render):
            # The table shows saved runs only: draw the numbers as one
            # (unsaved) example run column.
            from omrat_utils.accident_summary import ACCIDENT_TOTAL_KEYS
            from omrat_utils.result_compare import RunValues, accident_values
            totals = {k: float(v) for k, v in zip(ACCIDENT_TOTAL_KEYS, fake_probs)}
            render(None, [RunValues('Example run', accident_values(totals))])
        else:
            print('  WARN: TWAccidentResults missing / unconfigured; '
                  'ui_results_filled will show empty cells')

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
            lyr for lyr in QgsProject.instance().mapLayers().values() if lyr is not None
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

    # Wipe the model and layers left from earlier captures so the
    # "blank slate" shots are actually blank.
    project = QgsProject.instance()
    _clean_slate(plugin)

    dock = show_omrat_window(plugin)
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

    # The synthetic leg has no layer and half its fields: do not leave it
    # behind as the open model (closing OMRAT would offer to save it).
    _clean_slate(plugin)
    plugin.mark_project_saved()


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
    unsaved = getattr(plugin, 'has_unsaved_changes', None)
    if callable(unsaved) and unsaved():
        print('STOPPED: OMRAT has unsaved changes.  The capture replaces the '
              'open model with the example project -- save it or close '
              'OMRAT first.')
        return
    load_example(plugin)
    print(f'  [trace] after load_example: {_layer_count()} layers')

    # Landing image for the docs: QGIS with the loaded example and the
    # OMRAT window beside it.  Captured *before* the tab walk because
    # ``capture_tabs`` cycles through tabs and resizes the window.
    print('Landing shot (QGIS + OMRAT window, screen capture)...')
    dock = show_omrat_window(plugin)
    dock.tabWidget.setCurrentIndex(0)  # Routes tab
    _process(300)
    grab_desktop(plugin, 'ui_dock_overview')

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

    fit_omrat_window(plugin)
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
