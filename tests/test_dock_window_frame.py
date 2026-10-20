"""The OMRAT dock is a parentless top-level window: it must keep its frame.

Needs the QGIS conftest (``omrat`` fixture).
"""
from __future__ import annotations

import sys
from pathlib import Path

from qgis.PyQt.QtCore import QCoreApplication, Qt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _frameless(widget) -> bool:
    return bool(widget.windowFlags() & Qt.WindowType.FramelessWindowHint)


def _spin() -> None:
    for _ in range(5):
        QCoreApplication.processEvents()


def test_dock_has_no_parent(omrat):
    # The guard below only makes sense while the dock is not added to the
    # QGIS main window; if that changes, revisit ``_keep_window_frame``.
    assert omrat.main_widget.parentWidget() is None
    assert not _frameless(omrat.main_widget)


def test_docking_the_parentless_dock_gets_the_frame_back(omrat):
    dock = omrat.main_widget
    dock.setFloating(False)
    assert _frameless(dock)  # what Qt does on its own
    _spin()
    assert not _frameless(dock)
    assert dock.isFloating()


def test_toolbar_icon_repairs_an_open_frameless_dock(omrat):
    dock = omrat.main_widget
    dock.topLevelChanged.disconnect(dock._keep_window_frame)
    dock.setFloating(False)
    _spin()
    assert _frameless(dock)
    assert omrat.pluginIsActive is True
    omrat.run()
    assert not _frameless(dock)
    assert dock.isVisible()
