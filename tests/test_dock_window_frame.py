"""The OMRAT dock is a parentless top-level window: it must keep its frame.

Needs the QGIS conftest (``omrat`` fixture).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from qgis.PyQt.QtCore import QCoreApplication, QEvent, QPointF, Qt
from qgis.PyQt.QtGui import QMouseEvent
from qgis.PyQt.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import omrat_widget  # noqa: E402


def _frameless(widget) -> bool:
    return bool(widget.windowFlags() & Qt.WindowType.FramelessWindowHint)


def _normal_window(widget) -> bool:
    # A plain window, not the Qt.Tool that setFloating(True) makes: it has
    # minimise / maximise buttons and a taskbar entry.
    flags = widget.windowFlags()
    return (widget.windowType() == Qt.WindowType.Window
            and not flags & Qt.WindowType.FramelessWindowHint
            and bool(flags & Qt.WindowType.WindowMinimizeButtonHint)
            and bool(flags & Qt.WindowType.WindowMaximizeButtonHint))


def _spin(seconds: float = 0.0) -> None:
    end = time.monotonic() + seconds
    while True:
        for _ in range(5):
            QCoreApplication.processEvents()
        if time.monotonic() >= end:
            return
        time.sleep(0.01)


def _needs_native_frame() -> None:
    # On X11 (CI) Qt marks every floating dock frameless and draws the
    # title bar itself, so the frame-loss these tests reproduce cannot occur.
    if not omrat_widget.dock_uses_native_frame():
        pytest.skip('Qt draws the dock title bar itself on this platform (X11)')


def _title_bar_double_click(dock) -> None:
    """What Qt delivers for two clicks on the native title bar."""
    top = dock.frameGeometry().top() + 5
    global_pos = QPointF(dock.frameGeometry().center().x(), top)
    local_pos = QPointF(dock.mapFromGlobal(global_pos.toPoint()))
    for etype, buttons in (
            (QEvent.Type.NonClientAreaMouseButtonPress, Qt.MouseButton.LeftButton),
            (QEvent.Type.NonClientAreaMouseButtonRelease, Qt.MouseButton.NoButton),
            (QEvent.Type.NonClientAreaMouseButtonPress, Qt.MouseButton.LeftButton),
            (QEvent.Type.NonClientAreaMouseButtonDblClick, Qt.MouseButton.LeftButton),
            (QEvent.Type.NonClientAreaMouseButtonRelease, Qt.MouseButton.NoButton)):
        event = QMouseEvent(etype, local_pos, global_pos, Qt.MouseButton.LeftButton,
                            buttons, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(dock, event)


def test_dock_has_no_parent(omrat):
    # The guard below only makes sense while the dock is not added to the
    # QGIS main window; if that changes, revisit ``_keep_window_frame``.
    assert omrat.main_widget.parentWidget() is None
    assert not _frameless(omrat.main_widget)
    assert _normal_window(omrat.main_widget)


def test_title_bar_double_click_does_not_dock_the_window(omrat):
    # QDockWidget toggles docked / floating on it; without a parent that
    # stripped the frame while Windows maximised the window (2026-10-29).
    dock = omrat.main_widget
    changes: list[bool] = []
    dock.topLevelChanged.connect(changes.append)
    try:
        _title_bar_double_click(dock)
        _spin()
    finally:
        dock.topLevelChanged.disconnect(changes.append)
    assert changes == []
    assert _normal_window(dock)
    assert dock.isFloating()


def test_docking_the_parentless_dock_gets_the_frame_back(omrat):
    _needs_native_frame()
    dock = omrat.main_widget
    dock.setFloating(False)
    assert _frameless(dock)  # what Qt does on its own
    _spin()
    assert not _frameless(dock)
    assert dock.isFloating()
    assert _normal_window(dock)


def test_toolbar_icon_repairs_an_open_frameless_dock(omrat):
    _needs_native_frame()
    dock = omrat.main_widget
    dock.topLevelChanged.disconnect(dock._keep_window_frame)
    dock.setFloating(False)
    _spin()
    assert _frameless(dock)
    assert omrat.pluginIsActive is True
    omrat.run()
    assert not _frameless(dock)
    assert _normal_window(dock)
    assert dock.isVisible()


def test_toolbar_icon_turns_a_tool_window_back_into_a_window(omrat):
    # setFloating(True) -- the repair before 2026-10-29 -- makes a Qt.Tool
    # window: no minimise / maximise button, no taskbar entry.
    _needs_native_frame()
    dock = omrat.main_widget
    dock.topLevelChanged.disconnect(dock._keep_window_frame)
    dock.setFloating(False)
    dock.setFloating(True)
    _spin()
    assert dock.windowType() == Qt.WindowType.Tool
    omrat.run()
    assert _normal_window(dock)
    assert dock.isVisible()


def test_maximised_frameless_dock_comes_back_with_its_title_bar_on_screen(omrat):
    # The state the double-click left behind: maximised, no frame.  The
    # repair restores a normal window whose title bar is on the screen.
    _needs_native_frame()
    dock = omrat.main_widget
    dock.showMaximized()
    _spin(0.3)
    dock.topLevelChanged.disconnect(dock._keep_window_frame)
    dock.setFloating(False)
    _spin(0.3)
    assert _frameless(dock)
    omrat.run()
    _spin(0.3)
    assert _normal_window(dock)
    assert not dock.isMaximized()
    avail = dock.screen().availableGeometry()
    frame = dock.frameGeometry()
    assert frame.top() >= avail.top()
    assert frame.left() >= avail.left()
    assert frame.width() <= avail.width()


@pytest.mark.parametrize('native, expected', [(True, [omrat_widget.WINDOW_FLAGS]), (False, [])])
def test_frameless_dock_is_restyled_only_with_native_frames(omrat, monkeypatch, native, expected):
    # Runs on every platform: the window flags and the platform check are
    # faked, so this pins the X11 guard even where the tests above skip.
    dock = omrat.main_widget
    calls: list = []
    monkeypatch.setattr(omrat_widget, 'dock_uses_native_frame', lambda: native)
    monkeypatch.setattr(dock, 'windowFlags', lambda: Qt.WindowType.FramelessWindowHint)
    monkeypatch.setattr(dock, 'setWindowFlags', calls.append)
    monkeypatch.setattr(dock, 'isVisible', lambda: False)
    dock.ensure_window_frame()
    assert calls == expected
