"""The OMRAT window scrolls its contents instead of outgrowing the screen.

The tabs need 930-960 px of height; before 2026-10-29 that was the window's
minimum, so on a laptop / remote desktop screen its bottom hung off the
screen and a maximised window cut the contents off.

Needs the QGIS conftest (``omrat`` fixture).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from qgis.PyQt.QtCore import QCoreApplication, QPoint, QPointF, Qt
from qgis.PyQt.QtGui import QWheelEvent
from qgis.PyQt.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QMenuBar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import omrat_widget  # noqa: E402


def _spin(seconds: float = 0.0) -> None:
    end = time.monotonic() + seconds
    while True:
        for _ in range(5):
            QCoreApplication.processEvents()
        if time.monotonic() >= end:
            return
        time.sleep(0.01)


def _wheel_event(delta: int = -120) -> QWheelEvent:
    return QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, delta),
                       Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                       Qt.ScrollPhase.NoScrollPhase, False)


def test_contents_scroll_and_the_window_can_shrink(omrat):
    dock = omrat.main_widget
    assert dock.dockScrollArea.widget() is dock.dockWidgetContents
    assert dock.minimumSizeHint().height() < 300
    _spin(0.2)  # let the first-show sizing (deferred) run first
    dock.resize(720, 400)
    _spin(0.2)
    assert dock.height() == 400
    assert dock.dockScrollArea.verticalScrollBar().maximum() > 0


def test_menu_bar_stays_above_the_scrolling_contents(omrat):
    dock = omrat.main_widget
    bars = dock.findChildren(QMenuBar)
    assert len(bars) == 1
    bar = bars[0]
    assert not dock.dockScrollArea.isAncestorOf(bar)
    assert bar.parentWidget() is dock.dockScrollArea.parentWidget()
    assert bar.geometry().bottom() < dock.dockScrollArea.geometry().top()


def test_first_show_fits_the_screen(omrat):
    if not omrat_widget.dock_uses_native_frame():
        pytest.skip('Qt draws the dock title bar itself on this platform (X11)')
    dock = omrat.main_widget
    _spin(0.2)
    avail = dock.screen().availableGeometry()
    frame = dock.frameGeometry()
    assert frame.top() >= avail.top()
    assert frame.bottom() <= avail.bottom()
    # As tall as the contents want, as far as the screen allows.
    wanted = dock.preferred_size().height()
    assert dock.height() >= min(wanted, avail.height() - (frame.height() - dock.height())) - 1


def test_spin_and_combo_boxes_take_the_wheel_only_with_focus(omrat):
    dock = omrat.main_widget
    boxes = (dock.dockWidgetContents.findChildren(QAbstractSpinBox)
             + dock.dockWidgetContents.findChildren(QComboBox))
    assert boxes
    for box in boxes:
        assert box.focusPolicy() == Qt.FocusPolicy.StrongFocus, box.objectName()


def test_wheel_filter_passes_an_unfocused_box_on(monkeypatch):
    wheel_filter = omrat_widget.WheelScrollsPage()
    box = QComboBox()
    event = _wheel_event()
    assert wheel_filter.eventFilter(box, event) is True
    assert not event.isAccepted()  # so the parents (the page) get it
    monkeypatch.setattr(box, 'hasFocus', lambda: True)
    assert wheel_filter.eventFilter(box, _wheel_event()) is False


@pytest.mark.skipif(sys.platform != 'win32', reason='posts a native WM_MOUSEWHEEL')
def test_native_wheel_over_a_spin_box_scrolls_the_page(omrat):
    # End to end: Qt only passes a *spontaneous* wheel event on to the
    # parents, so this goes through the window system.
    import ctypes
    if QApplication.platformName() != 'windows':
        pytest.skip('needs the native Windows platform plugin')
    dock = omrat.main_widget
    _spin(0.2)  # let the first-show sizing (deferred) run first
    dock.resize(720, 450)
    dock.tabWidget.setCurrentIndex(0)
    _spin(0.3)
    page = dock.dockScrollArea.verticalScrollBar()
    page.setValue(0)
    box = dock.sbRouteId
    box.setValue(5)
    dock.raise_()
    _spin(0.2)
    centre = box.mapToGlobal(QPoint(box.width() // 2, box.height() // 2))
    under = QApplication.widgetAt(centre)
    if under is None or not (under is box or box.isAncestorOf(under)):
        # Qt hands a native wheel to whichever of its windows is under the point.
        pytest.skip('another window of the test session covers the spin box')
    dpr = dock.screen().devicePixelRatio()
    x, y = int(centre.x() * dpr), int(centre.y() * dpr)
    ctypes.windll.user32.PostMessageW(int(dock.winId()), 0x020A,  # WM_MOUSEWHEEL
                                      (-120 & 0xFFFF) << 16, (y << 16) | (x & 0xFFFF))
    _spin(0.4)
    assert box.value() == 5
    assert page.value() > 0
