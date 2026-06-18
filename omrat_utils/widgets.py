"""Shared Qt widget helpers for OMRAT.

The default ``QSpinBox`` / ``QDoubleSpinBox`` accept mouse-wheel events
whenever the cursor is over them, regardless of focus. In matrix views
(traffic, junctions, consequence) the user often scrolls the page while
the cursor happens to be over a numeric cell, silently mutating the
value. These subclasses only consume wheel events when the widget owns
keyboard focus; otherwise the event bubbles up to the surrounding
scroll view so the page scrolls as expected.
"""
from qgis.PyQt.QtWidgets import QDoubleSpinBox, QSpinBox


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()
