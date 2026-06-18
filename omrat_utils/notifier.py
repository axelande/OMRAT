"""Message-bar notifier for OMRAT.

Ports nextgis/qgis_devtools' MessageBarNotifier pattern: instead of a
blocking QMessageBox, exceptions surface as non-modal items in the
QGIS message bar with Details / Open logs / Let-us-know buttons.

The notifier is wired up once by ``OMRAT.__init__`` and exposed as
``self.notifier``.  Existing call sites that use
``OMRAT.show_error_popup`` are routed through here for free; new code
should raise :class:`omrat_utils.errors.OmratError` and let
``@handle_errors`` turn it into a notifier call.
"""
from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, Optional

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtCore import QObject, QUrl
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import QMessageBox, QPushButton, QWidget

from omrat_utils.errors import OmratError, OmratExceptionInfoMixin, OmratWarning


if TYPE_CHECKING:
    from qgis.gui import QgisInterface


_ITEM_OBJECT_NAME = "OmratMessageBarItem"
_ITEM_ID_PROPERTY = "OmratMessageId"


def let_us_know(tracker_url: str) -> None:
    """Open the OMRAT issue tracker in the user's default browser."""
    if not tracker_url:
        return
    QDesktopServices.openUrl(QUrl(tracker_url))


def open_logs(iface: "QgisInterface") -> None:
    """Open the QGIS message log panel."""
    iface.openMessageLog()


class MessageBarNotifier(QObject):
    """Non-modal notifier backed by ``iface.messageBar()``.

    Parameters
    ----------
    iface:
        Live ``QgisInterface`` -- the message bar is queried lazily so
        late-binding works in tests.
    plugin_name:
        Header text on each message-bar item (short -- shows next to the
        message itself).
    tracker_url:
        URL opened by the "Let us know" button on critical errors.  Usually
        read from ``metadata.txt`` (`tracker` field).
    parent:
        Standard QObject parent.
    """

    def __init__(
        self,
        iface: "QgisInterface",
        plugin_name: str = "OMRAT",
        tracker_url: str = "",
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._iface = iface
        self._plugin_name = plugin_name
        self._tracker_url = tracker_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def display_message(
        self,
        message: str,
        *,
        level: int = Qgis.MessageLevel.Info,
        widgets: Optional[list[QWidget]] = None,
        duration: int = 0,
    ) -> str:
        """Push a plain message to the QGIS message bar.

        Returns an opaque id that can be passed to :meth:`dismiss_message`.
        """
        message_bar = self._iface.messageBar()
        widget = message_bar.createMessage(self._plugin_name, message)

        for custom in widgets or []:
            custom.setParent(widget)
            widget.layout().addWidget(custom)

        item = message_bar.pushWidget(widget, level, duration)
        message_id = str(uuid.uuid4())
        item.setObjectName(_ITEM_OBJECT_NAME)
        item.setProperty(_ITEM_ID_PROPERTY, message_id)

        QgsMessageLog.logMessage(message, self._plugin_name, level)
        return message_id

    def display_exception(self, error: Exception) -> str:
        """Push an exception to the message bar.

        Plain ``Exception`` instances are wrapped in :class:`OmratError`
        so the message bar always gets a user_message and an error_id.
        ``OmratWarning`` instances render at warning level (no
        Let-us-know button).
        """
        wrapped = self._wrap(error)

        message = wrapped.user_message.rstrip(".") + "."
        message_bar = self._iface.messageBar()
        widget = message_bar.createMessage(self._plugin_name, message)

        is_warning = isinstance(wrapped, OmratWarning)
        if not is_warning:
            self._add_error_buttons(wrapped, widget)

        level = Qgis.MessageLevel.Warning if is_warning else Qgis.MessageLevel.Critical
        item = message_bar.pushWidget(widget, level)
        item.setObjectName(_ITEM_OBJECT_NAME)
        item.setProperty(_ITEM_ID_PROPERTY, wrapped.error_id)

        if is_warning:
            QgsMessageLog.logMessage(wrapped.user_message, self._plugin_name, level)
        else:
            # Include the original exception's traceback summary so the log
            # is actually useful for diagnosing.
            cause = wrapped.__cause__ if wrapped.__cause__ is not None else error
            log_text = f"{wrapped.log_message}: {type(cause).__name__}: {cause}"
            QgsMessageLog.logMessage(log_text, self._plugin_name, level)

        return wrapped.error_id

    def dismiss_message(self, message_id: str) -> None:
        """Remove a specific message from the bar."""
        bar = self._iface.messageBar()
        for item in list(bar.items()):
            if (
                item.objectName() == _ITEM_OBJECT_NAME
                and item.property(_ITEM_ID_PROPERTY) == message_id
            ):
                bar.popWidget(item)

    def dismiss_all(self) -> None:
        """Remove every OMRAT-pushed message from the bar."""
        bar = self._iface.messageBar()
        for item in list(bar.items()):
            if item.objectName() == _ITEM_OBJECT_NAME:
                bar.popWidget(item)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _wrap(self, error: Exception) -> OmratExceptionInfoMixin:
        """Return ``error`` as-is if it's already an OmratError/Warning,
        else wrap it so the notifier API stays uniform."""
        if isinstance(error, (OmratError, OmratWarning)):
            return error  # type: ignore[return-value]

        if isinstance(error, Warning):
            wrapped: OmratExceptionInfoMixin = OmratWarning(
                log_message=f"{type(error).__name__}: {error}",
                user_message=str(error) or "An unexpected warning occurred.",
            )
        else:
            wrapped = OmratError(
                log_message=f"{type(error).__name__}: {error}",
                user_message=str(error) or "An unexpected error occurred.",
            )
        # Preserve the original traceback chain.
        wrapped.__cause__ = error  # type: ignore[attr-defined]
        return wrapped

    def _add_error_buttons(self, error: OmratError, widget: QWidget) -> None:
        if error.try_again is not None:
            def _try_again() -> None:
                try:
                    error.try_again()  # type: ignore[misc]
                finally:
                    self._iface.messageBar().popWidget(widget)

            button = QPushButton(self.tr("Try again"))
            button.pressed.connect(_try_again)
            widget.layout().addWidget(button)

        for action_name, action_callback in error.actions:
            button = QPushButton(action_name)
            button.pressed.connect(action_callback)
            widget.layout().addWidget(button)

        if error.detail is not None:
            def _show_details() -> None:
                # Strip lightweight HTML tags from the title -- the user
                # may have passed <b>...</b> in user_message for the bar.
                title = re.sub(
                    r"</?(i|b)\b[^>]*?>", "",
                    error.user_message.rstrip("."),
                    flags=re.IGNORECASE,
                )
                QMessageBox.information(
                    self._iface.mainWindow(), title, error.detail or ""
                )

            button = QPushButton(self.tr("Details"))
            button.pressed.connect(_show_details)
            widget.layout().addWidget(button)
        elif error.need_logs:
            button = QPushButton(self.tr("Open logs"))
            button.pressed.connect(lambda: open_logs(self._iface))
            widget.layout().addWidget(button)

        # "Let us know" only on the generic OmratError -- domain
        # subclasses are expected to handle their own user guidance.
        if type(error) is OmratError and self._tracker_url:
            button = QPushButton(self.tr("Let us know"))
            button.pressed.connect(lambda: let_us_know(self._tracker_url))
            widget.layout().addWidget(button)
