"""User-facing exceptions for OMRAT.

OmratError and OmratWarning carry both a *log_message* (for the QGIS
log / debugging) and a *user_message* (terse, shown in the message
bar), plus an optional collapsible *detail* and recovery hooks
(`try_again` callback, named action buttons).

Style follows nextgis/qgis_devtools' DevToolsError pattern, trimmed
for Python 3.12+ -- the `<3.11` add_note shim is dropped because
OSGeo4W ships 3.12 and that's the only interpreter OMRAT supports.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable, Optional


_DEFAULT_MESSAGE = "An error occurred while running OMRAT"


class OmratExceptionInfoMixin:
    """Shared payload for OmratError / OmratWarning.

    Holds the IDs and messages the notifier renders, plus optional
    recovery hooks (`try_again`, named action buttons).  Inherit
    alongside Exception or UserWarning -- see OmratError / OmratWarning
    below.
    """

    def __init__(
        self,
        log_message: Optional[str] = None,
        *,
        user_message: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        self._error_id: str = str(uuid.uuid4())

        self._log_message = (log_message or _DEFAULT_MESSAGE).strip()
        self._user_message = (user_message or self._log_message).strip()

        # We're a mixin -- the concrete subclass calls Exception.__init__
        # with the log message; we just stash the args here.
        self._detail: Optional[str] = detail.strip() if detail else None

        self._try_again: Optional[Callable[[], Any]] = None
        self._actions: list[tuple[str, Callable[[], Any]]] = []
        self._need_logs: bool = True

    @property
    def error_id(self) -> str:
        return self._error_id

    @property
    def log_message(self) -> str:
        return self._log_message

    @property
    def user_message(self) -> str:
        return self._user_message

    @property
    def detail(self) -> Optional[str]:
        return self._detail

    @property
    def try_again(self) -> Optional[Callable[[], Any]]:
        return self._try_again

    @try_again.setter
    def try_again(self, callback: Optional[Callable[[], Any]]) -> None:
        self._try_again = callback

    @property
    def actions(self) -> list[tuple[str, Callable[[], Any]]]:
        return self._actions

    def add_action(self, name: str, callback: Callable[[], Any]) -> None:
        self._actions.append((name, callback))

    @property
    def need_logs(self) -> bool:
        return self._need_logs

    @need_logs.setter
    def need_logs(self, value: bool) -> None:
        self._need_logs = bool(value)


class OmratError(OmratExceptionInfoMixin, Exception):
    """Critical OMRAT error -- rendered as a red message-bar item.

    Subclass for domain-specific errors so the notifier can surface
    a meaningful class name in the log.  The notifier shows the
    user_message; the log_message goes to QGIS' message log.
    """

    def __init__(
        self,
        log_message: Optional[str] = None,
        *,
        user_message: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        OmratExceptionInfoMixin.__init__(
            self,
            log_message,
            user_message=user_message,
            detail=detail,
        )
        Exception.__init__(self, self._log_message)


class OmratWarning(OmratExceptionInfoMixin, UserWarning):
    """Non-critical OMRAT warning -- rendered as a yellow message-bar item.

    Use for recoverable issues where the user can keep working but
    should be aware of something (missing optional data, deprecated
    project format, etc.).
    """

    def __init__(
        self,
        log_message: Optional[str] = None,
        *,
        user_message: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        OmratExceptionInfoMixin.__init__(
            self,
            log_message,
            user_message=user_message,
            detail=detail,
        )
        UserWarning.__init__(self, self._log_message)


class OmratCalculationError(OmratError):
    """Raised by the calculation pipeline when a model fails.

    Default user_message points at the Run Analysis tab.
    """

    def __init__(
        self,
        log_message: Optional[str] = None,
        *,
        user_message: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(
            log_message=log_message or "Calculation failed",
            user_message=user_message or "The calculation could not be completed.",
            detail=detail,
        )


class OmratDataError(OmratError):
    """Raised when project data (segments, traffic, depths, objects) is invalid."""

    def __init__(
        self,
        log_message: Optional[str] = None,
        *,
        user_message: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(
            log_message=log_message or "Invalid project data",
            user_message=user_message or "Project data could not be parsed.",
            detail=detail,
        )
