"""Pure-Python tests for ``omrat_utils.error_handling``.

Verifies that ``@handle_errors`` routes through ``self.notifier`` if
present, then falls back to ``self.show_error_popup`` otherwise.

No QGIS dependency -- run with::

    pytest -p no:qgis --noconftest tests/test_error_handling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.error_handling import handle_errors  # noqa: E402
from omrat_utils.errors import OmratError, OmratWarning  # noqa: E402


class _RecordingNotifier:
    """Minimal stub matching MessageBarNotifier.display_exception."""

    def __init__(self) -> None:
        self.seen: list[Exception] = []

    def display_exception(self, exc: Exception) -> str:
        self.seen.append(exc)
        return "msg-id"


class _HostWithNotifier:
    def __init__(self) -> None:
        self.notifier = _RecordingNotifier()

    @handle_errors
    def will_raise(self) -> None:
        raise ValueError("nope")

    @handle_errors
    def will_raise_omrat(self) -> None:
        raise OmratError(log_message="boom", user_message="kaboom")

    @handle_errors
    def will_warn(self) -> None:
        raise OmratWarning("careful")

    @handle_errors
    def returns_value(self) -> int:
        return 7


class _HostWithLegacyPopup:
    def __init__(self) -> None:
        self.popups: list[tuple[str, str]] = []

    def show_error_popup(self, message: str, function_name: str) -> None:
        self.popups.append((message, function_name))

    @handle_errors
    def will_raise(self) -> None:
        raise RuntimeError("legacy fallback please")


class TestHandleErrorsDecorator:
    def test_returns_value_when_no_error(self):
        host = _HostWithNotifier()
        assert host.returns_value() == 7
        assert host.notifier.seen == []

    def test_routes_bare_exception_to_notifier(self):
        host = _HostWithNotifier()
        host.will_raise()  # should not re-raise
        assert len(host.notifier.seen) == 1
        assert isinstance(host.notifier.seen[0], ValueError)

    def test_routes_omrat_error_to_notifier(self):
        host = _HostWithNotifier()
        host.will_raise_omrat()
        assert len(host.notifier.seen) == 1
        assert isinstance(host.notifier.seen[0], OmratError)
        assert host.notifier.seen[0].user_message == "kaboom"

    def test_routes_omrat_warning_to_notifier(self):
        host = _HostWithNotifier()
        host.will_warn()
        assert len(host.notifier.seen) == 1
        assert isinstance(host.notifier.seen[0], OmratWarning)

    def test_falls_back_to_show_error_popup_without_notifier(self):
        host = _HostWithLegacyPopup()
        host.will_raise()
        assert len(host.popups) == 1
        message, function_name = host.popups[0]
        assert "legacy fallback" in message
        assert function_name == "will_raise"

    def test_reraises_if_no_handler_available(self):
        class Bare:
            @handle_errors
            def boom(self_inner) -> None:
                raise RuntimeError("nowhere to go")

        with pytest.raises(RuntimeError):
            Bare().boom()
