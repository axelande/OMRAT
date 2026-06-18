"""@handle_errors decorator -- the glue between OMRAT slot methods and
the MessageBarNotifier.

Usage:

    from omrat_utils.error_handling import handle_errors

    class OMRAT(...):
        @handle_errors
        def some_slot(self) -> None:
            ...

Any ``Exception`` raised inside ``some_slot`` is caught and shown via
``self.notifier.display_exception(exc)``.  If the plugin happens not
to have a notifier yet (during early init, or in tests), the call
falls back to ``self.show_error_popup`` so nothing is lost.

Decorated methods that need to signal "abort, but tell the user
this *specific* thing" should raise ``OmratError`` /
``OmratWarning`` -- the notifier picks up their user_message,
detail, try_again, etc.  Bare ``Exception`` is fine too; it gets
wrapped automatically.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from omrat_utils.errors import OmratError, OmratWarning


F = TypeVar("F", bound=Callable[..., Any])


def handle_errors(method: F) -> F:
    """Decorator: route exceptions through the plugin's MessageBarNotifier.

    Returns ``None`` on error rather than re-raising -- decorated slots
    are leaves (user-triggered actions), so propagating into Qt's event
    loop just produces an unhelpful traceback dialog.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except (OmratError, OmratWarning) as exc:
            _notify(self, exc, method.__name__)
            return None
        except Exception as exc:  # noqa: BLE001 -- intentional catch-all
            _notify(self, exc, method.__name__)
            return None

    return wrapper  # type: ignore[return-value]


def _notify(host: Any, exc: Exception, function_name: str) -> None:
    """Try the new notifier first, then fall back to legacy popup."""
    notifier = getattr(host, "notifier", None)
    if notifier is not None:
        try:
            notifier.display_exception(exc)
            return
        except Exception:  # noqa: BLE001 -- never let the notifier itself crash
            pass

    # Legacy fallback so partially-initialised plugins (or test doubles
    # without a notifier) still surface the error rather than swallow it.
    popup = getattr(host, "show_error_popup", None)
    if callable(popup):
        try:
            popup(str(exc), function_name)
            return
        except Exception:  # noqa: BLE001
            pass

    # Last-ditch: let Qt's default unhandled-exception path see it.
    raise exc
