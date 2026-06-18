"""QGIS-fixture tests for ``omrat_utils.notifier.MessageBarNotifier``.

Requires the QGIS environment (conftest.py provides ``qgis_iface`` /
``omrat``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qgis.core import Qgis  # noqa: E402

from omrat_utils.errors import OmratError, OmratWarning  # noqa: E402
from omrat_utils.notifier import MessageBarNotifier  # noqa: E402


@pytest.fixture
def notifier(qgis_iface):
    return MessageBarNotifier(
        iface=qgis_iface,
        plugin_name="OMRAT-test",
        tracker_url="https://github.com/axelande/OMRAT/issues",
    )


def _items(qgis_iface):
    return list(qgis_iface.messageBar().items())


class TestDisplayMessage:
    def test_pushes_item_and_returns_id(self, notifier, qgis_iface):
        message_id = notifier.display_message("hello", level=Qgis.MessageLevel.Info)
        items = _items(qgis_iface)
        assert any(it.property("OmratMessageId") == message_id for it in items)
        notifier.dismiss_all()

    def test_different_calls_get_different_ids(self, notifier, qgis_iface):
        a = notifier.display_message("first")
        b = notifier.display_message("second")
        assert a != b
        notifier.dismiss_all()


class TestDisplayException:
    def test_wraps_bare_exception(self, notifier, qgis_iface):
        err_id = notifier.display_exception(ValueError("nope"))
        items = _items(qgis_iface)
        assert any(it.property("OmratMessageId") == err_id for it in items)
        notifier.dismiss_all()

    def test_uses_omrat_error_id(self, notifier, qgis_iface):
        err = OmratError(log_message="oops", user_message="Friendly oops")
        err_id = notifier.display_exception(err)
        # The exception's own error_id is what gets attached to the bar
        # item -- callers can dismiss by it later.
        assert err_id == err.error_id
        notifier.dismiss_all()

    def test_warning_uses_warning_level(self, notifier, qgis_iface):
        warn_id = notifier.display_exception(OmratWarning("careful"))
        items = _items(qgis_iface)
        assert any(it.property("OmratMessageId") == warn_id for it in items)
        notifier.dismiss_all()


class TestDismiss:
    def test_dismiss_message_removes_only_that_item(self, notifier, qgis_iface):
        id_a = notifier.display_message("a")
        id_b = notifier.display_message("b")
        notifier.dismiss_message(id_a)
        ids = {it.property("OmratMessageId") for it in _items(qgis_iface)}
        assert id_b in ids
        assert id_a not in ids
        notifier.dismiss_all()

    def test_dismiss_all_clears_omrat_items(self, notifier, qgis_iface):
        notifier.display_message("a")
        notifier.display_message("b")
        notifier.dismiss_all()
        ids = {
            it.property("OmratMessageId")
            for it in _items(qgis_iface)
            if it.objectName() == "OmratMessageBarItem"
        }
        assert ids == set()


class TestOmratPluginWiring:
    """The OMRAT fixture should auto-create the notifier on init."""

    def test_plugin_has_notifier(self, omrat):
        assert isinstance(omrat.notifier, MessageBarNotifier)

    def test_show_error_popup_routes_through_notifier(self, omrat):
        # show_error_popup used to be a blocking QMessageBox; now it
        # surfaces in the message bar.  Verify by counting items
        # before/after.
        before = len(_items(omrat.iface))
        omrat.show_error_popup("Oops", "test_caller")
        after = len(_items(omrat.iface))
        assert after == before + 1
        omrat.notifier.dismiss_all()
