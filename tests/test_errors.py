"""Pure-Python tests for ``omrat_utils.errors``.

No QGIS dependency -- run with::

    pytest -p no:qgis --noconftest tests/test_errors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.errors import (  # noqa: E402
    OmratCalculationError,
    OmratDataError,
    OmratError,
    OmratWarning,
)


class TestOmratError:
    def test_defaults_when_no_args(self):
        err = OmratError()
        assert err.log_message  # populated with a default
        assert err.user_message == err.log_message
        assert err.detail is None
        assert err.try_again is None
        assert err.actions == []
        assert err.need_logs is True
        assert err.error_id  # uuid string

    def test_user_message_falls_back_to_log_message(self):
        err = OmratError(log_message="boom")
        assert err.user_message == "boom"

    def test_explicit_user_message(self):
        err = OmratError(
            log_message="internal boom",
            user_message="Something went wrong",
            detail="Stack frame: foo",
        )
        assert err.log_message == "internal boom"
        assert err.user_message == "Something went wrong"
        assert err.detail == "Stack frame: foo"

    def test_strips_whitespace(self):
        err = OmratError(
            log_message="  log  \n",
            user_message="  user  ",
            detail="  detail  ",
        )
        assert err.log_message == "log"
        assert err.user_message == "user"
        assert err.detail == "detail"

    def test_error_id_is_unique_per_instance(self):
        e1 = OmratError("a")
        e2 = OmratError("a")
        assert e1.error_id != e2.error_id

    def test_is_an_exception(self):
        with pytest.raises(OmratError):
            raise OmratError("kaboom")

    def test_add_action_appends(self):
        err = OmratError("boom")
        seen: list[str] = []
        err.add_action("Click me", lambda: seen.append("clicked"))
        assert len(err.actions) == 1
        name, callback = err.actions[0]
        assert name == "Click me"
        callback()
        assert seen == ["clicked"]

    def test_try_again_setter(self):
        err = OmratError("boom")
        assert err.try_again is None
        err.try_again = lambda: 42
        assert err.try_again() == 42

    def test_need_logs_toggle(self):
        err = OmratError("boom")
        assert err.need_logs is True
        err.need_logs = False
        assert err.need_logs is False


class TestOmratWarning:
    def test_is_a_warning(self):
        warn = OmratWarning("watch out")
        assert isinstance(warn, UserWarning)

    def test_carries_messages(self):
        warn = OmratWarning(
            log_message="deprecation",
            user_message="Use the new dialog",
        )
        assert warn.log_message == "deprecation"
        assert warn.user_message == "Use the new dialog"


class TestDomainSubclasses:
    def test_calculation_error_defaults(self):
        err = OmratCalculationError()
        assert "Calculation" in err.log_message
        assert "calculation" in err.user_message.lower()
        # Still an OmratError so notifier picks it up uniformly.
        assert isinstance(err, OmratError)

    def test_data_error_defaults(self):
        err = OmratDataError()
        assert "project data" in err.log_message.lower()
        assert isinstance(err, OmratError)

    def test_calculation_error_keeps_caller_overrides(self):
        err = OmratCalculationError(
            log_message="powered model failed",
            user_message="Powered grounding could not run",
            detail="Segment seg_3 has 0 ships",
        )
        assert err.log_message == "powered model failed"
        assert err.user_message == "Powered grounding could not run"
        assert err.detail == "Segment seg_3 has 0 ships"
