"""Smoke test: every menu entry and every dock button, used like a user would.

Each menu entry is triggered and each enabled button of the dock is clicked,
on an empty model and on a copy of ``proj.omrat``.  Modal dialogs answer
Cancel, file pickers return nothing, URLs and background tasks are recorded
but not opened / run (``tests/gui_helpers.ModalGuard``).  An action fails
when it raises (``sys.excepthook``), shows a caught exception on the message
bar, or logs a Critical message -- the failure names the action.  The menu
entries must also visibly *do* something (open a dialog, ask for a file,
show a message, open a URL, start a task or open a window): a menu entry
wired to nothing fails too.

The lists are read from the running dock, so a new menu entry or button is
covered without editing this file.  Needs the QGIS conftest.
"""
from __future__ import annotations

import traceback

import pytest

from qgis.PyQt import sip

from tests.gui_helpers import (
    ModalGuard,
    bring_to_front,
    button_label,
    click_inner_buttons,
    clickable,
    close_new_windows,
    dock_buttons,
    load_example,
    menu_entries,
    new_visible_windows,
    settle,
    window_addresses,
)

# Buttons that remove or delete go last so the others still find data.
_LAST_WORDS = ('remove', 'delete', 'clear', 'stop')


@pytest.fixture
def guard(omrat, monkeypatch, tmp_path):
    g = ModalGuard(monkeypatch)
    # File -> Load in testing mode reads tests/test_res.omrat and would then
    # make File -> Save overwrite it: load a private copy instead.
    from omrat_utils.storage import Storage
    from tests.gui_helpers import EXAMPLE
    import shutil
    copy = tmp_path / 'loaded_by_menu.omrat'
    shutil.copy(EXAMPLE, copy)
    monkeypatch.setattr(Storage, 'select_file', lambda self: str(copy))
    # The GEBCO button downloads from the internet when the box is filled.
    import omrat_utils.handle_object as ho

    def no_download(*_a, **_k):
        raise AssertionError('the smoke test must not download GEBCO data')
    monkeypatch.setattr(ho, 'download_geotiff', no_download)
    yield g
    g.close()
    omrat._current_task = None


def _model_state(omrat) -> tuple:
    """What a load / save changes without any dialog or message."""
    return (omrat.project_path, len(omrat.segment_data), len(omrat.traffic_data))


def _run(omrat, guard, label: str, action, failures: dict, idle: list | None = None,
         explore: bool = False) -> None:
    """Run one user action; record what failed (``failures``) and whether it
    did anything (``idle``).  ``explore``: also click every button inside the
    dialogs it opens, modal (from the stubbed ``exec``) or modeless."""
    before = window_addresses()
    snap = guard.snapshot()
    state = _model_state(omrat)
    guard.explore = explore
    try:
        action()
        settle()
        windows = new_visible_windows(before)
        for window in windows if explore else ():
            if window is not omrat.main_widget and not sip.isdeleted(window) and window.isVisible():
                click_inner_buttons(guard, window)
    except Exception:
        failures[label] = traceback.format_exc()
    finally:
        guard.explore = False
    opened_window = bool(new_visible_windows(before)) or bool(windows)
    close_new_windows(before, keep=(omrat.main_widget,))
    problems = guard.problems_since(snap)
    if problems:
        failures[label] = '\n'.join(problems)
    active = opened_window or guard.activity_since(snap) or _model_state(omrat) != state
    if idle is not None and not active:
        idle.append(label)


def _report(failures: dict) -> str:
    return '\n\n'.join(f'== {label}\n{text}' for label, text in failures.items())


def _trigger_every_menu_entry(omrat, guard) -> tuple[dict, list, int]:
    failures: dict[str, str] = {}
    idle: list[str] = []
    entries = menu_entries(omrat)
    for menu, entry, action in entries:
        _run(omrat, guard, f'{menu} > {entry}', action.trigger, failures, idle, explore=True)
    return failures, idle, len(entries)


def _click_every_button(omrat, guard) -> tuple[dict, list[str]]:
    dock = omrat.main_widget
    buttons = sorted(dock_buttons(dock),
                     key=lambda b: any(w in button_label(b).lower() for w in _LAST_WORDS))
    failures: dict[str, str] = {}
    clicked: list[str] = []
    for button in buttons:
        if not clickable(button, dock):
            continue
        label = button_label(button)
        bring_to_front(button, dock)
        clicked.append(label)
        _run(omrat, guard, label, button.click, failures, explore=True)
    return failures, clicked


class TestTheGuardItself:
    """The smoke tests only mean something if a broken slot is noticed."""

    def test_an_exception_in_a_slot_is_reported(self, omrat, guard):
        from qgis.PyQt.QtWidgets import QPushButton

        def broken():
            raise ValueError('broken slot')
        button = QPushButton('broken', omrat.main_widget)
        button.clicked.connect(broken)
        failures: dict = {}
        _run(omrat, guard, 'broken', button.click, failures)
        button.deleteLater()
        assert 'ValueError: broken slot' in failures.get('broken', '')

    def test_a_caught_exception_on_the_message_bar_is_reported(self, omrat, guard):
        failures: dict = {}
        _run(omrat, guard, 'bar', lambda: omrat.notifier.display_exception(KeyError('7')), failures)
        assert "KeyError: '7'" in failures.get('bar', '')

    def test_a_user_message_is_not_a_failure(self, omrat, guard):
        failures: dict = {}
        idle: list = []
        _run(omrat, guard, 'msg', lambda: omrat.show_error_popup('Enter a value', 'here'), failures, idle)
        assert failures == {} and idle == []

    def test_an_action_that_does_nothing_is_idle(self, omrat, guard):
        idle: list = []
        _run(omrat, guard, 'nothing', lambda: None, {}, idle)
        assert idle == ['nothing']


@pytest.mark.parametrize('button', ['pbLoadDepth', 'pbLoadObject'])
def test_an_unreadable_file_is_reported(omrat, guard, tmp_path, button):
    """Load depths / structures with a file that is no layer: a message on the
    bar (it was a ``print``, so the user saw nothing happen)."""
    bad = tmp_path / 'not_a_layer.shp'
    bad.write_text('not a shapefile', encoding='utf-8')
    guard.open_file = str(bad)
    getattr(omrat.main_widget, button).click()
    settle()
    assert any('Could not open not_a_layer.shp' in m for m in guard.messages), guard.messages
    assert not guard.errors


class TestMenuEntries:
    def test_every_entry_on_an_empty_model(self, omrat, guard):
        failures, idle, n = _trigger_every_menu_entry(omrat, guard)
        assert n >= 25, f'only {n} menu entries found'
        assert not failures, _report(failures)
        assert not idle, f'menu entries that did nothing: {idle}'
        # The buttons inside the dialogs were reached too (wizard pages included).
        assert len(guard.inner_clicked) >= 25, guard.inner_clicked

    def test_every_entry_on_a_loaded_project(self, omrat, guard, tmp_path):
        load_example(omrat, tmp_path)
        assert omrat.segment_data, 'proj.omrat did not load'
        failures, idle, _n = _trigger_every_menu_entry(omrat, guard)
        assert not failures, _report(failures)
        assert not idle, f'menu entries that did nothing: {idle}'


class TestDockButtons:
    def test_every_button_on_an_empty_model(self, omrat, guard):
        failures, clicked = _click_every_button(omrat, guard)
        assert len(clicked) >= 30, f'only {len(clicked)} buttons clicked: {clicked}'
        assert not failures, _report(failures)

    def test_every_button_on_a_loaded_project(self, omrat, guard, tmp_path):
        load_example(omrat, tmp_path)
        # Run model goes as far as handing the task to the task manager.
        omrat.main_widget.LEModelName.setText('smoke')
        omrat.main_widget.LEReportPath.setText(str(tmp_path))
        failures, clicked = _click_every_button(omrat, guard)
        assert 'pbRunModel' in clicked
        assert not failures, _report(failures)
