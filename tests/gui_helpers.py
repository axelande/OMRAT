"""Helpers for the tests that drive OMRAT through its GUI like a user.

:class:`ModalGuard` stubs everything that would block the test or reach
outside it -- a modal ``exec()``, file pickers, message boxes, input
dialogs, URL openers and the QGIS task manager -- and records what
happened, so a test can trigger a menu entry or click a button and then
check the outcome.  A dialog's ``exec()`` answers *Cancel* unless a handler
was queued with :meth:`ModalGuard.expect` (see :func:`accept_with` /
:func:`read_with`).

Errors are collected instead of crashing the process: an exception in a
slot reaches ``sys.excepthook`` (PyQt aborts with the default hook), and an
exception a handler caught and showed on the message bar reaches
``Notifier.display_exception``.  A plain :class:`OmratError` there is a
message to the user (``show_error_popup``: "enter valid coordinates..."),
anything else is a crash the user would see as a red bar.

Needs the QGIS conftest (``omrat`` fixture).
"""
from __future__ import annotations

import shutil
import sys
import traceback
import webbrowser
from pathlib import Path
from typing import Any, Callable, Iterable

from qgis.core import Qgis, QgsApplication, QgsTaskManager
from qgis.PyQt import sip
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QInputDialog,
    QMenu,
    QMenuBar,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollBar,
    QStackedWidget,
    QTabBar,
    QTableWidget,
    QTabWidget,
    QToolBar,
    QToolButton,
    QWidget,
    QWizard,
)

EXAMPLE = Path(__file__).resolve().parent / 'example_data' / 'proj.omrat'

_EXEC_CLASSES = (QDialog, QMessageBox, QFileDialog, QInputDialog, QWizard, QProgressDialog)
_REJECTED = 0


class ModalGuard:
    """Stub the blocking / outward-facing Qt calls and record them."""

    def __init__(self, monkeypatch) -> None:
        self._handlers: list[Callable[[QDialog], Any]] = []
        self.opened: list[str] = []            # dialogs whose exec() ran
        self.messages: list[str] = []          # message boxes + user messages on the bar
        self.file_dialogs: list[str] = []
        self.open_file = ''                    # what getOpenFileName answers
        self.save_file = ''                    # what getSaveFileName answers
        self.urls: list[str] = []
        self.tasks: list[Any] = []             # tasks handed to the task manager (never run)
        self.errors: list[str] = []            # unhandled exceptions + caught crashes
        self.critical_logs: list[str] = []
        self.explore = False                   # click the buttons inside every dialog
        self.where: list[str] = []             # breadcrumb of the buttons being clicked
        self.inner_clicked: list[str] = []     # 'Dialog > button' clicked while exploring
        self._in_user_message = False
        self._install(monkeypatch)

    # -- dialogs ----------------------------------------------------------
    def expect(self, handler: Callable[[QDialog], Any]) -> None:
        """Queue ``handler`` for the next dialog ``exec()``; it returns the result code."""
        self._handlers.append(handler)

    def _exec(self, dlg: Any) -> int:
        name = type(dlg).__name__
        if isinstance(dlg, QMessageBox):
            self.messages.append(f'{name}: {dlg.text()}')
        self.opened.append(name)
        if self._handlers:
            result = self._handlers.pop(0)(dlg)
            return int(dlg.result() if result is None else result)
        if self.explore and not isinstance(dlg, (QMessageBox, QFileDialog, QInputDialog)):
            click_inner_buttons(self, dlg)
        return _REJECTED

    # -- logs ---------------------------------------------------------------
    def _here(self) -> str:
        return f"[in {' > '.join(self.where)}] " if self.where else ''

    def _on_log(self, message: str, tag: str, level: Any) -> None:
        if level == Qgis.MessageLevel.Critical and not self._in_user_message:
            self.critical_logs.append(f'{self._here()}[{tag}] {message}')

    def _hook(self, etype, value, tb) -> None:
        self.errors.append(self._here() + ''.join(traceback.format_exception(etype, value, tb)))

    def close(self) -> None:
        """Disconnect from the message log (the patches go with ``monkeypatch``)."""
        try:
            QgsApplication.messageLog().messageReceived[str, str, Qgis.MessageLevel].disconnect(self._on_log)
        except (TypeError, RuntimeError, KeyError):
            pass

    def snapshot(self) -> tuple[int, ...]:
        """Lengths of every record list, to see what one action added."""
        return (len(self.opened), len(self.messages), len(self.file_dialogs), len(self.urls),
                len(self.tasks), len(self.errors), len(self.critical_logs))

    def problems_since(self, snap: tuple[int, ...]) -> list[str]:
        return self.errors[snap[5]:] + self.critical_logs[snap[6]:]

    def activity_since(self, snap: tuple[int, ...]) -> bool:
        """True when the action opened a dialog, showed a message, asked for a
        file, opened a URL or started a task."""
        return self.snapshot()[:5] != snap[:5]

    # -- install ---------------------------------------------------------------
    def _install(self, mp) -> None:
        guard = self

        def fake_exec(dlg, *_a, **_k):
            return guard._exec(dlg)

        for cls in _EXEC_CLASSES:
            mp.setattr(cls, 'exec', fake_exec, raising=False)
            mp.setattr(cls, 'exec_', fake_exec, raising=False)
        mp.setattr(QMenu, 'exec', lambda *_a, **_k: None, raising=False)

        def box(kind: str, answer: Any):
            def show(*args, **_kwargs):
                texts = [str(a) for a in args[1:3] if isinstance(a, str)]
                guard.messages.append(f'{kind}: ' + ' | '.join(texts))
                return answer
            return staticmethod(show)

        sb = QMessageBox.StandardButton
        mp.setattr(QMessageBox, 'question', box('question', sb.No))
        mp.setattr(QMessageBox, 'warning', box('warning', sb.Ok))
        mp.setattr(QMessageBox, 'information', box('information', sb.Ok))
        mp.setattr(QMessageBox, 'critical', box('critical', sb.Ok))
        mp.setattr(QMessageBox, 'about', box('about', None))

        def file_dialog(kind: str, answer: Callable[[], Any]):
            def ask(*_args, **_kwargs):
                guard.file_dialogs.append(kind)
                return answer()
            return staticmethod(ask)

        mp.setattr(QFileDialog, 'getOpenFileName', file_dialog('open', lambda: (guard.open_file, '')))
        mp.setattr(QFileDialog, 'getOpenFileNames', file_dialog('open many', lambda: ([], '')))
        mp.setattr(QFileDialog, 'getSaveFileName', file_dialog('save', lambda: (guard.save_file, '')))
        mp.setattr(QFileDialog, 'getExistingDirectory', file_dialog('folder', lambda: ''))
        for name, answer in (('getText', ('', False)), ('getItem', ('', False)), ('getInt', (0, False)),
                             ('getDouble', (0.0, False)), ('getMultiLineText', ('', False))):
            mp.setattr(QInputDialog, name, file_dialog(f'input {name}', lambda a=answer: a), raising=False)

        def open_url(url, *_a, **_k):
            guard.urls.append(url.toString() if hasattr(url, 'toString') else str(url))
            return True

        mp.setattr(QDesktopServices, 'openUrl', staticmethod(open_url))
        mp.setattr(webbrowser, 'open', lambda url, *a, **k: guard.urls.append(str(url)) or True)

        def add_task(_manager, task, *_a, **_k):
            guard.tasks.append(task)
            return len(guard.tasks)

        mp.setattr(QgsTaskManager, 'addTask', add_task)

        from omrat_utils.errors import OmratError
        from omrat_utils.notifier import MessageBarNotifier as Notifier
        original_display = Notifier.display_exception

        def display_exception(notifier, error):
            user_message = type(error) is OmratError and error.__cause__ is None
            if user_message:
                guard.messages.append(f'bar: {getattr(error, "user_message", error)}')
            else:
                detail = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
                guard.errors.append(f'{guard._here()}shown on the message bar: {detail}')
            guard._in_user_message = user_message
            try:
                return original_display(notifier, error)
            finally:
                guard._in_user_message = False

        mp.setattr(Notifier, 'display_exception', display_exception)
        original_message = Notifier.display_message

        def display_message(notifier, message, *args, **kwargs):
            guard.messages.append(f'bar: {message}')
            return original_message(notifier, message, *args, **kwargs)

        mp.setattr(Notifier, 'display_message', display_message)

        # The database wizard, migrations and the AIS ingest connect with
        # psycopg2 directly (the conftest only mocks handle_ais.DB).
        try:
            import psycopg2
        except ImportError:  # pragma: no cover - psycopg2 ships with OSGeo4W
            psycopg2 = None
        if psycopg2 is not None:
            def no_database(*_a, **_k):
                raise psycopg2.OperationalError('no database in the GUI tests')
            mp.setattr(psycopg2, 'connect', no_database)
        mp.setattr(sys, 'excepthook', self._hook)
        QgsApplication.messageLog().messageReceived[str, str, Qgis.MessageLevel].connect(self._on_log)


# ---------------------------------------------------------------------------
# Dialog handlers for ModalGuard.expect
# ---------------------------------------------------------------------------

def _click_role(dlg: QDialog, role, what: str) -> int:
    for box in dlg.findChildren(QDialogButtonBox):
        for button in box.buttons():
            if box.buttonRole(button) == role:
                button.click()
                QCoreApplication.processEvents()
                return int(dlg.result())
    raise AssertionError(f'{type(dlg).__name__} has no {what} button')


def click_accept(dlg: QDialog) -> int:
    """Press the dialog's OK / Save button (the button box's AcceptRole button)."""
    return _click_role(dlg, QDialogButtonBox.ButtonRole.AcceptRole, 'OK')


def click_reject(dlg: QDialog) -> int:
    """Press the dialog's Cancel / Discard button (the button box's RejectRole button)."""
    return _click_role(dlg, QDialogButtonBox.ButtonRole.RejectRole, 'Cancel')


def accept_with(edit: Callable[[QDialog], None]) -> Callable[[QDialog], int]:
    """Handler: change the widgets like a user (``edit``), then press OK."""
    def handler(dlg: QDialog) -> int:
        edit(dlg)
        return click_accept(dlg)
    return handler


def read_with(read: Callable[[QDialog], Any], into: dict) -> Callable[[QDialog], int]:
    """Handler: record what the open dialog shows in ``into['shown']``, then Cancel."""
    def handler(dlg: QDialog) -> int:
        into['shown'] = read(dlg)
        return _REJECTED
    return handler


# ---------------------------------------------------------------------------
# Menus, buttons, windows, projects
# ---------------------------------------------------------------------------

def menu_entries(omrat) -> list[tuple[str, str, Any]]:
    """``(menu, entry, QAction)`` for every entry of the dock's menu bar."""
    bars = omrat.main_widget.findChildren(QMenuBar)
    assert len(bars) == 1, 'the dock has one menu bar'
    out = []
    for top in bars[0].actions():
        menu = top.menu()
        if menu is None:
            continue
        for action in menu.actions():
            if action.isSeparator() or action.menu() is not None:
                continue
            out.append((top.text().replace('&', ''), action.text(), action))
    return out


def menu_action(omrat, menu: str, entry: str):
    for m, e, action in menu_entries(omrat):
        if m == menu and e == entry:
            return action
    raise AssertionError(f'no menu entry {menu} > {entry}')


def _inside(widget: QWidget, kinds: tuple[type, ...], stop: QWidget) -> bool:
    p = widget.parentWidget()
    while p is not None and p is not stop:
        if isinstance(p, kinds):
            return True
        p = p.parentWidget()
    return False


def _hidden_on_purpose(widget: QWidget, stop: QWidget) -> bool:
    """Hidden by the code (not merely on a tab that is not shown)."""
    w = widget
    while w is not None and w is not stop:
        if w.isHidden() and not isinstance(w.parentWidget(), QStackedWidget):
            return True
        w = w.parentWidget()
    return False


def button_label(button: QAbstractButton) -> str:
    text = button.text().replace('&', '') or type(button).__name__
    if button.objectName():
        return button.objectName()
    parent = button.parentWidget()
    viewport_owner = parent.parentWidget() if parent is not None else None
    if isinstance(viewport_owner, QTableWidget):          # a cell widget
        index = viewport_owner.indexAt(button.pos())
        return f'{viewport_owner.objectName() or "table"}[{index.row()},{index.column()}]:{text}'
    while parent is not None and not parent.objectName():
        parent = parent.parentWidget()
    where = parent.objectName() if parent is not None else '?'
    return f'{where}:{text}'


_NOT_OURS = (QTabBar, QToolBar, QDialogButtonBox, QAbstractSpinBox, QScrollBar, QMenuBar)


def dock_buttons(dock: QWidget) -> list[QAbstractButton]:
    """Push and tool buttons of the dock a user can click (Qt's own scroll
    arrows, toolbars, spin boxes and button boxes left out)."""
    out = []
    for b in dock.findChildren(QAbstractButton):
        if not isinstance(b, (QPushButton, QToolButton)):
            continue
        if _inside(b, _NOT_OURS, dock):
            continue
        out.append(b)
    return out


_SAFE_BOX_ROLES = (QDialogButtonBox.ButtonRole.ResetRole, QDialogButtonBox.ButtonRole.ApplyRole,
                   QDialogButtonBox.ButtonRole.HelpRole)


def dialog_buttons(dlg: QWidget) -> list[QAbstractButton]:
    """The buttons of a dialog except the ones that close it (OK / Cancel /
    Save / Discard ...) or step a wizard: every own button plus Reset /
    Apply / Help."""
    out = [b for b in dock_buttons(dlg) if not b.objectName().startswith('qt_wizard_')]
    for box in dlg.findChildren(QDialogButtonBox):
        out += [b for b in box.buttons() if box.buttonRole(b) in _SAFE_BOX_ROLES]
    return out


_MAX_DEPTH = 2
_MAX_WIZARD_PAGES = 10


def _click_buttons_of(guard: 'ModalGuard', root: QWidget) -> None:
    for button in dialog_buttons(root):
        if not clickable(button, root):
            continue
        bring_to_front(button, root)
        guard.where.append(button_label(button))
        guard.inner_clicked.append(' > '.join(guard.where))
        try:
            button.click()
            settle()
        except Exception:
            guard.errors.append(guard._here() + traceback.format_exc())
        finally:
            guard.where.pop()


def click_inner_buttons(guard: 'ModalGuard', dlg: QWidget) -> None:
    """Click every button of ``dlg`` (``dialog_buttons``) once, recording the
    path in ``guard.where``; a dialog opened by one of them is explored too,
    down to ``_MAX_DEPTH`` levels.  A wizard is walked page by page with
    Next, as far as its pages let a user go."""
    if len(guard.where) >= _MAX_DEPTH * 2:
        return
    guard.where.append(type(dlg).__name__)
    try:
        if not isinstance(dlg, QWizard):
            _click_buttons_of(guard, dlg)
            return
        dlg.restart()
        settle()
        for _ in range(_MAX_WIZARD_PAGES):
            page = dlg.currentPage()
            if page is None:
                break
            guard.where.append(f'page {dlg.currentId()} {type(page).__name__}')
            guard.inner_clicked.append(' > '.join(guard.where))
            try:
                _click_buttons_of(guard, page)
            finally:
                guard.where.pop()
            before = dlg.currentId()
            dlg.next()
            settle()
            if dlg.currentId() == before:
                break
    finally:
        guard.where.pop()


def bring_to_front(widget: QWidget, stop: QWidget) -> None:
    """Make every tab on the way to ``widget`` the current one, like a user would."""
    w = widget
    while w is not None and w is not stop:
        parent = w.parentWidget()
        if isinstance(parent, QStackedWidget) and isinstance(parent.parentWidget(), QTabWidget):
            parent.parentWidget().setCurrentWidget(w)
        w = parent


def clickable(button: QAbstractButton, dock: QWidget) -> bool:
    return (not sip.isdeleted(button) and button.isEnabled()
            and not _hidden_on_purpose(button, dock))


def window_addresses() -> set[int]:
    return {sip.unwrapinstance(w) for w in QApplication.topLevelWidgets()}


def new_visible_windows(before: set[int]) -> list[QWidget]:
    return [w for w in QApplication.topLevelWidgets()
            if sip.unwrapinstance(w) not in before and w.isVisible()]


def close_new_windows(before: set[int], keep: Iterable[QWidget] = ()) -> None:
    """Close the dialogs an action left open (``exec()`` is stubbed, so a
    dialog shown before it stays up); other new windows are hidden."""
    keep_ids = {sip.unwrapinstance(w) for w in keep}
    for w in new_visible_windows(before):
        if sip.unwrapinstance(w) in keep_ids:
            continue
        if isinstance(w, QDialog):
            w.close()
        else:
            w.hide()
    settle()


def settle() -> None:
    """Run the posted events and zero-delay timers an action queued."""
    for _ in range(3):
        QCoreApplication.processEvents()


def load_example(omrat, tmp_path: Path) -> Path:
    """Load a *copy* of ``proj.omrat`` (File -> Save then writes the copy)."""
    from omrat_utils.storage import Storage
    target = tmp_path / 'proj.omrat'
    shutil.copy(EXAMPLE, target)
    Storage(omrat).load_from_path(str(target))
    settle()
    return target
