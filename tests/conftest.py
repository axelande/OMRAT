import sys

import pytest
from pytest_qgis import qgis_iface  # noqa: F401 - fixture
from unittest.mock import patch, MagicMock

import qgis  # noqa: F401
from qgis.core import QgsProject
from qgis.PyQt import sip
from qgis.PyQt.QtCore import QCoreApplication, QEvent
from qgis.PyQt.QtWidgets import QApplication

from omrat import OMRAT


class actionAddFeature:
    def trigger(self):
        pass


class actionSaveActiveLayerEdits:
    def trigger(self):
        pass


class actionToggleEditing:
    def trigger(self):
        pass


def _window_addresses() -> set[int]:
    """C++ addresses of the open top-level windows (a wrapper made for a
    C++-created window is new on every call, so ``id()`` would not do)."""
    return {sip.unwrapinstance(w) for w in QApplication.topLevelWidgets()}


def _dispose(omrat: OMRAT, windows_before: set[int]) -> None:
    """Release one test's plugin: its project signal, every window opened
    since ``windows_before`` (the OMRAT window and any dialog a test left
    open) and its matplotlib figures.  Before this every test that used
    the fixture left its window open until pytest exited (300+ windows),
    and each one stayed connected to ``QgsProject.cleared``.

    ``hide()``, never ``close()``: closing runs the plugin's "save
    changes?" prompt and clears the model.
    """
    try:
        QgsProject.instance().cleared.disconnect(omrat._on_project_cleared)
    except (TypeError, RuntimeError):
        pass
    # Figures first: their managers own Qt windows / toolbars that the
    # loop below would delete under them.
    plt = sys.modules.get('matplotlib.pyplot')
    if plt is not None:
        plt.close('all')
    for window in QApplication.topLevelWidgets():
        try:
            if sip.unwrapinstance(window) in windows_before:
                continue
            window.hide()
            window.deleteLater()
        except RuntimeError:
            continue
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture(scope='function')
def omrat(qgis_iface, tmp_path_factory):  # noqa: F811
    qgis_iface.actionAddFeature = actionAddFeature
    qgis_iface.actionSaveActiveLayerEdits = actionSaveActiveLayerEdits
    qgis_iface.actionToggleEditing = actionToggleEditing

    # Patch DB before creating OMRAT/AIS.  The run history goes to a temp
    # file: opening the plugin refreshes Previous runs, which writes to
    # the history (schema, version backfill) -- never the user's own.
    with patch("omrat_utils.handle_ais.DB") as MockDB, \
            patch("omrat_utils.run_history.default_db_path",
                  return_value=tmp_path_factory.mktemp('history') / 'omrat_history.sqlite'):
        MockDB.return_value = MagicMock()
        windows_before = _window_addresses()
        omrat = OMRAT(qgis_iface, True)
        omrat.run()
        yield omrat
        # Clean up after the test: the layers first, while the plugin that
        # listens to their removal still has its widgets.
        QgsProject.instance().removeAllMapLayers()
        _dispose(omrat, windows_before)
