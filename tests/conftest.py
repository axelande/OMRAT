import pytest
from pytest_qgis import qgis_iface  # noqa: F401 - fixture
from unittest.mock import patch, MagicMock

import qgis  # noqa: F401
from qgis.core import QgsProject

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
        omrat = OMRAT(qgis_iface, True)
        omrat.run()
        yield omrat
        # Clean up after the test
        QgsProject.instance().removeAllMapLayers()
