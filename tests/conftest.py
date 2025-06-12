import pytest
from pytest_qgis import qgis_iface
from unittest.mock import patch, MagicMock

import qgis
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
def omrat(qgis_iface):
    qgis_iface.actionAddFeature = actionAddFeature
    qgis_iface.actionSaveActiveLayerEdits = actionSaveActiveLayerEdits
    qgis_iface.actionToggleEditing = actionToggleEditing
    
    # Patch DB before creating OMRAT/AIS
    with patch("omrat_utils.handle_ais.DB") as MockDB:
        MockDB.return_value = MagicMock()
        omrat = OMRAT(qgis_iface, True)
        omrat.run()
        yield omrat
        # Clean up after the test
        QgsProject.instance().removeAllMapLayers()