# import qgis libs so that ve set the correct sip api version
import pytest
from pytest_qgis import qgis_iface
import qgis   # pylint: disable=W0611  # NOQA
from qgis.core import QgsProject

from open_mrat import OpenMRAT


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
    omrat = OpenMRAT(qgis_iface, True)
    omrat.run()
    yield omrat
    # Clean up after the test
    QgsProject.instance().removeAllMapLayers()