from qgis.core import QgsVectorLayer, QgsPointXY, QgsGeometry, QgsField, QgsFeature
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtWidgets import QApplication

from ..omrat import OMRAT
from conftest import omrat


def create_object(points) -> QgsVectorLayer:
    lyr = QgsVectorLayer('Polygon?crs=epsg:4326', 'temp',"memory")
    pr = lyr.dataProvider()
    pr.addAttributes([QgsField("AreaId",  QVariant.Int)])
    lyr.updateFields()
    feat = QgsFeature(lyr.fields())
    feat.setAttribute("AreaId", 0)
    pointxys = []
    for coord in points:
        pointxys.append(QgsPointXY(coord[0], coord[1])) 
    geom = QgsGeometry.fromMultiPolygonXY([[pointxys]])
    feat.setGeometry(geom)
    return lyr


def test_add_object(omrat: OMRAT):
    omrat.main_widget.pbAddSimpleObject.click()
    obj = create_object([[17.5136058924, 56.8086347034],
                         [17.5137435342, 56.6331630685],
                         [17.5398315995, 56.6836007601],
                         [17.5362169055, 56.7324570938],
                         [17.5136058924, 56.8086347034]])
    omrat.object.object_area = obj
    omrat.main_widget.pbAddSimpleObject.click()
    assert omrat.main_widget.twObjectList.rowCount() == 1
