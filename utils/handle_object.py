from __future__ import annotations
import json
import os
from typing import TYPE_CHECKING

from qgis.PyQt.QtCore import QSettings, QVariant
from qgis.PyQt.QtWidgets import QFileDialog, QTableWidgetItem
from qgis.core import QgsProject, QgsVectorLayer, QgsField, QgsFeature, QgsPolygon, QgsGeometry


if TYPE_CHECKING:
    from open_mrat import OpenMRAT


class OObject:
    def __init__(self, parent: OpenMRAT) -> None:
        self.p = parent
        self.deph_id = 0
        self.depth_area = None
        self.object_id = 0
        self.object_area = None
        
    def add_area(self, name='area') -> QgsVectorLayer:
        area = QgsVectorLayer("Polygon?crs=epsg:4326", name, "memory")
        area.startEditing()
        self.p.iface.actionAddFeature().trigger()
        QgsProject.instance().addMapLayer(area)
        return area
    
    def load_area(self, name, wkt):
        area = QgsVectorLayer("Polygon?crs=epsg:4326", name, "memory")
        pr = area.dataProvider()
        fet = QgsFeature()
        fet.setGeometry(QgsGeometry.fromWkt(wkt))
        pr.addFeatures( [ fet ] )
        QgsProject.instance().addMapLayer(area)
        self.p.iface.actionSaveActiveLayerEdits().trigger()  
    
    def add_simple_depth(self):
        if self.p.dockwidget.pbAddSimpleDepth.text() == 'Save':
            self.store_depth()
            self.p.dockwidget.pbAddSimpleDepth.setText('Add manual')
        else:
            self.deph_id += 1
            self.depth_area = self.add_area(f'Depth_{self.deph_id}')
            self.p.dockwidget.pbAddSimpleDepth.setText('Save')
            
    def store_depth(self):
        self.p.dockwidget.twDepthList.setRowCount(self.deph_id)
        item1 = QTableWidgetItem(f'{self.deph_id}')
        item2 = QTableWidgetItem(f'10')
        polies = self.depth_area.getFeatures()
        for poly in polies:
            item3 = QTableWidgetItem(f'{poly.geometry().asWkt(precision=5)}')
            self.p.dockwidget.twDepthList.setItem(self.deph_id - 1, 0, item1)
            self.p.dockwidget.twDepthList.setItem(self.deph_id - 1, 1, item2)
            self.p.dockwidget.twDepthList.setItem(self.deph_id - 1, 2, item3)
        self.p.iface.actionSaveActiveLayerEdits().trigger()
        self.p.iface.actionToggleEditing().trigger()
        
    def add_simple_object(self):
        if self.p.dockwidget.pbAddSimpleObject.text() == 'Save':
            self.store_object()
            self.p.dockwidget.pbAddSimpleObject.setText('Add manual')
        else:
            self.object_id += 1
            self.object_area = self.add_area(f'Object_{self.object_id}')
            self.p.dockwidget.pbAddSimpleObject.setText('Save')
            
    def store_object(self):
        self.p.dockwidget.twObjectList.setRowCount(self.object_id)
        item1 = QTableWidgetItem(f'{self.object_id}')
        item2 = QTableWidgetItem(f'10')
        polies = self.object_area.getFeatures()
        for poly in polies:
            item3 = QTableWidgetItem(f'{poly.geometry().asWkt(precision=5)}')
            self.p.dockwidget.twObjectList.setItem(self.object_id - 1, 0, item1)
            self.p.dockwidget.twObjectList.setItem(self.object_id - 1, 1, item2)
            self.p.dockwidget.twObjectList.setItem(self.object_id - 1, 2, item3)
        self.p.iface.actionSaveActiveLayerEdits().trigger()
        self.p.iface.actionToggleEditing().trigger()

        