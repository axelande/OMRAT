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
        self.area = None
        self.object_id = 0
        self.area_type = ''
        self.loaded_areas = []
        
    def add_area(self, name='area') -> QgsVectorLayer:
        if self.area is not None:
            try:
                self.area.featureAdded.disconnect(self.on_feature_added)
            except Exception:
                pass
        self.area = QgsVectorLayer("Polygon?crs=epsg:4326", name, "memory")
        self.area.startEditing()
        self.area.featureAdded.connect(self.on_feature_added)
        self.p.iface.actionAddFeature().trigger()
        QgsProject.instance().addMapLayer(self.area)
    
    def on_feature_added(self, fid):
        """This will be called after the user right-clicks to finish the polygon"""
        try:
            self.area.featureAdded.disconnect(self.on_feature_added)
            print('dictonnected')
        except Exception as e:
            print(e)
        if self.area_type == 'depth':
            self.add_simple_depth()
        if self.area_type == 'object':
            self.add_simple_object()

    
    def load_area(self, name, wkt):
        area = QgsVectorLayer("Polygon?crs=epsg:4326", name, "memory")
        pr = area.dataProvider()
        fet = QgsFeature()
        fet.setGeometry(QgsGeometry.fromWkt(wkt))
        pr.addFeatures( [ fet ] )
        QgsProject.instance().addMapLayer(area)
        self.p.iface.actionSaveActiveLayerEdits().trigger()
        self.loaded_areas.append(area)
    
    def add_simple_depth(self):
        self.area_type = 'depth'
        if self.p.dockwidget.pbAddSimpleDepth.text() == 'Save':
            self.store_depth()
            self.p.dockwidget.pbAddSimpleDepth.setText('Add manual')
            self.loaded_areas.append(self.area)
        else:
            self.deph_id += 1
            self.add_area(f'Depth_{self.deph_id}')
            self.p.dockwidget.pbAddSimpleDepth.setText('Save')
            
    def store_depth(self):
        self.p.dockwidget.twDepthList.setRowCount(self.deph_id)
        item1 = QTableWidgetItem(f'{self.deph_id}')
        item2 = QTableWidgetItem(f'10')
        polies = self.area.getFeatures()
        for poly in polies:
            item3 = QTableWidgetItem(f'{poly.geometry().asWkt(precision=5)}')
            self.p.dockwidget.twDepthList.setItem(self.deph_id - 1, 0, item1)
            self.p.dockwidget.twDepthList.setItem(self.deph_id - 1, 1, item2)
            self.p.dockwidget.twDepthList.setItem(self.deph_id - 1, 2, item3)
        self.p.iface.actionSaveActiveLayerEdits().trigger()
        self.p.iface.actionToggleEditing().trigger()
        
    def add_simple_object(self):
        self.area_type = 'object'
        if self.p.dockwidget.pbAddSimpleObject.text() == 'Save':
            self.store_object()
            self.p.dockwidget.pbAddSimpleObject.setText('Add manual')
            self.loaded_areas.append(self.area)
        else:
            self.object_id += 1
            self.add_area(f'Object_{self.object_id}')
            self.p.dockwidget.pbAddSimpleObject.setText('Save')
            
    def store_object(self):
        self.p.dockwidget.twObjectList.setRowCount(self.object_id)
        item1 = QTableWidgetItem(f'{self.object_id}')
        item2 = QTableWidgetItem(f'10')
        polies = self.area.getFeatures()
        for poly in polies:
            item3 = QTableWidgetItem(f'{poly.geometry().asWkt(precision=5)}')
            self.p.dockwidget.twObjectList.setItem(self.object_id - 1, 0, item1)
            self.p.dockwidget.twObjectList.setItem(self.object_id - 1, 1, item2)
            self.p.dockwidget.twObjectList.setItem(self.object_id - 1, 2, item3)
        self.p.iface.actionSaveActiveLayerEdits().trigger()
        self.p.iface.actionToggleEditing().trigger()
        
    def unload(self):
        if self.area is not None:
            try:
                self.area.featureAdded.disconnect(self.on_feature_added)
            except TypeError:
                pass
        self.area = None

        for layer in self.loaded_areas:
            QgsProject.instance().removeMapLayer(layer.id())  # Remove the layer from QGIS
        self.loaded_areas = []

        