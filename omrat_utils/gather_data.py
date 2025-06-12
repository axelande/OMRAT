from __future__ import annotations
import json
import os
from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtWidgets import QTableWidget, QTableWidgetItem

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omrat import OMRAT
    
class GatherData:
    def __init__(self, parent: OMRAT) -> None:
        self.p = parent
        self.data = {}
               
    def get_segment_tbl(self):
        """Extends the segment_data in self.data, must be called after it is created."""
        for j, col in enumerate(['Segment Id', 'Route Id', 'Start Point', 'End Point', 'Width']):
            for i, key in enumerate(self.data['segment_data'].keys()):
                value = self.p.dockwidget.twRouteList.item(i, j).text()
                self.data["segment_data"][key][col] = value
        
    def get_all_for_save(self) -> dict:
        self.data['pc'] = self.p.causation_f.data
        self.data['drift'] = self.p.drift_values
        self.p.traffic.change_dist_segment() # Saves the current settings on the leg
        self.data['traffic_data'] = self.p.traffic_data
        self.data['segment_data'] = self.p.segment_data
        self.get_segment_tbl()
        self.data['depths'] = self.obtain_table_data(self.p.dockwidget.twDepthList)
        self.data['objects'] = self.obtain_table_data(self.p.dockwidget.twObjectList)
        return self.data
    
    def obtain_table_data(self, tbl) -> list:
        """Obtain data from a table"""
        tbl_data = []
        rows = tbl.rowCount()
        cols = tbl.columnCount()
        for row in range(rows):
            line = []
            for col in range(cols):
                value = tbl.item(row, col)
                if value is not None:
                    line.append(value.text())
            tbl_data.append(line)
        return tbl_data
    
    def populate(self, data):
        self.p.traffic_data = data['traffic_data'] 
        self.p.segment_data = data['segment_data']
        self.p.drift_values = data['drift']
        self.p.drift_settings.drift_values = data['drift']
        self.populate_segment_tbl(data['segment_data'], self.p.dockwidget.twRouteList)
        self.populate_cbTrafficSelectSeg()
        self.p.traffic.change_dist_segment()
        self.populate_tbl(data['depths'], self.p.dockwidget.twDepthList)
        self.populate_tbl(data['objects'], self.p.dockwidget.twObjectList)
        
        # Load data to canvas
        self.p.load_lines(data)
        for dep in data["depths"]:
            self.p.object.load_area('Depth - ' + dep[0], dep[2])
        for dep in data["objects"]:
            self.p.object.load_area('Structure - ' + dep[0], dep[2])
            
    def populate_cbTrafficSelectSeg(self):
        """Sets the segment names in cbTrafficSelectSeg"""
        self.p.dockwidget.cbTrafficSelectSeg.clear()
        for key in self.p.segment_data.keys():
            self.p.dockwidget.cbTrafficSelectSeg.addItem(str(key))
        self.p.traffic.c_seg = self.p.dockwidget.cbTrafficSelectSeg.currentText()

    def populate_tbl(self, data:list, tbl:QTableWidget):
        tbl.setRowCount(len(data))
        for i, line in enumerate(data):
            for j, value in enumerate(line):
                item = QTableWidgetItem(value)
                tbl.setItem(i, j, item)
    
    def populate_segment_tbl(self, data:dict, tbl:QTableWidget):
        tbl.setRowCount(len(data))
        print('data')
        print(data)
        for j, col in enumerate(['Segment Id', 'Route Id', 'Start Point', 'End Point', 'Width']):
            for i, key in enumerate(data.keys()):
                print(col)
                item = QTableWidgetItem(str(data[key][col]))
                print(data[key][col])
                self.p.dockwidget.twRouteList.setItem(i, j, item)
