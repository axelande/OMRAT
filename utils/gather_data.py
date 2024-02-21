from __future__ import annotations
import json
import os
from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtWidgets import QTableWidget, QTableWidgetItem

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from open_mrat import OpenMRAT
    
class GatherData:
    def __init__(self, parent: OpenMRAT) -> None:
        self.p = parent
        self.data = {}
        
    def gather_drift(self):
        n = float(self.p.dockwidget.leDriftN.text())
        ne = float(self.p.dockwidget.leDriftNE.text())
        e = float(self.p.dockwidget.leDriftE.text())
        se = float(self.p.dockwidget.leDriftSE.text())
        s = float(self.p.dockwidget.leDriftS.text())
        sw = float(self.p.dockwidget.leDriftSW.text())
        w = float(self.p.dockwidget.leDriftW.text())
        nw = float(self.p.dockwidget.leDriftNW.text())
        rose = {0: n, 45: ne, 90: e, 135: se, 180: s, 225: sw, 270: w, 315: nw}
        speed = float(self.p.dockwidget.leDriftSpeed.text())
        drift_p = float(self.p.dockwidget.leDriftProb.text())
        anchor_p = float(self.p.dockwidget.leAnchorProb.text())
        repair = {'func': self.p.dockwidget.leRepairFunc.toPlainText(),
                  'std': float(self.p.dockwidget.leRepairStd.text()),
                  'loc': float(self.p.dockwidget.leRepairLoc.text()),
                  'scale': float(self.p.dockwidget.leRepairScale.text()),
                  'active_window': self.p.dockwidget.tabRepair.currentIndex()}
        self.data['drift'] = {'drift_p': drift_p, 'anchor_p':anchor_p, 'speed':speed,
                              'rose': rose, 'repair': repair}
        
    def get_segment_tbl(self):
        """Extends the segment_data in self.data, must be called after it is created."""
        for j, col in enumerate(['Segment Id', 'Route Id', 'Start Point', 'End Point']):
            for i, key in enumerate(self.data['segment_data'].keys()):
                value = self.p.dockwidget.twRouteList.item(i, j).text()
                self.data["segment_data"][key][col] = value
        
    def get_causation_factors(self):
        p_pc = float(self.p.dockwidget.lePoweredPc.text())
        d_pc = float(self.p.dockwidget.leDriftingPc.text())
        ai = float(self.p.dockwidget.leMeanTimeBetChecks.text())
        self.data['pc'] = {'p_pc': p_pc, 'd_pc': d_pc, 'ai': ai}
        
    def get_all_for_save(self):
        self.gather_drift()
        self.get_causation_factors()
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
        self.populate_drift(data['drift'])
        self.populate_pc(data)
        self.p.traffic_data = data['traffic_data'] 
        self.p.segment_data = data['segment_data']
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
                
    def populate_pc(self, data):
        self.p.dockwidget.lePoweredPc.setText(f"{data['pc']['p_pc']}")
        self.p.dockwidget.leDriftingPc.setText(f"{data['pc']['d_pc']}")
        self.p.dockwidget.leMeanTimeBetChecks.setText(f"{data['pc']['ai']}")
    
    def populate_drift(self, data:dict):
        """Populates the drift fields with the "data" dict """
        self.p.dockwidget.leDriftN.setText(f"{data['rose']['0']}")
        self.p.dockwidget.leDriftNE.setText(f"{data['rose']['45']}")
        self.p.dockwidget.leDriftE.setText(f"{data['rose']['90']}")
        self.p.dockwidget.leDriftSE.setText(f"{data['rose']['135']}")
        self.p.dockwidget.leDriftS.setText(f"{data['rose']['180']}")
        self.p.dockwidget.leDriftSW.setText(f"{data['rose']['225']}")
        self.p.dockwidget.leDriftW.setText(f"{data['rose']['270']}")
        self.p.dockwidget.leDriftNW.setText(f"{data['rose']['315']}")
        self.p.dockwidget.leDriftSpeed.setText(f"{data['speed']}")
        self.p.dockwidget.leAnchorProb.setText(f"{data['anchor_p']}")
        self.p.dockwidget.leDriftProb.setText(f"{data['drift_p']}")
        self.p.dockwidget.leRepairFunc.setText(f"{data['repair']['func']}")
        self.p.dockwidget.leRepairStd.setText(f"{data['repair']['std']}")
        self.p.dockwidget.leRepairLoc.setText(f"{data['repair']['loc']}")
        self.p.dockwidget.leRepairScale.setText(f"{data['repair']['scale']}")
        self.p.dockwidget.tabRepair.setCurrentIndex(int(data['repair']['active_window']))

    def populate_tbl(self, data:list, tbl:QTableWidget):
        tbl.setRowCount(len(data))
        for i, line in enumerate(data):
            for j, value in enumerate(line):
                item = QTableWidgetItem(value)
                tbl.setItem(i, j, item)
    
    def populate_segment_tbl(self, data:dict, tbl:QTableWidget):
        tbl.setRowCount(len(data))
        for j, col in enumerate(['Segment Id', 'Route Id', 'Start Point', 'End Point']):
            for i, key in enumerate(data.keys()):
                item = QTableWidgetItem(data[key][col])
                self.p.dockwidget.twRouteList.setItem(i, j, item)
