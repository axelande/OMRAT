from dataclasses import dataclass, field
from operator import xor
import json
from functools import partial
from typing import Optional, Any, TYPE_CHECKING, cast, Union
if TYPE_CHECKING:
    from omrat import OMRAT, OMRATMainWidget

import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import numpy as np
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QTableWidgetItem, QSpinBox, QDoubleSpinBox, QLineEdit
from scipy import stats

from ui.traffic_data_widget import TrafficDataWidget
from geometries import isint


WidgetType = Union[QLineEdit, QSpinBox, QDoubleSpinBox]

class Traffic:
    def __init__(self, omrat: "OMRAT", dw:"OMRATMainWidget"):
        self.dw = dw
        self.omrat = omrat
        self.traffic_data: dict[str, dict[str, dict[str, Any]]] = self.omrat.traffic_data
        self.c_seg = "1"
        self.c_di = ""
        self.current_table = []
        self.variables = ['Frequency (ships/year)', 'Speed (knots)', 'Draught (meters)', 'Ship heights (meters)', 'Ship Beam (meters)']
        self.last_var = 'Frequency (ships/year)'
        self.run_update = False
        self.dw.cbSelectType.currentIndexChanged.connect(partial(self.update_traffic_tbl, 'type'))
        self.dw.cbTrafficSelectSeg.currentIndexChanged.connect(partial(self.update_traffic_tbl, 'segment'))
        self.dw.cbTrafficDirectionSelect.currentIndexChanged.connect(partial(self.update_traffic_tbl, 'dir'))            
        self.set_table_headings()
        self.run_update = True
        
    def fill_cbTrafficSelectSeg(self):
        """Sets the segment names in cbTrafficSelectSeg"""
        self.dw.cbTrafficSelectSeg.clear()
        nrs = self.dw.twRouteList.rowCount()
        for i in range(nrs):
            self.dw.cbTrafficSelectSeg.addItem(self.dw.twRouteList.item(i, 0).text())
        self.c_seg = self.dw.cbTrafficSelectSeg.currentText()

    def set_table_headings(self):
        """Sets the column and row names of the table"""
        types: list[str] = []
        for i in range(self.omrat.ship_cat.scw.cvTypes.rowCount()):
            it = self.omrat.ship_cat.scw.cvTypes.item(i, 0)
            text = it.text() if it is not None else ""
            types.append(text)
        sizes: list[str] = []
        for i in range(self.omrat.ship_cat.scw.twLengths.rowCount()):
            it1 = self.omrat.ship_cat.scw.twLengths.item(i, 0)
            text1 = it1.text() if it1 is not None else ""
            it2 = self.omrat.ship_cat.scw.twLengths.item(i, 1)
            text2 = it2.text() if it2 is not None else ""
            sizes.append(f'{text1} - {text2}')
        self.dw.twTrafficData.setColumnCount(len(sizes))
        self.dw.twTrafficData.setHorizontalHeaderLabels(sizes)
        self.dw.twTrafficData.setRowCount(len(types))
        self.dw.twTrafficData.setVerticalHeaderLabels(types)
        for row in range(len(types)):
            for col in range(len(sizes)):
                item = QSpinBox()
                item.setMaximum(100000)
                self.dw.twTrafficData.setCellWidget(row, col, item)
            
    def create_empty_dict(self, s_key:str, dirs:list[str]):
        """Creates an empty dict for the segment with all types"""
        self.traffic_data[s_key] = {}
        rows = self.dw.twTrafficData.rowCount()
        cols = self.dw.twTrafficData.columnCount()
        
        for di in dirs:
            self.traffic_data[s_key][di] = {}
            for idx, key in enumerate(self.variables):
                self.traffic_data[s_key][di][key] = []
                for _ in range(rows):
                    line:list[Any] = []
                    for _ in range(cols):
                        if idx == 0:
                            line.append(0)
                        else:
                            line.append([])
                    self.traffic_data[s_key][di][key].append(line)
            
    def update_traffic_tbl(self, caller:str):
        """Updates Traffic data table with the data from traffic_data, 
        using the correct type and segment"""
        if self.run_update:
            self.save()
        if caller == 'segment':
            self.c_seg = self.dw.cbTrafficSelectSeg.currentText()
            self.update_direction_select()
        rows = self.dw.twTrafficData.rowCount()
        cols = self.dw.twTrafficData.columnCount()
        self.last_var: str = self.dw.cbSelectType.currentText()
        self.c_di = self.dw.cbTrafficDirectionSelect.currentText()
        if any([self.c_seg== "", self.c_di== "", self.last_var== ""]):
            return    
        for row in range(rows):
            for col in range(cols):
                val:float|int|str = self.traffic_data[self.c_seg][self.c_di][self.last_var][row][col]

                if val == '':
                    item = QSpinBox()
                    val = 0
                elif val == np.inf:
                    item = QSpinBox()
                    item.setEnabled(False)
                    val = 0
                elif isint(val):
                    item = QSpinBox()
                    item.setMaximum(100000)
                    val = int(val)
                else:
                    item = QDoubleSpinBox()
                    val = float(val)
                item.setValue(val)
                #item.setItemDelegate(self.deligate)
                self.dw.twTrafficData.setCellWidget(row, col, item)
        
    def update_direction_select(self):
        self.run_update = False
        self.dw.cbTrafficDirectionSelect.clear()
        if len(self.traffic_data) == 0 or self.c_seg == '':
            return
        for key in self.traffic_data[self.c_seg].keys():
            self.dw.cbTrafficDirectionSelect.addItem(key)
        self.c_di = self.dw.cbTrafficDirectionSelect.currentText()
        self.run_update = True
    
    def save(self):
        """Saves the previous "table" in traffic_data"""
        if any([self.c_seg == "", self.c_di == "", self.last_var == ""]):
            return
        rows = self.dw.twTrafficData.rowCount()
        cols = self.dw.twTrafficData.columnCount()
        typ = self.last_var
        for row in range(rows):
            for col in range(cols):
                val = self.dw.twTrafficData.cellWidget(row, col)
                self.traffic_data[self.c_seg][self.c_di][typ][row][col] = val.value()
                    
    def commit_changes(self):
        """Copy the output from the traffic data within this module to omrats traffic_data"""
        self.save()
        self.omrat.traffic_data = self.traffic_data

    def unload(self):
        """Cleanup resources and disconnect signals."""
        # Clear traffic data
        self.traffic_data.clear()
        try:
            self.dw.cbSelectType.currentIndexChanged.disconnect()
            self.dw.cbTrafficSelectSeg.currentIndexChanged.disconnect()
            self.dw.cbTrafficDirectionSelect.currentIndexChanged.disconnect()
        except TypeError:
            pass
        # Remove reference to TrafficDataWidget
        print("Traffic resources cleaned up.")
