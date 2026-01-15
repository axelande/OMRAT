from __future__ import annotations
import json
import os
import copy
from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtWidgets import QTableWidget, QTableWidgetItem
from typing import Any
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omrat import OMRAT


def dict_ndarray_to_list(data: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, list]]:
    """Convert all np.ndarray values in a nested dict to lists."""
    result: dict[str, dict[str, list]] = {}
    for key, inner in data.items():
        result[key] = {}
        for subkey, value in inner.items():
            if isinstance(value, np.ndarray):
                result[key][subkey] = value.tolist()
            else:
                result[key][subkey] = value
    return result

def dict_list_to_ndarray(data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convert all list values in a nested dict to np.ndarray."""
    result: dict[str, dict[str, Any]] = {}
    for key, inner in data.items():
        result[key] = {}
        for subkey, value in inner.items():
            if isinstance(value, list):
                result[key][subkey] = np.array(value)
            else:
                result[key][subkey] = value
    return result
    
class GatherData:
    def __init__(self, parent: OMRAT) -> None:
        self.p = parent
        self.data = {}
               
    def get_segment_tbl(self):
        """Extends the segment_data in self.data, must be called after it is created."""
        for j, col in enumerate(['Segment_Id', 'Route_Id', 'Leg_name', 'Start_Point', 'End_Point', 'Width']):
            for i, key in enumerate(self.data['segment_data'].keys()):
                item = self.p.main_widget.twRouteList.item(i, j)
                if item is None:
                    return
                value = item.text()
                self.data["segment_data"][key][col] = copy.deepcopy(value)
        
    def get_all_for_save(self) -> dict[str, Any]:
        self.data['pc'] = copy.deepcopy(self.p.causation_f.data)
        self.data['drift'] = copy.deepcopy(self.p.drift_values)
        self.p.distributions.change_dist_segment(self.p.distributions.last_id) # Saves the current settings on the leg
        self.data['traffic_data'] = copy.deepcopy(self.p.traffic_data)
        self.data['segment_data'] = copy.deepcopy(self.p.segment_data)
        for key, item in self.data['segment_data'].items():
            self.data['segment_data'][key]['dist1'] = copy.deepcopy(item['dist1'].tolist())
            self.data['segment_data'][key]['dist2'] = copy.deepcopy(item['dist2'].tolist())
        self.get_segment_tbl()
        
        self.data['depths'] = []
        self.data['objects'] = []
        self.copy_depths_and_objects()
        # Persist the ship category scheme (types and size intervals) used to build traffic tables
        self.data['ship_categories'] = self.get_ship_categories_for_save()
        return self.data
    
    def copy_depths_and_objects(self):
        # Read rows from QTableWidget instead of iterating the widget
        depth_rows = self.obtain_table_data(self.p.main_widget.twDepthList)
        object_rows = self.obtain_table_data(self.p.main_widget.twObjectList)
        for row in depth_rows:
            if len(row) < 3:
                # Skip incomplete rows
                continue
            id_ = row[0]
            depths = row[1]
            polygon = row[2]
            # Support values like "48-...": take the first part as depth
            depth_val = depths.split('-')[0] if isinstance(depths, str) else str(depths)
            # Store as list [id, depth, polygon] per schema
            self.data['depths'].append([str(id_), str(depth_val), str(polygon)])
        for row in object_rows:
            if len(row) < 3:
                continue
            id_ = row[0]
            height = row[1]
            polygon = row[2]
            # Store as list [id, height, polygon] per schema
            self.data['objects'].append([str(id_), str(height), str(polygon)])

    def get_ship_categories_for_save(self) -> dict[str, Any]:
        """Capture the ship types (rows) and size intervals (columns) used for traffic tables.

        Returns a dict with keys:
        - types: list[str] of ship type labels (row headers from cvTypes)
        - length_intervals: list[dict] with keys {'min': float, 'max': float, 'label': str}
        - selection_mode: optional str indicating UI mode (e.g., 'simple_ais' or 'manual')
        """
        result: dict[str, Any] = {
            'types': [],
            'length_intervals': [],
        }
        try:
            # Extract ship types from Ship Categories widget
            scw = self.p.ship_cat.scw  # ShipCategoriesWidget
            types: list[str] = []
            if hasattr(scw, 'cvTypes') and scw.cvTypes is not None:
                rows = scw.cvTypes.rowCount()
                for i in range(rows):
                    it = scw.cvTypes.item(i, 0)
                    text = it.text() if it is not None else ''
                    if text:
                        types.append(text)
            result['types'] = types

            # Extract size intervals from twLengths (Min/Max columns)
            intervals: list[dict[str, Any]] = []
            if hasattr(scw, 'twLengths') and scw.twLengths is not None:
                rows = scw.twLengths.rowCount()
                for i in range(rows):
                    it_min = scw.twLengths.item(i, 0)
                    it_max = scw.twLengths.item(i, 1)
                    smin = it_min.text() if it_min is not None else ''
                    smax = it_max.text() if it_max is not None else ''
                    if smin == '' and smax == '':
                        continue
                    try:
                        vmin = float(smin)
                    except Exception:
                        vmin = smin
                    try:
                        vmax = float(smax)
                    except Exception:
                        vmax = smax
                    label = f"{smin} - {smax}".strip()
                    intervals.append({'min': vmin, 'max': vmax, 'label': label})
            result['length_intervals'] = intervals

            # Selection mode (optional)
            mode = None
            if hasattr(scw, 'radioButton') and hasattr(scw, 'radioButton_2'):
                try:
                    if scw.radioButton.isChecked():
                        mode = 'simple_ais'
                    elif scw.radioButton_2.isChecked():
                        mode = 'manual'
                except Exception:
                    mode = None
            if mode is not None:
                result['selection_mode'] = mode
        except Exception:
            # If widget not available, leave defaults (empty) to avoid breaking save
            pass
        return result
    
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

    def normalize_depths_for_ui(self, depths: list) -> list[list[str]]:
        """Undo copy_depths_and_objects for depths: dicts -> [id, depth, polygon].

        Accepts either a list of dicts (with keys id, depth, polygon) or
        an already UI-ready list of rows, and returns list of [id, depth, polygon].
        """
        rows: list[list[str]] = []
        if not isinstance(depths, list):
            return rows
        for item in depths:
            # Assume it's already a sequence like [id, depth, polygon]
            try:
                rows.append([str(item[0]), str(item[1]), str(item[2])])
            except Exception:
                rows.append([str(item), '', ''])
        return rows

    def normalize_objects_for_ui(self, objects: list) -> list[list[str]]:
        """Undo copy_depths_and_objects for objects: dicts -> [id, height, polygon].

        Accepts either a list of dicts (with keys id, height/heights, polygon) or
        an already UI-ready list of rows, and returns list of [id, height, polygon].
        """
        rows: list[list[str]] = []
        if not isinstance(objects, list):
            return rows
        for item in objects:
            try:
                rows.append([str(item[0]), str(item[1]), str(item[2])])
            except Exception:
                rows.append([str(item), '', ''])
        return rows
    
    def populate(self, data):
        self.p.traffic_data = data['traffic_data'] 
        self.p.segment_data = data['segment_data']
        self.p.drift_values = data['drift']
        self.p.drift_settings.drift_values = data['drift']
        for key, item in data['segment_data'].items():
            self.p.segment_data[key]['dist1'] = np.array(item['dist1'])
            self.p.segment_data[key]['dist2'] = np.array(item['dist2'])
        self.populate_segment_tbl(data['segment_data'], self.p.main_widget.twRouteList)
        self.populate_cbTrafficSelectSeg()
        self.p.main_widget.leNormMean1_1.setText('')
        self.p.distributions.change_dist_segment(self.p.distributions.last_id)
        depth_rows = self.normalize_depths_for_ui(data['depths'])
        object_rows = self.normalize_objects_for_ui(data['objects'])
        self.populate_tbl(depth_rows, self.p.main_widget.twDepthList)
        self.populate_tbl(object_rows, self.p.main_widget.twObjectList)
        
        # Load data to canvas
        self.p.load_lines(data)
        # Ensure layers get tracked for later unload
        self.p.object.area_type = 'depth'
        for i, dep in enumerate(depth_rows):
            self.p.object.load_area('Depth - ' + dep[0], dep[2], row=i)
        self.p.object.area_type = 'object'
        for j, obj in enumerate(object_rows):
            self.p.object.load_area('Structure - ' + obj[0], obj[2], row=j)
            
    def populate_cbTrafficSelectSeg(self):
        """Sets the segment names in cbTrafficSelectSeg"""
        self.p.main_widget.cbTrafficSelectSeg.clear()
        for key in self.p.segment_data.keys():
            self.p.main_widget.cbTrafficSelectSeg.addItem(str(key))
        self.p.traffic.c_seg = self.p.main_widget.cbTrafficSelectSeg.currentText()

    def populate_tbl(self, data:list, tbl:QTableWidget):
        tbl.setRowCount(len(data))
        for i, line in enumerate(data):
            for j, value in enumerate(line):
                item = QTableWidgetItem(value)
                tbl.setItem(i, j, item)
    
    def populate_segment_tbl(self, data:dict [str, dict[str, Any]], tbl:QTableWidget):
        tbl.setRowCount(len(data))
        for j, col in enumerate(['Segment_Id', 'Route_Id', 'Leg_name', 'Start_Point', 'End_Point', 'Width']):
            for i, key in enumerate(data.keys()):
                item = QTableWidgetItem(str(data[key][col]))
                self.p.main_widget.twRouteList.setItem(i, j, item)
