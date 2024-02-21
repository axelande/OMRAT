from operator import xor
import json
from functools import partial

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QTableWidgetItem, QSpinBox

from ui.traffic_data_widget import TrafficDataWidget

class Traffic:
    def __init__(self, omrat, dw):
        self.dw = dw
        self.omrat = omrat
        self.traffic_data = self.omrat.traffic_data
        self.c_seg = None
        self.current_table = []
        self.variables = ['Frequency (ships/year)', 'Speed (knots)', 'Draught (meters)', 'Ship heights (meters)']
        self.last_var = 'Frequency (ships/year)'
        self.last_dist_seg = '1'
        
        self.tdw = TrafficDataWidget()
                
        self.set_table_headings()
        self.run_update = True
        
    def fill_cbTrafficSelectSeg(self):
        """Sets the segment names in cbTrafficSelectSeg"""
        self.tdw.cbTrafficSelectSeg.clear()
        nrs = self.dw.twRouteList.rowCount()
        for i in range(nrs):
            self.tdw.cbTrafficSelectSeg.addItem(self.dw.twRouteList.item(i, 0).text())
        self.c_seg = self.tdw.cbTrafficSelectSeg.currentText()

    def set_table_headings(self):
        """Sets the column and row names of the table"""
        types = []
        for i in range(self.dw.cvTypes.rowCount()):
            it = self.dw.cvTypes.item(i, 0)
            text = it.text() if it is not None else ""
            types.append(text)
        sizes = []
        for i in range(self.dw.twLengths.rowCount()):
            it1 = self.dw.twLengths.item(i, 0)
            text1 = it1.text() if it is not None else ""
            it2 = self.dw.twLengths.item(i, 1)
            text2 = it2.text() if it is not None else ""
            sizes.append(f'{text1} - {text2}')
        self.tdw.twTrafficData.setColumnCount(len(sizes))
        self.tdw.twTrafficData.setHorizontalHeaderLabels(sizes)
        self.tdw.twTrafficData.setRowCount(len(types))
        self.tdw.twTrafficData.setVerticalHeaderLabels(types)
        for row in range(len(types)):
            for col in range(len(sizes)):
                item = QSpinBox()
                item.setMaximum(100000)
                self.tdw.twTrafficData.setCellWidget(row, col, item)
    
    def change_dist_segment(self):
        """Change the segment information in the main widget upon user changing the
        segment in the combobox"""
        seg_id = self.dw.cbTrafficSelectSeg.currentText()
        if seg_id == '':
            return
        l_id = self.last_dist_seg
        if self.dw.leNormMean.text() != '':
            self.omrat.segment_data[l_id]['mean'] = float(self.dw.leNormMean.text())
            self.omrat.segment_data[l_id]['std'] = float(self.dw.leNormStd.text())
            self.omrat.segment_data[l_id]['u_min'] = float(self.dw.leUniformMin.text())
            self.omrat.segment_data[l_id]['u_max'] = float(self.dw.leUniformMax.text())
            self.omrat.segment_data[l_id]['u_p'] = self.dw.sbUniformP.value()
            self.omrat.segment_data[l_id]['mean_2'] = float(self.dw.leNormMean_2.text())
            self.omrat.segment_data[l_id]['std_2'] = float(self.dw.leNormStd_2.text())
            self.omrat.segment_data[l_id]['u_min_2'] = float(self.dw.leUniformMin_2.text())
            self.omrat.segment_data[l_id]['u_max_2'] = float(self.dw.leUniformMax_2.text())
            self.omrat.segment_data[l_id]['u_p_2'] = self.dw.sbUniformP_2.value()
        if 'mean' not in self.omrat.segment_data[seg_id]:
            # Will happen the first time that cbTrafficSelectSeg is changed
            self.omrat.segment_data[seg_id].update({'mean': 0, 'std': 0, 'u_min': 0, 
                                                    'u_max':0, 'u_p': 0, 'mean_2': 0, 
                                                    'std_2': 0, 'u_min_2': 0, 
                                                    'u_max_2':0, 'u_p_2': 0,})
        self.dw.leNormMean.setText(str(self.omrat.segment_data[seg_id]['mean']))
        self.dw.leNormStd.setText(str(self.omrat.segment_data[seg_id]['std']))
        self.dw.leUniformMin.setText(str(self.omrat.segment_data[seg_id]['u_min']))
        self.dw.leUniformMax.setText(str(self.omrat.segment_data[seg_id]['u_max']))
        self.dw.sbUniformP.setValue(int(self.omrat.segment_data[seg_id]['u_p']))
        self.dw.leNormMean_2.setText(str(self.omrat.segment_data[seg_id]['mean_2']))
        self.dw.leNormStd_2.setText(str(self.omrat.segment_data[seg_id]['std_2']))
        self.dw.leUniformMin_2.setText(str(self.omrat.segment_data[seg_id]['u_min_2']))
        self.dw.leUniformMax_2.setText(str(self.omrat.segment_data[seg_id]['u_max_2']))
        self.dw.sbUniformP_2.setValue(int(self.omrat.segment_data[seg_id]['u_p_2']))
        self.last_dist_seg = seg_id
        
    def create_empty_dict(self, s_key, dirs):
        """Creates an empty dict for the segment with all types"""
        self.traffic_data[s_key] = {}
        rows = self.tdw.twTrafficData.rowCount()
        cols = self.tdw.twTrafficData.columnCount()
        for di in dirs:
            self.traffic_data[s_key][di] = {}
            for val, key in zip ([0, 0, 0, 0], self.variables):
                self.traffic_data[s_key][di][key] = []
                for row in range(rows):
                    line = []
                    for col in range(cols):
                        line.append(val)
                    self.traffic_data[s_key][di][key].append(line)
            
    def change_type(self, caller):
        """Updates Traffic data table with the data from traffic_data, 
        using the correct type and segment"""
        if not self.run_update:
            return
        self.save()
        if caller == 'segment':
            self.c_seg = self.tdw.cbTrafficSelectSeg.currentText()
            self.update_direction_select()
        rows = self.tdw.twTrafficData.rowCount()
        cols = self.tdw.twTrafficData.columnCount()
        self.last_var = self.tdw.cbSelectType.currentText()
        self.c_di = self.tdw.cbTrafficDirectionSelect.currentText()
        for row in range(rows):
            for col in range(cols):
                val = self.traffic_data[self.c_seg][self.c_di][self.last_var][row][col]
                if val == '':
                    val = 0
                if val not in ['0', 0]:
                    print(val, row, col)
                    
                item = QSpinBox()
                item.setMaximum(100000)
                item.setValue(float(val))
                #item.setItemDelegate(self.deligate)
                self.tdw.twTrafficData.setCellWidget(row, col, item)
        
    def update_direction_select(self):
        self.run_update = False
        self.tdw.cbTrafficDirectionSelect.clear()
        for key in self.traffic_data[self.c_seg].keys():
            self.tdw.cbTrafficDirectionSelect.addItem(key)
        self.c_di = self.tdw.cbTrafficDirectionSelect.currentText()
        self.run_update = True
    
    def save(self):
        """Saves the previous "table" in traffic_data"""
        rows = self.tdw.twTrafficData.rowCount()
        cols = self.tdw.twTrafficData.columnCount()
        typ = self.last_var
        for row in range(rows):
            for col in range(cols):
                val = self.tdw.twTrafficData.cellWidget(row, col)
                if val.value() not in ['0', 0]:
                    print(val, row, col)
                self.traffic_data[self.c_seg][self.c_di][typ][row][col] = val.value()
                    
    def commit_changes(self):
        """Copy the output from the traffic data within this module to omrats traffic_data"""
        self.save()
        self.omrat.traffic_data = self.traffic_data

    def run(self):
        self.tdw.show()
        self.last_var = 'Frequency (ships/year)'
        self.tdw.cbSelectType.currentIndexChanged.connect(partial(self.change_type, 'type'))
        self.tdw.cbTrafficSelectSeg.currentIndexChanged.connect(partial(self.change_type, 'segment'))
        self.tdw.cbTrafficDirectionSelect.currentIndexChanged.connect(partial(self.change_type, 'dir'))
        self.tdw.accepted.connect(self.commit_changes)
        self.tdw.exec_()