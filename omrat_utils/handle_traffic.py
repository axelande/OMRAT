from dataclasses import dataclass, field
from operator import xor
import json
from functools import partial
from typing import Union

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QTableWidgetItem, QSpinBox, QDoubleSpinBox
from scipy import stats

from ui.traffic_data_widget import TrafficDataWidget
from geometries import isint


@dataclass
class Normal:
    mean =  None
    std = 0
    probability = 0

@dataclass
class Uniform:
    lower = 0
    upper= 0
    probability = 100

@dataclass
class Params:
    normal1: Normal = field(default_factory=Normal)
    normal2: Normal = field(default_factory=Normal)
    normal3: Normal = field(default_factory=Normal)
    uniform: Uniform = field(default_factory=Uniform)

    def __iter__(self):
        # Combine regular cards and the super card for iteration
        return iter([self.normal1, self.normal2, self.normal3, self.uniform])
class Traffic:
    def __init__(self, omrat, dw):
        self.dw = dw
        self.omrat = omrat
        self.W = omrat.dockwidget
        self.traffic_data = self.omrat.traffic_data
        self.c_seg = None
        self.current_table = []
        self.variables = ['Frequency (ships/year)', 'Speed (knots)', 'Draught (meters)', 'Ship heights (meters)', 'Ship Beam (meters)']
        self.last_var = 'Frequency (ships/year)'
        self.last_dist_seg = '1'
        self.canvas = None
        self.dont_save = False
        
        self.tdw = TrafficDataWidget()
                
        self.set_table_headings()
        self.run_update = True
        self.main_connect()
        
    def main_connect(self):
        """Connects the values in the main widget"""
        connect_weights = [
            self.W.leNormWeight1_1, self.W.leNormWeight1_2, self.W.leNormWeight1_3,
            self.W.leNormWeight2_1, self.W.leNormWeight2_2, self.W.leNormWeight2_3
        ]
        for widget in connect_weights:
            widget.textChanged.connect(lambda _, w=widget: self.adjust_weights(w))
        self.W.sbUniformP1.valueChanged.connect(lambda _, w=self.W.sbUniformP1: self.adjust_weights(w))
        self.W.sbUniformP2.valueChanged.connect(lambda _, w=self.W.sbUniformP2: self.adjust_weights(w))

        # Connect other widgets to run_update_plot
        widgets_to_update = [
            self.W.leNormMean1_1, self.W.leNormMean1_2, self.W.leNormMean1_3,
            self.W.leNormStd1_1, self.W.leNormStd1_2, self.W.leNormStd1_3,
            self.W.leUniformMin1, self.W.leUniformMax1,
            self.W.leNormWeight2_1, self.W.leNormMean2_1, self.W.leNormMean2_2, self.W.leNormMean2_3,
            self.W.leNormStd2_1, self.W.leNormStd2_2, self.W.leNormStd2_3,
            self.W.leUniformMin2, self.W.leUniformMax2
        ]

        for widget in widgets_to_update:
            if hasattr(widget, 'textChanged'):
                widget.textChanged.connect(self.run_update_plot)
            elif hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self.run_update_plot)
        
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
        for i in range(self.omrat.ship_cat.scw.cvTypes.rowCount()):
            it = self.omrat.ship_cat.scw.cvTypes.item(i, 0)
            text = it.text() if it is not None else ""
            types.append(text)
        sizes = []
        for i in range(self.omrat.ship_cat.scw.twLengths.rowCount()):
            it1 = self.omrat.ship_cat.scw.twLengths.item(i, 0)
            text1 = it1.text() if it is not None else ""
            it2 = self.omrat.ship_cat.scw.twLengths.item(i, 1)
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
        if self.dw.leNormMean1_1.text() != '' or self.dont_save:
            # Update segment data for the last segment
            self.omrat.segment_data[l_id]['mean1_1'] = float(self.dw.leNormMean1_1.text())
            self.omrat.segment_data[l_id]['mean1_2'] = float(self.dw.leNormMean1_2.text())
            self.omrat.segment_data[l_id]['mean1_3'] = float(self.dw.leNormMean1_3.text())
            self.omrat.segment_data[l_id]['std1_1'] = float(self.dw.leNormStd1_1.text())
            self.omrat.segment_data[l_id]['std1_2'] = float(self.dw.leNormStd1_2.text())
            self.omrat.segment_data[l_id]['std1_3'] = float(self.dw.leNormStd1_3.text())
            self.omrat.segment_data[l_id]['mean2_1'] = float(self.dw.leNormMean2_1.text())
            self.omrat.segment_data[l_id]['mean2_2'] = float(self.dw.leNormMean2_2.text())
            self.omrat.segment_data[l_id]['mean2_3'] = float(self.dw.leNormMean2_3.text())
            self.omrat.segment_data[l_id]['std2_1'] = float(self.dw.leNormStd2_1.text())
            self.omrat.segment_data[l_id]['std2_2'] = float(self.dw.leNormStd2_2.text())
            self.omrat.segment_data[l_id]['std2_3'] = float(self.dw.leNormStd2_3.text())
            self.omrat.segment_data[l_id]['weight1_1'] = float(self.dw.leNormWeight1_1.text())
            self.omrat.segment_data[l_id]['weight1_2'] = float(self.dw.leNormWeight1_2.text())
            self.omrat.segment_data[l_id]['weight1_3'] = float(self.dw.leNormWeight1_3.text())
            self.omrat.segment_data[l_id]['weight2_1'] = float(self.dw.leNormWeight2_1.text())
            self.omrat.segment_data[l_id]['weight2_2'] = float(self.dw.leNormWeight2_2.text())
            self.omrat.segment_data[l_id]['weight2_3'] = float(self.dw.leNormWeight2_3.text())
            self.omrat.segment_data[l_id]['u_min1'] = float(self.dw.leUniformMin1.text())
            self.omrat.segment_data[l_id]['u_max1'] = float(self.dw.leUniformMax1.text())
            self.omrat.segment_data[l_id]['u_p1'] = self.dw.sbUniformP1.value()
            self.omrat.segment_data[l_id]['ai1'] = float(self.dw.LEMeanTimeSeconds1.text())
            self.omrat.segment_data[l_id]['u_min2'] = float(self.dw.leUniformMin2.text())
            self.omrat.segment_data[l_id]['u_max2'] = float(self.dw.leUniformMax2.text())
            self.omrat.segment_data[l_id]['u_p2'] = self.dw.sbUniformP2.value()
            self.omrat.segment_data[l_id]['ai2'] = float(self.dw.LEMeanTimeSeconds2.text())
        if 'mean1_1' not in self.omrat.segment_data[seg_id]:
            self.omrat.segment_data[seg_id].update({
                'mean1_1': 0,'mean2_1': 0, 'weight1_1': 100,
                'std1_1': 0,'std2_1': 0, 'weight2_1': 100})
        if 'mean1_2' not in self.omrat.segment_data[seg_id]:
            # Initialize default values for the new segment
            self.omrat.segment_data[seg_id].update({
                'mean1_2': 0, 'mean1_3': 0,
                'std1_2': 0, 'std1_3': 0,
                'mean2_2': 0, 'mean2_3': 0,
                'std2_2': 0, 'std2_3': 0,
                'weight1_2': 0, 'weight1_3': 0,
                'weight2_2': 0, 'weight2_3': 0,
                'u_min1': 0, 'u_max1': 0, 'u_p1': 0, 'ai1': 180,
                'u_min2': 0, 'u_max2': 0, 'u_p2': 0, 'ai2': 180
            })
        # Update the UI fields with the new segment's data
        if not self.dont_save:
            self.dw.leNormMean1_1.setText(str(self.omrat.segment_data[seg_id]['mean1_1']))
            self.dw.leNormMean1_2.setText(str(self.omrat.segment_data[seg_id]['mean1_2']))
            self.dw.leNormMean1_3.setText(str(self.omrat.segment_data[seg_id]['mean1_3']))
            self.dw.leNormStd1_1.setText(str(self.omrat.segment_data[seg_id]['std1_1']))
            self.dw.leNormStd1_2.setText(str(self.omrat.segment_data[seg_id]['std1_2']))
            self.dw.leNormStd1_3.setText(str(self.omrat.segment_data[seg_id]['std1_3']))
            self.dw.leNormMean2_1.setText(str(self.omrat.segment_data[seg_id]['mean2_1']))
            self.dw.leNormMean2_2.setText(str(self.omrat.segment_data[seg_id]['mean2_2']))
            self.dw.leNormMean2_3.setText(str(self.omrat.segment_data[seg_id]['mean2_3']))
            self.dw.leNormStd2_1.setText(str(self.omrat.segment_data[seg_id]['std2_1']))
            self.dw.leNormStd2_2.setText(str(self.omrat.segment_data[seg_id]['std2_2']))
            self.dw.leNormStd2_3.setText(str(self.omrat.segment_data[seg_id]['std2_3']))
            self.dw.leNormWeight1_1.setText(str(self.omrat.segment_data[seg_id]['weight1_1']))
            self.dw.leNormWeight1_2.setText(str(self.omrat.segment_data[seg_id]['weight1_2']))
            self.dw.leNormWeight1_3.setText(str(self.omrat.segment_data[seg_id]['weight1_3']))
            self.dw.leNormWeight2_1.setText(str(self.omrat.segment_data[seg_id]['weight2_1']))
            self.dw.leNormWeight2_2.setText(str(self.omrat.segment_data[seg_id]['weight2_2']))
            self.dw.leNormWeight2_3.setText(str(self.omrat.segment_data[seg_id]['weight2_3']))
            self.dw.leUniformMin1.setText(str(self.omrat.segment_data[seg_id]['u_min1']))
            self.dw.leUniformMax1.setText(str(self.omrat.segment_data[seg_id]['u_max1']))
            self.dw.sbUniformP1.setValue(int(self.omrat.segment_data[seg_id]['u_p1']))
            self.dw.LEMeanTimeSeconds1.setText(str(self.omrat.segment_data[seg_id]['ai1']))
            self.dw.leUniformMin2.setText(str(self.omrat.segment_data[seg_id]['u_min2']))
            self.dw.leUniformMax2.setText(str(self.omrat.segment_data[seg_id]['u_max2']))
            self.dw.sbUniformP2.setValue(int(self.omrat.segment_data[seg_id]['u_p2']))
            self.dw.LEMeanTimeSeconds2.setText(str(self.omrat.segment_data[seg_id]['ai2']))
            self.last_dist_seg = seg_id
            self.run_update_plot()
        self.dont_save= False
        
    def create_empty_dict(self, s_key, dirs):
        """Creates an empty dict for the segment with all types"""
        self.traffic_data[s_key] = {}
        rows = self.tdw.twTrafficData.rowCount()
        cols = self.tdw.twTrafficData.columnCount()
        for di in dirs:
            self.traffic_data[s_key][di] = {}
            for idx, key in enumerate(self.variables):
                self.traffic_data[s_key][di][key] = []
                for row in range(rows):
                    line = []
                    for col in range(cols):
                        if idx == 0:
                            line.append(0)
                        else:
                            line.append([])
                    self.traffic_data[s_key][di][key].append(line)
    
    def run_update_plot(self):
        leg_name = self.W.cbTrafficSelectSeg.currentText()
        if leg_name not in self.omrat.ais.dist_data:
            return
        d = self.omrat.ais.dist_data[leg_name]
        p1, p2 = self.get_leg_params()
        self.plot_data(d['line1'], d['line2'], p1, p2)
        
    @staticmethod
    def _assign(widget) -> float:
        if widget.text() == '-':
            return -0
        if widget.text() != "":
            return float(widget.text().replace(',', '.').replace('--',''))
        else:
            return None
            
    def get_leg_params(self) -> list:
        """Retrieve the values from the widget and pass them further as Params objs"""
        p1 = Params()
        p2 = Params()
        for di, p in zip(["1", "2"], [p1, p2]):
            for i in range(1, 4):
                item = getattr(p, f'normal{i}')
                item.mean = self._assign(getattr(self.W,f'leNormMean{di}_{i}'))
                item.std = self._assign(getattr(self.W,f'leNormStd{di}_{i}'))
                item.probability = self._assign(getattr(self.W, f'leNormWeight{di}_{i}'))
            p.uniform.lower = self._assign(getattr(self.W, f'leUniformMin{di}'))
            p.uniform.upper = self._assign(getattr(self.W, f'leUniformMax{di}'))
            p.uniform.probability = self._assign(getattr(self.W, f'sbUniformP{di}'))
        return [p1, p2]
    
    def plot_data(self, data, data2, parameters1: Params, parameters2: Params):
        """Makes the plot in the top left corner"""
        # Create the figure and axes
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.hist(data, bins=50, density=True, alpha=0.6, color='b', label=self.omrat.dockwidget.laDir1.text())
        ax.hist(data2, bins=50, density=True, alpha=0.6, color='g', label=self.omrat.dockwidget.laDir2.text())
        self.add_dist2plot(ax, parameters1, data, True)
        self.add_dist2plot(ax, parameters2, data2, False)

        # Add legend with fitted parameters
        plt.legend()
        ax.set_xlabel('Values')
        ax.set_ylabel('Density')

        # Remove the previous canvas if it exists
        if self.canvas is not None:
            self.W.DistributionWidget.removeWidget(self.canvas)
            self.canvas.deleteLater()  # Ensure the old canvas is deleted
            self.canvas = None

        # Add the new canvas
        fig.tight_layout()
        self.canvas = FigureCanvas(fig)
        self.W.DistributionWidget.addWidget(self.canvas)
        self.canvas.draw()
        plt.close(fig)
    
    def add_dist2plot(self, ax, parameters:Params, data:np.array, first:bool):
        """Adds the distributions to the plot"""
        try:
            if parameters.normal1.mean is None:
                parameters.normal1.mean, parameters.normal1.std = stats.norm.fit(data)
                parameters.normal1.mean = round(parameters.normal1.mean)
                parameters.normal1.std = round(parameters.normal1.std)
                if first:
                    self.W.leNormMean1_1.setText(str(parameters.normal1.mean))
                    self.W.leNormStd1_1.setText(str(parameters.normal1.std))
                    self.W.leNormWeight1_1.setText("1")
                else:
                    self.W.leNormMean2_1.setText(str(parameters.normal1.mean))
                    self.W.leNormStd2_1.setText(str(parameters.normal1.std))
                    self.W.leNormWeight2_1.setText("1")
                
                parameters.normal1.probability = 1
                parameters.uniform.lower = data.min()
                parameters.uniform.upper = data.max()
            x = np.linspace(data.min(), data.max(), 1000)
            tot_y = np.zeros(x.size)
            tot_p = 0
            
            for i, (dist, c) in enumerate(zip(parameters, ['r', 'm', 'b', 'y'])):
                if dist.probability > 0:
                    if isinstance(dist, Normal):
                        y = stats.norm.pdf(x, dist.mean, dist.std)
                        if first:
                            ax.plot(x, y, c, label=f'Normal_{i}')
                        else:
                            ax.plot(x, y, f'--{c}', label=f'Normal2_{i}')
                    elif isinstance(dist, Uniform):
                        y = stats.uniform.pdf(x, dist.lower, dist.upper-dist.lower)
                        if first:
                            ax.plot(x, y, 'y', label='Uniform')
                        else:
                            ax.plot(x, y, '--y', label='Uniform2')
                    tot_y += y * dist.probability
                    tot_p += dist.probability
            if first:
                ax.plot(x, tot_y / tot_p, 'k', label='Resulting P')
            else:
                ax.plot(x, tot_y / tot_p, '--k', label='Resulting P2')
        except Exception as e:
            print(e)
            
    def update_traffic_tbl(self, caller):
        """Updates Traffic data table with the data from traffic_data, 
        using the correct type and segment"""
        if self.run_update:
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
                self.traffic_data[self.c_seg][self.c_di][typ][row][col] = val.value()
                    
    def commit_changes(self):
        """Copy the output from the traffic data within this module to omrats traffic_data"""
        self.save()
        self.omrat.traffic_data = self.traffic_data

    def run(self):
        self.tdw.show()
        self.run_update = False
        self.last_var = 'Frequency (ships/year)'
        self.tdw.cbSelectType.currentIndexChanged.connect(partial(self.update_traffic_tbl, 'type'))
        self.tdw.cbTrafficSelectSeg.currentIndexChanged.connect(partial(self.update_traffic_tbl, 'segment'))
        self.tdw.cbTrafficDirectionSelect.currentIndexChanged.connect(partial(self.update_traffic_tbl, 'dir'))
        self.update_traffic_tbl('start')
        self.tdw.accepted.connect(self.commit_changes)
        self.run_update = True
        self.tdw.exec_()

    def adjust_weights(self, changed_widget):
        """Adjust the weights so that their sum equals 100 while maintaining order."""
        widgets1 = [
            self.W.leNormWeight1_1,
            self.W.leNormWeight1_2,
            self.W.leNormWeight1_3,
            self.W.sbUniformP1
        ]
        widgets2 = [
            self.W.leNormWeight2_1,
            self.W.leNormWeight2_2,
            self.W.leNormWeight2_3,
            self.W.sbUniformP2
        ]
        if changed_widget in widgets1:
            widgets = widgets1
        elif changed_widget in widgets2:
            widgets = widgets2
        else:
            print(widgets1, widgets2)
            print(changed_widget)
            print("Widget not found")

        # Get the total weight and the changed value
        total_weight = 100
        changed_value = float(changed_widget.text()) if hasattr(changed_widget, 'text') else changed_widget.value()

        # Calculate the remaining weight
        remaining_weight = total_weight - changed_value

        # Distribute the remaining weight proportionally among the other widgets
        other_widgets = [w for w in widgets if w != changed_widget]
        other_values = [
            float(w.text()) if hasattr(w, 'text') and w.text() != '' else w.value() if hasattr(w, 'value') else 0
            for w in other_widgets
        ]
        other_total = sum(other_values)

        if other_total == 0:
            # If all other values are zero, distribute equally
            for w in other_widgets:
                if hasattr(w, 'setText'):
                    w.setText(str(remaining_weight / len(other_widgets)))
                else:
                    w.setValue(int(remaining_weight / len(other_widgets)))
        else:
            # Adjust the other values proportionally
            for w, value in zip(other_widgets, other_values):
                adjusted_value = (value / other_total) * remaining_weight
                if hasattr(w, 'setText'):
                    w.setText(str(round(adjusted_value, 2)))
                else:
                    w.setValue(int(round(adjusted_value)))

        # Ensure the total sum is exactly 100
        self.ensure_total_sum(widgets)

    def ensure_total_sum(self, widgets):
        """Ensure the total sum of weights equals 100."""
        total = sum(
            float(w.text()) if hasattr(w, 'text') and w.text() != '' else w.value() if hasattr(w, 'value') else 0
            for w in widgets
        )
        difference = 100 - total

        # Adjust the last widget to make the total exactly 100
        last_widget = widgets[-1]
        if hasattr(last_widget, 'setText'):
            last_value = float(last_widget.text()) if last_widget.text() != '' else 0
            last_widget.setText(str(last_value + difference))
        else:
            last_widget.setValue(last_widget.value() + int(difference))

    def unload(self):
        """Cleanup resources and disconnect signals."""
        # Disconnect signals connected to self.W
        connect_weights = [
            self.W.leNormWeight1_1, self.W.leNormWeight1_2, self.W.leNormWeight1_3,
            self.W.leNormWeight2_1, self.W.leNormWeight2_2, self.W.leNormWeight2_3
        ]
        for widget in connect_weights:
            try:
                widget.textChanged.disconnect()
            except TypeError:
                pass

        try:
            self.W.sbUniformP1.valueChanged.disconnect()
            self.W.sbUniformP2.valueChanged.disconnect()
        except TypeError:
            pass

        # Disconnect other widgets connected to run_update_plot
        widgets_to_update = [
            self.W.leNormMean1_1, self.W.leNormMean1_2, self.W.leNormMean1_3,
            self.W.leNormStd1_1, self.W.leNormStd1_2, self.W.leNormStd1_3,
            self.W.leUniformMin1, self.W.leUniformMax1,
            self.W.leNormWeight2_1, self.W.leNormMean2_1, self.W.leNormMean2_2, self.W.leNormMean2_3,
            self.W.leNormStd2_1, self.W.leNormStd2_2, self.W.leNormStd2_3,
            self.W.leUniformMin2, self.W.leUniformMax2
        ]
        for widget in widgets_to_update:
            try:
                if hasattr(widget, 'textChanged'):
                    widget.textChanged.disconnect()
                elif hasattr(widget, 'valueChanged'):
                    widget.valueChanged.disconnect()
            except TypeError:
                pass

        # Disconnect signals from TrafficDataWidget (tdw)
        try:
            if hasattr(self, "tdw"):
                if self.tdw is not None:
                    self.tdw.cbSelectType.currentIndexChanged.disconnect()
                    self.tdw.cbTrafficDirectionSelect.currentIndexChanged.disconnect()
                    self.tdw.accepted.disconnect()
        except TypeError:
            pass

        # Remove all widgets from DistributionWidget
        if self.W.DistributionWidget is not None:
            while self.W.DistributionWidget.count() > 0:
                item = self.W.DistributionWidget.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        # Clear traffic data
        self.traffic_data.clear()

        # Remove reference to TrafficDataWidget
        self.tdw = None
        self.omrat = None
        self.W = None

        print("Traffic resources cleaned up.")
