from dataclasses import dataclass, field
from operator import xor
import json
from functools import partial
from typing import Optional, Any, TYPE_CHECKING, cast, Union
if TYPE_CHECKING:
    from omrat import OMRAT, OMRATMainWidget

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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

@dataclass
class Normal:
    mean: Optional[float] = None
    std: float = 0.0
    probability: float = 0.0

@dataclass
class Uniform:
    lower: float = 0.0
    upper: float = 0.0
    probability: float = 100.0


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
    def __init__(self, omrat: "OMRAT", dw:"OMRATMainWidget"):
        self.dw = dw
        self.omrat = omrat
        self.traffic_data: dict[str, dict[str, dict[str, Any]]] = self.omrat.traffic_data
        self.c_seg = "1"
        self.current_table = []
        self.variables = ['Frequency (ships/year)', 'Speed (knots)', 'Draught (meters)', 'Ship heights (meters)', 'Ship Beam (meters)']
        self.last_var = 'Frequency (ships/year)'
        self.last_dist_seg = '1'
        self.canvas: FigureCanvas | None = None
        self.dont_save = False
        
        self.tdw = TrafficDataWidget()
                
        self.set_table_headings()
        self.run_update = True
        self.main_connect()
        
    def main_connect(self):
        """Connects the values in the main widget"""
        connect_weights = [
            self.dw.leNormWeight1_1, self.dw.leNormWeight1_2, self.dw.leNormWeight1_3,
            self.dw.leNormWeight2_1, self.dw.leNormWeight2_2, self.dw.leNormWeight2_3
        ]
        for widget in connect_weights:
            widget.textChanged.connect(lambda _, w=widget: self.adjust_weights(w))
        self.dw.sbUniformP1.valueChanged.connect(lambda _, w=self.dw.sbUniformP1: self.adjust_weights(w))
        self.dw.sbUniformP2.valueChanged.connect(lambda _, w=self.dw.sbUniformP2: self.adjust_weights(w))

        # Connect other widgets to run_update_plot
        widgets_to_update:list[Any] = [
            self.dw.leNormMean1_1, self.dw.leNormMean1_2, self.dw.leNormMean1_3,
            self.dw.leNormStd1_1, self.dw.leNormStd1_2, self.dw.leNormStd1_3,
            self.dw.leUniformMin1, self.dw.leUniformMax1,
            self.dw.leNormWeight2_1, self.dw.leNormMean2_1, self.dw.leNormMean2_2, self.dw.leNormMean2_3,
            self.dw.leNormStd2_1, self.dw.leNormStd2_2, self.dw.leNormStd2_3,
            self.dw.leUniformMin2, self.dw.leUniformMax2
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
        
    def create_empty_dict(self, s_key:str, dirs:list[str]):
        """Creates an empty dict for the segment with all types"""
        self.traffic_data[s_key] = {}
        rows = self.tdw.twTrafficData.rowCount()
        cols = self.tdw.twTrafficData.columnCount()
        
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
    
    def run_update_plot(self):
        leg_name = self.dw.cbTrafficSelectSeg.currentText()
        if leg_name not in self.omrat.ais.dist_data:
            return
        d = self.omrat.ais.dist_data[leg_name]
        p1, p2 = self.get_leg_params()
        self.plot_data(d['line1'], d['line2'], p1, p2)
        
    def _assign(self, widget: WidgetType) -> float:
        if isinstance(widget, QLineEdit):
            return float(widget.text())
        elif isinstance(widget, QSpinBox):
            return float(widget.value())
        raise TypeError(f"Unsupported widget type: {type(widget)}")

            
    def get_leg_params(self) -> tuple[Params, Params]:
        """Retrieve the values from the widget and pass them further as Params objs"""
        p1 = Params()
        p2 = Params()
        for di, p in zip(["1", "2"], [p1, p2]):
            for i in range(1, 4):
                item = getattr(p, f'normal{i}')
                item.mean = self._assign(getattr(self.dw,f'leNormMean{di}_{i}'))
                item.std = self._assign(getattr(self.dw,f'leNormStd{di}_{i}'))
                item.probability = self._assign(getattr(self.dw, f'leNormWeight{di}_{i}'))
            p.uniform.lower = self._assign(getattr(self.dw, f'leUniformMin{di}'))
            p.uniform.upper = self._assign(getattr(self.dw, f'leUniformMax{di}'))
            p.uniform.probability = self._assign(getattr(self.dw, f'sbUniformP{di}'))
        return p1, p2
    
    def plot_data(self, data:np.ndarray, data2:np.ndarray, parameters1: Params, parameters2: Params):
        """Makes the plot in the top left corner"""
        # Create the figure and axes
        fig: Figure = plt.figure(figsize=(10, 6)) # type: ignore
        gs = gridspec.GridSpec(1, 1)
        ax: Axes = fig.add_subplot(gs[0, 0]) # type: ignore
        ax.hist(data, bins=50, density=True, alpha=0.6, color='b', label=self.omrat.main_widget.laDir1.text())
        ax.hist(data2, bins=50, density=True, alpha=0.6, color='g', label=self.omrat.main_widget.laDir2.text())
        self.add_dist2plot(ax, parameters1, data, True)
        self.add_dist2plot(ax, parameters2, data2, False)

        # Add legend with fitted parameters
        plt.legend()
        ax.set_xlabel('Values')
        ax.set_ylabel('Density')

        # Remove the previous canvas if it exists
        if self.canvas is not None:
            self.dw.DistributionWidget.removeWidget(self.canvas)
            self.canvas.deleteLater()  # Ensure the old canvas is deleted

        # Add the new canvas
        fig.tight_layout()
        self.canvas = FigureCanvas(fig)
        self.dw.DistributionWidget.addWidget(self.canvas)
        self.canvas.draw()
        plt.close(fig)
    
    def add_dist2plot(self, ax:Axes, parameters:Params, data:np.ndarray, first:bool):
        """Adds the distributions to the plot"""
        try:
            if parameters.normal1.mean is None:
                
                fit_result: tuple[float, float] = stats.norm.fit(data) # type: ignore
                mean, std = fit_result
                parameters.normal1.mean = round(mean)
                parameters.normal1.std = round(std)
                if first:
                    self.dw.leNormMean1_1.setText(str(parameters.normal1.mean))
                    self.dw.leNormStd1_1.setText(str(parameters.normal1.std))
                    self.dw.leNormWeight1_1.setText("1")
                else:
                    self.dw.leNormMean2_1.setText(str(parameters.normal1.mean))
                    self.dw.leNormStd2_1.setText(str(parameters.normal1.std))
                    self.dw.leNormWeight2_1.setText("1")
                
                parameters.normal1.probability = 1
                parameters.uniform.lower = data.min()
                parameters.uniform.upper = data.max()
            x: np.ndarray = np.linspace(data.min(), data.max(), 1000, dtype=float)
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
            
    def update_traffic_tbl(self, caller:str):
        """Updates Traffic data table with the data from traffic_data, 
        using the correct type and segment"""
        if self.run_update:
            self.save()
        if caller == 'segment':
            self.c_seg = self.tdw.cbTrafficSelectSeg.currentText()
            self.update_direction_select()
        assert self.tdw.twTrafficData is not None
        rows = self.tdw.twTrafficData.rowCount()
        cols = self.tdw.twTrafficData.columnCount()
        self.last_var: str = self.tdw.cbSelectType.currentText()
        self.c_di = self.tdw.cbTrafficDirectionSelect.currentText()
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

    def adjust_weights(self, changed_widget: QSpinBox | QLineEdit) -> None:
        """Adjust the weights so that their sum equals 100 while maintaining order."""
        widgets1: list[Any] = [
            self.dw.leNormWeight1_1,
            self.dw.leNormWeight1_2,
            self.dw.leNormWeight1_3,
            self.dw.sbUniformP1
        ]
        widgets2: list[Any] = [
            self.dw.leNormWeight2_1,
            self.dw.leNormWeight2_2,
            self.dw.leNormWeight2_3,
            self.dw.sbUniformP2
        ]
        if changed_widget in widgets1:
            widgets = widgets1
        elif changed_widget in widgets2:
            widgets = widgets2
        else:
            print("Widget not found")
            return

        # Get the total weight and the changed value
        total_weight = 100
        if hasattr(changed_widget, 'text'):
            changed_value = float(changed_widget.text())
        else:
            widget_sb: QSpinBox = cast(QSpinBox, changed_widget)
            changed_value = float(widget_sb.value())

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

    def ensure_total_sum(self, widgets:list[Any]):
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
        # Disconnect signals connected to self.dw
        connect_weights = [
            self.dw.leNormWeight1_1, self.dw.leNormWeight1_2, self.dw.leNormWeight1_3,
            self.dw.leNormWeight2_1, self.dw.leNormWeight2_2, self.dw.leNormWeight2_3
        ]
        for widget in connect_weights:
            try:
                widget.textChanged.disconnect()
            except TypeError:
                pass

        try:
            self.dw.sbUniformP1.valueChanged.disconnect()
            self.dw.sbUniformP2.valueChanged.disconnect()
        except TypeError:
            pass

        # Disconnect other widgets connected to run_update_plot
        widgets_to_update = [
            self.dw.leNormMean1_1, self.dw.leNormMean1_2, self.dw.leNormMean1_3,
            self.dw.leNormStd1_1, self.dw.leNormStd1_2, self.dw.leNormStd1_3,
            self.dw.leUniformMin1, self.dw.leUniformMax1,
            self.dw.leNormWeight2_1, self.dw.leNormMean2_1, self.dw.leNormMean2_2, self.dw.leNormMean2_3,
            self.dw.leNormStd2_1, self.dw.leNormStd2_2, self.dw.leNormStd2_3,
            self.dw.leUniformMin2, self.dw.leUniformMax2
        ]
        for widget in widgets_to_update:
            try:
                if hasattr(widget, 'textChanged'):
                    widget.textChanged.disconnect()
                elif hasattr(widget, 'valueChanged'):
                    widget.valueChanged.disconnect() # type: ignore
            except TypeError:
                pass

        # Disconnect signals from TrafficDataWidget (tdw)
        try:
            if hasattr(self, "tdw"):
                self.tdw.cbSelectType.currentIndexChanged.disconnect()
                self.tdw.cbTrafficDirectionSelect.currentIndexChanged.disconnect()
                self.tdw.accepted.disconnect()
        except TypeError:
            pass

        # Remove all widgets from DistributionWidget
        if self.dw.DistributionWidget is not None:
            while self.dw.DistributionWidget.count() > 0:
                item = self.dw.DistributionWidget.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        # Clear traffic data
        self.traffic_data.clear()

        # Remove reference to TrafficDataWidget
        self.tdw.deleteLater()
        print("Traffic resources cleaned up.")
