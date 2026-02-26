from dataclasses import dataclass, field
from typing import Optional, Any, TYPE_CHECKING, cast, Union
import numpy as np
from scipy import stats
from matplotlib.axes import Axes
from qgis.PyQt.QtWidgets import QLineEdit, QSpinBox, QDoubleSpinBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

if TYPE_CHECKING:
    from omrat import OMRAT, OMRATMainWidget
    
WidgetType = QLineEdit | QSpinBox | QDoubleSpinBox

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
        return iter([self.normal1, self.normal2, self.normal3, self.uniform])

        
class Distributions:
    def __init__(self, parent:"OMRAT"):
        self.omrat = parent
        self.dw = parent.main_widget
        self.canvas: FigureCanvas | None = None
        self.last_id: str = '1'
        self.new_id: str = '1'
        self.main_connect()


    def main_connect(self):
        """Connects the values in the main widget"""
        connect_weights = [
            self.dw.leNormWeight1_1, self.dw.leNormWeight1_2, self.dw.leNormWeight1_3,
            self.dw.leNormWeight2_1, self.dw.leNormWeight2_2, self.dw.leNormWeight2_3
        ]
        for widget in connect_weights:
            if hasattr(widget, 'editingFinished'):
                widget.editingFinished.connect(lambda w=widget: self.adjust_weights(w))
            elif hasattr(widget, 'leaveEvent'):
                widget.leaveEvent.connect(lambda w=widget: self.adjust_weights(w))
            
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
            if hasattr(widget, 'editingFinished'):
                widget.editingFinished.connect(lambda self=self: self.run_update_plot(self.last_id))
            elif hasattr(widget, 'leaveEvent'):
                widget.leaveEvent.connect(lambda self=self: self.run_update_plot(self.last_id))

    def _assign(self, widget: WidgetType) -> float:
        if isinstance(widget, QLineEdit):
            return float(widget.text())
        elif isinstance(widget, QSpinBox):
            return float(widget.value())
        elif isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        raise TypeError(f"Unsupported widget type: {type(widget)}")

    def get_leg_params(self) -> tuple[Params, Params]:
        """Retrieve the values from the widget and pass them further as Params objs"""
        p1 = Params()
        p2 = Params()
        for di, p in zip(["1", "2"], [p1, p2]):
            for i in range(1, 4):
                item = getattr(p, f'normal{i}')
                item.mean = self._assign(getattr(self.dw, f'leNormMean{di}_{i}'))
                item.std = self._assign(getattr(self.dw, f'leNormStd{di}_{i}'))
                item.probability = self._assign(getattr(self.dw, f'leNormWeight{di}_{i}'))
            p.uniform.lower = self._assign(getattr(self.dw, f'leUniformMin{di}'))
            p.uniform.upper = self._assign(getattr(self.dw, f'leUniformMax{di}'))
            p.uniform.probability = self._assign(getattr(self.dw, f'sbUniformP{di}'))
        return p1, p2
    
    def change_dist_segment(self, new_id:str | None):
        """Change the segment information in the main widget upon user changing the
        segment in the combobox"""
        if new_id == '' or new_id is None:
            return
        l_id = self.last_id
        if self.dw.leNormMean1_1.text() != '':
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
        if 'mean1_1' not in self.omrat.segment_data[new_id]:
            self.omrat.segment_data[new_id].update({
                'mean1_1': 0,'mean2_1': 0, 'weight1_1': 100,
                'std1_1': 0,'std2_1': 0, 'weight2_1': 100, 
                'mean1_2': 0, 'mean1_3': 0,
                'std1_2': 0, 'std1_3': 0,
                'mean2_2': 0, 'mean2_3': 0,
                'std2_2': 0, 'std2_3': 0,
                'weight1_2': 0, 'weight1_3': 0,
                'weight2_2': 0, 'weight2_3': 0,
                'u_min1': 0, 'u_max1': 0, 'u_p1': 0, 'ai1': 180,
                'u_min2': 0, 'u_max2': 0, 'u_p2': 0, 'ai2': 180})
        if 'mean1_2' not in self.omrat.segment_data[new_id]:
            # Initialize default values for the new segment
            self.omrat.segment_data[new_id].update({
                'mean1_2': 0, 'mean1_3': 0,
                'std1_2': 0, 'std1_3': 0,
                'mean2_2': 0, 'mean2_3': 0,
                'std2_2': 0, 'std2_3': 0,
                'weight1_2': 0, 'weight1_3': 0,
                'weight2_2': 0, 'weight2_3': 0,
                'u_min1': 0, 'u_max1': 0, 'u_p1': 0, 'ai1': 180,
                'u_min2': 0, 'u_max2': 0, 'u_p2': 0, 'ai2': 180
            })
        print(new_id, self.omrat.segment_data[new_id])
        # Update the UI fields with the new segment's data
        self.dw.leNormMean1_1.setText(str(self.omrat.segment_data[new_id]['mean1_1']))
        self.dw.leNormMean1_2.setText(str(self.omrat.segment_data[new_id]['mean1_2']))
        self.dw.leNormMean1_3.setText(str(self.omrat.segment_data[new_id]['mean1_3']))
        self.dw.leNormStd1_1.setText(str(self.omrat.segment_data[new_id]['std1_1']))
        self.dw.leNormStd1_2.setText(str(self.omrat.segment_data[new_id]['std1_2']))
        self.dw.leNormStd1_3.setText(str(self.omrat.segment_data[new_id]['std1_3']))
        self.dw.leNormMean2_1.setText(str(self.omrat.segment_data[new_id]['mean2_1']))
        self.dw.leNormMean2_2.setText(str(self.omrat.segment_data[new_id]['mean2_2']))
        self.dw.leNormMean2_3.setText(str(self.omrat.segment_data[new_id]['mean2_3']))
        self.dw.leNormStd2_1.setText(str(self.omrat.segment_data[new_id]['std2_1']))
        self.dw.leNormStd2_2.setText(str(self.omrat.segment_data[new_id]['std2_2']))
        self.dw.leNormStd2_3.setText(str(self.omrat.segment_data[new_id]['std2_3']))
        self.dw.leNormWeight1_1.setText(str(self.omrat.segment_data[new_id]['weight1_1']))
        self.dw.leNormWeight1_2.setText(str(self.omrat.segment_data[new_id]['weight1_2']))
        self.dw.leNormWeight1_3.setText(str(self.omrat.segment_data[new_id]['weight1_3']))
        self.dw.leNormWeight2_1.setText(str(self.omrat.segment_data[new_id]['weight2_1']))
        self.dw.leNormWeight2_2.setText(str(self.omrat.segment_data[new_id]['weight2_2']))
        self.dw.leNormWeight2_3.setText(str(self.omrat.segment_data[new_id]['weight2_3']))
        self.dw.leUniformMin1.setText(str(self.omrat.segment_data[new_id]['u_min1']))
        self.dw.leUniformMax1.setText(str(self.omrat.segment_data[new_id]['u_max1']))
        self.dw.sbUniformP1.setValue(int(self.omrat.segment_data[new_id]['u_p1']))
        self.dw.LEMeanTimeSeconds1.setText(str(self.omrat.segment_data[new_id]['ai1']))
        self.dw.leUniformMin2.setText(str(self.omrat.segment_data[new_id]['u_min2']))
        self.dw.leUniformMax2.setText(str(self.omrat.segment_data[new_id]['u_max2']))
        self.dw.sbUniformP2.setValue(int(self.omrat.segment_data[new_id]['u_p2']))
        self.dw.LEMeanTimeSeconds2.setText(str(self.omrat.segment_data[new_id]['ai2']))
        self.last_id = new_id

    def add_dist2plot(self, ax: Axes, parameters: Params, data: np.ndarray, first: bool, update_dist: bool=True):
        try:
            if update_dist:
                fit_result: tuple[float, float] = stats.norm.fit(data) # type: ignore
                mean, std = fit_result
                parameters.normal1.mean = round(mean)
                parameters.normal1.std = round(std)
                if first:
                    self.dw.leNormMean1_1.setText(str(parameters.normal1.mean))
                    self.dw.leNormStd1_1.setText(str(parameters.normal1.std))
                    self.dw.leNormWeight1_1.setText("100")
                else:
                    self.dw.leNormMean2_1.setText(str(parameters.normal1.mean))
                    self.dw.leNormStd2_1.setText(str(parameters.normal1.std))
                    self.dw.leNormWeight2_1.setText("100")
                parameters.normal1.probability = 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
                parameters.uniform.lower = data.min()
                parameters.uniform.upper = data.max()
            x: np.ndarray = np.linspace(data.min(), data.max(), 1000, dtype=float)
            tot_y = np.zeros(x.size)
            tot_p = 0
            current_ylim = ax.get_ylim()
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
            if tot_p > 0:
                result_y = tot_y / tot_p
                if first:
                    ax.plot(x, result_y, 'k', label='Resulting P')
                else:
                    ax.plot(x, result_y, '--k', label='Resulting P2')
                ax.set_ylim(current_ylim)
        except Exception as e:
            print(e)
            
    def run_update_plot(self, segment_id:str| None=None) -> None:
        self.change_dist_segment(segment_id)
        if segment_id is None:
            segment_id = self.last_id
            update_dist = True
        else:
            update_dist = False
        if segment_id not in self.omrat.ais.dist_data:
            return
        d = self.omrat.ais.dist_data[segment_id]
        p1, p2 = self.get_leg_params()
        self.plot_data(d['line1'], d['line2'], p1, p2, update_dist)
        self.last_id = segment_id
        
    def plot_data(self, data: np.ndarray, data2: np.ndarray, parameters1: Params, parameters2: Params, 
                  update_dist:bool=True) -> None:
        """Makes the plot in the top left corner"""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        fig: Figure = plt.figure(figsize=(10, 6)) # type: ignore
        gs = gridspec.GridSpec(1, 1)
        ax: Axes = fig.add_subplot(gs[0, 0]) # type: ignore
        ax.hist(data, bins=50, density=True, alpha=0.6, color='b', label=self.dw.laDir1.text())
        ax.hist(data2, bins=50, density=True, alpha=0.6, color='g', label=self.dw.laDir2.text())
        self.add_dist2plot(ax, parameters1, data, True, update_dist)
        self.add_dist2plot(ax, parameters2, data2, False, update_dist)

        plt.legend()
        ax.set_xlabel('Values')
        ax.set_ylabel('Density')

        # Remove the previous canvas if it exists
        if hasattr(self.dw, "DistributionWidget") and self.dw.DistributionWidget is not None:
            if hasattr(self, "canvas") and self.canvas is not None:
                self.dw.DistributionWidget.removeWidget(self.canvas)
                self.canvas.deleteLater()

        fig.tight_layout()
        self.canvas = FigureCanvas(fig)
        self.dw.DistributionWidget.addWidget(self.canvas)
        self.canvas.draw()
        plt.close(fig)

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
        self.run_update_plot(self.last_id)

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

