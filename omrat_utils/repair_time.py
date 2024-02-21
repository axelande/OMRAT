from __future__ import annotations
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
from numpy import exp, log, linspace
from scipy import stats


if TYPE_CHECKING:
    from open_mrat import OpenMRAT

class Repair:
    def __init__(self, parent: OpenMRAT) -> None:
        self.p = parent
        self.canvas = None
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.tick_params(axis="y",direction="in", pad=-10)
        self.ax.tick_params(axis="x",direction="in", pad=-10)
        self.figure.tight_layout()
        self.p.dockwidget.canRepairViewLay.addWidget(self.canvas)

    def test_evaluate(self):
        xs = linspace(0, 4, 20)
        ys = []
        self.ax.clear()
        for x in xs:
            try:
                ys.append(eval(self.p.dockwidget.leRepairFunc.toPlainText()))
            except Exception as e:
                ys.append(0)
                print(e)
        self.ax.plot(xs, ys)
        self.canvas.draw()
        
    def get_repair_prob(self, x):
        if self.p.dockwidget.tabRepair.currentIndex() == 1:
            return eval(self.p.dockwidget.leRepairFunc.toPlainText())
        else:
            std = float(self.p.dockwidget.leRepairStd.text())
            loc = float(self.p.dockwidget.leRepairLoc.text())
            scale = float(self.p.dockwidget.leRepairScale.text())
            drift = stats.lognorm(std, loc, scale)
            repaired = drift.cdf(x)
            return repaired