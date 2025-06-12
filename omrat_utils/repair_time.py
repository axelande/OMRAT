from __future__ import annotations
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
from numpy import exp, log, linspace
from scipy import stats


if TYPE_CHECKING:
    from omrat_utils.handle_settings import DriftSettings

class Repair:
    def __init__(self, settings: DriftSettings) -> None:
        self.sett = settings
        self.canvas = None
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.tick_params(axis="y",direction="in", pad=-10)
        self.ax.tick_params(axis="x",direction="in", pad=-10)
        self.figure.tight_layout()
        self.sett.dsw.canRepairViewLay.addWidget(self.canvas)

    def test_evaluate(self):
        xs = linspace(0, 4, 20)
        ys = []
        self.ax.clear()
        for x in xs:
            try:
                ys.append(eval(self.sett.dsw.leRepairFunc.toPlainText()))
            except Exception as e:
                ys.append(0)
                print(e)
        self.ax.plot(xs, ys)
        self.canvas.draw()
        
    def get_repair_prob(self, x):
        if self.sett.dsw.rbUserDefined.isChecked() == 1:
            return eval(self.sett.dsw.leRepairFunc.toPlainText())
        else:
            std = float(self.sett.dsw.leRepairStd.text())
            loc = float(self.sett.dsw.leRepairLoc.text())
            scale = float(self.sett.dsw.leRepairScale.text())
            drift = stats.lognorm(std, loc, scale)
            repaired = drift.cdf(x)
            return repaired