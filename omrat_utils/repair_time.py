from __future__ import annotations
from typing import TYPE_CHECKING

import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
from numpy import linspace
from scipy import stats

from compute.basic_equations import _safe_compile, _safe_eval

# QMessageBox is only available inside QGIS; standalone test runs import
# this module without Qt, so fall back to console output there.
try:
    from qgis.PyQt.QtWidgets import QMessageBox
except Exception:  # nosec B110 B112 - headless/test environment
    QMessageBox = None  # type: ignore[assignment]


if TYPE_CHECKING:
    from omrat_utils.handle_settings import DriftSettings

_EXAMPLE = "stats.norm(loc=0, scale=1).cdf(x)"


class Repair:
    def __init__(self, settings: DriftSettings) -> None:
        self.sett = settings
        self.canvas = None
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.tick_params(axis="y", direction="in", pad=-10)
        self.ax.tick_params(axis="x", direction="in", pad=-10)
        self.figure.tight_layout()
        self.sett.dsw.canRepairViewLay.addWidget(self.canvas)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_expression(
        self, code_str: str,
    ) -> tuple[list[float] | None, str | None]:
        """Evaluate *code_str* on a test grid and return ``(ys, error)``.

        ``ys`` is the list of evaluated values on ``x = linspace(0, 4, 20)``
        when the expression is a valid repair-time CDF candidate, else
        ``None`` with a human-readable ``error`` describing exactly what is
        wrong (parse error, evaluation error, non-numeric result, or a
        non-finite value) and how to fix it.
        """
        code_str = (code_str or "").strip()
        if not code_str:
            return None, (
                "The repair function is empty.\n\n"
                f"Enter an expression in x (drift time in hours), e.g.\n"
                f"    {_EXAMPLE}"
            )
        try:
            code = _safe_compile(code_str)
        except Exception as e:
            return None, (
                "Could not parse the repair function:\n"
                f"    {e}\n\n"
                "Only mathematical expressions are allowed (x, numbers, "
                "+-*/**, exp, log, sqrt, stats.norm, ...).\n"
                f"Example:  {_EXAMPLE}"
            )
        xs = linspace(0, 4, 20)
        ys: list[float] = []
        for x in xs:
            try:
                y = _safe_eval(code, float(x))
            except Exception as e:
                return None, (
                    f"The expression failed at x = {x:.2f} h:\n"
                    f"    {type(e).__name__}: {e}\n\n"
                    f"Example of a valid function:  {_EXAMPLE}"
                )
            try:
                y_f = float(y)
            except (TypeError, ValueError):
                hint = ""
                tname = type(y).__name__
                if "rv" in tname or "frozen" in tname:
                    hint = (
                        "\nIt looks like you created a distribution object "
                        "but did not evaluate it.\n"
                        "Did you forget to call .cdf(x)?"
                    )
                return None, (
                    f"The expression returned a {tname}, not a number, "
                    f"at x = {x:.2f} h.{hint}\n\n"
                    f"Example of a valid function:  {_EXAMPLE}"
                )
            if not math.isfinite(y_f):
                return None, (
                    f"The expression returned a non-finite value "
                    f"({y_f}) at x = {x:.2f} h.\n\n"
                    f"Example of a valid function:  {_EXAMPLE}"
                )
            ys.append(y_f)
        return ys, None

    def _show_error(self, message: str) -> None:
        """Show a popup in QGIS; fall back to console when Qt is absent."""
        print(message)
        if QMessageBox is not None:
            try:
                QMessageBox.warning(self.sett.dsw, "Repair function error", message)
            except Exception:  # nosec B110 B112 - never block on UI issues
                pass

    def _show_info(self, message: str) -> None:
        print(message)
        if QMessageBox is not None:
            try:
                QMessageBox.information(self.sett.dsw, "Repair function", message)
            except Exception:  # nosec B110 B112
                pass

    # ------------------------------------------------------------------
    # GUI actions
    # ------------------------------------------------------------------
    def test_evaluate(self):
        xs = linspace(0, 4, 20)
        self.ax.clear()
        ys, err = self.validate_expression(self.sett.dsw.leRepairFunc.toPlainText())
        if err is not None:
            # Leave the plot blank so the user sees the test did not pass.
            self.canvas.draw()
            self._show_error(err)
            return
        self.ax.plot(xs, ys)
        self.canvas.draw()
        # A repair-time function should behave like a CDF: values in [0, 1].
        if min(ys) < -1e-9 or max(ys) > 1.0 + 1e-9:
            self._show_info(
                "Note: a repair-time function should return the REPAIRED "
                "fraction as a probability between 0 and 1 "
                f"(got min = {min(ys):.3g}, max = {max(ys):.3g}).\n\n"
                f"Example:  {_EXAMPLE}"
            )

    def get_repair_prob(self, x):
        if self.sett.dsw.rbUserDefined.isChecked() == 1:
            code = _safe_compile(self.sett.dsw.leRepairFunc.toPlainText())
            return _safe_eval(code, x)
        else:
            std = float(self.sett.dsw.leRepairStd.text())
            loc = float(self.sett.dsw.leRepairLoc.text())
            scale = float(self.sett.dsw.leRepairScale.text())
            drift = stats.lognorm(std, loc, scale)
            repaired = drift.cdf(x)
            return repaired
