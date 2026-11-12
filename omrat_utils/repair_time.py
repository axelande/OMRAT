from __future__ import annotations
from typing import TYPE_CHECKING

import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
from numpy import linspace
from scipy import stats

from compute.basic_equations import _safe_compile, _safe_eval
from compute.drift_reach import REACH_QUANTILE, repair_time_quantile_h
from omrat_utils.number_input import parse_decimal

# QMessageBox is only available inside QGIS; standalone test runs import
# this module without Qt, so fall back to console output there.
try:
    from qgis.PyQt.QtWidgets import QMessageBox
except Exception:  # nosec B110 B112 - headless/test environment
    QMessageBox = None  # type: ignore[assignment]


if TYPE_CHECKING:
    from omrat_utils.handle_settings import DriftSettings

_EXAMPLE = "stats.norm(loc=0, scale=1).cdf(x)"

# A user expression is checked (and plotted) on 0..4 h.
_TEST_HOURS = 4.0
_TEST_POINTS = 20
# The lognormal curve is drawn smooth, at least over 0..4 h and far enough
# to show where REACH_QUANTILE of the ships are repaired (the drift model's
# corridor length).
_LOGNORM_POINTS = 200
_LOGNORM_MARGIN = 1.15

_LOGNORM_FIELDS = (
    ('leRepairStd', 'Standard deviation'),
    ('leRepairLoc', 'Location'),
    ('leRepairScale', 'Scale'),
)


def lognormal_params_from_text(
    std_text: str, loc_text: str, scale_text: str,
) -> tuple[tuple[float, float, float] | None, str | None]:
    """Parse the three Lognormal fields into ``((std, loc, scale), error)``.

    ``,`` is accepted as the decimal mark, like everywhere else in the
    drift settings.  ``std`` and ``scale`` must be > 0 (scipy's lognorm is
    undefined otherwise); ``loc`` is the time shift in hours and may be
    any finite number.
    """
    values: list[float] = []
    for (_, label), text in zip(_LOGNORM_FIELDS, (std_text, loc_text, scale_text)):
        try:
            value = parse_decimal(text)
        except ValueError:
            return None, (
                f"Lognormal {label.lower()}: '{str(text).strip()}' is not a number."
            )
        if not math.isfinite(value):
            return None, f"Lognormal {label.lower()} must be a finite number."
        values.append(value)
    std, loc, scale = values
    if std <= 0:
        return None, "Lognormal standard deviation must be greater than 0."
    if scale <= 0:
        return None, "Lognormal scale must be greater than 0."
    return (std, loc, scale), None


def lognormal_curve(
    std: float, loc: float, scale: float,
) -> tuple[list[float], list[float], float | None]:
    """Return ``(xs, ys, t_reach)`` for the lognormal repair CDF.

    ``ys`` is the share repaired after ``xs`` hours of drift; ``t_reach``
    is the time by which ``REACH_QUANTILE`` are repaired (same rule as the
    drift model, :mod:`compute.drift_reach`), or ``None``.
    """
    t_reach = repair_time_quantile_h(
        {'use_lognormal': True, 'std': std, 'loc': loc, 'scale': scale},
        REACH_QUANTILE)
    x_max = _TEST_HOURS
    if t_reach is not None and math.isfinite(t_reach):
        x_max = max(x_max, _LOGNORM_MARGIN * t_reach)
    xs = linspace(0.0, x_max, _LOGNORM_POINTS)
    ys = stats.lognorm(std, loc, scale).cdf(xs)
    return [float(x) for x in xs], [float(y) for y in ys], t_reach


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
        xs = linspace(0, _TEST_HOURS, _TEST_POINTS)
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

    def _user_defined(self) -> bool:
        return bool(self.sett.dsw.rbUserDefined.isChecked())

    def lognormal_params(self) -> tuple[tuple[float, float, float] | None, str | None]:
        """``((std, loc, scale), error)`` from the Lognormal fields."""
        dsw = self.sett.dsw
        return lognormal_params_from_text(
            *(getattr(dsw, name).text() for name, _ in _LOGNORM_FIELDS))

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
        """**Test eval**: plot the selected repair model, popup on a problem."""
        self._evaluate(popup=True)

    def refresh_plot(self):
        """Redraw while the user edits (fields, radio buttons).

        Same plot as :meth:`test_evaluate`, but a problem is written in
        the plot instead of a popup: these slots fire on every keystroke,
        and on both radio buttons of a switch.
        """
        self._evaluate(popup=False)

    def _evaluate(self, popup: bool) -> None:
        self.ax.clear()
        if self._user_defined():
            self._evaluate_user_defined(popup)
        else:
            self._evaluate_lognormal(popup)

    def _report_problem(self, message: str, popup: bool) -> None:
        """Leave the plot blank (the test did not pass) and say why."""
        if popup:
            self.canvas.draw()
            self._show_error(message)
            return
        self.ax.text(0.5, 0.5, message.split("\n\n")[0], ha="center",
                     va="center", wrap=True, transform=self.ax.transAxes)
        self.canvas.draw()

    def _evaluate_lognormal(self, popup: bool) -> None:
        params, err = self.lognormal_params()
        if err is not None:
            self._report_problem(err, popup)
            return
        xs, ys, t_reach = lognormal_curve(*params)
        self.ax.plot(xs, ys, label="Share repaired after x h of drift")
        self.ax.set_xlim(0.0, xs[-1])
        self.ax.set_ylim(0.0, 1.05)
        if t_reach is not None:
            self.ax.axvline(t_reach, linestyle="--", linewidth=1, color="grey",
                            label=f"{REACH_QUANTILE:.0%} repaired after {t_reach:.3g} h")
        self.ax.legend(loc="best", fontsize=8, frameon=False)
        self.canvas.draw()

    def _evaluate_user_defined(self, popup: bool) -> None:
        xs = linspace(0, _TEST_HOURS, _TEST_POINTS)
        ys, err = self.validate_expression(self.sett.dsw.leRepairFunc.toPlainText())
        if err is not None:
            self._report_problem(err, popup)
            return
        self.ax.plot(xs, ys)
        self.canvas.draw()
        # A repair-time function should behave like a CDF: values in [0, 1].
        if popup and (min(ys) < -1e-9 or max(ys) > 1.0 + 1e-9):
            self._show_info(
                "Note: a repair-time function should return the REPAIRED "
                "fraction as a probability between 0 and 1 "
                f"(got min = {min(ys):.3g}, max = {max(ys):.3g}).\n\n"
                f"Example:  {_EXAMPLE}"
            )

    def get_repair_prob(self, x):
        if self._user_defined():
            code = _safe_compile(self.sett.dsw.leRepairFunc.toPlainText())
            return _safe_eval(code, x)
        params, err = self.lognormal_params()
        if err is not None:
            raise ValueError(err)
        std, loc, scale = params
        return stats.lognorm(std, loc, scale).cdf(x)
