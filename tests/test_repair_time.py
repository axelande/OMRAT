"""Unit tests for omrat_utils/repair_time.py.

The ``Repair`` class is a thin wrapper around scipy.stats.lognorm + an
``eval``'d user-defined func.  Tests focus on:

- ``get_repair_prob`` returns a CDF probability in [0, 1]
- The user-defined path uses the leRepairFunc text as a Python expression
- The lognormal path uses std/loc/scale from the text widgets
- ``test_evaluate`` plots without crashing even when the user types
  an invalid expression
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Use the headless Agg backend so plt.figure() doesn't try to spin up Qt.
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot  # noqa: E402

from omrat_utils.repair_time import Repair  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Every Repair opens a pyplot figure; close them after each test."""
    yield
    matplotlib.pyplot.close('all')


def _make_repair(
    *,
    user_defined: bool,
    func_text: str = "",
    std: float = 1.0, loc: float = 0.0, scale: float = 1.0,
) -> Repair:
    """Build a Repair whose settings are driven by a MagicMock tree."""
    settings = MagicMock()
    dsw = settings.dsw
    dsw.leRepairFunc.toPlainText.return_value = func_text
    dsw.leRepairStd.text.return_value = str(std)
    dsw.leRepairLoc.text.return_value = str(loc)
    dsw.leRepairScale.text.return_value = str(scale)
    dsw.rbUserDefined.isChecked.return_value = 1 if user_defined else 0
    return Repair(settings)


class TestGetRepairProb:
    def test_lognormal_matches_scipy(self):
        r = _make_repair(user_defined=False, std=1.0, loc=0.0, scale=1.0)
        x = 2.5
        expected = float(stats.lognorm(1.0, 0.0, 1.0).cdf(x))
        assert r.get_repair_prob(x) == pytest.approx(expected, abs=1e-12)

    def test_lognormal_at_zero(self):
        r = _make_repair(user_defined=False, std=1.0, loc=0.0, scale=1.0)
        assert r.get_repair_prob(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_user_defined_function_evaluated(self):
        r = _make_repair(user_defined=True, func_text="x * 2")
        assert r.get_repair_prob(3.0) == 6.0

    def test_user_defined_with_scipy_call(self):
        # The repair-function expression has access to the ``stats`` and
        # ``norm`` names exposed by basic_equations._SAFE_EVAL_GLOBALS.
        # Direct ``__import__``/``open``/etc. are intentionally blocked.
        r = _make_repair(
            user_defined=True,
            func_text="stats.norm(loc=0, scale=1).cdf(x)",
        )
        got = r.get_repair_prob(1.0)
        expected = float(stats.norm(0, 1).cdf(1.0))
        assert got == pytest.approx(expected, abs=1e-12)

    def test_user_defined_blocks_dunder_imports(self):
        """The hardened evaluator must reject __import__ etc."""
        r = _make_repair(
            user_defined=True,
            func_text="__import__('os').system('echo pwn')",
        )
        with pytest.raises((NameError, ValueError)):
            r.get_repair_prob(1.0)


class TestTestEvaluate:
    def test_valid_expression_plots_and_draws(self):
        r = _make_repair(user_defined=True, func_text="x ** 2")
        # Replace the Matplotlib surface with mocks so the test stays
        # focused on the plumbing, not matplotlib's internals.
        r.ax = MagicMock()
        r.canvas = MagicMock()
        r.test_evaluate()
        r.ax.clear.assert_called_once()
        r.ax.plot.assert_called_once()
        # y values should be x**2 over 20 linspace points.
        _, y_plot = r.ax.plot.call_args.args
        xs = np.linspace(0, 4, 20)
        assert y_plot == pytest.approx(list(xs ** 2), abs=1e-12)
        r.canvas.draw.assert_called_once()

    def test_invalid_expression_shows_error_and_no_plot(self, capsys):
        r = _make_repair(user_defined=True, func_text="this will fail")
        r.ax = MagicMock()
        r.canvas = MagicMock()
        r.test_evaluate()
        # Validation fails -> nothing is plotted, canvas redrawn blank,
        # and a human-readable error is emitted (popup in QGIS, console here).
        r.ax.plot.assert_not_called()
        r.canvas.draw.assert_called_once()
        captured = capsys.readouterr()
        assert "Could not parse" in captured.out

    def test_distribution_object_gives_cdf_hint(self, capsys):
        """Forgetting .cdf(x) must produce a helpful message, not a
        matplotlib TypeError traceback (regression for the QGIS log crash)."""
        r = _make_repair(user_defined=True,
                         func_text="stats.norm(loc=0, scale=1)")
        r.ax = MagicMock()
        r.canvas = MagicMock()
        r.test_evaluate()
        r.ax.plot.assert_not_called()
        captured = capsys.readouterr()
        assert "not a number" in captured.out
        assert ".cdf(x)" in captured.out

    def test_valid_cdf_expression_no_warning(self, capsys):
        r = _make_repair(user_defined=True,
                         func_text="stats.norm(loc=0, scale=1).cdf(x)")
        r.ax = MagicMock()
        r.canvas = MagicMock()
        r.test_evaluate()
        r.ax.plot.assert_called_once()
        captured = capsys.readouterr()
        # In [0, 1] everywhere -> no range note, no error.
        assert captured.out == ""

    def test_out_of_range_expression_gets_note(self, capsys):
        r = _make_repair(user_defined=True, func_text="x * 2")
        r.ax = MagicMock()
        r.canvas = MagicMock()
        r.test_evaluate()
        # Still plotted (values are numbers), but a CDF-range note is shown.
        r.ax.plot.assert_called_once()
        captured = capsys.readouterr()
        assert "between 0 and 1" in captured.out

    def test_nonfinite_expression_shows_error(self, capsys):
        r = _make_repair(user_defined=True, func_text="log(x - 5)")
        r.ax = MagicMock()
        r.canvas = MagicMock()
        r.test_evaluate()
        r.ax.plot.assert_not_called()
        captured = capsys.readouterr()
        assert captured.out  # an error was reported (domain -> nan/error)

    def test_empty_expression_shows_error(self, capsys):
        r = _make_repair(user_defined=True, func_text="   ")
        r.ax = MagicMock()
        r.canvas = MagicMock()
        r.test_evaluate()
        r.ax.plot.assert_not_called()
        captured = capsys.readouterr()
        assert "empty" in captured.out


class TestLognormalEvaluate:
    """Test eval with *Lognormal* selected plots the lognormal CDF.

    It used to validate the (empty) user-defined expression box whatever
    radio button was selected, so Lognormal always answered "The repair
    function is empty".
    """

    def test_lognormal_plots_cdf_not_expression(self, capsys):
        r = _make_repair(user_defined=False, func_text="",
                         std=0.97, loc=0.2, scale=0.49)
        r.ax = MagicMock()
        r.canvas = MagicMock()
        r.test_evaluate()
        assert capsys.readouterr().out == ""   # no error, no note
        r.ax.plot.assert_called_once()
        xs, ys = r.ax.plot.call_args.args
        expected = stats.lognorm(0.97, 0.2, 0.49).cdf(np.asarray(xs))
        assert ys == pytest.approx(list(expected), abs=1e-12)
        r.canvas.draw.assert_called_once()

    def test_lognormal_range_reaches_the_drift_model_quantile(self):
        from compute.drift_reach import REACH_QUANTILE
        r = _make_repair(user_defined=False, std=0.95, loc=0.2, scale=0.85)
        r.ax = MagicMock()
        r.canvas = MagicMock()
        r.test_evaluate()
        xs, ys = r.ax.plot.call_args.args
        t99 = float(stats.lognorm(0.95, 0.2, 0.85).ppf(REACH_QUANTILE))
        assert xs[0] == 0.0
        assert xs[-1] > t99 > 4.0          # the default 0-4 h would cut it
        assert ys[-1] > REACH_QUANTILE
        r.ax.axvline.assert_called_once()
        assert r.ax.axvline.call_args.args[0] == pytest.approx(t99)

    def test_lognormal_accepts_decimal_comma(self, capsys):
        r = _make_repair(user_defined=False)
        r.sett.dsw.leRepairStd.text.return_value = "0,97"
        r.sett.dsw.leRepairLoc.text.return_value = "0,2"
        r.sett.dsw.leRepairScale.text.return_value = "0,49"
        r.ax = MagicMock()
        r.canvas = MagicMock()
        r.test_evaluate()
        assert capsys.readouterr().out == ""
        r.ax.plot.assert_called_once()
        assert r.get_repair_prob(1.0) == pytest.approx(
            float(stats.lognorm(0.97, 0.2, 0.49).cdf(1.0)), abs=1e-12)

    @pytest.mark.parametrize("field, text, words", [
        ("leRepairStd", "abc", "standard deviation: 'abc' is not a number"),
        ("leRepairStd", "0", "standard deviation must be greater than 0"),
        ("leRepairScale", "-1", "scale must be greater than 0"),
        ("leRepairLoc", "", "location: '' is not a number"),
        ("leRepairLoc", "inf", "location must be a finite number"),
    ])
    def test_bad_lognormal_parameter_is_named(self, capsys, field, text, words):
        r = _make_repair(user_defined=False)
        getattr(r.sett.dsw, field).text.return_value = text
        r.ax = MagicMock()
        r.canvas = MagicMock()
        r.test_evaluate()
        r.ax.plot.assert_not_called()
        r.canvas.draw.assert_called_once()
        out = capsys.readouterr().out
        assert words in out.lower()
        assert "repair function is empty" not in out


class TestRefreshPlot:
    """Edits redraw without popups; the problem goes in the plot."""

    def test_bad_lognormal_is_written_in_the_plot(self, capsys):
        r = _make_repair(user_defined=False, std=0.0)
        r.ax = MagicMock()
        r.canvas = MagicMock()
        r.refresh_plot()
        assert capsys.readouterr().out == ""   # no popup / console message
        r.ax.plot.assert_not_called()
        r.ax.text.assert_called_once()
        assert "greater than 0" in r.ax.text.call_args.args[2]
        r.canvas.draw.assert_called_once()

    def test_empty_expression_is_written_in_the_plot(self, capsys):
        r = _make_repair(user_defined=True, func_text="")
        r.ax = MagicMock()
        r.canvas = MagicMock()
        r.refresh_plot()
        assert capsys.readouterr().out == ""
        msg = r.ax.text.call_args.args[2]
        assert msg == "The repair function is empty."

    def test_out_of_range_note_only_on_test_eval(self, capsys):
        r = _make_repair(user_defined=True, func_text="x * 2")
        r.ax = MagicMock()
        r.canvas = MagicMock()
        r.refresh_plot()
        r.ax.plot.assert_called_once()
        assert capsys.readouterr().out == ""

    def test_lognormal_plots(self):
        r = _make_repair(user_defined=False, std=0.97, loc=0.2, scale=0.49)
        r.ax = MagicMock()
        r.canvas = MagicMock()
        r.refresh_plot()
        r.ax.plot.assert_called_once()
