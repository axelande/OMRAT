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
import matplotlib
matplotlib.use("Agg")

from omrat_utils.repair_time import Repair


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
        r = _make_repair(
            user_defined=True,
            func_text="__import__('scipy.stats', fromlist=['norm']).norm(loc=0, scale=1).cdf(x)",
        )
        got = r.get_repair_prob(1.0)
        expected = float(stats.norm(0, 1).cdf(1.0))
        assert got == pytest.approx(expected, abs=1e-12)


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

    def test_invalid_expression_yields_zeros(self, capsys):
        r = _make_repair(user_defined=True, func_text="this will fail")
        r.ax = MagicMock()
        r.canvas = MagicMock()
        r.test_evaluate()
        # Each failing eval appends 0.  All 20 y values should be 0.
        _, y_plot = r.ax.plot.call_args.args
        assert y_plot == [0] * 20
        # The function prints the exception per failing x; sanity-check it.
        captured = capsys.readouterr()
        assert captured.out  # something was printed
