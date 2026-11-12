"""Regression tests for GUI state roundtrips in the Drift-settings dialog.

These tests use **real Qt widgets** (not MagicMock) because the bugs they
cover depended on actual Qt behaviour that mocks can't reproduce:

1. Qt's auto-exclusive radio group refuses to uncheck the only checked
   button.  `populate_drift` calling `rbLogNormal.setChecked(False)`
   alone was a silent no-op, so files with `use_lognormal: false` were
   loaded into a dialog that still reported `isChecked() == True` and
   `commit_changes` then wrote `use_lognormal: True` back -- cascading
   into a 20,000x over-estimate of drifting-grounding risk.

2. `drift.speed` was being double-converted between m/s and knots at the
   GUI boundary.  The cascade and the IWRAP importer both treat
   `drift.speed` as knots; the GUI was interpreting it as m/s, so an
   IWRAP-imported `1.94` knots was displayed as `3.771` knots and
   re-saving corrupted the file.

Run standalone:
    /c/OSGeo4W/apps/Python312/python.exe -m pytest -p no:qgis \\
        --noconftest tests/test_regression_gui_state.py -v
"""
from __future__ import annotations

import sys
import pytest
from unittest.mock import MagicMock

from qgis.PyQt.QtWidgets import QApplication

# Module-level QApplication so every test reuses the same Qt event loop.
_app = QApplication.instance() or QApplication(sys.argv)

from omrat_utils.handle_settings import DriftSettings  # noqa: E402
from ui.drift_settings_widget import DriftSettingsWidget  # noqa: E402


@pytest.fixture
def real_dsw():
    """A real DriftSettingsWidget with live QRadioButton / QLineEdit state."""
    return DriftSettingsWidget(None)


@pytest.fixture
def drift_settings(real_dsw):
    """DriftSettings wired to the real widget and a dummy parent.

    We bypass DriftSettings.__init__ (which constructs another widget
    and a Repair matplotlib canvas) so the test stays lightweight.
    """
    parent = MagicMock()
    parent.drift_values = None
    ds = DriftSettings.__new__(DriftSettings)
    ds.parent = parent
    ds.dsw = real_dsw
    ds.repair = MagicMock()
    ds.drift_values = {}
    ds._blackout_table = None  # populated lazily in _ensure_blackout_table
    return ds


# ---------------------------------------------------------------------------
# Repair-distribution radio-button roundtrip (bug #1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_lognormal", [False, True])
def test_populate_drift_radio_buttons_reflect_use_lognormal(
    drift_settings, use_lognormal
):
    """After populate_drift, exactly one of rbLogNormal / rbUserDefined is
    checked, matching the loaded `use_lognormal` flag.

    Before the fix, calling `setChecked(False)` alone on `rbLogNormal`
    (which is default-checked in drift_settings.ui) was a no-op because
    Qt's auto-exclusive group wouldn't uncheck the only checked button.
    """
    drift_settings.drift_values = _minimal_drift_values(
        use_lognormal=use_lognormal, speed=1.94
    )

    drift_settings.populate_drift()

    assert drift_settings.dsw.rbLogNormal.isChecked() is use_lognormal
    assert drift_settings.dsw.rbUserDefined.isChecked() is (not use_lognormal)


def test_populate_then_commit_preserves_use_lognormal_false(drift_settings):
    """The full roundtrip: load a file with use_lognormal=False, open the
    dialog, click OK without changing anything -- and the saved value is
    still False.  This is the exact path that caused the 20,000x
    over-estimate.
    """
    drift_settings.drift_values = _minimal_drift_values(
        use_lognormal=False, speed=1.94
    )

    drift_settings.populate_drift()
    drift_settings.commit_changes()

    assert drift_settings.drift_values['repair']['use_lognormal'] is False


# ---------------------------------------------------------------------------
# drift.speed unit roundtrip (bug #2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("speed_kts", [0.5, 1.0, 1.47, 1.94, 3.771, 12.5])
def test_drift_speed_roundtrip_is_identity_in_knots(drift_settings, speed_kts):
    """The GUI stores / displays / reads drift.speed in knots throughout.

    Before the fix, setText multiplied by 3600/1852 (m/s -> kts) and
    the save path divided by the same factor, so any file whose stored
    speed wasn't numerically identical to its round-trip wasn't really
    being preserved.  Now: store X kts -> display X kts -> save X kts.
    """
    drift_settings.drift_values = _minimal_drift_values(speed=speed_kts)

    drift_settings.populate_drift()
    # The widget displays the exact number (rounded to 3 decimals).
    assert float(drift_settings.dsw.leDriftSpeed.text()) == pytest.approx(
        speed_kts, abs=1e-3
    )

    # And commit_changes reads it back without another conversion.
    drift_settings.commit_changes()
    assert drift_settings.drift_values['speed'] == pytest.approx(
        speed_kts, abs=1e-3
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_drift_values(speed: float = 1.94, use_lognormal: bool = False
                          ) -> dict:
    """Return a minimal drift-values dict sufficient for populate_drift.

    All rose directions are set to a finite value so no division-by-zero
    is triggered when the directions are re-read on commit.
    """
    from compute.basic_equations import default_blackout_by_ship_type

    rose = {str(a): 0.125 for a in (0, 45, 90, 135, 180, 225, 270, 315)}
    repair = {
        'func': "__import__('scipy.stats', fromlist=['norm'])"
                ".norm(loc=0, scale=1).cdf(x)",
        'std': 0.95, 'loc': 0.2, 'scale': 0.85,
        'use_lognormal': use_lognormal,
    }
    return {
        'drift_p': 1.0,
        'anchor_p': 0.70,
        'anchor_d': 7,
        'speed': speed,
        'start_from': 'leg_center',
        'rose': rose,
        'repair': repair,
        'blackout_by_ship_type': default_blackout_by_ship_type(),
    }


# ---------------------------------------------------------------------------
# Repair-time "Test eval" follows the selected radio button
# ---------------------------------------------------------------------------

def _curves(ax):
    """Plotted curves (the dashed reach marker is a 2-point line too)."""
    return [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 2]


@pytest.fixture
def drift_with_repair(drift_settings, monkeypatch):
    """``drift_settings`` with a real :class:`Repair` (canvas in the dialog)
    and the popups recorded instead of shown."""
    import matplotlib.pyplot as plt
    import omrat_utils.repair_time as repair_time
    from omrat_utils.repair_time import Repair

    popups: list[tuple[str, str]] = []
    box = MagicMock()
    box.warning.side_effect = lambda _w, title, msg: popups.append((title, msg))
    box.information.side_effect = lambda _w, title, msg: popups.append((title, msg))
    monkeypatch.setattr(repair_time, 'QMessageBox', box)
    drift_settings.repair = Repair(drift_settings)
    drift_settings._signals_wired = False
    drift_settings.popups = popups
    yield drift_settings
    plt.close(drift_settings.repair.figure)


def test_test_eval_with_lognormal_plots_the_lognormal(drift_with_repair):
    """Lognormal selected + empty expression box: Test eval draws the
    lognormal CDF.  It used to check the expression box and answer
    "The repair function is empty"."""
    ds = drift_with_repair
    values = _minimal_drift_values(use_lognormal=True)
    values['repair']['func'] = ''
    values['repair'].update(std=0.97, loc=0.2, scale=0.49)
    ds.drift_values = values
    ds.populate_drift()
    ds._wire_signals()
    ds.dsw.pbTestRepair.click()
    assert ds.popups == []
    curves = _curves(ds.repair.ax)
    assert len(curves) == 1
    assert max(curves[0].get_ydata()) > 0.99


def test_repair_signals_are_wired_once(drift_with_repair):
    """``run`` is called on every open; the signals must not pile up
    (each extra connection meant one more evaluation / popup per click)."""
    ds = drift_with_repair
    calls: list[int] = []
    ds.repair.test_evaluate = lambda: calls.append(1)
    ds._wire_signals()
    ds._wire_signals()
    ds.dsw.pbTestRepair.click()
    assert calls == [1]


def test_switching_and_typing_never_pops_up(drift_with_repair):
    """Radio switches and keystrokes only redraw: an empty expression or a
    half-typed number is written in the plot, not shown as a popup."""
    ds = drift_with_repair
    values = _minimal_drift_values(use_lognormal=True)
    values['repair']['func'] = ''
    ds.drift_values = values
    ds.populate_drift()
    ds._wire_signals()
    ds.dsw.rbUserDefined.click()      # empty expression box
    ds.dsw.rbLogNormal.click()
    ds.dsw.leRepairStd.setText('')    # mid-edit
    ds.dsw.leRepairStd.setText('0,9')
    assert ds.popups == []
    assert len(_curves(ds.repair.ax)) == 1
    # The explicit button still reports a problem with a popup.
    ds.dsw.leRepairScale.setText('0')
    ds.dsw.pbTestRepair.click()
    assert len(ds.popups) == 1
    assert 'scale must be greater than 0' in ds.popups[0][1]
