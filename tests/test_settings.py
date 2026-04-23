import pytest
from unittest.mock import MagicMock, patch
from omrat_utils.handle_settings import DriftSettings

# test_handle_settings.py


@pytest.fixture
def mock_parent():
    class Parent:
        drift_values = None
    return Parent()

@pytest.fixture
def mock_dsw():
    # Mock all UI elements used in DriftSettings
    dsw = MagicMock()
    # Line edits for drift directions
    dsw.leDriftN.text.return_value = "1"
    dsw.leDriftNE.text.return_value = "2"
    dsw.leDriftE.text.return_value = "3"
    dsw.leDriftSE.text.return_value = "4"
    dsw.leDriftS.text.return_value = "5"
    dsw.leDriftSW.text.return_value = "6"
    dsw.leDriftW.text.return_value = "7"
    dsw.leDriftNW.text.return_value = "8"
    dsw.leDriftSpeed.text.return_value = "9"
    dsw.leDriftProb.text.return_value = "0.5"
    dsw.leAnchorProb.text.return_value = "0.6"
    dsw.leAnchorMaxDepth.text.return_value = "10"
    dsw.leRepairFunc.toPlainText.return_value = "func"
    dsw.leRepairStd.text.return_value = "0.1"
    dsw.leRepairLoc.text.return_value = "0.2"
    dsw.leRepairScale.text.return_value = "0.3"
    dsw.rbLogNormal.isChecked.return_value = True
    # Setters for populate_drift
    dsw.leDriftN.setText = MagicMock()
    dsw.leDriftNE.setText = MagicMock()
    dsw.leDriftE.setText = MagicMock()
    dsw.leDriftSE.setText = MagicMock()
    dsw.leDriftS.setText = MagicMock()
    dsw.leDriftSW.setText = MagicMock()
    dsw.leDriftW.setText = MagicMock()
    dsw.leDriftNW.setText = MagicMock()
    dsw.leDriftSpeed.setText = MagicMock()
    dsw.leAnchorProb.setText = MagicMock()
    dsw.leAnchorMaxDepth.setText = MagicMock()
    dsw.leDriftProb.setText = MagicMock()
    dsw.leRepairFunc.setText = MagicMock()
    dsw.leRepairStd.setText = MagicMock()
    dsw.leRepairLoc.setText = MagicMock()
    dsw.leRepairScale.setText = MagicMock()
    dsw.rbLogNormal.setChecked = MagicMock()
    # For unload
    dsw.pbTestRepair.clicked.disconnect = MagicMock()
    count_values = [1, 0]
    dsw.canRepairViewLay.count.side_effect = lambda: count_values.pop(0)
    mock_item = MagicMock()
    mock_widget = MagicMock()
    mock_item.widget.return_value = mock_widget
    dsw.canRepairViewLay.takeAt.return_value = mock_item
    mock_widget.deleteLater = MagicMock()
    # For run
    dsw.show = MagicMock()
    dsw.exec_ = MagicMock()
    dsw.findChild.return_value = MagicMock()
    dsw.pbTestRepair.clicked.connect = MagicMock()
    dsw.rbLogNormal.toggled.connect = MagicMock()
    dsw.rbUserDefined.toggled.connect = MagicMock()
    dsw.leRepairStd.textChanged.connect = MagicMock()
    dsw.leRepairLoc.textChanged.connect = MagicMock()
    dsw.leRepairScale.textChanged.connect = MagicMock()
    return dsw

@pytest.fixture
def mock_repair():
    repair = MagicMock()
    repair.test_evaluate = MagicMock()
    return repair

@patch('omrat_utils.handle_settings.DriftSettingsWidget')
@patch('omrat_utils.handle_settings.Repair')
def test_run(mock_repair_cls, mock_dsw_cls, mock_parent, mock_dsw, mock_repair):
    # Patch widget and repair
    mock_dsw_cls.return_value = mock_dsw
    mock_repair_cls.return_value = mock_repair
    ds = DriftSettings(mock_parent)
    ds.populate_drift = MagicMock()
    ds.dsw = mock_dsw
    ds.repair = mock_repair
    # Simulate buttonBox with accepted/rejected signals
    button_box = MagicMock()
    mock_dsw.findChild.return_value = button_box
    button_box.accepted.connect = MagicMock()
    button_box.rejected.connect = MagicMock()
    ds.run()
    ds.populate_drift.assert_called_once()
    mock_dsw.show.assert_called_once()
    # PyQt6 renamed exec_() -> exec() (the underscore avoided the
    # Python 2 keyword collision, which Python 3 doesn't need).
    mock_dsw.exec.assert_called_once()
    button_box.accepted.connect.assert_called()
    button_box.rejected.connect.assert_called()
    mock_dsw.pbTestRepair.clicked.connect.assert_called_with(mock_repair.test_evaluate)

@patch('omrat_utils.handle_settings.DriftSettingsWidget')
@patch('omrat_utils.handle_settings.Repair')
def test_commit_changes(mock_repair_cls, mock_dsw_cls, mock_parent, mock_dsw, mock_repair):
    mock_dsw_cls.return_value = mock_dsw
    mock_repair_cls.return_value = mock_repair
    ds = DriftSettings(mock_parent)
    ds.dsw = mock_dsw
    ds.commit_changes()
    # Check drift_values updated
    assert ds.drift_values['drift_p'] == 0.5
    # anchor_p: UI is percentage, mock returns "0.6" so commit_changes keeps it as 0.6 (already <= 1.0).
    assert ds.drift_values['anchor_p'] == 0.6
    assert ds.drift_values['anchor_d'] == 10
    # drift.speed is stored in knots (no m/s conversion at the GUI boundary).
    assert ds.drift_values['speed'] == 9
    # Rose values are stored as fractions; commit_changes divides the UI
    # percentage by 100, so "1" -> 0.01 etc.
    assert ds.drift_values['rose'] == {
        '0': 0.01, '45': 0.02, '90': 0.03, '135': 0.04,
        '180': 0.05, '225': 0.06, '270': 0.07, '315': 0.08,
    }
    assert ds.drift_values['repair']['func'] == "func"
    assert ds.drift_values['repair']['std'] == 0.1
    assert ds.drift_values['repair']['loc'] == 0.2
    assert ds.drift_values['repair']['scale'] == 0.3
    assert ds.drift_values['repair']['use_lognormal'] is True
    # Check parent updated
    assert mock_parent.drift_values == ds.drift_values

@patch('omrat_utils.handle_settings.DriftSettingsWidget')
@patch('omrat_utils.handle_settings.Repair')
def test_populate_drift(mock_repair_cls, mock_dsw_cls, mock_parent, mock_dsw, mock_repair):
    mock_dsw_cls.return_value = mock_dsw
    mock_repair_cls.return_value = mock_repair
    ds = DriftSettings(mock_parent)
    ds.dsw = mock_dsw
    # _ensure_blackout_table constructs a real QTableWidget; in a
    # headless unit test with a MagicMock parent this hangs.  Patch it
    # out -- the blackout-table plumbing is covered in the real-Qt
    # regression tests under test_regression_gui_state.py.
    ds._ensure_blackout_table = MagicMock(return_value=None)
    # Rose values are stored as fractions and displayed as percentages
    # (populate_drift multiplies by 100).  Use exactly-representable
    # fractions (multiples of 1/8 = 0.125) so the percentage arithmetic
    # doesn't introduce floating-point noise like 7.000000000000001.
    ds.drift_values = {
        'drift_p': 0.1,
        'anchor_p': 0.2,
        'anchor_d': 3,
        'speed': 4,
        'rose': {str(a): 0.125 for a in (0, 45, 90, 135, 180, 225, 270, 315)},
        'repair': {'func': 'f', 'std': 0.1, 'loc': 0.2, 'scale': 0.3, 'use_lognormal': False}
    }
    ds.populate_drift()
    # Uniform rose -- every direction renders as "12.5".
    for name in ('leDriftN', 'leDriftNE', 'leDriftE', 'leDriftSE',
                 'leDriftS', 'leDriftSW', 'leDriftW', 'leDriftNW'):
        getattr(mock_dsw, name).setText.assert_called_with("12.5")
    # drift.speed is stored in knots; populate_drift displays it directly
    # (the value is passed through float() so "4" becomes "4.0").
    mock_dsw.leDriftSpeed.setText.assert_called_with("4.0")
    # anchor_p 0.2 is displayed as percentage "20.0" by populate_drift
    # (the display path converts fraction -> percent).
    mock_dsw.leAnchorProb.setText.assert_called_with("20.0")
    mock_dsw.leAnchorMaxDepth.setText.assert_called_with("3")
    mock_dsw.leDriftProb.setText.assert_called_with("0.1")
    mock_dsw.leRepairFunc.setText.assert_called_with("f")
    mock_dsw.leRepairStd.setText.assert_called_with("0.1")
    mock_dsw.leRepairLoc.setText.assert_called_with("0.2")
    mock_dsw.leRepairScale.setText.assert_called_with("0.3")
    # populate_drift sets BOTH radio buttons explicitly (Qt's auto-
    # exclusive group silently ignores setChecked(False) on the only
    # checked button -- see omrat_utils/handle_settings.py:267).
    mock_dsw.rbLogNormal.setChecked.assert_called_with(False)
    mock_dsw.rbUserDefined.setChecked.assert_called_with(True)

@patch('omrat_utils.handle_settings.DriftSettingsWidget')
@patch('omrat_utils.handle_settings.Repair')
def test_unload(mock_repair_cls, mock_dsw_cls, mock_parent, mock_dsw, mock_repair):
    mock_dsw_cls.return_value = mock_dsw
    mock_repair_cls.return_value = mock_repair
    ds = DriftSettings(mock_parent)
    ds.dsw = mock_dsw
    ds.unload()
    mock_dsw.pbTestRepair.clicked.disconnect.assert_called_once()
    mock_dsw.canRepairViewLay.takeAt.assert_called_with(0)
    mock_dsw.canRepairViewLay.count.assert_called()
    mock_dsw.canRepairViewLay.takeAt.return_value.widget.return_value.deleteLater.assert_called_once()

@patch('omrat_utils.handle_settings.DriftSettingsWidget')
@patch('omrat_utils.handle_settings.Repair')
def test_discard_changes(mock_repair_cls, mock_dsw_cls, mock_parent, mock_dsw, mock_repair):
    mock_dsw_cls.return_value = mock_dsw
    mock_repair_cls.return_value = mock_repair
    ds = DriftSettings(mock_parent)
    ds.discard_changes()  # Should not raise