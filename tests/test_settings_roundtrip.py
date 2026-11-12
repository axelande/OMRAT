"""A setting changed in its dialog reaches the calculation and the project file.

Each test follows one setting the way a user would:

1. load a copy of ``proj.omrat``;
2. open the setting's menu entry, change it in the real dialog, press OK;
3. **calculation**: click *Run model* -- the data handed to ``CalculationTask``
   carries the new value and the compute function that reads it sees it.
   For the collision model and the consequence step (about a second on
   ``proj.omrat``) the task is run and its result must move by the expected
   amount, e.g. a doubled head-on causation factor doubles head-on;
4. **file**: *File -> Save* writes the value into the ``.omrat`` file;
5. **load**: the setting is changed once more in the dialog, *File -> Load*
   reads the saved file back, and the reopened dialog shows the saved value.

Dialogs, file pickers and the task manager are stubbed by
``tests/gui_helpers.ModalGuard`` (the task is run in the test, synchronously).
Needs the QGIS conftest.
"""
from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import pytest
from qgis.PyQt.QtWidgets import QDoubleSpinBox

from tests.gui_helpers import (
    ModalGuard,
    accept_with,
    click_accept,
    click_reject,
    load_example,
    menu_action,
    read_with,
    settle,
)

COLLISION_TYPES = ('head_on', 'overtaking', 'crossing', 'merging', 'bend')


# ---------------------------------------------------------------------------
# The user's steps
# ---------------------------------------------------------------------------

@pytest.fixture
def project(omrat, monkeypatch, tmp_path):
    """``proj.omrat`` loaded from a temp copy, Run model ready to start."""
    import omrat as omrat_mod
    from compute.calculation_task import CalculationTask
    from omrat_utils.storage import Storage

    guard = ModalGuard(monkeypatch)
    path = load_example(omrat, tmp_path)
    tasks: list[CalculationTask] = []

    def make_task(**kwargs):
        tasks.append(CalculationTask(**kwargs))
        return tasks[-1]

    monkeypatch.setattr(omrat_mod, 'CalculationTask', make_task)
    load_from = {'path': str(path)}
    monkeypatch.setattr(Storage, 'select_file', lambda self: load_from['path'])
    monkeypatch.setattr(omrat, '_confirm_before_load', lambda _label: 'clear')
    omrat.main_widget.LEModelName.setText('roundtrip')
    omrat.main_widget.LEReportPath.setText(str(tmp_path))
    yield SimpleNamespace(omrat=omrat, guard=guard, path=path, tasks=tasks)
    guard.close()
    omrat._current_task = None


def in_dialog(p, menu: str, entry: str, edit) -> None:
    """Open ``menu > entry``, change the dialog with ``edit``, press OK."""
    p.guard.expect(accept_with(edit))
    menu_action(p.omrat, menu, entry).trigger()
    settle()
    assert not p.guard.errors, '\n'.join(p.guard.errors)


def shown_in_dialog(p, menu: str, entry: str, read):
    """Open ``menu > entry`` and return what ``read(dialog)`` finds, then Cancel."""
    into: dict = {}
    p.guard.expect(read_with(read, into))
    menu_action(p.omrat, menu, entry).trigger()
    settle()
    assert 'shown' in into, f'{menu} > {entry} opened no dialog'
    return into['shown']


def saved_now(p) -> dict:
    """What Save would write right now."""
    from omrat_utils.gather_data import GatherData
    return GatherData(p.omrat).get_all_for_save()


def run_model(p, phases=('collision',), execute: bool = True) -> tuple[dict, object]:
    """Click Run model; return the data the calculation received and, with
    ``execute``, the ``Calculation`` after the task ran (synchronously)."""
    p.omrat.set_run_phases(list(phases))
    p.omrat._current_task = None
    n = len(p.tasks)
    p.omrat.main_widget.pbRunModel.click()
    settle()
    assert len(p.tasks) == n + 1, 'Run model did not start a calculation'
    task = p.tasks[-1]
    sent = copy.deepcopy(task.data)
    p.omrat._current_task = None
    if execute:
        assert task.run(), task.error_msg
    return sent, p.omrat.calc


def save_and_read_file(p) -> dict:
    """File -> Save (onto the loaded copy); return the file's JSON."""
    menu_action(p.omrat, 'File', 'Save').trigger()
    settle()
    with open(p.path, encoding='utf-8') as f:
        return json.load(f)


def load_saved_file(p) -> None:
    menu_action(p.omrat, 'File', 'Load').trigger()
    settle()
    assert p.omrat.project_path == str(p.path)


def collision_totals(calc) -> dict:
    return {k: float(calc.collision_report['totals'][k]) for k in COLLISION_TYPES}


# ---------------------------------------------------------------------------
# Settings > Causation Factors
# ---------------------------------------------------------------------------

def _set_text(dlg, name: str, value) -> None:
    getattr(dlg, name).setText(str(value))


class TestCausationFactors:
    MENU = ('Settings', 'Causation Factors')

    def test_collision_factors_and_angle_limit(self, project):
        from compute.encounter_angle import min_angle_deg
        p = project
        base_pc = saved_now(p)['pc']
        _sent, calc = run_model(p)
        base = collision_totals(calc)
        assert base['head_on'] > 0 and base['overtaking'] > 0 and base['bend'] > 0

        headon, overtaking = 2 * base_pc['headon'], 3 * base_pc['overtaking']

        def edit(dlg):
            _set_text(dlg, 'leHeadOnCf', headon)
            _set_text(dlg, 'leOvertakingCf', overtaking)
            dlg.sbMinEncounterAngle.setValue(30.0)
        in_dialog(p, *self.MENU, edit)

        pc = saved_now(p)['pc']
        assert (pc['headon'], pc['overtaking'], pc['min_encounter_angle_deg']) == (headon, overtaking, 30.0)

        sent, calc = run_model(p)
        assert sent['pc']['headon'] == headon and sent['pc']['overtaking'] == overtaking
        assert min_angle_deg(sent['pc']) == 30.0
        after = collision_totals(calc)
        assert after['head_on'] == pytest.approx(2 * base['head_on'], rel=1e-9)
        assert after['overtaking'] == pytest.approx(3 * base['overtaking'], rel=1e-9)
        assert calc.collision_report['min_encounter_angle_deg'] == pytest.approx(30.0)
        assert after['bend'] <= base['bend']     # a higher limit only drops waypoint encounters

        written = save_and_read_file(p)['pc']
        assert (written['headon'], written['min_encounter_angle_deg']) == (headon, 30.0)

        in_dialog(p, *self.MENU, lambda dlg: (_set_text(dlg, 'leHeadOnCf', 1e-9),
                                              dlg.sbMinEncounterAngle.setValue(5.0)))
        load_saved_file(p)
        shown = shown_in_dialog(p, *self.MENU, lambda dlg: (
            float(dlg.leHeadOnCf.text()), dlg.sbMinEncounterAngle.value()))
        assert shown == (headon, 30.0)

    def test_powered_drifting_and_reduction_settings(self, project):
        from compute import passenger_reduction
        from compute.cat1_start_zone import start_zone_fraction
        from compute.squat import KEY_DRIFTING, KEY_POWERED, squat_on
        p = project

        def edit(dlg):
            _set_text(dlg, 'leGroundingCf', 1.1e-4)
            _set_text(dlg, 'leAllisionCat1Cf', 2.2e-4)
            dlg.cbSquatDrifting.setChecked(False)
            dlg.cbSquatPowered.setChecked(True)
            dlg.cbCat1SkipStart.setChecked(True)
            dlg.sbCat1SkipStartPct.setValue(25.0)
            dlg.sbPassengerReduction.setValue(5.0)
        in_dialog(p, *self.MENU, edit)

        expected = {'grounding': 1.1e-4, 'allision_cat1': 2.2e-4, KEY_DRIFTING: False, KEY_POWERED: True,
                    'cat1_skip_start': True, 'cat1_skip_start_pct': 25.0,
                    passenger_reduction.PASSENGER_KEY: 5.0}
        pc = saved_now(p)['pc']
        assert {k: pc[k] for k in expected} == expected

        sent, _calc = run_model(p, phases=('drifting', 'powered_grounding', 'powered_allision'), execute=False)
        assert {k: sent['pc'][k] for k in expected} == expected
        # The readers the models use agree with the dialog on the keys.
        assert squat_on(sent['pc'], KEY_DRIFTING) is False and squat_on(sent['pc'], KEY_POWERED) is True
        assert start_zone_fraction(sent['pc']) == pytest.approx(0.25)
        assert passenger_reduction.factor(sent['pc'], passenger_reduction.PASSENGER_KEY) == 5.0

        written = save_and_read_file(p)['pc']
        assert {k: written[k] for k in expected} == expected

        in_dialog(p, *self.MENU, lambda dlg: (dlg.sbCat1SkipStartPct.setValue(10.0),
                                              dlg.cbSquatDrifting.setChecked(True)))
        load_saved_file(p)
        shown = shown_in_dialog(p, *self.MENU, lambda dlg: (
            float(dlg.leGroundingCf.text()), dlg.cbSquatDrifting.isChecked(),
            dlg.sbCat1SkipStartPct.value(), dlg.sbPassengerReduction.value()))
        assert shown == (1.1e-4, False, 25.0, 5.0)

    def test_cancel_changes_nothing(self, project):
        p = project
        before = saved_now(p)['pc']
        p.guard.expect(lambda dlg: (_set_text(dlg, 'leHeadOnCf', 9.9), 0)[1])   # edit, then Cancel
        menu_action(p.omrat, *self.MENU).trigger()
        settle()
        assert saved_now(p)['pc'] == before


# ---------------------------------------------------------------------------
# Dialogs that live as long as the plugin: OK is connected once
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('entry, owner, ok_handler', [
    ('Causation Factors', 'omrat_utils.causation_factors.CausationFactors', 'commit_changes'),
    ('Ship Categories', 'omrat_utils.handle_ship_cat.ShipCategories', 'commit_changes'),
    ('AIS connection settings', 'omrat_utils.handle_ais.AIS', 'update_ais_settings'),
    ('Drift settings', 'omrat_utils.handle_settings.DriftSettings', 'commit_changes'),
])
def test_ok_runs_once_after_reopening(project, monkeypatch, entry, owner, ok_handler):
    """Each open used to connect OK again: after three opens one OK ran the
    handler four times (the AIS one opens a database connection each time)."""
    import importlib
    module, name = owner.rsplit('.', 1)
    calls: list[int] = []
    monkeypatch.setattr(getattr(importlib.import_module(module), name), ok_handler,
                        lambda self, *a: calls.append(1))
    action = menu_action(project.omrat, 'Settings', entry)
    for _ in range(3):
        action.trigger()                    # Cancel
        settle()
    project.guard.expect(click_accept)
    action.trigger()
    settle()
    assert calls == [1]


# ---------------------------------------------------------------------------
# Settings > Drift settings
# ---------------------------------------------------------------------------

class TestDriftSettings:
    MENU = ('Settings', 'Drift settings')

    def test_drift_settings(self, project):
        from scipy import stats
        from compute.drift_reach import REACH_QUANTILE, drift_reach_m
        from compute.drift_start import uses_distribution_center
        from omrat_utils.handle_settings import ROSE_FIELDS
        p = project
        rose_pct = {key: (30.0 if key == '0' else 10.0) for _name, key in ROSE_FIELDS}

        def edit(dlg):
            _set_text(dlg, 'leDriftSpeed', '3.5')
            _set_text(dlg, 'leAnchorProb', '80')          # percent in the dialog
            dlg.cbStartDriftingFrom.setCurrentIndex(1)     # Distribution center
            dlg.rbLogNormal.setChecked(True)
            _set_text(dlg, 'leRepairStd', '1.2')
            for name, key in ROSE_FIELDS:
                _set_text(dlg, name, rose_pct[key])
        in_dialog(p, *self.MENU, edit)

        def check(drift: dict) -> None:
            assert drift['speed'] == pytest.approx(3.5)
            assert drift['anchor_p'] == pytest.approx(0.80)
            assert drift['start_from'] == 'distribution_center'
            assert drift['repair']['std'] == pytest.approx(1.2) and drift['repair']['use_lognormal'] is True
            assert {k: float(v) for k, v in drift['rose'].items()} == pytest.approx(
                {k: v / 100.0 for k, v in rose_pct.items()})

        check(saved_now(p)['drift'])
        sent, _calc = run_model(p, phases=('drifting',), execute=False)
        check(sent['drift'])
        # The readers the drifting model uses (corridor reach, start point).
        assert uses_distribution_center(sent['drift']) is True
        rep = sent['drift']['repair']
        hours = stats.lognorm(rep['std'], loc=rep['loc'], scale=rep['scale']).ppf(REACH_QUANTILE)
        assert drift_reach_m(sent['drift']) == pytest.approx(3.5 * 1852.0 * hours, rel=1e-9)
        check(save_and_read_file(p)['drift'])

        in_dialog(p, *self.MENU, lambda dlg: (_set_text(dlg, 'leDriftSpeed', '1.0'),
                                              dlg.cbStartDriftingFrom.setCurrentIndex(0)))
        load_saved_file(p)
        check(p.omrat.drift_values)
        shown = shown_in_dialog(p, *self.MENU, lambda dlg: (
            float(dlg.leDriftSpeed.text()), float(dlg.leAnchorProb.text()),
            dlg.cbStartDriftingFrom.currentIndex(), float(dlg.leRepairStd.text())))
        assert shown == (3.5, 80.0, 1, 1.2)


# ---------------------------------------------------------------------------
# Settings > Ship Categories
# ---------------------------------------------------------------------------

def _busiest_length_class(traffic_data: dict) -> int:
    ships: dict[int, float] = {}
    for dirs in traffic_data.values():
        for block in dirs.values():
            for row in block.get('Frequency (ships/year)', []):
                for j, q in enumerate(row):
                    if isinstance(q, (int, float)) and math.isfinite(q):
                        ships[j] = ships.get(j, 0.0) + float(q)
    return max(ships, key=ships.get)


class TestShipCategories:
    MENU = ('Settings', 'Ship Categories')

    def test_length_interval(self, project):
        from compute.ship_collision_model import ShipCollisionModelMixin
        p = project
        _sent, calc = run_model(p)
        base = collision_totals(calc)
        col = _busiest_length_class(p.omrat.traffic_data)
        intervals = saved_now(p)['ship_categories']['length_intervals']
        new_max = float(intervals[col]['max']) + 20.0

        in_dialog(p, *self.MENU, lambda dlg: dlg.twLengths.item(col, 1).setText(f'{new_max:g}'))

        assert saved_now(p)['ship_categories']['length_intervals'][col]['max'] == new_max
        sent, calc = run_model(p)
        sent_intervals = sent['ship_categories']['length_intervals']
        assert sent_intervals[col]['max'] == new_max
        assert ShipCollisionModelMixin.get_loa_midpoint(col, sent_intervals) == pytest.approx(
            (float(sent_intervals[col]['min']) + new_max) / 2.0)
        # Longer ships in the busiest class: a larger collision diameter at the bends.
        assert collision_totals(calc)['bend'] > base['bend']
        assert save_and_read_file(p)['ship_categories']['length_intervals'][col]['max'] == new_max

        in_dialog(p, *self.MENU, lambda dlg: dlg.twLengths.item(col, 1).setText('999'))
        load_saved_file(p)
        shown = shown_in_dialog(p, *self.MENU, lambda dlg: float(dlg.twLengths.item(col, 1).text()))
        assert shown == new_max

    @pytest.mark.parametrize('how', ['Cancel button', 'Esc', 'window close'])
    def test_cancel_puts_the_tables_back(self, project, how):
        """The tables are the stored data, so Cancel has to undo the edits
        (before 2026-11-12 an edited length class was saved after Cancel)."""
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtTest import QTest
        p = project
        before = saved_now(p)['ship_categories']

        def edit_then_cancel(dlg):
            dlg.twLengths.item(0, 1).setText('33')
            dlg.cvTypes.item(0, 0).setText('Renamed type')
            if how == 'Cancel button':
                return click_reject(dlg)
            if how == 'Esc':
                QTest.keyClick(dlg, Qt.Key.Key_Escape)
            else:
                dlg.close()
            settle()
            return int(dlg.result())

        p.guard.expect(edit_then_cancel)
        menu_action(p.omrat, *self.MENU).trigger()
        settle()
        assert saved_now(p)['ship_categories'] == before
        shown = shown_in_dialog(p, *self.MENU, lambda dlg: (
            float(dlg.twLengths.item(0, 1).text()), dlg.cvTypes.item(0, 0).text()))
        assert shown == (before['length_intervals'][0]['max'], before['types'][0])

        # OK still keeps an edit.
        in_dialog(p, *self.MENU, lambda dlg: dlg.twLengths.item(0, 1).setText('33'))
        assert saved_now(p)['ship_categories']['length_intervals'][0]['max'] == 33.0


# ---------------------------------------------------------------------------
# Consequence menu
# ---------------------------------------------------------------------------

def _spin(tw, row: int, col: int) -> QDoubleSpinBox:
    w = tw.cellWidget(row, col)
    assert isinstance(w, QDoubleSpinBox), f'no spin box in cell ({row}, {col})'
    return w


def _matrix(tw) -> list[list[float]]:
    return [[_spin(tw, r, c).value() for c in range(tw.columnCount())] for r in range(tw.rowCount())]


class TestConsequence:
    def test_oil_onboard_scales_the_spill_volume(self, project):
        p = project
        menu = ('Consequence', 'Maximum oil onboard...')
        _sent, calc = run_model(p)
        base = calc.consequence_result['mean_spill_volume_per_year']
        assert base > 0
        # Doubling is exact in binary, so the matrices compare with ``==``
        # (``pytest.approx`` does not take nested lists).
        doubled = [[2 * v for v in row] for row in saved_now(p)['consequence']['oil_onboard']]

        def edit(dlg):
            for r, row in enumerate(doubled):
                for c, v in enumerate(row):
                    _spin(dlg.twOilOnboard, r, c).setValue(v)
        in_dialog(p, *menu, edit)

        assert saved_now(p)['consequence']['oil_onboard'] == doubled
        sent, calc = run_model(p)
        assert sent['consequence']['oil_onboard'] == doubled
        assert calc.consequence_result['mean_spill_volume_per_year'] == pytest.approx(2 * base, rel=1e-9)
        assert save_and_read_file(p)['consequence']['oil_onboard'] == doubled

        in_dialog(p, *menu, lambda dlg: _spin(dlg.twOilOnboard, 0, 0).setValue(1.0))
        load_saved_file(p)
        assert shown_in_dialog(p, *menu, lambda dlg: _matrix(dlg.twOilOnboard)) == doubled

    def test_spill_probability_fraction_and_levels(self, project):
        from compute.consequence import compute_catastrophe_exceedance
        from omrat_utils.consequence_defaults import ACCIDENT_TYPES
        p = project
        bend = ACCIDENT_TYPES.index('Bend collision')
        _sent, calc = run_model(p)
        report = calc.collision_report

        def exceedance(consequence: dict) -> dict:
            return compute_catastrophe_exceedance(consequence, collision_report=report)

        base = exceedance(saved_now(p)['consequence'])

        prob_row, frac_row = [94.0, 2.0, 2.0, 2.0], [0.0, 10.0, 30.0, 50.0]

        def edit_prob(dlg):
            for c, v in enumerate(prob_row):
                _spin(dlg.twSpillProbability, bend, c).setValue(v)

        def edit_frac(dlg):
            for c, v in enumerate(frac_row):
                _spin(dlg.twSpillFraction, bend, c).setValue(v)

        def edit_levels(dlg):
            dlg.pbAddLevel.click()                                  # a fourth level
            tw = dlg.twCatastropheLevels
            tw.item(tw.rowCount() - 1, 0).setText('Huge')
            _spin(tw, tw.rowCount() - 1, 1).setValue(20000.0)
            _spin(tw, 1, 1).setValue(400.0)                         # Major 500 -> 400

        in_dialog(p, 'Consequence', 'Spill probability per accident...', edit_prob)
        in_dialog(p, 'Consequence', 'Spill fraction per accident...', edit_frac)
        in_dialog(p, 'Consequence', 'Catastrophe levels...', edit_levels)

        def check(cons: dict) -> None:
            assert cons['spill_probability'][bend] == prob_row
            assert cons['spill_fraction'][bend] == frac_row
            levels = {lvl['name']: lvl['quantity'] for lvl in cons['catastrophe_levels']}
            assert levels['Major'] == 400.0 and levels['Huge'] == 20000.0

        check(saved_now(p)['consequence'])
        sent, _calc = run_model(p, execute=False)
        check(sent['consequence'])
        result = exceedance(sent['consequence'])
        assert [lvl['name'] for lvl in result['levels']][-1] == 'Huge'
        assert result['mean_spill_volume_per_year'] != pytest.approx(base['mean_spill_volume_per_year'])
        check(save_and_read_file(p)['consequence'])

        in_dialog(p, 'Consequence', 'Spill fraction per accident...',
                  lambda dlg: _spin(dlg.twSpillFraction, bend, 3).setValue(100.0))
        load_saved_file(p)
        shown = shown_in_dialog(p, 'Consequence', 'Spill fraction per accident...',
                                lambda dlg: _matrix(dlg.twSpillFraction)[bend])
        assert shown == frac_row

    def test_spill_probability_rows_must_sum_to_100(self, project):
        from omrat_utils.consequence_defaults import ACCIDENT_TYPES
        p = project
        bend = ACCIDENT_TYPES.index('Bend collision')
        before = saved_now(p)['consequence']['spill_probability']
        p.guard.expect(accept_with(lambda dlg: _spin(dlg.twSpillProbability, bend, 0).setValue(50.0)))
        menu_action(p.omrat, 'Consequence', 'Spill probability per accident...').trigger()
        settle()
        assert any('100' in m for m in p.guard.messages), 'OK with a bad row gave no warning'
        assert saved_now(p)['consequence']['spill_probability'] == before


# ---------------------------------------------------------------------------
# Dock fields: traffic scaling (Traffic tab), project name (Run Analysis tab)
# ---------------------------------------------------------------------------

class TestDockSettings:
    def test_global_traffic_scaling(self, project):
        p = project
        _sent, calc = run_model(p)
        base = collision_totals(calc)
        sb = p.omrat.main_widget.dsbGlobalTrafficScaling
        sb.setValue(130.0)
        sb.editingFinished.emit()           # Enter / leaving the field
        settle()

        def scaling_cells(traffic_data: dict) -> set[float]:
            return {float(v) for dirs in traffic_data.values() for block in dirs.values()
                    for row in block['Scaling (%)'] for v in row}

        saved = saved_now(p)
        assert saved['traffic_scaling']['global_percent'] == 130.0
        assert scaling_cells(saved['traffic_data']) == {130.0}
        sent, calc = run_model(p)
        assert scaling_cells(sent['traffic_data']) == {130.0}
        # Every encounter is a product of two frequencies: x 1.3^2.
        after = collision_totals(calc)
        for kind in COLLISION_TYPES:
            assert after[kind] == pytest.approx(1.69 * base[kind], rel=1e-9), kind
        written = save_and_read_file(p)
        assert written['traffic_scaling']['global_percent'] == 130.0

        sb.setValue(100.0)
        sb.editingFinished.emit()
        load_saved_file(p)
        assert p.omrat.main_widget.dsbGlobalTrafficScaling.value() == 130.0
        assert scaling_cells(saved_now(p)['traffic_data']) == {130.0}

    def test_project_name(self, project):
        p = project
        le = p.omrat.main_widget.LEProjectName
        le.setText('Round trip')
        assert saved_now(p)['project_name'] == 'Round trip'
        sent, _calc = run_model(p, execute=False)
        assert sent['project_name'] == 'Round trip'
        assert p.omrat._run_input_snapshot['project_name'] == 'Round trip'   # tags the run in Previous runs
        assert save_and_read_file(p)['project_name'] == 'Round trip'
        le.setText('Something else')
        load_saved_file(p)
        assert le.text() == 'Round trip'
