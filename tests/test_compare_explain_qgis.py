"""QGIS test: Compare tab -> **More info (collisions)...**.

The dialog runs the explanation as a task (here started synchronously),
fills its three sections per accident type, merges split legs, draws the
selected legs on the canvas and exports Markdown; the Compare-tab button
opens it for the picked snapshots, preselects the row's accident type and
reuses the cached explanation.  Runs with the QGIS conftest.
"""
from __future__ import annotations

import copy
import json

from omrat_utils import compare_explain as ce
from omrat_utils.compare_explain_dialog import CompareExplainDialog
from tests.test_compare_explain import _project


def _sync(task) -> None:
    task.finished(task.run())


def _scenario() -> tuple[dict, dict]:
    a = _project()
    b = copy.deepcopy(a)
    b['segment_data']['4']['suppressed'] = True
    b['segment_data']['4']['traffic_redirect'] = [
        {'from_dir': 0, 'leg': '3', 'dir': 0, 'share': 100.0},
        {'from_dir': 1, 'leg': '3', 'dir': 1, 'share': 100.0},
    ]
    b['segment_data']['1']['std1_1'] = 200.0
    return a, b


def _column(tw, col: int) -> list[str]:
    return [tw.item(r, col).text() for r in range(tw.rowCount()) if tw.item(r, col) is not None]


def test_dialog_runs_and_fills_sections(qgis_iface, tmp_path):
    a, b = _scenario()
    ready = []
    dlg = CompareExplainDialog(None, a, b, 'a.omrat', 'b.omrat', acc_type='crossing',
                               canvas=qgis_iface.mapCanvas(), on_ready=ready.append, task_starter=_sync)
    assert dlg.explanation is not None and ready == [dlg.explanation]
    assert dlg.current_type() == 'crossing'
    assert _column(dlg.twGroups, 0) == [ce.GROUP_LABELS['routing'], ce.GROUP_LABELS['lateral']]
    assert 'Run A' in dlg.lblSummary.text() and 'Run B' in dlg.lblSummary.text()
    assert 'Identical in A and B' in dlg.lblIdentical.text()
    assert dlg.twItems.rowCount() > 0
    assert 'Routing differs' in dlg.tbFindings.toPlainText()

    dlg.cbType.setCurrentIndex(dlg.cbType.findData('head_on'))
    assert 'LEG_1_1' in _column(dlg.twItems, 0)
    assert 'Lateral distributions differ' in dlg.tbFindings.toPlainText()

    dlg.cbType.setCurrentIndex(dlg.cbType.findData(ce.TOTAL))
    headers = [dlg.twItems.horizontalHeaderItem(c).text() for c in range(dlg.twItems.columnCount())]
    assert headers[:5] == ['Item', 'Type', 'Run A', 'Run B', 'Change']

    dlg.twItems.selectRow(0)
    assert dlg.show_selected_on_map() >= 1
    assert dlg._bands
    dlg.clear_map()
    assert not dlg._bands

    out = dlg.export_markdown(str(tmp_path / 'why.md'))
    assert out is not None and out.read_text(encoding='utf-8').startswith('# Why did the collisions change?')
    dlg.close()


def test_merge_split_checkbox(qgis_iface):
    a = _project()
    b = copy.deepcopy(a)
    for seg, name in (('1', 'LEG_1_1_a'), ('2', 'LEG_1_1_b')):
        a['segment_data'][seg]['Leg_name'] = name
        b['segment_data'][seg]['Leg_name'] = name
    b['segment_data']['1']['std1_1'] = 300.0
    b['segment_data']['2']['std1_1'] = 900.0
    dlg = CompareExplainDialog(None, a, b, 'a', 'b', acc_type='head_on', task_starter=_sync)
    assert {'LEG_1_1_a', 'LEG_1_1_b'} <= set(_column(dlg.twItems, 0))
    dlg.cbMergeSplit.setChecked(True)
    assert 'LEG_1_1 (2 sub-legs)' in _column(dlg.twItems, 0)
    dlg.close()


def test_compare_tab_button_opens_cached_explanation(omrat, tmp_path):
    w = omrat.main_widget
    a, b = _scenario()
    path_a, path_b = tmp_path / 'runA.omrat', tmp_path / 'runB.omrat'
    path_a.write_text(json.dumps(a), encoding='utf-8')
    path_b.write_text(json.dumps(b), encoding='utf-8')
    expl = ce.explain_collisions(a, b)
    for stem, totals in (('runA', expl.totals_a), ('runB', expl.totals_b)):
        (tmp_path / f'{stem}.collision.json').write_text(
            json.dumps({'totals': {t: totals[t] for t in ce.TYPES}}), encoding='utf-8')
    w.LECompareA.setText(str(path_a))
    w.LECompareB.setText(str(path_b))
    w.pbRunCompare.click()

    key = omrat._compare_explain_key(str(path_a), str(path_b))
    omrat._remember_compare_explanation(key, expl)
    rows = _column(w.TWCompareAccidents, 0)
    w.TWCompareAccidents.setCurrentCell(rows.index('Merging collision'), 0)
    w.pbCompareMoreInfo.click()
    dlg = omrat._compare_explain_dlg
    assert dlg is not None and dlg.explanation is expl
    assert dlg.current_type() == 'merging'

    # Same pair again: the open dialog is raised and switched, not rebuilt.
    w.TWCompareAccidents.setCurrentCell(rows.index('Head-on collision'), 0)
    w.pbCompareMoreInfo.click()
    assert omrat._compare_explain_dlg is dlg and dlg.current_type() == 'head_on'
    dlg.close()
