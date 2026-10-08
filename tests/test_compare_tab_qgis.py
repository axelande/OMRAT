"""QGIS test: the Compare tab fills every sub-tab from two runs on disk.

Two snapshots of the example project plus a collision sidecar each are
written to a temp folder; clicking **Compare** must fill the Totals,
Settings & inputs, Breakdown and Zone tables, the category filter must
hide rows, and changing the zone radius must recompute the Zone table.
Runs with the QGIS conftest (``omrat`` fixture).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parent / 'example_data' / 'proj.omrat'


def _collision(head_on_1: float, crossing: float) -> dict:
    return {
        'totals': {'head_on': head_on_1, 'overtaking': 0.0, 'crossing': crossing,
                   'merging': 0.0, 'bend': 0.0},
        'by_leg': {'1': {'head_on': head_on_1, 'overtaking': 0.0, 'bend': 0.0}},
        'by_waypoint': {'14.158810 55.187230': {'crossing': crossing, 'merging': 0.0, 'bend': 0.0}},
        'by_leg_pair': {},
    }


def _write_run(folder: Path, stem: str, snap: dict, collision: dict) -> Path:
    path = folder / f'{stem}.omrat'
    path.write_text(json.dumps(snap), encoding='utf-8')
    (folder / f'{stem}.collision.json').write_text(json.dumps(collision), encoding='utf-8')
    return path


def _column(tw, col: int) -> list[str]:
    return [tw.item(r, col).text() for r in range(tw.rowCount()) if tw.item(r, col) is not None]


def test_compare_fills_all_sub_tabs(omrat, tmp_path):
    w = omrat.main_widget
    with EXAMPLE.open('r', encoding='utf-8') as f:
        snap_a = json.load(f)
    snap_b = copy.deepcopy(snap_a)
    snap_b['segment_data']['1']['std1_1'] = 450.0
    path_a = _write_run(tmp_path, 'runA', snap_a, _collision(1e-3, 2e-3))
    path_b = _write_run(tmp_path, 'runB', snap_b, _collision(3e-3, 2e-3))
    w.LECompareA.setText(str(path_a))
    w.LECompareB.setText(str(path_b))
    w.pbRunCompare.click()

    accidents = _column(w.TWCompareAccidents, 0)
    assert 'Head-on collision' in accidents and 'All collisions' in accidents
    assert 'Lateral distribution' in _column(w.TWCompareInputs, 0)

    tw = w.TWCompareBreakdown
    assert tw.rowCount() >= 2
    assert w.lblCompareSummary.text() == 'Run A: runA.omrat   |   Run B: runB.omrat'
    assert 'Largest single change: LEG_1_1 (1) – head-on' in w.lblCompareDetails.text()
    assert 'Result files: A collision; B collision.' in w.lblCompareDetails.text()
    w.cbCompareBreakdownFilter.setCurrentText('Collisions per waypoint')
    visible = [r for r in range(tw.rowCount()) if not tw.isRowHidden(r)]
    assert visible and all(tw.item(r, 0).text() == 'Collisions per waypoint' for r in visible)
    w.cbCompareBreakdownFilter.setCurrentIndex(0)
    assert not any(tw.isRowHidden(r) for r in range(tw.rowCount()))

    assert w.lblCompareZoneInfo.text().startswith('Zone: within 20 km of 2 structure(s)')
    km_20 = w.TWCompareZone.item(0, 1).text()
    w.sbCompareZoneKm.setValue(2.0)
    assert w.lblCompareZoneInfo.text().startswith('Zone: within 2 km')
    assert w.TWCompareZone.item(0, 1).text() != km_20
