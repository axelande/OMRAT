"""Screenshot of the **Suppress leg** dialog for the user guide.

Builds the real dialog (``omrat_utils.suppress_leg_dialog._SuppressDialog``)
on a small stand-in project that matches Example 3 of the guide (route 7
moved onto route 2), fills it in exactly as the text describes and saves

    help/source/_static/screenshots/ui_suppress_leg_dialog.png

Run with the OSGeo4W Python from the repository root::

    C:/OSGeo4W/apps/Python312/python.exe help/figures_src/make_suppress_dialog_screenshot.py

The window is shown for a moment so ``grab()`` renders it like on screen.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qgis.core import QgsApplication  # noqa: E402

OUT = ROOT / 'help' / 'source' / '_static' / 'screenshots' / 'ui_suppress_leg_dialog.png'


def _leg(seg_id, name, start, end):
    return {'Segment_Id': seg_id, 'Leg_name': name, 'Start_Point': start, 'End_Point': end,
            'Dirs': ['East going', 'West going'], 'Width': 5000, 'Route_Id': 1}


def _traffic(east: float, west: float) -> dict:
    def block(q):
        return {'Frequency (ships/year)': [[q]], 'Speed (knots)': [[12.0]]}
    return {'East going': block(east), 'West going': block(west)}


def build_project():
    # Route 7 (to be suppressed) above route 2 (the detour), both drawn
    # west -> east.  Ids / names follow Example 3 of the user guide.
    route7 = [('28', 'LEG_7_3_c'), ('27', 'LEG_7_3_b'), ('26', 'LEG_7_3_a'), ('25', 'LEG_7_2')]
    route2 = [('4', 'LEG_2_3_a'), ('23', 'LEG_2_3_c'), ('16', 'LEG_2_3_b'), ('6', 'LEG_2_6')]
    segs, traffic = {}, {}
    for i, (sid, name) in enumerate(route7):
        segs[sid] = _leg(sid, name, f'{11.0 + 0.1 * i:.6f} 57.100000', f'{11.1 + 0.1 * i:.6f} 57.110000')
        traffic[sid] = _traffic(444, 497)
    for i, (sid, name) in enumerate(route2):
        segs[sid] = _leg(sid, name, f'{11.0 + 0.1 * i:.6f} 57.000000', f'{11.1 + 0.1 * i:.6f} 57.010000')
        traffic[sid] = _traffic(3500, 3600)
    segs['16'].update({'traffic_locked': True, 'traffic_source': '4'})
    segs['23'].update({'traffic_locked': True, 'traffic_source': '4'})
    return segs, traffic


def main() -> None:
    app = QgsApplication([], True)
    app.initQgis()
    from qgis.PyQt.QtWidgets import QApplication
    from omrat_utils.suppress_leg_dialog import _SuppressDialog

    segs, traffic = build_project()
    omrat = SimpleNamespace(main_widget=None, segment_data=segs, traffic_data=traffic,
                            tr=lambda text: text, qgis_geoms=None, notifier=None, testing=True)
    dlg = _SuppressDialog(omrat)
    dlg.fill_sources(select='27')                        # LEG_7_3_b, the lead
    for leg in ('28', '26', '25'):
        dlg.set_member(leg)
    for d in (0, 1):
        for target in ('6', '16', '23', '4'):
            row = dlg.add_row()
            dlg.table.cellWidget(row, 0).setCurrentIndex(d)
            cb_leg = dlg.table.cellWidget(row, 1)
            cb_leg.setCurrentIndex(cb_leg.findData(target))
            dlg.table.cellWidget(row, 3).setValue(100.0)
    dlg.resize(840, 800)
    dlg.show()
    for _ in range(20):
        QApplication.processEvents()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dlg.grab().save(str(OUT), 'PNG')
    print('wrote', OUT)
    dlg.close()
    app.exitQgis()


if __name__ == '__main__':
    main()
