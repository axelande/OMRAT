"""Screenshot of the **Waypoint editor** for the user guide.

Builds the real dialog (``omrat_utils.waypoint_editor_dialog``) on a
small stand-in project -- three legs meeting at one waypoint, a fourth
leg nearby -- selects the junction and the leg to detach, and saves

    help/source/_static/screenshots/ui_waypoint_editor.png

Run with the OSGeo4W Python from the repository root::

    C:/OSGeo4W/apps/Python312/python.exe help/figures_src/make_waypoint_editor_screenshot.py

The window is shown for a moment so ``grab()`` renders it like on screen.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qgis.core import QgsApplication  # noqa: E402

OUT = ROOT / 'help' / 'source' / '_static' / 'screenshots' / 'ui_waypoint_editor.png'


def build_project() -> dict:
    from geometries.route_validation import haversine_m
    pts = {
        'A': (11.000000, 57.000000), 'B': (11.100000, 57.050000), 'C': (11.250000, 57.000000),
        'D': (11.100000, 56.900000), 'E': (11.300000, 57.120000),
    }
    legs = {'1': ('LEG_1_1', 'A', 'B'), '2': ('LEG_1_2', 'B', 'C'),
            '3': ('LEG_2_1', 'B', 'D'), '4': ('LEG_3_1', 'C', 'E')}
    segs = {}
    for sid, (name, s, e) in legs.items():
        segs[sid] = {
            'Segment_Id': sid, 'Leg_name': name, 'Route_Id': int(name.split('_')[1]),
            'Start_Point': f'{pts[s][0]:.6f} {pts[s][1]:.6f}', 'End_Point': f'{pts[e][0]:.6f} {pts[e][1]:.6f}',
            'line_length': haversine_m(pts[s], pts[e]), 'Dirs': ['East going', 'West going'], 'Width': 5000,
        }
    return segs


def main() -> None:
    app = QgsApplication([], True)
    app.initQgis()
    from qgis.PyQt.QtWidgets import QApplication
    from geometries.waypoints import find_waypoint_at
    from omrat_utils.handle_junctions import Junctions
    from omrat_utils.waypoint_editor_dialog import WaypointEditorDialog

    omrat = SimpleNamespace(main_widget=None, segment_data=build_project(), waypoints={},
                            tr=lambda text: text, qgis_geoms=None, notifier=None, iface=None, testing=True)
    omrat.junctions = Junctions(omrat)
    dlg = WaypointEditorDialog(omrat)
    dlg.select_waypoint(find_waypoint_at(omrat.waypoints, (11.1, 57.05)), zoom=False)
    for row in range(dlg.lst_legs.count()):
        if 'LEG_1_2' in dlg.lst_legs.item(row).text():
            dlg.lst_legs.setCurrentRow(row)
    dlg.resize(820, 520)
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
