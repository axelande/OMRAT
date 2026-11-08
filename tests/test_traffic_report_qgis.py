"""QGIS tests for **Traffic report...**: the button, the File menu entry,
writing the report from the live model and the *Traffic report* button in
the messages after Update AIS, Copy traffic and Suppress leg.  Needs the
QGIS conftest."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest


class _Notifier:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def display_message(self, message, **kwargs):
        self.calls.append((message, kwargs))
        return 'id'


@pytest.fixture
def legs(omrat):
    """Leg 91 with ships both ways, leg 92 continuing it with none."""
    om = omrat
    om.testing = True
    base = {'Width': 2000, 'Route_Id': 1, 'Tangent_Pos': 0.5, 'Dirs': ['North going', 'South going']}
    om.segment_data['91'] = dict(copy.deepcopy(base), Segment_Id='91', Leg_name='LEG_1_91',
                                 Start_Point='14.000000 55.000000', End_Point='14.000000 55.200000')
    om.segment_data['92'] = dict(copy.deepcopy(base), Segment_Id='92', Leg_name='LEG_1_92',
                                 Start_Point='14.000000 55.200000', End_Point='14.000000 55.400000')
    om.traffic.create_empty_dict('91', ['North going', 'South going'])
    om.traffic.create_empty_dict('92', ['North going', 'South going'])
    om.traffic_data['91']['North going']['Frequency (ships/year)'][0][0] = 50
    om.traffic_data['91']['South going']['Frequency (ships/year)'][1][1] = 30
    yield om
    for k in ('91', '92'):
        om.segment_data.pop(k, None)
        om.traffic_data.pop(k, None)


def test_button_and_file_menu_open_the_report(omrat, monkeypatch):
    from qgis.PyQt.QtWidgets import QMenuBar
    calls = []
    monkeypatch.setattr(omrat, 'open_traffic_report', lambda: calls.append('open'))
    omrat.main_widget.pbTrafficReport.click()
    assert calls == ['open']
    menubar = omrat.main_widget.findChild(QMenuBar)
    file_menu = next(a.menu() for a in menubar.actions() if a.text() == 'File')
    assert 'Traffic report...' in [a.text() for a in file_menu.actions()]


def test_write_report_from_live_model(legs, tmp_path):
    from omrat_utils.traffic_report_dialog import write_traffic_report
    out = write_traffic_report(legs, str(tmp_path / 'report.html'))
    page = (tmp_path / 'report.html').read_text(encoding='utf-8')
    assert out == str(tmp_path / 'report.html')
    assert 'LEG_1_91' in page and 'LEG_1_92' in page
    assert 'id="leg-91"' in page
    assert 'LEG_1_92 has no ships.' in page


def test_open_with_a_path_writes_and_does_not_open_a_browser_in_tests(legs, tmp_path, monkeypatch):
    from omrat_utils import traffic_report_dialog as trd
    opened = []
    monkeypatch.setattr(trd.QDesktopServices, 'openUrl', lambda url: opened.append(url))
    legs.notifier = _Notifier()
    path = trd.open_traffic_report(legs, str(tmp_path / 'r.html'))
    assert path == str(tmp_path / 'r.html')
    assert (tmp_path / 'r.html').exists()
    assert opened == []
    assert 'Traffic report saved to' in legs.notifier.calls[-1][0]


def test_open_asks_for_a_file_name(legs, tmp_path, monkeypatch):
    from omrat_utils import traffic_report_dialog as trd
    asked = []

    def fake_dialog(parent, title, start, filt):
        asked.append(start)
        return str(tmp_path / 'chosen'), filt
    monkeypatch.setattr(trd.QFileDialog, 'getSaveFileName', fake_dialog)
    legs.notifier = _Notifier()
    assert trd.open_traffic_report(legs) == str(tmp_path / 'chosen.html')
    assert '_traffic_report_' in asked[0] and asked[0].endswith('.html')
    monkeypatch.setattr(trd.QFileDialog, 'getSaveFileName', lambda *a: ('', ''))
    assert trd.open_traffic_report(legs) is None


def test_default_path_uses_the_project_file(omrat, tmp_path, monkeypatch):
    from datetime import datetime
    from omrat_utils.traffic_report_dialog import default_report_path
    monkeypatch.setattr(omrat, 'project_path', str(tmp_path / 'kattegat.omrat'))
    path = default_report_path(omrat, now=datetime(2026, 11, 8, 9, 30))
    assert path == str(tmp_path / 'kattegat_traffic_report_20261108_0930.html')


def test_no_legs_says_so(omrat, monkeypatch):
    from omrat_utils import traffic_report_dialog as trd
    shown = []
    monkeypatch.setattr(trd.QMessageBox, 'information', lambda *a: shown.append(a[2]))
    monkeypatch.setattr(omrat, 'segment_data', {})
    assert trd.open_traffic_report(omrat) is None
    assert 'no legs' in shown[0]


def test_message_button_opens_the_report(omrat, monkeypatch):
    from qgis.PyQt.QtWidgets import QPushButton
    from omrat_utils import traffic_report_dialog as trd
    opened = []
    monkeypatch.setattr(trd, 'open_traffic_report', lambda om, path=None: opened.append(om))
    omrat.notifier = _Notifier()
    trd.notify_with_report(omrat, 'Copied traffic', duration=10)
    message, kwargs = omrat.notifier.calls[-1]
    assert message == 'Copied traffic' and kwargs['duration'] == 10
    (btn,) = kwargs['widgets']
    assert isinstance(btn, QPushButton) and btn.text() == 'Traffic report'
    btn.click()
    assert opened == [omrat]


def test_message_without_widget_support_falls_back(omrat):
    from omrat_utils.traffic_report_dialog import notify_with_report

    class OldNotifier(_Notifier):
        def display_message(self, message, duration=0):
            self.calls.append((message, {'duration': duration}))
    omrat.notifier = OldNotifier()
    notify_with_report(omrat, 'Leg suppressed')
    assert omrat.notifier.calls == [('Leg suppressed', {'duration': 10})]


def test_ais_update_and_suppress_messages_carry_the_button(legs):
    from omrat_utils.ais_update_task import AisUpdateTask
    from omrat_utils.suppress_leg_dialog import _SuppressDialog
    legs.notifier = _Notifier()
    AisUpdateTask._notify_done(SimpleNamespace(results={'91': {}, 'gone': {}}, bulk=True), legs)
    message, kwargs = legs.notifier.calls[-1]
    assert message == 'Update all distributions finished: new traffic on 1 leg(s).'
    assert kwargs['widgets'][0].text() == 'Traffic report'
    AisUpdateTask._notify_done(SimpleNamespace(results={}, bulk=False), legs)
    assert len(legs.notifier.calls) == 1                   # nothing updated -> no message
    _SuppressDialog._notify(SimpleNamespace(omrat=legs), 'Leg LEG_1_91 suppressed')
    assert legs.notifier.calls[-1][0] == 'Leg LEG_1_91 suppressed'
    assert 'widgets' in legs.notifier.calls[-1][1]
