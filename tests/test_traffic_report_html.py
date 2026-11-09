"""Pure tests for ``omrat_utils.traffic_report_html`` (the Traffic report page)."""
from __future__ import annotations

import copy
import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.traffic_report_html import render_traffic_report_html  # noqa: E402
from omrat_utils.traffic_summary import build_traffic_summary, legs_csv, types_csv  # noqa: E402

VOID = {'meta', 'input', 'br', 'img', 'line', 'circle', 'polygon', 'link'}


class _Collector(HTMLParser):
    """Ids, tags and a check that every non-void tag is closed in order."""

    def __init__(self):
        super().__init__()
        self.ids: list[str] = []
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.tags: dict[str, int] = {}
        self.links: list[dict] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags[tag] = self.tags.get(tag, 0) + 1
        if 'id' in a:
            self.ids.append(a['id'])
        if tag == 'a':
            self.links.append(a)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")
            return
        self.stack.pop()


def _parse(page: str) -> _Collector:
    c = _Collector()
    c.feed(page)
    c.close()
    return c


@pytest.fixture
def data():
    return json.loads((ROOT / 'tests' / 'example_data' / 'proj.omrat').read_text(encoding='utf-8'))


def _scenario(data):
    """Leg 3 suppressed onto leg 4, leg 2 named with HTML characters."""
    data = copy.deepcopy(data)
    data['segment_data']['3']['suppressed'] = True
    data['segment_data']['3']['traffic_redirect'] = [{'from_dir': 0, 'leg': '4', 'dir': 0, 'share': 100}]
    data['segment_data']['2']['Leg_name'] = 'LEG_<b>2</b> & co'
    return data


def test_page_is_well_formed_and_self_contained(data):
    page = render_traffic_report_html(build_traffic_summary(data))
    parsed = _parse(page)
    assert parsed.errors == [] and parsed.stack == []
    assert page.startswith('<!DOCTYPE html>')
    for bad in ('<link', 'src="http', "src='http", '@import', 'href="http'):
        assert bad not in page
    for section in ('figures', 'checks', 'map', 'legs', 'types', 'details', 'about', 'map-svg', 'tip'):
        assert section in parsed.ids


def test_every_leg_has_details_and_table_row(data):
    summary = build_traffic_summary(data)
    parsed = _parse(render_traffic_report_html(summary))
    for leg in summary['legs']:
        assert f"leg-{leg['id']}" in parsed.ids
    hrefs = [a.get('href') for a in parsed.links]
    assert hrefs.count('#leg-1') >= 3           # map band(s), leg table, ship-type table


def test_names_are_escaped(data):
    page = render_traffic_report_html(build_traffic_summary(_scenario(data)))
    assert '<b>2</b>' not in page
    assert 'LEG_&lt;b&gt;2&lt;/b&gt; &amp; co' in page
    assert _parse(page).errors == []


def test_map_bands_follow_the_data(data):
    summary = build_traffic_summary(data)
    page = render_traffic_report_html(summary)
    n_dirs = sum(1 for leg in summary['legs'] for d in leg['dirs'] if d['ships'] > 0)
    assert page.count('class="band k-') == 2 * n_dirs     # as stored + in the calculation
    assert 'type="radio" name="map-view"' not in page                  # nothing differs -> no toggle


def test_calculation_toggle_and_suppressed_leg(data):
    summary = build_traffic_summary(_scenario(data))
    page = render_traffic_report_html(summary)
    assert 'type="radio" name="map-view"' in page
    assert 'class="band k-received' in page                # leg 4 gets ships from leg 3
    assert 'leg-line suppressed v-calc' in page
    assert 'In the calculation</th>' in page
    assert '>suppressed</td>' in page


def test_csv_links_hold_the_tables(data):
    summary = build_traffic_summary(data)
    page = render_traffic_report_html(summary, decimal=',', csv_stem='proj')
    parsed = _parse(page)
    downloads = {a['download']: a['href'] for a in parsed.links if 'download' in a}
    assert set(downloads) == {'proj_legs.csv', 'proj_ship_types.csv'}
    prefix = 'data:text/csv;charset=utf-8,'
    legs = unquote(downloads['proj_legs.csv'][len(prefix):])
    assert legs == '﻿' + legs_csv(summary, decimal=',')
    types = unquote(downloads['proj_ship_types.csv'][len(prefix):])
    assert types == '﻿' + types_csv(summary, decimal=',')


def test_checks_section_lists_findings_or_ok(data):
    summary = build_traffic_summary(data)
    page = render_traffic_report_html(summary)
    assert 'meet at a waypoint with no other leg' in page
    summary['checks'] = []
    for leg in summary['legs']:
        leg['checks'] = []
    assert 'No problems found.' in render_traffic_report_html(summary)


def test_legs_without_coordinates():
    data = {'segment_data': {'1': {'Leg_name': 'LEG_1', 'Dirs': ['North going', 'South going']}},
            'traffic_data': {}}
    page = render_traffic_report_html(build_traffic_summary(data))
    assert 'The legs have no coordinates.' in page
    assert _parse(page).errors == []


def test_many_legs_hide_names_by_default(data):
    data = copy.deepcopy(data)
    base = data['segment_data']['1']
    for i in range(30):
        seg = str(100 + i)
        data['segment_data'][seg] = dict(copy.deepcopy(base), Segment_Id=seg, Leg_name=f'LEG_X_{i}',
                                         Start_Point=f'{13 + i * 0.1:.6f} 56.000000',
                                         End_Point=f'{13 + i * 0.1 + 0.05:.6f} 56.000000')
        data['traffic_data'][seg] = copy.deepcopy(data['traffic_data']['1'])
    page = render_traffic_report_html(build_traffic_summary(data))
    assert 'class="map nolabels"' in page
    assert 'id="map-names">' in page                       # unticked


def test_check_badges_tell_what_they_are_about_on_hover(data):
    """The leg table / leg details badges carry the checks' own text (shown on hover)."""
    import html as html_mod
    import re
    data = _scenario(data)
    data['segment_data']['1']['traffic_locked'] = True        # a note on leg 1 as well
    summary = build_traffic_summary(data)
    page = render_traffic_report_html(summary)
    tips = [html_mod.unescape(t) for t in re.findall(r'<span class="badge \w+"[^>]*data-tip="([^"]*)"', page)]
    checks = summary['checks']
    for leg in summary['legs']:
        for i in leg['checks']:
            # once in the leg table, once in the leg's details
            assert sum(checks[i]['text'] in t for t in tips) >= 2, checks[i]['text']
    several = [t for t in tips if '\n• ' in t]
    single = [t for t in tips if t.startswith(('Warning: ', 'Note: '))]
    assert single, tips
    if several:
        assert re.match(r'\d+ (warnings|notes):\n• ', several[0])
    assert 'tabindex="0" data-tip=' in page                   # keyboard users get it too
    assert _parse(page).errors == []
