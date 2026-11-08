"""Render a :func:`omrat_utils.traffic_summary.build_traffic_summary` as one HTML page.

Pure (no Qt / QGIS) and self-contained: inline CSS, a little inline
JavaScript (sorting, filter, map toggles, tooltips), the map as inline SVG
and the CSV tables as ``data:`` links -- so the file opens in any browser,
offline, and can be mailed or attached as it is.  Print / *Save as PDF*
opens every leg's details first.

Colours follow the report's own small palette (defined once as CSS custom
properties, light and dark): own data blue, copied traffic aqua (OMRAT's
Traffic links layer draws copies green), ships moved from a suppressed leg
orange (orange in Traffic links too), suppressed legs grey and dashed, a leg
without ships a dashed red line.  Magnitudes use one blue ramp.
"""
from __future__ import annotations

import html
import math
from typing import Any
from urllib.parse import quote

from omrat_utils.traffic_summary import CHECK_TEXT, fmt_ships, legs_csv, short_type, types_csv

__all__ = ['render_traffic_report_html']

_WARN_ICON = '⚠'
_NOTE_ICON = 'ℹ'
_OK_ICON = '✓'

KIND_LABEL = {
    'own': 'Own data',
    'copy': 'Copied traffic',
    'received': 'Gets ships from a suppressed leg',
    'suppressed': 'Suppressed (not in the calculation)',
    'none': 'No ships',
}

# Sequential blue ramp, light -> dark (steps 100 ... 700).
_RAMP = ('#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
         '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b')
_RAMP_WHITE_FROM = 7          # white text from step 450 on

MAP_W = 960
MAP_PAD = 48
BAND_MAX_PX = 18.0
BAND_MIN_PX = 1.5
LABEL_DEFAULT_MAX_LEGS = 25


def _e(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _num1(value: float | None, unit: str = '') -> str:
    if value is None:
        return '-'
    return f"{value:.1f}{unit}"


def _heat_style(value: float, vmax: float) -> str:
    if value <= 0 or vmax <= 0:
        return ''
    idx = min(len(_RAMP) - 1, max(0, int(math.ceil(value / vmax * len(_RAMP))) - 1))
    ink = '#ffffff' if idx >= _RAMP_WHITE_FROM else '#0b0b0b'
    return f' style="background:{_RAMP[idx]};color:{ink}"'


def _csv_href(text: str) -> str:
    return 'data:text/csv;charset=utf-8,' + quote('﻿' + text, safe='')


def _leg_link(leg: dict[str, Any]) -> str:
    return f'<a href="#leg-{_e(leg["id"])}">{_e(leg["name"])}</a>'


# ---------------------------------------------------------------------------
# Page pieces
# ---------------------------------------------------------------------------

_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --own: #2a78d6; --copy: #1baf7a; --received: #eb6834; --suppressed: #898781;
  --warning: #fab219; --critical: #d03b3b; --good: #0ca30c; --bar: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --own: #3987e5; --copy: #199e70; --received: #d95926; --bar: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --own: #3987e5; --copy: #199e70; --received: #d95926; --bar: #3987e5;
}
* { box-sizing: border-box; }
html { background: var(--page); }
body { margin: 0 auto; max-width: 1200px; padding: 24px 16px 64px; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 26px; margin: 0 0 4px; font-weight: 600; }
h2 { font-size: 18px; margin: 0 0 12px; font-weight: 600; }
h3 { font-size: 15px; margin: 16px 0 8px; font-weight: 600; }
a { color: var(--own); }
td a, .chips a, .tile a { text-decoration: none; }
td a:hover, .chips a:hover, .tile a:hover { text-decoration: underline; text-underline-offset: 0.3em; }
.meta { color: var(--ink-2); margin: 0; }
.meta span + span::before { content: " \\00b7 "; color: var(--muted); }
.lead { color: var(--ink-2); margin: 8px 0 0; max-width: 75ch; }
nav.toc { display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 16px 0 0; align-items: center; }
nav.toc a { text-decoration: none; }
section { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px;
  margin-top: 16px; }
button, .btn { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--axis);
  border-radius: 6px; padding: 3px 10px; cursor: pointer; text-decoration: none; display: inline-block; }
button:hover, .btn:hover { background: var(--grid); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.tile { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.tile .label { color: var(--ink-2); font-size: 13px; }
.tile .value { font-size: 24px; font-weight: 600; margin-top: 2px; overflow-wrap: anywhere; }
.tile .sub { color: var(--ink-2); font-size: 12px; }
ul.checks { list-style: none; margin: 0; padding: 0; }
ul.checks li { padding: 8px 0; border-top: 1px solid var(--grid); }
ul.checks li:first-child { border-top: 0; }
.badge { display: inline-block; font-size: 12px; font-weight: 600; border-radius: 4px; padding: 0 6px;
  margin-right: 6px; white-space: nowrap; border: 1px solid var(--border); }
.badge.warning::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: var(--warning); margin-right: 4px; }
.badge.note::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: var(--muted); margin-right: 4px; }
.badge.good::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: var(--good); margin-right: 4px; }
.chips { margin-top: 4px; font-size: 13px; }
.chips a { margin-right: 8px; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; margin-bottom: 8px; }
.toolbar label { white-space: nowrap; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--ink-2); margin-top: 8px; }
.legend .key { display: inline-block; width: 22px; height: 8px; border-radius: 2px; vertical-align: middle;
  margin-right: 6px; }
.legend .key.dashed { height: 0; border-top: 2px dashed var(--suppressed); border-radius: 0; }
.legend .key.none { height: 0; border-top: 2px dashed var(--critical); border-radius: 0; }
.mapwrap { width: 100%; border: 1px solid var(--grid); border-radius: 6px; background: var(--surface); }
svg.map { display: block; width: 100%; height: auto; }
svg.map .leg-line { stroke: var(--axis); stroke-width: 1; }
svg.map .leg-line.suppressed { stroke: var(--suppressed); stroke-dasharray: 6 4; stroke-width: 1.5; }
svg.map .leg-line.none { stroke: var(--critical); stroke-dasharray: 5 4; stroke-width: 1.5; }
svg.map .band { stroke-linecap: butt; }
svg.map .k-own { stroke: var(--own); }
svg.map .k-copy { stroke: var(--copy); }
svg.map .k-received { stroke: var(--received); }
svg.map .k-suppressed { stroke: var(--suppressed); }
svg.map .hit { stroke: transparent; stroke-linecap: butt; pointer-events: stroke; cursor: pointer; }
svg.map .arrow { fill: var(--surface); pointer-events: none; }
svg.map .label { font-size: 11px; fill: var(--ink-2); paint-order: stroke; stroke: var(--surface);
  stroke-width: 3px; stroke-linejoin: round; pointer-events: none; }
svg.map .wp { fill: var(--surface); stroke: var(--muted); stroke-width: 1; }
svg.map .chrome { fill: var(--ink-2); font-size: 11px; }
svg.map .chrome-line { stroke: var(--ink-2); stroke-width: 1.5; }
svg.map[data-view="stored"] .v-calc, svg.map[data-view="calc"] .v-stored { display: none; }
svg.map.nolabels .label { display: none; }
svg.map a:focus-visible .hit { stroke: var(--ink); stroke-opacity: 0.25; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 8px; border-bottom: 1px solid var(--grid); text-align: left; vertical-align: top; }
th { font-weight: 600; color: var(--ink-2); font-size: 13px; white-space: nowrap; background: var(--surface);
  position: sticky; top: 0; }
td.n, th.n { text-align: right; white-space: nowrap; }
table.sortable th { cursor: pointer; user-select: none; }
table.sortable th[aria-sort="ascending"]::after { content: " \\25b2"; font-size: 10px; }
table.sortable th[aria-sort="descending"]::after { content: " \\25bc"; font-size: 10px; }
tr.suppressed td { color: var(--muted); }
td.legcell { white-space: nowrap; }
td.src { min-width: 170px; max-width: 240px; }
.tt { display: block; white-space: nowrap; }
table.heat td:first-child, table.heat th:first-child { position: sticky; left: 0; background: var(--surface);
  z-index: 1; }
table.heat th:first-child { z-index: 2; }
.dirlabel { color: var(--ink-2); font-size: 12px; display: block; }
.barcell { min-width: 150px; }
.bar { display: inline-block; height: 8px; background: var(--bar); border-radius: 0 4px 4px 0;
  vertical-align: middle; margin-right: 6px; }
.kind { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.kind.own { background: var(--own); } .kind.copy { background: var(--copy); }
.kind.received { background: var(--received); } .kind.suppressed { background: var(--suppressed); }
.kind.none { background: transparent; border: 1.5px solid var(--critical); }
.small { font-size: 12px; color: var(--ink-2); }
.changed { font-weight: 600; }
table.heat td.n { min-width: 56px; }
table.heat th.type { white-space: normal; min-width: 64px; max-width: 110px; vertical-align: bottom; }
table.matrix { width: auto; }
table.matrix td.total, table.matrix th.total { font-weight: 600; }
table.matrix tr.total td { font-weight: 600; border-top: 1px solid var(--axis); }
details.leg { border-top: 1px solid var(--grid); padding: 6px 0; }
details.leg > summary { cursor: pointer; padding: 4px 0; }
details.leg > summary .sum { color: var(--ink-2); margin-left: 8px; }
details.leg[open] > summary { font-weight: 600; }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 4px 16px;
  margin: 8px 0; }
.facts div span { color: var(--ink-2); }
.dirgrid { display: grid; grid-template-columns: minmax(0, 1fr); gap: 4px; }
table.matrix td:first-child { white-space: nowrap; }
#tip { position: fixed; z-index: 10; pointer-events: none; display: none; max-width: 320px;
  background: var(--surface); color: var(--ink); border: 1px solid var(--axis); border-radius: 6px;
  padding: 6px 8px; font-size: 13px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
input[type="search"] { font: inherit; padding: 3px 8px; border: 1px solid var(--axis); border-radius: 6px;
  background: var(--surface); color: var(--ink); min-width: 220px; }
@media (max-width: 640px) { .tile .value { font-size: 20px; } }
@media print {
  html, body { background: #fff; }
  body { max-width: none; padding: 0; }
  section { break-inside: auto; border: 0; padding: 8px 0; }
  nav.toc, .toolbar, .noprint { display: none !important; }
  th { position: static; }
  details.leg { break-inside: avoid; }
}
"""

_JS = """
(function () {
  var tip = document.getElementById('tip');
  function showTip(text, x, y) {
    tip.textContent = text; tip.style.display = 'block';
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = Math.min(x + 14, window.innerWidth - w - 8), top = y + 14;
    if (top + h > window.innerHeight - 8) top = y - h - 10;
    tip.style.left = Math.max(8, left) + 'px'; tip.style.top = Math.max(8, top) + 'px';
  }
  document.addEventListener('mousemove', function (ev) {
    var el = ev.target.closest ? ev.target.closest('[data-tip]') : null;
    if (el) showTip(el.getAttribute('data-tip'), ev.clientX, ev.clientY); else tip.style.display = 'none';
  });
  document.addEventListener('focusin', function (ev) {
    var el = ev.target.closest ? ev.target.closest('[data-tip]') || ev.target.querySelector('[data-tip]') : null;
    if (!el) { tip.style.display = 'none'; return; }
    var r = el.getBoundingClientRect(); showTip(el.getAttribute('data-tip'), r.left + r.width / 2, r.bottom);
  });
  document.addEventListener('focusout', function () { tip.style.display = 'none'; });

  function cellValue(cell) {
    var v = cell.getAttribute('data-v');
    return v === null ? cell.textContent.trim() : v;
  }
  document.querySelectorAll('table.sortable').forEach(function (table) {
    var heads = table.tHead.rows[0].cells;
    Array.prototype.forEach.call(heads, function (th, idx) {
      th.tabIndex = 0;
      function sort() {
        var num = th.getAttribute('data-type') === 'num';
        var asc = th.getAttribute('aria-sort') !== 'ascending';
        Array.prototype.forEach.call(heads, function (h) { h.removeAttribute('aria-sort'); });
        th.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
        var body = table.tBodies[0], rows = Array.prototype.slice.call(body.rows);
        rows.sort(function (a, b) {
          var x = cellValue(a.cells[idx]), y = cellValue(b.cells[idx]), r;
          if (num) { x = parseFloat(x); y = parseFloat(y); x = isNaN(x) ? -Infinity : x;
                     y = isNaN(y) ? -Infinity : y; r = x - y; }
          else { r = String(x).localeCompare(String(y), undefined, {numeric: true, sensitivity: 'base'}); }
          return asc ? r : -r;
        });
        rows.forEach(function (row) { body.appendChild(row); });
      }
      th.addEventListener('click', sort);
      th.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); sort(); }
      });
    });
  });

  var filter = document.getElementById('leg-filter');
  if (filter) filter.addEventListener('input', function () {
    var q = filter.value.toLowerCase();
    document.querySelectorAll('[data-filter-rows] tbody tr').forEach(function (row) {
      row.style.display = row.textContent.toLowerCase().indexOf(q) >= 0 ? '' : 'none';
    });
  });

  var map = document.getElementById('map-svg');
  document.querySelectorAll('input[name="map-view"]').forEach(function (r) {
    r.addEventListener('change', function () { if (r.checked && map) map.setAttribute('data-view', r.value); });
  });
  var names = document.getElementById('map-names');
  if (names && map) names.addEventListener('change', function () { map.classList.toggle('nolabels', !names.checked); });

  function openTarget() {
    var id = decodeURIComponent(location.hash.slice(1));
    var el = id && document.getElementById(id);
    if (el && el.tagName === 'DETAILS') el.open = true;
  }
  window.addEventListener('hashchange', openTarget); openTarget();
  function setAll(open) { document.querySelectorAll('details.leg').forEach(function (d) { d.open = open; }); }
  var ex = document.getElementById('expand-all'), co = document.getElementById('collapse-all');
  if (ex) ex.addEventListener('click', function () { setAll(true); });
  if (co) co.addEventListener('click', function () { setAll(false); });
  var wasOpen = [];
  window.addEventListener('beforeprint', function () {
    wasOpen = Array.prototype.map.call(document.querySelectorAll('details.leg'), function (d) { return d.open; });
    setAll(true);
  });
  window.addEventListener('afterprint', function () {
    document.querySelectorAll('details.leg').forEach(function (d, i) { d.open = !!wasOpen[i]; });
  });
  var pr = document.getElementById('print');
  if (pr) pr.addEventListener('click', function () { window.print(); });
})();
"""


def _header(summary: dict[str, Any]) -> str:
    proj = summary['project']
    title = proj['name'] or (proj['file'].replace('\\', '/').rsplit('/', 1)[-1] if proj['file'] else '')
    meta = [f"<span>Made {_e(proj['generated'])}</span>"]
    if proj['file']:
        meta.append(f"<span>Project file {_e(proj['file'])}</span>")
    if proj['omrat_version']:
        meta.append(f"<span>OMRAT {_e(proj['omrat_version'])}</span>")
    return (
        f"<header><h1>Traffic report{': ' + _e(title) if title else ''}</h1>"
        f"<p class=\"meta\">{''.join(meta)}</p>"
        "<p class=\"lead\">The traffic on every leg as it was in OMRAT when this report was made, unsaved "
        "changes included. <em>As stored</em> is what the Traffic Data tab shows; <em>in the calculation</em> "
        "is what Run model uses after traffic scaling and suppressed legs.</p>"
        "<nav class=\"toc noprint\"><a href=\"#checks\">Things to check</a><a href=\"#map\">Map</a>"
        "<a href=\"#legs\">Legs</a><a href=\"#types\">Ship types</a><a href=\"#details\">Leg details</a>"
        "<a href=\"#about\">About the numbers</a>"
        "<button type=\"button\" id=\"print\">Print / Save as PDF</button></nav></header>"
    )


def _tiles(summary: dict[str, Any]) -> str:
    f = summary['figures']
    tiles = []

    def tile(label: str, value: str, sub: str = '') -> None:
        tiles.append(f"<div class=\"tile\"><div class=\"label\">{_e(label)}</div>"
                     f"<div class=\"value\">{value}</div>"
                     + (f"<div class=\"sub\">{sub}</div>" if sub else '') + "</div>")

    tile('Legs', str(f['legs']), f"{f['in_calc']} in the calculation, {f['suppressed']} suppressed")
    if f['busiest']:
        seg, name, ships = f['busiest']
        tile('Busiest leg', f"<a href=\"#leg-{_e(seg)}\">{_e(name)}</a>", f"{fmt_ships(ships)} ships/year")
    else:
        tile('Busiest leg', '-', 'no leg has ships')
    tile('Legs without ships', str(f['no_traffic']), 'in the calculation')
    tile('Copied traffic', str(f['copies']), f"{f['locked']} leg(s) locked")
    tile('Warnings', str(f['warnings']), 'probably wrong')
    tile('Notes', str(f['notes']), 'check that it is expected')
    return f"<section id=\"figures\" aria-label=\"Key figures\"><div class=\"tiles\">{''.join(tiles)}</div></section>"


def _checks(summary: dict[str, Any]) -> str:
    by_id = {leg['id']: leg for leg in summary['legs']}
    items = []
    for c in summary['checks']:
        icon, label = (_WARN_ICON, 'Warning') if c['level'] == 'warning' else (_NOTE_ICON, 'Note')
        chips = ''
        legs = [by_id[s] for s in c['legs'] if s in by_id]
        if legs:
            shown = ''.join(_leg_link(leg) for leg in legs[:15])
            more = f"<span class=\"small\">and {len(legs) - 15} more</span>" if len(legs) > 15 else ''
            chips = f"<div class=\"chips\">{shown}{more}</div>"
        items.append(f"<li><span class=\"badge {c['level']}\">{icon} {label}</span>{_e(c['text'])}{chips}</li>")
    if not items:
        items.append(f"<li><span class=\"badge good\">{_OK_ICON} OK</span>No problems found.</li>")
    counts: dict[str, int] = {}
    for c in summary['checks']:
        counts[c['code']] = counts.get(c['code'], 0) + 1
    ran = ''.join(
        f"<li>{_e(text)}: {'<b>' + str(counts[code]) + ' found</b>' if code in counts else _OK_ICON + ' none'}</li>"
        for code, text in CHECK_TEXT.items())
    return (
        "<section id=\"checks\"><h2>Things to check</h2>"
        f"<ul class=\"checks\">{''.join(items)}</ul>"
        f"<details class=\"small\" style=\"margin-top:8px\"><summary>What was checked</summary><ul>{ran}</ul>"
        "</details></section>"
    )


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def _project(legs: list[dict[str, Any]]):
    pts = [p for leg in legs for p in (leg['start'], leg['end']) if p is not None]
    if not pts:
        return None
    lat0 = sum(p[1] for p in pts) / len(pts)
    kx = math.cos(math.radians(lat0)) or 1e-9
    xs = [p[0] * kx for p in pts]
    ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    height = min(720.0, max(320.0, MAP_W * dy / dx))
    scale = min((MAP_W - 2 * MAP_PAD) / dx, (height - 2 * MAP_PAD) / dy)
    ox = (MAP_W - dx * scale) / 2.0
    oy = (height - dy * scale) / 2.0

    def to_px(p: tuple[float, float]) -> tuple[float, float]:
        return ox + (p[0] * kx - x0) * scale, height - (oy + (p[1] - y0) * scale)
    return to_px, height, scale


def _nice(value: float) -> float:
    if value <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(value))
    for m in (1, 2, 5, 10):
        if m * exp >= value * 0.75:
            return m * exp
    return 10 * exp


def _f(v: float) -> str:
    return f"{v:.1f}"


def _map(summary: dict[str, Any]) -> str:
    legs = summary['legs']
    proj = _project(legs)
    if proj is None:
        return "<section id=\"map\"><h2>Map</h2><p>The legs have no coordinates.</p></section>"
    to_px, height, scale = proj
    qmax = max([d['ships'] for leg in legs for d in leg['dirs']]
               + [d['calc'] or 0.0 for leg in legs for d in leg['dirs']] + [0.0])

    def width(q: float) -> float:
        return 0.0 if q <= 0 or qmax <= 0 else max(BAND_MIN_PX, BAND_MAX_PX * q / qmax)

    lines, bands, labels, nodes = [], [], [], set()
    kinds_seen: set[str] = set()
    for leg in legs:
        if leg['start'] is None or leg['end'] is None:
            continue
        (xa, ya), (xb, yb) = to_px(leg['start']), to_px(leg['end'])
        nodes.update({(round(xa, 1), round(ya, 1)), (round(xb, 1), round(yb, 1))})
        length = math.hypot(xb - xa, yb - ya)
        if length < 1e-6:
            continue
        ux, uy = (xb - xa) / length, (yb - ya) / length
        nx, ny = -uy, ux                  # screen right of the drawn direction (y down)
        inset = min(6.0, 0.12 * length)
        line_cls = {'suppressed': ' suppressed', 'none': ' none'}
        for view, kind in (('stored', leg['kind']), ('calc', leg['calc_kind'])):
            cls = line_cls.get(kind, '')
            lines.append(f"<line class=\"leg-line{cls} v-{view}\" x1=\"{_f(xa)}\" y1=\"{_f(ya)}\" "
                         f"x2=\"{_f(xb)}\" y2=\"{_f(yb)}\"/>")
            if kind == 'none':
                kinds_seen.add('none')
            for d in leg['dirs']:
                q = d['ships'] if view == 'stored' else (d['calc'] or 0.0)
                if view == 'calc' and leg['suppressed']:
                    q = 0.0
                w = width(q)
                if w <= 0:
                    continue
                kinds_seen.add(kind)
                side = 1.0 if d['index'] == 0 else -1.0
                off = side * (1.0 + w / 2.0)
                sx, sy = nx * off, ny * off
                p0 = (xa + ux * inset + sx, ya + uy * inset + sy)
                p1 = (xb - ux * inset + sx, yb - uy * inset + sy)
                coords = f"x1=\"{_f(p0[0])}\" y1=\"{_f(p0[1])}\" x2=\"{_f(p1[0])}\" y2=\"{_f(p1[1])}\""
                where = 'in the calculation' if view == 'calc' else 'as stored'
                tip = f"{leg['name']} {d['label']}: {fmt_ships(q)} ships/year {where}. {leg['source_text']}"
                arrow = ''
                if w >= 6 and length > 40:
                    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
                    ax, ay = ux * side, uy * side
                    s = min(4.0, w / 2.5)
                    tri = [(mx + ax * s, my + ay * s), (mx - ax * s + nx * s, my - ay * s + ny * s),
                           (mx - ax * s - nx * s, my - ay * s - ny * s)]
                    arrow = "<polygon class=\"arrow\" points=\"" + ' '.join(f"{_f(x)},{_f(y)}" for x, y in tri) + "\"/>"
                bands.append(
                    f"<a href=\"#leg-{_e(leg['id'])}\" class=\"v-{view}\" aria-label=\"{_e(tip)}\">"
                    f"<line class=\"band k-{kind}\" {coords} stroke-width=\"{_f(w)}\"/>{arrow}"
                    f"<line class=\"hit\" {coords} stroke-width=\"{_f(max(w, 12.0))}\" data-tip=\"{_e(tip)}\"/></a>"
                )
        widest = max([width(d['ships']) for d in leg['dirs']] + [width(d['calc'] or 0.0) for d in leg['dirs']])
        angle = math.degrees(math.atan2(uy, ux))
        flip = angle > 90 or angle < -90
        if flip:
            angle -= 180 if angle > 0 else -180
        off = -(widest + 1.0 + 9.0)       # beyond the direction-2 band
        lx, ly = (xa + xb) / 2 + nx * off, (ya + yb) / 2 + ny * off
        labels.append(f"<text class=\"label\" text-anchor=\"middle\" dominant-baseline=\"middle\" "
                      f"transform=\"translate({_f(lx)} {_f(ly)}) rotate({angle:.1f})\">{_e(leg['name'])}</text>")
    wps = ''.join(f"<circle class=\"wp\" cx=\"{x}\" cy=\"{y}\" r=\"2.5\"/>" for x, y in sorted(nodes))

    # Chrome: north arrow and scale bar.
    km_per_px = 111.32 / scale
    bar_km = _nice(120 * km_per_px)
    bar_px = bar_km / km_per_px
    bx, by = 16.0, height - 18.0
    chrome = (
        f"<line class=\"chrome-line\" x1=\"{_f(bx)}\" y1=\"{_f(by)}\" x2=\"{_f(bx + bar_px)}\" y2=\"{_f(by)}\"/>"
        f"<line class=\"chrome-line\" x1=\"{_f(bx)}\" y1=\"{_f(by - 4)}\" x2=\"{_f(bx)}\" y2=\"{_f(by + 4)}\"/>"
        f"<line class=\"chrome-line\" x1=\"{_f(bx + bar_px)}\" y1=\"{_f(by - 4)}\" x2=\"{_f(bx + bar_px)}\" "
        f"y2=\"{_f(by + 4)}\"/>"
        f"<text class=\"chrome\" x=\"{_f(bx)}\" y=\"{_f(by - 8)}\">{bar_km:g} km</text>"
        f"<text class=\"chrome\" x=\"{MAP_W - 22}\" y=\"20\" text-anchor=\"middle\">N</text>"
        f"<line class=\"chrome-line\" x1=\"{MAP_W - 22}\" y1=\"42\" x2=\"{MAP_W - 22}\" y2=\"26\"/>"
        f"<polygon class=\"chrome\" points=\"{MAP_W - 22},24 {MAP_W - 26},31 {MAP_W - 18},31\"/>"
    )
    toggle = ''
    if summary['calc_differs']:
        toggle = ("<span role=\"radiogroup\" aria-label=\"Traffic shown\">"
                  "<label><input type=\"radio\" name=\"map-view\" value=\"stored\" checked> As stored</label> "
                  "<label><input type=\"radio\" name=\"map-view\" value=\"calc\"> In the calculation</label></span>")
    checked = ' checked' if len(legs) <= LABEL_DEFAULT_MAX_LEGS else ''
    order = ('own', 'copy', 'received')
    legend = []
    for kind in order:
        if kind in kinds_seen:
            legend.append(f"<span><span class=\"key\" style=\"background:var(--{kind})\"></span>"
                          f"{_e(KIND_LABEL[kind])}</span>")
    if any(leg['suppressed'] for leg in legs):
        legend.append("<span><span class=\"key\" style=\"background:var(--suppressed)\"></span>"
                      "<span class=\"key dashed\"></span>Suppressed leg (not in the calculation)</span>")
    if 'none' in kinds_seen:
        legend.append(f"<span><span class=\"key none\"></span>{_WARN_ICON} No ships</span>")
    sample = _nice(qmax / 2) if qmax > 0 else 0
    if sample > 0:
        legend.append(f"<span><span class=\"key\" style=\"background:var(--muted);height:{_f(width(sample))}px\">"
                      f"</span>{fmt_ships(sample)} ships/year (band width scales with ships)</span>")
    view = 'stored'
    return (
        "<section id=\"map\"><h2>Map</h2>"
        f"<div class=\"toolbar noprint\">{toggle}<label><input type=\"checkbox\" id=\"map-names\"{checked}> "
        "Leg names</label><span class=\"small\">Each leg has one band per direction, on the side the ships "
        "keep to (starboard), as wide as the ships per year; the white arrow points the way they sail. "
        "Hover a band for the numbers, click it for the leg's details.</span></div>"
        f"<div class=\"mapwrap\"><svg id=\"map-svg\" class=\"map{'' if checked else ' nolabels'}\" "
        f"data-view=\"{view}\" viewBox=\"0 0 {MAP_W} {_f(height)}\" role=\"img\" "
        "aria-label=\"Map of the legs with the ships per year per direction\">"
        f"<g>{''.join(lines)}</g><g>{''.join(bands)}</g><g>{wps}</g><g>{''.join(labels)}</g>"
        f"<g>{chrome}</g></svg></div>"
        f"<div class=\"legend\">{''.join(legend)}</div></section>"
    )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _check_counts(leg: dict[str, Any], checks: list[dict[str, Any]]) -> tuple[int, int]:
    warn = sum(1 for i in leg['checks'] if checks[i]['level'] == 'warning')
    note = sum(1 for i in leg['checks'] if checks[i]['level'] == 'note')
    return warn, note


def _leg_table(summary: dict[str, Any], decimal: str, stem: str) -> str:
    legs, checks, types = summary['legs'], summary['checks'], summary['types']
    show_calc = summary['calc_differs']
    tmax = max([leg['ships'] for leg in legs] + [0.0])
    head = ['<th>Leg</th>', '<th class="n" data-type="num">Length (km)</th>',
            '<th class="n" data-type="num">Direction 1</th>', '<th class="n" data-type="num">Direction 2</th>',
            '<th data-type="num">Ships/year</th>']
    if show_calc:
        head.append('<th class="n" data-type="num">In the calculation</th>')
    head += ['<th>Traffic from</th>', '<th>Main ship types</th>', '<th class="n" data-type="num">Speed (kn)</th>',
             '<th class="n" data-type="num">Draught (m)</th>', '<th class="n" data-type="num">Checks</th>']
    rows = []
    for leg in legs:
        d0, d1 = leg['dirs']
        warn, note = _check_counts(leg, checks)
        bar = (100.0 * leg['ships'] / tmax) if tmax > 0 else 0.0
        cells = [
            f"<td class=\"legcell\" data-v=\"{_e(leg['name'])}\"><span class=\"kind {leg['kind']}\" "
            f"title=\"{_e(KIND_LABEL[leg['kind']])}\"></span>{_leg_link(leg)}"
            f"<span class=\"dirlabel\">id {_e(leg['id'])}"
            + (f" &middot; route {_e(leg['route'])}" if leg['route'] else '') + "</span></td>",
            f"<td class=\"n\" data-v=\"{(leg['length_m'] or 0) / 1000:.3f}\">"
            f"{_num1((leg['length_m'] or 0) / 1000 if leg['length_m'] else None)}</td>",
            f"<td class=\"n\" data-v=\"{d0['ships']:.3f}\">{fmt_ships(d0['ships'])}"
            f"<span class=\"dirlabel\">{_e(d0['label'])}</span></td>",
            f"<td class=\"n\" data-v=\"{d1['ships']:.3f}\">{fmt_ships(d1['ships'])}"
            f"<span class=\"dirlabel\">{_e(d1['label'])}</span></td>",
            f"<td class=\"barcell\" data-v=\"{leg['ships']:.3f}\"><span class=\"bar\" "
            f"style=\"width:{bar * 0.7:.1f}px\"></span>{fmt_ships(leg['ships'])}</td>",
        ]
        if show_calc:
            if leg['calc'] is None:
                cells.append("<td class=\"n\" data-v=\"-1\">suppressed</td>")
            else:
                changed = abs(leg['calc'] - leg['ships']) > 1e-6 * max(1.0, leg['ships'])
                cells.append(f"<td class=\"n{' changed' if changed else ''}\" data-v=\"{leg['calc']:.3f}\">"
                             f"{fmt_ships(leg['calc'])}</td>")
        top = ''.join(f"<span class=\"tt\">{_e(short_type(types[i]))} {share * 100:.0f} %</span>"
                      for i, _q, share in leg['top_types'])
        badges = []
        if warn:
            badges.append(f"<span class=\"badge warning\">{_WARN_ICON} {warn}</span>")
        if note:
            badges.append(f"<span class=\"badge note\">{_NOTE_ICON} {note}</span>")
        cells += [
            f"<td class=\"src\">{_e(leg['source_text'])}</td>",
            f"<td>{top or '-'}</td>",
            f"<td class=\"n\" data-v=\"{leg['mean_speed'] or 0:.3f}\">{_num1(leg['mean_speed'])}</td>",
            f"<td class=\"n\" data-v=\"{leg['mean_draught'] or 0:.3f}\">{_num1(leg['mean_draught'])}</td>",
            f"<td class=\"n\" data-v=\"{warn * 1000 + note}\">{''.join(badges) or '-'}</td>",
        ]
        cls = ' class="suppressed"' if leg['suppressed'] else ''
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    csv = _csv_href(legs_csv(summary, decimal))
    return (
        "<section id=\"legs\"><h2>Legs</h2>"
        "<div class=\"toolbar noprint\"><input type=\"search\" id=\"leg-filter\" placeholder=\"Filter legs\" "
        "aria-label=\"Filter legs\">"
        f"<a class=\"btn\" download=\"{_e(stem)}_legs.csv\" href=\"{csv}\">Download CSV (Excel)</a>"
        "<span class=\"small\">Click a column header to sort. Ships/year as stored, both directions.</span>"
        "</div><div class=\"scroll\"><table class=\"sortable\" data-filter-rows>"
        f"<thead><tr>{''.join(head)}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def _type_table(summary: dict[str, Any], decimal: str, stem: str) -> str:
    legs, types, idx = summary['legs'], summary['types'], summary['type_index']
    if not idx:
        return "<section id=\"types\"><h2>Ship types per leg</h2><p>No leg has ships.</p></section>"
    vmax = max([leg['by_type'][i] for leg in legs for i in idx] + [0.0])
    head = ['<th>Leg</th>'] + [f"<th class=\"n type\" data-type=\"num\" title=\"{_e(types[i])}\">"
                               f"{_e(short_type(types[i]))}</th>" for i in idx]
    head.append('<th class="n" data-type="num">Total</th>')
    rows = []
    for leg in legs:
        cells = [f"<td data-v=\"{_e(leg['name'])}\">{_leg_link(leg)}</td>"]
        for i in idx:
            q = leg['by_type'][i]
            if q <= 0:
                cells.append("<td class=\"n\" data-v=\"0\"></td>")
                continue
            share = q / leg['ships'] * 100 if leg['ships'] > 0 else 0
            tip = f"{leg['name']}, {short_type(types[i])}: {fmt_ships(q)} ships/year ({share:.0f} % of the leg)"
            cells.append(f"<td class=\"n\" data-v=\"{q:.3f}\" data-tip=\"{_e(tip)}\"{_heat_style(q, vmax)}>"
                         f"{fmt_ships(q)}</td>")
        cells.append(f"<td class=\"n\" data-v=\"{leg['ships']:.3f}\"><b>{fmt_ships(leg['ships'])}</b></td>")
        cls = ' class="suppressed"' if leg['suppressed'] else ''
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    csv = _csv_href(types_csv(summary, decimal))
    return (
        "<section id=\"types\"><h2>Ship types per leg</h2>"
        "<div class=\"toolbar noprint\">"
        f"<a class=\"btn\" download=\"{_e(stem)}_ship_types.csv\" href=\"{csv}\">Download CSV (Excel)</a>"
        "<span class=\"small\">Ships/year as stored, both directions; darker = more ships. Only ship types "
        "with ships on some leg are shown.</span></div>"
        "<div class=\"scroll\"><table class=\"sortable heat\">"
        f"<thead><tr>{''.join(head)}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def _direction_matrix(d: dict[str, Any], types: list[str], lengths: list[str]) -> str:
    matrix = d['matrix']
    rows = [i for i in range(len(matrix)) if sum(matrix[i]) > 0]
    cols = [j for j in range(len(lengths)) if any(j < len(matrix[i]) and matrix[i][j] > 0 for i in rows)]
    title = f"<h3>{_e(d['label'])}: {fmt_ships(d['ships'])} ships/year</h3>"
    if not rows:
        return title + "<p class=\"small\">No ships.</p>"
    vmax = max(matrix[i][j] for i in rows for j in cols)
    head = (['<th>Ship type</th>'] + [f"<th class=\"n\">{_e(lengths[j])}</th>" for j in cols]
            + ['<th class="n total">Total</th>', '<th class="n">Speed (kn)</th>', '<th class="n">Draught (m)</th>'])
    body = []
    for i in rows:
        cells = [f"<td>{_e(short_type(types[i]))}</td>"]
        for j in cols:
            q = matrix[i][j]
            cells.append(f"<td class=\"n\"{_heat_style(q, vmax)}>{fmt_ships(q) if q > 0 else ''}</td>")
        cells += [f"<td class=\"n total\">{fmt_ships(sum(matrix[i]))}</td>",
                  f"<td class=\"n\">{_num1(d['type_speed'][i])}</td>",
                  f"<td class=\"n\">{_num1(d['type_draught'][i])}</td>"]
        body.append(f"<tr>{''.join(cells)}</tr>")
    totals = ['<td>Total</td>'] + [f"<td class=\"n\">{fmt_ships(sum(matrix[i][j] for i in rows))}</td>"
                                   for j in cols]
    totals += [f"<td class=\"n total\">{fmt_ships(d['ships'])}</td>", f"<td class=\"n\">{_num1(d['mean_speed'])}</td>",
               f"<td class=\"n\">{_num1(d['mean_draught'])}</td>"]
    body.append(f"<tr class=\"total\">{''.join(totals)}</tr>")
    note = ''
    if d['no_speed'] > 0 or d['no_draught'] > 0:
        note = (f"<p class=\"small\">{_WARN_ICON} Without speed: {fmt_ships(d['no_speed'])}, without draught: "
                f"{fmt_ships(d['no_draught'])} ships/year (Run model fills them).</p>")
    return (title + "<div class=\"scroll\"><table class=\"matrix\">"
            f"<thead><tr>{''.join(head)}</tr></thead><tbody>{''.join(body)}</tbody></table></div>" + note)


def _details(summary: dict[str, Any]) -> str:
    types, lengths, checks = summary['types'], summary['lengths'], summary['checks']
    out = []
    for leg in summary['legs']:
        warn, note = _check_counts(leg, checks)
        badges = ''
        if warn:
            badges += f" <span class=\"badge warning\">{_WARN_ICON} {warn}</span>"
        if note:
            badges += f" <span class=\"badge note\">{_NOTE_ICON} {note}</span>"
        d0, d1 = leg['dirs']
        calc = ('not in the calculation' if leg['calc'] is None else
                f"{fmt_ships(d0['calc'] or 0.0)} / {fmt_ships(d1['calc'] or 0.0)} ships/year")
        facts = [
            ('Id', leg['id']), ('Route', leg['route'] or '-'),
            ('Length', f"{(leg['length_m'] or 0) / 1000:.1f} km" if leg['length_m'] else '-'),
            ('Traffic from', leg['source_text']),
            ('In the calculation (dir 1 / dir 2)', calc),
            ('Mean speed', _num1(leg['mean_speed'], ' kn')), ('Mean draught', _num1(leg['mean_draught'], ' m')),
        ]
        facts_html = ''.join(f"<div><span>{_e(k)}:</span> {_e(v)}</div>" for k, v in facts)
        leg_checks = ''.join(
            f"<li><span class=\"badge {checks[i]['level']}\">"
            f"{_WARN_ICON if checks[i]['level'] == 'warning' else _NOTE_ICON} "
            f"{'Warning' if checks[i]['level'] == 'warning' else 'Note'}</span>{_e(checks[i]['text'])}</li>"
            for i in leg['checks'])
        moves = ''.join(
            f"<li>{_e(mv['src_name'])} {_e(mv['from_label'])} -> {_e(mv['dir_label'])}: {mv['share']:g} % "
            f"({fmt_ships(mv['ships'])} ships/year)</li>" for mv in leg['received_moves'])
        out.append(
            f"<details class=\"leg\" id=\"leg-{_e(leg['id'])}\"><summary><span class=\"kind {leg['kind']}\"></span>"
            f"{_e(leg['name'])}<span class=\"sum\">{fmt_ships(leg['ships'])} ships/year "
            f"({_e(d0['label'])} {fmt_ships(d0['ships'])}, {_e(d1['label'])} {fmt_ships(d1['ships'])})</span>"
            f"{badges}</summary><div class=\"facts\">{facts_html}</div>"
            + (f"<ul class=\"checks\">{leg_checks}</ul>" if leg_checks else '')
            + (f"<p><b>Ships moved here from suppressed legs</b> (added in the calculation):</p><ul>{moves}</ul>"
               if moves else '')
            + f"<div class=\"dirgrid\"><div>{_direction_matrix(d0, types, lengths)}</div>"
            f"<div>{_direction_matrix(d1, types, lengths)}</div></div></details>"
        )
    return (
        "<section id=\"details\"><h2>Leg details</h2>"
        "<div class=\"toolbar noprint\"><button type=\"button\" id=\"expand-all\">Expand all</button>"
        "<button type=\"button\" id=\"collapse-all\">Collapse all</button>"
        "<span class=\"small\">Ships/year per ship type and length (as stored); speed and draught are "
        "ship-weighted means.</span></div>"
        f"{''.join(out)}</section>"
    )


def _about(summary: dict[str, Any]) -> str:
    scaling = summary.get('scaling') or {}
    glob = scaling.get('global_percent')
    scale_txt = ''
    if isinstance(glob, (int, float)) and abs(float(glob) - 100.0) > 1e-9:
        scale_txt = f" The project's global traffic scaling is {float(glob):g} %."
    return (
        "<section id=\"about\"><h2>About the numbers</h2><ul>"
        "<li>Ships are counted where they cross each leg's passage line (the tangent line). One ship that "
        "sails three legs is counted on each of them, so adding up the legs does not give a number of ships.</li>"
        "<li><b>Direction 1</b> sails the way the leg was drawn (start to end), <b>direction 2</b> the "
        "opposite way.</li>"
        "<li><b>As stored</b> is what the Traffic Data tab holds (from Update all distributions, Copy traffic "
        "or typed in). <b>In the calculation</b> is what Run model uses: every cell times its traffic scaling, "
        "suppressed legs left out and their ships added to the target legs." + _e(scale_txt) + "</li>"
        "<li>Speed and draught are means weighted by ships per year. Cells with ships but no speed or draught "
        "are filled by Run model with the mean of the same ship type and length on the other legs.</li>"
        "<li>The CSV files use the decimal separator of the computer that made the report.</li>"
        "</ul></section>"
    )


def render_traffic_report_html(summary: dict[str, Any], *, decimal: str = '.', csv_stem: str = 'traffic') -> str:
    """The whole report page.  ``decimal`` is used in the CSV downloads
    (``','`` also switches their separator to ``';'``); ``csv_stem`` names
    the downloaded files."""
    proj = summary['project']
    title = 'Traffic report' + (f" - {proj['name']}" if proj['name'] else '')
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<meta name=\"generator\" content=\"OMRAT traffic report\">"
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>"
        + _header(summary) + _tiles(summary) + _checks(summary) + _map(summary)
        + _leg_table(summary, decimal, csv_stem) + _type_table(summary, decimal, csv_stem)
        + _details(summary) + _about(summary)
        + f"<div id=\"tip\" role=\"tooltip\"></div><script>{_JS}</script></body></html>\n"
    )
