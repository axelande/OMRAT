"""Draw help/source/_static/images/collision_types.svg (collisions.rst).

The five ship-ship encounter types as OMRAT counts them: head-on and
overtaking along a leg, crossing / merging / bend per pair of routes at a
shared waypoint, with the formulas of compute/basic_equations.py and
compute/ship_collision_model.py.  Re-run after a formula, a rule or a
default causation factor (compute/iwrap_defaults.py) changes:

    C:/OSGeo4W/apps/Python312/python.exe help/figures_src/make_collision_types_figure.py
"""
from __future__ import annotations

import math
import os
import re
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'source', '_static', 'images', 'collision_types.svg')

W, H = 900, 828
BLUE, RED, GREY, INK = '#1976D2', '#D32F2F', '#90A4AE', '#263238'

PANELS = {
    # name: (fill, stroke, header colour)
    'head_on': ('#e3f2fd', '#1976D2', '#0D47A1'),
    'overtaking': ('#e8f5e9', '#4CAF50', '#2E7D32'),
    'crossing': ('#fff3e0', '#FF9800', '#E65100'),
    'merging': ('#fffde7', '#F9A825', '#8D6E00'),
    'bend': ('#f3e5f5', '#9C27B0', '#6A1B9A'),
}

_SCRIPT = re.compile(r'([_^])\{([^}]*)\}|([_^])(\w)')


def rich(s: str, size: float) -> str:
    """``V_{ij}`` / ``V_1`` / ``s^2`` -> tspans.  dy (not baseline-shift),
    so every browser draws the sub- and superscripts."""
    out, pos, small = [], 0, size * 0.75
    for m in _SCRIPT.finditer(s):
        out.append(escape(s[pos:m.start()]))
        kind = m.group(1) or m.group(3)
        body = m.group(2) if m.group(1) else m.group(4)
        dy = size * 0.3 if kind == '_' else -size * 0.4
        out.append('<tspan dy="%.1f" font-size="%.1f">%s</tspan><tspan dy="%.1f">\u200b</tspan>'
                   % (dy, small, escape(body), -dy))
        pos = m.end()
    out.append(escape(s[pos:]))
    return ''.join(out)


def text(x, y, s, size=11, fill=INK, anchor='middle', weight='normal', style='normal'):
    return ('<text x="%g" y="%g" text-anchor="%s" font-size="%g" font-weight="%s" '
            'font-style="%s" fill="%s">%s</text>' % (x, y, anchor, size, weight, style, fill, rich(s, size)))


def line(points, colour, width=3, dash=None, arrow=None):
    pts = ' '.join('%g,%g' % p for p in points)
    extra = ''
    if dash:
        extra += ' stroke-dasharray="%s"' % dash
    if arrow:
        extra += ' marker-end="url(#%s)"' % arrow
    return ('<polyline points="%s" fill="none" stroke="%s" stroke-width="%g" '
            'stroke-linejoin="round"%s/>' % (pts, colour, width, extra))


def arc(cx, cy, r, a0, a1, colour):
    """Short arc from math angle a0 to a1 (degrees, counter-clockwise, y up)."""
    def pt(a):
        return cx + r * math.cos(math.radians(a)), cy - r * math.sin(math.radians(a))
    (x0, y0), (x1, y1) = pt(a0), pt(a1)
    sweep = 0 if (a1 - a0) % 360 < 180 else 1  # screen y points down
    return ('<path d="M %.1f,%.1f A %g,%g 0 0,%d %.1f,%.1f" fill="none" stroke="%s" '
            'stroke-width="1.5"/>' % (x0, y0, r, r, sweep, x1, y1, colour))


def waypoint(cx, cy):
    return '<circle cx="%g" cy="%g" r="5.5" fill="#FF9800" stroke="#E65100" stroke-width="2"/>' % (cx, cy)


def panel(x, y, w, h, kind, title):
    fill, stroke, head = PANELS[kind]
    return [
        '<rect x="%g" y="%g" width="%g" height="%g" rx="10" fill="%s" stroke="%s" stroke-width="2"/>'
        % (x, y, w, h, fill, stroke),
        text(x + w / 2, y + 24, title, 16, head, weight='bold'),
    ]


def formulas(x, y, rows, colour, size=11, step=19):
    out = []
    for i, row in enumerate(rows):
        bold = row.startswith('!')
        out.append(text(x, y + i * step, row.lstrip('!'), size, colour if bold else INK,
                        weight='bold' if bold else 'normal'))
    return out


def head_on(x, y):
    cx = x + 210
    p = panel(x, y, 420, 270, 'head_on', 'Head-on')
    p.append('<rect x="%g" y="%g" width="340" height="78" fill="#ffffff" stroke="%s" '
             'stroke-dasharray="5 3"/>' % (x + 40, y + 40, GREY))
    p.append(text(x + 46, y + 53, 'leg', 9, '#78909C', anchor='start', style='italic'))
    p.append(line([(x + 70, y + 64), (x + 250, y + 64)], BLUE, arrow='blue'))
    p.append(text(x + 160, y + 58, 'direction 1 (Q_1, V_1)', 10, '#1565C0'))
    p.append(line([(x + 350, y + 96), (x + 170, y + 96)], RED, arrow='red'))
    p.append(text(x + 260, y + 111, 'direction 2 (Q_2, V_2)', 10, '#C62828'))
    p.append(text(cx, y + 138, 'Opposite directions on the same leg', 11, style='italic'))
    p += formulas(cx, y + 162, [
        '!V_{ij} = V_1 + V_2',
        'P_G = \u03a3_k \u03a3_l w_k w_l P_G(k, l)  over the lateral mixture components',
        'P_G(k, l) = \u03a6((\u0394\u03bc + B)/\u03c3) \u2212 \u03a6((\u0394\u03bc \u2212 B)/\u03c3)',
        '\u0394\u03bc = \u03bc_k \u2212 \u03bc_l,  \u03c3 = \u221a(\u03c3_k^2 + \u03c3_l^2),  B = (B_1 + B_2)/2',
        '!N_G = (Q_1/V_1) (Q_2/V_2) V_{ij} P_G L_w',
    ], '#0D47A1')
    return p


def overtaking(x, y):
    cx = x + 210
    p = panel(x, y, 420, 270, 'overtaking', 'Overtaking')
    p.append('<rect x="%g" y="%g" width="340" height="78" fill="#ffffff" stroke="%s" '
             'stroke-dasharray="5 3"/>' % (x + 40, y + 40, GREY))
    p.append(text(x + 46, y + 53, 'leg', 9, '#78909C', anchor='start', style='italic'))
    p.append(line([(x + 70, y + 66), (x + 300, y + 66)], BLUE, arrow='blue'))
    p.append(text(x + 185, y + 60, 'fast ship (Q_f, V_f)', 10, '#1565C0'))
    p.append(line([(x + 90, y + 94), (x + 230, y + 94)], RED, width=2.5, arrow='red'))
    p.append(text(x + 160, y + 110, 'slow ship (Q_s, V_s)', 10, '#C62828'))
    p.append(text(cx, y + 138, 'Same direction, different speeds', 11, style='italic'))
    p += formulas(cx, y + 162, [
        '!V_{ij} = V_f \u2212 V_s   (no candidates unless V_f > V_s)',
        'Both ships drawn from the same direction\u2019s mixture,',
        'P_G summed over the component pairs as for head-on',
        '',
        '!N_G = (Q_f/V_f) (Q_s/V_s) V_{ij} P_G L_w',
    ], '#2E7D32')
    return p


def crossing(x, y):
    cx, cy = x + 140, y + 112
    p = panel(x, y, 280, 240, 'crossing', 'Crossing')
    p.append(line([(cx - 110, cy), (cx + 105, cy)], BLUE, arrow='blue'))
    p.append(line([(cx - 55, cy + 70), (cx + 52, cy - 66)], RED, arrow='red'))
    p.append(arc(cx, cy, 30, 0, math.degrees(math.atan2(66, 52)), '#E65100'))
    p.append(text(cx + 36, cy - 12, '\u03b8', 13, '#E65100', anchor='start', weight='bold'))
    p.append(waypoint(cx, cy))
    p.append(text(cx - 70, cy - 8, 'route 1', 10, '#1565C0'))
    p.append(text(cx - 50, cy + 62, 'route 2', 10, '#C62828', anchor='end'))
    p.append(text(cx, y + 200, 'Two routes over four legs', 11, style='italic'))
    p.append(text(cx, y + 217, 'whose paths cross at the waypoint', 11, style='italic'))
    return p


def merging(x, y):
    cx, cy = x + 150, y + 112
    p = panel(x, y, 280, 240, 'merging', 'Merging')
    p.append(line([(cx - 115, cy - 52), (cx, cy - 3), (cx + 105, cy - 3)], BLUE, arrow='blue'))
    p.append(line([(cx - 115, cy + 52), (cx, cy + 3), (cx + 105, cy + 3)], RED, arrow='red'))
    a = math.degrees(math.atan2(52, 115))
    p.append(arc(cx, cy, 34, 180 - a, 180 + a, '#8D6E00'))
    p.append(text(cx - 40, cy + 5, '\u03b8', 13, '#8D6E00', anchor='end', weight='bold'))
    p.append(waypoint(cx, cy))
    p.append(text(cx - 95, cy - 52, 'route 1', 10, '#1565C0', anchor='start'))
    p.append(text(cx - 95, cy + 66, 'route 2', 10, '#C62828', anchor='start'))
    p.append(text(cx + 55, cy - 12, 'same out-leg', 10, '#5D4037'))
    p.append(text(x + 140, y + 200, 'Two routes from different legs', 11, style='italic'))
    p.append(text(x + 140, y + 217, 'leaving on the same leg', 11, style='italic'))
    return p


def bend(x, y):
    cx, cy = x + 120, y + 120
    p = panel(x, y, 280, 240, 'bend', 'Bend')
    bx, by = cx + 115, cy - 80                     # far end of leg B
    p.append(line([(cx + 8, cy), (cx + 140, cy)], BLUE, width=1.5, dash='5 4', arrow='blue'))
    p.append(text(cx + 140, cy + 16, 'does not turn', 10, '#1565C0', anchor='end'))
    p.append(line([(cx - 100, cy - 3), (cx, cy - 3), (bx - 4, by - 3)], BLUE, arrow='blue'))
    p.append(line([(bx + 4, by + 5), (cx + 3, cy + 4), (cx - 100, cy + 4)], RED, arrow='red'))
    turn = math.degrees(math.atan2(80, 115))
    p.append(arc(cx, cy, 34, 0, turn, '#6A1B9A'))
    p.append(text(cx + 40, cy - 8, 'turn', 10, '#6A1B9A', anchor='start', weight='bold'))
    p.append(waypoint(cx, cy))
    p.append(text(cx - 60, cy - 10, 'A \u2192 B', 10, '#1565C0'))
    p.append(text(cx - 60, cy + 20, 'B \u2192 A', 10, '#C62828'))
    p.append(text(x + 140, y + 200, 'The two reverse routes of one turn,', 11, style='italic'))
    p.append(text(x + 140, y + 217, 'every ship of both, at \u03b8 = 180\u00b0 \u2212 turn', 11, style='italic'))
    return p


def waypoint_box(x, y):
    w, h = 860, 150
    out = ['<rect x="%g" y="%g" width="%g" height="%g" rx="10" fill="#fafafa" stroke="#9E9E9E" '
           'stroke-width="1.5"/>' % (x, y, w, h),
           text(x + w / 2, y + 24, 'One formula for crossing, merging and bend (Pedersen 1995), '
                'per pair of routes and pair of ship cells i, j', 13, '#37474F', weight='bold')]
    out += formulas(x + w / 2, y + 50, [
        '!N_G = \u03a3_{i,j} (Q_i T_1 / V_i) (Q_j T_2 / V_j) V_{ij} D_{ij} / |sin \u03b8|',
        'V_{ij} = \u221a(V_i^2 + V_j^2 \u2212 2 V_i V_j cos \u03b8)',
        'D_{ij} = (L_i V_j + L_j V_i) |sin \u03b8| / V_{ij}  +  B_i \u221a(1 \u2212 (V_j sin \u03b8 / V_{ij})^2)'
        '  +  B_j \u221a(1 \u2212 (V_i sin \u03b8 / V_{ij})^2)',
        'T_1, T_2 = the routes\u2019 junction-matrix shares;  \u03b8 = angle between the arrival courses;'
        '  speeds at the waypoint;  L = length-class midpoint',
        'Route pairs within the angle limit (default 10\u00b0) of parallel or opposite are skipped',
    ], '#37474F', step=21)
    return out


def build() -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="Arial, Helvetica, sans-serif">'
        % (W, H),
        '<!-- Generated by help/figures_src/make_collision_types_figure.py; edit that, not this file. -->',
        '<defs>',
    ]
    for name, colour in (('blue', BLUE), ('red', RED)):
        parts.append('<marker id="%s" markerWidth="12" markerHeight="8" refX="10" refY="4" orient="auto" '
                     'markerUnits="userSpaceOnUse"><polygon points="0 0, 12 4, 0 8" fill="%s"/></marker>'
                     % (name, colour))
    parts.append('</defs>')
    parts.append('<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))
    parts.append(text(W / 2, 30, 'Ship-ship collision types in OMRAT', 20, '#1a1a2e', weight='bold'))
    parts.append(text(W / 2, 64, 'Along a leg (Hansen 2008)', 13, '#455A64', weight='bold'))
    parts += head_on(20, 75)
    parts += overtaking(460, 75)
    parts.append(text(W / 2, 374, 'At a shared waypoint, per pair of routes '
                      '(in-leg \u2192 out-leg, shares from the junction matrix)', 13, '#455A64', weight='bold'))
    parts += crossing(20, 385)
    parts += merging(310, 385)
    parts += bend(600, 385)
    parts += waypoint_box(20, 638)
    parts.append('<rect x="20" y="798" width="860" height="24" rx="5" fill="#efebe9" stroke="#795548"/>')
    parts.append(text(W / 2, 814, 'Accident frequency = N_G \u00d7 P_C.   Default P_C (IWRAP): head-on 4.9e-5, '
                      'overtaking 1.1e-4, crossing / merging / bend 1.3e-4;  '
                      '\u00f7\u221aF per passenger ship or fast ferry', 10.5, '#4E342E'))
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


if __name__ == '__main__':
    with open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(build())
    print('wrote', os.path.normpath(OUT))
