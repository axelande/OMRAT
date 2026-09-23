"""Draw the schematic figures for the "Moving traffic" section of the user
guide (help/source/user_guide.rst, ``suppress-leg``).

Run with the OSGeo4W Python; writes SVG files into
``help/source/_static/images``::

    C:/OSGeo4W/apps/Python312/python.exe help/figures_src/make_suppress_figures.py

Colours and line styles match the *Traffic links* map layer
(``HandleQGISIface._LINK_STYLE``): orange dashed = traffic moved from a
suppressed leg, grey dotted = suppressed together with a lead leg.  Legs
are blue like OMRAT's leg layers; a suppressed leg is dashed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'source' / '_static' / 'images'

LEG = '#1f3fd6'
MOVED = '#f07c00'
WITH = '#7f7f7f'
INK = '#222222'
MUTED = '#666666'
FARM = '#d9d9d9'

plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})


def _leg(ax, p0, p1, name, suppressed=False, label_offset=(0, 0.14), ha='center'):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=LEG, lw=3.2 if not suppressed else 2.6,
            ls=(0, (5, 3)) if suppressed else '-', solid_capstyle='round', zorder=2)
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    ax.text(mx + label_offset[0], my + label_offset[1], name, ha=ha, va='center', color=INK, fontsize=9,
            zorder=5)
    return (mx, my)


def _arrow(ax, a, b, colour, style, rad=0.25, label=None, label_at=0.5, label_dy=0.0):
    ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>,head_length=7,head_width=4',
                                 color=colour, lw=1.6, ls=style, zorder=3, shrinkA=4, shrinkB=4))
    if label:
        x = a[0] + (b[0] - a[0]) * label_at
        y = a[1] + (b[1] - a[1]) * label_at + label_dy
        ax.text(x, y, label, color=INK, fontsize=8.5, ha='center', va='center', zorder=6,
                bbox={'boxstyle': 'round,pad=0.2', 'fc': 'white', 'ec': colour, 'lw': 0.8})


def _farm(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, fc=FARM, ec='#9e9e9e', lw=0.8, hatch='..', zorder=1))
    ax.text(x + w / 2, y + h + 0.06, 'wind farm', ha='center', va='bottom', color=MUTED, fontsize=8, zorder=5)


def _legend(ax, y=-1.05):
    ax.plot([0.0, 0.5], [y, y], color=LEG, lw=2.6, ls=(0, (5, 3)))
    ax.text(0.6, y, 'suppressed leg', va='center', color=INK, fontsize=8.5)
    ax.plot([2.4, 2.9], [y, y], color=MOVED, lw=1.6, ls='--')
    ax.text(3.0, y, 'traffic moved (share)', va='center', color=INK, fontsize=8.5)


def _finish(fig, ax, name, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(OUT / name, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)
    print('wrote', OUT / name)


def fig_series():
    """Example 1: one leg, one detour of two legs in a row."""
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    _farm(ax, 2.5, 0.7, 1.0, 0.6)
    a = _leg(ax, (0, 1), (6, 1), 'LEG_A  (suppressed)', suppressed=True, label_offset=(-2.0, 0.18))
    b = _leg(ax, (0, 1), (3, -0.4), 'LEG_B', label_offset=(-0.35, -0.2))
    c = _leg(ax, (3, -0.4), (6, 1), 'LEG_C', label_offset=(0.35, -0.2))
    _arrow(ax, a, b, MOVED, '--', rad=0.3)
    _arrow(ax, a, c, MOVED, '--', rad=-0.3)
    for x in (1.35, 4.65):
        ax.text(x, 0.72, 'E 100 %, W 100 %', color=INK, fontsize=8.5, ha='center', va='center', zorder=6,
                bbox={'boxstyle': 'round,pad=0.2', 'fc': 'white', 'ec': MOVED, 'lw': 0.8})
    ax.text(3, 1.75, 'LEG_A carries 100 ships/year East going and 40 West going',
            ha='center', color=INK, fontsize=9)
    _legend(ax)
    _finish(fig, ax, 'suppress_series.svg', (-0.3, 6.3), (-1.3, 2.0))


def fig_alternatives():
    """Example 2: the ships split between two alternative routes."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _farm(ax, 2.5, 0.7, 1.0, 0.6)
    a = _leg(ax, (0, 1), (6, 1), 'LEG_A  (suppressed)', suppressed=True, label_offset=(-2.0, 0.18))
    n1 = _leg(ax, (0, 1), (3, 2.4), 'LEG_N1', label_offset=(-0.35, 0.2))
    n2 = _leg(ax, (3, 2.4), (6, 1), 'LEG_N2', label_offset=(0.35, 0.2))
    s1 = _leg(ax, (0, 1), (3, -0.4), 'LEG_S1', label_offset=(-0.35, -0.2))
    s2 = _leg(ax, (3, -0.4), (6, 1), 'LEG_S2', label_offset=(0.35, -0.2))
    _arrow(ax, a, n1, MOVED, '--', rad=-0.3, label='E 80 %', label_at=0.8)
    _arrow(ax, a, n2, MOVED, '--', rad=0.3, label='E 80 %', label_at=0.8)
    _arrow(ax, a, s1, MOVED, '--', rad=0.3, label='E 20 %', label_at=0.8)
    _arrow(ax, a, s2, MOVED, '--', rad=-0.3, label='E 20 %', label_at=0.8)
    ax.text(3, 2.95, 'Northern route (N1 + N2) takes 80 %, southern route (S1 + S2) 20 %',
            ha='center', color=INK, fontsize=9)
    _legend(ax, y=-1.05)
    _finish(fig, ax, 'suppress_alternatives.svg', (-0.3, 6.3), (-1.3, 3.2))


def _route_panel(ax, title, right: bool):
    r7 = [(0, 2.0), (1.6, 2.5), (3.2, 3.0), (4.8, 3.5), (6.4, 4.0)]
    r2 = [(0, 0.0), (1.6, 0.5), (3.2, 1.0), (4.8, 1.5), (6.4, 2.0)]
    names7 = ['LEG_7_3_c', 'LEG_7_3_b  (lead)' if right else 'LEG_7_3_b', 'LEG_7_3_a', 'LEG_7_2']
    names2 = ['LEG_2_3_a', 'LEG_2_3_c', 'LEG_2_3_b', 'LEG_2_6']
    mid7 = [_leg(ax, r7[i], r7[i + 1], names7[i], suppressed=True, label_offset=(-0.3, 0.42))
            for i in range(4)]
    mid2 = [_leg(ax, r2[i], r2[i + 1], names2[i], label_offset=(0.25, -0.3)) for i in range(4)]
    ax.text(3.2, 5.0, title, ha='center', color=INK, fontsize=10, fontweight='bold')
    if right:
        lead = mid7[1]
        # Grey arcs run below route 7, the names sit above it.
        for i in (0, 2, 3):
            _arrow(ax, mid7[i], lead, WITH, ':', rad=0.5 if i == 0 else -0.35)
        for m in mid2:
            _arrow(ax, lead, m, MOVED, '--', rad=0.15)
        ax.text(3.2, -0.95, 'Each route-2 leg: +941 ships/year  (correct)', ha='center', color=INK, fontsize=9)
    else:
        for m in mid7:
            _arrow(ax, m, mid2[3], MOVED, '--', rad=0.12)
        ax.text(3.2, -0.95, 'LEG_2_6: +3,764 ships/year  (4 x too many)', ha='center', color=INK, fontsize=9)


def fig_route():
    """Example 3: a whole route -- one lead leg vs targets on every leg."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    _route_panel(ax1, 'Right: one lead leg, the others "with" it', right=True)
    _route_panel(ax2, 'Wrong: targets on every leg', right=False)
    for ax in (ax1, ax2):
        ax.set_xlim(-0.6, 7.0)
        ax.set_ylim(-1.3, 5.3)
        ax.set_aspect('equal')
        ax.axis('off')
    y = -1.25
    ax1.plot([0.0, 0.5], [y - 0.35, y - 0.35], color=WITH, lw=1.6, ls=':')
    ax1.text(0.6, y - 0.35, 'suppressed together with the lead', va='center', color=INK, fontsize=8.5)
    ax1.plot([4.2, 4.7], [y - 0.35, y - 0.35], color=MOVED, lw=1.6, ls='--')
    ax1.text(4.8, y - 0.35, '100 % moved', va='center', color=INK, fontsize=8.5)
    fig.savefig(OUT / 'suppress_route.svg', bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)
    print('wrote', OUT / 'suppress_route.svg')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig_series()
    fig_alternatives()
    fig_route()
