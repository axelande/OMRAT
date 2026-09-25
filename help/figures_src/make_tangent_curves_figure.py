"""Figure for the user guide: how the lateral distribution curves are
drawn on the tangent line (``help/source/user_guide.rst``,
``distribution-curves``).

The curve geometry comes from :mod:`geometries.distribution_curves` --
the same functions the plugin uses for the *Tangent Line* layer -- so the
picture cannot drift from the code.  Run with the OSGeo4W Python::

    C:/OSGeo4W/apps/Python312/python.exe help/figures_src/make_tangent_curves_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from geometries.distribution_curves import HEIGHT_FRACTION, curve_points, curve_profiles  # noqa: E402

OUT = ROOT / 'help' / 'source' / '_static' / 'images' / 'tangent_distribution_curves.svg'
LEG = '#1f3fd6'
DIR1 = '#1f5fd6'     # HandleQGISIface.CURVE_COLOURS
DIR2 = '#1a9641'
INK = '#222222'
MUTED = '#666666'

plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})

WIDTH = 6000.0                     # leg width = tangent line length (m)
SEG = {                            # a leg drawn northwards (Dirs = North going, South going)
    'mean1_1': -1200.0, 'std1_1': 600.0, 'weight1_1': 100,   # North going: starboard = east
    'mean2_1': 900.0, 'std2_1': 900.0, 'weight2_1': 100,     # South going: its starboard = west
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    mid, unit = (0.0, 0.0), (0.0, 1.0)            # tangent point, leg drawn south -> north
    half = WIDTH / 2
    # Leg and drawn direction.
    ax.plot([0, 0], [-4200, 4200], color=LEG, lw=3, zorder=2)
    ax.add_patch(FancyArrowPatch((0, 3000), (0, 4300), arrowstyle='-|>,head_length=9,head_width=5',
                                 color=LEG, lw=0, zorder=3))
    ax.text(250, 3600, 'leg, drawn northwards\n(Start -> End)', color=INK, fontsize=9, va='center')
    # Tangent line: from the starboard (east) end to the port (west) end.
    ax.plot([half, -half], [0, 0], color='black', lw=1.6, zorder=2)
    ax.text(half, -260, 'x = -3000 m\nstarboard of the\ndrawn direction', ha='right', va='top',
            color=MUTED, fontsize=8.5)
    ax.text(-half, 200, 'x = +3000 m, port', ha='left', va='bottom', color=MUTED, fontsize=8.5)
    ax.text(-half, 800, 'tangent line', ha='left', va='bottom', color=INK, fontsize=9)

    prof = curve_profiles(SEG, WIDTH)
    for d, colour, name, arrow in ((0, DIR1, 'North going (direction 1)', 1),
                                   (1, DIR2, 'South going (direction 2)', -1)):
        pts = curve_points(mid, unit, prof[d], d)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, lw=2, zorder=4)
        tip = max(pts, key=lambda p: arrow * p[1])
        ax.annotate(name, xy=tip, xytext=(tip[0] + (900 if d == 0 else -900), tip[1] + arrow * 700),
                    ha='left' if d == 0 else 'right', va='center', color=INK, fontsize=9,
                    arrowprops={'arrowstyle': '-', 'color': colour, 'lw': 0.8})
    # Height scale.
    top = HEIGHT_FRACTION * WIDTH
    x_bar = half + 350
    ax.annotate('', xy=(x_bar, top), xytext=(x_bar, 0),
                arrowprops={'arrowstyle': '<->', 'color': MUTED, 'lw': 0.8})
    ax.text(x_bar + 120, top / 2, 'taller peak =\n1/4 of the\nleg width', va='center', color=MUTED, fontsize=8.5)

    ax.set_xlim(-half - 400, half + 2300)
    ax.set_ylim(-4400, 4500)
    ax.set_aspect('equal')           # easting / northing in metres, east to the right
    ax.axis('off')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
