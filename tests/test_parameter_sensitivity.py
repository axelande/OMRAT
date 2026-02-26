"""
Tests verifying how calculation parameters affect drift allision/grounding/anchoring.

Uses the cascade formula from compute/run_calculations.py with controlled inputs
to verify that each parameter produces the expected effect on the result.

The tests are split into two groups:

1. **TestAnchoringEffect** – uses *effective overlap fractions* (hole_pct)
   that represent what the corridor-based computation would give for a large
   anchor zone.  This demonstrates the cascade behaviour: an anchor area
   between the leg and a structure absorbs probability and reduces allision.

2. **TestParameterSensitivity** – verifies scaling laws and monotonicity
   for ship frequency, ship speed, drift speed, anchor success rate,
   anchor-area size, and grounding-area size.

The ``_cascade`` helper replicates the inner loop of
``_iterate_traffic_and_sum`` in ``compute/run_calculations.py`` (lines
~1497-1559)::

    base = (line_length / (speed_kts * 1852)) * freq * (drift_p / (365 * 24))

    Obstacles sorted by distance (closest first):
        anchoring:  anchor_contrib = base * rp * remaining * anchor_p * hole_pct
                    remaining *= (1 - anchor_p * hole_pct)
        allision:   contrib = base * rp * remaining * hole_pct * p_nr
                    remaining *= (1 - hole_pct)
        grounding:  contrib = base * rp * remaining * hole_pct * p_nr
                    remaining *= (1 - hole_pct)

Run with::

    python -m pytest tests/test_parameter_sensitivity.py -v -s --noconftest
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from scipy.stats import norm
from shapely.geometry import Polygon, LineString
import geopandas as gpd

from compute.basic_equations import get_not_repaired
from geometries.calculate_probability_holes import compute_probability_holes

# Set to True (or env SHOW_PLOT=1) to display visual plots during tests
SHOW_PLOT = os.environ.get('SHOW_PLOT', '').lower() in ('1', 'true', 'yes')


# ---------------------------------------------------------------------------
# Default parameters (matching proj.omrat)
# ---------------------------------------------------------------------------

REPAIR_PARAMS = {
    'use_lognormal': 1,
    'std': 0.95,
    'loc': 0.2,
    'scale': 0.85,
}

LINE_LENGTH = 10_000.0       # metres (10 km leg)
DRIFT_SPEED_MS = 1.0         # m/s
FREQ = 1000                  # ships / year
SHIP_SPEED_KTS = 12.0        # knots
DRIFT_P = 1.0                # annual blackout probability
WIND_PROB = 0.125            # uniform wind rose (1/8)

# ---------------------------------------------------------------------------
# Effective overlap fractions for the cascade tests.
#
# In a real scenario the anchor zone is the union of *all* depth cells
# where anchoring is possible – often a very large area that covers most
# of the drift corridor.  The hole_pct for such a zone can be large
# (e.g. 0.30 means 30 % of the lateral distribution hits the anchor zone).
#
# The structure is smaller and farther away, so its hole_pct is lower.
# ---------------------------------------------------------------------------

STRUCT_HOLE     = 0.20    # 20 % – structure is 2 km wide / 10 km leg
STRUCT_DIST     = 7000.0  # metres from leg to structure

ANCHOR_LARGE_HOLE = 1.00  # 100 % – large anchor zone fully covers structure width (3 km ⊃ 2 km)
ANCHOR_SMALL_HOLE = 0.50  # 50 % – small anchor (1 km) covers half the structure width (2 km)
ANCHOR_DIST       = 3000.0  # metres from leg to anchor area (closer than structure)

GROUNDING_LARGE_HOLE = 0.40  # 40 % – grounding large is 4 km wide / 10 km leg
GROUNDING_SMALL_HOLE = 0.20  # 20 % – grounding small is 2 km wide / 10 km leg
GROUNDING_DIST       = 5500.0  # metres – grounding areas sit between anchor and structure

# ---------------------------------------------------------------------------
# Geometry used only by the area-size verification tests
# (these call compute_probability_holes to confirm larger area → larger hole)
# ---------------------------------------------------------------------------

LEG = LineString([(0, 0), (10000, 0)])

ANCHOR_GEOM_LARGE = Polygon([
    (3500, 3000), (6500, 3000), (6500, 5000), (3500, 5000),
])  # 3 km × 2 km

ANCHOR_GEOM_SMALL = Polygon([
    (4500, 3000), (5500, 3000), (5500, 4000), (4500, 4000),
])  # 1 km × 1 km

GROUNDING_GEOM_LARGE = Polygon([
    (3000, 5500), (7000, 5500), (7000, 7000), (3000, 7000),
])  # 4 km × 1.5 km – between anchor zone and structure

GROUNDING_GEOM_SMALL = Polygon([
    (4000, 5500), (6000, 5500), (6000, 6500), (4000, 6500),
])  # 2 km × 1 km

STRUCTURE_GEOM = Polygon([
    (4000, 7000), (6000, 7000), (6000, 9000), (4000, 9000),
])

LATERAL_STD = 1500.0
DRIFT_DISTANCE = 20_000.0
NORTH_DIR_IDX = 2  # 90° math convention


# ---------------------------------------------------------------------------
# Plot helpers (enabled with SHOW_PLOT=1)
# ---------------------------------------------------------------------------

def _plot_test_setup():
    """
    Show the spatial layout used in the tests:
    leg, structure, anchor areas (large & small), drift direction arrow.
    """
    if not SHOW_PLOT:
        return
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # -- Leg -----------------------------------------------------------------
    lx, ly = LEG.xy
    ax.plot(lx, ly, color='black', linewidth=4, solid_capstyle='butt',
            label='Leg (10 km)')
    ax.plot(lx[0], ly[0], 'o', color='black', markersize=8)
    ax.plot(lx[-1], ly[-1], 'o', color='black', markersize=8)

    # -- Structure -----------------------------------------------------------
    sx, sy = STRUCTURE_GEOM.exterior.xy
    ax.fill(sx, sy, alpha=0.35, fc='salmon', ec='red', linewidth=2,
            label=f'Structure (allision target, d={STRUCT_DIST:.0f} m)')

    # -- Anchor large --------------------------------------------------------
    ax_l, ay_l = ANCHOR_GEOM_LARGE.exterior.xy
    ax.fill(ax_l, ay_l, alpha=0.25, fc='dodgerblue', ec='blue', linewidth=2,
            label=f'Anchor LARGE (d={ANCHOR_DIST:.0f} m)')

    # -- Anchor small --------------------------------------------------------
    ax_s, ay_s = ANCHOR_GEOM_SMALL.exterior.xy
    ax.fill(ax_s, ay_s, alpha=0.35, fc='cornflowerblue', ec='blue',
            linewidth=2, linestyle='--',
            label='Anchor SMALL')

    # -- Grounding large -----------------------------------------------------
    gx_l, gy_l = GROUNDING_GEOM_LARGE.exterior.xy
    ax.fill(gx_l, gy_l, alpha=0.20, fc='orange', ec='darkorange', linewidth=2,
            label=f'Grounding LARGE (d={GROUNDING_DIST:.0f} m)')

    # -- Grounding small -----------------------------------------------------
    gx_s, gy_s = GROUNDING_GEOM_SMALL.exterior.xy
    ax.fill(gx_s, gy_s, alpha=0.35, fc='gold', ec='darkorange',
            linewidth=2, linestyle='--',
            label='Grounding SMALL')

    # -- Drift direction arrow -----------------------------------------------
    arrow_x = 1500
    ax.annotate('', xy=(arrow_x, 6500), xytext=(arrow_x, 500),
                arrowprops=dict(arrowstyle='->', color='green',
                                lw=3, mutation_scale=20))
    ax.text(arrow_x + 200, 3500, 'Drift\n(North)', fontsize=11,
            color='green', fontweight='bold', va='center')

    # -- Distance annotations ------------------------------------------------
    mid_x = 8500
    ax.annotate('', xy=(mid_x, 3000), xytext=(mid_x, 0),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(mid_x + 150, 1500, f'{ANCHOR_DIST:.0f} m', fontsize=9,
            color='gray', va='center')

    ax.annotate('', xy=(mid_x, 7000), xytext=(mid_x, 0),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(mid_x + 150, 3800, f'{STRUCT_DIST:.0f} m', fontsize=9,
            color='gray', va='center')

    ax.set_xlim(-500, 11500)
    ax.set_ylim(-1500, 10500)
    ax.set_aspect('equal')
    ax.set_xlabel('x  (metres)')
    ax.set_ylabel('y  (metres)')
    ax.set_title('Test Geometry Setup – Drift North from Leg')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_anchoring_cascade(r_no: dict, r_yes: dict, anchor_p: float):
    """
    Bar chart comparing cascade results without / with anchoring.
    """
    if not SHOW_PLOT:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    labels = ['No anchoring', f'With anchoring\n(anchor_p={anchor_p})']
    allision_vals = [r_no['allision'], r_yes['allision']]
    anchor_vals = [0.0, r_yes['anchoring']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_a = ax.bar(x - width / 2, allision_vals, width, label='Allision',
                    color='salmon', edgecolor='red')
    bars_anch = ax.bar(x + width / 2, anchor_vals, width, label='Anchoring absorbed',
                       color='dodgerblue', edgecolor='blue')

    # annotate values
    for bar in bars_a:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                f'{h:.2e}', ha='center', va='bottom', fontsize=9)
    for bar in bars_anch:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                    f'{h:.2e}', ha='center', va='bottom', fontsize=9)

    # reduction annotation
    reduction_pct = (1.0 - r_yes['allision'] / r_no['allision']) * 100
    ax.annotate(
        f'{reduction_pct:.1f} % reduction',
        xy=(1 - width / 2, r_yes['allision']),
        xytext=(0.3, (r_no['allision'] + r_yes['allision']) / 2),
        fontsize=11, fontweight='bold', color='green',
        arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
    )

    ax.set_ylabel('Annual probability')
    ax.set_title(
        f'Anchoring Effect on Allision\n'
        f'(anchor hole_pct={ANCHOR_LARGE_HOLE}, '
        f'struct hole_pct={STRUCT_HOLE})'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_parameter_bars(title: str, labels: list[str], values: list[float],
                         ylabel: str = 'Annual probability'):
    """Generic side-by-side bar chart for parameter comparisons."""
    if not SHOW_PLOT:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color='steelblue', edgecolor='navy', width=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                f'{h:.2e}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_area_comparison(
    title: str,
    geom_small,
    geom_large,
    area_label: str,
    area_color: str,
    area_ec: str,
    bar_labels: list[str],
    bar_values: list[float],
    bar_title: str,
    show_structure: bool = True,
):
    """Side-by-side: geometry layout (left) + bar chart (right) for area-size tests."""
    if not SHOW_PLOT:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax_geo, ax_bar) = plt.subplots(1, 2, figsize=(15, 6))

    # -- Left: geometry layout -----------------------------------------------
    lx, ly = LEG.xy
    ax_geo.plot(lx, ly, color='black', linewidth=4, solid_capstyle='butt',
                label='Leg (10 km)')

    # Large area
    gx, gy = geom_large.exterior.xy
    ax_geo.fill(gx, gy, alpha=0.25, fc=area_color, ec=area_ec, linewidth=2,
                label=f'{area_label} LARGE')

    # Small area
    sx, sy = geom_small.exterior.xy
    ax_geo.fill(sx, sy, alpha=0.45, fc=area_color, ec=area_ec, linewidth=2,
                linestyle='--', label=f'{area_label} SMALL')

    if show_structure:
        stx, sty = STRUCTURE_GEOM.exterior.xy
        ax_geo.fill(stx, sty, alpha=0.30, fc='salmon', ec='red', linewidth=2,
                    label='Structure')

    # Drift arrow
    ax_geo.annotate('', xy=(1500, 6500), xytext=(1500, 500),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax_geo.text(1700, 3500, 'Drift\n(North)', fontsize=10,
                color='green', fontweight='bold', va='center')

    ax_geo.set_xlim(-500, 11000)
    ax_geo.set_ylim(-1500, 10500)
    ax_geo.set_aspect('equal')
    ax_geo.set_xlabel('x  (metres)')
    ax_geo.set_ylabel('y  (metres)')
    ax_geo.set_title('Geometry Layout')
    ax_geo.legend(loc='upper left', fontsize=8)
    ax_geo.grid(True, alpha=0.3)

    # -- Right: bar chart ----------------------------------------------------
    x = np.arange(len(bar_labels))
    bars = ax_bar.bar(x, bar_values, color='steelblue', edgecolor='navy', width=0.5)

    for bar in bars:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                    f'{h:.2e}', ha='center', va='bottom', fontsize=10)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bar_labels)
    ax_bar.set_ylabel('Annual probability')
    ax_bar.set_title(bar_title)
    ax_bar.grid(True, axis='y', alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def _print_equations(
    obstacles: list[tuple[str, float, float, float]],
    *,
    freq: float = FREQ,
    speed_kts: float = SHIP_SPEED_KTS,
    drift_p: float = DRIFT_P,
    rp: float = WIND_PROB,
    repair: dict | None = None,
    drift_speed_ms: float = DRIFT_SPEED_MS,
    line_length: float = LINE_LENGTH,
    label: str = '',
):
    """Print step-by-step cascade equations with numeric values."""
    if repair is None:
        repair = REPAIR_PARAMS

    hours = line_length / (speed_kts * 1852.0)
    blackout_rate = drift_p / (365.0 * 24.0)
    base = hours * freq * blackout_rate

    print(f"\n{'='*70}")
    if label:
        print(f"  CASCADE EQUATIONS — {label}")
    else:
        print(f"  CASCADE EQUATIONS")
    print(f"{'='*70}")
    print(f"  base = (L / (v * 1852)) * N * (P_drift / (365 * 24))")
    print(f"       = ({line_length:.0f} / ({speed_kts:.1f} * 1852)) "
          f"* {freq:.0f} * ({drift_p:.4f} / 8760)")
    print(f"       = {hours:.6f} * {freq:.0f} * {blackout_rate:.8f}")
    print(f"       = {base:.6f}")
    print()

    remaining = 1.0
    allision = grounding = anchoring = 0.0
    sorted_obs = sorted(obstacles, key=lambda x: x[1])

    for obs_type, dist, hole_pct, anchor_p in sorted_obs:
        if remaining <= 0.0:
            break

        if obs_type == 'anchoring':
            c = base * rp * remaining * anchor_p * hole_pct
            anchoring += c
            print(f"  [{obs_type.upper()}]  distance = {dist:.0f} m")
            print(f"    contrib = base * rp * remaining * anchor_p * hole_pct")
            print(f"            = {base:.6f} * {rp:.4f} * {remaining:.6f} "
                  f"* {anchor_p:.2f} * {hole_pct:.2f}")
            print(f"            = {c:.6e}")
            new_remaining = remaining * (1.0 - anchor_p * hole_pct)
            print(f"    remaining = {remaining:.6f} * (1 - {anchor_p:.2f} * {hole_pct:.2f})")
            print(f"              = {remaining:.6f} * {1.0 - anchor_p * hole_pct:.6f}")
            print(f"              = {new_remaining:.6f}")
            remaining = new_remaining
        else:
            p_nr = get_not_repaired(repair, drift_speed_ms, dist)
            c = base * rp * remaining * hole_pct * p_nr
            drift_time_h = (dist / drift_speed_ms) / 3600.0
            if obs_type == 'allision':
                allision += c
            else:
                grounding += c

            print(f"  [{obs_type.upper()}]  distance = {dist:.0f} m")
            print(f"    drift_time = {dist:.0f} / {drift_speed_ms:.1f} / 3600 "
                  f"= {drift_time_h:.4f} hours")
            print(f"    P(not repaired) = 1 - lognorm.cdf({drift_time_h:.4f}, "
                  f"s={repair['std']}, loc={repair['loc']}, scale={repair['scale']})")
            print(f"                    = {p_nr:.6f}")
            print(f"    contrib = base * rp * remaining * hole_pct * P_nr")
            print(f"            = {base:.6f} * {rp:.4f} * {remaining:.6f} "
                  f"* {hole_pct:.2f} * {p_nr:.6f}")
            print(f"            = {c:.6e}")
            new_remaining = remaining * (1.0 - hole_pct)
            print(f"    remaining = {remaining:.6f} * (1 - {hole_pct:.2f}) "
                  f"= {new_remaining:.6f}")
            remaining = new_remaining
        print()

    print(f"  TOTALS:  allision = {allision:.6e}")
    print(f"           grounding = {grounding:.6e}")
    print(f"           anchoring = {anchoring:.6e}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hole(
    leg: LineString,
    obstacle,
    lateral_std: float = LATERAL_STD,
    distance: float = DRIFT_DISTANCE,
    dir_idx: int = NORTH_DIR_IDX,
) -> float:
    """Compute probability-hole for a single (leg, obstacle, direction)."""
    dist_obj = norm(loc=0, scale=lateral_std)
    holes = compute_probability_holes(
        [leg],
        [[dist_obj]],
        [[1.0]],
        [gpd.GeoDataFrame(geometry=[obstacle])],
        distance=distance,
    )
    return holes[0][dir_idx][0]


def _cascade(
    obstacles: list[tuple[str, float, float, float]],
    *,
    freq: float = FREQ,
    speed_kts: float = SHIP_SPEED_KTS,
    drift_p: float = DRIFT_P,
    rp: float = WIND_PROB,
    repair: dict | None = None,
    drift_speed_ms: float = DRIFT_SPEED_MS,
    line_length: float = LINE_LENGTH,
) -> dict[str, float]:
    """
    Run the cascade for one leg / direction.

    Mirrors ``_iterate_traffic_and_sum`` in ``compute/run_calculations.py``.

    Parameters
    ----------
    obstacles : list of (obs_type, distance_m, hole_pct, anchor_p)
        ``obs_type`` in ``{'anchoring', 'allision', 'grounding'}``.
        Sorted by distance internally (closest first).
    """
    if repair is None:
        repair = REPAIR_PARAMS

    blackout_per_hour = drift_p / (365.0 * 24.0)
    hours_present = (line_length / (speed_kts * 1852.0)) * freq
    base = hours_present * blackout_per_hour

    remaining = 1.0
    allision = grounding = anchoring = 0.0

    for obs_type, dist, hole_pct, anchor_p in sorted(obstacles, key=lambda x: x[1]):
        if remaining <= 0.0:
            break

        if obs_type == 'anchoring':
            c = base * rp * remaining * anchor_p * hole_pct
            anchoring += c
            remaining *= (1.0 - anchor_p * hole_pct)
        else:
            p_nr = get_not_repaired(repair, drift_speed_ms, dist)
            c = base * rp * remaining * hole_pct * p_nr
            if obs_type == 'allision':
                allision += c
            else:
                grounding += c
            remaining *= (1.0 - hole_pct)

    return {'allision': allision, 'grounding': grounding, 'anchoring': anchoring}


# ---------------------------------------------------------------------------
# TestAnchoringEffect
# ---------------------------------------------------------------------------

class TestAnchoringEffect:
    """
    Verify that an anchoring area placed between the leg and a structure
    reduces the allision probability.

    Uses effective overlap fractions (hole_pct) that represent a realistic
    large anchor zone covering most of the drift corridor.
    """

    def test_baseline_allision_positive(self):
        """Without anchoring, allision probability is positive."""
        _plot_test_setup()

        r = _cascade([
            ('allision', STRUCT_DIST, STRUCT_HOLE, 0.0),
        ])
        assert r['allision'] > 0
        print(f"\nBaseline allision (no anchoring): {r['allision']:.4e}")

    def test_anchoring_reduces_allision(self):
        """
        An anchor area (hole_pct=1.0, covering 100% of structure width) with
        anchor_p=0.95 should dramatically reduce allision.

        Expected: remaining = 1 - 0.95 * 1.0 = 0.05
        → allision drops to ~5 % of the baseline.
        """
        obs_no = [('allision', STRUCT_DIST, STRUCT_HOLE, 0.0)]
        r_no = _cascade(obs_no)

        anchor_p = 0.95
        obs_yes = [
            ('anchoring', ANCHOR_DIST, ANCHOR_LARGE_HOLE, anchor_p),
            ('allision', STRUCT_DIST, STRUCT_HOLE, 0.0),
        ]
        r_yes = _cascade(obs_yes)

        ratio = r_yes['allision'] / r_no['allision']
        expected_remaining = 1.0 - anchor_p * ANCHOR_LARGE_HOLE  # 0.715

        # Show full equation breakdown
        _print_equations(obs_no, label='WITHOUT anchoring')
        _print_equations(obs_yes, label='WITH anchoring (anchor_p=0.95)')

        print(f"\nAllision without anchoring: {r_no['allision']:.4e}")
        print(f"Allision with anchoring:    {r_yes['allision']:.4e}")
        print(f"Anchoring absorbed:         {r_yes['anchoring']:.4e}")
        print(f"Allision ratio:             {ratio:.4f}  (expected ~{expected_remaining:.3f})")

        assert r_yes['allision'] < r_no['allision'], \
            "Anchoring area must reduce allision probability"
        assert abs(ratio - expected_remaining) < 0.01, \
            f"Ratio should be ~{expected_remaining:.3f}, got {ratio:.4f}"
        assert r_yes['anchoring'] > 0, \
            "Some probability must be absorbed by anchoring"

        _plot_anchoring_cascade(r_no, r_yes, anchor_p)

    def test_anchoring_absorbs_probability(self):
        """
        The probability absorbed by anchoring should explain the allision
        reduction.  With anchor_p=0.95 and hole_pct=1.0 (100% coverage) the
        anchor zone absorbs ~95 % of the drifting ships before they reach
        the structure.
        """
        anchor_p = 0.95

        r_no = _cascade([
            ('allision', STRUCT_DIST, STRUCT_HOLE, 0.0),
        ])
        r_yes = _cascade([
            ('anchoring', ANCHOR_DIST, ANCHOR_LARGE_HOLE, anchor_p),
            ('allision', STRUCT_DIST, STRUCT_HOLE, 0.0),
        ])

        reduction = r_no['allision'] - r_yes['allision']
        assert reduction > 0, "Allision must decrease with anchoring"
        assert r_yes['anchoring'] > 0, "Anchoring contribution must be positive"

        # The reduction should be a substantial fraction of the original
        pct_reduction = reduction / r_no['allision'] * 100
        print(f"\nAllision reduction:  {reduction:.4e}  ({pct_reduction:.1f} %)")
        print(f"Anchoring absorbed:  {r_yes['anchoring']:.4e}")
        assert pct_reduction > 20, \
            f"Expected >20 % reduction, got {pct_reduction:.1f} %"


# ---------------------------------------------------------------------------
# TestParameterSensitivity
# ---------------------------------------------------------------------------

class TestParameterSensitivity:
    """
    Verify that each calculation parameter affects the result
    in the expected direction and with the correct scaling.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up reference obstacle lists with effective overlap fractions."""
        self.obs_allision_only = [
            ('allision', STRUCT_DIST, STRUCT_HOLE, 0.0),
        ]
        self.obs_with_anchor = [
            ('anchoring', ANCHOR_DIST, ANCHOR_LARGE_HOLE, 0.95),
            ('allision', STRUCT_DIST, STRUCT_HOLE, 0.0),
        ]

    # -- ship frequency -------------------------------------------------

    def test_frequency_scales_linearly(self):
        """Doubling ship frequency doubles the allision probability."""
        r1 = _cascade(self.obs_allision_only, freq=500)
        r2 = _cascade(self.obs_allision_only, freq=1000)

        ratio = r2['allision'] / r1['allision']

        print(f"\nfreq=500  -> allision={r1['allision']:.4e}")
        print(f"freq=1000 -> allision={r2['allision']:.4e}")
        print(f"ratio = {ratio:.6f} (expected 2.0)")

        assert abs(ratio - 2.0) < 1e-6, f"Expected ratio 2.0, got {ratio}"

        _plot_parameter_bars(
            'Ship Frequency Effect (linear)',
            ['freq = 500', 'freq = 1000'],
            [r1['allision'], r2['allision']],
        )

    # -- ship speed -----------------------------------------------------

    def test_ship_speed_inversely_proportional(self):
        """Doubling ship speed halves the allision probability."""
        r1 = _cascade(self.obs_allision_only, speed_kts=10.0)
        r2 = _cascade(self.obs_allision_only, speed_kts=20.0)

        ratio = r1['allision'] / r2['allision']

        print(f"\nspeed=10 kts -> allision={r1['allision']:.4e}")
        print(f"speed=20 kts -> allision={r2['allision']:.4e}")
        print(f"ratio = {ratio:.6f} (expected 2.0)")

        assert abs(ratio - 2.0) < 1e-6, f"Expected ratio 2.0, got {ratio}"

        _plot_parameter_bars(
            'Ship Speed Effect (inverse)',
            ['speed = 10 kts', 'speed = 20 kts'],
            [r1['allision'], r2['allision']],
        )

    # -- drift speed ----------------------------------------------------

    def test_drift_speed_increases_probability(self):
        """Faster drift -> less time to repair -> higher P(not repaired) -> higher allision."""
        r_slow = _cascade(self.obs_allision_only, drift_speed_ms=0.5)
        r_fast = _cascade(self.obs_allision_only, drift_speed_ms=2.0)

        print(f"\ndrift=0.5 m/s -> allision={r_slow['allision']:.4e}")
        print(f"drift=2.0 m/s -> allision={r_fast['allision']:.4e}")

        assert r_fast['allision'] > r_slow['allision'], \
            "Faster drift speed should increase allision probability"

        _plot_parameter_bars(
            'Drift Speed Effect (faster drift = higher probability)',
            ['drift = 0.5 m/s', 'drift = 2.0 m/s'],
            [r_slow['allision'], r_fast['allision']],
        )

    # -- anchor success rate --------------------------------------------

    def test_anchor_success_rate_reduces_allision(self):
        """Higher anchor_p absorbs more probability -> less allision behind."""
        obs_50 = [
            ('anchoring', ANCHOR_DIST, ANCHOR_LARGE_HOLE, 0.50),
            ('allision', STRUCT_DIST, STRUCT_HOLE, 0.0),
        ]
        obs_95 = [
            ('anchoring', ANCHOR_DIST, ANCHOR_LARGE_HOLE, 0.95),
            ('allision', STRUCT_DIST, STRUCT_HOLE, 0.0),
        ]

        r_50 = _cascade(obs_50)
        r_95 = _cascade(obs_95)

        print(f"\nanchor_p=0.50 -> allision={r_50['allision']:.4e}, "
              f"anchoring={r_50['anchoring']:.4e}")
        print(f"anchor_p=0.95 -> allision={r_95['allision']:.4e}, "
              f"anchoring={r_95['anchoring']:.4e}")

        assert r_95['allision'] < r_50['allision'], \
            "Higher anchor success rate should reduce allision probability"
        assert r_95['anchoring'] > r_50['anchoring'], \
            "Higher anchor success rate should increase anchoring absorption"

        _plot_parameter_bars(
            'Anchor Success Rate Effect on Allision',
            ['anchor_p = 0.50', 'anchor_p = 0.95'],
            [r_50['allision'], r_95['allision']],
        )

    # -- anchor area size -----------------------------------------------

    def test_anchor_area_size_reduces_allision(self):
        """
        Larger anchor area -> larger hole_pct -> more anchoring -> less allision.

        Part 1: verify via compute_probability_holes that a geometrically
        larger polygon produces a larger hole_pct.
        Part 2: verify the cascade effect with the two different sizes.
        """
        # Part 1 – geometric verification
        small_geo_hole = _compute_hole(LEG, ANCHOR_GEOM_SMALL)
        large_geo_hole = _compute_hole(LEG, ANCHOR_GEOM_LARGE)

        print(f"\nGeometric check (compute_probability_holes):")
        print(f"  Small anchor hole_pct: {small_geo_hole:.6f}")
        print(f"  Large anchor hole_pct: {large_geo_hole:.6f}")
        assert large_geo_hole > small_geo_hole, \
            "Larger anchor geometry should have larger probability hole"

        # Part 2 – cascade with effective fractions
        anchor_p = 0.95
        r_small = _cascade([
            ('anchoring', ANCHOR_DIST, ANCHOR_SMALL_HOLE, anchor_p),
            ('allision', STRUCT_DIST, STRUCT_HOLE, 0.0),
        ])
        r_large = _cascade([
            ('anchoring', ANCHOR_DIST, ANCHOR_LARGE_HOLE, anchor_p),
            ('allision', STRUCT_DIST, STRUCT_HOLE, 0.0),
        ])

        print(f"\nCascade effect:")
        print(f"  Small anchor (hole={ANCHOR_SMALL_HOLE}) -> allision={r_small['allision']:.4e}")
        print(f"  Large anchor (hole={ANCHOR_LARGE_HOLE}) -> allision={r_large['allision']:.4e}")

        assert r_large['allision'] < r_small['allision'], \
            "Larger anchor area should reduce allision probability more"

        _plot_area_comparison(
            title='Anchor Area Size Effect on Allision',
            geom_small=ANCHOR_GEOM_SMALL,
            geom_large=ANCHOR_GEOM_LARGE,
            area_label='Anchor',
            area_color='dodgerblue',
            area_ec='blue',
            bar_labels=[f'Small anchor\n(hole={ANCHOR_SMALL_HOLE})',
                        f'Large anchor\n(hole={ANCHOR_LARGE_HOLE})'],
            bar_values=[r_small['allision'], r_large['allision']],
            bar_title='Allision probability',
            show_structure=True,
        )

    # -- grounding area size --------------------------------------------

    def test_grounding_area_size_increases_grounding(self):
        """
        Larger grounding area -> larger hole_pct -> more grounding.

        Part 1: geometric verification.
        Part 2: cascade verification.
        """
        # Part 1 – geometric verification
        small_geo_hole = _compute_hole(LEG, GROUNDING_GEOM_SMALL)
        large_geo_hole = _compute_hole(LEG, GROUNDING_GEOM_LARGE)

        print(f"\nGeometric check (compute_probability_holes):")
        print(f"  Small grounding hole: {small_geo_hole:.6f}")
        print(f"  Large grounding hole: {large_geo_hole:.6f}")
        assert large_geo_hole > small_geo_hole, \
            "Larger grounding geometry should have larger probability hole"

        # Part 2 – cascade with effective fractions
        r_small = _cascade([
            ('grounding', GROUNDING_DIST, GROUNDING_SMALL_HOLE, 0.0),
        ])
        r_large = _cascade([
            ('grounding', GROUNDING_DIST, GROUNDING_LARGE_HOLE, 0.0),
        ])

        print(f"\nCascade effect:")
        print(f"  Small grounding (hole={GROUNDING_SMALL_HOLE}) -> {r_small['grounding']:.4e}")
        print(f"  Large grounding (hole={GROUNDING_LARGE_HOLE}) -> {r_large['grounding']:.4e}")

        assert r_large['grounding'] > r_small['grounding'], \
            "Larger grounding area should produce more grounding probability"

        _plot_area_comparison(
            title='Grounding Area Size Effect',
            geom_small=GROUNDING_GEOM_SMALL,
            geom_large=GROUNDING_GEOM_LARGE,
            area_label='Grounding',
            area_color='orange',
            area_ec='darkorange',
            bar_labels=[f'Small grounding\n(hole={GROUNDING_SMALL_HOLE})',
                        f'Large grounding\n(hole={GROUNDING_LARGE_HOLE})'],
            bar_values=[r_small['grounding'], r_large['grounding']],
            bar_title='Grounding probability',
            show_structure=False,
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '--noconftest'])
