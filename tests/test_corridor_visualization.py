# -*- coding: utf-8 -*-
"""
Visual test for drift corridors with shadows, blocking and anchoring areas.

Loads proj.omrat data and renders corridors for all 8 directions per leg,
showing:
    - Depth obstacles (grounding risk) coloured by depth
    - Structure obstacles (allision risk) highlighted
    - Full (unclipped) corridor outline
    - Blocked / shadow zone (red)
    - GREEN corridor zone: deep water where the ship drifts freely and cannot
      anchor (depth >= anchor_d * draught)
    - BLUE corridor zone: water shallow enough for anchoring (depth < anchor_d
      * draught).  The ship CAN attempt to anchor here (anchor_p = 95 %
      success in proj.omrat), but may also fail and keep drifting.

Run with:
    SHOW_PLOT=1 python -m pytest tests/test_corridor_visualization.py -s -k test_all_corridors

Ship parameters (configurable at the top of the file):
    SHIP_HEIGHT = 12  metres (air draught, determines structure threshold)
    SHIP_DRAUGHT = 6  metres (determines depth threshold)
"""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, MultiPolygon, LineString, box
from shapely.ops import transform as shapely_transform, unary_union
from shapely.validation import make_valid
from shapely import wkt as shapely_wkt

from geometries.drift.constants import DIRECTIONS
from geometries.drift.coordinates import compass_to_vector, get_utm_crs, transform_geometry
from geometries.drift.corridor import create_projected_corridor
from geometries.drift.shadow import create_obstacle_shadow, extract_polygons
from geometries.drift.clipping import clip_corridor_at_obstacles, split_corridor_by_anchor_zone
from geometries.drift.distribution import get_projection_distance, get_distribution_width

# ---------------------------------------------------------------------------
# Ship parameters -- adjust as needed
# ---------------------------------------------------------------------------
SHIP_HEIGHT = 12    # metres – structures with height <= this are obstacles
SHIP_DRAUGHT = 6    # metres – depth cells with max_depth <= this are obstacles

# Lateral distribution std (metres) – from segment_data Width / coverage
LATERAL_STD = 1500  # reasonable default; proj.omrat Width=15000 -> ~15000/5.15=2913

# Show plots flag (set env SHOW_PLOT=1 to enable)
SHOW_PLOT = os.environ.get('SHOW_PLOT', '').lower() in ('1', 'true', 'yes')


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_proj_omrat():
    """Load and return the parsed proj.omrat JSON."""
    proj_file = Path(__file__).parent / "example_data" / "proj.omrat"
    if not proj_file.exists():
        pytest.skip("proj.omrat not found")
    with open(proj_file, "r") as f:
        return json.load(f)


def _get_legs_wgs84(data: dict) -> dict[str, LineString]:
    """Extract legs as {name: LineString} in WGS84."""
    legs = {}
    for seg_id, seg in data['segment_data'].items():
        sp = seg['Start_Point'].split()
        ep = seg['End_Point'].split()
        legs[f'Leg{seg_id}'] = LineString([
            (float(sp[0]), float(sp[1])),
            (float(ep[0]), float(ep[1])),
        ])
    return legs


def _get_depth_obstacles_wgs84(data: dict, depth_threshold: float):
    """
    Return depth obstacles as [(Polygon/MultiPolygon, depth_value), ...].

    Depths are bin-lower-bounds with bin_width=3m.
    A depth cell is an obstacle when (depth_value + bin_width) <= threshold.
    """
    bin_width = 3.0
    obstacles = []
    for d in data.get('depths', []):
        depth_value = float(d[1])
        max_depth = depth_value + bin_width
        if max_depth > depth_threshold:
            continue
        geom = shapely_wkt.loads(d[2])
        geom = make_valid(geom)
        if not geom.is_empty:
            obstacles.append((geom, depth_value))
    return obstacles


def _get_structure_obstacles_wgs84(data: dict, height_threshold: float):
    """
    Return structure obstacles as [(Polygon, height), ...].

    Structures with height <= threshold are obstacles.
    """
    obstacles = []
    for obj in data.get('objects', []):
        height = float(obj[1])
        if height > height_threshold:
            continue
        geom = shapely_wkt.loads(obj[2])
        geom = make_valid(geom)
        if not geom.is_empty:
            obstacles.append((geom, height))
    return obstacles


def _to_utm(geom, wgs84, utm_crs):
    """Transform a Shapely geometry from WGS84 to UTM."""
    g = transform_geometry(geom, wgs84, utm_crs)
    return make_valid(g)


def _flatten_multipolygons(obstacles):
    """Split MultiPolygon entries into individual polygons, keeping values."""
    flat = []
    for geom, val in obstacles:
        if isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                if not p.is_empty:
                    flat.append((make_valid(p), val))
        elif isinstance(geom, Polygon) and not geom.is_empty:
            flat.append((geom, val))
    return flat


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_geom(ax, geom, **kwargs):
    """Plot a Polygon or MultiPolygon on ax."""
    if geom is None or geom.is_empty:
        return
    polys = extract_polygons(geom) if not isinstance(geom, Polygon) else [geom]
    for p in polys:
        x, y = p.exterior.xy
        ax.fill(x, y, **kwargs)


def _add_compass_arrow(ax, angle_deg, label, xc, yc, arrow_len):
    """Draw a small compass arrow showing drift direction."""
    dx, dy = compass_to_vector(angle_deg, arrow_len)
    ax.annotate(
        '', xy=(xc + dx, yc + dy), xytext=(xc, yc),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
    )
    ax.text(xc + dx * 1.15, yc + dy * 1.15, label,
            ha='center', va='center', fontsize=7, fontweight='bold')


# ---------------------------------------------------------------------------
# The test class
# ---------------------------------------------------------------------------

class TestCorridorVisualization:
    """
    Visual integration test that loads proj.omrat data, builds corridors
    for all legs / directions, clips them at depth + structure obstacles,
    and renders the result with matplotlib.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load data and transform to UTM."""
        data = _load_proj_omrat()

        # --- coordinate systems ---
        self.wgs84 = CRS("EPSG:4326")
        # use a representative point to pick the UTM zone
        first_leg = list(_get_legs_wgs84(data).values())[0]
        c = first_leg.centroid
        self.utm_crs = get_utm_crs(c.x, c.y)

        # --- legs ---
        self.legs_wgs84 = _get_legs_wgs84(data)
        self.legs_utm = {
            name: _to_utm(leg, self.wgs84, self.utm_crs)
            for name, leg in self.legs_wgs84.items()
        }

        # --- depth obstacles (grounding) ---
        depth_obs_wgs = _get_depth_obstacles_wgs84(data, SHIP_DRAUGHT)
        self.depth_obstacles_utm = [
            (_to_utm(g, self.wgs84, self.utm_crs), v)
            for g, v in depth_obs_wgs
        ]
        # also keep all depths for context
        all_depth_wgs = [(shapely_wkt.loads(d[2]), float(d[1])) for d in data['depths']]
        self.all_depths_utm = [
            (_to_utm(make_valid(g), self.wgs84, self.utm_crs), v)
            for g, v in all_depth_wgs
        ]

        # --- structure obstacles (allision) ---
        struct_obs_wgs = _get_structure_obstacles_wgs84(data, SHIP_HEIGHT)
        self.structure_obstacles_utm = [
            (_to_utm(g, self.wgs84, self.utm_crs), v)
            for g, v in struct_obs_wgs
        ]

        # --- combined obstacle list (for clipping) ---
        self.combined_obstacles_utm = (
            _flatten_multipolygons(self.depth_obstacles_utm) +
            _flatten_multipolygons(self.structure_obstacles_utm)
        )

        # --- drift parameters ---
        drift = data['drift']
        repair_params = {
            'use_lognormal': drift['repair'].get('use_lognormal', True),
            'std': drift['repair'].get('std', 0.95),
            'loc': drift['repair'].get('loc', 0.2),
            'scale': drift['repair'].get('scale', 0.85),
        }
        drift_speed_ms = drift['speed'] * 1852.0 / 3600.0  # knots -> m/s

        self.half_width = get_distribution_width(LATERAL_STD, 0.99) / 2
        self.projection_dist = get_projection_distance(
            repair_params, drift_speed_ms, target_prob=1e-3
        )
        self.projection_dist = min(self.projection_dist, 50_000)

        # --- anchoring zone ---
        # anchor_d * draught = threshold depth for anchoring
        # (see compute/run_calculations.py line ~961)
        anchor_d = float(drift.get('anchor_d', 0))
        self.anchor_p = float(drift.get('anchor_p', 0))
        self.anchor_threshold = anchor_d * SHIP_DRAUGHT  # metres

        # Collect all depth bins whose lower-bound < anchor_threshold
        # Their union defines the area where anchoring is possible
        anchor_geoms_utm = []
        for d in data.get('depths', []):
            depth_value = float(d[1])
            if depth_value < self.anchor_threshold:
                geom = make_valid(shapely_wkt.loads(d[2]))
                geom_utm = _to_utm(geom, self.wgs84, self.utm_crs)
                if not geom_utm.is_empty:
                    anchor_geoms_utm.append(geom_utm)

        if anchor_geoms_utm:
            self.anchor_zone_utm = make_valid(unary_union(anchor_geoms_utm))
        else:
            self.anchor_zone_utm = Polygon()

        print(f"\n--- Corridor parameters ---")
        print(f"  Ship height:  {SHIP_HEIGHT} m (structure threshold)")
        print(f"  Ship draught: {SHIP_DRAUGHT} m (depth threshold)")
        print(f"  Half-width:   {self.half_width:.0f} m")
        print(f"  Projection:   {self.projection_dist:.0f} m")
        print(f"  Depth obstacles:     {len(self.depth_obstacles_utm)}")
        print(f"  Structure obstacles: {len(self.structure_obstacles_utm)}")
        print(f"  Combined (flat):     {len(self.combined_obstacles_utm)}")
        print(f"  Anchor threshold:    {self.anchor_threshold:.0f} m "
              f"(anchor_d={anchor_d}, draught={SHIP_DRAUGHT}m)")
        print(f"  Anchor zone area:    {self.anchor_zone_utm.area / 1e6:.1f} km\u00b2")
        print(f"  Anchor success prob: {self.anchor_p}")

    # ------------------------------------------------------------------
    # Test: build all corridors and render
    # ------------------------------------------------------------------

    def test_all_corridors(self):
        """
        Build corridors for every leg/direction, clip at obstacles,
        and create a multi-page visualization.
        """
        # direction names in compass convention (matches DIRECTIONS keys)
        dir_names = list(DIRECTIONS.keys())   # N, NW, W, SW, S, SE, E, NE
        dir_angles = list(DIRECTIONS.values())

        # --- compute corridors ---
        corridors = {}   # (leg_name, dir_name) -> {'full': ..., 'clipped': ..., 'blue': ..., 'green': ...}
        for leg_name, leg_utm in self.legs_utm.items():
            for dname, dangle in zip(dir_names, dir_angles):
                full_corridor = create_projected_corridor(
                    leg_utm, self.half_width, dangle, self.projection_dist
                )
                clipped_corridor = clip_corridor_at_obstacles(
                    full_corridor, self.combined_obstacles_utm, dangle
                )
                blue, green = split_corridor_by_anchor_zone(
                    clipped_corridor, self.anchor_zone_utm,
                    dangle, full_corridor.bounds,
                )
                corridors[(leg_name, dname)] = {
                    'full': full_corridor,
                    'clipped': clipped_corridor,
                    'blue': blue,
                    'green': green,
                    'angle': dangle,
                }

        # --- print summary ---
        print("\n--- Corridor summary ---")
        for leg_name in self.legs_utm:
            for dname in dir_names:
                key = (leg_name, dname)
                full_a = corridors[key]['full'].area
                clip_a = corridors[key]['clipped'].area if not corridors[key]['clipped'].is_empty else 0
                pct = clip_a / full_a * 100 if full_a > 0 else 0
                if full_a - clip_a > 1:
                    print(f"  {leg_name:6} {dname:3}: full={full_a/1e6:.2f} km2, "
                          f"clipped={clip_a/1e6:.2f} km2 ({pct:.0f}% remaining)")

        # --- assertions ---
        # At least one corridor should be clipped (obstacles exist)
        any_clipped = any(
            corridors[k]['clipped'].area < corridors[k]['full'].area * 0.99
            for k in corridors
            if corridors[k]['full'].area > 0
        )
        assert any_clipped, "Expected at least one corridor to be clipped by obstacles"

        # Clipped area should never exceed full area
        for k, v in corridors.items():
            if v['full'].area > 0:
                assert v['clipped'].area <= v['full'].area + 1, \
                    f"Clipped area exceeds full area for {k}"

        # --- visualise ---
        self._plot_overview(corridors, dir_names, dir_angles)
        self._plot_per_leg(corridors, dir_names, dir_angles)

    # ------------------------------------------------------------------
    # Plotting: overview
    # ------------------------------------------------------------------

    def _plot_overview(self, corridors, dir_names, dir_angles):
        """Single figure showing all legs, obstacles, and corridors."""
        if not SHOW_PLOT:
            return
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
            import matplotlib.colors as mcolors
        except ImportError:
            print("matplotlib not available")
            return

        fig, ax = plt.subplots(figsize=(18, 14))

        # -- depth obstacles (all, coloured by depth) --
        cmap = plt.cm.YlOrBr
        max_d = max((v for _, v in self.all_depths_utm), default=1)
        for geom, depth_val in self.all_depths_utm:
            norm = depth_val / max_d if max_d > 0 else 0
            _plot_geom(ax, geom, fc=cmap(norm), ec='none', alpha=0.25)

        # -- highlight obstacle depths in blue --
        for geom, depth_val in self.depth_obstacles_utm:
            _plot_geom(ax, geom, fc='royalblue', ec='navy', alpha=0.35, linewidth=0.5)

        # -- structure obstacles in red --
        for geom, height in self.structure_obstacles_utm:
            _plot_geom(ax, geom, fc='tomato', ec='darkred', alpha=0.6, linewidth=1)

        # -- legs --
        leg_colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
        for idx, (leg_name, leg_utm) in enumerate(self.legs_utm.items()):
            lx, ly = leg_utm.xy
            c = leg_colors[idx % len(leg_colors)]
            ax.plot(lx, ly, color=c, linewidth=3, label=leg_name, zorder=5)
            ax.plot(lx[0], ly[0], 'o', color=c, markersize=8, zorder=6)
            ax.plot(lx[-1], ly[-1], '^', color=c, markersize=8, zorder=6)

        # -- corridors: green (deep) / blue (anchorable) --
        corr_alpha = 0.10
        for leg_name in self.legs_utm:
            for dname in dir_names:
                v = corridors[(leg_name, dname)]
                blue = v['blue']
                green = v['green']
                if not green.is_empty:
                    _plot_geom(ax, green, fc='#2ecc71', ec='none',
                               alpha=corr_alpha, linewidth=0)
                if not blue.is_empty:
                    _plot_geom(ax, blue, fc='#3498db', ec='none',
                               alpha=corr_alpha, linewidth=0)

        # -- legend --
        legend_patches = [
            Patch(fc='royalblue', ec='navy', alpha=0.5,
                  label=f'Depth obstacle (max depth \u2264 {SHIP_DRAUGHT}m)'),
            Patch(fc='tomato', ec='darkred', alpha=0.6,
                  label=f'Structure obstacle (height \u2264 {SHIP_HEIGHT}m)'),
            Patch(fc='#2ecc71', ec='#2ecc71', alpha=0.4,
                  label='Corridor \u2013 deep water (no anchoring)'),
            Patch(fc='#3498db', ec='#3498db', alpha=0.4,
                  label=f'Corridor \u2013 anchorable (depth < {self.anchor_threshold:.0f}m)'),
        ]
        ax.legend(handles=legend_patches + ax.get_legend_handles_labels()[0],
                  loc='upper left', fontsize=9)
        ax.set_title(
            f'Overview: All legs and obstacles\n'
            f'Ship draught={SHIP_DRAUGHT}m, height={SHIP_HEIGHT}m  |  '
            f'half-width={self.half_width:.0f}m, projection={self.projection_dist:.0f}m',
            fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Plotting: per-leg detail (8 subplots per leg)
    # ------------------------------------------------------------------

    def _plot_per_leg(self, corridors, dir_names, dir_angles):
        """One figure per leg with 8 subplots (one per direction)."""
        if not SHOW_PLOT:
            return
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
        except ImportError:
            print("matplotlib not available")
            return

        for leg_name, leg_utm in self.legs_utm.items():
            fig, axes = plt.subplots(2, 4, figsize=(22, 12))
            axes = axes.flatten()

            for ax_idx, (dname, dangle) in enumerate(zip(dir_names, dir_angles)):
                ax = axes[ax_idx]
                v = corridors[(leg_name, dname)]
                full_corr = v['full']
                clip_corr = v['clipped']

                # Determine view extent from full corridor + some margin
                if not full_corr.is_empty:
                    bx = full_corr.bounds
                    margin = max(bx[2] - bx[0], bx[3] - bx[1]) * 0.1
                    ax.set_xlim(bx[0] - margin, bx[2] + margin)
                    ax.set_ylim(bx[1] - margin, bx[3] + margin)

                # -- precomputed green / blue zones --
                blue_zone = v['blue']
                green_zone = v['green']

                # -- full corridor outline (dashed) --
                if not full_corr.is_empty:
                    for p in extract_polygons(full_corr):
                        fx, fy = p.exterior.xy
                        ax.plot(fx, fy, '--', color='gray', linewidth=1, alpha=0.6)

                # -- blocked area = full minus clipped (red fill) --
                if not full_corr.is_empty and not clip_corr.is_empty:
                    try:
                        blocked = full_corr.difference(clip_corr)
                        blocked = make_valid(blocked)
                        if not blocked.is_empty:
                            _plot_geom(ax, blocked, fc='#e74c3c', ec='none',
                                       alpha=0.2)
                    except Exception:
                        pass

                # -- green zone (deep water, free drift) --
                if not green_zone.is_empty:
                    _plot_geom(ax, green_zone, fc='#2ecc71', ec='#27ae60',
                               alpha=0.35, linewidth=0.5)

                # -- blue zone (anchorable water) --
                if not blue_zone.is_empty:
                    _plot_geom(ax, blue_zone, fc='#3498db', ec='#2980b9',
                               alpha=0.35, linewidth=0.5)

                # -- depth obstacles --
                for geom, depth_val in self.depth_obstacles_utm:
                    _plot_geom(ax, geom, fc='royalblue', ec='navy',
                               alpha=0.4, linewidth=0.3)

                # -- structure obstacles --
                for geom, height in self.structure_obstacles_utm:
                    _plot_geom(ax, geom, fc='tomato', ec='darkred',
                               alpha=0.6, linewidth=0.8)

                # -- the leg itself --
                lx, ly = leg_utm.xy
                ax.plot(lx, ly, color='black', linewidth=2.5, zorder=5)
                ax.plot(lx[0], ly[0], 'ko', markersize=6, zorder=6)
                ax.plot(lx[-1], ly[-1], 'k^', markersize=6, zorder=6)

                # -- compass arrow --
                cx = (leg_utm.bounds[0] + leg_utm.bounds[2]) / 2
                cy = (leg_utm.bounds[1] + leg_utm.bounds[3]) / 2
                arrow_len = min(full_corr.bounds[2] - full_corr.bounds[0],
                                full_corr.bounds[3] - full_corr.bounds[1]) * 0.08 if not full_corr.is_empty else 500
                _add_compass_arrow(ax, dangle, dname, cx, cy, arrow_len)

                # -- stats annotation --
                full_a = full_corr.area / 1e6
                clip_a = clip_corr.area / 1e6 if not clip_corr.is_empty else 0
                green_a = green_zone.area / 1e6 if not green_zone.is_empty else 0
                blue_a = blue_zone.area / 1e6 if not blue_zone.is_empty else 0
                pct = clip_a / full_a * 100 if full_a > 0 else 0
                ax.text(0.02, 0.97,
                        f'Full:    {full_a:.1f} km\u00b2\n'
                        f'Green:   {green_a:.1f} km\u00b2 (deep)\n'
                        f'Blue:    {blue_a:.1f} km\u00b2 (anchor)\n'
                        f'Blocked: {full_a - clip_a:.1f} km\u00b2\n'
                        f'Remain:  {pct:.0f}%',
                        transform=ax.transAxes, fontsize=7, va='top',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round', fc='wheat', alpha=0.85))

                ax.set_title(f'{dname} ({dangle}\u00b0)', fontsize=10)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.2)

            # -- figure legend --
            legend_patches = [
                Patch(fc='#2ecc71', ec='#27ae60', alpha=0.5,
                      label='Deep water \u2013 free drift (no anchoring)'),
                Patch(fc='#3498db', ec='#2980b9', alpha=0.5,
                      label=f'Anchorable water (depth < {self.anchor_threshold:.0f}m, '
                            f'P(anchor)={self.anchor_p})'),
                Patch(fc='#e74c3c', ec='none', alpha=0.3,
                      label='Blocked / shadow'),
                Patch(fc='royalblue', ec='navy', alpha=0.5,
                      label=f'Depth obstacle (grounding \u2264 {SHIP_DRAUGHT}m)'),
                Patch(fc='tomato', ec='darkred', alpha=0.6,
                      label=f'Structure (\u2264 {SHIP_HEIGHT}m)'),
            ]
            fig.legend(handles=legend_patches, loc='lower center',
                       ncol=3, fontsize=9, framealpha=0.9)
            fig.suptitle(
                f'{leg_name} \u2014 drift corridors (8 directions)\n'
                f'Ship: draught={SHIP_DRAUGHT}m, height={SHIP_HEIGHT}m  |  '
                f'half-width={self.half_width:.0f}m, '
                f'projection={self.projection_dist:.0f}m  |  '
                f'anchor threshold={self.anchor_threshold:.0f}m',
                fontsize=13, fontweight='bold')
            plt.tight_layout(rect=[0, 0.06, 1, 0.93])
            plt.show()

    # ------------------------------------------------------------------
    # Standalone test: depth obstacle coverage sanity
    # ------------------------------------------------------------------

    def test_depth_obstacle_thresholds(self):
        """Verify correct number of depth obstacles for the chosen draught."""
        # With bin_width=3 and draught=6:
        #   depth=0 -> max=3 <= 6 -> include
        #   depth=3 -> max=6 <= 6 -> include
        #   depth=6 -> max=9 > 6  -> exclude
        assert len(self.depth_obstacles_utm) >= 1, \
            f"Expected at least 1 depth obstacle for draught {SHIP_DRAUGHT}m"

        # All included depths should have (value + 3) <= SHIP_DRAUGHT
        for geom, val in self.depth_obstacles_utm:
            assert val + 3 <= SHIP_DRAUGHT, \
                f"Depth {val}m (max {val + 3}m) should not exceed draught {SHIP_DRAUGHT}m"

        print(f"\nDepth obstacles included: {len(self.depth_obstacles_utm)}")
        for geom, val in self.depth_obstacles_utm:
            print(f"  depth={val}m (max={val + 3}m), area={geom.area / 1e6:.2f} km2")

    def test_structure_obstacle_thresholds(self):
        """Verify correct structures are included for the chosen ship height."""
        assert len(self.structure_obstacles_utm) >= 1, \
            f"Expected at least 1 structure obstacle for height {SHIP_HEIGHT}m"

        for geom, height in self.structure_obstacles_utm:
            assert height <= SHIP_HEIGHT, \
                f"Structure height {height}m exceeds ship height {SHIP_HEIGHT}m"

        print(f"\nStructure obstacles included: {len(self.structure_obstacles_utm)}")
        for geom, height in self.structure_obstacles_utm:
            print(f"  height={height}m, area={geom.area / 1e6:.2f} km2")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '--noconftest'])
