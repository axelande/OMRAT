"""Unit tests for the pure-Python helpers in omrat_utils/handle_object.py.

The ``OObject`` class is QGIS-dependent and exercised via
``tests/test_load_data.py`` + the live cascade.  This file focuses on
the module-level helpers that don't touch QGIS widgets.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.handle_object import (
    build_gebco_url,
    expand_bbox,
    get_bbox,
    get_depth_color,
    get_leg_coordinates,
)


# ---------------------------------------------------------------------------
# get_leg_coordinates
# ---------------------------------------------------------------------------

class TestGetLegCoordinates:
    def test_reads_start_and_end_from_table(self):
        tbl = MagicMock()
        tbl.rowCount.return_value = 2
        def item(row, col):
            grid = {
                (0, 3): '14.0 55.0', (0, 4): '14.5 55.1',
                (1, 3): '14.5 55.1', (1, 4): '15.0 55.2',
            }
            return SimpleNamespace(text=lambda row=row, col=col: grid[(row, col)])
        tbl.item.side_effect = item

        coords = get_leg_coordinates(tbl)
        assert coords == [(14.0, 55.0), (14.5, 55.1), (14.5, 55.1), (15.0, 55.2)]

    def test_none_cells_skipped(self):
        tbl = MagicMock()
        tbl.rowCount.return_value = 1
        tbl.item.side_effect = lambda r, c: None
        assert get_leg_coordinates(tbl) == []

    def test_malformed_coord_skipped(self):
        tbl = MagicMock()
        tbl.rowCount.return_value = 1
        def item(row, col):
            grid = {
                (0, 3): 'nonsense', (0, 4): '14.5 55.1',
            }
            return SimpleNamespace(text=lambda r=row, c=col: grid.get((r, c), ''))
        tbl.item.side_effect = item
        # Malformed start is skipped; end is taken.
        assert get_leg_coordinates(tbl) == [(14.5, 55.1)]


# ---------------------------------------------------------------------------
# get_bbox / expand_bbox
# ---------------------------------------------------------------------------

class TestGetBbox:
    def test_returns_min_max_lat_lon(self):
        coords = [(14.0, 55.0), (14.5, 55.5), (13.8, 54.9)]
        min_lat, max_lat, min_lon, max_lon = get_bbox(coords)
        assert min_lat == 54.9 and max_lat == 55.5
        assert min_lon == 13.8 and max_lon == 14.5

    def test_single_point(self):
        assert get_bbox([(10.0, 20.0)]) == (20.0, 20.0, 10.0, 10.0)


class TestExpandBbox:
    def test_zero_percent_expansion_returns_same_bbox(self):
        assert expand_bbox(10, 20, 30, 40, 0) == (10.0, 20.0, 30.0, 40.0)

    def test_10_percent_expansion(self):
        # lat range = 10, lon range = 10, 10% = 1 each side.
        out = expand_bbox(10, 20, 30, 40, 10)
        assert out == (9.0, 21.0, 29.0, 41.0)

    def test_handles_zero_range(self):
        # Degenerate lat range -> no expansion.
        out = expand_bbox(55.0, 55.0, 14.0, 15.0, 10)
        assert out[0] == out[1] == 55.0

# ---------------------------------------------------------------------------
# get_depth_color
# ---------------------------------------------------------------------------

class TestGetDepthColor:
    def test_zero_depth_is_dark_blue(self, qgis_app):
        c = get_depth_color(0.0, max_depth=50.0)
        # dark blue ~ (0, 0, 139)
        assert c.red() == 0 and c.green() == 0
        assert 100 < c.blue() < 160

    def test_max_depth_is_light_blue(self, qgis_app):
        c = get_depth_color(50.0, max_depth=50.0)
        # light blue ~ (200, 220, 255)
        assert c.red() >= 180 and c.green() >= 200 and c.blue() >= 240

    def test_intermediate_depth_blends_proportionally(self, qgis_app):
        c0 = get_depth_color(0.0, max_depth=50.0)
        c_half = get_depth_color(25.0, max_depth=50.0)
        c_full = get_depth_color(50.0, max_depth=50.0)
        # Half should fall between the two endpoints.
        assert c0.red() <= c_half.red() <= c_full.red()
        assert c0.blue() <= c_half.blue() <= c_full.blue()

    def test_depth_above_max_clamped(self, qgis_app):
        """A depth above max_depth uses max_depth's color."""
        c_above = get_depth_color(1e6, max_depth=50.0)
        c_max = get_depth_color(50.0, max_depth=50.0)
        assert (c_above.red(), c_above.green(), c_above.blue()) == (
            c_max.red(), c_max.green(), c_max.blue())

    def test_negative_depth_clamped_to_zero(self, qgis_app):
        c_neg = get_depth_color(-5.0, max_depth=50.0)
        c_zero = get_depth_color(0.0, max_depth=50.0)
        assert (c_neg.red(), c_neg.green(), c_neg.blue()) == (
            c_zero.red(), c_zero.green(), c_zero.blue())


# ---------------------------------------------------------------------------
# build_gebco_url
# ---------------------------------------------------------------------------

class TestBuildGebcoUrl:
    def test_builds_url_with_all_params(self):
        url = build_gebco_url(
            min_lat=54.0, max_lat=56.0, min_lon=13.0, max_lon=16.0,
            api_key='SECRET',
        )
        assert 'south=54.0' in url
        assert 'north=56.0' in url
        assert 'west=13.0' in url
        assert 'east=16.0' in url
        assert 'API_Key=SECRET' in url
        assert url.startswith('https://portal.opentopography.org/')

    def test_empty_api_key_still_forms_url(self):
        url = build_gebco_url(0, 1, 0, 1, '')
        assert 'API_Key=' in url
