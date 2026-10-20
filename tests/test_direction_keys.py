"""Traffic direction keys vs a re-derived ``Dirs`` (2026-10-16).

A vertex drag re-derives ``segment_data[leg]['Dirs']`` from the new
bearing (90-degree naming sectors on the raw lon/lat azimuth).  The keys
of ``traffic_data[leg]`` kept their old names, so a leg dragged across a
sector boundary -- or any north/south leg of an IWRAP-imported project,
which names every leg East/West -- exported to IWRAP with no traffic.
Compute reads the blocks by position and was never affected.

Example pinned here: ``testEO08v2.omrat`` LEG_2_6 (7 467 ships/year),
end vertex dragged ~1.6 km east, true bearing 27.2 -> 29.3 deg, names
North/South -> East/West.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.iwrap_convertion import generate_iwrap_xml  # noqa: E402
from omrat_utils.project_sanitize import rename_direction_keys  # noqa: E402

EXAMPLE = ROOT / 'tests' / 'example_data' / 'testEO08v2.omrat'
LEG = '6'


def _exported(data: dict, leg: str) -> dict[str, float]:
    root = generate_iwrap_xml(data)
    out = {}
    for td in root.iter('traffic_distribution'):
        name = td.get('name')
        if name in (f'TD_{leg}_FTL', f'TD_{leg}_LTF'):
            out[name[-3:]] = round(sum(float(c.get('freq', 0)) for c in td.iter('category')), 1)
    return out


@pytest.fixture(scope='module')
def project() -> dict:
    with open(EXAMPLE, encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def dragged(project) -> dict:
    """LEG_2_6 after the drag: ``Dirs`` re-derived, traffic keys not."""
    data = copy.deepcopy(project)
    seg = data['segment_data'][LEG]
    sx, sy = map(float, seg['Start_Point'].split())
    _ex, ey = map(float, seg['End_Point'].split())
    ex = sx + (ey - sy) * math.tan(math.radians(45.5))     # raw azimuth 45.5 deg
    seg['End_Point'] = f'{ex:.6f} {ey:.6f}'
    seg['Dirs'] = ['East going', 'West going']
    return data


class TestRenameDirectionKeys:
    def test_renames_by_position_in_place(self):
        a, b = {'Q': 1}, {'Q': 2}
        block = {'North going': a, 'South going': b}
        same = block
        assert rename_direction_keys(block, ['East going', 'West going']) is True
        assert block is same
        assert list(block) == ['East going', 'West going']
        assert block['East going'] is a and block['West going'] is b

    def test_no_change(self):
        block = {'East going': {}, 'West going': {}}
        assert rename_direction_keys(block, ['East going', 'West going']) is False
        assert list(block) == ['East going', 'West going']

    def test_reversed_names_keep_the_data_at_its_position(self):
        a, b = {'Q': 1}, {'Q': 2}
        block = {'North going': a, 'South going': b}
        assert rename_direction_keys(block, ['South going', 'North going']) is True
        assert list(block) == ['South going', 'North going']
        assert block['South going'] is a and block['North going'] is b

    def test_non_direction_entries_untouched(self):
        block = {'categories': [1, 2], 'North going': {'Q': 1}, 'South going': {'Q': 2}}
        rename_direction_keys(block, ['East going', 'West going'])
        assert list(block) == ['categories', 'East going', 'West going']
        assert block['categories'] == [1, 2]

    @pytest.mark.parametrize('block, dirs', [(None, ['A', 'B']), ({}, ['A', 'B']), ({'A': {}}, [])])
    def test_nothing_to_do(self, block, dirs):
        assert rename_direction_keys(block, dirs) is False


class TestIwrapExportByPosition:
    def test_example_before_the_drag(self, project):
        assert _exported(project, LEG) == {'FTL': 4185.0, 'LTF': 3282.0}

    def test_dragged_leg_keeps_its_traffic(self, dragged):
        # Before the fix: {'FTL': 0, 'LTF': 0}.
        assert list(dragged['traffic_data'][LEG]) == ['North going', 'South going']
        assert _exported(dragged, LEG) == {'FTL': 4185.0, 'LTF': 3282.0}

    def test_renamed_keys_export_the_same(self, dragged):
        rename_direction_keys(dragged['traffic_data'][LEG], dragged['segment_data'][LEG]['Dirs'])
        assert list(dragged['traffic_data'][LEG]) == ['East going', 'West going']
        assert _exported(dragged, LEG) == {'FTL': 4185.0, 'LTF': 3282.0}

    def test_follows_block_order_like_compute(self, project):
        # Keys stored opposite to the Dirs order: FTL is block 0, as in compute.
        data = copy.deepcopy(project)
        blk = data['traffic_data'][LEG]
        data['traffic_data'][LEG] = {'South going': blk['South going'], 'North going': blk['North going']}
        assert _exported(data, LEG) == {'FTL': 3282.0, 'LTF': 4185.0}


class TestComputeUnaffected:
    def test_collision_totals_identical_after_rename(self, dragged):
        from omrat_utils.compare_explain import run_collision_model
        renamed = copy.deepcopy(dragged)
        rename_direction_keys(renamed['traffic_data'][LEG], renamed['segment_data'][LEG]['Dirs'])
        before = run_collision_model(copy.deepcopy(dragged))
        after = run_collision_model(renamed)
        keys = ('head_on', 'overtaking', 'crossing', 'merging', 'bend')
        tb, ta = before.get('totals', {}), after.get('totals', {})
        assert any(tb.get(k) for k in keys)
        for k in keys:
            assert ta.get(k) == pytest.approx(tb.get(k), rel=1e-12)
