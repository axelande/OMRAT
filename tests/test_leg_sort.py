"""Pure tests for ``omrat_utils.leg_sort`` (natural ordering of legs)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.leg_sort import (  # noqa: E402
    SORTABLE_COLUMNS, natural_key, sort_segment_data,
)


class TestNaturalKey:
    def test_numbers_compare_numerically(self):
        names = ['LEG_1_10', 'LEG_1_2', 'LEG_1_1', 'LEG_10_1', 'LEG_2_1']
        assert sorted(names, key=natural_key) == ['LEG_1_1', 'LEG_1_2', 'LEG_1_10', 'LEG_2_1', 'LEG_10_1']

    def test_sub_leg_suffixes(self):
        names = ['LEG_5_12_b', 'LEG_5_12', 'LEG_5_12_a', 'LEG_5_12_c']
        assert sorted(names, key=natural_key) == ['LEG_5_12', 'LEG_5_12_a', 'LEG_5_12_b', 'LEG_5_12_c']

    def test_case_insensitive_and_none_first(self):
        assert sorted(['b', 'A', None, 'a'], key=natural_key) == [None, 'A', 'a', 'b']

    def test_plain_ints_work(self):
        assert sorted([10, 2, 1], key=natural_key) == [1, 2, 10]


class TestSortSegmentData:
    SD = {
        '3': {'Segment_Id': '3', 'Route_Id': 1, 'Leg_name': 'LEG_1_10'},
        '2': {'Segment_Id': '2', 'Route_Id': 2, 'Leg_name': 'LEG_2_1'},
        '1': {'Segment_Id': '1', 'Route_Id': 1, 'Leg_name': 'LEG_1_2'},
        '10': {'Segment_Id': '10', 'Route_Id': 1, 'Leg_name': 'LEG_1_1'},
    }

    def test_by_leg_name(self):
        out = sort_segment_data(self.SD, 'Leg_name')
        assert list(out) == ['10', '1', '3', '2']

    def test_by_leg_name_descending(self):
        out = sort_segment_data(self.SD, 'Leg_name', reverse=True)
        assert list(out) == ['2', '3', '1', '10']

    def test_by_segment_id_is_numeric(self):
        assert list(sort_segment_data(self.SD, 'Segment_Id')) == ['1', '2', '3', '10']

    def test_by_route_id_ties_broken_by_segment_id(self):
        assert list(sort_segment_data(self.SD, 'Route_Id')) == ['1', '3', '10', '2']

    def test_entries_and_values_preserved(self):
        out = sort_segment_data(self.SD, 'Leg_name')
        assert out == self.SD            # same mapping, different order
        assert out['3'] is self.SD['3']  # not copied

    def test_non_dict_entries_kept_at_end(self):
        sd = dict(self.SD)
        sd['junk'] = 'not a leg'
        out = sort_segment_data(sd, 'Leg_name')
        assert list(out)[-1] == 'junk'

    def test_sortable_columns(self):
        assert SORTABLE_COLUMNS == {0: 'Segment_Id', 1: 'Route_Id', 2: 'Leg_name'}
