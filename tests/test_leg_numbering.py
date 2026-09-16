"""Pure tests for ``omrat_utils.leg_numbering`` -- the split between the
global ``Segment_Id`` key and the per-route leg number in ``Leg_name``."""
from omrat_utils.leg_numbering import (
    leg_name, max_leg_number, next_free_segment_id, parse_leg_number,
)


def _sd(*legs):
    out = {}
    for sid, route, name in legs:
        out[str(sid)] = {'Segment_Id': sid, 'Route_Id': route, 'Leg_name': name}
    return out


def test_next_free_segment_id_skips_every_used_key():
    sd = _sd((1, 1, 'LEG_1_1'), (2, 1, 'LEG_1_2'), (7, 1, 'LEG_1_3'))
    assert next_free_segment_id(sd) == 8
    # A reset "next id" of 1 (floor 0) must not resurrect key 1.
    assert next_free_segment_id(sd, floor=0) == 8
    # The floor wins when the handler has already gone past the dict.
    assert next_free_segment_id(sd, floor=12) == 13


def test_next_free_segment_id_reads_segment_id_values_too():
    sd = {'3': {'Segment_Id': 9}, 'x': {'Segment_Id': 'y'}, '4': 'not a dict'}
    assert next_free_segment_id(sd) == 10


def test_next_free_segment_id_empty_project_starts_at_one():
    assert next_free_segment_id({}) == 1
    assert next_free_segment_id(None) == 1


def test_parse_leg_number_handles_split_suffixes():
    assert parse_leg_number('LEG_2_5') == 5
    assert parse_leg_number('LEG_2_5_a') == 5
    assert parse_leg_number('LEG_2_5_a_b') == 5
    assert parse_leg_number('LEG_2_5', route_id=2) == 5
    assert parse_leg_number('LEG_2_5', route_id=1) is None
    assert parse_leg_number('Kattegat north') is None
    assert parse_leg_number(None) is None


def test_max_leg_number_is_per_route():
    sd = _sd((1, 1, 'LEG_1_1'), (2, 1, 'LEG_1_2'), (3, 2, 'LEG_2_1'), (4, 2, 'LEG_2_2_b'))
    assert max_leg_number(sd, 1) == 2
    assert max_leg_number(sd, 2) == 2
    assert max_leg_number(sd, 3) == 0
    assert max_leg_number({}, 1) == 0


def test_max_leg_number_ignores_free_text_names_on_the_route():
    sd = _sd((1, 1, 'Approach'), (2, 1, 'LEG_1_4'))
    assert max_leg_number(sd, 1) == 4


def test_max_leg_number_trusts_name_when_route_id_disagrees():
    # Hand-edited file: Route_Id says 1 but the name says route 2.
    sd = _sd((1, 1, 'LEG_2_7'))
    assert max_leg_number(sd, 2) == 7
    assert max_leg_number(sd, 1) == 0


def test_leg_name_format():
    assert leg_name(3, 1) == 'LEG_3_1'
    assert leg_name('3', '12') == 'LEG_3_12'
