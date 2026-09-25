"""QGIS tests: the junction matrix editor opens after **Update all
distributions** when legs merge or cross (needs the QGIS conftest)."""
from __future__ import annotations

import pytest


def _leg(start, end, name):
    return {'Start_Point': start, 'End_Point': end, 'Leg_name': name, 'Width': 2000, 'Route_Id': 1,
            'Dirs': ['East going', 'West going'], 'Tangent_Pos': 0.5}


@pytest.fixture
def junction_project(omrat):
    """A Y junction (three legs at 14.2 55.0) and a plain bend (two legs at
    15.2 55.0)."""
    saved = omrat.segment_data
    omrat.segment_data = {
        '1': _leg('14.000000 55.000000', '14.200000 55.000000', 'LEG_1_1'),
        '2': _leg('14.200000 55.000000', '14.400000 55.100000', 'LEG_1_2'),
        '3': _leg('14.200000 55.000000', '14.400000 54.900000', 'LEG_2_1'),
        '4': _leg('15.000000 55.000000', '15.200000 55.000000', 'LEG_3_1'),
        '5': _leg('15.200000 55.000000', '15.400000 55.100000', 'LEG_3_2'),
    }
    omrat.junctions.rebuild_from_segments(omrat.segment_data, prefer_user=False)
    yield omrat
    dlg = getattr(omrat, '_junction_matrix_dialog', None)
    if dlg is not None:
        dlg.close()
        omrat._junction_matrix_dialog = None
    omrat.segment_data = saved
    omrat.junctions.rebuild_from_segments(saved, prefer_user=False)


def test_only_merging_or_crossing_junctions_need_review(junction_project):
    from omrat_utils.junction_matrix_dialog import junctions_to_review
    reg = junction_project.junctions.registry
    assert len(reg) == 2
    review = junctions_to_review(reg)
    assert len(review) == 1
    assert set(reg[review[0]].legs) == {'1', '2', '3'}


def test_opens_on_the_junction_to_review(junction_project):
    from omrat_utils.junction_matrix_dialog import junctions_to_review, open_after_update
    om = junction_project
    assert open_after_update(om) is True
    dlg = om._junction_matrix_dialog
    assert dlg.isVisible() and not dlg.isModal()
    wanted = junctions_to_review(om.junctions.registry)[0]
    assert dlg._junction_ids[dlg.cmb.currentIndex()] == wanted


def test_plain_bends_do_not_open_it(junction_project):
    from omrat_utils.junction_matrix_dialog import open_after_update
    om = junction_project
    del om.segment_data['3']                       # the Y becomes a bend
    om.junctions.rebuild_from_segments(om.segment_data, prefer_user=False)
    assert open_after_update(om) is False
    dlg = getattr(om, '_junction_matrix_dialog', None)
    assert dlg is None or not dlg.isVisible()


def test_an_open_editor_is_reloaded(junction_project):
    from omrat_utils.junction_matrix_dialog import open_after_update
    om = junction_project
    open_after_update(om)
    dlg = om._junction_matrix_dialog
    assert dlg.cmb.count() == 2
    om.segment_data['6'] = _leg('15.200000 55.000000', '15.400000 54.900000', 'LEG_4_1')   # bend -> Y
    om.junctions.rebuild_from_segments(om.segment_data, prefer_user=True)
    assert open_after_update(om) is True
    assert om._junction_matrix_dialog is dlg
    assert dlg.cmb.count() == 2
    labels = [dlg.cmb.itemText(i) for i in range(dlg.cmb.count())]
    assert any('LEG_4_1' in lbl for lbl in labels)
