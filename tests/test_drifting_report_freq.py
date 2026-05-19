"""Regression test for the inflated `freq` column in the drifting Ship
Category Breakdown table.

Bug: ``_update_report`` was called once per (cell × obstacle edge hit) and
accumulated ``scat['freq'] += freq`` — so a ship category passing legs
with more obstacles in a given (leg, dir, wind angle) had its annual
frequency multiplied by the obstacle-edge count.  Risk math was
unaffected (cells contribute via ``base`` once per cell).

Run::

    /c/OSGeo4W/apps/Python312/python.exe -m pytest -p no:qgis \
        --noconftest tests/test_drifting_report_freq.py -v
"""
from __future__ import annotations

from compute.drifting_report_builder import DriftingReportBuilderMixin


def _empty_report() -> dict:
    return {
        'by_leg_direction': {},
        'by_object': {},
        'by_structure_legdir': {},
    }


def test_freq_is_invariant_across_obstacle_calls():
    """Calling _update_report multiple times for the same (leg, dir,
    angle, cat) with the same freq must NOT accumulate freq.

    Mimics one ship cell (freq=1000 ships/yr) drifting in one wind
    angle and hitting 5 obstacle edges -- the breakdown table should
    still show 1000 ships/yr, not 5000.
    """
    mixin = DriftingReportBuilderMixin()
    report = _empty_report()
    cell = {'direction': 'East going'}

    for _ in range(5):  # 5 obstacle edges hit
        mixin._update_report(
            report=report,
            event='grounding',
            contrib=0.001,
            idx=0,
            structures=[],
            depths=[{'id': 'd0'}],
            seg_id='1',
            cell=cell,
            d_idx=2,  # wind angle 90 deg
            dist=100.0,
            base=10.0,
            rp=0.125,
            anchor_factor=0.0,
            p_nr=0.5,
            ov_frac=0.05,
            freq=1000.0,
            ship_type=18,
            ship_size=2,
        )

    rec = report['by_leg_direction']['1:East going:90']
    cat = rec['ship_categories']['18-2']
    assert cat['freq'] == 1000.0, (
        f"freq inflated by obstacle-edge count: got {cat['freq']}, "
        f"expected 1000.0"
    )
    # Risk should still accumulate normally.
    assert abs(cat['grounding'] - 5 * 0.001) < 1e-12


def test_freq_independent_across_legs_dirs_angles():
    """Different (leg, dir, angle) records have independent freq values."""
    mixin = DriftingReportBuilderMixin()
    report = _empty_report()
    cell_east = {'direction': 'East going'}
    cell_west = {'direction': 'West going'}

    # Three obstacle hits on leg 1 east-going, angle 0, freq=500.
    for _ in range(3):
        mixin._update_report(
            report=report, event='grounding', contrib=0.0, idx=0,
            structures=[], depths=[{'id': 'd0'}], seg_id='1',
            cell=cell_east, d_idx=0, dist=10.0, base=1.0, rp=0.125,
            anchor_factor=0.0, p_nr=0.0, ov_frac=0.0,
            freq=500.0, ship_type=18, ship_size=2,
        )
    # One hit on leg 2 west-going, angle 45, freq=80.
    mixin._update_report(
        report=report, event='grounding', contrib=0.0, idx=0,
        structures=[], depths=[{'id': 'd0'}], seg_id='2',
        cell=cell_west, d_idx=1, dist=10.0, base=1.0, rp=0.125,
        anchor_factor=0.0, p_nr=0.0, ov_frac=0.0,
        freq=80.0, ship_type=18, ship_size=2,
    )

    assert report['by_leg_direction']['1:East going:0']['ship_categories']['18-2']['freq'] == 500.0
    assert report['by_leg_direction']['2:West going:45']['ship_categories']['18-2']['freq'] == 80.0
