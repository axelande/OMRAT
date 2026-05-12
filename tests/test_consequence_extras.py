"""Extra Consequence-module tests covering audit-flagged gaps.

Adds:

* Coverage for the new public :func:`validate_consequence` function.
* Edge cases the audit flagged as missing in
  :func:`compute_catastrophe_exceedance` itself: malformed cell keys,
  duplicate catastrophe thresholds, all-zero traffic, NaN/inf inputs.
"""

from __future__ import annotations

import math

import pytest

from compute.consequence import (
    ConsequenceValidation,
    compute_catastrophe_exceedance,
    validate_consequence,
)
from omrat_utils.consequence_defaults import (
    default_catastrophe_levels,
    default_oil_onboard,
    default_spill_fraction,
    default_spill_probability,
)


# ---------------------------------------------------------------------------
# validate_consequence
# ---------------------------------------------------------------------------


def _good_block() -> dict:
    return {
        'oil_onboard': default_oil_onboard(['Cargo'], [{'min': 0, 'max': 50}]),
        'spill_probability': default_spill_probability(),
        'spill_fraction': default_spill_fraction(),
        'catastrophe_levels': default_catastrophe_levels(),
    }


def test_validate_passes_default_block():
    rep = validate_consequence(_good_block())
    assert isinstance(rep, ConsequenceValidation)
    assert rep.ok
    assert not rep.errors


def test_validate_fails_when_input_not_dict():
    rep = validate_consequence(None)
    assert not rep.ok
    assert any('missing' in e for e in rep.errors)


def test_validate_flags_spill_probability_row_not_summing_100():
    block = _good_block()
    block['spill_probability'][0] = [50, 25, 25, 0]  # sums to 100 (good)
    block['spill_probability'][1] = [40, 20, 20, 10]  # sums to 90 (bad)
    rep = validate_consequence(block)
    assert not rep.ok
    assert any('row 1' in e and 'sums to 90' in e for e in rep.errors)


def test_validate_flags_minimum_two_catastrophe_levels():
    block = _good_block()
    block['catastrophe_levels'] = [{'name': 'only', 'quantity': 100}]
    rep = validate_consequence(block)
    assert not rep.ok
    assert any('at least 2' in e for e in rep.errors)


def test_validate_flags_negative_quantity():
    block = _good_block()
    block['catastrophe_levels'].append({'name': 'bad', 'quantity': -1})
    rep = validate_consequence(block)
    assert not rep.ok
    assert any('positive' in e for e in rep.errors)


def test_validate_warns_on_duplicate_quantity():
    block = _good_block()
    q = block['catastrophe_levels'][0]['quantity']
    block['catastrophe_levels'].append({'name': 'dup', 'quantity': q})
    rep = validate_consequence(block)
    assert rep.ok  # duplicates are warnings, not errors
    assert any('duplicate' in w for w in rep.warnings)


def test_validate_flags_negative_oil_onboard():
    block = _good_block()
    block['oil_onboard'][0][0] = -10.0
    rep = validate_consequence(block)
    assert not rep.ok
    assert any('negative' in e for e in rep.errors)


def test_validate_flags_spill_fraction_out_of_range():
    block = _good_block()
    block['spill_fraction'][0] = [0, 10, 30, 150]  # 150% > 100
    rep = validate_consequence(block)
    assert not rep.ok
    assert any('outside [0, 100]' in e for e in rep.errors)


# ---------------------------------------------------------------------------
# Edge cases in compute_catastrophe_exceedance
# ---------------------------------------------------------------------------


def _trivial_consequence() -> dict:
    return {
        'oil_onboard': [[100.0, 200.0], [50.0, 75.0]],
        'spill_probability': [[100, 0, 0, 0]] * 8,
        'spill_fraction': [[0, 10, 30, 100]] * 8,
        'catastrophe_levels': [
            {'name': 'small', 'quantity': 50.0},
            {'name': 'big', 'quantity': 500.0},
        ],
    }


def test_compute_handles_malformed_cell_key():
    cons = _trivial_consequence()
    # Cell key with wrong format -> silently skipped, no crash.
    drifting_report = {
        'by_cell_grounding': {'not_a_cell': 0.5, '0_0': 1.0, '0': 0.5},
    }
    out = compute_catastrophe_exceedance(
        cons, drifting_report=drifting_report,
    )
    assert isinstance(out, dict)
    assert 'levels' in out


def test_compute_with_all_zero_traffic_returns_zero_exceedance():
    cons = _trivial_consequence()
    drifting_report = {'by_cell_grounding': {'0_0': 0.0, '1_1': 0.0}}
    out = compute_catastrophe_exceedance(cons, drifting_report=drifting_report)
    assert all(lvl['exceedance'] == 0.0 for lvl in out['levels'])


def test_compute_handles_negative_freq_silently():
    cons = _trivial_consequence()
    drifting_report = {'by_cell_grounding': {'0_0': -1.0, '1_1': 1.0}}
    out = compute_catastrophe_exceedance(cons, drifting_report=drifting_report)
    # Negative is treated like zero (skipped); only the positive cell counts.
    assert out['by_accident']['Drifting grounding']['frequency'] == pytest.approx(1.0)


def test_compute_duplicate_threshold_double_counts():
    """Documented behaviour: a duplicate level is processed twice."""
    cons = _trivial_consequence()
    cons['catastrophe_levels'].append({'name': 'big_dup', 'quantity': 500.0})
    drifting_report = {'by_cell_grounding': {'0_0': 1.0}}
    out = compute_catastrophe_exceedance(cons, drifting_report=drifting_report)
    big_levels = [l for l in out['levels'] if l['quantity'] == 500.0]
    assert len(big_levels) == 2  # both produce a row
    # And both fire identically.
    assert big_levels[0]['exceedance'] == big_levels[1]['exceedance']


def test_compute_with_empty_reports_returns_zero_exceedance():
    cons = _trivial_consequence()
    out = compute_catastrophe_exceedance(cons)
    assert all(lvl['exceedance'] == 0.0 for lvl in out['levels'])
    assert out['total_spill_frequency'] == 0.0


def test_compute_with_no_consequence_block_returns_empty_result():
    out = compute_catastrophe_exceedance({})
    assert out['levels'] == []
    assert out['total_spill_frequency'] == 0.0


def test_compute_skips_cells_with_invalid_freq_value():
    cons = _trivial_consequence()
    # Non-numeric freq -> skipped via try/except.
    drifting_report = {'by_cell_grounding': {'0_0': 'not_a_number'}}
    out = compute_catastrophe_exceedance(cons, drifting_report=drifting_report)
    assert all(lvl['exceedance'] == 0.0 for lvl in out['levels'])


def test_compute_lookup_oil_uses_zero_for_out_of_bounds_cell():
    cons = _trivial_consequence()  # oil_onboard is 2x2
    # Cell key references shipped type 5 / length 5 -> beyond table -> oil=0.
    drifting_report = {'by_cell_grounding': {'5_5': 1.0}}
    out = compute_catastrophe_exceedance(cons, drifting_report=drifting_report)
    # Frequency still counted but spill volume 0 -> no exceedance.
    assert out['by_accident']['Drifting grounding']['frequency'] == pytest.approx(1.0)
    assert all(lvl['exceedance'] == 0.0 for lvl in out['levels'])


def test_compute_handles_inf_freq_value():
    cons = _trivial_consequence()
    drifting_report = {'by_cell_grounding': {'0_0': math.inf}}
    out = compute_catastrophe_exceedance(cons, drifting_report=drifting_report)
    # inf propagates through linear math, but the function should still
    # return a structured result rather than crashing.
    assert isinstance(out, dict)
    assert 'levels' in out
