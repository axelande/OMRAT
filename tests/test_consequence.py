"""Unit tests for ``compute/consequence.py`` and the consequence defaults.

Standalone (no QGIS); run with::

    /mnt/c/OSGeo4W/apps/Python312/python.exe -m pytest \\
        -p no:qgis --noconftest tests/test_consequence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from anywhere -- mirrors what other standalone tests do.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.consequence import compute_catastrophe_exceedance  # noqa: E402
from omrat_utils.consequence_defaults import (  # noqa: E402
    ACCIDENT_KEYS, ACCIDENT_TYPES,
    default_catastrophe_levels,
    default_oil_onboard,
    default_spill_fraction,
    default_spill_probability,
    reshape_oil_onboard,
)


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------
def test_default_oil_onboard_tanker_uses_avg_length():
    types = ['Other', 'Tanker', 'Chemical/gas tanker']
    intervals = [
        {'min': 0.0, 'max': 50.0, 'label': '0 - 50'},
        {'min': 200.0, 'max': 'inf', 'label': '200 - inf'},
    ]
    matrix = default_oil_onboard(types, intervals)
    # Non-tanker row defaults to 100 in every column.
    assert matrix[0] == [100.0, 100.0]
    # Tanker row uses 80 * midpoint, with open interval => min + 50.
    assert matrix[1] == [80.0 * 25.0, 80.0 * 250.0]
    # Substring match: "Chemical/gas tanker" is also a tanker.
    assert matrix[2] == [80.0 * 25.0, 80.0 * 250.0]


def test_default_spill_probability_rows_sum_to_100():
    matrix = default_spill_probability()
    assert len(matrix) == len(ACCIDENT_TYPES)
    for row in matrix:
        assert abs(sum(row) - 100.0) < 1e-6
    # Drifting rows must have zero major / total spill.
    assert matrix[0][2:] == [0.0, 0.0]  # Drifting allision
    assert matrix[1][2:] == [0.0, 0.0]  # Drifting grounding
    # Other rows allow >0 in major / total.
    for row in matrix[2:]:
        assert row[2] > 0.0
        assert row[3] > 0.0


def test_default_catastrophe_levels_ordered_and_named():
    levels = default_catastrophe_levels()
    assert len(levels) == 3
    quantities = [lvl['quantity'] for lvl in levels]
    assert quantities == sorted(quantities)
    assert {lvl['name'] for lvl in levels} == {'Minor', 'Major', 'Catastrophic'}


def test_reshape_oil_onboard_preserves_overlap():
    types = ['A', 'B']
    intervals = [
        {'min': 0.0, 'max': 50.0, 'label': '0 - 50'},
        {'min': 50.0, 'max': 100.0, 'label': '50 - 100'},
    ]
    existing = [[1.0, 2.0], [3.0, 4.0]]
    out = reshape_oil_onboard(existing, types, intervals)
    assert out == [[1.0, 2.0], [3.0, 4.0]]

    # Add a row + a column; the new cells fall back to defaults (100 for
    # non-tankers).
    types_grown = ['A', 'B', 'C']
    intervals_grown = intervals + [{'min': 100.0, 'max': 200.0, 'label': '100 - 200'}]
    out = reshape_oil_onboard(existing, types_grown, intervals_grown)
    assert out[0][:2] == [1.0, 2.0]
    assert out[1][:2] == [3.0, 4.0]
    assert out[0][2] == 100.0  # New column, default
    assert out[2] == [100.0, 100.0, 100.0]  # New row, default


# ----------------------------------------------------------------------
# Catastrophe exceedance
# ----------------------------------------------------------------------
def _baseline_consequence(oil_value: float = 1000.0) -> dict:
    """Build a consequence block where the first cell has ``oil_value`` m^3
    and every other cell has zero, so we can target a single (i, j) pair.
    """
    return {
        'oil_onboard': [
            [oil_value, 0.0],
            [0.0, 0.0],
        ],
        # Every accident: 100% chance of "total loss" -> spill 100% of tank.
        'spill_probability': [[0.0, 0.0, 0.0, 100.0] for _ in ACCIDENT_TYPES],
        'spill_fraction':    [default_spill_fraction()[0] for _ in ACCIDENT_TYPES],
        'catastrophe_levels': [
            {'name': 'Minor', 'quantity': 50.0},
            {'name': 'Major', 'quantity': 500.0},
            {'name': 'Catastrophic', 'quantity': 5000.0},
        ],
    }


def test_exceedance_counts_only_levels_below_spill_volume():
    """A single 1000 m^3 spill at frequency 1e-3 should exceed the 50 m^3
    and 500 m^3 thresholds but NOT the 5000 m^3 threshold."""
    consequence = _baseline_consequence(oil_value=1000.0)
    drifting_report = {
        'by_cell_allision': {'0_0': 1e-3},
        'by_cell_grounding': {},
    }

    out = compute_catastrophe_exceedance(
        consequence, drifting_report=drifting_report,
    )
    levels_by_name = {lvl['name']: lvl for lvl in out['levels']}
    assert levels_by_name['Minor']['exceedance'] > 0
    assert levels_by_name['Major']['exceedance'] > 0
    assert levels_by_name['Catastrophic']['exceedance'] == 0


def test_exceedance_sums_across_accident_types():
    """A drifting allision + powered allision both contributing to the
    same ship cell should add up in the exceedance for thresholds they
    both cross.
    """
    consequence = _baseline_consequence(oil_value=1000.0)
    drifting_report = {'by_cell_allision': {'0_0': 1e-3}, 'by_cell_grounding': {}}
    powered_allision = {'by_cell': {'0_0': 2e-3}}

    out = compute_catastrophe_exceedance(
        consequence,
        drifting_report=drifting_report,
        powered_allision_report=powered_allision,
    )
    minor = [lvl for lvl in out['levels'] if lvl['name'] == 'Minor'][0]
    # Both accidents have P(total loss) = 100% and spill 1000 m^3 (>50 m^3),
    # so the sum of frequencies feeds the Minor threshold.
    assert abs(minor['exceedance'] - 3e-3) < 1e-9


def test_exceedance_zero_when_oil_onboard_is_zero():
    """If a ship category has no oil onboard, its accidents contribute
    zero to every catastrophe level, regardless of spill fractions.
    """
    consequence = _baseline_consequence(oil_value=0.0)
    powered_grounding = {'by_cell': {'0_0': 5e-2}}

    out = compute_catastrophe_exceedance(
        consequence, powered_grounding_report=powered_grounding,
    )
    for lvl in out['levels']:
        assert lvl['exceedance'] == 0.0


def test_exceedance_handles_collision_by_cell_routing():
    """Each collision sub-category in the report should feed the matching
    accident row.  Use spill_probability designed to highlight head-on
    only, then check that overtaking contributions don't inflate the
    head-on accident in by_accident."""
    consequence = _baseline_consequence(oil_value=1000.0)
    collision_report = {
        'by_cell': {
            'head_on':   {'0_0': 1.0},
            'overtaking': {'0_0': 2.0},
            'crossing':   {},
            'merging':    {},
            'bend':       {},
        },
    }
    out = compute_catastrophe_exceedance(
        consequence, collision_report=collision_report,
    )
    assert out['by_accident']['Head-on collision']['frequency'] == 1.0
    assert out['by_accident']['Overtaking collision']['frequency'] == 2.0


def test_exceedance_levels_sorted_by_quantity():
    consequence = _baseline_consequence(oil_value=1000.0)
    # Provide thresholds out of order on input.
    consequence['catastrophe_levels'] = [
        {'name': 'High', 'quantity': 800.0},
        {'name': 'Low',  'quantity': 10.0},
        {'name': 'Mid',  'quantity': 100.0},
    ]
    out = compute_catastrophe_exceedance(consequence)
    names = [lvl['name'] for lvl in out['levels']]
    assert names == ['Low', 'Mid', 'High']


# ----------------------------------------------------------------------
# Cross-check: ACCIDENT_KEYS lines up with model report sources
# ----------------------------------------------------------------------
def test_accident_keys_routed_correctly():
    """Each accident key in the canonical order must produce a
    non-zero contribution from the matching report dict and zero
    from any other.  Catches mistakes when ACCIDENT_KEYS or the routing
    in ``_by_cell_for_accident`` drifts.
    """
    base = _baseline_consequence(oil_value=1000.0)

    # Map accident key -> kwargs that put 1.0 frequency in cell 0_0.
    cases = {
        'drifting_allision':  {'drifting_report': {'by_cell_allision': {'0_0': 1.0}, 'by_cell_grounding': {}}},
        'drifting_grounding': {'drifting_report': {'by_cell_allision': {}, 'by_cell_grounding': {'0_0': 1.0}}},
        'powered_allision':   {'powered_allision_report': {'by_cell': {'0_0': 1.0}}},
        'powered_grounding':  {'powered_grounding_report': {'by_cell': {'0_0': 1.0}}},
        'overtaking':         {'collision_report': {'by_cell': {'head_on': {}, 'overtaking': {'0_0': 1.0}, 'crossing': {}, 'merging': {}, 'bend': {}}}},
        'head_on':            {'collision_report': {'by_cell': {'head_on': {'0_0': 1.0}, 'overtaking': {}, 'crossing': {}, 'merging': {}, 'bend': {}}}},
        'crossing':           {'collision_report': {'by_cell': {'head_on': {}, 'overtaking': {}, 'crossing': {'0_0': 1.0}, 'merging': {}, 'bend': {}}}},
        'merging':            {'collision_report': {'by_cell': {'head_on': {}, 'overtaking': {}, 'crossing': {}, 'merging': {'0_0': 1.0}, 'bend': {}}}},
    }
    for accident_key, kwargs in cases.items():
        out = compute_catastrophe_exceedance(base, **kwargs)
        idx = ACCIDENT_KEYS.index(accident_key)
        accident_label = ACCIDENT_TYPES[idx]
        assert out['by_accident'][accident_label]['frequency'] == 1.0, (
            f"{accident_key} not routed to {accident_label}"
        )
        # Every *other* accident row must be zero.
        for other_label in ACCIDENT_TYPES:
            if other_label == accident_label:
                continue
            assert out['by_accident'][other_label]['frequency'] == 0.0, (
                f"{accident_key} leaked into {other_label}"
            )
