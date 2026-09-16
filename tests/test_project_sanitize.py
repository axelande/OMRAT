"""A project saved before its first AIS pass must load again.

Regression for ``kattegatt.omrat`` (2026-09-16): 62 legs were drawn and
the file saved without ever running *Update all distributions*.  Every
Speed / Draught / Height / Beam cell was still the empty observation
list from ``Traffic.create_empty_dict`` and 52 legs had never been shown
in the Distributions tab, so ``RootModelSchema`` rejected the file and
**Load** returned without a word.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.project_sanitize import (  # noqa: E402
    DISTRIBUTION_DEFAULTS, NO_OBSERVATION, collapse_sample_cells,
    sanitize_project, seed_distribution_defaults,
)
from omrat_utils.storage import Storage  # noqa: E402
from omrat_utils.validate_data import RootModelSchema  # noqa: E402

VARS = ['Frequency (ships/year)', 'Speed (knots)', 'Draught (meters)',
        'Ship heights (meters)', 'Ship Beam (meters)', 'Scaling (%)']


def _empty_block(rows=2, cols=3):
    """Same shape ``Traffic.create_empty_dict`` produces: numbers for
    Frequency / Scaling, a fresh empty list in every observation cell."""
    block = {}
    for var in VARS:
        if var == 'Frequency (ships/year)':
            default = 0
        elif var == 'Scaling (%)':
            default = 100.0
        else:
            default = None
        block[var] = [[([] if default is None else default) for _ in range(cols)] for _ in range(rows)]
    return block


def _drawn_leg(sid, route=1):
    """Exactly what ``HandleQGISIface.update_segment_data`` writes."""
    return {
        'Start_Point': '10.5 57.8', 'End_Point': '10.8 57.8',
        'Dirs': ['East going', 'West going'], 'Width': 5000, 'line_length': 18000.0,
        'Tangent_Pos': 0.5, 'Route_Id': route, 'Segment_Id': str(sid),
        'Leg_name': f'LEG_{route}_{sid}',
    }


def _drawn_project(n_legs=3):
    return {
        'pc': {'p_pc': 0.00016, 'd_pc': 1},
        'drift': {'drift_p': 1, 'anchor_p': 0.5, 'anchor_d': 1, 'speed': 2.0,
                  'rose': {'0': 12.5}, 'repair': {'func': 'lognorm', 'std': 1, 'loc': 0,
                                                  'scale': 1, 'use_lognormal': True}},
        'segment_data': {str(i): _drawn_leg(i) for i in range(1, n_legs + 1)},
        'traffic_data': {str(i): {'East going': _empty_block(), 'West going': _empty_block()}
                         for i in range(1, n_legs + 1)},
        'depths': [], 'objects': [],
        'ship_categories': {'types': ['A', 'B'],
                            'length_intervals': [{'min': 0, 'max': 25, 'label': '0-25'},
                                                 {'min': 25, 'max': 50, 'label': '25-50'},
                                                 {'min': 50, 'max': '', 'label': '50-'}]},
    }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_collapse_empty_list_cell_becomes_inf_like_the_ais_task():
    td = {'1': {'East going': _empty_block(), 'West going': _empty_block()}}
    n = collapse_sample_cells(td)
    assert n == 2 * 4 * 2 * 3  # two dirs, four observation matrices, 2x3 cells
    assert td['1']['East going']['Speed (knots)'][0][0] == NO_OBSERVATION
    assert math.isinf(td['1']['West going']['Ship Beam (meters)'][1][2])
    # Numbers are untouched.
    assert td['1']['East going']['Frequency (ships/year)'][0][0] == 0
    assert td['1']['East going']['Scaling (%)'][0][0] == 100.0


def test_collapse_non_empty_list_cell_becomes_mean():
    td = {'1': {'E': {'Speed (knots)': [[[10.0, 12.0, 14.0], 7.5]]}}}
    assert collapse_sample_cells(td) == 1
    assert td['1']['E']['Speed (knots)'][0] == [12.0, 7.5]


def test_collapse_is_idempotent_and_tolerates_junk():
    td = {'1': {'E': {'Speed (knots)': [[1.0, 2.0]]}}, 'x': 'junk', '2': {'E': 'junk', 'W': {'v': 'junk'}}}
    assert collapse_sample_cells(td) == 0
    assert collapse_sample_cells(None) == 0


def test_seed_distribution_defaults_only_fills_gaps():
    sd = {'1': _drawn_leg(1), '2': {**_drawn_leg(2), 'mean1_1': 250.0, 'std1_1': 40.0}, '3': 'junk'}
    assert seed_distribution_defaults(sd) == 2
    assert sd['1']['mean1_1'] == 0 and sd['1']['ai1'] == 180 and sd['1']['weight1_1'] == 100
    # Existing values survive; only the missing keys are added.
    assert sd['2']['mean1_1'] == 250.0 and sd['2']['std1_1'] == 40.0
    assert sd['2']['weight2_1'] == 100
    assert set(DISTRIBUTION_DEFAULTS) <= set(sd['1'])
    # Second pass changes nothing.
    assert seed_distribution_defaults(sd) == 0


# ---------------------------------------------------------------------------
# End to end: a drawn-but-not-updated project validates after either end
# ---------------------------------------------------------------------------

def test_drawn_project_fails_schema_without_sanitising():
    with pytest.raises(ValidationError):
        RootModelSchema.model_validate(_drawn_project())


def test_sanitized_drawn_project_passes_schema():
    data = sanitize_project(_drawn_project())
    RootModelSchema.model_validate(data)  # must not raise


def test_storage_normaliser_makes_a_pre_ais_file_loadable():
    store = Storage(MagicMock())
    out = store._normalize_legacy_to_schema(_drawn_project(n_legs=5))
    RootModelSchema.model_validate(out)  # must not raise
    assert math.isinf(out['traffic_data']['3']['West going']['Draught (meters)'][0][1])
    assert out['segment_data']['5']['ai2'] == 180


def test_validation_summary_groups_by_field():
    try:
        RootModelSchema.model_validate(_drawn_project(n_legs=4))
    except ValidationError as exc:
        text = Storage.summarize_validation_error(exc, limit=3)
    else:  # pragma: no cover
        pytest.fail('expected a validation error')
    lines = text.splitlines()
    assert len(lines) == 4  # 3 grouped lines + "... and N more"
    assert lines[0].startswith('traffic_data.*.')
    assert 'places' in lines[0]
    assert lines[-1].startswith('... and')
    # A leg id must not appear: the summary is per field, not per cell.
    assert 'traffic_data.1.' not in text and 'segment_data.1.' not in text
