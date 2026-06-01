"""Unit tests for :func:`compute.data_preparation.apply_traffic_scaling`.

Scaling is the "easy option" the user wires up on the Traffic tab: each
``traffic_data[seg][dir]['Scaling (%)'][type][len]`` cell multiplies the
matching Frequency cell by ``value / 100``.  ``apply_traffic_scaling``
must mutate the data dict in place and be a no-op whenever Scaling is
missing or already 100% -- that way legacy projects compute identically
to before, and a 30 % bump shows up in *every* model (ship-ship,
powered, drifting, consequence) for free, because they all read
``Frequency (ships/year)`` after this pass.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.data_preparation import apply_traffic_scaling


def _leg(freq, scaling=None):
    direction = {'Frequency (ships/year)': freq}
    if scaling is not None:
        direction['Scaling (%)'] = scaling
    return {'East going': direction}


class TestApplyTrafficScaling:
    def test_missing_scaling_is_noop(self):
        data = {'traffic_data': {'1': _leg([[10.0, 20.0], [0.0, 5.0]])}}
        apply_traffic_scaling(data)
        assert data['traffic_data']['1']['East going']['Frequency (ships/year)'] == [
            [10.0, 20.0], [0.0, 5.0],
        ]

    def test_all_100_percent_is_noop(self):
        data = {'traffic_data': {'1': _leg(
            [[10.0, 20.0]], scaling=[[100.0, 100.0]],
        )}}
        apply_traffic_scaling(data)
        assert data['traffic_data']['1']['East going']['Frequency (ships/year)'] == [
            [10.0, 20.0],
        ]

    def test_uniform_30_percent_bump(self):
        """130 % in every cell scales Q by 1.30 uniformly."""
        data = {'traffic_data': {'1': _leg(
            [[10.0, 20.0], [0.0, 5.0]],
            scaling=[[130.0, 130.0], [130.0, 130.0]],
        )}}
        apply_traffic_scaling(data)
        freq = data['traffic_data']['1']['East going']['Frequency (ships/year)']
        assert freq[0][0] == pytest.approx(13.0)
        assert freq[0][1] == pytest.approx(26.0)
        # Zero stays zero -- multiplying by any finite factor still gives 0.
        assert freq[1][0] == pytest.approx(0.0)
        assert freq[1][1] == pytest.approx(6.5)

    def test_per_cell_scaling(self):
        """Each (type, len) cell can use a different multiplier."""
        data = {'traffic_data': {'1': _leg(
            [[10.0, 20.0], [40.0, 50.0]],
            scaling=[[150.0, 100.0], [50.0, 200.0]],
        )}}
        apply_traffic_scaling(data)
        freq = data['traffic_data']['1']['East going']['Frequency (ships/year)']
        assert freq[0][0] == pytest.approx(15.0)
        assert freq[0][1] == pytest.approx(20.0)
        assert freq[1][0] == pytest.approx(20.0)
        assert freq[1][1] == pytest.approx(100.0)

    def test_per_direction_independence(self):
        """A different scaling matrix per direction must not bleed across."""
        data = {'traffic_data': {'1': {
            'East going': {
                'Frequency (ships/year)': [[10.0]],
                'Scaling (%)': [[200.0]],
            },
            'West going': {
                'Frequency (ships/year)': [[10.0]],
                'Scaling (%)': [[50.0]],
            },
        }}}
        apply_traffic_scaling(data)
        td = data['traffic_data']['1']
        assert td['East going']['Frequency (ships/year)'][0][0] == pytest.approx(20.0)
        assert td['West going']['Frequency (ships/year)'][0][0] == pytest.approx(5.0)

    def test_zero_percent_zeroes_out(self):
        """0 % scaling effectively removes those ships from the calc."""
        data = {'traffic_data': {'1': _leg(
            [[10.0, 20.0]], scaling=[[0.0, 100.0]],
        )}}
        apply_traffic_scaling(data)
        freq = data['traffic_data']['1']['East going']['Frequency (ships/year)']
        assert freq[0][0] == pytest.approx(0.0)
        assert freq[0][1] == pytest.approx(20.0)

    def test_partial_scaling_matrix_falls_back_to_one(self):
        """A scaling matrix smaller than Frequency leaves overflow cells alone."""
        data = {'traffic_data': {'1': _leg(
            [[10.0, 20.0], [30.0, 40.0]],
            scaling=[[150.0]],  # only row 0, col 0
        )}}
        apply_traffic_scaling(data)
        freq = data['traffic_data']['1']['East going']['Frequency (ships/year)']
        # Row 0, col 0 scaled; everything else stays at Q.
        assert freq[0][0] == pytest.approx(15.0)
        assert freq[0][1] == pytest.approx(20.0)
        assert freq[1][0] == pytest.approx(30.0)
        assert freq[1][1] == pytest.approx(40.0)

    def test_string_freq_value_skipped(self):
        """Empty-string cells (used by the table editor for "not set") are
        left untouched -- compute treats them as 0 elsewhere."""
        data = {'traffic_data': {'1': _leg(
            [['', 20.0]], scaling=[[200.0, 200.0]],
        )}}
        apply_traffic_scaling(data)
        freq = data['traffic_data']['1']['East going']['Frequency (ships/year)']
        assert freq[0][0] == ''  # unchanged
        assert freq[0][1] == pytest.approx(40.0)

    def test_inf_value_skipped(self):
        """``np.inf`` markers (left by AIS import on empty cells) survive."""
        import numpy as np
        data = {'traffic_data': {'1': _leg(
            [[np.inf, 20.0]], scaling=[[200.0, 200.0]],
        )}}
        apply_traffic_scaling(data)
        freq = data['traffic_data']['1']['East going']['Frequency (ships/year)']
        assert math.isinf(freq[0][0])
        assert freq[0][1] == pytest.approx(40.0)

    def test_empty_traffic_data_is_noop(self):
        data: dict = {'traffic_data': {}}
        apply_traffic_scaling(data)
        assert data == {'traffic_data': {}}

    def test_missing_traffic_data_key_does_not_raise(self):
        data: dict = {}
        apply_traffic_scaling(data)  # should not raise
        assert data == {}
