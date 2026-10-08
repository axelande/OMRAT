"""The Distributions tab must not write stale values over an AIS update.

B156 LEG_1_1 (2026-10-07): after Clear model the tab held weights '0' for
``last_id = '1'``, so the first leg drawn got weight 0.  Every Update AIS
on it then wrote the new mean / std into the widgets and switched leg,
which flushed the widgets -- weight 0 included -- back over the AIS
result: the numbers changed but the leg never got a curve on the tangent
line.  The same flush wrote the updated leg's mean / std into whichever
other leg the tab showed.

Needs the QGIS conftest (real widgets).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _seg(mean: float = 0.0, std: float = 0.0, weight: float = 100.0, ai: float = 180.0) -> dict:
    seg: dict = {'Dirs': ['North going', 'South going']}
    for d in (1, 2):
        seg[f'mean{d}_1'] = mean
        seg[f'std{d}_1'] = std
        seg[f'weight{d}_1'] = weight
        for i in (2, 3):
            seg[f'mean{d}_{i}'] = 0
            seg[f'std{d}_{i}'] = 0
            seg[f'weight{d}_{i}'] = 0
        seg[f'u_min{d}'] = 0
        seg[f'u_max{d}'] = 0
        seg[f'u_p{d}'] = 0
        seg[f'ai{d}'] = ai
    return seg


@pytest.fixture
def plugin(omrat, monkeypatch):
    """``omrat`` with the plot and the weight re-balancing stubbed."""
    monkeypatch.setattr(omrat.distributions, 'plot_data', MagicMock())
    monkeypatch.setattr(omrat.distributions, 'adjust_weights', MagicMock())
    omrat.segment_data.clear()
    return omrat


def _show(plugin, leg: str) -> None:
    """Load ``leg`` into the tab without flushing anything."""
    plugin.main_widget.leNormMean1_1.setText('')
    plugin.distributions.change_dist_segment(leg)


def _apply_ais(plugin, leg: str, line1: list[float], line2: list[float]) -> None:
    from omrat_utils.ais_update_task import AisUpdateTask
    task = AisUpdateTask(
        plugin.ais, legs={}, seg_table={}, rows=0, cols=0,
        variables=[], var_defaults={}, leg_dirs={},
    )
    task.results = {leg: {
        'traffic': {'North going': {}, 'South going': {}},
        'line1': np.array(line1), 'line2': np.array(line2),
    }}
    task.last_key = leg
    task._apply_results(plugin)


class TestAisUpdateVsDistributionsTab:
    def test_shown_leg_keeps_the_ais_weights(self, plugin):
        plugin.segment_data['1'] = _seg(weight=0.0)
        _show(plugin, '1')

        _apply_ais(plugin, '1', [-900.0, -1000.0], [1400.0, 1500.0])

        seg = plugin.segment_data['1']
        assert seg['weight1_1'] == 100
        assert seg['weight2_1'] == 100
        assert seg['mean1_1'] == pytest.approx(-950.0)
        assert seg['mean2_1'] == pytest.approx(1450.0)
        assert plugin.main_widget.leNormWeight1_1.text() == '100'
        assert plugin.distributions.last_id == '1'

    def test_other_shown_leg_is_not_overwritten(self, plugin):
        plugin.segment_data['1'] = _seg(weight=0.0)
        plugin.segment_data['2'] = _seg(mean=300.0, std=50.0)
        _show(plugin, '2')

        _apply_ais(plugin, '1', [-900.0, -1000.0], [1400.0, 1500.0])

        assert plugin.segment_data['2']['mean1_1'] == 300.0
        assert plugin.segment_data['2']['std1_1'] == 50.0
        assert plugin.segment_data['1']['weight1_1'] == 100
        # The tab now shows the updated leg.
        assert plugin.distributions.last_id == '1'
        assert float(plugin.main_widget.leNormMean1_1.text()) == pytest.approx(-950.0)

    def test_pending_edit_on_shown_leg_is_saved_first(self, plugin):
        plugin.segment_data['1'] = _seg()
        plugin.segment_data['2'] = _seg(mean=300.0, std=50.0)
        _show(plugin, '2')
        # The check-interval fields are only flushed on a leg switch.
        plugin.main_widget.LEMeanTimeSeconds1.setText('60')

        _apply_ais(plugin, '1', [-900.0, -1000.0], [1400.0, 1500.0])

        assert plugin.segment_data['2']['ai1'] == 60.0


class TestClearModel:
    def test_first_leg_after_clear_keeps_its_weights(self, plugin):
        plugin.clear_model()
        assert plugin.distributions.last_id == ''
        plugin.segment_data['1'] = _seg(mean=-500.0, std=400.0)
        plugin.segment_data['2'] = _seg()

        plugin.distributions.change_dist_segment('2')

        seg = plugin.segment_data['1']
        assert seg['weight1_1'] == 100.0
        assert seg['mean1_1'] == -500.0
