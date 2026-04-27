"""Tests for the ``Distributions`` class methods.

Dataclasses (Normal, Uniform, Params) are tested in
``test_distributions_dataclasses.py``.  This file exercises the methods
that interact with Qt widgets via the ``omrat`` fixture from conftest.

Setting widget values normally cascades through ``valueChanged`` and
``editingFinished`` signals into ``adjust_weights`` →
``run_update_plot`` → matplotlib re-render.  The ``dist_quiet`` fixture
mocks the cascading methods so widget mutations stay local to the test.
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


@pytest.fixture
def dist(omrat):
    """Raw Distributions instance with all signals connected."""
    return omrat.distributions


@pytest.fixture
def dist_quiet(omrat):
    """Distributions with run_update_plot + adjust_weights stubbed.

    Use this when the test sets widget text/values; otherwise the wired
    Qt signals re-enter ``run_update_plot`` and either recurse forever
    or block on the matplotlib backend.
    """
    d = omrat.distributions
    d.run_update_plot = MagicMock()
    d.adjust_weights = MagicMock()
    d.plot_data = MagicMock()
    return d


# ---------------------------------------------------------------------------
# _assign  -- widget-type dispatch
# ---------------------------------------------------------------------------

class TestAssign:
    def test_qlineedit_branch(self, dist, qgis_iface):
        from qgis.PyQt.QtWidgets import QLineEdit
        w = QLineEdit()
        w.setText('3.14')
        assert dist._assign(w) == pytest.approx(3.14)

    def test_qspinbox_branch(self, dist, qgis_iface):
        from qgis.PyQt.QtWidgets import QSpinBox
        w = QSpinBox(); w.setMaximum(100); w.setValue(42)
        assert dist._assign(w) == 42.0

    def test_qdoublespinbox_branch(self, dist, qgis_iface):
        from qgis.PyQt.QtWidgets import QDoubleSpinBox
        w = QDoubleSpinBox(); w.setMaximum(100); w.setValue(2.5)
        assert dist._assign(w) == 2.5

    def test_unsupported_widget_raises(self, dist):
        class FakeWidget:
            pass
        with pytest.raises(TypeError):
            dist._assign(FakeWidget())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_leg_params
# ---------------------------------------------------------------------------

class TestGetLegParams:
    def test_reads_widgets_into_params(self, dist_quiet):
        d = dist_quiet
        d.dw.leNormMean1_1.setText('5')
        d.dw.leNormStd1_1.setText('10')
        d.dw.leNormWeight1_1.setText('80')
        d.dw.leNormMean1_2.setText('0')
        d.dw.leNormStd1_2.setText('0')
        d.dw.leNormWeight1_2.setText('0')
        d.dw.leNormMean1_3.setText('0')
        d.dw.leNormStd1_3.setText('0')
        d.dw.leNormWeight1_3.setText('0')
        d.dw.leUniformMin1.setText('0')
        d.dw.leUniformMax1.setText('5')
        d.dw.sbUniformP1.setValue(20)
        # Direction 2 -- match dimensions but zero everything.
        d.dw.leNormMean2_1.setText('0')
        d.dw.leNormStd2_1.setText('0')
        d.dw.leNormWeight2_1.setText('100')
        d.dw.leNormMean2_2.setText('0')
        d.dw.leNormStd2_2.setText('0')
        d.dw.leNormWeight2_2.setText('0')
        d.dw.leNormMean2_3.setText('0')
        d.dw.leNormStd2_3.setText('0')
        d.dw.leNormWeight2_3.setText('0')
        d.dw.leUniformMin2.setText('0')
        d.dw.leUniformMax2.setText('0')
        d.dw.sbUniformP2.setValue(0)

        p1, p2 = d.get_leg_params()
        assert p1.normal1.mean == 5.0
        assert p1.normal1.std == 10.0
        assert p1.normal1.probability == 80.0
        assert p1.uniform.upper == 5.0
        assert p1.uniform.probability == 20.0
        assert p2.normal1.probability == 100.0


# ---------------------------------------------------------------------------
# change_dist_segment
# ---------------------------------------------------------------------------

class TestChangeDistSegment:
    def test_blank_id_returns_silently(self, dist_quiet):
        # Should not raise.
        dist_quiet.change_dist_segment(None)
        dist_quiet.change_dist_segment('')

    def test_initialises_new_segment_with_defaults(self, dist_quiet):
        d = dist_quiet
        d.omrat.segment_data['1'] = {}
        d.omrat.segment_data['2'] = {}
        # mean1_1 empty so save block is skipped.
        d.dw.leNormMean1_1.setText('')
        d.last_id = '1'
        d.change_dist_segment('2')
        seg = d.omrat.segment_data['2']
        assert seg['mean1_1'] == 0
        assert seg['weight1_1'] == 100
        assert seg['ai1'] == 180

    def test_persists_current_widget_values(self, dist_quiet):
        d = dist_quiet
        d.omrat.segment_data['1'] = {}
        d.omrat.segment_data['2'] = {}
        d.dw.leNormMean1_1.setText('7')
        d.dw.leNormMean1_2.setText('0')
        d.dw.leNormMean1_3.setText('0')
        d.dw.leNormStd1_1.setText('3')
        d.dw.leNormStd1_2.setText('0')
        d.dw.leNormStd1_3.setText('0')
        d.dw.leNormMean2_1.setText('0')
        d.dw.leNormMean2_2.setText('0')
        d.dw.leNormMean2_3.setText('0')
        d.dw.leNormStd2_1.setText('0')
        d.dw.leNormStd2_2.setText('0')
        d.dw.leNormStd2_3.setText('0')
        d.dw.leNormWeight1_1.setText('100')
        d.dw.leNormWeight1_2.setText('0')
        d.dw.leNormWeight1_3.setText('0')
        d.dw.leNormWeight2_1.setText('100')
        d.dw.leNormWeight2_2.setText('0')
        d.dw.leNormWeight2_3.setText('0')
        d.dw.leUniformMin1.setText('0')
        d.dw.leUniformMax1.setText('0')
        d.dw.sbUniformP1.setValue(0)
        d.dw.LEMeanTimeSeconds1.setText('120')
        d.dw.leUniformMin2.setText('0')
        d.dw.leUniformMax2.setText('0')
        d.dw.sbUniformP2.setValue(0)
        d.dw.LEMeanTimeSeconds2.setText('120')
        d.last_id = '1'

        d.change_dist_segment('2')
        # Saved into '1'.
        assert d.omrat.segment_data['1']['mean1_1'] == 7.0
        assert d.omrat.segment_data['1']['std1_1'] == 3.0
        assert d.omrat.segment_data['1']['ai1'] == 120

    def test_pre_existing_segment_skips_default_init(self, dist_quiet):
        """A segment that already has 'mean1_1' but missing 'mean1_2' triggers
        the second-stage default init (lines 153-164)."""
        d = dist_quiet
        d.omrat.segment_data['1'] = {}
        d.omrat.segment_data['2'] = {
            'mean1_1': 5, 'std1_1': 1, 'weight1_1': 50,
            'mean2_1': 0, 'std2_1': 0, 'weight2_1': 50,
        }
        d.dw.leNormMean1_1.setText('')
        d.last_id = '1'
        d.change_dist_segment('2')
        # Default init for missing fields.
        assert d.omrat.segment_data['2']['mean1_2'] == 0
        assert d.omrat.segment_data['2']['ai1'] == 180


# ---------------------------------------------------------------------------
# ensure_total_sum
# ---------------------------------------------------------------------------

class TestEnsureTotalSum:
    def test_brings_total_to_100(self, dist, qgis_iface):
        from qgis.PyQt.QtWidgets import QLineEdit, QSpinBox
        a, b, c = QLineEdit(), QLineEdit(), QSpinBox()
        c.setMaximum(100)
        a.setText('40'); b.setText('40'); c.setValue(10)
        # Total = 90 -> last (c) gets +10 to reach 100.
        dist.ensure_total_sum([a, b, c])
        assert c.value() == 20

    def test_handles_blank_text(self, dist, qgis_iface):
        from qgis.PyQt.QtWidgets import QLineEdit
        a, b, c = QLineEdit(), QLineEdit(), QLineEdit()
        a.setText('30'); b.setText(''); c.setText('30')
        dist.ensure_total_sum([a, b, c])
        # Last (c) was '30'; total = 60 -> bumped by 40 -> 70.
        assert c.text() == '70.0'


# ---------------------------------------------------------------------------
# adjust_weights -- with quiet fixture so the inner run_update_plot is stubbed
# ---------------------------------------------------------------------------

class TestAdjustWeights:
    def test_distribute_remaining_proportionally(self, dist_quiet):
        d = dist_quiet
        d.adjust_weights = type(d).adjust_weights.__get__(d)  # un-stub
        d.dw.leNormWeight1_1.setText('60')
        d.dw.leNormWeight1_2.setText('20')
        d.dw.leNormWeight1_3.setText('20')
        d.dw.sbUniformP1.setValue(0)
        d.adjust_weights(d.dw.leNormWeight1_1)
        total = sum([
            float(d.dw.leNormWeight1_1.text()),
            float(d.dw.leNormWeight1_2.text()),
            float(d.dw.leNormWeight1_3.text()),
            float(d.dw.sbUniformP1.value()),
        ])
        assert total == pytest.approx(100.0, abs=1e-6)

    def test_unknown_widget_returns_silently(self, dist_quiet, qgis_iface, capsys):
        from qgis.PyQt.QtWidgets import QLineEdit
        d = dist_quiet
        d.adjust_weights = type(d).adjust_weights.__get__(d)
        unknown = QLineEdit()
        d.adjust_weights(unknown)
        out = capsys.readouterr().out
        assert 'Widget not found' in out

    def test_dir2_widget_dispatch(self, dist_quiet):
        d = dist_quiet
        d.adjust_weights = type(d).adjust_weights.__get__(d)
        d.dw.leNormWeight2_1.setText('50')
        d.dw.leNormWeight2_2.setText('30')
        d.dw.leNormWeight2_3.setText('20')
        d.dw.sbUniformP2.setValue(0)
        d.adjust_weights(d.dw.leNormWeight2_1)
        total = sum([
            float(d.dw.leNormWeight2_1.text()),
            float(d.dw.leNormWeight2_2.text()),
            float(d.dw.leNormWeight2_3.text()),
            float(d.dw.sbUniformP2.value()),
        ])
        assert total == pytest.approx(100.0, abs=1e-6)

    def test_other_total_zero_distributes_equally(self, dist_quiet):
        d = dist_quiet
        d.adjust_weights = type(d).adjust_weights.__get__(d)
        d.dw.leNormWeight1_1.setText('40')
        d.dw.leNormWeight1_2.setText('0')
        d.dw.leNormWeight1_3.setText('0')
        d.dw.sbUniformP1.setValue(0)
        d.adjust_weights(d.dw.leNormWeight1_1)
        # Remaining 60 split among 3 -> 20 each.
        assert float(d.dw.leNormWeight1_2.text()) == pytest.approx(20.0, abs=1.0)


# ---------------------------------------------------------------------------
# add_dist2plot
# ---------------------------------------------------------------------------

class TestAddDist2Plot:
    def test_plots_against_axes(self, dist_quiet, qgis_iface):
        from matplotlib.figure import Figure
        from omrat_utils.handle_distributions import Params, Normal
        d = dist_quiet
        ax = Figure().subplots()
        params = Params()
        params.normal1 = Normal(mean=0, std=1, probability=100)
        np.random.seed(0)
        data = np.random.normal(0, 1, 200)
        # Should not raise.
        d.add_dist2plot(ax, params, data, first=True)
        assert len(ax.lines) >= 1

    def test_first_false_uses_dir2_widgets(self, dist_quiet, qgis_iface):
        from matplotlib.figure import Figure
        from omrat_utils.handle_distributions import Params, Normal
        d = dist_quiet
        ax = Figure().subplots()
        params = Params()
        params.normal1 = Normal(mean=0, std=1, probability=100)
        np.random.seed(0)
        data = np.random.normal(0, 1, 100)
        # first=False routes through the dir-2 widget setText path.
        d.add_dist2plot(ax, params, data, first=False)

    def test_uniform_component_plotted(self, dist_quiet, qgis_iface):
        from matplotlib.figure import Figure
        from omrat_utils.handle_distributions import Params, Uniform
        d = dist_quiet
        ax = Figure().subplots()
        params = Params()
        params.uniform = Uniform(lower=-1, upper=1, probability=100)
        np.random.seed(0)
        data = np.random.uniform(-1, 1, 100)
        d.add_dist2plot(ax, params, data, first=True, update_dist=False)
        # Uniform line plotted (label='Uniform').
        labels = [line.get_label() for line in ax.lines]
        assert 'Uniform' in labels

    def test_exception_swallowed(self, dist_quiet, qgis_iface, capsys):
        """Bad data -> stats.norm.fit raises -> caught and printed."""
        from matplotlib.figure import Figure
        from omrat_utils.handle_distributions import Params
        d = dist_quiet
        ax = Figure().subplots()
        # Passing a string forces float conversion failure -> exception caught.
        d.add_dist2plot(ax, Params(), 'not-an-array', first=True)
        # No raise; capsys may have an error printed.


# ---------------------------------------------------------------------------
# unload
# ---------------------------------------------------------------------------

class TestUnload:
    def test_unload_does_not_raise(self, dist):
        dist.unload()


# ---------------------------------------------------------------------------
# run_update_plot
# ---------------------------------------------------------------------------

class TestRunUpdatePlot:
    def test_segment_id_none_uses_last_id(self, dist_quiet):
        """run_update_plot(None) defaults segment_id to last_id and sets
        update_dist=True."""
        d = dist_quiet
        called = []
        d.plot_data = lambda data, data2, p1, p2, update_dist=True: called.append(update_dist)
        d.change_dist_segment = MagicMock()
        d.last_id = '1'
        d.omrat.ais.dist_data = {'1': {'line1': np.array([1.0]), 'line2': np.array([2.0])}}
        d.run_update_plot = type(d).run_update_plot.__get__(d)
        d.run_update_plot(None)
        assert called and called[-1] is True

    def test_segment_id_explicit_sets_update_dist_false(self, dist_quiet):
        d = dist_quiet
        called = []
        d.plot_data = lambda data, data2, p1, p2, update_dist=True: called.append(update_dist)
        # Stub change_dist_segment to avoid the widget-cascade.
        d.change_dist_segment = MagicMock()
        d.last_id = '1'
        d.omrat.ais.dist_data = {'1': {'line1': np.array([1.0]), 'line2': np.array([2.0])}}
        d.run_update_plot = type(d).run_update_plot.__get__(d)
        d.run_update_plot('1')
        assert called and called[-1] is False

    def test_uses_segment_data_fallback_when_no_ais(self, dist_quiet):
        d = dist_quiet
        seen_args = []
        d.plot_data = lambda data, data2, p1, p2, update_dist=True: seen_args.append((data, data2))
        d.change_dist_segment = MagicMock()
        d.last_id = '1'
        d.omrat.ais.dist_data = {}
        d.omrat.segment_data = {'1': {
            'dist1': np.array([10.0, 20.0]),
            'dist2': np.array([30.0, 40.0]),
        }}
        d.run_update_plot = type(d).run_update_plot.__get__(d)
        d.run_update_plot('1')
        assert seen_args
        line1, line2 = seen_args[-1]
        assert list(line1) == [10.0, 20.0]
        assert list(line2) == [30.0, 40.0]

    def test_no_dist_data_returns_silently(self, dist_quiet):
        d = dist_quiet
        d.plot_data = MagicMock()
        d.change_dist_segment = MagicMock()
        d.last_id = '1'
        d.omrat.ais.dist_data = {}
        d.omrat.segment_data = {'1': {}}  # no dist1/dist2
        d.run_update_plot = type(d).run_update_plot.__get__(d)
        d.run_update_plot('1')
        d.plot_data.assert_not_called()


# ---------------------------------------------------------------------------
# plot_data
# ---------------------------------------------------------------------------

class TestPlotData:
    def test_creates_canvas_and_plots(self, dist_quiet):
        """plot_data renders a histogram + distribution overlays into the
        DistributionWidget container."""
        from omrat_utils.handle_distributions import Params, Normal
        d = dist_quiet
        # Restore real plot_data; un-stub.
        d.plot_data = type(d).plot_data.__get__(d)
        np.random.seed(0)
        d.plot_data(
            data=np.random.normal(0, 1, 50),
            data2=np.random.normal(0, 1, 50),
            parameters1=Params(),
            parameters2=Params(),
            update_dist=True,
        )
        # canvas attribute set.
        assert d.canvas is not None

    def test_plot_data_replaces_existing_canvas(self, dist_quiet):
        """Calling plot_data twice removes the prior canvas first."""
        from omrat_utils.handle_distributions import Params
        d = dist_quiet
        d.plot_data = type(d).plot_data.__get__(d)
        np.random.seed(0)
        data = np.random.normal(0, 1, 50)
        d.plot_data(data, data, Params(), Params(), update_dist=True)
        first = d.canvas
        d.plot_data(data, data, Params(), Params(), update_dist=True)
        # Second call replaces canvas.
        assert d.canvas is not first
