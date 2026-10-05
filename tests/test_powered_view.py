"""Powered View dialog: rebuilt report, panels and leg names (QGIS conftest run).

``geometries/get_powered_overlap.py`` imports matplotlib's Qt backend,
so this file runs with the conftest like ``test_get_powered_overlap.py``.

* ``allision_report_from_computations`` must give exactly the figures of
  ``run_powered_allision_model`` (old runs have no powered sidecar, the
  View rebuilds the shares from its own ray casts).
* The figure (overview + info, Cat I | Cat II geometry over Cat II
  probability) names legs as the route table does, draws the local panels
  to true scale zoomed to the hits (``local_view``) and keeps them a
  rotation of the map, never a mirror image.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.cat1_start_zone import start_zone_fraction  # noqa: E402
from compute.powered_breakdown import (  # noqa: E402
    allision_report_from_computations, build_sidebar_rows, leg_direction_frequencies,
)
from compute.powered_model import PoweredModelMixin  # noqa: E402
from geometries.get_powered_overlap import (  # noqa: E402
    INSET_FRAC, PoweredOverlapVisualizer, SimpleProjector, _build_legs_and_obstacles, _parse_point,
    _run_all_computations, _side_sign, local_view,
)

# Leg "1" runs east along lat 55.2.  ``s_in`` straddles the leg's
# centreline (Cat I for both directions); ``s_north`` lies north of the
# leg, i.e. left of the East-going ships and right of the West-going ones;
# ``s_east`` lies past the east end (Cat II for the East-going ships).
_S_IN = 'POLYGON((14.09 55.199, 14.11 55.199, 14.11 55.201, 14.09 55.201, 14.09 55.199))'
_S_NORTH = 'POLYGON((14.05 55.2015, 14.07 55.2015, 14.07 55.203, 14.05 55.203, 14.05 55.2015))'
_S_EAST = 'POLYGON((14.21 55.195, 14.24 55.195, 14.24 55.205, 14.21 55.205, 14.21 55.195))'


def _cell(freq: float = 100.0, speed: float = 10.0, height: float = 20.0) -> dict:
    def grid():
        return [[0.0, 0.0] for _ in range(21)]
    f, s, d, h = grid(), grid(), grid(), grid()
    f[18][1], s[18][1], d[18][1], h[18][1] = freq, speed, 13.0, height
    return {'Frequency (ships/year)': f, 'Speed (knots)': s,
            'Draught (meters)': d, 'Ship heights (meters)': h}


def _data(**pc) -> dict:
    seg = {
        'Start_Point': '14.0 55.2', 'End_Point': '14.2 55.2',
        'Dirs': ['East going', 'West going'], 'Width': 1000,
        'Leg_name': 'LEG_1_1', 'Segment_Id': '1',
        'mean1_1': 0.0, 'std1_1': 200.0, 'weight1_1': 1.0,
        'mean2_1': 0.0, 'std2_1': 200.0, 'weight2_1': 1.0,
        'ai1': 180.0, 'ai2': 180.0,
    }
    return {
        'pc': {'allision': 1.9e-4, 'allision_cat1': 2.5e-4, **pc},
        'segment_data': {'1': seg},
        'traffic_data': {'1': {'East going': _cell(speed=12.0), 'West going': _cell(freq=60.0)}},
        'objects': [['s_in', '30', _S_IN], ['s_north', '0', _S_NORTH], ['s_east', '10', _S_EAST]],
    }


def _view_inputs(data: dict, mode: str = 'allision'):
    seg = data['segment_data'][next(iter(data['segment_data']))]
    proj = SimpleProjector(*_parse_point(seg['Start_Point']))
    legs, obs, dg, dgd, og = _build_legs_and_obstacles(data, proj, mode, 15.0)
    comps = _run_all_computations(legs, obs, cat1_start_zone_frac=start_zone_fraction(data.get('pc')))
    return legs, obs, dg, dgd, og, comps


def _model_report(data: dict) -> dict:
    class Host(PoweredModelMixin):
        def __init__(self):
            self.p = MagicMock()
    host = Host()
    host.run_powered_allision_model(data)
    return host.powered_allision_report


class TestRebuiltAllisionReport:
    @pytest.mark.parametrize('pc', [{}, {'cat1_skip_start': False}, {'cat1_skip_start_pct': 40.0}])
    def test_matches_the_model(self, qgis_iface, pc):
        data = _data(**pc)
        *_, og, comps = _view_inputs(data)
        rebuilt = allision_report_from_computations(data, comps, og)
        model = _model_report(data)
        assert model['totals']['allision'] > 0
        for key in ('allision', 'cat1', 'cat2'):
            assert rebuilt['totals'][key] == pytest.approx(model['totals'][key], rel=1e-12)
        assert leg_direction_frequencies(rebuilt) == pytest.approx(leg_direction_frequencies(model), rel=1e-12)

    def test_height_clearance_is_applied(self, qgis_iface):
        """Ships of 20 m pass under the 30 m in-lane structure: no Cat I from it."""
        data = _data()
        *_, og, comps = _view_inputs(data)
        rebuilt = allision_report_from_computations(data, comps, og)
        assert 's_in' not in rebuilt['by_obstacle']
        assert rebuilt['by_obstacle']['s_north'] > 0


def _figure_viz(data: dict, rows=None) -> PoweredOverlapVisualizer:
    legs, obs, dg, dgd, og, comps = _view_inputs(data)
    if rows is None:
        rows = build_sidebar_rows(comps, allision_report_from_computations(data, comps, og),
                                  data['segment_data'], 'allision')
    fig, axes = PoweredOverlapVisualizer.build_figure()
    return PoweredOverlapVisualizer(fig, axes, legs, obs, dg, dgd, og, comps, 'allision',
                                    segment_data=data['segment_data'], breakdown_rows=rows)


def _comp_index(viz: PoweredOverlapVisualizer, dir_idx: int) -> int:
    return next(i for i, c in enumerate(viz.computations) if c['dir_idx'] == dir_idx)


class TestLocalView:
    """``local_view``: true-scale windows zoomed to the hits."""

    def test_first_hit_always_shown_and_ratio_matches_the_box(self):
        x0, x1, y0, y1 = local_view((-400.0, 400.0), [(100_000.0, 103_000.0, -5_000.0, -2_500.0)],
                                    anchors=(0.0, 112_000.0), box_ratio=0.8)
        assert (y1 - y0) / (x1 - x0) == pytest.approx(0.8)
        assert y0 < -5_000.0 and y1 > 400.0
        # The hit is right of the inset strip; the far ray start is not in view.
        assert x0 + INSET_FRAC * (x1 - x0) < 100_000.0 and x1 > 103_000.0
        assert x0 > 0.0

    def test_near_anchor_joins_the_window(self):
        x0, x1, _y0, _y1 = local_view((-400.0, 400.0), [(2_000.0, 2_500.0, -300.0, 300.0)],
                                      anchors=(0.0,), box_ratio=0.5)
        assert x0 + INSET_FRAC * (x1 - x0) <= 0.0 < 2_500.0 < x1

    def test_far_second_hit_is_left_to_the_toolbar(self):
        hits = [(1_000.0, 1_200.0, -200.0, 200.0), (60_000.0, 61_000.0, -200.0, 200.0)]
        _x0, x1, _y0, _y1 = local_view((-300.0, 300.0), hits, box_ratio=0.6)
        assert x1 < 60_000.0

    def test_without_hits_starts_at_the_anchor(self):
        x0, x1, y0, y1 = local_view((-300.0, 300.0), [], anchors=(0.0,), box_ratio=0.6)
        assert x0 + INSET_FRAC * (x1 - x0) <= 0.0 < x1
        assert (y1 - y0) / (x1 - x0) == pytest.approx(0.6)


class TestFigurePanels:
    def test_figure_panels(self, qgis_iface):
        _fig, axes = PoweredOverlapVisualizer.build_figure()
        assert set(axes) == {'overview', 'info', 'detail', 'cat2', 'waterfall'}
        assert axes['waterfall'].get_shared_x_axes().joined(axes['waterfall'], axes['cat2'])

    def test_default_selection_is_the_largest_contributor(self, qgis_iface):
        viz = _figure_viz(_data())
        viz.run_visualization()
        assert viz._selected_comp_idx == viz.breakdown_rows[0]['comp_idx']

    def test_titles_use_the_leg_name_and_share(self, qgis_iface):
        viz = _figure_viz(_data())
        viz.run_visualization()
        for dir_idx, dir_name in ((0, 'East going'), (1, 'West going')):
            idx = _comp_index(viz, dir_idx)
            viz._select_computation(idx)
            row = viz._rows_by_comp[idx]
            title = viz.axes['overview'].get_title()
            assert f'LEG_1_1 {dir_name}' in title
            assert f"{row['share']:.1f}%" in title
            assert viz.axes['detail'].get_title().startswith('Cat I')
            assert viz.axes['cat2'].get_title().startswith('Cat II')
            assert 'LEG_1_1' in viz.axes['detail'].get_title()

    def test_overview_extent_does_not_move_on_selection(self, qgis_iface):
        viz = _figure_viz(_data())
        viz.run_visualization()
        before = (viz.axes['overview'].get_xlim(), viz.axes['overview'].get_ylim())
        viz._select_computation(_comp_index(viz, 1))
        assert (viz.axes['overview'].get_xlim(), viz.axes['overview'].get_ylim()) == before

    def test_local_panels_are_true_scale_and_show_the_largest_hit(self, qgis_iface):
        viz = _figure_viz(_data())
        viz.run_visualization()
        for dir_idx in (0, 1):
            viz._select_computation(_comp_index(viz, dir_idx))
            comp = viz.computations[viz._selected_comp_idx]
            for key, category in (('detail', 'cat1'), ('cat2', 'cat2')):
                ax = viz.axes[key]
                assert ax.get_aspect() == 1.0
                summaries = viz._summaries(comp, category)
                if not summaries:
                    continue
                top = max(summaries.values(), key=lambda s: s['p_approx'])
                sign = viz._frame(comp, category)[3]
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                assert x0 <= min(top['ray_dists']) and max(top['ray_dists']) <= x1
                assert y0 <= min(sign * o for o in top['ray_offsets'])
                assert max(sign * o for o in top['ray_offsets']) <= y1
                assert viz.views[category] == pytest.approx((x0, x1, y0, y1))

    def test_probability_panel_follows_the_cat2_window(self, qgis_iface):
        viz = _figure_viz(_data())
        viz._select_computation(_comp_index(viz, 0))
        assert viz.axes['waterfall'].get_xlim() == pytest.approx(viz.axes['cat2'].get_xlim())

    def test_overview_outlines_both_panel_windows(self, qgis_iface):
        viz = _figure_viz(_data())
        viz._select_computation(_comp_index(viz, 0))
        labels = {a.get_text() for a in viz._overview_highlight if hasattr(a, 'get_text')}
        assert {'Cat I', 'Cat II', 'LEG_1_1 East going'} <= labels

    def test_no_click_selection_on_the_figure(self, qgis_iface):
        """Both directions share the map line, so the table is the only selector."""
        viz = _figure_viz(_data())
        viz.run_visualization()
        assert not hasattr(viz, '_on_overview_click')
        assert not hasattr(viz, '_connect_events')

    def test_local_panels_are_rotated_not_mirrored(self, qgis_iface):
        """``s_north`` is left of the East-going ships and right of the West-going ones."""
        viz = _figure_viz(_data())
        signs = {}
        for dir_idx in (0, 1):
            comp = viz.computations[_comp_index(viz, dir_idx)]
            s = comp['cat1']['summaries'][('object', 's_north')]
            sign = _side_sign(comp['cat1']['along_dir'], comp['perp'])
            signs[dir_idx] = sign * np.mean(s['ray_offsets'])
        assert signs[0] > 0 > signs[1]

    def test_works_without_report_or_optional_panels(self, qgis_iface):
        """Old three-axes figure and no breakdown rows (grounding of an old run)."""
        from matplotlib.figure import Figure
        legs, obs, dg, dgd, og, comps = _view_inputs(_data())
        fig = Figure()
        axes = {'overview': fig.add_subplot(2, 2, (1, 2)), 'detail': fig.add_subplot(2, 2, 3),
                'waterfall': fig.add_subplot(2, 2, 4)}
        viz = PoweredOverlapVisualizer(fig, axes, legs, obs, dg, dgd, og, comps, 'grounding')
        viz.run_visualization()
        assert viz._selected_comp_idx == 0
        for i in range(len(comps)):
            viz._select_computation(i)

    def test_cat2_off_is_explained(self, qgis_iface):
        data = _data()
        data['segment_data']['1']['ai1'] = 0.0
        viz = _figure_viz(data)
        viz._select_computation(_comp_index(viz, 0))
        texts = [t.get_text() for t in viz.axes['waterfall'].texts]
        assert any('Cat II is off' in t for t in texts)


class TestPoweredSidecar:
    """``<stem>.powered.json``: written after a run, read back by the View."""

    def test_written_next_to_the_run_and_loaded(self, qgis_iface, tmp_path):
        from omrat_utils.run_history import RunMeta
        from omrat_utils.run_history_mixin import RunHistoryMixin, load_powered_sidecar
        report = _model_report(_data())
        calc = SimpleNamespace(powered_allision_report=report, powered_grounding_report=None)
        path = RunHistoryMixin._write_powered_report_sidecar(
            calc_object=calc, out_dir=tmp_path, base_filename='r_20261002_120000.gpkg',
        )
        assert path == tmp_path / 'r_20261002_120000.powered.json'
        run = RunMeta(run_id=1, name='r', timestamp='t', output_dir=str(tmp_path),
                      output_filename='r_20261002_120000.gpkg')
        payload = load_powered_sidecar(run)
        assert set(payload) == {'allision'}
        assert payload['allision']['totals']['allision'] == pytest.approx(report['totals']['allision'])
        assert payload['allision']['by_leg_direction'] == pytest.approx(leg_direction_frequencies(report))

    def test_nothing_written_without_reports(self, qgis_iface, tmp_path):
        from omrat_utils.run_history_mixin import RunHistoryMixin
        calc = SimpleNamespace(powered_allision_report=None, powered_grounding_report={})
        assert RunHistoryMixin._write_powered_report_sidecar(
            calc_object=calc, out_dir=tmp_path, base_filename='x.gpkg') is None
        assert not list(tmp_path.iterdir())

    def test_missing_or_unusable_run(self, qgis_iface, tmp_path):
        from omrat_utils.run_history import RunMeta
        from omrat_utils.run_history_mixin import load_powered_sidecar
        assert load_powered_sidecar(RunMeta(run_id=1, name='r', timestamp='t', output_dir=str(tmp_path),
                                            output_filename='none.gpkg')) is None
        assert load_powered_sidecar(RunMeta(run_id=2, name='x', timestamp='t')) is None
        assert load_powered_sidecar(MagicMock()) is None

    def test_view_passes_the_report_and_restores(self, qgis_iface):
        from omrat_utils.accident_results_mixin import AccidentResultsMixin
        host = SimpleNamespace(calc=SimpleNamespace(drifting_report=None, powered_view_reports='live'))
        seen = {}

        def method(_data):
            seen['reports'] = host.calc.powered_view_reports

        AccidentResultsMixin._invoke_drift_or_powered_visualiser(
            host, method, {}, None, powered_reports={'allision': {'totals': {}}},
        )
        assert seen['reports'] == {'allision': {'totals': {}}}
        assert host.calc.powered_view_reports == 'live'
        # A run without a sidecar must not see a previous run's reports.
        AccidentResultsMixin._invoke_drift_or_powered_visualiser(host, method, {}, None)
        assert seen['reports'] is None

    def test_visualization_mixin_hands_the_block_to_the_dialog(self, qgis_iface):
        from compute.visualization import VisualizationMixin
        host = VisualizationMixin()
        assert host._powered_view_report('allision') is None
        host.powered_view_reports = {'allision': {'totals': {'allision': 1.0}}, 'grounding': 'bad'}
        assert host._powered_view_report('allision') == {'totals': {'allision': 1.0}}
        assert host._powered_view_report('grounding') is None


class TestDialogTables:
    """The Qt side: the sidebar tables of the powered and drifting View dialogs."""

    def test_powered_dialog_table(self, qgis_iface):
        from ui.show_geom_res import ShowGeomRes
        dialog = ShowGeomRes()  # keep a reference: Qt deletes the table with its dialog
        viz = PoweredOverlapVisualizer.show_in_dialog(dialog, _data(), mode='allision')
        sb = viz.sidebar
        headers = [sb.horizontalHeaderItem(i).text() for i in range(sb.columnCount())]
        assert headers == ['Leg', 'Direction', 'Frequency (/yr)', '% of total']
        assert {sb.item(r, 0).text() for r in range(sb.rowCount())} == {'LEG_1_1'}
        shares = [float(sb.item(r, 3).text().rstrip('%')) for r in range(sb.rowCount())]
        assert shares == sorted(shares, reverse=True)
        assert sum(shares) == pytest.approx(100.0, abs=0.2)
        assert viz._selected_comp_idx == viz.sidebar_index_map[0]
        sb.selectRow(1)
        assert viz._selected_comp_idx == viz.sidebar_index_map[1]

    def test_powered_dialog_uses_the_sidecar_block(self, qgis_iface):
        from ui.show_geom_res import ShowGeomRes
        block = {'totals': {'allision': 4.0}, 'by_leg_direction': {'1:0': 3.0, '1:1': 1.0}}
        dialog = ShowGeomRes()
        viz = PoweredOverlapVisualizer.show_in_dialog(dialog, _data(), mode='allision', report=block)
        sb = viz.sidebar
        rows = {(sb.item(r, 1).text(), sb.item(r, 3).text()) for r in range(sb.rowCount())}
        assert rows == {('East going', '75.0%'), ('West going', '25.0%')}

    def test_powered_grounding_without_report(self, qgis_iface):
        from ui.show_geom_res import ShowGeomRes
        data = _data()
        data['depths'] = [['d1', '5', _S_EAST]]
        dialog = ShowGeomRes()
        viz = PoweredOverlapVisualizer.show_in_dialog(dialog, data, mode='grounding')
        sb = viz.sidebar
        assert sb.horizontalHeaderItem(2).text() == 'P(hit) per ship'
        assert all(sb.item(r, 3).text() == '--' for r in range(sb.rowCount()))

    def test_drifting_sidebar_names_and_shares(self, qgis_iface):
        from geometries.drift_overlap_sidebar import build_overlap_sidebar
        report = {
            'totals': {'allision': 4e-4},
            'by_leg_direction': {'1:East going:90': {'contrib_allision': 1e-4},
                                 '1:East going:45': {'contrib_allision': 3e-4}},
        }
        sb = build_overlap_sidebar(['Leg 1-East going'], report, 'allision',
                                   segment_data=_data()['segment_data'])
        assert sb.horizontalHeaderItem(3).text() == '% of total'
        by_dir = {sb.item(r, 1).text(): sb.item(r, 3).text() for r in range(sb.rowCount())}
        assert by_dir['East'] == '25.0%' and by_dir['North-East'] == '75.0%'
        assert {sb.item(r, 0).text() for r in range(sb.rowCount())} == {'LEG_1_1 East going'}


class TestResizeRefit:
    def test_resize_redraws_the_selection_with_the_new_panel_shape(self, qgis_iface):
        viz = _figure_viz(_data())
        viz.run_visualization()
        before = viz.views['cat1']
        viz.fig.set_size_inches(20, 8)
        viz._refit_after_resize()
        after = viz.views['cat1']
        ratio = (after[3] - after[2]) / (after[1] - after[0])
        assert ratio == pytest.approx(viz._box_ratio(viz.axes['detail']), rel=1e-6)
        assert ratio != pytest.approx((before[3] - before[2]) / (before[1] - before[0]), rel=1e-3)

    def test_resize_events_are_debounced(self, qgis_iface):
        viz = _figure_viz(_data())
        viz.run_visualization()
        calls = []
        viz._resize_timer = SimpleNamespace(stop=lambda: calls.append('stop'), start=lambda: calls.append('start'))
        viz._schedule_refit()
        viz._schedule_refit()
        assert calls == ['stop', 'start', 'stop', 'start']
