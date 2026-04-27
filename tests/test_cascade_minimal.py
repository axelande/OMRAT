"""End-to-end cascade tests on a minimal synthetic project.

Rather than running the cascade on the 4-MB ``proj.omrat`` fixture
(which takes ~minutes), we build a tiny project programmatically:

* 1 leg (east-going, 10 km long near Sweden)
* 1 depth polygon (12 m, in front of the leg) and 1 structure (tall
  building next to the leg, for allision)
* 1 ship category with 2 LOA bins, and a handful of traffic frequencies

This exercises:
* ``compute/drifting_model.py`` (the main DriftingModelMixin)
* ``compute/powered_model.py`` (PoweredModelMixin)
* ``compute/ship_collision_model.py`` (ShipCollisionModelMixin)
* ``compute/drifting_report.py`` (report generation on live data)
* ``compute/data_preparation.py`` (clean_traffic, split_structures)

Run time: < 5 s on a modest laptop, so these are NOT marked ``@slow``.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Minimal synthetic project builder
# ---------------------------------------------------------------------------

def _minimal_traffic_cell() -> dict:
    """One ship category (Cargo=index 18), 2 LOA bins.

    The cell produces a handful of crossings per year at moderate speed
    so the cascade has non-zero contributions but still finishes fast.
    """
    # 21 rows x 2 cols.  Populate only Cargo (row 18) * LOA bin 1.
    def _grid(init=0.0):
        return [[init, init] for _ in range(21)]

    freq = _grid(0.0)
    speed = _grid(0.0)
    draught = _grid(0.0)
    height = _grid(0.0)
    beam = _grid(0.0)
    # Cargo (18), bin 1 (25-50m) -- 100 ships/yr at 10 kts, draft 13 m
    freq[18][1] = 100
    speed[18][1] = 10.0
    draught[18][1] = 13.0
    height[18][1] = 20.0
    beam[18][1] = 12.0

    return {
        'Frequency (ships/year)': freq,
        'Speed (knots)': speed,
        'Draught (meters)': draught,
        'Ship heights (meters)': height,
        'Ship Beam (meters)': beam,
    }


def _build_minimal_project() -> dict:
    """Build an OMRAT project dict that's valid for the full cascade.

    Single leg east-going near Sweden, one 12 m depth polygon NW of the
    leg (reachable by NW drift), one structure north of the leg.
    """
    return {
        'pc': {
            'p_pc': 0.00016,
            'd_pc': 1e-4,
            'headon': 4.9e-5, 'overtaking': 1.1e-4, 'crossing': 1.3e-4,
            'bend': 1.3e-4, 'grounding': 1.6e-4, 'allision': 1.9e-4,
        },
        'drift': {
            'drift_p': 1,
            'anchor_p': 0.7,
            'anchor_d': 7,
            'speed': 1.94,   # knots
            'rose': {str(a): 0.125 for a in
                     (0, 45, 90, 135, 180, 225, 270, 315)},
            'repair': {
                'func': "__import__('scipy.stats', fromlist=['norm'])"
                        ".norm(loc=0, scale=1).cdf(x)",
                'std': 1.0, 'loc': 0.0, 'scale': 1.0,
                'use_lognormal': False,
                'dist_type': 'normal',
                'norm_mean': 0.0, 'norm_std': 1.0,
            },
        },
        'segment_data': {
            '1': {
                'Start_Point': '14.0 55.2',
                'End_Point': '14.2 55.2',       # 10-km east-going leg
                'Dirs': ['East going', 'West going'],
                'Width': 1000,
                'line_length': 10_000.0,
                'Route_Id': 1, 'Leg_name': 'leg 1', 'Segment_Id': '1',
                # Single Gaussian around centerline.
                'mean1_1': 0.0, 'std1_1': 200.0, 'weight1_1': 1.0,
                'mean1_2': 0.0, 'std1_2': 0.0, 'weight1_2': 0.0,
                'mean1_3': 0.0, 'std1_3': 0.0, 'weight1_3': 0.0,
                'mean2_1': 0.0, 'std2_1': 200.0, 'weight2_1': 1.0,
                'mean2_2': 0.0, 'std2_2': 0.0, 'weight2_2': 0.0,
                'mean2_3': 0.0, 'std2_3': 0.0, 'weight2_3': 0.0,
                'u_min1': 0.0, 'u_max1': 0.0, 'u_p1': 0.0,
                'u_min2': 0.0, 'u_max2': 0.0, 'u_p2': 0.0,
                'ai1': 180.0, 'ai2': 180.0,
                'dist1': [], 'dist2': [],
            },
        },
        'traffic_data': {
            '1': {
                'East going': _minimal_traffic_cell(),
                'West going': _minimal_traffic_cell(),
            },
        },
        # Depth polygon 12m, ~1.5 km north of the leg.
        'depths': [
            ['d1', '12',
             'POLYGON((14.08 55.22, 14.10 55.22, 14.10 55.23, 14.08 55.23, 14.08 55.22))'],
        ],
        # Structure 20m, ~1 km north of the leg (near the middle).
        'objects': [
            ['s1', '20',
             'POLYGON((14.09 55.208, 14.10 55.208, 14.10 55.212, 14.09 55.212, 14.09 55.208))'],
        ],
        'ship_categories': {
            'types': [
                'Fishing', 'Towing', 'Dredging or underwater ops',
                'Diving ops', 'Military ops', 'Sailing', 'Pleasure Craft',
                'High speed craft (HSC)', 'Pilot Vessel',
                'Search and Rescue vessel', 'Tug', 'Port Tender',
                'Anti-pollution equipment', 'Law Enforcement', 'Spare',
                'Medical Transport',
                'Noncombatant ship according to RR Resolution No. 18',
                'Passenger, all ships of this type',
                'Cargo, all ships of this type',
                'Tanker, all ships of this type',
                'Other Type, all ships of this type',
            ],
            'length_intervals': [
                {'min': 0.0,  'max': 25.0, 'label': '0-25'},
                {'min': 25.0, 'max': 50.0, 'label': '25-50'},
            ],
            'selection_mode': 'ais',
        },
    }


def _mock_parent():
    mp = MagicMock()
    mp.main_widget = MagicMock()
    mp.main_widget.LEPDriftAllision.setText = MagicMock()
    mp.main_widget.LEPDriftingGrounding.setText = MagicMock()
    mp.main_widget.cbShipCategories.count = MagicMock(return_value=0)
    mp.main_widget.LEReportPath.text = MagicMock(return_value='')
    return mp


@pytest.fixture(scope="module")
def minimal_project() -> dict:
    return _build_minimal_project()


@pytest.fixture(scope="module")
def cascade(minimal_project):
    """Run the full drifting cascade once and return (calc, allision, grounding)."""
    from compute.basic_equations import default_blackout_by_ship_type
    from compute.run_calculations import Calculation

    data = copy.deepcopy(minimal_project)
    data['drift'].setdefault('blackout_by_ship_type', default_blackout_by_ship_type())
    calc = Calculation(_mock_parent())
    calc.set_progress_callback(lambda c, t, m: True)
    allision, grounding = calc.run_drifting_model(data)
    return calc, data, allision, grounding


# ---------------------------------------------------------------------------
# Cascade end-to-end tests
# ---------------------------------------------------------------------------

class TestDriftingCascadeE2E:
    def test_cascade_returns_finite_probabilities(self, cascade):
        calc, data, allision, grounding = cascade
        assert 0.0 <= allision < 1.0
        assert 0.0 <= grounding < 1.0

    def test_cascade_produces_drifting_report(self, cascade):
        calc, *_ = cascade
        report = calc.drifting_report
        assert report is not None
        assert 'totals' in report
        totals = report['totals']
        assert set(totals.keys()) >= {'allision', 'grounding', 'anchoring'}

    def test_report_totals_match_returned_values(self, cascade):
        calc, _, allision, grounding = cascade
        totals = calc.drifting_report['totals']
        assert totals['allision'] == pytest.approx(allision, abs=1e-15)
        assert totals['grounding'] == pytest.approx(grounding, abs=1e-15)

    def test_report_has_by_leg_direction(self, cascade):
        calc, *_ = cascade
        bld = calc.drifting_report.get('by_leg_direction', {})
        # At least one key present for the single leg x some directions.
        assert len(bld) >= 1

    def test_cascade_respects_anchor_p(self, minimal_project):
        """Varying anchor_p from 0 -> 0.7 must strictly reduce the
        grounding probability (regression guard for the anchor-lookup
        fix we landed earlier)."""
        from compute.basic_equations import default_blackout_by_ship_type
        from compute.run_calculations import Calculation

        def _run(anchor_p: float) -> float:
            data = copy.deepcopy(minimal_project)
            data['drift']['anchor_p'] = anchor_p
            data['drift'].setdefault(
                'blackout_by_ship_type', default_blackout_by_ship_type())
            calc = Calculation(_mock_parent())
            calc.set_progress_callback(lambda c, t, m: True)
            _, grounding = calc.run_drifting_model(data)
            return grounding

        g0 = _run(0.0)
        g7 = _run(0.7)
        # With anchor_p=0 we get no anchor reduction; with 0.7 we should
        # get a strictly smaller grounding total (if any grounding at all).
        if g0 > 0:
            assert g7 < g0

    def test_cascade_scales_with_traffic_volume(self, minimal_project):
        """Doubling ship frequency must roughly double the grounding
        probability (the exposure factor in the cascade is linear)."""
        from compute.basic_equations import default_blackout_by_ship_type
        from compute.run_calculations import Calculation

        def _run(scale: float) -> float:
            data = copy.deepcopy(minimal_project)
            data['drift'].setdefault(
                'blackout_by_ship_type', default_blackout_by_ship_type())
            for leg in data['traffic_data'].values():
                for di in leg.values():
                    freq = di['Frequency (ships/year)']
                    di['Frequency (ships/year)'] = [
                        [v * scale for v in row] for row in freq
                    ]
            calc = Calculation(_mock_parent())
            calc.set_progress_callback(lambda c, t, m: True)
            _, grounding = calc.run_drifting_model(data)
            return grounding

        g1 = _run(1.0)
        g2 = _run(2.0)
        if g1 > 0:
            # Allow 1% numerical slack.
            assert g2 == pytest.approx(2.0 * g1, rel=0.01)


# ---------------------------------------------------------------------------
# Ship-ship collision model
# ---------------------------------------------------------------------------

class TestShipCollisionCascade:
    def test_run_ship_collision_model_returns_summary(self, minimal_project):
        from compute.run_calculations import Calculation

        data = copy.deepcopy(minimal_project)
        # Add a second segment crossing the first so the crossing path
        # can do something.
        data['segment_data']['2'] = copy.deepcopy(data['segment_data']['1'])
        data['segment_data']['2']['Start_Point'] = '14.09 55.15'
        data['segment_data']['2']['End_Point']   = '14.11 55.25'
        data['segment_data']['2']['Segment_Id']  = '2'
        data['segment_data']['2']['Leg_name']    = 'leg 2'
        data['traffic_data']['2'] = copy.deepcopy(data['traffic_data']['1'])

        calc = Calculation(_mock_parent())
        calc.set_progress_callback(lambda c, t, m: True)
        result = calc.run_ship_collision_model(data)

        # result is a dict summarising the collision totals.
        assert isinstance(result, dict)
        # Typical top-level keys -- exact set depends on which collision
        # types fired; at minimum these should all be present as zeros.
        expected_keys = {
            'head_on', 'overtaking', 'crossing', 'bend', 'total',
        }
        assert expected_keys.issubset(result.keys())
        for k in expected_keys:
            assert result[k] >= 0


# ---------------------------------------------------------------------------
# Powered model
# ---------------------------------------------------------------------------

class TestPoweredCascade:
    def test_powered_grounding_returns_finite(self, minimal_project):
        from compute.basic_equations import default_blackout_by_ship_type
        from compute.run_calculations import Calculation

        data = copy.deepcopy(minimal_project)
        data['drift'].setdefault(
            'blackout_by_ship_type', default_blackout_by_ship_type())
        calc = Calculation(_mock_parent())
        calc.set_progress_callback(lambda c, t, m: True)
        result = calc.run_powered_grounding_model(data)
        assert isinstance(result, (int, float))
        assert 0.0 <= result < 1.0

    def test_powered_allision_returns_finite(self, minimal_project):
        from compute.basic_equations import default_blackout_by_ship_type
        from compute.run_calculations import Calculation

        data = copy.deepcopy(minimal_project)
        data['drift'].setdefault(
            'blackout_by_ship_type', default_blackout_by_ship_type())
        calc = Calculation(_mock_parent())
        calc.set_progress_callback(lambda c, t, m: True)
        result = calc.run_powered_allision_model(data)
        assert isinstance(result, (int, float))
        assert 0.0 <= result < 1.0


# ---------------------------------------------------------------------------
# Drifting report markdown generation on live cascade data
# ---------------------------------------------------------------------------

class TestDriftingReportOnLiveData:
    def test_markdown_generated_from_live_report(self, cascade, tmp_path):
        calc, data, *_ = cascade
        md = calc.generate_drifting_report_markdown(data)
        assert isinstance(md, str) and len(md) > 100
        assert '# Drifting Model Appendix Report' in md

    def test_markdown_writes_to_disk(self, cascade, tmp_path):
        calc, data, *_ = cascade
        out = tmp_path / 'report.md'
        content = calc.write_drifting_report_markdown(str(out), data)
        assert out.exists()
        assert out.read_text(encoding='utf-8') == content
