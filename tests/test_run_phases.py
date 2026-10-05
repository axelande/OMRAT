"""Phase selection behind the Calculation matrix (``compute/run_phases.py``).

Standalone: run with ``--noconftest -p no:qgis``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute import run_phases as rp  # noqa: E402


class TestNormalize:
    def test_none_means_all_phases(self):
        assert rp.normalize_phases(None) == rp.PHASES

    def test_keeps_run_order_and_drops_unknown(self):
        out = rp.normalize_phases(['powered_allision', 'bogus', 'drifting'])
        assert out == ('drifting', 'powered_allision')

    def test_empty_selection_stays_empty(self):
        assert rp.normalize_phases([]) == ()

    def test_duplicates_collapse(self):
        assert rp.normalize_phases(['collision', 'collision']) == ('collision',)


class TestSkipped:
    def test_full_run_skips_nothing(self):
        assert rp.skipped_phases(None) == ()
        assert rp.skipped_totals(rp.PHASES) == ()
        assert rp.is_partial(None) is False

    def test_partial_run(self):
        sel = ('powered_allision',)
        assert rp.is_partial(sel) is True
        assert rp.skipped_phases(sel) == ('drifting', 'collision', 'powered_grounding')
        assert rp.skipped_totals(sel) == (
            'drift_allision', 'drift_grounding',
            'overtaking', 'head_on', 'crossing', 'merging', 'bend',
            'powered_grounding',
        )

    def test_phase_totals_cover_every_accident_row_once(self):
        """Every row of the accident table belongs to exactly one phase,
        so skipping phases can never leave a row unaccounted for."""
        from compute.sensitivity import ACCIDENT_KEYS
        keys = [k for p in rp.PHASES for k in rp.PHASE_TOTALS[p]]
        assert sorted(keys) == sorted(ACCIDENT_KEYS)
        assert len(keys) == len(set(keys))


class TestDescribe:
    def test_all_and_nothing(self):
        assert rp.describe_selection(None) == 'all parts'
        assert rp.describe_selection([]) == 'nothing'

    def test_labels_in_run_order(self):
        sel = ['powered_allision', 'drifting']
        assert rp.describe_selection(sel) == 'Drifting, Powered allision'
        assert rp.phase_labels(sel) == ['Drifting', 'Powered allision']


class TestResetResults:
    def test_resets_only_the_given_phases(self):
        calc = SimpleNamespace(
            drifting_report={'totals': {'allision': 1.0}},
            collision_report={'totals': {}},
            powered_allision_report={'totals': {'allision': 2.0}},
            _last_structures=[{'id': '1'}],
        )
        rp.reset_results(calc, ['drifting', 'powered_allision'])
        assert calc.drifting_report is None
        assert calc._last_structures == []
        assert calc.powered_allision_report is None
        assert calc.collision_report == {'totals': {}}

    def test_list_defaults_are_not_shared(self):
        a, b = SimpleNamespace(), SimpleNamespace()
        rp.reset_results(a)
        rp.reset_results(b)
        a._last_structures.append('x')
        assert b._last_structures == []
        assert rp.PHASE_RESULT_ATTRS['drifting']['_last_structures'] == []

    def test_every_phase_resets_its_report(self):
        for phase in rp.PHASES:
            assert any(attr.endswith('_report') for attr in rp.PHASE_RESULT_ATTRS[phase]), phase


def test_sensitivity_shares_the_phase_vocabulary():
    from compute import sensitivity
    assert sensitivity.PHASES is rp.PHASES
    assert sensitivity.PHASE_TOTALS is rp.PHASE_TOTALS
    assert sensitivity.PHASE_LABELS is rp.PHASE_LABELS
