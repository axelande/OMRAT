"""Tests for compute/drifting_report.py::DriftingReportMixin.

The mixin builds a Markdown appendix from ``self.drifting_report``.
Tests exercise the rendering pipeline with minimal synthetic reports so
formatting regressions surface.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.drifting_report import DriftingReportMixin


class _Host(DriftingReportMixin):
    """Bare host class that supplies the mixin's single required attribute."""
    def __init__(self, report):
        self.drifting_report = report


@pytest.fixture
def tiny_report():
    """A plausible drifting_report with one leg, one direction, two objects."""
    return {
        'totals': {
            'allision': 1.234e-5,
            'grounding': 5.678e-7,
            'anchoring': 9.0e-8,
        },
        'by_leg_direction': {
            '1:East going:0': {
                'contrib_allision': 1.0e-6,
                'contrib_grounding': 2.0e-7,
                'base': 50.0,
                'rp': 0.125,
                'allision': 1.0e-6,
                'grounding': 2.0e-7,
            },
        },
        'by_object': {
            'Structure - X': {'allision': 1.0e-6, 'grounding': 0.0},
            'Depth - D1': {'allision': 0.0, 'grounding': 2.0e-7},
        },
        'by_structure_legdir': {
            'Structure - X': {'1:East going:0': 1.0e-6},
        },
    }


# ---------------------------------------------------------------------------
# get_drifting_report
# ---------------------------------------------------------------------------

class TestGetDriftingReport:
    def test_returns_stored_report(self, tiny_report):
        host = _Host(tiny_report)
        assert host.get_drifting_report() is tiny_report

    def test_returns_none_if_not_run(self):
        host = _Host(None)
        assert host.get_drifting_report() is None


# ---------------------------------------------------------------------------
# generate_drifting_report_markdown
# ---------------------------------------------------------------------------

class TestGenerateMarkdown:
    def test_empty_report_still_produces_header(self):
        host = _Host({})
        md = host.generate_drifting_report_markdown()
        assert '# Drifting Model Appendix Report' in md
        assert 'Summary' in md

    def test_totals_section_rendered(self, tiny_report):
        host = _Host(tiny_report)
        md = host.generate_drifting_report_markdown()
        # Header.
        assert '# Drifting Model Appendix Report' in md
        # Totals values in scientific notation.
        assert '1.234e-05' in md
        assert '5.678e-07' in md

    def test_with_data_parameter_renders_drift_settings(self, tiny_report):
        host = _Host(tiny_report)
        data = {
            'drift': {
                'speed': 1.94,
                'drift_p': 1.0,
                'anchor_p': 0.7,
                'anchor_d': 7,
                'repair': {'use_lognormal': False},
                'rose': {str(a): 0.125 for a in (0, 45, 90, 135, 180, 225, 270, 315)},
            },
        }
        md = host.generate_drifting_report_markdown(data)
        assert '1.47' in md or '1.94' in md  # drift speed rendered
        assert '0.7' in md  # anchor prob

    def test_lognormal_repair_rendered(self, tiny_report):
        host = _Host(tiny_report)
        data = {
            'drift': {
                'speed': 1.0,
                'repair': {
                    'use_lognormal': True,
                    'std': 0.95, 'loc': 0.2, 'scale': 0.85,
                },
                'rose': {},
            },
        }
        md = host.generate_drifting_report_markdown(data)
        assert 'Repair lognormal' in md
        assert 'std=0.95' in md

    def test_not_lognormal_rendered_as_disabled(self, tiny_report):
        host = _Host(tiny_report)
        data = {
            'drift': {'speed': 1.0, 'repair': {'use_lognormal': False},
                      'rose': {}},
        }
        md = host.generate_drifting_report_markdown(data)
        assert 'lognormal disabled' in md or 'not specified' in md

    def test_none_report_handled(self):
        host = _Host(None)
        md = host.generate_drifting_report_markdown()
        assert isinstance(md, str) and len(md) > 0


# ---------------------------------------------------------------------------
# write_drifting_report_markdown
# ---------------------------------------------------------------------------

class TestWriteMarkdown:
    def test_writes_file_and_returns_content(self, tiny_report, tmp_path):
        host = _Host(tiny_report)
        out = tmp_path / 'appendix.md'
        content = host.write_drifting_report_markdown(str(out))
        assert out.exists()
        assert out.read_text(encoding='utf-8') == content
        assert '# Drifting Model Appendix Report' in content
