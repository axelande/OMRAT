"""Unit tests for omrat_utils/causation_factors.py."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.causation_factors import CausationFactors


@pytest.fixture
def cf(qgis_iface):
    cf_obj = CausationFactors(MagicMock())
    return cf_obj


class TestCausationFactorDefaults:
    def test_default_data_populated(self, cf):
        d = cf.data
        assert d['p_pc'] == pytest.approx(1.6e-4)
        assert d['d_pc'] == 1
        assert d['headon'] == pytest.approx(4.9e-5)
        assert d['overtaking'] == pytest.approx(1.1e-4)
        assert d['crossing'] == pytest.approx(1.3e-4)
        assert d['bend'] == pytest.approx(1.3e-4)
        assert d['grounding'] == pytest.approx(1.6e-4)
        assert d['allision'] == pytest.approx(1.9e-4)


class TestSetValuesAndCommitChanges:
    def test_set_values_writes_to_widgets(self, cf):
        """set_values pushes the 8 values to the corresponding widgets."""
        cf.data = dict(
            p_pc=0.1, d_pc=0.2,
            headon=0.3, overtaking=0.4, crossing=0.5,
            bend=0.6, grounding=0.7, allision=0.8,
        )
        cf.set_values()
        assert cf.cfw.lePoweredPc.text() == '0.1'
        assert cf.cfw.leDriftingPc.text() == '0.2'
        assert cf.cfw.leHeadOnCf.text() == '0.3'
        assert cf.cfw.leOvertakingCf.text() == '0.4'
        assert cf.cfw.leCrossingCf.text() == '0.5'
        assert cf.cfw.leBendCf.text() == '0.6'
        assert cf.cfw.leGroundingCf.text() == '0.7'
        assert cf.cfw.leAllisionCf.text() == '0.8'

    def test_commit_changes_reads_back_from_widgets(self, cf):
        """Commit reads back the current widget text and overwrites data."""
        cf.cfw.lePoweredPc.setText('0.11')
        cf.cfw.leDriftingPc.setText('0.22')
        cf.cfw.leHeadOnCf.setText('0.33')
        cf.cfw.leOvertakingCf.setText('0.44')
        cf.cfw.leCrossingCf.setText('0.55')
        cf.cfw.leBendCf.setText('0.66')
        cf.cfw.leGroundingCf.setText('0.77')
        cf.cfw.leAllisionCf.setText('0.88')

        cf.commit_changes()
        assert cf.data == {
            'p_pc': 0.11, 'd_pc': 0.22,
            'headon': 0.33, 'overtaking': 0.44,
            'crossing': 0.55, 'bend': 0.66,
            'grounding': 0.77, 'allision': 0.88,
        }

    def test_roundtrip_set_then_commit_preserves_data(self, cf):
        """set_values followed by commit_changes must be an identity."""
        original = dict(
            p_pc=0.00016, d_pc=1.0,
            headon=4.9e-5, overtaking=1.1e-4, crossing=1.3e-4,
            bend=1.3e-4, grounding=1.6e-4, allision=1.9e-4,
        )
        cf.data = dict(original)
        cf.set_values()
        cf.commit_changes()
        assert cf.data == original
