"""Regression test for the anchor-shadow lookup fix in the drifting
cascade.

The bug: `_precompute_bucket_memo` at compute/drifting_model.py:828 (and
the fallback path at :2471) looked up the shadow with
`shadows_map.get((obs_type, obs_idx))`, but `_precompute_shadow_layer`
stores shadows under `('depth', idx)` -- never `('anchoring', idx)`.  The
lookup therefore returned `None`, `anchor_union` never populated, and
`h_in_anchor` was 0 for every grounding entry, making the `a_p`
multiplication in `h_eff = h_reach - a_p * h_in_anchor` dead code.

The fix maps `obs_type == 'anchoring'` to `lookup_type == 'depth'` at
the lookup site.  With the fix applied, `a_p` has an observable effect
on the grounding total: polygons that sit inside their own anchor
shadow get `h_eff = (1 - a_p) * h_reach`.

Test strategy: run the production cascade twice on the same data,
varying only `drift.anchor_p` between 0.0 and 0.7.  With the fix,
increasing `anchor_p` MUST decrease the grounding total.  Without the
fix, the two runs return exactly the same number.

Run standalone:
    /c/OSGeo4W/apps/Python312/python.exe -m pytest -p no:qgis \\
        --noconftest tests/test_regression_anchor_lookup.py -v
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mock_parent():
    mp = MagicMock()
    mp.main_widget = MagicMock()
    mp.main_widget.LEPDriftAllision.setText = MagicMock()
    mp.main_widget.LEPDriftingGrounding.setText = MagicMock()
    mp.main_widget.cbShipCategories.count = MagicMock(return_value=0)
    mp.main_widget.LEReportPath.text = MagicMock(return_value='')
    return mp


def _load_project(path: Path) -> dict:
    with path.open(encoding='utf-8') as fh:
        return json.load(fh)


def _run_cascade(data: dict) -> tuple[float, float]:
    """Run the drifting cascade and return (allision, grounding) totals."""
    from compute.basic_equations import default_blackout_by_ship_type
    from compute.run_calculations import Calculation

    drift = data.setdefault('drift', {})
    drift.setdefault('blackout_by_ship_type', default_blackout_by_ship_type())

    calc = Calculation(_mock_parent())
    calc.set_progress_callback(lambda completed, total, msg: True)
    return calc.run_drifting_model(data)


@pytest.fixture
def proj_data():
    """Load the small example project; it has at least one depth polygon
    close enough to a leg to exercise the anchor overlay.
    """
    return _load_project(ROOT / 'tests' / 'example_data' / 'proj.omrat')


@pytest.mark.slow
def test_anchor_p_actually_reduces_grounding(proj_data):
    """Increasing anchor_p from 0 to 0.7 MUST reduce the grounding total.

    If this test fails with identical numbers, the anchor-shadow lookup
    bug is back.
    """
    data_zero = copy.deepcopy(proj_data)
    data_zero['drift']['anchor_p'] = 0.0

    data_seven = copy.deepcopy(proj_data)
    data_seven['drift']['anchor_p'] = 0.7

    _, grounding_zero = _run_cascade(data_zero)
    _, grounding_seven = _run_cascade(data_seven)

    assert grounding_zero > 0, (
        "Sanity check: baseline grounding total must be positive "
        "(otherwise the test can't distinguish the two runs)."
    )
    assert grounding_seven < grounding_zero, (
        f"anchor_p=0.7 should REDUCE the grounding total, but got "
        f"{grounding_seven:.4e} vs {grounding_zero:.4e} at anchor_p=0.  "
        f"If these are equal, the anchor-shadow lookup bug is back: "
        f"check the `lookup_type = 'depth' if obs_type == 'anchoring' "
        f"else obs_type` remap in compute/drifting_model.py near lines "
        f"828 and 2471."
    )


def test_shadow_lookup_remaps_anchoring_to_depth_in_source():
    """Static guard: the cascade MUST remap `obs_type='anchoring'` to
    `lookup_type='depth'` before calling `shadows_map.get(...)`, because
    shadows are stored under `('depth', idx)` by
    `_precompute_shadow_layer`.

    This fast test complements the slow cascade-based characterization
    test below.  It catches the bug purely by inspecting the source --
    no cascade run required -- so the regression gate still closes even
    if the characterization test is skipped for speed.
    """
    import re
    src_path = ROOT / 'compute' / 'drifting_model.py'
    src = src_path.read_text(encoding='utf-8')

    # Find every `shadows_map.get((...))` call and verify the preceding
    # few lines remap 'anchoring' to 'depth'.
    lookups = [(m.start(), m.group(0))
               for m in re.finditer(r"shadows_map\.get\(\(", src)]
    assert lookups, "shadows_map.get((...)) not found in drifting_model.py"
    for offset, _ in lookups:
        # Look at the 500 chars preceding this call: the remap line
        # should appear.
        preceding = src[max(0, offset - 500):offset]
        assert "lookup_type" in preceding and "'anchoring'" in preceding, (
            f"shadows_map.get at offset {offset}: preceding code must "
            f"define `lookup_type = 'depth' if obs_type == 'anchoring' "
            f"else obs_type`; otherwise anchor_union is never populated. "
            f"Preceding 500 chars were:\n---\n{preceding}\n---"
        )


@pytest.mark.slow
def test_anchor_p_reduction_matches_cascade_formula(proj_data):
    """The reduction factor `grounding(a_p=0.7) / grounding(a_p=0)` must
    be in a physically plausible range [0.0, 1.0].

    For a scene where every depth polygon sits inside its own anchor
    shadow (which `create_obstacle_shadow` guarantees), the ratio tends
    toward `1 - a_p = 0.3`.  For a scene where only some polygons are
    anchor candidates, the ratio is between 0.3 and 1.0.
    """
    data_zero = copy.deepcopy(proj_data)
    data_zero['drift']['anchor_p'] = 0.0

    data_seven = copy.deepcopy(proj_data)
    data_seven['drift']['anchor_p'] = 0.7

    _, grounding_zero = _run_cascade(data_zero)
    _, grounding_seven = _run_cascade(data_seven)

    if grounding_zero <= 0:
        pytest.skip("Fixture produced zero grounding -- can't ratio-test.")

    ratio = grounding_seven / grounding_zero
    # Allow a small tolerance above 0.3 for polygons not in their own
    # anchor shadow; cap below 1.0 to ensure SOME reduction happened.
    assert 0.29 <= ratio <= 0.999, (
        f"Anchor-p=0.7 reduction ratio is {ratio:.3f}, expected in "
        f"[0.29, 0.999].  A ratio of 1.0 exactly means the anchor "
        f"reduction was not applied (the bug).  A ratio below 0.29 "
        f"would indicate the cascade over-applied the reduction."
    )
