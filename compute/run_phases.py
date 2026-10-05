"""Model phases of one **Run model**, and which of them a run calculates.

``CalculationTask`` runs the selected phases in ``PHASES`` order; the
**Calculation matrix** dialog (``omrat_utils/calculation_matrix_dialog.py``)
edits the selection.  A phase a run skips leaves its results *empty*
(``None`` report, ``None`` history total, blank table row) instead of
zero, so nothing claims a value that was never calculated.

Pure Python -- no QGIS imports, so the sensitivity analysis and the
standalone tests share this vocabulary.
"""
from __future__ import annotations

import copy
from typing import Any, Iterable

# Model phases, in the order ``CalculationTask`` runs them.
PHASES: tuple[str, ...] = (
    'drifting', 'collision', 'powered_grounding', 'powered_allision',
)
ALL_PHASES: tuple[str, ...] = PHASES

PHASE_LABELS: dict[str, str] = {
    'drifting': 'Drifting',
    'collision': 'Ship-ship collisions',
    'powered_grounding': 'Powered grounding',
    'powered_allision': 'Powered allision',
}

# Which accident totals (``RunHistory`` keys) each phase produces.
PHASE_TOTALS: dict[str, tuple[str, ...]] = {
    'drifting': ('drift_allision', 'drift_grounding'),
    'collision': ('overtaking', 'head_on', 'crossing', 'merging', 'bend'),
    'powered_grounding': ('powered_grounding',),
    'powered_allision': ('powered_allision',),
}

# ``Calculation`` method that runs each phase.
PHASE_METHODS: dict[str, str] = {
    'drifting': 'run_drifting_model',
    'collision': 'run_ship_collision_model',
    'powered_grounding': 'run_powered_grounding_model',
    'powered_allision': 'run_powered_allision_model',
}

# Attributes each phase writes on ``Calculation`` and their value before
# the phase has run ("not calculated").
PHASE_RESULT_ATTRS: dict[str, dict[str, Any]] = {
    'drifting': {
        'drifting_report': None,
        'drifting_allision_prob': 0.0,
        'drifting_grounding_prob': 0.0,
        'allision_result_layer': None,
        'grounding_result_layer': None,
        '_last_structures': [],
        '_last_depths': [],
    },
    'collision': {
        'collision_report': None,
        'ship_collision_prob': 0.0,
        'collision_line_layer': None,
        'collision_point_layer': None,
    },
    'powered_grounding': {
        'powered_grounding_report': None,
        'powered_grounding_layer': None,
    },
    'powered_allision': {
        'powered_allision_report': None,
        'powered_allision_layer': None,
        '_last_powered_allision_structs': [],
    },
}


def normalize_phases(phases: Iterable[str] | None) -> tuple[str, ...]:
    """``phases`` as a tuple in ``PHASES`` order.

    ``None`` means every phase; unknown names are dropped, so an empty
    tuple comes back when nothing valid is selected.
    """
    if phases is None:
        return PHASES
    wanted = {str(p) for p in phases}
    return tuple(p for p in PHASES if p in wanted)


def skipped_phases(phases: Iterable[str] | None) -> tuple[str, ...]:
    """The phases a run with selection ``phases`` leaves out."""
    selected = normalize_phases(phases)
    return tuple(p for p in PHASES if p not in selected)


def is_partial(phases: Iterable[str] | None) -> bool:
    return bool(skipped_phases(phases))


def skipped_totals(phases: Iterable[str] | None) -> tuple[str, ...]:
    """Accident-total keys (``drift_allision`` ...) a run leaves empty."""
    return tuple(key for p in skipped_phases(phases) for key in PHASE_TOTALS[p])


def phase_labels(phases: Iterable[str] | None) -> list[str]:
    return [PHASE_LABELS[p] for p in normalize_phases(phases)]


def describe_selection(phases: Iterable[str] | None) -> str:
    """``'all parts'``, ``'nothing'`` or the selected parts' labels."""
    selected = normalize_phases(phases)
    if len(selected) == len(PHASES):
        return 'all parts'
    if not selected:
        return 'nothing'
    return ', '.join(PHASE_LABELS[p] for p in selected)


def reset_results(calc: Any, phases: Iterable[str] | None = None) -> None:
    """Put the result attributes of ``phases`` (default: all) on ``calc``
    back to their not-calculated values."""
    for phase in normalize_phases(phases):
        for attr, value in PHASE_RESULT_ATTRS[phase].items():
            setattr(calc, attr, copy.copy(value))
