"""Pure helpers shared by the accident-results table and its comparison
columns (no Qt / QGIS imports so they are unit-testable standalone).

``ACCIDENT_TOTAL_KEYS`` is the row order of the nine accident rows in
``TWAccidentResults`` and must stay in step with
``AccidentResultsMixin._ACCIDENT_ROWS``.  ``SUMMARY_ROWS`` are the three
aggregate rows appended below them.
"""
from __future__ import annotations

from typing import Any, Mapping

ACCIDENT_TOTAL_KEYS: tuple[str, ...] = (
    'drift_allision', 'drift_grounding',
    'powered_allision', 'powered_grounding',
    'overtaking', 'head_on', 'crossing', 'merging', 'bend',
)

# (Row label, accident keys summed into that row).
SUMMARY_ROWS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('All grounding', ('drift_grounding', 'powered_grounding')),
    ('All allision', ('drift_allision', 'powered_allision')),
    ('All collisions', ('overtaking', 'head_on', 'crossing', 'merging', 'bend')),
)


def parse_probability(text: Any) -> float | None:
    """``'1.234e-05'`` -> ``1.234e-05``; blanks / dashes -> ``None``."""
    if text is None:
        return None
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def summary_values(totals: Mapping[str, Any]) -> list[float | None]:
    """One aggregate per ``SUMMARY_ROWS`` entry.

    A row is ``None`` only when *every* component is missing or
    unparsable; otherwise missing components count as zero so a run
    that never computed e.g. powered grounding still gets a grounding
    total.
    """
    out: list[float | None] = []
    for _label, keys in SUMMARY_ROWS:
        acc: float | None = None
        for key in keys:
            value = parse_probability(totals.get(key))
            if value is None:
                continue
            acc = value if acc is None else acc + value
        out.append(acc)
    return out


def format_probability(value: float | None) -> str:
    """Table text for a probability (matches the LEP* ``.3e`` format)."""
    if value is None:
        return ''
    if value == 0:
        return '0'
    return f'{value:.3e}'
