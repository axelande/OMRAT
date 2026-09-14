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


# ---------------------------------------------------------------------------
# Result display mode: annual frequency vs. years between incidents
# ---------------------------------------------------------------------------

DISPLAY_FREQUENCY = 'frequency'
DISPLAY_RETURN_PERIOD = 'return_period'
DISPLAY_MODES: tuple[str, ...] = (DISPLAY_FREQUENCY, DISPLAY_RETURN_PERIOD)

#: QSettings key remembering the chosen mode between sessions.
RESULT_DISPLAY_SETTING = 'omrat/result_display_mode'

#: Combo-box text per mode (order == ``DISPLAY_MODES``).
DISPLAY_MODE_LABELS: dict[str, str] = {
    DISPLAY_FREQUENCY: 'Frequency (per year)',
    DISPLAY_RETURN_PERIOD: 'Years between incidents',
}

INFINITY_TEXT = '∞'


def normalize_display_mode(mode: Any) -> str:
    """Coerce a stored / user value to one of ``DISPLAY_MODES``."""
    text = str(mode or '').strip().lower()
    return text if text in DISPLAY_MODES else DISPLAY_FREQUENCY


def return_period_years(value: float | None) -> float | None:
    """``1 / frequency`` -- the mean number of years between incidents.

    ``None`` stays ``None``; a zero (or negative) frequency gives
    ``inf`` because the incident never happens.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    if f <= 0:
        return float('inf')
    return 1.0 / f


def format_years(years: float | None) -> str:
    """Human-readable return period: ``12,345`` / ``45.6`` / ``2.34`` / ``∞``."""
    if years is None:
        return ''
    if years == float('inf'):
        return INFINITY_TEXT
    if years >= 100:
        return f'{years:,.0f}'
    if years >= 10:
        return f'{years:.1f}'
    return f'{years:.2f}'


def format_result(value: float | None, mode: str = DISPLAY_FREQUENCY) -> str:
    """Table text for an annual frequency in the requested display mode.

    The stored quantity is always the frequency (accidents / year); the
    return-period mode only changes how it is *shown*.
    """
    value = parse_probability(value) if not isinstance(value, (int, float)) else value
    if normalize_display_mode(mode) == DISPLAY_RETURN_PERIOD:
        return format_years(return_period_years(value))
    return format_probability(value)


def result_header(mode: str = DISPLAY_FREQUENCY) -> str:
    """Header of the value column in ``TWAccidentResults``."""
    if normalize_display_mode(mode) == DISPLAY_RETURN_PERIOD:
        return 'Years between incidents'
    return 'Probability'


def catastrophe_header(mode: str = DISPLAY_FREQUENCY) -> str:
    """Header of the value column in ``TWCatastropheResults``."""
    if normalize_display_mode(mode) == DISPLAY_RETURN_PERIOD:
        return 'Years between exceedances'
    return 'Exceedance (events/year)'


def catastrophe_label(mode: str = DISPLAY_FREQUENCY) -> str:
    """Caption above ``TWCatastropheResults``."""
    if normalize_display_mode(mode) == DISPLAY_RETURN_PERIOD:
        return 'Catastrophe-level exceedance (years between exceedances)'
    return 'Catastrophe-level exceedance (events/year)'
