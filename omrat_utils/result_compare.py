"""Run columns of the Run Analysis result tables (pure: no Qt / QGIS).

The accident table and the catastrophe table on the Run Analysis tab show
saved runs side by side (user's design, 2026-11-09)::

    labels | <baseline> (baseline) | run A | Change (%) | run B | Change (%) | View

* The **baseline** is the run ticked in the Baseline column of Previous
  runs.  It is shown once, values only, as the first run column, whether
  or not its row is selected.
* Every other selected run gets its value and its change against the
  baseline.  Without a baseline the runs show their values only.
* The change is computed in the unit shown: the annual frequency, or the
  years between incidents (``1 / frequency``).  A frequency that doubles
  is +100 % as a frequency and -50 % in years.
* There is no column for the live model: a finished run reaches the
  tables through the run history (the new run is selected).

Copy to Excel takes numbers, not the cell text: :func:`copy_number`
writes the locale's decimal point and six significant digits
(``1,234e-05`` in Swedish), and a change is copied as a plain number of
percent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from omrat_utils.accident_summary import (
    ACCIDENT_TOTAL_KEYS, DISPLAY_FREQUENCY, DISPLAY_RETURN_PERIOD, INFINITY_TEXT,
    format_result, normalize_display_mode, parse_probability, return_period_years,
    summary_values,
)

VALUE = 'value'
CHANGE = 'change'
DASH = '—'
BASELINE_SUFFIX = ' (baseline)'
CHANGE_HEADER = 'Change (%)'

NO_CONSEQUENCE_NOTE = (
    'No catastrophe data was saved for this run (recorded before the '
    '.consequence.json sidecar existed). Re-run the model to record it.'
)


@dataclass
class RunValues:
    """One run's numbers, row by row: annual frequencies, ``None`` = no
    value.  ``values=None`` means nothing was saved for the run at all
    (shown as dashes); ``note`` then says why."""

    name: str
    values: Sequence[float | None] | None
    run_id: int | None = None
    note: str = ''


@dataclass
class CompareColumn:
    """One run column of a result table.

    ``cells`` holds annual frequencies for a ``VALUE`` column and percent
    for a ``CHANGE`` column (``None`` = no value / not defined).
    """

    kind: str
    header: str
    tooltip: str
    cells: list[float | None] = field(default_factory=list)
    run_id: int | None = None
    is_baseline: bool = False
    has_data: bool = True


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def change_pct(value: Any, base: Any, mode: str = DISPLAY_FREQUENCY) -> float | None:
    """Change of ``value`` against ``base`` in percent, in the unit shown.

    Both arguments are annual frequencies.  In the years mode the change
    is that of ``1 / frequency``; a zero frequency (never happens) makes
    the years infinite and the change undefined.  ``None`` when either
    value is missing or the base is zero.
    """
    v = _as_float(value)
    b = _as_float(base)
    if v is None or b is None:
        return None
    if normalize_display_mode(mode) == DISPLAY_RETURN_PERIOD:
        if v <= 0 or b <= 0:
            return None
        return (b / v - 1.0) * 100.0
    if b == 0:
        return None
    return (v - b) / b * 100.0


def change_text(pct: float | None) -> str:
    """Cell text of a change: ``+12.3%`` (one decimal), dash when undefined."""
    if pct is None or not math.isfinite(pct):
        return DASH
    return f'{pct:+.1f}%'


def value_text(value: Any, mode: str = DISPLAY_FREQUENCY, *, has_data: bool = True,
               missing: str = '') -> str:
    """Cell text of a run value in the display mode.

    ``has_data=False`` (nothing saved for the run) shows a dash; a single
    missing value (a part the run did not calculate) shows ``missing``.
    """
    if not has_data:
        return DASH
    v = _as_float(value)
    if v is None:
        return missing
    return format_result(v, mode)


def copy_number(value: Any, decimal_point: str = '.') -> str:
    """Clipboard text of a number for Excel: six significant digits with
    the locale's decimal point; blank for ``None``, ``∞`` for infinity."""
    v = _as_float(value)
    if v is None:
        return ''
    if math.isinf(v):
        return INFINITY_TEXT
    return f'{v:.6g}'.replace('.', decimal_point)


def value_copy_text(value: Any, mode: str = DISPLAY_FREQUENCY, decimal_point: str = '.') -> str:
    """Clipboard text of a run value, in the unit shown (frequency or years)."""
    v = _as_float(value)
    if v is None:
        return ''
    if normalize_display_mode(mode) == DISPLAY_RETURN_PERIOD:
        return copy_number(return_period_years(v), decimal_point)
    return copy_number(v, decimal_point)


def _cells(run: RunValues, n_rows: int) -> list[float | None]:
    values = [_as_float(v) for v in (run.values or [])][:n_rows]
    return values + [None] * (n_rows - len(values))


def build_compare_columns(
    baseline: RunValues | None,
    runs: Sequence[RunValues],
    n_rows: int,
    mode: str = DISPLAY_FREQUENCY,
) -> list[CompareColumn]:
    """The run columns, left to right: the baseline (values only), then a
    value column per run and -- with a baseline -- its change column.

    A selected run that is the baseline is not repeated.
    """
    columns: list[CompareColumn] = []
    base_cells: list[float | None] | None = None
    if baseline is not None:
        base_cells = _cells(baseline, n_rows)
        columns.append(CompareColumn(
            kind=VALUE,
            header=f'{baseline.name}{BASELINE_SUFFIX}',
            tooltip=baseline.note or (
                f'Baseline run "{baseline.name}": the change columns are '
                'computed against it.  Untick Baseline in Previous runs to remove it.'),
            cells=base_cells,
            run_id=baseline.run_id,
            is_baseline=True,
            has_data=baseline.values is not None,
        ))
    for run in runs:
        if (baseline is not None and run.run_id is not None
                and run.run_id == baseline.run_id):
            continue
        cells = _cells(run, n_rows)
        columns.append(CompareColumn(
            kind=VALUE,
            header=run.name,
            tooltip=run.note or f'Run "{run.name}"',
            cells=cells,
            run_id=run.run_id,
            has_data=run.values is not None,
        ))
        if base_cells is None:
            continue
        columns.append(CompareColumn(
            kind=CHANGE,
            header=CHANGE_HEADER,
            tooltip=(f'Change of "{run.name}" compared with the baseline '
                     f'"{baseline.name}", in % of the baseline value as shown'),
            cells=[change_pct(v, b, mode) for v, b in zip(cells, base_cells)],
            run_id=run.run_id,
        ))
    return columns


def compare_hint(baseline_name: str | None, n_runs: int) -> str:
    """One line above the tables saying what they show and what to do."""
    if baseline_name:
        return f'Change compared with the baseline: {baseline_name}'
    if n_runs:
        return 'Tick Baseline in Previous runs to add change columns'
    return 'Select runs in Previous runs to show their results'


# ---------------------------------------------------------------------------
# Row values
# ---------------------------------------------------------------------------

def accident_values(totals: Mapping[str, Any] | None) -> list[float | None]:
    """The nine accident rows (``ACCIDENT_TOTAL_KEYS``) followed by the
    three summary rows, as annual frequencies."""
    totals = totals or {}
    values = [parse_probability(totals.get(k)) for k in ACCIDENT_TOTAL_KEYS]
    return values + summary_values(totals)


def _levels(result: Any) -> list[dict]:
    if not isinstance(result, dict):
        return []
    return [lvl for lvl in (result.get('levels') or []) if isinstance(lvl, dict)]


def catastrophe_rows(results: Sequence[Any]) -> list[tuple[str, float | None]]:
    """``(level name, threshold m3)`` for every level in ``results``, in
    order of first appearance (baseline first when it is passed first)."""
    rows: list[tuple[str, float | None]] = []
    seen: set[str] = set()
    for result in results:
        for lvl in _levels(result):
            name = str(lvl.get('name', ''))
            if name in seen:
                continue
            seen.add(name)
            rows.append((name, _as_float(lvl.get('quantity'))))
    return rows


def exceedance_values(result: Any, names: Sequence[str]) -> list[float | None] | None:
    """Exceedance per level name (matched by name); ``None`` without a result."""
    if not isinstance(result, dict):
        return None
    by_name = {str(lvl.get('name', '')): _as_float(lvl.get('exceedance')) for lvl in _levels(result)}
    return [by_name.get(name) for name in names]
