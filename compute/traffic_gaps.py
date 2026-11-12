"""Fill traffic cells that carry ships but no draught or speed (pure; no QGIS).

An AIS update collapses each cell's observations to their mean and writes
``inf`` when there were none (``AisUpdateTask._collapse_td_block``).  For
the draught that read as *unlimited*: those ships grounded on every depth
class, and the drift model let them anchor everywhere (KNOWN_ISSUES item 9;
one model had 5,903 ships/yr like that, mostly sailing and pleasure craft).
For the speed it turned every ship-ship encounter of the leg into
``0 * inf = NaN``, so the head-on, overtaking and merging totals of the
whole model came out NaN (KNOWN_ISSUES item 14; 27 ships/yr on one leg
whose frequencies had been edited by hand were enough).

:func:`fill_traffic_gaps` runs on the private copy of the data at the
start of a calculation (``CalculationTask.run``,
``sensitivity_task.run_phases_on_stub`` and
``compare_explain.run_collision_model`` -- keep them in step).  A missing
value (non-finite or <= 0) is filled, in this order, with the
traffic-weighted mean of

1. the same ship type and length class on the other legs,
2. the same ship type over all lengths,
3. the same length class over all types,
4. :data:`DEFAULT_DRAUGHT_M` / :data:`DEFAULT_SPEED_KN`.

The user is told how a gap is solved (:data:`FILL_RULE`) in three places:
the Traffic tab warns when an edit leaves a cell with ships but no speed /
draught and explains it in the cell tooltips (:func:`gap_cells`,
:func:`describe_new_gaps`, :func:`describe_gap_cell`), the result report's
Input completeness table counts the gaps (:func:`preview_traffic_gaps`), and
the run logs what it filled (:func:`describe_fill`).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

DRAUGHT_KEY = 'Draught (meters)'
SPEED_KEY = 'Speed (knots)'
FREQ_KEY = 'Frequency (ships/year)'
DEFAULT_DRAUGHT_M = 5.0
"""Last resort; the same value the powered model used for a missing draught."""
DEFAULT_SPEED_KN = 10.0
"""Last resort; the same value the powered and collision models use for a missing speed."""

SOURCES = ('type_and_length', 'type', 'length', 'default')

GAP_FIELDS = (('speed', SPEED_KEY), ('draught', DRAUGHT_KEY))
"""The values a run fills: ``(name used in messages, traffic matrix key)``."""

FILL_RULE = (
    "Run model fills the missing value with the traffic-weighted mean of the same ship type and "
    "length on the other legs (else the same type, else the same length, else "
    f"{DEFAULT_SPEED_KN:g} kn speed / {DEFAULT_DRAUGHT_M:g} m draught)."
)
"""How a gap is solved -- the one wording for the Traffic tab, its tooltips and the report."""


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _valid(value: Any) -> bool:
    v = _num(value)
    return math.isfinite(v) and v > 0.0


def _block_cells(block: Any):
    """Yield ``(i, j, q)`` for every cell of one direction block that carries ships."""
    freq = block.get(FREQ_KEY) if isinstance(block, dict) else None
    if not isinstance(freq, list):
        return
    for i, row in enumerate(freq):
        if not isinstance(row, list):
            continue
        for j, q in enumerate(row):
            q = _num(q)
            if math.isfinite(q) and q > 0.0:
                yield i, j, q


def _cells(traffic: dict[str, Any]):
    """Yield ``(block, i, j, q)`` for every cell that carries ships."""
    for dirs in traffic.values():
        if not isinstance(dirs, dict):
            continue
        for block in dirs.values():
            for i, j, q in _block_cells(block):
                yield block, i, j, q


def _fill_missing(data: dict[str, Any], field: str, default: float,
                  quantity: str, unit: str, apply: bool = True) -> dict[str, Any]:
    """Fill the missing ``field`` values of cells with ships, in place.

    Returns ``{'ships': float, 'cells': int, 'by_source': {source: ships}}``
    for the ships (per year) whose value was filled, plus the ``quantity``,
    ``default`` and ``unit`` :func:`describe_fill` words its line with.
    ``apply=False`` only counts (:func:`preview_traffic_gaps`).
    """
    traffic = data.get('traffic_data') or {}
    summary: dict[str, Any] = {'ships': 0.0, 'cells': 0, 'by_source': {s: 0.0 for s in SOURCES},
                               'quantity': quantity, 'default': default, 'unit': unit}
    if not isinstance(traffic, dict):
        return summary

    num: dict[tuple, float] = defaultdict(float)
    den: dict[tuple, float] = defaultdict(float)
    missing: list[tuple[dict, int, int, float]] = []
    for block, i, j, q in _cells(traffic):
        values = block.get(field)
        try:
            v = values[i][j]
        except (TypeError, IndexError, KeyError):
            continue
        if _valid(v):
            for key in ((i, j), (i, None), (None, j)):
                num[key] += q * float(v)
                den[key] += q
        else:
            missing.append((block, i, j, q))

    for block, i, j, q in missing:
        value, source = default, 'default'
        for key, name in (((i, j), 'type_and_length'), ((i, None), 'type'), ((None, j), 'length')):
            if den[key] > 0.0:
                value, source = num[key] / den[key], name
                break
        if apply:
            block[field][i][j] = round(value, 2)
        summary['ships'] += q
        summary['cells'] += 1
        summary['by_source'][source] += q
    return summary


def fill_missing_draughts(data: dict[str, Any]) -> dict[str, Any]:
    """Fill missing draughts in ``data['traffic_data']`` in place (see the module docstring)."""
    return _fill_missing(data, DRAUGHT_KEY, DEFAULT_DRAUGHT_M, 'draught', 'm')


def fill_missing_speeds(data: dict[str, Any]) -> dict[str, Any]:
    """Fill missing speeds in ``data['traffic_data']`` in place (see the module docstring).

    A speed of 0 counts as missing too: the ship crossed the passage line,
    and every encounter model divides by the speed.
    """
    return _fill_missing(data, SPEED_KEY, DEFAULT_SPEED_KN, 'speed', 'kn')


def fill_traffic_gaps(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Fill missing draughts and speeds; one summary each, for :func:`describe_fill`."""
    return [fill_missing_draughts(data), fill_missing_speeds(data)]


def preview_traffic_gaps(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The summaries :func:`fill_traffic_gaps` would return, without changing ``data``."""
    return [_fill_missing(data, DRAUGHT_KEY, DEFAULT_DRAUGHT_M, 'draught', 'm', apply=False),
            _fill_missing(data, SPEED_KEY, DEFAULT_SPEED_KN, 'speed', 'kn', apply=False)]


def describe_sources(summary: dict[str, Any]) -> str:
    """``'same type and length: 20, same type: 7'`` -- ships per fill source."""
    default = summary.get('default', DEFAULT_DRAUGHT_M)
    unit = summary.get('unit', 'm')
    labels = {'type_and_length': 'same type and length', 'type': 'same type', 'length': 'same length',
              'default': f'default {default:g} {unit}'}
    return ', '.join(f"{labels[name]}: {ships:.0f}" for name, ships in summary['by_source'].items() if ships)


def describe_fill(summary: dict[str, Any]) -> str:
    """One log line for a :func:`fill_missing_draughts` / :func:`fill_missing_speeds`
    summary ('' when nothing was filled)."""
    if not summary.get('cells'):
        return ''
    quantity = summary.get('quantity', 'draught')
    return (f"{summary['ships']:.0f} ships/year in {summary['cells']} traffic cells had no {quantity} and "
            f"got the mean {quantity} of the other legs ({describe_sources(summary)})")


# ---------------------------------------------------------------------------
# Traffic tab: cells an edit leaves with ships but no speed / draught
# ---------------------------------------------------------------------------

def cell_gaps(block: Any, i: int, j: int) -> list[str]:
    """Names (``'speed'``, ``'draught'``) cell ``(i, j)`` of one direction
    block lacks while it carries ships -- what a run will fill there."""
    try:
        q = _num(block[FREQ_KEY][i][j])
    except (TypeError, IndexError, KeyError):
        return []
    if not (math.isfinite(q) and q > 0.0):
        return []
    missing: list[str] = []
    for name, field in GAP_FIELDS:
        try:
            value = block[field][i][j]
        except (TypeError, IndexError, KeyError):
            continue
        if not _valid(value):
            missing.append(name)
    return missing


def gap_cells(block: Any) -> dict[str, set[tuple[int, int]]]:
    """``{'speed': cells, 'draught': cells}`` of one direction block that carry
    ships but no valid value -- the cells a run will fill."""
    out: dict[str, set[tuple[int, int]]] = {name: set() for name, _ in GAP_FIELDS}
    for i, j, _q in _block_cells(block):
        for name in cell_gaps(block, i, j):
            out[name].add((i, j))
    return out


def new_gaps(before: dict[str, set[tuple[int, int]]],
             after: dict[str, set[tuple[int, int]]]) -> dict[str, set[tuple[int, int]]]:
    """Gaps in ``after`` that were not in ``before`` (an edit made them)."""
    return {name: set(cells) - set(before.get(name, ())) for name, cells in after.items()}


def describe_gap_cell(missing: Sequence[str]) -> str:
    """Tooltip for one cell whose ships have no ``missing`` values ('' when none)."""
    if not missing:
        return ''
    return f"These ships have no {' or '.join(missing)}. {FILL_RULE} Type a value to use your own."


def describe_new_gaps(new: dict[str, set[tuple[int, int]]],
                      type_labels: Sequence[str] = (),
                      length_labels: Sequence[str] = (),
                      max_examples: int = 3) -> str:
    """Traffic-tab warning for the cells an edit left with ships but no
    speed / draught, and how the run solves it ('' when there are none)."""
    cells = sorted(set().union(*new.values())) if new else []
    if not cells:
        return ''
    missing = [name for name, _ in GAP_FIELDS if new.get(name)]

    def label(i: int, j: int) -> str:
        t = type_labels[i] if i < len(type_labels) else f'type {i + 1}'
        length = length_labels[j] if j < len(length_labels) else f'length class {j + 1}'
        return f'{t} {length}'

    shown = ', '.join(label(i, j) for i, j in cells[:max_examples])
    more = f' and {len(cells) - max_examples} more' if len(cells) > max_examples else ''
    fields = ' / '.join(field for name, field in GAP_FIELDS if name in missing)
    return (f"{len(cells)} cell(s) now have ships but no {' or '.join(missing)} ({shown}{more}). "
            f"{FILL_RULE} Enter {fields} for these cells to use your own values.")


WEIGHT_FIELDS = tuple(f'weight{d}_{i}' for d in (1, 2) for i in (1, 2, 3)) + ('u_p1', 'u_p2')
"""Mixture weights of the lateral distribution (normal components and the uniform part)."""


def negative_weights(segment_data: dict[str, Any] | None) -> list[tuple[str, str, float]]:
    """``(leg name, field, value)`` for every negative mixture weight.

    ``compute.data_preparation.get_distribution`` drops a component whose
    weight is not positive, so a negative weight silently changes the
    lateral distribution (weights of -4900 / -3300 have been seen, KNOWN_ISSUES 6).
    """
    found: list[tuple[str, str, float]] = []
    for seg_id, seg in (segment_data or {}).items():
        if not isinstance(seg, dict):
            continue
        for field in WEIGHT_FIELDS:
            try:
                value = float(seg.get(field, 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if value < 0:
                found.append((str(seg.get('Leg_name') or seg_id), field, value))
    return found


def describe_negative_weights(found: list[tuple[str, str, float]]) -> str:
    """One log line for :func:`negative_weights` ('' when there are none)."""
    if not found:
        return ''
    shown = ', '.join(f'{leg} {field} = {value:g}' for leg, field, value in found[:8])
    more = f' and {len(found) - 8} more' if len(found) > 8 else ''
    return (f'{len(found)} lateral-distribution weight(s) are negative and are ignored by the calculation: '
            f'{shown}{more}. Set them on the Distributions tab.')
