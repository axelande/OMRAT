"""Text <-> number and clipboard (TSV) helpers for the editable matrix
tables, so a block can go to Excel and back.

Pure Python (no Qt) so the standalone test suite covers it; the Traffic
tab (``handle_traffic.Traffic``) supplies the locale's decimal point and
group separator from ``QLocale``.

Excel copies a range as tab-separated rows using the *system* locale --
``12,5`` in Swedish, ``12.5`` / ``1,234.5`` in English -- so
:func:`parse_number` accepts both separators and drops grouping, and
:func:`format_number` writes the locale's decimal point so Excel reads the
cells as numbers.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# Characters Excel / Qt use to group thousands that never mean anything else.
_SPACES = (' ', ' ', ' ', ' ')


def parse_number(text: str, decimal_point: str = '.', group_separator: str = ',') -> float | None:
    """``text`` as a float; ``None`` for a blank cell.

    Raises ``ValueError`` for anything that is not a finite number.  A
    trailing ``%`` is ignored (Scaling cells).  With both ``,`` and ``.``
    present the last one is the decimal separator; a lone separator is a
    decimal separator unless it is the locale's group separator in a
    thousands pattern (``1,234`` in English).
    """
    s = str(text).strip()
    for ch in _SPACES:
        s = s.replace(ch, '')
    if s.endswith('%'):
        s = s[:-1]
    if s == '':
        return None
    if ',' in s and '.' in s:
        dec = ',' if s.rfind(',') > s.rfind('.') else '.'
        other = '.' if dec == ',' else ','
        s = s.replace(other, '').replace(dec, '.')
    elif ',' in s or '.' in s:
        sep = ',' if ',' in s else '.'
        thousands = re.fullmatch(r'[+-]?\d{1,3}(' + re.escape(sep) + r'\d{3})+', s)
        if sep == group_separator and sep != decimal_point and thousands:
            s = s.replace(sep, '')
        else:
            s = s.replace(sep, '.')
    value = float(s)
    if not math.isfinite(value):
        raise ValueError(f"not a finite number: {text!r}")
    return value


def format_number(value, decimal_point: str = '.', decimals: int = 2) -> str:
    """Cell text for ``value``: integers without decimals, others with at
    most ``decimals`` (trailing zeros dropped), blank for ``''`` / ``None``
    / non-finite / an uncollapsed observation list (no observation)."""
    if value is None or isinstance(value, (list, tuple)) or value == '':
        return ''
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return ''
    if v.is_integer():
        return str(int(v))
    text = f'{v:.{decimals}f}'.rstrip('0').rstrip('.')
    if text in ('', '-0'):
        text = '0'
    return text.replace('.', decimal_point)


def to_tsv(rows: list[list[str]]) -> str:
    """Tab-separated rows, the format Excel puts on / takes off the clipboard."""
    return '\n'.join('\t'.join(r) for r in rows)


def parse_tsv(text: str) -> list[list[str]]:
    """Clipboard text -> rows of cell strings.  Accepts ``\\r\\n`` / ``\\r``
    line ends, drops the trailing line end Excel adds, and strips the
    quotes Excel puts around a cell holding a tab or line break."""
    t = str(text or '').replace('\r\n', '\n').replace('\r', '\n')
    if t.endswith('\n'):
        t = t[:-1]
    if t == '':
        return []
    out = []
    for line in t.split('\n'):
        cells = []
        for c in line.split('\t'):
            if len(c) >= 2 and c[0] == '"' and c[-1] == '"':
                c = c[1:-1].replace('""', '"')
            cells.append(c)
        out.append(cells)
    return out


def _is_number(text: str, decimal_point: str, group_separator: str) -> bool:
    try:
        return parse_number(text, decimal_point, group_separator) is not None
    except ValueError:
        return False


def strip_headers(
    rows: list[list[str]], decimal_point: str = '.', group_separator: str = ',',
) -> tuple[list[list[str]], bool, bool]:
    """Drop a header row / header column (``Copy with headers``, or a
    labelled block from Excel).  Returns ``(rows, had_row, had_col)``.

    A header is text in *every* cell of the first row (after a blank or
    text corner) / of the first column (with more columns after it) --
    length-interval and ship-type labels are never blank -- so a typo in
    a numeric block is reported as invalid, never mistaken for a header.
    """
    def label(c: str) -> bool:
        return bool(c.strip()) and not _is_number(c, decimal_point, group_separator)

    had_row = had_col = False
    if (rows and len(rows[0]) > 1 and all(label(c) for c in rows[0][1:])
            and not _is_number(rows[0][0], decimal_point, group_separator)):
        rows = rows[1:]
        had_row = True
    if rows and any(len(r) > 1 for r in rows) and all(r and label(r[0]) for r in rows):
        rows = [r[1:] for r in rows]
        had_col = True
    return rows, had_row, had_col


@dataclass
class PastePlan:
    """Cells a paste writes, as ``(row, col, text)``, plus what it left out."""
    cells: list[tuple[int, int, str]] = field(default_factory=list)
    clipped: int = 0


def plan_paste(
    rows: list[list[str]], n_rows: int, n_cols: int, anchor: tuple[int, int],
    selection: list[tuple[int, int]] | None = None,
) -> PastePlan:
    """Where a clipboard block lands in an ``n_rows`` x ``n_cols`` table.

    One value with several cells selected fills the selection (Excel
    behaviour); otherwise the block starts at ``anchor`` and anything past
    the table edge is counted in ``clipped``.
    """
    plan = PastePlan()
    if not rows:
        return plan
    if len(rows) == 1 and len(rows[0]) == 1 and selection and len(selection) > 1:
        plan.cells = [(r, c, rows[0][0]) for r, c in selection if 0 <= r < n_rows and 0 <= c < n_cols]
        return plan
    r0, c0 = anchor
    for i, line in enumerate(rows):
        for j, text in enumerate(line):
            r, c = r0 + i, c0 + j
            if 0 <= r < n_rows and 0 <= c < n_cols:
                plan.cells.append((r, c, text))
            else:
                plan.clipped += 1
    return plan
