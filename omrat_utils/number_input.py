"""Lenient numeric input for the settings dialogs.

Two small, QGIS-free helpers shared by the Drift settings dialog:

* :func:`parse_decimal` accepts both ``.`` and ``,`` as the decimal
  separator (``"12,5"`` -> ``12.5``) so users on a Swedish / German
  keyboard layout are not rejected by ``float()``.
* :func:`normalise_weights` rescales a list of weights so they sum to a
  target (100 % for the wind rose) without touching their proportions.
  It replaces the old per-field auto-adjust that rewrote the seven other
  compass fields every time one lost focus, which made entering a whole
  rose by hand practically impossible.
"""
from __future__ import annotations

from typing import Sequence

# Sums closer to the target than this are left alone by the check.
SUM_TOLERANCE = 1e-6


def parse_decimal(text: str, default: float | None = None) -> float:
    """Parse ``text`` as a float, accepting ``,`` as the decimal mark.

    Whitespace is stripped.  An empty string returns ``default`` when one
    is given and raises ``ValueError`` otherwise, like ``float('')``.
    A number with a thousands separator (``"1,234.5"``) is *not*
    supported: it contains both marks and raises ``ValueError``.
    """
    s = str(text).strip()
    if not s:
        if default is not None:
            return float(default)
        raise ValueError("empty number")
    if ',' in s and '.' in s:
        raise ValueError(f"could not convert string to float: {text!r}")
    return float(s.replace(',', '.'))


def normalise_weights(values: Sequence[float], target: float = 100.0,
                      ndigits: int = 2) -> list[float]:
    """Scale ``values`` proportionally so they sum to ``target``.

    The result is rounded to ``ndigits`` and the rounding residue is put
    on the largest weight so the rounded values still sum exactly to
    ``target``.  Negative inputs raise ``ValueError``; an all-zero input
    becomes a uniform split.
    """
    vals = [float(v) for v in values]
    if not vals:
        return []
    if any(v < 0 for v in vals):
        raise ValueError("weights must be non-negative")
    total = sum(vals)
    if total <= 0:
        vals = [1.0] * len(vals)
        total = float(len(vals))
    scaled = [round(v * target / total, ndigits) for v in vals]
    residue = round(target - sum(scaled), ndigits)
    if residue:
        idx = max(range(len(scaled)), key=lambda i: scaled[i])
        scaled[idx] = round(scaled[idx] + residue, ndigits)
    return scaled


def weights_sum_ok(values: Sequence[float], target: float = 100.0,
                   tol: float = SUM_TOLERANCE) -> bool:
    """True when ``values`` already sum to ``target`` within ``tol``."""
    return abs(sum(float(v) for v in values) - target) <= tol


def format_weight(value: float, ndigits: int = 2) -> str:
    """Render a weight for a line edit: ``12.5`` not ``12.50``, ``0`` -> ``0.0``."""
    text = f"{round(float(value), ndigits):.{ndigits}f}".rstrip('0')
    if text.endswith('.'):
        text += '0'
    return text
