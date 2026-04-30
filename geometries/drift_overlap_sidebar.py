"""QTableWidget sidebar for the drifting-overlap dialog.

The sidebar lists every ``(leg, direction)`` pair next to the per-pair
contribution to drift allision / grounding probability.  Clicking a
row selects that leg+direction in the matplotlib axes.

The pre-conputed contribution comes from the calc's ``drifting_report``
under ``by_leg_direction[<seg>:<legdir>:<compass>]``.  When the
report is missing we leave ``--`` cells rather than substitute a
geometric coverage that has different semantics from
``contrib_grounding`` / ``contrib_allision`` (and confused users).
"""
from __future__ import annotations

from typing import Any

# Compass-direction labels for the 8 drift polygons.
# ``extend_polygon_in_directions`` iterates math angles 0,45,...,315
# (CCW from East).  Polygon index N corresponds to compass angle
# ``(90 - N*45) % 360``.
DIRECTION_LABELS: tuple[str, ...] = (
    "East",       # math 0
    "North-East", # math 45
    "North",      # math 90
    "North-West", # math 135
    "West",       # math 180
    "South-West", # math 225
    "South",      # math 270
    "South-East", # math 315
)


def polygon_to_compass(direction_index: int) -> int:
    """Compass-angle for sidebar row's polygon index."""
    return (90 - direction_index * 45) % 360


def split_leg_name(name: str) -> tuple[str, str]:
    """``"Leg 1-East going"`` -> ``("1", "East going")``."""
    try:
        head, _, tail = str(name).partition('-')
        seg = head.replace('Leg', '').strip()
        return seg, tail.strip()
    except Exception:
        return str(name), ''


def lookup_contribution(
    bld: dict[str, Any],
    seg_id: str,
    legdir: str,
    compass: int,
    contrib_key: str,
) -> float | None:
    """Pull a single contribution value from ``by_leg_direction``."""
    if not bld:
        return None
    rec = bld.get(f"{seg_id}:{legdir}:{compass}", {}) or {}
    try:
        return float(rec.get(contrib_key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return None


def make_numeric_item_class():
    """Build a ``QTableWidgetItem`` subclass that sorts by float UserRole.

    Returns a class -- callers instantiate it.  Defined as a factory
    so the whole module stays importable without a Qt event loop in
    the way (the class definition still imports Qt though).
    """
    from qgis.PyQt.QtCore import Qt as _Qt
    from qgis.PyQt.QtWidgets import QTableWidgetItem

    class _NumericItem(QTableWidgetItem):
        """Sort by float value stored on UserRole."""

        def __lt__(self, other):
            try:
                return (
                    self.data(_Qt.ItemDataRole.UserRole)
                    < other.data(_Qt.ItemDataRole.UserRole)
                )
            except Exception:
                return super().__lt__(other)

    return _NumericItem


def _format_prob(value: float | None) -> tuple[str, float]:
    """``(display_text, sort_value)`` for the contribution column."""
    if value is None:
        return '--', -1.0
    if value <= 0.0:
        return '0', 0.0
    return f"{value:.3e}", value


def build_overlap_sidebar(
    line_names: list[str],
    drifting_report: dict[str, Any] | None,
    accident_kind: str,
):
    """Build the ``QTableWidget`` for the overlap dialog sidebar.

    Each row is one ``(leg, direction)`` pair; sortable by
    contribution value.  The ``Leg`` cell carries the original
    ``(leg_idx, dir_idx)`` tuple in its ``UserRole`` payload so click
    handlers can recover the original index after the user sorts the
    column.
    """
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtWidgets import (
        QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem,
    )

    NumericItem = make_numeric_item_class()

    bld = (drifting_report or {}).get('by_leg_direction', {}) or {}
    contrib_key = (
        'contrib_grounding' if accident_kind == 'grounding'
        else 'contrib_allision'
    )

    n_dirs = len(DIRECTION_LABELS)
    sidebar = QTableWidget(len(line_names) * n_dirs, 3)
    prob_header = (
        'Grounding' if accident_kind == 'grounding' else 'Allision'
    )
    sidebar.setHorizontalHeaderLabels(['Leg', 'Direction', prob_header])
    sidebar.verticalHeader().setVisible(False)
    sidebar.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    sidebar.setSelectionBehavior(
        QAbstractItemView.SelectionBehavior.SelectRows,
    )
    sidebar.setSelectionMode(
        QAbstractItemView.SelectionMode.SingleSelection,
    )

    for li, leg_name in enumerate(line_names):
        seg_id, legdir = split_leg_name(leg_name)
        for di, dir_label in enumerate(DIRECTION_LABELS):
            row = li * n_dirs + di
            leg_item = QTableWidgetItem(str(leg_name))
            leg_item.setData(Qt.ItemDataRole.UserRole, (li, di))
            sidebar.setItem(row, 0, leg_item)
            sidebar.setItem(row, 1, QTableWidgetItem(dir_label))

            prob_value = lookup_contribution(
                bld, seg_id, legdir,
                polygon_to_compass(di), contrib_key,
            )
            prob_text, sort_value = _format_prob(prob_value)
            prob_item = NumericItem(prob_text)
            prob_item.setData(Qt.ItemDataRole.UserRole, sort_value)
            prob_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter,
            )
            sidebar.setItem(row, 2, prob_item)

    # Enable sorting AFTER population so the rows don't reorder mid-fill.
    sidebar.setSortingEnabled(True)
    sidebar.horizontalHeader().setSectionResizeMode(
        0, QHeaderView.ResizeMode.ResizeToContents,
    )
    sidebar.horizontalHeader().setSectionResizeMode(
        1, QHeaderView.ResizeMode.Stretch,
    )
    sidebar.horizontalHeader().setSectionResizeMode(
        2, QHeaderView.ResizeMode.ResizeToContents,
    )
    sidebar.setMinimumWidth(220)
    sidebar.setMaximumWidth(360)
    return sidebar
