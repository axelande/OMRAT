"""Junction transition-matrix editor dialog.

Lists every junction in the project and lets the user pick one to edit.
The selected junction's matrix is rendered as a square table whose row
labels are inbound legs and column labels are outbound legs; cells are
spinboxes accepting percentages 0-100.  On OK each row is normalised
back to 1.0 and the junction is marked ``source="user"`` so the next
"Update all distributions" pass leaves it alone.

Kept import-light so the standalone test suite can import the helpers
without a Qt event loop — the dialog construction itself is wrapped in
``_build_dialog`` and invoked from ``run`` only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from omrat_utils.widgets import NoWheelDoubleSpinBox

if TYPE_CHECKING:
    from omrat import OMRAT
    from geometries.junctions import Junction


# Resolver signature: leg_id -> display label (e.g. "LEG_1_3").
LegNameResolver = Callable[[str], str]


def _identity_name(leg_id: str) -> str:
    return leg_id


def build_leg_name_resolver(
    segment_data: dict[str, Any] | None,
) -> LegNameResolver:
    """Return a function mapping a leg id to its ``LEG_{route}_{leg}`` name.

    Falls back to the bare leg id when the segment is missing or the
    expected fields aren't populated -- keeps the dialog robust against
    legacy projects and tests that don't supply ``segment_data``.
    """
    if not segment_data:
        return _identity_name

    def _resolve(leg_id: str) -> str:
        seg = segment_data.get(leg_id) or segment_data.get(str(leg_id))
        if not isinstance(seg, dict):
            return str(leg_id)
        name = seg.get('Leg_name')
        if name:
            return str(name)
        route = seg.get('Route_Id')
        seg_id = seg.get('Segment_Id', leg_id)
        if route is not None:
            return f"LEG_{route}_{seg_id}"
        return str(leg_id)

    return _resolve


def _format_junction_label(
    junction: "Junction",
    name_for: LegNameResolver = _identity_name,
) -> str:
    """Pretty label for the junction picker combo.

    Shows only the leg names involved in the junction (e.g.
    ``"LEG_1_1, LEG_2_1"``).  The junction coordinate and matrix source
    are surfaced separately by the dialog (``lbl_source``) so they
    don't clutter the picker.
    """
    return ", ".join(name_for(lid) for lid in sorted(junction.legs.keys()))


def matrix_from_table(
    table: QTableWidget,
    leg_ids: list[str],
) -> dict[str, dict[str, float]]:
    """Read percentage spinboxes back into a row-major matrix.

    Pure helper so tests can drive the read path without spinning up
    the full dialog.  Rows are normalised in
    :meth:`omrat_utils.handle_junctions.Junctions.set_row`.
    """
    out: dict[str, dict[str, float]] = {}
    for r, in_leg in enumerate(leg_ids):
        row: dict[str, float] = {}
        for c, out_leg in enumerate(leg_ids):
            if r == c:
                continue
            widget = table.cellWidget(r, c)
            if isinstance(widget, QDoubleSpinBox):
                row[out_leg] = float(widget.value())
        out[in_leg] = row
    return out


def populate_table(
    table: QTableWidget,
    leg_ids: list[str],
    transitions: dict[str, dict[str, float]],
    name_for: LegNameResolver = _identity_name,
) -> None:
    """Render the junction's leg ids as headers and pre-fill spinboxes."""
    n = len(leg_ids)
    table.clear()
    table.setRowCount(n)
    table.setColumnCount(n)
    table.setHorizontalHeaderLabels([f"-> {name_for(lid)}" for lid in leg_ids])
    table.setVerticalHeaderLabels([f"from {name_for(lid)}" for lid in leg_ids])
    for r, in_leg in enumerate(leg_ids):
        row = transitions.get(in_leg) or {}
        for c, out_leg in enumerate(leg_ids):
            if r == c:
                cell = QTableWidgetItem("—")
                cell.setFlags(Qt.ItemIsEnabled)
                cell.setTextAlignment(int(Qt.AlignCenter))
                table.setItem(r, c, cell)
                continue
            spin = NoWheelDoubleSpinBox()
            spin.setRange(0.0, 100.0)
            spin.setDecimals(2)
            spin.setSuffix(" %")
            value = float(row.get(out_leg, 0.0)) * 100.0
            spin.setValue(value)
            table.setCellWidget(r, c, spin)


class JunctionMatrixDialog(QDialog):
    """Dialog to view/edit one junction's transition matrix at a time."""

    def __init__(self, omrat: "OMRAT", parent=None):
        super().__init__(parent)
        self.omrat = omrat
        self.setWindowTitle("Junction transition matrix")
        self.resize(640, 480)

        # Resolver for pretty leg names (``LEG_{route}_{leg}``) used in
        # the picker combo and table headers.  Built once at dialog
        # open so the lookup is cheap per render; segment_data won't
        # change while the modal dialog is up.
        segment_data = getattr(omrat, 'segment_data', None) or {}
        self._name_for: LegNameResolver = build_leg_name_resolver(segment_data)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Pick a junction; rows are inbound legs, columns outbound. "
            "Each row should sum to 100%; values are normalised on save."
        ))
        self.cmb = QComboBox()
        layout.addWidget(self.cmb)

        self.lbl_source = QLabel("")
        layout.addWidget(self.lbl_source)

        self.table = QTableWidget(0, 0)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        self.bb = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Close
        )
        layout.addWidget(self.bb)
        self.bb.button(QDialogButtonBox.Save).setText("Save row")
        self.bb.button(QDialogButtonBox.Close).clicked.connect(self.accept)
        self.bb.button(QDialogButtonBox.Save).clicked.connect(self._save_current)

        self._junction_ids: list[str] = []
        self._populate_combo()
        self.cmb.currentIndexChanged.connect(self._on_junction_changed)
        if self._junction_ids:
            self._render_for_index(0)

    # ------------------------------------------------------------------

    def _populate_combo(self) -> None:
        self.cmb.clear()
        handler = getattr(self.omrat, 'junctions', None)
        if handler is None or not handler.registry:
            self.cmb.addItem("(no junctions found)")
            self.cmb.setEnabled(False)
            return
        self.cmb.setEnabled(True)
        # Sort by id so the picker stays stable across project loads.
        self._junction_ids = sorted(handler.registry.keys())
        for jid in self._junction_ids:
            self.cmb.addItem(
                _format_junction_label(handler.registry[jid], self._name_for),
            )

    def _on_junction_changed(self, index: int) -> None:
        if 0 <= index < len(self._junction_ids):
            self._render_for_index(index)

    def _render_for_index(self, index: int) -> None:
        handler = self.omrat.junctions
        jid = self._junction_ids[index]
        j = handler.registry[jid]
        self.lbl_source.setText(f"Source: {j.source}")
        leg_ids = sorted(j.legs.keys())
        populate_table(self.table, leg_ids, j.transitions, self._name_for)

    def _save_current(self) -> None:
        handler = getattr(self.omrat, 'junctions', None)
        if handler is None or not self._junction_ids:
            return
        index = self.cmb.currentIndex()
        if not (0 <= index < len(self._junction_ids)):
            return
        jid = self._junction_ids[index]
        j = handler.registry[jid]
        leg_ids = sorted(j.legs.keys())
        matrix = matrix_from_table(self.table, leg_ids)
        handler.set_matrix(jid, matrix)
        # Re-render so the user sees the normalised values.
        self._render_for_index(index)
        QMessageBox.information(
            self, "Junction matrix",
            f"Saved transitions for junction {jid}.",
        )


def open_junction_dialog(omrat: "OMRAT") -> None:
    """Convenience entry point used by the menu action."""
    parent = getattr(omrat, 'main_widget', None)
    dlg = JunctionMatrixDialog(omrat, parent)
    dlg.exec()
