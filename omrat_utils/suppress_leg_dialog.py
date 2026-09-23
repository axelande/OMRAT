"""Dialog for **Suppress leg...** on the Routes tab.

Pick a leg, then list where its traffic goes: per direction of the
suppressed leg, a target leg, the target's direction (pre-filled from the
bearings) and the share in percent.  The data model and the compute step
live in :mod:`compute.traffic_redirect`.

``apply_suppression`` is the headless core (flags + map + labels) so the
QGIS test-suite can exercise it; ``run`` only builds the dialog.  The
dialog is modeless so the map can be panned while picking the detour.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
)

from compute.traffic_redirect import (
    auto_target_dir, direction_label, dir_key, get_redirect, group_lead, group_members, is_suppressed,
    normalise_redirect, restore_group, set_group, set_suppressed, total_frequency,
)
from omrat_utils.copy_traffic import describe_targets
from omrat_utils.traffic_links import leg_label

if TYPE_CHECKING:
    from omrat import OMRAT

COL_FROM, COL_LEG, COL_DIR, COL_SHARE = range(4)


def _leg_label(seg_id: str, segment_data: dict[str, Any]) -> str:
    return leg_label(str(seg_id), segment_data)


def direction_totals(omrat: "OMRAT", seg: str) -> list[float]:
    """Ships / year of direction 0 and 1 of ``seg`` (unscaled)."""
    block = (getattr(omrat, 'traffic_data', None) or {}).get(str(seg))
    seg_d = (getattr(omrat, 'segment_data', None) or {}).get(str(seg)) or {}
    out = []
    for d in (0, 1):
        key = dir_key(block, seg_d, d) if isinstance(block, dict) else None
        out.append(total_frequency(block.get(key)) if key else 0.0)
    return out


def apply_suppression(
    omrat: "OMRAT", seg: str, suppressed: bool, redirect: list[dict[str, Any]] | None = None,
    members: list[str] | None = None,
) -> bool:
    """Store the flag / redirect on ``seg`` and refresh the map and labels.

    ``redirect=None`` keeps the stored redirect (used by *Restore*, so a
    later suppress brings the same detour back).  ``members`` are the legs
    suppressed *with* ``seg`` (same ships, nothing moved again);
    ``None`` keeps the current members.  Restoring ``seg`` restores its
    members too.
    """
    seg = str(seg)
    segs = omrat.segment_data
    if seg not in segs:
        return False
    if suppressed:
        set_suppressed(segs, seg, True, redirect)
        # A member suppressed here on its own becomes a lead: set_group
        # clears its own ``suppressed_with``.
        changed = [seg] + set_group(segs, seg, group_members(segs, seg) if members is None else members)
    else:
        changed = restore_group(segs, seg)
    geoms = getattr(omrat, 'qgis_geoms', None)
    if geoms is not None:
        for leg in changed:
            try:
                geoms.sync_suppressed_style(leg)
            except Exception:  # nosec B110 B112
                pass
        if hasattr(geoms, 'refresh_traffic_link_views'):
            geoms.refresh_traffic_link_views()
    return True


class _SuppressDialog(QDialog):
    def __init__(self, omrat: "OMRAT") -> None:
        super().__init__(omrat.main_widget)
        self.omrat = omrat
        self.setWindowTitle(omrat.tr("Suppress leg and move its traffic"))
        self.setModal(False)
        self.resize(720, 460)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(omrat.tr("Leg to suppress:")))
        self.cb_src = QComboBox()
        layout.addWidget(self.cb_src)
        self.lbl_totals = QLabel()
        layout.addWidget(self.lbl_totals)

        help_txt = QLabel(omrat.tr(
            "Where does the traffic go?  Share = % of that direction's ships sailing the target leg.\n"
            "Legs of one detour in series each get the full share (e.g. 100 % on every leg); "
            "alternative routes split it (e.g. 80 % / 20 %).\n"
            "Whole route: give only this leg (the one with the best AIS sample) the targets and tick the "
            "route's other legs below -- they carry the same ships, so nothing is moved twice.\n"
            "Suppressed legs are left out of the calculation, drawn dashed, and can be restored."
        ))
        help_txt.setWordWrap(True)
        layout.addWidget(help_txt)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([
            omrat.tr("From direction"), omrat.tr("To leg"), omrat.tr("To direction"), omrat.tr("Share (%)"),
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_LEG, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        row_btns = QHBoxLayout()
        self.pb_add = QPushButton(omrat.tr("Add target"))
        self.pb_remove = QPushButton(omrat.tr("Remove target"))
        row_btns.addWidget(self.pb_add)
        row_btns.addWidget(self.pb_remove)
        row_btns.addStretch(1)
        layout.addLayout(row_btns)

        layout.addWidget(QLabel(omrat.tr(
            "Suppress together with this leg (same ships -- left out, nothing moved again):")))
        self.lst_with = QListWidget()
        self.lst_with.setMaximumHeight(120)
        layout.addWidget(self.lst_with)
        self.lbl_note = QLabel()
        self.lbl_note.setWordWrap(True)
        layout.addWidget(self.lbl_note)

        self.buttons = QDialogButtonBox()
        self.pb_suppress = self.buttons.addButton(
            omrat.tr("Suppress && move traffic"), QDialogButtonBox.ButtonRole.AcceptRole)
        self.pb_restore = self.buttons.addButton(
            omrat.tr("Restore leg"), QDialogButtonBox.ButtonRole.ActionRole)
        self.buttons.addButton(QDialogButtonBox.StandardButton.Close)
        layout.addWidget(self.buttons)

        self.cb_src.currentIndexChanged.connect(self._load_source)
        self.pb_add.clicked.connect(lambda: self.add_row())
        self.pb_remove.clicked.connect(self._remove_rows)
        self.pb_suppress.clicked.connect(self._on_suppress)
        self.pb_restore.clicked.connect(self._on_restore)
        self.buttons.rejected.connect(self.close)
        self.fill_sources()

    # -- data helpers ---------------------------------------------------

    @property
    def segs(self) -> dict[str, Any]:
        return getattr(self.omrat, 'segment_data', None) or {}

    def source(self) -> str:
        return str(self.cb_src.currentData())

    def fill_sources(self, select: str | None = None) -> None:
        self.cb_src.blockSignals(True)
        self.cb_src.clear()
        for seg_id, seg_d in self.segs.items():
            if isinstance(seg_d, dict):
                self.cb_src.addItem(_leg_label(str(seg_id), self.segs), str(seg_id))
        if select is not None:
            idx = self.cb_src.findData(str(select))
            if idx >= 0:
                self.cb_src.setCurrentIndex(idx)
        self.cb_src.blockSignals(False)
        self._load_source()

    def _target_ids(self) -> list[str]:
        src = self.source()
        return [str(k) for k, v in self.segs.items()
                if isinstance(v, dict) and str(k) != src and not is_suppressed(self.segs, str(k))]

    def _load_source(self, *_args) -> None:
        src = self.source()
        tot = direction_totals(self.omrat, src)
        self.lbl_totals.setText("   ".join(
            f"{direction_label(src, d, self.segs)}: {tot[d]:,.0f} ships/year" for d in (0, 1)
        ))
        self.table.setRowCount(0)
        lead = group_lead(self.segs, src)
        if lead is None:
            for e in get_redirect(self.segs, src):
                self.add_row(e)
        self._fill_with_list()
        members = group_members(self.segs, src)
        if lead is not None:
            self.lbl_note.setText(self.omrat.tr(
                "This leg is suppressed together with {lead}.  Restore it to take it out of that group, "
                "or give it its own targets and suppress it here.").format(lead=leg_label(lead, self.segs)))
        else:
            self.lbl_note.setText('')
        self.pb_restore.setEnabled(is_suppressed(self.segs, src))
        self.pb_restore.setText(
            self.omrat.tr("Restore leg (+{n} with it)").format(n=len(members)) if members
            else self.omrat.tr("Restore leg"))

    def _fill_with_list(self) -> None:
        """Checkable list of the other legs; ticked = suppressed with the
        current leg.  Legs suppressed on their own or with another lead are
        shown but cannot be ticked."""
        src = self.source()
        self.lst_with.clear()
        for seg_id, seg_d in self.segs.items():
            seg_id = str(seg_id)
            if seg_id == src or not isinstance(seg_d, dict):
                continue
            item = QListWidgetItem(leg_label(seg_id, self.segs))
            item.setData(Qt.ItemDataRole.UserRole, seg_id)
            mine = group_lead(self.segs, seg_id) == src
            free = not is_suppressed(self.segs, seg_id)
            flags = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable
            if mine or free:
                flags |= Qt.ItemFlag.ItemIsEnabled
            item.setFlags(flags)
            item.setCheckState(Qt.CheckState.Checked if mine else Qt.CheckState.Unchecked)
            self.lst_with.addItem(item)

    def members(self) -> list[str]:
        out = []
        for i in range(self.lst_with.count()):
            item = self.lst_with.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return out

    def set_member(self, seg_id: str, checked: bool = True) -> None:
        """Tick / untick ``seg_id`` in the "together with" list."""
        for i in range(self.lst_with.count()):
            item = self.lst_with.item(i)
            if str(item.data(Qt.ItemDataRole.UserRole)) == str(seg_id):
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    # -- table rows -----------------------------------------------------

    def add_row(self, entry: dict[str, Any] | None = None) -> int:
        src = self.source()
        row = self.table.rowCount()
        self.table.insertRow(row)

        cb_from = QComboBox()
        for d in (0, 1):
            cb_from.addItem(direction_label(src, d, self.segs), d)
        cb_leg = QComboBox()
        for seg_id in self._target_ids():
            cb_leg.addItem(_leg_label(seg_id, self.segs), seg_id)
        cb_dir = QComboBox()
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 100.0)
        spin.setDecimals(1)
        spin.setSuffix(" %")
        spin.setValue(100.0)

        self.table.setCellWidget(row, COL_FROM, cb_from)
        self.table.setCellWidget(row, COL_LEG, cb_leg)
        self.table.setCellWidget(row, COL_DIR, cb_dir)
        self.table.setCellWidget(row, COL_SHARE, spin)

        def _refresh_dirs() -> None:
            dst = cb_leg.currentData()
            cb_dir.clear()
            if dst is None:
                return
            for d in (0, 1):
                cb_dir.addItem(direction_label(str(dst), d, self.segs), d)
            cb_dir.setCurrentIndex(auto_target_dir(self.segs, src, int(cb_from.currentData()), str(dst)))

        if entry is not None:
            cb_from.setCurrentIndex(int(entry['from_dir']))
            idx = cb_leg.findData(str(entry['leg']))
            if idx >= 0:
                cb_leg.setCurrentIndex(idx)
            spin.setValue(float(entry['share']))
        _refresh_dirs()
        if entry is not None and cb_leg.findData(str(entry['leg'])) >= 0:
            cb_dir.setCurrentIndex(int(entry['dir']))
        cb_from.currentIndexChanged.connect(_refresh_dirs)
        cb_leg.currentIndexChanged.connect(_refresh_dirs)
        return row

    def _remove_rows(self) -> None:
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()}, reverse=True)
        if not rows and self.table.rowCount():
            rows = [self.table.rowCount() - 1]
        for r in rows:
            self.table.removeRow(r)

    def entries(self) -> list[dict[str, Any]]:
        out = []
        for r in range(self.table.rowCount()):
            cb_from = self.table.cellWidget(r, COL_FROM)
            cb_leg = self.table.cellWidget(r, COL_LEG)
            cb_dir = self.table.cellWidget(r, COL_DIR)
            spin = self.table.cellWidget(r, COL_SHARE)
            if cb_leg is None or cb_leg.currentData() is None or spin.value() <= 0:
                continue
            out.append({
                'from_dir': cb_from.currentData(), 'leg': str(cb_leg.currentData()),
                'dir': cb_dir.currentData() if cb_dir.currentData() is not None else 0,
                'share': spin.value(),
            })
        return normalise_redirect(out)

    # -- actions --------------------------------------------------------

    def _on_suppress(self) -> None:
        src = self.source()
        if src not in self.segs:
            self.fill_sources()
            return
        entries = [e for e in self.entries() if e['leg'] in self.segs]
        members = [m for m in self.members() if m in self.segs]
        both = sorted(set(members) & {e['leg'] for e in entries})
        if both:
            QMessageBox.information(
                self, self.omrat.tr("Suppress leg"),
                self.omrat.tr("A leg cannot be both a target and suppressed with this leg: {legs}").format(
                    legs=describe_targets(both, self.segs)))
            return
        tot = direction_totals(self.omrat, src)
        lost = [direction_label(src, d, self.segs) for d in (0, 1)
                if tot[d] > 0 and not any(e['from_dir'] == d for e in entries)]
        if lost:
            answer = QMessageBox.question(
                self, self.omrat.tr("Traffic not moved"),
                self.omrat.tr(
                    "No target is given for: {dirs}.\n\nThose ships are removed from the "
                    "calculation.  Suppress the leg anyway?").format(dirs=", ".join(lost)),
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        apply_suppression(self.omrat, src, True, entries, members=members)
        with_txt = self.omrat.tr(" (+{n} leg(s) with it)").format(n=len(members)) if members else ""
        self._notify(self.omrat.tr("Leg {leg}{with_txt} suppressed; traffic moved to {targets}").format(
            leg=describe_targets([src], self.segs), with_txt=with_txt,
            targets=describe_targets(sorted({e['leg'] for e in entries}), self.segs) or '-',
        ))
        self.fill_sources(select=src)

    def _on_restore(self) -> None:
        src = self.source()
        apply_suppression(self.omrat, src, False)
        self._notify(self.omrat.tr("Leg {leg} restored").format(leg=describe_targets([src], self.segs)))
        self.fill_sources(select=src)

    def _notify(self, text: str) -> None:
        notifier = getattr(self.omrat, 'notifier', None)
        if notifier is not None:
            try:
                notifier.display_message(text, duration=10)
            except Exception:  # nosec B110 B112
                pass


def run(omrat: "OMRAT") -> None:
    """Open (or raise) the modeless Suppress leg dialog."""
    existing = getattr(omrat, '_suppress_leg_dlg', None)
    if existing is not None:
        try:
            existing.fill_sources(select=existing.source())
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return
        except RuntimeError:
            omrat._suppress_leg_dlg = None
    if len(getattr(omrat, 'segment_data', None) or {}) < 2:
        QMessageBox.information(
            omrat.main_widget, omrat.tr("Suppress leg"),
            omrat.tr("At least two legs are needed to move traffic from one to another."),
        )
        return
    dlg = _SuppressDialog(omrat)
    # Keep a Python reference, otherwise the modeless dialog is collected.
    omrat._suppress_leg_dlg = dlg
    dlg.show()
