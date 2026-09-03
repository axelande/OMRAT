"""Dialog for **Copy traffic...** on the Routes tab.

Lets the user pick a source leg and one or more target legs, then copies
the traffic matrices (and optionally the lateral distributions) with
:func:`omrat_utils.copy_traffic.copy_leg_traffic` and locks the targets
against AIS refreshes.

``apply_copy`` is the headless core -- it does the copying and the UI
refresh without any modal dialog -- so the QGIS test-suite can exercise
the full effect on the plugin.  ``run`` only builds the dialog and
forwards the choices.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
)

from omrat_utils.copy_traffic import (
    LOCK_KEY, copy_leg_traffic, describe_targets, is_locked,
)

if TYPE_CHECKING:
    from omrat import OMRAT


def _leg_label(seg_id: str, seg_d: dict[str, Any]) -> str:
    name = seg_d.get('Leg_name') or f'LEG_{seg_id}'
    suffix = '  [locked]' if seg_d.get(LOCK_KEY) is True else ''
    return f"{name}  (id {seg_id}){suffix}"


def _legs_with_traffic(omrat: "OMRAT") -> list[str]:
    td = getattr(omrat, 'traffic_data', None) or {}
    return [k for k, v in td.items() if isinstance(v, dict) and v]


def apply_copy(
    omrat: "OMRAT",
    src: str,
    targets: list[str],
    *,
    swap_dirs: bool = False,
    copy_distributions: bool = True,
    lock: bool = True,
) -> list[str]:
    """Copy ``src`` onto every leg in ``targets`` and refresh the UI.

    Returns the list of targets actually written.  Skips a target that
    equals the source.  The Traffic tab, the lateral-distribution panel
    and the route-table lock boxes are refreshed so the copied values
    show immediately and are not overwritten by stale widget contents.
    """
    src = str(src)
    done: list[str] = []

    # Flush whatever the user typed for the current leg into segment_data
    # first, otherwise the panel would write stale values over the copy
    # the next time the leg selection changes.
    dists = getattr(omrat, 'distributions', None)
    if dists is not None:
        try:
            dists.change_dist_segment(dists.last_id)
        except Exception:  # nosec B110 B112
            pass
    traffic = getattr(omrat, 'traffic', None)
    if traffic is not None:
        try:
            traffic.save()
        except Exception:  # nosec B110 B112
            pass

    for dst in targets:
        dst = str(dst)
        if dst == src:
            continue
        copy_leg_traffic(
            omrat.traffic_data, omrat.segment_data, src, dst,
            swap_dirs=swap_dirs, copy_distributions=copy_distributions, lock=lock,
        )
        # Keep the AIS plot cache in step so the distribution plot shows
        # the copied samples rather than the target's old AIS pull.
        ais = getattr(omrat, 'ais', None)
        if ais is not None and isinstance(getattr(ais, 'dist_data', None), dict):
            seg_d = omrat.segment_data.get(dst) or {}
            if 'dist1' in seg_d and 'dist2' in seg_d:
                ais.dist_data[dst] = {'line1': seg_d['dist1'], 'line2': seg_d['dist2']}
            else:
                ais.dist_data.pop(dst, None)
        done.append(dst)

    if not done:
        return done

    # Route table lock boxes.
    geoms = getattr(omrat, 'qgis_geoms', None)
    if geoms is not None and hasattr(geoms, 'sync_lock_column'):
        for dst in done:
            geoms.sync_lock_column(dst)

    # Traffic tab: same dict object, but the direction combo and the
    # matrix must be re-rendered if the shown leg was a target.
    if traffic is not None:
        try:
            traffic.traffic_data = omrat.traffic_data
            if traffic.c_seg in done:
                traffic.run_update = False
                traffic.update_direction_select()
                traffic.run_update = False
                traffic.update_traffic_tbl('dir')
                traffic.run_update = True
        except Exception:  # nosec B110 B112
            pass

    # Distribution panel: repopulate from segment_data for the shown leg.
    # Clearing leNormMean1_1 first is the documented way to skip the
    # "flush widgets -> segment_data" step inside change_dist_segment.
    if dists is not None and dists.last_id in done:
        try:
            omrat.main_widget.leNormMean1_1.setText('')
            dists.change_dist_segment(dists.last_id)
            dists.run_update_plot(dists.last_id)
        except Exception:  # nosec B110 B112
            pass
    return done


def run(omrat: "OMRAT") -> None:
    """Open the modal dialog and apply the user's choice."""
    segment_data = getattr(omrat, 'segment_data', None) or {}
    sources = _legs_with_traffic(omrat)
    if not sources:
        QMessageBox.information(
            omrat.main_widget, omrat.tr("Copy traffic"),
            omrat.tr("No leg has traffic data yet. Fetch AIS traffic or import a model first."),
        )
        return
    if len(segment_data) < 2:
        QMessageBox.information(
            omrat.main_widget, omrat.tr("Copy traffic"),
            omrat.tr("At least two legs are needed to copy traffic between them."),
        )
        return

    dlg = QDialog(omrat.main_widget)
    dlg.setWindowTitle(omrat.tr("Copy traffic between legs"))
    layout = QVBoxLayout(dlg)

    layout.addWidget(QLabel(omrat.tr("Copy traffic from leg:")))
    cb_src = QComboBox()
    for seg_id in sources:
        cb_src.addItem(_leg_label(seg_id, segment_data.get(seg_id) or {}), seg_id)
    layout.addWidget(cb_src)

    layout.addWidget(QLabel(omrat.tr("To leg(s)  (Ctrl-click to pick several):")))
    lst = QListWidget()
    lst.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    layout.addWidget(lst)

    def _fill_targets() -> None:
        lst.clear()
        src = cb_src.currentData()
        for seg_id, seg_d in segment_data.items():
            if str(seg_id) == str(src) or not isinstance(seg_d, dict):
                continue
            item = QListWidgetItem(_leg_label(str(seg_id), seg_d))
            item.setData(Qt.ItemDataRole.UserRole, str(seg_id))
            lst.addItem(item)

    cb_src.currentIndexChanged.connect(_fill_targets)
    _fill_targets()

    cb_dists = QCheckBox(omrat.tr("Also copy the lateral distributions (mean / std / weights / AI)"))
    cb_dists.setChecked(True)
    cb_swap = QCheckBox(omrat.tr("Swap directions (target leg is drawn the opposite way)"))
    cb_swap.setToolTip(omrat.tr(
        "Exchanges direction 1 and 2 and mirrors the lateral axis "
        "(means, samples and uniform bounds change sign)."
    ))
    cb_lock = QCheckBox(omrat.tr("Lock target legs so 'Update AIS' leaves them untouched"))
    cb_lock.setChecked(True)
    for w in (cb_dists, cb_swap, cb_lock):
        layout.addWidget(w)

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    if dlg.exec() != QDialog.DialogCode.Accepted:
        return

    src = str(cb_src.currentData())
    targets = [str(it.data(Qt.ItemDataRole.UserRole)) for it in lst.selectedItems()]
    if not targets:
        QMessageBox.information(omrat.main_widget, omrat.tr("Copy traffic"), omrat.tr("No target leg selected."))
        return

    already = [t for t in targets if is_locked(segment_data, t)]
    if already:
        answer = QMessageBox.question(
            omrat.main_widget, omrat.tr("Overwrite locked legs?"),
            omrat.tr(
                "These target legs are locked (they already hold copied or protected traffic):\n\n"
                f"{describe_targets(already, segment_data)}\n\nOverwrite them?"
            ),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

    done = apply_copy(
        omrat, src, targets,
        swap_dirs=cb_swap.isChecked(),
        copy_distributions=cb_dists.isChecked(),
        lock=cb_lock.isChecked(),
    )
    notifier = getattr(omrat, 'notifier', None)
    if notifier is not None and done:
        try:
            lock_txt = omrat.tr(" and locked") if cb_lock.isChecked() else ""
            notifier.display_message(
                omrat.tr("Copied traffic from {src} to {n} leg(s){lock}: {names}").format(
                    src=describe_targets([src], segment_data), n=len(done), lock=lock_txt,
                    names=describe_targets(done, segment_data),
                ),
                duration=10,
            )
        except Exception:  # nosec B110 B112
            pass
