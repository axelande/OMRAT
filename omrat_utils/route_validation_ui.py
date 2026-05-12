"""Modal-prompt driver for the route-validation pass.

Walks the :class:`ValidationReport` returned by
``geometries.route_validation.validate_routes`` and asks the user, one
candidate at a time, what to do with each close-waypoint pair and each
leg-leg X-crossing.  The dialog zooms the QGIS canvas to the candidate
and offers four choices for waypoints (point 1 / point 2 / midpoint /
skip) and two for crossings (split into 4 sub-legs / skip).

Kept thin: the actual mutation of ``segment_data`` / ``traffic_data``
lives in :mod:`geometries.route_validation`; this module is just the
glue between that and the QGIS UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
)
from qgis.PyQt.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from geometries.route_validation import (
    CloseWaypointPair,
    LegIntersection,
    apply_intersection_split,
    apply_waypoint_merge,
    validate_routes,
)

if TYPE_CHECKING:
    from omrat import OMRAT


# Buffer (degrees) added around the zoom rectangle so the user can see
# the surrounding context, not just the two near-identical points.
_ZOOM_PADDING_DEG = 0.005


@dataclass
class _MergeChoice:
    """User's selection for one close-waypoint pair."""
    target: tuple[float, float] | None  # None = skip


def _zoom_canvas_to_points(
    omrat: "OMRAT", points: list[tuple[float, float]],
) -> None:
    """Pan/zoom the canvas so every ``points`` entry is visible."""
    if not points:
        return
    canvas = omrat.iface.mapCanvas() if omrat.iface else None
    if canvas is None:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    rect = QgsRectangle(
        min(xs) - _ZOOM_PADDING_DEG,
        min(ys) - _ZOOM_PADDING_DEG,
        max(xs) + _ZOOM_PADDING_DEG,
        max(ys) + _ZOOM_PADDING_DEG,
    )
    # Convert from EPSG:4326 to whatever the canvas uses.
    canvas_crs = canvas.mapSettings().destinationCrs()
    src_crs = QgsCoordinateReferenceSystem("EPSG:4326")
    if canvas_crs.authid() != src_crs.authid():
        try:
            tr = QgsCoordinateTransform(src_crs, canvas_crs, QgsProject.instance())
            ll = tr.transform(QgsPointXY(rect.xMinimum(), rect.yMinimum()))
            ur = tr.transform(QgsPointXY(rect.xMaximum(), rect.yMaximum()))
            rect = QgsRectangle(ll.x(), ll.y(), ur.x(), ur.y())
        except Exception:  # nosec B110
            # Reprojection can fail when the canvas CRS lacks a transform
            # path back to 4326 (custom local CRSs etc.).  In that case we
            # fall through and use the un-projected rectangle, which is
            # still better than crashing the validation pass.
            pass
    canvas.setExtent(rect)
    canvas.refresh()


class _MergePromptDialog(QDialog):
    """Modal dialog asking which of two near-coincident locations to keep."""

    def __init__(self, pair: CloseWaypointPair, parent=None):
        super().__init__(parent)
        self.pair = pair
        self._choice: _MergeChoice = _MergeChoice(target=None)
        self.setWindowTitle("Merge close waypoints?")

        layout = QVBoxLayout(self)
        msg = QLabel(
            f"Two waypoints are {pair.distance_m:.1f} m apart "
            f"(threshold {pair.threshold_m:.1f} m).\n\n"
            f"Point 1: {pair.point_a[0]:.6f}, {pair.point_a[1]:.6f}\n"
            f"Point 2: {pair.point_b[0]:.6f}, {pair.point_b[1]:.6f}\n\n"
            "Which location should both legs snap to?"
        )
        msg.setWordWrap(True)
        layout.addWidget(msg)

        btn_row = QHBoxLayout()
        b_p1 = QPushButton("Use point 1")
        b_p2 = QPushButton("Use point 2")
        b_mid = QPushButton("Use midpoint")
        b_skip = QPushButton("Skip")
        for b in (b_p1, b_p2, b_mid, b_skip):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        b_p1.clicked.connect(lambda: self._set_and_accept(pair.point_a))
        b_p2.clicked.connect(lambda: self._set_and_accept(pair.point_b))
        b_mid.clicked.connect(lambda: self._set_and_accept(pair.midpoint))
        b_skip.clicked.connect(self.reject)

    def _set_and_accept(self, target: tuple[float, float]) -> None:
        self._choice = _MergeChoice(target=target)
        self.accept()

    def choice(self) -> _MergeChoice:
        return self._choice


class _SplitPromptDialog(QDialog):
    """Modal dialog asking whether to split a crossing into 4 sub-legs."""

    def __init__(self, intersection: LegIntersection, parent=None):
        super().__init__(parent)
        self.intersection = intersection
        self._accept_split: bool = False
        self.setWindowTitle("Split crossing legs?")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"Legs {intersection.leg1_id} and {intersection.leg2_id} cross "
            f"at {intersection.point[0]:.6f}, {intersection.point[1]:.6f}.\n\n"
            "Split each leg in two at the crossing point so the model treats "
            "the crossing as a real four-leg junction?"
        ))
        bb = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        bb.button(QDialogButtonBox.Ok).setText("Split")
        bb.button(QDialogButtonBox.Cancel).setText("Skip")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _accept(self) -> None:
        self._accept_split = True
        self.accept()

    def accepted_split(self) -> bool:
        return self._accept_split


@dataclass
class ValidationOutcome:
    """Summary returned by :func:`run_validation_pass`."""
    merges_applied: int = 0
    splits_applied: int = 0
    skipped: int = 0


def _hide_dock_for_prompts(omrat: "OMRAT"):
    """Temporarily hide the OMRAT dock so the prompts don't cover the canvas.

    Returns the original visibility state so :func:`_restore_dock` can
    bring the dock back at the end of the validation pass.
    """
    dock = getattr(omrat, 'main_widget', None)
    if dock is None or not hasattr(dock, 'isVisible'):
        return None
    was_visible = dock.isVisible()
    if was_visible:
        try:
            dock.hide()
        except Exception:
            pass
    return was_visible


def _restore_dock(omrat: "OMRAT", was_visible) -> None:
    if was_visible:
        dock = getattr(omrat, 'main_widget', None)
        if dock is not None:
            try:
                dock.show()
                dock.raise_()
            except Exception:
                pass


def run_validation_pass(
    omrat: "OMRAT",
    *,
    tol_frac: float = 0.05,
    show_dialog: Callable | None = None,
) -> ValidationOutcome:
    """Run validate_routes and walk the report interactively.

    Returns counts of operations applied so the caller can show a final
    "X waypoints merged, Y crossings split" toast.  ``show_dialog`` is
    a hook for tests so they can drive the UI without spinning up Qt
    event loops.

    The OMRAT dock is hidden while the prompts are showing (it
    otherwise covers the very canvas the user needs to see when
    deciding which waypoint to keep) and restored at the end.

    After all merges / splits have been applied, the leg vector
    layers, route table, and offset lines are torn down and rebuilt
    from ``segment_data`` -- without that, the QGIS canvas would still
    draw the old geometry and ``GatherData.get_segment_tbl`` would
    later overwrite the merged endpoints from the stale table cells.
    """
    out = ValidationOutcome()
    sd = getattr(omrat, 'segment_data', {}) or {}
    td = getattr(omrat, 'traffic_data', {}) or {}
    report = validate_routes(sd, tol_frac=tol_frac)

    parent = None  # not omrat.main_widget -- the parent gets hidden below

    # Hide the dock only when there's actually something to prompt
    # about (so a no-op validation pass does not flicker the UI).
    needs_prompt = bool(report.close_pairs or report.intersections)
    dock_was_visible = (
        _hide_dock_for_prompts(omrat) if (needs_prompt and show_dialog is None)
        else None
    )

    def _show_merge(pair: CloseWaypointPair) -> _MergeChoice:
        if show_dialog is not None:
            return show_dialog('merge', pair)
        _zoom_canvas_to_points(omrat, [pair.point_a, pair.point_b])
        dlg = _MergePromptDialog(pair, parent)
        dlg.exec()
        return dlg.choice()

    def _show_split(intersection: LegIntersection) -> bool:
        if show_dialog is not None:
            return show_dialog('split', intersection)
        _zoom_canvas_to_points(omrat, [intersection.point])
        dlg = _SplitPromptDialog(intersection, parent)
        dlg.exec()
        return dlg.accepted_split()

    try:
        # Process merges first — they may collapse a candidate intersection
        # into a junction, which the second pass will skip naturally.
        for pair in report.close_pairs:
            choice = _show_merge(pair)
            if choice.target is None:
                out.skipped += 1
                continue
            moved = apply_waypoint_merge(sd, pair, choice.target)
            if moved > 0:
                out.merges_applied += 1

        # Re-run intersection detection on the (possibly mutated) data so
        # any merge-resolved crossings disappear from the list.
        fresh_intersections = validate_routes(sd, tol_frac=tol_frac).intersections
        for hit in fresh_intersections:
            if not _show_split(hit):
                out.skipped += 1
                continue
            apply_intersection_split(sd, hit, traffic_data=td)
            out.splits_applied += 1
    finally:
        _restore_dock(omrat, dock_was_visible)

    # Refresh canvas layers, route table, and tangent lines so the
    # mutated segment_data is reflected on screen *and* survives the
    # next save (GatherData reads endpoints from twRouteList).
    if out.merges_applied or out.splits_applied:
        qgis_geoms = getattr(omrat, 'qgis_geoms', None)
        if qgis_geoms is not None and hasattr(qgis_geoms, 'reload_legs_from_segment_data'):
            try:
                qgis_geoms.reload_legs_from_segment_data()
            except Exception:  # nosec B110
                # Reloading is best-effort -- a failure here means the
                # dict is updated but the UI is stale; better that than
                # losing the merge entirely.
                pass

    # Refresh the junction registry after the structural edits so the
    # transition matrices have correct leg references.
    handler = getattr(omrat, 'junctions', None)
    if handler is not None:
        handler.rebuild_from_segments(sd, prefer_user=True)

    return out
