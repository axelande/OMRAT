import math
from functools import partial
from typing import TYPE_CHECKING

from qgis._core import QgsFeatureRenderer, QgsVectorDataProvider
if TYPE_CHECKING:
    from omrat import OMRAT

from qgis.core import (
    QgsVectorLayer, QgsFeature, QgsGeometry, QgsLineString, QgsPoint, QgsProject,
    QgsField, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsFields,
    QgsPalLayerSettings, QgsVectorLayerSimpleLabeling, QgsSingleSymbolRenderer,
    QgsLineSymbol, QgsPointXY, QgsWkbTypes,
)
from qgis.gui import QgsRubberBand
from qgis.PyQt.QtCore import QMetaType, Qt, QTimer
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QTableWidgetItem, QPushButton


from omrat_utils import PointTool
from omrat_utils.copy_traffic import LOCK_KEY, SOURCE_KEY, is_locked, release_copy, set_locked
from omrat_utils.layer_styles import apply_stored_style
from omrat_utils.leg_numbering import leg_name, max_leg_number, next_free_segment_id
from omrat_utils.leg_sort import SORTABLE_COLUMNS, sort_segment_data
from geometries.waypoints import (
    WP_END, WP_START, ensure_waypoint, find_waypoint_at, find_waypoint_near,
    merge_waypoints, move_waypoint, points_equal, prune_unused_waypoints,
)
from geometries.tangent_position import (
    DEFAULT_TANGENT_POS, TANGENT_POS_KEY, fraction_from_percent, normalize_tangent_pos,
    percent_from_fraction, point_along, project_fraction,
)


def is_valid_point_pair(start: QgsPointXY, end: QgsPointXY) -> bool:
    return not (
        (-1 <= start.x() <= 1 and -1 <= start.y() <= 1) or
        (-1 <= end.x() <= 1 and -1 <= end.y() <= 1)
    )


def calculate_tangent_line(
    mid: QgsPointXY, start: QgsPointXY, end: QgsPointXY, offset: float
) -> tuple[QgsPointXY, QgsPointXY] | None:
    dx = end.x() - start.x()
    dy = end.y() - start.y()
    length = (dx**2 + dy**2)**0.5
    if length < 1e-9:
        return None
    unit_dx = dx / length
    unit_dy = dy / length
    perp_dx = -unit_dy
    perp_dy = unit_dx

    start_tangent = QgsPointXY(mid.x() - perp_dx * offset, mid.y() - perp_dy * offset)
    end_tangent = QgsPointXY(mid.x() + perp_dx * offset, mid.y() + perp_dy * offset)
    return start_tangent, end_tangent


def _layer_tree_view():
    """The QGIS layer-tree view, or ``None`` headless."""
    try:
        from qgis.utils import iface
        return iface.layerTreeView() if iface is not None else None
    except Exception:  # nosec B110 B112
        return None


def _refresh_link_views(handler, method: str) -> None:
    """Call the traffic-links refresh ``method`` on ``handler`` if it has one
    (tests drive some handler methods with a bare namespace as ``self``)."""
    fn = getattr(handler, method, None)
    if callable(fn):
        fn()


def unwire_leg_layer(handler, layer) -> None:
    """Disconnect the slot ``wire_leg_layer`` attached to ``layer`` and
    forget the layer.  Module-level so callers that only hold a duck-typed
    handler (tests, the AIS task) can use it too."""
    slots = getattr(handler, '_leg_geom_slots', None)
    slot = None
    if isinstance(slots, dict):
        try:
            slot = slots.pop(layer.id(), None)
        except Exception:  # nosec B110 B112
            slot = None
    if slot is not None:
        try:
            layer.geometryChanged.disconnect(slot)
        except (TypeError, RuntimeError):
            pass
    tracked = getattr(handler, 'buffer_edits', None)
    if isinstance(tracked, list) and layer in tracked:
        tracked.remove(layer)


class HandleQGISIface:
    def __init__(self, omrat: "OMRAT"):
        """Initialize the HandleQGISIface class."""
        self.omrat = omrat
        self.tangent_layer: QgsVectorLayer | None = None
        self.vector_layers: list[QgsVectorLayer] = []
        self.current_start_point: QgsPointXY | None = None
        # ``segment_id`` is the last *global* leg id handed out -- the
        # ``segment_data`` key.  ``route_leg_no`` is the last leg number on
        # the current route, the ``n`` in ``LEG_{route}_{n}``; it restarts
        # at 0 for every new route.  Only the number is user-editable.
        self.segment_id = 0
        self.cur_route_id = 1
        self.route_leg_no = 0
        self.item_changed_connected = False
        # Leg layers whose ``geometryChanged`` we listen to, and the exact
        # slot connected per layer id so teardown can disconnect *only*
        # ours.  (Kept under the historical name: it used to hold edit
        # buffers -- see ``wire_leg_layer`` for why that was a bug.)
        self.buffer_edits: list[QgsVectorLayer] = []
        self._leg_geom_slots: dict[str, object] = {}
        self.leg_dirs: dict[str, list[str]] = {}
        self._rubber_band: QgsRubberBand | None = None
        # "Traffic links" view (omrat_utils/traffic_links.py): the memory
        # layer, whether the user switched it on, and the highlight bands
        # of the leg last clicked in the route table.
        self.traffic_links_layer: QgsVectorLayer | None = None
        self._links_shown = False
        self._link_bands: list[QgsRubberBand] = []
        self._highlight_seg: str | None = None
        # Re-entrancy guards: our own tangent redraws / table writes must
        # not be mistaken for user edits.
        self._tangent_guard = False
        self._table_sync_guard = False
        # (column, descending) of the last header-click sort.
        self._route_sort: tuple[int, bool] | None = None
        self.omrat.main_widget.twRouteList.cellClicked.connect(self.on_route_table_cell_clicked)
        self._wire_route_table_header()
        # Spinboxes for current route / next-leg IDs (declared in the .ui).
        mw = self.omrat.main_widget
        if hasattr(mw, 'sbRouteId'):
            mw.sbRouteId.setValue(self.cur_route_id)
            mw.sbRouteId.valueChanged.connect(self._on_route_id_changed)
        if hasattr(mw, 'sbNextLegId'):
            mw.sbNextLegId.setValue(self.route_leg_no + 1)
            mw.sbNextLegId.valueChanged.connect(self._on_next_leg_id_changed)

    def add_new_route(self):
        """Starts the editing for a new route."""
        self.current_start_point = None
        self.point_layer = None
        self.mapTool = PointTool(self.omrat.iface.mapCanvas())
        canvas = self.omrat.iface.mapCanvas()
        if canvas is not None:
            canvas.setMapTool(self.mapTool)
        self.mapTool.canvasClicked.connect(self.onMapClick)
        self.mapTool.canvasMoved.connect(self._on_canvas_moved)
        self.omrat.main_widget.pbStopRoute.setEnabled(True)

    def onMapClick(self, point: QgsPoint):
        q_point: QgsPoint = self.point4326_from_wkt(point.asWkt())
        # Reuse an existing node when the click lands on one, so the new
        # leg is joined to it instead of ending a few metres away.
        snapped, _wid = self.snap_to_waypoint((q_point.x(), q_point.y()))
        q_point = QgsPoint(snapped[0], snapped[1])
        if self.current_start_point is None:
            self.create_point(q_point)
        else:
            self.create_line(q_point)

    # ------------------------------------------------------------------
    # Rubber-band preview helpers
    # ------------------------------------------------------------------

    def _init_rubber_band(self) -> None:
        """Create (or recreate) the dashed rubber-band line on the canvas."""
        self._clear_rubber_band()
        canvas = self.omrat.iface.mapCanvas() if self.omrat.iface else None
        if canvas is None:
            return
        rb = QgsRubberBand(canvas, QgsWkbTypes.GeometryType.LineGeometry)
        rb.setColor(QColor(80, 80, 80, 180))
        rb.setWidth(1)
        rb.setLineStyle(getattr(Qt, 'DashLine', None) or Qt.PenStyle.DashLine)
        self._rubber_band = rb

    def _on_canvas_moved(self, point: QgsPointXY) -> None:
        """Update the rubber-band end-point as the cursor moves."""
        if self._rubber_band is None or self.current_start_point is None:
            return
        canvas = self.omrat.iface.mapCanvas()
        start = self._to_canvas_crs(self.current_start_point, canvas)
        self._rubber_band.reset(QgsWkbTypes.GeometryType.LineGeometry)
        self._rubber_band.addPoint(start, False)
        self._rubber_band.addPoint(point, True)

    def _to_canvas_crs(self, pt_4326: QgsPointXY, canvas) -> QgsPointXY:
        """Transform a point from EPSG:4326 to the canvas CRS."""
        canvas_crs = canvas.mapSettings().destinationCrs()
        src_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        if canvas_crs.authid() == src_crs.authid():
            return pt_4326
        try:
            tr = QgsCoordinateTransform(src_crs, canvas_crs, QgsProject.instance())
            return tr.transform(pt_4326)
        except Exception:  # nosec B110 B112
            return pt_4326

    def _clear_rubber_band(self) -> None:
        """Remove the rubber-band line from the canvas."""
        if self._rubber_band is not None:
            try:
                self._rubber_band.reset(QgsWkbTypes.GeometryType.LineGeometry)
                canvas = self.omrat.iface.mapCanvas() if self.omrat.iface else None
                if canvas is not None:
                    canvas.scene().removeItem(self._rubber_band)
            except Exception:  # nosec B110 B112
                pass
            self._rubber_band = None

    # ------------------------------------------------------------------
    # Spinbox sync helpers
    # ------------------------------------------------------------------

    def sync_drawing_spinboxes(self) -> None:
        """Push current route/leg IDs into the UI spinboxes (blocks signals to avoid loops)."""
        mw = self.omrat.main_widget
        for name, value in (
            ('sbRouteId', self.cur_route_id),
            ('sbNextLegId', self.route_leg_no + 1),
        ):
            sb = getattr(mw, name, None)
            if sb is None:
                continue
            sb.blockSignals(True)
            try:
                sb.setValue(value)
            finally:
                sb.blockSignals(False)

    def _on_route_id_changed(self, value: int) -> None:
        self.begin_route(value)

    def _on_next_leg_id_changed(self, value: int) -> None:
        """The spinbox edits the per-route leg *number* only.  The global
        ``Segment_Id`` is minted by ``next_free_segment_id`` so a manual
        reset to 1 can never overwrite an existing leg."""
        self.route_leg_no = value - 1

    def begin_route(self, route_id: int | None = None) -> None:
        """Make ``route_id`` (default: the next unused route) the route
        that newly drawn legs belong to and restart the leg numbering so
        the first new leg is ``LEG_{route}_1`` -- or continues after the
        highest number the route already has."""
        if route_id is None:
            route_id = self.cur_route_id + 1
        self.cur_route_id = int(route_id)
        segment_data = getattr(self.omrat, 'segment_data', None) or {}
        self.route_leg_no = max_leg_number(segment_data, self.cur_route_id)
        self.sync_drawing_spinboxes()

    def current_leg_name(self) -> str:
        """``LEG_{route}_{n}`` for the leg being drawn."""
        return leg_name(self.cur_route_id, self.route_leg_no)

    def create_point(self, point: QgsPoint):
        self.point_layer = QgsVectorLayer("Point?crs=EPSG:4326", "StartPoint", "memory")
        prov: QgsVectorDataProvider | None = self.point_layer.dataProvider()
        QgsProject.instance().addMapLayer(self.point_layer)  # Add the layer to the project

        if not self.point_layer.isEditable():
            self.point_layer.startEditing()  # Start editing

        feat = QgsFeature()
        feat.setGeometry(point)
        if isinstance(prov, QgsVectorDataProvider):
            prov.addFeature(feat)
        self.current_start_point = QgsPointXY(point.x(), point.y())
        # Create the rubber-band preview line for the next leg.
        self._init_rubber_band()

    def create_fields(self) -> QgsFields:
        fields = QgsFields()
        fields.append(QgsField("segmentId", QMetaType.Type.Int))
        fields.append(QgsField("routeId", QMetaType.Type.Int))
        fields.append(QgsField("startPoint", QMetaType.Type.QString))
        fields.append(QgsField("endPoint", QMetaType.Type.QString))
        fields.append(QgsField("label", QMetaType.Type.QString))  # Field for labeling
        return fields

    def create_line(self, point: QgsPoint):
        """Create a line layer, style it, label it, and create offset lines."""
        # Mint the global id above every id in use (never reuse a key that
        # is still in ``segment_data``) and advance the per-route number
        # that the layer name carries from the moment it appears in the
        # QGIS layer panel.
        self.segment_id = next_free_segment_id(
            getattr(self.omrat, 'segment_data', None), self.segment_id,
        )
        self.route_leg_no += 1
        layer_name = self.current_leg_name()
        vl = QgsVectorLayer("LineString?crs=EPSG:4326", layer_name, "memory")
        if not vl.isValid():
            print("Error: Line layer is not valid")
            return

        # Add the layer to the project
        QgsProject.instance().addMapLayer(vl)

        # Start editing the layer
        if not vl.isEditable():
            vl.startEditing()

        # Add fields to the layer
        pr: QgsVectorDataProvider | None = vl.dataProvider()
        fields = self.create_fields()
        if pr is not None:
            pr.addAttributes(fields.toList())
        vl.updateFields()

        # Create the feature
        fet = QgsFeature(fields)
        if isinstance(self.current_start_point, QgsPointXY):
            start_point = self.current_start_point
            end_point = point
            fet.setGeometry(QgsLineString([QgsPoint(start_point.x(), start_point.y()), end_point]))
            fet.setAttributes([
                self.segment_id,
                self.cur_route_id,
                self.current_start_point.asWkt(),
                point.asWkt(),
                layer_name,  # Label value
            ])
            # Style the layer (project style if one is stored, else default)
            self.style_layer(vl)
            apply_stored_style(self.omrat, 'legs', vl)
            fet.setId(self.segment_id)

            # Add the feature to the layer
            if pr is not None:
                pr.addFeature(fet)

            # Label the layer
            self.label_layer(vl)

            # Refresh the layer to ensure changes are applied
            vl.triggerRepaint()

            # Create offset lines
            self.create_offset_lines(start_point, QgsPointXY(end_point.x(), end_point.y()), 2500, self.segment_id)

            # Stop editing and remove the point layer
            if not self.omrat.testing and self.point_layer is not None:
                if self.point_layer.isEditable():
                    self.point_layer.commitChanges()  # Save changes
                QgsProject.instance().removeMapLayer(self.point_layer)

            # Update segment data and save the route
            self.update_segment_data(point)

            # Auto-split both legs if the new one crosses an existing leg.
            if self._auto_split_on_intersection(str(self.segment_id), point, vl):
                return

            self.vector_layers.append(vl)

            # Leave the layer editable so the user can drag vertices right
            # away; the signal hookup itself does not depend on it.
            if not vl.isEditable():
                vl.startEditing()
            self.wire_leg_layer(vl, self.segment_id)
            self.current_start_point = QgsPointXY(point.x(), point.y())
            self.point_layer = None
            self.save_route(QgsPoint(start_point.x(), start_point.y()), end_point)

            vl.setCustomProperty("segment_id", self.segment_id)
            self.sync_drawing_spinboxes()

    def wire_leg_layer(self, layer: QgsVectorLayer, fid: int) -> None:
        """Route vertex edits on ``layer`` (leg ``fid``) to
        ``on_geometry_changed``.

        Connects to the *layer's* ``geometryChanged``, which is emitted
        for every edit session.  Until v0.15.2 the handler hung off
        ``layer.editBuffer()``; QGIS destroys that buffer on every
        commit (**Stop route**, toggling edit mode, Save Layer Edits), so
        a vertex dragged afterwards moved on the canvas while
        ``segment_data``, the route table and the tangent stayed put --
        and the next save wrote the stale coordinates.
        """
        slot = partial(self.on_geometry_changed_wrapper, fid)
        layer.geometryChanged.connect(slot)
        self._leg_geom_slots[layer.id()] = slot
        self.buffer_edits.append(layer)

    def unwire_all_leg_layers(self) -> None:
        for layer in list(self.buffer_edits):
            unwire_leg_layer(self, layer)
        self.buffer_edits = []
        self._leg_geom_slots = {}

    def unload(self):
        """Remove temporary layers and disconnect signals."""
        # Remove the point layer
        print('unloading_qgis')
        try:
            if hasattr(self, 'point_layer') and self.point_layer is not None:
                QgsProject.instance().removeMapLayer(self.point_layer)
        except RuntimeError:
            pass
        self.point_layer = None
        # Remove vector layers and disconnect geometryChanged signals
        self.unwire_all_leg_layers()
        for layer in self.vector_layers:
            QgsProject.instance().removeMapLayer(layer.id())  # Remove the layer from QGIS

        # Disconnect itemChanged signal from twRouteList
        try:
            self.omrat.main_widget.twRouteList.itemChanged.disconnect()
        except TypeError:
            pass

        # Disconnect custom signals
        if hasattr(self, "mapTool"):
            if hasattr(self.mapTool, 'canvasClicked'):
                try:
                    self.mapTool.canvasClicked.disconnect()
                except TypeError:
                    pass
        try:
            self.omrat.main_widget.twRouteList.disconnect()
        except TypeError:
            pass
        self._disconnect_tangent_signal()
        # Break circular references
        self._clear_rubber_band()
        self.tangent_layer = None
        self.mapTool = None
        self.vector_layers = []
        self.current_start_point = None
        self.leg_dirs = {}

    def clear(self) -> None:
        """Remove all route/segment layers and reset state.

        Unlike unload(), this keeps the plugin operational by preserving
        permanent signal connections (cellClicked, canvasClicked, etc.).
        """
        # Remove the point layer
        try:
            if hasattr(self, 'point_layer') and self.point_layer is not None:
                QgsProject.instance().removeMapLayer(self.point_layer)
        except RuntimeError:
            pass
        self.point_layer = None

        # Disconnect our geometry-changed slots
        self.unwire_all_leg_layers()

        # Remove vector layers from QGIS project
        for layer in self.vector_layers:
            try:
                QgsProject.instance().removeMapLayer(layer.id())
            except Exception:  # nosec B110 B112
                pass
        self.vector_layers = []

        # Remove tangent layer
        if self.tangent_layer is not None:
            try:
                QgsProject.instance().removeMapLayer(self.tangent_layer.id())
            except Exception:  # nosec B110 B112
                pass
            self.tangent_layer = None

        # Traffic links layer + highlight (rebuilt after the next load when
        # the view is still switched on).
        self._remove_traffic_links_layer()
        self._clear_link_highlight()

        # Disconnect itemChanged if connected (will be reconnected on next load)
        if self.item_changed_connected:
            try:
                self.omrat.main_widget.twRouteList.itemChanged.disconnect(self.on_width_changed)
            except TypeError:
                pass
            self.item_changed_connected = False

        # Reset state
        self.current_start_point = None
        self.segment_id = 0
        self.cur_route_id = 1
        self.route_leg_no = 0
        self.leg_dirs = {}
        self._clear_rubber_band()
        self.sync_drawing_spinboxes()

    def _find_layer_for_seg_id(self, seg_id: int) -> "QgsVectorLayer | None":
        """Return the vector layer whose first feature has segmentId == seg_id."""
        for layer in self.vector_layers:
            try:
                for feat in layer.getFeatures():
                    if feat["segmentId"] == seg_id:
                        return layer
            except Exception:  # nosec B110 B112
                pass
        return None

    def remove_leg(self) -> None:
        """Remove selected leg(s) from the route table, QGIS layers, and all data dicts."""
        table = self.omrat.main_widget.twRouteList
        selected_rows: set[int] = {item.row() for item in table.selectedItems()}
        if not selected_rows:
            return

        for row in sorted(selected_rows, reverse=True):
            seg_id_item = table.item(row, 0)
            if seg_id_item is None:
                continue
            try:
                seg_id = int(seg_id_item.text())
            except ValueError:
                continue
            seg_key = str(seg_id)

            layer_to_remove = self._find_layer_for_seg_id(seg_id)

            if layer_to_remove is not None:
                unwire_leg_layer(self, layer_to_remove)
                self.vector_layers.remove(layer_to_remove)
                try:
                    QgsProject.instance().removeMapLayer(layer_to_remove.id())
                except Exception:  # nosec B110 B112
                    pass

            # Remove tangent lines for this segment.
            if self.tangent_layer is not None:
                self.remove_existing_tangent(seg_id)

            # Remove from all data dicts.
            self.omrat.segment_data.pop(seg_key, None)
            self.omrat.traffic_data.pop(seg_key, None)
            self.leg_dirs.pop(seg_key, None)

            table.removeRow(row)

        prune_unused_waypoints(getattr(self.omrat, 'waypoints', None) or {}, self.omrat.segment_data)

        # Rebuild junction registry to remove references to deleted legs.
        if hasattr(self.omrat, 'junctions') and self.omrat.junctions is not None:
            self.omrat.junctions.rebuild_from_segments(self.omrat.segment_data, prefer_user=True)

        # Refresh traffic UI so the removed leg no longer appears in the selector.
        self.omrat.run_traffic_module()
        _refresh_link_views(self, 'refresh_traffic_links')

        canvas = self.omrat.iface.mapCanvas() if self.omrat.iface else None
        if canvas is not None:
            canvas.refresh()

    def _auto_split_on_intersection(
        self,
        new_leg_id: str,
        endpoint: QgsPoint,
        orphan_layer: "QgsVectorLayer | None" = None,
    ) -> bool:
        """Split any existing leg that the newly added leg crosses.

        When a true X-intersection is found the two legs are replaced by
        four sub-legs meeting at the crossing point, the canvas and route
        table are rebuilt, and the junction registry is refreshed.

        ``orphan_layer`` is the transient vector layer created for the new
        leg before we knew a split was needed; it is removed from the QGIS
        project before ``reload_legs_from_segment_data`` recreates everything.

        Returns ``True`` if at least one split was applied so the caller can
        return early (skipping the normal post-draw bookkeeping that
        ``reload_legs_from_segment_data`` already handles).
        """
        from geometries.route_validation import (
            _segments_intersect,
            LegIntersection,
            split_leg_at_points,
            _make_id_provider,
        )
        sd = self.omrat.segment_data
        new_leg = sd.get(new_leg_id)
        if not isinstance(new_leg, dict):
            return False

        new_start = self._parse_wkt_xy(new_leg.get('Start_Point'))
        new_end = self._parse_wkt_xy(new_leg.get('End_Point'))
        if new_start is None or new_end is None:
            return False

        intersections: list[LegIntersection] = []
        for other_id, other_leg in list(sd.items()):
            if other_id == new_leg_id or not isinstance(other_leg, dict):
                continue
            other_start = self._parse_wkt_xy(other_leg.get('Start_Point'))
            other_end = self._parse_wkt_xy(other_leg.get('End_Point'))
            if other_start is None or other_end is None:
                continue
            # Shared endpoint == junction, not a crossing.
            if new_start in (other_start, other_end) or new_end in (other_start, other_end):
                continue
            hit = _segments_intersect(new_start, new_end, other_start, other_end)
            if hit is None:
                continue
            t1, t2, pt = hit
            intersections.append(LegIntersection(
                leg1_id=new_leg_id, leg2_id=other_id, point=pt, t1=t1, t2=t2,
            ))

        if not intersections:
            return False

        td = getattr(self.omrat, 'traffic_data', None)

        # Sort intersections by their position along the new leg so that
        # sub-legs are created in left-to-right (start-to-end) order.
        intersections.sort(key=lambda ix: ix.t1)

        # One shared id-provider so all new IDs are allocated without gaps.
        id_prov = _make_id_provider(sd)

        # Split the new leg at ALL crossing points in one atomic pass,
        # producing clean _a / _b / _c sub-legs instead of cascading _a_b_a.
        split_leg_at_points(
            sd, new_leg_id,
            [ix.point for ix in intersections],
            id_prov, td,
        )

        # Split each crossed leg once at its own intersection point.
        # Use split_leg_at_points so suffix-stripping and collision-avoidance
        # apply (same logic as for the new leg above).
        for ix in intersections:
            split_leg_at_points(sd, ix.leg2_id, [ix.point], id_prov, td)

        # Remove the transient layer that was already added to the QGIS
        # project — reload_legs_from_segment_data recreates all legs from
        # the now-mutated segment_data.
        if orphan_layer is not None:
            try:
                QgsProject.instance().removeMapLayer(orphan_layer.id())
            except Exception:  # nosec B110 B112
                pass

        self.reload_legs_from_segment_data()

        # Set start point for the next interactive leg draw.
        self.current_start_point = QgsPointXY(endpoint.x(), endpoint.y())
        self.point_layer = None

        # Sync the traffic segment combo box from the rebuilt route table.
        traffic = getattr(self.omrat, 'traffic', None)
        if traffic is not None and hasattr(traffic, 'fill_cbTrafficSelectSeg'):
            try:
                traffic.fill_cbTrafficSelectSeg()
            except Exception:  # nosec B110 B112
                pass

        # Rebuild junction registry so new crossing node gets a matrix.
        handler = getattr(self.omrat, 'junctions', None)
        if handler is not None:
            handler.rebuild_from_segments(sd, prefer_user=True)

        return True

    def reload_legs_from_segment_data(self) -> None:
        """Tear down all leg vector layers and rebuild them from ``segment_data``.

        Used by the route-validation pass after merges or splits have
        mutated ``segment_data``: simply rewriting the dict isn't enough
        because (a) the line layers on the canvas still draw the old
        geometry and (b) ``GatherData.get_segment_tbl`` reads endpoints
        back from ``twRouteList`` on save and would silently overwrite
        the merged values.

        Reuses ``OMRAT.load_lines`` to recreate the leg vector layers
        from the in-memory dict; rebuilds the route table rows
        directly so the actual ``Width`` and ``Leg_name`` values are
        preserved (``save_route`` would hardcode ``5000`` and
        ``LEG_{route}_{id}``).
        """
        widget = self.omrat.main_widget
        if widget is None:
            return

        # Disconnect geometry-changed slots on the existing layers.
        self.unwire_all_leg_layers()
        # Remove the existing leg layers from the QGIS project.
        for layer in list(self.vector_layers):
            try:
                QgsProject.instance().removeMapLayer(layer.id())
            except Exception:  # nosec B110 B112
                pass
        self.vector_layers = []
        # Tear down the tangent layer too -- ``ensure_tangent_layer``
        # rebuilds it on demand inside ``create_offset_lines``.
        if self.tangent_layer is not None:
            try:
                QgsProject.instance().removeMapLayer(self.tangent_layer.id())
            except Exception:  # nosec B110 B112
                pass
            self.tangent_layer = None

        segment_data = getattr(self.omrat, 'segment_data', {}) or {}

        # Rebuild the leg vector layers via the same path used on file
        # load -- one source of truth for the layer construction.
        widget.twRouteList.setRowCount(0)
        self.omrat.load_lines({'segment_data': segment_data})

        self.rebuild_route_table_rows()

        canvas = self.omrat.iface.mapCanvas() if self.omrat.iface else None
        if canvas is not None:
            canvas.refresh()

    def rebuild_route_table_rows(self, *, redraw_tangents: bool = True) -> None:
        """Rewrite every row of ``twRouteList`` from ``segment_data`` in
        its current order -- data cells, Tangent (%) cell, Update AIS
        button and AIS-lock box -- and reconnect the edit handler.

        Used after reloads and sorts.  Rebuilding rather than moving rows
        is what keeps the cell-widget buttons attached to the right leg:
        Qt's own table sorting moves items but leaves cell widgets where
        they were.
        """
        widget = self.omrat.main_widget
        if widget is None:
            return
        segment_data = getattr(self.omrat, 'segment_data', {}) or {}

        # Keep the edit handler quiet while rows are written.
        prev_item_changed = self.item_changed_connected
        self.suspend_route_table_signal()
        widget.twRouteList.setRowCount(0)

        max_id = 0
        for seg_id, seg in segment_data.items():
            if not isinstance(seg, dict):
                continue
            sp = self._parse_wkt_xy(seg.get('Start_Point'))
            ep = self._parse_wkt_xy(seg.get('End_Point'))
            if sp is None or ep is None:
                continue
            try:
                fid = int(str(seg_id))
                if fid > max_id:
                    max_id = fid
            except ValueError:
                continue
            try:
                route_id = int(seg.get('Route_Id', 1))
            except (TypeError, ValueError):
                route_id = 1
            try:
                width = float(seg.get('Width', 5000))
            except (TypeError, ValueError):
                width = 5000.0
            leg_name = str(seg.get('Leg_name', f'LEG_{route_id}_{fid}'))

            row_id = widget.twRouteList.rowCount()
            widget.twRouteList.setRowCount(row_id + 1)
            widget.twRouteList.setItem(row_id, 0, QTableWidgetItem(str(fid)))
            widget.twRouteList.setItem(row_id, 1, QTableWidgetItem(str(route_id)))
            widget.twRouteList.setItem(row_id, 2, QTableWidgetItem(leg_name))
            widget.twRouteList.setItem(
                row_id, 3, QTableWidgetItem(self.format_wkt(QgsPoint(sp[0], sp[1]))),
            )
            widget.twRouteList.setItem(
                row_id, 4, QTableWidgetItem(self.format_wkt(QgsPoint(ep[0], ep[1]))),
            )
            widget.twRouteList.setItem(row_id, 5, QTableWidgetItem(f"{int(width)}"))
            widget.twRouteList.setItem(
                row_id, 6,
                QTableWidgetItem(percent_from_fraction(normalize_tangent_pos(seg.get(TANGENT_POS_KEY)))),
            )
            btn = QPushButton("Update AIS")
            btn.clicked.connect(
                lambda _checked=False, k=str(seg_id): self.omrat.ais.update_legs(k)
            )
            widget.twRouteList.setCellWidget(row_id, 7, btn)
            widget.twRouteList.setItem(row_id, 8, self._make_lock_item(seg))

            if redraw_tangents:
                try:
                    self.create_offset_lines(
                        QgsPointXY(sp[0], sp[1]),
                        QgsPointXY(ep[0], ep[1]),
                        width / 2,
                        fid,
                    )
                except Exception:  # nosec B110 B112
                    pass

        # Bump the leg-id counter past the highest current id so future
        # interactive draws don't collide.
        self.segment_id = max(max_id, self.segment_id)
        self.sync_drawing_spinboxes()

        if prev_item_changed:
            self.ensure_route_table_signal()

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    def _wire_route_table_header(self) -> None:
        try:
            header = self.omrat.main_widget.twRouteList.horizontalHeader()
            header.setSectionsClickable(True)
            header.setSortIndicatorShown(False)
            header.sectionClicked.connect(self.sort_route_table)
        except Exception:  # nosec B110 B112
            pass

    def sort_route_table(self, column: int) -> None:
        """Header click on Segment_Id / Route_Id / Leg_name: order the legs
        naturally (``LEG_1_2`` before ``LEG_1_10``) by that column.

        The order is applied to ``segment_data`` itself, so the route
        table, the Traffic tab's leg selector and the saved project all
        agree.  A second click on the same column reverses the order.
        """
        key = SORTABLE_COLUMNS.get(int(column))
        if key is None:
            return
        prev_col, prev_rev = self._route_sort if self._route_sort else (None, True)
        reverse = not prev_rev if prev_col == column else False
        self._route_sort = (column, reverse)

        sd = getattr(self.omrat, 'segment_data', None)
        if not isinstance(sd, dict) or not sd:
            return
        ordered = sort_segment_data(sd, key, reverse=reverse)
        sd.clear()
        sd.update(ordered)

        self.rebuild_route_table_rows(redraw_tangents=False)
        try:
            header = self.omrat.main_widget.twRouteList.horizontalHeader()
            header.setSortIndicatorShown(True)
            header.setSortIndicator(
                column, Qt.SortOrder.DescendingOrder if reverse else Qt.SortOrder.AscendingOrder,
            )
        except Exception:  # nosec B110 B112
            pass
        # Traffic tab: the leg selector mirrors the table order.
        run_traffic = getattr(self.omrat, 'run_traffic_module', None)
        if callable(run_traffic):
            run_traffic()

    def on_geometry_changed(self, fid: int, geom: QgsGeometry):
        """Handle geometry changes for a feature."""
        # Get the segment ID from the feature's attributes
        polyline: list[QgsGeometry] = geom.asPolyline()
        assert isinstance(polyline, list)  # nosec B101
        start_point_: QgsGeometry = polyline[0]
        end_point_: QgsGeometry = polyline[-1]
        if isinstance(start_point_, QgsPointXY) and isinstance(end_point_, QgsPointXY):
            start_point = QgsPoint(start_point_.x(), start_point_.y())
            end_point = QgsPoint(end_point_.x(), end_point_.y())
            start_pointXY = start_point_
            end_pointXY = end_point_
        elif isinstance(start_point_, QgsPoint) and isinstance(end_point_, QgsPoint):
            start_point = start_point_
            end_point = end_point_
            start_pointXY = QgsPointXY(start_point_.x(), start_point_.y())
            end_pointXY = QgsPointXY(end_point_.x(), end_point_.y())
        else:
            raise TypeError("Unknown data point")

        assert isinstance(start_point, QgsPoint)  # nosec B101
        assert isinstance(end_point, QgsPoint)  # nosec B101
        # Update the start and end points in the table
        assert (self.omrat.main_widget is not None)  # nosec B101

        # Capture the OLD endpoints of this leg before we overwrite them
        # so the shared-vertex propagation (below) can match siblings.
        old_seg = self.omrat.segment_data.get(str(fid)) or {}
        old_start = self._parse_wkt_xy(old_seg.get('Start_Point'))
        old_end = self._parse_wkt_xy(old_seg.get('End_Point'))

        for row in range(self.omrat.main_widget.twRouteList.rowCount()):
            if int(self.omrat.main_widget.twRouteList.item(row, 0).text()) == fid:
                self.omrat.main_widget.twRouteList.item(row, 3).setText(self.format_wkt(start_point))
                self.omrat.main_widget.twRouteList.item(row, 4).setText(self.format_wkt(end_point))

                # Get the width from the table
                width = float(self.omrat.main_widget.twRouteList.item(row, 5).text())

                # Update the tangent line for this segment
                self.create_offset_lines(start_pointXY, end_pointXY, width / 2, fid)

                # Keep backing segment_data in sync so save/export uses edited geometry.
                seg_key = str(fid)
                if seg_key in self.omrat.segment_data:
                    self.omrat.segment_data[seg_key]['Start_Point'] = self.format_wkt(start_point)
                    self.omrat.segment_data[seg_key]['End_Point'] = self.format_wkt(end_point)

                    # Recompute heading-based direction labels and line length in meters.
                    # Use the matching QgsPointXY overload of azimuth — passing a
                    # QgsPointXY to QgsPoint.azimuth raises a type-mismatch error
                    # in QGIS 4 / Qt 6.
                    degrees: float = (start_pointXY.azimuth(end_pointXY) + 360) % 360
                    if degrees > 315 or degrees <= 45:
                        dirs = ['North going', 'South going']
                    elif degrees > 45 and degrees <= 135:
                        dirs = ['East going', 'West going']
                    elif degrees > 135 and degrees <= 225:
                        dirs = ['South going', 'North going']
                    else:
                        dirs = ['West going', 'East going']
                    self.omrat.segment_data[seg_key]['Dirs'] = dirs
                    self.leg_dirs[seg_key] = dirs

                    longitude = (start_pointXY.x() + end_pointXY.x()) / 2
                    utm_zone = int((longitude + 180) / 6) + 1
                    is_northern = start_point.y() >= 0
                    utm_crs = QgsCoordinateReferenceSystem(
                        f"EPSG:{32600 + utm_zone if is_northern else 32700 + utm_zone}"
                    )
                    transform_to_utm = QgsCoordinateTransform(
                        QgsCoordinateReferenceSystem("EPSG:4326"),
                        utm_crs,
                        QgsProject.instance(),
                    )
                    start_utm = transform_to_utm.transform(start_pointXY)
                    end_utm = transform_to_utm.transform(end_pointXY)
                    self.omrat.segment_data[seg_key]['line_length'] = start_utm.distance(end_utm)

                # Move the node(s) this vertex belongs to, so every leg
                # sharing the junction follows (and snap onto another node
                # when dropped on one).
                self._apply_vertex_move(
                    fid, old_start, old_end,
                    (start_pointXY.x(), start_pointXY.y()),
                    (end_pointXY.x(), end_pointXY.y()),
                )

                # Stop processing once the correct row is updated
                return

    @staticmethod
    def label_layer(layer: QgsVectorLayer) -> None:
        """Label the layer with the 'label' field."""
        settings = QgsPalLayerSettings()
        settings.fieldName = "label"  # Use the 'label' field for labeling
        settings.placement = QgsPalLayerSettings.Placement.Line
        settings.enabled = True

        labeling = QgsVectorLayerSimpleLabeling(settings)
        layer.setLabeling(labeling)
        layer.setLabelsEnabled(True)

        # Trigger a refresh of the layer's labeling
        layer.triggerRepaint()

    @staticmethod
    def style_layer(layer: QgsVectorLayer) -> None:
        """Style the layer with a thicker line."""
        # Get the layer's renderer and symbol
        renderer: QgsFeatureRenderer | None = layer.renderer()
        if (renderer is None):
            renderer = QgsSingleSymbolRenderer(QgsLineSymbol())
            layer.setRenderer(renderer)

        symbol = renderer.symbol()
        assert (isinstance(symbol, QgsLineSymbol))  # nosec B101
        symbol.setWidth(1.5)  # Set line thickness
        symbol.setColor(QColor("blue"))  # Optional: Set line color

        # Trigger a refresh of the layer's symbology
        layer.triggerRepaint()

    def save_route(self, point1: QgsPoint, point2: QgsPoint):
        """Save route information to the twRouteList table."""
        assert (self.omrat.main_widget is not None)  # nosec B101
        row_id = self.omrat.main_widget.twRouteList.rowCount()
        self.omrat.main_widget.twRouteList.setRowCount(row_id + 1)

        # Create table items
        item1 = QTableWidgetItem(f'{self.segment_id}')
        item2 = QTableWidgetItem(f'{self.cur_route_id}')
        # Six decimals like every other endpoint writer: save reads the
        # endpoints back from this table, and junctions are detected by
        # exact coordinate match, so a 5-decimal copy here silently
        # detached a leg drawn from an existing junction.
        item3 = QTableWidgetItem(self.format_wkt(point1))
        item4 = QTableWidgetItem(self.format_wkt(point2))
        item5 = QTableWidgetItem('5000')  # Default width
        item6 = QTableWidgetItem(self.current_leg_name())  # Leg name
        item7 = QTableWidgetItem(percent_from_fraction(DEFAULT_TANGENT_POS))  # Tangent position

        # Add items to the table
        self.omrat.main_widget.twRouteList.setItem(row_id, 0, item1)
        self.omrat.main_widget.twRouteList.setItem(row_id, 1, item2)
        self.omrat.main_widget.twRouteList.setItem(row_id, 2, item6)
        self.omrat.main_widget.twRouteList.setItem(row_id, 3, item3)
        self.omrat.main_widget.twRouteList.setItem(row_id, 4, item4)
        self.omrat.main_widget.twRouteList.setItem(row_id, 5, item5)
        self.omrat.main_widget.twRouteList.setItem(row_id, 6, item7)
        btn_update_ais = QPushButton("Update AIS")
        btn_update_ais.clicked.connect(lambda: self.omrat.ais.update_legs(str(self.segment_id)))
        self.omrat.main_widget.twRouteList.setCellWidget(row_id, 7, btn_update_ais)
        self.omrat.main_widget.twRouteList.setItem(row_id, 8, self._make_lock_item(None))

        self.ensure_route_table_signal()

    # ------------------------------------------------------------------
    # Route table wiring shared by drawing, reload and file load
    # ------------------------------------------------------------------

    def ensure_route_table_signal(self) -> None:
        """Connect ``twRouteList.itemChanged`` -> ``on_width_changed`` once."""
        if self.item_changed_connected:
            return
        self.omrat.main_widget.twRouteList.itemChanged.connect(self.on_width_changed)
        self.item_changed_connected = True

    def suspend_route_table_signal(self) -> None:
        """Disconnect the edit handler before a bulk repopulation so
        ``setItem`` calls are not mistaken for user edits."""
        if not self.item_changed_connected:
            return
        try:
            self.omrat.main_widget.twRouteList.itemChanged.disconnect(self.on_width_changed)
        except TypeError:
            pass
        self.item_changed_connected = False

    def finish_route_table_rows(self) -> None:
        """Complete rows written by a bulk population (file load).

        ``GatherData.populate_segment_tbl`` writes the six data columns
        only.  This adds the Tangent (%) cell and the Update AIS button
        to every row that lacks them and (re)connects the edit signal,
        so width / tangent edits on a loaded project reach the canvas
        exactly like on a freshly drawn one.
        """
        widget = self.omrat.main_widget
        if widget is None:
            return
        table = widget.twRouteList
        for row in range(table.rowCount()):
            id_item = table.item(row, 0)
            if id_item is None:
                continue
            seg_key = id_item.text()
            if table.item(row, 6) is None:
                table.setItem(
                    row, 6, QTableWidgetItem(percent_from_fraction(self.stored_tangent_pos(seg_key))),
                )
            if table.cellWidget(row, 7) is None:
                btn = QPushButton("Update AIS")
                btn.clicked.connect(
                    lambda _checked=False, k=seg_key: self.omrat.ais.update_legs(k)
                )
                table.setCellWidget(row, 7, btn)
            if table.item(row, 8) is None:
                seg = (getattr(self.omrat, 'segment_data', None) or {}).get(seg_key)
                table.setItem(row, 8, self._make_lock_item(seg if isinstance(seg, dict) else None))
        self.ensure_route_table_signal()

    # ------------------------------------------------------------------
    # AIS lock column (col 8)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_lock_item(seg: dict | None) -> QTableWidgetItem:
        """Checkbox cell mirroring ``segment_data[seg]['traffic_locked']``."""
        item = QTableWidgetItem()
        item.setFlags(
            Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        )
        locked = isinstance(seg, dict) and seg.get(LOCK_KEY) is True
        item.setCheckState(Qt.CheckState.Checked if locked else Qt.CheckState.Unchecked)
        tip = "Tick to keep this leg's traffic and distributions when AIS data are updated."
        if isinstance(seg, dict) and seg.get(SOURCE_KEY):
            tip += f"\nTraffic copied from leg {seg.get(SOURCE_KEY)}."
        item.setToolTip(tip)
        return item

    def set_traffic_locked(self, segment_id: int | str, locked: bool) -> None:
        """Store the lock flag and mirror it into the route table.
        Unlocking a copy also releases its copy link (:meth:`_apply_lock`)."""
        self._apply_lock(segment_id, locked)
        self.sync_lock_column(segment_id)

    def _apply_lock(self, segment_id: int | str, locked: bool) -> str | None:
        """Lock / unlock one leg.  Unlocking a copy drops its
        ``traffic_source`` and re-derives the junctions it touches (the
        link forced 100 % continuation there, see
        ``geometries.junctions.linked_partners``).  Returns the released
        source leg, or ``None``."""
        seg = str(segment_id)
        src = None
        if locked:
            set_locked(self.omrat.segment_data, seg, True)
        else:
            src = release_copy(self.omrat.segment_data, seg)
        if src is not None:
            handler = getattr(self.omrat, 'junctions', None)
            if handler is not None:
                try:
                    # AIS-sourced junctions fall back to geometry (and are
                    # re-counted by the next update); geometry ones are
                    # recomputed without the link by the rebuild.
                    handler.invalidate_legs([seg], self.omrat.segment_data)
                    handler.rebuild_from_segments(self.omrat.segment_data, prefer_user=True)
                except Exception:  # nosec B110 B112
                    pass
        _refresh_link_views(self, 'refresh_traffic_link_views')
        return src

    def sync_lock_column(self, segment_id: int | str) -> None:
        """Redraw the lock checkbox of one row from ``segment_data``."""
        row = self._tangent_row_for_segment(segment_id)
        if row is None:
            return
        table = self.omrat.main_widget.twRouteList
        seg = (getattr(self.omrat, 'segment_data', None) or {}).get(str(segment_id))
        self._table_sync_guard = True
        try:
            table.setItem(row, 8, self._make_lock_item(seg if isinstance(seg, dict) else None))
        finally:
            self._table_sync_guard = False

    def open_copy_traffic_dialog(self) -> None:
        from omrat_utils import copy_traffic_dialog
        copy_traffic_dialog.run(self.omrat)

    def open_suppress_leg_dialog(self) -> None:
        from omrat_utils import suppress_leg_dialog
        suppress_leg_dialog.run(self.omrat)

    # ------------------------------------------------------------------
    # Suppressed legs: drawn dashed (compute/traffic_redirect.py)
    # ------------------------------------------------------------------

    def leg_layer_for(self, segment_id: int | str) -> "QgsVectorLayer | None":
        """Leg layer whose feature carries ``segmentId == segment_id``
        (compared as text: the attribute is an int or a str by origin)."""
        key = str(segment_id)
        for layer in list(self.vector_layers):
            try:
                for feat in layer.getFeatures():
                    if str(feat["segmentId"]) == key:
                        return layer
            except Exception:  # nosec B110 B112
                continue
        return None

    @staticmethod
    def set_layer_dashed(layer: QgsVectorLayer, dashed: bool) -> bool:
        """Switch the line symbol layers of ``layer`` between dashed and
        solid, keeping colour and width.  Returns ``True`` when a symbol
        layer was changed."""
        renderer = layer.renderer() if layer is not None else None
        if renderer is None:
            return False
        if isinstance(renderer, QgsSingleSymbolRenderer):
            symbols = [renderer.symbol()]
        else:
            from qgis.core import QgsRenderContext
            symbols = list(renderer.symbols(QgsRenderContext()))
        style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
        changed = False
        for symbol in symbols:
            if symbol is None:
                continue
            for sl in symbol.symbolLayers():
                if hasattr(sl, 'setPenStyle'):
                    sl.setPenStyle(style)
                    changed = True
        layer.triggerRepaint()
        try:
            view = _layer_tree_view()
            if view is not None:
                view.refreshLayerSymbology(layer.id())
        except Exception:  # nosec B110 B112
            pass
        return changed

    def sync_suppressed_style(self, segment_id: int | str) -> None:
        """Dash (or restore) one leg on the map from ``segment_data``."""
        from compute.traffic_redirect import is_suppressed
        layer = self.leg_layer_for(segment_id)
        if layer is not None:
            self.set_layer_dashed(layer, is_suppressed(self.omrat.segment_data, str(segment_id)))

    def refresh_suppressed_styles(self) -> None:
        """Dash every suppressed leg (after load / a project style apply)."""
        from compute.traffic_redirect import suppressed_legs
        for seg_id in suppressed_legs(getattr(self.omrat, 'segment_data', None) or {}):
            self.sync_suppressed_style(seg_id)

    # ------------------------------------------------------------------
    # Traffic links: which leg's traffic refers to which
    # (omrat_utils/traffic_links.py)
    # ------------------------------------------------------------------

    LINKS_LAYER_NAME = "Traffic links"
    _LINK_STYLE = {
        # kind: (colour, pen style, legend label)
        'copy': ('#1a9641', Qt.PenStyle.SolidLine, 'Copied traffic (source -> copy)'),
        'redirect': ('#f07c00', Qt.PenStyle.DashLine, 'Moved traffic (suppressed -> target)'),
        'with': ('#7f7f7f', Qt.PenStyle.DotLine, 'Suppressed with (member -> lead)'),
    }

    def refresh_traffic_link_views(self) -> None:
        """Bring every view of the copy / lock / suppress links up to date:
        the Traffic tab leg labels, the links layer and the highlight."""
        from omrat_utils.traffic_links import status_suffix
        segs = getattr(self.omrat, 'segment_data', None) or {}
        try:
            cb = self.omrat.main_widget.cbTrafficSelectSeg
            for i in range(cb.count()):
                base = cb.itemText(i).split('  [')[0]
                cb.setItemText(i, f"{base}{status_suffix(str(cb.itemData(i)), segs)}")
        except Exception:  # nosec B110 B112
            pass
        self.refresh_traffic_links()

    def show_traffic_links(self, show: bool) -> None:
        """Switch the "Traffic links" map layer on / off (toggle button)."""
        self._links_shown = bool(show)
        if self._links_shown:
            self.refresh_traffic_links()
        else:
            self._remove_traffic_links_layer()
            self._clear_link_highlight()

    def refresh_traffic_links(self) -> None:
        """Rebuild the links layer from ``segment_data`` (no-op when off)."""
        if not self._links_shown:
            return
        from omrat_utils.traffic_links import link_geometries
        layer = self._ensure_traffic_links_layer()
        if layer is None:
            return
        provider = layer.dataProvider()
        provider.truncate()
        feats = []
        for link, pts in link_geometries(getattr(self.omrat, 'segment_data', None) or {}):
            feat = QgsFeature(layer.fields())
            feat.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(x, y) for x, y in pts]))
            feat.setAttributes([link.kind, link.src, link.dst, link.label])
            feats.append(feat)
        provider.addFeatures(feats)
        layer.updateExtents()
        layer.triggerRepaint()
        if self._highlight_seg is not None:
            self.highlight_traffic_links(self._highlight_seg)

    def _ensure_traffic_links_layer(self) -> "QgsVectorLayer | None":
        layer = self.traffic_links_layer
        try:
            if layer is not None and QgsProject.instance().mapLayer(layer.id()) is not None:
                return layer
        except RuntimeError:
            pass
        layer = QgsVectorLayer("LineString?crs=EPSG:4326", self.LINKS_LAYER_NAME, "memory")
        fields = QgsFields()
        for name in ('kind', 'src', 'dst', 'label'):
            fields.append(QgsField(name, QMetaType.Type.QString))
        layer.dataProvider().addAttributes(fields.toList())
        layer.updateFields()
        self._style_traffic_links_layer(layer)
        QgsProject.instance().addMapLayer(layer)
        self.traffic_links_layer = layer
        return layer

    def _style_traffic_links_layer(self, layer: QgsVectorLayer) -> None:
        """Categorised on ``kind``: coloured line + arrow head at the target."""
        from qgis.core import (
            Qgis, QgsCategorizedSymbolRenderer, QgsMarkerLineSymbolLayer, QgsMarkerSymbol,
            QgsRendererCategory,
        )
        cats = []
        for kind, (colour, pen, legend) in self._LINK_STYLE.items():
            symbol = QgsLineSymbol()
            line = symbol.symbolLayer(0)
            line.setColor(QColor(colour))
            line.setWidth(0.7)
            line.setPenStyle(pen)
            head = QgsMarkerLineSymbolLayer()
            try:
                head.setPlacements(Qgis.MarkerLinePlacements(Qgis.MarkerLinePlacement.LastVertex))
            except (AttributeError, TypeError):
                pass
            head.setSubSymbol(QgsMarkerSymbol.createSimple(
                {'name': 'filled_arrowhead', 'color': colour, 'outline_style': 'no', 'size': '3.5'}))
            symbol.appendSymbolLayer(head)
            cats.append(QgsRendererCategory(kind, symbol, legend))
        layer.setRenderer(QgsCategorizedSymbolRenderer('kind', cats))
        settings = QgsPalLayerSettings()
        settings.fieldName = "label"
        settings.placement = QgsPalLayerSettings.Placement.Curved
        settings.enabled = True
        layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
        layer.setLabelsEnabled(True)

    def _remove_traffic_links_layer(self) -> None:
        layer = self.traffic_links_layer
        self.traffic_links_layer = None
        if layer is None:
            return
        try:
            if QgsProject.instance().mapLayer(layer.id()) is not None:
                QgsProject.instance().removeMapLayer(layer.id())
        except RuntimeError:
            pass

    def highlight_traffic_links(self, segment_id: int | str) -> None:
        """Highlight leg ``segment_id`` (yellow) and every leg its traffic
        is linked with (orange) on the canvas."""
        from omrat_utils.traffic_links import related_legs
        self._clear_link_highlight()
        seg = str(segment_id)
        segs = getattr(self.omrat, 'segment_data', None) or {}
        related = related_legs(seg, segs)
        self._highlight_seg = seg
        if not related:
            return
        canvas = self.omrat.iface.mapCanvas() if self.omrat.iface else None
        if canvas is None:
            return
        crs = QgsCoordinateReferenceSystem("EPSG:4326")
        for leg, colour in [(seg, QColor(255, 220, 0, 200))] + [(r, QColor(255, 120, 0, 170)) for r in sorted(related)]:
            seg_d = segs.get(leg)
            if not isinstance(seg_d, dict):
                continue
            try:
                geom = QgsGeometry.fromWkt(f"LINESTRING({seg_d['Start_Point']}, {seg_d['End_Point']})")
                band = QgsRubberBand(canvas, QgsWkbTypes.GeometryType.LineGeometry)
                band.setColor(colour)
                band.setWidth(8)
                band.setToGeometry(geom, crs)
                self._link_bands.append(band)
            except Exception:  # nosec B110 B112
                continue

    def highlighted_legs(self) -> int:
        """Number of highlight bands on the canvas (for tests)."""
        return len(self._link_bands)

    def _clear_link_highlight(self) -> None:
        canvas = self.omrat.iface.mapCanvas() if self.omrat.iface else None
        for band in self._link_bands:
            try:
                band.reset(QgsWkbTypes.GeometryType.LineGeometry)
                if canvas is not None:
                    canvas.scene().removeItem(band)
            except Exception:  # nosec B110 B112
                pass
        self._link_bands = []
        self._highlight_seg = None

    def start_move_tangent(self) -> None:
        """One-click entry point for dragging a tangent line.

        Activates the *Tangent Line* layer, makes sure it is in edit
        mode and starts QGIS's own **Move Feature** tool, so the user
        does not have to find the Advanced Digitizing toolbar.  The
        drag itself is handled by ``_on_tangent_geometry_changed``.
        """
        self.ensure_tangent_layer()
        self.ensure_tangent_fields()
        layer = self.tangent_layer
        if layer is None:
            return
        if layer.featureCount() == 0:
            self._notify(self.omrat.tr(
                "There are no tangent lines to move yet. Draw or load a route first."
            ), duration=6)
            return
        iface = self.omrat.iface
        try:
            iface.setActiveLayer(layer)
        except Exception:  # nosec B110 B112
            pass
        if not layer.isEditable():
            layer.startEditing()
        getter = getattr(iface, 'actionMoveFeature', None)
        action = getter() if callable(getter) else None
        if action is None:
            self._notify(self.omrat.tr(
                "Tangent Line layer is ready for editing. Pick 'Move Feature' on the "
                "Advanced Digitizing toolbar and drag a tangent line along its leg."
            ), duration=10)
            return
        action.trigger()
        self._notify(self.omrat.tr(
            "Drag a tangent line along its leg. It snaps back onto the leg when released. "
            "Choose the Pan tool when you are done."
        ), duration=10)

    def _notify(self, message: str, *, duration: int = 8) -> None:
        notifier = getattr(self.omrat, 'notifier', None)
        if notifier is None:
            return
        try:
            notifier.display_message(message, duration=duration)
        except Exception:  # nosec B110 B112
            pass

    def on_route_table_cell_clicked(self, row: int, column: int):
        """Called when any cell in the route table is clicked."""
        segment_id_item = self.omrat.main_widget.twRouteList.item(row, 0)
        if segment_id_item is not None:
            try:
                segment_id = segment_id_item.text()
                self.omrat.distributions.run_update_plot(segment_id)
            except ValueError:
                pass  # Handle or log invalid segment_id if needed
            if getattr(self, '_links_shown', False):
                self.highlight_traffic_links(segment_id_item.text())

    def update_segment_data(self, point: QgsPoint) -> None:
        main_widget = self.omrat.main_widget
        assert (self.current_start_point is not None)  # nosec B101
        pointXY = QgsPointXY(point.x(), point.y())
        degrees: float = (self.current_start_point.azimuth(pointXY) + 360) % 360
        if degrees > 315 or degrees <= 45:
            main_widget.laDir1.setText('North going')
            main_widget.laDir2.setText('South going')
            dirs: list[str] = ['North going', 'South going']
        elif degrees > 45 and degrees <= 135:
            main_widget.laDir1.setText('East going')
            main_widget.laDir2.setText('West going')
            dirs: list[str] = ['East going', 'West going']
        elif degrees > 135 and degrees <= 225:
            main_widget.laDir1.setText('South going')
            main_widget.laDir2.setText('North going')
            dirs: list[str] = ['South going', 'North going']
        elif degrees > 225 and degrees <= 315:
            main_widget.laDir1.setText('West going')
            main_widget.laDir2.setText('East going')
            dirs: list[str] = ['West going', 'East going']
        else:
            return
        longitude = (pointXY.x() + self.current_start_point.x()) / 2
        utm_zone = int((longitude + 180) / 6) + 1
        is_northern = self.current_start_point.y() >= 0
        utm_epsg = 32600 + utm_zone if is_northern else 32700 + utm_zone
        utm_crs = QgsCoordinateReferenceSystem(f"EPSG:{utm_epsg}")

        transform_to_utm = QgsCoordinateTransform(
            QgsCoordinateReferenceSystem("EPSG:4326"), utm_crs, QgsProject.instance()
        )
        start_utm = transform_to_utm.transform(self.current_start_point)
        dist = start_utm.distance(transform_to_utm.transform(pointXY))
        seg_key = f'{self.segment_id}'
        # "lon lat" with six decimals -- the one endpoint format the file,
        # the route table, the AIS passage line and the waypoint registry
        # share.  (Until v0.15.2 a freshly drawn leg carried WKT
        # ``Point (...)`` text here until the first save rewrote it.)
        start_wkt = self.format_wkt(QgsPoint(self.current_start_point.x(), self.current_start_point.y()))
        end_wkt = self.format_wkt(QgsPoint(point.x(), point.y()))
        if seg_key in self.omrat.segment_data:
            self.omrat.segment_data[seg_key]['Start_Point'] = start_wkt
            self.omrat.segment_data[seg_key]['End_Point'] = end_wkt
            self.omrat.segment_data[seg_key]['Dirs'] = dirs
            self.omrat.segment_data[seg_key]['line_length'] = dist
        else:
            self.omrat.segment_data[seg_key] = {
                'Start_Point': start_wkt,
                'End_Point': end_wkt,
                'Dirs': dirs, 'Width': 5000, 'line_length': dist,
                TANGENT_POS_KEY: DEFAULT_TANGENT_POS,
                'Route_Id': self.cur_route_id,
                'Segment_Id': self.segment_id,
                'Leg_name': self.current_leg_name(),
            }
            wps = self._waypoints()
            self.omrat.segment_data[seg_key][WP_START] = ensure_waypoint(
                wps, (self.current_start_point.x(), self.current_start_point.y()),
            )
            self.omrat.segment_data[seg_key][WP_END] = ensure_waypoint(wps, (point.x(), point.y()))
            # Initialise an empty traffic block so save() and the UI
            # don't crash with KeyError before AIS data are loaded.
            traffic = getattr(self.omrat, 'traffic', None)
            if traffic is not None and seg_key not in self.omrat.traffic_data:
                traffic.create_empty_dict(seg_key, dirs)
            # Stamp the write-once import baseline for the audit report.
            imported = getattr(self.omrat, 'segments_imported', None)
            if isinstance(imported, dict) and f'{self.segment_id}' not in imported:
                imported[f'{self.segment_id}'] = {
                    'Start_Point': QgsPoint(
                        self.current_start_point.x(),
                        self.current_start_point.y(),
                    ).asWkt(),
                    'End_Point': point.asWkt(),
                }
        self.leg_dirs[f'{self.segment_id}'] = dirs
        main_widget.cbTrafficSelectSeg.addItem(self.current_leg_name(), f'{self.segment_id}')
        self.omrat.traffic.c_seg = f'{self.segment_id}'
        self.current_start_point = None

    def _tangent_layer_in_project(self) -> bool:
        """False when the cached tangent layer was removed from the
        project behind our back (Layers panel delete / new project)."""
        if self.tangent_layer is None:
            return False
        try:
            layer_id = self.tangent_layer.id()
        except RuntimeError:
            return False
        project = QgsProject.instance()
        return project is not None and project.mapLayer(layer_id) is not None

    def ensure_tangent_layer(self):
        if self.tangent_layer is not None and not self._tangent_layer_in_project():
            stale = self.tangent_layer
            self.tangent_layer = None
            self.vector_layers = [lyr for lyr in self.vector_layers if lyr is not stale]
        if self.tangent_layer is None:
            self.tangent_layer = QgsVectorLayer("LineString?crs=EPSG:4326", "Tangent Line", "memory")
            if not self.tangent_layer.isValid():
                raise RuntimeError("Tangent line layer is not valid")
            QgsProject.instance().addMapLayer(self.tangent_layer)
            self.vector_layers.append(self.tangent_layer)
            # The layer stays in edit mode so the user can drag a tangent
            # with QGIS's own Move Feature / vertex tools; we catch the
            # edit and snap the line back onto the leg.
            self.tangent_layer.geometryChanged.connect(self._on_tangent_geometry_changed)
            apply_stored_style(self.omrat, 'tangent', self.tangent_layer)
        self.tangent_layer.startEditing()

    def ensure_tangent_fields(self):
        if self.tangent_layer.fields().lookupField("type") < 0:
            pr = self.tangent_layer.dataProvider()
            pr.addAttributes([QgsField("type", QMetaType.Type.QString)])
            self.tangent_layer.updateFields()

    def calculate_midpoint_utm(
        self, start: QgsPointXY, end: QgsPointXY, tangent_pos: float = DEFAULT_TANGENT_POS,
    ) -> tuple[QgsPointXY, QgsCoordinateTransform, QgsCoordinateTransform]:
        """UTM point at fraction ``tangent_pos`` along ``start -> end``
        (``0.5`` = midpoint) plus the to/from-UTM transforms."""
        longitude = (start.x() + end.x()) / 2
        utm_zone = int((longitude + 180) / 6) + 1
        is_northern = start.y() >= 0
        utm_epsg = 32600 + utm_zone if is_northern else 32700 + utm_zone
        utm_crs = QgsCoordinateReferenceSystem(f"EPSG:{utm_epsg}")

        transform_to_utm = QgsCoordinateTransform(
            QgsCoordinateReferenceSystem("EPSG:4326"), utm_crs, QgsProject.instance()
        )
        transform_to_canvas = QgsCoordinateTransform(
            utm_crs, QgsCoordinateReferenceSystem("EPSG:4326"), QgsProject.instance()
        )

        start_utm = transform_to_utm.transform(start)
        end_utm = transform_to_utm.transform(end)
        mx, my = point_along(
            (start_utm.x(), start_utm.y()), (end_utm.x(), end_utm.y()), tangent_pos,
        )
        mid_utm = QgsPointXY(mx, my)

        return mid_utm, transform_to_utm, transform_to_canvas

    def remove_existing_tangent(self, segment_id: int):
        ids = [f.id() for f in self.tangent_layer.getFeatures() if f["type"] == f"Tangent Line {segment_id}"]
        if not ids:
            return
        # Use the data provider directly — consistent with add_tangent_feature
        # which also uses dataProvider().addFeatures(), so no edit mode needed.
        self.tangent_layer.dataProvider().deleteFeatures(ids)
        self.tangent_layer.triggerRepaint()

    def add_tangent_feature(self, start: QgsPointXY, end: QgsPointXY, segment_id: int):
        self.remove_existing_tangent(segment_id)
        fet = QgsFeature()
        fet.setGeometry(QgsLineString([QgsPoint(start.x(), start.y()), QgsPoint(end.x(), end.y())]))
        fet.setAttributes([f"Tangent Line {segment_id}"])
        self.tangent_layer.dataProvider().addFeatures([fet])

    def stored_tangent_pos(self, segment_id: int | str) -> float:
        """``Tangent_Pos`` fraction for ``segment_id`` from ``segment_data``
        (midpoint when the leg is unknown or the value is malformed)."""
        seg = (getattr(self.omrat, 'segment_data', None) or {}).get(str(segment_id))
        if not isinstance(seg, dict):
            return DEFAULT_TANGENT_POS
        return normalize_tangent_pos(seg.get(TANGENT_POS_KEY))

    def create_offset_lines(
        self, start_point: QgsPointXY, end_point: QgsPointXY, offset_distance: float, segment_id: int,
        tangent_pos: float | None = None,
    ):
        """(Re)draw the tangent line of ``segment_id``.

        ``tangent_pos`` is the fraction along the leg; ``None`` reads the
        value stored in ``segment_data`` so every existing caller (width
        edits, vertex drags, reloads) keeps a moved tangent in place.
        """
        if not is_valid_point_pair(start_point, end_point):
            return

        if tangent_pos is None:
            tangent_pos = self.stored_tangent_pos(segment_id)
        tangent_pos = normalize_tangent_pos(tangent_pos)

        self.ensure_tangent_layer()
        self.ensure_tangent_fields()

        mid_utm, to_utm, to_canvas = self.calculate_midpoint_utm(start_point, end_point, tangent_pos)
        start_utm = to_utm.transform(start_point)
        end_utm = to_utm.transform(end_point)

        result = calculate_tangent_line(mid_utm, start_utm, end_utm, offset_distance)
        if result is None:
            return
        tangent_start_utm, tangent_end_utm = result
        tangent_start = to_canvas.transform(tangent_start_utm)
        tangent_end = to_canvas.transform(tangent_end_utm)

        self.add_tangent_feature(tangent_start, tangent_end, segment_id)
        self.tangent_layer.commitChanges()
        self.tangent_layer.triggerRepaint()

    # ------------------------------------------------------------------
    # Movable tangent line
    # ------------------------------------------------------------------

    def _disconnect_tangent_signal(self) -> None:
        if self.tangent_layer is None:
            return
        try:
            self.tangent_layer.geometryChanged.disconnect(self._on_tangent_geometry_changed)
        except (TypeError, RuntimeError):
            pass

    @staticmethod
    def _segment_id_from_tangent_type(type_text: object) -> int | None:
        """``'Tangent Line 3' -> 3``; ``None`` for anything else."""
        if not isinstance(type_text, str):
            return None
        prefix = 'Tangent Line '
        if not type_text.startswith(prefix):
            return None
        try:
            return int(type_text[len(prefix):].strip())
        except ValueError:
            return None

    def _leg_endpoints(self, segment_id: int) -> tuple[QgsPointXY, QgsPointXY, float] | None:
        """``(start, end, width)`` of a leg from ``segment_data``, falling
        back to the route table.  ``None`` when the leg is unknown."""
        seg_key = str(segment_id)
        seg = (getattr(self.omrat, 'segment_data', None) or {}).get(seg_key)
        sp = ep = None
        width = 5000.0
        if isinstance(seg, dict):
            sp = self._parse_wkt_xy(seg.get('Start_Point'))
            ep = self._parse_wkt_xy(seg.get('End_Point'))
            try:
                width = float(seg.get('Width', 5000) or 5000)
            except (TypeError, ValueError):
                width = 5000.0
        if sp is None or ep is None:
            row = self._tangent_row_for_segment(segment_id)
            if row is None:
                return None
            table = self.omrat.main_widget.twRouteList
            sp = self._parse_wkt_xy(table.item(row, 3).text() if table.item(row, 3) else None)
            ep = self._parse_wkt_xy(table.item(row, 4).text() if table.item(row, 4) else None)
            try:
                width = float(table.item(row, 5).text())
            except (AttributeError, TypeError, ValueError):
                width = 5000.0
        if sp is None or ep is None:
            return None
        return QgsPointXY(sp[0], sp[1]), QgsPointXY(ep[0], ep[1]), width

    def _tangent_row_for_segment(self, segment_id: int) -> int | None:
        widget = self.omrat.main_widget
        if widget is None:
            return None
        table = widget.twRouteList
        wanted = str(segment_id).strip()
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is None:
                continue
            text = item.text().strip()
            if text == wanted:
                return row
            try:
                if int(text) == int(wanted):
                    return row
            except ValueError:
                continue
        return None

    def tangent_fraction_from_geometry(self, segment_id: int, geom: QgsGeometry) -> float | None:
        """Project the midpoint of a (dragged) tangent geometry onto the
        leg and return the clamped fraction along it."""
        ends = self._leg_endpoints(segment_id)
        if ends is None or geom is None or geom.isEmpty():
            return None
        start, end, _width = ends
        pts = geom.asPolyline()
        if not pts or len(pts) < 2:
            return None
        first, last = QgsPointXY(pts[0]), QgsPointXY(pts[-1])
        drag_mid = QgsPointXY((first.x() + last.x()) / 2.0, (first.y() + last.y()) / 2.0)
        _mid, to_utm, _to_canvas = self.calculate_midpoint_utm(start, end)
        s = to_utm.transform(start)
        e = to_utm.transform(end)
        m = to_utm.transform(drag_mid)
        return project_fraction((s.x(), s.y()), (e.x(), e.y()), (m.x(), m.y()))

    def _on_tangent_geometry_changed(self, fid: int, geom: QgsGeometry) -> None:
        """User moved a tangent feature with a QGIS editing tool.

        Only the along-track component of the move is kept: the new
        fraction is stored and the line is redrawn perpendicular
        through that point on the leg.  The snap-back is deferred to
        the next event-loop pass so the map tool has finished its own
        edit before we roll the buffer back.
        """
        if self._tangent_guard or self.tangent_layer is None:
            return
        try:
            feat = self.tangent_layer.getFeature(fid)
            segment_id = self._segment_id_from_tangent_type(feat['type'])
        except Exception:  # nosec B110 B112
            return
        if segment_id is None:
            return
        t = self.tangent_fraction_from_geometry(segment_id, geom)
        if t is None:
            return
        if getattr(self.omrat, 'testing', False):
            self._apply_tangent_drag(segment_id, t)
        else:
            QTimer.singleShot(0, lambda: self._apply_tangent_drag(segment_id, t))

    def _apply_tangent_drag(self, segment_id: int, tangent_pos: float) -> None:
        """Discard the user's raw edit and redraw the tangent at ``tangent_pos``."""
        self._tangent_guard = True
        try:
            layer = self.tangent_layer
            if layer is not None:
                try:
                    if layer.isEditable():
                        layer.rollBack()
                except RuntimeError:
                    pass
            self.set_tangent_pos(segment_id, tangent_pos, notify=True)
        finally:
            self._tangent_guard = False

    def set_tangent_pos(
        self, segment_id: int, tangent_pos: float, *, redraw: bool = True, notify: bool = False,
    ) -> float:
        """Store ``tangent_pos`` for a leg, mirror it into the route table
        and (optionally) redraw the tangent.  Returns the clamped value."""
        t = normalize_tangent_pos(tangent_pos)
        seg_key = str(segment_id)
        seg = (getattr(self.omrat, 'segment_data', None) or {}).get(seg_key)
        if isinstance(seg, dict):
            if seg.get(TANGENT_POS_KEY) != t:
                self._invalidate_junction_ais([seg_key])
            seg[TANGENT_POS_KEY] = t

        row = self._tangent_row_for_segment(segment_id)
        if row is not None:
            table = self.omrat.main_widget.twRouteList
            text = percent_from_fraction(t)
            self._table_sync_guard = True
            try:
                cell = table.item(row, 6)
                if cell is None:
                    table.setItem(row, 6, QTableWidgetItem(text))
                elif cell.text() != text:
                    cell.setText(text)
            finally:
                self._table_sync_guard = False

        if redraw:
            ends = self._leg_endpoints(segment_id)
            if ends is not None:
                start, end, width = ends
                self._tangent_guard = True
                try:
                    self.create_offset_lines(start, end, width / 2, int(segment_id), tangent_pos=t)
                finally:
                    self._tangent_guard = False

        if notify:
            self._notify_tangent_moved(segment_id, t)
        return t

    def _notify_tangent_moved(self, segment_id: int, tangent_pos: float) -> None:
        self._notify(
            self.omrat.tr(
                "Tangent line for leg {leg} moved to {pct} % of the leg. "
                "Press 'Update AIS' on that leg to resample the traffic."
            ).format(leg=segment_id, pct=percent_from_fraction(tangent_pos)),
            duration=8,
        )

    def on_width_changed(self, item: QTableWidgetItem):
        """Handle edits to the route table: width (col 5), Leg_name (col 2)
        and tangent position in percent (col 6)."""
        if self._table_sync_guard:
            return
        column = item.column()
        row = item.row()
        seg_id_item = self.omrat.main_widget.twRouteList.item(row, 0)
        if seg_id_item is None:
            return
        try:
            segment_id = int(seg_id_item.text())
        except ValueError:
            return

        if column == 5:  # Width
            start_point_wkt = self.omrat.main_widget.twRouteList.item(row, 3).text()
            end_point_wkt = self.omrat.main_widget.twRouteList.item(row, 4).text()
            width = float(self.omrat.main_widget.twRouteList.item(row, 5).text())
            seg_key = str(segment_id)
            if seg_key in self.omrat.segment_data:
                self.omrat.segment_data[seg_key]['Width'] = int(width)
            self._invalidate_junction_ais([seg_key])
            start_point_geom = QgsGeometry.fromWkt(f"Point ({start_point_wkt})")
            end_point_geom = QgsGeometry.fromWkt(f"Point ({end_point_wkt})")
            if not start_point_geom.isEmpty() and not end_point_geom.isEmpty():
                start_point: QgsPointXY = start_point_geom.asPoint()
                end_point: QgsPointXY = end_point_geom.asPoint()
                self.create_offset_lines(start_point, end_point, width / 2, segment_id)

        elif column == 8:  # AIS lock checkbox
            locked = item.checkState() == Qt.CheckState.Checked
            if locked != is_locked(self.omrat.segment_data, str(segment_id)):
                src = self._apply_lock(segment_id, locked)
                msg = self.omrat.tr("Leg {leg} is now {state} for AIS updates.").format(
                    leg=segment_id, state=self.omrat.tr("locked") if locked else self.omrat.tr("unlocked"),
                )
                if src is not None:
                    msg += " " + self.omrat.tr(
                        "Its link to leg {src} was removed; the next Update AIS replaces the copied traffic."
                    ).format(src=src)
                self._notify(
                    msg,
                    duration=8 if src is not None else 5,
                )

        elif column == 6:  # Tangent position (%)
            t = fraction_from_percent(item.text())
            if t is None:
                # Not a number: put the stored value back.
                self._table_sync_guard = True
                try:
                    item.setText(percent_from_fraction(self.stored_tangent_pos(segment_id)))
                finally:
                    self._table_sync_guard = False
                return
            self.set_tangent_pos(segment_id, t)

        elif column == 2:  # Leg_name
            new_name = item.text()
            seg_key = str(segment_id)
            if seg_key in self.omrat.segment_data:
                self.omrat.segment_data[seg_key]['Leg_name'] = new_name
            layer = self._find_layer_for_seg_id(segment_id)
            if layer is not None:
                layer.setName(new_name)
                label_idx = layer.fields().lookupField("label")
                if label_idx >= 0:
                    pr = layer.dataProvider()
                    for feat in layer.getFeatures():
                        if feat["segmentId"] == segment_id:
                            pr.changeAttributeValues({feat.id(): {label_idx: new_name}})
                            break
                layer.triggerRepaint()

    def point4326_from_wkt(self, wkt: str) -> QgsPoint:
        """Converts a WKT string to a QgsGeometry in EPSG:4326."""
        q_point_base: QgsGeometry = QgsGeometry.fromWkt(wkt)
        assert isinstance(q_point_base, QgsGeometry)  # nosec B101
        pointXY = q_point_base.asPoint()
        q_point = QgsPoint(pointXY.x(), pointXY.y())
        crs = self.omrat.iface.mapCanvas().mapSettings().destinationCrs().authid()
        tr = QgsCoordinateTransform(QgsCoordinateReferenceSystem(crs),
                                    QgsCoordinateReferenceSystem("EPSG:4326"),
                                    QgsProject.instance())
        q_point.transform(tr)
        return q_point

    def format_wkt(self, point: QgsPoint):
        """Formats a point as a WKT string with six decimal places."""
        return f'{point.x():.6f} {point.y():.6f}'

    @staticmethod
    def _parse_wkt_xy(text: str | None) -> tuple[float, float] | None:
        """Parse the OMRAT segment-table point format ``"lon lat"``.

        Tolerant of leading/trailing whitespace, of comma-separated forms,
        and of the full WKT shapes ``"Point (lon lat)"`` /
        ``"POINT(lon lat)"`` that ``update_segment_data`` writes for
        freshly-drawn legs via ``QgsPoint.asWkt()``.  Returns ``None`` for
        missing / malformed input.
        """
        if not isinstance(text, str):
            return None
        s = text.strip()
        if '(' in s and ')' in s:
            s = s.split('(', 1)[1].split(')', 1)[0]
        parts = s.replace(',', ' ').split()
        if len(parts) < 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Waypoints: legs reference shared nodes; dragging a node moves them all
    # ------------------------------------------------------------------

    #: Canvas snapping radius for a click or a dragged vertex.
    SNAP_PIXELS = 12
    #: Headless fallback (tests / no canvas) in metres.
    SNAP_FALLBACK_M = 10.0

    def _waypoints(self) -> dict:
        wps = getattr(self.omrat, 'waypoints', None)
        if not isinstance(wps, dict):
            wps = {}
            self.omrat.waypoints = wps
        return wps

    def snap_to_waypoint(
        self, xy: tuple[float, float], exclude: tuple[str, ...] = (),
    ) -> tuple[tuple[float, float], str | None]:
        """Nearest existing node within ``SNAP_PIXELS`` of ``xy`` on the
        canvas (``SNAP_FALLBACK_M`` when there is no usable canvas or
        the plugin runs headless).  Returns ``(coordinate, node id)``;
        the coordinate is the node's when snapped, ``xy`` otherwise."""
        wps = self._waypoints()
        if not wps:
            return xy, None
        skip = {str(e) for e in exclude}
        best: tuple[float, str] | None = None
        canvas = self.omrat.iface.mapCanvas() if self.omrat.iface else None
        if canvas is not None and not getattr(self.omrat, 'testing', False):
            try:
                m2p = canvas.getCoordinateTransform()
                px = m2p.transform(self._to_canvas_crs(QgsPointXY(xy[0], xy[1]), canvas))
                for wid, (wx, wy) in wps.items():
                    if str(wid) in skip:
                        continue
                    q = m2p.transform(self._to_canvas_crs(QgsPointXY(wx, wy), canvas))
                    d = math.hypot(px.x() - q.x(), px.y() - q.y())
                    if d <= self.SNAP_PIXELS and (best is None or d < best[0]):
                        best = (d, str(wid))
            except Exception:  # nosec B110 B112
                best = None
        if best is None:
            hit = find_waypoint_near(wps, xy, self.SNAP_FALLBACK_M, exclude=skip)
            if hit is not None:
                best = (hit[1], hit[0])
        if best is None:
            return xy, None
        wxy = wps[best[1]]
        return (float(wxy[0]), float(wxy[1])), best[1]

    def _apply_vertex_move(
        self,
        moved_fid: int,
        old_start: tuple[float, float] | None,
        old_end: tuple[float, float] | None,
        new_start: tuple[float, float],
        new_end: tuple[float, float],
    ) -> None:
        """A vertex of leg ``moved_fid`` was dragged: move its *node*, so
        every leg sharing that node follows (model, table, tangent and
        canvas line).  A node dragged onto another node is merged into it.

        Replaces the pre-v0.15.2 coordinate matching, which could only
        find siblings whose text coordinates happened to be identical.
        The re-entrancy flag ignores the ``geometryChanged`` signals our
        own canvas rewrites raise.
        """
        if getattr(self, '_propagating_vertex_move', False):
            return
        sd = self.omrat.segment_data
        seg = sd.get(str(moved_fid))
        if not isinstance(seg, dict):
            return
        wps = self._waypoints()
        touched: set[str] = set()
        self._propagating_vertex_move = True
        try:
            for ref, old_xy, new_xy in (
                (WP_START, old_start, new_start),
                (WP_END, old_end, new_end),
            ):
                if old_xy is None or points_equal(old_xy, new_xy):
                    continue
                wid = seg.get(ref)
                if wid is None or str(wid) not in wps or not points_equal(wps[str(wid)], old_xy):
                    # Refs out of step with the coordinates (leg made by a
                    # path that did not register nodes): give this end
                    # its own node and carry on.
                    wid = find_waypoint_at(wps, old_xy)
                    if wid is None:
                        wid = ensure_waypoint(wps, new_xy)
                        seg[ref] = wid
                        continue
                    seg[ref] = wid
                wid = str(wid)
                _snapped, target = self.snap_to_waypoint(new_xy, exclude=(wid,))
                if target is not None:
                    affected = merge_waypoints(wps, sd, keep_id=target, drop_id=wid)
                else:
                    affected = move_waypoint(wps, sd, wid, new_xy)
                for leg_id in affected:
                    touched.add(str(leg_id))
                    self._refresh_leg_views(int(leg_id))
        finally:
            self._propagating_vertex_move = False
        self._invalidate_junction_ais(touched)

    def _invalidate_junction_ais(self, leg_ids) -> None:
        """A leg's passage line changed: drop the AIS junction counts it fed.

        Junctions touching ``leg_ids`` fall back to their geometric
        matrix, so the next **Update AIS** on any leg re-counts them
        instead of skipping the junction pass (``AIS.junction_pass_needed``).
        """
        # A moved / resized leg moves its traffic-link arrows too.
        _refresh_link_views(self, 'refresh_traffic_links')
        handler = getattr(self.omrat, 'junctions', None)
        if handler is None or not leg_ids:
            return
        try:
            handler.invalidate_legs(leg_ids, self.omrat.segment_data)
        except Exception:  # nosec B110 B112
            pass

    def _refresh_leg_views(self, fid: int) -> None:
        """Bring the route-table row, the tangent and the canvas line of
        leg ``fid`` in line with ``segment_data``."""
        seg = self.omrat.segment_data.get(str(fid))
        if not isinstance(seg, dict):
            return
        sp = self._parse_wkt_xy(seg.get('Start_Point'))
        ep = self._parse_wkt_xy(seg.get('End_Point'))
        if sp is None or ep is None:
            return
        widget = self.omrat.main_widget
        width = 0.0
        if widget is not None:
            row = self._tangent_row_for_segment(fid)
            if row is not None:
                try:
                    widget.twRouteList.item(row, 3).setText(seg['Start_Point'])
                    widget.twRouteList.item(row, 4).setText(seg['End_Point'])
                except Exception:  # nosec B110 B112
                    pass
                try:
                    width = float(widget.twRouteList.item(row, 5).text())
                except (AttributeError, TypeError, ValueError):
                    width = 0.0
        if not width:
            try:
                width = float(seg.get('Width', 0) or 0)
            except (TypeError, ValueError):
                width = 0.0
        start_xy = QgsPointXY(sp[0], sp[1])
        end_xy = QgsPointXY(ep[0], ep[1])
        try:
            self.create_offset_lines(start_xy, end_xy, width / 2 if width else 0.0, fid)
        except Exception:  # nosec B110 B112
            pass
        self._rewrite_leg_geometry(fid, start_xy, end_xy)

    def _rewrite_leg_geometry(self, fid: int, start_xy: QgsPointXY, end_xy: QgsPointXY) -> None:
        """Push new endpoints into leg ``fid``'s canvas feature.  The
        caller holds ``_propagating_vertex_move`` so the ``geometryChanged``
        this raises is not propagated again."""
        layer = self._find_layer_for_seg_id(fid)
        if layer is None:
            return
        try:
            feat = next(layer.getFeatures())
        except StopIteration:
            return
        current = feat.geometry().asPolyline() if feat.hasGeometry() else []
        if (
            len(current) >= 2
            and current[0].distance(start_xy) < 1e-9
            and current[-1].distance(end_xy) < 1e-9
        ):
            return
        geom = QgsGeometry.fromPolylineXY([start_xy, end_xy])
        try:
            if layer.isEditable():
                layer.changeGeometry(feat.id(), geom)
            else:
                prov = layer.dataProvider()
                if prov is not None:
                    prov.changeGeometryValues({feat.id(): geom})
            layer.updateExtents()
            layer.triggerRepaint()
        except Exception:  # nosec B110 B112
            pass

    def on_geometry_changed_wrapper(self, segment_id: int, fid: int, geom: QgsGeometry):
        """Wrapper for the geometryChanged signal to pass the segment ID."""
        self.on_geometry_changed(segment_id, geom)
