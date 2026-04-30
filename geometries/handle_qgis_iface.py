from functools import partial
from typing import TYPE_CHECKING, Any, cast

from qgis._core import QgsFeatureRenderer, QgsVectorDataProvider, QgsVectorLayerEditBuffer
if TYPE_CHECKING:
    from omrat import OMRAT

from omrat_widget import OMRATMainWidget
from qgis.core import (
    QgsVectorLayer, QgsFeature, QgsGeometry, QgsLineString, QgsPoint, QgsProject,
    QgsField, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsFields,
    QgsPalLayerSettings, QgsVectorLayerSimpleLabeling, QgsSingleSymbolRenderer,
    QgsLineSymbol, QgsPointXY, QgsSymbol
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QTableWidgetItem, QPushButton
from qgis.gui import QgsMapToolPan


from omrat_utils import PointTool



def is_valid_point_pair(start: QgsPointXY, end: QgsPointXY) -> bool:
    return not (
        (-1 <= start.x() <= 1 and -1 <= start.y() <= 1) or
        (-1 <= end.x() <= 1 and -1 <= end.y() <= 1)
    )

def calculate_tangent_line(mid: QgsPointXY, start: QgsPointXY, end: QgsPointXY, offset: float) -> tuple[QgsPointXY, QgsPointXY]:
    dx = end.x() - start.x()
    dy = end.y() - start.y()
    length = (dx**2 + dy**2)**0.5
    unit_dx = dx / length
    unit_dy = dy / length
    perp_dx = -unit_dy
    perp_dy = unit_dx

    start_tangent = QgsPointXY(mid.x() - perp_dx * offset, mid.y() - perp_dy * offset)
    end_tangent = QgsPointXY(mid.x() + perp_dx * offset, mid.y() + perp_dy * offset)
    return start_tangent, end_tangent

class HandleQGISIface:
    def __init__(self, omrat: "OMRAT"):
        """Initialize the HandleQGISIface class."""
        self.omrat = omrat
        self.tangent_layer: QgsVectorLayer | None = None
        self.vector_layers: list[QgsVectorLayer] = []
        self.current_start_point: QgsPointXY | None = None
        self.segment_id = 0
        self.cur_route_id = 1
        self.item_changed_connected = False
        self.buffer_edits: list[QgsVectorLayerEditBuffer] = []
        self.leg_dirs: dict[str, list[str]] = {}
        self.omrat.main_widget.twRouteList.cellClicked.connect(self.on_route_table_cell_clicked)

    def add_new_route(self):
        """Starts the editing for a new route."""
        self.current_start_point = None
        self.point_layer = None
        self.mapTool = PointTool(self.omrat.iface.mapCanvas())
        canvas = self.omrat.iface.mapCanvas()
        if canvas is not None:
            canvas.setMapTool(self.mapTool)
        self.mapTool.canvasClicked.connect(self.onMapClick)
        self.omrat.main_widget.pbStopRoute.setEnabled(True)

    def onMapClick(self, point: QgsPoint):
        q_point: QgsPoint = self.point4326_from_wkt(point.asWkt())
        if self.current_start_point is None:
            self.create_point(q_point)
        else:
            self.create_line(q_point)

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
        # Create the memory layer
        vl = QgsVectorLayer("LineString?crs=EPSG:4326", "Segment", "memory")
        if not vl.isValid():
            print("Error: Line layer is not valid")
            return

        # Increment segment ID
        self.segment_id += 1

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
                f"LEG_{self.segment_id}_{self.cur_route_id}"  # Label value
            ])
            # Style the layer
            self.style_layer(vl)
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
            self.vector_layers.append(vl)

            # Ensure the layer is in editing mode before connecting the signal
            if not vl.isEditable():
                vl.startEditing()

            edit_buffer: QgsVectorLayerEditBuffer | None = vl.editBuffer()
            if edit_buffer is None:
                print("Error: editBuffer is None")
                return
            
            edit_buffer.geometryChanged.connect(partial(self.on_geometry_changed_wrapper, self.segment_id))
            self.buffer_edits.append(edit_buffer)
            self.current_start_point = QgsPointXY(point.x(), point.y())
            self.point_layer = None
            self.save_route(QgsPoint(start_point.x(), start_point.y()), end_point)
            
            vl.setCustomProperty("segment_id", self.segment_id)
    
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
        for obj in self.buffer_edits:
            try:
                obj.geometryChanged.disconnect()
            except:
                pass
        self.buffer_edits = []
        for layer in self.vector_layers:
            try:
                edit_buffer = layer.editBuffer()
                if edit_buffer is not None:
                    edit_buffer.geometryChanged.disconnect()
                    print(f"Disconnected geometryChanged signal for layer {layer.name()}")
            except TypeError:
                print(f"No connection for geometryChanged signal in layer {layer.name()}")
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
        # Break circular references
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

        # Disconnect geometry-changed signals from edit buffers
        for obj in self.buffer_edits:
            try:
                obj.geometryChanged.disconnect()
            except Exception:
                pass
        self.buffer_edits = []

        # Remove vector layers from QGIS project
        for layer in self.vector_layers:
            try:
                edit_buffer = layer.editBuffer()
                if edit_buffer is not None:
                    try:
                        edit_buffer.geometryChanged.disconnect()
                    except TypeError:
                        pass
                QgsProject.instance().removeMapLayer(layer.id())
            except Exception:
                pass
        self.vector_layers = []

        # Remove tangent layer
        if self.tangent_layer is not None:
            try:
                QgsProject.instance().removeMapLayer(self.tangent_layer.id())
            except Exception:
                pass
            self.tangent_layer = None

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
        self.leg_dirs = {}

    def on_geometry_changed(self, fid:int, geom:QgsGeometry):
        """Handle geometry changes for a feature."""
        # Get the segment ID from the feature's attributes
        polyline: list[QgsGeometry] = geom.asPolyline()
        assert isinstance(polyline, list)
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

        assert isinstance(start_point, QgsPoint)
        assert isinstance(end_point, QgsPoint)
        # Update the start and end points in the table
        assert(self.omrat.main_widget is not None)

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

                # Propagate the move to any other leg that shared the
                # endpoint that just moved (so curved routes stay
                # connected when the user drags a junction vertex).
                self._propagate_shared_vertex_move(
                    moved_fid=fid,
                    old_start=old_start,
                    old_end=old_end,
                    new_start=(start_pointXY.x(), start_pointXY.y()),
                    new_end=(end_pointXY.x(), end_pointXY.y()),
                )

                # Stop processing once the correct row is updated
                return

    @staticmethod
    def label_layer(layer: QgsVectorLayer) -> None:
        """Label the layer with the 'label' field."""
        settings = QgsPalLayerSettings()
        settings.fieldName = "label"  # Use the 'label' field for labeling
        settings.placement = QgsPalLayerSettings.Line
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
        assert (isinstance(symbol, QgsLineSymbol))
        symbol.setWidth(1.5)  # Set line thickness
        symbol.setColor(QColor("blue"))  # Optional: Set line color

        # Trigger a refresh of the layer's symbology
        layer.triggerRepaint()

    def save_route(self, point1: QgsPoint, point2: QgsPoint):
        """Save route information to the twRouteList table."""
        assert(self.omrat.main_widget is not None)
        row_id = self.omrat.main_widget.twRouteList.rowCount()
        self.omrat.main_widget.twRouteList.setRowCount(row_id + 1)

        # Create table items
        item1 = QTableWidgetItem(f'{self.segment_id}')
        item2 = QTableWidgetItem(f'{self.cur_route_id}')
        item3 = QTableWidgetItem(f'{point1.asWkt(precision=5).split("(")[1].split(")")[0]}')
        item4 = QTableWidgetItem(f'{point2.asWkt(precision=5).split("(")[1].split(")")[0]}')
        item5 = QTableWidgetItem(f'5000')  # Default width
        item6 = QTableWidgetItem(f'LEG_{self.segment_id}_{self.cur_route_id}')  # Leg name

        # Add items to the table
        self.omrat.main_widget.twRouteList.setItem(row_id, 0, item1)
        self.omrat.main_widget.twRouteList.setItem(row_id, 1, item2)
        self.omrat.main_widget.twRouteList.setItem(row_id, 2, item6)
        self.omrat.main_widget.twRouteList.setItem(row_id, 3, item3)
        self.omrat.main_widget.twRouteList.setItem(row_id, 4, item4)
        self.omrat.main_widget.twRouteList.setItem(row_id, 5, item5)
        btn_update_ais = QPushButton("Update AIS")
        btn_update_ais.clicked.connect(lambda: self.omrat.ais.update_legs(str(self.segment_id)))
        self.omrat.main_widget.twRouteList.setCellWidget(row_id, 6, btn_update_ais)

        # Connect the itemChanged signal to a handler
        if not self.item_changed_connected:
            self.omrat.main_widget.twRouteList.itemChanged.connect(self.on_width_changed)
            self.item_changed_connected = True
            
    def on_route_table_cell_clicked(self, row: int, column: int):
        """Called when any cell in the route table is clicked."""
        segment_id_item = self.omrat.main_widget.twRouteList.item(row, 0)
        if segment_id_item is not None:
            try:
                segment_id = segment_id_item.text()
                self.omrat.distributions.run_update_plot(segment_id)
            except ValueError:
                pass  # Handle or log invalid segment_id if needed

    def update_segment_data(self, point:QgsPoint) -> None:
        main_widget = self.omrat.main_widget
        assert(self.current_start_point is not None)
        pointXY = QgsPointXY(point.x(), point.y())
        degrees:float = (self.current_start_point.azimuth(pointXY) + 360) % 360
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
        utm_crs = QgsCoordinateReferenceSystem(f"EPSG:{32600 + utm_zone if is_northern else 32700 + utm_zone}")
        
        transform_to_utm = QgsCoordinateTransform(QgsCoordinateReferenceSystem("EPSG:4326"), utm_crs, QgsProject.instance())
        start_utm = transform_to_utm.transform(self.current_start_point)
        dist = start_utm.distance(transform_to_utm.transform(pointXY))
        if f'{self.segment_id}' in self.omrat.segment_data:
            self.omrat.segment_data[f'{self.segment_id}']['Start_Point'] = QgsPoint(self.current_start_point.x(),
                                                                                    self.current_start_point.y()).asWkt()
            self.omrat.segment_data[f'{self.segment_id}']['End_Point'] = point.asWkt()
            self.omrat.segment_data[f'{self.segment_id}']['Dirs'] = dirs
            self.omrat.segment_data[f'{self.segment_id}']['line_length'] = dist
        else:
            self.omrat.segment_data[f'{self.segment_id}'] = {'Start_Point': QgsPoint(self.current_start_point.x(),
                                                                                     self.current_start_point.y()).asWkt(),
                                                'End_Point': point.asWkt(),
                                                'Dirs': dirs, 'Width': 5000, 'line_length': dist,
                                                'Route_Id': self.cur_route_id, 
                                                'Leg_name': f'LEG_{self.segment_id}_{self.cur_route_id}'}
        self.leg_dirs[f'{self.segment_id}'] = dirs
        main_widget.cbTrafficSelectSeg.addItem(f'{self.segment_id}')
        self.omrat.traffic.c_seg = f'{self.segment_id}'
        self.current_start_point = None
        
    def ensure_tangent_layer(self):
        if self.tangent_layer is None:
            self.tangent_layer = QgsVectorLayer("LineString?crs=EPSG:4326", "Tangent Line", "memory")
            if not self.tangent_layer.isValid():
                raise RuntimeError("Tangent line layer is not valid")
            QgsProject.instance().addMapLayer(self.tangent_layer)
            self.vector_layers.append(self.tangent_layer)
        self.tangent_layer.startEditing()


    def ensure_tangent_fields(self):
        if self.tangent_layer.fields().lookupField("type") < 0:
            pr = self.tangent_layer.dataProvider()
            pr.addAttributes([QgsField("type", QMetaType.Type.QString)])
            self.tangent_layer.updateFields()

    def calculate_midpoint_utm(self, start: QgsPointXY, end: QgsPointXY) -> tuple[QgsPointXY, QgsCoordinateTransform, QgsCoordinateTransform]:
        longitude = (start.x() + end.x()) / 2
        utm_zone = int((longitude + 180) / 6) + 1
        is_northern = start.y() >= 0
        utm_crs = QgsCoordinateReferenceSystem(f"EPSG:{32600 + utm_zone if is_northern else 32700 + utm_zone}")
        
        transform_to_utm = QgsCoordinateTransform(QgsCoordinateReferenceSystem("EPSG:4326"), utm_crs, QgsProject.instance())
        transform_to_canvas = QgsCoordinateTransform(utm_crs, QgsCoordinateReferenceSystem("EPSG:4326"), QgsProject.instance())

        start_utm = transform_to_utm.transform(start)
        end_utm = transform_to_utm.transform(end)
        mid_utm = QgsPointXY((start_utm.x() + end_utm.x()) / 2, (start_utm.y() + end_utm.y()) / 2)

        return mid_utm, transform_to_utm, transform_to_canvas
    
    def remove_existing_tangent(self, segment_id: int):
        ids = [f.id() for f in self.tangent_layer.getFeatures() if f["type"] == f"Tangent Line {segment_id}"]
        if ids:
            self.tangent_layer.deleteFeatures(ids)

    def add_tangent_feature(self, start: QgsPointXY, end: QgsPointXY, segment_id: int):
        self.remove_existing_tangent(segment_id)
        fet = QgsFeature()
        fet.setGeometry(QgsLineString([QgsPoint(start.x(), start.y()), QgsPoint(end.x(), end.y())]))
        fet.setAttributes([f"Tangent Line {segment_id}"])
        self.tangent_layer.dataProvider().addFeatures([fet])

    def create_offset_lines(self, start_point: QgsPointXY, end_point: QgsPointXY, offset_distance: float, segment_id: int):
        if not is_valid_point_pair(start_point, end_point):
            return

        self.ensure_tangent_layer()
        self.ensure_tangent_fields()

        mid_utm, to_utm, to_canvas = self.calculate_midpoint_utm(start_point, end_point)
        start_utm = to_utm.transform(start_point)
        end_utm = to_utm.transform(end_point)

        tangent_start_utm, tangent_end_utm = calculate_tangent_line(mid_utm, start_utm, end_utm, offset_distance)
        tangent_start = to_canvas.transform(tangent_start_utm)
        tangent_end = to_canvas.transform(tangent_end_utm)

        self.add_tangent_feature(tangent_start, tangent_end, segment_id)
        self.tangent_layer.commitChanges()
        self.tangent_layer.triggerRepaint()

        
    def on_width_changed(self, item: QTableWidgetItem):
        """Handle changes to the width in the twRouteList table."""
        column = item.column()
        row = item.row()
        if column == 5:  # Assuming the 'Width' column is at index 5
            segment_id = int(self.omrat.main_widget.twRouteList.item(row, 0).text())
            start_point_wkt = self.omrat.main_widget.twRouteList.item(row, 3).text()
            end_point_wkt = self.omrat.main_widget.twRouteList.item(row, 4).text()
            width = float(self.omrat.main_widget.twRouteList.item(row, 5).text())

            # Convert WKT to QgsGeometry
            start_point_geom = QgsGeometry.fromWkt(f"Point ({start_point_wkt})")
            end_point_geom = QgsGeometry.fromWkt(f"Point ({end_point_wkt})")

            # Extract QgsPoint from QgsGeometry
            if not start_point_geom.isEmpty() and not end_point_geom.isEmpty():
                start_point: QgsPointXY = start_point_geom.asPoint()
                end_point: QgsPointXY = end_point_geom.asPoint()
                # Remove the existing tangent line and redraw it with the updated width
                self.create_offset_lines(start_point, end_point, width / 2, segment_id)

    def point4326_from_wkt(self, wkt:str) -> QgsPoint:
        """Converts a WKT string to a QgsGeometry in EPSG:4326."""
        q_point_base: QgsGeometry = QgsGeometry.fromWkt(wkt)
        assert isinstance(q_point_base, QgsGeometry)
        pointXY = q_point_base.asPoint()
        q_point = QgsPoint(pointXY.x(), pointXY.y())
        crs = self.omrat.iface.mapCanvas().mapSettings().destinationCrs().authid()
        tr = QgsCoordinateTransform(QgsCoordinateReferenceSystem(crs),
                                     QgsCoordinateReferenceSystem("EPSG:4326"),
                                     QgsProject.instance())
        q_point.transform(tr)
        return q_point

    def format_wkt(self, point:QgsPoint):
        """Formats a point as a WKT string with six decimal places."""
        return f'{point.x():.6f} {point.y():.6f}'

    @staticmethod
    def _parse_wkt_xy(text: str | None) -> tuple[float, float] | None:
        """Parse the OMRAT segment-table point format ``"lon lat"``.

        Tolerant of leading/trailing whitespace and of comma-separated
        forms.  Returns ``None`` for missing / malformed input.
        """
        if not isinstance(text, str):
            return None
        parts = text.replace(',', ' ').split()
        if len(parts) < 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None

    def _propagate_shared_vertex_move(
        self,
        *,
        moved_fid: int,
        old_start: tuple[float, float] | None,
        old_end: tuple[float, float] | None,
        new_start: tuple[float, float],
        new_end: tuple[float, float],
    ) -> None:
        """Move sibling legs that shared the endpoint just dragged.

        For every other leg whose start or end point matched ``old_start``
        or ``old_end`` (within :data:`_SHARED_VERTEX_TOL` degrees),
        update the stored endpoint to the matching ``new_*`` and rewrite
        the matching row of ``twRouteList``.

        A re-entrancy flag suppresses the propagation triggered by the
        QGIS ``geometryChanged`` signals fired by our own writes — those
        signals would otherwise cause a chain of moves that drift away
        from the user's original drag.
        """
        if getattr(self, '_propagating_vertex_move', False):
            return

        # Only propagate when the endpoint actually moved.
        moved: list[tuple[tuple[float, float], tuple[float, float]]] = []
        if old_start is not None and not self._xy_close(old_start, new_start):
            moved.append((old_start, new_start))
        if old_end is not None and not self._xy_close(old_end, new_end):
            moved.append((old_end, new_end))
        if not moved:
            return

        self._propagating_vertex_move = True
        try:
            for old_xy, new_xy in moved:
                self._move_matching_endpoints(
                    skip_fid=moved_fid, old_xy=old_xy, new_xy=new_xy,
                )
        finally:
            self._propagating_vertex_move = False

    @staticmethod
    def _xy_close(
        a: tuple[float, float],
        b: tuple[float, float],
        tol: float = 1e-7,
    ) -> bool:
        return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol

    def _move_matching_endpoints(
        self,
        *,
        skip_fid: int,
        old_xy: tuple[float, float],
        new_xy: tuple[float, float],
    ) -> None:
        """For every leg != ``skip_fid`` whose start or end equals
        ``old_xy``, update that endpoint to ``new_xy``.

        Updates ``segment_data`` and the ``twRouteList`` row in step,
        and rewrites the offset / leg layer features so the canvas
        catches up without waiting for the next edit-buffer commit.
        """
        widget = self.omrat.main_widget
        if widget is None:
            return
        tol = 1e-7
        for row in range(widget.twRouteList.rowCount()):
            try:
                fid = int(widget.twRouteList.item(row, 0).text())
            except (AttributeError, ValueError):
                continue
            if fid == skip_fid:
                continue

            seg_key = str(fid)
            seg = self.omrat.segment_data.get(seg_key)
            if seg is None:
                continue

            sp = self._parse_wkt_xy(seg.get('Start_Point'))
            ep = self._parse_wkt_xy(seg.get('End_Point'))
            if sp is None or ep is None:
                continue

            updated = False
            if abs(sp[0] - old_xy[0]) <= tol and abs(sp[1] - old_xy[1]) <= tol:
                sp = new_xy
                updated = True
            if abs(ep[0] - old_xy[0]) <= tol and abs(ep[1] - old_xy[1]) <= tol:
                ep = new_xy
                updated = True
            if not updated:
                continue

            new_start_pt = QgsPoint(sp[0], sp[1])
            new_end_pt = QgsPoint(ep[0], ep[1])
            new_start_xy = QgsPointXY(sp[0], sp[1])
            new_end_xy = QgsPointXY(ep[0], ep[1])

            seg['Start_Point'] = self.format_wkt(new_start_pt)
            seg['End_Point'] = self.format_wkt(new_end_pt)
            try:
                widget.twRouteList.item(row, 3).setText(seg['Start_Point'])
                widget.twRouteList.item(row, 4).setText(seg['End_Point'])
            except Exception:
                pass

            try:
                width = float(widget.twRouteList.item(row, 5).text())
            except (AttributeError, TypeError, ValueError):
                width = 0.0
            try:
                self.create_offset_lines(
                    new_start_xy, new_end_xy, width / 2 if width else 0.0, fid,
                )
            except Exception:
                pass

    def on_geometry_changed_wrapper(self, segment_id:int, fid:int, geom:QgsGeometry):
        """Wrapper for the geometryChanged signal to pass the segment ID."""
        self.on_geometry_changed(segment_id, geom)
        