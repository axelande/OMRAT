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
from qgis.PyQt.QtCore import QMetaType, QVariant
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
        fields.append(QgsField("segmentId", QMetaType.Int))
        fields.append(QgsField("routeId", QMetaType.Int))
        fields.append(QgsField("startPoint", QMetaType.QString))
        fields.append(QgsField("endPoint", QMetaType.QString))
        fields.append(QgsField("label", QMetaType.QString))  # Field for labeling
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
        for row in range(self.omrat.main_widget.twRouteList.rowCount()):
            if int(self.omrat.main_widget.twRouteList.item(row, 0).text()) == fid:
                self.omrat.main_widget.twRouteList.item(row, 3).setText(self.format_wkt(start_point))
                self.omrat.main_widget.twRouteList.item(row, 4).setText(self.format_wkt(end_point))

                # Get the width from the table
                width = float(self.omrat.main_widget.twRouteList.item(row, 5).text())

                # Update the tangent line for this segment
                self.create_offset_lines(start_pointXY, end_pointXY, width / 2, fid)

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
            pr.addAttributes([QgsField("type", QVariant.String)])
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

    def on_geometry_changed_wrapper(self, segment_id:int, fid:int, geom:QgsGeometry):
        """Wrapper for the geometryChanged signal to pass the segment ID."""
        self.on_geometry_changed(segment_id, geom)
        