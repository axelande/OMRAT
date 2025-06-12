from qgis.core import (
    QgsVectorLayer, QgsFeature, QgsGeometry, QgsLineString, QgsPoint, QgsProject,
    QgsField, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsFields,
    QgsPalLayerSettings, QgsVectorLayerSimpleLabeling, QgsSingleSymbolRenderer,
    QgsLineSymbol, QgsPointXY
)
from qgis.PyQt.QtCore import QMetaType, QVariant
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QTableWidgetItem
from qgis.gui import QgsMapToolPan


from omrat_utils import PointTool

class HandleQGISIface:
    def __init__(self, omrat):
        """Initialize the HandleQGISIface class."""
        self.omrat = omrat
        self.tangent_layer = None
        self.vector_layers = []
        self.current_start_point = None
        self.segment_id = 0
        self.cur_route_id = 1
        self.item_changed_connected = False
        self.buffer_edits = []
        self.leg_dirs = {}

    def add_new_route(self):
        """Starts the editing for a new route."""
        self.current_start_point = None
        self.point_layer = None
        self.mapTool = PointTool(self.omrat.iface.mapCanvas())
        self.omrat.iface.mapCanvas().setMapTool(self.mapTool)
        self.mapTool.canvasClicked.connect(self.onMapClick)
        self.omrat.dockwidget.pbStopRoute.setEnabled(True)

    def onMapClick(self, point):
        q_point = self.point4326_from_wkt(point.asWkt())
        if self.current_start_point is None:
            self.create_point(q_point)
        else:
            self.create_line(q_point)

    def create_point(self, point):
        self.point_layer = QgsVectorLayer("Point?crs=EPSG:4326", "StartPoint", "memory")
        prov = self.point_layer.dataProvider()
        QgsProject.instance().addMapLayer(self.point_layer)  # Add the layer to the project

        if not self.point_layer.isEditable():
            self.point_layer.startEditing()  # Start editing

        feat = QgsFeature()
        feat.setGeometry(point)
        prov.addFeatures([feat])
        self.current_start_point = point

    def create_line(self, point):
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
        pr = vl.dataProvider()
        fields = QgsFields()
        fields.append(QgsField("segmentId", QMetaType.Int))
        fields.append(QgsField("routeId", QMetaType.Int))
        fields.append(QgsField("startPoint", QMetaType.QString))
        fields.append(QgsField("endPoint", QMetaType.QString))
        fields.append(QgsField("label", QMetaType.QString))  # Field for labeling
        pr.addAttributes(fields.toList())
        vl.updateFields()

        # Create the feature
        fet = QgsFeature(fields)
        start_point = QgsPoint(self.current_start_point.asPoint())
        end_point = QgsPoint(point.asPoint())
        fet.setGeometry(QgsLineString([start_point, end_point]))
        fet.setAttributes([
            self.segment_id,
            self.cur_route_id,
            self.current_start_point.asPoint().asWkt(),
            point.asPoint().asWkt(),
            f"LEG_{self.segment_id}_{self.cur_route_id}"  # Label value
        ])
        # Style the layer
        self.style_layer(vl)
        fet.setId(self.segment_id)
              
        # Add the feature to the layer
        pr.addFeatures([fet])

        # Label the layer
        self.label_layer(vl)

        # Refresh the layer to ensure changes are applied
        vl.triggerRepaint()

        # Create offset lines
        self.create_offset_lines(start_point, end_point, 2500, self.segment_id)

        # Stop editing and remove the point layer
        if not self.omrat.testing and self.point_layer is not None:
            if self.point_layer.isEditable():
                self.point_layer.commitChanges()  # Save changes
            QgsProject.instance().removeMapLayer(self.point_layer)

        # Update segment data and save the route
        self.update_segment_data(point.asPoint())
        self.vector_layers.append(vl)

        # Ensure the layer is in editing mode before connecting the signal
        if not vl.isEditable():
            vl.startEditing()

        edit_buffer = vl.editBuffer()
        if edit_buffer is None:
            print("Error: editBuffer is None")
            return
        
        edit_buffer.geometryChanged.connect(self.on_geometry_changed_wrapper(self.segment_id))
        self.buffer_edits.append(edit_buffer)
        self.current_start_point = point
        self.point_layer = None
        self.save_route(start_point, end_point)
        
        vl.setCustomProperty("segment_id", self.segment_id)
    
    def unload(self):
        """Remove temporary layers and disconnect signals."""
        # Remove the point layer
        print('unloading_qgis')
        if hasattr(self, 'point_layer') and self.point_layer is not None:
            QgsProject.instance().removeMapLayer(self.point_layer)
            self.point_layer = None
        print(self.vector_layers)
        # Remove vector layers and disconnect geometryChanged signals
        for obj in self.buffer_edits:
            try:
                obj.geometryChanged.disconnect()
            except:
                pass
        self.buffer_edits = []
        if hasattr(self, 'vector_layers'):
            for layer in self.vector_layers:
                try:
                    if layer.editBuffer() is not None:
                        layer.editBuffer().geometryChanged.disconnect()
                        print(f"Disconnected geometryChanged signal for layer {layer.name()}")
                except TypeError:
                    print(f"No connection for geometryChanged signal in layer {layer.name()}")
                QgsProject.instance().removeMapLayer(layer.id())  # Remove the layer from QGIS
            self.vector_layers = []

        # Disconnect itemChanged signal from twRouteList
        try:
            if hasattr(self.omrat, "dockwidget") and self.omrat.dockwidget is not None:
                self.omrat.dockwidget.twRouteList.itemChanged.disconnect()
        except TypeError:
            pass

        # Disconnect custom signals
        if hasattr(self, "mapTool"):
            if hasattr(self.mapTool, 'canvasClicked'):
                try:
                    self.mapTool.canvasClicked.disconnect()
                except TypeError:
                    pass

        # Break circular references
        self.tangent_layer = None
        self.omrat = None
        self.mapTool = None
        self.vector_layers = None
        self.current_start_point = None
        self.leg_dirs = None

    def on_geometry_changed(self, fid, geom):
        """Handle geometry changes for a feature."""

        # Get the segment ID from the feature's attributes
        start_point = geom.asPolyline()[0]
        end_point = geom.asPolyline()[-1]
        
        # Update the start and end points in the table
        for row in range(self.omrat.dockwidget.twRouteList.rowCount()):
            if int(self.omrat.dockwidget.twRouteList.item(row, 0).text()) == fid:
                self.omrat.dockwidget.twRouteList.item(row, 2).setText(self.format_wkt(start_point))
                self.omrat.dockwidget.twRouteList.item(row, 3).setText(self.format_wkt(end_point))

                # Get the width from the table
                width = float(self.omrat.dockwidget.twRouteList.item(row, 4).text())

                # Update the tangent line for this segment
                self.create_offset_lines(start_point, end_point, width / 2, fid)

                # Stop processing once the correct row is updated
                return

    @staticmethod
    def label_layer(layer):
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
    def style_layer(layer):
        """Style the layer with a thicker line."""
        # Get the layer's renderer and symbol
        renderer = layer.renderer()
        if (renderer is None):
            renderer = QgsSingleSymbolRenderer(QgsLineSymbol())
            layer.setRenderer(renderer)

        symbol = renderer.symbol()
        symbol.setWidth(1.5)  # Set line thickness
        symbol.setColor(QColor("blue"))  # Optional: Set line color

        # Trigger a refresh of the layer's symbology
        layer.triggerRepaint()

    def save_route(self, point1, point2):
        """Save route information to the twRouteList table."""
        row_id = self.omrat.dockwidget.twRouteList.rowCount()
        self.omrat.dockwidget.twRouteList.setRowCount(row_id + 1)

        # Create table items
        item1 = QTableWidgetItem(f'{self.segment_id}')
        item2 = QTableWidgetItem(f'{self.cur_route_id}')
        item3 = QTableWidgetItem(f'{point1.asWkt(precision=5).split("(")[1].split(")")[0]}')
        item4 = QTableWidgetItem(f'{point2.asWkt(precision=5).split("(")[1].split(")")[0]}')
        item5 = QTableWidgetItem(f'5000')  # Default width

        # Add items to the table
        self.omrat.dockwidget.twRouteList.setItem(row_id, 0, item1)
        self.omrat.dockwidget.twRouteList.setItem(row_id, 1, item2)
        self.omrat.dockwidget.twRouteList.setItem(row_id, 2, item3)
        self.omrat.dockwidget.twRouteList.setItem(row_id, 3, item4)
        self.omrat.dockwidget.twRouteList.setItem(row_id, 4, item5)

        # Connect the itemChanged signal to a handler
        if not self.item_changed_connected:
            self.omrat.dockwidget.twRouteList.itemChanged.connect(self.on_width_changed)
            self.item_changed_connected = True

    def update_segment_data(self, point):
        degrees = (self.current_start_point.asPoint().azimuth(point) + 360) % 360
        if degrees > 315 or degrees <= 45:
            self.omrat.dockwidget.laDir1.setText('North going')
            self.omrat.dockwidget.laDir2.setText('South going')
            dirs = ['North going', 'South going']
        if degrees > 45 and degrees <= 135:
            self.omrat.dockwidget.laDir1.setText('East going')
            self.omrat.dockwidget.laDir2.setText('West going')
            dirs = ['East going', 'West going']
        if degrees > 135 and degrees <= 225:
            self.omrat.dockwidget.laDir1.setText('South going')
            self.omrat.dockwidget.laDir2.setText('North going')
            dirs = ['South going', 'North going']
        if degrees > 225 and degrees <= 315:
            self.omrat.dockwidget.laDir1.setText('West going')
            self.omrat.dockwidget.laDir2.setText('East going')
            dirs = ['West going', 'East going']
        if f'{self.segment_id}' in self.omrat.segment_data:
            self.omrat.segment_data[f'{self.segment_id}']['Start Point'] = self.current_start_point.asPoint().asWkt()
            self.omrat.segment_data[f'{self.segment_id}']['End Point'] = point.asWkt()
            self.omrat.segment_data[f'{self.segment_id}']['Dirs'] = dirs
        else:
            self.omrat.segment_data[f'{self.segment_id}'] = {'Start Point': self.current_start_point.asPoint().asWkt(), 
                                                  'End Point': point.asWkt(),
                                                  'Dirs': dirs, 'Width': 5000}
        self.leg_dirs[f'{self.segment_id}'] = dirs
        self.omrat.dockwidget.cbTrafficSelectSeg.addItem(f'{self.segment_id}')
        self.omrat.traffic.c_seg = f'{self.segment_id}'
        self.current_start_point = None

    def create_offset_lines(self, start_point, end_point, offset_distance, segment_id:int):
        """Create a single offset line across the center of the original line."""
        # Check if a tangent layer already exists
        if (-1 <= start_point.x() <= 1 and -1 <= start_point.y() <= 1 or
        -1 <= end_point.x() <= 1 and -1 <= end_point.y() <= 1):
            return
        
        # If no tangent layer exists, create a new one
        if self.tangent_layer is None:
            self.tangent_layer = QgsVectorLayer("LineString?crs=EPSG:4326", "Tangent Line", "memory")
            if not self.tangent_layer.isValid():
                print("Error: Tangent line layer is not valid")
                return
            QgsProject.instance().addMapLayer(self.tangent_layer)
            self.vector_layers.append(self.tangent_layer)

        self.tangent_layer.startEditing()

        # Add fields to the tangent layer if they don't already exist
        if self.tangent_layer.fields().lookupField("type") < 0:
            pr = self.tangent_layer.dataProvider()
            pr.addAttributes([QgsField("type", QVariant.String)])
            self.tangent_layer.updateFields()

        # Calculate the midpoint of the original line in the UTM CRS
        canvas_crs = self.omrat.iface.mapCanvas().mapSettings().destinationCrs()
        longitude = (start_point.x() + end_point.x()) / 2  # Approximate central longitude
        utm_zone = int((longitude + 180) / 6) + 1
        is_northern = start_point.y() >= 0  # Determine if the point is in the northern hemisphere
        utm_crs = QgsCoordinateReferenceSystem(f"EPSG:{32600 + utm_zone if is_northern else 32700 + utm_zone}")
        
        transform_to_utm = QgsCoordinateTransform(QgsCoordinateReferenceSystem("EPSG:4326"), utm_crs, QgsProject.instance())
        transform_to_canvas = QgsCoordinateTransform(utm_crs, QgsCoordinateReferenceSystem("EPSG:4326"), QgsProject.instance())

        start_point_utm = transform_to_utm.transform(QgsPointXY(start_point))
        end_point_utm = transform_to_utm.transform(QgsPointXY(end_point))

        mid_point_utm = QgsPointXY(
            (start_point_utm.x() + end_point_utm.x()) / 2,
            (start_point_utm.y() + end_point_utm.y()) / 2
        )

        dx = end_point_utm.x() - start_point_utm.x()
        dy = end_point_utm.y() - start_point_utm.y()
        length = (dx**2 + dy**2)**0.5

        unit_dx = dx / length
        unit_dy = dy / length

        perp_dx = -unit_dy
        perp_dy = unit_dx

        tangent_start_utm = QgsPointXY(
            mid_point_utm.x() - perp_dx * offset_distance,
            mid_point_utm.y() - perp_dy * offset_distance
        )
        tangent_end_utm = QgsPointXY(
            mid_point_utm.x() + perp_dx * offset_distance,
            mid_point_utm.y() + perp_dy * offset_distance
        )

        tangent_start = transform_to_canvas.transform(tangent_start_utm)
        tangent_end = transform_to_canvas.transform(tangent_end_utm)

        # Remove existing tangent lines for this segment
        features_to_remove = [
            f.id() for f in self.tangent_layer.getFeatures() if f["type"] == f"Tangent Line {segment_id}"
        ]
        if features_to_remove:
            self.tangent_layer.deleteFeatures(features_to_remove)

        # Add the new tangent line as a feature
        tangent_fet = QgsFeature()
        tangent_fet.setGeometry(QgsLineString([QgsPoint(tangent_start), QgsPoint(tangent_end)]))
        tangent_fet.setAttributes([f"Tangent Line {segment_id}"])
        self.tangent_layer.dataProvider().addFeatures([tangent_fet])

        self.tangent_layer.commitChanges()
        self.tangent_layer.triggerRepaint()
        
    def on_width_changed(self, item):
        """Handle changes to the width in the twRouteList table."""
        column = item.column()
        row = item.row()
        if column == 4:  # Assuming the 'Width' column is at index 4
            segment_id = int(self.omrat.dockwidget.twRouteList.item(row, 0).text())
            start_point_wkt = self.omrat.dockwidget.twRouteList.item(row, 2).text()
            end_point_wkt = self.omrat.dockwidget.twRouteList.item(row, 3).text()
            width = float(self.omrat.dockwidget.twRouteList.item(row, 4).text())

            # Convert WKT to QgsGeometry
            start_point_geom = QgsGeometry.fromWkt(f"POINT({start_point_wkt})")
            end_point_geom = QgsGeometry.fromWkt(f"POINT({end_point_wkt})")

            # Extract QgsPoint from QgsGeometry
            if not start_point_geom.isEmpty() and not end_point_geom.isEmpty():
                start_point = start_point_geom.asPoint()
                end_point = end_point_geom.asPoint()
                # Remove the existing tangent line and redraw it with the updated width
                self.create_offset_lines(start_point, end_point, width / 2, segment_id)

    def point4326_from_wkt(self, wkt):
        """Converts a WKT string to a QgsGeometry in EPSG:4326."""
        q_point = QgsGeometry.fromWkt(wkt)
        crs = self.omrat.iface.mapCanvas().mapSettings().destinationCrs().authid()
        tr = QgsCoordinateTransform(QgsCoordinateReferenceSystem(crs),
                                     QgsCoordinateReferenceSystem("EPSG:4326"),
                                     QgsProject.instance())
        q_point.transform(tr)
        return q_point

    def format_wkt(self, point):
        """Formats a point as a WKT string with six decimal places."""
        return f'{point.x():.6f} {point.y():.6f}'

    def on_geometry_changed_wrapper(self, segment_id):
        """Wrapper for the geometryChanged signal to pass the segment ID."""
        def handler(fid, geom):
            self.on_geometry_changed(segment_id, geom)
        return handler