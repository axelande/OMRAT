from __future__ import annotations
import os
from typing import TYPE_CHECKING

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtWidgets import QFileDialog, QTableWidgetItem, QTableWidget
from qgis._core import QgsVectorDataProvider
from qgis.core import (QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsFeatureRequest, QgsField,
                       QgsFillSymbol, QgsGraduatedSymbolRenderer, QgsRendererRange)
from qgis.PyQt.QtGui import QColor
import requests
import processing
import tempfile


if TYPE_CHECKING:
    from omrat import OMRAT

def get_leg_coordinates(tbl: QTableWidget) -> list[tuple[float, float]]:
    """Extract all start/end coordinates from the route table."""
    coords: list[tuple[float, float]] = []
    for row in range(tbl.rowCount()):
        start_str = tbl.item(row, 3)
        end_str = tbl.item(row, 4)
        if start_str is None or end_str is None:
            continue
        for coord_str in [start_str.text(), end_str.text()]:
            try:
                # WKT: 'POINT (lon lat)'
                coord = coord_str.split(' ')
                lon, lat = float(coord[0]), float(coord[1])
                coords.append((lon, lat))
            except Exception:
                continue
    return coords

def get_bbox(coords: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Return min/max lat/lon from a list of (lon, lat) tuples."""
    lats = [lat for _, lat in coords]
    lons = [lon for lon, _ in coords]
    return min(lats), max(lats), min(lons), max(lons)

def expand_bbox(min_lat: float, max_lat: float, min_lon: float, max_lon: float, extension_percent: float) -> tuple[float, float, float, float]:
    """Expand the bounding box by a percentage."""
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    lat_ext = lat_range * extension_percent / 100.0
    lon_ext = lon_range * extension_percent / 100.0
    return min_lat - lat_ext, max_lat + lat_ext, min_lon - lon_ext, max_lon + lon_ext

def get_depth_color(depth: float, max_depth: float = 50.0) -> QColor:
    """Get a blue color for depth: 0m = dark blue (danger), max_depth = light/white (safe).

    Args:
        depth: The depth value in meters (shallower = darker blue = more danger)
        max_depth: The depth at which color is lightest (default 50m)

    Returns:
        QColor ranging from dark blue (shallow/danger) to light blue/white (deep/safe)
    """
    # Clamp depth to 0-max_depth range
    depth = max(0, min(depth, max_depth))
    # Calculate ratio (0 = shallow/dark, 1 = deep/light)
    ratio = depth / max_depth

    # Interpolate from dark blue (0, 0, 139) to light blue/white (200, 220, 255)
    # Shallow (0m) = dark blue (danger), Deep (max_depth) = light blue (safe)
    r = int(0 + 200 * ratio)      # 0 -> 200
    g = int(0 + 220 * ratio)      # 0 -> 220
    b = int(139 + (255 - 139) * ratio)  # 139 -> 255

    return QColor(r, g, b)

def build_gebco_url(min_lat: float, max_lat: float, min_lon: float, max_lon: float, api_key: str) -> str:
    return (
        f"https://portal.opentopography.org/API/globaldem?"
        f"demtype=GEBCOIceTopo&south={min_lat}&north={max_lat}&west={min_lon}&east={max_lon}"
        f"&outputFormat=GTiff&API_Key={api_key}"
    )

def download_geotiff(url: str, save_path: str = "gebco_download.tif") -> str:
    """
    Download a GeoTIFF file from the given URL and save it to disk.
    Returns the path to the saved file.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raises an error for bad responses

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return save_path

class OObject:
    def __init__(self, parent: "OMRAT") -> None:
        self.p = parent
        self.deph_id = 0
        self.area: QgsVectorLayer | None = None
        self.object_id = 0
        self.area_type = ''
        # Single consolidated depth layer (all depth polygons as features)
        self.depth_layer: QgsVectorLayer | None = None
        self.depth_feature_row: dict[int, int] = {}  # feature ID -> table row
        self._depth_edit_buffer = None
        # Object layers remain per-layer (not consolidated)
        self.loaded_object_areas: list[QgsVectorLayer] = []
        self.object_layer_row: dict[str, int] = {}
        self.object_buffer_edits = []
        
    # ------------------------------------------------------------------
    # Consolidated depth layer helpers
    # ------------------------------------------------------------------

    def _ensure_depth_layer(self) -> QgsVectorLayer:
        """Create the consolidated 'Depth Areas' layer if it doesn't exist, or return it."""
        if self.depth_layer is not None:
            return self.depth_layer

        layer = QgsVectorLayer("Polygon?crs=epsg:4326", "Depth Areas", "memory")
        pr = layer.dataProvider()
        if pr is not None:
            pr.addAttributes([
                QgsField("id", QVariant.Int),      # type: ignore[arg-type]
                QgsField("depth", QVariant.Double), # type: ignore[arg-type]
            ])
            layer.updateFields()

        QgsProject.instance().addMapLayer(layer)

        if not layer.isEditable():
            layer.startEditing()

        buf = layer.editBuffer()
        if buf is not None:
            buf.geometryChanged.connect(self._on_depth_geometry_changed)
            self._depth_edit_buffer = buf

        self.depth_layer = layer
        return layer

    def _on_depth_geometry_changed(self, fid: int, geom: QgsGeometry) -> None:
        """Sync a geometry edit on a consolidated depth feature back to twDepthList."""
        wkt = geom.asWkt(precision=5)
        row = self.depth_feature_row.get(fid)
        if row is not None and 0 <= row < self.p.main_widget.twDepthList.rowCount():
            self.p.main_widget.twDepthList.setItem(row, 2, QTableWidgetItem(wkt))

    def _add_depth_feature(self, depth_id: int, depth_value: float, wkt: str,
                           row: int, defer_style: bool = False) -> None:
        """Add a single depth polygon as a feature in the consolidated layer."""
        layer = self._ensure_depth_layer()
        pr = layer.dataProvider()
        if pr is None:
            return

        feat = QgsFeature(layer.fields())
        feat.setGeometry(QgsGeometry.fromWkt(wkt))
        feat.setAttribute("id", depth_id)
        feat.setAttribute("depth", depth_value)

        success, added_features = pr.addFeatures([feat])
        if success and added_features:
            fid = added_features[0].id()
            self.depth_feature_row[fid] = row

        layer.updateExtents()
        if not defer_style:
            self._apply_depth_graduated_style()

    def _apply_depth_graduated_style(self) -> None:
        """Apply graduated blue symbology based on the 'depth' field."""
        if self.depth_layer is None:
            return

        depths: list[float] = []
        for feat in self.depth_layer.getFeatures():
            d = feat.attribute("depth")
            if d is not None:
                try:
                    depths.append(float(d))
                except (ValueError, TypeError):
                    pass

        if not depths:
            return

        min_d = min(depths)
        max_d = max(depths) if max(depths) > min_d else min_d + 1.0
        unique_depths = sorted(set(depths))
        num_classes = min(10, len(unique_depths))
        step = (max_d - min_d) / num_classes

        ranges = []
        for i in range(num_classes):
            lower = min_d + i * step
            upper = min_d + (i + 1) * step
            mid = (lower + upper) / 2
            color = get_depth_color(mid, max_d)
            symbol = QgsFillSymbol.createSimple({
                'color': color.name(),
                'outline_color': '#000080',
                'outline_width': '0.26'
            })
            label = f"{lower:.1f} - {upper:.1f}m"
            rng = QgsRendererRange(lower, upper, symbol, label)
            ranges.append(rng)

        renderer = QgsGraduatedSymbolRenderer('depth', ranges)
        renderer.setMode(QgsGraduatedSymbolRenderer.Custom)
        self.depth_layer.setRenderer(renderer)
        self.depth_layer.triggerRepaint()

    def _rebuild_depth_feature_row_map(self) -> None:
        """Re-establish depth_feature_row by matching feature 'id' attrs to table row IDs."""
        self.depth_feature_row.clear()
        if self.depth_layer is None:
            return
        table = self.p.main_widget.twDepthList
        for feat in self.depth_layer.getFeatures():
            feat_id_attr = str(feat.attribute("id"))
            for row in range(table.rowCount()):
                id_item = table.item(row, 0)
                if id_item is not None and id_item.text() == feat_id_attr:
                    self.depth_feature_row[feat.id()] = row
                    break

    def add_area(self, name: str = 'area', value_field: str | None = None) -> None:
        if self.area is not None:
            try:
                self.area.featureAdded.disconnect(self.on_feature_added)
            except Exception:
                pass
        self.area = QgsVectorLayer("Polygon?crs=epsg:4326", name, "memory")

        # Add value field (Depth or Height) if specified
        if value_field:
            pr = self.area.dataProvider()
            if pr is not None:
                pr.addAttributes([QgsField(value_field, QVariant.Double)])  # type: ignore[arg-type]
                self.area.updateFields()

        self.area.startEditing()
        self.area.featureAdded.connect(self.on_feature_added)
        self.p.iface.actionAddFeature().trigger()
        QgsProject.instance().addMapLayer(self.area)
    
    def on_feature_added(self, fid):
        """This will be called after the user right-clicks to finish the polygon"""
        try:
            self.area.featureAdded.disconnect(self.on_feature_added)
            print('dictonnected')
        except Exception as e:
            print(e)
        if self.area_type == 'depth':
            self.add_simple_depth()
        if self.area_type == 'object':
            self.add_simple_object()
    
    def load_area(self, name: str, wkt: str, row: int | None = None,
                  value: str | None = None, value_field: str | None = None,
                  defer_style: bool = False):
        # Depth areas: add as feature to the consolidated depth layer
        if self.area_type == 'depth':
            if row is None:
                row = self.p.main_widget.twDepthList.rowCount() - 1
            depth_val = 0.0
            if value is not None:
                try:
                    depth_val = float(value.split('-')[-1]) if '-' in str(value) else float(value)
                except (ValueError, AttributeError):
                    pass
            depth_id = int(row + 1)
            self._add_depth_feature(depth_id, depth_val, wkt, row, defer_style=defer_style)
            return

        # Object areas: keep per-layer behaviour (unchanged)
        area = QgsVectorLayer("Polygon?crs=epsg:4326", name, "memory")
        pr: QgsVectorDataProvider | None = area.dataProvider()

        if pr is not None and value_field:
            pr.addAttributes([QgsField(value_field, QVariant.Double)])  # type: ignore[arg-type]
            area.updateFields()

        fet = QgsFeature(area.fields())
        fet.setGeometry(QgsGeometry.fromWkt(wkt))

        if value_field and value is not None:
            try:
                fet.setAttribute(value_field, float(value.split('-')[-1]) if '-' in value else float(value))
            except (ValueError, AttributeError):
                pass

        if pr is not None:
            pr.addFeature(fet)
        QgsProject.instance().addMapLayer(area)
        if not area.isEditable():
            area.startEditing()
        buf = area.editBuffer()
        layer_id = area.id()
        if self.area_type == 'object':
            if row is None:
                row = self.p.main_widget.twObjectList.rowCount() - 1
            self.object_layer_row[layer_id] = row
            if buf is not None:
                buf.geometryChanged.connect(lambda fid, geom, lid=layer_id: self.on_area_geometry_changed_wrapper(lid, 'object', fid, geom))
                self.object_buffer_edits.append(buf)
        self.p.iface.actionSaveActiveLayerEdits().trigger()
        if self.area_type == 'object':
            self.loaded_object_areas.append(area)

    def on_area_geometry_changed_wrapper(self, layer_id: str, kind: str, fid: int, geom: QgsGeometry):
        # Update WKT in the object table when geometry changes (depth handled by _on_depth_geometry_changed)
        wkt = geom.asWkt(precision=5)
        row = self.object_layer_row.get(layer_id)
        if row is not None and 0 <= row < self.p.main_widget.twObjectList.rowCount():
            self.p.main_widget.twObjectList.setItem(row, 2, QTableWidgetItem(wkt))

    def update_depth_intervals(self) -> None:
        """Update TWDepthIntervals with intervals from 0 to LEMaxDepth using SBDepthInterval."""
        try:
            interval = int(self.p.main_widget.SBDepthInterval.value())
            max_depth = int(float(self.p.main_widget.LEMaxDepth.text()))
        except Exception:
            self.p.show_error_popup("Please enter valid numbers for interval and max depth.", "update_depth_intervals")
            return

        self.p.main_widget.TWDepthIntervals.setRowCount(0)
        value = 0
        row = 0
        while value <= max_depth:
            self.p.main_widget.TWDepthIntervals.insertRow(row)
            self.p.main_widget.TWDepthIntervals.setItem(row, 0, QTableWidgetItem(str(value)))
            value += interval
            row += 1
        # If last value is less than max_depth, add max_depth as final interval
        if value - interval < max_depth:
            self.p.main_widget.TWDepthIntervals.insertRow(row)
            self.p.main_widget.TWDepthIntervals.setItem(row, 0, QTableWidgetItem(str(max_depth)))
            
    def obtain_gebco_data(self):
        tbl = self.p.main_widget.twRouteList
        coords = get_leg_coordinates(tbl)
        if not coords:
            self.p.show_error_popup("You need to create legs first.", "obtain_gebco_data")
            return

        min_lat, max_lat, min_lon, max_lon = get_bbox(coords)
        extension_str = self.p.main_widget.LEGebcoExtension.text()
        try:
            extension = float(extension_str)
        except ValueError:
            self.p.show_error_popup("Extension must be a number.", "obtain_gebco_data")
            return

        min_lat, max_lat, min_lon, max_lon = expand_bbox(min_lat, max_lat, min_lon, max_lon, extension)
        api_key = self.p.main_widget.LEOpenTopoAPIKey.text()
        url = build_gebco_url(min_lat, max_lat, min_lon, max_lon, api_key)

        # Download geotiff to local temp folder
        temp_dir = tempfile.gettempdir()
        geotiff_path = download_geotiff(url, save_path=os.path.join(temp_dir, 'gebco_download.tif'))
        intervals = [float(self.p.main_widget.TWDepthIntervals.item(i, 0).text()) for i in range(self.p.main_widget.TWDepthIntervals.rowCount())]
        self.vectorize_and_add_geotiff(geotiff_path, intervals)
        
    def vectorize_and_add_geotiff(
        self,
        geotiff_path: str,
        intervals: list[float],
    ) -> None:
        """
        Vectorize the GeoTIFF raster using depth intervals and add the resulting polygons to QGIS.
        """
        # Create a unique layer name for the memory layer
        
        temp_path = os.path.join(tempfile.gettempdir(), "tmp.gpkg")

        processing.run(
            "gdal:polygonize",
            {
                "INPUT": geotiff_path,
                "BAND": 1,
                "FIELD": "VALUE",
                "EIGHT_CONNECTEDNESS": False,
                "OUTPUT": temp_path
            }
        )

        vector_layer = QgsVectorLayer(temp_path, "temp_depths", "ogr")

        if not vector_layer.isValid():
            self.p.show_error_popup("Polygonized layer could not be loaded.", "vectorize_and_add_geotiff")
            return

        # Filter polygons by intervals and add as features to the consolidated depth layer
        for idx, interval in enumerate(intervals[:-1]):
            expr = f'"VALUE" <= {-interval} and "VALUE" > {-intervals[idx + 1]}'
            request = QgsFeatureRequest().setFilterExpression(expr)
            features = [feat for feat in vector_layer.getFeatures(request)]
            if not features:
                continue

            # Merge geometries
            merged_geom = features[0].geometry()
            for feat in features[1:]:
                merged_geom = merged_geom.combine(feat.geometry())

            wkt = merged_geom.asWkt(precision=5)

            # Update the table with WKT
            row = self.p.main_widget.twDepthList.rowCount()
            self.p.main_widget.twDepthList.insertRow(row)
            self.p.main_widget.twDepthList.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.p.main_widget.twDepthList.setItem(row, 1, QTableWidgetItem(f"{interval}-{intervals[idx+1]}"))
            self.p.main_widget.twDepthList.setItem(row, 2, QTableWidgetItem(wkt))

            # Add to consolidated depth layer (defer styling until after loop)
            depth_val = intervals[idx + 1]  # upper bound of interval
            self._add_depth_feature(row + 1, depth_val, wkt, row, defer_style=True)

        self._apply_depth_graduated_style()
    
    def add_simple_depth(self):
        self.area_type = 'depth'
        if self.p.main_widget.pbAddSimpleDepth.text() == 'Save':
            self.store_depth()
            self.p.main_widget.pbAddSimpleDepth.setText('Add manual')
            # Add drawn polygon to consolidated depth layer
            assert self.area is not None
            row = self.deph_id - 1
            wkt_item = self.p.main_widget.twDepthList.item(row, 2)
            depth_item = self.p.main_widget.twDepthList.item(row, 1)
            if wkt_item is not None:
                wkt = wkt_item.text()
                depth_val = float(depth_item.text()) if depth_item else 10.0
                self._add_depth_feature(self.deph_id, depth_val, wkt, row)
            # Remove the temporary drawing layer from the project
            QgsProject.instance().removeMapLayer(self.area.id())
            self.area = None
        else:
            self.deph_id += 1
            # Create layer with Depth field - will be renamed after user enters depth value
            self.add_area(f'Depth - {self.deph_id} (enter depth)', value_field='Depth')
            self.p.main_widget.pbAddSimpleDepth.setText('Save')

    def store_depth(self):
        # Default depth value - user can edit in table
        default_depth = 10.0

        self.p.main_widget.twDepthList.setRowCount(self.deph_id)
        item1 = QTableWidgetItem(f'{self.deph_id}')
        item2 = QTableWidgetItem(f'{default_depth}')
        polies = self.area.getFeatures()
        for poly in polies:
            item3 = QTableWidgetItem(f'{poly.geometry().asWkt(precision=5)}')
            self.p.main_widget.twDepthList.setItem(self.deph_id - 1, 0, item1)
            self.p.main_widget.twDepthList.setItem(self.deph_id - 1, 1, item2)
            self.p.main_widget.twDepthList.setItem(self.deph_id - 1, 2, item3)

        self.p.iface.actionSaveActiveLayerEdits().trigger()
        self.p.iface.actionToggleEditing().trigger()
        
    def add_simple_object(self):
        self.area_type = 'object'
        if self.p.main_widget.pbAddSimpleObject.text() == 'Save':
            self.store_object()
            self.p.main_widget.pbAddSimpleObject.setText('Add manual')
            self.loaded_object_areas.append(self.area)
        else:
            self.object_id += 1
            # Create layer with Height field
            self.add_area(f'Structure - {self.object_id} (enter height)', value_field='Height')
            self.p.main_widget.pbAddSimpleObject.setText('Save')

    def store_object(self):
        # Default height value - user can edit in table
        default_height = 10.0

        self.p.main_widget.twObjectList.setRowCount(self.object_id)
        item1 = QTableWidgetItem(f'{self.object_id}')
        item2 = QTableWidgetItem(f'{default_height}')
        polies = self.area.getFeatures()
        for poly in polies:
            item3 = QTableWidgetItem(f'{poly.geometry().asWkt(precision=5)}')
            self.p.main_widget.twObjectList.setItem(self.object_id - 1, 0, item1)
            self.p.main_widget.twObjectList.setItem(self.object_id - 1, 1, item2)
            self.p.main_widget.twObjectList.setItem(self.object_id - 1, 2, item3)

        # Rename the layer to show the height value
        if self.area is not None:
            self.area.setName(f'Structure - {default_height}m')

        self.p.iface.actionSaveActiveLayerEdits().trigger()
        self.p.iface.actionToggleEditing().trigger()

    def _select_file(self, title: str) -> str | None:
        file_path, _ = QFileDialog.getOpenFileName(
            self.p.main_widget,
            title,
            "",
            "Shapefiler (*.shp)"
        )
        return file_path if file_path else None

    def _load_layer(self, file_path: str, layer_name: str, target_list: list) -> QgsVectorLayer | None:
        layer = QgsVectorLayer(file_path, layer_name, "ogr")
        if not layer.isValid():
            print(f"Ogiltigt lager: {layer_name}")
            return None
        QgsProject.instance().addMapLayer(layer)
        target_list.append(layer)
        return layer

    def _populate_table(self, layer: QgsVectorLayer, table_widget, attr_name: str) -> None:
        row_index = table_widget.rowCount()
        for feature in layer.getFeatures():
            geom = feature.geometry()
            wkt: str = geom.asWkt(precision=5)
            value = feature.attribute(attr_name) if attr_name in layer.fields().names() else 0.0

            table_widget.insertRow(row_index)
            table_widget.setItem(row_index, 0, QTableWidgetItem(str(row_index + 1)))
            table_widget.setItem(row_index, 1, QTableWidgetItem(str(value)))
            table_widget.setItem(row_index, 2, QTableWidgetItem(wkt))
            row_index += 1

    def load_objects(self) -> None:
        file_path = self._select_file("Välj shapefil med objektområden")
        if file_path is None:
            return

        layer = self._load_layer(file_path, "Loaded Objects", self.loaded_object_areas)
        if layer is None:
            return

        self._populate_table(layer, self.p.main_widget.twObjectList, "object")  # Use the correct attribute name if different

    def load_depths(self) -> None:
        file_path = self._select_file("Välj shapefil med djupområden")
        if file_path is None:
            return

        # Load shapefile as temporary layer (not added to project)
        temp_layer = QgsVectorLayer(file_path, "temp_depths", "ogr")
        if not temp_layer.isValid():
            print(f"Ogiltigt lager: Loaded Depths")
            return

        # Extract features and add to consolidated layer + table
        row_index = self.p.main_widget.twDepthList.rowCount()
        for feature in temp_layer.getFeatures():
            geom = feature.geometry()
            wkt = geom.asWkt(precision=5)
            value = feature.attribute("depth") if "depth" in temp_layer.fields().names() else 0.0

            self.p.main_widget.twDepthList.insertRow(row_index)
            self.p.main_widget.twDepthList.setItem(row_index, 0, QTableWidgetItem(str(row_index + 1)))
            self.p.main_widget.twDepthList.setItem(row_index, 1, QTableWidgetItem(str(value)))
            self.p.main_widget.twDepthList.setItem(row_index, 2, QTableWidgetItem(wkt))

            depth_val = float(value) if value else 0.0
            self._add_depth_feature(row_index + 1, depth_val, wkt, row_index, defer_style=True)
            row_index += 1

        self._apply_depth_graduated_style()

    
    def remove_depth(self) -> None:
        table = self.p.main_widget.twDepthList
        selected_rows: set[int] = {item.row() for item in table.selectedItems()}
        if not selected_rows:
            return

        # Build reverse map: row -> feature ID
        row_to_fid: dict[int, int] = {row: fid for fid, row in self.depth_feature_row.items()}

        # Collect feature IDs to remove
        fids_to_remove = [row_to_fid[row] for row in selected_rows if row in row_to_fid]

        # Remove features from the consolidated layer
        if self.depth_layer is not None and fids_to_remove:
            pr = self.depth_layer.dataProvider()
            if pr is not None:
                pr.deleteFeatures(fids_to_remove)
            self.depth_layer.updateExtents()

        # Remove rows from table (reverse order to maintain indices)
        for row in sorted(selected_rows, reverse=True):
            table.removeRow(row)

        # Rebuild the feature-to-row mapping after deletion
        self._rebuild_depth_feature_row_map()
        self._apply_depth_graduated_style()

    def remove_object(self) -> None:
        table = self.p.main_widget.twObjectList
        selected_rows: set[int] = {item.row() for item in table.selectedItems()}

        for row in sorted(selected_rows, reverse=True):
            table.removeRow(row)

            # Remove the layers if they are selected
            if row < len(self.loaded_object_areas):
                layer = self.loaded_object_areas.pop(row)
                QgsProject.instance().removeMapLayer(layer.id())

    def _cleanup_depth_layer(self) -> None:
        """Disconnect signals and remove the consolidated depth layer."""
        if self._depth_edit_buffer is not None:
            try:
                self._depth_edit_buffer.geometryChanged.disconnect()
            except Exception:
                pass
            self._depth_edit_buffer = None
        if self.depth_layer is not None:
            try:
                QgsProject.instance().removeMapLayer(self.depth_layer.id())
            except Exception:
                pass
            self.depth_layer = None
        self.depth_feature_row = {}

    def unload(self):
        if self.area is not None:
            try:
                self.area.featureAdded.disconnect(self.on_feature_added)
            except TypeError:
                pass
        self.area = None
        # Clean up consolidated depth layer
        self._cleanup_depth_layer()
        # Clean up object layers
        try:
            for buf in self.object_buffer_edits:
                buf.geometryChanged.disconnect()
        except Exception:
            pass
        self.object_buffer_edits = []
        self.object_layer_row = {}
        for layer in self.loaded_object_areas:
            QgsProject.instance().removeMapLayer(layer.id())
        self.loaded_object_areas = []

    def clear(self) -> None:
        """Remove all loaded depth/object layers and reset state.

        Unlike unload(), this keeps the plugin operational so new data
        can be loaded afterwards.
        """
        # Disconnect featureAdded signal from current area layer
        if self.area is not None:
            try:
                self.area.featureAdded.disconnect(self.on_feature_added)
            except (TypeError, RuntimeError):
                pass
        self.area = None

        # Clean up consolidated depth layer
        self._cleanup_depth_layer()

        # Disconnect object geometry-changed signals
        for buf in self.object_buffer_edits:
            try:
                buf.geometryChanged.disconnect()
            except Exception:
                pass
        self.object_buffer_edits = []

        # Remove object layers from QGIS
        for layer in self.loaded_object_areas:
            try:
                QgsProject.instance().removeMapLayer(layer.id())
            except Exception:
                pass
        self.loaded_object_areas = []
        self.object_layer_row = {}

        # Reset counters
        self.deph_id = 0
        self.object_id = 0
        self.area_type = ''

        # Reset button text in case user was mid-add
        try:
            self.p.main_widget.pbAddSimpleDepth.setText('Add manual')
            self.p.main_widget.pbAddSimpleObject.setText('Add manual')
        except Exception:
            pass
