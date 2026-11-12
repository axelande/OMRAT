from __future__ import annotations
import math
import os
from typing import TYPE_CHECKING

import re as _re
from qgis.PyQt.QtCore import QVariant, Qt
from qgis.PyQt.QtWidgets import QFileDialog, QTableWidgetItem, QTableWidget
from qgis._core import QgsVectorDataProvider
from qgis.core import (Qgis, QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsField,
                       QgsFillSymbol, QgsGraduatedSymbolRenderer, QgsRendererRange,
                       QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsMessageLog)
from qgis.PyQt.QtGui import QColor
import requests
import tempfile


if TYPE_CHECKING:
    from omrat import OMRAT


def _wkt_summary(wkt: str) -> str:
    """Return a short label for a WKT geometry (e.g. 'Polygon (42 pts)')."""
    geom_type = wkt.split('(')[0].strip() if '(' in wkt else wkt[:30]
    n = len(_re.findall(r'[-\d.]+\s+[-\d.]+', wkt))
    if n > 5:
        return f"{geom_type}({n} pts)"
    return wkt if len(wkt) <= 60 else wkt[:57] + '...'


_USER_ROLE = getattr(Qt, 'UserRole', None) or Qt.ItemDataRole.UserRole


def _wkt_table_item(wkt: str) -> QTableWidgetItem:
    """QTableWidgetItem that shows a compact summary but stores the full WKT in UserRole."""
    item = QTableWidgetItem(_wkt_summary(wkt))
    item.setData(_USER_ROLE, wkt)
    item.setToolTip(wkt[:400] + ('...' if len(wkt) > 400 else ''))
    return item


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
            except Exception:  # nosec B110 B112
                continue
    return coords


def get_bbox(coords: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Return min/max lat/lon from a list of (lon, lat) tuples."""
    lats = [lat for _, lat in coords]
    lons = [lon for lon, _ in coords]
    return min(lats), max(lats), min(lons), max(lons)


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
        "https://portal.opentopography.org/API/globaldem?"
        f"demtype=GEBCOIceTopo&south={min_lat}&north={max_lat}&west={min_lon}&east={max_lon}"
        f"&outputFormat=GTiff&API_Key={api_key}"
    )


def download_geotiff(url: str, save_path: str = "gebco_download.ti", timeout: float = 60.0) -> str:
    """
    Download a GeoTIFF file from the given URL and save it to disk.
    Returns the path to the saved file.

    A connect/read timeout (default 60s) is enforced so a stalled GEBCO
    response can't hang the QGIS UI thread indefinitely.
    """
    response = requests.get(url, stream=True, timeout=timeout)
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

    def _depth_layer_in_project(self) -> bool:
        """Return True if the cached depth layer is still registered in the QGIS project.

        The user may delete "Depth Areas" from the Layers panel, or open a
        new QGIS project, without going through the plugin.  In both cases
        ``self.depth_layer`` keeps pointing at a layer that is no longer in
        the layer tree (and possibly a deleted C++ object).  Reusing it
        would silently add features to a layer nobody can see.
        """
        if self.depth_layer is None:
            return False
        try:
            layer_id = self.depth_layer.id()
        except RuntimeError:
            # Underlying C++ object already deleted by QGIS.
            return False
        project = QgsProject.instance()
        if project is None:
            return False
        return project.mapLayer(layer_id) is not None

    def _ensure_depth_layer(self) -> QgsVectorLayer:
        """Create the consolidated 'Depth Areas' layer if it doesn't exist, or return it.

        If a previously created layer has been removed from the project
        behind our back, the stale reference is dropped and a fresh layer
        is created and added to the project.
        """
        if self.depth_layer is not None:
            if self._depth_layer_in_project():
                return self.depth_layer
            self.depth_layer = None
            self._depth_edit_buffer = None
            self.depth_feature_row = {}

        layer = QgsVectorLayer("Polygon?crs=epsg:4326", "Depth Areas", "memory")
        pr = layer.dataProvider()
        if pr is not None:
            pr.addAttributes([
                QgsField("id", QVariant.Int),      # type: ignore[arg-type]
                QgsField("depth", QVariant.Double),  # type: ignore[arg-type]
            ])
            layer.updateFields()

        QgsProject.instance().addMapLayer(layer)

        if not layer.isEditable():
            layer.startEditing()

        # Layer-level signal: survives commits / edit-mode toggles, unlike
        # the edit buffer's (see HandleQGISIface.wire_leg_layer).
        layer.geometryChanged.connect(self._on_depth_geometry_changed)
        self._depth_edit_buffer = layer

        self.depth_layer = layer
        try:
            from omrat_utils.layer_styles import apply_stored_style
            apply_stored_style(self.p, 'depths', layer)
        except Exception:  # nosec B110 B112
            pass
        return layer

    def _on_depth_geometry_changed(self, fid: int, geom: QgsGeometry) -> None:
        """Sync a geometry edit on a consolidated depth feature back to twDepthList."""
        wkt = geom.asWkt(precision=5)
        row = self.depth_feature_row.get(fid)
        if row is not None and 0 <= row < self.p.main_widget.twDepthList.rowCount():
            self.p.main_widget.twDepthList.setItem(row, 2, _wkt_table_item(wkt))

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
        renderer.setMode(QgsGraduatedSymbolRenderer.Mode.Custom)
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
            except Exception:  # nosec B110 B112
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
        except (TypeError, RuntimeError):  # nosec B110 - already disconnected
            pass
        if self.area_type == 'depth':
            self.add_simple_depth()
        if self.area_type == 'object':
            self.add_simple_object()

    def load_area(self, name: str, wkt: str, row: int | None = None,
                  value: str | None = None, value_field: str | None = None,
                  defer_style: bool = False):
        from geometries.depth_simplify import depth_value
        # Depth areas: add as feature to the consolidated depth layer
        if self.area_type == 'depth':
            if row is None:
                row = self.p.main_widget.twDepthList.rowCount() - 1
            depth_val = 0.0
            if value is not None:
                # The class value compute uses: an interval's shallower end.
                v = depth_value(value)
                if not math.isnan(v):
                    depth_val = v
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
            v = depth_value(value)
            if not math.isnan(v):
                fet.setAttribute(value_field, v)

        if pr is not None:
            pr.addFeature(fet)
        QgsProject.instance().addMapLayer(area)
        if self.area_type == 'object':
            try:
                from omrat_utils.layer_styles import apply_stored_style
                apply_stored_style(self.p, 'structures', area)
            except Exception:  # nosec B110 B112
                pass
        if not area.isEditable():
            area.startEditing()
        layer_id = area.id()
        if self.area_type == 'object':
            if row is None:
                row = self.p.main_widget.twObjectList.rowCount() - 1
            self.object_layer_row[layer_id] = row

            def _slot(fid, geom, lid=layer_id):
                self.on_area_geometry_changed_wrapper(lid, 'object', fid, geom)
            # Layer-level signal (survives commits); remember the slot so
            # teardown disconnects only ours.
            area.geometryChanged.connect(_slot)
            self.object_buffer_edits.append((area, _slot))
        self.p.iface.actionSaveActiveLayerEdits().trigger()
        if self.area_type == 'object':
            self.loaded_object_areas.append(area)

    def on_area_geometry_changed_wrapper(self, layer_id: str, kind: str, fid: int, geom: QgsGeometry):
        # Update WKT in the object table when geometry changes (depth handled by _on_depth_geometry_changed)
        wkt = geom.asWkt(precision=5)
        row = self.object_layer_row.get(layer_id)
        if row is not None and 0 <= row < self.p.main_widget.twObjectList.rowCount():
            self.p.main_widget.twObjectList.setItem(row, 2, _wkt_table_item(wkt))

    def update_depth_intervals(self) -> None:
        """Update TWDepthIntervals with intervals from 0 to LEMaxDepth using SBDepthInterval."""
        try:
            interval = int(self.p.main_widget.SBDepthInterval.value())
            max_depth = int(float(self.p.main_widget.LEMaxDepth.text()))
        except Exception:  # nosec B110 B112
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
        w = self.p.main_widget
        try:
            min_lat = float(w.LEBboxLLLat.text())
            min_lon = float(w.LEBboxLLLon.text())
            max_lat = float(w.LEBboxURLat.text())
            max_lon = float(w.LEBboxURLon.text())
        except ValueError:
            self.p.show_error_popup(
                "Enter valid lower-left and upper-right coordinates before fetching.",
                "obtain_gebco_data"
            )
            return

        if min_lat >= max_lat or min_lon >= max_lon:
            self.p.show_error_popup(
                "Lower-left must be south-west of upper-right (lat_ll < lat_ur, lon_ll < lon_ur).",
                "obtain_gebco_data"
            )
            return

        api_key = w.LEOpenTopoAPIKey.text()
        url = build_gebco_url(min_lat, max_lat, min_lon, max_lon, api_key)

        temp_dir = tempfile.gettempdir()
        geotiff_path = download_geotiff(url, save_path=os.path.join(temp_dir, 'gebco_download.tif'))
        tw = w.TWDepthIntervals
        intervals = [float(tw.item(i, 0).text()) for i in range(tw.rowCount())]
        self.vectorize_and_add_geotiff(geotiff_path, intervals)

    def vectorize_and_add_geotiff(
        self,
        geotiff_path: str,
        intervals: list[float],
    ) -> None:
        """Vectorize a GEBCO GeoTIFF by depth interval and add to QGIS.

        Performance strategy:
        - Reclassify the raster into N interval bins with numpy (in-memory,
          fast) before polygonizing.  Raw GEBCO has thousands of distinct
          integer depth values; after reclassification there are only N
          distinct values (one per interval), so polygonize produces orders
          of magnitude fewer polygons.
        - Polygonize with GDAL directly (no QGIS processing framework
          overhead) into an OGR in-memory layer.
        - Merge features per interval with QgsGeometry.unaryUnion (C++ GEOS)
          instead of a sequential Python combine loop.
        """
        from osgeo import gdal, ogr, osr
        import numpy as np

        # ── 1. Read raster and reclassify into interval bins ──────────────
        ds = gdal.Open(geotiff_path)
        if ds is None:
            self.p.show_error_popup("Could not open GeoTIFF.", "vectorize_and_add_geotiff")
            return
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray().astype(float)
        nodata = band.GetNoDataValue()
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        ny, nx = data.shape
        ds = None

        # GEBCO stores depth below sea level as negative values.
        # Interval [low, high] metres → raster range (-high, -low].
        # Bin index starts at 1 so 0 can serve as "outside all intervals".
        classified = np.zeros((ny, nx), dtype=np.int32)
        for idx in range(len(intervals) - 1):
            low, high = intervals[idx], intervals[idx + 1]
            mask = (data > -high) & (data <= -low)
            if nodata is not None:
                mask &= (data != nodata)
            classified[mask] = idx + 1

        if not classified.any():
            return

        # ── 2. Write classified raster to GDAL virtual filesystem ─────────
        # /vsimem/ keeps everything in RAM so there is no disk round-trip.
        vsimem_raster = '/vsimem/gebco_classified.tif'
        drv = gdal.GetDriverByName("GTiff")
        out_ds = drv.Create(vsimem_raster, nx, ny, 1, gdal.GDT_Int32)
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(classified)
        out_band.SetNoDataValue(0)
        out_ds.FlushCache()
        out_ds = None

        # Build a mask band (non-zero = valid pixel to polygonize).
        vsimem_mask = '/vsimem/gebco_mask.tif'
        msk_ds = drv.Create(vsimem_mask, nx, ny, 1, gdal.GDT_Byte)
        msk_ds.SetGeoTransform(geotransform)
        msk_ds.SetProjection(projection)
        msk_ds.GetRasterBand(1).WriteArray((classified != 0).astype(np.uint8) * 255)
        msk_ds.FlushCache()
        msk_ds = None

        # ── 3. Polygonize directly into an OGR memory layer ───────────────
        src_ds = gdal.Open(vsimem_raster)
        src_band = src_ds.GetRasterBand(1)
        msk_src = gdal.Open(vsimem_mask)
        mask_band = msk_src.GetRasterBand(1)

        mem_drv = ogr.GetDriverByName("Memory")
        mem_ds = mem_drv.CreateDataSource("polys")
        srs = osr.SpatialReference()
        srs.ImportFromWkt(projection)
        dst_layer = mem_ds.CreateLayer("polygons", srs=srs)
        dst_layer.CreateField(ogr.FieldDefn("VALUE", ogr.OFTInteger))
        val_idx = dst_layer.GetLayerDefn().GetFieldIndex("VALUE")

        gdal.Polygonize(src_band, mask_band, dst_layer, val_idx, [], callback=None)

        src_ds = None
        msk_src = None
        gdal.Unlink(vsimem_raster)
        gdal.Unlink(vsimem_mask)

        # ── 4. Merge per interval with unaryUnion and add to table/layer ──
        for idx in range(len(intervals) - 1):
            dst_layer.SetAttributeFilter(f"VALUE = {idx + 1}")
            qgs_geoms = []
            for feat in dst_layer:
                geom_ref = feat.GetGeometryRef()
                if geom_ref is not None:
                    qgs_geoms.append(QgsGeometry.fromWkt(geom_ref.ExportToWkt()))
            dst_layer.ResetReading()

            if not qgs_geoms:
                continue

            merged_geom = QgsGeometry.unaryUnion(qgs_geoms)
            wkt = merged_geom.asWkt(precision=5)

            interval = intervals[idx]
            row = self.p.main_widget.twDepthList.rowCount()
            self.p.main_widget.twDepthList.insertRow(row)
            self.p.main_widget.twDepthList.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.p.main_widget.twDepthList.setItem(row, 1, QTableWidgetItem(f"{interval}-{intervals[idx + 1]}"))
            self.p.main_widget.twDepthList.setItem(row, 2, _wkt_table_item(wkt))

            # Coloured by the class value compute uses: the shallower end.
            depth_val = min(intervals[idx], intervals[idx + 1])
            self._add_depth_feature(row + 1, depth_val, wkt, row, defer_style=True)

        self._apply_depth_graduated_style()

    def add_simple_depth(self):
        self.area_type = 'depth'
        if self.p.main_widget.pbAddSimpleDepth.text() == 'Save':
            self.store_depth()
            self.p.main_widget.pbAddSimpleDepth.setText('Add manually')
            # Add drawn polygon to consolidated depth layer
            assert self.area is not None  # nosec B101
            row = self.deph_id - 1
            wkt_item = self.p.main_widget.twDepthList.item(row, 2)
            depth_item = self.p.main_widget.twDepthList.item(row, 1)
            if wkt_item is not None:
                wkt = wkt_item.data(_USER_ROLE) or wkt_item.text()
                depth_val = float(depth_item.text()) if depth_item else 10.0
                self._add_depth_feature(self.deph_id, depth_val, wkt, row)
            # Remove the temporary drawing layer from the project
            QgsProject.instance().removeMapLayer(self.area.id())
            self.area = None
        else:
            # Next row of the table: loads (file, shapefile, GEBCO) fill the
            # table without this counter, so counting on would overwrite rows.
            self.deph_id = self.p.main_widget.twDepthList.rowCount() + 1
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
            wkt_poly = poly.geometry().asWkt(precision=5)
            self.p.main_widget.twDepthList.setItem(self.deph_id - 1, 0, item1)
            self.p.main_widget.twDepthList.setItem(self.deph_id - 1, 1, item2)
            self.p.main_widget.twDepthList.setItem(self.deph_id - 1, 2, _wkt_table_item(wkt_poly))

        self.p.iface.actionSaveActiveLayerEdits().trigger()
        self.p.iface.actionToggleEditing().trigger()

    def add_simple_object(self):
        self.area_type = 'object'
        if self.p.main_widget.pbAddSimpleObject.text() == 'Save':
            self.store_object()
            self.p.main_widget.pbAddSimpleObject.setText('Add manually')
            # The drawn structure becomes the same one-feature layer a loaded
            # structure gets (``load_area``: Height field, stored style,
            # geometry edits written to the table); the drawing layer goes.
            table = self.p.main_widget.twObjectList
            row = self.object_id - 1
            wkt_item = table.item(row, 2)
            height_item = table.item(row, 1)
            if self.area is not None:
                QgsProject.instance().removeMapLayer(self.area.id())
                self.area = None
            if wkt_item is not None:
                wkt = wkt_item.data(_USER_ROLE) or wkt_item.text()
                height = height_item.text() if height_item is not None else '10.0'
                self.load_area(f'Structure - {height}m', wkt, row=row, value=height, value_field='Height')
        else:
            # Next row of the table: a project or shapefile load fills it
            # without this counter, so counting on would overwrite rows.
            self.object_id = self.p.main_widget.twObjectList.rowCount() + 1
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
            wkt_poly = poly.geometry().asWkt(precision=5)
            self.p.main_widget.twObjectList.setItem(self.object_id - 1, 0, item1)
            self.p.main_widget.twObjectList.setItem(self.object_id - 1, 1, item2)
            self.p.main_widget.twObjectList.setItem(self.object_id - 1, 2, _wkt_table_item(wkt_poly))

        self.p.iface.actionSaveActiveLayerEdits().trigger()
        self.p.iface.actionToggleEditing().trigger()

    def _select_file(self, title: str) -> str | None:
        file_path, _ = QFileDialog.getOpenFileName(
            self.p.main_widget,
            title,
            "",
            "Shapefiles (*.shp)"
        )
        return file_path if file_path else None

    def _report_invalid_layer(self, file_path: str) -> None:
        """Tell the user a chosen file could not be read as a vector layer."""
        msg = f"Could not open {os.path.basename(file_path)} as a vector layer -- nothing was loaded."
        QgsMessageLog.logMessage(f"{msg} ({file_path})", 'OMRAT', Qgis.MessageLevel.Warning)
        notifier = getattr(self.p, 'notifier', None)
        if notifier is not None:
            try:
                notifier.display_message(msg, level=Qgis.MessageLevel.Warning, duration=10)
            except Exception:  # nosec B110 B112 - the log line above already has it
                pass

    def _wgs84_transform(self, layer: QgsVectorLayer) -> QgsCoordinateTransform | None:
        # Object/depth WKTs are stored as EPSG:4326 throughout OMRAT, so a
        # shapefile in any other CRS must be reprojected before its WKT is
        # extracted -- otherwise projected meters land in the table and the
        # calculation later fails the forward-transform check.
        src = layer.crs()
        if not src.isValid() or src.authid() == 'EPSG:4326':
            return None
        wgs84 = QgsCoordinateReferenceSystem('EPSG:4326')
        return QgsCoordinateTransform(src, wgs84, QgsProject.instance().transformContext())

    @staticmethod
    def _attribute(feature, layer: QgsVectorLayer, names: tuple[str, ...]):
        """The first of ``names`` the layer has as a field (any case), or 0.0."""
        by_lower = {n.lower(): n for n in layer.fields().names()}
        for name in names:
            field = by_lower.get(name.lower())
            if field is not None:
                return feature.attribute(field)
        return 0.0

    # Shapefile fields read as a structure's height, in order.  ``object``
    # was the only one read until 2026-11-12; it stays for files made then.
    STRUCTURE_HEIGHT_FIELDS = ('height', 'object')
    DEPTH_FIELDS = ('depth',)

    def load_objects(self) -> None:
        """Add the polygons of a shapefile as structures.

        Height from the field ``height`` (any case; ``object`` also read).
        Every polygon becomes its own row and its own one-feature layer, as
        when a project is loaded (``load_area``), so removing a row removes
        its layer and geometry edits reach the table.
        """
        file_path = self._select_file("Select a shapefile with structures")
        if file_path is None:
            return
        layer = QgsVectorLayer(file_path, "temp_structures", "ogr")
        if not layer.isValid():
            self._report_invalid_layer(file_path)
            return

        tr = self._wgs84_transform(layer)
        table = self.p.main_widget.twObjectList
        self.area_type = 'object'
        for feature in layer.getFeatures():
            geom = feature.geometry()
            if tr is not None:
                geom = QgsGeometry(geom)
                geom.transform(tr)
            wkt = geom.asWkt(precision=5)
            value = self._attribute(feature, layer, self.STRUCTURE_HEIGHT_FIELDS)
            height = str(value) if value not in (None, '') else '0.0'
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            table.setItem(row, 1, QTableWidgetItem(height))
            table.setItem(row, 2, _wkt_table_item(wkt))
            self.load_area(f'Structure - {height}m', wkt, row=row, value=height, value_field='Height')

    def load_depths(self) -> None:
        file_path = self._select_file("Select a shapefile with depth areas")
        if file_path is None:
            return

        # Load shapefile as temporary layer (not added to project)
        temp_layer = QgsVectorLayer(file_path, "temp_depths", "ogr")
        if not temp_layer.isValid():
            self._report_invalid_layer(file_path)
            return

        tr = self._wgs84_transform(temp_layer)

        # Extract features and add to consolidated layer + table
        row_index = self.p.main_widget.twDepthList.rowCount()
        for feature in temp_layer.getFeatures():
            geom = feature.geometry()
            if tr is not None:
                geom = QgsGeometry(geom)
                geom.transform(tr)
            wkt = geom.asWkt(precision=5)
            value = self._attribute(feature, temp_layer, self.DEPTH_FIELDS)

            self.p.main_widget.twDepthList.insertRow(row_index)
            self.p.main_widget.twDepthList.setItem(row_index, 0, QTableWidgetItem(str(row_index + 1)))
            self.p.main_widget.twDepthList.setItem(row_index, 1, QTableWidgetItem(str(value)))
            self.p.main_widget.twDepthList.setItem(row_index, 2, _wkt_table_item(wkt))

            depth_val = float(value) if value else 0.0
            self._add_depth_feature(row_index + 1, depth_val, wkt, row_index, defer_style=True)
            row_index += 1

        self._apply_depth_graduated_style()

    @staticmethod
    def _table_rows(table: QTableWidget) -> list[list[str]]:
        """``[id, value, wkt]`` rows of a depth / structure table (full WKT, not the summary)."""
        rows: list[list[str]] = []
        for row in range(table.rowCount()):
            cells = [table.item(row, col) for col in range(3)]
            if any(c is None for c in cells):
                continue
            wkt = cells[2].data(_USER_ROLE) or cells[2].text()
            rows.append([cells[0].text(), cells[1].text(), str(wkt)])
        return rows

    def depth_rows(self) -> list[list[str]]:
        """The depth table as ``[id, depth, wkt]`` rows."""
        return self._table_rows(self.p.main_widget.twDepthList)

    def object_rows(self) -> list[list[str]]:
        """The structure table as ``[id, height, wkt]`` rows."""
        return self._table_rows(self.p.main_widget.twObjectList)

    def replace_depths(self, rows: list[list[str]]) -> None:
        """Replace every depth polygon (table and "Depth Areas" layer) with ``rows``."""
        table = self.p.main_widget.twDepthList
        if self.depth_layer is not None and self._depth_layer_in_project():
            pr = self.depth_layer.dataProvider()
            if pr is not None:
                pr.deleteFeatures([f.id() for f in self.depth_layer.getFeatures()])
            self.depth_layer.updateExtents()
        self.depth_feature_row = {}
        table.setRowCount(0)
        for row_index, (depth_id, depth, wkt) in enumerate(rows):
            table.insertRow(row_index)
            table.setItem(row_index, 0, QTableWidgetItem(str(depth_id)))
            table.setItem(row_index, 1, QTableWidgetItem(str(depth)))
            table.setItem(row_index, 2, _wkt_table_item(wkt))
            try:
                depth_val = float(depth)
            except (TypeError, ValueError):
                depth_val = 0.0
            self._add_depth_feature(row_index + 1, depth_val, wkt, row_index, defer_style=True)
        self.deph_id = len(rows)
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

    def _disconnect_object_geometry_slots(self) -> None:
        for entry in self.object_buffer_edits:
            try:
                layer, slot = entry
                layer.geometryChanged.disconnect(slot)
            except Exception:  # nosec B110 B112
                pass
        self.object_buffer_edits = []

    def _cleanup_depth_layer(self) -> None:
        """Disconnect signals and remove the consolidated depth layer."""
        if self._depth_edit_buffer is not None:
            try:
                self._depth_edit_buffer.geometryChanged.disconnect(self._on_depth_geometry_changed)
            except Exception:  # nosec B110 B112
                pass
            self._depth_edit_buffer = None
        if self.depth_layer is not None:
            try:
                QgsProject.instance().removeMapLayer(self.depth_layer.id())
            except Exception:  # nosec B110 B112
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
        self._disconnect_object_geometry_slots()
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
        self._disconnect_object_geometry_slots()

        # Remove object layers from QGIS
        for layer in self.loaded_object_areas:
            try:
                QgsProject.instance().removeMapLayer(layer.id())
            except Exception:  # nosec B110 B112
                pass
        self.loaded_object_areas = []
        self.object_layer_row = {}

        # Reset counters
        self.deph_id = 0
        self.object_id = 0
        self.area_type = ''

        # Reset button text in case user was mid-add
        try:
            self.p.main_widget.pbAddSimpleDepth.setText('Add manually')
            self.p.main_widget.pbAddSimpleObject.setText('Add manually')
        except Exception:  # nosec B110 B112
            pass
