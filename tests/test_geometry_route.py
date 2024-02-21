from qgis.core import (QgsVectorLayer, QgsPointXY, QgsGeometry, QgsField, QgsProject, 
                       QgsFeature, QgsCoordinateReferenceSystem, QgsCoordinateTransform)
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtWidgets import QApplication

from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from geometries.route import get_mean_distance

app = QApplication(sys.argv)
def get_best_utm(l_obj:list) -> CRS:
    ll, ur = get_ll_ur(l_obj)
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=ll[0],
            south_lat_degree=ll[1],
            east_lon_degree=ur[0],
            north_lat_degree=ur[1],
        ),
    )
    utm_crs = CRS.from_epsg(utm_crs_list[0].code)
    return utm_crs.srs

def get_ll_ur(l_obj):
    """Get lower left and upper right based on a list of object"""
    ll = [90, 180]
    ur = [-90, -180]
    for obj in l_obj:
        xx = obj[0]
        yy = obj[1]
        if xx < ll[0]:
            ll[0] = xx
        if yy < ll[1]:
            ll[1] = yy
        if xx > ur[0]:
            ur[0] = xx
        if yy > ur[1]:
            ur[1] = yy
        
    return [ll, ur]

def create_route(coords:list, route_id: int) -> QgsVectorLayer:
    utm_srs = get_best_utm(coords)
    sourceCrs = QgsCoordinateReferenceSystem.fromEpsgId(4326)
    destCrs = QgsCoordinateReferenceSystem.fromEpsgId(int(utm_srs.split(':')[1]))
    tr = QgsCoordinateTransform(sourceCrs, destCrs, QgsProject.instance())
    if len(coords) == 2:
        vect = QgsVectorLayer(f"LineString?crs=EPSG:4326", "temp", "memory")
    else:
        vect = QgsVectorLayer(f"Polygon?crs=epsg:4326", "temp", "memory")
    
    pr = vect.dataProvider()
    if len(coords) == 2:
        pr.addAttributes([QgsField("RouteId",  QVariant.Int)])
    else:
        pr.addAttributes([QgsField("AreaId",  QVariant.Int)])
    vect.updateFields()
    # QgsProject.instance().addMapLayer(geom) # Add the layer in QGIS project
    vect.startEditing()
    feat = QgsFeature(vect.fields()) # Create the feature

    if len(coords) == 2:
        feat.setAttribute("routeId", route_id) # set attributes
        geom = QgsGeometry.fromPolylineXY([QgsPointXY(coords[0][0], coords[0][1]),
                                           QgsPointXY(coords[1][0], coords[1][1])])

    else:
        feat.setAttribute("AreaId", route_id) # set attributes
        pointxys = []
        for coord in coords:
            pointxys.append(QgsPointXY(coord[0], coord[1])) 
        geom = QgsGeometry.fromMultiPolygonXY([[pointxys]])
    geom.transform(tr)
    feat.setGeometry(geom)
    vect.addFeature(feat) # add the feature to the layer    
    vect.endEditCommand() # Stop editing
    vect.commitChanges() # Save changes
    return vect
line0 = create_route([[17.483303, 56.6999912],
                      [17.4832501, 56.7166186]], 0)

ground = create_route([[17.5136058924, 56.8086347034],
                       [17.5137435342, 56.6331630685],
                       [17.5398315995, 56.6836007601],
                       [17.5362169055, 56.7324570938],
                       [17.5136058924, 56.8086347034]], 0)

# get_mean_distance(line0, ground, 100, 50)