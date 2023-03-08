import numpy as np
import math
from scipy import stats
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import transform, split
from qgis.core import QgsPointXY, QgsLineString, QgsGeometry
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info


def get_best_utm(l_obj:list) -> CRS:
    """returns the best CRS text for the project."""
    ll, ur = _get_ll_ur(l_obj)
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
    return utm_crs


def _get_ll_ur(l_obj):
    """Get lower left and upper right based on a list of object"""
    ll = [90, 180]
    ur = [-90, -180]
    for obj in l_obj:
        if isinstance(obj, Polygon):
            xx, yy = obj.exterior.coords.xy
        else:
            xx, yy = obj.xy
        if min(xx) < ll[0]:
            ll[0] = min(xx)
        if min(yy) < ll[1]:
            ll[1] = min(yy)
        if max(xx) > ur[0]:
            ur[0] = max(xx)
        if max(yy) > ur[1]:
            ur[1] = max(yy)
        
    return [ll, ur]


def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    # https://shapely.readthedocs.io/en/stable/manual.html#linear-referencing-methods
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

def proj_point(point:Point, distance:float, direction:float) -> Point:
    """Project a point at a disatnce and bearing"""
    new_y = point.xy[1][0] + math.cos(math.radians(direction)) * distance
    new_x = point.xy[0][0] + math.sin(math.radians(direction)) * distance
    return Point(new_x, new_y)

def get_angle(pt1, pt2):
    """Returns the angle between p1, and p2 in degrees"""
    x_diff = pt2[1] - pt1[1]
    y_diff = pt2[0] - pt1[0]
    return math.degrees(math.atan2(y_diff, x_diff))

def create_line_grid(line_utm: LineString, mu, std, width=100, height=100) -> list:
    line_utm_parts = []
    line_dir = get_angle(line_utm.coords[1], line_utm.coords[0])
    org_length = line_utm.length
    while line_utm.length > org_length / (height - 1):
        part, line_utm = cut(line_utm, org_length / height)
        line_utm_parts.append(part)
    line_utm_parts.append(line_utm)

    dist_90_line = []
    for i in range(1, width + 1):
        dist_90_line.append(stats.norm.ppf(i / width, mu, std))
    points = []
    for part in line_utm_parts:
        for dist in dist_90_line:
            dist_point = proj_point(part.centroid, dist, line_dir+90)
            points.append(dist_point)
    return points


def get_mean_distance(route:LineString, obj:Polygon, mu: float, std: float) -> float:
    max_distance = 50_000
    directions = []
    distances = []
    lines = []
    points: list[QgsPointXY] = create_line_grid(route, mu, std)
    for point in points:
        for direction in range(0, 360, 45):
            #v.distance
            #point.distance()
            p2 = proj_point(point, max_distance, direction)
            new_line = QgsLineString((point, p2))
            dist = QgsGeometry.intersects(new_line, obj)
            # dist = split(new_line, object_utm).geoms[0]
            if dist.length < max_distance:
                distances.append(dist.length)
                lines.append(dist)
                directions.append(direction)
            else:
                distances.append(0)
                directions.append(-1)
    return np.array(distances), lines, points, directions

def get_proj_tansformer(line):
    wgs84 = CRS('EPSG:4326')
    utm = get_best_utm([line])
    project = Transformer.from_crs(wgs84, utm, always_xy=True).transform
    return project

def get_multiple_ed(line:LineString, objs:list, mu:float, std:float, 
                       max_distance:float, width: int =100):
    distribution_line = []
    for i in range(1, width + 1):
        distribution_line.append(stats.norm.ppf(i / 100, mu, std))
    project = get_proj_tansformer(line)
    line_utm = transform(project, line)
    
    line_dir = get_angle(line_utm.coords[1], line_utm.coords[0])
    distances = {}
    lines = {}
    points = []
    for key in range(len(objs)):
        distances[key] = []
        lines[key] = []
    for dist in distribution_line:
        dist_point = proj_point(Point(line_utm.coords[1]), dist, line_dir-90)
        points.append(dist_point)
        for key, obj in enumerate(objs):
            object_utm = transform(project, obj["wkt"])
            new_line = LineString((dist_point, proj_point(dist_point, max_distance, line_dir+180)))
            dist = split(new_line, object_utm).geoms[0]
            lines[key].append(dist)
            if dist.length < max_distance - 1:
                distances[key].append(dist.length)
                break
    return distances, lines, points

def get_multi_drift_distance(line:LineString, objs:list, mu: float, std: float, 
                             width:int = 100, height:int = 100) -> float:
    project = get_proj_tansformer(line)
    line_utm = transform(project, line)
    utm_objs = []
    for obj in objs:
        utm_objs.append(transform(project, obj["wkt"]))
    max_distance = 50_000
    distances = {}
    directions = {}
    for key in range(len(objs)):
        distances[key] = []
        directions[key] = []
    lines = []
    points = create_line_grid(line_utm, mu, std, width, height)
    for point in points:
        for direction in range(0, 360, 45):
            for key, obj in enumerate(utm_objs):
                p2 = proj_point(point, max_distance, direction)
                new_line = LineString((point, p2))    
                if new_line.intersects(obj):
                    distances[key].append(point.distance(new_line.intersection(obj)))
                    directions[key].append(direction)
            lines.append(new_line)
                    
    return distances, lines, points, directions

