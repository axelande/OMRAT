import json
import sys
sys.path.append('.')

from numpy import exp, log
from pyproj import CRS, Transformer
from qgis.PyQt.QtWidgets import QTableWidget, QTableWidgetItem, QTreeWidgetItem, QTreeWidget
from scipy import stats
import shapely.wkt as sw
from shapely.ops import transform

from basic_equations import get_drifting_prob, get_drift_time, get_Fcoll, powered_na
from geometries.route import get_multiple_ed
from geometries.route import get_multi_drift_distance, get_best_utm 

from ui.result_widget import ResultWidget
    
def clean_traffic(data):
    """List all ships that on each segment/direction"""
    # TODO: Optimise this so each line is just estimated once.
    traffics = []
    for segment, dirs in data["traffic_data"].items():
        for k, (di, var) in enumerate(dirs.items()):
            if k == 0:
                geom = sw.loads(f'LineString({data["segment_data"][segment]["Start Point"]}, {data["segment_data"][segment]["End Point"]})')
                mean = data["segment_data"][segment]["mean"]
                std = data["segment_data"][segment]["std"]
            else:
                geom = sw.loads(f'LineString({data["segment_data"][segment]["End Point"]}, {data["segment_data"][segment]["Start Point"]})')
                mean = data["segment_data"][segment]["mean_2"]
                std = data["segment_data"][segment]["std_2"]
            for i, row in enumerate(var['Frequency (ships/year)']):
                for j, value in enumerate(row):
                    if value == '':
                        continue
                    if isinstance(value, str):
                        value = int(value)
                    if value > 0:
                        info = {'freq': value,
                                'speed': var['Speed (knots)'][i][j], 
                                'draught': var['Draught (meters)'][i][j], 
                                'height': var['Ship heights (meters)'][i][j],
                                'ship_type': i,
                                'ship_size': j,
                                'direction': di,
                                'segment': segment,
                                'geom': geom,
                                'mean': mean,
                                'std': std}
                        traffics.append(info)
    return traffics

def drift_accidents(data, w:int = 100, h:int = 100) -> dict:
    """Estimates the probability of drifting accidents"""
    objs = load_areas(data)
    wind_rose_tot = sum(data["drift"]["rose"].values())
    not_anchor = 1 - data['drift']['anchor_p']
    drift_speed = data['drift']["speed"] * 1852 # m/h
    drift = None
    if data['drift']['repair']['active_window'] == 0:
        drift = stats.lognorm(data['drift']['repair']['std'], data['drift']['repair']['loc'], data['drift']['repair']['scale'])
    res = {'tot_sum': 0, 'l':{}, 'o':{}, 'all':{}}
    for lin in clean_traffic(data):
        # TODO: Check that the height and darught are relevant for the ship on the segment
        line_length = get_line_length(lin)
        drift_prob = get_drifting_prob(float(data['drift']['drift_p']), line_length, float(lin['speed']) * 1852)
        distances, _, _, directions = get_multi_drift_distance(lin["geom"], objs, lin["mean"], lin["std"], h, w)
        add_empty_segment_to_res_dict(res, lin)
        for i in range(len(objs)):
            o_name = f"{objs[i]['type']} - {objs[i]['id']}"
            obj_sum = 0
            for dist, direction in zip(distances[i], directions[i]):
                prob_not_repaired = get_not_repaired(data, drift_speed, dist, drift)
                obj_sum += (lin['freq'] * drift_prob * prob_not_repaired * not_anchor * 
                            (data['drift']['rose'][direction] / wind_rose_tot / (w * h)))
            write_res2dict(res, lin, o_name, obj_sum) 
    return res

def get_line_length(lin):
    """Returns the segment length in meters."""
    utm = get_best_utm([lin["geom"]])
    project = Transformer.from_crs(CRS('EPSG:4326'), utm, always_xy=True).transform
    line_length = transform(project, lin["geom"]).length
    return line_length

def load_areas(data):
    """Loads the objects and depths into one objs dict"""
    objs = []
    for id, height, wkt in data['objects']:
        objs.append({'type': 'Structure', 'id': id, 'height': height, 'wkt': sw.loads(wkt)})
    for id, depth, wkt in data['depths']:
        objs.append({'type': 'Depth', 'id': id, 'depth': depth, 'wkt': sw.loads(wkt)})
    return objs

def get_not_repaired(data, drift_speed, dist, drift=None):
    """Get the probability that the ship isn't repaired"""
    drift_time = get_drift_time(dist, drift_speed)
    if data['drift']['repair']['active_window'] == 1:
        x = drift_time # used in the eval func
        prob_not_repaired = 1 - eval(data['drift']['repair']['func'])
    else:
        prob_not_repaired = 1 - drift.cdf(drift_time)
    return prob_not_repaired

def write_res2dict(res, lin, o_name, obj_sum):
    """Write results to the res_dict"""
    if o_name not in res['o']:
        res['o'][o_name] = obj_sum
    else:
        res['o'][o_name] += obj_sum
    res['l'][lin['segment']][lin['direction']] += obj_sum
    res['l'][lin['segment']]['lin_sum'] += obj_sum
    res['all'][f"{lin['segment']} - {lin['direction']}"][o_name] = obj_sum
    res['tot_sum'] += obj_sum

def add_empty_segment_to_res_dict(res, lin):
    """Adds the segment and direction to the res dict"""
    if lin['segment'] not in res['l']:
        res['l'][lin['segment']] = {'lin_sum': 0}
        res['l'][lin['segment']][lin['direction']] = 0
    else:
        res['l'][lin['segment']][lin['direction']] = 0
    res['all'][f"{lin['segment']} - {lin['direction']}"] = {}

def populate_details(twRes: QTableWidget, res_dict: dict):
    twRes.clear()
    twRes.setColumnCount(len(res_dict['o']))
    twRes.setHorizontalHeaderLabels(list(res_dict['o'].keys()))
    twRes.setRowCount(len(res_dict['all']))
    twRes.setVerticalHeaderLabels(list(res_dict['all'].keys()))
    for row, r_key in enumerate(res_dict['all'].keys()):
        for col, c_key in enumerate(res_dict['o'].keys()):
            item = QTableWidgetItem(f"{res_dict['all'][r_key][c_key]:.2e}")
            twRes.setItem(row, col, item)

def populate_segment(tree: QTreeWidget, res_dict):
    tree.clear()
    tree.setColumnCount(3)
    tree.setHeaderLabels(['Segment', 'Direction', 'Probability'])
    tree.setColumnWidth(0, 55)
    tree.setColumnWidth(1, 75)
    tree.setColumnWidth(2, 35)
    main_item = QTreeWidgetItem(tree)
    main_item.setText(0, 'All')
    main_item.setText(2, f"{res_dict['tot_sum']:.2e}")
    for segment, s_val in res_dict['l'].items():
        segment_item = QTreeWidgetItem()
        segment_item.setText(0, segment)
        segment_item.setText(2, f"{s_val['lin_sum']:.2e}")
        main_item.addChild(segment_item)
        # set the child
        for di, val in s_val.items():
            dir_item = QTreeWidgetItem()
            dir_item.setText(1, di)
            dir_item.setText(2, f"{val:.2e}")
            segment_item.addChild(dir_item)
 
def populate_object(tree: QTreeWidget, res_dict):
    tree.clear()
    tree.setColumnCount(2)
    tree.setHeaderLabels(['Object', 'Probability'])
    tree.setColumnWidth(0, 75)
    tree.setColumnWidth(1, 35)
    main_item = QTreeWidgetItem(tree)
    main_item.setText(0, 'All')
    main_item.setText(1, f"{res_dict['tot_sum']:.2e}")
    for object, s_val in res_dict['o'].items():
        item = QTreeWidgetItem()
        item.setText(0, object)
        item.setText(1, f"{s_val:.2e}")
        main_item.addChild(item)

def powered_accidents(data) -> dict:
    width = 100
    max_distance = 25_000
    mean_time = data['pc']['ai']
    objs = load_areas(data)
    res = {'tot_sum': 0, 'l':{}, 'o':{}, 'all':{}}
    for lin in clean_traffic(data):
        distances, lines, points = get_multiple_ed(lin["geom"], objs, lin["mean"], lin["std"], max_distance, width)
        add_empty_segment_to_res_dict(res, lin)
        for i in range(len(objs)):
            o_name = f"{objs[i]['type']} - {objs[i]['id']}"
            powered_na_sum = 0
            for dist in distances[i]:
                if dist == 0:
                    continue
                powered = powered_na(dist, mean_time, float(lin['speed']) * 1852 / 3600)
                powered_na_sum += powered * lin['freq'] * data['pc']['p_pc'] / width
            write_res2dict(res, lin, o_name, powered_na_sum)
    return res

class Calculation:
    def __init__(self, data, height) -> None:
        self.drift_dict = drift_accidents(data, h=height)
        self.powered_dict = powered_accidents(data)

    def run_model(self):
        self.rw = ResultWidget()
        self.rw.show()
        self.rw.cbResType.currentIndexChanged.connect(self.populate)
        self.populate(self.drift_dict)
        self.rw.exec_()

    def populate(self, res_dict: dict = None):
        if res_dict == 0:
            res_dict = self.drift_dict
        if res_dict == 1:
            res_dict = self.powered_dict
        populate_details(self.rw.twRes, res_dict)
        populate_segment(self.rw.treewSegment, res_dict)
        populate_object(self.rw.treewObject, res_dict)


if __name__ == '__main__':
    file_path = 'c:\\Users\\axa\\Documents\\proj.omrat'
    with open(file_path, 'r') as f:
        data = json.load(f)
        drift_accidents(data)
