from _collections_abc import dict_items
import os
from typing import Any, cast, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from omrat import OMRAT

import numpy as np
from qgis.core import QgsProject
from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtWidgets import QTableWidget
from shapely import wkt
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from compute.database import DB
from ui.ais_connection_widget import AISConnectionWidget

def update_ais_settings_file(db_host:str, db_user:str, db_pass:str, db_name:str):
    ais_settings_path = os.path.join(os.path.dirname(__file__), '..', 'ui', 'ais_settings.py')
    with open(ais_settings_path, 'w', encoding='utf-8') as f:
        f.write(f"""db_host = '{db_host}'
db_user = '{db_user}'
db_password = '{db_pass}'
db_name = '{db_name}'
""")


def get_pl(db:DB, lat1:float, lat2:float, lon1:float, lon2:float, l_width:float) -> str:
    """Collects the passage line as text"""
    sql = f"""SELECT st_astext(st_makeline(ST_Project(ST_Centroid(ST_GeomFromText('LINESTRING({lon1} {lat1}, {lon2} {lat2})', 4326))::geography, 
    {l_width/2}, ST_Azimuth(ST_Point({lon1}, {lat1})::geography, ST_Point({lon2}, {lat2})::geography) + radians(90))::geometry,
    ST_Project(ST_Centroid(ST_GeomFromText('LINESTRING({lon1} {lat1}, {lon2} {lat2})', 4326))::geography, 
    {l_width/2}, ST_Azimuth(ST_Point({lon1}, {lat1})::geography, ST_Point({lon2}, {lat2})::geography) + radians(270))::geometry
    ))"""
    ok, res = cast(tuple[bool, list[list[Any]]], db.execute_and_return(sql, return_error=True))
    if ok:
        pl: str = res[0][0]
    else:
        pl = ""
    return pl

def get_type(toc:float, sh_type:str) -> int:
    """Return ship type in accordance with OMRAT."""
    if toc > 79 and toc < 90 and 'product' in sh_type.lower():
        type__cat = 1
    elif toc > 79 and toc < 90 and 'chemical' in sh_type.lower():
        type__cat = 2
    elif toc > 79 and toc < 90 and (
            'gas' in sh_type.lower() or 'lng' in sh_type.lower()):
        type__cat = 3
    elif toc > 79 and toc < 90:
        type__cat = 0
    elif toc > 69 and toc < 80 and 'container' in sh_type.lower():
        type__cat = 4
    elif toc > 69 and toc < 80 and 'bulk' in sh_type.lower():
        type__cat = 6
    elif toc > 69 and toc < 80 and 'ro-ro' in sh_type.lower() and 'passenger' not in sh_type.lower():
        type__cat = 7
    elif toc > 69 and toc < 80:
        type__cat = 5
    elif toc > 59 and toc < 70:
        type__cat = 8
    elif toc == 36 or toc == 37:
        type__cat = 12
    elif (toc > 32 and toc < 40) or (toc > 49 and toc < 60):
        type__cat = 9
    elif toc > 40 and toc < 50:
        type__cat = 10
    elif toc == 30:
        type__cat = 11
    else:
        type__cat = 13
    return type__cat

def close_to_line(bearing:float, cog:float, max_angle:float) -> bool:
    """Returns True if the cog is less than the max_angle towards the bearing else False"""
    if max_angle > 180:
        raise ValueError("Maximum deviation must be lower than 180 degrees")
        return
    if bearing > 360:
        bearing = np.mod(bearing, 360)
    if cog > 360:
        cog = np.mod(cog, 360)
    if bearing - max_angle > 0 and bearing + max_angle < 360:
        if bearing - max_angle < cog < bearing + max_angle:
            return True
        else:
            return False
    else:
        if np.mod(bearing - max_angle, 360) < cog or cog < np.mod(
                bearing + max_angle, 360):
            return True
        else:
            return False

class AIS:
    def __init__(self, omrat:"OMRAT"):
        self.settings: QSettings = QSettings()
        self.db: DB | None = None
        self.omrat = omrat
        self.acw = AISConnectionWidget()
        self.set_start_ais_settings()
        self.dist_data: dict[str, dict[str, np.ndarray]] = {}
        
    def unload(self):
        """If signals from omrat was used they are disconnected"""
        pass
        
    def run(self):
        self.acw.show()
        self.acw.accepted.connect(self.update_ais_settings)
        self.acw.exec_()
    
    def set_start_ais_settings(self):
        db_host: str = self.settings.value("omrat/db_host", "")
        db_name: str = self.settings.value("omrat/db_name", "")
        db_user: str = self.settings.value("omrat/db_user", "")
        db_pass: str = self.settings.value("omrat/db_pass", "")
        self.acw.leDBHost.setText(db_host)
        self.acw.leDBName.setText(db_name)
        self.acw.leUserName.setText(db_user)
        self.acw.lePassword.setText(db_pass)
        self.schema = self.acw.leProvider.text()
        self.year = self.acw.SBYear.value()
        self.months: list[int] = []
        for i in range(1, 13):
            cb = getattr(self.acw, f'CB_{i}')
            if cb.isChecked():
                self.months.append(i)
        try:
            self.db = DB(db_host=db_host, 
                        db_name=db_name, 
                        db_user=db_user, 
                        db_pass=db_pass)
        except Exception:
            self.db = None
        self.max_deviation = float(self.acw.leMaxDev.text())
    
    def update_ais_settings(self):
        db_host = self.acw.leDBHost.text()
        db_name = self.acw.leDBName.text()
        db_user = self.acw.leUserName.text()
        db_pass = self.acw.lePassword.text()
        self.schema = self.acw.leProvider.text()
        self.year = self.acw.SBYear.value()
        self.months = []
        for i in range(1, 13):
            cb = getattr(self.acw, f'CB_{i}')
            if cb.isChecked():
                self.months.append(i)
        self.db = DB(db_host=db_host, db_name=db_name, db_user=db_user, db_pass=db_pass)
        self.max_deviation = float(self.acw.leMaxDev.text())
        for key, value in zip(["db_host", "db_user", "db_pass", "db_name"], [db_host, db_user, db_pass, db_name]):
            self.settings.setValue(f"omrat/{key}", value)
        
    def update_legs(self, key:str|None=None):
        """Update AIS data for all legs or a specific leg."""
        if self.db is None:
            return
        segment_data: dict[str, dict[str, str]] = self.get_segment_data_from_table()
        if key is None:
            legs: dict_items[str, dict[str, str]] = segment_data.items()
        else:
            legs = [[key, segment_data[key]]] # type: ignore
        for leg_key, leg_d in legs:
            if leg_key not in self.omrat.qgis_geoms.leg_dirs.keys():
                # Use the current leg_key to look up segment direction labels
                self.omrat.qgis_geoms.leg_dirs[leg_key] = self.omrat.segment_data[leg_key]["Dirs"]
            dirs = self.omrat.qgis_geoms.leg_dirs[leg_key]
            self.omrat.traffic.create_empty_dict(leg_key, dirs)
            start_p = wkt.loads(f"Point ({leg_d['Start_Point']})")
            assert isinstance(start_p, Point)
            end_p = wkt.loads(f"Point ({leg_d['End_Point']})")
            assert isinstance(end_p, Point)
            pl = get_pl(self.db, lat1=start_p.y,
                        lat2=end_p.y,
                        lon1=start_p.x,
                        lon2=end_p.x, 
                        l_width=float(leg_d['Width']))
            try:
                ais_data = self.run_sql(pl)
            except Exception as e:
                self.omrat.show_error_popup(str(e), "AIS.update_legs")
                return
            sql = f"""select degrees(ST_Azimuth(ST_Point({start_p.x}, {start_p.y})::geography, 
            ST_Point({end_p.x}, {end_p.y})::geography))
            """
            _, leg_bearing = cast(tuple[bool, list[list[Any]]], self.db.execute_and_return(sql, return_error=True))
            line1, line2 = self.update_ais_data(leg_key, ais_data, leg_bearing[0][0], dirs)
            self.convert_list2avg()
            self.update_dist_data(line1, line2, leg_key)
            key = leg_key
        if key is not None:
            self.omrat.main_widget.leNormMean1_1.setText('')
            self.omrat.distributions.run_update_plot(key)
            self.omrat.main_widget.cbTrafficSelectSeg.setCurrentIndex(self.omrat.main_widget.cbTrafficSelectSeg.count()-1)
        
    def convert_list2avg(self):
        for key1 in self.omrat.traffic.traffic_data.keys():
            for key2 in self.omrat.traffic.traffic_data[key1].keys():
                for key3 in self.omrat.traffic.traffic_data[key1][key2].keys():
                    for idx1, _ in enumerate(self.omrat.traffic.traffic_data[key1][key2][key3]):
                        for idx2, val in enumerate(self.omrat.traffic.traffic_data[key1][key2][key3][idx1]):
                            if isinstance(val, list):
                                if len(val) > 0:
                                    self.omrat.traffic.traffic_data[key1][key2][key3][idx1][idx2] = np.array(val).mean()
                                else:
                                    self.omrat.traffic.traffic_data[key1][key2][key3][idx1][idx2] = np.inf

    def update_dist_data(self, line1:np.ndarray, line2:np.ndarray, key:str) -> None:
        self.omrat.segment_data[key]['mean1_1'] = line1.mean()
        self.omrat.segment_data[key]['std1_1'] = line1.std()
        self.omrat.segment_data[key]['mean2_1'] = line2.mean()
        self.omrat.segment_data[key]['std2_1'] = line2.std()
        self.omrat.segment_data[key]['weight1_1'] = 100
        self.omrat.segment_data[key]['weight2_1'] = 100
        self.omrat.segment_data[key]['dist1'] = line1
        self.omrat.segment_data[key]['dist2'] = line2
        if float(self.omrat.main_widget.leNormMean1_1.text()) == 0.0:
            self.omrat.main_widget.leNormMean1_1.setText(str(line1.mean()))
            self.omrat.main_widget.leNormMean2_1.setText(str(line2.mean()))
            self.omrat.main_widget.leNormStd1_1.setText(str(line1.std()))
            self.omrat.main_widget.leNormStd2_1.setText(str(line2.std()))
        for j in range(1, 3):
            for i in range(2, 4):
                self.omrat.segment_data[key][f'mean{j}_{i}'] = 0
                self.omrat.segment_data[key][f'std{j}_{i}'] = 0
                self.omrat.segment_data[key][f'weight{j}_{i}'] = 0
            self.omrat.segment_data[key][f'u_min{j}'] = 0
            self.omrat.segment_data[key][f'u_max{j}'] = 0
            self.omrat.segment_data[key][f'u_p{j}'] = 0
        self.omrat.segment_data[key]['ai1'] = 180
        self.omrat.segment_data[key]['ai2'] = 180
    
    def run_sql(self, pl:str) -> list[list[Any]]:
        """Runs the SQL query to get the passages"""
        if self.db is None:
            # Should not occur
            raise RuntimeError(self.omrat.tr("No database connection was found"))
        sql = "with segments as ("
        if len(self.months) > 0:
            for month in self.months:
                sql += f"""select ss.mmsi, segment, cog, sog, draught, type_and_cargo, date1, dim_a, dim_b, dim_c, dim_d
                FROM {self.schema}.segments_{self.year}_{month} ss
                JOIN {self.schema}.states_{self.year} st on st.rowid=ss.state_id
                JOIN {self.schema}.statics_{self.year} si on si.rowid=st.static_id
                WHERE ST_intersects(segment, ST_geomfromtext('{pl}', 4326))
                UNION """
        else:
            sql += f"""select ss.mmsi, segment, cog, sog, draught, type_and_cargo, date1, dim_a, dim_b, dim_c, dim_d
                                FROM {self.schema}.segments_{self.year} ss
                                JOIN {self.schema}.states_{self.year} st on st.rowid=ss.state_id
                                JOIN {self.schema}.statics_{self.year} si on si.rowid=st.static_id
                                WHERE ST_intersects(segment, ST_geomfromtext('{pl}', 4326))
            """

        sql = sql[:-6] + f"""), get_vessel_info as(select mmsi, ship_type, loa, breadth_moulded as beam, height as air_draught
        FROM vessels.seaweb_data)
        SELECT case when dim_a + dim_b < 2 or dim_a > 510 or dim_b > 510 then loa else dim_a + dim_b end as loa, case when dim_c + dim_d < 2 or dim_c > 62 or dim_d > 62 then loa else dim_c + dim_d end as beam, type_and_cargo, draught, ship_type, date1, sog, air_draught, st_distance(st_intersection(segment, st_geomfromtext('{pl}',4326))::geography, st_startpoint(st_geomfromtext('{pl}', 4326))::geography)-st_length(st_geomfromtext('{pl}', 4326)::geography)/2 as dist_from_start, cog
        FROM segments ss
        left outer JOIN get_vessel_info sd on ss.mmsi=sd.mmsi
        """
        ok, ais_data = cast(tuple[bool, list[list[Any]]], self.db.execute_and_return(sql, return_error=True))
        if ok:
            return ais_data
        else:
            raise TypeError(ais_data[0][0])

    def update_ais_data(self, leg_key:str, ais_data:list[list[Any]], leg_bearing:float, dirs:list[str]) -> tuple[np.ndarray, np.ndarray]:
        line1:list[float] = []
        line2:list[float] = []
        for loa, beam, toc, draugt, sh_type, _, sog, air_draught, dist, cog in ais_data:
            if close_to_line(leg_bearing + 180, cog, self.max_deviation):
                line1.append(dist)
                l1 = True
            elif close_to_line(leg_bearing, cog, self.max_deviation):
                line2.append(dist)
                l1 = False
            else:
                # print(leg_bearing, cog, self.max_deviation)
                continue
            dir_ = dirs[0] if l1 else dirs[1]
            if loa is None:
                loa = 100
            loa = int(loa)
            loa_cat = -1
            for loa_i in range(len(self.omrat.traffic.traffic_data[leg_key][dir_]['Frequency (ships/year)'])):
                if loa_i * 25 < loa <= (loa_i * 25 + 25):
                    loa_cat = loa_i
                    continue
            if loa_cat < 0:
                loa_cat = 4
            if sh_type is None:
                sh_type = ''
            type_cat = get_type(toc, sh_type)
            self.omrat.traffic.traffic_data[leg_key][dir_]['Frequency (ships/year)'][loa_cat][type_cat] += 1
            if sog is not None:
                self.omrat.traffic.traffic_data[leg_key][dir_]['Speed (knots)'][loa_cat][type_cat].append(float(sog))
            if air_draught is not None:
                self.omrat.traffic.traffic_data[leg_key][dir_]['Ship heights (meters)'][loa_cat][type_cat].append(
                    float(air_draught))
            if beam is not None:
                self.omrat.traffic.traffic_data[leg_key][dir_]['Ship Beam (meters)'][loa_cat][type_cat].append(float(beam))
            if draugt is not None:
                self.omrat.traffic.traffic_data[leg_key][dir_]['Draught (meters)'][loa_cat][type_cat].append(
                    float(draugt))
        np_line1 = np.array(line1)
        np_line2 = np.array(line2)
        return np_line1, np_line2

    def get_segment_data_from_table(self) -> dict[str, dict[str, str]]:
        """Extract segment data from the QTableWidget (twRouteList)."""
        segment_data: dict[str, dict[str, str]] = {}
        table:QTableWidget = self.omrat.main_widget.twRouteList
        row_count: int = table.rowCount()

        for row in range(row_count):
            segment_id: str | None = table.item(row, 0).text() if table.item(row, 0) else None
            route_id: str | None = table.item(row, 1).text() if table.item(row, 1) else None
            leg_name: str | None = table.item(row, 2).text() if table.item(row, 2) else None
            start_point: str | None = table.item(row, 3).text() if table.item(row, 3) else None
            end_point: str | None = table.item(row, 4).text() if table.item(row, 4) else None
            width: str|None = table.item(row, 5).text() if table.item(row, 5) else None

            if segment_id and route_id and start_point and end_point and width:
                segment_data[segment_id] = {
                    'Route_Id': route_id,
                    'Leg_name': leg_name,
                    'Start_Point': start_point,
                    'End_Point': end_point,
                    'Width': width
                }

        return segment_data
