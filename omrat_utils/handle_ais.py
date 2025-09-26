import os

import numpy as np
from shapely import wkt

from compute.database import DB
from ui.ais_connection_widget import AISConnectionWidget
from ui import ais_settings

def update_ais_settings_file(db_host:str, db_user:str, db_pass:str, db_name:str):
    ais_settings_path = os.path.join(os.path.dirname(__file__), '..', 'ui', 'ais_settings.py')
    with open(ais_settings_path, 'w', encoding='utf-8') as f:
        f.write(f"""db_host = '{db_host}'
db_user = '{db_user}'
db_password = '{db_pass}'
db_name = '{db_name}'
""")


def get_pl(db, lat1, lat2, lon1, lon2, l_width):
    """Collects the passage line as text"""
    sql = f"""SELECT st_astext(st_makeline(ST_Project(ST_Centroid(ST_GeomFromText('LINESTRING({lon1} {lat1}, {lon2} {lat2})', 4326))::geography, 
    {l_width/2}, ST_Azimuth(ST_Point({lon1}, {lat1})::geography, ST_Point({lon2}, {lat2})::geography) + radians(90))::geometry,
    ST_Project(ST_Centroid(ST_GeomFromText('LINESTRING({lon1} {lat1}, {lon2} {lat2})', 4326))::geography, 
    {l_width/2}, ST_Azimuth(ST_Point({lon1}, {lat1})::geography, ST_Point({lon2}, {lat2})::geography) + radians(270))::geometry
    ))"""
    pl = db.execute_and_return(sql)[0][0]
    return pl

def get_type(toc, sh_type):
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

def close_to_line(bearing, cog, max_angle) -> bool:
    """Returns True if the cog is less than the max_angle towards the bearing else False"""
    if max_angle > 180:
        return 'Error'
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
    def __init__(self, omrat):
        self.db = DB()
        self.omrat = omrat
        self.acw = AISConnectionWidget()
        self.set_start_ais_settings()
        self.dist_data = {}
        
    def unload(self):
        """If signals from omrat was used they are disconnected"""
        pass
        
    def run(self):
        self.acw.show()
        self.acw.accepted.connect(self.update_ais_settings)
        self.acw.exec_()
    
    def set_start_ais_settings(self):
        self.acw.leDBHost.setText(ais_settings.db_host)
        self.acw.leDBName.setText(ais_settings.db_name)
        self.acw.leUserName.setText(ais_settings.db_user)
        self.acw.lePassword.setText(ais_settings.db_password)
        self.schema = self.acw.leProvider.text()
        self.year = self.acw.SBYear.value()
        self.months = []
        for i in range(1, 13):
            cb = getattr(self.acw, f'CB_{i}')
            if cb.isChecked():
                self.months.append(i)
        try:
            self.db = DB(db_host=ais_settings.db_host, 
                        db_name=ais_settings.db_name, 
                        db_user=ais_settings.db_user, 
                        db_pass=ais_settings.db_password)
        except Exception:
            pass
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
        update_ais_settings_file(db_host, db_user, db_pass, db_name)
        
    def update_legs(self, key=None, leg_d=None):
        """Update AIS data for all legs or a specific leg."""
        segment_data = self.get_segment_data_from_table()
        legs = [(key, leg_d)] if key and leg_d else segment_data.items()
        print(legs)
        for key, leg_d in legs:
            dirs = self.omrat.qgis_geoms.leg_dirs[key]
            self.omrat.traffic.create_empty_dict(key, dirs)
            start_p = wkt.loads(f"POINT({leg_d['Start Point']})")
            end_p = wkt.loads(f"POINT({leg_d['End Point']})")
            pl = get_pl(self.db, lat1=start_p.y,
                        lat2=end_p.y,
                        lon1=start_p.x,
                        lon2=end_p.x, 
                        l_width=leg_d['Width'])
            ais_data = self.run_sql(pl)
            sql = f"""select degrees(ST_Azimuth(ST_Point({start_p.x}, {start_p.y})::geography, 
            ST_Point({end_p.x}, {end_p.y})::geography))
            """
            leg_bearing = self.db.execute_and_return(sql)[0][0]
            line1, line2 = self.update_ais_data(key, ais_data, leg_bearing, dirs)
            self.convert_list2avg()
            self.update_dist_data(line1, line2, key)
        self.omrat.traffic.dont_save = True
        self.omrat.traffic.run_update_plot()
        
    def convert_list2avg(self):
        for key1 in self.omrat.traffic.traffic_data.keys():
            for key2 in self.omrat.traffic.traffic_data[key1].keys():
                for key3 in self.omrat.traffic.traffic_data[key1][key2].keys():
                    for idx1, row in enumerate(self.omrat.traffic.traffic_data[key1][key2][key3]):
                        for idx2, val in enumerate(self.omrat.traffic.traffic_data[key1][key2][key3][idx1]):
                            if isinstance(val, list):
                                if len(val) > 0:
                                    self.omrat.traffic.traffic_data[key1][key2][key3][idx1][idx2] = np.array(val).mean()
                                else:
                                    self.omrat.traffic.traffic_data[key1][key2][key3][idx1][idx2] = np.inf

    def update_dist_data(self, line1, line2, key):
        self.dist_data[key] = {'line1': line1, 'line2': line2}
        self.omrat.segment_data[key]['mean1_1'] = line1.mean()
        self.omrat.segment_data[key]['std1_1'] = line1.std()
        self.omrat.segment_data[key]['mean2_1'] = line2.mean()
        self.omrat.segment_data[key]['std2_1'] = line2.std()
        self.omrat.segment_data[key]['weight1_1'] = 100
        self.omrat.segment_data[key]['weight2_1'] = 100
        if float(self.omrat.dockwidget.leNormMean1_1.text()) == 0.0:
            self.omrat.dockwidget.leNormMean1_1.setText(str(line1.mean()))
            self.omrat.dockwidget.leNormMean2_1.setText(str(line2.mean()))
            self.omrat.dockwidget.leNormStd1_1.setText(str(line1.std()))
            self.omrat.dockwidget.leNormStd2_1.setText(str(line2.std()))
    
    def run_sql(self, pl):
        """Runs the SQL query to get the passages"""
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
        ais_data = self.db.execute_and_return(sql)
        return ais_data

    def update_ais_data(self, leg_key, ais_data, leg_bearing, dirs) -> list:
        line1 = []
        line2 = []
        for loa, beam, toc, draugt, sh_type, date1, sog, air_draught, dist, cog in ais_data:
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
            found_loa = False
            if loa is None:
                loa = 100
            loa = int(loa)
            for loa_i in range(len(self.omrat.traffic.traffic_data[leg_key][dir_]['Frequency (ships/year)'])):
                if loa_i * 25 < loa <= (loa_i * 25 + 25):
                    loa_cat = loa_i
                    found_loa = True
                    continue
            if not found_loa:
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
        line1 = np.array(line1)
        line2 = np.array(line2)
        return [line1, line2]

    def get_segment_data_from_table(self):
        """Extract segment data from the QTableWidget (twRouteList)."""
        segment_data = {}
        table = self.omrat.dockwidget.twRouteList
        row_count = table.rowCount()

        for row in range(row_count):
            segment_id = table.item(row, 0).text() if table.item(row, 0) else None
            route_id = table.item(row, 1).text() if table.item(row, 1) else None
            start_point = table.item(row, 2).text() if table.item(row, 2) else None
            end_point = table.item(row, 3).text() if table.item(row, 3) else None
            width = table.item(row, 4).text() if table.item(row, 4) else None

            if segment_id and route_id and start_point and end_point and width:
                segment_data[segment_id] = {
                    'Route Id': route_id,
                    'Start Point': start_point,
                    'End Point': end_point,
                    'Width': float(width)
                }

        return segment_data
