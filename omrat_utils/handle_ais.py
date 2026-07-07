import os
import re
from typing import Any, cast, TYPE_CHECKING
if TYPE_CHECKING:
    from omrat import OMRAT

import numpy as np
from qgis.core import QgsApplication
from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtWidgets import QMessageBox, QTableWidget
from shapely import wkt
from shapely.geometry import Point

from compute.database import DB
from omrat_utils.vessel_lookup import VesselLookupConfig
from ui.ais_connection_widget import AISConnectionWidget


_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_AIS_SELECT_TEMPLATE = (
    "select ss.mmsi, segment, cog, sog, draught, type_and_cargo, "
    "date1, dim_a, dim_b, dim_c, dim_d "
    "FROM {tbl} ss "
    "JOIN {schema}.states_{year} st on st.rowid=ss.state_id "
    "JOIN {schema}.statics_{year} si on si.rowid=st.static_id "
    "WHERE ST_intersects(segment, ST_geomfromtext(%(pl)s, 4326))"
)


def _build_ais_union_block(schema: str, year: int, months: list[int]) -> str:
    if months:
        tables = [f"{schema}.segments_{year}_{month}" for month in months]
    else:
        tables = [f"{schema}.segments_{year}"]
    return " UNION ".join(
        _AIS_SELECT_TEMPLATE.format(tbl=tbl, schema=schema, year=year)
        for tbl in tables
    )


def _assemble_ais_query(
    union_block: str, cte_block: str, ext_join: str,
    loa_fb: str, beam_fb: str, ship_type_expr: str, air_draught_expr: str,
) -> str:
    return (  # nosec B608
        "with segments as (" + union_block + ")" + cte_block + " "
        "SELECT case when dim_a + dim_b < 2 or dim_a > 510 or dim_b > 510 "
        f"then {loa_fb} else dim_a + dim_b end as loa, "
        "case when dim_c + dim_d < 2 or dim_c > 62 or dim_d > 62 "
        f"then {beam_fb} else dim_c + dim_d end as beam, "
        "type_and_cargo, draught, "
        f"{ship_type_expr}, "
        "date1, sog, "
        f"{air_draught_expr}, "
        "st_distance("
        "st_intersection(segment, st_geomfromtext(%(pl)s,4326))::geography, "
        "st_startpoint(st_geomfromtext(%(pl)s, 4326))::geography"
        ") - st_length(st_geomfromtext(%(pl)s, 4326)::geography) / 2 "
        "as dist_from_start, cog "
        "FROM segments ss"
        + ext_join
    )


def _validate_sql_identifier(name: str) -> str:
    """Reject anything that isn't a plain SQL identifier.

    Used so that the AIS schema name (which the user types in the GUI) can be
    safely interpolated into table references — table/schema names cannot be
    bound as query parameters, so we whitelist instead.
    """
    if not isinstance(name, str) or not _SQL_IDENT_RE.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name

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

def get_type(toc: float) -> int:
    """Return ship type index matching the UI ship category list.

    Maps AIS Type-of-Cargo (TOC) codes to indices 0-20 corresponding to:
        0: Fishing (TOC 30)
        1: Towing (TOC 31-32)
        2: Dredging or underwater ops (TOC 33)
        3: Diving ops (TOC 34)
        4: Military ops (TOC 35)
        5: Sailing (TOC 36)
        6: Pleasure Craft (TOC 37)
        7: High speed craft (TOC 40-49)
        8: Pilot Vessel (TOC 50)
        9: Search and Rescue vessel (TOC 51)
        10: Tug (TOC 52)
        11: Port Tender (TOC 53)
        12: Anti-pollution equipment (TOC 54)
        13: Law Enforcement (TOC 55)
        14: Spare (TOC 56-57)
        15: Medical Transport (TOC 58)
        16: Noncombatant ship (TOC 59)
        17: Passenger (TOC 60-69)
        18: Cargo (TOC 70-79)
        19: Tanker (TOC 80-89)
        20: Other Type (everything else)
    """
    # NULL / unparseable type_and_cargo is common — any MMSI whose Type-5
    # statics never came through arrives here as None.  Bucket those into
    # "Other Type" rather than crashing the whole leg's traffic build.
    if toc is None:
        return 20
    try:
        toc_int = int(toc)
    except (TypeError, ValueError):
        return 20
    _TOC_MAP = {
        30: 0, 31: 1, 32: 1, 33: 2, 34: 3, 35: 4, 36: 5, 37: 6,
        50: 8, 51: 9, 52: 10, 53: 11, 54: 12, 55: 13,
        56: 14, 57: 14, 58: 15, 59: 16,
    }
    if toc_int in _TOC_MAP:
        return _TOC_MAP[toc_int]
    if 40 <= toc_int <= 49:
        return 7
    if 60 <= toc_int <= 69:
        return 17
    if 70 <= toc_int <= 79:
        return 18
    if 80 <= toc_int <= 89:
        return 19
    return 20

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
        # Optional LEFT JOIN against an external vessel-metadata table.
        # The default-constructed config has ``enabled=False`` so
        # ``is_valid()`` is False and ``run_sql`` skips the JOIN.
        self.vessel_lookup: VesselLookupConfig = VesselLookupConfig.from_qsettings()
        # When True, ``update_legs`` divides the year-of-seconds by the
        # actual coverage seen in the data and uses that as a multiplier
        # on the per-ping frequency increment, so a partial-year ingest
        # is reported as an annualised rate.
        self.recalc_to_full_year: bool = self._read_recalc_setting()
        self.set_start_ais_settings()
        self.dist_data: dict[str, dict[str, np.ndarray]] = {}

    def _read_recalc_setting(self) -> bool:
        """Read ``omrat/recalc_to_full_year`` and coerce to bool.

        QSettings on Windows stores booleans as the strings ``"true"`` /
        ``"false"`` rather than Python bools, so we accept either.
        """
        raw = self.settings.value("omrat/recalc_to_full_year", False)
        if isinstance(raw, str):
            return raw.strip().lower() == "true"
        return bool(raw)
        
    def unload(self):
        """If signals from omrat was used they are disconnected"""
        pass
        
    def run(self):
        self.acw.show()
        self.acw.accepted.connect(self.update_ais_settings)
        self.acw.exec()
    
    def set_start_ais_settings(self):
        db_host: str = self.settings.value("omrat/db_host", "")
        db_name: str = self.settings.value("omrat/db_name", "")
        db_user: str = self.settings.value("omrat/db_user", "")
        db_pass: str = self.settings.value("omrat/db_pass", "")
        # ``omrat/db_port`` is new (older installs won't have it).  Fall back
        # to 5432, the Postgres default.
        try:
            db_port = int(self.settings.value("omrat/db_port", 5432) or 5432)
        except (TypeError, ValueError):
            db_port = 5432
        self.acw.leDBHost.setText(db_host)
        self.acw.SBPort.setValue(db_port)
        self.acw.leDBName.setText(db_name)
        self.acw.leUserName.setText(db_user)
        self.acw.lePassword.setText(db_pass)
        # External vessel-data lookup: prefill from QSettings.
        self.acw.gbExtVessel.setChecked(bool(self.vessel_lookup.enabled))
        self.acw.leExtSchema.setText(self.vessel_lookup.schema)
        self.acw.leExtTable.setText(self.vessel_lookup.table)
        self.acw.leExtMmsiCol.setText(self.vessel_lookup.mmsi_col or "mmsi")
        self.acw.leExtLoaCol.setText(self.vessel_lookup.loa_col)
        self.acw.leExtBeamCol.setText(self.vessel_lookup.beam_col)
        self.acw.leExtShipTypeCol.setText(self.vessel_lookup.ship_type_col)
        self.acw.leExtAirDraughtCol.setText(self.vessel_lookup.air_draught_col)
        self.acw.cbRecalcFullYear.setChecked(bool(self.recalc_to_full_year))
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
                        db_pass=db_pass,
                        db_port=db_port)
        except Exception:
            self.db = None
        self.max_deviation = float(self.acw.leMaxDev.text())
    
    def _read_months_from_ui(self) -> list[int]:
        months = []
        for i in range(1, 13):
            if getattr(self.acw, f'CB_{i}').isChecked():
                months.append(i)
        return months

    def _connect_db_from_ui(self, db_host, db_port, db_name, db_user, db_pass) -> bool:
        try:
            self.db = DB(db_host=db_host, db_name=db_name, db_user=db_user, db_pass=db_pass, db_port=db_port)
            return True
        except Exception as e:
            self.db = None
            QMessageBox.warning(self.omrat.main_widget, self.omrat.tr("Could not connect to AIS database"), str(e))
            return False

    def _capture_vessel_lookup_from_ui(self) -> None:
        self.vessel_lookup = VesselLookupConfig(
            enabled=bool(self.acw.gbExtVessel.isChecked()),
            schema=self.acw.leExtSchema.text().strip(),
            table=self.acw.leExtTable.text().strip(),
            mmsi_col=self.acw.leExtMmsiCol.text().strip() or "mmsi",
            loa_col=self.acw.leExtLoaCol.text().strip(),
            beam_col=self.acw.leExtBeamCol.text().strip(),
            ship_type_col=self.acw.leExtShipTypeCol.text().strip(),
            air_draught_col=self.acw.leExtAirDraughtCol.text().strip(),
        )
        self.vessel_lookup.to_qsettings()

    def update_ais_settings(self):
        db_host = self.acw.leDBHost.text()
        db_port = int(self.acw.SBPort.value())
        db_name = self.acw.leDBName.text()
        db_user = self.acw.leUserName.text()
        db_pass = self.acw.lePassword.text()
        self.schema = self.acw.leProvider.text()
        self.year = self.acw.SBYear.value()
        self.months = self._read_months_from_ui()
        if not self._connect_db_from_ui(db_host, db_port, db_name, db_user, db_pass):
            return
        self.max_deviation = float(self.acw.leMaxDev.text())
        for key, value in zip(
            ["db_host", "db_port", "db_user", "db_pass", "db_name"],
            [db_host, db_port, db_user, db_pass, db_name],
        ):
            self.settings.setValue(f"omrat/{key}", value)
        self._capture_vessel_lookup_from_ui()
        self.recalc_to_full_year = bool(self.acw.cbRecalcFullYear.isChecked())
        self.settings.setValue("omrat/recalc_to_full_year", self.recalc_to_full_year)

    # ----------------------------------------------------------- coverage

    # Year-of-seconds (non-leap) — the reference window the multiplier
    # normalises to.  The 6 h discrepancy from a leap year is well below
    # the resolution of "ships per year" estimates and not worth a
    # branch.
    _YEAR_SECONDS = 365 * 24 * 3600

    # Gaps in the data longer than this are treated as receiver outages
    # rather than valid empty periods, and excluded from the coverage
    # window.  12 h matches the user's spec.
    _GAP_THRESHOLD_HOURS = 12

    def compute_year_multiplier(self) -> tuple[float, float, float] | None:
        """Inspect the segments table and return ``(multiplier, coverage_s, gap_s)``.

        ``coverage_s`` = ``max(date1) - min(date1)`` minus the time spent
        inside any inter-ping gap longer than
        ``_GAP_THRESHOLD_HOURS``.  ``multiplier`` =
        ``_YEAR_SECONDS / coverage_s`` so callers can scale a per-ping
        count to an annualised rate.

        Returns ``None`` when the DB is unset, the query fails, or the
        table is empty (no rows → no coverage to extrapolate from).
        """
        if self.db is None:
            return None
        from psycopg2 import sql as psql

        schema = _validate_sql_identifier(str(self.schema))
        year = int(self.year)
        gap_h = int(self._GAP_THRESHOLD_HOURS)
        sql = psql.SQL(
            "WITH ordered AS ("
            "SELECT date1, date1 - LAG(date1) OVER (ORDER BY date1) AS dt "
            "FROM {schema}.{segments}"
            ") "
            "SELECT EXTRACT(EPOCH FROM (MAX(date1) - MIN(date1))), "
            "COALESCE(SUM(EXTRACT(EPOCH FROM dt)) "
            "FILTER (WHERE dt > make_interval(hours => %s)), 0) "
            "FROM ordered"
        ).format(
            schema=psql.Identifier(schema),
            segments=psql.Identifier(f"segments_{year}"),
        )
        ok, rows = cast(
            tuple[bool, list[list[Any]]],
            self.db.execute_and_return(sql, return_error=True, params=(gap_h,)),
        )
        if not ok or not rows or rows[0][0] is None:
            return None
        try:
            span_s = float(rows[0][0])
            gap_s = float(rows[0][1])
        except (TypeError, ValueError):
            return None
        coverage_s = span_s - gap_s
        if coverage_s <= 0:
            return None
        multiplier = self._YEAR_SECONDS / coverage_s
        return multiplier, coverage_s, gap_s

    def _ensure_leg_dirs(self, legs: dict) -> dict[str, list[str]]:
        for leg_key in legs:
            if leg_key not in self.omrat.qgis_geoms.leg_dirs:
                self.omrat.qgis_geoms.leg_dirs[leg_key] = self.omrat.segment_data[leg_key]["Dirs"]
        return {k: list(v) for k, v in self.omrat.qgis_geoms.leg_dirs.items()}

    def update_legs(self, key: str | None = None) -> None:
        """Launch a background task to fetch AIS data for all legs or one leg."""
        if self.db is None:
            QMessageBox.information(
                self.omrat.main_widget,
                self.omrat.tr("AIS database not connected"),
                self.omrat.tr(
                    "No AIS database connection is configured.\n\n"
                    "Open Settings -> AIS Connection and provide a host, "
                    "database, user and password before fetching AIS traffic."
                ),
            )
            return
        seg_table: dict[str, dict[str, str]] = self.get_segment_data_from_table()
        legs: dict[str, dict[str, str]] = (
            {key: seg_table[key]} if key is not None else dict(seg_table)
        )
        if not legs:
            return
        leg_dirs = self._ensure_leg_dirs(legs)
        tw = self.omrat.main_widget.twTrafficData
        from omrat_utils.ais_update_task import AisUpdateTask
        task = AisUpdateTask(
            ais=self,
            legs=legs,
            seg_table=seg_table,
            rows=tw.rowCount(),
            cols=tw.columnCount(),
            variables=list(self.omrat.traffic.variables),
            var_defaults=dict(self.omrat.traffic._var_cell_defaults),
            leg_dirs=leg_dirs,
        )
        btn = getattr(self.omrat.main_widget, 'pbUpdateAIS', None)
        if btn is not None:
            btn.setEnabled(False)
        QgsApplication.taskManager().addTask(task)
        
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
        # Keep a dedicated distribution cache used by Distributions.run_update_plot.
        self.dist_data[key] = {
            'line1': line1,
            'line2': line2
        }
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
    
    def _validate_months(self) -> list[int]:
        months: list[int] = []
        for m in self.months:
            mi = int(m)
            if not 1 <= mi <= 12:
                raise ValueError(f"Invalid AIS month value: {m!r}")
            months.append(mi)
        return months

    def _build_ext_vessel_fragments(self) -> tuple[str, str, str, str, str, str]:
        if not self.vessel_lookup.is_valid():
            return "", "", "NULL", "NULL", "NULL::int as ship_type", "NULL::double precision as air_draught"
        for ident in (
            self.vessel_lookup.schema, self.vessel_lookup.table,
            self.vessel_lookup.mmsi_col, self.vessel_lookup.loa_col,
            self.vessel_lookup.beam_col, self.vessel_lookup.ship_type_col,
            self.vessel_lookup.air_draught_col,
        ):
            if ident:
                _validate_sql_identifier(ident)
        cte_block = ", " + self.vessel_lookup.build_cte()
        ext_join = " LEFT OUTER JOIN external_vessels ext ON ss.mmsi = ext.mmsi"
        return (
            cte_block, ext_join,
            "ext.ext_loa", "ext.ext_beam",
            "ext.ext_ship_type as ship_type",
            "ext.ext_air_draught as air_draught",
        )

    def run_sql(self, pl: str) -> list[list[Any]]:
        """Runs the SQL query to get the passages."""
        if self.db is None:
            raise RuntimeError(self.omrat.tr("No database connection was found"))
        schema = _validate_sql_identifier(str(self.schema))
        year = int(self.year)
        months = self._validate_months()
        union_block = _build_ais_union_block(schema, year, months)
        cte_block, ext_join, loa_fb, beam_fb, ship_type_expr, air_draught_expr = (
            self._build_ext_vessel_fragments()
        )
        sql = _assemble_ais_query(
            union_block, cte_block, ext_join, loa_fb, beam_fb, ship_type_expr, air_draught_expr,
        )
        ok, ais_data = cast(
            tuple[bool, list[list[Any]]],
            self.db.execute_and_return(sql, return_error=True, params={"pl": pl}),
        )
        if ok:
            return ais_data
        else:
            raise TypeError(ais_data[0][0])

    def _bin_one_ping(
        self, leg_key: str, dir_: str,
        loa, toc, sog, air_draught, beam, draugt, multiplier: float,
    ) -> None:
        if loa is None:
            loa = 100
        loa = int(loa)
        loa_cat = -1
        freq_data = self.omrat.traffic.traffic_data[leg_key][dir_]['Frequency (ships/year)']
        n_loa_cats = len(freq_data[0]) if freq_data else 0
        for loa_i in range(n_loa_cats):
            if loa_i * 25 < loa <= (loa_i * 25 + 25):
                loa_cat = loa_i
                continue
        if loa_cat < 0:
            loa_cat = n_loa_cats - 1 if n_loa_cats > 0 else 0
        type_cat = get_type(toc)
        td = self.omrat.traffic.traffic_data[leg_key][dir_]
        td['Frequency (ships/year)'][type_cat][loa_cat] += multiplier
        if sog is not None:
            td['Speed (knots)'][type_cat][loa_cat].append(float(sog))
        if air_draught is not None:
            td['Ship heights (meters)'][type_cat][loa_cat].append(float(air_draught))
        if beam is not None:
            td['Ship Beam (meters)'][type_cat][loa_cat].append(float(beam))
        if draugt is not None:
            td['Draught (meters)'][type_cat][loa_cat].append(float(draugt))

    def update_ais_data(
        self, leg_key: str, ais_data: list[list[Any]],
        leg_bearing: float, dirs: list[str],
        multiplier: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bin every passing ship into the per-(type, loa) frequency matrix.

        ``multiplier`` (default 1.0) scales the per-ping increment so a
        partial-year ingest can be reported as an annualised rate.  See
        :meth:`compute_year_multiplier`.  Speed/Beam/Draught/Heights
        distributions are *not* scaled — they are observations, not
        counts, and ``convert_list2avg`` averages them downstream.
        """
        line1: list[float] = []
        line2: list[float] = []
        for loa, beam, toc, draugt, sh_type, _, sog, air_draught, dist, cog in ais_data:
            if close_to_line(leg_bearing + 180, cog, self.max_deviation):
                line1.append(dist)
                l1 = True
            elif close_to_line(leg_bearing, cog, self.max_deviation):
                line2.append(dist)
                l1 = False
            else:
                continue
            dir_ = dirs[0] if l1 else dirs[1]
            self._bin_one_ping(leg_key, dir_, loa, toc, sog, air_draught, beam, draugt, multiplier)
        np_line1 = np.array(line1)
        np_line2 = np.array(line2)
        return np_line1, np_line2

    # ----------------------------------------------------------- junction transitions

    def fetch_passages_for_leg(
        self, leg_d: dict[str, str], near_radius_m: float | None = None,
    ) -> dict[str, list[float]]:
        """Return ``{mmsi: [unix_timestamp, ...]}`` for AIS pings inside the leg.

        ``near_radius_m`` is currently unused — the existing ``run_sql``
        builds a passage line spanning the whole width of the leg, which
        is the same data the per-leg traffic update consumes.  When AIS-
        derived junction transitions become production-grade we will
        switch to a near-junction sub-polygon to better reflect "ships
        actually transiting the junction" instead of "ships anywhere
        on the leg".  For now the wider polygon is good enough because
        the ``transition_counts_from_passages`` algorithm self-filters
        by the per-MMSI temporal window.
        """
        if self.db is None:
            return {}
        start_p = wkt.loads(f"Point ({leg_d['Start_Point']})")
        end_p = wkt.loads(f"Point ({leg_d['End_Point']})")
        if not isinstance(start_p, Point) or not isinstance(end_p, Point):
            return {}
        pl = get_pl(
            self.db,
            lat1=start_p.y, lat2=end_p.y,
            lon1=start_p.x, lon2=end_p.x,
            l_width=float(leg_d.get('Width', 5000)),
        )
        try:
            rows = self.run_sql(pl)
        except Exception:
            return {}
        # ``run_sql`` returns the per-ping row used by ``update_ais_data``;
        # we only need the (mmsi, date1) pair to build the transition
        # counts.  ``date1`` is at column index 5 in the SELECT order
        # (loa, beam, type_and_cargo, draught, ship_type, date1, sog,
        # air_draught, dist_from_start, cog).  ``mmsi`` is not in the
        # public SELECT list — we replay the same query joined back to
        # ``ss.mmsi`` here for the targeted use case.
        # Keep the implementation pragmatic: re-issue a lighter query
        # focused on (mmsi, date1) so we don't have to refactor run_sql.
        return self._fetch_mmsi_passages(pl)

    def _fetch_mmsi_passages(self, pl: str) -> dict[str, list[float]]:
        """Return ``{mmsi: [unix_timestamp, ...]}`` for the given passage line."""
        if self.db is None:
            return {}
        schema = _validate_sql_identifier(str(self.schema))
        year = int(self.year)
        months: list[int] = []
        for m in (self.months or []):
            mi = int(m)
            if 1 <= mi <= 12:
                months.append(mi)
        if months:
            tables = [f"{schema}.segments_{year}_{month}" for month in months]
        else:
            tables = [f"{schema}.segments_{year}"]
        # ``schema`` is regex-validated, ``year``/``month`` are coerced to
        # int, and ``pl`` is bound through the ``%(pl)s`` placeholder, so
        # f-string interpolation of the table name is safe here.
        select_template = (  # nosec B608
            "select ss.mmsi, "
            "extract(epoch from date1) as t "
            "FROM {tbl} ss "
            "JOIN {schema}.states_{year} st on st.rowid=ss.state_id "
            "WHERE ST_intersects(segment, ST_geomfromtext(%(pl)s, 4326))"
        )
        sql = " UNION ".join(  # nosec B608
            select_template.format(tbl=tbl, schema=schema, year=year)
            for tbl in tables
        )
        try:
            ok, rows = cast(
                tuple[bool, list[list[Any]]],
                self.db.execute_and_return(
                    sql, return_error=True, params={"pl": pl},
                ),
            )
        except Exception:
            return {}
        if not ok or not rows:
            return {}
        out: dict[str, list[float]] = {}
        for mmsi, t in rows:
            if mmsi is None or t is None:
                continue
            out.setdefault(str(mmsi), []).append(float(t))
        return out

    def compute_junction_transitions(
        self,
        time_window_s: float | None = None,
        seg_table: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, dict[str, dict[str, int]]]:
        """Build a ``{junction_id: {in_leg: {out_leg: count}}}`` table from AIS.

        Returns an empty dict if the database is offline or the project
        has no junctions.  Junctions are read from the live
        :class:`Junctions` handler on the parent OMRAT instance.

        ``seg_table`` may be pre-supplied (e.g. by a background task that
        already read the route table on the main thread) to avoid accessing
        the UI widget from a non-main thread.  When ``None`` the method reads
        the table itself (original behaviour, requires main thread).
        """
        from compute.junction_transitions import (
            DEFAULT_TIME_WINDOW_S,
            transition_counts_from_passages,
        )
        handler = getattr(self.omrat, 'junctions', None)
        if handler is None or not handler.registry:
            return {}
        if self.db is None:
            return {}
        if seg_table is None:
            seg_table = self.get_segment_data_from_table()
        # First fetch passages once per leg so junctions sharing legs
        # don't query the database twice.
        leg_ids: set[str] = set()
        for j in handler.registry.values():
            for leg_id in j.legs:
                leg_ids.add(leg_id)
        passages_by_leg: dict[str, dict[str, list[float]]] = {}
        for leg_id in leg_ids:
            leg_d = seg_table.get(str(leg_id))
            if leg_d is None:
                passages_by_leg[leg_id] = {}
                continue
            passages_by_leg[leg_id] = self.fetch_passages_for_leg(leg_d)
        # Build per-junction count tables by restricting the global
        # passage map to the legs that touch each junction.
        out: dict[str, dict[str, dict[str, int]]] = {}
        window = (
            DEFAULT_TIME_WINDOW_S if time_window_s is None
            else float(time_window_s)
        )
        for jid, j in handler.registry.items():
            local = {
                leg_id: passages_by_leg.get(leg_id, {})
                for leg_id in j.legs
            }
            out[jid] = transition_counts_from_passages(
                local, time_window_s=window,
            )
        return out

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
