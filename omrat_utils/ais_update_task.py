"""Background QgsTask for fetching and processing AIS data.

All database queries and per-leg data processing happen in ``run()``
(background thread).  ``finished()`` is called by Qt on the main thread
and is the only place that touches UI widgets or mutates shared plugin state.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
from qgis.core import QgsTask, QgsMessageLog, Qgis
from qgis.PyQt.QtWidgets import QMessageBox

if TYPE_CHECKING:
    from omrat_utils.handle_ais import AIS


class AisUpdateTask(QgsTask):
    """Fetch AIS traffic for one or more legs in a background thread."""

    def __init__(
        self,
        ais: "AIS",
        legs: dict[str, dict[str, Any]],
        seg_table: dict[str, dict[str, str]],
        rows: int,
        cols: int,
        variables: list[str],
        var_defaults: dict[str, Any],
        leg_dirs: dict[str, list[str]],
    ) -> None:
        super().__init__("OMRAT: Fetching AIS data", QgsTask.CanCancel)
        self.ais = ais
        self.legs = legs
        self.seg_table = seg_table
        self.rows = rows
        self.cols = cols
        self.variables = variables
        self.var_defaults = var_defaults
        self.leg_dirs = leg_dirs

        # Populated in run(), consumed in finished()
        self.results: dict[str, dict] = {}
        self.multiplier: float = 1.0
        self.multiplier_info: tuple | None = None
        self.junction_counts: dict | None = None
        self.last_key: str | None = None
        self.error: str | None = None

    # ------------------------------------------------------------------
    # Helpers

    def _make_empty_block(self, dirs: list[str]) -> dict:
        """Build the same empty per-leg traffic dict as create_empty_dict."""
        block: dict = {}
        for di in dirs:
            block[di] = {}
            for var in self.variables:
                default = self.var_defaults.get(var, [])
                block[di][var] = []
                for _ in range(self.rows):
                    row: list = []
                    for _ in range(self.cols):
                        row.append(default if not isinstance(default, list) else [])
                    block[di][var].append(row)
        return block

    # ------------------------------------------------------------------
    # Background thread

    def _bin_ping(
        self, td_block: dict, dirs: list, row: tuple,
        leg_bearing: float, line1: list, line2: list,
    ) -> None:
        from omrat_utils.handle_ais import close_to_line, get_type
        loa, beam, toc, draugt, _, _, sog, air_draught, dist, cog = row
        if close_to_line(leg_bearing + 180, cog, self.ais.max_deviation):
            line1.append(dist)
            l1 = True
        elif close_to_line(leg_bearing, cog, self.ais.max_deviation):
            line2.append(dist)
            l1 = False
        else:
            return
        dir_ = dirs[0] if l1 else dirs[1]
        if loa is None:
            loa = 100
        loa = int(loa)
        freq_data = td_block[dir_]['Frequency (ships/year)']
        n_cats = len(freq_data[0]) if freq_data else 0
        loa_cat = next((i for i in range(n_cats) if i * 25 < loa <= i * 25 + 25), max(n_cats - 1, 0))
        type_cat = get_type(toc)
        td_block[dir_]['Frequency (ships/year)'][type_cat][loa_cat] += self.multiplier
        if sog is not None:
            td_block[dir_]['Speed (knots)'][type_cat][loa_cat].append(float(sog))
        if air_draught is not None:
            td_block[dir_]['Ship heights (meters)'][type_cat][loa_cat].append(float(air_draught))
        if beam is not None:
            td_block[dir_]['Ship Beam (meters)'][type_cat][loa_cat].append(float(beam))
        if draugt is not None:
            td_block[dir_]['Draught (meters)'][type_cat][loa_cat].append(float(draugt))

    def _collapse_td_block(self, td_block: dict) -> None:
        for di in td_block:
            for var in td_block[di]:
                for r, row in enumerate(td_block[di][var]):
                    for c, val in enumerate(row):
                        if isinstance(val, list):
                            td_block[di][var][r][c] = float(np.mean(val)) if val else np.inf

    def _process_leg(self, leg_key: str, leg_d: dict) -> None:
        from omrat_utils.handle_ais import get_pl
        from shapely import wkt
        ais = self.ais
        dirs = self.leg_dirs.get(leg_key, [])
        start_p = wkt.loads(f"Point ({leg_d['Start_Point']})")
        end_p = wkt.loads(f"Point ({leg_d['End_Point']})")
        pl = get_pl(ais.db, lat1=start_p.y, lat2=end_p.y, lon1=start_p.x, lon2=end_p.x, l_width=float(leg_d['Width']))
        ais_data = ais.run_sql(pl)
        sql = (f"select degrees(ST_Azimuth(ST_Point({start_p.x}, {start_p.y})::geography, "
               f"ST_Point({end_p.x}, {end_p.y})::geography))")
        _, bearing_rows = ais.db.execute_and_return(sql, return_error=True)
        leg_bearing: float = bearing_rows[0][0]
        td_block = self._make_empty_block(dirs)
        line1: list[float] = []
        line2: list[float] = []
        for row in ais_data:
            self._bin_ping(td_block, dirs, row, leg_bearing, line1, line2)
        self._collapse_td_block(td_block)
        self.results[leg_key] = {'traffic': td_block, 'dirs': dirs, 'line1': np.array(line1), 'line2': np.array(line2)}
        self.last_key = leg_key

    def _fetch_junction_counts(self) -> None:
        ais = self.ais
        if ais.db is None:
            return
        try:
            self.junction_counts = ais.compute_junction_transitions(seg_table=self.seg_table)
        except Exception as exc:
            QgsMessageLog.logMessage(f"Junction transition fetch skipped: {exc}", "OMRAT", Qgis.Warning)

    def run(self) -> bool:
        try:
            ais = self.ais
            if ais.recalc_to_full_year:
                info = ais.compute_year_multiplier()
                if info is not None:
                    self.multiplier = info[0]
                    self.multiplier_info = info
            n = len(self.legs)
            for idx, (leg_key, leg_d) in enumerate(self.legs.items()):
                if self.isCanceled():
                    return False
                self.setProgress(int(idx / n * 80))
                self.setDescription(f"OMRAT: AIS fetch {idx + 1}/{n}")
                self._process_leg(leg_key, leg_d)
            if self.isCanceled():
                return False
            self.setProgress(85)
            self.setDescription("OMRAT: Junction transitions")
            self._fetch_junction_counts()
            self.setProgress(100)
            return True
        except Exception as exc:
            import traceback
            self.error = f"{exc}\n{traceback.format_exc()}"
            QgsMessageLog.logMessage(f"AIS update task failed: {self.error}", "OMRAT", Qgis.Critical)
            return False

    # ------------------------------------------------------------------
    # Main thread — called after run() completes

    def _apply_leg_result(self, omrat: Any, leg_key: str, res: dict) -> None:
        ais = self.ais
        omrat.traffic.traffic_data[leg_key] = res['traffic']
        ais.dist_data[leg_key] = {'line1': res['line1'], 'line2': res['line2']}
        line1, line2 = res['line1'], res['line2']
        sd = omrat.segment_data.get(leg_key)
        if sd is None:
            return
        sd['mean1_1'] = float(line1.mean()) if len(line1) else 0.0
        sd['std1_1']  = float(line1.std())  if len(line1) else 0.0
        sd['mean2_1'] = float(line2.mean()) if len(line2) else 0.0
        sd['std2_1']  = float(line2.std())  if len(line2) else 0.0
        sd['weight1_1'] = 100
        sd['weight2_1'] = 100
        sd['dist1'] = line1
        sd['dist2'] = line2
        for j in range(1, 3):
            for i in range(2, 4):
                sd[f'mean{j}_{i}'] = 0
                sd[f'std{j}_{i}']  = 0
                sd[f'weight{j}_{i}'] = 0
            sd[f'u_min{j}'] = 0
            sd[f'u_max{j}'] = 0
            sd[f'u_p{j}']   = 0
        sd['ai1'] = 180
        sd['ai2'] = 180

    def _update_last_leg_display(self, omrat: Any) -> None:
        if not self.last_key or self.last_key not in self.results:
            return
        line1 = self.results[self.last_key]['line1']
        line2 = self.results[self.last_key]['line2']
        mw = omrat.main_widget
        mw.leNormMean1_1.setText('')
        if len(line1) > 0:
            mw.leNormMean1_1.setText(str(line1.mean()))
            mw.leNormMean2_1.setText(str(line2.mean()))
            mw.leNormStd1_1.setText(str(line1.std()))
            mw.leNormStd2_1.setText(str(line2.std()))
        omrat.distributions.run_update_plot(self.last_key)
        mw.cbTrafficSelectSeg.setCurrentIndex(mw.cbTrafficSelectSeg.count() - 1)

    def _refresh_junction_registry(self, omrat: Any) -> None:
        try:
            handler = getattr(omrat, 'junctions', None)
            if handler is not None:
                handler.rebuild_from_segments(omrat.segment_data, prefer_user=True)
                if self.junction_counts:
                    handler.apply_ais_counts(self.junction_counts)
        except Exception as exc:
            QgsMessageLog.logMessage(f"Junction transition refresh skipped: {exc}", "OMRAT", Qgis.Warning)

    def finished(self, result: bool) -> None:
        ais = self.ais
        omrat = ais.omrat
        btn = getattr(omrat.main_widget, 'pbUpdateAIS', None)
        if btn is not None:
            btn.setEnabled(True)
        if not result or self.isCanceled():
            if self.error:
                QMessageBox.critical(omrat.main_widget, omrat.tr("AIS update failed"), self.error[:500])
            return
        if self.multiplier_info is not None:
            mult, coverage_s, gap_s = self.multiplier_info
            QMessageBox.information(
                omrat.main_widget, omrat.tr("Annualised AIS frequency"),
                omrat.tr(f"Coverage: {coverage_s / 3600:.1f} h (skipped {gap_s / 3600:.1f} h of gaps > 12 h).\n"
                         f"Multiplier: {mult:.2f}x ({ais._YEAR_SECONDS / 3600:.0f} h / {coverage_s / 3600:.1f} h)."),
            )
        for leg_key, res in self.results.items():
            self._apply_leg_result(omrat, leg_key, res)
        self._update_last_leg_display(omrat)
        self._refresh_junction_registry(omrat)
