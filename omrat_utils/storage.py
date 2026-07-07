from __future__ import annotations
import json
import os
from typing import TYPE_CHECKING

from pydantic import ValidationError
from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtWidgets import QFileDialog

from .gather_data import GatherData
from .validate_data import RootModelSchema

if TYPE_CHECKING:
    from omrat import OMRAT

class Storage:
    def __init__(self, parent: OMRAT) -> None:
        self.p = parent
        
    def store_all(self):
        file_path = self.new_file_path(True, "Save Project", self.last_used_dir(),
                                       "proj.omrat", "shapefiles (*.omrat *.OMRAT)" )[0]
        if file_path == "":
            return
        gather = GatherData(self.p)
        data = gather.get_all_for_save()
        try:
            RootModelSchema.model_validate(data)
        except ValidationError as e:
            # This should not happen when everything works
            print(e)
        with open(file_path, 'w') as f:
            f.write(json.dumps(data, indent=2))
        
    def select_file(self) -> str:
        """Open a file dialog to select a .omrat file (or use test path).

        Returns the selected file path, or empty string if cancelled.
        """
        if self.p.testing:
            dp = os.path.dirname(__file__)
            return os.path.join(dp, '..', 'tests', 'test_res.omrat')
        else:
            file_path = self.new_file_path(False, "Load Project", self.last_used_dir(),
                                           "proj.omrat", "shapefiles (*.omrat *.OMRAT)")[0]
            return file_path

    def load_from_path(self, file_path: str) -> None:
        """Load and populate data from the given .omrat file path."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Normalize legacy project files to match RootModelSchema
            data = self._normalize_legacy_to_schema(data)
            try:
                RootModelSchema.model_validate(data)
            except ValidationError as e:
                # Show error to user, log, etc.
                print("Validation error:", e)
                return
            gather = GatherData(self.p)
            gather.populate(data)

    def load_all(self):
        """Select a file and load it. Legacy convenience method."""
        file_path = self.select_file()
        if not file_path:
            return
        self.load_from_path(file_path)

    def _normalize_legacy_to_schema(self, data: dict) -> dict:
        """Convert older saved formats/keys to match RootModelSchema."""
        out = dict(data)
        out['depths'] = self._normalize_depths(out.get('depths', []))
        out['objects'] = self._normalize_objects(out.get('objects', []))
        out['segment_data'] = self._normalize_segment_data(out.get('segment_data', {}) or {})
        out['traffic_data'] = self._normalize_traffic_data(out.get('traffic_data', {}) or {})
        out['traffic_scaling'] = self._normalize_traffic_scaling(out.get('traffic_scaling'))
        out['drift'] = self._normalize_drift(out.get('drift', {}) or {})
        if 'junctions' not in out or not isinstance(out.get('junctions'), dict):
            out['junctions'] = {}
        if 'consequence' not in out or not isinstance(out.get('consequence'), dict):
            out['consequence'] = self._default_consequence(out.get('ship_categories') or {})
        return out

    @staticmethod
    def _normalize_depths(depths: list) -> list:
        result = []
        for dep in depths:
            if isinstance(dep, dict):
                result.append([str(dep.get('id', '')), str(dep.get('depth', '')), str(dep.get('polygon', ''))])
            else:
                try:
                    did, depth, poly = dep
                    result.append([str(did), str(depth), str(poly)])
                except Exception:
                    pass
        return result

    @staticmethod
    def _normalize_objects(objects: list) -> list:
        result = []
        for obj in objects:
            if isinstance(obj, dict):
                height = obj.get('height', obj.get('heights', ''))
                result.append([str(obj.get('id', '')), str(height), str(obj.get('polygon', ''))])
            else:
                try:
                    oid, height, poly = obj
                    result.append([str(oid), str(height), str(poly)])
                except Exception:
                    pass
        return result

    @staticmethod
    def _normalize_segment_data(segs: dict) -> dict:
        for sid, seg in segs.items():
            if 'Start Point' in seg and 'Start_Point' not in seg:
                seg['Start_Point'] = seg.pop('Start Point')
            if 'End Point' in seg and 'End_Point' not in seg:
                seg['End_Point'] = seg.pop('End Point')
            for key, default in [
                ('line_length', 0.0), ('Route_Id', 0), ('Leg_name', ''), ('Segment_Id', str(sid)),
                ('u_min1', 0.0), ('u_max1', 0.0), ('ai1', 0.0), ('u_min2', 0.0), ('u_max2', 0.0), ('ai2', 0.0),
                ('Width', seg.get('Width', 0)),
            ]:
                seg.setdefault(key, default)
            seg.setdefault('dist1', [])
            seg.setdefault('dist2', [])
        return segs

    @staticmethod
    def _normalize_traffic_data(td: dict) -> dict:
        for _leg_id, dirs in td.items():
            for _dir_name, dir_data in dirs.items():
                if 'Ship Beam (meters)' not in dir_data:
                    sp = dir_data.get('Speed (knots)', [])
                    dir_data['Ship Beam (meters)'] = [
                        [0 for _ in row] if hasattr(row, '__iter__') else []
                        for row in sp
                    ]
                freq = dir_data.get('Frequency (ships/year)', [])
                existing = dir_data.get('Scaling (%)')
                tgt_rows = len(freq) if freq else 0
                tgt_cols = len(freq[0]) if (tgt_rows and hasattr(freq[0], '__len__')) else 0
                if not isinstance(existing, list) or len(existing) != tgt_rows:
                    dir_data['Scaling (%)'] = [[100.0] * tgt_cols for _ in range(tgt_rows)]
                else:
                    for r in range(tgt_rows):
                        if r >= len(existing) or not isinstance(existing[r], list):
                            existing[r] = [100.0] * tgt_cols
                            continue
                        row_list = existing[r]
                        if len(row_list) < tgt_cols:
                            row_list.extend([100.0] * (tgt_cols - len(row_list)))
                        elif len(row_list) > tgt_cols:
                            del row_list[tgt_cols:]
        return td

    @staticmethod
    def _normalize_traffic_scaling(scaling_block) -> dict:
        if not isinstance(scaling_block, dict):
            return {'global_percent': 100.0, 'follow_global': []}
        scaling_block.setdefault('global_percent', 100.0)
        scaling_block.setdefault('follow_global', [])
        return scaling_block

    @staticmethod
    def _normalize_drift(drift: dict) -> dict:
        repair = drift.get('repair', {}) or {}
        try:
            combi = str(repair.get('combi', ''))
            rep_type = str(repair.get('type', ''))
            use_ln = bool(repair.get('use_lognormal', False))
            if use_ln and (rep_type.lower() == 'normal' or ('Mean' in combi and 'Std' in combi and 'Lower' not in combi)):
                mean = float(repair.get('param_0', repair.get('loc', 0.0)))
                std = float(repair.get('param_1', repair.get('scale', 1.0)))
                if std <= 0:
                    std = 1e-6
                repair['use_lognormal'] = False
                repair['dist_type'] = 'normal'
                repair['norm_mean'] = mean
                repair['norm_std'] = std
                repair['func'] = (
                    f"__import__('scipy.stats', fromlist=['norm'])"
                    f".norm(loc={mean}, scale={std}).cdf(x)"
                )
        except Exception:
            pass
        repair.setdefault('use_lognormal', False)
        drift['repair'] = repair
        if 'anchor_p' not in drift:
            drift['anchor_p'] = 0.7
        try:
            anchor_p = float(drift.get('anchor_p', 0.7))
            if anchor_p > 1.0:
                anchor_p = anchor_p / 100.0
            drift['anchor_p'] = max(0.0, min(1.0, anchor_p))
        except Exception:
            drift['anchor_p'] = 0.7
        drift.setdefault('anchor_d', drift.get('anchor_depth', 0))
        drift.setdefault('start_from', 'leg_center')
        drift.setdefault('squat_mode', 'average_speed')
        if 'blackout_by_ship_type' in drift and isinstance(drift['blackout_by_ship_type'], dict):
            converted: dict[int, float] = {}
            for k, v in drift['blackout_by_ship_type'].items():
                try:
                    converted[int(k)] = float(v)
                except Exception:
                    continue
            drift['blackout_by_ship_type'] = converted
        else:
            try:
                from compute.basic_equations import default_blackout_by_ship_type
                drift['blackout_by_ship_type'] = default_blackout_by_ship_type()
            except Exception:
                drift['blackout_by_ship_type'] = {}
        return drift

    @staticmethod
    def _default_consequence(ship_cats: dict) -> dict:
        try:
            from omrat_utils.consequence_defaults import (
                default_oil_onboard,
                default_spill_probability,
                default_spill_fraction,
                default_catastrophe_levels,
            )
            types = list(ship_cats.get('types', []))
            intervals = list(ship_cats.get('length_intervals', []))
            return {
                'oil_onboard': default_oil_onboard(types, intervals),
                'spill_probability': default_spill_probability(),
                'spill_fraction': default_spill_fraction(),
                'catastrophe_levels': default_catastrophe_levels(),
            }
        except Exception:
            return {'oil_onboard': [], 'spill_probability': [], 'spill_fraction': [], 'catastrophe_levels': []}

    def new_file_path(self, save, show_msg, dir_path, generic_name, filter_text):
        """Open the QFileDialog and return a string with the folder and name of 
        the new file.
        """
        if save:
            output_filename = QFileDialog.getSaveFileName(None, show_msg,
                                                      dir_path + os.sep + generic_name,
                                                      filter_text)
        else:
            output_filename = QFileDialog.getOpenFileName(None, show_msg,
                                                      dir_path + os.sep + generic_name,
                                                      filter_text)
        if not output_filename:
            return ''
        else:
            return output_filename
    
    def last_used_dir(self):
        """A function that remembers where you last open a vector file"""
        settings = QSettings()
        return settings.value("/QGIS_tools/lastDir", "", type=str)
