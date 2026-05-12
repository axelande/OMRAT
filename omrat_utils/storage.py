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
        """Convert older saved formats/keys to match RootModelSchema.

        - Ensure depths/objects are lists: [id, value, polygon]
        - Map segment_data keys like 'Start Point' -> 'Start_Point'
        - Fill missing segment fields with safe defaults
        - Ensure traffic_data has 'Ship Beam (meters)' arrays
        - Ensure drift.repair.use_lognormal and drift.anchor_d exist
        """
        out = dict(data)
        # depths
        depths = out.get('depths', [])
        new_depths = []
        for dep in depths:
            if isinstance(dep, dict):
                did = dep.get('id', '')
                depth = dep.get('depth', '')
                poly = dep.get('polygon', '')
                new_depths.append([str(did), str(depth), str(poly)])
            else:
                try:
                    did, depth, poly = dep
                    new_depths.append([str(did), str(depth), str(poly)])
                except Exception:
                    pass
        out['depths'] = new_depths
        # objects
        objects = out.get('objects', [])
        new_objects = []
        for obj in objects:
            if isinstance(obj, dict):
                oid = obj.get('id', '')
                height = obj.get('height', obj.get('heights', ''))
                poly = obj.get('polygon', '')
                new_objects.append([str(oid), str(height), str(poly)])
            else:
                try:
                    oid, height, poly = obj
                    new_objects.append([str(oid), str(height), str(poly)])
                except Exception:
                    pass
        out['objects'] = new_objects
        # segment_data key mapping and defaults
        segs = out.get('segment_data', {}) or {}
        for sid, seg in segs.items():
            if 'Start Point' in seg and 'Start_Point' not in seg:
                seg['Start_Point'] = seg.pop('Start Point')
            if 'End Point' in seg and 'End_Point' not in seg:
                seg['End_Point'] = seg.pop('End Point')
            # Defaults for required numeric fields
            for key, default in [
                ('line_length', 0.0), ('Route_Id', 0), ('Leg_name', ''), ('Segment_Id', str(sid)),
                ('u_min1', 0.0), ('u_max1', 0.0), ('ai1', 0.0), ('u_min2', 0.0), ('u_max2', 0.0), ('ai2', 0.0),
                ('Width', seg.get('Width', 0)),
            ]:
                seg.setdefault(key, default)
            # Ensure dist arrays exist for UI processing
            seg.setdefault('dist1', [])
            seg.setdefault('dist2', [])
        out['segment_data'] = segs
        # traffic_data ensure Ship Beam (meters)
        td = out.get('traffic_data', {}) or {}
        for leg_id, dirs in td.items():
            for dir_name, dir_data in dirs.items():
                if 'Ship Beam (meters)' not in dir_data:
                    # create zeros of same shape as Speed (knots)
                    sp = dir_data.get('Speed (knots)', [])
                    beam = []
                    for row in sp:
                        try:
                            beam.append([0 for _ in row])
                        except Exception:
                            beam.append([])
                    dir_data['Ship Beam (meters)'] = beam
        out['traffic_data'] = td
        # drift defaults
        drift = out.get('drift', {}) or {}
        repair = drift.get('repair', {}) or {}
        # Backward-compatible repair migration:
        # legacy files may store IWRAP type=Normal together with use_lognormal=True.
        # In that case, switch to an explicit normal CDF model so loaded .omrat
        # behaves like current XML import logic.
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
            # Guard against legacy files storing anchor probability in percentage.
            if anchor_p > 1.0:
                anchor_p = anchor_p / 100.0
            drift['anchor_p'] = max(0.0, min(1.0, anchor_p))
        except Exception:
            drift['anchor_p'] = 0.7
        if 'anchor_d' not in drift:
            # try alternate key or default
            drift['anchor_d'] = drift.get('anchor_depth', 0)
        if 'start_from' not in drift:
            drift['start_from'] = 'leg_center'
        if 'squat_mode' not in drift:
            drift['squat_mode'] = 'average_speed'
        # Per-ship-type blackout rate: default to IWRAP-compatible values
        # (1.0 for most, 0.1 for Passenger / Ro-ro / Ro-pax).  JSON keys are
        # serialised as strings; convert to int keys on load.
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
        out['drift'] = drift

        # Seed the junctions block on legacy projects that pre-date the
        # transition-matrix model.  An empty dict is a valid block —
        # downstream code falls back to "all-traffic-counted" defaults
        # when a junction has no recorded matrix.
        if 'junctions' not in out or not isinstance(out.get('junctions'), dict):
            out['junctions'] = {}

        # Seed the consequence block on legacy projects that pre-date the
        # Consequence menu.  ``handle_consequence`` will reshape on load if
        # the user's ship-category dimensions changed since the file was
        # written, but we still need a valid block so RootModelSchema
        # validates.
        if 'consequence' not in out or not isinstance(out.get('consequence'), dict):
            try:
                from omrat_utils.consequence_defaults import (
                    default_oil_onboard,
                    default_spill_probability,
                    default_spill_fraction,
                    default_catastrophe_levels,
                )
                ship_cats = out.get('ship_categories') or {}
                types = list(ship_cats.get('types', []))
                intervals = list(ship_cats.get('length_intervals', []))
                out['consequence'] = {
                    'oil_onboard': default_oil_onboard(types, intervals),
                    'spill_probability': default_spill_probability(),
                    'spill_fraction': default_spill_fraction(),
                    'catastrophe_levels': default_catastrophe_levels(),
                }
            except Exception:
                out['consequence'] = {
                    'oil_onboard': [],
                    'spill_probability': [],
                    'spill_fraction': [],
                    'catastrophe_levels': [],
                }
        return out

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
