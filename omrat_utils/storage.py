from __future__ import annotations
import json
import os
from typing import TYPE_CHECKING

from qgis.PyQt.QtCore import QSettings
from qgis.PyQt.QtWidgets import QFileDialog

from .gather_data import GatherData

if TYPE_CHECKING:
    from open_mrat import OpenMRAT

class Storage:
    def __init__(self, parent: OpenMRAT) -> None:
        self.p = parent
        
    def store_all(self):
        file_path = self.new_file_path(True, "Save Project", self.last_used_dir(),
                                       "proj.omrat", "shapefiles (*.omrat *.OMRAT)" )[0]
        if file_path == "":
            return
        gather = GatherData(self.p)
        data = gather.get_all_for_save()
        print(data)
        with open(file_path, 'w') as f:
            f.write(json.dumps(data, indent=2))
        
    def load_all(self):
        if self.p.testing:
            dp = os.path.dirname(__file__)
            file_path = os.path.join(dp, '..', 'tests', 'test_res.omrat')
        else:
            file_path = self.new_file_path(False, "Load Project", self.last_used_dir(),
                                           "proj.omrat", "shapefiles (*.omrat *.OMRAT)")[0]
        if file_path == "":
            return
        with open(file_path, 'r') as f:
            data = json.load(f)
            gather = GatherData(self.p)
            gather.populate(data)

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
