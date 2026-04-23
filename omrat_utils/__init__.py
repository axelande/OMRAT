from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import QLineEdit, QStyledItemDelegate
from qgis.gui import QgsMapToolEmitPoint, QgsMapTool

from helpers.qt_conversions import create_regex_validator


class PointTool(QgsMapToolEmitPoint): 
    canvasClicked = pyqtSignal('QgsPointXY')
    
    def __init__(self, canvas):
        super(QgsMapTool, self).__init__(canvas)

    def canvasReleaseEvent(self, event):
        point_canvas_crs = event.mapPoint()
        self.canvasClicked.emit(point_canvas_crs)
        
class NumericDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = super(NumericDelegate, self).createEditor(parent, option, index)
        if isinstance(editor, QLineEdit):
            validator = create_regex_validator(r"[0-9]+(?:\.[0-9]{0,2})?", editor)
            editor.setValidator(validator)
        return editor