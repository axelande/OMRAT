from qgis.PyQt.QtCore import pyqtSignal, QRegExp
from qgis.PyQt.QtWidgets import QLineEdit, QStyledItemDelegate
from qgis.PyQt.QtGui import QRegExpValidator
from qgis.gui import QgsMapToolEmitPoint, QgsMapTool


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
            reg_ex = QRegExp("[0-9]+.?[0-9]{,2}")
            validator = QRegExpValidator(reg_ex, editor)
            editor.setValidator(validator)
        return editor