import pytest
from qgis.PyQt.QtWidgets import QApplication
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from open_mrat_dockwidget import OpenMRATDockWidget

app = QApplication(sys.argv)
def test_add_route():
    print(132)
    dockwidget = OpenMRATDockWidget(None)
    print(dockwidget.pbRunModel.text())
    
    
test_add_route()