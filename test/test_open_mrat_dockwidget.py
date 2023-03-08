# coding=utf-8
"""DockWidget test.

.. note:: This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

"""

__author__ = 'axel.horteborn@ri.se'
__date__ = '2022-12-22'
__copyright__ = 'Copyright 2022, Axel Hörteborn'

import unittest

from qgis.PyQt.QtWidgets import QDockWidget
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from open_mrat_dockwidget import OpenMRATDockWidget

from .utilities import get_qgis_app

QGIS_APP = get_qgis_app()


class OpenMRATDockWidgetTest(unittest.TestCase):
    """Test dockwidget works."""

    def setUp(self):
        """Runs before each test."""
        self.dockwidget = OpenMRATDockWidget()

    def tearDown(self):
        """Runs after each test."""
        self.dockwidget = None

    def test_dockwidget_ok(self):
        """Test we can click OK. 123"""
        pass

if __name__ == "__main__":
    suite = unittest.makeSuite(OpenMRATDockWidgetTest)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

