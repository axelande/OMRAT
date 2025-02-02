# -*- coding: utf-8 -*-
"""
/***************************************************************************
 OMRAT
                                 A QGIS plugin
 This is an open source implementation of Pedersens equations from 1995
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2022-12-22
        git sha              : $Format:%H$
        copyright            : (C) 2022 by Axel Hörteborn
        email                : axel.horteborn@ri.se
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt, QVariant
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QTableWidgetItem
from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry, QgsLineString, QgsPoint, QgsProject, 
                       QgsField, QgsCoordinateReferenceSystem, QgsCoordinateTransform)
import sys
import os
sys.path.append('.')
# Initialize Qt resources from file resources.py
from resources import *

# Import the code for the DockWidget
from compute.run_calculations import Calculation
from omrat_utils import PointTool
from omrat_utils.handle_traffic import Traffic
from omrat_utils.repair_time import Repair
from omrat_utils.storage import Storage
from omrat_utils.handle_object import OObject
from omrat_utils.gather_data import GatherData
from open_mrat_dockwidget import OpenMRATDockWidget
from operator import xor
import os.path


class OpenMRAT:
    """QGIS Plugin Implementation."""

    def __init__(self, iface, testing=False):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        self.testing = testing
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        if not self.testing:
            # initialize locale
            locale = QSettings().value('locale/userLocale')[0:2]
            locale_path = os.path.join(
                self.plugin_dir,
                'i18n',
                f'OpenMRAT_{locale}.qm')

            if os.path.exists(locale_path):
                self.translator = QTranslator()
                self.translator.load(locale_path)
                QCoreApplication.installTranslator(self.translator)

            # Declare instance attributes
            self.actions = []
            self.menu = self.tr(u'&Open Maritime Risk Analysis Tool')
            # TODO: We are going to let the user set this up in a future iteration
            self.toolbar = self.iface.addToolBar(u'OMRAT')
            self.toolbar.setObjectName(u'OMRAT')

            #print "** INITIALIZING OMRAT"

        self.pluginIsActive = False
        self.dockwidget = None
        self.segment_id = 0
        self.traffic_data = {}
        self.segment_data = {}
        self.traffic = None
        self.calc = None

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('OMRAT', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action


    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = self.plugin_dir + '/icon.png'
        print(icon_path)
        self.add_action(
            icon_path,
            text=self.tr(u'Omrat'),
            callback=self.run,
            parent=self.iface.mainWindow())

    #--------------------------------------------------------------------------

    def onClosePlugin(self):
        """Cleanup necessary items here when plugin dockwidget is closed"""

        #print "** CLOSING OMRAT"

        # disconnects
        self.dockwidget.closingPlugin.disconnect(self.onClosePlugin)

        # remove this statement if dockwidget is to remain
        # for reuse if plugin is reopened
        # Commented next statement since it causes QGIS crashe
        # when closing the docked window:
        # self.dockwidget = None

        self.pluginIsActive = False


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""

        #print "** UNLOAD OMRAT"

        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Open Maritime Risk Analysis Tool'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar
    
    def add_new_route(self):
        """Starts the editing for a new route, using the setMapTool and PointTool"""
        self.current_start_point = None
        self.point_layer = None
        self.mapTool = PointTool(self.iface.mapCanvas())
        self.iface.mapCanvas().setMapTool(self.mapTool)
        self.mapTool.canvasClicked.connect(self.onMapClick)
        
    def onMapClick(self, point):
        q_point = self.point4326_from_wkt(point.asWkt())
        if self.current_start_point is None:
            self.create_point(q_point)
        else:
            self.create_line(q_point)

    def create_point(self, point):
        self.point_layer = QgsVectorLayer("Point", "StartPoint", "memory")
        prov = self.point_layer.dataProvider()
        self.point_layer.startEditing()
        feat = QgsFeature()
        feat.setGeometry(point)
        prov.addFeatures([feat])
        QgsProject.instance().addMapLayer(self.point_layer)
        self.current_start_point = point
    
    def create_line(self, point):
        vl = QgsVectorLayer("LineString", "Segment", "memory")
        self.segment_id += 1
        pr = vl.dataProvider()
        vl.startEditing()
        seg_ids = [QgsField("segmentId", QVariant.Int),
                   QgsField("routeId", QVariant.Int),
                   QgsField("startPoint",  QVariant.String),
                   QgsField("endPoint", QVariant.String)]
        pr.addAttributes(seg_ids)
        fet = QgsFeature()
        fet.setAttributes([self.segment_id, 1, self.current_start_point.asPoint().asWkt(), point.asPoint().asWkt()])
        start_point = QgsPoint(self.current_start_point.asPoint())
        end_point = QgsPoint(point.asPoint())
        fet.setGeometry(QgsLineString([start_point, end_point]))
        pr.addFeatures( [ fet ] )
        self.iface.actionToggleEditing().trigger()
        # Show in project
        if not self.testing:
            QgsProject.instance().removeMapLayer(self.point_layer)
        self.save_route(start_point, end_point)
        QgsProject.instance().addMapLayer(vl)
        self.update_segment_data(point.asPoint())
        
    def point4326_from_wkt(self, wkt) -> QgsGeometry:
        q_point = QgsGeometry.fromWkt(wkt)
        crs = self.iface.mapCanvas().mapSettings().destinationCrs().authid()
        tr = QgsCoordinateTransform(QgsCoordinateReferenceSystem(crs),
                                    QgsCoordinateReferenceSystem("EPSG:4326"),
                                    QgsProject.instance())
        q_point.transform(tr)
        return q_point
        
    def load_lines(self, data):
        self.segment_id = 0
        for key, seg_data in data["segment_data"].items():
            s_wkt = f'Point({seg_data["Start Point"]})'
            e_wkt = f'Point({seg_data["End Point"]})'
            vl = QgsVectorLayer("LineString", "Segment", "memory")
            self.segment_id += 1
            pr = vl.dataProvider()
            vl.startEditing()
            seg_ids = [QgsField("segmentId", QVariant.Int),
                    QgsField("routeId", QVariant.Int),
                    QgsField("startPoint",  QVariant.String),
                    QgsField("endPoint", QVariant.String)]
            pr.addAttributes(seg_ids)
            fet = QgsFeature()
            fet.setAttributes([self.segment_id, seg_data["Route Id"], s_wkt, e_wkt])
            start = self.point4326_from_wkt(s_wkt).asPoint()
            end = self.point4326_from_wkt(e_wkt).asPoint()
            fet.setGeometry(QgsLineString([start, end]))
            pr.addFeatures( [ fet ] )
            QgsProject.instance().addMapLayer(vl)
            self.iface.actionSaveActiveLayerEdits().trigger() 
            self.iface.actionToggleEditing().trigger()    
        
    def update_segment_data(self, point):
        degrees = (self.current_start_point.asPoint().azimuth(point) + 360) % 360
        if degrees > 315 or degrees <= 45:
            self.dockwidget.laDir1.setText('North going')
            self.dockwidget.laDir2.setText('South going')
            dirs = ['North going', 'South going']
        if degrees > 45 and degrees <= 135:
            self.dockwidget.laDir1.setText('East going')
            self.dockwidget.laDir2.setText('West going')
            dirs = ['East going', 'West going']
        if degrees > 135 and degrees <= 225:
            self.dockwidget.laDir1.setText('South going')
            self.dockwidget.laDir2.setText('North going')
            dirs = ['South going', 'North going']
        if degrees > 225 and degrees <= 315:
            self.dockwidget.laDir1.setText('West going')
            self.dockwidget.laDir2.setText('East going')
            dirs = ['West going', 'East going']
        if f'{self.segment_id}' in self.segment_data:
            self.segment_data[f'{self.segment_id}']['Start Point'] = self.current_start_point.asPoint().asWkt()
            self.segment_data[f'{self.segment_id}']['End Point'] = point.asWkt()
            self.segment_data[f'{self.segment_id}']['Dirs'] = dirs
        else:
            self.segment_data[f'{self.segment_id}'] = {'Start Point': self.current_start_point.asPoint().asWkt(), 
                                                  'End Point': point.asWkt(),
                                                  'Dirs': dirs}
        self.traffic.create_empty_dict(f'{self.segment_id}', dirs)
        self.dockwidget.cbTrafficSelectSeg.addItem(f'{self.segment_id}')
        self.traffic.c_seg = f'{self.segment_id}'
        self.current_start_point = None
        
    def get_length_and_dir_from_line(self, p1, p2):
        degrees = p1.azimuth(p2)
        
    def save_route(self, point1, point2):
        row_id = self.dockwidget.twRouteList.rowCount()
        self.dockwidget.twRouteList.setRowCount(row_id + 1)
        item1 = QTableWidgetItem(f'{self.segment_id}')
        item2 = QTableWidgetItem(f'1')
        item3 = QTableWidgetItem(f'{point1.asWkt(precision=5).split("(")[1].split(")")[0]}')
        item4 = QTableWidgetItem(f'{point2.asWkt(precision=5).split("(")[1].split(")")[0]}')
        self.dockwidget.twRouteList.setItem(row_id, 0, item1)
        self.dockwidget.twRouteList.setItem(row_id, 1, item2)
        self.dockwidget.twRouteList.setItem(row_id, 2, item3)
        self.dockwidget.twRouteList.setItem(row_id, 3, item4)
        self.iface.actionSaveActiveLayerEdits().trigger()
        self.iface.actionToggleEditing().trigger()

    def reset_route_table(self):
        self.dockwidget.twRouteList.setColumnCount(4)
        self.dockwidget.twRouteList.setHorizontalHeaderLabels(['Segment Id', 'Route Id', 
                                                               'Start Point', 'End Point'])
        self.dockwidget.twRouteList.setColumnWidth(1, 75)
        self.dockwidget.twRouteList.setColumnWidth(2, 125)
        self.dockwidget.twRouteList.setColumnWidth(3, 125)
        self.dockwidget.twRouteList.setRowCount(0)
    
    def show_traffic_widget(self):
        self.traffic.traffic_data = self.traffic_data
        self.traffic.fill_cbTrafficSelectSeg()
        self.traffic.update_direction_select()
        self.traffic.change_type('omrat')
        self.traffic.run()
        
    def save_work(self):
        store = Storage(self)
        store.store_all()

    def load_work(self):
        store = Storage(self)
        store.load_all()
        
    def test_evalute_repair(self):
        self.repair.test_evaluate()
        
    def run_calculation(self):
        gd = GatherData(self)
        height = self.dockwidget.sbHeight.value()
        data = gd.get_all_for_save()
        self.calc = Calculation(self, data, height)
        self.calc.run_model()
        
        

    #--------------------------------------------------------------------------

    def run(self):
        """Run method that loads and starts the plugin"""

        if not self.pluginIsActive:
            self.pluginIsActive = True

            #print "** STARTING OMRAT"

            # dockwidget may not exist if:
            #    first run of plugin
            #    removed on close (see self.onClosePlugin method)
            if self.dockwidget == None:
                # Create the dockwidget (after translation) and keep reference
                self.dockwidget = OpenMRATDockWidget()
            self.repair = Repair(self)
            self.traffic = Traffic(self, self.dockwidget)
            self.object = OObject(self)
            # connect to provide cleanup on closing of dockwidget
            self.dockwidget.closingPlugin.connect(self.onClosePlugin)

            # show the dockwidget
            # TODO: fix to allow choice of dock location
            if not self.testing:
                self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dockwidget)
            self.dockwidget.pbAddRoute.clicked.connect(self.add_new_route)
            self.dockwidget.pbEditTrafficData.clicked.connect(self.show_traffic_widget)
            self.dockwidget.pbSaveProject.clicked.connect(self.save_work)
            self.dockwidget.pbLoadProject.clicked.connect(self.load_work)
            self.dockwidget.pbTestRepair.clicked.connect(self.test_evalute_repair)
            self.dockwidget.cbTrafficSelectSeg.currentIndexChanged.connect(self.traffic.change_dist_segment)
            self.dockwidget.pbAddSimpleDepth.clicked.connect(self.object.add_simple_depth)
            self.dockwidget.pbAddSimpleObject.clicked.connect(self.object.add_simple_object)
            self.dockwidget.pbRunModel.clicked.connect(self.run_calculation)
            
            self.reset_route_table()
            self.dockwidget.show()
