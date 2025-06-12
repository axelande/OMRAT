import pytest
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsPointXY, QgsGeometry, QgsFeature
)
from qgis.PyQt.QtWidgets import QTableWidgetItem
from geometries.handle_qgis_iface import HandleQGISIface
from unittest.mock import MagicMock
from conftest import omrat


def test_add_new_route(omrat):
    """Test the add_new_route method."""
    omrat.qgis_geoms.add_new_route()
    assert omrat.qgis_geoms.current_start_point is None
    assert omrat.qgis_geoms.point_layer is None
    assert omrat.qgis_geoms.mapTool is not None

def test_create_point(omrat):
    """Test the create_point method."""
    
    point = QgsGeometry.fromPointXY(QgsPointXY(10, 20))
    omrat.qgis_geoms.create_point(point)

    # Check that the point layer was created and added to the project
    assert omrat.qgis_geoms.point_layer is not None
    assert omrat.qgis_geoms.current_start_point == point
    assert QgsProject.instance().mapLayersByName("StartPoint")


def test_create_line(omrat):
    """Test the create_line method."""
    start_point = QgsGeometry.fromPointXY(QgsPointXY(10, 20))
    end_point = QgsGeometry.fromPointXY(QgsPointXY(30, 40))
    omrat.qgis_geoms.current_start_point = start_point
    omrat.qgis_geoms.create_line(end_point)

    # Check that the line layer was created and added to the project
    assert len(omrat.qgis_geoms.vector_layers) == 2
    assert QgsProject.instance().mapLayersByName("Segment")


def test_create_offset_lines(omrat):
    """Test the create_offset_lines method."""
    start_point = QgsPointXY(10, 20)
    end_point = QgsPointXY(30, 40)
    segment_id = 1

    omrat.qgis_geoms.create_offset_lines(start_point, end_point, 2500, segment_id)

    # Check that the tangent layer was created and added to the project
    tangent_layer = QgsProject.instance().mapLayersByName("Tangent Line")
    assert tangent_layer
    assert len(tangent_layer) == 1

    # Verify that the tangent line has the correct attributes
    feature = next(tangent_layer[0].getFeatures())
    assert feature["type"] == f"Tangent Line {segment_id}"


def test_on_geometry_changed(omrat):
    """Test the on_geometry_changed method."""
    # Create a mock layer and feature
    start_point = QgsPointXY(14.31942998, 55.20514187)
    end_point = QgsPointXY(14.46021114, 55.30168824)
    end_point2 = QgsPointXY(14.61358249, 55.41424602)
    omrat.qgis_geoms.current_start_point = QgsGeometry.fromPointXY(start_point)

    # Create the initial line and tangent
    omrat.qgis_geoms.create_line(QgsGeometry.fromPointXY(end_point))
    tangent_layer = QgsProject.instance().mapLayersByName("Tangent Line")[0]
    tangent_feature = next(tangent_layer.getFeatures())
    original_tangent_geom = tangent_feature.geometry().asPolyline()

    # Verify the original tangent geometry
    assert original_tangent_geom == [QgsPointXY(14.41994445867451624, 55.23905443055804199),
        QgsPointXY(14.35950436006096886, 55.26780867173166456)
    ]

    # Change the endpoint of the segment
    segment_layer = omrat.qgis_geoms.vector_layers[1]
    segment_feature = next(segment_layer.getFeatures())
    segment_feature.setGeometry(QgsGeometry.fromPolylineXY([start_point, end_point2]))
    segment_layer.dataProvider().changeGeometryValues({segment_feature.id(): segment_feature.geometry()})

    # Call the on_geometry_changed method
    omrat.qgis_geoms.on_geometry_changed(segment_feature.id(), segment_feature.geometry())

    # Verify that the tangent line was updated
    updated_tangent_feature = next(tangent_layer.getFeatures())
    updated_tangent_geom = updated_tangent_feature.geometry().asPolyline()
    assert updated_tangent_geom == [
        QgsPointXY(14.49682801649846553, 55.29571991224211303),
        QgsPointXY(14.43538957075596407, 55.32383741183878101)
    ]


def test_on_width_changed(omrat):
    """Test the on_width_changed method."""
    start_point = QgsPointXY(14.31942998, 55.20514187)
    end_point = QgsPointXY(14.46021114, 55.30168824)
    omrat.qgis_geoms.current_start_point = QgsGeometry.fromPointXY(start_point)

    # Create the initial line and tangent
    omrat.qgis_geoms.create_line(QgsGeometry.fromPointXY(end_point))
    tangent_layer = QgsProject.instance().mapLayersByName("Tangent Line")[0]
    tangent_feature = next(tangent_layer.getFeatures())
    original_tangent_geom = tangent_feature.geometry().asPolyline()

    # Verify the original tangent geometry
    assert original_tangent_geom == [QgsPointXY(14.41994445867451624, 55.23905443055804199),
                                     QgsPointXY(14.35950436006096886, 55.26780867173166456)
                                    ]
    
    item5 = QTableWidgetItem(f'8000')
    omrat.dockwidget.twRouteList.setItem(0, 4, item5)
    tangent_layer = QgsProject.instance().mapLayersByName("Tangent Line")[0]
    tangent_feature = next(tangent_layer.getFeatures())
    updated_tangent_geom = tangent_feature.geometry().asPolyline()
    assert updated_tangent_geom != original_tangent_geom
    

def test_modify_second_segment(omrat):
    """Test modifying the second segment and verifying all tangents are updated correctly."""
    # Create a mock layer and features
    start_point = QgsPointXY(14.31942998, 55.20514187)
    mid_point = QgsPointXY(14.46021114, 55.30168824)
    end_point = QgsPointXY(14.61358249, 55.41424602)
    end_point_mod = QgsPointXY(14.54719788, 55.41359631)
    end_point2 = QgsPointXY(14.77725490, 55.46813418)
    omrat.qgis_geoms.current_start_point = QgsGeometry.fromPointXY(start_point)

    # Create the initial line and tangent
    omrat.qgis_geoms.create_line(QgsGeometry.fromPointXY(mid_point))
    omrat.qgis_geoms.create_line(QgsGeometry.fromPointXY(end_point))
    omrat.qgis_geoms.create_line(QgsGeometry.fromPointXY(end_point2))

    # Get the tangent layer and verify initial tangent geometries
    tangent_layer = QgsProject.instance().mapLayersByName("Tangent Line")[0]
    tangent_features = list(tangent_layer.getFeatures())

    # Verify tangent geometries for all three segments
    assert len(tangent_features) == 3
    for tangent in tangent_features:
        if tangent.attributes()[0] == 'Tangent Line 1':
            assert tangent.geometry().asPolyline() == [
                QgsPointXY(14.41994445867451624, 55.23905443055804199),
                QgsPointXY(14.35950436006096886, 55.26780867173166456)
            ]
        elif tangent.attributes()[0] == 'Tangent Line 2':
            assert tangent.geometry().asPolyline() == [
                QgsPointXY(14.56792552892891379, 55.34421538279835318),
                QgsPointXY(14.50562546536714592, 55.37176484007880362)
            ]
        elif tangent.attributes()[0] == 'Tangent Line 3':
            assert tangent.geometry().asPolyline() == [
                QgsPointXY(14.71515210670152562, 55.42177566433891656),
                QgsPointXY(14.67554678409647018, 55.46065575566534278)
            ]
        else:
            assert False, "unknown line"

    # Modify the second segment
    segment_layer = omrat.qgis_geoms.vector_layers[2]
    segment_feature = next(segment_layer.getFeatures())
    segment_feature.setGeometry(QgsGeometry.fromPolylineXY([mid_point, end_point_mod]))
    segment_layer.dataProvider().changeGeometryValues({segment_feature.id(): segment_feature.geometry()})

    # Call the on_geometry_changed method
    omrat.qgis_geoms.on_geometry_changed(segment_feature.attributes()[0], segment_feature.geometry())

    # Verify that only the second tangent was updated
    tangent_layer = QgsProject.instance().mapLayersByName("Tangent Line")[0]
    updated_tangent_features = list(tangent_layer.getFeatures())
    assert len(updated_tangent_features) == 3
    for tangent in updated_tangent_features:
        if tangent.attributes()[0] == 'Tangent Line 1':
            assert tangent.geometry().asPolyline() == [
                QgsPointXY(14.41994445867451624, 55.23905443055804199), 
                QgsPointXY(14.35950436006096886, 55.26780867173166456)
                ]
        
        elif tangent.attributes()[0] == 'Tangent Line 2':
            assert tangent.geometry().asPolyline() == [
                QgsPointXY(14.539690916698774, 55.34854897137823571),
                QgsPointXY(14.46757931842909528, 55.36674064821078645)
            ]
            
        elif tangent.attributes()[0] == 'Tangent Line 3':
            assert tangent.geometry().asPolyline() == [
                QgsPointXY(14.71515210670152562, 55.42177566433891656),
                QgsPointXY(14.67554678409647018, 55.46065575566534278)
                ]  # Third tangent remains unchanged
        else:
            assert False, "unknown line"



def test_unload(omrat):
    """Test the unload method."""
    # Add a mock layer to the project
    layer = QgsVectorLayer("Point?crs=EPSG:4326", "TestLayer", "memory")
    QgsProject.instance().addMapLayer(layer)
    omrat.qgis_geoms.vector_layers.append(layer)

    # Call the unload method
    omrat.qgis_geoms.unload()

    # Verify that the layer was removed
    assert not QgsProject.instance().mapLayersByName("TestLayer")
    assert omrat.qgis_geoms.vector_layers is None