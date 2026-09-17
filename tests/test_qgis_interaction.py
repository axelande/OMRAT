import pytest
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsPointXY, QgsGeometry, QgsPoint
)


def test_add_new_route(omrat):
    """Test the add_new_route method."""
    omrat.qgis_geoms.add_new_route()
    assert omrat.qgis_geoms.current_start_point is None
    assert omrat.qgis_geoms.point_layer is None
    assert omrat.qgis_geoms.mapTool is not None


def test_create_point(omrat):
    """Test the create_point method."""
    point = QgsPoint(10, 20)
    omrat.qgis_geoms.create_point(point)

    # Check that the point layer was created and added to the project
    assert omrat.qgis_geoms.point_layer is not None
    assert omrat.qgis_geoms.current_start_point == QgsPointXY(point.x(), point.y())
    assert QgsProject.instance().mapLayersByName("StartPoint")


def test_create_line(omrat):
    """Test the create_line method."""
    start_point = QgsPointXY(10, 20)
    end_point = QgsPoint(30, 40)
    omrat.qgis_geoms.current_start_point = start_point
    omrat.qgis_geoms.create_line(end_point)

    # Check that the line layer was created and added to the project
    assert len(omrat.qgis_geoms.vector_layers) == 2
    route_id = omrat.qgis_geoms.cur_route_id
    leg_no = omrat.qgis_geoms.route_leg_no
    assert QgsProject.instance().mapLayersByName(f"LEG_{route_id}_{leg_no}")


def _draw_leg(omrat, start, end):
    omrat.qgis_geoms.current_start_point = QgsPointXY(*start)
    omrat.qgis_geoms.create_line(QgsPoint(*end))
    return str(omrat.qgis_geoms.segment_id)


def test_next_leg_number_reset_never_overwrites_existing_legs(omrat):
    """Regression: setting the *Next leg* spinbox back to 1 for a second
    route used to reuse ``Segment_Id`` 1 and overwrite the first route's
    leg in ``segment_data`` (the legs then vanished when a split rebuilt
    the canvas from the dict).  The spinbox now only drives the leg
    number in the name; the key is always fresh."""
    qg = omrat.qgis_geoms
    first = _draw_leg(omrat, (10.0, 20.0), (11.0, 20.0))
    second = _draw_leg(omrat, (11.0, 20.0), (12.0, 20.0))
    before = {k: dict(v) for k, v in omrat.segment_data.items()}
    assert set(before) >= {first, second}

    # User starts route 2 and resets the leg number to 1 by hand.
    qg._on_route_id_changed(2)
    assert qg.route_leg_no == 0
    qg._on_next_leg_id_changed(1)
    third = _draw_leg(omrat, (10.0, 25.0), (11.0, 25.0))
    fourth = _draw_leg(omrat, (11.0, 25.0), (12.0, 25.0))

    assert len({first, second, third, fourth}) == 4
    for key, seg in before.items():
        assert omrat.segment_data[key]['Start_Point'] == seg['Start_Point']
        assert omrat.segment_data[key]['End_Point'] == seg['End_Point']
    assert omrat.segment_data[third]['Leg_name'] == 'LEG_2_1'
    assert omrat.segment_data[fourth]['Leg_name'] == 'LEG_2_2'
    assert omrat.segment_data[third]['Route_Id'] == 2
    assert QgsProject.instance().mapLayersByName('LEG_2_1')
    assert QgsProject.instance().mapLayersByName('LEG_1_1')


def test_stop_route_restarts_leg_numbering(omrat):
    qg = omrat.qgis_geoms
    _draw_leg(omrat, (10.0, 20.0), (11.0, 20.0))
    _draw_leg(omrat, (11.0, 20.0), (12.0, 20.0))
    assert qg.route_leg_no == 2
    omrat.stop_route()
    assert qg.cur_route_id == 2
    assert qg.route_leg_no == 0
    assert omrat.main_widget.sbNextLegId.value() == 1
    # Switching the route spinbox back continues after the last number.
    qg._on_route_id_changed(1)
    assert omrat.main_widget.sbNextLegId.value() == 3


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


def _leg_layer_and_fid(omrat, seg_id):
    layer = omrat.qgis_geoms._find_layer_for_seg_id(seg_id)
    assert layer is not None
    return layer, next(layer.getFeatures()).id()


def _tangent_midpoint(seg_id):
    tangent_layer = QgsProject.instance().mapLayersByName("Tangent Line")[0]
    for feat in tangent_layer.getFeatures():
        if feat["type"] == f"Tangent Line {seg_id}":
            a, b = feat.geometry().asPolyline()
            return QgsPointXY((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
    raise AssertionError(f"no tangent for leg {seg_id}")


def _tangent_length(seg_id):
    tangent_layer = QgsProject.instance().mapLayersByName("Tangent Line")[0]
    for feat in tangent_layer.getFeatures():
        if feat["type"] == f"Tangent Line {seg_id}":
            a, b = feat.geometry().asPolyline()
            return a.distance(b)
    raise AssertionError(f"no tangent for leg {seg_id}")


def test_vertex_move_after_commit_updates_model_table_and_tangent(omrat):
    """Regression (kattegatt2.omrat, 2026-09-16): the geometry handler hung
    off the layer's *edit buffer*, which QGIS destroys on commit.  After
    **Stop route** a dragged vertex moved on the canvas only; the model,
    the route table and the tangent kept the old position and the next
    save wrote stale coordinates.  The handler now listens to the layer."""
    start_point = QgsPointXY(14.31942998, 55.20514187)
    end_point = QgsPoint(14.46021114, 55.30168824)
    new_end = QgsPointXY(14.61358249, 55.41424602)
    omrat.qgis_geoms.current_start_point = start_point
    omrat.qgis_geoms.create_line(end_point)
    seg_id = omrat.qgis_geoms.segment_id
    layer, fid = _leg_layer_and_fid(omrat, seg_id)

    # Stop route commits every leg layer -> old edit buffer is gone.
    omrat.stop_route()
    assert not layer.isEditable()

    # A new edit session, as the vertex tool would start.
    assert layer.startEditing()
    assert layer.changeGeometry(fid, QgsGeometry.fromPolylineXY([start_point, new_end]))

    seg = omrat.segment_data[str(seg_id)]
    assert seg['End_Point'] == f"{new_end.x():.6f} {new_end.y():.6f}"
    table = omrat.main_widget.twRouteList
    row = next(r for r in range(table.rowCount()) if table.item(r, 0).text() == str(seg_id))
    assert table.item(row, 4).text() == seg['End_Point']
    # Tangent redrawn on the new leg: its midpoint sits at the new leg
    # midpoint (interpolated in UTM, hence the loose tolerance of about
    # 100 m) and nowhere near the old one.
    mid = _tangent_midpoint(seg_id)
    expect = QgsPointXY((start_point.x() + new_end.x()) / 2, (start_point.y() + new_end.y()) / 2)
    old_mid = QgsPointXY((start_point.x() + end_point.x()) / 2, (start_point.y() + end_point.y()) / 2)
    assert mid.distance(expect) < 1e-3
    assert mid.distance(old_mid) > 0.05


def test_on_width_changed(omrat):
    """Editing the Width cell redraws the tangent with the new half-width."""
    start_point = QgsPointXY(14.31942998, 55.20514187)
    end_point = QgsPoint(14.46021114, 55.30168824)
    omrat.qgis_geoms.current_start_point = start_point
    omrat.qgis_geoms.create_line(end_point)
    seg_id = omrat.qgis_geoms.segment_id
    before = _tangent_length(seg_id)

    table = omrat.main_widget.twRouteList
    row = next(r for r in range(table.rowCount()) if table.item(r, 0).text() == str(seg_id))
    table.item(row, 5).setText('8000')

    after = _tangent_length(seg_id)
    assert after == pytest.approx(before * 8000 / 5000, rel=1e-3)
    assert float(omrat.segment_data[str(seg_id)]['Width']) == 8000


def test_moving_shared_vertex_propagates_to_neighbour(omrat):
    """Dragging the junction between leg 2 and leg 3 (after a commit) moves
    leg 3's start too, in the model, the table and the tangent."""
    p0 = QgsPointXY(14.31942998, 55.20514187)
    p1 = QgsPointXY(14.46021114, 55.30168824)
    p2 = QgsPointXY(14.61358249, 55.41424602)
    p3 = QgsPointXY(14.77725490, 55.46813418)
    p2_new = QgsPointXY(14.54719788, 55.41359631)
    omrat.qgis_geoms.current_start_point = p0
    ids = []
    for pt in (p1, p2, p3):
        omrat.qgis_geoms.create_line(QgsPoint(pt.x(), pt.y()))
        ids.append(omrat.qgis_geoms.segment_id)
    leg2, leg3 = ids[1], ids[2]
    tangent_layer = QgsProject.instance().mapLayersByName("Tangent Line")[0]
    assert tangent_layer.featureCount() == 3

    omrat.stop_route()
    layer, fid = _leg_layer_and_fid(omrat, leg2)
    assert layer.startEditing()
    assert layer.changeGeometry(fid, QgsGeometry.fromPolylineXY([p1, p2_new]))

    new_wkt = f"{p2_new.x():.6f} {p2_new.y():.6f}"
    assert omrat.segment_data[str(leg2)]['End_Point'] == new_wkt
    assert omrat.segment_data[str(leg3)]['Start_Point'] == new_wkt
    table = omrat.main_widget.twRouteList
    row3 = next(r for r in range(table.rowCount()) if table.item(r, 0).text() == str(leg3))
    assert table.item(row3, 3).text() == new_wkt
    expect = QgsPointXY((p2_new.x() + p3.x()) / 2, (p2_new.y() + p3.y()) / 2)
    old_mid = QgsPointXY((p2.x() + p3.x()) / 2, (p2.y() + p3.y()) / 2)
    assert _tangent_midpoint(leg3).distance(expect) < 1e-3
    assert _tangent_midpoint(leg3).distance(old_mid) > 0.02
    # The neighbour's blue line on the canvas follows too (node
    # coordinates are kept at six decimals, hence the 1e-6 tolerance).
    layer3, _ = _leg_layer_and_fid(omrat, leg3)
    first_vertex = next(layer3.getFeatures()).geometry().asPolyline()[0]
    assert first_vertex.distance(p2_new) < 1e-6


def test_unload(omrat):
    """Test the unload method."""
    layer = QgsVectorLayer("Point?crs=EPSG:4326", "TestLayer", "memory")
    QgsProject.instance().addMapLayer(layer)
    omrat.qgis_geoms.vector_layers.append(layer)

    omrat.qgis_geoms.unload()

    assert not QgsProject.instance().mapLayersByName("TestLayer")
    assert omrat.qgis_geoms.vector_layers == []


# ---------------------------------------------------------------------------
# Waypoints (v0.15.2): legs hang on shared nodes
# ---------------------------------------------------------------------------

def _wkt(p):
    return f"{p.x():.6f} {p.y():.6f}"


def test_click_near_existing_node_snaps_and_shares_it(omrat):
    qg = omrat.qgis_geoms
    p0 = QgsPointXY(14.30, 55.20)
    p1 = QgsPointXY(14.40, 55.30)
    p2 = QgsPointXY(14.50, 55.40)
    first = _draw_leg(omrat, (p0.x(), p0.y()), (p1.x(), p1.y()))
    omrat.stop_route()
    # New route, first click 0.6 m from the end of the first leg.
    qg.onMapClick(QgsPoint(p1.x() + 0.00001, p1.y()))
    assert qg.current_start_point == p1
    qg.onMapClick(QgsPoint(p2.x(), p2.y()))
    second = str(qg.segment_id)
    sd = omrat.segment_data
    assert sd[second]['Start_Point'] == _wkt(p1)
    assert sd[second]['start_wp'] == sd[first]['end_wp']
    assert len(omrat.waypoints) == 3


def test_dragging_a_junction_moves_every_leg_on_it(omrat):
    a = QgsPointXY(14.30, 55.20)
    b = QgsPointXY(14.40, 55.30)
    c = QgsPointXY(14.50, 55.40)
    d = QgsPointXY(14.50, 55.20)
    b_new = QgsPointXY(14.42, 55.31)
    leg1 = _draw_leg(omrat, (a.x(), a.y()), (b.x(), b.y()))
    leg2 = _draw_leg(omrat, (b.x(), b.y()), (c.x(), c.y()))
    leg3 = _draw_leg(omrat, (b.x(), b.y()), (d.x(), d.y()))
    sd = omrat.segment_data
    node = sd[leg1]['end_wp']
    assert sd[leg2]['start_wp'] == node and sd[leg3]['start_wp'] == node
    omrat.stop_route()

    layer, fid = _leg_layer_and_fid(omrat, int(leg1))
    assert layer.startEditing()
    assert layer.changeGeometry(fid, QgsGeometry.fromPolylineXY([a, b_new]))

    assert omrat.waypoints[node] == (b_new.x(), b_new.y())
    for leg in (leg2, leg3):
        assert sd[leg]['Start_Point'] == _wkt(b_new)
        lyr, _ = _leg_layer_and_fid(omrat, int(leg))
        assert next(lyr.getFeatures()).geometry().asPolyline()[0].distance(b_new) < 1e-9
        assert _tangent_midpoint(int(leg)).distance(b) > 0.005
    table = omrat.main_widget.twRouteList
    row3 = next(r for r in range(table.rowCount()) if table.item(r, 0).text() == leg3)
    assert table.item(row3, 3).text() == _wkt(b_new)
    # Untouched node stays where it was.
    assert sd[leg2]['End_Point'] == _wkt(c)


def test_dropping_a_vertex_on_another_node_merges_them(omrat):
    qg = omrat.qgis_geoms
    a = QgsPointXY(14.30, 55.20)
    b = QgsPointXY(14.40, 55.30)
    c = QgsPointXY(14.60, 55.20)
    d = QgsPointXY(14.50, 55.40)
    leg1 = _draw_leg(omrat, (a.x(), a.y()), (b.x(), b.y()))
    omrat.stop_route()
    leg2 = _draw_leg(omrat, (c.x(), c.y()), (d.x(), d.y()))
    omrat.stop_route()
    assert len(omrat.waypoints) == 4

    layer, fid = _leg_layer_and_fid(omrat, int(leg1))
    assert layer.startEditing()
    # Drop leg 1's end 0.6 m from leg 2's end.
    near_d = QgsPointXY(d.x() + 0.00001, d.y())
    assert layer.changeGeometry(fid, QgsGeometry.fromPolylineXY([a, near_d]))

    sd = omrat.segment_data
    assert sd[leg1]['End_Point'] == _wkt(d)
    assert sd[leg1]['end_wp'] == sd[leg2]['end_wp']
    assert len(omrat.waypoints) == 3
    # The canvas line of leg 1 was pulled onto the node too.
    assert next(layer.getFeatures()).geometry().asPolyline()[-1].distance(d) < 1e-9
    assert qg.snap_to_waypoint((d.x(), d.y()))[1] == sd[leg2]['end_wp']


def test_remove_leg_prunes_unused_nodes(omrat):
    qg = omrat.qgis_geoms
    leg1 = _draw_leg(omrat, (14.30, 55.20), (14.40, 55.30))
    _draw_leg(omrat, (14.40, 55.30), (14.50, 55.40))
    assert len(omrat.waypoints) == 3
    table = omrat.main_widget.twRouteList
    row = next(r for r in range(table.rowCount()) if table.item(r, 0).text() == leg1)
    table.selectRow(row)
    qg.remove_leg()
    assert leg1 not in omrat.segment_data
    assert len(omrat.waypoints) == 2
