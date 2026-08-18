"""Standalone tests for HandleQGISIface.remove_leg.

Imports the real HandleQGISIface (OSGeo4W Python has QGIS available).
QgsProject.instance().removeMapLayer is patched per-test to avoid requiring
a running QGIS application.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries.handle_qgis_iface import HandleQGISIface  # noqa: E402

_PATCH_PROJECT = "geometries.handle_qgis_iface.QgsProject"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_layer(seg_id: int) -> MagicMock:
    """Mock QGIS vector layer whose first feature reports segmentId == seg_id."""
    feat = MagicMock()
    feat.__getitem__ = MagicMock(side_effect=lambda k: seg_id if k == "segmentId" else None)
    layer = MagicMock()
    layer.getFeatures.return_value = [feat]
    layer.editBuffer.return_value = None
    return layer


def _make_table(row_to_seg: dict[int, int]) -> MagicMock:
    """Mock QTableWidget with specific rows selected and seg IDs in column 0."""
    items = []
    for row in row_to_seg:
        item = MagicMock()
        item.row.return_value = row
        items.append(item)

    def _item(row, col):
        if col == 0 and row in row_to_seg:
            m = MagicMock()
            m.text.return_value = str(row_to_seg[row])
            return m
        return None

    table = MagicMock()
    table.selectedItems.return_value = items
    table.item = MagicMock(side_effect=_item)
    return table


def _make_handler(
    row_to_seg: dict[int, int],
    segment_data: dict | None = None,
    traffic_data: dict | None = None,
    leg_dirs: dict | None = None,
    layers: list | None = None,
    buffer_edits: list | None = None,
    tangent_layer=None,
    has_junctions: bool = True,
) -> SimpleNamespace:
    """Build a minimal stub that HandleQGISIface.remove_leg can run against."""
    if segment_data is None:
        segment_data = {str(s): {} for s in row_to_seg.values()}
    if traffic_data is None:
        traffic_data = {str(s): {} for s in row_to_seg.values()}
    if leg_dirs is None:
        leg_dirs = {str(s): [] for s in row_to_seg.values()}
    if layers is None:
        layers = [_make_layer(s) for s in row_to_seg.values()]

    junctions_mock = MagicMock() if has_junctions else None
    run_traffic = MagicMock()
    canvas_mock = MagicMock()
    iface_mock = MagicMock()
    iface_mock.mapCanvas.return_value = canvas_mock

    omrat = SimpleNamespace(
        segment_data=segment_data,
        traffic_data=traffic_data,
        junctions=junctions_mock,
        run_traffic_module=run_traffic,
        iface=iface_mock,
        main_widget=SimpleNamespace(twRouteList=_make_table(row_to_seg)),
    )

    handler = SimpleNamespace(
        omrat=omrat,
        vector_layers=list(layers),
        buffer_edits=list(buffer_edits or []),
        tangent_layer=tangent_layer,
        leg_dirs=leg_dirs,
        _find_layer_for_seg_id=lambda seg_id: next(
            (l for l in layers or [] if any(
                f["segmentId"] == seg_id
                for f in l.getFeatures()
            )),
            None,
        ),
    )
    return handler


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRemoveLegNoOp:
    def test_no_selection_does_nothing(self):
        handler = _make_handler({})
        handler.omrat.main_widget.twRouteList.selectedItems.return_value = []
        segment_data_before = dict(handler.omrat.segment_data)
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        assert handler.omrat.segment_data == segment_data_before

    def test_missing_table_item_skipped(self):
        handler = _make_handler({0: 1})
        handler.omrat.main_widget.twRouteList.item = MagicMock(return_value=None)
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        assert '1' in handler.omrat.segment_data


class TestRemoveLegDataCleanup:
    def test_removes_from_segment_data(self):
        handler = _make_handler({0: 1})
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        assert '1' not in handler.omrat.segment_data

    def test_removes_from_traffic_data(self):
        handler = _make_handler({0: 1})
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        assert '1' not in handler.omrat.traffic_data

    def test_removes_from_leg_dirs(self):
        handler = _make_handler({0: 1})
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        assert '1' not in handler.leg_dirs

    def test_unrelated_keys_preserved(self):
        handler = _make_handler(
            {0: 1},
            segment_data={'1': {}, '2': {}},
            traffic_data={'1': {}, '2': {}},
            leg_dirs={'1': [], '2': []},
        )
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        assert '2' in handler.omrat.segment_data
        assert '2' in handler.omrat.traffic_data
        assert '2' in handler.leg_dirs


class TestRemoveLegLayerCleanup:
    def test_matching_layer_removed_from_list(self):
        layer = _make_layer(1)
        handler = _make_handler({0: 1}, layers=[layer])
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        assert layer not in handler.vector_layers

    def test_non_matching_layer_preserved(self):
        layer1 = _make_layer(1)
        layer2 = _make_layer(2)
        handler = _make_handler({0: 1}, layers=[layer1, layer2])
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        assert layer2 in handler.vector_layers
        assert layer1 not in handler.vector_layers

    def test_edit_buffer_disconnected(self):
        edit_buffer = MagicMock()
        layer = _make_layer(1)
        layer.editBuffer.return_value = edit_buffer
        handler = _make_handler({0: 1}, layers=[layer])
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        edit_buffer.geometryChanged.disconnect.assert_called_once()

    def test_edit_buffer_removed_from_buffer_edits(self):
        edit_buffer = MagicMock()
        layer = _make_layer(1)
        layer.editBuffer.return_value = edit_buffer
        handler = _make_handler({0: 1}, layers=[layer], buffer_edits=[edit_buffer])
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        assert edit_buffer not in handler.buffer_edits

    def test_qgsproject_removeMapLayer_called(self):
        layer = _make_layer(1)
        handler = _make_handler({0: 1}, layers=[layer])
        with patch(_PATCH_PROJECT) as mock_proj:
            HandleQGISIface.remove_leg(handler)
            mock_proj.instance().removeMapLayer.assert_called_with(layer.id())

    def test_no_matching_layer_does_not_crash(self):
        layer = _make_layer(99)
        handler = _make_handler({0: 1}, layers=[layer])
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        assert layer in handler.vector_layers


class TestRemoveLegTangentCleanup:
    def test_remove_existing_tangent_called(self):
        tangent = MagicMock()
        handler = _make_handler({0: 1}, tangent_layer=tangent)
        handler.remove_existing_tangent = MagicMock()
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        handler.remove_existing_tangent.assert_called_once_with(1)

    def test_no_tangent_layer_does_not_crash(self):
        handler = _make_handler({0: 1}, tangent_layer=None)
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)


class TestRemoveLegSideEffects:
    def test_junction_registry_rebuilt(self):
        handler = _make_handler({0: 1})
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        handler.omrat.junctions.rebuild_from_segments.assert_called_once_with(
            handler.omrat.segment_data, prefer_user=True
        )

    def test_no_junctions_does_not_crash(self):
        handler = _make_handler({0: 1}, has_junctions=False)
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)

    def test_traffic_module_refreshed(self):
        handler = _make_handler({0: 1})
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        handler.omrat.run_traffic_module.assert_called_once()

    def test_canvas_refreshed(self):
        handler = _make_handler({0: 1})
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        handler.omrat.iface.mapCanvas().refresh.assert_called_once()

    def test_table_row_removed(self):
        handler = _make_handler({0: 1})
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        handler.omrat.main_widget.twRouteList.removeRow.assert_called_once_with(0)


class TestRemoveLegMultipleRows:
    def test_removes_two_selected_legs(self):
        layer1 = _make_layer(1)
        layer2 = _make_layer(2)
        handler = _make_handler(
            {0: 1, 1: 2},
            segment_data={'1': {}, '2': {}, '3': {}},
            traffic_data={'1': {}, '2': {}, '3': {}},
            leg_dirs={'1': [], '2': [], '3': []},
            layers=[layer1, layer2],
        )
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        assert '1' not in handler.omrat.segment_data
        assert '2' not in handler.omrat.segment_data
        assert '3' in handler.omrat.segment_data

    def test_table_rows_removed_in_reverse_order(self):
        """Rows must be removed high-to-low to keep earlier indices valid."""
        removed = []
        handler = _make_handler({0: 1, 1: 2})
        handler.omrat.main_widget.twRouteList.removeRow = MagicMock(
            side_effect=lambda r: removed.append(r)
        )
        with patch(_PATCH_PROJECT):
            HandleQGISIface.remove_leg(handler)
        assert removed == sorted(removed, reverse=True)


class TestFindLayerForSegId:
    def test_returns_matching_layer(self):
        layer = _make_layer(5)
        handler = SimpleNamespace(vector_layers=[layer])
        result = HandleQGISIface._find_layer_for_seg_id(handler, 5)
        assert result is layer

    def test_returns_none_when_no_match(self):
        layer = _make_layer(99)
        handler = SimpleNamespace(vector_layers=[layer])
        result = HandleQGISIface._find_layer_for_seg_id(handler, 1)
        assert result is None

    def test_returns_none_for_empty_list(self):
        handler = SimpleNamespace(vector_layers=[])
        result = HandleQGISIface._find_layer_for_seg_id(handler, 1)
        assert result is None
