"""Tests for the Settings > Ship type mapping... controller.

The widget is a MagicMock, the AIS handler a SimpleNamespace with a
mocked DB, and the file / message dialogs are patched, so this runs
without a Qt event loop (``--noconftest -p no:qgis``).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omrat_utils.ship_type_map import MappingRow, ShipTypeMapConfig  # noqa: E402
from omrat_utils.ship_type_map_dialog import PREVIEW_LIMIT, ShipTypeMapping  # noqa: E402

MOD = "omrat_utils.ship_type_map_dialog"


def _db(count=(3, 2, 3), rows=None, fail=False):
    db = MagicMock()

    def _exec(sql, return_error=False, params=None):
        text = repr(sql)
        if fail:
            return (False, [["relation does not exist"]])
        if "count(*)" in text:
            return (True, [count])
        return (True, rows if rows is not None else [
            (265123000, 9123456, 19, "MT X"), (265999000, None, 18, None), (None, 9000001, 17, None),
        ])

    db.execute_and_return.side_effect = _exec
    return db


@pytest.fixture
def ctl():
    widget = MagicMock()
    widget.cbEnabled.isChecked.return_value = False
    widget.leSchema.text.return_value = "omrat"
    widget.leTable.text.return_value = ""
    ais = SimpleNamespace(
        db=_db(), type_map=ShipTypeMapConfig(enabled=True, schema="omrat", table="ship_type_map"),
        set_type_map=MagicMock(),
    )
    omrat = SimpleNamespace(ais=ais)
    c = ShipTypeMapping(omrat, widget_factory=lambda: widget)
    return c


class TestRunAndApply:
    def test_run_prefills_from_ais_config_and_execs(self, ctl):
        ctl.run()
        ctl.dlg.cbEnabled.setChecked.assert_called_with(True)
        ctl.dlg.leSchema.setText.assert_called_with("omrat")
        ctl.dlg.leTable.setText.assert_called_with("ship_type_map")
        ctl.dlg.exec.assert_called_once()

    def test_apply_hands_config_to_ais(self, ctl):
        ctl.dlg.cbEnabled.isChecked.return_value = True
        ctl.dlg.leSchema.text.return_value = " ais "
        ctl.dlg.leTable.text.return_value = "my_map"
        cfg = ctl.apply()
        assert cfg == ShipTypeMapConfig(enabled=True, schema="ais", table="my_map")
        ctl.omrat.ais.set_type_map.assert_called_once_with(cfg)

    def test_blank_table_falls_back_to_default(self, ctl):
        assert ctl.apply().table == "ship_type_map"


class TestRefresh:
    def test_fills_status_and_preview(self, ctl):
        ctl.refresh()
        status = ctl.dlg.lblStatus.setText.call_args.args[0]
        assert "omrat.ship_type_map" in status and "3 rows" in status and "2 with IMO" in status
        ctl.dlg.twPreview.setRowCount.assert_called_with(3)
        # first row: mmsi, imo, "19: Tanker...", note
        texts = [c.args[2].text() if hasattr(c.args[2], "text") else c.args[2]
                 for c in ctl.dlg.twPreview.setItem.call_args_list[:4]]
        assert texts[0] == "265123000" and texts[1] == "9123456"
        assert texts[2].startswith("19: Tanker") and texts[3] == "MT X"

    def test_missing_table_message(self, ctl):
        ctl.omrat.ais.db = _db(fail=True)
        ctl.refresh()
        assert "does not exist yet" in ctl.dlg.lblStatus.setText.call_args.args[0]
        ctl.dlg.twPreview.setRowCount.assert_called_with(0)

    def test_no_db_message(self, ctl):
        ctl.omrat.ais.db = None
        ctl.refresh()
        assert "No AIS database connection" in ctl.dlg.lblStatus.setText.call_args.args[0]

    def test_quiet_refresh_with_blank_schema_hints_instead_of_complaining(self, ctl):
        ctl.dlg.leSchema.text.return_value = ""
        ctl.refresh(quiet=True)
        assert "import a CSV" in ctl.dlg.lblStatus.setText.call_args.args[0]
        ctl.refresh()
        assert "plain SQL names" in ctl.dlg.lblStatus.setText.call_args.args[0]

    def test_large_table_says_first_n_shown(self, ctl):
        ctl.omrat.ais.db = _db(count=(PREVIEW_LIMIT + 5, 0, PREVIEW_LIMIT + 5))
        ctl.refresh()
        assert f"first {PREVIEW_LIMIT} shown" in ctl.dlg.lblStatus.setText.call_args.args[0]


class TestImport:
    def _csv(self, tmp_path, text):
        p = tmp_path / "map.csv"
        p.write_text(text, encoding="utf-8")
        return str(p)

    def test_import_writes_rows_enables_and_refreshes(self, ctl, tmp_path):
        path = self._csv(tmp_path, "mmsi,ship_type\n265123000,Tanker\n265123001,Cargo\n")
        with patch(f"{MOD}.QFileDialog.getOpenFileName", return_value=(path, "")), \
             patch(f"{MOD}.import_mapping", return_value=2) as imp, \
             patch(f"{MOD}.QgsMessageLog"):
            written = ctl.import_csv()
        assert written == 2
        db, cfg, rows = imp.call_args.args
        assert db is ctl.omrat.ais.db
        assert cfg == ShipTypeMapConfig(enabled=True, schema="omrat", table="ship_type_map")
        assert [r.ship_type for r in rows] == [19, 18]
        assert imp.call_args.kwargs == {"replace": True}
        ctl.dlg.cbEnabled.setChecked.assert_called_with(True)

    def test_cancelled_file_dialog_does_nothing(self, ctl):
        with patch(f"{MOD}.QFileDialog.getOpenFileName", return_value=("", "")), \
             patch(f"{MOD}.import_mapping") as imp:
            assert ctl.import_csv() is None
        imp.assert_not_called()

    def test_bad_rows_ask_before_import_and_no_aborts(self, ctl, tmp_path):
        path = self._csv(tmp_path, "mmsi,ship_type\n265123000,Tanker\nabc,Tanker\n")
        with patch(f"{MOD}.QFileDialog.getOpenFileName", return_value=(path, "")), \
             patch(f"{MOD}.QMessageBox") as mb, \
             patch(f"{MOD}.import_mapping") as imp:
            mb.question.return_value = mb.StandardButton.No
            assert ctl.import_csv() is None
        mb.question.assert_called_once()
        imp.assert_not_called()

    def test_unreadable_file_is_reported(self, ctl, tmp_path):
        path = self._csv(tmp_path, "name,foo\nx,y\n")
        with patch(f"{MOD}.QFileDialog.getOpenFileName", return_value=(path, "")), \
             patch(f"{MOD}.QMessageBox") as mb, \
             patch(f"{MOD}.import_mapping") as imp:
            assert ctl.import_csv() is None
        mb.critical.assert_called_once()
        imp.assert_not_called()

    def test_invalid_schema_blocks_import(self, ctl):
        ctl.dlg.leSchema.text.return_value = "bad-schema"
        with patch(f"{MOD}.QMessageBox") as mb, patch(f"{MOD}.QFileDialog") as fd:
            assert ctl.import_csv() is None
        mb.warning.assert_called_once()
        fd.getOpenFileName.assert_not_called()

    def test_no_db_blocks_import(self, ctl):
        ctl.omrat.ais.db = None
        with patch(f"{MOD}.QMessageBox") as mb, patch(f"{MOD}.QFileDialog") as fd:
            assert ctl.import_csv() is None
        mb.warning.assert_called_once()
        fd.getOpenFileName.assert_not_called()

    def test_db_error_is_reported(self, ctl, tmp_path):
        path = self._csv(tmp_path, "mmsi,ship_type\n265123000,Tanker\n")
        with patch(f"{MOD}.QFileDialog.getOpenFileName", return_value=(path, "")), \
             patch(f"{MOD}.QMessageBox") as mb, \
             patch(f"{MOD}.import_mapping", side_effect=RuntimeError("permission denied")):
            assert ctl.import_csv() is None
        assert "permission denied" in mb.critical.call_args.args[2]


class TestExport:
    def test_export_writes_csv(self, ctl, tmp_path):
        out = str(tmp_path / "out.csv")
        with patch(f"{MOD}.QFileDialog.getSaveFileName", return_value=(out, "")):
            n = ctl.export_csv()
        assert n == 3
        from omrat_utils.ship_type_map import read_mapping_csv
        rows, errors = read_mapping_csv(out)
        assert errors == []
        assert rows[0] == MappingRow(265123000, 9123456, 19, "MT X")
        assert "Wrote 3 rows" in ctl.dlg.lblStatus.setText.call_args.args[0]

    def test_export_of_missing_table_warns(self, ctl):
        ctl.omrat.ais.db = _db(fail=True)
        with patch(f"{MOD}.QMessageBox") as mb, patch(f"{MOD}.QFileDialog") as fd:
            assert ctl.export_csv() is None
        mb.warning.assert_called_once()
        fd.getSaveFileName.assert_not_called()
