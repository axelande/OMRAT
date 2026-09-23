"""Standalone tests for ``omrat_utils/ship_type_map.py``.

Covers the value parser, the SQL fragments, the CSV reader and the
importer (against a mocked psycopg2 connection).  No QGIS needed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omrat_utils.ship_type_map import (  # noqa: E402
    N_SHIP_TYPES, OTHER_TYPE, MappingRow, ShipTypeMapConfig, count_mapping, fetch_mapping_rows,
    get_type, import_mapping, parse_ship_type, read_mapping_csv, resolve_ship_type,
    statics_have_imo, write_mapping_csv,
)


# --------------------------------------------------------------------------- parsing

class TestParseShipType:
    @pytest.mark.parametrize("value, expected", [
        (0, 0), (19, 19), (20, 20), ("7", 7), (" 18 ", 18), (17.0, 17),
    ])
    def test_index_passthrough(self, value, expected):
        assert parse_ship_type(value) == expected

    @pytest.mark.parametrize("value, expected", [
        (30, 0), ("70", 18), (89, 19), (99, OTHER_TYPE), (45, 7),
    ])
    def test_ais_codes_are_converted(self, value, expected):
        assert parse_ship_type(value) == expected == get_type(value)

    @pytest.mark.parametrize("value, expected", [
        ("Tanker", 19), ("tanker", 19), ("CARGO", 18), ("ferry", 17),
        ("Passenger, all ships of this type", 17), ("hsc", 7), ("Pleasure", 6),
        ("High speed craft", 7), ("other", 20), ("Fishing", 0), ("LNG", 19),
    ])
    def test_names_and_aliases(self, value, expected):
        assert parse_ship_type(value) == expected

    @pytest.mark.parametrize("value", [
        None, "", "   ", -1, 21, 29, 100, 3.5, "spaceship", True, "Tank er",
    ])
    def test_unparseable_returns_none(self, value):
        assert parse_ship_type(value) is None

    def test_every_full_name_round_trips(self):
        from omrat_utils.ship_type_map import SHIP_TYPE_NAMES
        assert len(SHIP_TYPE_NAMES) == N_SHIP_TYPES == 21
        for idx, name in enumerate(SHIP_TYPE_NAMES):
            assert parse_ship_type(name) == idx


class TestResolveShipType:
    def test_mapped_value_wins_over_ais_code(self):
        assert resolve_ship_type(19, 70) == 19

    def test_none_falls_back_to_ais_code(self):
        assert resolve_ship_type(None, 70) == 18

    def test_garbage_falls_back_to_ais_code(self):
        assert resolve_ship_type("??", 80) == 19
        assert resolve_ship_type(-5, 60) == 17

    def test_both_missing_is_other(self):
        assert resolve_ship_type(None, None) == OTHER_TYPE


# --------------------------------------------------------------------------- config / SQL

class TestShipTypeMapConfig:
    def test_default_is_disabled_and_invalid(self):
        cfg = ShipTypeMapConfig()
        assert cfg.enabled is False
        assert cfg.table == "ship_type_map"
        assert not cfg.is_valid()

    @pytest.mark.parametrize("schema, table", [
        ("", "t"), ("omrat", ""), ("om-rat", "t"), ("omrat", "t;drop"), ("1abc", "t"),
    ])
    def test_bad_identifiers_are_invalid(self, schema, table):
        assert not ShipTypeMapConfig(enabled=True, schema=schema, table=table).is_valid()

    def test_valid_config(self):
        cfg = ShipTypeMapConfig(enabled=True, schema="omrat", table="ship_type_map")
        assert cfg.is_valid()
        assert cfg.qualified_name() == "omrat.ship_type_map"

    def test_build_ctes_dedupes_both_keys(self):
        cfg = ShipTypeMapConfig(enabled=True, schema="omrat", table="ship_type_map")
        ctes = cfg.build_ctes()
        assert "type_map_mmsi AS (SELECT DISTINCT ON (mmsi) mmsi, ship_type FROM omrat.ship_type_map" in ctes
        assert "type_map_imo AS (SELECT DISTINCT ON (imo) imo, ship_type FROM omrat.ship_type_map" in ctes
        assert "WHERE mmsi IS NOT NULL" in ctes
        assert "WHERE imo IS NOT NULL" in ctes

    def test_build_ctes_refuses_invalid_config(self):
        with pytest.raises(ValueError):
            ShipTypeMapConfig(enabled=False, schema="omrat").build_ctes()

    def test_joins_and_expr_without_imo(self):
        assert ShipTypeMapConfig.build_joins(False) == (
            " LEFT OUTER JOIN type_map_mmsi tmm ON tmm.mmsi = ss.mmsi"
        )
        assert ShipTypeMapConfig.build_ship_type_expr("NULL::int", False) == (
            "COALESCE(tmm.ship_type, NULL::int)"
        )

    def test_joins_and_expr_with_imo_put_imo_first(self):
        joins = ShipTypeMapConfig.build_joins(True)
        assert joins.index("tmi.imo = ss.imo_num") < joins.index("tmm.mmsi = ss.mmsi")
        assert ShipTypeMapConfig.build_ship_type_expr("ext.ext_ship_type", True) == (
            "COALESCE(tmi.ship_type, tmm.ship_type, ext.ext_ship_type)"
        )

    def test_dict_round_trip_ignores_unknown_keys(self):
        cfg = ShipTypeMapConfig(enabled=True, schema="s", table="t")
        data = cfg.to_dict()
        data["bogus"] = 1
        assert ShipTypeMapConfig.from_dict(data) == cfg


# --------------------------------------------------------------------------- CSV

def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


class TestReadMappingCsv:
    def test_comma_file_with_both_keys(self, tmp_path):
        path = _write(tmp_path, "m.csv",
                      "mmsi,imo,ship_type,note\n"
                      "265123000,9123456,Tanker,MT Example\n"
                      "265999000,,18,\n"
                      ",9555555,80,\n")
        rows, errors = read_mapping_csv(path)
        assert errors == []
        assert rows == [
            MappingRow(mmsi=265123000, imo=9123456, ship_type=19, note="MT Example"),
            MappingRow(mmsi=265999000, imo=None, ship_type=18, note=None),
            MappingRow(mmsi=None, imo=9555555, ship_type=19, note=None),
        ]

    def test_semicolon_file_and_header_variants(self, tmp_path):
        path = _write(tmp_path, "m.csv",
                      "IMO Number;Type\n"
                      "IMO 9123456;ferry\n"
                      "9000001;cargo\n")
        rows, errors = read_mapping_csv(path)
        assert errors == []
        assert [(r.imo, r.ship_type) for r in rows] == [(9123456, 17), (9000001, 18)]
        assert all(r.mmsi is None for r in rows)

    def test_bad_rows_are_reported_and_skipped(self, tmp_path):
        path = _write(tmp_path, "m.csv",
                      "mmsi,ship_type\n"
                      "265123000,Tanker\n"
                      "abc,Tanker\n"
                      "265123001,spaceship\n"
                      ",Cargo\n"
                      "0,Cargo\n"
                      "\n"
                      "265123002,7\n")
        rows, errors = read_mapping_csv(path)
        assert [r.mmsi for r in rows] == [265123000, 265123002]
        assert len(errors) == 4
        assert errors[0].startswith("line 3:")
        assert "spaceship" in errors[1]
        assert "no MMSI or IMO" in errors[2]
        assert "no MMSI or IMO" in errors[3]

    def test_missing_type_column(self, tmp_path):
        path = _write(tmp_path, "m.csv", "mmsi,name\n1,2\n")
        rows, errors = read_mapping_csv(path)
        assert rows == [] and len(errors) == 1 and "ship type column" in errors[0]

    def test_missing_key_columns(self, tmp_path):
        path = _write(tmp_path, "m.csv", "name,ship_type\nx,Tanker\n")
        rows, errors = read_mapping_csv(path)
        assert rows == [] and "mmsi" in errors[0]

    def test_empty_file(self, tmp_path):
        path = _write(tmp_path, "m.csv", "")
        assert read_mapping_csv(path) == ([], ["The file is empty"])

    def test_bom_is_tolerated(self, tmp_path):
        p = tmp_path / "bom.csv"
        p.write_bytes(b"\xef\xbb\xbfmmsi,ship_type\r\n1,Tug\r\n")
        rows, errors = read_mapping_csv(str(p))
        assert errors == [] and rows[0].ship_type == 10


# --------------------------------------------------------------------------- database

def _fake_db():
    db = MagicMock()
    cur = MagicMock()
    db.conn.cursor.return_value = cur
    return db, cur


def _executed_sql(cur) -> list[str]:
    return [repr(call.args[0]) for call in cur.execute.call_args_list]


class TestImportMapping:
    def test_creates_table_truncates_and_inserts_in_chunks(self):
        db, cur = _fake_db()
        cfg = ShipTypeMapConfig(enabled=True, schema="omrat", table="ship_type_map")
        rows = [MappingRow(mmsi=i, imo=None, ship_type=19) for i in range(1, 8)]
        written = import_mapping(db, cfg, rows, replace=True, chunk=3)
        assert written == 7
        sqls = _executed_sql(cur)
        assert any("CREATE TABLE IF NOT EXISTS" in s and "'ship_type_map'" in s for s in sqls)
        assert sum("CREATE INDEX IF NOT EXISTS" in s for s in sqls) == 2
        assert any("TRUNCATE" in s for s in sqls)
        inserts = [c for c in cur.execute.call_args_list if "INSERT INTO" in repr(c.args[0])]
        assert len(inserts) == 3  # 3 + 3 + 1
        # first chunk: 3 rows x 4 params
        assert inserts[0].args[1] == [1, None, 19, None, 2, None, 19, None, 3, None, 19, None]
        db.conn.commit.assert_called_once()
        db.conn.rollback.assert_not_called()

    def test_append_mode_skips_truncate(self):
        db, cur = _fake_db()
        cfg = ShipTypeMapConfig(enabled=True, schema="omrat", table="ship_type_map")
        import_mapping(db, cfg, [MappingRow(1, None, 0)], replace=False)
        assert not any("TRUNCATE" in s for s in _executed_sql(cur))

    def test_failure_rolls_back_and_reraises(self):
        db, cur = _fake_db()
        cur.execute.side_effect = RuntimeError("boom")
        cfg = ShipTypeMapConfig(enabled=True, schema="omrat", table="ship_type_map")
        with pytest.raises(RuntimeError):
            import_mapping(db, cfg, [MappingRow(1, None, 0)])
        db.conn.rollback.assert_called_once()
        db.conn.commit.assert_not_called()

    def test_invalid_config_raises_before_touching_db(self):
        db, cur = _fake_db()
        with pytest.raises(ValueError):
            import_mapping(db, ShipTypeMapConfig(enabled=True, schema="bad-schema"), [])
        cur.execute.assert_not_called()


class TestCountAndProbe:
    def test_count_mapping(self):
        db = MagicMock()
        db.execute_and_return.return_value = (True, [(12, 5, 10)])
        cfg = ShipTypeMapConfig(enabled=True, schema="omrat")
        assert count_mapping(db, cfg) == (12, 5, 10)

    def test_count_mapping_none_when_missing_or_unconfigured(self):
        db = MagicMock()
        db.execute_and_return.return_value = (False, [["relation does not exist"]])
        assert count_mapping(db, ShipTypeMapConfig(enabled=True, schema="omrat")) is None
        assert count_mapping(db, ShipTypeMapConfig()) is None
        assert count_mapping(None, ShipTypeMapConfig(enabled=True, schema="omrat")) is None

    def test_statics_have_imo(self):
        db = MagicMock()
        db.execute_and_return.return_value = (True, [(1,)])
        assert statics_have_imo(db, "ais", 2024) is True
        params = db.execute_and_return.call_args.kwargs["params"]
        assert params == ("ais", "statics_2024")
        db.execute_and_return.return_value = (True, [])
        assert statics_have_imo(db, "ais", 2024) is False
        assert statics_have_imo(None, "ais", 2024) is False


class TestFetchAndExport:
    def test_fetch_mapping_rows_with_limit(self):
        db = MagicMock()
        db.execute_and_return.return_value = (True, [(265123000, 9123456, 19, "MT X"), (None, 9000001, 18, None)])
        cfg = ShipTypeMapConfig(enabled=True, schema="omrat")
        rows = fetch_mapping_rows(db, cfg, limit=50)
        assert rows == [
            MappingRow(265123000, 9123456, 19, "MT X"),
            MappingRow(None, 9000001, 18, None),
        ]
        call = db.execute_and_return.call_args
        assert "LIMIT" in repr(call.args[0]) and call.kwargs["params"] == (50,)

    def test_fetch_mapping_rows_without_limit_and_failure(self):
        db = MagicMock()
        db.execute_and_return.return_value = (True, [])
        cfg = ShipTypeMapConfig(enabled=True, schema="omrat")
        assert fetch_mapping_rows(db, cfg, limit=None) == []
        assert "LIMIT" not in repr(db.execute_and_return.call_args.args[0])
        db.execute_and_return.return_value = (False, [["no table"]])
        assert fetch_mapping_rows(db, cfg) is None
        assert fetch_mapping_rows(None, cfg) is None

    def test_write_then_read_round_trips(self, tmp_path):
        rows = [MappingRow(265123000, 9123456, 19, "MT X"), MappingRow(None, 9000001, 7, None)]
        path = str(tmp_path / "out.csv")
        write_mapping_csv(path, rows)
        text = open(path, encoding="utf-8").read()
        assert text.splitlines()[0] == "mmsi,imo,ship_type,ship_type_name,note"
        assert "Tanker, all ships of this type" in text
        back, errors = read_mapping_csv(path)
        assert errors == []
        assert back == rows
