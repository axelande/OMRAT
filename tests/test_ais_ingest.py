"""Headless tests for the AIS ingestion pipeline.

These tests do NOT touch a real Postgres database — they mock the psycopg2
cursor and verify the SQL the pipeline emits.  Run them with::

    /mnt/c/OSGeo4W/apps/Python312/python.exe -m pytest -p no:qgis --noconftest tests/test_ais_ingest.py
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from unittest.mock import MagicMock, call

import pytest

from omrat_utils.db_setup import (
    DEFAULT_MAX_GAP_S,
    DEFAULT_MIN_SED_M,
    DEFAULT_MIN_SVD_KN,
    DEFAULT_SPEED_FLOOR_KN,
    DEFAULT_SPEED_TOLERANCE,
    ConnectionProfile,
    IngestionSettings,
)
from omrat_utils.handle_ais_ingest import (
    IngestionPipeline,
    IngestionResult,
    split_track_at_gaps,
)


def _render(sql) -> str:
    """Render a psycopg2.sql.Composable (or plain string) for substring tests.

    The ingestion pipeline now composes SQL with ``psycopg2.sql.Identifier``
    so SAST tools can see safe identifier handling.  Tests that historically
    asserted against the raw f-string still want to verify that the right
    schema/table identifiers are present, so this walks the Composable tree
    and emits the bare identifier names (no double-quoting) plus the literal
    SQL fragments.
    """
    from psycopg2 import sql as _psql
    if isinstance(sql, str):
        return sql
    if isinstance(sql, _psql.Composed):
        return "".join(_render(s) for s in sql.seq)
    if isinstance(sql, _psql.SQL):
        return sql.string
    if isinstance(sql, _psql.Identifier):
        return ".".join(sql.strings)
    return str(sql)


# ---------------------------------------------------------------------------
# IngestionSettings
# ---------------------------------------------------------------------------


class TestConnectionProfileDsn:
    """``ConnectionProfile.to_dsn`` must always set client_encoding=UTF8."""

    def test_to_dsn_forces_utf8_client_encoding(self):
        # Without this, a Swedish Windows locale's cp1252-encoded
        # PostgreSQL messages crash psycopg2 with UnicodeDecodeError.
        # Regression test for the user's observed 0xf6 (`ö`) failure.
        p = ConnectionProfile(
            host="h", database="d", user="u", password="p", schema="omrat",
        )
        dsn = p.to_dsn()
        assert dsn["client_encoding"] == "UTF8"

    def test_to_dsn_preserves_other_fields(self):
        p = ConnectionProfile(
            host="h", port=5500, database="d", user="u", password="p",
            schema="omrat", sslmode="require",
        )
        dsn = p.to_dsn()
        assert dsn["host"] == "h"
        assert dsn["port"] == 5500
        assert dsn["dbname"] == "d"
        assert dsn["user"] == "u"
        assert dsn["password"] == "p"
        assert dsn["sslmode"] == "require"


class TestDecodeLibpqMessage:
    """``decode_libpq_message`` cp1252 fallback for libpq's pre-startup messages.

    Background: even with ``client_encoding='UTF8'`` set, PostgreSQL emits
    some auth/role/db rejection messages in the OS ``lc_messages`` encoding
    *before* it processes startup parameters, so psycopg2 hits a UTF-8
    decode error.  The helper recovers the actual text by decoding the
    bytes as cp1252 (or latin-1 with replacement as a last resort).
    """

    def _decode_err(self, raw: bytes) -> UnicodeDecodeError:
        """Build a real UnicodeDecodeError carrying the given bytes."""
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as err:
            return err
        raise AssertionError("test bytes must be invalid UTF-8")

    def test_returns_decoded_swedish_authentication_message(self):
        from omrat_utils.db_setup import decode_libpq_message

        # The user's actual failure mode: Swedish "password authentication
        # failed for user X" emitted in cp1252.
        bytes_ = (
            'FATAL:  l\xf6senordsautentisering misslyckades '
            'f\xf6r anv\xe4ndaren "u"'
        ).encode("cp1252")
        out = decode_libpq_message(self._decode_err(bytes_))
        assert "lösenordsautentisering misslyckades" in out
        assert 'användaren "u"' in out

    def test_returns_empty_string_for_empty_bytes(self):
        from omrat_utils.db_setup import decode_libpq_message

        err = UnicodeDecodeError("utf-8", b"", 0, 1, "x")
        assert decode_libpq_message(err) == ""

    def test_strips_trailing_whitespace(self):
        from omrat_utils.db_setup import decode_libpq_message

        bytes_ = "FATAL:  ogiltig roll \xf6\n".encode("cp1252")
        out = decode_libpq_message(self._decode_err(bytes_))
        assert not out.endswith(("\n", " "))
        assert "ö" in out

    def test_falls_back_to_latin1_for_non_cp1252_bytes(self):
        """cp1252 has a few invalid byte slots (e.g. 0x81); the helper
        must never itself raise on weird bytes."""
        from omrat_utils.db_setup import decode_libpq_message

        # 0x81 is an undefined cp1252 codepoint; cp1252 strict mode raises.
        bytes_ = b"hello \x81 world"
        err = UnicodeDecodeError("utf-8", bytes_, 0, 1, "x")
        out = decode_libpq_message(err)
        # latin-1 maps 0x81 to U+0081 cleanly, so we get the bytes back.
        assert "hello" in out and "world" in out


class TestIngestionSettings:
    def test_default_values_are_omrat_tuned(self):
        s = IngestionSettings()
        assert s.min_sed_m == DEFAULT_MIN_SED_M == 30.0
        assert s.min_svd_kn == DEFAULT_MIN_SVD_KN == 0.3
        assert s.max_gap_s == DEFAULT_MAX_GAP_S == 3600.0
        assert s.speed_tolerance == DEFAULT_SPEED_TOLERANCE == 0.3
        assert s.speed_floor_kn == DEFAULT_SPEED_FLOOR_KN == 1.0

    def test_to_aissegments_kwargs(self):
        s = IngestionSettings(min_sed_m=12.5, min_svd_kn=0.42)
        assert s.to_aissegments_kwargs() == {
            "min_sed_m": 12.5,
            "min_svd_kn": 0.42,
        }

    def test_to_splitter_kwargs(self):
        s = IngestionSettings(
            max_gap_s=600.0, speed_tolerance=0.5, speed_floor_kn=2.0
        )
        assert s.to_splitter_kwargs() == {
            "max_gap_s": 600.0,
            "speed_tolerance": 0.5,
            "speed_floor_kn": 2.0,
        }

    def test_from_dict_filters_unknown_keys(self):
        s = IngestionSettings.from_dict({
            "min_sed_m": 50.0,
            "min_svd_kn": 0.5,
            "max_gap_s": 1200.0,
            "speed_tolerance": 0.4,
            "speed_floor_kn": 0.5,
            "bogus": "ignored",
        })
        assert s.min_sed_m == 50.0
        assert s.min_svd_kn == 0.5
        assert s.max_gap_s == 1200.0
        assert s.speed_tolerance == 0.4
        assert s.speed_floor_kn == 0.5


# ---------------------------------------------------------------------------
# IngestionResult
# ---------------------------------------------------------------------------


class TestIngestionResult:
    def test_summary_includes_all_counts(self):
        r = IngestionResult(
            n_files=2, n_tracks=10, n_segments=60,
            n_static_rows=5, n_state_rows=10,
            elapsed_seconds=3.14,
        )
        s = r.summary()
        assert "2 file" in s
        assert "10 tracks" in s
        assert "60 segments" in s
        assert "3.1s" in s
        assert "errors" not in s

    def test_summary_flags_errors(self):
        r = IngestionResult(elapsed_seconds=0.0, errors=["x", "y"])
        assert "(2 errors)" in r.summary()


# ---------------------------------------------------------------------------
# IngestionPipeline helpers (no DB needed)
# ---------------------------------------------------------------------------


def _pipeline() -> IngestionPipeline:
    profile = ConnectionProfile(host="x", database="y", user="u", schema="omrat")
    return IngestionPipeline(profile, IngestionSettings())


class TestTsToDt:
    def test_none_passes_through(self):
        assert IngestionPipeline._ts_to_dt(None) is None

    def test_valid_unix_seconds(self):
        dt = IngestionPipeline._ts_to_dt(1625097600.0)  # 2021-07-01 00:00:00 UTC
        assert dt is not None
        assert dt.tzinfo is timezone.utc
        assert dt.year == 2021 and dt.month == 7 and dt.day == 1

    def test_invalid_value_returns_none(self):
        assert IngestionPipeline._ts_to_dt(float("inf")) is None


class TestReadStaticRows:
    def test_picks_up_aisdb_static_table(self, tmp_path):
        db = tmp_path / "test.db"
        with sqlite3.connect(str(db)) as con:
            con.execute("""
                CREATE TABLE ais_202107_static (
                    mmsi INTEGER, time INTEGER,
                    vessel_name TEXT, ship_type INTEGER,
                    call_sign TEXT, imo INTEGER,
                    dim_bow REAL, dim_stern REAL, dim_port REAL, dim_star REAL,
                    draught REAL, destination TEXT,
                    ais_version INTEGER, fixing_device INTEGER,
                    eta_month INTEGER, eta_day INTEGER, eta_hour INTEGER, eta_minute INTEGER,
                    source TEXT
                )
            """)
            con.execute(
                "INSERT INTO ais_202107_static VALUES "
                "(123456, 1625097600, 'TEST', 70, 'XYZ', 9999, "
                "10, 80, 5, 7, 4.5, 'PORT', 0, 0, 7, 1, 12, 0, 'T')"
            )
            con.execute(
                "INSERT INTO ais_202107_static VALUES "
                "(123456, 1625097700, 'TEST_NEWER', 70, 'XYZ', 9999, "
                "10, 80, 5, 7, 4.6, 'PORT', 0, 0, 7, 1, 12, 0, 'T')"
            )
            con.commit()

        result = _pipeline()._read_static_rows(db, year=2021)
        assert 123456 in result
        # Latest record per MMSI wins.
        assert result[123456]["vessel_name"] == "TEST_NEWER"
        assert result[123456]["draught"] == 4.6

    def test_missing_table_returns_empty(self, tmp_path):
        db = tmp_path / "empty.db"
        with sqlite3.connect(str(db)) as con:
            con.execute("CREATE TABLE unrelated (x INTEGER)")
            con.commit()
        assert _pipeline()._read_static_rows(db, year=2021) == {}


class TestInsertHelpers:
    def _segment(
        self, *, t_start: float = 0.0, t_end: float = 60.0,
        lon_start: float = 12.0, lat_start: float = 55.0,
        lon_end: float = 12.001, lat_end: float = 55.0,
    ):
        from aissegments import Segment
        return Segment(
            mmsi=999, t_start=t_start, t_end=t_end,
            lon_start=lon_start, lat_start=lat_start,
            lon_end=lon_end, lat_end=lat_end,
            cog_mean=90.0, sog_mean=10.0, n_points=2,
        )

    def test_insert_static_writes_placeholder_when_rec_is_none(self):
        """rec=None now writes a NULL-fields placeholder row (was: bailed out).
        This guarantees callers always have a non-NULL static_id for the
        states.static_id FK."""
        cur = MagicMock()
        cur.fetchone.return_value = (5,)
        result = _pipeline()._insert_static(cur, "omrat", 2021, mmsi=123, rec=None)
        assert result == 5
        sql_obj, params = cur.execute.call_args[0]
        sql = _render(sql_obj)
        assert "INSERT INTO omrat.statics_2021" in sql
        # All identity fields NULL.
        assert params[2] is None and params[3] is None
        assert params[4] is None and params[5] is None
        assert params[6] is None      # imo_num

    def test_insert_static_writes_identity_columns_only(self):
        """Statics now hold IDENTITY data only — dimensions + IMO, not voyage info."""
        cur = MagicMock()
        cur.fetchone.return_value = (42,)
        rec = {
            "mmsi": 123, "imo": 999, "vessel_name": "TEST",
            "call_sign": "XX", "ship_type": 70,
            "dim_bow": 10, "dim_stern": 80, "dim_port": 5, "dim_star": 7,
            "draught": 4.5, "destination": "PORT", "time": 1625097600,
        }
        sid = _pipeline()._insert_static(cur, "omrat", 2021, mmsi=123, rec=rec)
        assert sid == 42
        sql_obj, params = cur.execute.call_args[0]
        sql = _render(sql_obj)
        assert "INSERT INTO omrat.statics_2021" in sql
        # Voyage fields must NOT appear in statics column list any more.
        assert "draught" not in sql
        assert "destination" not in sql
        assert "type_and_cargo" not in sql
        # Identity fields ARE present.
        assert "dim_a" in sql and "dim_b" in sql and "dim_c" in sql and "dim_d" in sql
        assert "imo_num" in sql

    def test_insert_static_clamps_dim_to_smallint(self):
        cur = MagicMock()
        cur.fetchone.return_value = (1,)
        rec = {"dim_bow": 99999, "dim_stern": 80, "time": 0}  # bow out of smallint range
        _pipeline()._insert_static(cur, "omrat", 2021, mmsi=1, rec=rec)
        params = cur.execute.call_args[0][1]
        # Param ordering: (mmsi, date, dim_a, dim_b, dim_c, dim_d, imo_num)
        assert params[2] is None         # dim_a clamped (bow=99999 out of smallint)
        assert params[3] == 80           # dim_b (stern) preserved

    def test_insert_static_always_inserts_even_with_none(self):
        """Verify the rec=None placeholder INSERT runs (the dedup decision
        is the caller's job — see TestEnsureStatic)."""
        cur = MagicMock()
        cur.fetchone.return_value = (1,)
        result = _pipeline()._insert_static(cur, "omrat", 2021, mmsi=1, rec=None)
        assert result == 1
        cur.execute.assert_called_once()

    def test_insert_state_writes_voyage_columns(self):
        """States now carry the voyage data (draught, type_and_cargo, etc.)."""
        cur = MagicMock()
        cur.fetchone.return_value = (7,)
        rec = {
            "draught": 4.5, "ship_type": 70, "destination": "ROTTERDAM",
            "time": 1625097600,
        }
        sid = _pipeline()._insert_state(
            cur, "omrat", 2021, mmsi=123, static_id=1, rec=rec, fallback_t=0.0,
        )
        assert sid == 7
        sql_obj, params = cur.execute.call_args[0]
        sql = _render(sql_obj)
        assert "INSERT INTO omrat.states_2021" in sql
        assert "draught" in sql and "type_and_cargo" in sql and "destination" in sql
        assert "static_id" in sql
        # Param ordering: (mmsi, date, draught, type_and_cargo, eta, destination, static_id)
        assert params[2] == 4.5          # draught
        assert params[3] == 70           # type_and_cargo
        assert params[4] is None         # eta — v0.1 stores NULL
        assert params[5] == "ROTTERDAM"  # destination
        assert params[6] == 1            # static_id

    def test_insert_state_truncates_destination_to_20_chars(self):
        cur = MagicMock()
        cur.fetchone.return_value = (1,)
        long_dest = "A" * 50
        _pipeline()._insert_state(
            cur, "omrat", 2021, mmsi=1, static_id=None,
            rec={"destination": long_dest, "time": 0},
            fallback_t=0.0,
        )
        params = cur.execute.call_args[0][1]
        assert params[5] == "A" * 20

    def test_insert_state_uses_fallback_t_when_no_static_time(self):
        cur = MagicMock()
        cur.fetchone.return_value = (1,)
        _pipeline()._insert_state(
            cur, "omrat", 2021, mmsi=1, static_id=None,
            rec=None, fallback_t=1625097600.0,
        )
        params = cur.execute.call_args[0][1]
        assert params[1] is not None     # date came from fallback_t

    def test_bulk_insert_segments_groups_by_month(self):
        from datetime import datetime as _dt
        from datetime import timezone as _tz
        # Two segments: one in Jan, one in Feb.
        jan = self._segment(
            t_start=_dt(2021, 1, 5, tzinfo=_tz.utc).timestamp(),
            t_end=_dt(2021, 1, 5, 0, 1, tzinfo=_tz.utc).timestamp(),
        )
        feb = self._segment(
            t_start=_dt(2021, 2, 5, tzinfo=_tz.utc).timestamp(),
            t_end=_dt(2021, 2, 5, 0, 1, tzinfo=_tz.utc).timestamp(),
        )
        from unittest.mock import patch
        cur = MagicMock()
        with patch("omrat_utils.handle_ais_ingest.execute_values") as ev:
            _pipeline()._bulk_insert_segments(
                cur, "omrat", 2021, mmsi=999, state_id=1, segments=[jan, feb],
            )
        assert ev.call_count == 2
        sqls = [_render(c.args[1]) for c in ev.call_args_list]
        assert any("segments_2021_1" in s for s in sqls)
        assert any("segments_2021_2" in s for s in sqls)

    def test_bulk_insert_columns_match_legacy_schema(self):
        """Inserted columns must align with the legacy segments_YYYY layout."""
        from unittest.mock import patch
        seg = self._segment()
        cur = MagicMock()
        with patch("omrat_utils.handle_ais_ingest.execute_values") as ev:
            _pipeline()._bulk_insert_segments(
                cur, "omrat", 2021, mmsi=999, state_id=42, segments=[seg],
            )
        sql = _render(ev.call_args.args[1])
        for col in ("mmsi", "date1", "date2", "segment", "cog", "sog",
                    "route_id", "state_id", "heading"):
            assert col in sql, f"missing column {col!r} in INSERT"
        # n_points must NOT appear (it was a v0.1 column we dropped).
        assert "n_points" not in sql

    def test_bulk_insert_route_id_and_heading_are_null(self):
        """v0.1 ingestion leaves route_id and heading NULL."""
        from unittest.mock import patch
        seg = self._segment()
        cur = MagicMock()
        with patch("omrat_utils.handle_ais_ingest.execute_values") as ev:
            _pipeline()._bulk_insert_segments(
                cur, "omrat", 2021, mmsi=999, state_id=42, segments=[seg],
            )
        rows = ev.call_args.args[2]
        assert len(rows) == 1
        # Tuple ordering: (mmsi, date1, date2, wkt, cog, sog, route_id, state_id, heading)
        row = rows[0]
        assert row[6] is None  # route_id
        assert row[7] == 42    # state_id
        assert row[8] is None  # heading

    def test_bulk_insert_cog_clamped_to_smallint_range(self):
        """COG is rounded and wrapped to [0, 359] for smallint storage."""
        from unittest.mock import patch
        from aissegments import Segment
        seg = Segment(
            mmsi=1, t_start=0.0, t_end=60.0,
            lon_start=0.0, lat_start=0.0, lon_end=0.001, lat_end=0.0,
            cog_mean=359.7, sog_mean=10.0, n_points=2,
        )
        cur = MagicMock()
        with patch("omrat_utils.handle_ais_ingest.execute_values") as ev:
            _pipeline()._bulk_insert_segments(
                cur, "omrat", 2021, mmsi=1, state_id=1, segments=[seg],
            )
        cog_value = ev.call_args.args[2][0][4]
        # round(359.7) = 360 → wrapped to 0 (smallint, valid AIS COG)
        assert cog_value == 0

    def test_bulk_insert_skips_empty(self):
        from unittest.mock import patch
        with patch("omrat_utils.handle_ais_ingest.execute_values") as ev:
            _pipeline()._bulk_insert_segments(
                MagicMock(), "omrat", 2021, mmsi=1, state_id=1, segments=[],
            )
        ev.assert_not_called()

    def test_update_watermark_uses_upsert(self):
        cur = MagicMock()
        _pipeline()._update_watermark(cur, mmsi=123, last_t=1625097600.0, n_segs=5)
        sql = cur.execute.call_args[0][0]
        assert "INSERT INTO omrat_meta.segment_watermark" in sql
        assert "ON CONFLICT (mmsi)" in sql
        assert "GREATEST" in sql


# ---------------------------------------------------------------------------
# IngestionPipeline.run — defensive paths
# ---------------------------------------------------------------------------


class TestFormatDetection:
    def test_nm4_extension(self, tmp_path):
        f = tmp_path / "log.nm4"
        f.write_bytes(b"!AIVDM,1,1,,A,...")
        assert IngestionPipeline._detect_format(f) == "aisdb_nmea"

    def test_nmea_extension(self, tmp_path):
        f = tmp_path / "feed.nmea"
        f.write_bytes(b"!AIVDM,...")
        assert IngestionPipeline._detect_format(f) == "aisdb_nmea"

    def test_aisdb_csv_header(self, tmp_path):
        f = tmp_path / "ais.csv"
        f.write_text(
            "MMSI,Message_ID,Repeat_indicator,Time,Millisecond,Region\n"
            "123,1,0,20210701_000000,0,66\n",
            encoding="utf-8",
        )
        assert IngestionPipeline._detect_format(f) == "aisdb_csv"

    def test_marine_cadastre_header(self, tmp_path):
        f = tmp_path / "mc.csv"
        f.write_text(
            "MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,VesselName\n"
            "123,2019-01-01T14:15:12,26.1,-80.1,0.0,360.0,511.0,X\n",
            encoding="utf-8",
        )
        assert IngestionPipeline._detect_format(f) == "simple_csv"

    def test_canonical_simple_csv_header(self, tmp_path):
        f = tmp_path / "simple.csv"
        f.write_text("mmsi,time,lon,lat,sog,cog\n100,0,12,55,10,90\n", encoding="utf-8")
        assert IngestionPipeline._detect_format(f) == "simple_csv"

    def test_gzipped_simple_csv(self, tmp_path):
        import gzip
        f = tmp_path / "x.csv.gz"
        with gzip.open(f, "wt", encoding="utf-8") as g:
            g.write("mmsi,time,lon,lat,sog,cog\n100,0,12,55,10,90\n")
        assert IngestionPipeline._detect_format(f) == "simple_csv"


class TestFilterToYear:
    def _make_track(self, timestamps):
        from aissegments import Track
        n = len(timestamps)
        import numpy as np
        return Track.from_arrays(
            mmsi=1, t=timestamps,
            lon=np.linspace(0, 0.01, n),
            lat=np.full(n, 55.0),
            sog=np.full(n, 10.0),
            cog=np.full(n, 90.0),
        )

    def test_track_fully_in_year(self):
        from datetime import datetime as _dt
        from datetime import timezone as _tz
        ts = [_dt(2021, 6, 15, tzinfo=_tz.utc).timestamp() + i * 60 for i in range(5)]
        track = self._make_track(ts)
        out = IngestionPipeline._filter_to_year(track, 2021)
        assert out is track  # no slicing needed

    def test_track_fully_outside_year_returns_none(self):
        from datetime import datetime as _dt
        from datetime import timezone as _tz
        ts = [_dt(2020, 6, 15, tzinfo=_tz.utc).timestamp() + i * 60 for i in range(5)]
        out = IngestionPipeline._filter_to_year(self._make_track(ts), 2021)
        assert out is None

    def test_track_partially_in_year_is_sliced(self):
        from datetime import datetime as _dt
        from datetime import timezone as _tz
        ts = [
            _dt(2020, 12, 31, 23, 59, tzinfo=_tz.utc).timestamp(),
            _dt(2021, 1, 1, 0, 1, tzinfo=_tz.utc).timestamp(),
            _dt(2021, 6, 15, tzinfo=_tz.utc).timestamp(),
            _dt(2022, 1, 1, 0, 1, tzinfo=_tz.utc).timestamp(),
        ]
        out = IngestionPipeline._filter_to_year(self._make_track(ts), 2021)
        assert out is not None
        assert len(out) == 2  # only the two pings in 2021


class TestCombineEta:
    def test_all_fields_combine(self):
        from datetime import datetime, timezone
        eta = IngestionPipeline._combine_eta(
            {"eta_month": 7, "eta_day": 15, "eta_hour": 14, "eta_minute": 30},
            year_hint=2021,
        )
        assert eta == datetime(2021, 7, 15, 14, 30, tzinfo=timezone.utc)

    def test_missing_fields_returns_none(self):
        eta = IngestionPipeline._combine_eta(
            {"eta_month": 7, "eta_day": 15}, year_hint=2021,
        )
        assert eta is None

    def test_spec_sentinels_yield_none(self):
        # AIS uses month=0 / day=0 / hour=24 / minute=60 to mean "n/a"
        for sentinel in [
            {"eta_month": 0, "eta_day": 1, "eta_hour": 0, "eta_minute": 0},
            {"eta_month": 7, "eta_day": 0, "eta_hour": 0, "eta_minute": 0},
            {"eta_month": 7, "eta_day": 1, "eta_hour": 24, "eta_minute": 0},
            {"eta_month": 7, "eta_day": 1, "eta_hour": 0, "eta_minute": 60},
        ]:
            assert IngestionPipeline._combine_eta(sentinel, year_hint=2021) is None

    def test_invalid_calendar_date_returns_none(self):
        # Feb 30 is invalid.
        eta = IngestionPipeline._combine_eta(
            {"eta_month": 2, "eta_day": 30, "eta_hour": 12, "eta_minute": 0},
            year_hint=2021,
        )
        assert eta is None

    def test_out_of_range_components(self):
        for rec in [
            {"eta_month": 13, "eta_day": 1, "eta_hour": 0, "eta_minute": 0},
            {"eta_month": 7, "eta_day": 32, "eta_hour": 0, "eta_minute": 0},
        ]:
            assert IngestionPipeline._combine_eta(rec, year_hint=2021) is None

    def test_non_numeric_returns_none(self):
        eta = IngestionPipeline._combine_eta(
            {"eta_month": "not_a_number", "eta_day": 1, "eta_hour": 0, "eta_minute": 0},
            year_hint=2021,
        )
        assert eta is None


class TestStaticIdentityTuple:
    def test_none_yields_all_nulls(self):
        t = IngestionPipeline._static_identity_tuple(None)
        assert t == (None, None, None, None, None)

    def test_extracts_dimensions_and_imo(self):
        t = IngestionPipeline._static_identity_tuple(
            {"dim_bow": 10, "dim_stern": 80, "dim_port": 5, "dim_star": 7, "imo": 9999}
        )
        assert t == (10, 80, 5, 7, 9999)

    def test_clamps_out_of_range_dims(self):
        t = IngestionPipeline._static_identity_tuple(
            {"dim_bow": 99999, "dim_stern": 80, "dim_port": 5, "dim_star": 7, "imo": None}
        )
        # dim_bow out of smallint range → None
        assert t == (None, 80, 5, 7, None)


class TestStateVoyageTuple:
    def test_none_keeps_static_id(self):
        t = IngestionPipeline._state_voyage_tuple(None, static_id=42)
        assert t == (None, None, None, None, 42)

    def test_extracts_voyage_fields(self):
        t = IngestionPipeline._state_voyage_tuple(
            {"draught": 4.5, "ship_type": 70, "destination": "ROTTERDAM"},
            static_id=1,
        )
        assert t == (4.5, 70, "ROTTERDAM", None, 1)

    def test_truncates_long_destination(self):
        t = IngestionPipeline._state_voyage_tuple(
            {"destination": "A" * 50}, static_id=1
        )
        assert t[2] == "A" * 20

    def test_includes_combined_eta_when_year_hint_given(self):
        from datetime import datetime, timezone
        t = IngestionPipeline._state_voyage_tuple(
            {"eta_month": 7, "eta_day": 15, "eta_hour": 14, "eta_minute": 30},
            static_id=1,
            year_hint=2021,
        )
        assert t[3] == datetime(2021, 7, 15, 14, 30, tzinfo=timezone.utc)

    def test_eta_change_invalidates_dedup(self):
        """A different ETA should produce a different voyage tuple, so it
        triggers a new state row even when draught/destination match."""
        a = IngestionPipeline._state_voyage_tuple(
            {"draught": 4.5, "destination": "ROTTERDAM",
             "eta_month": 7, "eta_day": 15, "eta_hour": 14, "eta_minute": 30},
            static_id=1, year_hint=2021,
        )
        b = IngestionPipeline._state_voyage_tuple(
            {"draught": 4.5, "destination": "ROTTERDAM",
             "eta_month": 7, "eta_day": 16, "eta_hour": 14, "eta_minute": 30},
            static_id=1, year_hint=2021,
        )
        assert a != b


class TestEnsureStatic:
    def _result(self) -> IngestionResult:
        return IngestionResult()

    def test_cache_hit_same_tuple_reuses(self):
        pipeline = _pipeline()
        pipeline._statics_cache[123] = (42, (10, 80, 5, 7, 9999))
        cur = MagicMock()
        result = self._result()
        rec = {"dim_bow": 10, "dim_stern": 80, "dim_port": 5, "dim_star": 7, "imo": 9999}
        sid = pipeline._ensure_static(cur, "omrat", 2021, 123, rec, result)
        assert sid == 42
        cur.execute.assert_not_called()
        assert result.n_static_reused == 1
        assert result.n_static_rows == 0

    def test_cache_hit_different_tuple_inserts(self):
        pipeline = _pipeline()
        pipeline._statics_cache[123] = (42, (10, 80, 5, 7, 9999))
        cur = MagicMock()
        cur.fetchone.return_value = (99,)
        result = self._result()
        rec = {"dim_bow": 11, "dim_stern": 80, "dim_port": 5, "dim_star": 7, "imo": 9999}
        sid = pipeline._ensure_static(cur, "omrat", 2021, 123, rec, result)
        assert sid == 99
        assert pipeline._statics_cache[123] == (99, (11, 80, 5, 7, 9999))
        assert result.n_static_rows == 1
        assert result.n_static_reused == 0

    def test_cache_miss_with_rec_inserts(self):
        pipeline = _pipeline()
        cur = MagicMock()
        cur.fetchone.return_value = (5,)
        result = self._result()
        rec = {"dim_bow": 10}
        sid = pipeline._ensure_static(cur, "omrat", 2021, 1, rec, result)
        assert sid == 5
        assert 1 in pipeline._statics_cache
        assert result.n_static_rows == 1

    def test_none_rec_with_cache_reuses_existing(self):
        """When no new info is provided, reuse the cached row instead of inserting NULLs."""
        pipeline = _pipeline()
        pipeline._statics_cache[1] = (42, (10, 80, 5, 7, 9999))
        cur = MagicMock()
        result = self._result()
        sid = pipeline._ensure_static(cur, "omrat", 2021, 1, None, result)
        assert sid == 42
        cur.execute.assert_not_called()
        assert result.n_static_reused == 1

    def test_none_rec_without_cache_inserts_placeholder(self):
        pipeline = _pipeline()
        cur = MagicMock()
        cur.fetchone.return_value = (1,)
        result = self._result()
        sid = pipeline._ensure_static(cur, "omrat", 2021, 999, None, result)
        assert sid == 1
        # Placeholder cached.
        assert pipeline._statics_cache[999] == (1, (None, None, None, None, None))
        assert result.n_static_rows == 1


class TestEnsureState:
    def test_cache_hit_same_voyage_reuses(self):
        pipeline = _pipeline()
        pipeline._states_cache[1] = (7, (4.5, 70, "ROTTERDAM", None, 100))
        cur = MagicMock()
        result = IngestionResult()
        rec = {"draught": 4.5, "ship_type": 70, "destination": "ROTTERDAM"}
        sid = pipeline._ensure_state(
            cur, "omrat", 2021, 1, static_id=100, rec=rec,
            fallback_t=0.0, result=result,
        )
        assert sid == 7
        cur.execute.assert_not_called()
        assert result.n_state_reused == 1

    def test_cache_hit_different_voyage_inserts(self):
        pipeline = _pipeline()
        pipeline._states_cache[1] = (7, (4.5, 70, "ROTTERDAM", None, 100))
        cur = MagicMock()
        cur.fetchone.return_value = (8,)
        result = IngestionResult()
        rec = {"draught": 8.0, "ship_type": 70, "destination": "ROTTERDAM"}  # draught changed
        sid = pipeline._ensure_state(
            cur, "omrat", 2021, 1, static_id=100, rec=rec,
            fallback_t=0.0, result=result,
        )
        assert sid == 8
        assert pipeline._states_cache[1][0] == 8
        assert result.n_state_rows == 1

    def test_static_id_change_invalidates_cache(self):
        """A new static_id (vessel re-registered) means a new state row."""
        pipeline = _pipeline()
        pipeline._states_cache[1] = (7, (4.5, 70, "ROTTERDAM", None, 100))
        cur = MagicMock()
        cur.fetchone.return_value = (8,)
        result = IngestionResult()
        rec = {"draught": 4.5, "ship_type": 70, "destination": "ROTTERDAM"}
        sid = pipeline._ensure_state(
            cur, "omrat", 2021, 1, static_id=200, rec=rec,  # static_id changed
            fallback_t=0.0, result=result,
        )
        assert sid == 8                  # new row
        assert result.n_state_rows == 1

    def test_none_rec_with_cache_reuses(self):
        pipeline = _pipeline()
        pipeline._states_cache[1] = (7, (4.5, 70, "ROTTERDAM", None, 100))
        cur = MagicMock()
        result = IngestionResult()
        sid = pipeline._ensure_state(
            cur, "omrat", 2021, 1, static_id=100, rec=None,
            fallback_t=0.0, result=result,
        )
        assert sid == 7
        cur.execute.assert_not_called()
        assert result.n_state_reused == 1

    def test_none_rec_without_cache_inserts_placeholder(self):
        pipeline = _pipeline()
        cur = MagicMock()
        cur.fetchone.return_value = (3,)
        result = IngestionResult()
        sid = pipeline._ensure_state(
            cur, "omrat", 2021, 1, static_id=100, rec=None,
            fallback_t=1625097600.0, result=result,
        )
        assert sid == 3
        assert pipeline._states_cache[1][0] == 3
        assert pipeline._states_cache[1][1] == (None, None, None, None, 100)
        assert result.n_state_rows == 1


class TestBatchLoadCaches:
    def test_load_latest_statics_populates_cache(self):
        pipeline = _pipeline()
        cur = MagicMock()
        cur.fetchall.return_value = [
            (123, 1, 10, 80, 5, 7, 9999),
            (456, 2, 20, 100, 8, 10, None),
        ]
        pipeline._load_latest_statics(cur, "omrat", 2021)
        sql = _render(cur.execute.call_args[0][0])
        assert "DISTINCT ON (mmsi)" in sql
        assert "omrat.statics_2021" in sql
        assert pipeline._statics_cache[123] == (1, (10, 80, 5, 7, 9999))
        assert pipeline._statics_cache[456] == (2, (20, 100, 8, 10, None))

    def test_load_latest_statics_handles_db_error(self):
        pipeline = _pipeline()
        cur = MagicMock()
        cur.execute.side_effect = Exception("table not found")
        pipeline._load_latest_statics(cur, "omrat", 2021)
        assert pipeline._statics_cache == {}

    def test_load_latest_states_populates_cache(self):
        pipeline = _pipeline()
        cur = MagicMock()
        cur.fetchall.return_value = [
            (123, 7, 4.5, 70, "ROTTERDAM", None, 100),
        ]
        pipeline._load_latest_states(cur, "omrat", 2021)
        sql = _render(cur.execute.call_args[0][0])
        assert "DISTINCT ON (mmsi)" in sql
        assert pipeline._states_cache[123] == (7, (4.5, 70, "ROTTERDAM", None, 100))

    def test_load_latest_states_truncates_destination(self):
        pipeline = _pipeline()
        cur = MagicMock()
        cur.fetchall.return_value = [
            (123, 7, 4.5, 70, "A" * 50, None, 100),
        ]
        pipeline._load_latest_states(cur, "omrat", 2021)
        # Stored destination should match the 20-char trim used by ingestion.
        assert pipeline._states_cache[123][1][2] == "A" * 20


class TestWatermark:
    def test_filter_passes_track_through_when_no_watermark(self):
        from aissegments import Track
        import numpy as np
        track = Track.from_arrays(
            mmsi=1, t=[0, 60, 120], lon=[0, 0.001, 0.002],
            lat=[55, 55, 55], sog=[10, 10, 10], cog=[90, 90, 90],
        )
        assert IngestionPipeline._filter_after_watermark(track, None) is track

    def test_filter_returns_none_when_track_before_watermark(self):
        from aissegments import Track
        track = Track.from_arrays(
            mmsi=1, t=[0, 60, 120], lon=[0, 0.001, 0.002],
            lat=[55, 55, 55], sog=[10, 10, 10], cog=[90, 90, 90],
        )
        # Watermark in the future of all pings.
        assert IngestionPipeline._filter_after_watermark(track, 1000) is None

    def test_filter_slices_partially_overlapping_track(self):
        from aissegments import Track
        track = Track.from_arrays(
            mmsi=1, t=[0, 60, 120, 180, 240], lon=[0, 0.001, 0.002, 0.003, 0.004],
            lat=[55] * 5, sog=[10] * 5, cog=[90] * 5,
        )
        out = IngestionPipeline._filter_after_watermark(track, 100)
        assert out is not None
        assert len(out) == 3                    # t=120, 180, 240
        assert float(out.t[0]) == 120.0

    def test_filter_passes_track_unchanged_when_all_after(self):
        from aissegments import Track
        track = Track.from_arrays(
            mmsi=1, t=[100, 200, 300], lon=[0, 0.001, 0.002],
            lat=[55, 55, 55], sog=[10, 10, 10], cog=[90, 90, 90],
        )
        out = IngestionPipeline._filter_after_watermark(track, 50)
        assert out is track  # no slicing

    def test_load_watermarks_parses_results(self):
        cur = MagicMock()
        from datetime import datetime, timezone
        cur.fetchall.return_value = [
            (123, datetime(2021, 7, 1, tzinfo=timezone.utc)),
            (456, datetime(2021, 8, 15, tzinfo=timezone.utc)),
        ]
        wm = IngestionPipeline._load_watermarks(cur)
        sql = cur.execute.call_args[0][0]
        assert "omrat_meta.segment_watermark" in sql
        assert wm[123] == datetime(2021, 7, 1, tzinfo=timezone.utc).timestamp()
        assert wm[456] == datetime(2021, 8, 15, tzinfo=timezone.utc).timestamp()

    def test_load_watermarks_empty_on_db_error(self):
        cur = MagicMock()
        cur.execute.side_effect = Exception("table not found")
        assert IngestionPipeline._load_watermarks(cur) == {}

    def test_load_watermarks_drops_null_rows(self):
        cur = MagicMock()
        cur.fetchall.return_value = [(None, None), (1, None), (2, None)]
        # No valid (mmsi, last_t) pairs → empty.
        assert IngestionPipeline._load_watermarks(cur) == {}


class TestIngestionResult2:
    def test_summary_mentions_watermark_skips(self):
        r = IngestionResult(
            n_files=1, n_tracks=10, n_segments=20,
            n_static_rows=5, n_state_rows=10,
            n_tracks_skipped_watermark=42,
            elapsed_seconds=1.0,
        )
        s = r.summary()
        assert "42 skipped (watermark)" in s


class TestRunDefensive:
    def test_no_files_yields_error(self):
        result = _pipeline().run([], year=2021)
        assert result.n_files == 0
        assert result.n_tracks == 0
        assert "No input files" in result.errors[0]

    def test_run_dispatches_simple_csv_to_correct_path(self, tmp_path):
        """A Marine-Cadastre-style CSV must go through the simple-CSV path,
        not aisdb's decoder.  We patch both paths and assert which one fires."""
        from unittest.mock import patch

        f = tmp_path / "mc.csv"
        f.write_text(
            "MMSI,BaseDateTime,LAT,LON,SOG,COG\n"
            "123,2021-06-01T12:00:00,55.0,12.0,10.0,90.0\n"
            "123,2021-06-01T12:01:00,55.0,12.001,10.0,90.0\n",
            encoding="utf-8",
        )
        pipeline = _pipeline()
        with patch.object(pipeline, "_ingest_simple_csv_files") as mock_simple, \
             patch.object(pipeline, "_ingest_aisdb_files") as mock_aisdb, \
             patch("omrat_utils.handle_ais_ingest.Migrator") as mock_migrator:
            pipeline.run([f], year=2021, create_indexes_after=False)
        mock_simple.assert_called_once()
        mock_aisdb.assert_not_called()

    def test_run_dispatches_aisdb_csv_to_correct_path(self, tmp_path):
        from unittest.mock import patch

        f = tmp_path / "ais.csv"
        f.write_text(
            "MMSI,Message_ID,Repeat_indicator,Time,Millisecond\n"
            "123,1,0,20210701_000000,0\n",
            encoding="utf-8",
        )
        pipeline = _pipeline()
        with patch.object(pipeline, "_ingest_simple_csv_files") as mock_simple, \
             patch.object(pipeline, "_ingest_aisdb_files") as mock_aisdb, \
             patch("omrat_utils.handle_ais_ingest.Migrator") as mock_migrator:
            pipeline.run([f], year=2021, create_indexes_after=False)
        mock_aisdb.assert_called_once()
        mock_simple.assert_not_called()

    def test_simple_csv_passes_extracted_static_to_ingest_one_track(self, tmp_path):
        """Marine Cadastre static fields (Length/Width/Draft/IMO/...) must
        flow through to ``_ingest_one_track`` instead of ``rec=None``.
        """
        from unittest.mock import patch, MagicMock

        f = tmp_path / "mc.csv"
        # One MMSI with full static info, three position rows so TDKC has
        # something meaningful to compress.
        f.write_text(
            "MMSI,BaseDateTime,LAT,LON,SOG,COG,VesselName,IMO,Length,Width,Draft\n"
            "100,2021-06-01T12:00:00,55.0,12.0,10.0,90.0,TESTVESSEL,9999,30,8,4.5\n"
            "100,2021-06-01T12:01:00,55.0,12.001,10.0,90.0,,,,,\n"
            "100,2021-06-01T12:02:00,55.0,12.002,10.0,90.0,,,,,\n",
            encoding="utf-8",
        )

        captured: list[dict | None] = []

        def capture(cur, track, rec, year, result):
            captured.append(rec)

        pipeline = _pipeline()
        # Mock the connection so we don't need a real DB.
        mock_cur = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        with patch("omrat_utils.handle_ais_ingest.psycopg2.connect") as mock_connect, \
             patch.object(pipeline, "_ingest_one_track", side_effect=capture):
            mock_connect.return_value.__enter__.return_value = mock_conn
            result = IngestionResult()
            pipeline._ingest_simple_csv_files(
                [f], year=2021, result=result, incremental=True,
            )

        assert len(captured) == 1
        rec = captured[0]
        assert rec is not None, "Static record was not propagated through"
        assert rec["mmsi"] == 100
        assert rec["vessel_name"] == "TESTVESSEL"
        assert rec["imo"] == 9999
        assert rec["draught"] == 4.5
        assert rec["dim_bow"] == 15.0  # 30/2
        assert rec["dim_port"] == 4.0  # 8/2

    def test_simple_csv_passes_none_when_no_static_columns(self, tmp_path):
        """Bare kinematic CSVs (no static columns) should pass rec=None."""
        from unittest.mock import patch, MagicMock

        f = tmp_path / "minimal.csv"
        f.write_text(
            "mmsi,time,lon,lat,sog,cog\n"
            "100,2021-06-01T12:00:00,12.0,55.0,10.0,90.0\n"
            "100,2021-06-01T12:01:00,12.001,55.0,10.0,90.0\n",
            encoding="utf-8",
        )

        captured: list[dict | None] = []

        def capture(cur, track, rec, year, result):
            captured.append(rec)

        pipeline = _pipeline()
        mock_cur = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        with patch("omrat_utils.handle_ais_ingest.psycopg2.connect") as mock_connect, \
             patch.object(pipeline, "_ingest_one_track", side_effect=capture):
            mock_connect.return_value.__enter__.return_value = mock_conn
            result = IngestionResult()
            pipeline._ingest_simple_csv_files(
                [f], year=2021, result=result, incremental=True,
            )

        assert len(captured) == 1
        assert captured[0] is None  # _ensure_static handles None correctly


# ---------------------------------------------------------------------------
# Track splitter — gap + implausible-speed
# ---------------------------------------------------------------------------


def _make_track(timestamps, *, lon=None, lat=None, sog=None, cog=None, mmsi=1):
    """Build an aissegments.Track with sane defaults for splitter tests."""
    from aissegments import Track
    import numpy as np
    n = len(timestamps)
    return Track.from_arrays(
        mmsi=mmsi,
        t=np.asarray(timestamps, dtype=np.float64),
        lon=np.asarray(lon if lon is not None else np.linspace(0.0, 0.001, n),
                       dtype=np.float64),
        lat=np.asarray(lat if lat is not None else np.full(n, 55.0),
                       dtype=np.float64),
        sog=np.asarray(sog if sog is not None else np.full(n, 10.0),
                       dtype=np.float64),
        cog=np.asarray(cog if cog is not None else np.full(n, 90.0),
                       dtype=np.float64),
    )


class TestSplitTrackAtGaps:
    """Pre-TDKC plausibility filter — see ``split_track_at_gaps``."""

    def test_short_track_passed_through(self):
        track = _make_track([0.0])
        assert split_track_at_gaps(
            track, max_gap_s=3600.0, speed_tolerance=0.3, speed_floor_kn=1.0,
        ) == [track]

    def test_clean_track_returns_single_subtrack(self):
        # Five pings 10 s apart, ~10 m apart each → ~2 kn.  Reported 10 kn,
        # implied speed below the tolerance limit.  Must NOT split.
        track = _make_track(
            timestamps=[0, 10, 20, 30, 40],
            lon=[0.0, 0.00005, 0.00010, 0.00015, 0.00020],
            lat=[55.0] * 5,
            sog=[10.0] * 5,
            cog=[90.0] * 5,
        )
        out = split_track_at_gaps(
            track, max_gap_s=3600.0, speed_tolerance=0.3, speed_floor_kn=1.0,
        )
        assert len(out) == 1
        assert len(out[0]) == 5

    def test_time_gap_triggers_split(self):
        # 4 pings at 0, 60, 7200 (2 h gap), 7260.  max_gap_s=3600 → split
        # between idx 1 and 2.
        track = _make_track(timestamps=[0, 60, 7200, 7260])
        out = split_track_at_gaps(
            track, max_gap_s=3600.0, speed_tolerance=10.0, speed_floor_kn=1000.0,
        )
        assert len(out) == 2
        assert len(out[0]) == 2 and len(out[1]) == 2
        assert sum(len(s) for s in out) == 4

    def test_speed_jump_triggers_split_for_zero_sog_teleport(self):
        # The user's reported red-segment case: vessel at 0 kn appears 1100 km
        # away 69 s later.  Implied ~30,000 kn — must split, even though
        # avg_sog=0 and the percentage check would otherwise be undefined.
        track = _make_track(
            timestamps=[0, 60, 129, 189],
            lon=[-90.0, -90.0001, -80.0, -80.0001],   # ~1100 km lon jump
            lat=[35.0, 35.0, 35.0, 35.0],
            sog=[0.0] * 4,
            cog=[0.0] * 4,
        )
        out = split_track_at_gaps(
            track, max_gap_s=3600.0, speed_tolerance=0.3, speed_floor_kn=1.0,
        )
        assert len(out) >= 2
        # First sub-track holds the pre-jump cluster, last holds the post-jump.
        assert float(out[0].lon[0]) == -90.0
        assert float(out[-1].lon[-1]) == -80.0001

    def test_speed_jump_with_running_vessel(self):
        # 12 kn vessel (~6 m/s).  Pings 60 s apart → ~360 m / step.  Inject
        # a step of ~3.6 km at the third interval — implied ~120 kn vs
        # avg_sog ~12 kn (limit 12*1.3+1=16.6 kn) → split.
        track = _make_track(
            timestamps=[0, 60, 120, 180, 240],
            lon=[0.0, 0.0054, 0.0108, 0.0700, 0.0754],
            lat=[55.0] * 5,
            sog=[12.0] * 5,
            cog=[90.0] * 5,
        )
        out = split_track_at_gaps(
            track, max_gap_s=3600.0, speed_tolerance=0.3, speed_floor_kn=1.0,
        )
        assert len(out) == 2
        # Break is between idx 2 and 3 (the 3.6 km step).
        assert len(out[0]) == 3 and len(out[1]) == 2

    def test_decelerating_vessel_does_not_split(self):
        # Implied speed BELOW reported SOG is plausible (slowing for port).
        # 8 kn for 60 s → ~250 m, but the vessel was reported doing 12 kn.
        track = _make_track(
            timestamps=[0, 60, 120],
            lon=[0.0, 0.0039, 0.0078],   # ~250 m / 60 s = ~8 kn implied
            lat=[55.0] * 3,
            sog=[12.0] * 3,
            cog=[90.0] * 3,
        )
        out = split_track_at_gaps(
            track, max_gap_s=3600.0, speed_tolerance=0.3, speed_floor_kn=1.0,
        )
        assert len(out) == 1

    def test_floor_absorbs_gps_jitter_at_zero_sog(self):
        # Moored vessel reporting 0 kn, GPS jitter ~0.3 kn.  speed_floor_kn=1
        # must absorb this so the splitter doesn't fragment the track.
        # 30 m every 60 s → ~1.0 kn implied.
        track = _make_track(
            timestamps=[0, 60, 120, 180],
            lon=[0.0, 0.000470, 0.000940, 0.001410],
            lat=[55.0] * 4,
            sog=[0.0] * 4,
            cog=[0.0] * 4,
        )
        out = split_track_at_gaps(
            track, max_gap_s=3600.0, speed_tolerance=0.3, speed_floor_kn=1.0,
        )
        assert len(out) == 1

    def test_multiple_breaks_partition_track_contiguously(self):
        # Two independent gaps: one time-based, one speed-based.
        track = _make_track(
            timestamps=[0, 60, 7200, 7260, 7320],   # gap between 1 and 2
            lon=[0.0, 0.001, 0.002, 0.5, 0.501],    # speed jump between 2 and 3
            lat=[55.0] * 5,
            sog=[10.0] * 5,
            cog=[90.0] * 5,
        )
        out = split_track_at_gaps(
            track, max_gap_s=3600.0, speed_tolerance=0.3, speed_floor_kn=1.0,
        )
        assert len(out) >= 2
        assert sum(len(s) for s in out) == 5    # no points dropped


class TestIngestOneTrackSplits:
    """Verify that ``_ingest_one_track`` honours the splitter."""

    def test_split_increments_n_track_splits(self):
        from unittest.mock import patch

        track = _make_track(timestamps=[0, 60, 7200, 7260])  # one time gap
        pipeline = _pipeline()
        cur = MagicMock()
        result = IngestionResult()
        with patch.object(pipeline, "_ensure_static", return_value=1), \
             patch.object(pipeline, "_ensure_state", return_value=2), \
             patch.object(pipeline, "_bulk_insert_segments"), \
             patch.object(pipeline, "_update_watermark"):
            pipeline._ingest_one_track(cur, track, None, 2021, result)
        assert result.n_tracks == 1     # one MMSI track…
        assert result.n_track_splits == 1   # …split into two sub-tracks

    def test_clean_track_has_no_splits(self):
        from unittest.mock import patch

        track = _make_track(timestamps=[0, 60, 120, 180])
        pipeline = _pipeline()
        cur = MagicMock()
        result = IngestionResult()
        with patch.object(pipeline, "_ensure_static", return_value=1), \
             patch.object(pipeline, "_ensure_state", return_value=2), \
             patch.object(pipeline, "_bulk_insert_segments"), \
             patch.object(pipeline, "_update_watermark"):
            pipeline._ingest_one_track(cur, track, None, 2021, result)
        assert result.n_tracks == 1
        assert result.n_track_splits == 0


class TestMergeSimpleCsvTracks:
    """Cross-file MMSI merging — see ``_merge_simple_csv_tracks``."""

    def test_single_track_passes_through(self):
        track = _make_track(timestamps=[0, 60, 120])
        merged = list(IngestionPipeline._merge_simple_csv_tracks({1: [track]}))
        assert len(merged) == 1
        # Single-track shortcut keeps the same instance.
        assert merged[0] is track

    def test_two_files_merge_into_time_sorted_track(self):
        # File A has the later half, file B has the earlier half — i.e.
        # file order != time order, the failure mode that the watermark
        # used to silently drop.
        a = _make_track(
            timestamps=[200, 260, 320],
            lon=[0.005, 0.006, 0.007],
        )
        b = _make_track(
            timestamps=[0, 60, 120],
            lon=[0.000, 0.001, 0.002],
        )
        merged = list(IngestionPipeline._merge_simple_csv_tracks({1: [a, b]}))
        assert len(merged) == 1
        out = merged[0]
        assert len(out) == 6
        # Time-sorted ascending.
        assert list(out.t) == [0.0, 60.0, 120.0, 200.0, 260.0, 320.0]
        # Longitudes follow the time order (proves we sorted, not concatenated).
        assert list(out.lon) == [0.0, 0.001, 0.002, 0.005, 0.006, 0.007]

    def test_duplicate_timestamps_deduped(self):
        # Same MMSI, same ping in both files — the merge should keep one.
        a = _make_track(timestamps=[0, 60, 120], lon=[0.0, 0.001, 0.002])
        b = _make_track(timestamps=[60, 180], lon=[0.001, 0.003])
        merged = list(IngestionPipeline._merge_simple_csv_tracks({1: [a, b]}))
        out = merged[0]
        assert len(out) == 4
        assert list(out.t) == [0.0, 60.0, 120.0, 180.0]


class TestIngestionResult3:
    def test_summary_mentions_track_splits(self):
        r = IngestionResult(
            n_files=1, n_tracks=10, n_segments=20,
            n_static_rows=2, n_state_rows=5,
            n_track_splits=3, elapsed_seconds=1.0,
        )
        assert "3 gap-splits" in r.summary()

    def test_summary_mentions_year_skipped(self):
        r = IngestionResult(
            n_files=1, n_tracks=0, n_segments=0,
            n_tracks_skipped_year=42, elapsed_seconds=1.0,
        )
        assert "42 skipped (outside target year)" in r.summary()

    def test_summary_flags_cancelled_run(self):
        r = IngestionResult(
            n_files=1, n_tracks=5, n_segments=20,
            cancelled=True, elapsed_seconds=10.0,
        )
        s = r.summary()
        assert s.startswith("Cancelled after ingesting")


class TestProgressFormatting:
    """``_format_progress`` / ``_fmt_duration`` — string-only, no DB needed."""

    def test_fmt_duration_handles_zero_and_negative(self):
        assert IngestionPipeline._fmt_duration(0.0) == "0s"
        assert IngestionPipeline._fmt_duration(-5.0) == "0s"

    def test_fmt_duration_seconds_only(self):
        assert IngestionPipeline._fmt_duration(42.0) == "42s"

    def test_fmt_duration_minutes(self):
        assert IngestionPipeline._fmt_duration(125.0) == "2m 5s"

    def test_fmt_duration_hours(self):
        assert IngestionPipeline._fmt_duration(3 * 3600 + 7 * 60 + 12) == "3h 7m"

    def test_format_progress_unknown_total_omits_percent(self):
        out = IngestionPipeline._format_progress(processed=100, total=0, t_loop_start=0.0)
        assert out == "100 tracks"

    def test_format_progress_warming_up_when_no_processed(self):
        # We deliberately keep elapsed/processed=0 ambiguous so the user
        # gets something sensible during the cold-start window.
        import time as _time
        out = IngestionPipeline._format_progress(
            processed=0, total=10_000, t_loop_start=_time.monotonic(),
        )
        assert "warming up" in out
        assert "10000" in out

    def test_format_progress_emits_pct_rate_eta(self):
        # Pin a synthetic clock so the rate/ETA are deterministic.
        from unittest.mock import patch
        # Pretend the loop started 10 seconds ago.
        loop_start = 1000.0
        now = 1010.0
        with patch("omrat_utils.handle_ais_ingest.time_mod.monotonic", return_value=now):
            out = IngestionPipeline._format_progress(
                processed=2000, total=10_000, t_loop_start=loop_start,
            )
        # 2000 / 10s = 200 tr/s, 8000 remaining → 40 s ETA.
        assert "2000/10000" in out
        assert "20.0%" in out
        assert "200 tr/s" in out
        assert "ETA 40s" in out


class TestCountAisdbDistinctMmsis:
    def test_returns_zero_when_no_dynamic_tables(self, tmp_path):
        db = tmp_path / "empty.db"
        with sqlite3.connect(str(db)) as con:
            con.execute("CREATE TABLE unrelated (x INTEGER)")
            con.commit()
        assert _pipeline()._count_aisdb_distinct_mmsis(db, year=2021) == 0

    def test_counts_distinct_mmsis_across_months(self, tmp_path):
        db = tmp_path / "test.db"
        with sqlite3.connect(str(db)) as con:
            # Two monthly dynamic tables, overlapping MMSIs — the count is
            # the size of the union, NOT the sum, since one MMSI yielded
            # in TrackGen across multiple months still produces one track.
            con.execute(
                "CREATE TABLE ais_202106_dynamic ("
                "mmsi INTEGER, time INTEGER, longitude REAL, latitude REAL)"
            )
            con.execute(
                "CREATE TABLE ais_202107_dynamic ("
                "mmsi INTEGER, time INTEGER, longitude REAL, latitude REAL)"
            )
            for mmsi in (100, 101, 102, 102):
                con.execute(
                    "INSERT INTO ais_202106_dynamic VALUES (?, ?, ?, ?)",
                    (mmsi, 0, 0.0, 0.0),
                )
            for mmsi in (102, 103):  # 102 also in June, 103 only in July
                con.execute(
                    "INSERT INTO ais_202107_dynamic VALUES (?, ?, ?, ?)",
                    (mmsi, 0, 0.0, 0.0),
                )
            con.commit()
        assert _pipeline()._count_aisdb_distinct_mmsis(db, year=2021) == 4

    def test_skips_null_mmsis(self, tmp_path):
        db = tmp_path / "null.db"
        with sqlite3.connect(str(db)) as con:
            con.execute(
                "CREATE TABLE ais_202101_dynamic "
                "(mmsi INTEGER, time INTEGER, longitude REAL, latitude REAL)"
            )
            con.execute("INSERT INTO ais_202101_dynamic VALUES (NULL, 0, 0, 0)")
            con.execute("INSERT INTO ais_202101_dynamic VALUES (100, 0, 0, 0)")
            con.commit()
        assert _pipeline()._count_aisdb_distinct_mmsis(db, year=2021) == 1


class TestPipelineCancellation:
    """Cooperative cancellation — see ``IngestionPipeline.cancel``."""

    def test_cancel_flag_starts_clear(self):
        assert _pipeline()._is_cancelled() is False

    def test_cancel_sets_flag(self):
        p = _pipeline()
        p.cancel()
        assert p._is_cancelled() is True

    def test_run_resets_cancel_flag(self):
        """A second .run() on the same pipeline must clear a stale flag."""
        from unittest.mock import patch
        p = _pipeline()
        p.cancel()
        # Drive run() with no files — early-exit path, but it MUST clear
        # the flag before doing so.  We don't actually want it to hit the
        # DB, so the empty-files short-circuit suffices.
        with patch("omrat_utils.handle_ais_ingest.Migrator"):
            p.run([], year=2021)
        assert p._is_cancelled() is False

    def test_cancel_during_simple_csv_loop_short_circuits(self, tmp_path):
        """A cancel between merged tracks must abort the per-track loop."""
        from unittest.mock import patch, MagicMock

        f = tmp_path / "many.csv"
        # 3 vessels, 3 pings each — enough that the cancel-after-first
        # track leaves the other two unprocessed.
        rows = ["mmsi,time,lon,lat,sog,cog"]
        for mmsi in (100, 101, 102):
            for k in range(3):
                rows.append(f"{mmsi},2021-06-01T12:0{k}:00,12.0,55.0,10.0,90.0")
        f.write_text("\n".join(rows) + "\n", encoding="utf-8")

        pipeline = _pipeline()
        n_called = {"count": 0}

        def cancel_after_first(*_args, **_kwargs):
            n_called["count"] += 1
            if n_called["count"] == 1:
                pipeline.cancel()

        mock_cur = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        with patch("omrat_utils.handle_ais_ingest.psycopg2.connect") as mock_connect, \
             patch.object(pipeline, "_ingest_one_track", side_effect=cancel_after_first):
            mock_connect.return_value.__enter__.return_value = mock_conn
            result = IngestionResult()
            pipeline._ingest_simple_csv_files(
                [f], year=2021, result=result, incremental=True,
            )

        # Exactly one track ingested before the cancel took effect.
        assert n_called["count"] == 1


class TestTruncateYear:
    """``Migrator.truncate_year`` SQL-level checks (no live DB needed)."""

    def _migrator(self):
        from omrat_utils.db_setup import Migrator
        return Migrator(ConnectionProfile(
            host="x", database="y", user="u", schema="omrat"
        ))

    @staticmethod
    def _ctx_conn(cur):
        """Build a MagicMock connection whose ``with conn:`` yields itself,
        and whose ``conn.cursor()`` context yields ``cur``.

        Mirrors how :meth:`Migrator._connect` is consumed::

            with self._connect() as conn, conn.cursor() as cur:
                ...
        """
        conn = MagicMock()
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = False
        conn.cursor.return_value.__enter__.return_value = cur
        conn.cursor.return_value.__exit__.return_value = False
        return conn

    def test_count_year_data_handles_missing_tables(self):
        from unittest.mock import patch

        m = self._migrator()
        cur = MagicMock()
        # Every SELECT raises (table missing) — count returns an empty dict.
        cur.execute.side_effect = Exception("does not exist")
        conn = self._ctx_conn(cur)
        with patch.object(m, "_connect", return_value=conn):
            counts = m.count_year_data(2021)
        assert counts == {}

    def test_count_year_data_returns_per_table_counts(self):
        from unittest.mock import patch

        m = self._migrator()
        cur = MagicMock()
        # Three SELECT COUNTs for the year tables, then one for watermarks.
        cur.fetchone.side_effect = [(123,), (5,), (2,), (10,)]
        conn = self._ctx_conn(cur)
        with patch.object(m, "_connect", return_value=conn):
            counts = m.count_year_data(2021)
        assert counts == {
            "omrat.segments_2021": 123,
            "omrat.states_2021": 5,
            "omrat.statics_2021": 2,
            "omrat_meta.segment_watermark": 10,
        }

    def test_count_year_data_rejects_invalid_year(self):
        from omrat_utils.db_setup import MigrationError
        with pytest.raises(MigrationError):
            self._migrator().count_year_data(1850)

    def test_truncate_year_emits_expected_statements(self):
        from unittest.mock import patch

        m = self._migrator()
        cur = MagicMock()
        conn = self._ctx_conn(cur)
        with patch.object(m, "_connect", return_value=conn):
            m.truncate_year(2021)

        # Collect every SQL statement actually executed.
        sqls = [c.args[0] for c in cur.execute.call_args_list if c.args]
        joined = " ".join(sqls)
        assert "TRUNCATE omrat.segments_2021" in joined
        assert (
            "TRUNCATE omrat.states_2021, omrat.statics_2021 CASCADE" in joined
        )
        # Watermark cleanup.
        assert any(
            "DELETE FROM omrat_meta.segment_watermark" in s for s in sqls
        )
        conn.commit.assert_called_once()

    def test_truncate_year_skips_missing_year_tables(self):
        """Calling truncate on a never-provisioned year is a no-op (no raise)."""
        from unittest.mock import patch
        import psycopg2

        m = self._migrator()
        cur = MagicMock()
        # Every TRUNCATE raises UndefinedTable; SAVEPOINT/RELEASE/ROLLBACK
        # and the final watermark DELETE all succeed (return None).
        def execute(sql, *args, **kwargs):
            if "TRUNCATE" in sql:
                raise psycopg2.errors.UndefinedTable("nope")
        cur.execute.side_effect = execute
        conn = self._ctx_conn(cur)
        with patch.object(m, "_connect", return_value=conn):
            m.truncate_year(2021)
        # Never raised, transaction committed.
        conn.commit.assert_called_once()

    def test_truncate_year_rejects_invalid_year(self):
        from omrat_utils.db_setup import MigrationError
        with pytest.raises(MigrationError):
            self._migrator().truncate_year(3000)


class TestTracksSkippedYear:
    def test_simple_csv_increments_year_skip_counter(self, tmp_path):
        """Every ping outside the target year is silently dropped, but
        the result counter records it so the user can see why ingestion
        produced 0 segments."""
        from unittest.mock import patch, MagicMock

        # File contains 2019 data; we'll target year 2026 (the user's
        # actual reported failure mode).
        f = tmp_path / "wrong_year.csv"
        f.write_text(
            "mmsi,time,lon,lat,sog,cog\n"
            "100,2019-06-01T12:00:00,12.0,55.0,10.0,90.0\n"
            "100,2019-06-01T12:01:00,12.001,55.0,10.0,90.0\n"
            "200,2019-07-15T08:00:00,13.0,56.0,8.0,180.0\n"
            "200,2019-07-15T08:01:00,13.001,56.0,8.0,180.0\n",
            encoding="utf-8",
        )

        pipeline = _pipeline()
        mock_cur = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        with patch("omrat_utils.handle_ais_ingest.psycopg2.connect") as mock_connect, \
             patch.object(pipeline, "_ingest_one_track") as mock_ingest:
            mock_connect.return_value.__enter__.return_value = mock_conn
            result = IngestionResult()
            pipeline._ingest_simple_csv_files(
                [f], year=2026, result=result, incremental=True,
            )

        # No tracks ingested (all 2019 data, target year 2026).
        mock_ingest.assert_not_called()
        # Both MMSIs counted as year-skipped.
        assert result.n_tracks_skipped_year == 2
