"""Unit tests for the pure-Python helpers in omrat_utils/handle_ais.py.

Covers: get_pl, get_type, close_to_line.
``update_ais_settings_file`` is I/O + QSettings -- tested separately.
The ``AIS`` class itself is QGIS-dependent and is covered by
test_ais_data_retrival.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.handle_ais import (
    get_pl, get_type, close_to_line, update_ais_settings_file,
)


def _render(sql) -> str:
    """Render a psycopg2.sql.Composable (or plain string) for substring tests.

    The AIS query path now composes SQL with ``psycopg2.sql.Identifier`` for
    safer identifier handling.  Tests assert against the raw SQL shape, so
    this walks the Composable and emits identifier names without surrounding
    double-quotes — the substring assertions stay readable.
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
# get_type (AIS type_and_cargo -> OMRAT ship type index)
# ---------------------------------------------------------------------------

class TestGetType:
    @pytest.mark.parametrize("toc, expected", [
        (30, 0),                 # Fishing
        (31, 1), (32, 1),        # Towing
        (33, 2),                 # Dredging
        (34, 3),                 # Diving
        (35, 4), (36, 5), (37, 6),
        (40, 7), (45, 7), (49, 7),  # High-speed range
        (50, 8), (51, 9), (52, 10), (53, 11),
        (54, 12), (55, 13),
        (56, 14), (57, 14),
        (58, 15), (59, 16),
        (60, 17), (65, 17), (69, 17),  # Passenger range
        (70, 18), (75, 18), (79, 18),  # Cargo range
        (80, 19), (85, 19), (89, 19),  # Tanker range
        (90, 20), (99, 20),            # Other
    ])
    def test_toc_to_type_index(self, toc, expected):
        assert get_type(toc) == expected

    def test_float_toc_accepted(self):
        assert get_type(79.0) == 18

    def test_none_toc_returns_other(self):
        """NULL ``type_and_cargo`` is common for vessels that never
        broadcast Type-5 statics — bucket into 'Other Type' (20) instead
        of crashing the whole traffic build."""
        assert get_type(None) == 20

    def test_unparseable_toc_returns_other(self):
        """Defensive: a junk string from a hand-edited registry row
        shouldn't bring the AIS pipeline down either."""
        assert get_type("not-a-number") == 20


# ---------------------------------------------------------------------------
# close_to_line
# ---------------------------------------------------------------------------

class TestCloseToLine:
    def test_bearing_cog_exact_match(self):
        assert close_to_line(bearing=90.0, cog=90.0, max_angle=10.0)

    def test_cog_within_window_returns_true(self):
        assert close_to_line(bearing=90.0, cog=85.0, max_angle=10.0)

    def test_cog_outside_window_returns_false(self):
        assert not close_to_line(bearing=90.0, cog=120.0, max_angle=10.0)

    def test_wraparound_bearing_near_north(self):
        """When bearing=5° and max_angle=10°, accepting cogs in 355°-15°."""
        assert close_to_line(5.0, 358.0, max_angle=10.0)
        assert close_to_line(5.0, 12.0, max_angle=10.0)
        assert not close_to_line(5.0, 200.0, max_angle=10.0)

    def test_max_angle_above_180_raises(self):
        with pytest.raises(ValueError):
            close_to_line(90.0, 90.0, max_angle=200.0)

    def test_bearing_over_360_modded(self):
        # bearing = 450 -> 90; cog = 85; within 10 deg -> True.
        assert close_to_line(450.0, 85.0, max_angle=10.0)

    def test_cog_over_360_modded(self):
        assert close_to_line(90.0, 445.0, max_angle=10.0)  # 445 mod 360 = 85


# ---------------------------------------------------------------------------
# get_pl (line-widening SQL helper)
# ---------------------------------------------------------------------------

class TestGetPl:
    def test_returns_linestring_wkt(self):
        """get_pl queries the DB for a linestring widened to a given width.
        The DB returns a WKT string; we mock it."""
        mock_db = MagicMock()
        expected_wkt = 'LINESTRING(14 55, 15 56)'
        mock_db.execute_and_return.return_value = [True, [[expected_wkt]]]
        result = get_pl(mock_db, lat1=55.0, lat2=56.0, lon1=14.0, lon2=15.0,
                        l_width=1000.0)
        assert result == expected_wkt
        # Verify it issued one query containing the lat/lon/half-width.
        assert mock_db.execute_and_return.call_count == 1
        # SQL uses `l_width/2` -> 500.0 for width=1000.
        call_sql = mock_db.execute_and_return.call_args.args[0]
        assert '14.0' in call_sql and '55.0' in call_sql
        assert '500.0' in call_sql  # l_width / 2

    def test_returns_empty_when_db_query_fails(self):
        """When the DB reports an error (ok=False), get_pl returns ''."""
        mock_db = MagicMock()
        mock_db.execute_and_return.return_value = [False, 'error msg']
        assert get_pl(mock_db, 55.0, 56.0, 14.0, 15.0, l_width=500) == ''

    def test_buffer_uses_width(self):
        """Different widths produce different half-widths in the SQL."""
        mock_db = MagicMock()
        mock_db.execute_and_return.return_value = [True, [['LINESTRING(...)']]]
        get_pl(mock_db, 55.0, 56.0, 14.0, 15.0, l_width=500)
        sql_a = mock_db.execute_and_return.call_args.args[0]
        get_pl(mock_db, 55.0, 56.0, 14.0, 15.0, l_width=2000)
        sql_b = mock_db.execute_and_return.call_args.args[0]
        assert sql_a != sql_b
        assert '250.0' in sql_a        # 500 / 2
        assert '1000.0' in sql_b       # 2000 / 2


# ---------------------------------------------------------------------------
# update_ais_settings_file
# ---------------------------------------------------------------------------

class TestUpdateAisSettingsFile:
    def test_writes_the_four_fields(self, tmp_path, monkeypatch):
        """Function writes a ui/ais_settings.py module with the 4 values."""
        # The production function writes to an absolute path next to the
        # module.  Instead of changing cwd, monkey-patch ``open`` to write
        # into tmp_path.
        target = tmp_path / 'ais_settings.py'
        import builtins
        real_open = builtins.open

        def fake_open(path, *args, **kwargs):
            if str(path).endswith('ais_settings.py'):
                return real_open(str(target), *args, **kwargs)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, 'open', fake_open)
        update_ais_settings_file('my-host', 'alice', 's3cret', 'ais_db')
        content = target.read_text()
        assert "db_host = 'my-host'" in content
        assert "db_user = 'alice'" in content
        assert "db_password = 's3cret'" in content
        assert "db_name = 'ais_db'" in content


# ---------------------------------------------------------------------------
# AIS class methods -- exercised via mocks so no real DB / QGIS is needed.
# ---------------------------------------------------------------------------

@pytest.fixture
def ais_with_mocks(monkeypatch):
    """Build an AIS instance with the DB and QSettings classes patched."""
    from unittest.mock import patch

    # Patch DB to a no-op before importing AIS so ``set_start_ais_settings``
    # succeeds.  AISConnectionWidget is also a QWidget; patch it as a MagicMock.
    # Also patch ``VesselLookupConfig.to_qsettings`` so the external-vessel
    # round-trip doesn't try to serialise MagicMock values into a real
    # QSettings backend (which under ``pytest-qgis`` is the live QGIS one
    # and can hang the run).
    from omrat_utils.vessel_lookup import VesselLookupConfig

    with patch('omrat_utils.handle_ais.DB') as MockDB, \
         patch('omrat_utils.handle_ais.AISConnectionWidget') as MockACW, \
         patch('omrat_utils.handle_ais.QSettings') as MockSettings, \
         patch('omrat_utils.vessel_lookup.VesselLookupConfig.to_qsettings'), \
         patch('omrat_utils.vessel_lookup.VesselLookupConfig.from_qsettings',
               return_value=VesselLookupConfig()):
        MockSettings.return_value.value.return_value = ''
        acw = MagicMock()
        # Month checkboxes return False so ``months`` stays empty.
        for i in range(1, 13):
            cb = MagicMock()
            cb.isChecked.return_value = False
            setattr(acw, f'CB_{i}', cb)
        acw.leMaxDev.text.return_value = '10.0'
        # Port spinbox defaults to the Postgres standard.
        acw.SBPort.value.return_value = 5432
        # Recalc-to-full-year checkbox starts off; tests opt in by
        # setting ``ais_with_mocks.recalc_to_full_year = True``.
        acw.cbRecalcFullYear.isChecked.return_value = False
        # External-vessel-lookup widgets return concrete strings so
        # ``VesselLookupConfig`` is built from real text rather than from
        # MagicMock objects (which break dataclass equality and any
        # downstream string handling).
        acw.gbExtVessel.isChecked.return_value = False
        for name in (
            'leExtSchema', 'leExtTable', 'leExtMmsiCol',
            'leExtLoaCol', 'leExtBeamCol', 'leExtShipTypeCol',
            'leExtAirDraughtCol',
        ):
            getattr(acw, name).text.return_value = ''
        MockACW.return_value = acw
        MockDB.return_value = MagicMock()

        from omrat_utils.handle_ais import AIS
        omrat = MagicMock()
        ais = AIS(omrat)
        ais.acw = acw  # ensure the mock is visible to tests
        yield ais


class TestAISRunAndUnload:
    def test_unload_is_noop(self, ais_with_mocks):
        # Simply ensure the method exists and returns None.
        assert ais_with_mocks.unload() is None

    def test_run_shows_and_executes_dialog(self, ais_with_mocks):
        ais_with_mocks.run()
        ais_with_mocks.acw.show.assert_called_once()
        # PyQt6 (QGIS 4) dropped ``exec_``; the production code calls the
        # underscore-free ``exec`` so it works on both PyQt5 and PyQt6.
        ais_with_mocks.acw.exec.assert_called_once()

    def test_set_start_ais_settings_db_exception_sets_none(
        self, ais_with_mocks, monkeypatch
    ):
        """If DB(...) raises, db is set to None."""
        import omrat_utils.handle_ais as mod

        def boom(**kwargs):
            raise RuntimeError("no db")

        monkeypatch.setattr(mod, 'DB', boom)
        # Re-run the settings loader; should not raise.
        ais_with_mocks.set_start_ais_settings()
        assert ais_with_mocks.db is None


class TestAISUpdateSettings:
    def test_update_ais_settings_persists_to_qsettings(self, ais_with_mocks):
        """The method reads widget text and stores db_host/port/user/pass/name
        into QSettings."""
        import omrat_utils.handle_ais as mod
        from unittest.mock import patch

        ais_with_mocks.acw.leDBHost.text.return_value = 'new-host'
        ais_with_mocks.acw.SBPort.value.return_value = 6543
        ais_with_mocks.acw.leDBName.text.return_value = 'new-db'
        ais_with_mocks.acw.leUserName.text.return_value = 'u'
        ais_with_mocks.acw.lePassword.text.return_value = 'p'
        ais_with_mocks.acw.leProvider.text.return_value = 'ais'
        ais_with_mocks.acw.SBYear.value.return_value = 2024
        ais_with_mocks.acw.leMaxDev.text.return_value = '15.0'

        # Re-patch DB locally: under pytest-qgis the fixture's outer patch
        # context can be unwound by the time this test runs, so the real
        # DB constructor would attempt to connect to ``new-host`` and hang.
        with patch.object(mod, 'DB') as MockDB:
            MockDB.return_value = MagicMock()
            ais_with_mocks.update_ais_settings()
        # QSettings.setValue called 6 times: host, port, user, pass, name,
        # plus the recalc-to-full-year flag (the vessel-lookup config
        # writes to its own QSettings instance, not this mock).
        setval = ais_with_mocks.settings.setValue
        assert setval.call_count == 6
        stored = {c.args[0]: c.args[1] for c in setval.call_args_list}
        assert stored['omrat/db_host'] == 'new-host'
        assert stored['omrat/db_port'] == 6543
        assert stored['omrat/db_name'] == 'new-db'
        assert stored['omrat/recalc_to_full_year'] is False

    def test_update_ais_settings_passes_port_to_db(self, ais_with_mocks):
        """The chosen port flows through to the DB constructor."""
        import omrat_utils.handle_ais as mod
        from unittest.mock import patch

        ais_with_mocks.acw.leDBHost.text.return_value = 'h'
        ais_with_mocks.acw.SBPort.value.return_value = 6543
        ais_with_mocks.acw.leDBName.text.return_value = 'n'
        ais_with_mocks.acw.leUserName.text.return_value = 'u'
        ais_with_mocks.acw.lePassword.text.return_value = 'p'
        ais_with_mocks.acw.leProvider.text.return_value = 'ais'
        ais_with_mocks.acw.SBYear.value.return_value = 2024
        ais_with_mocks.acw.leMaxDev.text.return_value = '10'

        with patch.object(mod, 'DB') as MockDB:
            MockDB.return_value = MagicMock()
            ais_with_mocks.update_ais_settings()
        kwargs = MockDB.call_args.kwargs
        assert kwargs['db_port'] == 6543
        assert kwargs['db_host'] == 'h'

    def test_checked_months_collected(self, ais_with_mocks):
        """Checkboxes that return isChecked=True append their month index."""
        # Check months 3 and 7.
        ais_with_mocks.acw.CB_3.isChecked.return_value = True
        ais_with_mocks.acw.CB_7.isChecked.return_value = True
        ais_with_mocks.acw.leDBHost.text.return_value = 'h'
        ais_with_mocks.acw.leDBName.text.return_value = 'n'
        ais_with_mocks.acw.leUserName.text.return_value = 'u'
        ais_with_mocks.acw.lePassword.text.return_value = 'p'
        ais_with_mocks.acw.leProvider.text.return_value = 'prov'
        ais_with_mocks.acw.SBYear.value.return_value = 2024
        ais_with_mocks.acw.leMaxDev.text.return_value = '10'
        ais_with_mocks.update_ais_settings()
        assert ais_with_mocks.months == [3, 7]

    def test_update_ais_settings_db_failure_shows_popup(
        self, ais_with_mocks, monkeypatch
    ):
        """When the user clicks Save and DB() raises (wrong password,
        missing role, server down), a QMessageBox.warning popup must
        appear with the actual server message — *not* a Python-error
        traceback dialog.  Regression test for the user's reported
        ``UnicodeDecodeError: 0xf6`` failure mode."""
        import omrat_utils.handle_ais as mod
        from unittest.mock import patch

        ais_with_mocks.acw.leDBHost.text.return_value = 'localhost'
        ais_with_mocks.acw.leDBName.text.return_value = 'omrat'
        ais_with_mocks.acw.leUserName.text.return_value = 'omrat'
        ais_with_mocks.acw.lePassword.text.return_value = 'wrong'
        ais_with_mocks.acw.leProvider.text.return_value = 'omrat'
        ais_with_mocks.acw.SBYear.value.return_value = 2024
        ais_with_mocks.acw.leMaxDev.text.return_value = '10'

        # The legacy DB._connect surfaces a friendly cp1252-decoded
        # message when libpq pre-startup fails — simulate that here.
        decoded = (
            "Error connecting to database on 'localhost'.\n"
            "Server message (decoded as cp1252):\n"
            "  FATAL:  Lösenordsautentisering misslyckades "
            'för användare "omrat"'
        )

        def fail(**kwargs):
            raise Exception(decoded)

        monkeypatch.setattr(mod, 'DB', fail)
        # Reset the QSettings mock so we can check it's NOT called.
        ais_with_mocks.settings.setValue.reset_mock()

        with patch.object(mod, 'QMessageBox') as MockMsg:
            ais_with_mocks.update_ais_settings()

        # Popup was shown with the decoded message as its body.
        MockMsg.warning.assert_called_once()
        body = MockMsg.warning.call_args.args[2]
        assert 'Lösenordsautentisering misslyckades' in body
        assert 'omrat' in body
        # No connection means no credentials persisted to QSettings —
        # bad creds shouldn't overwrite previously-working ones.
        ais_with_mocks.settings.setValue.assert_not_called()
        # And db is reset to None so update_legs hits its existing guard.
        assert ais_with_mocks.db is None


class TestAISUpdateLegsGuard:
    def test_update_legs_returns_early_when_db_is_none(self, ais_with_mocks):
        """If self.db is None the method shows a popup and returns without
        running the segment-data pipeline."""
        from unittest.mock import patch
        import omrat_utils.handle_ais as mod

        ais_with_mocks.db = None
        # Patch QMessageBox so the real Qt binding doesn't reject the
        # MagicMock parent widget — we only care that the segment-data
        # pipeline is skipped.
        with patch.object(mod, 'QMessageBox') as MockMsg:
            ais_with_mocks.update_legs()
        MockMsg.information.assert_called_once()
        assert not ais_with_mocks.omrat.traffic.create_empty_dict.called

    def test_update_legs_with_key_uses_single_leg(self, ais_with_mocks, monkeypatch):
        """Calling update_legs(key='L1') iterates only that one leg
        (L175 branch) and falls into the ``leg_key not in leg_dirs`` init
        (L179)."""
        import numpy as np
        segment = {
            'L1': {
                'Start_Point': '14.0 55.0',
                'End_Point': '14.1 55.0',
                'Width': '1000',
            },
        }
        monkeypatch.setattr(
            ais_with_mocks, 'get_segment_data_from_table',
            lambda: segment,
        )
        # Force leg_dirs to be empty so the ``leg_key not in leg_dirs`` branch fires.
        ais_with_mocks.omrat.qgis_geoms.leg_dirs = {}
        ais_with_mocks.omrat.segment_data = {'L1': {'Dirs': ['East', 'West']}}
        # Stub run_sql + update_ais_data + update_dist_data to trivial values.
        monkeypatch.setattr(ais_with_mocks, 'run_sql', lambda pl: [])
        monkeypatch.setattr(
            ais_with_mocks, 'update_ais_data',
            lambda *a, **k: (np.array([]), np.array([])),
        )
        monkeypatch.setattr(ais_with_mocks, 'update_dist_data', lambda *a, **k: None)
        monkeypatch.setattr(ais_with_mocks, 'convert_list2avg', lambda: None)

        ais_with_mocks.db = MagicMock()
        ais_with_mocks.db.execute_and_return.return_value = (True, [[270.0]])

        # Widget stubs used at the end of update_legs.
        ais_with_mocks.omrat.main_widget.leNormMean1_1.setText = MagicMock()
        ais_with_mocks.omrat.distributions.run_update_plot = MagicMock()
        ais_with_mocks.omrat.main_widget.cbTrafficSelectSeg.count.return_value = 1

        ais_with_mocks.update_legs(key='L1')

        # leg_dirs now has the one leg populated.
        assert ais_with_mocks.omrat.qgis_geoms.leg_dirs['L1'] == ['East', 'West']

    def test_update_legs_runsql_exception_shows_popup(
        self, ais_with_mocks, monkeypatch
    ):
        """A RuntimeError from run_sql triggers show_error_popup and return."""
        segment = {'L1': {'Start_Point': '14.0 55.0', 'End_Point': '14.1 55.0',
                          'Width': '1000'}}
        monkeypatch.setattr(
            ais_with_mocks, 'get_segment_data_from_table', lambda: segment,
        )
        ais_with_mocks.omrat.qgis_geoms.leg_dirs = {'L1': ['E', 'W']}
        ais_with_mocks.omrat.segment_data = {'L1': {'Dirs': ['E', 'W']}}

        def bad_run_sql(pl):
            raise RuntimeError('database refused')

        monkeypatch.setattr(ais_with_mocks, 'run_sql', bad_run_sql)
        ais_with_mocks.db = MagicMock()
        # get_pl (called before run_sql) needs a valid 2-tuple response.
        ais_with_mocks.db.execute_and_return.return_value = (True, [['LINESTRING(...)']])

        ais_with_mocks.update_legs(key='L1')
        ais_with_mocks.omrat.show_error_popup.assert_called_once()


class TestAISRunSql:
    def test_run_sql_raises_when_no_db(self, ais_with_mocks):
        """With self.db == None, run_sql raises RuntimeError."""
        ais_with_mocks.db = None
        with pytest.raises(RuntimeError):
            ais_with_mocks.run_sql('LINESTRING(0 0, 1 1)')

    def test_run_sql_raises_on_db_error(self, ais_with_mocks):
        """DB returning ok=False -> run_sql raises TypeError with the error."""
        ais_with_mocks.db = MagicMock()
        ais_with_mocks.db.execute_and_return.return_value = (False, [['sql broke']])
        ais_with_mocks.schema = 'ais'
        ais_with_mocks.year = 2024
        ais_with_mocks.months = []
        with pytest.raises(TypeError):
            ais_with_mocks.run_sql('LINESTRING(0 0, 1 1)')

    def test_run_sql_with_months_builds_union(self, ais_with_mocks):
        """months = [1,2] builds a SQL with 2 UNION'd segment_year_month tables."""
        ais_with_mocks.db = MagicMock()
        ais_with_mocks.db.execute_and_return.return_value = (True, [])
        ais_with_mocks.schema = 'ais'
        ais_with_mocks.year = 2024
        ais_with_mocks.months = [1, 2]
        ais_with_mocks.run_sql('LINESTRING(0 0, 1 1)')
        sql = ais_with_mocks.db.execute_and_return.call_args.args[0]
        assert 'segments_2024_1' in sql
        assert 'segments_2024_2' in sql

    def test_run_sql_does_not_join_external_table_by_default(self, ais_with_mocks):
        """Without a configured external vessel-lookup table, ``run_sql``
        derives loa/beam straight from ``dim_a..d`` in the statics, and
        emits NULL for ship_type / air_draught — no extra LEFT JOIN."""
        ais_with_mocks.db = MagicMock()
        ais_with_mocks.db.execute_and_return.return_value = (True, [])
        ais_with_mocks.schema = 'ais'
        ais_with_mocks.year = 2024
        ais_with_mocks.months = [1]
        ais_with_mocks.run_sql('LINESTRING(0 0, 1 1)')
        sql = ais_with_mocks.db.execute_and_return.call_args.args[0]
        assert 'external_vessels' not in sql
        assert 'LEFT OUTER JOIN' not in sql
        # Dimensions still computed from statics.
        assert 'dim_a + dim_b' in sql
        assert 'dim_c + dim_d' in sql
        # Two NULL placeholder columns to keep the consumer's 10-tuple
        # unpack signature stable.
        assert 'NULL::int as ship_type' in sql
        assert 'NULL::double precision as air_draught' in sql

    def test_run_sql_with_external_vessel_lookup_emits_join(self, ais_with_mocks):
        """When the user has configured an external vessel-lookup table
        (via the AIS Settings dialog), ``run_sql`` injects a CTE +
        LEFT JOIN so loa/beam fall back to the user's columns when the
        AIS dim arithmetic looks bogus, and ship_type/air_draught are
        populated from there."""
        from omrat_utils.vessel_lookup import VesselLookupConfig
        ais_with_mocks.db = MagicMock()
        ais_with_mocks.db.execute_and_return.return_value = (True, [])
        ais_with_mocks.schema = 'ais'
        ais_with_mocks.year = 2024
        ais_with_mocks.months = [1]
        ais_with_mocks.vessel_lookup = VesselLookupConfig(
            enabled=True, schema='vessels', table='ship_registry',
            mmsi_col='mmsi',
            loa_col='loa', beam_col='breadth_moulded',
            ship_type_col='ship_type', air_draught_col='height',
        )
        ais_with_mocks.run_sql('LINESTRING(0 0, 1 1)')
        sql = ais_with_mocks.db.execute_and_return.call_args.args[0]
        # CTE + JOIN are present.
        assert 'external_vessels AS (' in sql
        assert 'FROM vessels.ship_registry' in sql
        assert 'LEFT OUTER JOIN external_vessels ext ON ss.mmsi = ext.mmsi' in sql
        # The CASE bodies now reference the external columns instead of NULL.
        assert 'then ext.ext_loa else dim_a + dim_b' in sql
        assert 'then ext.ext_beam else dim_c + dim_d' in sql
        # ship_type and air_draught come from the external table now.
        assert 'ext.ext_ship_type as ship_type' in sql
        assert 'ext.ext_air_draught as air_draught' in sql

    def test_run_sql_skips_join_when_lookup_disabled(self, ais_with_mocks):
        """``enabled=False`` keeps the legacy statics-only path, even if
        the rest of the config is filled in."""
        from omrat_utils.vessel_lookup import VesselLookupConfig
        ais_with_mocks.db = MagicMock()
        ais_with_mocks.db.execute_and_return.return_value = (True, [])
        ais_with_mocks.schema = 'ais'
        ais_with_mocks.year = 2024
        ais_with_mocks.months = [1]
        ais_with_mocks.vessel_lookup = VesselLookupConfig(
            enabled=False,  # all other fields filled in but disabled
            schema='vessels', table='ship_registry', mmsi_col='mmsi',
            loa_col='loa',
        )
        ais_with_mocks.run_sql('LINESTRING(0 0, 1 1)')
        sql = ais_with_mocks.db.execute_and_return.call_args.args[0]
        assert 'external_vessels' not in sql
        assert 'LEFT OUTER JOIN' not in sql

    def test_compute_year_multiplier_returns_none_when_no_db(self, ais_with_mocks):
        ais_with_mocks.db = None
        assert ais_with_mocks.compute_year_multiplier() is None

    def test_compute_year_multiplier_for_48h_clean(self, ais_with_mocks):
        """48 h of data with no gaps > 12 h should give a multiplier of
        ~365*24/48 = 182.5x (the example from the user's spec)."""
        ais_with_mocks.db = MagicMock()
        ais_with_mocks.schema = 'ais'
        ais_with_mocks.year = 2024
        # span_s = 48 h, gap_s = 0 → coverage = 48 h, multiplier = 365*24/48
        ais_with_mocks.db.execute_and_return.return_value = (
            True, [[48 * 3600, 0]],
        )
        out = ais_with_mocks.compute_year_multiplier()
        assert out is not None
        multiplier, coverage_s, gap_s = out
        assert coverage_s == 48 * 3600
        assert gap_s == 0
        assert multiplier == pytest.approx(365 * 24 / 48, rel=1e-6)
        # SQL was issued with the schema/year; the 12-hour gap threshold is
        # now passed as a bound integer parameter rather than literal text.
        call = ais_with_mocks.db.execute_and_return.call_args
        sql = _render(call.args[0])
        assert 'ais.segments_2024' in sql
        assert 'make_interval(hours => %s)' in sql
        assert call.kwargs.get('params') == (12,)

    def test_compute_year_multiplier_subtracts_long_gaps(self, ais_with_mocks):
        """A 200 h span with a 100 h receiver outage gives a 100 h
        coverage, so the multiplier is 365*24/100."""
        ais_with_mocks.db = MagicMock()
        ais_with_mocks.schema = 'ais'
        ais_with_mocks.year = 2024
        ais_with_mocks.db.execute_and_return.return_value = (
            True, [[200 * 3600, 100 * 3600]],
        )
        multiplier, coverage_s, gap_s = ais_with_mocks.compute_year_multiplier()
        assert coverage_s == 100 * 3600
        assert gap_s == 100 * 3600
        assert multiplier == pytest.approx(365 * 24 / 100, rel=1e-6)

    def test_compute_year_multiplier_returns_none_on_empty_table(self, ais_with_mocks):
        """An empty segments table gives MAX/MIN = NULL — bail rather
        than divide by zero."""
        ais_with_mocks.db = MagicMock()
        ais_with_mocks.schema = 'ais'
        ais_with_mocks.year = 2024
        ais_with_mocks.db.execute_and_return.return_value = (True, [[None, 0]])
        assert ais_with_mocks.compute_year_multiplier() is None

    def test_compute_year_multiplier_returns_none_on_query_error(self, ais_with_mocks):
        ais_with_mocks.db = MagicMock()
        ais_with_mocks.schema = 'ais'
        ais_with_mocks.year = 2024
        ais_with_mocks.db.execute_and_return.return_value = (False, [['boom']])
        assert ais_with_mocks.compute_year_multiplier() is None

    def test_run_sql_returns_10_columns(self, ais_with_mocks):
        """``update_ais_data`` unpacks 10 fields per row, so the SELECT
        must keep emitting that many across all SQL-shape changes.
        Counting commas at the SELECT top-level is fragile but enough as a
        guard-rail — adding a column should re-trigger this assertion."""
        ais_with_mocks.db = MagicMock()
        ais_with_mocks.db.execute_and_return.return_value = (True, [])
        ais_with_mocks.schema = 'ais'
        ais_with_mocks.year = 2024
        ais_with_mocks.months = [1]
        ais_with_mocks.run_sql('LINESTRING(0 0, 1 1)')
        sql = ais_with_mocks.db.execute_and_return.call_args.args[0]
        # Each output column appears once in the final SELECT.
        for col in (
            "as loa,",
            "as beam,",
            "type_and_cargo,",
            "draught,",
            "as ship_type,",
            "date1,",
            "sog,",
            "as air_draught,",
            "as dist_from_start,",
            "cog ",
        ):
            assert col in sql, f"output column {col!r} missing from SELECT"


class TestAISUpdateAisData:
    def test_none_loa_defaults_to_100(self, ais_with_mocks):
        """A row with loa=None uses fallback 100 -> bucket 3 (loa_i*25 < 100
        fails for i=3 due to strict <)."""
        ais_with_mocks.max_deviation = 45.0
        ais_with_mocks.omrat.traffic.traffic_data = {
            'L1': {
                'East': {
                    'Frequency (ships/year)': [[0] * 5 for _ in range(21)],
                    'Speed (knots)': [[[] for _ in range(5)] for _ in range(21)],
                    'Ship heights (meters)': [[[] for _ in range(5)] for _ in range(21)],
                    'Ship Beam (meters)': [[[] for _ in range(5)] for _ in range(21)],
                    'Draught (meters)': [[[] for _ in range(5)] for _ in range(21)],
                },
                'West': {
                    'Frequency (ships/year)': [[0] * 5 for _ in range(21)],
                    'Speed (knots)': [[[] for _ in range(5)] for _ in range(21)],
                    'Ship heights (meters)': [[[] for _ in range(5)] for _ in range(21)],
                    'Ship Beam (meters)': [[[] for _ in range(5)] for _ in range(21)],
                    'Draught (meters)': [[[] for _ in range(5)] for _ in range(21)],
                },
            },
        }
        # One AIS row: loa=None, beam=20, toc=70, draugt=6.0, sh_type=None,
        # date1=..., sog=12.0, air_draught=20.0, dist=0.0, cog=90.0
        row = [None, 20, 70, 6.0, None, '2024-01-01', 12.0, 20.0, 0.0, 90.0]
        l1, l2 = ais_with_mocks.update_ais_data(
            'L1', [row], leg_bearing=270.0, dirs=['East', 'West'],
        )
        # Cog 90 vs leg_bearing+180=450%360=90 -> matches line1.
        assert len(l1) == 1 and len(l2) == 0

    def test_multiplier_scales_frequency_only(self, ais_with_mocks):
        """The multiplier is added to ``Frequency (ships/year)`` per ping
        but does NOT scale Speed/Beam/Draught/Heights — those are
        observations later averaged by ``convert_list2avg``."""
        ais_with_mocks.max_deviation = 45.0
        # Row layout: loa, beam, toc, draught, sh_type, _, sog, air_draught, dist, cog
        row = [100, 20, 70, 6.0, None, '2024-01-01', 12.0, 20.0, 0.0, 90.0]
        ais_with_mocks.omrat.traffic.traffic_data = {
            'L1': {
                'East': {
                    'Frequency (ships/year)': [[0] * 5 for _ in range(21)],
                    'Speed (knots)': [[[] for _ in range(5)] for _ in range(21)],
                    'Ship heights (meters)': [[[] for _ in range(5)] for _ in range(21)],
                    'Ship Beam (meters)': [[[] for _ in range(5)] for _ in range(21)],
                    'Draught (meters)': [[[] for _ in range(5)] for _ in range(21)],
                },
                'West': {
                    'Frequency (ships/year)': [[0] * 5 for _ in range(21)],
                    'Speed (knots)': [[[] for _ in range(5)] for _ in range(21)],
                    'Ship heights (meters)': [[[] for _ in range(5)] for _ in range(21)],
                    'Ship Beam (meters)': [[[] for _ in range(5)] for _ in range(21)],
                    'Draught (meters)': [[[] for _ in range(5)] for _ in range(21)],
                },
            },
        }
        ais_with_mocks.update_ais_data(
            'L1', [row, row, row], leg_bearing=270.0, dirs=['East', 'West'],
            multiplier=182.5,  # 365*24/48 - the user's 48 h example
        )
        td = ais_with_mocks.omrat.traffic.traffic_data['L1']['East']
        # 3 pings × 182.5 = 547.5 ships/year, all in the toc=70→cargo
        # bucket (type_cat=18) and loa=100 → loa_cat=3.
        assert td['Frequency (ships/year)'][18][3] == pytest.approx(547.5)
        # Observation lists carry the raw values, not scaled by 182.5.
        assert td['Speed (knots)'][18][3] == [12.0, 12.0, 12.0]
        assert td['Ship Beam (meters)'][18][3] == [20.0, 20.0, 20.0]
        assert td['Draught (meters)'][18][3] == [6.0, 6.0, 6.0]
        assert td['Ship heights (meters)'][18][3] == [20.0, 20.0, 20.0]

    def test_default_multiplier_is_one(self, ais_with_mocks):
        """When the recalc-to-full-year flag is off, every ping
        increments the frequency by exactly 1."""
        ais_with_mocks.max_deviation = 45.0
        row = [100, 20, 70, 6.0, None, '2024-01-01', 12.0, 20.0, 0.0, 90.0]
        ais_with_mocks.omrat.traffic.traffic_data = {
            'L1': {
                'East': {
                    'Frequency (ships/year)': [[0] * 5 for _ in range(21)],
                    'Speed (knots)': [[[] for _ in range(5)] for _ in range(21)],
                    'Ship heights (meters)': [[[] for _ in range(5)] for _ in range(21)],
                    'Ship Beam (meters)': [[[] for _ in range(5)] for _ in range(21)],
                    'Draught (meters)': [[[] for _ in range(5)] for _ in range(21)],
                },
                'West': {
                    'Frequency (ships/year)': [[0] * 5 for _ in range(21)],
                    'Speed (knots)': [[[] for _ in range(5)] for _ in range(21)],
                    'Ship heights (meters)': [[[] for _ in range(5)] for _ in range(21)],
                    'Ship Beam (meters)': [[[] for _ in range(5)] for _ in range(21)],
                    'Draught (meters)': [[[] for _ in range(5)] for _ in range(21)],
                },
            },
        }
        ais_with_mocks.update_ais_data(
            'L1', [row, row], leg_bearing=270.0, dirs=['East', 'West'],
        )
        td = ais_with_mocks.omrat.traffic.traffic_data['L1']['East']
        assert td['Frequency (ships/year)'][18][3] == 2

    def test_cog_outside_deviation_is_skipped(self, ais_with_mocks):
        """cog not matching either leg direction falls through the continue."""
        ais_with_mocks.max_deviation = 5.0
        ais_with_mocks.omrat.traffic.traffic_data = {
            'L1': {
                'East': {
                    'Frequency (ships/year)': [[0] * 5 for _ in range(21)],
                    'Speed (knots)': [[[] for _ in range(5)] for _ in range(21)],
                    'Ship heights (meters)': [[[] for _ in range(5)] for _ in range(21)],
                    'Ship Beam (meters)': [[[] for _ in range(5)] for _ in range(21)],
                    'Draught (meters)': [[[] for _ in range(5)] for _ in range(21)],
                },
                'West': {
                    'Frequency (ships/year)': [[0] * 5 for _ in range(21)],
                    'Speed (knots)': [[[] for _ in range(5)] for _ in range(21)],
                    'Ship heights (meters)': [[[] for _ in range(5)] for _ in range(21)],
                    'Ship Beam (meters)': [[[] for _ in range(5)] for _ in range(21)],
                    'Draught (meters)': [[[] for _ in range(5)] for _ in range(21)],
                },
            },
        }
        # cog 180 doesn't match bearing 90 +- 5 and doesn't match 270 +- 5.
        row = [100, 20, 70, 6.0, 'cargo', '2024-01-01', 12.0, 20.0, 0.0, 180.0]
        l1, l2 = ais_with_mocks.update_ais_data(
            'L1', [row], leg_bearing=270.0, dirs=['East', 'West'],
        )
        assert len(l1) == 0 and len(l2) == 0

    def test_oversize_loa_bucketed_into_last(self, ais_with_mocks):
        """A ship with LOA larger than any bucket's upper bound gets the
        last bucket (loa_cat = n_loa_cats - 1)."""
        ais_with_mocks.max_deviation = 45.0
        # 2 loa-cats: 0-25 and 25-50.  A loa=9999 falls outside both.
        ais_with_mocks.omrat.traffic.traffic_data = {
            'L1': {
                'East': {
                    'Frequency (ships/year)': [[0] * 2 for _ in range(21)],
                    'Speed (knots)': [[[] for _ in range(2)] for _ in range(21)],
                    'Ship heights (meters)': [[[] for _ in range(2)] for _ in range(21)],
                    'Ship Beam (meters)': [[[] for _ in range(2)] for _ in range(21)],
                    'Draught (meters)': [[[] for _ in range(2)] for _ in range(21)],
                },
                'West': {
                    'Frequency (ships/year)': [[0] * 2 for _ in range(21)],
                    'Speed (knots)': [[[] for _ in range(2)] for _ in range(21)],
                    'Ship heights (meters)': [[[] for _ in range(2)] for _ in range(21)],
                    'Ship Beam (meters)': [[[] for _ in range(2)] for _ in range(21)],
                    'Draught (meters)': [[[] for _ in range(2)] for _ in range(21)],
                },
            },
        }
        row = [9999, 20, 70, 6.0, 'cargo', '2024-01-01', 12.0, 20.0, 0.0, 90.0]
        l1, l2 = ais_with_mocks.update_ais_data(
            'L1', [row], leg_bearing=270.0, dirs=['East', 'West'],
        )
        # Cargo (row 18), last-bucket (col 1) should be 1.
        assert ais_with_mocks.omrat.traffic.traffic_data['L1']['East'][
            'Frequency (ships/year)'][18][1] == 1


class TestAISUpdateDistDataUiReset:
    def test_dist_data_sets_ui_when_empty(self, ais_with_mocks):
        """If leNormMean1_1 is currently 0.0, update_dist_data populates it
        from the new line means.  Exercises L236-239."""
        import numpy as np
        ais_with_mocks.omrat.segment_data = {'L1': {}}
        # Initial mean field is '0.0' -> triggers the setText branch.
        ais_with_mocks.omrat.main_widget.leNormMean1_1.text.return_value = '0.0'

        l1 = np.array([1.0, 2.0, 3.0])
        l2 = np.array([4.0, 5.0, 6.0])
        ais_with_mocks.update_dist_data(l1, l2, 'L1')

        # Four widget setters were called (mean1_1, mean2_1, std1_1, std2_1).
        mw = ais_with_mocks.omrat.main_widget
        mw.leNormMean1_1.setText.assert_called_with(str(l1.mean()))
        mw.leNormMean2_1.setText.assert_called_with(str(l2.mean()))
        mw.leNormStd1_1.setText.assert_called_with(str(l1.std()))
        mw.leNormStd2_1.setText.assert_called_with(str(l2.std()))
