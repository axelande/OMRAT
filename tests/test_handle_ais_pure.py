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
    with patch('omrat_utils.handle_ais.DB') as MockDB, \
         patch('omrat_utils.handle_ais.AISConnectionWidget') as MockACW, \
         patch('omrat_utils.handle_ais.QSettings') as MockSettings:
        MockSettings.return_value.value.return_value = ''
        acw = MagicMock()
        # Month checkboxes return False so ``months`` stays empty.
        for i in range(1, 13):
            cb = MagicMock()
            cb.isChecked.return_value = False
            setattr(acw, f'CB_{i}', cb)
        acw.leMaxDev.text.return_value = '10.0'
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
        ais_with_mocks.acw.exec_.assert_called_once()

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
        """The method reads widget text and stores db_host/user/pass/name
        into QSettings."""
        ais_with_mocks.acw.leDBHost.text.return_value = 'new-host'
        ais_with_mocks.acw.leDBName.text.return_value = 'new-db'
        ais_with_mocks.acw.leUserName.text.return_value = 'u'
        ais_with_mocks.acw.lePassword.text.return_value = 'p'
        ais_with_mocks.acw.leProvider.text.return_value = 'ais'
        ais_with_mocks.acw.SBYear.value.return_value = 2024
        ais_with_mocks.acw.leMaxDev.text.return_value = '15.0'

        ais_with_mocks.update_ais_settings()
        # QSettings.setValue called 4 times, one per field.
        setval = ais_with_mocks.settings.setValue
        assert setval.call_count == 4
        stored = {c.args[0]: c.args[1] for c in setval.call_args_list}
        assert stored['omrat/db_host'] == 'new-host'
        assert stored['omrat/db_name'] == 'new-db'

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


class TestAISUpdateLegsGuard:
    def test_update_legs_returns_early_when_db_is_none(self, ais_with_mocks):
        """If self.db is None the method returns without touching anything."""
        ais_with_mocks.db = None
        # get_segment_data_from_table would raise if called on the MagicMock
        # in a way we can detect, so observe via the omrat mock not being
        # touched.
        ais_with_mocks.update_legs()
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
