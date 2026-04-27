"""Tests for ``compute.database.DB``.

The ``DB`` class wraps psycopg2 + pandas for AIS queries.  These tests
exercise the wrapper logic (return shapes, error handling, reconnect
behaviour) by patching ``psycopg2.connect`` so no real database is
needed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute.database import DB


# ---------------------------------------------------------------------------
# Construction / connect / disconnect
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_no_connect_when_host_blank(self):
        db = DB()  # db_host defaults to ''
        assert db.conn is None

    def test_attributes_stored(self):
        db = DB(db_user='u', db_pass='p', db_name='n', time_out=3600)
        assert db.db_user == 'u'
        assert db.db_pass == 'p'
        assert db.db_name == 'n'
        assert db.time_out == 3600
        assert db.db_port == '5432'

    def test_connect_called_when_host_provided(self):
        with patch('compute.database.psycopg2.connect') as mock_connect:
            mock_connect.return_value = MagicMock()
            db = DB(db_host='myhost', db_name='ais', db_user='u', db_pass='p')
            mock_connect.assert_called_once()
            assert db.conn is not None

    def test_connect_failure_raises(self):
        import psycopg2
        with patch('compute.database.psycopg2.connect',
                   side_effect=psycopg2.OperationalError('refused')):
            with pytest.raises(Exception, match='Error connecting to database'):
                DB(db_host='badhost')


class TestDisconnect:
    def test_disconnect_with_no_conn(self):
        db = DB()
        # Should not raise.
        db._disconnect()

    def test_disconnect_closes_open_conn(self):
        db = DB()
        db.conn = MagicMock()
        db._disconnect()
        db.conn.close.assert_called_once()

    def test_reconnect_disconnects_then_connects(self):
        with patch('compute.database.psycopg2.connect') as mock_connect:
            mock_connect.return_value = MagicMock()
            db = DB(db_host='myhost')
            mock_connect.reset_mock()
            db._reconnect()
            mock_connect.assert_called_once()


# ---------------------------------------------------------------------------
# execute_and_return
# ---------------------------------------------------------------------------

class TestExecuteAndReturn:
    def test_success_returns_data_only(self):
        db = DB()
        cursor = MagicMock()
        cursor.fetchall.return_value = [[1, 2], [3, 4]]
        db.conn = MagicMock()
        db.conn.cursor.return_value = cursor
        result = db.execute_and_return('SELECT 1')
        assert result == [[1, 2], [3, 4]]

    def test_success_with_return_error_true(self):
        db = DB()
        cursor = MagicMock()
        cursor.fetchall.return_value = [[1, 2]]
        db.conn = MagicMock()
        db.conn.cursor.return_value = cursor
        ok, data = db.execute_and_return('SELECT 1', return_error=True)
        assert ok is True
        assert data == [[1, 2]]

    def test_failure_with_return_error_false_returns_false(self):
        db = DB()
        cursor = MagicMock()
        cursor.execute.side_effect = RuntimeError('bad sql')
        db.conn = MagicMock()
        db.conn.cursor.return_value = cursor
        # Patch reconnect to avoid trying to reconnect.
        db._reconnect = MagicMock()
        result = db.execute_and_return('BAD SQL')
        assert result == [[False]]

    def test_failure_with_return_error_true_returns_error_msg(self):
        db = DB()
        cursor = MagicMock()
        cursor.execute.side_effect = RuntimeError('bad sql')
        db.conn = MagicMock()
        db.conn.cursor.return_value = cursor
        db._reconnect = MagicMock()
        ok, data = db.execute_and_return('BAD SQL', return_error=True)
        assert ok is False
        assert isinstance(data[0][0], RuntimeError)


# ---------------------------------------------------------------------------
# execute_and_get_pd
# ---------------------------------------------------------------------------

class TestExecuteAndGetPd:
    def test_success_returns_dataframe(self):
        import pandas as pd
        db = DB()
        db.conn = MagicMock()
        with patch('compute.database.pd.read_sql',
                   return_value=pd.DataFrame({'x': [1, 2]})):
            result = db.execute_and_get_pd('SELECT 1')
            assert len(result) == 1
            assert isinstance(result[0], pd.DataFrame)

    def test_success_with_return_error(self):
        import pandas as pd
        db = DB()
        db.conn = MagicMock()
        with patch('compute.database.pd.read_sql',
                   return_value=pd.DataFrame({'x': [1]})):
            ok, df = db.execute_and_get_pd('SELECT 1', return_error=True)
            assert ok is True
            assert isinstance(df, pd.DataFrame)

    def test_failure_returns_false(self):
        db = DB()
        db.conn = MagicMock()
        db._reconnect = MagicMock()
        with patch('compute.database.pd.read_sql',
                   side_effect=RuntimeError('bad')):
            assert db.execute_and_get_pd('BAD') == [False]

    def test_failure_with_return_error_returns_exception(self):
        db = DB()
        db.conn = MagicMock()
        db._reconnect = MagicMock()
        with patch('compute.database.pd.read_sql',
                   side_effect=RuntimeError('bad')):
            ok, err = db.execute_and_get_pd('BAD', return_error=True)
            assert ok is False
            assert isinstance(err, RuntimeError)


# ---------------------------------------------------------------------------
# execute  (DML)
# ---------------------------------------------------------------------------

class TestExecute:
    def test_success_no_return(self):
        db = DB()
        cursor = MagicMock()
        db.conn = MagicMock()
        db.conn.cursor.return_value = cursor
        result = db.execute('UPDATE x SET y=1')
        # When commit=True (default) and no error: function returns None.
        assert result is None
        cursor.execute.assert_called_once()
        db.conn.commit.assert_called_once()

    def test_success_no_commit(self):
        db = DB()
        cursor = MagicMock()
        db.conn = MagicMock()
        db.conn.cursor.return_value = cursor
        db.execute('UPDATE x', commit=False)
        db.conn.commit.assert_not_called()

    def test_success_with_return_error_returns_true_empty(self):
        db = DB()
        cursor = MagicMock()
        db.conn = MagicMock()
        db.conn.cursor.return_value = cursor
        result = db.execute('UPDATE x', return_error=True)
        assert result == [True, '']

    def test_failure_returns_false(self):
        db = DB()
        cursor = MagicMock()
        cursor.execute.side_effect = RuntimeError('boom')
        db.conn = MagicMock()
        db.conn.cursor.return_value = cursor
        db._reconnect = MagicMock()
        assert db.execute('BAD') is False

    def test_failure_with_return_error_returns_exception(self):
        db = DB()
        cursor = MagicMock()
        cursor.execute.side_effect = RuntimeError('boom')
        db.conn = MagicMock()
        db.conn.cursor.return_value = cursor
        db._reconnect = MagicMock()
        ok, err = db.execute('BAD', return_error=True)
        assert ok is False
        assert isinstance(err, RuntimeError)


# ---------------------------------------------------------------------------
# commit
# ---------------------------------------------------------------------------

class TestCommit:
    def test_commit_calls_underlying_conn(self):
        db = DB()
        db.conn = MagicMock()
        db.commit()
        db.conn.commit.assert_called_once()


# ---------------------------------------------------------------------------
# execute_get_row_count
# ---------------------------------------------------------------------------

class TestExecuteGetRowCount:
    def test_success_returns_row_count(self):
        db = DB()
        cursor = MagicMock()
        cursor.rowcount = 7
        db.conn = MagicMock()
        db.conn.cursor.return_value = cursor
        assert db.execute_get_row_count('UPDATE') == 7

    def test_success_with_return_error_returns_pair(self):
        db = DB()
        cursor = MagicMock()
        cursor.rowcount = 3
        db.conn = MagicMock()
        db.conn.cursor.return_value = cursor
        result = db.execute_get_row_count('UPDATE', return_error=True)
        assert result == [3, '']

    def test_failure_returns_false(self):
        db = DB()
        cursor = MagicMock()
        cursor.execute.side_effect = RuntimeError('boom')
        db.conn = MagicMock()
        db.conn.cursor.return_value = cursor
        db._reconnect = MagicMock()
        assert db.execute_get_row_count('BAD') is False

    def test_failure_with_return_error_returns_exception(self):
        db = DB()
        cursor = MagicMock()
        cursor.execute.side_effect = RuntimeError('boom')
        db.conn = MagicMock()
        db.conn.cursor.return_value = cursor
        db._reconnect = MagicMock()
        ok, err = db.execute_get_row_count('BAD', return_error=True)
        assert ok is False
        assert isinstance(err, RuntimeError)
