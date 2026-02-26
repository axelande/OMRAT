"""A database class DB"""
from typing import Any
import psycopg2
from psycopg2._psycopg import connection
import pandas as pd

__author__ = 'Axel Hörteborn'


class DB:
    """DB is a database class for running queries to a Postgres database
    Parameters
    ----------
    db_user: str
        The database username
    db_pass: str
        The database password
    db_name: str
        The database you want to access
    time_out: int
        Set the timeout (sec) for a connection, default 24 hours.
    """
    def __init__(self, db_user: str = "", db_pass: str = "",
                 db_name: str = "", db_host: str = "", time_out: int = 86400):
        self.db_host = db_host
        self.db_name = db_name
        self.db_user = db_user
        self.db_pass = db_pass
        self.db_port = "5432"
        self.time_out = time_out
        self.conn: connection | None = None
        if self.db_host != "":
            self._connect()

    def __del__(self):
        self._disconnect()

    def _connect(self):
        try:
            self.conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                port=self.db_port,
                user=self.db_user,
                password=self.db_pass,
                options="-c statement_timeout=" + str(self.time_out) + "000"
            )
        except psycopg2.OperationalError as e:
            raise Exception(f"Error connecting to database on '{self.db_host}'. {e!s}")

    def _disconnect(self):
        if hasattr(self, 'conn'):
            if self.conn != None:
                self.conn.close()

    def _reconnect(self):
        self._disconnect()
        self._connect()

    def execute_and_get_pd(self, sql: str, return_error: bool = False) -> list:
        """Execute the query and returns the result as a pandas dataframe
        Parameters
        ----------
        sql: str
            Your query
        return_error: bool
            If True the function returns [bool, pd/error_message]
            else it just return pd/False
        Returns
        -------
        list
            the list is organized as follow: [pd]
        """
        try:
            df = pd.read_sql(sql, con=self.conn)
            if return_error:
                return [True, df]
            else:
                return [df]
        except Exception as e:
            self._reconnect()
            if return_error:
                return [False, e]
            else:
                return [False]

    def execute_and_return(self, sql: str, return_error: bool = False) -> list[list[Any]] | tuple[bool, list[list[Any]]]:
        """Execute the query and returns the result as a list of list
        Parameters
        ----------
        sql: str
            Your query
        return_error: bool
            If True the function returns [bool, data/error_message]
            else it just return data/False
        Returns
        -------
        list
            the list is organised as follow: [row][column]
        Examples
        --------
        >>> execute_and_return('''SELECT 
                               FROM 
                               WHERE ''')
        """
        c = self.conn.cursor()
        try:
            c.execute(sql)
            data: list[list[Any]] = c.fetchall()
            if return_error:
                return True, data
            else:
                return data
        except Exception as e:
            self._reconnect()
            if return_error:
                return False, [[e]]
            else:
                return [[False]]

    def execute(self, sql: str, commit: bool = True, return_error: bool = False):
        """Execute the query
        Parameters
        ----------
        sql: str
            Your query
        commit: bool
            Could be set to False if a db.commit is run later (in order to save 
            time)
        return_error: bool
            If true the function returns False, error_failure else it just return False
        Examples
        --------
        >>> execute('''INSERT INTO 
                    SELECT 
                    FROM 
                    WHERE''')
        
        """
        c = self.conn.cursor()
        try:
            c.execute(sql)
            if commit:
                self.conn.commit()
        except Exception as e:
            self._reconnect()
            if return_error:
                return [False, e]
            else:
                return False
        if return_error:
            return [True, '']

    def commit(self):
        """Commits previously executed data"""
        self.conn.commit()

    def execute_get_row_count(self, sql, return_error: bool = False) -> int:
        """Execute the query
        Parameters
        ----------
        sql: str
            Your query
        return_error: bool
            If true the function returns False, error_failure else it just
            return False

        Returns
        -------
        notices: list
            A list of notices
        Examples
        --------
        >>> execute('''INSERT INTO
                    SELECT
                    FROM
                    WHERE''')
        """
        c = self.conn.cursor()
        try:
            c.execute(sql)
            self.conn.commit()
            row_count = c.rowcount
            if return_error:
                return [row_count, '']
            else:
                return row_count
        except Exception as e:
            self._reconnect()
            if return_error:
                return [False, e]
            else:
                return False

