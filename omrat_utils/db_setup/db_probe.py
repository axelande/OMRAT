"""State-detection probe for an OMRAT Postgres backend.

Runs a short series of read-only queries to determine which setup steps the
user still needs to complete.  Each capability check is independent and
records its own failure into ``ProbeResult.error_messages`` rather than
aborting the probe, so the wizard can show every gap in one pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import psycopg2
from psycopg2 import sql

from omrat_utils.db_setup.connection_profile import (
    ConnectionProfile,
    decode_libpq_message,
)


_PROBE_TIMEOUT_SEC = 5


@dataclass
class ProbeResult:
    server_reachable: bool = False
    can_login: bool = False
    server_version: Optional[str] = None
    postgis_installed: bool = False
    postgis_version: Optional[str] = None
    timescaledb_installed: bool = False
    timescaledb_version: Optional[str] = None
    target_schema_present: bool = False
    omrat_meta_present: bool = False
    schema_version: Optional[int] = None
    is_superuser: bool = False
    can_create_schema: bool = False
    can_create_extension: bool = False
    error_messages: list[str] = field(default_factory=list)

    @property
    def ready_for_omrat(self) -> bool:
        return (
            self.server_reachable
            and self.can_login
            and self.postgis_installed
            and self.target_schema_present
            and self.omrat_meta_present
            and self.schema_version is not None
        )


class DbProbe:
    """Capability probe for a Postgres server using a ``ConnectionProfile``."""

    def __init__(self, profile: ConnectionProfile):
        self.profile = profile

    # ---------------------------------------------------------------- public

    def probe(self) -> ProbeResult:
        result = ProbeResult()
        conn = self._connect(result)
        if conn is None:
            return result
        try:
            self._probe_server_version(conn, result)
            self._probe_extensions(conn, result)
            self._probe_target_schema(conn, result)
            self._probe_omrat_meta(conn, result)
            self._probe_privileges(conn, result)
        finally:
            conn.close()
        return result

    # --------------------------------------------------------------- helpers

    def _connect(self, result: ProbeResult):
        dsn = self.profile.to_dsn()
        dsn["connect_timeout"] = _PROBE_TIMEOUT_SEC
        try:
            conn = psycopg2.connect(**dsn)
        except psycopg2.OperationalError as e:
            msg = str(e).lower()
            # Distinguish "server unreachable" from "auth rejected" so the
            # wizard can show the right next step.  libpq's error strings
            # are stable enough to grep for these markers.
            if "could not translate host name" in msg or "could not connect" in msg or "timeout expired" in msg:
                result.server_reachable = False
            else:
                result.server_reachable = True
            result.error_messages.append(f"Connect failed: {e}")
            return None
        except UnicodeDecodeError as e:
            # libpq returns error messages in the OS locale (e.g. cp1252 on a
            # Swedish Windows install).  psycopg2 hard-decodes them as UTF-8
            # and crashes here before raising OperationalError, so we have to
            # swallow the decode error AND surface the actual server message
            # by decoding the bytes ourselves via the shared cp1252 fallback.
            server_msg = decode_libpq_message(e)
            reachable = self._tcp_reachable()
            result.server_reachable = reachable
            if reachable:
                hint = (
                    "Server is reachable but libpq returned a non-UTF8 error "
                    "message (likely auth or database/role error in a "
                    "non-English Windows locale)."
                )
                if server_msg:
                    hint += f"  Server message (cp1252): {server_msg}"
            else:
                hint = (
                    f"Could not reach {self.profile.host}:{self.profile.port}. "
                    "Is Postgres running?  If you are using the bundled docker "
                    "stack, run `docker compose -f docker/docker-compose.yml "
                    "up -d` and wait for the healthcheck."
                )
            result.error_messages.append(f"Connect failed (encoding): {hint}")
            return None
        result.server_reachable = True
        result.can_login = True
        return conn

    def _tcp_reachable(self) -> bool:
        """Plain TCP probe — used when psycopg2 itself blows up before we get
        a structured error back."""
        import socket
        try:
            with socket.create_connection(
                (self.profile.host, int(self.profile.port or 5432)),
                timeout=_PROBE_TIMEOUT_SEC,
            ):
                return True
        except OSError:
            return False

    def _probe_server_version(self, conn, result: ProbeResult) -> None:
        try:
            with conn.cursor() as cur:
                cur.execute("SHOW server_version")
                result.server_version = cur.fetchone()[0]
        except Exception as e:
            result.error_messages.append(f"server_version: {e}")

    def _probe_extensions(self, conn, result: ProbeResult) -> None:
        for extname, attr_installed, attr_version in (
            ("postgis", "postgis_installed", "postgis_version"),
            ("timescaledb", "timescaledb_installed", "timescaledb_version"),
        ):
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT extversion FROM pg_extension WHERE extname = %s",
                        (extname,),
                    )
                    row = cur.fetchone()
                    if row is not None:
                        setattr(result, attr_installed, True)
                        setattr(result, attr_version, row[0])
            except Exception as e:
                result.error_messages.append(f"{extname}: {e}")

    def _probe_target_schema(self, conn, result: ProbeResult) -> None:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
                    (self.profile.schema,),
                )
                result.target_schema_present = cur.fetchone() is not None
        except Exception as e:
            result.error_messages.append(f"target_schema: {e}")

    def _probe_omrat_meta(self, conn, result: ProbeResult) -> None:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'omrat_meta' AND table_name = 'schema_version'"
                )
                if cur.fetchone() is None:
                    return
                result.omrat_meta_present = True
                cur.execute("SELECT max(version) FROM omrat_meta.schema_version")
                row = cur.fetchone()
                if row and row[0] is not None:
                    result.schema_version = int(row[0])
        except Exception as e:
            result.error_messages.append(f"omrat_meta: {e}")

    def _probe_privileges(self, conn, result: ProbeResult) -> None:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT current_setting('is_superuser')")
                result.is_superuser = (cur.fetchone()[0] == "on")

                cur.execute(
                    "SELECT has_database_privilege(current_user, current_database(), 'CREATE')"
                )
                result.can_create_schema = bool(cur.fetchone()[0])

                # Creating extensions usually needs superuser, but a role with
                # CREATEDB / membership in pg_create_extension can also do it.
                # Treat superuser as the authoritative signal; the rest is a
                # best-effort check the wizard can override on user request.
                result.can_create_extension = result.is_superuser
        except Exception as e:
            result.error_messages.append(f"privileges: {e}")
