"""Idempotent SQL migration runner for the OMRAT Postgres schema.

Migration files live in ``omrat_utils/db_setup/migrations/`` and follow the
``V{NNN}__{name}.sql`` convention.  Each file runs in a single transaction;
its version is recorded in ``omrat_meta.schema_version`` on success so reruns
skip already-applied migrations.

The year-partitioned tables (``segments_YYYY``, ``states_YYYY``,
``statics_YYYY``) are created on demand via :meth:`Migrator.ensure_year_partition`,
which renders ``templates/year_partition.sql.tpl`` for the requested schema and
year.  This keeps schema/year out of the static migration files.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import psycopg2
from psycopg2 import sql as psql
from psycopg2.extensions import connection as PgConnection

from omrat_utils.db_setup.connection_profile import ConnectionProfile


_MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_TEMPLATES_DIR = _MIGRATIONS_DIR / "templates"
_VERSIONED_FILE_RE = re.compile(r"^V(\d{3})__(.+)\.sql$")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MigrationError(RuntimeError):
    """Raised for any failure during migration discovery or application."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path

    @property
    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")


def _validate_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _IDENT_RE.match(value):
        raise MigrationError(f"Invalid {label}: {value!r}")
    return value


def discover_migrations(directory: Path = _MIGRATIONS_DIR) -> list[Migration]:
    """List versioned migration files in ascending version order."""
    if not directory.is_dir():
        raise MigrationError(f"Migration directory not found: {directory}")
    found: list[Migration] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        m = _VERSIONED_FILE_RE.match(path.name)
        if m is None:
            continue
        found.append(Migration(int(m.group(1)), m.group(2), path))
    found.sort(key=lambda mig: mig.version)
    return found


class Migrator:
    """Apply pending migrations and provision year partitions for a schema."""

    def __init__(self, profile: ConnectionProfile):
        self.profile = profile

    # ----------------------------------------------------------------- query

    def applied_versions(self) -> list[int]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'omrat_meta' AND table_name = 'schema_version'"
            )
            if cur.fetchone() is None:
                return []
            cur.execute("SELECT version FROM omrat_meta.schema_version ORDER BY version")
            return [int(r[0]) for r in cur.fetchall()]

    def pending_migrations(self) -> list[Migration]:
        applied = set(self.applied_versions())
        return [m for m in discover_migrations() if m.version not in applied]

    # ---------------------------------------------------------------- apply

    def apply_pending(self) -> list[Migration]:
        """Run every pending migration, in order, each in its own transaction."""
        applied: list[Migration] = []
        for mig in self.pending_migrations():
            self._apply_one(mig)
            applied.append(mig)
        return applied

    def _apply_one(self, mig: Migration) -> None:
        rendered = mig.sql.replace("{schema}", _validate_identifier(self.profile.schema, "schema"))
        with self._connect() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(rendered)
                    cur.execute(
                        "INSERT INTO omrat_meta.schema_version (version, name) "
                        "VALUES (%s, %s) ON CONFLICT (version) DO NOTHING",
                        (mig.version, mig.name),
                    )
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise MigrationError(f"Migration V{mig.version:03d} ({mig.name}) failed: {e}") from e

    # --------------------------------------------------------- year-partitions

    def ensure_year_partition(self, year: int) -> None:
        """Create ``segments_YYYY`` (+ monthly partitions), ``states_YYYY``, ``statics_YYYY``.

        Idempotent: every DDL statement uses ``IF NOT EXISTS`` so calling this
        repeatedly for the same year is a no-op.  The heavy ``segments``
        indexes are NOT created here — call :meth:`create_year_indexes`
        after bulk ingestion to build them.
        """
        self._render_and_run("year_partition.sql.tpl", year, "ensure_year_partition")

    def create_year_indexes(self, year: int) -> None:
        """Build the GiST + btree indexes on ``segments_YYYY`` after ingestion.

        Maintaining these during bulk INSERT slows ingestion by 5-10x;
        building them once at the end on populated data is the standard ETL
        pattern.  Idempotent: re-running on already-indexed tables is a no-op.
        Also runs ``ANALYZE`` so the planner picks them up immediately.
        """
        self._render_and_run(
            "year_partition_indexes.sql.tpl", year, "create_year_indexes"
        )

    def count_year_data(self, year: int) -> dict[str, int]:
        """Return ``{table: row_count}`` for the year's segments/states/statics
        plus the count of watermark rows whose ``last_t`` falls in that year.

        Tables that don't exist (e.g. the year was never provisioned) are
        omitted from the dict.  Used by the wizard to summarise what a
        :meth:`truncate_year` call is about to wipe.
        """
        if not (1900 <= int(year) <= 2999):
            raise MigrationError(f"Year out of range: {year!r}")
        schema = _validate_identifier(self.profile.schema, "schema")
        year_int = int(year)
        out: dict[str, int] = {}
        targets = [
            (schema, f"segments_{year_int}"),
            (schema, f"states_{year_int}"),
            (schema, f"statics_{year_int}"),
        ]
        with self._connect() as conn, conn.cursor() as cur:
            for schema_name, table_name in targets:
                tbl_label = f"{schema_name}.{table_name}"
                try:
                    cur.execute(
                        psql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                            psql.Identifier(schema_name),
                            psql.Identifier(table_name),
                        )
                    )
                    out[tbl_label] = int(cur.fetchone()[0])
                except Exception:
                    conn.rollback()
                    # Table missing or otherwise unqueryable — omit silently;
                    # the caller treats absence as "nothing to wipe".
                    continue
            try:
                cur.execute(
                    "SELECT COUNT(*) FROM omrat_meta.segment_watermark "
                    "WHERE last_t >= %s AND last_t < %s",
                    (f"{int(year)}-01-01 00:00:00+00",
                     f"{int(year) + 1}-01-01 00:00:00+00"),
                )
                out["omrat_meta.segment_watermark"] = int(cur.fetchone()[0])
            except Exception:
                conn.rollback()
        return out

    def truncate_year(self, year: int) -> None:
        """Wipe every row in the year's segments/states/statics + matching watermarks.

        Idempotent and safe to run on a never-populated year (skipped
        cleanly when the tables don't exist).  The schema itself is
        preserved — only data is cleared, so the year remains ready for a
        fresh ingestion run.

        FK ordering: states → statics, so we ``TRUNCATE`` the states +
        statics pair with ``CASCADE`` to keep the FK happy.  segments is
        wiped via the partition parent (cascades to all monthly children).
        Watermarks are filtered by ``last_t`` in ``[year, year+1)``.

        Raises :class:`MigrationError` on any DB error.
        """
        if not (1900 <= int(year) <= 2999):
            raise MigrationError(f"Year out of range: {year!r}")
        schema = _validate_identifier(self.profile.schema, "schema")
        y = int(year)
        # Build statements; each is wrapped in a savepoint so a missing
        # table (e.g. segments_YYYY was never provisioned) doesn't abort
        # the whole operation.
        steps = [
            ("segments", f"TRUNCATE {schema}.segments_{y}"),
            ("states+statics",
             f"TRUNCATE {schema}.states_{y}, {schema}.statics_{y} CASCADE"),
        ]
        with self._connect() as conn:
            try:
                with conn.cursor() as cur:
                    for label, stmt in steps:
                        cur.execute("SAVEPOINT s")
                        try:
                            cur.execute(stmt)
                            cur.execute("RELEASE SAVEPOINT s")
                        except psycopg2.errors.UndefinedTable:
                            cur.execute("ROLLBACK TO SAVEPOINT s")
                            cur.execute("RELEASE SAVEPOINT s")
                            # Year not provisioned — nothing to clear here.
                            continue
                    cur.execute(
                        "DELETE FROM omrat_meta.segment_watermark "
                        "WHERE last_t >= %s AND last_t < %s",
                        (f"{y}-01-01 00:00:00+00",
                         f"{y + 1}-01-01 00:00:00+00"),
                    )
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise MigrationError(f"truncate_year({year}) failed: {e}") from e

    def _render_and_run(self, template_name: str, year: int, label: str) -> None:
        if not (1900 <= int(year) <= 2999):
            raise MigrationError(f"Year out of range: {year!r}")
        template = _TEMPLATES_DIR / template_name
        if not template.is_file():
            raise MigrationError(f"Template missing: {template}")
        schema = _validate_identifier(self.profile.schema, "schema")
        # Render {year}+1 BEFORE {year} so "{year}+1" doesn't get partially
        # substituted as e.g. "2024+1" then misparsed.
        next_year = int(year) + 1
        rendered = (
            template.read_text(encoding="utf-8")
            .replace("{year}+1", str(next_year))
            .replace("{schema}", schema)
            .replace("{year}", str(int(year)))
        )
        with self._connect() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(rendered)
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise MigrationError(f"{label}({year}) failed: {e}") from e

    # ------------------------------------------------------------- internals

    def _connect(self) -> PgConnection:
        try:
            return psycopg2.connect(**self.profile.to_dsn())
        except psycopg2.OperationalError as e:
            raise MigrationError(f"Could not connect to {self.profile.host}: {e}") from e
