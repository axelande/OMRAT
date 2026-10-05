"""Persistent history of OMRAT model runs (slim master DB).

Each completed Run Model invocation produces:

* one **per-run GeoPackage** in the user-selected output folder, named
  ``<model_name>_<YYYYMMDD_HHMMSS>.gpkg``, holding the actual result
  layers (drifting allision/grounding, powered allision/grounding,
  ship-collision lines and points);
* one row in the **master history database**
  (``omrat_history.sqlite`` under the user app-data folder) holding only
  metadata: run name, timestamp, elapsed duration, total probabilities
  for every accident type, and a pointer (``output_dir`` +
  ``output_filename``) to the per-run GeoPackage.

Splitting the spatial features into per-run files prevents the master
file from growing without bound and makes it trivial for the user to
archive, share, or delete a single run.

The class is intentionally QGIS-soft: every method here uses stdlib
``sqlite3`` only, so it can be unit-tested without a QGIS instance.
The actual GeoPackage *writer* (which needs ``QgsVectorFileWriter``)
lives in :mod:`omrat_utils.run_persistence`.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage location
# ---------------------------------------------------------------------------

def default_db_path() -> Path:
    """Return the platform-appropriate path of ``omrat_history.sqlite``.

    Resolves to ``<user-data>/OMRAT/omrat_history.sqlite`` where
    ``<user-data>`` is:

    * ``%APPDATA%`` on Windows
    * ``$XDG_DATA_HOME`` (or ``~/.local/share``) on Linux
    * ``~/Library/Application Support`` on macOS
    * ``~`` as a final fallback.

    QStandardPaths is **not** used here: inside QGIS it returns a
    profile-namespaced path (``…/QGIS/QGIS4/…``) which would hide the
    history under QGIS's own data folder rather than at the platform
    root. The README and the user mental model both expect
    ``%APPDATA%\\OMRAT`` directly.

    Two legacy paths are migrated in place if found (no data loss --
    same SQLite format):

    * older ``.gpkg`` filenames in the same folder, or
    * the QGIS-namespaced location written by an earlier build.
    """
    if os.name == 'nt':
        base = os.environ.get('APPDATA', '') or str(Path.home())
    elif os.name == 'posix':
        import sys as _sys
        if _sys.platform == 'darwin':
            base = str(Path.home() / 'Library' / 'Application Support')
        else:
            base = os.environ.get(
                'XDG_DATA_HOME', str(Path.home() / '.local' / 'share'),
            )
    else:
        base = str(Path.home())

    target = Path(base) / 'OMRAT' / 'omrat_history.sqlite'

    if not target.exists():
        # Legacy filenames in the same folder (renamed extension).
        for legacy_name in ('omrat_history.gpkg', 'omrat_runs.gpkg'):
            legacy = target.parent / legacy_name
            if legacy.exists():
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    legacy.rename(target)
                    logger.info(
                        f"Migrated legacy run-history file {legacy} -> {target}"
                    )
                    return target
                except Exception as exc:
                    logger.warning(f"Could not rename {legacy}: {exc}")

        # Earlier builds wrote under the QGIS-namespaced AppDataLocation;
        # pick that up too if present so the user keeps prior runs.
        try:
            from qgis.PyQt.QtCore import QStandardPaths
            loc = getattr(QStandardPaths, 'AppDataLocation', None)
            if loc is None:
                loc = QStandardPaths.StandardLocation.AppDataLocation
            qt_base = QStandardPaths.writableLocation(loc) or ''
        except Exception:
            qt_base = ''
        if qt_base:
            for legacy_name in (
                'omrat_history.sqlite', 'omrat_history.gpkg', 'omrat_runs.gpkg',
            ):
                legacy = Path(qt_base) / 'OMRAT' / legacy_name
                if legacy.exists() and legacy.resolve() != target.resolve():
                    try:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        legacy.rename(target)
                        logger.info(
                            f"Migrated QGIS-namespaced run-history "
                            f"{legacy} -> {target}"
                        )
                        break
                    except Exception as exc:
                        logger.warning(f"Could not move {legacy}: {exc}")

    return target


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# One metadata table.  No spatial features here -- those live in the
# per-run GeoPackage referenced by output_dir + output_filename.
_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS omrat_runs (
        run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
        name                TEXT    NOT NULL,
        timestamp           TEXT    NOT NULL,
        duration_seconds    REAL,
        drift_allision      REAL,
        drift_grounding     REAL,
        drift_anchoring     REAL,
        powered_grounding   REAL,
        powered_allision    REAL,
        head_on             REAL,
        overtaking          REAL,
        crossing            REAL,
        merging             REAL,
        bend                REAL,
        ship_collision_total REAL,
        output_dir          TEXT,
        output_filename     TEXT,
        notes               TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_runs_timestamp ON omrat_runs(timestamp)",
    # Project tag + OMRAT version per run (2026-10-02).  A side table, not
    # columns on ``omrat_runs``: ``CREATE TABLE IF NOT EXISTS`` adds it to
    # every existing history database, so runs recorded before it keep
    # loading (they simply have no row here), and an older OMRAT reading
    # this database still finds ``omrat_runs`` exactly as it expects.
    """
    CREATE TABLE IF NOT EXISTS omrat_run_info (
        run_id          INTEGER PRIMARY KEY,
        project         TEXT,
        omrat_version   TEXT
    )
    """,
]

# ``list_runs`` & co. read the side table along with every run.
_SELECT_RUNS = (
    "SELECT r.*, i.project AS project, i.omrat_version AS omrat_version "
    "FROM omrat_runs r LEFT JOIN omrat_run_info i ON i.run_id = r.run_id"
)

# Shown for a run with neither a project tag nor an output folder.
NO_PROJECT_LABEL = '(no project)'


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunMeta:
    """Lightweight summary returned by :meth:`RunHistory.list_runs`."""

    run_id: int
    name: str
    timestamp: str
    duration_seconds: float | None = None
    drift_allision: float | None = None
    drift_grounding: float | None = None
    drift_anchoring: float | None = None
    powered_grounding: float | None = None
    powered_allision: float | None = None
    head_on: float | None = None
    overtaking: float | None = None
    crossing: float | None = None
    merging: float | None = None
    bend: float | None = None
    ship_collision_total: float | None = None
    output_dir: str | None = None
    output_filename: str | None = None
    notes: str | None = None
    # From ``omrat_run_info``; ``None`` for runs recorded before it.
    project: str | None = None
    omrat_version: str | None = None

    def project_label(self) -> str:
        """The project this run is grouped under: its tag, else the name
        of the folder its results were written to."""
        tag = (self.project or '').strip()
        return tag or folder_project(self.output_dir) or NO_PROJECT_LABEL

    def snapshot_path(self) -> Path | None:
        """The ``.omrat`` input snapshot written next to the GeoPackage."""
        gpkg = self.gpkg_path()
        return gpkg.with_suffix('.omrat') if gpkg is not None else None

    def totals_dict(self) -> dict[str, float | None]:
        """All scalar totals as a plain dict (preserves ordering)."""
        return {
            'drift_allision': self.drift_allision,
            'drift_grounding': self.drift_grounding,
            'drift_anchoring': self.drift_anchoring,
            'powered_grounding': self.powered_grounding,
            'powered_allision': self.powered_allision,
            'head_on': self.head_on,
            'overtaking': self.overtaking,
            'crossing': self.crossing,
            'merging': self.merging,
            'bend': self.bend,
            'ship_collision_total': self.ship_collision_total,
        }

    def gpkg_path(self) -> Path | None:
        """Resolved per-run GeoPackage path, or None if absent."""
        if not self.output_dir or not self.output_filename:
            return None
        return Path(self.output_dir) / self.output_filename


# ---------------------------------------------------------------------------
# RunHistory
# ---------------------------------------------------------------------------

class RunHistory:
    """Read / write the slim history database (sqlite-only)."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create ``omrat_runs`` / ``omrat_run_info`` if absent.

        There is no column-migration step: ``_SCHEMA`` is the schema.  A
        history database written by an OMRAT older than the current
        ``_SCHEMA`` must be deleted (or the column added by hand) --
        adding an accident type means adding it to ``_SCHEMA`` and
        accepting that older databases are incompatible.  Per-run
        metadata that is not an accident total goes in a side table
        instead (see ``omrat_run_info``), which needs no migration.
        """
        with self._connect() as conn:
            for stmt in _SCHEMA:
                conn.execute(stmt)
            conn.commit()

    # ----------------- writes ----------------- #

    def save_run(
        self,
        name: str,
        *,
        timestamp: str | None = None,
        duration_seconds: float | None = None,
        totals: dict[str, float | None] | None = None,
        output_dir: str | Path | None = None,
        output_filename: str | None = None,
        notes: str | None = None,
        project: str | None = None,
        omrat_version: str | None = None,
    ) -> int:
        """Persist a finished run's metadata.  Returns its run_id.

        ``project`` empty / ``None`` leaves the run untagged: it is then
        grouped under its output folder name (``RunMeta.project_label``).
        """
        ts = timestamp or time.strftime('%Y-%m-%d %H:%M:%S')
        totals = totals or {}
        row = {
            'name': name or 'unnamed',
            'timestamp': ts,
            'duration_seconds': duration_seconds,
            'drift_allision': _f(totals.get('drift_allision')),
            'drift_grounding': _f(totals.get('drift_grounding')),
            'drift_anchoring': _f(totals.get('drift_anchoring')),
            'powered_grounding': _f(totals.get('powered_grounding')),
            'powered_allision': _f(totals.get('powered_allision')),
            'head_on': _f(totals.get('head_on')),
            'overtaking': _f(totals.get('overtaking')),
            'crossing': _f(totals.get('crossing')),
            'merging': _f(totals.get('merging')),
            'bend': _f(totals.get('bend')),
            'ship_collision_total': _f(totals.get('ship_collision_total')),
            'output_dir': str(output_dir) if output_dir is not None else None,
            'output_filename': output_filename,
            'notes': notes,
        }
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO omrat_runs (
                    name, timestamp, duration_seconds,
                    drift_allision, drift_grounding, drift_anchoring,
                    powered_grounding, powered_allision,
                    head_on, overtaking, crossing, merging, bend,
                    ship_collision_total,
                    output_dir, output_filename, notes
                ) VALUES (
                    :name, :timestamp, :duration_seconds,
                    :drift_allision, :drift_grounding, :drift_anchoring,
                    :powered_grounding, :powered_allision,
                    :head_on, :overtaking, :crossing, :merging, :bend,
                    :ship_collision_total,
                    :output_dir, :output_filename, :notes
                )""",
                row,
            )
            run_id = int(cursor.lastrowid)
            tag = (project or '').strip() or None
            if tag is not None or omrat_version:
                conn.execute(
                    "INSERT OR REPLACE INTO omrat_run_info (run_id, project, omrat_version) "
                    "VALUES (?, ?, ?)",
                    (run_id, tag, omrat_version or None),
                )
            conn.commit()
        logger.info(f"Saved OMRAT run {run_id} ('{name}') metadata to {self.db_path}")
        return run_id

    # ----------------- reads ----------------- #

    def list_runs(self) -> list[RunMeta]:
        with self._connect() as conn:
            rows = conn.execute(
                _SELECT_RUNS + " ORDER BY r.run_id DESC"
            ).fetchall()
        return [RunMeta(**dict(r)) for r in rows]

    def get_run(self, run_id: int) -> RunMeta | None:
        with self._connect() as conn:
            row = conn.execute(
                _SELECT_RUNS + " WHERE r.run_id = ?", (run_id,),
            ).fetchone()
        return RunMeta(**dict(row)) if row is not None else None

    def compare_runs(self, run_ids: Iterable[int]) -> list[RunMeta]:
        ids = list(run_ids)
        if not ids:
            return []
        # ``placeholders`` is built from a fixed ``?`` token per id; only the
        # number of placeholders varies.  All values are bound through the
        # ``ids`` parameter list, so no user data is interpolated into SQL.
        placeholders = ','.join('?' for _ in ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"{_SELECT_RUNS} WHERE r.run_id IN ({placeholders})",  # nosec B608
                ids,
            ).fetchall()
        by_id = {int(r['run_id']): RunMeta(**dict(r)) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    # ----------------- project / version ----------------- #

    def set_project(self, run_ids: Iterable[int], project: str | None) -> None:
        """Tag ``run_ids`` with ``project``; empty / ``None`` removes the
        tag so the runs fall back to their output folder name."""
        tag = (project or '').strip() or None
        with self._connect() as conn:
            for run_id in run_ids:
                conn.execute(
                    "INSERT INTO omrat_run_info (run_id, project) VALUES (?, ?) "
                    "ON CONFLICT(run_id) DO UPDATE SET project = excluded.project",
                    (int(run_id), tag),
                )
            conn.commit()

    def set_version(self, run_id: int, version: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO omrat_run_info (run_id, omrat_version) VALUES (?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET omrat_version = excluded.omrat_version",
                (int(run_id), version),
            )
            conn.commit()

    def backfill_versions(self, runs: Iterable[RunMeta]) -> int:
        """Fill in the OMRAT version of runs recorded before it was stored.

        Read from the run's ``.omrat`` snapshot (which carries
        ``omrat_version`` since 0.15), stored once and set on the
        ``RunMeta`` too.  A snapshot without the key stores ``''`` so it
        is not read again; a missing snapshot is left for a later try.
        Returns the number of runs updated.
        """
        updated = 0
        for run in runs:
            if run.omrat_version is not None:
                continue
            path = run.snapshot_path()
            if path is None or not path.is_file():
                continue
            version = snapshot_version(path)
            try:
                self.set_version(run.run_id, version)
            except sqlite3.Error as exc:
                logger.warning(f"Could not store the version of run {run.run_id}: {exc}")
                continue
            run.omrat_version = version
            updated += 1
        return updated

    # ----------------- delete ----------------- #

    def delete_run(self, run_id: int, *, delete_gpkg: bool = False) -> None:
        """Remove a run from the master DB.

        When ``delete_gpkg`` is True and the row's per-run GeoPackage
        is present on disk, the file is also removed.  Default is
        False so accidental clicks don't lose the spatial data.
        """
        run = self.get_run(run_id) if delete_gpkg else None
        with self._connect() as conn:
            conn.execute("DELETE FROM omrat_runs WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM omrat_run_info WHERE run_id = ?", (run_id,))
            conn.commit()
        if delete_gpkg and run is not None:
            path = run.gpkg_path()
            if path is not None and path.is_file():
                try:
                    path.unlink()
                    logger.info(f"Deleted per-run GeoPackage {path}")
                except Exception as exc:
                    logger.warning(f"Could not delete {path}: {exc}")
        logger.info(f"Deleted OMRAT run {run_id} from master DB")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def folder_project(output_dir: str | Path | None) -> str:
    """Name of the last folder in ``output_dir`` ('' if none).

    Splits on both separators so a history written on Windows reads the
    same everywhere.
    """
    parts = [p for p in re.split(r'[\\/]', str(output_dir or '')) if p]
    return parts[-1] if parts else ''


_VERSION_RE = re.compile(r'"omrat_version"\s*:\s*"([^"]*)"')


def snapshot_version(path: str | Path, head_bytes: int = 16384) -> str:
    """``omrat_version`` from the start of a ``.omrat`` file, or ''.

    The key is written first (``GatherData.get_all_for_save``), so the
    first few KB are enough -- snapshots are often 10+ MB.
    """
    try:
        with open(path, 'rb') as f:
            head = f.read(head_bytes).decode('utf-8', errors='ignore')
    except OSError:
        return ''
    match = _VERSION_RE.search(head)
    return match.group(1) if match else ''


def project_counts(runs: Iterable[RunMeta]) -> dict[str, int]:
    """``{project label: number of runs}``, sorted by label (case-insensitive)."""
    counts: dict[str, int] = {}
    for run in runs:
        label = run.project_label()
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0].lower()))


def _f(v) -> float | None:
    """Coerce numeric-ish input to ``float`` (or ``None``)."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def totals_from_calc(calc) -> dict[str, float | None]:
    """Extract the standard scalar totals dict from a Calculation object.

    Used by :mod:`omrat.py` and :mod:`omrat_utils.run_persistence` so
    the same key vocabulary is used everywhere.  A model the run did not
    calculate (Calculation matrix) has no report: its totals are
    ``None`` -- an empty history cell, not a zero.
    """
    def _totals(attr: str) -> dict | None:
        report = getattr(calc, attr, None)
        if report is None:
            return None
        return report.get('totals', {}) or {}

    def _get(totals: dict | None, key: str) -> float | None:
        if totals is None:
            return None
        return float(totals.get(key, 0.0) or 0.0)

    drift = _totals('drifting_report')
    coll = _totals('collision_report')
    pg = _totals('powered_grounding_report')
    pa = _totals('powered_allision_report')
    return {
        'drift_allision': _get(drift, 'allision'),
        'drift_grounding': _get(drift, 'grounding'),
        'drift_anchoring': _get(drift, 'anchoring'),
        'powered_grounding': _get(pg, 'grounding'),
        'powered_allision': _get(pa, 'allision'),
        'head_on': _get(coll, 'head_on'),
        'overtaking': _get(coll, 'overtaking'),
        'crossing': _get(coll, 'crossing'),
        'merging': _get(coll, 'merging'),
        'bend': _get(coll, 'bend'),
        'ship_collision_total': _get(coll, 'total'),
    }


def slug(name: str) -> str:
    """File-system-safe form of ``name``.

    Used to build the per-run GeoPackage filename.  Strips characters
    that would be illegal on Windows / mount points and folds
    whitespace to ``_``.
    """
    bad = '<>:"/\\|?*\0'
    out = ''.join('_' if c in bad else c for c in (name or '').strip())
    out = '_'.join(out.split())  # collapse whitespace runs
    return out or 'run'


def make_run_filename(name: str, ts: time.struct_time | None = None) -> str:
    """``<slug>_<YYYYMMDD_HHMMSS>.gpkg``."""
    if ts is None:
        ts = time.localtime()
    stamp = time.strftime('%Y%m%d_%H%M%S', ts)
    return f"{slug(name)}_{stamp}.gpkg"
