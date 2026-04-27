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

    Uses Qt's :class:`QStandardPaths` when available, otherwise falls
    back to ``%APPDATA%`` / ``$XDG_DATA_HOME`` / ``~/.local/share`` so
    the module is importable for tests outside QGIS.

    If a legacy ``omrat_history.gpkg`` (or older ``omrat_runs.gpkg``)
    is found alongside the new path it is auto-renamed in place so a
    user upgrading from an earlier release keeps their run history.
    """
    base = ''
    try:
        from qgis.PyQt.QtCore import QStandardPaths
        # Try Qt 5 flat enum first; fall back to Qt 6 scoped form.
        loc = getattr(QStandardPaths, 'AppDataLocation', None)
        if loc is None:
            loc = QStandardPaths.StandardLocation.AppDataLocation
        base = QStandardPaths.writableLocation(loc) or ''
    except Exception:
        base = ''
    if not base:
        if os.name == 'nt':
            base = os.environ.get('APPDATA', '')
        else:
            base = os.environ.get(
                'XDG_DATA_HOME', str(Path.home() / '.local' / 'share'),
            )
        if not base:
            base = str(Path.home())
    target = Path(base) / 'OMRAT' / 'omrat_history.sqlite'

    # Migrate from the older ``.gpkg`` filename if a previous
    # release left one behind.  No data is lost -- the file format
    # is identical, only the extension changed.
    if not target.exists():
        for legacy_name in ('omrat_history.gpkg', 'omrat_runs.gpkg'):
            legacy = target.parent / legacy_name
            if legacy.exists():
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    legacy.rename(target)
                    logger.info(
                        f"Migrated legacy run-history file {legacy} -> {target}"
                    )
                    break
                except Exception as exc:
                    logger.warning(f"Could not rename {legacy}: {exc}")

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
        bend                REAL,
        ship_collision_total REAL,
        output_dir          TEXT,
        output_filename     TEXT,
        notes               TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_runs_timestamp ON omrat_runs(timestamp)",
]


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
    bend: float | None = None
    ship_collision_total: float | None = None
    output_dir: str | None = None
    output_filename: str | None = None
    notes: str | None = None

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
    ) -> int:
        """Persist a finished run's metadata.  Returns its run_id."""
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
                    head_on, overtaking, crossing, bend, ship_collision_total,
                    output_dir, output_filename, notes
                ) VALUES (
                    :name, :timestamp, :duration_seconds,
                    :drift_allision, :drift_grounding, :drift_anchoring,
                    :powered_grounding, :powered_allision,
                    :head_on, :overtaking, :crossing, :bend, :ship_collision_total,
                    :output_dir, :output_filename, :notes
                )""",
                row,
            )
            run_id = int(cursor.lastrowid)
            conn.commit()
        logger.info(f"Saved OMRAT run {run_id} ('{name}') metadata to {self.db_path}")
        return run_id

    # ----------------- reads ----------------- #

    def list_runs(self) -> list[RunMeta]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM omrat_runs ORDER BY run_id DESC"""
            ).fetchall()
        return [RunMeta(**dict(r)) for r in rows]

    def get_run(self, run_id: int) -> RunMeta | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM omrat_runs WHERE run_id = ?", (run_id,),
            ).fetchone()
        return RunMeta(**dict(row)) if row is not None else None

    def compare_runs(self, run_ids: Iterable[int]) -> list[RunMeta]:
        ids = list(run_ids)
        if not ids:
            return []
        placeholders = ','.join('?' for _ in ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM omrat_runs WHERE run_id IN ({placeholders})",
                ids,
            ).fetchall()
        by_id = {int(r['run_id']): RunMeta(**dict(r)) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

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

def _f(v) -> float | None:
    """Coerce numeric-ish input to ``float`` (or ``None``)."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def totals_from_calc(calc) -> dict[str, float]:
    """Extract the standard scalar totals dict from a Calculation object.

    Used by :mod:`omrat.py` and :mod:`omrat_utils.run_persistence` so
    the same key vocabulary is used everywhere.
    """
    drift = (getattr(calc, 'drifting_report', None) or {}).get('totals', {}) or {}
    coll = (getattr(calc, 'collision_report', None) or {}).get('totals', {}) or {}
    pg = (getattr(calc, 'powered_grounding_report', None) or {}).get('totals', {}) or {}
    pa = (getattr(calc, 'powered_allision_report', None) or {}).get('totals', {}) or {}
    return {
        'drift_allision': float(drift.get('allision', 0.0) or 0.0),
        'drift_grounding': float(drift.get('grounding', 0.0) or 0.0),
        'drift_anchoring': float(drift.get('anchoring', 0.0) or 0.0),
        'powered_grounding': float(pg.get('grounding', 0.0) or 0.0),
        'powered_allision': float(pa.get('allision', 0.0) or 0.0),
        'head_on': float(coll.get('head_on', 0.0) or 0.0),
        'overtaking': float(coll.get('overtaking', 0.0) or 0.0),
        'crossing': float(coll.get('crossing', 0.0) or 0.0),
        'bend': float(coll.get('bend', 0.0) or 0.0),
        'ship_collision_total': float(coll.get('total', 0.0) or 0.0),
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
