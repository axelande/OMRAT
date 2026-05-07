"""AIS ingestion pipeline: raw NMEA / aisdb-CSV → TDKC linestring segments.

Architecture
------------
:class:`IngestionPipeline` is **headless** and Qt-free — it owns the full
pipeline so the worker logic stays unit-testable without a running QGIS.

:class:`IngestionWorker` is a thin :class:`~qgis.PyQt.QtCore.QThread`
wrapper that runs the pipeline off the UI thread and forwards progress /
log signals.  It is imported lazily by the wizard so headless callers
never pay the Qt cost.

Pipeline steps
--------------
1. Decode every input file into a per-run temp SQLite via ``aisdb.decode_msgs``.
2. Open a fresh ``SQLiteDBConn`` so aisdb's cached ``db_daterange`` is correct.
3. Pull dynamic tracks via ``aisdb.TrackGen`` and the matching static rows
   directly from the ``ais_YYYYMM_static`` table.
4. For each MMSI:
   - upsert one ``{schema}.statics_{year}`` row → ``static_id``;
   - insert one ``{schema}.states_{year}`` row covering ``[min(t), max(t)]``
     for that MMSI in this ingestion run → ``state_id``;
   - run ``aissegments.tdkc_segments(track, **settings.to_aissegments_kwargs())``;
   - bulk-insert the resulting linestrings into the right monthly partition
     of ``{schema}.segments_{year}`` with the ``state_id`` attached.
5. Update ``omrat_meta.segment_watermark`` with each MMSI's last-seen time.

This *first cut* assigns one static + one state per MMSI per ingestion run.
Vessels that change registration mid-year get a less precise mapping; the
schema supports proper time-windowed linkage and a future iteration can
populate that without changing the wire format.
"""
from __future__ import annotations

import contextlib
import gc
import logging
import re
import tempfile
import time as time_mod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import psycopg2
from psycopg2 import sql as psql
from psycopg2.extras import execute_values

from omrat_utils.db_setup import (
    ConnectionProfile,
    IngestionSettings,
    Migrator,
)

_LOG = logging.getLogger(__name__)

_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_sqlite_ident(name: str) -> str:
    """Validate ``name`` as a plain SQL identifier and return it double-quoted.

    SQLite has no parameter binding for identifiers; the only safe way to
    interpolate a table or column name is to whitelist its characters and
    wrap it in double quotes (per the SQLite grammar).  Raises
    ``ValueError`` for anything that isn't ``[A-Za-z_][A-Za-z0-9_]*``.
    """
    if not isinstance(name, str) or not _SQL_IDENT_RE.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return f'"{name}"'

# WGS84 mean radius (metres) — same value used inside aissegments.tdkc.
_EARTH_RADIUS_M = 6_371_008.8

# Conversion: 1 knot = 0.514444 m/s
_MPS_TO_KN = 1.0 / 0.514444


def _haversine_pairwise_m(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Great-circle distance (m) between consecutive points of two arrays.

    Returns an array of length ``len(lon) - 1``.  Empty input yields an
    empty array.
    """
    if len(lon) < 2:
        return np.zeros(0)
    lon1 = np.deg2rad(lon[:-1])
    lat1 = np.deg2rad(lat[:-1])
    lon2 = np.deg2rad(lon[1:])
    lat2 = np.deg2rad(lat[1:])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def split_track_at_gaps(
    track,
    *,
    max_gap_s: float,
    speed_tolerance: float,
    speed_floor_kn: float,
):
    """Split a track wherever a coverage gap or implausible jump is detected.

    A break is inserted between consecutive pings ``i`` and ``i+1`` when
    *either*:

    - ``t[i+1] - t[i] > max_gap_s`` — receiver dropout, or
    - the *implied* speed (``haversine(p_i, p_{i+1}) / dt`` in knots)
      exceeds ``avg_sog * (1 + speed_tolerance) + speed_floor_kn``,
      where ``avg_sog = (sog[i] + sog[i+1]) / 2``.

    The check is one-sided (``implied > limit``): a vessel decelerating
    *below* its reported SOG is plausible (slowing for a port, anchoring),
    only physically-impossible *speed-ups* indicate a teleport.

    Tracks shorter than 2 points are returned as a single-element list.

    Parameters
    ----------
    track : aissegments.Track
        Input track for one MMSI.
    max_gap_s : float
        Maximum allowed inter-ping time gap, in seconds.
    speed_tolerance : float
        Allowed fractional excess over the average reported SOG.
    speed_floor_kn : float
        Additive slack (knots) on top of the percentage limit, to absorb
        GPS jitter for slow / moored vessels.

    Returns
    -------
    list[aissegments.Track]
        One or more sub-tracks covering the original points contiguously.
        ``sum(len(s) for s in out) == len(track)`` always holds.
    """
    n = len(track)
    if n < 2:
        return [track]

    t = np.asarray(track.t, dtype=np.float64)
    lon = np.asarray(track.lon, dtype=np.float64)
    lat = np.asarray(track.lat, dtype=np.float64)
    sog = np.asarray(track.sog, dtype=np.float64)

    dt = np.diff(t)
    # Gap on time axis.
    gap_time = dt > float(max_gap_s)

    # Implied speed in knots; guard against duplicate timestamps (dt=0).
    dist_m = _haversine_pairwise_m(lon, lat)
    safe_dt = np.where(dt > 0.0, dt, np.inf)
    implied_kn = (dist_m / safe_dt) * _MPS_TO_KN

    avg_sog = (sog[:-1] + sog[1:]) / 2.0
    limit_kn = avg_sog * (1.0 + float(speed_tolerance)) + float(speed_floor_kn)
    gap_speed = implied_kn > limit_kn

    gap = gap_time | gap_speed
    if not gap.any():
        return [track]

    # Split indices: gap[i] True means break between i and i+1.
    break_after = np.where(gap)[0]
    sub_tracks = []
    start = 0
    for i in break_after:
        end_excl = int(i) + 1
        sub_tracks.append(track.take(np.arange(start, end_excl).tolist()))
        start = end_excl
    sub_tracks.append(track.take(np.arange(start, n).tolist()))
    return sub_tracks


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    n_files: int = 0
    n_tracks: int = 0
    n_segments: int = 0
    n_static_rows: int = 0          # newly INSERTed static rows
    n_static_reused: int = 0        # static lookups satisfied from existing data
    n_state_rows: int = 0           # newly INSERTed state rows
    n_state_reused: int = 0         # state lookups satisfied from existing data
    n_tracks_skipped_watermark: int = 0
    n_tracks_skipped_year: int = 0  # tracks dropped because no ping fell in target year
    n_track_splits: int = 0         # extra sub-tracks produced by the gap splitter
    cancelled: bool = False         # True if the user requested cancellation mid-run
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        skip = (
            f", {self.n_tracks_skipped_watermark} skipped (watermark)"
            if self.n_tracks_skipped_watermark
            else ""
        )
        skip_year = (
            f", {self.n_tracks_skipped_year} skipped (outside target year)"
            if self.n_tracks_skipped_year
            else ""
        )
        reused = (
            f", reused {self.n_static_reused} static + {self.n_state_reused} state"
            if (self.n_static_reused or self.n_state_reused)
            else ""
        )
        splits = (
            f", {self.n_track_splits} gap-splits"
            if self.n_track_splits
            else ""
        )
        prefix = "Cancelled after ingesting" if self.cancelled else "Ingested"
        return (
            f"{prefix} {self.n_files} file(s) → "
            f"{self.n_tracks} tracks, "
            f"{self.n_segments} segments, "
            f"{self.n_static_rows} static, "
            f"{self.n_state_rows} state rows"
            f"{reused}{splits}{skip}{skip_year} in {self.elapsed_seconds:.1f}s"
            + (f"  ({len(self.errors)} errors)" if self.errors else "")
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IngestionPipeline:
    """Decode AIS files, compress with TDKC, insert into PostGIS.

    Parameters
    ----------
    profile : ConnectionProfile
        Connection details for the OMRAT Postgres backend.
    settings : IngestionSettings
        TDKC threshold floors (``min_sed_m``, ``min_svd_kn``).
    progress_cb : callable, optional
        Invoked with a single human-readable string at each notable step
        (file decoded, MMSI batch finished, etc.).  Use this from a Qt
        worker to forward to the UI; ignored otherwise.
    """

    def __init__(
        self,
        profile: ConnectionProfile,
        settings: IngestionSettings,
        progress_cb: Callable[[str], None] | None = None,
    ) -> None:
        self.profile = profile
        self.settings = settings
        self._progress_cb = progress_cb
        # Dedup caches: ``mmsi → (rowid, identity_tuple)``.  Populated lazily
        # by ``_load_latest_statics`` / ``_load_latest_states`` at the start
        # of each ingestion path, then maintained in-memory as we insert.
        self._statics_cache: dict[int, tuple[int, tuple]] = {}
        self._states_cache: dict[int, tuple[int, tuple]] = {}
        # Cooperative cancellation: a worker thread can call ``cancel()`` at
        # any time; the per-track loops in both ingestion paths check
        # ``_is_cancelled()`` and stop after the current track commits.
        self._cancelled = False

    def cancel(self) -> None:
        """Request a clean stop after the current track.

        Safe to call from any thread.  The pipeline will commit the
        current batch and exit; the per-MMSI watermark already in the DB
        means a follow-up run with the same files picks up exactly where
        this one left off.
        """
        self._cancelled = True

    def _is_cancelled(self) -> bool:
        return self._cancelled

    # ------------------------------------------------------------ public API

    def run(
        self,
        files: Iterable[str | Path],
        *,
        year: int,
        source_tag: str = "OMRAT",
        create_indexes_after: bool = True,
        incremental: bool = True,
    ) -> IngestionResult:
        """End-to-end ingestion for one calendar year of AIS data.

        Files are auto-classified by extension + header sniff:

        - ``*.nm4`` / ``*.nmea`` → routed through ``aisdb.decode_msgs`` (the
          full Type-1/Type-5 decoder; populates statics + states from
          static-AIS messages).
        - ``*.csv`` / ``*.csv.gz`` whose header looks like aisdb's own
          AIS-message dump (``Message_ID``, ``Repeat_indicator``) → also
          routed through aisdb.
        - ``*.csv`` / ``*.csv.gz`` with a generic AIS header (Marine
          Cadastre's ``BaseDateTime`` / ``LAT`` / ``LON``, or our own
          ``time``/``lon``/``lat`` schema) → routed directly through
          ``aissegments.read_csv_tracks``.  Static info is unavailable
          from this format, so ``statics_YYYY`` rows have NULL identity
          fields and ``states_YYYY`` rows have NULL voyage fields.
          Downstream AIS queries derive loa/beam directly from
          ``dim_a + dim_b`` / ``dim_c + dim_d`` (AIS Type-5) and use the
          ``type_and_cargo`` column for ship-type classification.

        Parameters
        ----------
        files : iterable of path
        year : int
            Target year — picks the matching ``{schema}.segments_YYYY``
            partition family.  Pings outside ``[year, year+1)`` are skipped.
        source_tag : str
            Label written to aisdb's decoded rows; ignored for simple-CSV
            input.  Default ``"OMRAT"``.
        create_indexes_after : bool
            When True (default), call :meth:`Migrator.create_year_indexes`
            once ingestion finishes.  Set to False if you plan to ingest
            more batches into the same year before indexing.
        incremental : bool
            When True (default), the pipeline reads
            ``omrat_meta.segment_watermark`` for each MMSI and skips pings
            with ``time <= last_t`` so re-running on overlapping data is
            cheap and idempotent.  Set to False to force re-processing
            (after wiping the relevant ``segments_YYYY_M`` partitions).
        """
        # Allow re-using one pipeline across runs: clear any cancel flag
        # left over from a previous run before doing anything else (so
        # even the empty-files early return doesn't leave a stale flag).
        self._cancelled = False
        files = [Path(f) for f in files]
        result = IngestionResult(n_files=len(files))
        if not files:
            result.errors.append("No input files")
            return result

        t0 = time_mod.monotonic()
        migrator = Migrator(self.profile)
        migrator.ensure_year_partition(year)
        self._log(f"Year-{year} tables ready in schema {self.profile.schema!r}")
        # Reset dedup caches for this run.  They get repopulated from the DB
        # at the start of each ingestion path so cross-run dedup works too.
        self._statics_cache = {}
        self._states_cache = {}

        aisdb_files: list[Path] = []
        simple_csv_files: list[Path] = []
        for f in files:
            try:
                fmt = self._detect_format(f)
            except Exception as e:
                result.errors.append(f"format detect {f.name}: {e}")
                continue
            if fmt == "simple_csv":
                simple_csv_files.append(f)
            else:
                aisdb_files.append(f)
        self._log(
            f"Format split: {len(aisdb_files)} aisdb, {len(simple_csv_files)} simple-CSV"
        )

        if aisdb_files:
            self._ingest_aisdb_files(aisdb_files, year, source_tag, result, incremental)
        if simple_csv_files and not self._is_cancelled():
            self._ingest_simple_csv_files(simple_csv_files, year, result, incremental)

        if self._is_cancelled():
            result.cancelled = True
            self._log("Cancelled by user — committed batches are preserved.")

        # Build the heavy segment indexes once on populated data.
        if create_indexes_after and result.n_segments > 0:
            try:
                self._log(f"Creating segments_{year} indexes (GiST + btree)...")
                migrator.create_year_indexes(year)
                self._log("Indexes created and statistics refreshed")
            except Exception as e:
                result.errors.append(f"create_year_indexes({year}) failed: {e}")

        result.elapsed_seconds = time_mod.monotonic() - t0
        self._log(result.summary())
        return result

    # -- format detection + dispatch ----------------------------------------

    @staticmethod
    def _detect_format(path: Path) -> str:
        """Return ``"aisdb_nmea"``, ``"aisdb_csv"``, or ``"simple_csv"``."""
        suffix = path.suffix.lower()
        if suffix in (".nm4", ".nmea"):
            return "aisdb_nmea"
        # Pull the header line.
        if suffix == ".gz":
            import gzip
            opener = gzip.open(path, "rt", encoding="utf-8")
        else:
            opener = path.open("r", encoding="utf-8")
        with opener as f:
            header = f.readline()
        cols_lower = {c.strip().lower() for c in header.split(",")}
        # aisdb's own CSV dump has these distinctive columns.
        if "message_id" in cols_lower or "repeat_indicator" in cols_lower:
            return "aisdb_csv"
        return "simple_csv"

    def _ingest_aisdb_files(
        self,
        files: list[Path],
        year: int,
        source_tag: str,
        result: IngestionResult,
        incremental: bool,
    ) -> None:
        try:
            tmpdir, sqlite_path = self._decode_to_sqlite(files, source_tag)
        except Exception as e:  # pragma: no cover - aisdb itself errors
            result.errors.append(f"decode_msgs failed: {e}")
            return
        try:
            statics_by_mmsi = self._read_static_rows(sqlite_path, year)
            self._log(f"Loaded {len(statics_by_mmsi)} static rows from decoded data")
            total_estimate = self._count_aisdb_distinct_mmsis(sqlite_path, year)
            self._log(
                f"Estimated {total_estimate} distinct MMSI(s) to process"
            )
            with psycopg2.connect(**self.profile.to_dsn()) as conn:
                with conn.cursor() as cur:
                    self._load_latest_statics(cur, self.profile.schema, year)
                    self._load_latest_states(cur, self.profile.schema, year)
                    self._log(
                        f"Dedup caches: {len(self._statics_cache)} statics, "
                        f"{len(self._states_cache)} states pre-loaded"
                    )
                    watermarks = self._load_watermarks(cur) if incremental else {}
                    if incremental:
                        self._log(
                            f"Incremental mode: {len(watermarks)} MMSIs have a watermark"
                        )
                    n_in_batch = 0
                    n_visited = 0
                    t_loop = time_mod.monotonic()
                    for raw in self._iter_tracks(sqlite_path, year):
                        if self._is_cancelled():
                            conn.commit()
                            return
                        n_visited += 1
                        try:
                            from aissegments.adapters import from_aisdb_track
                            track = from_aisdb_track(raw)
                        except Exception as e:
                            result.errors.append(
                                f"adapter mmsi={raw.get('mmsi')}: {e}"
                            )
                            continue
                        track = self._filter_after_watermark(
                            track, watermarks.get(int(raw["mmsi"]))
                        )
                        if track is None:
                            result.n_tracks_skipped_watermark += 1
                            continue
                        rec = statics_by_mmsi.get(int(raw["mmsi"]))
                        self._ingest_one_track(cur, track, rec, year, result)
                        n_in_batch += 1
                        if n_in_batch % 200 == 0:
                            self._log(
                                f"Progress: "
                                f"{self._format_progress(n_visited, total_estimate, t_loop)}"
                                f" — {result.n_segments} segments so far"
                            )
                            conn.commit()
                conn.commit()
        finally:
            gc.collect()
            with contextlib.suppress(Exception):
                tmpdir.cleanup()

    def _ingest_simple_csv_files(
        self,
        files: list[Path],
        year: int,
        result: IngestionResult,
        incremental: bool,
    ) -> None:
        from aissegments import read_csv_static_records, read_csv_tracks

        # Pre-scan every input file for static data (Marine Cadastre carries
        # VesselType / Length / Width / Draft / IMO; bare custom CSVs may
        # not).  Latest record wins per MMSI when files overlap.
        # NOTE: simple-CSV files are scanned twice (once here for statics,
        # once below for kinematics).  On large Marine-Cadastre dumps each
        # pass takes minutes — we log per-file timing so the UI doesn't
        # appear stuck during the silent pre-pass.
        self._log(
            f"Pre-scanning {len(files)} simple-CSV file(s) for static info "
            f"(this is a full file pass; large files may take minutes)..."
        )
        statics_by_mmsi: dict[int, dict[str, Any]] = {}
        for idx, f in enumerate(files, start=1):
            if self._is_cancelled():
                return
            t_pre = time_mod.monotonic()
            try:
                file_statics = read_csv_static_records(f)
            except Exception as e:
                result.errors.append(f"read_csv_static_records {f.name}: {e}")
                self._log(
                    f"  [{idx}/{len(files)}] static-scan {f.name} FAILED: {e}"
                )
                continue
            self._log(
                f"  [{idx}/{len(files)}] static-scan {f.name}: "
                f"{len(file_statics)} MMSIs in {time_mod.monotonic() - t_pre:.1f}s"
            )
            for mmsi, rec in file_statics.items():
                prev = statics_by_mmsi.get(mmsi)
                if prev is None or rec.get("time", 0) >= prev.get("time", 0):
                    statics_by_mmsi[mmsi] = rec
        if statics_by_mmsi:
            self._log(
                f"Extracted static records for {len(statics_by_mmsi)} MMSIs "
                f"from simple-CSV input"
            )

        # Merge kinematic data across all input files so one MMSI's pings
        # spanning multiple files become a single time-sorted track.  Without
        # this step, the watermark would silently drop pings from any file
        # processed *after* a later-timestamped file in the same run.
        self._log(
            f"Reading {len(files)} simple-CSV file(s) for kinematic data..."
        )
        per_mmsi_tracks: dict[int, list] = defaultdict(list)
        for idx, f in enumerate(files, start=1):
            if self._is_cancelled():
                return
            t_read = time_mod.monotonic()
            try:
                file_tracks = read_csv_tracks(f)
            except Exception as e:
                result.errors.append(f"read_csv_tracks {f.name}: {e}")
                self._log(
                    f"  [{idx}/{len(files)}] read {f.name} FAILED: {e}"
                )
                continue
            n_pings = sum(len(tr) for tr in file_tracks)
            self._log(
                f"  [{idx}/{len(files)}] read {f.name}: "
                f"{len(file_tracks)} vessels / {n_pings} pings "
                f"in {time_mod.monotonic() - t_read:.1f}s"
            )
            for tr in file_tracks:
                per_mmsi_tracks[int(tr.mmsi)].append(tr)

        merged_tracks = list(self._merge_simple_csv_tracks(per_mmsi_tracks))
        if files:
            self._log(
                f"Merged into {len(merged_tracks)} per-MMSI track(s) "
                f"across {len(files)} file(s)"
            )

        with psycopg2.connect(**self.profile.to_dsn()) as conn:
            with conn.cursor() as cur:
                self._load_latest_statics(cur, self.profile.schema, year)
                self._load_latest_states(cur, self.profile.schema, year)
                self._log(
                    f"Dedup caches: {len(self._statics_cache)} statics, "
                    f"{len(self._states_cache)} states pre-loaded"
                )
                watermarks = self._load_watermarks(cur) if incremental else {}
                if incremental:
                    self._log(
                        f"Incremental mode: {len(watermarks)} MMSIs have a watermark"
                    )
                n_in_batch = 0
                n_visited = 0
                total = len(merged_tracks)
                t_loop = time_mod.monotonic()
                for track in merged_tracks:
                    if self._is_cancelled():
                        conn.commit()
                        return
                    n_visited += 1
                    in_year = self._filter_to_year(track, year)
                    if in_year is None:
                        result.n_tracks_skipped_year += 1
                        continue
                    in_year = self._filter_after_watermark(
                        in_year, watermarks.get(int(track.mmsi))
                    )
                    if in_year is None:
                        result.n_tracks_skipped_watermark += 1
                        continue
                    rec = statics_by_mmsi.get(int(track.mmsi))
                    self._ingest_one_track(cur, in_year, rec, year, result)
                    n_in_batch += 1
                    if n_in_batch % 200 == 0:
                        self._log(
                            f"Progress: "
                            f"{self._format_progress(n_visited, total, t_loop)}"
                            f" — {result.n_segments} segments so far"
                        )
                        conn.commit()
            conn.commit()

    @staticmethod
    def _merge_simple_csv_tracks(per_mmsi: dict[int, list]):
        """Concatenate all per-file tracks for the same MMSI into one track.

        Yields one merged :class:`aissegments.Track` per MMSI.  Points are
        time-sorted (stable) and deduplicated on identical timestamps so
        overlapping files don't produce duplicate pings.
        """
        from aissegments import Track

        for mmsi, tracks in per_mmsi.items():
            if not tracks:
                continue
            if len(tracks) == 1 and len(tracks[0]) >= 2:
                # Common case: single file → already time-sorted by
                # ``read_csv_tracks``.  Pass through unchanged.
                yield tracks[0]
                continue
            t = np.concatenate([np.asarray(tr.t, dtype=np.float64) for tr in tracks])
            lon = np.concatenate([np.asarray(tr.lon, dtype=np.float64) for tr in tracks])
            lat = np.concatenate([np.asarray(tr.lat, dtype=np.float64) for tr in tracks])
            sog = np.concatenate([np.asarray(tr.sog, dtype=np.float64) for tr in tracks])
            cog = np.concatenate([np.asarray(tr.cog, dtype=np.float64) for tr in tracks])
            order = np.argsort(t, kind="stable")
            t, lon, lat, sog, cog = (
                t[order], lon[order], lat[order], sog[order], cog[order]
            )
            if len(t) > 1:
                # Drop runs of identical timestamps (keep the first occurrence
                # to give a stable, deterministic merge).
                keep = np.concatenate([[True], np.diff(t) > 0])
                if not keep.all():
                    t, lon, lat, sog, cog = (
                        t[keep], lon[keep], lat[keep], sog[keep], cog[keep]
                    )
            yield Track(mmsi=mmsi, t=t, lon=lon, lat=lat, sog=sog, cog=cog)

    # -- watermark helpers ---------------------------------------------------

    @staticmethod
    def _load_watermarks(cur) -> dict[int, float]:
        """Return ``{mmsi: last_t_unix_seconds}`` for every MMSI in the watermark table.

        Used in incremental mode to skip pings already represented in the
        segments table.  Empty dict if the table is empty (or missing).
        """
        try:
            cur.execute("SELECT mmsi, last_t FROM omrat_meta.segment_watermark")
        except Exception:
            return {}
        out: dict[int, float] = {}
        for row in cur.fetchall():
            mmsi_val, last_t = row[0], row[1]
            if mmsi_val is None or last_t is None:
                continue
            out[int(mmsi_val)] = last_t.timestamp()
        return out

    @staticmethod
    def _filter_after_watermark(track, last_t_unix: float | None):
        """Return ``track`` restricted to pings strictly after ``last_t_unix``.

        Returns the original track unchanged when there is no watermark
        (``last_t_unix is None``) or the whole track is newer than the
        watermark; ``None`` when nothing remains after filtering.
        """
        if last_t_unix is None:
            return track
        import numpy as np

        mask = track.t > float(last_t_unix)
        if not mask.any():
            return None
        if mask.all():
            return track
        return track.take(np.where(mask)[0].tolist())

    @staticmethod
    def _filter_to_year(track, year: int):
        """Return ``track`` restricted to ``[year, year+1)`` UTC, or None."""
        import numpy as np

        start = datetime(year, 1, 1, tzinfo=timezone.utc).timestamp()
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp()
        mask = (track.t >= start) & (track.t < end)
        if not mask.any():
            return None
        if mask.all():
            return track
        return track.take(np.where(mask)[0].tolist())

    # --------------------------------------------------------------- helpers

    def _log(self, msg: str) -> None:
        _LOG.info(msg)
        if self._progress_cb is not None:
            self._progress_cb(msg)

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        """Format a duration in seconds as ``"Hh Mm"``, ``"Mm Ss"``, or ``"Ss"``."""
        if seconds <= 0 or not (seconds == seconds):  # NaN guard
            return "0s"
        s_int = int(round(seconds))
        h, rem = divmod(s_int, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m}m"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    @classmethod
    def _format_progress(
        cls, processed: int, total: int, t_loop_start: float
    ) -> str:
        """Build a one-liner like ``"1234/9876 (12.5%), 412 tr/s, ETA 21s"``.

        ``processed`` is the count of tracks the loop has visited so far
        (including ones skipped by year/watermark filters — anything we
        looked at counts).  ``total`` is the upfront upper bound; if it
        is unknown or zero we omit the % / ETA columns since they would
        be meaningless.

        ``t_loop_start`` is the ``time_mod.monotonic()`` reading taken
        immediately before the loop began — *not* the overall pipeline
        start.  Pre-loop file reading time would otherwise pollute the
        rate.
        """
        elapsed = time_mod.monotonic() - t_loop_start
        if total <= 0:
            return f"{processed} tracks"
        if processed <= 0 or elapsed <= 0.0:
            return f"0/{total} (warming up)"
        rate = processed / elapsed
        remaining = max(total - processed, 0)
        eta_s = remaining / rate if rate > 0 else 0.0
        pct = 100.0 * processed / max(total, 1)
        return (
            f"{processed}/{total} ({pct:.1f}%), "
            f"{rate:.0f} tr/s, ETA {cls._fmt_duration(eta_s)}"
        )

    def _decode_to_sqlite(
        self, files: list[Path], source_tag: str
    ) -> tuple[tempfile.TemporaryDirectory, Path]:
        """Decode all files into a fresh per-run SQLite DB; return (dir, dbpath)."""
        import aisdb

        tmpdir = tempfile.TemporaryDirectory(prefix="omrat_ingest_")
        sqlite_path = Path(tmpdir.name) / "ingest.db"
        with aisdb.SQLiteDBConn(str(sqlite_path)) as dbconn:
            aisdb.decode_msgs(
                [str(f) for f in files],
                dbconn=dbconn,
                source=source_tag,
                verbose=False,
            )
        self._log(f"Decoded {len(files)} file(s) → {sqlite_path.name}")
        return tmpdir, sqlite_path

    def _iter_tracks(self, sqlite_path: Path, year: int):
        """Yield ``aisdb`` track dicts (one per MMSI) for the target year."""
        import aisdb
        from aisdb.database import sqlfcn_callbacks

        with aisdb.SQLiteDBConn(str(sqlite_path)) as dbconn:
            q = aisdb.DBQuery(
                callback=sqlfcn_callbacks.in_timerange_validmmsi,
                dbconn=dbconn,
                start=datetime(year, 1, 1),
                end=datetime(year + 1, 1, 1),
            )
            yield from aisdb.TrackGen(q.gen_qry(), decimate=False)

    def _count_aisdb_distinct_mmsis(self, sqlite_path: Path, year: int) -> int:
        """Estimate ``TrackGen``'s output cardinality by counting distinct MMSIs.

        Reads ``ais_YYYYMM_dynamic`` for every month of the target year
        (skipping tables that don't exist) and returns the size of the
        union of ``mmsi`` columns.  Used purely to drive the % / ETA
        progress display, so a slight over-count (e.g. invalid MMSIs
        that ``in_timerange_validmmsi`` will later filter out) is fine —
        the estimate just settles in a few percent low as the loop runs.
        Returns 0 when no monthly table is present.
        """
        import sqlite3

        seen: set[int] = set()
        with contextlib.closing(sqlite3.connect(str(sqlite_path))) as con:
            cur = con.cursor()
            for month in range(1, 13):
                tbl = _quote_sqlite_ident(f"ais_{int(year)}{month:02d}_dynamic")
                try:
                    cur.execute(f"SELECT DISTINCT mmsi FROM {tbl}")  # nosec B608
                except sqlite3.OperationalError:
                    continue
                for row in cur:
                    if row[0] is not None:
                        seen.add(int(row[0]))
        return len(seen)

    def _read_static_rows(
        self, sqlite_path: Path, year: int
    ) -> dict[int, dict[str, Any]]:
        """Return ``mmsi → static-row dict`` (latest record per MMSI)."""
        import sqlite3

        out: dict[int, dict[str, Any]] = {}
        with contextlib.closing(sqlite3.connect(str(sqlite_path))) as con:
            cur = con.cursor()
            for month in range(1, 13):
                tbl = _quote_sqlite_ident(f"ais_{int(year)}{month:02d}_static")
                try:
                    cur.execute(f"SELECT * FROM {tbl}")  # nosec B608
                except sqlite3.OperationalError:
                    continue
                cols = [d[0] for d in cur.description]
                for row in cur.fetchall():
                    rec = dict(zip(cols, row, strict=True))
                    mmsi_val = rec.get("mmsi")
                    if mmsi_val is None:
                        continue
                    mmsi = int(mmsi_val)
                    # Keep the row with the latest ``time`` per MMSI.
                    prev = out.get(mmsi)
                    if prev is None or (rec.get("time") or 0) >= (prev.get("time") or 0):
                        out[mmsi] = rec
        return out

    def _ingest_one_track(
        self,
        cur,
        track,
        static_rec: dict[str, Any] | None,
        year: int,
        result: IngestionResult,
    ) -> None:
        """Insert one (statics?, state, N segments) bundle and bump counters.

        Statics and states are deduplicated against the most recent existing
        row for this MMSI — if the identity tuple (dimensions+IMO) or
        voyage tuple (draught+type+destination+...) matches what we
        already have, the existing rowid is reused and no INSERT runs.

        The track is first run through :func:`split_track_at_gaps` so that
        coverage dropouts and physically-impossible jumps become independent
        sub-tracks (each TDKC-compressed on its own); without this, TDKC
        would happily emit a single straight-line segment spanning the gap.

        Skips tracks shorter than 2 points.  TDKC errors are recorded in
        ``result.errors`` and do not abort the run.
        """
        from aissegments import tdkc_segments

        if len(track) < 2:
            return
        mmsi = int(track.mmsi)
        result.n_tracks += 1

        static_id = self._ensure_static(
            cur, self.profile.schema, year, mmsi, static_rec, result
        )
        state_id = self._ensure_state(
            cur, self.profile.schema, year, mmsi, static_id, static_rec,
            fallback_t=float(track.t[0]),
            result=result,
        )

        sub_tracks = split_track_at_gaps(track, **self.settings.to_splitter_kwargs())
        if len(sub_tracks) > 1:
            result.n_track_splits += len(sub_tracks) - 1

        all_segments: list = []
        for sub in sub_tracks:
            if len(sub) < 2:
                continue
            try:
                all_segments.extend(
                    tdkc_segments(sub, **self.settings.to_aissegments_kwargs())
                )
            except Exception as e:
                result.errors.append(f"tdkc mmsi={mmsi}: {e}")

        if all_segments:
            self._bulk_insert_segments(
                cur, self.profile.schema, year, mmsi, state_id, all_segments
            )
        result.n_segments += len(all_segments)
        self._update_watermark(cur, mmsi, float(track.t[-1]), len(all_segments))

    # -- dedup logic for statics + states ------------------------------------

    @staticmethod
    def _static_identity_tuple(rec: dict[str, Any] | None) -> tuple:
        """Build the identity tuple used for ``statics_YYYY`` deduplication."""
        if rec is None:
            return (None, None, None, None, None)
        return (
            IngestionPipeline._to_smallint(rec.get("dim_bow")),
            IngestionPipeline._to_smallint(rec.get("dim_stern")),
            IngestionPipeline._to_smallint(rec.get("dim_port")),
            IngestionPipeline._to_smallint(rec.get("dim_star")),
            rec.get("imo"),
        )

    @staticmethod
    def _state_voyage_tuple(
        rec: dict[str, Any] | None,
        static_id: int | None,
        *,
        year_hint: int | None = None,
    ) -> tuple:
        """Build the voyage tuple used for ``states_YYYY`` deduplication.

        When ``year_hint`` is given, the AIS ``eta_*`` fields are combined
        into a timestamptz and included in the comparison so an ETA change
        (alongside an unchanged draught/destination) still triggers a new
        row.
        """
        if rec is None:
            return (None, None, None, None, static_id)
        destination = rec.get("destination")
        if isinstance(destination, str):
            destination = destination.strip()[:20]
        eta = (
            IngestionPipeline._combine_eta(rec, year_hint)
            if year_hint is not None
            else None
        )
        return (
            rec.get("draught"),
            rec.get("ship_type"),
            destination,
            eta,
            static_id,
        )

    def _load_latest_statics(self, cur, schema: str, year: int) -> None:
        """Populate ``_statics_cache`` with the most recent statics row per MMSI."""
        try:
            cur.execute(
                psql.SQL(
                    """
                    SELECT DISTINCT ON (mmsi)
                        mmsi, rowid, dim_a, dim_b, dim_c, dim_d, imo_num
                    FROM {schema}.{table}
                    WHERE mmsi IS NOT NULL
                    ORDER BY mmsi, rowid DESC
                    """
                ).format(
                    schema=psql.Identifier(schema),
                    table=psql.Identifier(f"statics_{int(year)}"),
                )
            )
        except Exception:
            return
        for row in cur.fetchall():
            mmsi = int(row[0])
            rowid = int(row[1])
            self._statics_cache[mmsi] = (rowid, (row[2], row[3], row[4], row[5], row[6]))

    def _load_latest_states(self, cur, schema: str, year: int) -> None:
        """Populate ``_states_cache`` with the most recent states row per MMSI."""
        try:
            cur.execute(
                psql.SQL(
                    """
                    SELECT DISTINCT ON (mmsi)
                        mmsi, rowid, draught, type_and_cargo, destination, eta, static_id
                    FROM {schema}.{table}
                    WHERE mmsi IS NOT NULL
                    ORDER BY mmsi, rowid DESC
                    """
                ).format(
                    schema=psql.Identifier(schema),
                    table=psql.Identifier(f"states_{int(year)}"),
                )
            )
        except Exception:
            return
        for row in cur.fetchall():
            mmsi = int(row[0])
            rowid = int(row[1])
            destination = row[4]
            if isinstance(destination, str):
                destination = destination.strip()[:20]
            tup = (row[2], row[3], destination, row[5], row[6])
            self._states_cache[mmsi] = (rowid, tup)

    def _ensure_static(
        self,
        cur,
        schema: str,
        year: int,
        mmsi: int,
        rec: dict[str, Any] | None,
        result: IngestionResult,
    ) -> int:
        """Return a ``static_id``, inserting only when the data is new.

        - ``rec is None`` and the cache has an entry → reuse the cached id.
        - ``rec is None`` and no cache entry → insert a NULL-fields
          placeholder so ``state.static_id`` can be non-NULL.
        - ``rec`` provided and tuple matches cache → reuse.
        - ``rec`` provided and tuple differs (or no cache) → insert.
        """
        cached = self._statics_cache.get(mmsi)

        if rec is None:
            if cached is not None:
                result.n_static_reused += 1
                return cached[0]
            new_id = self._insert_static(cur, schema, year, mmsi, None)
            self._statics_cache[mmsi] = (new_id, self._static_identity_tuple(None))
            result.n_static_rows += 1
            return new_id

        new_tuple = self._static_identity_tuple(rec)
        if cached is not None and cached[1] == new_tuple:
            result.n_static_reused += 1
            return cached[0]
        new_id = self._insert_static(cur, schema, year, mmsi, rec)
        self._statics_cache[mmsi] = (new_id, new_tuple)
        result.n_static_rows += 1
        return new_id

    def _ensure_state(
        self,
        cur,
        schema: str,
        year: int,
        mmsi: int,
        static_id: int,
        rec: dict[str, Any] | None,
        *,
        fallback_t: float,
        result: IngestionResult,
    ) -> int:
        """Return a ``state_id``, inserting only when the voyage data is new."""
        cached = self._states_cache.get(mmsi)

        if rec is None:
            if cached is not None:
                result.n_state_reused += 1
                return cached[0]
            new_id = self._insert_state(
                cur, schema, year, mmsi, static_id, None, fallback_t=fallback_t,
            )
            self._states_cache[mmsi] = (
                new_id, self._state_voyage_tuple(None, static_id, year_hint=year)
            )
            result.n_state_rows += 1
            return new_id

        new_tuple = self._state_voyage_tuple(rec, static_id, year_hint=year)
        if cached is not None and cached[1] == new_tuple:
            result.n_state_reused += 1
            return cached[0]
        new_id = self._insert_state(
            cur, schema, year, mmsi, static_id, rec, fallback_t=fallback_t,
        )
        self._states_cache[mmsi] = (new_id, new_tuple)
        result.n_state_rows += 1
        return new_id

    # -- per-row insert helpers ---------------------------------------------

    @staticmethod
    def _ts_to_dt(value: float | None) -> datetime | None:
        if value is None:
            return None
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    @staticmethod
    def _combine_eta(rec: dict[str, Any], year_hint: int) -> datetime | None:
        """Combine AIS Type-5 ``eta_month/day/hour/minute`` into a timestamptz.

        The AIS spec doesn't carry an ETA year — it's implicit in the
        reporting time.  Convention here: use ``year_hint`` (typically the
        ingestion target year), and if the resulting date is before
        midwinter we assume next year's voyage and bump it.

        Returns ``None`` whenever any field is missing, the spec sentinel
        for "not available" (month=0, day=0, hour=24, minute=60), or the
        combined date is invalid (e.g. Feb 30).
        """
        m = rec.get("eta_month")
        d = rec.get("eta_day")
        h = rec.get("eta_hour")
        mi = rec.get("eta_minute")
        if m is None or d is None or h is None or mi is None:
            return None
        try:
            m, d, h, mi = int(m), int(d), int(h), int(mi)
        except (TypeError, ValueError):
            return None
        # Spec sentinels: month=0 / day=0 / hour=24 / minute=60 mean "n/a".
        if m == 0 or d == 0 or h == 24 or mi == 60:
            return None
        if not (1 <= m <= 12) or not (1 <= d <= 31):
            return None
        if not (0 <= h <= 23) or not (0 <= mi <= 59):
            return None
        try:
            return datetime(int(year_hint), m, d, h, mi, tzinfo=timezone.utc)
        except ValueError:
            return None

    @staticmethod
    def _to_smallint(value: Any) -> int | None:
        """Coerce dim_a-d to PG ``smallint`` range, clamping silently."""
        if value is None:
            return None
        try:
            v = int(value)
        except (TypeError, ValueError):
            return None
        if v < -32768 or v > 32767:
            return None
        return v

    def _insert_static(
        self,
        cur,
        schema: str,
        year: int,
        mmsi: int,
        rec: dict[str, Any] | None,
    ) -> int:
        """Insert one IDENTITY-only row (mmsi, dimensions, IMO).

        Voyage-changing fields (draught, type_and_cargo, eta, destination)
        live in ``states_{year}`` — see :meth:`_insert_state`.

        ``rec=None`` writes a placeholder row with NULL identity fields,
        so callers (``_ensure_static``) can guarantee a non-NULL
        ``state.static_id`` even for input formats that carry no static
        AIS messages.  Always returns the new ``static_id``.
        """
        if rec is None:
            rec = {}
        cur.execute(
            psql.SQL(
                """
                INSERT INTO {schema}.{table}
                    (mmsi, "date", dim_a, dim_b, dim_c, dim_d, imo_num)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING rowid
                """
            ).format(
                schema=psql.Identifier(schema),
                table=psql.Identifier(f"statics_{int(year)}"),
            ),
            (
                mmsi,
                self._ts_to_dt(rec.get("time")),
                self._to_smallint(rec.get("dim_bow")),
                self._to_smallint(rec.get("dim_stern")),
                self._to_smallint(rec.get("dim_port")),
                self._to_smallint(rec.get("dim_star")),
                rec.get("imo"),
            ),
        )
        return int(cur.fetchone()[0])

    def _insert_state(
        self,
        cur,
        schema: str,
        year: int,
        mmsi: int,
        static_id: int | None,
        rec: dict[str, Any] | None,
        *,
        fallback_t: float,
    ) -> int:
        """Insert one VOYAGE row (draught, type_and_cargo, eta, destination).

        ``fallback_t`` is used as the row's ``date`` when the static record
        has no usable ``time`` (or ``rec`` is ``None``); typically the first
        timestamp of the dynamic track.
        """
        if rec is None:
            rec = {}
        eta_value = self._combine_eta(rec, year_hint=year)
        destination = rec.get("destination")
        if isinstance(destination, str):
            destination = destination.strip()[:20]  # legacy varchar(20)

        cur.execute(
            psql.SQL(
                """
                INSERT INTO {schema}.{table}
                    (mmsi, "date", draught, type_and_cargo, eta, destination, static_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING rowid
                """
            ).format(
                schema=psql.Identifier(schema),
                table=psql.Identifier(f"states_{int(year)}"),
            ),
            (
                mmsi,
                self._ts_to_dt(rec.get("time")) or self._ts_to_dt(fallback_t),
                rec.get("draught"),
                rec.get("ship_type"),
                eta_value,
                destination,
                static_id,
            ),
        )
        return int(cur.fetchone()[0])

    def _bulk_insert_segments(
        self,
        cur,
        schema: str,
        year: int,
        mmsi: int,
        state_id: int,
        segments,
    ) -> None:
        """Insert all segments for one MMSI, grouped by month for partition routing.

        Column layout matches the legacy schema:
        ``(mmsi, date1, date2, segment, cog, sog, route_id, state_id, heading)``.
        ``route_id`` and ``heading`` are NULL on initial ingestion — they're
        populated downstream (route classification, separate Type-1 dynamic
        feed) in later passes.
        """
        if not segments:
            return
        by_month: dict[int, list[tuple]] = defaultdict(list)
        for seg in segments:
            month = self._ts_to_dt(seg.t_start).month  # type: ignore[union-attr]
            wkt = (
                f"LINESTRING({seg.lon_start} {seg.lat_start}, "
                f"{seg.lon_end} {seg.lat_end})"
            )
            by_month[month].append(
                (
                    mmsi,
                    self._ts_to_dt(seg.t_start),
                    self._ts_to_dt(seg.t_end),
                    wkt,
                    int(round(float(seg.cog_mean))) % 360,  # smallint, [0,359]
                    float(seg.sog_mean),
                    None,            # route_id — set by downstream route tagging
                    state_id,
                    None,            # heading — populated from a Type-1 feed in v0.2
                )
            )
        for month, rows in by_month.items():
            execute_values(
                cur,
                psql.SQL(
                    """
                    INSERT INTO {schema}.{table}
                        (mmsi, date1, date2, segment, cog, sog,
                         route_id, state_id, heading)
                    VALUES %s
                    """
                ).format(
                    schema=psql.Identifier(schema),
                    table=psql.Identifier(f"segments_{int(year)}_{int(month)}"),
                ),
                rows,
                template=(
                    "(%s, %s, %s, "
                    "ST_GeomFromText(%s, 4326), "
                    "%s, %s, %s, %s, %s)"
                ),
            )

    def _update_watermark(self, cur, mmsi: int, last_t: float, n_segs: int) -> None:
        cur.execute(
            """
            INSERT INTO omrat_meta.segment_watermark (mmsi, last_t, n_segments)
            VALUES (%s, %s, %s)
            ON CONFLICT (mmsi) DO UPDATE SET
                last_t = GREATEST(omrat_meta.segment_watermark.last_t, EXCLUDED.last_t),
                last_run_at = now(),
                n_segments = omrat_meta.segment_watermark.n_segments + EXCLUDED.n_segments
            """,
            (mmsi, self._ts_to_dt(last_t), n_segs),
        )


# ---------------------------------------------------------------------------
# Qt thread wrapper (lazy-imported by the wizard)
# ---------------------------------------------------------------------------


def make_worker(
    profile: ConnectionProfile,
    settings: IngestionSettings,
    files: list[str | Path],
    year: int,
    source_tag: str = "OMRAT",
    incremental: bool = True,
):
    """Build a ``QThread`` that runs ``IngestionPipeline`` and emits signals.

    Imports Qt lazily so the headless pipeline (and its tests) never load it.
    """
    from qgis.PyQt.QtCore import QThread, pyqtSignal

    class _IngestionWorker(QThread):
        message = pyqtSignal(str)
        finished_with_result = pyqtSignal(object)  # IngestionResult
        failed = pyqtSignal(str)

        def __init__(self) -> None:
            super().__init__()
            self._files = [Path(f) for f in files]
            self._pipeline: IngestionPipeline | None = None

        def cancel(self) -> None:
            """Forward a cancellation request to the running pipeline.

            No-op if the pipeline hasn't started yet — the next ``run()``
            call will reset its own flag, so this is safe to call early.
            """
            if self._pipeline is not None:
                self._pipeline.cancel()

        def run(self) -> None:  # noqa: D401 - QThread API
            self._pipeline = IngestionPipeline(
                profile, settings, progress_cb=self.message.emit
            )
            try:
                result = self._pipeline.run(
                    self._files,
                    year=year,
                    source_tag=source_tag,
                    incremental=incremental,
                )
            except Exception as e:
                self.failed.emit(str(e))
                return
            self.finished_with_result.emit(result)

    return _IngestionWorker()
