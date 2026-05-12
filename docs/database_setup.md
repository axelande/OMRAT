# OMRAT database setup & AIS ingestion

End-to-end guide for getting from "nothing" to "OMRAT querying real AIS
linestring segments out of PostGIS."

## Architecture in one paragraph

OMRAT's risk calculations consume **constant-COG/SOG linestring segments**
out of PostGIS, one row per `(MMSI, time-window, course/speed)`.  Those
rows are produced by feeding raw AIS pings through the **TDKC compression
algorithm** (Guo et al. 2024) implemented in the
[AISsegments](https://github.com/axelande/AISsegments) sister package.
The full pipeline is:

```
raw AIS files (NMEA / CSV)
     │
     ▼   aisdb.decode_msgs()
per-vessel point streams (temp SQLite)
     │
     ▼   aissegments.tdkc_segments()
constant-COG/SOG linestring segments
     │
     ▼   psycopg2 + PostGIS bulk insert
{schema}.segments_YYYY_M  +  {schema}.statics_YYYY  +  {schema}.states_YYYY
     │
     ▼   omrat_utils.handle_ais.run_sql()
risk-analysis queries
```

## 1. Stand up the database

The easiest path on a developer machine: the bundled Docker stack
([docker/](../docker/README.md)).

```bash
cd <OMRAT repo root>
cp docker/env.example docker/.env       # edit credentials if you want
docker compose -f docker/docker-compose.yml up -d
```

That gives you `localhost:5432` with PostGIS 3.4 enabled, running as user
`omrat` / database `omrat`.

For a remote / institutional database, just enable PostGIS and create a
role with `CREATE` privileges on the database — see the wizard's
"Database capabilities" page for what it checks.

## 2. Install OMRAT's Python dependencies

OMRAT's [requirements.txt](../requirements.txt) now lists
[`aissegments`](https://github.com/axelande/AISsegments) and
[`aisdb`](https://github.com/AISViz/AISdb) alongside the existing deps.
QGIS plugin loads pull these in via `qpip` (configured in
[metadata.txt](../metadata.txt)).

For development outside QGIS:

```bash
pip install aissegments aisdb
```

## Supported AIS file formats

The ingestion pipeline auto-classifies each input file by extension + a
header sniff and routes it to the right decoder:

| File pattern | Header signature | Routed to | Static AIS data? |
|---|---|---|---|
| `*.nm4`, `*.nmea` | (binary NMEA) | `aisdb.decode_msgs()` | ✓ from Type-5 messages |
| `*.csv`, `*.csv.gz` with `Message_ID` / `Repeat_indicator` columns | aisdb's own CSV dump | `aisdb.decode_msgs()` | ✓ |
| `*.csv`, `*.csv.gz` without those columns (Marine Cadastre, custom exports) | generic AIS CSV | `aissegments.read_csv_tracks()` + `read_csv_static_records()` | ✓ when static columns are present (Length/Width/Draft/IMO/...) |

For the **simple-CSV** path (Marine Cadastre and similar), AISsegments
recognises a wide range of column-name aliases — `BaseDateTime` /
`timestamp` / `time`, `LAT` / `latitude` / `lat`, `LON` / `longitude` /
`lon`, etc. — and parses ISO 8601 timestamps as well as Unix seconds.

If the CSV also carries vessel-info columns (Marine Cadastre's
`VesselName` / `IMO` / `CallSign` / `VesselType` / `Length` / `Width` /
`Draft`), the pipeline extracts them via
`aissegments.read_csv_static_records()` and populates `statics_YYYY`
and `states_YYYY` accordingly.  Length and Width get split half/half
into AISdb's per-quadrant antenna offsets (`dim_a` = `dim_b` = Length/2
and `dim_c` = `dim_d` = Width/2 — a centred-antenna approximation).

If the CSV is bare (just `mmsi`/`time`/`lon`/`lat`/`sog`/`cog`), the
identity and voyage fields land as NULL placeholders.  Downstream AIS
queries derive vessel length and beam directly from the AIS Type-5
dimensions (`loa = dim_a + dim_b`, `beam = dim_c + dim_d`); ship-type
classification uses the AIS `type_and_cargo` field.  Air-draught
distributions stay empty unless you supply a richer vessel registry of
your own and reinstate a JOIN in
[handle_ais.py:run_sql](../omrat_utils/handle_ais.py).

## 3. Walk the wizard

Open QGIS, load the OMRAT plugin, then **Settings → Database setup
wizard…**.  The wizard has five pages:

1. **Intro** — quick orientation; offers a button to open the Docker
   quickstart (`docker/README.md`).
2. **Connection** — host / port / db / user / password / schema / sslmode.
   Click *Test connection*.  Once a probe succeeds, the *Next* button
   activates.
3. **Database capabilities** — runs `DbProbe` and lists what's in place vs
   what's missing.  Buttons:
   - *Enable PostGIS* (only if the user is superuser; otherwise displays
     the SQL for a DBA to run).
   - *Apply OMRAT schema migrations* — creates `omrat_meta` (version
     table + ingestion watermark) and the AIS schema you configured.
   - *Create year-partitioned tables for `<year>`* — provisions
     `statics_YYYY`, `states_YYYY`, and the 12 monthly partitions of
     `segments_YYYY`.
4. **Ingest AIS data (optional)** — the new ingestion page:
   - Pick AIS files (NMEA `.nm4`, aisdb `.csv`, gzipped variants).
   - Set `min_sed_m` (default **30 m**) and `min_svd_kn` (default
     **0.3 kn**) — these are the OMRAT-tuned TDKC threshold floors.
   - Set the target year and a source label.
   - Click *Run ingestion*.  The job runs on a `QThread`; progress
     messages stream into the log view.  Per-MMSI: insert one
     `statics_YYYY` row, one `states_YYYY` row, and a batch of
     `segments_YYYY_M` linestring rows from `tdkc_segments(...)`.
5. **Done** — saves the connection profile + ingestion settings to
   QSettings (`omrat/db_profiles/default/*` and
   `omrat/ingest_profiles/default/*`).  The legacy flat keys read by
   [`handle_ais.py`](../omrat_utils/handle_ais.py) are mirrored
   automatically, so existing AIS-traffic queries pick up the new
   credentials transparently.

## 4. Verify with a smoke test

After the wizard finishes, you can run a one-shot end-to-end check by
pointing OMRAT at a small fixture.  AISdb's bundled
`test_data_20210701.csv` is the easiest:

```python
from pathlib import Path
import aisdb

from omrat_utils.db_setup import ConnectionProfile, IngestionSettings, Migrator
from omrat_utils.handle_ais_ingest import IngestionPipeline

profile = ConnectionProfile.from_qsettings()           # picked up from the wizard
settings = IngestionSettings.from_qsettings(profile.name)  # OMRAT defaults

# Make sure the year tables exist for the dataset's date range.
Migrator(profile).apply_pending()
Migrator(profile).ensure_year_partition(2021)

aisdb_csv = Path(aisdb.__file__).parent / "tests" / "testdata" / "test_data_20210701.csv"
result = IngestionPipeline(profile, settings).run([aisdb_csv], year=2021)
print(result.summary())
```

Then in `psql`:

```sql
SELECT count(*) FROM omrat.segments_2021;
SELECT count(*) FROM omrat.statics_2021;
SELECT count(*) FROM omrat_meta.segment_watermark;
```

You should see non-zero counts in all three.

## 5. Run risk analysis as before

Once the segments table is populated, [`omrat_utils.handle_ais.run_sql`](../omrat_utils/handle_ais.py)
queries `{schema}.segments_{year}_{month}` exactly as it always did —
the schema layout is intentionally compatible.  The "AIS connection
settings" menu in OMRAT's main dialog reads from the same QSettings keys
the wizard saved, so traffic-fetching for legs / segments works without
any additional config.

## Database schema reference

The schema matches OMRAT's legacy (sjfv) layout used by
[handle_ais.py](../omrat_utils/handle_ais.py).  Two metadata tables in
`omrat_meta` plus a year-partitioned trio in the user-configured schema:

```
omrat_meta.schema_version       (version PK, name, applied_at)
omrat_meta.segment_watermark    (mmsi PK, last_t, last_run_at, n_segments)

{schema}.statics_YYYY            ← per-vessel IDENTITY (changes ~never)
   ├ rowid  bigserial PK
   ├ mmsi   bigint
   ├ date   timestamptz          (when this identity row was reported)
   ├ dim_a  smallint             (antenna→bow, m)
   ├ dim_b  smallint             (antenna→stern)
   ├ dim_c  smallint             (antenna→port)
   ├ dim_d  smallint             (antenna→starboard)
   └ imo_num bigint

{schema}.states_YYYY             ← per-VOYAGE static data (changes per leg)
   ├ rowid  bigserial PK
   ├ mmsi   bigint
   ├ date   timestamptz
   ├ draught         double precision
   ├ type_and_cargo  smallint
   ├ eta             timestamptz   (NULL in v0.1; populated in v0.2)
   ├ destination     varchar(20)
   └ static_id       bigint  → statics_YYYY.rowid  ON DELETE SET NULL

{schema}.segments_YYYY           ← TDKC linestring rows (PARTITION BY RANGE date1)
   ├ rowid    bigserial
   ├ mmsi     bigint
   ├ date1    timestamptz NOT NULL  (segment start; partition key)
   ├ date2    timestamptz           (segment end)
   ├ segment  geometry(LineString, 4326)
   ├ cog      smallint              (mean course, [0, 359])
   ├ sog      double precision      (mean speed, knots)
   ├ route_id bigint                (NULL on initial ingestion)
   ├ state_id bigint                (logical link to states_YYYY.rowid)
   ├ heading  smallint              (NULL on initial ingestion)
   └ PRIMARY KEY (rowid, date1)
       ↳ segments_YYYY_1   PARTITION OF segments_YYYY  for Jan
       ↳ segments_YYYY_2   PARTITION OF segments_YYYY  for Feb
       ↳ ... segments_YYYY_12  for Dec
```

Differences from the legacy schema:

- **Postgres declarative partitioning** on `segments_YYYY` (was a single
  yearly table or per-month UNION).  Queries against
  `segments_YYYY_M` continue to work; `segments_YYYY` now also works
  thanks to partition pruning.  The composite PK `(rowid, date1)` is a
  Postgres requirement for partitioned-table unique constraints — `rowid`
  alone is still unique by virtue of `bigserial`.
- **`segment` is constrained** to `geometry(LineString, 4326)`; legacy
  used the unconstrained `public.geometry`.  The constraint catches
  insert errors but is invisible to existing `ST_Intersects` queries.
- **Foreign key `states_YYYY.static_id → statics_YYYY.rowid`** is
  declared; `segments_YYYY.state_id` is NOT a hard FK (FKs from a
  partitioned table get nuanced in older PG).

## Static + state row deduplication

The pipeline does NOT insert a new `statics_YYYY` or `states_YYYY` row
when the data is identical to the most recent existing row for that
MMSI.  Each ingestion path:

1. **Pre-loads** the latest `(mmsi, identity_tuple)` pairs for statics
   and the latest `(mmsi, voyage_tuple)` pairs for states from the DB,
   in one query each (using PostgreSQL's `DISTINCT ON (mmsi)`).
2. **For every track**, computes the new identity / voyage tuple and
   compares to the cached one.  Match → reuse the existing rowid, no
   INSERT.  Mismatch → INSERT a new row and update the cache.

Identity tuple for `statics_YYYY`: `(dim_a, dim_b, dim_c, dim_d, imo_num)`.

Voyage tuple for `states_YYYY`:
`(draught, type_and_cargo, destination, eta, static_id)`.

This means:

- A vessel reported with the same dimensions across multiple ingestion
  runs gets one `statics_YYYY` row total.
- A vessel reporting the same draught + destination + ETA on every
  message gets one `states_YYYY` row.
- A draught change, destination change, ETA change, or new IMO triggers
  exactly one new row.  Older segments still link to the older row;
  newer segments link to the newer one.  This gives **time-windowed
  static linking** for free as voyage data evolves.

The summary line counts both inserts and reuses, e.g.:

```
Ingested 5 file(s) → 1234 tracks, 24500 segments, 12 static, 145 state rows,
reused 1222 static + 1089 state in 412.3s
```

## ETA combination

The legacy schema's `states_YYYY.eta` is a `timestamptz`.  AIS Type-5
static messages encode ETA as four separate fields (`eta_month`,
`eta_day`, `eta_hour`, `eta_minute`) with no year — the year is
implicit in the report.  The pipeline combines them using the
ingestion target year as the year hint, with the AIS spec sentinels
(`month=0`, `day=0`, `hour=24`, `minute=60`) and invalid calendar
dates (Feb 30, etc.) all mapped to NULL.

ETA is part of the voyage tuple, so a voyage with the same
draught/destination but a corrected ETA still produces a new
`states_YYYY` row.

## Incremental ingestion (re-runs are cheap)

Each successful per-MMSI insert updates `omrat_meta.segment_watermark`
with the latest AIS timestamp seen for that vessel.  On the next run
(`incremental=True`, the wizard's default), the pipeline:

1. Reads the watermark table at the start of each ingestion path.
2. For every track it sees, drops any pings with `time <= last_t` for
   that MMSI before passing the track into TDKC.
3. If nothing remains after the filter, the track is recorded under
   `n_tracks_skipped_watermark` and skipped silently.

This makes overlapping or repeated ingestions safely idempotent — point
the wizard at the same files twice and the second run inserts zero new
segments (you'll see `… skipped (watermark)` in the summary).

To force a full re-ingest, uncheck **Incremental** in the wizard or pass
`incremental=False` to `IngestionPipeline.run(...)`.  Typically you'd
also `TRUNCATE` the relevant `segments_YYYY_M` partitions first,
otherwise you'll duplicate rows.

## Index strategy: build after bulk load

The `segments_YYYY` table can hold tens of millions of rows after a
single year of dense AIS data.  Maintaining the GiST + btree indexes
during bulk INSERT slows ingestion by a factor of 5-10x — so the
migration that creates the tables deliberately leaves them **unindexed**
for `segments_YYYY`.  Indexes on `statics_YYYY` and `states_YYYY` (which
hold thousands of rows, not millions) are created inline since their
maintenance cost is negligible.

After ingestion finishes, the pipeline runs
[`Migrator.create_year_indexes(year)`](../omrat_utils/db_setup/migrations.py)
once on the populated tables, building:

| Index | Column(s) | Used by |
|-------|-----------|---------|
| `segments_YYYY_geom_gix`        | GiST on `segment`    | `ST_Intersects` corridor queries in `handle_ais.py` |
| `segments_YYYY_mmsi_date_idx`   | `(mmsi, date1)`      | per-vessel time-window lookups |
| `segments_YYYY_state_id_idx`    | `state_id`           | JOIN to `states_YYYY` for ship metadata |
| `segments_YYYY_route_id_idx`    | `route_id`           | downstream route-tagging queries |

The index step also runs `ANALYZE` on all three tables so the planner
picks the new indexes up immediately.

`IngestionPipeline.run(...)` does this automatically when finished
(`create_indexes_after=True` by default).  If you ingest several batches
into the same year and want to delay indexing until the end, pass
`create_indexes_after=False` on every batch except the last.  Calling
`create_year_indexes(year)` more than once is safe — every CREATE uses
`IF NOT EXISTS`.

## Threshold tuning

`min_sed_m` and `min_svd_kn` are **per-ingestion** — they're settings,
not constants.  Edit them in the wizard's Ingest page or override
programmatically:

```python
settings = IngestionSettings(min_sed_m=10.0, min_svd_kn=0.1)
```

Sensible regimes:

| Use case | min_sed_m | min_svd_kn | Effect |
| --- | --- | --- | --- |
| OMRAT default | 30 m | 0.3 kn | Risk-grade compression, filters AIS jitter |
| High fidelity research | 5 m | 0.05 kn | Preserves more detail, larger DB footprint |
| Long-term archive | 100 m | 1.0 kn | Aggressive compression, ~order-of-magnitude smaller |
| Paper-faithful (Guo et al.) | 0 | 0 | No floor; uses pure adaptive thresholds |

See [AISsegments docs/algorithm.md](https://github.com/axelande/AISsegments/blob/main/docs/algorithm.md)
for the algorithm details and parameter visualisations.

## Operational notes

- **Idempotent**: re-running the wizard or `apply_pending()` is safe.
  Schema migrations use `IF NOT EXISTS`; the migrator records its
  version in `omrat_meta.schema_version`.
- **Watermark**: `omrat_meta.segment_watermark` records the last AIS
  timestamp seen per MMSI.  A future `IngestionPipeline` iteration can
  use this to skip already-ingested data on reruns.
- **Threading**: the ingestion worker uses `QThread`; the UI stays
  responsive.  Cancellation isn't wired up yet — close the wizard or
  stop QGIS to abort.
- **Batch commits**: the worker commits every 200 tracks so a long run
  doesn't sit in a single transaction the entire time.
- **Static-data caveat**: in the v0.1 cut each MMSI gets one
  `statics_YYYY` row + one `states_YYYY` row per ingestion run.  Vessels
  that change registration mid-period get a less precise mapping; the
  schema supports proper time-windowed linkage and a future iteration
  can populate it without changing the on-disk format.

## Troubleshooting

**"PostGIS extension is missing"** — Either run `CREATE EXTENSION
postgis;` as a DB superuser, or use the bundled Docker stack which
preloads it.

**"Year-partition tables missing"** — The wizard's Capabilities page has
a year-spinbox + *Create tables* button.  Or call
`Migrator(profile).ensure_year_partition(year)` programmatically.

**Ingestion is slow** — Expected; profiling and parallelisation are on
the v0.2 roadmap.  For one Baltic month (~10M points) on a developer
laptop, expect tens of minutes.  Bulk inserts already use
`psycopg2.extras.execute_values`; the next bottleneck will likely be
TDKC itself, where Numba acceleration inside AISsegments is a clear
candidate.
