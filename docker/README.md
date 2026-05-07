# OMRAT local Postgres/PostGIS stack

This is the **lowest-friction option** for users who do not have a Postgres
server and do not want to install one natively. It stands up a containerized
PostgreSQL 16 + PostGIS 3.4 instance that the OMRAT plugin can connect to
out of the box.

## What you get

- A single container (`omrat-db`) running `postgis/postgis:16-3.4`.
- Database `omrat`, role `omrat`, password `omrat` (override in `.env`).
- PostGIS, PostGIS Topology, and `btree_gist` extensions enabled on first
  start.
- A named Docker volume (`omrat-db-data`) so your data survives container
  restarts.

## Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine + Compose plugin (Linux).
- Port 5432 free on the host (override with `OMRAT_DB_PORT` in `.env` if not).

## Quickstart

```bash
cp docker/.env.example docker/.env
# edit docker/.env if you want non-default credentials
docker compose -f docker/docker-compose.yml up -d
```

Wait ~30 seconds for the healthcheck to go green, then connect from OMRAT
(Settings → AIS connection settings) using:

| Field    | Value                              |
| -------- | ---------------------------------- |
| Host     | `localhost`                        |
| Database | `omrat` (or your `OMRAT_DB_NAME`)  |
| User     | `omrat` (or your `OMRAT_DB_USER`)  |
| Password | `omrat` (or your `OMRAT_DB_PASSWORD`) |
| Schema   | `omrat`                            |

Then run the OMRAT schema migrations (the wizard does this for you, or
from a Python shell):

```python
from omrat_utils.db_setup import ConnectionProfile, Migrator
p = ConnectionProfile(host="localhost", database="omrat",
                      user="omrat", password="omrat", schema="omrat")
Migrator(p).apply_pending()
Migrator(p).ensure_year_partition(2024)
```

## Stopping / wiping

```bash
docker compose -f docker/docker-compose.yml down            # stop
docker compose -f docker/docker-compose.yml down -v         # stop + DELETE data
```

## Why not TimescaleDB out of the box?

The `postgis/postgis` image is multi-arch (Apple Silicon, x86_64, ARM Linux),
small, and stable across PostGIS releases. TimescaleDB compression is useful
for raw AIS point storage but is not required by OMRAT itself — the
linestring-segments table is small enough that PostGIS partitioning alone is
sufficient. To swap in TimescaleDB later, replace the `image:` line with
`timescale/timescaledb-ha:pg16` and add `CREATE EXTENSION IF NOT EXISTS
timescaledb;` to `init/01_extensions.sql` before the first run.

## Not for production

This compose file binds to `0.0.0.0:5432` with a default password and no TLS.
It is fine for single-user local development. For shared/remote deployments,
put it behind a reverse proxy with TLS, set a strong `OMRAT_DB_PASSWORD`, and
restrict `ports:` to a private interface.
