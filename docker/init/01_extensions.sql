-- Runs once on first container start (postgres-entrypoint convention).
-- Enables PostGIS in the OMRAT database; the OMRAT schema and year tables
-- are created later by omrat_utils.db_setup.Migrator (idempotent).

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS btree_gist;
