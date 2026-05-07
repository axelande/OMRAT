-- V001: bootstrap omrat_meta schema (version tracking + ingestion watermark).
--
-- Idempotent: every CREATE uses IF NOT EXISTS so a partial earlier run is
-- safe to retry.  The Migrator records the version row only after this whole
-- script commits, so failure leaves the database unchanged.
--
-- {schema} is substituted by the migrator with the user-configured AIS schema
-- (validated against [A-Za-z_][A-Za-z0-9_]*).  This file does NOT create the
-- year-partitioned tables — those are provisioned on demand from
-- templates/year_partition.sql.tpl.

CREATE SCHEMA IF NOT EXISTS omrat_meta;

CREATE TABLE IF NOT EXISTS omrat_meta.schema_version (
    version    integer PRIMARY KEY,
    name       text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
);

-- Per-vessel high-water mark used by the TDKC segmentation worker so it can
-- resume incrementally instead of reprocessing already-vectorised history.
CREATE TABLE IF NOT EXISTS omrat_meta.segment_watermark (
    mmsi          bigint PRIMARY KEY,
    last_t        timestamptz NOT NULL,
    last_run_at   timestamptz NOT NULL DEFAULT now(),
    n_segments    bigint NOT NULL DEFAULT 0
);

-- Target AIS schema where the year-partitioned tables will live.  Created
-- here so subsequent year-partition DDL doesn't have to.
CREATE SCHEMA IF NOT EXISTS {schema};
