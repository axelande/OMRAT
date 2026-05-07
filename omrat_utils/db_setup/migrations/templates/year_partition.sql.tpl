-- Year-partition template.  Rendered by Migrator.ensure_year_partition with
-- {schema} (validated SQL identifier) and {year} (int 1900-2999).
--
-- Layout matches the legacy OMRAT (sjfv) schema queried by handle_ais.py:
--   {schema}.statics_{year}            ← per-vessel IDENTITY only
--                                        (mmsi, dimensions, imo_num)
--   {schema}.states_{year}             ← per-VOYAGE static data
--                                        (draught, type_and_cargo, eta,
--                                         destination, static_id → statics)
--   {schema}.segments_{year}_{month}   ← linestring rows (TDKC output)
--                                        with a state_id back-reference
--
-- Indexes on segments_{year} are deliberately NOT created here — they're
-- built after bulk ingestion via ``Migrator.create_year_indexes(year)``.
-- Building indexes once on a populated table is much faster than
-- maintaining them incrementally during bulk INSERT.  Statics and states
-- are small enough that the savings don't matter, so their MMSI index
-- stays inline.
--
-- segments_{year} is a Postgres-declarative range partition on date1; the
-- monthly child tables retain the legacy names so existing queries
-- (`segments_2024_6`, etc.) keep working unchanged.

CREATE TABLE IF NOT EXISTS {schema}.statics_{year} (
    rowid    bigserial PRIMARY KEY,
    mmsi     bigint,
    "date"   timestamptz,
    dim_a    smallint,
    dim_b    smallint,
    dim_c    smallint,
    dim_d    smallint,
    imo_num  bigint
);
CREATE INDEX IF NOT EXISTS statics_{year}_mmsi_idx
    ON {schema}.statics_{year} (mmsi);

CREATE TABLE IF NOT EXISTS {schema}.states_{year} (
    rowid          bigserial PRIMARY KEY,
    mmsi           bigint,
    "date"         timestamptz,
    draught        double precision,
    type_and_cargo smallint,
    eta            timestamptz,
    destination    varchar(20),
    static_id      bigint,
    CONSTRAINT states_{year}_static_fkey
        FOREIGN KEY (static_id) REFERENCES {schema}.statics_{year}(rowid)
        ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS states_{year}_mmsi_idx
    ON {schema}.states_{year} (mmsi);

CREATE TABLE IF NOT EXISTS {schema}.segments_{year} (
    rowid     bigserial,
    mmsi      bigint,
    date1     timestamptz NOT NULL,
    date2     timestamptz,
    segment   geometry(LineString, 4326),
    cog       smallint,
    sog       double precision,
    route_id  bigint,
    state_id  bigint,
    heading   smallint,
    PRIMARY KEY (rowid, date1)
) PARTITION BY RANGE (date1);

CREATE TABLE IF NOT EXISTS {schema}.segments_{year}_1
    PARTITION OF {schema}.segments_{year}
    FOR VALUES FROM ('{year}-01-01 00:00:00+00') TO ('{year}-02-01 00:00:00+00');
CREATE TABLE IF NOT EXISTS {schema}.segments_{year}_2
    PARTITION OF {schema}.segments_{year}
    FOR VALUES FROM ('{year}-02-01 00:00:00+00') TO ('{year}-03-01 00:00:00+00');
CREATE TABLE IF NOT EXISTS {schema}.segments_{year}_3
    PARTITION OF {schema}.segments_{year}
    FOR VALUES FROM ('{year}-03-01 00:00:00+00') TO ('{year}-04-01 00:00:00+00');
CREATE TABLE IF NOT EXISTS {schema}.segments_{year}_4
    PARTITION OF {schema}.segments_{year}
    FOR VALUES FROM ('{year}-04-01 00:00:00+00') TO ('{year}-05-01 00:00:00+00');
CREATE TABLE IF NOT EXISTS {schema}.segments_{year}_5
    PARTITION OF {schema}.segments_{year}
    FOR VALUES FROM ('{year}-05-01 00:00:00+00') TO ('{year}-06-01 00:00:00+00');
CREATE TABLE IF NOT EXISTS {schema}.segments_{year}_6
    PARTITION OF {schema}.segments_{year}
    FOR VALUES FROM ('{year}-06-01 00:00:00+00') TO ('{year}-07-01 00:00:00+00');
CREATE TABLE IF NOT EXISTS {schema}.segments_{year}_7
    PARTITION OF {schema}.segments_{year}
    FOR VALUES FROM ('{year}-07-01 00:00:00+00') TO ('{year}-08-01 00:00:00+00');
CREATE TABLE IF NOT EXISTS {schema}.segments_{year}_8
    PARTITION OF {schema}.segments_{year}
    FOR VALUES FROM ('{year}-08-01 00:00:00+00') TO ('{year}-09-01 00:00:00+00');
CREATE TABLE IF NOT EXISTS {schema}.segments_{year}_9
    PARTITION OF {schema}.segments_{year}
    FOR VALUES FROM ('{year}-09-01 00:00:00+00') TO ('{year}-10-01 00:00:00+00');
CREATE TABLE IF NOT EXISTS {schema}.segments_{year}_10
    PARTITION OF {schema}.segments_{year}
    FOR VALUES FROM ('{year}-10-01 00:00:00+00') TO ('{year}-11-01 00:00:00+00');
CREATE TABLE IF NOT EXISTS {schema}.segments_{year}_11
    PARTITION OF {schema}.segments_{year}
    FOR VALUES FROM ('{year}-11-01 00:00:00+00') TO ('{year}-12-01 00:00:00+00');
CREATE TABLE IF NOT EXISTS {schema}.segments_{year}_12
    PARTITION OF {schema}.segments_{year}
    FOR VALUES FROM ('{year}-12-01 00:00:00+00') TO ('{year}+1-01-01 00:00:00+00');
