-- Post-ingestion index template for {schema}.segments_{year}.
--
-- Run via ``Migrator.create_year_indexes(year)`` AFTER bulk-loading the
-- monthly partitions.  Maintaining these indexes during ingestion would
-- multiply insert cost by 5-10x; building them once on populated data is
-- the standard ETL pattern.
--
-- Each index targets a column group queried by handle_ais.py or by the
-- broader OMRAT analytics path:
--   - segment (GiST)            — ST_Intersects with the corridor polygon
--   - (mmsi, date1)             — vessel-history lookups, partition pruning
--   - state_id                  — JOIN to states_{year} for ship metadata
--   - route_id                  — when downstream code tags segments by route
--
-- Idempotent — every CREATE uses IF NOT EXISTS.

CREATE INDEX IF NOT EXISTS segments_{year}_geom_gix
    ON {schema}.segments_{year} USING GIST (segment);

CREATE INDEX IF NOT EXISTS segments_{year}_mmsi_date_idx
    ON {schema}.segments_{year} (mmsi, date1);

CREATE INDEX IF NOT EXISTS segments_{year}_state_id_idx
    ON {schema}.segments_{year} (state_id);

CREATE INDEX IF NOT EXISTS segments_{year}_route_id_idx
    ON {schema}.segments_{year} (route_id);

-- Refresh planner statistics so the new indexes get used right away.
ANALYZE {schema}.segments_{year};
ANALYZE {schema}.states_{year};
ANALYZE {schema}.statics_{year};
