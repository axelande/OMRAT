"""Database setup, probing and migration for OMRAT's Postgres/PostGIS backend."""
from omrat_utils.db_setup.connection_profile import (
    ConnectionProfile,
    decode_libpq_message,
)
from omrat_utils.db_setup.db_probe import DbProbe, ProbeResult
from omrat_utils.db_setup.ingestion_settings import (
    DEFAULT_MAX_GAP_S,
    DEFAULT_MIN_SED_M,
    DEFAULT_MIN_SVD_KN,
    DEFAULT_SPEED_FLOOR_KN,
    DEFAULT_SPEED_TOLERANCE,
    IngestionSettings,
)
from omrat_utils.db_setup.migrations import Migrator, MigrationError

__all__ = [
    "ConnectionProfile",
    "decode_libpq_message",
    "DbProbe",
    "ProbeResult",
    "Migrator",
    "MigrationError",
    "IngestionSettings",
    "DEFAULT_MIN_SED_M",
    "DEFAULT_MIN_SVD_KN",
    "DEFAULT_MAX_GAP_S",
    "DEFAULT_SPEED_TOLERANCE",
    "DEFAULT_SPEED_FLOOR_KN",
]
