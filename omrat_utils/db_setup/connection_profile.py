"""Typed connection profile with QSettings persistence and back-compat shims.

QSettings layout used here:

    omrat/db_profiles/{name}/host
    omrat/db_profiles/{name}/port
    omrat/db_profiles/{name}/database
    omrat/db_profiles/{name}/user
    omrat/db_profiles/{name}/password
    omrat/db_profiles/{name}/schema
    omrat/db_profiles/{name}/sslmode

The flat legacy keys written by ``omrat_utils.handle_ais`` (``omrat/db_host``,
``omrat/db_name``, ``omrat/db_user``, ``omrat/db_pass``) are still read as a
fallback for the profile named ``"default"`` so existing installs keep working.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

# QSettings is part of QGIS' PyQt bundle.  Importing it lazily keeps the module
# usable from headless contexts (CLI, tests) where Qt is unavailable.
try:  # pragma: no cover - import guard exercised only in non-Qt envs
    from qgis.PyQt.QtCore import QSettings
    _HAS_QT = True
except Exception:
    QSettings = None  # type: ignore[assignment]
    _HAS_QT = False


DEFAULT_PROFILE = "default"
_PROFILE_PREFIX = "omrat/db_profiles"
# NOTE: values below are QSettings KEY PATHS, not credentials.  The literal
# ``"omrat/db_pass"`` is the registry path under which the legacy plain-text
# password was stored — used only to read existing values back so older
# installs migrate cleanly.  Flagged-as-secret false positive.
_LEGACY_KEYS = {
    "host": "omrat/db_host",
    "port": "omrat/db_port",
    "database": "omrat/db_name",
    "user": "omrat/db_user",
    "password": "omrat/db_pass",  # pragma: allowlist secret - QSettings key path, not a secret
}


def decode_libpq_message(exc: UnicodeDecodeError) -> str:
    """Decode the bytes inside a ``UnicodeDecodeError`` from libpq.

    libpq emits some pre-startup error messages (notably auth/role/db
    rejections) in the OS ``lc_messages`` encoding regardless of the
    ``client_encoding`` startup parameter, so psycopg2 hits a UTF-8
    decode failure before raising ``OperationalError``.  When that
    happens the actual server message is stored in ``exc.object``;
    decoding it as cp1252 (Windows Western European, the most common
    case for a Swedish/German/French Windows install) recovers a
    readable string.  ``errors='replace'`` guarantees the call cannot
    itself raise — the worst case is a few ``�`` replacement
    characters in an already-degraded code path.

    Returns the trimmed message, or an empty string when ``exc.object``
    is empty/missing.
    """
    raw = getattr(exc, "object", None) or b""
    if not raw:
        return ""
    try:
        text = bytes(raw).decode("cp1252")
    except UnicodeDecodeError:
        text = bytes(raw).decode("latin-1", errors="replace")
    return text.strip()


@dataclass
class ConnectionProfile:
    """A named Postgres connection profile.

    ``schema`` is the AIS data schema (e.g. ``sjfv`` for legacy installs,
    ``omrat`` for fresh setups).  ``password`` is held in plaintext for
    parity with the existing ``handle_ais`` path; QgsAuthManager-backed
    storage is the long-term replacement.
    """

    name: str = DEFAULT_PROFILE
    host: str = ""
    port: int = 5432
    database: str = ""
    user: str = ""
    password: str = ""
    schema: str = "omrat"
    sslmode: str = "prefer"

    # ---------------------------------------------------------------- helpers

    def to_dsn(self) -> dict[str, Any]:
        """Return a kwargs dict suitable for ``psycopg2.connect(**dsn)``.

        ``client_encoding='UTF8'`` is set unconditionally so the server
        transcodes its messages to UTF-8 before sending them — without
        this, a non-English Windows locale (e.g. Swedish ``lc_messages``)
        emits cp1252-encoded notices that psycopg2 then crashes on with
        ``UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf6``.
        """
        return {
            "host": self.host,
            "port": int(self.port) if self.port else 5432,
            "dbname": self.database,
            "user": self.user,
            "password": self.password,
            "sslmode": self.sslmode or "prefer",
            "client_encoding": "UTF8",
        }

    def is_complete(self) -> bool:
        return bool(self.host and self.database and self.user)

    # ------------------------------------------------------------ persistence

    @classmethod
    def from_qsettings(cls, name: str = DEFAULT_PROFILE) -> "ConnectionProfile":
        if not _HAS_QT:
            return cls(name=name)
        s = QSettings()
        prefix = f"{_PROFILE_PREFIX}/{name}"
        host = s.value(f"{prefix}/host", "", type=str)
        # Fall back to legacy flat keys for the default profile so existing
        # installs (handle_ais.py wrote those keys) are picked up automatically.
        if not host and name == DEFAULT_PROFILE:
            host = s.value(_LEGACY_KEYS["host"], "", type=str)
            database = s.value(_LEGACY_KEYS["database"], "", type=str)
            user = s.value(_LEGACY_KEYS["user"], "", type=str)
            password = s.value(_LEGACY_KEYS["password"], "", type=str)
            # Legacy port key was added later than the others; treat
            # missing as the Postgres default rather than 0.
            try:
                port = int(s.value(_LEGACY_KEYS["port"], 5432) or 5432)
            except (TypeError, ValueError):
                port = 5432
        else:
            database = s.value(f"{prefix}/database", "", type=str)
            user = s.value(f"{prefix}/user", "", type=str)
            password = s.value(f"{prefix}/password", "", type=str)
            port = int(s.value(f"{prefix}/port", 5432, type=int) or 5432)
        return cls(
            name=name,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            schema=s.value(f"{prefix}/schema", "omrat", type=str) or "omrat",
            sslmode=s.value(f"{prefix}/sslmode", "prefer", type=str) or "prefer",
        )

    def to_qsettings(self) -> None:
        if not _HAS_QT:
            return
        s = QSettings()
        prefix = f"{_PROFILE_PREFIX}/{self.name}"
        s.setValue(f"{prefix}/host", self.host)
        s.setValue(f"{prefix}/port", int(self.port))
        s.setValue(f"{prefix}/database", self.database)
        s.setValue(f"{prefix}/user", self.user)
        s.setValue(f"{prefix}/password", self.password)
        s.setValue(f"{prefix}/schema", self.schema)
        s.setValue(f"{prefix}/sslmode", self.sslmode)
        # Mirror to legacy flat keys for the default profile so handle_ais.py
        # keeps reading the same credentials the wizard last saved.
        if self.name == DEFAULT_PROFILE:
            s.setValue(_LEGACY_KEYS["host"], self.host)
            s.setValue(_LEGACY_KEYS["port"], int(self.port))
            s.setValue(_LEGACY_KEYS["database"], self.database)
            s.setValue(_LEGACY_KEYS["user"], self.user)
            s.setValue(_LEGACY_KEYS["password"], self.password)

    @classmethod
    def list_profiles(cls) -> list[str]:
        if not _HAS_QT:
            return []
        s = QSettings()
        s.beginGroup(_PROFILE_PREFIX)
        names = list(s.childGroups())
        s.endGroup()
        if DEFAULT_PROFILE not in names:
            # Surface the legacy flat-key install as the default profile.
            if s.value(_LEGACY_KEYS["host"], "", type=str):
                names.insert(0, DEFAULT_PROFILE)
        return names

    @classmethod
    def delete_profile(cls, name: str) -> None:
        if not _HAS_QT:
            return
        s = QSettings()
        s.beginGroup(f"{_PROFILE_PREFIX}/{name}")
        s.remove("")
        s.endGroup()

    # ----------------------------------------------------------- dict helpers

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConnectionProfile":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
