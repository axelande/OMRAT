"""User-configurable LEFT JOIN against an external vessel-metadata table.

Vessel length and beam are normally derived directly from the AIS
Type-5 dimensions (``loa = dim_a + dim_b``, ``beam = dim_c + dim_d``)
and ship type from ``type_and_cargo``.  Some users maintain their own
institutional vessel registry and want richer metadata (proper ``loa``,
beam, air draught) when those broadcast values are missing or
unreliable.

This module lets the user point the plugin at any compatible Postgres
table via the AIS Settings dialog.  All identifiers are validated as
plain SQL names so they can be safely interpolated into the query
(table/column names cannot be bound as query parameters).  Per-column
fields are individually optional: the loa/beam columns fall back to the
AIS dim arithmetic when blank, and ship_type/air_draught simply emit
NULL when blank.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

# Lazy Qt import: the dataclass and its SQL builder are useful headless
# (tests, future CLI ingestion); only ``from_qsettings`` / ``to_qsettings``
# need Qt.
try:  # pragma: no cover - import guard exercised only in non-Qt envs
    from qgis.PyQt.QtCore import QSettings
    _HAS_QT = True
except Exception:
    QSettings = None  # type: ignore[assignment]
    _HAS_QT = False


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PREFIX = "omrat/vessel_lookup"


def _is_ident(value: str) -> bool:
    return bool(value) and bool(_IDENT_RE.match(value))


@dataclass
class VesselLookupConfig:
    """Where (and how) to LEFT JOIN extra per-MMSI vessel metadata.

    All strings are plain SQL identifiers — no dotted paths, no quoting.
    The config is considered *active* only when ``enabled`` is True AND
    :meth:`is_valid` returns True; otherwise the AIS query falls back to
    the statics-only path.

    Attributes
    ----------
    enabled : bool
        Master switch.  When False the lookup is skipped regardless of
        the other fields, which lets the user keep their config around
        but turn it off temporarily.
    schema, table, mmsi_col : str
        Required when ``enabled``.  Together they identify the source
        table and the join column.
    loa_col, beam_col, ship_type_col, air_draught_col : str
        Optional per-output-field column names.  A blank entry means
        "skip this field" — the query emits NULL for ship_type /
        air_draught, and falls back to the AIS dim arithmetic for
        loa / beam.
    """

    enabled: bool = False
    schema: str = ""
    table: str = ""
    mmsi_col: str = "mmsi"
    loa_col: str = ""
    beam_col: str = ""
    ship_type_col: str = ""
    air_draught_col: str = ""

    # ------------------------------------------------------------ validation

    def is_valid(self) -> bool:
        """True when the JOIN is usable.

        Schema, table and MMSI column are mandatory and must be plain SQL
        identifiers.  Each of the optional output columns must EITHER be
        blank (meaning "skip") OR a plain SQL identifier — we never
        accept "interesting" strings that could escape the query.

        ``enabled=False`` short-circuits to False so callers can write a
        single ``if cfg.is_valid():`` check.
        """
        if not self.enabled:
            return False
        for required in (self.schema, self.table, self.mmsi_col):
            if not _is_ident(required):
                return False
        for optional in (
            self.loa_col, self.beam_col,
            self.ship_type_col, self.air_draught_col,
        ):
            if optional and not _is_ident(optional):
                return False
        # If every output column is blank, there's nothing to fetch — the
        # JOIN would produce only NULLs.  Treat that as "not configured"
        # so the caller skips the JOIN entirely.
        if not any((
            self.loa_col, self.beam_col,
            self.ship_type_col, self.air_draught_col,
        )):
            return False
        return True

    # ------------------------------------------------------------ SQL builder

    def build_cte(self) -> str:
        """Return the ``external_vessels AS (...)`` CTE body for the JOIN.

        Caller is responsible for checking :meth:`is_valid` first; this
        method assumes all identifiers are already validated.  Output
        column names are stable so the consuming SELECT can reference
        them: ``ext_loa``, ``ext_beam``, ``ext_ship_type``,
        ``ext_air_draught``.

        For each optional column that is blank, we emit a literal NULL
        of the appropriate type rather than skipping the alias — that
        way the consuming SELECT's ``ext.ext_loa`` etc. always resolves.
        """
        loa = self.loa_col or "NULL::double precision"
        beam = self.beam_col or "NULL::double precision"
        ship = self.ship_type_col or "NULL::int"
        air = self.air_draught_col or "NULL::double precision"
        return (
            "external_vessels AS ("
            f"SELECT {self.mmsi_col} AS mmsi, "
            f"{loa} AS ext_loa, "
            f"{beam} AS ext_beam, "
            f"{ship} AS ext_ship_type, "
            f"{air} AS ext_air_draught "
            f"FROM {self.schema}.{self.table})"
        )

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VesselLookupConfig":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    @classmethod
    def from_qsettings(cls) -> "VesselLookupConfig":
        if not _HAS_QT:
            return cls()
        s = QSettings()
        # ``QSettings.value`` returns Python bools for type=bool but also
        # accepts the string "true"/"false" (Windows registry quirk) —
        # ``str(...).lower() == "true"`` covers both.
        enabled = s.value(f"{_PREFIX}/enabled", False)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() == "true"
        return cls(
            enabled=bool(enabled),
            schema=str(s.value(f"{_PREFIX}/schema", "") or ""),
            table=str(s.value(f"{_PREFIX}/table", "") or ""),
            mmsi_col=str(s.value(f"{_PREFIX}/mmsi_col", "mmsi") or "mmsi"),
            loa_col=str(s.value(f"{_PREFIX}/loa_col", "") or ""),
            beam_col=str(s.value(f"{_PREFIX}/beam_col", "") or ""),
            ship_type_col=str(s.value(f"{_PREFIX}/ship_type_col", "") or ""),
            air_draught_col=str(s.value(f"{_PREFIX}/air_draught_col", "") or ""),
        )

    def to_qsettings(self) -> None:
        if not _HAS_QT:
            return
        s = QSettings()
        s.setValue(f"{_PREFIX}/enabled", bool(self.enabled))
        s.setValue(f"{_PREFIX}/schema", self.schema)
        s.setValue(f"{_PREFIX}/table", self.table)
        s.setValue(f"{_PREFIX}/mmsi_col", self.mmsi_col)
        s.setValue(f"{_PREFIX}/loa_col", self.loa_col)
        s.setValue(f"{_PREFIX}/beam_col", self.beam_col)
        s.setValue(f"{_PREFIX}/ship_type_col", self.ship_type_col)
        s.setValue(f"{_PREFIX}/air_draught_col", self.air_draught_col)
