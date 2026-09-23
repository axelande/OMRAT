"""Custom ship-type mapping keyed by IMO number or MMSI.

The AIS Type-5 ``type_and_cargo`` code is often wrong or missing, and
some studies need a specific hull in a specific OMRAT category (a
ferry that broadcasts *Cargo*, a tanker in ballast reported as
*Other*).  This module lets the user keep a small mapping table in the
AIS PostgreSQL database::

    <schema>.<table> (mmsi bigint, imo bigint, ship_type smallint, note text)

``ship_type`` holds the OMRAT category index (0-20, see
:data:`SHIP_TYPE_NAMES`).  The passage query LEFT JOINs the table twice
and resolves the category as

    IMO match  ->  MMSI match  ->  external vessel lookup  ->  AIS code

The table is normally filled from a CSV through the AIS connection
dialog (:func:`read_mapping_csv` + :func:`import_mapping`); any tool
that writes the same four columns works as well.

Everything here except ``from_qsettings`` / ``to_qsettings`` runs
without Qt or QGIS so it can be unit tested standalone.
"""
from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from string import Template as _Template
from typing import Any, Iterable

try:  # pragma: no cover - import guard exercised only in non-Qt envs
    from qgis.PyQt.QtCore import QSettings
    _HAS_QT = True
except Exception:
    QSettings = None  # type: ignore[assignment]
    _HAS_QT = False


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PREFIX = "omrat/ship_type_map"
DEFAULT_TABLE = "ship_type_map"

#: OMRAT ship categories in traffic-matrix row order.  Index == the
#: value stored in the mapping table and returned by :func:`get_type`.
SHIP_TYPE_NAMES: tuple[str, ...] = (
    'Fishing', 'Towing', 'Dredging or underwater ops', 'Diving ops', 'Military ops',
    'Sailing', 'Pleasure Craft', 'High speed craft (HSC)', 'Pilot Vessel',
    'Search and Rescue vessel', 'Tug', 'Port Tender', 'Anti-pollution equipment',
    'Law Enforcement', 'Spare', 'Medical Transport',
    'Noncombatant ship according to RR Resolution No. 18',
    'Passenger, all ships of this type', 'Cargo, all ships of this type',
    'Tanker, all ships of this type', 'Other Type, all ships of this type',
)
N_SHIP_TYPES = len(SHIP_TYPE_NAMES)
OTHER_TYPE = N_SHIP_TYPES - 1

# Short spellings accepted in CSV files next to the full names above.
_NAME_ALIASES: dict[str, int] = {
    'fishing': 0, 'fish': 0,
    'towing': 1,
    'dredging': 2, 'dredger': 2, 'underwater ops': 2,
    'diving': 3, 'diving ops': 3,
    'military': 4, 'military ops': 4, 'naval': 4,
    'sailing': 5, 'sail': 5,
    'pleasure': 6, 'pleasure craft': 6, 'yacht': 6,
    'hsc': 7, 'high speed craft': 7, 'high speed': 7, 'fast ferry': 7,
    'pilot': 8, 'pilot vessel': 8,
    'sar': 9, 'search and rescue': 9, 'rescue': 9,
    'tug': 10,
    'port tender': 11, 'tender': 11,
    'anti-pollution': 12, 'anti pollution': 12, 'antipollution': 12,
    'law enforcement': 13, 'police': 13, 'coast guard': 13,
    'spare': 14,
    'medical': 15, 'medical transport': 15,
    'noncombatant': 16, 'non-combatant': 16,
    'passenger': 17, 'ferry': 17, 'ro-pax': 17, 'ropax': 17, 'cruise': 17,
    'cargo': 18, 'container': 18, 'bulk': 18, 'bulk carrier': 18, 'general cargo': 18,
    'ro-ro': 18, 'roro': 18,
    'tanker': 19, 'oil tanker': 19, 'chemical tanker': 19, 'lng': 19, 'lpg': 19,
    'gas carrier': 19, 'product tanker': 19,
    'other': 20, 'other type': 20, 'unknown': 20,
}


def _is_ident(value: str) -> bool:
    return bool(value) and bool(_IDENT_RE.match(value))


# --------------------------------------------------------------------------- types

def get_type(toc: Any) -> int:
    """Return the OMRAT category index for an AIS Type-of-Cargo code.

    Maps AIS Type-of-Cargo (TOC) codes to indices 0-20 corresponding to:
        0: Fishing (TOC 30)
        1: Towing (TOC 31-32)
        2: Dredging or underwater ops (TOC 33)
        3: Diving ops (TOC 34)
        4: Military ops (TOC 35)
        5: Sailing (TOC 36)
        6: Pleasure Craft (TOC 37)
        7: High speed craft (TOC 40-49)
        8: Pilot Vessel (TOC 50)
        9: Search and Rescue vessel (TOC 51)
        10: Tug (TOC 52)
        11: Port Tender (TOC 53)
        12: Anti-pollution equipment (TOC 54)
        13: Law Enforcement (TOC 55)
        14: Spare (TOC 56-57)
        15: Medical Transport (TOC 58)
        16: Noncombatant ship (TOC 59)
        17: Passenger (TOC 60-69)
        18: Cargo (TOC 70-79)
        19: Tanker (TOC 80-89)
        20: Other Type (everything else)

    NULL / unparseable codes are common (any MMSI whose Type-5 statics
    never came through) and land in *Other Type* rather than raising.
    """
    if toc is None:
        return OTHER_TYPE
    try:
        toc_int = int(toc)
    except (TypeError, ValueError):
        return OTHER_TYPE
    _TOC_MAP = {
        30: 0, 31: 1, 32: 1, 33: 2, 34: 3, 35: 4, 36: 5, 37: 6,
        50: 8, 51: 9, 52: 10, 53: 11, 54: 12, 55: 13,
        56: 14, 57: 14, 58: 15, 59: 16,
    }
    if toc_int in _TOC_MAP:
        return _TOC_MAP[toc_int]
    if 40 <= toc_int <= 49:
        return 7
    if 60 <= toc_int <= 69:
        return 17
    if 70 <= toc_int <= 79:
        return 18
    if 80 <= toc_int <= 89:
        return 19
    return OTHER_TYPE


def parse_ship_type(value: Any) -> int | None:
    """Turn a mapping-file cell into an OMRAT category index.

    Accepted spellings:

    * an integer ``0..20`` -- the OMRAT index itself;
    * an integer ``30..99`` -- an AIS type code, converted with
      :func:`get_type` (the two ranges do not overlap);
    * a category name, full (``Tanker, all ships of this type``) or
      short (``tanker``, ``ferry``, ``hsc`` ...), case-insensitive.

    Returns ``None`` for anything else (blank, negative, ``21..29``,
    unknown words) so the caller can report the row.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        num: float | None = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            num = float(text)
        except ValueError:
            num = None
    if num is not None:
        if num != int(num):
            return None
        code = int(num)
        if 0 <= code < N_SHIP_TYPES:
            return code
        if 30 <= code <= 99:
            return get_type(code)
        return None
    key = re.sub(r"\s+", " ", text.lower()).strip()
    for idx, name in enumerate(SHIP_TYPE_NAMES):
        if key == name.lower():
            return idx
    if key in _NAME_ALIASES:
        return _NAME_ALIASES[key]
    for idx, name in enumerate(SHIP_TYPE_NAMES):
        if name.lower().startswith(key):
            return idx
    return None


def resolve_ship_type(mapped: Any, toc: Any) -> int:
    """Category index for one AIS row: the mapped value wins over the code.

    ``mapped`` is whatever the passage query returned in its
    ``ship_type`` column (custom mapping, external vessel lookup or
    NULL).  It goes through :func:`parse_ship_type`, so both an OMRAT
    index and an AIS code are understood; anything unparseable falls
    back to :func:`get_type` on ``toc``.
    """
    idx = parse_ship_type(mapped)
    if idx is None:
        return get_type(toc)
    return idx


# --------------------------------------------------------------------------- config

@dataclass
class ShipTypeMapConfig:
    """Where the mapping table lives.

    ``enabled`` is the master switch; ``schema`` and ``table`` must be
    plain SQL identifiers (they are interpolated into the query, so they
    are whitelisted rather than bound).
    """

    enabled: bool = False
    schema: str = ""
    table: str = DEFAULT_TABLE

    def is_valid(self) -> bool:
        return bool(self.enabled) and _is_ident(self.schema) and _is_ident(self.table)

    def qualified_name(self) -> str:
        return f"{self.schema}.{self.table}"

    # ------------------------------------------------------------ SQL fragments

    # ``string.Template`` rather than f-strings so Bandit does not flag
    # the build; every substituted value is a validated identifier.
    _CTE_TEMPLATE = _Template(
        "type_map_mmsi AS ("
        "SELECT DISTINCT ON (mmsi) mmsi, ship_type FROM $schema.$table "
        "WHERE mmsi IS NOT NULL ORDER BY mmsi, ship_type), "
        "type_map_imo AS ("
        "SELECT DISTINCT ON (imo) imo, ship_type FROM $schema.$table "
        "WHERE imo IS NOT NULL ORDER BY imo, ship_type)"
    )

    def build_ctes(self) -> str:
        """Two de-duplicated lookups, one per key.

        ``DISTINCT ON`` guarantees a ping is never multiplied by a
        mapping table that lists the same MMSI or IMO twice.
        """
        if not self.is_valid():
            raise ValueError("ship type mapping is not configured")
        return self._CTE_TEMPLATE.substitute(schema=self.schema, table=self.table)

    @staticmethod
    def build_joins(with_imo: bool) -> str:
        joins = ""
        if with_imo:
            joins += " LEFT OUTER JOIN type_map_imo tmi ON tmi.imo = ss.imo_num"
        joins += " LEFT OUTER JOIN type_map_mmsi tmm ON tmm.mmsi = ss.mmsi"
        return joins

    @staticmethod
    def build_ship_type_expr(fallback_expr: str, with_imo: bool) -> str:
        """``COALESCE(<imo match>, <mmsi match>, <fallback>)``."""
        parts = []
        if with_imo:
            parts.append("tmi.ship_type")
        parts.append("tmm.ship_type")
        parts.append(fallback_expr)
        return "COALESCE(" + ", ".join(parts) + ")"

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShipTypeMapConfig":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    @classmethod
    def from_qsettings(cls) -> "ShipTypeMapConfig":
        if not _HAS_QT:
            return cls()
        s = QSettings()
        enabled = s.value(f"{_PREFIX}/enabled", False)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() == "true"
        return cls(
            enabled=bool(enabled),
            schema=str(s.value(f"{_PREFIX}/schema", "") or ""),
            table=str(s.value(f"{_PREFIX}/table", DEFAULT_TABLE) or DEFAULT_TABLE),
        )

    def to_qsettings(self) -> None:
        if not _HAS_QT:
            return
        s = QSettings()
        s.setValue(f"{_PREFIX}/enabled", bool(self.enabled))
        s.setValue(f"{_PREFIX}/schema", self.schema)
        s.setValue(f"{_PREFIX}/table", self.table)


# --------------------------------------------------------------------------- CSV

_MMSI_HEADERS = ('mmsi',)
# Compared after lower-casing and turning spaces / dashes into underscores.
_IMO_HEADERS = ('imo', 'imo_num', 'imo_number', 'imonumber', 'imo_no')
_TYPE_HEADERS = ('ship_type', 'shiptype', 'type', 'category', 'omrat_type', 'omrat_category',
                 'ship_category')
_NOTE_HEADERS = ('note', 'notes', 'name', 'vessel_name', 'ship_name', 'comment')


@dataclass
class MappingRow:
    mmsi: int | None
    imo: int | None
    ship_type: int
    note: str | None = None

    def as_params(self) -> tuple[int | None, int | None, int, str | None]:
        return (self.mmsi, self.imo, self.ship_type, self.note)


def _find_column(headers: list[str], candidates: Iterable[str]) -> int | None:
    norm = [re.sub(r'[\s-]+', '_', h.strip().lower()) for h in headers]
    for cand in candidates:
        if cand in norm:
            return norm.index(cand)
    return None


def _parse_id(value: str | None) -> int | None:
    """MMSI / IMO cell -> int, ``None`` when blank or zero."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper().startswith("IMO"):
        text = text[3:].strip()
    try:
        num = int(float(text))
    except ValueError:
        raise ValueError(f"not a number: {value!r}")
    if num <= 0:
        return None
    return num


def read_mapping_csv(path: str, encoding: str = "utf-8-sig") -> tuple[list[MappingRow], list[str]]:
    """Read ``mmsi`` / ``imo`` / ``ship_type`` (+ optional ``note``) rows.

    Header names are matched case-insensitively; ``;``, ``,`` and tab
    delimiters are detected.  A row needs at least one of MMSI / IMO
    and a resolvable ship type (see :func:`parse_ship_type`).  Bad rows
    are skipped and described in the returned error list, so the caller
    can show them and still import the good ones.
    """
    rows: list[MappingRow] = []
    errors: list[str] = []
    with open(path, "r", encoding=encoding, newline="") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(fh, dialect)
        try:
            headers = next(reader)
        except StopIteration:
            return rows, ["The file is empty"]
        mmsi_i = _find_column(headers, _MMSI_HEADERS)
        imo_i = _find_column(headers, _IMO_HEADERS)
        type_i = _find_column(headers, _TYPE_HEADERS)
        note_i = _find_column(headers, _NOTE_HEADERS)
        if type_i is None:
            return rows, [
                "No ship type column found (expected one of: " + ", ".join(_TYPE_HEADERS) + ")"
            ]
        if mmsi_i is None and imo_i is None:
            return rows, ["Neither an 'mmsi' nor an 'imo' column was found"]
        for line_no, rec in enumerate(reader, start=2):
            if not any(cell.strip() for cell in rec):
                continue

            def cell(i: int | None) -> str | None:
                return rec[i] if i is not None and i < len(rec) else None

            try:
                mmsi = _parse_id(cell(mmsi_i))
                imo = _parse_id(cell(imo_i))
            except ValueError as exc:
                errors.append(f"line {line_no}: {exc}")
                continue
            if mmsi is None and imo is None:
                errors.append(f"line {line_no}: no MMSI or IMO")
                continue
            ship_type = parse_ship_type(cell(type_i))
            if ship_type is None:
                errors.append(f"line {line_no}: unknown ship type {cell(type_i)!r}")
                continue
            note_raw = cell(note_i)
            note = note_raw.strip() or None if note_raw is not None else None
            rows.append(MappingRow(mmsi=mmsi, imo=imo, ship_type=ship_type, note=note))
    return rows, errors


# --------------------------------------------------------------------------- database

def import_mapping(db: Any, cfg: ShipTypeMapConfig, rows: list[MappingRow],
                   replace: bool = True, chunk: int = 500) -> int:
    """Create the mapping table if needed and load ``rows`` into it.

    ``db`` is a :class:`compute.database.DB` (anything exposing
    ``conn`` with a psycopg2-style cursor works).  With ``replace`` the
    table is emptied first; otherwise rows are appended.  Runs in one
    transaction and returns the number of rows written.
    """
    from psycopg2 import sql as psql

    if not cfg.is_valid():
        raise ValueError("Ship type mapping: schema and table must be plain SQL identifiers")
    schema = psql.Identifier(cfg.schema)
    table = psql.Identifier(cfg.table)
    conn = db.conn
    cur = conn.cursor()
    try:
        cur.execute(psql.SQL(
            "CREATE TABLE IF NOT EXISTS {schema}.{table} ("
            "mmsi bigint, imo bigint, ship_type smallint NOT NULL, note text)"
        ).format(schema=schema, table=table))
        for col in ("mmsi", "imo"):
            cur.execute(psql.SQL(
                "CREATE INDEX IF NOT EXISTS {idx} ON {schema}.{table} ({col})"
            ).format(
                idx=psql.Identifier(f"{cfg.table}_{col}_idx"),
                schema=schema, table=table, col=psql.Identifier(col),
            ))
        if replace:
            cur.execute(psql.SQL("TRUNCATE {schema}.{table}").format(schema=schema, table=table))
        written = 0
        for start in range(0, len(rows), chunk):
            batch = rows[start:start + chunk]
            values = psql.SQL(", ").join([psql.SQL("(%s, %s, %s, %s)")] * len(batch))
            params: list[Any] = []
            for r in batch:
                params.extend(r.as_params())
            cur.execute(
                psql.SQL("INSERT INTO {schema}.{table} (mmsi, imo, ship_type, note) VALUES ")
                .format(schema=schema, table=table) + values,
                params,
            )
            written += len(batch)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return written


def fetch_mapping_rows(db: Any, cfg: ShipTypeMapConfig, limit: int | None = 1000) -> list[MappingRow] | None:
    """Rows of the mapping table (IMO-keyed first), ``None`` when unreadable."""
    from psycopg2 import sql as psql

    if db is None or not cfg.is_valid():
        return None
    query = psql.SQL(
        "SELECT mmsi, imo, ship_type, note FROM {schema}.{table} "
        "ORDER BY imo NULLS LAST, mmsi NULLS LAST"
    ).format(schema=psql.Identifier(cfg.schema), table=psql.Identifier(cfg.table))
    params: tuple | None = None
    if limit is not None:
        query = query + psql.SQL(" LIMIT %s")
        params = (int(limit),)
    ok, data = db.execute_and_return(query, return_error=True, params=params)
    if not ok:
        return None
    rows: list[MappingRow] = []
    for mmsi, imo, ship_type, note in data:
        rows.append(MappingRow(
            mmsi=None if mmsi is None else int(mmsi),
            imo=None if imo is None else int(imo),
            ship_type=int(ship_type),
            note=None if note is None else str(note),
        ))
    return rows


def write_mapping_csv(path: str, rows: Iterable[MappingRow]) -> None:
    """Write rows as ``mmsi,imo,ship_type,ship_type_name,note``.

    The ``ship_type_name`` column is informational; :func:`read_mapping_csv`
    picks ``ship_type`` first, so the file round-trips unchanged.
    """
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mmsi", "imo", "ship_type", "ship_type_name", "note"])
        for r in rows:
            name = SHIP_TYPE_NAMES[r.ship_type] if 0 <= r.ship_type < N_SHIP_TYPES else ""
            writer.writerow([
                "" if r.mmsi is None else r.mmsi,
                "" if r.imo is None else r.imo,
                r.ship_type, name, r.note or "",
            ])


def count_mapping(db: Any, cfg: ShipTypeMapConfig) -> tuple[int, int, int] | None:
    """``(rows, with_imo, with_mmsi)`` or ``None`` when the table is unreachable."""
    from psycopg2 import sql as psql

    if db is None or not cfg.is_valid():
        return None
    query = psql.SQL(
        "SELECT count(*), count(imo), count(mmsi) FROM {schema}.{table}"
    ).format(schema=psql.Identifier(cfg.schema), table=psql.Identifier(cfg.table))
    res = db.execute_and_return(query, return_error=True)
    ok, data = res
    if not ok or not data:
        return None
    total, n_imo, n_mmsi = data[0]
    return int(total), int(n_imo), int(n_mmsi)


def statics_have_imo(db: Any, schema: str, year: int) -> bool:
    """True when ``<schema>.statics_<year>`` has an ``imo_num`` column.

    Databases built by OMRAT's ingestion wizard have it; older or
    third-party schemas may not, in which case the passage query joins
    the mapping on MMSI only.
    """
    if db is None:
        return False
    res = db.execute_and_return(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s AND column_name = 'imo_num'",
        return_error=True, params=(schema, f"statics_{int(year)}"),
    )
    ok, data = res
    return bool(ok and data)
