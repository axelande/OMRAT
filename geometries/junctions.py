"""Junction registry and transition-matrix derivation.

A *junction* is a point where two or more legs meet (i.e. a coordinate
shared by at least two leg endpoints).  Each junction carries a
**transition matrix** ``T[in_leg_id][out_leg_id] -> fraction`` describing
the share of traffic that, after arriving at the junction along
``in_leg``, continues out along ``out_leg``.  Rows must sum to 1.0
(modulo the floating-point tolerance below).

The matrix has three possible provenances:

* ``"ais"`` — derived by walking AIS tracks across both legs' near-
  junction zones.
* ``"geometry"`` — fallback heuristic using the deflection angle between
  inbound and outbound bearings.  Used when AIS data are absent.
* ``"user"`` — entered or edited via the matrix UI; the validation pass
  preserves these and never overwrites them.

This module is QGIS-free so the matrix-derivation logic can be tested
under the standalone interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, degrees, exp, radians
from typing import Any, Iterable

from geometries.route_validation import parse_wkt_point


# Tolerance for treating two coordinates as the same junction.  Tighter
# than the close-waypoint snap tolerance because by the time the
# junction registry is built, snaps have already been applied.
_COORD_EPS = 1e-7

# Tolerance for accepting a row-sum as "close enough to 1.0".  Generous
# enough to swallow floating-point noise from heuristic normalisation.
_ROW_SUM_TOL = 1e-6


# ---------------------------------------------------------------------------
# Junction dataclass
# ---------------------------------------------------------------------------


@dataclass
class Junction:
    """A point where two or more legs meet.

    ``legs`` records, for each connected leg, which side of that leg
    touches the junction (``"start"`` or ``"end"``).  ``transitions`` is
    the row-stochastic matrix of in-leg -> out-leg shares; missing
    entries default to 0.
    """

    junction_id: str
    point: tuple[float, float]
    legs: dict[str, str] = field(default_factory=dict)
    transitions: dict[str, dict[str, float]] = field(default_factory=dict)
    source: str = "geometry"

    def degree(self) -> int:
        """Number of legs meeting at this junction."""
        return len(self.legs)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            'point': list(self.point),
            'legs': dict(self.legs),
            'transitions': {
                k: dict(v) for k, v in self.transitions.items()
            },
            'source': self.source,
        }

    @classmethod
    def from_dict(cls, junction_id: str, data: dict[str, Any]) -> "Junction":
        """Inverse of :meth:`to_dict`; tolerant of missing keys."""
        pt_raw = data.get('point', [0.0, 0.0])
        try:
            pt = (float(pt_raw[0]), float(pt_raw[1]))
        except (TypeError, ValueError, IndexError):
            pt = (0.0, 0.0)
        legs = {
            str(k): str(v) for k, v in (data.get('legs') or {}).items()
        }
        trans: dict[str, dict[str, float]] = {}
        for in_leg, row in (data.get('transitions') or {}).items():
            inner: dict[str, float] = {}
            for out_leg, frac in (row or {}).items():
                try:
                    inner[str(out_leg)] = float(frac)
                except (TypeError, ValueError):
                    continue
            trans[str(in_leg)] = inner
        return cls(
            junction_id=str(junction_id),
            point=pt,
            legs=legs,
            transitions=trans,
            source=str(data.get('source', 'geometry')),
        )


# ---------------------------------------------------------------------------
# Junction id helpers
# ---------------------------------------------------------------------------


def junction_id_for_point(point: tuple[float, float]) -> str:
    """Stable string id derived from the coordinate.

    Six-decimal precision matches OMRAT's WKT formatter so an id stays
    invariant across save / reload cycles (assuming the underlying
    waypoint hasn't actually moved).
    """
    return f"j_{point[0]:.6f}_{point[1]:.6f}"


def _points_equal(
    a: tuple[float, float], b: tuple[float, float],
    tol: float = _COORD_EPS,
) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


# ---------------------------------------------------------------------------
# Bearings
# ---------------------------------------------------------------------------


def _outward_bearing(
    junction_pt: tuple[float, float],
    other_pt: tuple[float, float],
) -> float:
    """Compass bearing (deg, 0=N, CW) from ``junction_pt`` to ``other_pt``.

    Uses planar arctan in the lon/lat plane — adequate at the scale of
    the deflection-angle heuristic, which is itself only used for
    rough defaults that the user can override.
    """
    dx = other_pt[0] - junction_pt[0]
    dy = other_pt[1] - junction_pt[1]
    return degrees(atan2(dx, dy)) % 360.0


def _angular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two compass bearings, in [0, 180]."""
    d = (a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d


def deflection_deg(in_bearing_outward: float, out_bearing_outward: float) -> float:
    """Heading change for a ship that comes in along ``in`` and leaves along ``out``.

    Both bearings point *outward* from the junction (i.e. toward each
    leg's far endpoint).  A ship arriving along the in-leg has the
    opposite heading (``in_bearing - 180``), so the heading change is::

        |((out - (in - 180)) wrapped to [-180, 180])|
      = |angular_diff(out, in - 180)|

    A straight continuation has a deflection of 0; a hairpin reversal
    has 180.
    """
    in_arrival = (in_bearing_outward + 180.0) % 360.0
    return _angular_diff_deg(out_bearing_outward, in_arrival)


# ---------------------------------------------------------------------------
# Build registry from segment_data
# ---------------------------------------------------------------------------


def build_junctions(
    segment_data: dict[str, Any],
    tol: float = _COORD_EPS,
) -> dict[str, Junction]:
    """Discover every junction in ``segment_data``.

    A junction is any coordinate that appears as the endpoint of at
    least two distinct legs (or twice in a single leg, which would be
    a degenerate self-loop — ignored).  Returned junctions have empty
    ``transitions`` matrices; call :func:`apply_geometric_defaults`
    to populate them.
    """
    # Group endpoints by exact coordinate (within tol).
    seen: list[tuple[tuple[float, float], dict[str, str]]] = []

    def _add(loc: tuple[float, float], leg_id: str, side: str) -> None:
        for stored_pt, leg_map in seen:
            if _points_equal(stored_pt, loc, tol):
                # Last-write-wins is fine because a leg only ever has
                # one start and one end.
                leg_map[leg_id] = side
                return
        seen.append((loc, {leg_id: side}))

    for leg_id, seg in (segment_data or {}).items():
        if not isinstance(seg, dict):
            continue
        sp = parse_wkt_point(seg.get('Start_Point'))
        ep = parse_wkt_point(seg.get('End_Point'))
        if sp is not None:
            _add(sp, str(leg_id), 'start')
        if ep is not None:
            _add(ep, str(leg_id), 'end')

    junctions: dict[str, Junction] = {}
    for pt, leg_map in seen:
        if len(leg_map) < 2:
            continue
        jid = junction_id_for_point(pt)
        junctions[jid] = Junction(
            junction_id=jid,
            point=pt,
            legs=dict(leg_map),
        )
    return junctions


# ---------------------------------------------------------------------------
# Geometric default matrix
# ---------------------------------------------------------------------------


def _leg_outward_bearing(
    junction: Junction, leg_id: str, segment_data: dict[str, Any],
) -> float | None:
    """Bearing from the junction toward ``leg_id``'s far endpoint."""
    seg = (segment_data or {}).get(leg_id)
    if not isinstance(seg, dict):
        return None
    sp = parse_wkt_point(seg.get('Start_Point'))
    ep = parse_wkt_point(seg.get('End_Point'))
    if sp is None or ep is None:
        return None
    side = junction.legs.get(leg_id)
    if side == 'start':
        # Junction is at this leg's start, so far end = End_Point.
        return _outward_bearing(junction.point, ep)
    elif side == 'end':
        return _outward_bearing(junction.point, sp)
    return None


def compute_geometric_transition_matrix(
    junction: Junction,
    segment_data: dict[str, Any],
    *,
    deflection_scale_deg: float = 30.0,
) -> dict[str, dict[str, float]]:
    """Per-(in, out) transition shares from inbound/outbound bearings.

    For each pair (in_leg, out_leg) with ``in_leg != out_leg`` the
    deflection angle is mapped through ``exp(-deflection / scale)``
    so a straight continuation gets the highest score, a hairpin the
    lowest.  Scores are normalised per row to sum to 1.

    Special case: if the junction has exactly two legs, the result is
    the trivial 100/100 matrix regardless of the bearings — there's
    only one place to go.
    """
    bearings: dict[str, float] = {}
    for leg_id in junction.legs:
        b = _leg_outward_bearing(junction, leg_id, segment_data)
        if b is not None:
            bearings[leg_id] = b
    matrix: dict[str, dict[str, float]] = {}
    if not bearings:
        return matrix
    leg_ids = list(bearings.keys())
    if len(leg_ids) == 2:
        a, b = leg_ids
        matrix[a] = {b: 1.0}
        matrix[b] = {a: 1.0}
        return matrix
    for in_leg in leg_ids:
        scores: dict[str, float] = {}
        for out_leg in leg_ids:
            if out_leg == in_leg:
                continue
            d = deflection_deg(bearings[in_leg], bearings[out_leg])
            scores[out_leg] = exp(-d / max(deflection_scale_deg, 1e-3))
        total = sum(scores.values())
        if total <= 0:
            # Pathological: all-zero scores → uniform distribution.
            n = len(scores)
            matrix[in_leg] = {k: 1.0 / n for k in scores}
        else:
            matrix[in_leg] = {k: v / total for k, v in scores.items()}
    return matrix


def apply_geometric_defaults(
    junctions: dict[str, Junction],
    segment_data: dict[str, Any],
    *,
    overwrite_user: bool = False,
) -> int:
    """Fill in transitions for any junction whose source isn't ``user``.

    Returns the number of junctions actually updated.  ``user`` rows are
    preserved unless ``overwrite_user=True``.
    """
    updated = 0
    for j in junctions.values():
        if j.source == 'user' and not overwrite_user:
            continue
        new_matrix = compute_geometric_transition_matrix(j, segment_data)
        if new_matrix:
            j.transitions = new_matrix
            j.source = 'geometry'
            updated += 1
    return updated


# ---------------------------------------------------------------------------
# AIS-derived defaults — pure-data shape
# ---------------------------------------------------------------------------


def transition_matrix_from_counts(
    counts: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Normalise per-row to convert raw transition counts into shares.

    ``counts[in_leg][out_leg] = number of AIS tracks that went in_leg -> out_leg``.
    Empty rows (``sum == 0``) are preserved as empty rows so callers
    can detect "no AIS evidence" and fall back to the geometric default.
    """
    out: dict[str, dict[str, float]] = {}
    for in_leg, row in counts.items():
        total = sum(v for v in row.values() if v > 0)
        if total <= 0:
            out[in_leg] = {}
            continue
        out[in_leg] = {k: v / total for k, v in row.items() if v > 0}
    return out


def apply_ais_defaults(
    junctions: dict[str, Junction],
    counts_by_junction: dict[str, dict[str, dict[str, float]]],
    segment_data: dict[str, Any],
    *,
    overwrite_user: bool = False,
) -> int:
    """Replace each non-user junction's matrix with AIS-derived shares.

    ``counts_by_junction`` is keyed by ``junction_id`` and contains the
    raw transition-count tables emitted by :func:`AIS.junction_counts`
    (or whichever AIS query the host plugin uses).  Junctions without
    any rows are left untouched so the geometric default still applies.
    """
    updated = 0
    for jid, j in junctions.items():
        if j.source == 'user' and not overwrite_user:
            continue
        raw = counts_by_junction.get(jid)
        if not raw:
            continue
        normalised = transition_matrix_from_counts(raw)
        # If every row is empty after normalisation, treat as no data.
        if all(not v for v in normalised.values()):
            continue
        # Fill in any missing rows with a geometric fallback so the
        # matrix is fully specified.
        geo = compute_geometric_transition_matrix(j, segment_data)
        merged: dict[str, dict[str, float]] = {}
        for leg_id in j.legs:
            row_ais = normalised.get(leg_id, {})
            if row_ais:
                merged[leg_id] = row_ais
            else:
                merged[leg_id] = geo.get(leg_id, {})
        j.transitions = merged
        j.source = 'ais'
        updated += 1
    return updated


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class JunctionWarning:
    junction_id: str
    leg_id: str
    kind: str  # "row_sum" | "missing_row" | "unknown_leg"
    detail: str


def validate_junctions(
    junctions: dict[str, Junction],
    segment_data: dict[str, Any],
) -> list[JunctionWarning]:
    """Spot-check transition-matrix conservation and references.

    Reports junctions where:

    * a row-sum is materially off 1.0 (``> _ROW_SUM_TOL``);
    * an in-leg or out-leg id isn't present in ``segment_data``;
    * a junction has degree >= 2 but a leg with no outgoing row at all.
    """
    warnings: list[JunctionWarning] = []
    sd = segment_data or {}
    for j in junctions.values():
        for leg_id in j.legs:
            if leg_id not in sd:
                warnings.append(JunctionWarning(
                    junction_id=j.junction_id,
                    leg_id=leg_id,
                    kind='unknown_leg',
                    detail=f"leg {leg_id} listed at junction but missing from segment_data",
                ))
        if j.degree() < 2:
            continue
        for in_leg in j.legs:
            row = j.transitions.get(in_leg)
            if not row:
                warnings.append(JunctionWarning(
                    junction_id=j.junction_id,
                    leg_id=in_leg,
                    kind='missing_row',
                    detail=f"no outgoing shares defined for leg {in_leg}",
                ))
                continue
            s = sum(row.values())
            if abs(s - 1.0) > _ROW_SUM_TOL:
                warnings.append(JunctionWarning(
                    junction_id=j.junction_id,
                    leg_id=in_leg,
                    kind='row_sum',
                    detail=f"row sum {s:.6f} != 1.0",
                ))
    return warnings


# ---------------------------------------------------------------------------
# Lookup helpers used by the compute pipeline
# ---------------------------------------------------------------------------


def transition_share(
    junctions: dict[str, Junction],
    junction_pt: tuple[float, float],
    in_leg_id: str,
    out_leg_id: str,
    *,
    default: float = 1.0,
) -> float:
    """Return ``T[in_leg][out_leg]`` for the junction at ``junction_pt``.

    If no junction is registered at that point (or no transition row
    exists for the given in-leg) the function returns ``default``.
    Compute callers pass ``default=1.0`` to recover the legacy
    "all-traffic-counted" behaviour when no junction info is present.
    """
    jid = junction_id_for_point(junction_pt)
    j = junctions.get(jid)
    if j is None:
        # Linear scan fallback for junctions whose stored coords differ
        # by more than the format-rounding could explain.
        for cand in junctions.values():
            if _points_equal(cand.point, junction_pt):
                j = cand
                break
        if j is None:
            return default
    row = j.transitions.get(str(in_leg_id))
    if not row:
        return default
    return float(row.get(str(out_leg_id), 0.0))


def serialize_junctions(
    junctions: dict[str, Junction],
) -> dict[str, dict[str, Any]]:
    """Project a junction registry to a JSON-friendly dict.

    Used by ``omrat_utils/storage.py`` when writing the ``junctions``
    block of an ``.omrat`` file.
    """
    return {jid: j.to_dict() for jid, j in junctions.items()}


def deserialize_junctions(
    payload: dict[str, dict[str, Any]] | None,
) -> dict[str, Junction]:
    """Inverse of :func:`serialize_junctions`."""
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Junction] = {}
    for jid, data in payload.items():
        if not isinstance(data, dict):
            continue
        out[str(jid)] = Junction.from_dict(str(jid), data)
    return out


def refresh_junction_registry(
    junctions: dict[str, Junction],
    segment_data: dict[str, Any],
) -> dict[str, Junction]:
    """Rebuild the registry while preserving user-edited matrices.

    Called after the validation pass (which may have merged or split
    legs) to bring the registry back into sync.  Junctions whose
    ``source`` is ``"user"`` keep their transitions if all referenced
    legs still exist; everything else is regenerated from geometry.
    """
    fresh = build_junctions(segment_data)
    for jid, new_j in fresh.items():
        old = junctions.get(jid)
        if old is not None and old.source == 'user':
            # Preserve user edits when every referenced leg still exists.
            ok = all(
                in_leg in new_j.legs and all(
                    out_leg in new_j.legs for out_leg in row
                )
                for in_leg, row in old.transitions.items()
            )
            if ok and old.transitions:
                new_j.transitions = old.transitions
                new_j.source = 'user'
                continue
        new_j.transitions = compute_geometric_transition_matrix(
            new_j, segment_data,
        )
        new_j.source = 'geometry'
    return fresh
