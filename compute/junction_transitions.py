"""Pure-Python helpers for deriving junction transition counts from AIS passages.

The expensive part — querying the PostGIS database for which MMSIs
transit the near-junction zone of each leg — lives in
:mod:`omrat_utils.handle_ais`.  Once that returns ``passages_by_leg``
(one timestamp list per MMSI per leg), the rest of the work is pure
data manipulation, which is what this module owns.

Splitting the responsibilities like this keeps the database-dependent
code out of the standalone test suite while still letting us exercise
the transition logic end-to-end.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

# Default tolerance for matching an inbound and outbound passage as
# belonging to the same trip.  Tracks that pass two legs more than this
# apart are not a single junction transit.  Two hours covers most port
# manoeuvres while excluding return trips.
DEFAULT_TIME_WINDOW_S = 2 * 60 * 60


def transition_counts_from_passages(
    passages_by_leg: dict[str, dict[str, list[float]]],
    *,
    time_window_s: float = DEFAULT_TIME_WINDOW_S,
) -> dict[str, dict[str, int]]:
    """Count MMSI-level (in_leg -> out_leg) transitions across legs.

    ``passages_by_leg[leg_id][mmsi]`` is a list of UNIX-epoch timestamps
    for every AIS ping by that vessel inside ``leg_id``'s near-junction
    zone.  An "in_leg -> out_leg" transition is recorded when the same
    MMSI's earliest timestamp on ``out_leg`` is later than its earliest
    timestamp on ``in_leg`` *and* the gap is no greater than
    ``time_window_s``.  Each MMSI contributes at most one count to
    (in_leg, out_leg).

    Pairs where ``in_leg == out_leg`` are skipped — a ship on the same
    leg twice is a U-turn, not a junction transition.
    """
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    leg_ids = list(passages_by_leg.keys())
    if len(leg_ids) < 2:
        return {leg_id: {} for leg_id in leg_ids}

    # Collect the earliest timestamp per MMSI per leg up front.
    first_seen: dict[str, dict[str, float]] = {}
    for leg_id, by_mmsi in passages_by_leg.items():
        slot: dict[str, float] = {}
        for mmsi, times in (by_mmsi or {}).items():
            if not times:
                continue
            try:
                slot[str(mmsi)] = float(min(times))
            except (TypeError, ValueError):
                continue
        first_seen[leg_id] = slot

    # Identify the universe of MMSIs that touched at least two legs.
    mmsi_legs: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for leg_id, by_mmsi in first_seen.items():
        for mmsi, t in by_mmsi.items():
            mmsi_legs[mmsi].append((leg_id, t))

    for mmsi, hits in mmsi_legs.items():
        if len(hits) < 2:
            continue
        # Sort by timestamp so the first occurrence is the inbound leg.
        hits.sort(key=lambda x: x[1])
        in_leg, in_t = hits[0]
        recorded: set[str] = set()
        for out_leg, out_t in hits[1:]:
            if out_leg == in_leg:
                continue
            if out_leg in recorded:
                continue
            if out_t - in_t > time_window_s:
                # Vessel returned much later -- treat as a separate trip.
                # Reset the inbound slot to the current pass.
                in_leg, in_t = out_leg, out_t
                recorded = set()
                continue
            out[in_leg][out_leg] += 1
            recorded.add(out_leg)

    # Materialise the defaultdict-of-defaultdict for callers.
    result: dict[str, dict[str, int]] = {}
    for leg_id in leg_ids:
        row = out.get(leg_id)
        result[leg_id] = dict(row) if row else {}
    return result


def normalise_counts_to_shares(
    counts: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    """Convenience: row-normalise integer counts to shares in [0, 1].

    Empty rows survive as empty dicts so callers can detect "no AIS
    evidence" and fall back to the geometric default.
    """
    out: dict[str, dict[str, float]] = {}
    for in_leg, row in counts.items():
        total = sum(v for v in row.values() if v > 0)
        if total <= 0:
            out[in_leg] = {}
        else:
            out[in_leg] = {k: v / total for k, v in row.items() if v > 0}
    return out
