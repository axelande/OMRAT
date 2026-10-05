"""Pure-Python helpers for deriving junction transition counts from AIS crossings.

The expensive part -- querying the PostGIS database for the AIS segments
that cross each leg's passage line -- lives in
:mod:`omrat_utils.handle_ais`.  It hands over one :class:`Crossing` per
segment, already sorted into the leg's two directions by COG with the
same rule the traffic binning uses.  Pairing those crossings into
``in_leg -> out_leg`` transits is pure data manipulation, which is what
this module owns.

Crossings are paired per **trip**.  The segments table carries a
``route_id`` per port-to-port trip, so every crossing of one vessel on
one ``route_id`` belongs to the same trip however long it took.  Rows
without a ``route_id`` fall back to the vessel (MMSI), and then the two
crossings must be at most ``time_window_s`` apart.

Within a trip the crossings at a junction are walked in time order: an
*inbound* crossing (heading towards the junction) on one leg directly
followed by an *outbound* crossing on another leg is one transit.
Everything else -- a trip that starts or ends between the passage line
and the junction, a U-turn, a crossing whose partner was not observed --
is unmatched and adds nothing, so the shares are normalised over the
matched transits only.

Splitting the responsibilities like this keeps the database-dependent
code out of the standalone test suite while still letting us exercise
the transition logic end-to-end.
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

# Fallback for crossings without a ``route_id``: the inbound and the
# outbound crossing of one vessel must be at most this far apart to be a
# single junction transit.  Two hours covers most port manoeuvres while
# excluding return trips.
DEFAULT_TIME_WINDOW_S = 2 * 60 * 60


class Crossing(NamedTuple):
    """One AIS segment crossing a leg's passage line."""

    mmsi: str
    route_id: int | None
    t: float
    # True: cog ~ leg bearing (the drawn direction, Dirs[0]); False: reciprocal.
    along_drawn: bool


class MatchSummary(NamedTuple):
    """How much of a junction's AIS evidence was paired into transits."""

    matched: int        # in -> out transits counted
    inbound: int        # crossings heading towards the junction
    with_route_id: int  # crossings that carried a route_id


class _Event(NamedTuple):
    t: float
    outbound: bool  # False sorts first, so inbound wins a tie on t
    leg_id: str


def is_inbound(side: str | None, along_drawn: bool) -> bool | None:
    """Whether a crossing heads towards the junction at ``side`` of its leg.

    Flow in the drawn direction (``Start_Point -> End_Point``) arrives at
    the leg's end, so it is inbound at an ``"end"`` junction and outbound
    at a ``"start"`` one.  ``None`` for an unknown side.
    """
    if side == 'end':
        return along_drawn
    if side == 'start':
        return not along_drawn
    return None


def _trip_events(
    crossings_by_leg: dict[str, list[Crossing]],
    leg_sides: dict[str, str],
) -> dict[tuple[str, int | None], list[_Event]]:
    trips: dict[tuple[str, int | None], list[_Event]] = defaultdict(list)
    for leg_id, crossings in crossings_by_leg.items():
        side = leg_sides.get(leg_id)
        for c in crossings or ():
            inbound = is_inbound(side, c.along_drawn)
            if inbound is None:
                continue
            trips[(str(c.mmsi), c.route_id)].append(_Event(float(c.t), not inbound, leg_id))
    return trips


def transition_counts_from_crossings(
    crossings_by_leg: dict[str, list[Crossing]],
    leg_sides: dict[str, str],
    *,
    time_window_s: float = DEFAULT_TIME_WINDOW_S,
) -> dict[str, dict[str, int]]:
    """Count ``in_leg -> out_leg`` transits at one junction.

    ``crossings_by_leg`` holds the passage-line crossings of every leg
    meeting at the junction and ``leg_sides`` which end of each leg
    touches it (``Junction.legs``).  Crossings are grouped into trips by
    ``(mmsi, route_id)`` and sorted by time; an inbound crossing on leg
    *i* directly followed by an outbound crossing on leg *j != i* counts
    one *i -> j* transit.  A trip that passes the junction twice counts
    twice.  Crossings without a ``route_id`` must also be at most
    ``time_window_s`` apart.

    Every leg in ``crossings_by_leg`` gets a row.  A row without matched
    transits stays empty, so the caller keeps the geometric default.
    """
    out: dict[str, dict[str, int]] = {leg_id: {} for leg_id in crossings_by_leg}
    for (_mmsi, route_id), events in _trip_events(crossings_by_leg, leg_sides).items():
        events.sort()
        for prev, nxt in zip(events, events[1:]):
            if prev.outbound or not nxt.outbound or prev.leg_id == nxt.leg_id:
                continue
            if route_id is None and nxt.t - prev.t > time_window_s:
                continue
            row = out.setdefault(prev.leg_id, {})
            row[nxt.leg_id] = row.get(nxt.leg_id, 0) + 1
    return out


def match_summary(
    crossings_by_leg: dict[str, list[Crossing]],
    leg_sides: dict[str, str],
    counts: dict[str, dict[str, int]],
) -> MatchSummary:
    """Matched transits vs inbound crossings at one junction (for the log)."""
    inbound = 0
    with_route_id = 0
    for leg_id, crossings in crossings_by_leg.items():
        side = leg_sides.get(leg_id)
        for c in crossings or ():
            if is_inbound(side, c.along_drawn):
                inbound += 1
            if c.route_id is not None:
                with_route_id += 1
    matched = sum(sum(row.values()) for row in counts.values())
    return MatchSummary(matched, inbound, with_route_id)


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
