"""Leg ids versus leg numbers.

OMRAT keys ``segment_data`` by a stringified integer, ``Segment_Id``.
That id is a *global* key: the tangent feature, the layer custom property
and every traffic / distribution lookup use it, so it must never be
reused while a leg with that id exists.  The number the user sees in the
leg name, ``LEG_{route}_{n}``, is a different thing: it restarts at 1 for
every route and is purely cosmetic.

Before v0.15.2 the **Next leg ID** spinbox drove the key.  Setting it to
``1`` for a second route made the next drawn leg overwrite the first
leg's entry in ``segment_data`` and, once an intersection split rebuilt
the canvas from the dict, whole routes disappeared.  The spinbox now
drives the per-route leg *number* and the key is always minted here.

Pure Python, no Qt.
"""
from __future__ import annotations

import re
from typing import Any

# ``LEG_{route}_{n}`` with optional split suffixes (``_a``, ``_b``, ...).
_LEG_NAME_RE = re.compile(r'^LEG_(\d+)_(\d+)(?:_[a-z]+)*$')


def leg_name(route_id: int | str, leg_no: int | str) -> str:
    """Canonical name of leg number ``leg_no`` on route ``route_id``."""
    return f'LEG_{route_id}_{leg_no}'


def next_free_segment_id(segment_data: dict[str, Any] | None, floor: int = 0) -> int:
    """First integer id above every id already in use.

    Looks at both the dict keys and the ``Segment_Id`` values (they should
    agree, but a hand-edited file may disagree) and at ``floor``, the
    last id the drawing handler used, so ids are never handed out twice.
    """
    max_id = int(floor)
    for key, seg in (segment_data or {}).items():
        for candidate in (key, seg.get('Segment_Id') if isinstance(seg, dict) else None):
            try:
                value = int(str(candidate))
            except (TypeError, ValueError):
                continue
            if value > max_id:
                max_id = value
    return max_id + 1


def parse_leg_number(name: Any, route_id: int | str | None = None) -> int | None:
    """Leg number ``n`` from a ``LEG_{route}_{n}`` style name.

    Returns ``None`` when the name does not follow the convention, or when
    ``route_id`` is given and the name belongs to another route.
    """
    m = _LEG_NAME_RE.match(str(name or '').strip())
    if m is None:
        return None
    if route_id is not None and int(m.group(1)) != int(route_id):
        return None
    return int(m.group(2))


def max_leg_number(segment_data: dict[str, Any] | None, route_id: int | str) -> int:
    """Highest leg number already used on ``route_id``; ``0`` when the
    route has no legs yet (so the next leg becomes number 1).

    Only the name is consulted: it is the thing being numbered, and a
    hand-edited ``Route_Id`` must not pull another route's numbers in.
    """
    best = 0
    for seg in (segment_data or {}).values():
        if not isinstance(seg, dict):
            continue
        n = parse_leg_number(seg.get('Leg_name'), route_id)
        if n is not None and n > best:
            best = n
    return best
