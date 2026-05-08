"""Defaults and constants for the oil-spill consequence module.

The Consequence menu adds four project-level data structures:

* ``oil_onboard``        -- m^3 of oil on board, per (ship_type, length_interval).
* ``spill_probability``  -- conditional P(spill_level | accident), per accident,
                            in percent (rows must sum to 100).
* ``spill_fraction``     -- fraction of full tank that ends up as spill at each
                            spill level, per accident, in percent.
* ``catastrophe_levels`` -- user-editable list of named spill thresholds (m^3).

These defaults are seeded when a project is created or loaded without an
existing block, and are also used by the dialog widgets when the underlying
ship-category dimensions change (rows/columns added or removed).
"""
from __future__ import annotations

from typing import Any

# Order matches ``AccidentResultsMixin._ACCIDENT_ROWS`` so the rows of the
# spill-probability and spill-fraction matrices line up with the on-screen
# accident-results table.
ACCIDENT_TYPES: tuple[str, ...] = (
    'Drifting allision',
    'Drifting grounding',
    'Powered allision',
    'Powered grounding',
    'Overtaking collision',
    'Head-on collision',
    'Crossing collision',
    'Merging collision',
)

# Internal accident keys used by ``compute/consequence.py`` to look up the
# per-cell breakdowns that the model mixins emit.  Same order as
# ``ACCIDENT_TYPES`` so ``zip(...)`` is the canonical mapping.
ACCIDENT_KEYS: tuple[str, ...] = (
    'drifting_allision',
    'drifting_grounding',
    'powered_allision',
    'powered_grounding',
    'overtaking',
    'head_on',
    'crossing',
    'merging',
)

SPILL_LEVELS: tuple[str, ...] = (
    'No spill',
    'Minor spill',
    'Major spill',
    'Total loss',
)

# Tanker ship-type label used by ``default_oil_onboard``.  Match against the
# label coming out of the Ship Categories widget; we use a substring check
# so "Tanker", "Chemical/gas tanker", etc. all use the tanker formula.
TANKER_KEYWORDS: tuple[str, ...] = ('tanker',)

DEFAULT_OIL_NON_TANKER_M3: float = 100.0
TANKER_OIL_LENGTH_FACTOR: float = 80.0
OPEN_INTERVAL_DELTA: float = 50.0  # length used for an open-ended top bin


def _interval_average_length(interval: dict[str, Any]) -> float:
    """Return the representative length (m) for an interval entry.

    Interval entries come from ``gather_data.get_ship_categories_for_save``;
    ``min`` / ``max`` are floats when parseable and the original strings
    otherwise (e.g. ``"inf"``, ``"1000+"``).  We treat any non-numeric
    ``max`` as an open interval and use ``min + OPEN_INTERVAL_DELTA``;
    otherwise we use the midpoint of [min, max].
    """
    import math
    raw_min = interval.get('min', 0.0)
    raw_max = interval.get('max', 0.0)
    try:
        vmin = float(raw_min)
    except (TypeError, ValueError):
        vmin = 0.0
    # ``float('inf')`` parses successfully, so we explicitly test for
    # infinite / non-finite values and treat them like an open interval.
    try:
        vmax = float(raw_max)
        if not math.isfinite(vmax):
            return vmin + OPEN_INTERVAL_DELTA
        return (vmin + vmax) / 2.0
    except (TypeError, ValueError):
        # Open-ended top bin -- e.g. max == "1000+" or other non-numeric
        return vmin + OPEN_INTERVAL_DELTA


def _is_tanker(ship_type_label: str) -> bool:
    label = ship_type_label.lower()
    return any(kw in label for kw in TANKER_KEYWORDS)


def default_oil_onboard(
    ship_types: list[str],
    length_intervals: list[dict[str, Any]],
) -> list[list[float]]:
    """Build the default oil-onboard matrix for a project.

    Returns a 2D list shaped ``[ship_type_idx][length_idx]`` with values in m^3:

    * Tankers (label contains "tanker"):  ``80 * average_length``
    * All other ship types:               ``100``

    The shape mirrors ``traffic_data[seg][dir]['Frequency (ships/year)']`` so
    the consequence dialogs can use the same row/column indexing as the
    frequency matrix.
    """
    matrix: list[list[float]] = []
    for stype in ship_types:
        row: list[float] = []
        is_tanker = _is_tanker(stype)
        for interval in length_intervals:
            if is_tanker:
                row.append(TANKER_OIL_LENGTH_FACTOR * _interval_average_length(interval))
            else:
                row.append(DEFAULT_OIL_NON_TANKER_M3)
        matrix.append(row)
    return matrix


def default_spill_probability() -> list[list[float]]:
    """Default conditional spill-level probabilities per accident, in percent.

    Drifting groundings / allisions produce 0% major or total-loss spills
    (low-energy contact); other accident types use ``[97, 1, 1, 1]``.  Every
    row sums to 100.  Index order matches ``ACCIDENT_TYPES``.
    """
    drifting = [98.0, 2.0, 0.0, 0.0]
    other = [97.0, 1.0, 1.0, 1.0]
    return [
        list(drifting),  # Drifting allision
        list(drifting),  # Drifting grounding
        list(other),     # Powered allision
        list(other),     # Powered grounding
        list(other),     # Overtaking
        list(other),     # Head-on
        list(other),     # Crossing
        list(other),     # Merging
    ]


def default_spill_fraction() -> list[list[float]]:
    """Default percent-of-tank-spilt per (accident, spill level).

    Same row for every accident type: ``[0%, 10%, 30%, 100%]``.  Users can
    override per accident if they want lower-energy events (e.g. drifting)
    to spill less even at the major / total-loss bands.
    """
    row = [0.0, 10.0, 30.0, 100.0]
    return [list(row) for _ in ACCIDENT_TYPES]


def default_catastrophe_levels() -> list[dict[str, Any]]:
    """Default catastrophe-level definitions.

    Returns three rows in ascending volume order.  The Consequence dialog
    enforces a minimum of two rows; quantities are in m^3.
    """
    return [
        {'name': 'Minor', 'quantity': 50.0},
        {'name': 'Major', 'quantity': 500.0},
        {'name': 'Catastrophic', 'quantity': 5000.0},
    ]


def reshape_oil_onboard(
    existing: list[list[float]] | None,
    ship_types: list[str],
    length_intervals: list[dict[str, Any]],
) -> list[list[float]]:
    """Reshape ``existing`` to the current ship-category dimensions.

    Preserves cells that map onto the new shape, fills the rest from the
    defaults.  Used by the oil-onboard dialog when the user adds / removes
    ship types or length intervals between save and re-open.
    """
    defaults = default_oil_onboard(ship_types, length_intervals)
    if not existing:
        return defaults
    out: list[list[float]] = []
    for i, default_row in enumerate(defaults):
        if i < len(existing) and isinstance(existing[i], list):
            new_row: list[float] = []
            for j, default_val in enumerate(default_row):
                if j < len(existing[i]):
                    try:
                        new_row.append(float(existing[i][j]))
                    except (TypeError, ValueError):
                        new_row.append(default_val)
                else:
                    new_row.append(default_val)
            out.append(new_row)
        else:
            out.append(list(default_row))
    return out
