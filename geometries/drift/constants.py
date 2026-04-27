# -*- coding: utf-8 -*-
"""
Direction constants using standard nautical compass convention.

Clockwise from North:
    0° = North (ships drift northward, +Y in UTM)
    45° = NorthEast
    90° = East
    135° = SouthEast
    180° = South
    225° = SouthWest
    270° = West
    315° = NorthWest
"""

DIRECTIONS = {
    'N': 0,       # North
    'NE': 45,     # NorthEast
    'E': 90,      # East
    'SE': 135,    # SouthEast
    'S': 180,     # South
    'SW': 225,    # SouthWest
    'W': 270,     # West
    'NW': 315,    # NorthWest
}
