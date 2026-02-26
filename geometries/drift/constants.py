# -*- coding: utf-8 -*-
"""
Direction constants using nautical/compass convention.

The convention follows counter-clockwise from North:
    0° = North (ships drift northward, +Y in UTM)
    45° = NorthWest
    90° = West
    135° = SouthWest
    180° = South
    225° = SouthEast
    270° = East
    315° = NorthEast
"""

DIRECTIONS = {
    'N': 0,       # North
    'NW': 45,     # NorthWest
    'W': 90,      # West
    'SW': 135,    # SouthWest
    'S': 180,     # South
    'SE': 225,    # SouthEast
    'E': 270,     # East
    'NE': 315,    # NorthEast
}
