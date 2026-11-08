"""AIS updates size their matrices from the ship categories, not the table.

KNOWN_ISSUES item 1: the Traffic table has an extra Total row and column;
using its size gave 22 x 15 matrices whose last column held every ship longer
than the last length class.  Needs the QGIS conftest (``omrat`` fixture).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omrat_utils.ais_update_task import AisUpdateTask  # noqa: E402


def test_matrix_shape_excludes_the_total_row_and_column(omrat):
    omrat.traffic.set_table_headings()
    tw = omrat.main_widget.twTrafficData
    rows, cols = omrat.ais._matrix_shape()
    assert (rows, cols) == (tw.rowCount() - 1, tw.columnCount() - 1)
    assert len(omrat.ais._length_bounds(cols)) == cols


def test_long_ship_lands_in_the_last_length_class(omrat):
    omrat.traffic.set_table_headings()
    rows, cols = omrat.ais._matrix_shape()
    task = AisUpdateTask(
        omrat.ais, legs={}, seg_table={}, rows=rows, cols=cols,
        variables=list(omrat.traffic.variables),
        var_defaults=dict(omrat.traffic._var_cell_defaults),
        leg_dirs={}, length_bounds=omrat.ais._length_bounds(cols),
    )
    block = task._make_empty_block(['East going', 'West going'])
    freq = block['East going']['Frequency (ships/year)']
    assert (len(freq), len(freq[0])) == (rows, cols)
    omrat.ais.max_deviation = 45.0
    # loa, beam, toc, draught, mapped type, -, sog, air draught, dist, cog
    ping = (400, 60, 70, 15.0, None, None, 16.0, None, 0.0, 90.0)
    task._bin_ping(block, ['East going', 'West going'], ping, 90.0, [], [])
    cargo = 18
    assert block['East going']['Frequency (ships/year)'][cargo][cols - 1] == 1
