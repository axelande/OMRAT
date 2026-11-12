"""Search-and-rescue aircraft are left out of the AIS traffic (KNOWN_ISSUES 23).

ITU-R M.585 gives SAR aircraft the MMSI range ``111MIDxxx``.  They were
binned as "Search and Rescue vessel" at 130-241 kn; on the Kattegat model one
such cell carried 8.7 % of the powered allision.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from omrat_utils.ais_binning import SAR_AIRCRAFT_SQL, is_sar_aircraft


@pytest.mark.parametrize('mmsi, expected', [
    ('111219500', True), (111265001, True), (' 111219500 ', True),
    ('219000111', False), ('1112195', False), ('1112195001', False), (None, False), (265547250, False),
])
def test_is_sar_aircraft(mmsi, expected):
    assert is_sar_aircraft(mmsi) is expected


def test_sql_filter_is_safe_for_psycopg_parameters():
    # A literal '%' would be read as a placeholder next to %(pl)s.
    assert '%' not in SAR_AIRCRAFT_SQL
    assert "'111______'" in SAR_AIRCRAFT_SQL


def test_traffic_query_leaves_out_sar_aircraft():
    from omrat_utils.handle_ais import _build_ais_union_block
    sql = _build_ais_union_block('sjfv', 2024, [1, 2, 3])
    assert sql.count(SAR_AIRCRAFT_SQL) == 3          # one per monthly table
    assert _build_ais_union_block('sjfv', 2024, []).count(SAR_AIRCRAFT_SQL) == 1


def _stub_ais(rows):
    from omrat_utils.handle_ais import AIS
    with patch("omrat_utils.handle_ais.DB"), patch("omrat_utils.handle_ais.AISConnectionWidget"):
        omrat = MagicMock()
        omrat.traffic = SimpleNamespace(traffic_data={})
        ais = AIS(omrat)
    ais.schema, ais.year, ais.months = 'test', 2023, []
    ais.db = MagicMock()
    ais.db.execute_and_return.return_value = (True, rows)
    return ais


def test_crossing_query_and_rows_leave_out_sar_aircraft():
    ais = _stub_ais([
        ['219000001', 7, 1000.0, 90.0],
        ['111219500', 8, 1001.0, 92.0],
    ])
    out = ais._fetch_crossings('LINESTRING(10 56, 10.1 56)')
    sql = ais.db.execute_and_return.call_args[0][0]
    assert SAR_AIRCRAFT_SQL in sql
    assert [m for m, *_ in out] == ['219000001']
